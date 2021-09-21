#include <h2opus/core/thrust_runtime.h>
#include <h2opus/core/tlr/tlr_functors.h>
#include <h2opus/util/thrust_wrappers.h>

template <class T, int hw>
TTLR_Matrix<T, hw>::TTLR_Matrix(int n, H2OpusTLRType type, int block_size, const int *index_map, H2OpusTLRAlloc alloc,
                                h2opusComputeStream_t stream)
{
    // max rank is the block size for now
    init<hw>(n, type, block_size, block_size, index_map, alloc, stream);
}

template <class T, int hw> TTLR_Matrix<T, hw>::~TTLR_Matrix()
{
    for (size_t i = 0; i < block_data_U.size(); i++)
        freeVector<T, hw>(block_data_U[i]);

    for (size_t i = 0; i < block_data_V.size(); i++)
        freeVector<T, hw>(block_data_V[i]);
}

template <class T, int hw>
template <int other_hw>
void TTLR_Matrix<T, hw>::init(int n, H2OpusTLRType type, int max_rank, int block_size, const int *index_map,
                              H2OpusTLRAlloc alloc, h2opusComputeStream_t stream)
{
    this->n = n;
    this->type = type;
    this->alloc = alloc;
    this->block_size = block_size;
    this->max_rank = max_rank;

    this->n_block = (n + block_size - 1) / block_size;

    // Allocate dense diagonal blocks
    this->block_diagonal.resize(n_block * block_size * block_size);
    this->diagonal_block_ptrs.resize(n_block);
    this->diagonal_block_host_ptrs.resize(n_block);

    // Allocate block pointers
    this->block_U_ptrs.resize(n_block * n_block);
    this->block_V_ptrs.resize(n_block * n_block);
    fillArray(vec_ptr(this->block_U_ptrs), n_block * n_block, NULL, stream, hw);
    fillArray(vec_ptr(this->block_V_ptrs), n_block * n_block, NULL, stream, hw);

    // Allocate block ranks
    this->block_ranks.resize(n_block * n_block);
    fillArray(vec_ptr(this->block_ranks), this->block_ranks.size(), 0, stream, hw);

    size_t data_entries;
    if (alloc == H2OpusTLRColumn)
    {
        // Allocate the block low rank factor source memory by column
        data_entries = n_block;

        // Temporary array for prefix sums whenever the ranks of a block column changes
        this->column_rank_prefix_sum.resize(n_block);
    }
    else
    {
        // Allocate the block low rank factor source memory by tile
        data_entries = n_block * n_block;

        // Temporary array to copy the ranks to the host
        this->host_block_ranks.resize(n_block);
    }

    this->block_data_U.resize(data_entries, NULL);
    this->block_data_V.resize(data_entries, NULL);
    this->block_alloc_U.resize(data_entries, 0);
    this->block_alloc_V.resize(data_entries, 0);

    // Copy over the index map
    this->index_map.resize(n);
    if (index_map)
        copyVector(vec_ptr(this->index_map), hw, index_map, other_hw, n);
    else
        generateSequence(vec_ptr(this->index_map), n, 0, stream, hw);

    generateArrayOfPointers(vec_ptr(block_diagonal), vec_ptr(diagonal_block_ptrs), block_size * block_size, n_block,
                            stream, hw);
    copyVector(diagonal_block_host_ptrs, diagonal_block_ptrs);
}

template <class T, int hw>
template <int other_hw>
void TTLR_Matrix<T, hw>::copy(const TTLR_Matrix<T, other_hw> &A, h2opusComputeStream_t stream)
{
    init<other_hw>(A.n, A.type, A.max_rank, A.block_size, vec_ptr(A.index_map), A.alloc, stream);

    bool sym = (A.type == H2OpusTLR_Symmetric);
    bool update_lower_only = (sym || A.type == H2OpusTLR_LowerTriangular);
    bool update_triangular = (update_lower_only || A.type == H2OpusTLR_UpperTriangular);

    for (int col_index = 0; col_index < A.n_block; col_index++)
    {
        int block_row_start = (update_lower_only ? col_index + 1 : 0);
        int num_blocks = n_block - (update_triangular ? col_index + 1 : 0);

        const int *src_ranks = vec_ptr(A.block_ranks) + block_row_start + col_index * A.n_block;
        int *dest_ranks = vec_ptr(this->block_ranks) + block_row_start + col_index * A.n_block;
        copyVector(dest_ranks, hw, src_ranks, other_hw, num_blocks);

        // Set the pointers and allocate the necessary memory by setting the ranks
        this->allocateBlockColumn(col_index, dest_ranks, stream);

        // Copy the actual data over
        if (alloc == H2OpusTLRColumn)
        {
            copyVector(block_data_U[col_index], hw, A.block_data_U[col_index], other_hw, block_alloc_U[col_index]);
            copyVector(block_data_V[col_index], hw, A.block_data_V[col_index], other_hw, block_alloc_V[col_index]);
        }
        else
        {
            // TODO: use batch copy if on the same hardware
            for (int i = 0; i < num_blocks; i++)
            {
                int tile_index = col_index * n_block + i + block_row_start;
                copyVector(block_data_U[tile_index], hw, A.block_data_U[tile_index], other_hw,
                           block_alloc_U[tile_index]);
                copyVector(block_data_V[tile_index], hw, A.block_data_V[tile_index], other_hw,
                           block_alloc_V[tile_index]);
            }
        }
    }
    // TODO: Deal with pivoted diagonal data
    assert(this->block_diagonal.size() == A.block_diagonal.size());
    copyVector(this->block_diagonal, A.block_diagonal);
}

template <class T, int hw>
void TTLR_Matrix<T, hw>::allocateBlockColumn(int col_index, const int *ranks, h2opusComputeStream_t stream)
{
    assert(col_index < n_block);

    bool sym = (type == H2OpusTLR_Symmetric);
    bool update_lower_only = (sym || type == H2OpusTLR_LowerTriangular);
    bool update_triangular = (update_lower_only || type == H2OpusTLR_UpperTriangular);

    // Each block i in the block column will need block_size * ranks[i] memory for U and V
    // First compute the prefix sum of ranks so we can allocate the total amount needed
    // and then update the pointers in the block_*_ptrs arrays for that column
    int block_row_start = (update_lower_only ? col_index + 1 : 0);
    int num_blocks = n_block - (update_triangular ? col_index + 1 : 0);

    if (num_blocks == 0)
        return;

    if (alloc == H2OpusTLRColumn)
    {
        int *prefix_sum_ptr = vec_ptr(column_rank_prefix_sum);
        int total_ranks = inclusiveScan(ranks, num_blocks, prefix_sum_ptr, stream, hw);

        block_alloc_U[col_index] = block_size * total_ranks;
        block_alloc_V[col_index] = block_size * total_ranks;

        // Free the old tiles
        freeVector<T, hw>(block_data_U[col_index]);
        freeVector<T, hw>(block_data_V[col_index]);

        // Reallocate
        block_data_U[col_index] = allocateVector<T, hw>(block_alloc_U[col_index]);
        block_data_V[col_index] = allocateVector<T, hw>(block_alloc_V[col_index]);

        // Now we can update the pointers
        TLR_ColRankUpdate<T> ptr_updater(block_data_U[col_index], block_data_V[col_index], vec_ptr(block_U_ptrs),
                                         vec_ptr(block_V_ptrs), vec_ptr(block_ranks), ranks, prefix_sum_ptr, block_size,
                                         col_index, n_block, block_row_start, sym);

        thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(num_blocks), ptr_updater);
    }
    else
    {
        int linear_offset = block_row_start + col_index * n_block;
        // Copy the ranks
        int *block_ranks_offset = vec_ptr(block_ranks) + linear_offset;
        if (block_ranks_offset != ranks)
            copyVector(block_ranks_offset, hw, ranks, hw, num_blocks);

        // Copy the ranks over to the host and allocate the tiles
        copyVector(vec_ptr(host_block_ranks), H2OPUS_HWTYPE_CPU, ranks, hw, num_blocks);
        for (int i = 0; i < num_blocks; i++)
        {
            int tile_index = col_index * n_block + i + block_row_start;
            block_alloc_U[tile_index] = block_size * host_block_ranks[i];
            block_alloc_V[tile_index] = block_size * host_block_ranks[i];

            // Free the old tiles
            freeVector<T, hw>(block_data_U[tile_index]);
            freeVector<T, hw>(block_data_V[tile_index]);

            // Reallocate
            block_data_U[tile_index] = allocateVector<T, hw>(block_alloc_U[tile_index]);
            block_data_V[tile_index] = allocateVector<T, hw>(block_alloc_V[tile_index]);
        }

        // Copy the pointers over to block_*_ptrs
        T **src_U = vec_ptr(block_data_U) + linear_offset;
        T **src_V = vec_ptr(block_data_V) + linear_offset;
        T **dest_U = vec_ptr(block_U_ptrs) + linear_offset;
        T **dest_V = vec_ptr(block_V_ptrs) + linear_offset;

        copyVector(dest_U, hw, src_U, H2OPUS_HWTYPE_CPU, num_blocks);
        copyVector(dest_V, hw, src_V, H2OPUS_HWTYPE_CPU, num_blocks);

        // Transpose the pointers for the symmetric part
        if (sym)
        {
            TLR_SetTransposePtrs<T> ptr_updater(vec_ptr(block_U_ptrs), vec_ptr(block_V_ptrs), vec_ptr(block_ranks),
                                                col_index, block_row_start, n_block);

            thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                             thrust::counting_iterator<int>(num_blocks), ptr_updater);
        }
    }
}

template <class T, int hw> void TTLR_Matrix<T, hw>::swapDiagonalBlocks(int i, int j, h2opusComputeStream_t stream)
{
    swap_vectors(1, vec_ptr(diagonal_block_ptrs) + i, 1, vec_ptr(diagonal_block_ptrs) + j, 1, hw, stream);
    swap_vectors(1, vec_ptr(diagonal_block_host_ptrs) + i, 1, vec_ptr(diagonal_block_host_ptrs) + j, 1,
                 H2OPUS_HWTYPE_CPU, stream);
}

template <class T, int hw>
void TTLR_Matrix<T, hw>::swapBlocks(int blocks, int b1_row, int b1_col, int inc_b1, int b2_row, int b2_col, int inc_b2,
                                    bool swap_uv, h2opusComputeStream_t stream)
{
    assert(alloc == H2OpusTLRTile);

    if (blocks <= 0)
        return;

    int b1_index = b1_row + b1_col * n_block;
    int b2_index = b2_row + b2_col * n_block;

    // On hw
    T **b1_U = vec_ptr(block_U_ptrs) + b1_index;
    T **b1_V = vec_ptr(block_V_ptrs) + b1_index;
    int *b1_ranks = vec_ptr(block_ranks) + b1_index;

    T **b2_U = vec_ptr(block_U_ptrs) + b2_index;
    T **b2_V = vec_ptr(block_V_ptrs) + b2_index;
    int *b2_ranks = vec_ptr(block_ranks) + b2_index;

    if (swap_uv)
        std::swap(b2_U, b2_V);

    swap_vectors(blocks, b1_U, inc_b1, b2_U, inc_b2, hw, stream);
    swap_vectors(blocks, b1_V, inc_b1, b2_V, inc_b2, hw, stream);
    swap_vectors(blocks, b1_ranks, inc_b1, b2_ranks, inc_b2, hw, stream);

    // Host only data
    T **b1_data_U = vec_ptr(block_data_U) + b1_index;
    T **b1_data_V = vec_ptr(block_data_V) + b1_index;
    int *b1_alloc_U = vec_ptr(block_alloc_U) + b1_index;
    int *b1_alloc_V = vec_ptr(block_alloc_V) + b1_index;

    T **b2_data_U = vec_ptr(block_data_U) + b2_index;
    T **b2_data_V = vec_ptr(block_data_V) + b2_index;
    int *b2_alloc_U = vec_ptr(block_alloc_U) + b2_index;
    int *b2_alloc_V = vec_ptr(block_alloc_V) + b2_index;

    if (swap_uv)
    {
        std::swap(b2_U, b2_V);
        std::swap(b2_alloc_U, b2_alloc_V);
    }

    swap_vectors(blocks, b1_data_U, inc_b1, b2_data_U, inc_b2, H2OPUS_HWTYPE_CPU, stream);
    swap_vectors(blocks, b1_data_V, inc_b1, b2_data_V, inc_b2, H2OPUS_HWTYPE_CPU, stream);
    swap_vectors(blocks, b1_alloc_U, inc_b1, b2_alloc_U, inc_b2, H2OPUS_HWTYPE_CPU, stream);
    swap_vectors(blocks, b1_alloc_V, inc_b1, b2_alloc_V, inc_b2, H2OPUS_HWTYPE_CPU, stream);
}

template <class T, int hw> void TTLR_Matrix<T, hw>::transposeBlock(int i, int j, h2opusComputeStream_t stream)
{
    assert(alloc == H2OpusTLRTile);

    int index = i + j * n_block;

    // On hw
    T **U = vec_ptr(block_U_ptrs) + index;
    T **V = vec_ptr(block_V_ptrs) + index;

    swap_vectors(1, U, 1, V, 1, hw, stream);

    // Host only data
    T **data_U = vec_ptr(block_data_U) + index;
    T **data_V = vec_ptr(block_data_U) + index;
    int *alloc_U = vec_ptr(block_alloc_U) + index;
    int *alloc_V = vec_ptr(block_alloc_V) + index;

    swap_vectors(1, data_U, 1, data_V, 1, hw, stream);
    swap_vectors(1, alloc_U, 1, alloc_V, 1, hw, stream);
}

template <class T, int hw> size_t TTLR_Matrix<T, hw>::memoryUsage()
{
    size_t dense_bytes = denseMemoryUsage();

    size_t lr_bytes = 0;
    for (size_t i = 0; i < block_alloc_U.size(); i++)
        lr_bytes += sizeof(T) * block_alloc_U[i];
    for (size_t i = 0; i < block_alloc_V.size(); i++)
        lr_bytes += sizeof(T) * block_alloc_V[i];

    size_t ptr_bytes = (2 * n_block * n_block + n_block) * sizeof(T *);
    size_t int_bytes = (n_block * n_block + 2 * n_block) * sizeof(int);

    return dense_bytes + lr_bytes + ptr_bytes + int_bytes;
}

template <class T, int hw> size_t TTLR_Matrix<T, hw>::denseMemoryUsage()
{
    size_t dense_bytes = block_diagonal.size() * sizeof(T);

    return dense_bytes;
}
