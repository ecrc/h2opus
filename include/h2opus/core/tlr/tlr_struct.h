#ifndef __H2OPUS_TLR_STRUCT_H__
#define __H2OPUS_TLR_STRUCT_H__

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/tlr/tlr_defs.h>

template <class T, int hw> struct TTLR_Matrix
{
    template <class other_T, int other_hw> friend struct TTLR_Matrix;

  private:
    typedef typename VectorContainer<hw, T *>::type RealPointerArray;
    typedef typename VectorContainer<hw, T>::type RealVector;
    typedef typename VectorContainer<hw, int>::type IntVector;
    typedef typename VectorContainer<H2OPUS_HWTYPE_CPU, T *>::type HostRealPointerArray;
    typedef typename VectorContainer<H2OPUS_HWTYPE_CPU, T>::type HostRealVector;
    typedef typename VectorContainer<H2OPUS_HWTYPE_CPU, int>::type HostIntVector;

    // Host pointers for allocations of the block data for U and V
    // if alloc is by tiles, then the pointers are a host copy of the block_U_ptrs and block_V_ptrs
    //                       i.e. it's a column major matrix of the pointers of the tiles
    // if alloc is by column, then the pointers are the allocations for each block column of tiles
    //                       i.e. there are n_block pointers for each block column and the block pointers
    //                       are assigned as offsets from each column pointer based on their rank
    HostRealPointerArray block_data_U, block_data_V;
    HostIntVector block_alloc_U, block_alloc_V;

    // Host data for setting ranks
    HostIntVector host_block_ranks;

    // Host pointers for the diagonal blocks
    HostRealPointerArray diagonal_block_host_ptrs;

    RealVector block_diagonal;
    IntVector column_rank_prefix_sum;

    TTLR_Matrix(const TTLR_Matrix &A);
    TTLR_Matrix &operator=(const TTLR_Matrix &A);
    template <int other_hw> TTLR_Matrix(const TTLR_Matrix<T, other_hw> &A);
    template <int other_hw> TTLR_Matrix &operator=(const TTLR_Matrix<T, other_hw> &A);

    template <int other_hw>
    void init(int n, H2OpusTLRType type, int max_rank, int block_size, const int *index_map, H2OpusTLRAlloc alloc,
              h2opusComputeStream_t stream);

  public:
    RealPointerArray block_U_ptrs, block_V_ptrs;
    RealPointerArray diagonal_block_ptrs;

    IntVector block_ranks;
    IntVector index_map;

    int n, block_size, max_rank;
    int n_block;
    H2OpusTLRType type;
    H2OpusTLRAlloc alloc;

    // Update the ranks of the block column
    // Ranks should be of size n_block if the matrix is non-symmetric and
    // (n_block - col_index - 1) for symmetric matrices
    // diagonal block rank should be set to zero for non-symmetric matrices
    // Clears out all low rank data stored in the block column
    void allocateBlockColumn(int col_index, const int *ranks, h2opusComputeStream_t stream);

    // Get host pointer for diagonal block
    T *getDiagonalBlockHostPtr(int block)
    {
        return diagonal_block_host_ptrs[block];
    }

    int getPaddedDim()
    {
        return n_block * block_size;
    }

    TTLR_Matrix(int n, H2OpusTLRType type, int block_size, const int *index_map, H2OpusTLRAlloc alloc,
                h2opusComputeStream_t stream);

    TTLR_Matrix(const TTLR_Matrix &A, h2opusComputeStream_t stream)
    {
        copy(A, stream);
    }

    ~TTLR_Matrix();

    template <int other_hw> TTLR_Matrix(const TTLR_Matrix<T, other_hw> &A, h2opusComputeStream_t stream)
    {
        copy(A, stream);
    }

    template <int other_hw> void copy(const TTLR_Matrix<T, other_hw> &A, h2opusComputeStream_t stream);

    size_t memoryUsage();

    size_t denseMemoryUsage();

    void swapDiagonalBlocks(int i, int j, h2opusComputeStream_t stream);

    void swapBlocks(int blocks, int b1_row, int b1_col, int inc_b1, int b2_row, int b2_col, int inc_b2, bool swap_uv,
                    h2opusComputeStream_t stream);

    void transposeBlock(int i, int j, h2opusComputeStream_t stream);
};

#include <h2opus/core/tlr/tlr_struct.cuh>

#endif
