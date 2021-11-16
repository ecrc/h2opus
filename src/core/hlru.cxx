#include <h2opus/core/hlru.h>
#include <h2opus/marshal/hlru_marshal.h>

#include <h2opus/util/batch_wrappers.h>
#include <h2opus/util/debug_routines.h>
#include <h2opus/util/gpu_err_check.h>
#include <h2opus/util/thrust_wrappers.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Updating low rank update pointers to allow splitting into multiple updates
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class H2Opus_Real> struct HLRU_Offset_Pointers
{
    H2Opus_Real **U, **V;
    int rank, ldu, ldv;
    HLRU_Offset_Pointers(H2Opus_Real **U, int ldu, H2Opus_Real **V, int ldv, int rank)
    {
        this->U = U;
        this->ldu = ldu;
        this->V = V;
        this->ldv = ldv;
        this->rank = rank;
    }

    __host__ __device__ void operator()(int index)
    {
        U[index] += rank * ldu;
        V[index] += rank * ldv;
    }
};

template <int hw> void hlru_advance_pointers(TLowRankUpdate<hw> &update, int rank, h2opusComputeStream_t stream)
{
    HLRU_Offset_Pointers<H2Opus_Real> offset_pointers(vec_ptr(update.U), update.ldu, vec_ptr(update.V), update.ldv,
                                                      rank);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(update.num_updates), offset_pointers);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Core low rank update routines
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <int hw>
void hlru_downsweep_hnode_updates(THNodeTree<hw> &hnodes, int update_level, int num_updates, int *updated_hnodes,
                                  int *hnode_flags, h2opusComputeStream_t stream)
{
    HNodeTreeLevelData &hnode_level_data = hnodes.level_data;
    int *parent = vec_ptr(hnodes.parent);

    int num_levels = hnode_level_data.depth;
    assert(update_level < num_levels && update_level >= 0);

    // Clear out the temporary indexing data
    fillArray(hnode_flags, hnodes.num_nodes, -1, stream, hw);

    // Initialize the level of the update
    hlru_init_hnode_update<hw>(hnode_flags, updated_hnodes, num_updates, stream);

    // Sweep down the matrix tree
    for (int level = update_level + 1; level < num_levels; level++)
    {
        int level_start = hnode_level_data.getLevelStart(level);
        int level_end = hnode_level_data.getLevelEnd(level);

        hlru_downsweep_hnode_update<hw>(hnode_flags, parent, level_start, level_end, stream);
    }
}

template <int hw>
void hlru_update_coupling_matrices(THNodeTree<hw> &hnodes, int update_level, int update_rank, int *hnode_flags,
                                   h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    typedef typename VectorContainer<hw, H2Opus_Real *>::type RealPointerArray;

    HNodeTreeLevelData &hnode_level_data = hnodes.level_data;

    int num_levels = hnodes.depth;
    assert(update_level < num_levels && update_level >= 0);

    // Pointers used in the batch routines
    int max_nodes = hnode_level_data.getMaxLevelCouplingNodes();
    RealPointerArray original_node_ptrs(max_nodes), new_node_ptrs(max_nodes);
    RealPointerArray flagged_ptrs(max_nodes);

    // Set S = [S 0; 0 I]
    for (int level = update_level; level < hnode_level_data.depth; level++)
    {

        int level_rank = hnode_level_data.getLevelRank(level);
        int level_nodes = hnode_level_data.getCouplingLevelSize(level);
        int level_start = hnode_level_data.getCouplingLevelStart(level);

        int new_rank = update_rank + level_rank;

        RealVector old_coupling_level;
        copyVector(old_coupling_level, hnodes.rank_leaf_mem[level]);
        hnodes.rank_leaf_mem[level].resize(level_nodes * new_rank * new_rank);
        initVector(hnodes.rank_leaf_mem[level], (H2Opus_Real)0, stream);

        // Copy the original data into the appropriate sub-block of the new coupling matrices
        generateArrayOfPointers(vec_ptr(old_coupling_level), vec_ptr(original_node_ptrs), level_rank * level_rank,
                                level_nodes, stream, hw);
        generateArrayOfPointers(vec_ptr(hnodes.rank_leaf_mem[level]), vec_ptr(new_node_ptrs), new_rank * new_rank,
                                level_nodes, stream, hw);

        check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copyBlock)(
            stream, level_rank, level_rank, vec_ptr(new_node_ptrs), 0, 0, new_rank, vec_ptr(original_node_ptrs), 0, 0,
            level_rank, level_nodes));

        // Now we need to reduce the pointer array to include just the flagged hnodes
        int flagged_nodes = hlru_flagged_coupling_marshal_batch<H2Opus_Real, hw>(
            vec_ptr(new_node_ptrs), vec_ptr(flagged_ptrs), hnode_flags, vec_ptr(hnodes.rank_leaf_tree_index),
            level_rank, new_rank, level_start, level_nodes, stream);

        // Set the lower right block of the flagged nodes to the identity
        H2OpusBatched<H2Opus_Real, hw>::setIdentity(stream, update_rank, update_rank, vec_ptr(flagged_ptrs), new_rank,
                                                    flagged_nodes);
    }
}

template <int hw>
void hlru_update_basis_leaves(TBasisTree<hw> &basis_tree, H2Opus_Real **basis_update, int update_rank,
                              int *basis_update_index, int *basis_update_row, int update_ld,
                              h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    typedef typename VectorContainer<hw, int>::type IntVector;
    typedef typename VectorContainer<hw, H2Opus_Real *>::type RealPointerArray;

    BasisTreeLevelData &level_data = basis_tree.level_data;

    int num_levels = basis_tree.depth;
    int num_leaves = basis_tree.basis_leaves;
    int leaf_size = basis_tree.leaf_size;
    int leaf_rank = level_data.getLevelRank(num_levels - 1);
    int new_rank = update_rank + leaf_rank;

    int leaf_start, leaf_end;
    level_data.getLevelRange(num_levels - 1, leaf_start, leaf_end);

    RealVector old_basis_leaves;
    copyVector(old_basis_leaves, basis_tree.basis_mem);
    basis_tree.basis_mem.resize(num_leaves * leaf_size * new_rank);
    initVector(basis_tree.basis_mem, (H2Opus_Real)0, stream);

    RealPointerArray dest_ptrs(num_leaves), origin_ptrs(num_leaves);
    RealPointerArray flagged_dest_ptrs(num_leaves), flagged_origin_ptrs(num_leaves);
    IntVector rows_array(num_leaves), cols_array(num_leaves);
    IntVector ld_src_array(num_leaves), ld_dest_array(num_leaves);

    // First copy the original leaves
    generateArrayOfPointers(vec_ptr(old_basis_leaves), vec_ptr(origin_ptrs), leaf_size * leaf_rank, num_leaves, stream,
                            hw);
    generateArrayOfPointers(vec_ptr(basis_tree.basis_mem), vec_ptr(dest_ptrs), leaf_size * new_rank, num_leaves, stream,
                            hw);

    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copyBlock)(stream, leaf_size, leaf_rank, vec_ptr(dest_ptrs), 0,
                                                                  0, leaf_size, vec_ptr(origin_ptrs), 0, 0, leaf_size,
                                                                  num_leaves));

    // Generate pointers for the flagged nodes
    fillArray(vec_ptr(ld_dest_array), num_leaves, leaf_size, stream, hw);
    fillArray(vec_ptr(ld_src_array), num_leaves, update_ld, stream, hw);

    int flagged_nodes = hlru_flagged_basis_marshal_batch<H2Opus_Real, hw>(
        vec_ptr(dest_ptrs), basis_update, vec_ptr(flagged_dest_ptrs), vec_ptr(flagged_origin_ptrs), vec_ptr(rows_array),
        vec_ptr(cols_array), basis_update_index, basis_update_row, vec_ptr(basis_tree.node_len), leaf_size, leaf_rank,
        update_rank, leaf_start, num_leaves, stream);

    // Copy over the flagged blocks
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copyBlock)(
        stream, vec_ptr(rows_array), vec_ptr(cols_array), leaf_size, update_rank, vec_ptr(flagged_dest_ptrs),
        vec_ptr(ld_dest_array), vec_ptr(flagged_origin_ptrs), vec_ptr(ld_src_array), flagged_nodes));
}

template <int hw>
void hlru_update_transfer_matrices(TBasisTree<hw> &basis_tree, int rank, int update_level, int *basis_update_index,
                                   h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    typedef typename VectorContainer<hw, H2Opus_Real *>::type RealPointerArray;

    BasisTreeLevelData &level_data = basis_tree.level_data;

    int num_levels = basis_tree.depth;
    assert(update_level < num_levels && update_level >= 0);

    int max_nodes = level_data.getLevelSize(level_data.getLargestLevel());
    RealPointerArray dest_ptrs(max_nodes), origin_ptrs(max_nodes);

    // Set F = [F 0; 0 I]
    for (int level = update_level; level < num_levels; level++)
    {
        // Get the transfer matrix dimensions for this level
        int rows, cols;
        level_data.getTransferDims(level, rows, cols);
        int level_nodes = level_data.getLevelSize(level);
        int level_start = level_data.getLevelStart(level);

        int new_rows = rows + rank, new_cols = cols + rank;
        if (level == update_level)
            new_cols = cols;

        RealVector old_transfer_level;
        copyVector(old_transfer_level, basis_tree.trans_mem[level]);
        basis_tree.trans_mem[level].resize(level_nodes * new_rows * new_cols);
        initVector(basis_tree.trans_mem[level], (H2Opus_Real)0, stream);

        // Copy the original blocks
        generateArrayOfPointers(vec_ptr(old_transfer_level), vec_ptr(origin_ptrs), rows * cols, level_nodes, stream,
                                hw);
        generateArrayOfPointers(vec_ptr(basis_tree.trans_mem[level]), vec_ptr(dest_ptrs), new_rows * new_cols,
                                level_nodes, stream, hw);

        check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copyBlock)(
            stream, rows, cols, vec_ptr(dest_ptrs), 0, 0, new_rows, vec_ptr(origin_ptrs), 0, 0, rows, level_nodes));

        // Set the lower right block to the identity matrix
        if (level != update_level)
        {
            // Reuse the pointer array - referenced for clarity
            H2Opus_Real **flagged_ptrs = vec_ptr(origin_ptrs);

            int flagged_nodes = hlru_flagged_transfer_marshal_batch<H2Opus_Real, hw>(
                vec_ptr(dest_ptrs), flagged_ptrs, basis_update_index, level_start, rows + cols * new_rows, level_nodes,
                stream);

            H2OpusBatched<H2Opus_Real, hw>::setIdentity(stream, rank, rank, flagged_ptrs, new_rows, flagged_nodes);
        }
    }
}

template <int hw>
void hlru_update_dense_blocks(THNodeTree<hw> &hnodes, TBasisTree<hw> &u_basis_tree, TBasisTree<hw> &v_basis_tree,
                              int *basis_update_row, int *basis_update_col, H2Opus_Real **U, int ldu, H2Opus_Real **V,
                              int ldv, int rank, int *hnode_update_index, h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, int>::type IntVector;
    typedef typename VectorContainer<hw, H2Opus_Real *>::type RealPointerArray;

    int num_dense_leaves = hnodes.num_dense_leaves;
    int dense_dim = hnodes.leaf_size;

    RealPointerArray dense_ptrs(num_dense_leaves), flagged_m_ptrs(num_dense_leaves);
    RealPointerArray flagged_u_ptrs(num_dense_leaves), flagged_v_ptrs(num_dense_leaves);

    IntVector rows_array(num_dense_leaves), cols_array(num_dense_leaves), ranks_array(num_dense_leaves);
    IntVector ldm_array(num_dense_leaves), ldv_array(num_dense_leaves), ldu_array(num_dense_leaves);

    generateArrayOfPointers(vec_ptr(hnodes.dense_leaf_mem), vec_ptr(dense_ptrs), dense_dim * dense_dim,
                            num_dense_leaves, stream, hw);

    int flagged_blocks = hlru_dense_update_marshal_batch<H2Opus_Real, hw>(
        vec_ptr(dense_ptrs), dense_dim, U, ldu, V, ldv, rank, vec_ptr(flagged_m_ptrs), vec_ptr(flagged_u_ptrs),
        vec_ptr(flagged_v_ptrs), vec_ptr(rows_array), vec_ptr(cols_array), vec_ptr(ranks_array), vec_ptr(ldm_array),
        vec_ptr(ldu_array), vec_ptr(ldv_array), vec_ptr(hnodes.dense_leaf_tree_index), vec_ptr(hnodes.node_u_index),
        vec_ptr(hnodes.node_v_index), vec_ptr(u_basis_tree.node_len), vec_ptr(v_basis_tree.node_len), basis_update_row,
        basis_update_col, hnode_update_index, num_dense_leaves, stream);

    // M += U * V^T
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
        stream, H2Opus_NoTrans, H2Opus_Trans, vec_ptr(rows_array), vec_ptr(cols_array), vec_ptr(ranks_array), dense_dim,
        dense_dim, rank, (H2Opus_Real)1, (const H2Opus_Real **)vec_ptr(flagged_u_ptrs), vec_ptr(ldu_array),
        (const H2Opus_Real **)vec_ptr(flagged_v_ptrs), vec_ptr(ldv_array), (H2Opus_Real)1, vec_ptr(flagged_m_ptrs),
        vec_ptr(ldm_array), flagged_blocks));
}

template <int hw> void hlru_init_temp_data(THMatrix<hw> &hmatrix, TLowRankUpdate<hw> &update)
{
    update.u_basis_update_index.resize(hmatrix.u_basis_tree.num_nodes);
    update.u_basis_update_row.resize(hmatrix.u_basis_tree.num_nodes);

    if (!hmatrix.sym)
    {
        update.v_basis_update_index.resize(hmatrix.v_basis_tree.num_nodes);
        update.v_basis_update_row.resize(hmatrix.v_basis_tree.num_nodes);
    }

    update.hnode_update_index.resize(hmatrix.hnodes.num_nodes);
}

template <int hw>
void hlru_downsweep_basis_updates(TBasisTree<hw> &basis_tree, int *hnode_basis_index, int update_level, int num_updates,
                                  int *updated_hnodes, int *basis_update_index, int *basis_update_row,
                                  h2opusComputeStream_t stream)
{
    BasisTreeLevelData &level_data = basis_tree.level_data;

    int num_levels = basis_tree.depth;
    assert(update_level < num_levels && update_level >= 0);

    // Clear out the temporary indexing data
    fillArray(basis_update_index, basis_tree.num_nodes, -1, stream, hw);
    fillArray(basis_update_row, basis_tree.num_nodes, -1, stream, hw);

    // Initialize the level of the update
    hlru_init_basis_update<hw>(basis_update_index, basis_update_row, hnode_basis_index, updated_hnodes, num_updates,
                               stream);

    // Sweep down the basis tree
    for (int level = update_level + 1; level < num_levels; level++)
    {
        int level_start = level_data.getLevelStart(level);
        int level_size = level_data.getLevelSize(level);

        hlru_downsweep_basis_update<hw>(basis_tree.parent_ptr(), vec_ptr(basis_tree.node_start), basis_update_index,
                                        basis_update_row, level_start, level_size, stream);
    }
}

template <int hw> int hlru_sym_template(THMatrix<hw> &hmatrix, TLowRankUpdate<hw> &update, h2opusHandle_t handle)
{
    assert(hmatrix.sym == true);

    if (update.applied_rank == update.total_rank)
        return H2OPUS_LRU_DONE;

    int update_rank = std::min(update.rank_per_update, update.total_rank - update.applied_rank);

    HNodeTreeLevelData &hnode_level_data = hmatrix.hnodes.level_data;
    BasisTreeLevelData &u_level_data = hmatrix.u_basis_tree.level_data;
    h2opusComputeStream_t main_stream = handle->getMainStream();

    // Update basis trees
    if (update.applied_rank == 0)
    {
        hlru_init_temp_data<hw>(hmatrix, update);

        hlru_downsweep_basis_updates<hw>(hmatrix.u_basis_tree, vec_ptr(hmatrix.hnodes.node_u_index), update.level,
                                         update.num_updates, vec_ptr(update.hnode_indexes),
                                         vec_ptr(update.u_basis_update_index), vec_ptr(update.u_basis_update_row),
                                         main_stream);
    }

    hlru_update_basis_leaves<hw>(hmatrix.u_basis_tree, vec_ptr(update.U), update_rank,
                                 vec_ptr(update.u_basis_update_index), vec_ptr(update.u_basis_update_row), update.ldu,
                                 main_stream);

    hlru_update_transfer_matrices<hw>(hmatrix.u_basis_tree, update_rank, update.level,
                                      vec_ptr(update.u_basis_update_index), main_stream);

    // Update hnodetree
    if (update.applied_rank == 0)
    {
        hlru_downsweep_hnode_updates<hw>(hmatrix.hnodes, update.level, update.num_updates,
                                         vec_ptr(update.hnode_indexes), vec_ptr(update.hnode_update_index),
                                         main_stream);
    }

    hlru_update_coupling_matrices<hw>(hmatrix.hnodes, update.level, update_rank, vec_ptr(update.hnode_update_index),
                                      main_stream);

    hlru_update_dense_blocks<hw>(hmatrix.hnodes, hmatrix.u_basis_tree, hmatrix.u_basis_tree,
                                 vec_ptr(update.u_basis_update_row), vec_ptr(update.u_basis_update_row),
                                 vec_ptr(update.U), update.ldu, vec_ptr(update.V), update.ldv, update_rank,
                                 vec_ptr(update.hnode_update_index), main_stream);

    // Update ranks
    // TODO: Assuming all levels below the update level are affected by the low rank update
    std::vector<int> new_ranks(hmatrix.u_basis_tree.depth);
    for (int i = 0; i < (int)new_ranks.size(); i++)
        new_ranks[i] = u_level_data.getLevelRank(i) + (i >= update.level ? update_rank : 0);

    u_level_data.setLevelRanks(vec_ptr(new_ranks));
    hnode_level_data.setRankFromBasis(u_level_data, 0);

    u_level_data.nested_root_level = update.level;

    // Advance the pointers within the low rank update if necessary
    update.applied_rank += update_rank;
    if (update.applied_rank != update.total_rank)
    {
        hlru_advance_pointers(update, update_rank, main_stream);
        return H2OPUS_LRU_NOT_DONE;
    }
    else
        return H2OPUS_LRU_DONE;
}

template <int hw>
void hlru_dense_block_update_template(THMatrix<hw> &hmatrix, TDenseBlockUpdate<hw> &update, h2opusHandle_t handle)
{
    typedef typename VectorContainer<hw, H2Opus_Real *>::type RealPointerArray;
    typedef typename VectorContainer<hw, int>::type IntVector;

    int leaf_size = hmatrix.u_basis_tree.leaf_size;
    assert(leaf_size == update.block_dim);

    TBasisTree<hw> &u_basis_tree = hmatrix.u_basis_tree;
    TBasisTree<hw> &v_basis_tree = (hmatrix.sym ? u_basis_tree : hmatrix.v_basis_tree);
    h2opusComputeStream_t main_stream = handle->getMainStream();

    RealPointerArray dense_ptrs(update.num_updates);
    IntVector rows_array(update.num_updates), cols_array(update.num_updates);
    IntVector ld_src_array(update.num_updates), ld_dest_array(update.num_updates);

    fillArray(vec_ptr(ld_src_array), update.num_updates, update.update_ld, main_stream, hw);
    fillArray(vec_ptr(ld_dest_array), update.num_updates, leaf_size, main_stream, hw);

    hlru_dense_block_update_marshal_batch<H2Opus_Real, hw>(
        vec_ptr(dense_ptrs), vec_ptr(hmatrix.hnodes.dense_leaf_mem), vec_ptr(rows_array), vec_ptr(cols_array),
        vec_ptr(hmatrix.hnodes.node_u_index), vec_ptr(hmatrix.hnodes.node_v_index), vec_ptr(u_basis_tree.node_len),
        vec_ptr(v_basis_tree.node_len), vec_ptr(update.hnode_indexes), vec_ptr(hmatrix.hnodes.node_type),
        vec_ptr(hmatrix.hnodes.node_to_leaf), leaf_size * leaf_size, update.num_updates, main_stream);

    // M = M + D
    H2OpusBatched<H2Opus_Real, hw>::add_matrix(main_stream, vec_ptr(rows_array), vec_ptr(cols_array), leaf_size,
                                               leaf_size, 1, vec_ptr(dense_ptrs), vec_ptr(ld_dest_array), 1,
                                               vec_ptr(update.M), vec_ptr(ld_src_array), vec_ptr(dense_ptrs),
                                               vec_ptr(ld_dest_array), update.num_updates);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Interface routines
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int hlru_sym(HMatrix &hmatrix, LowRankUpdate &update, h2opusHandle_t handle)
{
    return hlru_sym_template<H2OPUS_HWTYPE_CPU>(hmatrix, update, handle);
}

void hlru_dense_block_update(HMatrix &hmatrix, DenseBlockUpdate &update, h2opusHandle_t handle)
{
    hlru_dense_block_update_template<H2OPUS_HWTYPE_CPU>(hmatrix, update, handle);
}

#ifdef H2OPUS_USE_GPU
int hlru_sym(HMatrix_GPU &hmatrix, LowRankUpdate_GPU &update, h2opusHandle_t handle)
{
    return hlru_sym_template<H2OPUS_HWTYPE_GPU>(hmatrix, update, handle);
}

void hlru_dense_block_update(HMatrix_GPU &hmatrix, DenseBlockUpdate_GPU &update, h2opusHandle_t handle)
{
    hlru_dense_block_update_template<H2OPUS_HWTYPE_GPU>(hmatrix, update, handle);
}
#endif
