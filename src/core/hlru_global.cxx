#include <h2opus/core/hlru.h>
#include <h2opus/marshal/hlru_marshal_global.h>

#include <h2opus/util/batch_wrappers.h>
#include <h2opus/util/debug_routines.h>
#include <h2opus/util/gpu_err_check.h>
#include <h2opus/util/thrust_wrappers.h>

////////////////////////////////////////////////////////////////
// Global low rank update
////////////////////////////////////////////////////////////////
template <int hw>
void hlru_update_dense_blocks_global(THNodeTree<hw> &hnodes, TBasisTree<hw> &u_basis_tree, TBasisTree<hw> &v_basis_tree,
                                     const H2Opus_Real *U, int ldu, const H2Opus_Real *V, int ldv, int rank, H2Opus_Real s,
                                     h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, int>::type IntVector;
    typedef typename VectorContainer<hw, H2Opus_Real *>::type RealPointerArray;

    int num_dense_leaves = hnodes.num_dense_leaves;
    int dense_dim = hnodes.leaf_size;

    RealPointerArray dense_ptrs(num_dense_leaves);
    RealPointerArray U_ptrs(num_dense_leaves), V_ptrs(num_dense_leaves);

    IntVector rows_array(num_dense_leaves), cols_array(num_dense_leaves), ranks_array(num_dense_leaves);
    IntVector ldm_array(num_dense_leaves), ldv_array(num_dense_leaves), ldu_array(num_dense_leaves);

    generateArrayOfPointers(vec_ptr(hnodes.dense_leaf_mem), vec_ptr(dense_ptrs), dense_dim * dense_dim,
                            num_dense_leaves, stream, hw);

    hlru_dense_update_global_marshal_batch<H2Opus_Real, hw>(
        dense_dim, U, ldu, V, ldv, rank, vec_ptr(U_ptrs), vec_ptr(V_ptrs), vec_ptr(rows_array), vec_ptr(cols_array),
        vec_ptr(ranks_array), vec_ptr(ldm_array), vec_ptr(ldu_array), vec_ptr(ldv_array),
        vec_ptr(hnodes.dense_leaf_tree_index), vec_ptr(hnodes.node_u_index), vec_ptr(hnodes.node_v_index),
        vec_ptr(u_basis_tree.node_start), vec_ptr(v_basis_tree.node_start), vec_ptr(u_basis_tree.node_len),
        vec_ptr(v_basis_tree.node_len), num_dense_leaves, stream);

    // M += U * V^T
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
        stream, H2Opus_NoTrans, H2Opus_Trans, vec_ptr(rows_array), vec_ptr(cols_array), vec_ptr(ranks_array), dense_dim,
        dense_dim, rank, s, (const H2Opus_Real **)vec_ptr(U_ptrs), vec_ptr(ldu_array),
        (const H2Opus_Real **)vec_ptr(V_ptrs), vec_ptr(ldv_array), (H2Opus_Real)1, vec_ptr(dense_ptrs),
        vec_ptr(ldm_array), num_dense_leaves));
}

template <int hw>
void hlru_update_coupling_matrices_global(THNodeTree<hw> &hnodes, H2Opus_Real s, int update_rank,
                                          h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, H2Opus_Real *>::type RealPointerArray;
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

    HNodeTreeLevelData &hnode_level_data = hnodes.level_data;
    int num_levels = hnode_level_data.depth;

    // Pointers used in the batch routines
    int max_nodes = hnode_level_data.getMaxLevelCouplingNodes();
    RealPointerArray original_node_ptrs(max_nodes), new_node_ptrs(max_nodes);

    // Set S = [S 0; 0 sI]
    for (int level = 0; level < num_levels; level++)
    {
        int level_rank = hnode_level_data.getLevelRank(level);
        int level_nodes = hnode_level_data.getCouplingLevelSize(level);
        // int level_start = hnode_level_data.getCouplingLevelStart(level);

        if (level_nodes == 0)
            continue;

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

        // Set the lower right block of the new coupling matrices to the scaled identity
        hlru_offset_pointer_array<H2Opus_Real, hw>(vec_ptr(new_node_ptrs), new_rank, level_rank, level_rank,
                                                   level_nodes, stream);

        H2OpusBatched<H2Opus_Real, hw>::setDiagonal(stream, update_rank, update_rank, vec_ptr(new_node_ptrs), new_rank,
                                                    s, level_nodes);
    }
}

template <int hw>
void hlru_update_transfer_matrices_global(TBasisTree<hw> &basis_tree, int update_rank, h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, H2Opus_Real *>::type RealPointerArray;
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

    BasisTreeLevelData &level_data = basis_tree.level_data;

    // Temporary pointer data
    int max_nodes = level_data.getLevelSize(level_data.getLargestLevel());
    RealPointerArray new_node_ptrs(max_nodes), original_node_ptrs(max_nodes);
    int num_levels = level_data.depth;

    // First update all the transfer matrices by padding with an identity matrix
    for (int level = 0; level < num_levels; level++)
    {
        // Get the transfer matrix dimensions for this level
        int rows, cols;
        level_data.getTransferDims(level, rows, cols);
        int level_nodes = level_data.getLevelSize(level);
        // int level_start = level_data.getLevelStart(level);

        int new_rows = rows + update_rank, new_cols = cols + update_rank;

        RealVector old_transfer_level;
        copyVector(old_transfer_level, basis_tree.trans_mem[level]);
        basis_tree.trans_mem[level].resize(level_nodes * new_rows * new_cols);
        initVector(basis_tree.trans_mem[level], (H2Opus_Real)0, stream);

        // Copy the original blocks
        generateArrayOfPointers(vec_ptr(old_transfer_level), vec_ptr(original_node_ptrs), rows * cols, level_nodes,
                                stream, hw);
        generateArrayOfPointers(vec_ptr(basis_tree.trans_mem[level]), vec_ptr(new_node_ptrs), new_rows * new_cols,
                                level_nodes, stream, hw);

        check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copyBlock)(stream, rows, cols, vec_ptr(new_node_ptrs), 0, 0,
                                                                      new_rows, vec_ptr(original_node_ptrs), 0, 0, rows,
                                                                      level_nodes));

        // Set the lower right block of the new transfer matrices to the identity
        hlru_offset_pointer_array<H2Opus_Real, hw>(vec_ptr(new_node_ptrs), new_rows, rows, cols, level_nodes, stream);

        H2OpusBatched<H2Opus_Real, hw>::setIdentity(stream, update_rank, update_rank, vec_ptr(new_node_ptrs), new_rows,
                                                    level_nodes);
    }
}

template <int hw>
void hlru_sym_update_basis_leaves_global(TBasisTree<hw> &basis_tree, const H2Opus_Real *A, int lda, int update_rank,
                                         h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    typedef typename VectorContainer<hw, int>::type IntVector;
    typedef typename VectorContainer<hw, H2Opus_Real *>::type RealPointerArray;

    BasisTreeLevelData &level_data = basis_tree.level_data;

    // Update the leaves
    int num_levels = level_data.depth;
    int num_leaves = basis_tree.basis_leaves;
    int leaf_size = basis_tree.leaf_size;
    int leaf_rank = level_data.getLevelRank(num_levels - 1);
    int new_rank = update_rank + leaf_rank;

    int leaf_start, leaf_end;
    level_data.getLevelRange(num_levels - 1, leaf_start, leaf_end);

    RealPointerArray new_basis_ptrs(num_leaves), original_basis_ptrs(num_leaves);
    RealPointerArray updated_basis_ptrs(num_leaves), update_ptrs(num_leaves);

    IntVector rows_array(num_leaves), cols_array(num_leaves);
    IntVector ld_src_array(num_leaves), ld_dest_array(num_leaves);

    RealVector old_leaves;
    copyVector(old_leaves, basis_tree.basis_mem);
    basis_tree.basis_mem.resize(num_leaves * leaf_size * new_rank);
    initVector(basis_tree.basis_mem, (H2Opus_Real)0, stream);

    // First copy the original leaves
    generateArrayOfPointers(vec_ptr(old_leaves), vec_ptr(original_basis_ptrs), leaf_size * leaf_rank, num_leaves,
                            stream, hw);
    generateArrayOfPointers(vec_ptr(basis_tree.basis_mem), vec_ptr(new_basis_ptrs), leaf_size * new_rank, num_leaves,
                            stream, hw);

    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copyBlock)(stream, leaf_size, leaf_rank, vec_ptr(new_basis_ptrs),
                                                                  0, 0, leaf_size, vec_ptr(original_basis_ptrs), 0, 0,
                                                                  leaf_size, num_leaves));

    // Generate pointers for the flagged nodes
    fillArray(vec_ptr(ld_dest_array), num_leaves, leaf_size, stream, hw);
    fillArray(vec_ptr(ld_src_array), num_leaves, lda, stream, hw);

    hlru_global_basis_leaf_marshal_batch<H2Opus_Real, hw>(
        vec_ptr(new_basis_ptrs), A, vec_ptr(updated_basis_ptrs), vec_ptr(update_ptrs), vec_ptr(rows_array),
        vec_ptr(cols_array), vec_ptr(basis_tree.node_start), vec_ptr(basis_tree.node_len), leaf_start, leaf_size,
        leaf_rank, update_rank, num_leaves, stream);

    // Copy over the update blocks
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copyBlock)(
        stream, vec_ptr(rows_array), vec_ptr(cols_array), leaf_size, update_rank, vec_ptr(updated_basis_ptrs),
        vec_ptr(ld_dest_array), vec_ptr(update_ptrs), vec_ptr(ld_src_array), num_leaves));
}

template <int hw>
void hlru_sym_global_template(THMatrix<hw> &hmatrix, const H2Opus_Real *U, int ldu, int rank, H2Opus_Real s,
                              h2opusHandle_t handle)
{
    assert(hmatrix.sym == true);

    HNodeTreeLevelData &hnode_level_data = hmatrix.hnodes.level_data;
    BasisTreeLevelData &u_level_data = hmatrix.u_basis_tree.level_data;
    h2opusComputeStream_t main_stream = handle->getMainStream();

    // Update basis tree
    hlru_sym_update_basis_leaves_global<hw>(hmatrix.u_basis_tree, U, ldu, rank, main_stream);
    hlru_update_transfer_matrices_global<hw>(hmatrix.u_basis_tree, rank, main_stream);

    // Update hnodetree
    hlru_update_coupling_matrices_global<hw>(hmatrix.hnodes, s, rank, main_stream);

    hlru_update_dense_blocks_global<hw>(hmatrix.hnodes, hmatrix.u_basis_tree, hmatrix.u_basis_tree, U, ldu, U, ldu,
                                        rank, s, main_stream);

    // Update ranks
    std::vector<int> new_ranks(hmatrix.u_basis_tree.depth);
    for (int i = 0; i < (int)new_ranks.size(); i++)
        new_ranks[i] = u_level_data.getLevelRank(i) + rank;

    u_level_data.setLevelRanks(vec_ptr(new_ranks));
    hnode_level_data.setRankFromBasis(u_level_data, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Interface routines
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void hlru_sym_global(HMatrix &hmatrix, const H2Opus_Real *U, int ldu, int rank, H2Opus_Real s, h2opusHandle_t handle)
{
    hlru_sym_global_template<H2OPUS_HWTYPE_CPU>(hmatrix, U, ldu, rank, s, handle);
}

#ifdef H2OPUS_USE_GPU
void hlru_sym_global(HMatrix_GPU &hmatrix, const H2Opus_Real *U, int ldu, int rank, H2Opus_Real s, h2opusHandle_t handle)
{
    hlru_sym_global_template<H2OPUS_HWTYPE_GPU>(hmatrix, U, ldu, rank, s, handle);
}
#endif
