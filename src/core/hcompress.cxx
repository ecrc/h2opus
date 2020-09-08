#include <h2opus/core/hcompress.h>
#include <h2opus/marshal/hcompress_marshal.h>

#include <h2opus/util/batch_wrappers.h>
#include <h2opus/util/debug_routines.h>
#include <h2opus/util/gpu_err_check.h>
#include <h2opus/util/perf_counter.h>
#include <h2opus/util/thrust_wrappers.h>
#include <h2opus/util/timer.h>

// Leaves are referred to as U here
template <int hw>
int hcompress_compressed_basis_leaf_rank_template(TBasisTree<hw> &basis_tree, H2Opus_Real eps,
                                                  HcompressUpsweepWorkspace &workspace, h2opusComputeStream_t stream)
{
    BasisTreeLevelData &level_data = basis_tree.level_data;
    size_t num_leaves = basis_tree.basis_leaves;
    int leaf_size = basis_tree.leaf_size;
    int leaf_rank = level_data.getLevelRank(level_data.depth - 1);

    H2Opus_Real *U = vec_ptr(basis_tree.basis_mem);
    H2Opus_Real *Z_hat_leaves = workspace.Z_hat[level_data.depth - 1];

    // Calculate the weighted leaves UZ = U * Z'
    H2Opus_Real alpha = 1, beta = 0;
    H2Opus_Real **ptr_U = workspace.ptr_A, **ptr_Z = workspace.ptr_B, **ptr_UZ = workspace.ptr_C;
    H2Opus_Real *UZ_data = workspace.UZ_data;

    generateArrayOfPointers(U, ptr_U, leaf_size * leaf_rank, num_leaves, stream, hw);
    generateArrayOfPointers(Z_hat_leaves, ptr_Z, leaf_rank * leaf_rank, num_leaves, stream, hw);
    generateArrayOfPointers(UZ_data, ptr_UZ, leaf_size * leaf_rank, num_leaves, stream, hw);

    // Clear UZ_data
    fillArray(UZ_data, num_leaves * leaf_size * leaf_rank, 0, stream, hw);

    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
        stream, H2Opus_NoTrans, H2Opus_Trans, leaf_size, leaf_rank, leaf_rank, alpha, (const H2Opus_Real **)ptr_U,
        leaf_size, (const H2Opus_Real **)ptr_Z, leaf_rank, beta, ptr_UZ, leaf_size, num_leaves));

    // Calculate a basis Q for the approximation of UZ using column pivoted QR
    // UZ is overwritten by Q
    H2Opus_Real *tau_data = workspace.tau_data, **ptr_tau = workspace.ptr_D;
    int *ranks_array = workspace.ranks_array;

    fillArray(tau_data, leaf_rank * num_leaves, 0, stream, hw);
    generateArrayOfPointers(tau_data, ptr_tau, leaf_rank, num_leaves, stream, hw);

    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::geqp2)(stream, leaf_size, leaf_rank, UZ_data, leaf_size,
                                                              leaf_size * leaf_rank, tau_data, leaf_rank, ranks_array,
                                                              eps, num_leaves));

    // The new rank for the leaf level is the max rank of all leaves
    return getMaxElement(ranks_array, num_leaves, stream, hw);
}

template <int hw>
void hcompress_truncate_basis_leaves_template(TBasisTree<hw> &basis_tree, int new_rank,
                                              HcompressUpsweepWorkspace &workspace, h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

    BasisTreeLevelData &level_data = basis_tree.level_data;
    size_t num_leaves = basis_tree.basis_leaves;
    int leaf_size = basis_tree.leaf_size;
    int leaf_rank = level_data.getLevelRank(level_data.depth - 1);
    H2Opus_Real *U = vec_ptr(basis_tree.basis_mem);

    // Workspace is assumed to be in a valid state as set by
    // hcompress_compressed_basis_leaf_rank_template
    H2Opus_Real alpha = 1, beta = 0;
    int max_rank = std::min(leaf_rank, leaf_size);

    H2Opus_Real **ptr_U = workspace.ptr_A, **ptr_Z = workspace.ptr_B, **ptr_UZ = workspace.ptr_C;
    H2Opus_Real *UZ_data = workspace.UZ_data;
    H2Opus_Real **ptr_tau = workspace.ptr_D;

    check_kblas_error(
        (H2OpusBatched<H2Opus_Real, hw>::orgqr)(stream, leaf_size, new_rank, ptr_UZ, leaf_size, ptr_tau, num_leaves));

    // Calculate the projection matrices as T = Q^t * U
    // Reuse the pointer array from Z for T (same count)
    H2Opus_Real **ptr_T = ptr_Z, *T_hat_leaves = workspace.T_hat[basis_tree.depth - 1];
    fillArray(T_hat_leaves, leaf_rank * leaf_rank * num_leaves, 0, stream, hw);
    generateArrayOfPointers(T_hat_leaves, ptr_T, leaf_rank * leaf_rank, num_leaves, stream, hw);

    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
        stream, H2Opus_Trans, H2Opus_NoTrans, max_rank, leaf_rank, leaf_size, alpha, (const H2Opus_Real **)ptr_UZ,
        leaf_size, (const H2Opus_Real **)ptr_U, leaf_size, beta, ptr_T, leaf_rank, num_leaves));

    // Now copy over the truncated leaf data
    basis_tree.basis_mem = RealVector(num_leaves * leaf_size * new_rank, 0);
    U = vec_ptr(basis_tree.basis_mem);

    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copyBlock)(stream, leaf_size, new_rank, U, 0, 0, leaf_size,
                                                                  leaf_size * new_rank, UZ_data, 0, 0, leaf_size,
                                                                  leaf_size * leaf_rank, num_leaves));

    // Set the rank for the level
    workspace.new_ranks[basis_tree.depth - 1] = new_rank;
}

template <int hw>
int hcompress_compressed_basis_level_rank_template(TBasisTree<hw> &basis_tree, H2Opus_Real eps, int level,
                                                   HcompressUpsweepWorkspace &workspace, h2opusComputeStream_t stream)
{
    BasisTreeLevelData &level_data = basis_tree.level_data;

    int child_new_rank = workspace.new_ranks[level + 1];
    int max_children = basis_tree.max_children;
    int te_rows = max_children * child_new_rank;

    // Temporary memory for TE = [T_1 E_1; T_2 E_2] and UZ = TE * Z
    H2Opus_Real *TE_data = workspace.TE_data;
    H2Opus_Real *UZ_data = workspace.UZ_data;

    // Temporary memory for the new ranks for the nodes within a level
    int *level_ranks = workspace.ranks_array;

    H2Opus_Real alpha = 1, beta = 0;

    int level_rank = level_data.getLevelRank(level);
    size_t num_nodes = level_data.getLevelSize(level);
    size_t level_start = level_data.getLevelStart(level);

    // Child level info
    int child_rank = level_data.getLevelRank(level + 1);
    size_t num_child_nodes = level_data.getLevelSize(level + 1);
    size_t child_level_start = level_data.getLevelStart(level + 1);

    // Clear out the projection matrix level
    H2Opus_Real *T_hat_level = workspace.T_hat[level];
    fillArray(T_hat_level, num_nodes * level_rank * level_rank, 0, stream, hw);

    if (level_rank == 0 || child_new_rank == 0)
        return 0;

    H2Opus_Real *child_transfer = vec_ptr(basis_tree.trans_mem[level + 1]);
    H2Opus_Real *Z_hat_level = workspace.Z_hat[level];
    H2Opus_Real *T_hat_child_data = workspace.T_hat[level + 1];

    ////////////////////////////////////////////////////////////////
    // Form TE = [T_c1 E_c1; T_c2 E_c2]
    ////////////////////////////////////////////////////////////////
    // Marshal upsweep pointers
    hcompress_upsweep_batch_marshal<H2Opus_Real, hw>(T_hat_child_data, child_transfer, TE_data, workspace.ptr_A,
                                                     workspace.ptr_B, workspace.ptr_C, child_new_rank, child_rank,
                                                     level_rank, child_level_start, level_start, max_children,
                                                     basis_tree.head_ptr(), basis_tree.next_ptr(), num_nodes, stream);

    // Now execute the batch gemm
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
        stream, H2Opus_NoTrans, H2Opus_NoTrans, child_new_rank, level_rank, child_rank, alpha,
        (const H2Opus_Real **)workspace.ptr_A, child_rank, (const H2Opus_Real **)workspace.ptr_B, child_rank, beta,
        workspace.ptr_C, te_rows, num_child_nodes));

    ////////////////////////////////////////////////////////////////
    // Apply the weights of each basis node to the stacked TE nodes - ie UZ = TE * Z'
    ////////////////////////////////////////////////////////////////
    H2Opus_Real **ptr_UZ = workspace.ptr_A, **ptr_TE = workspace.ptr_B, **ptr_Z = workspace.ptr_C;
    generateArrayOfPointers(TE_data, ptr_TE, te_rows * level_rank, num_nodes, stream, hw);
    generateArrayOfPointers(Z_hat_level, ptr_Z, level_rank * level_rank, num_nodes, stream, hw);
    generateArrayOfPointers(UZ_data, ptr_UZ, te_rows * level_rank, num_nodes, stream, hw);

    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
        stream, H2Opus_NoTrans, H2Opus_Trans, te_rows, level_rank, level_rank, alpha, (const H2Opus_Real **)ptr_TE,
        te_rows, (const H2Opus_Real **)ptr_Z, level_rank, beta, ptr_UZ, te_rows, num_nodes));

    // Get new approximate basis Q for UZ - overwrites UZ
    H2Opus_Real *tau_data = workspace.tau_data, **ptr_tau = workspace.ptr_D;
    fillArray(tau_data, level_rank * num_nodes, 0, stream, hw);
    generateArrayOfPointers(tau_data, ptr_tau, level_rank, num_nodes, stream, hw);

    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::geqp2)(stream, te_rows, level_rank, UZ_data, te_rows,
                                                              te_rows * level_rank, tau_data, level_rank, level_ranks,
                                                              eps, num_nodes));

    return getMaxElement(level_ranks, num_nodes, stream, hw);
}

template <int hw>
void hcompress_truncate_basis_level_template(TBasisTree<hw> &basis_tree, int new_rank, int level,
                                             HcompressUpsweepWorkspace &workspace, h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

    if (new_rank == 0)
        return;

    BasisTreeLevelData &level_data = basis_tree.level_data;

    int child_new_rank = workspace.new_ranks[level + 1];
    int max_children = basis_tree.max_children;
    int te_rows = max_children * child_new_rank;

    // Workspace is assumed to be in a valid state as set by
    // hcompress_compressed_basis_level_rank
    H2Opus_Real *UZ_data = workspace.UZ_data;
    H2Opus_Real **ptr_UZ = workspace.ptr_A, **ptr_TE = workspace.ptr_B;
    H2Opus_Real **ptr_tau = workspace.ptr_D;

    H2Opus_Real alpha = 1, beta = 0;

    int level_rank = level_data.getLevelRank(level);
    size_t num_nodes = level_data.getLevelSize(level);
    size_t level_start = level_data.getLevelStart(level);
    int max_rank = std::min(level_rank, te_rows);

    // Child level info
    int child_rank = level_data.getLevelRank(level + 1);
    size_t num_child_nodes = level_data.getLevelSize(level + 1);
    size_t child_level_start = level_data.getLevelStart(level + 1);

    H2Opus_Real *T_hat_level = workspace.T_hat[level];

    check_kblas_error(
        (H2OpusBatched<H2Opus_Real, hw>::orgqr)(stream, te_rows, new_rank, ptr_UZ, te_rows, ptr_tau, num_nodes));

    // Calculate the projection matrices as T = Q^t * TE
    H2Opus_Real **T_ptrs = workspace.ptr_D;
    generateArrayOfPointers(T_hat_level, T_ptrs, level_rank * level_rank, num_nodes, stream, hw);

    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
        stream, H2Opus_Trans, H2Opus_NoTrans, max_rank, level_rank, te_rows, alpha, (const H2Opus_Real **)ptr_UZ,
        te_rows, (const H2Opus_Real **)ptr_TE, te_rows, beta, T_ptrs, level_rank, num_nodes));

    ////////////////////////////////////////////////////////////////
    // Resize original array and copy over the truncated transfer matrices
    ////////////////////////////////////////////////////////////////
    basis_tree.trans_mem[level + 1] = RealVector(num_child_nodes * child_new_rank * new_rank, 0);

    // First marhsal the pointers...
    H2Opus_Real **ptr_E = workspace.ptr_D;
    hcompress_copyBlock_marshal_batch<H2Opus_Real, hw>(
        vec_ptr(basis_tree.trans_mem[level + 1]), UZ_data, ptr_E, ptr_UZ, child_new_rank, level_rank, new_rank,
        child_level_start, level_start, max_children, basis_tree.head_ptr(), basis_tree.next_ptr(), num_nodes, stream);

    // ...and then copy
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copyBlock)(
        stream, child_new_rank, new_rank, ptr_E, 0, 0, child_new_rank, ptr_UZ, 0, 0, te_rows, num_child_nodes));

    workspace.new_ranks[level] = new_rank;
}

template <int hw>
void hcompress_compress_leaves(TBasisTree<hw> &basis_tree, H2Opus_Real eps, HcompressUpsweepWorkspace &workspace,
                               h2opusComputeStream_t stream)
{
    int new_rank = hcompress_compressed_basis_leaf_rank_template<hw>(basis_tree, eps, workspace, stream);
    hcompress_truncate_basis_leaves_template<hw>(basis_tree, new_rank, workspace, stream);
}

template <int hw>
void hcompress_truncate_basis(TBasisTree<hw> &basis_tree, H2Opus_Real eps, HcompressUpsweepWorkspace &workspace,
                              h2opusComputeStream_t stream)
{
    BasisTreeLevelData &level_data = basis_tree.level_data;

    int num_levels = basis_tree.depth;
    int stop_level = level_data.nested_root_level;

    // Compress the leaves
    hcompress_compress_leaves<hw>(basis_tree, eps, workspace, stream);

    // Now sweep up the tree
    for (int level = num_levels - 2; level >= stop_level; level--)
    {
        int level_new_rank =
            hcompress_compressed_basis_level_rank_template<hw>(basis_tree, eps, level, workspace, stream);
        hcompress_truncate_basis_level_template<hw>(basis_tree, level_new_rank, level, workspace, stream);
    }

    // Copy over the ranks above the top level
    for (int i = 0; i < stop_level; i++)
        workspace.new_ranks[i] = level_data.getLevelRank(i);
}

template <int hw>
void hcompress_appyly_weight_packet(TBasisTree<hw> &basis_tree, HcompressUpsweepWorkspace &workspace,
                                    TWeightAccelerationPacket<hw> &weight_packet, h2opusComputeStream_t stream)
{
    BasisTreeLevelData &basis_level_data = basis_tree.level_data;
    int top_level = basis_level_data.nested_root_level;
    assert(top_level >= weight_packet.level);

    int level_rank = basis_level_data.getLevelRank(weight_packet.level);
    H2Opus_Real *Z_hat_level = workspace.Z_hat[weight_packet.level];

    if (weight_packet.rank != level_rank && weight_packet.level == 0)
    {
        size_t level_nodes = basis_level_data.getLevelSize(0);
        fillArray(Z_hat_level, level_nodes * level_rank * level_rank, 0, stream, hw);
    }
    else
    {
        assert(weight_packet.rank == level_rank);

        size_t level_nodes = basis_level_data.getLevelSize(weight_packet.level);
        size_t num_entries = level_nodes * level_rank * level_rank;

        assert(num_entries == weight_packet.Z_level.size());

        copyArray(vec_ptr(weight_packet.Z_level), Z_hat_level, num_entries, stream, hw);
    }
}

template <int hw>
void hcompress_generate_optimal_basis_template(THNodeTree<hw> &hnodes, THNodeTree<hw> *offdiagonal_hnodes,
                                               BSNPointerDirection direction, TBasisTree<hw> &basis_tree,
                                               HcompressUpsweepWorkspace &upsweep_workspace,
                                               HcompressOptimalBGenWorkspace &bgen_workspace, int start_level,
                                               h2opusComputeStream_t stream)
{
    typedef typename THNodeTree<hw>::HNodeTreeBSNData BSNData;
    // typedef typename VectorContainer<hw, H2Opus_Real>::type  RealVector;

    HNodeTreeLevelData &hnode_level_data = hnodes.level_data;
    BasisTreeLevelData &basis_level_data = basis_tree.level_data;
    BSNData &bsn_data = (direction == BSN_DIRECTION_COLUMN ? hnodes.bsn_col_data : hnodes.bsn_row_data);
    BSNData *offdiag_bsn_data = NULL;
    if (offdiagonal_hnodes)
        offdiag_bsn_data = (direction == BSN_DIRECTION_COLUMN ? &(offdiagonal_hnodes->bsn_col_data)
                                                              : &(offdiagonal_hnodes->bsn_row_data));

    int depth = hnode_level_data.depth;

    std::vector<H2Opus_Real *> &Z_hat = upsweep_workspace.Z_hat;
    std::vector<int> &max_nodes = bsn_data.max_nodes;

    // Temporary buffers to hold node data and tau for the QR
    H2Opus_Real *stacked_node_data = bgen_workspace.stacked_node_data;
    H2Opus_Real *stacked_tau_data = bgen_workspace.stacked_tau_data;

    H2Opus_Real **stacked_node_ptrs = bgen_workspace.stacked_node_ptrs;
    H2Opus_Real **stacked_node_tau_ptrs = bgen_workspace.stacked_node_tau_ptrs;

    int *stacked_node_row = bgen_workspace.stacked_node_row;
    int *stacked_node_col = bgen_workspace.stacked_node_col;
    int *stacked_node_ld = bgen_workspace.stacked_node_ld;

    // Pointers for GEMMS and transpose
    H2Opus_Real **ptr_ZE = bgen_workspace.ptr_ZE, **ptr_Z = bgen_workspace.ptr_Z;
    H2Opus_Real **ptr_E = bgen_workspace.ptr_E;
    H2Opus_Real **ptr_row_data = bgen_workspace.ptr_row_data, **ptr_S = bgen_workspace.ptr_S;

    // Now go through each level and calculate the weights
    for (int level = start_level + 1; level < depth; level++)
    {
        int level_rank = hnode_level_data.getLevelRank(level);
        int parent_rank = hnode_level_data.getLevelRank(level - 1);
        size_t level_rows = basis_level_data.getLevelSize(level);
        size_t level_start = basis_level_data.getLevelStart(level);
        size_t parent_level_start = basis_level_data.getLevelStart(level - 1);
        size_t coupling_level_start = hnode_level_data.getCouplingLevelStart(level);
        size_t offdiag_coupling_level_start = 0;

        int ld_ZE = parent_rank + level_rank * max_nodes[level];
        if (offdiagonal_hnodes)
        {
            offdiag_coupling_level_start = offdiagonal_hnodes->level_data.getCouplingLevelStart(level);
            ld_ZE += level_rank * offdiag_bsn_data->max_nodes[level];
        }

        H2Opus_Real *Z_hat_level = Z_hat[level];
        H2Opus_Real *Z_hat_parent_level = Z_hat[level - 1];
        H2Opus_Real *transfer_level = vec_ptr(basis_tree.trans_mem[level]);

        fillArray(Z_hat_level, level_rows * level_rank * level_rank, 0, stream, hw);

        if (level_rank == 0 || ld_ZE == 0)
            continue;

        std::vector<int> &cached_node_ptrs = bsn_data.cached_coupling_ptrs[level];

        size_t start_index = 0;
        size_t increment = std::min((size_t)COMPRESSION_BASIS_GEN_MAX_NODES, level_rows);

        fillArray(stacked_node_ld, increment, ld_ZE, stream, hw);

        generateArrayOfPointers(stacked_node_data, stacked_node_ptrs, ld_ZE * level_rank, increment, stream, hw);
        generateArrayOfPointers(stacked_tau_data, stacked_node_tau_ptrs, level_rank, increment, stream, hw);

        size_t batch_index = 0;
        while (start_index != level_rows)
        {
            size_t batch_size = std::min(increment, level_rows - start_index);

            // Clear the scratch memory that we're going to use
            fillArray(stacked_node_data, ld_ZE * level_rank * batch_size, 0, stream, hw);
            fillArray(stacked_tau_data, level_rank * batch_size, 0, stream, hw);

            ////////////////////////////////////////////////////////////////////////////////
            // For the stacked matrices for each basis node t: SN_t = [Z_t+ E^T_t; S_t1^T .... S_tn^T]
            // where S_ti are all the coupling matrices corresponding to t
            // and Z^+_t is the parent node in Z_hat corresponding to the parent of t
            ////////////////////////////////////////////////////////////////////////////////
            // The first block of the stacked matrix data is the weighted parent contribution ZE^t
            hcompress_parent_weight_batch_marshal<H2Opus_Real, hw>(
                ptr_ZE, stacked_node_data, ptr_Z, Z_hat_parent_level, ptr_E, transfer_level, basis_tree.parent_ptr(),
                parent_level_start, level_start, start_index, ld_ZE, parent_rank, level_rank, batch_size, stream);

            check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
                stream, H2Opus_NoTrans, H2Opus_Trans, parent_rank, level_rank, parent_rank, (H2Opus_Real)1,
                (const H2Opus_Real **)ptr_Z, parent_rank, (const H2Opus_Real **)ptr_E, level_rank, (H2Opus_Real)0,
                ptr_ZE, ld_ZE, batch_size));

            // The blocks after that are the transpose of the coupling matrices within a block row
            // First marshal the coupling nodes for each row
            H2Opus_Real *coupling_level = vec_ptr(hnodes.rank_leaf_mem[level]);
            int *coupling_ptrs = vec_ptr(bsn_data.coupling_ptrs[level]);
            int *coupling_indexes = vec_ptr(bsn_data.coupling_node_indexes[level]);
            int *coupling_node_to_leaf = vec_ptr(hnodes.node_to_leaf);

            int coupling_start = cached_node_ptrs[batch_index], coupling_end = cached_node_ptrs[batch_index + 1];

            H2Opus_Real *offdiag_coupling_level = NULL;
            int *offdiag_coupling_ptrs = NULL, *offdiag_coupling_indexes = NULL;
            int *offdiag_coupling_node_to_leaf = NULL;
            int offdiag_coupling_start = 0, offdiag_coupling_end = 0;
            if (offdiagonal_hnodes)
            {
                offdiag_coupling_level = vec_ptr(offdiagonal_hnodes->rank_leaf_mem[level]);
                offdiag_coupling_ptrs = vec_ptr(offdiag_bsn_data->coupling_ptrs[level]);
                offdiag_coupling_indexes = vec_ptr(offdiag_bsn_data->coupling_node_indexes[level]);
                offdiag_coupling_node_to_leaf = vec_ptr(offdiagonal_hnodes->node_to_leaf);
                offdiag_coupling_start = offdiag_bsn_data->cached_coupling_ptrs[level][batch_index];
                offdiag_coupling_end = offdiag_bsn_data->cached_coupling_ptrs[level][batch_index + 1];
            }

            hcompress_stack_coupling_data_batch_marshal<H2Opus_Real, hw>(
                ptr_row_data, stacked_node_data, ptr_S, coupling_level, coupling_ptrs, coupling_indexes,
                offdiag_coupling_level, offdiag_coupling_ptrs, offdiag_coupling_indexes, coupling_node_to_leaf,
                coupling_level_start, offdiag_coupling_node_to_leaf, offdiag_coupling_level_start, stacked_node_row,
                stacked_node_col, parent_rank, level_rank, ld_ZE, start_index, coupling_start, coupling_end,
                offdiag_coupling_start, offdiag_coupling_end, batch_size, stream);

            // Now stack them up using the batch transpose
            int total_coupling_blocks = coupling_end - coupling_start + offdiag_coupling_end - offdiag_coupling_start;

            check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::transpose)(
                stream, level_rank, level_rank, ptr_S, level_rank, ptr_row_data, ld_ZE, total_coupling_blocks));

            ////////////////////////////////////////////////////////////////////////////////
            // Now we can form the Z nodes of the current level by taking the triangular factor
            // of the QR decomposition of SN: [~, Z_t] = qr(SN_t);
            ////////////////////////////////////////////////////////////////////////////////
            check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::tsqrf)(stream, stacked_node_row, stacked_node_col, ld_ZE,
                                                                      level_rank, stacked_node_ptrs, stacked_node_ld,
                                                                      stacked_node_tau_ptrs, batch_size));

            check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copy_upper)(
                stream, ld_ZE, level_rank, stacked_node_data, ld_ZE, ld_ZE * level_rank, Z_hat_level, level_rank,
                level_rank * level_rank, batch_size));

            start_index += batch_size;
            Z_hat_level += batch_size * level_rank * level_rank;
            batch_index++;
        }
    }
}

template <int hw>
void hcompress_project_top_level(TBasisTree<hw> &basis_tree, HcompressUpsweepWorkspace &upsweep_ws,
                                 HcompressProjectTopLevelWorkspace &top_level_ws, h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

    BasisTreeLevelData &level_data = basis_tree.level_data;
    int top_level = level_data.nested_root_level;
    if (top_level == 0)
        return;

    int level_truncated_rank = upsweep_ws.new_ranks[top_level];
    int level_rank = level_data.getLevelRank(top_level);
    int parent_rank = level_data.getLevelRank(top_level - 1);
    size_t num_nodes = level_data.getLevelSize(top_level);

    if (num_nodes == 0 || parent_rank == 0 || level_rank == 0 || level_truncated_rank == 0)
        return;

    H2Opus_Real *T_hat_level = upsweep_ws.T_hat[top_level];
    H2Opus_Real *old_level = top_level_ws.old_transfer;
    copyArray(vec_ptr(basis_tree.trans_mem[top_level]), old_level, basis_tree.trans_mem[top_level].size(), stream, hw);
    basis_tree.trans_mem[top_level] = RealVector(num_nodes * level_truncated_rank * parent_rank);

    H2Opus_Real *truncated_level = vec_ptr(basis_tree.trans_mem[top_level]);
    H2Opus_Real **T_ptrs = top_level_ws.ptr_A, **E_ptrs = top_level_ws.ptr_B, **TE_ptrs = top_level_ws.ptr_C;

    generateArrayOfPointers(T_hat_level, T_ptrs, level_rank * level_rank, num_nodes, stream, hw);
    generateArrayOfPointers(old_level, E_ptrs, level_rank * parent_rank, num_nodes, stream, hw);
    generateArrayOfPointers(truncated_level, TE_ptrs, level_truncated_rank * parent_rank, num_nodes, stream, hw);

    H2Opus_Real alpha = 1, beta = 0;

    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
        stream, H2Opus_NoTrans, H2Opus_NoTrans, level_truncated_rank, parent_rank, level_rank, alpha,
        (const H2Opus_Real **)T_ptrs, level_rank, (const H2Opus_Real **)E_ptrs, level_rank, beta, TE_ptrs,
        level_truncated_rank, num_nodes));

    // The basis is now completely nested again
    level_data.nested_root_level = 0;
}

template <int hw>
void hcompress_project_coupling_template(THNodeTree<hw> &hnodes, TBasisTree<hw> &u_basis_tree,
                                         TBasisTree<hw> &v_basis_tree, HcompressUpsweepWorkspace &u_upsweep_ws,
                                         HcompressUpsweepWorkspace &v_upsweep_ws, HcompressProjectionWorkspace &proj_ws,
                                         h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

    BasisTreeLevelData &u_level_data = u_basis_tree.level_data;
    BasisTreeLevelData &v_level_data = v_basis_tree.level_data;
    std::vector<int> &u_new_ranks = u_upsweep_ws.new_ranks;
    // std::vector<int>& v_new_ranks = v_upsweep_ws.new_ranks;
    HNodeTreeLevelData &hnode_level_data = hnodes.level_data;

    assert(u_level_data.nested_root_level == v_level_data.nested_root_level);

    int num_levels = hnodes.depth;
    int top_level = u_level_data.nested_root_level;

    H2Opus_Real **TC_array = proj_ws.ptr_A, **C_array = proj_ws.ptr_B;
    H2Opus_Real **Tu_array = proj_ws.ptr_C, **Tv_array = proj_ws.ptr_D;
    H2Opus_Real *TC = proj_ws.TC;

    H2Opus_Real alpha = 1, beta = 0;

    // Now go through the levels of the tree and compute the projection
    // of the coupling matrices into the new basis
    for (int level = num_levels - 1; level >= top_level; level--)
    {
        int level_rank = hnode_level_data.getLevelRank(level);
        int projection_rank = u_new_ranks[level];
        size_t u_level_start = u_level_data.getLevelStart(level);
        size_t v_level_start = v_level_data.getLevelStart(level);
        size_t level_nodes = hnode_level_data.getCouplingLevelSize(level);

        if (level_nodes == 0)
            continue;

        H2Opus_Real *Tu_level = u_upsweep_ws.T_hat[level];
        H2Opus_Real *Tv_level = v_upsweep_ws.T_hat[level];
        H2Opus_Real *C_level = vec_ptr(hnodes.rank_leaf_mem[level]);

        // Generate an array of pointers so that we can use the cublas batch gemm routines
        generateArrayOfPointers(TC, TC_array, projection_rank * level_rank, level_nodes, stream, hw);
        generateArrayOfPointers(C_level, C_array, level_rank * level_rank, level_nodes, stream, hw);

        size_t coupling_start = hnode_level_data.getCouplingLevelStart(level);

        hcompress_project_batch_marshal<H2Opus_Real, hw>(
            vec_ptr(hnodes.rank_leaf_tree_index), vec_ptr(hnodes.node_u_index), Tu_level, Tu_array,
            level_rank * level_rank, coupling_start, u_level_start, level_nodes, stream);
        hcompress_project_batch_marshal<H2Opus_Real, hw>(
            vec_ptr(hnodes.rank_leaf_tree_index), vec_ptr(hnodes.node_v_index), Tv_level, Tv_array,
            level_rank * level_rank, coupling_start, v_level_start, level_nodes, stream);

        // First calculate TC_{ts} = Tu_{t} C_{ts}
        check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
            stream, H2Opus_NoTrans, H2Opus_NoTrans, projection_rank, level_rank, level_rank, alpha,
            (const H2Opus_Real **)Tu_array, level_rank, (const H2Opus_Real **)C_array, level_rank, beta, TC_array,
            projection_rank, level_nodes));

        // Now we can resize the coupling matrix data to hold the new projected nodes
        hnodes.rank_leaf_mem[level] = RealVector(projection_rank * projection_rank * level_nodes);

        // Regenerate the level pointers
        C_level = vec_ptr(hnodes.rank_leaf_mem[level]);
        generateArrayOfPointers(C_level, C_array, projection_rank * projection_rank, level_nodes, stream, hw);

        // Now calculate C_{ts} = TC_{ts} * Pv_{t}^t
        check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
            stream, H2Opus_NoTrans, H2Opus_Trans, projection_rank, projection_rank, level_rank, alpha,
            (const H2Opus_Real **)TC_array, projection_rank, (const H2Opus_Real **)Tv_array, level_rank, beta, C_array,
            projection_rank, level_nodes));
    }
}

template <int hw>
void hcompress_template(THMatrix<hw> &hmatrix, TWeightAccelerationPacket<hw> &weight_packet, H2Opus_Real eps,
                        h2opusHandle_t h2opus_handle)
{
    // Only symmetric matrices for now
    assert(hmatrix.sym);

    H2OpusWorkspaceState ws_needed = hcompress_workspace(hmatrix);
    H2OpusWorkspaceState ws_allocated = h2opus_handle->getWorkspaceState();

    if (ws_allocated < ws_needed)
    {
        // printf("Insufficient workspace for hcompress...allocating...");
        h2opus_handle->setWorkspaceState(ws_needed);
        // printf("done.\n");
    }

    BasisTreeLevelData &u_level_data = hmatrix.u_basis_tree.level_data;
    HNodeTreeLevelData &hnode_level_data = hmatrix.hnodes.level_data;

    h2opusComputeStream_t main_stream = h2opus_handle->getMainStream();
    HcompressWorkspace workspace;
    hcompress_get_workspace(hmatrix, workspace, h2opus_handle);

#ifdef H2OPUS_PROFILING_ENABLED
    Timer<hw> timer;
    timer.init();

    H2Opus_Real gather_gflops, project_gflops, trunc_gflops, stitch_gflops;
    H2Opus_Real gather_time, project_time, trunc_time, stitch_time;
    PerformanceCounter::clearCounters();
#endif

    ////////////////////////////////////////////////////////////////////////////////
    // Gather weights for the basis nodes from the matrix coupling data
    ////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_PROFILING_ENABLED
    timer.start();
#endif
    // Apply the weight acceleration packet to insert its Zhat level into the the Zhat tree
    hcompress_appyly_weight_packet<hw>(hmatrix.u_basis_tree, workspace.u_upsweep, weight_packet, main_stream);

    hcompress_generate_optimal_basis_template<hw>(hmatrix.hnodes, NULL, BSN_DIRECTION_ROW, hmatrix.u_basis_tree,
                                                  workspace.u_upsweep, workspace.optimal_bgen, weight_packet.level,
                                                  main_stream);

    // dumpMatrixTreeContainer(u_level_data, workspace.u_upsweep.Z_hat, 4, hw);

#ifdef H2OPUS_PROFILING_ENABLED
    gather_time = timer.stop();
    gather_gflops = PerformanceCounter::getOpCount(PerformanceCounter::GEMM) +
                    PerformanceCounter::getOpCount(PerformanceCounter::QR);
    PerformanceCounter::clearCounters();
    HLibProfile::addRun(HLibProfile::HCOMPRESS_BASIS_GEN, gather_gflops, gather_time);
#endif

    ////////////////////////////////////////////////////////////////////////////////
    // First truncate the basis
    ////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_PROFILING_ENABLED
    timer.start();
#endif
    hcompress_truncate_basis<hw>(hmatrix.u_basis_tree, eps, workspace.u_upsweep, main_stream);

    // dumpMatrixTreeContainer(u_level_data, workspace.u_upsweep.T_hat, 4, hw);

#ifdef H2OPUS_PROFILING_ENABLED
    trunc_time = timer.stop();
    trunc_gflops = PerformanceCounter::getOpCount(PerformanceCounter::GEMM) +
                   PerformanceCounter::getOpCount(PerformanceCounter::QR) +
                   PerformanceCounter::getOpCount(PerformanceCounter::SVD);
    PerformanceCounter::clearCounters();
    HLibProfile::addRun(HLibProfile::HCOMPRESS_TRUNCATE_BASIS, trunc_gflops, trunc_time);
#endif

    ////////////////////////////////////////////////////////////////////////////////
    // Now use the truncated projection trees to project the coupling matrices
    ////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_PROFILING_ENABLED
    timer.start();
#endif
    hcompress_project_coupling_template<hw>(hmatrix.hnodes, hmatrix.u_basis_tree, hmatrix.u_basis_tree,
                                            workspace.u_upsweep, workspace.u_upsweep, workspace.projection,
                                            main_stream);
#ifdef H2OPUS_PROFILING_ENABLED
    project_time = timer.stop();
    project_gflops = PerformanceCounter::getOpCount(PerformanceCounter::GEMM);
    PerformanceCounter::clearCounters();
    HLibProfile::addRun(HLibProfile::HCOMPRESS_PROJECTION, project_gflops, project_time);
#endif

    ////////////////////////////////////////////////////////////////////////////////
    // Update the weight packet to include the Z_hat level that was not affected in
    // this truncation, i.e. to the level = nested_root_level - 1
    ////////////////////////////////////////////////////////////////////////////////
    std::vector<int> &u_new_ranks = workspace.u_upsweep.new_ranks;

    int acc_level = u_level_data.nested_root_level - 1;
    if (acc_level < 0)
        acc_level = 0;
    int acc_node_rank = u_level_data.getLevelRank(acc_level);
    size_t acc_level_nodes = u_level_data.getLevelSize(acc_level);
    size_t acc_level_entries = acc_level_nodes * acc_node_rank * acc_node_rank;
    weight_packet.Z_level.resize(acc_level_entries);
    copyArray(workspace.u_upsweep.Z_hat[acc_level], vec_ptr(weight_packet.Z_level), acc_level_entries, main_stream, hw);
    weight_packet.level = acc_level;
    weight_packet.rank = u_new_ranks[acc_level];

    ////////////////////////////////////////////////////////////////////////////////
    // Project the top level to make sure the basis is nested again
    ////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_PROFILING_ENABLED
    timer.start();
#endif
    hcompress_project_top_level<hw>(hmatrix.u_basis_tree, workspace.u_upsweep, workspace.u_top_level, main_stream);
#ifdef H2OPUS_PROFILING_ENABLED
    stitch_time = timer.stop();
    stitch_gflops = PerformanceCounter::getOpCount(PerformanceCounter::GEMM);
    PerformanceCounter::clearCounters();
    HLibProfile::addRun(HLibProfile::HCOMPRESS_STITCH, stitch_gflops, stitch_time);
#endif

    ////////////////////////////////////////////////////////////////////////////////
    // Update the level data
    ////////////////////////////////////////////////////////////////////////////////
    // printf("Level nodes\tOld_rank\tNew_rank\n");
    // for(int i = u_level_data.depth - 1; i >= 0; i--)
    // 	printf("%d\t\t%d\t\t%d\n", hnode_level_data.getCouplingLevelSize(i), hnode_level_data.getLevelRank(i),
    // u_new_ranks[i]); printf("\n");

    u_level_data.setLevelRanks(vec_ptr(u_new_ranks));
    hnode_level_data.setRankFromBasis(u_level_data, 0);
}

template <int hw> void hcompress_template(THMatrix<hw> &hmatrix, H2Opus_Real eps, h2opusHandle_t h2opus_handle)
{
    TWeightAccelerationPacket<hw> packet(0, 0, 0);
    hcompress_template(hmatrix, packet, eps, h2opus_handle);
}

////////////////////////////////////////////////////////////////////////////////
// Interface routines
////////////////////////////////////////////////////////////////////////////////
void hcompress_generate_optimal_basis(HNodeTree &hnodes, HNodeTree *offdiagonal_hnodes, BSNPointerDirection direction,
                                      BasisTree &basis_tree, HcompressUpsweepWorkspace &upsweep_workspace,
                                      HcompressOptimalBGenWorkspace &bgen_workspace, int start_level,
                                      h2opusComputeStream_t stream)
{
    hcompress_generate_optimal_basis_template<H2OPUS_HWTYPE_CPU>(
        hnodes, offdiagonal_hnodes, direction, basis_tree, upsweep_workspace, bgen_workspace, start_level, stream);
}

int hcompress_compressed_basis_leaf_rank(BasisTree &basis_tree, H2Opus_Real eps, HcompressUpsweepWorkspace &workspace,
                                         h2opusComputeStream_t stream)
{
    return hcompress_compressed_basis_leaf_rank_template<H2OPUS_HWTYPE_CPU>(basis_tree, eps, workspace, stream);
}

void hcompress_truncate_basis_leaves(BasisTree &basis_tree, int new_rank, HcompressUpsweepWorkspace &workspace,
                                     h2opusComputeStream_t stream)
{
    hcompress_truncate_basis_leaves_template<H2OPUS_HWTYPE_CPU>(basis_tree, new_rank, workspace, stream);
}

void hcompress_truncate_basis_level(BasisTree &basis_tree, int new_rank, int level,
                                    HcompressUpsweepWorkspace &workspace, h2opusComputeStream_t stream)
{
    hcompress_truncate_basis_level_template<H2OPUS_HWTYPE_CPU>(basis_tree, new_rank, level, workspace, stream);
}

int hcompress_compressed_basis_level_rank(BasisTree &basis_tree, H2Opus_Real eps, int level,
                                          HcompressUpsweepWorkspace &workspace, h2opusComputeStream_t stream)
{
    return hcompress_compressed_basis_level_rank_template<H2OPUS_HWTYPE_CPU>(basis_tree, eps, level, workspace, stream);
}

void hcompress_project_coupling(HNodeTree &hnodes, BasisTree &u_basis_tree, BasisTree &v_basis_tree,
                                HcompressUpsweepWorkspace &u_upsweep_ws, HcompressUpsweepWorkspace &v_upsweep_ws,
                                HcompressProjectionWorkspace &proj_ws, h2opusComputeStream_t stream)
{
    hcompress_project_coupling_template<H2OPUS_HWTYPE_CPU>(hnodes, u_basis_tree, v_basis_tree, u_upsweep_ws,
                                                           v_upsweep_ws, proj_ws, stream);
}

void hcompress(HMatrix &hmatrix, H2Opus_Real eps, h2opusHandle_t h2opus_handle)
{
    hcompress_template<H2OPUS_HWTYPE_CPU>(hmatrix, eps, h2opus_handle);
}

void hcompress(HMatrix &hmatrix, WeightAccelerationPacket &packet, H2Opus_Real eps, h2opusHandle_t h2opus_handle)
{
    hcompress_template<H2OPUS_HWTYPE_CPU>(hmatrix, packet, eps, h2opus_handle);
}

#ifdef H2OPUS_USE_GPU
void hcompress_generate_optimal_basis(HNodeTree_GPU &hnodes, HNodeTree_GPU *offdiagonal_hnodes,
                                      BSNPointerDirection direction, BasisTree_GPU &basis_tree,
                                      HcompressUpsweepWorkspace &upsweep_workspace,
                                      HcompressOptimalBGenWorkspace &bgen_workspace, int start_level,
                                      h2opusComputeStream_t stream)
{
    hcompress_generate_optimal_basis_template<H2OPUS_HWTYPE_GPU>(
        hnodes, offdiagonal_hnodes, direction, basis_tree, upsweep_workspace, bgen_workspace, start_level, stream);
}

int hcompress_compressed_basis_leaf_rank(BasisTree_GPU &basis_tree, H2Opus_Real eps,
                                         HcompressUpsweepWorkspace &workspace, h2opusComputeStream_t stream)
{
    return hcompress_compressed_basis_leaf_rank_template<H2OPUS_HWTYPE_GPU>(basis_tree, eps, workspace, stream);
}

void hcompress_truncate_basis_leaves(BasisTree_GPU &basis_tree, int new_rank, HcompressUpsweepWorkspace &workspace,
                                     h2opusComputeStream_t stream)
{
    hcompress_truncate_basis_leaves_template<H2OPUS_HWTYPE_GPU>(basis_tree, new_rank, workspace, stream);
}

void hcompress_truncate_basis_level(BasisTree_GPU &basis_tree, int new_rank, int level,
                                    HcompressUpsweepWorkspace &workspace, h2opusComputeStream_t stream)
{
    hcompress_truncate_basis_level_template<H2OPUS_HWTYPE_GPU>(basis_tree, new_rank, level, workspace, stream);
}

int hcompress_compressed_basis_level_rank(BasisTree_GPU &basis_tree, H2Opus_Real eps, int level,
                                          HcompressUpsweepWorkspace &workspace, h2opusComputeStream_t stream)
{
    return hcompress_compressed_basis_level_rank_template<H2OPUS_HWTYPE_GPU>(basis_tree, eps, level, workspace, stream);
}

void hcompress_project_coupling(HNodeTree_GPU &hnodes, BasisTree_GPU &u_basis_tree, BasisTree_GPU &v_basis_tree,
                                HcompressUpsweepWorkspace &u_upsweep_ws, HcompressUpsweepWorkspace &v_upsweep_ws,
                                HcompressProjectionWorkspace &proj_ws, h2opusComputeStream_t stream)
{
    hcompress_project_coupling_template<H2OPUS_HWTYPE_GPU>(hnodes, u_basis_tree, v_basis_tree, u_upsweep_ws,
                                                           v_upsweep_ws, proj_ws, stream);
}

void hcompress(HMatrix_GPU &hmatrix, H2Opus_Real eps, h2opusHandle_t h2opus_handle)
{
    hcompress_template<H2OPUS_HWTYPE_GPU>(hmatrix, eps, h2opus_handle);
}

void hcompress(HMatrix_GPU &hmatrix, WeightAccelerationPacket_GPU &packet, H2Opus_Real eps,
               h2opusHandle_t h2opus_handle)
{
    hcompress_template<H2OPUS_HWTYPE_GPU>(hmatrix, packet, eps, h2opus_handle);
}
#endif
