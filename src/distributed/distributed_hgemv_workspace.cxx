#include <h2opus/distributed/distributed_hgemv_workspace.h>

//////////////////////////////////////////////
// Utility routines
//////////////////////////////////////////////
template <int hw>
inline void getOffDiagonalMarshalNodes(TDistributedHMatrix<hw> &dist_hmatrix, size_t &dense_nodes, size_t &max_nodes)
{
    TDistributedCompressedBSNData<hw> &bsn_data = dist_hmatrix.compressed_basis_tree_data.dense_compressed_bsn_data;
    dense_nodes = bsn_data.send_process_nodes.size();

    max_nodes = dense_nodes;
    for (int level = 0; level < dist_hmatrix.basis_tree.basis_branch.depth; level++)
    {
        TDistributedCompressedBSNData<hw> &bsn_data =
            dist_hmatrix.compressed_basis_tree_data.coupling_compressed_bsn_data[level];
        size_t level_nodes = bsn_data.send_process_nodes.size();
        max_nodes = std::max(level_nodes, max_nodes);
    }
}

//////////////////////////////////////////////
// Templates
//////////////////////////////////////////////
template <int hw>
void distributed_hgemv_workspace_template(TDistributedHMatrix<hw> &dist_hmatrix, int num_vectors,
                                          H2OpusWorkspaceState &branch_ws_needed,
                                          H2OpusWorkspaceState &top_level_ws_needed,
                                          distributedH2OpusHandle_t dist_h2opus_handle)
{
    // See how much workspace is needed for the diagonal block
    BasisTreeLevelData &branch_level_data = dist_hmatrix.basis_tree.basis_branch.level_data;
    HNodeTreeLevelData &hnode_level_data = dist_hmatrix.hnodes.diagonal_block.level_data;
    HNodeTreeLevelData &offdiag_hnode_level_data = dist_hmatrix.hnodes.off_diagonal_blocks.level_data;

    int dense_nodes = dist_hmatrix.hnodes.diagonal_block.num_dense_leaves;
    int coupling_nodes = dist_hmatrix.hnodes.diagonal_block.num_rank_leaves;

    // We don't process the diagonal and offdiagonal concurrently, so the dense and coupling
    // nodes needed in the workspace is the max of the two
    dense_nodes = std::max(dense_nodes, dist_hmatrix.hnodes.off_diagonal_blocks.num_dense_leaves);
    coupling_nodes = std::max(coupling_nodes, dist_hmatrix.hnodes.off_diagonal_blocks.num_rank_leaves);

    branch_ws_needed = hgemv_workspace(branch_level_data, branch_level_data, hnode_level_data,
                                       &offdiag_hnode_level_data, dense_nodes, coupling_nodes, num_vectors, hw);

    //////////////////////////////////////////////////////////////////
    // Additional memory allocations
    //////////////////////////////////////////////////////////////////
    // Align the workspace to reals
    size_t needed_data_bytes, needed_ptr_bytes;
    branch_ws_needed.getBytes(needed_data_bytes, needed_ptr_bytes, hw);
    needed_data_bytes += (needed_data_bytes % sizeof(H2Opus_Real));

    // Add memory for the top level yhat scatter
    needed_data_bytes += branch_level_data.getLevelRank(0) * num_vectors * sizeof(H2Opus_Real);

    // Align to integers
    needed_data_bytes += (needed_data_bytes % sizeof(int));

    // Add memory for the marshaling of the offdiagonal blocks
    size_t dense_offdiag_nodes, max_offdiag_nodes;
    getOffDiagonalMarshalNodes<hw>(dist_hmatrix, dense_offdiag_nodes, max_offdiag_nodes);
    needed_data_bytes += 4 * dense_offdiag_nodes * sizeof(int);

    // Add memory for the marshaling of the pointers for the block copies
    needed_ptr_bytes += 2 * max_offdiag_nodes * sizeof(H2Opus_Real *);

    branch_ws_needed.setBytes(needed_data_bytes, needed_ptr_bytes, hw);
    //////////////////////////////////////////////////////////////////

    // Get the workspace needed for the top level, which is only on the master process
    if (dist_h2opus_handle->rank == 0)
    {
        BasisTreeLevelData &basis_top_level_data = dist_hmatrix.basis_tree.top_level.level_data;
        HNodeTreeLevelData &hnode_top_level_data = dist_hmatrix.hnodes.top_level.level_data;

        int dense_nodes = dist_hmatrix.hnodes.top_level.num_dense_leaves;
        int coupling_nodes = dist_hmatrix.hnodes.top_level.num_rank_leaves;

        top_level_ws_needed = hgemv_workspace(basis_top_level_data, basis_top_level_data, hnode_top_level_data,
                                              dense_nodes, coupling_nodes, num_vectors, hw);
    }
}

template <int hw>
void distributed_hgemv_get_workspace_template(TDistributedHMatrix<hw> &dist_hmatrix, int num_vectors,
                                              DistributedHgemvWorkspace &dist_workspace,
                                              distributedH2OpusHandle_t dist_h2opus_handle)
{
    // See how much workspace is needed for the regular hgemv
    h2opusHandle_t h2opus_handle = dist_h2opus_handle->handle;
    h2opusComputeStream_t stream = h2opus_handle->getMainStream();
    h2opusWorkspace_t h2opus_ws = h2opus_handle->getWorkspace();

    BasisTreeLevelData &branch_level_data = dist_hmatrix.basis_tree.basis_branch.level_data;
    HNodeTreeLevelData &hnode_level_data = dist_hmatrix.hnodes.diagonal_block.level_data;
    HNodeTreeLevelData &offdiag_hnode_level_data = dist_hmatrix.hnodes.off_diagonal_blocks.level_data;

    int dense_nodes = dist_hmatrix.hnodes.diagonal_block.num_dense_leaves;
    int coupling_nodes = dist_hmatrix.hnodes.diagonal_block.num_rank_leaves;

    // We don't process the diagonal and offdiagonal concurrently, so the dense and coupling
    // nodes needed in the workspace is the max of the two
    dense_nodes = std::max(dense_nodes, dist_hmatrix.hnodes.off_diagonal_blocks.num_dense_leaves);
    coupling_nodes = std::max(coupling_nodes, dist_hmatrix.hnodes.off_diagonal_blocks.num_rank_leaves);

    H2OpusWorkspaceState ws_allocated;

    hgemv_get_workspace(branch_level_data, branch_level_data, hnode_level_data, &offdiag_hnode_level_data, dense_nodes,
                        coupling_nodes, num_vectors, hw, ws_allocated, dist_workspace.branch_workspace, stream,
                        h2opus_ws);

    //////////////////////////////////////////////////////////////////
    // Additional memory allocations
    //////////////////////////////////////////////////////////////////
    // Align the workspace to reals
    size_t allocated_data_bytes, allocated_ptr_bytes;
    ws_allocated.getBytes(allocated_data_bytes, allocated_ptr_bytes, hw);
    allocated_data_bytes += (allocated_data_bytes % sizeof(H2Opus_Real));

    H2Opus_Real *ws_base = (H2Opus_Real *)((unsigned char *)h2opus_ws->getData(hw) + allocated_data_bytes);
    H2Opus_Real **ws_ptr_base = (H2Opus_Real **)((unsigned char *)h2opus_ws->getPtrs(hw) + allocated_ptr_bytes);

    // Allocate the yhat top level scatter buffer
    dist_workspace.yhat_scatter_root = ws_base;
    allocated_data_bytes += branch_level_data.getLevelRank(0) * num_vectors * sizeof(H2Opus_Real);

    size_t dense_offdiag_nodes, max_offdiag_nodes;
    getOffDiagonalMarshalNodes<hw>(dist_hmatrix, dense_offdiag_nodes, max_offdiag_nodes);

    // Align to ints
    allocated_data_bytes += (allocated_data_bytes % sizeof(int));
    int *ws_int_base = (int *)((unsigned char *)h2opus_ws->getData(hw) + allocated_data_bytes);

    dist_workspace.ptr_m = ws_int_base;
    dist_workspace.ptr_n = ws_int_base + dense_offdiag_nodes;
    dist_workspace.ptr_lda = ws_int_base + 2 * dense_offdiag_nodes;
    dist_workspace.ptr_ldb = ws_int_base + 3 * dense_offdiag_nodes;

    dist_workspace.ptr_A = ws_ptr_base;
    dist_workspace.ptr_B = ws_ptr_base + max_offdiag_nodes;
    //////////////////////////////////////////////////////////////////

    // Get the workspace needed for the top level, which is only on the master process
    if (dist_h2opus_handle->rank == 0)
    {
        BasisTreeLevelData &basis_top_level_data = dist_hmatrix.basis_tree.top_level.level_data;
        HNodeTreeLevelData &hnode_top_level_data = dist_hmatrix.hnodes.top_level.level_data;

        int dense_nodes = dist_hmatrix.hnodes.top_level.num_dense_leaves;
        int coupling_nodes = dist_hmatrix.hnodes.top_level.num_rank_leaves;

        h2opusComputeStream_t stream = dist_h2opus_handle->top_level_handle->getMainStream();
        h2opusWorkspace_t h2opus_ws = dist_h2opus_handle->top_level_handle->getWorkspace();

        H2OpusWorkspaceState top_level_ws_allocated;

        hgemv_get_workspace(basis_top_level_data, basis_top_level_data, hnode_top_level_data, dense_nodes,
                            coupling_nodes, num_vectors, hw, top_level_ws_allocated, dist_workspace.top_level_workspace,
                            stream, h2opus_ws);
    }
}

//////////////////////////////////////////////
// Interface routines
//////////////////////////////////////////////
void distributed_hgemv_get_workspace(DistributedHMatrix &dist_hmatrix, int num_vectors,
                                     DistributedHgemvWorkspace &dist_workspace,
                                     distributedH2OpusHandle_t dist_h2opus_handle)
{
    distributed_hgemv_get_workspace_template<H2OPUS_HWTYPE_CPU>(dist_hmatrix, num_vectors, dist_workspace,
                                                                dist_h2opus_handle);
}

void distributed_hgemv_workspace(DistributedHMatrix &dist_hmatrix, int num_vectors,
                                 H2OpusWorkspaceState &branch_ws_needed, H2OpusWorkspaceState &top_level_ws_needed,
                                 distributedH2OpusHandle_t dist_h2opus_handle)
{
    distributed_hgemv_workspace_template<H2OPUS_HWTYPE_CPU>(dist_hmatrix, num_vectors, branch_ws_needed,
                                                            top_level_ws_needed, dist_h2opus_handle);
}

#ifdef H2OPUS_USE_GPU
void distributed_hgemv_get_workspace(DistributedHMatrix_GPU &dist_hmatrix, int num_vectors,
                                     DistributedHgemvWorkspace &dist_workspace,
                                     distributedH2OpusHandle_t dist_h2opus_handle)
{
    distributed_hgemv_get_workspace_template<H2OPUS_HWTYPE_GPU>(dist_hmatrix, num_vectors, dist_workspace,
                                                                dist_h2opus_handle);
}

void distributed_hgemv_workspace(DistributedHMatrix_GPU &dist_hmatrix, int num_vectors,
                                 H2OpusWorkspaceState &branch_ws_needed, H2OpusWorkspaceState &top_level_ws_needed,
                                 distributedH2OpusHandle_t dist_h2opus_handle)
{
    distributed_hgemv_workspace_template<H2OPUS_HWTYPE_GPU>(dist_hmatrix, num_vectors, branch_ws_needed,
                                                            top_level_ws_needed, dist_h2opus_handle);
}
#endif
