#include <h2opus/distributed/distributed_horthog_workspace.h>

//////////////////////////////////////////////
// Utility routines
//////////////////////////////////////////////
template <int hw> inline void getOffDiagonalMarshalNodes(TDistributedHMatrix<hw> &dist_hmatrix, size_t &max_nodes)
{
    max_nodes = 0;
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
void distributed_horthog_workspace_template(TDistributedHMatrix<hw> &dist_hmatrix,
                                            H2OpusWorkspaceState &branch_ws_needed,
                                            H2OpusWorkspaceState &top_level_ws_needed,
                                            distributedH2OpusHandle_t dist_h2opus_handle)
{
    // See how much workspace is needed for the diagonal block
    BasisTreeLevelData &branch_level_data = dist_hmatrix.basis_tree.basis_branch.level_data;
    HNodeTreeLevelData &hnode_level_data = dist_hmatrix.hnodes.diagonal_block.level_data;
    HNodeTreeLevelData &offdiag_hnode_level_data = dist_hmatrix.hnodes.off_diagonal_blocks.level_data;

    int max_children = dist_hmatrix.basis_tree.basis_branch.max_children;
    bool symmetric = true; // Only symmetric matrices for now

    std::vector<int> branch_u_new_ranks(branch_level_data.depth, -1);
    std::vector<int> branch_v_new_ranks(branch_level_data.depth, -1);

    branch_ws_needed =
        horthog_workspace(branch_level_data, branch_level_data, hnode_level_data, &offdiag_hnode_level_data,
                          branch_u_new_ranks, branch_v_new_ranks, max_children, symmetric, hw);

    //////////////////////////////////////////////////////////////////
    // Additional memory allocations
    //////////////////////////////////////////////////////////////////
    size_t needed_data_bytes, needed_ptr_bytes;
    branch_ws_needed.getBytes(needed_data_bytes, needed_ptr_bytes, hw);

    // Add memory for the marshaling of the offdiagonal blocks
    size_t max_offdiag_nodes;
    getOffDiagonalMarshalNodes<hw>(dist_hmatrix, max_offdiag_nodes);

    // Add memory for the marshaling of the pointers for the block copies
    needed_ptr_bytes += 2 * max_offdiag_nodes * sizeof(H2Opus_Real *);

    branch_ws_needed.setBytes(needed_data_bytes, needed_ptr_bytes, hw);
    //////////////////////////////////////////////////////////////////

    // Get the workspace needed for the top level, which is only on the master process
    if (dist_h2opus_handle->rank == 0)
    {
        BasisTreeLevelData &basis_top_level_data = dist_hmatrix.basis_tree.top_level.level_data;
        HNodeTreeLevelData &hnode_top_level_data = dist_hmatrix.hnodes.top_level.level_data;

        std::vector<int> top_u_new_ranks(basis_top_level_data.depth, -1);
        std::vector<int> top_v_new_ranks(basis_top_level_data.depth, -1);

        top_u_new_ranks[basis_top_level_data.depth - 1] = branch_u_new_ranks[0];
        top_v_new_ranks[basis_top_level_data.depth - 1] = branch_v_new_ranks[0];

        top_level_ws_needed = horthog_workspace(basis_top_level_data, basis_top_level_data, hnode_top_level_data, NULL,
                                                top_u_new_ranks, top_v_new_ranks, max_children, symmetric, hw);
    }
}

template <int hw>
void distributed_horthog_get_workspace_template(TDistributedHMatrix<hw> &dist_hmatrix,
                                                DistributedHorthogWorkspace &dist_workspace,
                                                distributedH2OpusHandle_t dist_h2opus_handle)
{
    // See how much workspace is needed for the regular horthog
    h2opusHandle_t h2opus_handle = dist_h2opus_handle->handle;
    // h2opusComputeStream_t stream = h2opus_handle->getMainStream();
    h2opusWorkspace_t h2opus_ws = h2opus_handle->getWorkspace();

    BasisTreeLevelData &branch_level_data = dist_hmatrix.basis_tree.basis_branch.level_data;
    HNodeTreeLevelData &hnode_level_data = dist_hmatrix.hnodes.diagonal_block.level_data;
    HNodeTreeLevelData &offdiag_hnode_level_data = dist_hmatrix.hnodes.off_diagonal_blocks.level_data;

    int max_children = dist_hmatrix.basis_tree.basis_branch.max_children;
    bool symmetric = true; // Only symmetric matrices for now

    HorthogWorkspace &branch_workspace = dist_workspace.branch_workspace;
    branch_workspace.u_new_ranks.resize(branch_level_data.depth, -1);
    branch_workspace.v_new_ranks.resize(branch_level_data.depth, -1);

    H2OpusWorkspaceState ws_allocated =
        horthog_get_workspace(branch_level_data, branch_level_data, hnode_level_data, &offdiag_hnode_level_data,
                              max_children, symmetric, branch_workspace, h2opus_handle, hw);

    //////////////////////////////////////////////////////////////////
    // Additional memory allocations
    //////////////////////////////////////////////////////////////////
    size_t allocated_data_bytes, allocated_ptr_bytes;
    ws_allocated.getBytes(allocated_data_bytes, allocated_ptr_bytes, hw);

    H2Opus_Real **ws_ptr_base = (H2Opus_Real **)((unsigned char *)h2opus_ws->getPtrs(hw) + allocated_ptr_bytes);

    size_t max_offdiag_nodes;
    getOffDiagonalMarshalNodes<hw>(dist_hmatrix, max_offdiag_nodes);

    dist_workspace.ptr_A = ws_ptr_base;
    dist_workspace.ptr_B = ws_ptr_base + max_offdiag_nodes;
    //////////////////////////////////////////////////////////////////

    // Get the workspace needed for the top level, which is only on the master process
    if (dist_h2opus_handle->rank == 0)
    {
        BasisTreeLevelData &basis_top_level_data = dist_hmatrix.basis_tree.top_level.level_data;
        HNodeTreeLevelData &hnode_top_level_data = dist_hmatrix.hnodes.top_level.level_data;

        dist_workspace.top_level_workspace.u_new_ranks.resize(basis_top_level_data.depth, -1);
        dist_workspace.top_level_workspace.v_new_ranks.resize(basis_top_level_data.depth, -1);

        HorthogWorkspace &top_level_workspace = dist_workspace.top_level_workspace;

        top_level_workspace.u_new_ranks[basis_top_level_data.depth - 1] = branch_workspace.u_new_ranks[0];
        top_level_workspace.v_new_ranks[basis_top_level_data.depth - 1] = branch_workspace.v_new_ranks[0];

        horthog_get_workspace(basis_top_level_data, basis_top_level_data, hnode_top_level_data, NULL, max_children,
                              symmetric, top_level_workspace, dist_h2opus_handle->top_level_handle, hw);
    }
}

//////////////////////////////////////////////
// Interface routines
//////////////////////////////////////////////
void distributed_horthog_get_workspace(DistributedHMatrix &dist_hmatrix, DistributedHorthogWorkspace &dist_workspace,
                                       distributedH2OpusHandle_t dist_h2opus_handle)
{
    distributed_horthog_get_workspace_template<H2OPUS_HWTYPE_CPU>(dist_hmatrix, dist_workspace, dist_h2opus_handle);
}

void distributed_horthog_workspace(DistributedHMatrix &dist_hmatrix, H2OpusWorkspaceState &branch_ws_needed,
                                   H2OpusWorkspaceState &top_level_ws_needed,
                                   distributedH2OpusHandle_t dist_h2opus_handle)
{
    distributed_horthog_workspace_template<H2OPUS_HWTYPE_CPU>(dist_hmatrix, branch_ws_needed, top_level_ws_needed,
                                                              dist_h2opus_handle);
}

#ifdef H2OPUS_USE_GPU
void distributed_horthog_get_workspace(DistributedHMatrix_GPU &dist_hmatrix,
                                       DistributedHorthogWorkspace &dist_workspace,
                                       distributedH2OpusHandle_t dist_h2opus_handle)
{
    distributed_horthog_get_workspace_template<H2OPUS_HWTYPE_GPU>(dist_hmatrix, dist_workspace, dist_h2opus_handle);
}

void distributed_horthog_workspace(DistributedHMatrix_GPU &dist_hmatrix, H2OpusWorkspaceState &branch_ws_needed,
                                   H2OpusWorkspaceState &top_level_ws_needed,
                                   distributedH2OpusHandle_t dist_h2opus_handle)
{
    distributed_horthog_workspace_template<H2OPUS_HWTYPE_GPU>(dist_hmatrix, branch_ws_needed, top_level_ws_needed,
                                                              dist_h2opus_handle);
}
#endif
