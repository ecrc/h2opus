#include <h2opus/core/hcompress_workspace.h>
#include <h2opus/util/thrust_wrappers.h>

// TODO: deal with unsymmetric case and when the ranks of the row and column basis are not the same

//////////////////////////////////////////////////////////////////////////////////////////
// Helper routines
//////////////////////////////////////////////////////////////////////////////////////////
void hcompress_projection_tree_workspace(BasisTreeLevelData &level_data, HcompressUpsweepWorkspace *workspace,
                                         void *ws_base, size_t &required_bytes)
{
    size_t entries = 0;

    //////////////////////////////////////////////////////////////////////////////////////////
    // Workspace for projection tree
    //////////////////////////////////////////////////////////////////////////////////////////
    int num_levels = level_data.depth;
    for (int level = 0; level < num_levels; level++)
    {
        size_t node_rank = level_data.getLevelRank(level);
        size_t level_entries = level_data.getLevelSize(level) * node_rank * node_rank;

        if (workspace)
            workspace->T_hat[level] = (H2Opus_Real *)ws_base + entries;

        entries += level_entries;
    }
    required_bytes = entries * sizeof(H2Opus_Real);
}

void hcompress_project_top_level_workspace(BasisTreeLevelData &level_data, HcompressProjectTopLevelWorkspace *workspace,
                                           void *ws_base, void **ptr_base, size_t &required_bytes,
                                           size_t &required_ptr_bytes)
{
    required_bytes = required_ptr_bytes = 0;

    int top_level = level_data.nested_root_level;
    if (top_level == 0)
        return;

    int num_nodes = level_data.getLevelSize(top_level);
    int level_rank = level_data.getLevelRank(top_level);
    int parent_rank = level_data.getLevelRank(top_level - 1);

    if (num_nodes == 0 || parent_rank == 0 || level_rank == 0)
        return;

    if (workspace)
    {
        workspace->old_transfer = (H2Opus_Real *)ws_base;
        workspace->ptr_A = (H2Opus_Real **)ptr_base;
        workspace->ptr_B = workspace->ptr_A + num_nodes;
        workspace->ptr_C = workspace->ptr_B + num_nodes;
    }

    required_bytes = num_nodes * level_rank * parent_rank * sizeof(H2Opus_Real);
    required_ptr_bytes = num_nodes * 3 * sizeof(H2Opus_Real *);
}

void hcompress_weight_tree_workspace(BasisTreeLevelData &level_data, HcompressUpsweepWorkspace *workspace,
                                     void *ws_base, size_t &required_bytes)
{
    size_t entries = 0;

    //////////////////////////////////////////////////////////////////////////////////////////
    // Workspace for projection tree
    //////////////////////////////////////////////////////////////////////////////////////////
    int num_levels = level_data.depth;
    for (int level = 0; level < num_levels; level++)
    {
        size_t node_rank = level_data.getLevelRank(level);
        size_t level_entries = level_data.getLevelSize(level) * node_rank * node_rank;

        if (workspace)
            workspace->Z_hat[level] = (H2Opus_Real *)ws_base + entries;

        entries += level_entries;
    }
    required_bytes = entries * sizeof(H2Opus_Real);
}

//////////////////////////////////////////////////////////////////////////////////////////
// Template helper routines
//////////////////////////////////////////////////////////////////////////////////////////
void hcompress_hnode_projection_workspace(HNodeTreeLevelData &hnode_level_data,
                                          HNodeTreeLevelData *offdiag_hnode_level_data,
                                          HcompressProjectionWorkspace *workspace, void *ws_base, void **ptr_base,
                                          size_t &required_bytes, size_t &required_ptr_bytes)
{
    size_t max_level_nodes = hnode_level_data.getMaxLevelCouplingNodes();
    size_t max_level_size = hnode_level_data.getMaxCouplingLevelSize();

    if (offdiag_hnode_level_data)
    {
        max_level_nodes = std::max(max_level_nodes, offdiag_hnode_level_data->getMaxLevelCouplingNodes());
        max_level_size = std::max(max_level_size, offdiag_hnode_level_data->getMaxCouplingLevelSize());
    }

    if (workspace)
    {
        workspace->TC = (H2Opus_Real *)ws_base;

        workspace->ptr_A = (H2Opus_Real **)ptr_base;
        workspace->ptr_B = workspace->ptr_A + max_level_nodes;
        workspace->ptr_C = workspace->ptr_B + max_level_nodes;
        workspace->ptr_D = workspace->ptr_C + max_level_nodes;
        ;
    }

    required_bytes = max_level_size * sizeof(H2Opus_Real);
    required_ptr_bytes = 4 * max_level_nodes * sizeof(H2Opus_Real *);
}

void hcompress_upsweep_workspace(BasisTreeLevelData &level_data, void *ws_base, void **ptr_base,
                                 HcompressUpsweepWorkspace *workspace, size_t &required_bytes,
                                 size_t &required_ptr_bytes)
{
    size_t num_leaves = level_data.basis_leaves, leaf_size = level_data.leaf_size;
    size_t leaf_rank = level_data.getLevelRank(level_data.depth - 1);
    size_t max_children = level_data.max_children;

    // Temporary matrix data to form the matrix TE = [T_1 E_1; T_2 E_2]
    size_t te_data_size = level_data.getLargestChildStackSize(max_children);

    // Temporary memory for the UZ and S nodes
    size_t leaf_usize = num_leaves * leaf_size * leaf_rank;
    size_t u_size = std::max(leaf_usize, te_data_size), tau_size = level_data.getLargestLevelSizeByRank();

    if (workspace)
    {
        workspace->UZ_data = (H2Opus_Real *)ws_base;
        workspace->TE_data = workspace->UZ_data + u_size;
        workspace->tau_data = workspace->TE_data + te_data_size;
    }
    required_bytes = (te_data_size + u_size + tau_size) * sizeof(H2Opus_Real);

    // Temporary memory for ranks and matrix dimenion data for non-uniform batches
    // Align memory to sizeof(int)
    required_bytes += (required_bytes % sizeof(int));
    size_t largest_level_size = level_data.getLevelSize(level_data.getLargestLevel());

    if (workspace)
    {
        workspace->row_array = (int *)((unsigned char *)ws_base + required_bytes);
        workspace->col_array = workspace->row_array + largest_level_size;
        workspace->ld_array = workspace->col_array + largest_level_size;
        workspace->ranks_array = workspace->ld_array + largest_level_size;
    }
    required_bytes += 4 * largest_level_size * sizeof(int);

    // Pointer data
    size_t largest_level_nodes = max_children * level_data.getLevelSize(level_data.getLargestParentLevel());
    largest_level_nodes = std::max(largest_level_nodes, num_leaves);

    required_ptr_bytes = 4 * largest_level_nodes * sizeof(H2Opus_Real *);
    if (workspace)
    {
        workspace->ptr_A = (H2Opus_Real **)ptr_base;
        workspace->ptr_B = workspace->ptr_A + largest_level_nodes;
        workspace->ptr_C = workspace->ptr_B + largest_level_nodes;
        workspace->ptr_D = workspace->ptr_C + largest_level_nodes;
    }
}

void hcompress_generate_basis_workspace(HNodeTreeLevelData &hnode_level_data, std::vector<int> &bsn_max_nodes,
                                        HNodeTreeLevelData *offdiag_hnode_level_data,
                                        std::vector<int> *offdiag_bsn_max_nodes, BasisTreeLevelData &basis_level_data,
                                        void *ws_base, void **ptr_base, HcompressOptimalBGenWorkspace *workspace,
                                        size_t &required_bytes, size_t &required_ptr_bytes)
{
    int depth = hnode_level_data.depth;
    int start_level = 0; // basis_level_data.nested_root_level;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // First we need some memory for the stacked up block rows with the top block being
    // the weighted parent contribution
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    int max_rows = 0;
    size_t row_data_entries = 0, tau_entries = 0;
    int max_coupling_nodes = hnode_level_data.getMaxLevelCouplingNodes();
    if (offdiag_hnode_level_data)
        max_coupling_nodes += offdiag_hnode_level_data->getMaxLevelCouplingNodes();

    for (int level = start_level; level < depth; level++)
    {
        int level_rank = hnode_level_data.getLevelRank(level);
        int level_rows = basis_level_data.getLevelSize(level);

        int parent_rank = (level > 0 ? hnode_level_data.getLevelRank(level - 1) : 0);
        size_t ZE_entries = parent_rank * level_rank + level_rank * level_rank * bsn_max_nodes[level];
        if (offdiag_bsn_max_nodes)
            ZE_entries += level_rank * level_rank * (*offdiag_bsn_max_nodes)[level];

        size_t level_node_increment = std::min(COMPRESSION_BASIS_GEN_MAX_NODES, level_rows);

        row_data_entries = std::max(row_data_entries, level_node_increment * ZE_entries);
        tau_entries = std::max(tau_entries, level_node_increment * level_rank);
        max_rows = std::max(max_rows, level_rows);
    }

    size_t allocated_node_increment = std::min(COMPRESSION_BASIS_GEN_MAX_NODES, max_rows);
    if (workspace)
    {
        workspace->stacked_node_data = (H2Opus_Real *)ws_base;
        workspace->stacked_tau_data = (H2Opus_Real *)ws_base + row_data_entries;
    }
    required_bytes = (row_data_entries + tau_entries) * sizeof(H2Opus_Real);
    // Align bytes to integer array
    required_bytes += (required_bytes % sizeof(int));

    size_t dimension_entries = 3 * allocated_node_increment;
    if (workspace)
    {
        int *ws_base_int = (int *)((unsigned char *)ws_base + required_bytes);
        workspace->stacked_node_row = ws_base_int;
        workspace->stacked_node_col = ws_base_int + allocated_node_increment;
        workspace->stacked_node_ld = ws_base_int + 2 * allocated_node_increment;
    }
    required_bytes += dimension_entries * sizeof(int);

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // Pointer data
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    required_ptr_bytes = (allocated_node_increment * 2 + // stacked node pointers and tau pointers
                          3 * max_rows +                 // GEMM pointers to form Z * E
                          max_coupling_nodes * 2         // Transpose pointers
                          ) *
                         sizeof(H2Opus_Real *);

    if (workspace)
    {
        workspace->stacked_node_ptrs = (H2Opus_Real **)ptr_base;
        workspace->stacked_node_tau_ptrs = workspace->stacked_node_ptrs + allocated_node_increment;
        workspace->ptr_ZE = workspace->stacked_node_tau_ptrs + allocated_node_increment;
        workspace->ptr_Z = workspace->ptr_ZE + max_rows;
        workspace->ptr_E = workspace->ptr_Z + max_rows;
        workspace->ptr_row_data = workspace->ptr_E + max_rows;
        workspace->ptr_S = workspace->ptr_row_data + max_coupling_nodes;
    }
}

H2OpusWorkspaceState hcompress_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                                         HNodeTreeLevelData &hnode_level_data, std::vector<int> &diagonal_max_nodes,
                                         HNodeTreeLevelData *offdiag_hnode_level_data,
                                         std::vector<int> *offdiag_max_nodes, bool symmetric, int hw)
{
    // The only workspace we need throughout the whole compression is the Z_hat and T_hat
    size_t required_u_proj_bytes = 0, required_u_weight_bytes = 0;
    size_t required_v_proj_bytes = 0, required_v_weight_bytes = 0;
    hcompress_projection_tree_workspace(u_level_data, NULL, NULL, required_u_proj_bytes);
    hcompress_weight_tree_workspace(u_level_data, NULL, NULL, required_u_weight_bytes);
    if (!symmetric)
    {
        hcompress_projection_tree_workspace(v_level_data, NULL, NULL, required_v_proj_bytes);
        hcompress_weight_tree_workspace(v_level_data, NULL, NULL, required_v_weight_bytes);
    }
    size_t required_proj_bytes = required_u_proj_bytes + required_v_proj_bytes;
    size_t required_weight_bytes = std::max(required_u_weight_bytes, required_v_weight_bytes);

    // These are all temporary storage requirements so we take the max of them all
    size_t required_bgen_bytes, required_bgen_ptr_bytes;
    size_t required_upsweep_bytes, required_upsweep_ptr_bytes;
    size_t required_projection_bytes, required_projection_ptr_bytes;
    size_t required_top_level_bytes, required_top_level_ptr_bytes;

    hcompress_upsweep_workspace(u_level_data, NULL, NULL, NULL, required_upsweep_bytes, required_upsweep_ptr_bytes);

    hcompress_generate_basis_workspace(hnode_level_data, diagonal_max_nodes, offdiag_hnode_level_data,
                                       offdiag_max_nodes, u_level_data, NULL, NULL, NULL, required_bgen_bytes,
                                       required_bgen_ptr_bytes);

    hcompress_hnode_projection_workspace(hnode_level_data, offdiag_hnode_level_data, NULL, NULL, NULL,
                                         required_projection_bytes, required_projection_ptr_bytes);

    hcompress_project_top_level_workspace(u_level_data, NULL, NULL, NULL, required_top_level_bytes,
                                          required_top_level_ptr_bytes);

    size_t required_bytes = std::max(required_bgen_bytes, required_upsweep_bytes);
    size_t required_ptr_bytes = std::max(required_bgen_ptr_bytes, required_upsweep_ptr_bytes);

    required_bytes = std::max(required_bytes, required_projection_bytes);
    required_ptr_bytes = std::max(required_ptr_bytes, required_projection_ptr_bytes);

    required_bytes = std::max(required_bytes, required_top_level_bytes);
    required_ptr_bytes = std::max(required_ptr_bytes, required_top_level_ptr_bytes);

    required_bytes += required_proj_bytes + required_weight_bytes;

    H2OpusWorkspaceState ws_needed;
    ws_needed.setBytes(required_bytes, required_ptr_bytes, hw);

    return ws_needed;
}

H2OpusWorkspaceState hcompress_get_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                                             HNodeTreeLevelData &hnode_level_data, std::vector<int> &diagonal_max_nodes,
                                             HNodeTreeLevelData *offdiag_hnode_level_data,
                                             std::vector<int> *offdiag_max_nodes, bool symmetric,
                                             HcompressWorkspace &workspace, h2opusHandle_t h2opus_handle, int hw)
{
    h2opusWorkspace_t h2opus_ws = h2opus_handle->getWorkspace();
    workspace.symmetric = symmetric;

    // Allocate the workspace pointers from the handle
    void *ws_entries_base = (void *)h2opus_ws->getData(hw);
    void **ws_ptr_base = (void **)h2opus_ws->getPtrs(hw);

    // Allocate the host vectors for the pointers to the projection tree levels
    // and the new ranks for each level
    workspace.u_upsweep.new_ranks.resize(u_level_data.depth, 0);
    workspace.u_upsweep.T_hat.resize(u_level_data.depth);
    workspace.u_upsweep.Z_hat.resize(u_level_data.depth);

    if (!symmetric)
    {
        workspace.v_upsweep.new_ranks.resize(v_level_data.depth, 0);
        workspace.v_upsweep.T_hat.resize(v_level_data.depth);
        workspace.v_upsweep.Z_hat.resize(v_level_data.depth);
    }

    // Allocate Z_hat and T_hat for each basis
    size_t required_u_proj_bytes = 0, required_u_weight_bytes = 0;
    size_t required_v_proj_bytes = 0, required_v_weight_bytes = 0;

    void *temp_ws_base = ws_entries_base;
    hcompress_projection_tree_workspace(u_level_data, &workspace.u_upsweep, temp_ws_base, required_u_proj_bytes);
    temp_ws_base = (void *)((unsigned char *)temp_ws_base + required_u_proj_bytes);
    if (!symmetric)
    {
        hcompress_projection_tree_workspace(v_level_data, &workspace.v_upsweep, temp_ws_base, required_v_proj_bytes);
        temp_ws_base = (void *)((unsigned char *)temp_ws_base + required_v_proj_bytes);
    }
    size_t required_proj_bytes = required_u_proj_bytes + required_v_proj_bytes;

    // Only need one Z_hat, so we allocate the maximum of the two
    hcompress_weight_tree_workspace(u_level_data, &workspace.u_upsweep, temp_ws_base, required_u_weight_bytes);
    if (!symmetric)
        hcompress_weight_tree_workspace(v_level_data, &workspace.v_upsweep, temp_ws_base, required_v_weight_bytes);
    size_t required_weight_bytes = std::max(required_u_weight_bytes, required_v_weight_bytes);

    temp_ws_base = (void *)((unsigned char *)temp_ws_base + required_weight_bytes);

    // These are all temporary storage
    size_t required_bgen_bytes, required_bgen_ptr_bytes;
    size_t required_upsweep_bytes, required_upsweep_ptr_bytes;
    size_t required_projection_bytes, required_projection_ptr_bytes;
    size_t required_top_level_bytes, required_top_level_ptr_bytes;

    hcompress_upsweep_workspace(u_level_data, temp_ws_base, ws_ptr_base, &(workspace.u_upsweep), required_upsweep_bytes,
                                required_upsweep_ptr_bytes);

    hcompress_generate_basis_workspace(hnode_level_data, diagonal_max_nodes, offdiag_hnode_level_data,
                                       offdiag_max_nodes, u_level_data, temp_ws_base, ws_ptr_base,
                                       &(workspace.optimal_bgen), required_bgen_bytes, required_bgen_ptr_bytes);

    hcompress_hnode_projection_workspace(hnode_level_data, offdiag_hnode_level_data, &(workspace.projection),
                                         temp_ws_base, ws_ptr_base, required_projection_bytes,
                                         required_projection_ptr_bytes);

    hcompress_project_top_level_workspace(u_level_data, &workspace.u_top_level, temp_ws_base, ws_ptr_base,
                                          required_top_level_bytes, required_top_level_ptr_bytes);

    size_t required_bytes = std::max(required_bgen_bytes, required_upsweep_bytes);
    size_t required_ptr_bytes = std::max(required_bgen_ptr_bytes, required_upsweep_ptr_bytes);

    required_bytes = std::max(required_bytes, required_projection_bytes);
    required_ptr_bytes = std::max(required_ptr_bytes, required_projection_ptr_bytes);

    required_bytes = std::max(required_bytes, required_top_level_bytes);
    required_ptr_bytes = std::max(required_ptr_bytes, required_top_level_ptr_bytes);

    required_bytes += required_proj_bytes + required_weight_bytes;

    H2OpusWorkspaceState ws_needed;
    ws_needed.setBytes(required_bytes, required_ptr_bytes, hw);

    return ws_needed;
}

template <int hw> H2OpusWorkspaceState hcompress_workspace_template(THMatrix<hw> &hmatrix)
{
    // Add up the entries needed for the basis trees upsweeps
    TBasisTree<hw> &u_basis_tree = hmatrix.u_basis_tree;
    TBasisTree<hw> &v_basis_tree = (hmatrix.sym ? hmatrix.u_basis_tree : hmatrix.v_basis_tree);

    BasisTreeLevelData &u_level_data = u_basis_tree.level_data;
    BasisTreeLevelData &v_level_data = v_basis_tree.level_data;
    HNodeTreeLevelData &hnode_level_data = hmatrix.hnodes.level_data;
    std::vector<int> &max_nodes = hmatrix.hnodes.bsn_row_data.max_nodes;

    return hcompress_workspace(u_level_data, v_level_data, hnode_level_data, max_nodes, NULL, NULL, hmatrix.sym, hw);
}

template <int hw>
void hcompress_get_workspace_template(THMatrix<hw> &hmatrix, HcompressWorkspace &workspace,
                                      h2opusHandle_t h2opus_handle)
{
    TBasisTree<hw> &u_basis_tree = hmatrix.u_basis_tree;
    TBasisTree<hw> &v_basis_tree = (hmatrix.sym ? hmatrix.u_basis_tree : hmatrix.v_basis_tree);

    BasisTreeLevelData &u_level_data = u_basis_tree.level_data;
    BasisTreeLevelData &v_level_data = v_basis_tree.level_data;

    HNodeTreeLevelData &hnode_level_data = hmatrix.hnodes.level_data;
    std::vector<int> &max_nodes = hmatrix.hnodes.bsn_row_data.max_nodes;

    hcompress_get_workspace(u_level_data, v_level_data, hnode_level_data, max_nodes, NULL, NULL, hmatrix.sym, workspace,
                            h2opus_handle, hw);
}

//////////////////////////////////////////////////////////////////////////////////////////
// Interface routines
//////////////////////////////////////////////////////////////////////////////////////////
H2OpusWorkspaceState hcompress_workspace(HMatrix &hmatrix)
{
    return hcompress_workspace_template<H2OPUS_HWTYPE_CPU>(hmatrix);
}

void hcompress_get_workspace(HMatrix &hmatrix, HcompressWorkspace &workspace, h2opusHandle_t h2opus_handle)
{
    hcompress_get_workspace_template<H2OPUS_HWTYPE_CPU>(hmatrix, workspace, h2opus_handle);
}

#ifdef H2OPUS_USE_GPU
H2OpusWorkspaceState hcompress_workspace(HMatrix_GPU &hmatrix)
{
    return hcompress_workspace_template<H2OPUS_HWTYPE_GPU>(hmatrix);
}

void hcompress_get_workspace(HMatrix_GPU &hmatrix, HcompressWorkspace &workspace, h2opusHandle_t h2opus_handle)
{
    hcompress_get_workspace_template<H2OPUS_HWTYPE_GPU>(hmatrix, workspace, h2opus_handle);
}
#endif
