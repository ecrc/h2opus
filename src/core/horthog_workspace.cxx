#include <h2opus/core/horthog_workspace.h>
#include <h2opus/util/thrust_wrappers.h>

// TODO: deal with unsymmetric case when the ranks of the row and column basis are not the same

//////////////////////////////////////////////////////////////////////////////////////////
// Helper routines
//////////////////////////////////////////////////////////////////////////////////////////
size_t horthog_projection_tree_workspace(BasisTreeLevelData &level_data, std::vector<H2Opus_Real *> *T_hat = NULL,
                                         H2Opus_Real *ws_entries_base = NULL)
{
    size_t entries = 0;

    //////////////////////////////////////////////////////////////////////////////////////////
    // Workspace for projection tree
    //////////////////////////////////////////////////////////////////////////////////////////
    int num_levels = level_data.depth;
    for (int level = 0; level < num_levels; level++)
    {
        size_t node_rank = level_data.getLevelRank(level);
        size_t level_nodes = level_data.getLevelSize(level);
        size_t level_entries = level_nodes * node_rank * node_rank;

        if (T_hat)
            (*T_hat)[level] = ws_entries_base + entries;

        entries += level_entries;
    }

    return entries;
}

size_t horthog_hnode_projection_workspace(HNodeTreeLevelData &hnode_level_data,
                                          HNodeTreeLevelData *offdiag_hnode_level_data)
{
    size_t max_level_nodes = hnode_level_data.getMaxLevelCouplingNodes();
    size_t max_level_rank = hnode_level_data.getLargestRank();

    if (offdiag_hnode_level_data)
    {
        max_level_nodes = std::max(max_level_nodes, offdiag_hnode_level_data->getMaxLevelCouplingNodes());
        max_level_rank = std::max(max_level_rank, (size_t)offdiag_hnode_level_data->getLargestRank());
    }

    size_t increment = std::min((size_t)PROJECTION_MAX_NODES, max_level_nodes);
    return increment * max_level_rank * max_level_rank;
}

//////////////////////////////////////////////////////////////////////////////////////////
// Template helper routines
//////////////////////////////////////////////////////////////////////////////////////////
size_t horthog_ptr_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                             HNodeTreeLevelData &hnode_level_data, HNodeTreeLevelData *offdiag_hnode_level_data,
                             int max_children, bool symmetric, HorthogWorkspace *workspace = NULL,
                             H2Opus_Real **ws_ptr_base = NULL)
{
    size_t max_ptr_entries = 0;
    //////////////////////////////////////////////////////////////////////////////////////////
    // Temporary memory for the pointers that we generate to pass to batch gemm routines
    // Pointers are reused after each operation
    //////////////////////////////////////////////////////////////////////////////////////////
    size_t largest_level_nodes = max_children * u_level_data.getLevelSize(u_level_data.getLargestParentLevel());
    max_ptr_entries = std::max(max_ptr_entries, 3 * largest_level_nodes);

    if (!symmetric)
    {
        largest_level_nodes = max_children * v_level_data.getLevelSize(v_level_data.getLargestParentLevel());
        max_ptr_entries = std::max(max_ptr_entries, 3 * largest_level_nodes);
    }

    if (workspace)
    {
        workspace->ptr_TE = ws_ptr_base;
        workspace->ptr_T = workspace->ptr_TE + largest_level_nodes;
        workspace->ptr_E = workspace->ptr_T + largest_level_nodes;
    }

    size_t max_coupling_level_nodes = hnode_level_data.getMaxLevelCouplingNodes();
    if (offdiag_hnode_level_data)
        max_coupling_level_nodes =
            std::max(max_coupling_level_nodes, offdiag_hnode_level_data->getMaxLevelCouplingNodes());

    size_t increment = std::min((size_t)PROJECTION_MAX_NODES, max_coupling_level_nodes);

    max_ptr_entries = std::max(max_ptr_entries, increment + 4 * max_coupling_level_nodes);

    if (workspace)
    {
        workspace->TS_array = ws_ptr_base;
        workspace->Tu_array = workspace->TS_array + increment;
        workspace->Tv_array = workspace->Tu_array + max_coupling_level_nodes;
        workspace->S_array = workspace->Tv_array + max_coupling_level_nodes;
        workspace->S_new_array = workspace->S_array + max_coupling_level_nodes;
    }

    return max_ptr_entries;
}

size_t horthog_upsweep_realloc_workspace(BasisTreeLevelData &level_data, std::vector<int> &new_ranks)
{
    size_t entries = 0;
    // Temporary reallcation workspace for the upsweep when the rank changes
    size_t max_temp_entries = 0;

    if (new_ranks[level_data.depth - 1] == -1)
    {
        int leaf_rows = level_data.leaf_size;
        int leaf_rank = level_data.getLevelRank(level_data.depth - 1);

        // Check if we need to reduce the rank of the leaves
        if (leaf_rank > leaf_rows)
        {
            new_ranks[level_data.depth - 1] = leaf_rows;
            size_t level_entries = level_data.basis_leaves * leaf_rows * leaf_rank;
            max_temp_entries = std::max(max_temp_entries, level_entries);
        }
        else
            new_ranks[level_data.depth - 1] = leaf_rank;
    }

    // Sweep up the tree and see if the rank on each level needs to be reduced
    int top_level = level_data.nested_root_level;
    int num_levels = level_data.depth;

    for (int level = num_levels - 2; level >= top_level; level--)
    {
        int child_rank = level_data.getLevelRank(level + 1);
        int level_rank = level_data.getLevelRank(level);
        int child_new_rank = new_ranks[level + 1];

        new_ranks[level] = level_rank;

        if (child_rank == 0 || level_rank == 0)
            continue;

        int te_rows = level_data.max_children * child_new_rank;

        if (child_new_rank != child_rank || te_rows < level_rank)
        {
            size_t level_entries = level_data.getLevelSize(level + 1) * level_rank * child_rank;
            max_temp_entries = std::max(max_temp_entries, level_entries);
            new_ranks[level] = std::min(te_rows, level_rank);
        }
    }

    entries += max_temp_entries;

    return entries;
}

size_t horthog_realloc_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                                 HNodeTreeLevelData &hnode_level_data, HNodeTreeLevelData *offdiag_hnode_level_data,
                                 int max_children, bool symmetric, std::vector<int> &u_new_ranks,
                                 std::vector<int> &v_new_ranks)
{
    // Check if the basis tree data needs reallocation
    size_t upsweep_realloc_entries = horthog_upsweep_realloc_workspace(u_level_data, u_new_ranks);
    if (!symmetric)
    {
        upsweep_realloc_entries =
            std::max(upsweep_realloc_entries, horthog_upsweep_realloc_workspace(v_level_data, v_new_ranks));
    }
    // Check if the hnodes need to allocate temporary memory
    size_t proj_realloc_entries = 0;
    int num_levels = hnode_level_data.depth;
    int top_level = u_level_data.nested_root_level;
    assert(u_level_data.nested_root_level == v_level_data.nested_root_level);

    for (int level = num_levels - 1; level >= top_level; level--)
    {
        int level_rank = hnode_level_data.getLevelRank(level);
        int level_new_rank = u_new_ranks[level];

        if (level_rank != level_new_rank)
        {
            size_t level_nodes = hnode_level_data.getLevelSize(level);
            if (offdiag_hnode_level_data)
                level_nodes = std::max(level_nodes, (size_t)offdiag_hnode_level_data->getLevelSize(level));

            proj_realloc_entries = std::max(proj_realloc_entries, level_nodes * level_rank * level_rank);
        }
    }

    // Check if we need to stitch the top level
    size_t stitch_entries = 0;
    int u_stitch_level = u_level_data.nested_root_level;
    if (u_stitch_level != 0)
    {
        int stitch_level_rank = u_level_data.getLevelRank(u_stitch_level);
        stitch_entries = u_level_data.getLevelSize(u_stitch_level) * stitch_level_rank * stitch_level_rank;
    }

    int v_stitch_level = v_level_data.nested_root_level;
    if (!symmetric && v_stitch_level != 0)
    {
        int stitch_level_rank = v_level_data.getLevelRank(v_stitch_level);
        size_t v_stitch_entries = v_level_data.getLevelSize(v_stitch_level) * stitch_level_rank * stitch_level_rank;
        stitch_entries = std::max(stitch_entries, v_stitch_entries);
    }
    return std::max(stitch_entries, std::max(upsweep_realloc_entries, proj_realloc_entries));
}

H2OpusWorkspaceState horthog_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                                       HNodeTreeLevelData &hnode_level_data,
                                       HNodeTreeLevelData *offdiag_hnode_level_data, std::vector<int> &u_new_ranks,
                                       std::vector<int> &v_new_ranks, int max_children, bool symmetric, int hw)
{
    // We need both projection trees so we add instead of taking the max
    size_t projection_tree_entries = horthog_projection_tree_workspace(u_level_data);
    if (!symmetric)
        projection_tree_entries += horthog_projection_tree_workspace(v_level_data);

    // Get the necessary reallocation workspace - happens at the same time as the projection/upsweep
    // so it has to be added to the final result
    size_t realloc_entries =
        horthog_realloc_workspace(u_level_data, v_level_data, hnode_level_data, offdiag_hnode_level_data, max_children,
                                  symmetric, u_new_ranks, v_new_ranks);

    // Upsweep could be done on both trees in parallel, but for now we do them one at a time
    // so we take the max of the required entries
    size_t upsweep_te_entries = u_level_data.getLargestChildStackSize(max_children);
    if (!symmetric)
        upsweep_te_entries = std::max(upsweep_te_entries, v_level_data.getLargestChildStackSize(max_children));

    size_t upsweep_tau_entries = u_level_data.getLargestLevelSizeByRank();
    if (!symmetric)
        upsweep_tau_entries = std::max(upsweep_tau_entries, v_level_data.getLargestLevelSizeByRank());

    // Get the entries for the projection phase - happens after the upsweeps are completed,
    // so the workspace here is the larger of the two

    size_t proj_entries = horthog_hnode_projection_workspace(hnode_level_data, offdiag_hnode_level_data);

    // The total needed entries
    size_t total_entries =
        projection_tree_entries + realloc_entries + std::max(proj_entries, upsweep_te_entries + upsweep_tau_entries);
    size_t total_ptrs = horthog_ptr_workspace(u_level_data, v_level_data, hnode_level_data, offdiag_hnode_level_data,
                                              max_children, symmetric);

    H2OpusWorkspaceState ws_needed;
    ws_needed.setBytes(total_entries * sizeof(H2Opus_Real), total_ptrs * sizeof(H2Opus_Real *), hw);

    return ws_needed;
}

H2OpusWorkspaceState horthog_get_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                                           HNodeTreeLevelData &hnode_level_data,
                                           HNodeTreeLevelData *offdiag_hnode_level_data, int max_children,
                                           bool symmetric, HorthogWorkspace &workspace, h2opusHandle_t h2opus_handle,
                                           int hw)
{
    h2opusComputeStream_t stream = h2opus_handle->getMainStream();
    h2opusWorkspace_t h2opus_ws = h2opus_handle->getWorkspace();

    // Allocate the host vectors for the pointers to the projection tree levels
    workspace.Tu_hat.resize(u_level_data.depth);
    workspace.symmetric = symmetric;

    if (!symmetric)
        workspace.Tv_hat.resize(u_level_data.depth);

    // Allocate the workspace pointers from the handle
    H2Opus_Real *ws_entries_base = (H2Opus_Real *)h2opus_ws->getData(hw);
    H2Opus_Real **ws_ptr_base = (H2Opus_Real **)h2opus_ws->getPtrs(hw);

    size_t projection_tree_entries =
        horthog_projection_tree_workspace(u_level_data, &(workspace.Tu_hat), ws_entries_base);
    if (!symmetric)
        projection_tree_entries += horthog_projection_tree_workspace(v_level_data, &(workspace.Tv_hat),
                                                                     ws_entries_base + projection_tree_entries);

    ws_entries_base += projection_tree_entries;

    // Get the necessary reallocation workspace - happens at the same time as the projection/upsweep
    // so it has to be added to the final result
    size_t realloc_entries =
        horthog_realloc_workspace(u_level_data, v_level_data, hnode_level_data, offdiag_hnode_level_data, max_children,
                                  symmetric, workspace.u_new_ranks, workspace.v_new_ranks);

    workspace.realloc_buffer = ws_entries_base;
    ws_entries_base += realloc_entries;

    // Upsweep could be done on both trees in parallel, but for now we do them one at a time
    // so we take the max of the required entries
    size_t upsweep_te_entries = u_level_data.getLargestChildStackSize(max_children);
    if (!symmetric)
        upsweep_te_entries = std::max(upsweep_te_entries, v_level_data.getLargestChildStackSize(max_children));

    size_t upsweep_tau_entries = u_level_data.getLargestLevelSizeByRank();
    if (!symmetric)
        upsweep_tau_entries = std::max(upsweep_tau_entries, v_level_data.getLargestLevelSizeByRank());

    workspace.TE_data = ws_entries_base;
    workspace.TE_tau = ws_entries_base + upsweep_te_entries;

    // Get the entries for the projection phase - happens after the upsweeps are completed,
    // so the workspace here is the larger of the two
    size_t proj_entries = horthog_hnode_projection_workspace(hnode_level_data, offdiag_hnode_level_data);

    workspace.TS = ws_entries_base;

    size_t total_entries =
        projection_tree_entries + realloc_entries + std::max(proj_entries, upsweep_te_entries + upsweep_tau_entries);

    // Clear the workspace
    fillArray((H2Opus_Real *)h2opus_ws->getData(hw), total_entries, 0, stream, hw);

    size_t total_ptrs = horthog_ptr_workspace(u_level_data, v_level_data, hnode_level_data, offdiag_hnode_level_data,
                                              max_children, symmetric, &workspace, ws_ptr_base);

    H2OpusWorkspaceState ws_allocated;
    ws_allocated.setBytes(total_entries * sizeof(H2Opus_Real), total_ptrs * sizeof(H2Opus_Real *), hw);

    return ws_allocated;
}

template <int hw> H2OpusWorkspaceState horthog_workspace_template(THMatrix<hw> &hmatrix)
{
    bool symmetric = hmatrix.sym;

    // Add up the entries needed for the basis trees upsweeps
    TBasisTree<hw> &u_basis_tree = hmatrix.u_basis_tree;
    TBasisTree<hw> &v_basis_tree = (symmetric ? hmatrix.u_basis_tree : hmatrix.v_basis_tree);

    BasisTreeLevelData &u_level_data = u_basis_tree.level_data;
    BasisTreeLevelData &v_level_data = v_basis_tree.level_data;
    HNodeTreeLevelData &hnode_level_data = hmatrix.hnodes.level_data;

    int max_children = u_basis_tree.max_children;

    std::vector<int> u_new_ranks(u_level_data.depth, -1);
    std::vector<int> v_new_ranks(v_level_data.depth, -1);

    return horthog_workspace(u_level_data, v_level_data, hnode_level_data, NULL, u_new_ranks, v_new_ranks, max_children,
                             symmetric, hw);
}

template <int hw>
void horthog_get_workspace_template(THMatrix<hw> &hmatrix, HorthogWorkspace &workspace, h2opusHandle_t h2opus_handle)
{
    bool symmetric = hmatrix.sym;

    TBasisTree<hw> &u_basis_tree = hmatrix.u_basis_tree;
    TBasisTree<hw> &v_basis_tree = (symmetric ? hmatrix.u_basis_tree : hmatrix.v_basis_tree);

    BasisTreeLevelData &u_level_data = u_basis_tree.level_data;
    BasisTreeLevelData &v_level_data = v_basis_tree.level_data;
    HNodeTreeLevelData &hnode_level_data = hmatrix.hnodes.level_data;

    int max_children = u_basis_tree.max_children;

    workspace.u_new_ranks.resize(u_level_data.depth, -1);
    if (!symmetric)
        workspace.v_new_ranks.resize(v_level_data.depth, -1);

    horthog_get_workspace(u_level_data, v_level_data, hnode_level_data, NULL, max_children, symmetric, workspace,
                          h2opus_handle, hw);
}

//////////////////////////////////////////////////////////////////////////////////////////
// Interface routines
//////////////////////////////////////////////////////////////////////////////////////////
H2OpusWorkspaceState horthog_workspace(HMatrix &hmatrix)
{
    return horthog_workspace_template<H2OPUS_HWTYPE_CPU>(hmatrix);
}

void horthog_get_workspace(HMatrix &hmatrix, HorthogWorkspace &workspace, h2opusHandle_t h2opus_handle)
{
    horthog_get_workspace_template<H2OPUS_HWTYPE_CPU>(hmatrix, workspace, h2opus_handle);
}

#ifdef H2OPUS_USE_GPU
H2OpusWorkspaceState horthog_workspace(HMatrix_GPU &hmatrix)
{
    return horthog_workspace_template<H2OPUS_HWTYPE_GPU>(hmatrix);
}

void horthog_get_workspace(HMatrix_GPU &hmatrix, HorthogWorkspace &workspace, h2opusHandle_t h2opus_handle)
{
    horthog_get_workspace_template<H2OPUS_HWTYPE_GPU>(hmatrix, workspace, h2opus_handle);
}
#endif
