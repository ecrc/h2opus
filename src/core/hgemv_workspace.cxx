#include <h2opus/core/hgemv_workspace.h>
#include <h2opus/core/h2opus_defs.h>

#include <h2opus/util/thrust_wrappers.h>
#include <thrust/fill.h>

//////////////////////////////////////////////
// utility
//////////////////////////////////////////////
void setGemmMarshalledData(BatchGemmMarshalledData &data, int *dim_base_ptr, H2Opus_Real **pointers_base_ptr, int ops)
{
    data.m_batch = dim_base_ptr;
    dim_base_ptr += ops;
    data.n_batch = dim_base_ptr;
    dim_base_ptr += ops;
    data.k_batch = dim_base_ptr;
    dim_base_ptr += ops;
    data.lda_batch = dim_base_ptr;
    dim_base_ptr += ops;
    data.ldb_batch = dim_base_ptr;
    dim_base_ptr += ops;
    data.ldc_batch = dim_base_ptr;

    data.A_ptrs = pointers_base_ptr;
    data.B_ptrs = data.A_ptrs + ops;
    data.C_ptrs = data.B_ptrs + ops;
}

void getHgemvVectorTree(VectorTree &vector_tree, BasisTreeLevelData &level_data, int num_vectors, H2Opus_Real *ws_base,
                        h2opusComputeStream_t stream, int hw)
{
    int tree_depth = level_data.depth;
    size_t total_entries = 0;

    vector_tree.data.resize(tree_depth);
    for (int level = 0; level < tree_depth; level++)
    {
        size_t node_size = level_data.getLevelRank(level) * num_vectors;
        size_t level_entries = level_data.getLevelSize(level) * node_size;

        vector_tree.data[level] = ws_base + total_entries;
        total_entries += level_entries;
    }
    vector_tree.allocated_entries = total_entries;

    // Clear the tree data
    fillArray(ws_base, total_entries, 0, stream, hw);
}

int hgemv_getMaxGemmOps(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                        HNodeTreeLevelData &hnode_level_data, HNodeTreeLevelData *offdiag_hnode_level_data)
{
    // Start with the upsweep...
    int max_gemm_ops = u_level_data.max_children * u_level_data.getLevelSize(u_level_data.depth - 1);

    // mult...
    for (int level = hnode_level_data.depth - 1; level >= 0; level--)
    {
        int level_size = hnode_level_data.getCouplingLevelSize(level);
        if (offdiag_hnode_level_data)
            level_size = std::max(level_size, offdiag_hnode_level_data->getCouplingLevelSize(level));
        max_gemm_ops = std::max(max_gemm_ops, level_size);
    }

    // downsweep...
    max_gemm_ops = std::max(max_gemm_ops, v_level_data.getLevelSize(v_level_data.depth - 1));

    return max_gemm_ops;
}

H2OpusWorkspaceState hgemv_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                                     HNodeTreeLevelData &hnode_level_data, HNodeTreeLevelData *offdiag_hnode_level_data,
                                     int dense_nodes, int coupling_nodes, int num_vectors, int hw)
{
    ////////////////////////////////////////////////////////////////////////////////////
    // Data bytes
    ////////////////////////////////////////////////////////////////////////////////////
    // Workspace for xhat and yhat
    size_t xhat_entries = 0, yhat_entries = 0;
    int tree_depth = u_level_data.depth;
    for (int level = 0; level < tree_depth; level++)
    {
        size_t xnode_size = v_level_data.getLevelRank(level) * num_vectors;
        size_t ynode_size = u_level_data.getLevelRank(level) * num_vectors;

        xhat_entries += v_level_data.getLevelSize(level) * xnode_size;
        yhat_entries += u_level_data.getLevelSize(level) * ynode_size;
    }

    // GEMM dimension data for the batches - needs 6 integers per gemm operation
    // One gemm per per low rank gemm and one gemm per dense block, since the
    // dense blocks are carried out on a separate stream in parallel
    int max_gemm_ops = hgemv_getMaxGemmOps(u_level_data, v_level_data, hnode_level_data, offdiag_hnode_level_data);
    max_gemm_ops += dense_nodes;

    // Align the xhat and yhat tree entries to the size of integers and add 6 integers per gemm op
    size_t total_data_bytes = (xhat_entries + yhat_entries) * sizeof(H2Opus_Real);
    total_data_bytes += (total_data_bytes % sizeof(int));
    total_data_bytes += 6 * max_gemm_ops * sizeof(int);

    ////////////////////////////////////////////////////////////////////////////////////
    // Ptr bytes - 3 per gemm operation
    ////////////////////////////////////////////////////////////////////////////////////
    size_t total_ptr_bytes = 3 * max_gemm_ops * sizeof(H2Opus_Real *);

    H2OpusWorkspaceState ws_needed;
    ws_needed.setBytes(total_data_bytes, total_ptr_bytes, hw);

    return ws_needed;
}

void hgemv_get_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                         HNodeTreeLevelData &hnode_level_data, HNodeTreeLevelData *offdiag_hnode_level_data,
                         int dense_nodes, int coupling_nodes, int num_vectors, int hw,
                         H2OpusWorkspaceState &ws_allocated, HgemvWorkspace &workspace, h2opusComputeStream_t stream,
                         h2opusWorkspace_t h2opus_ws)
{
    size_t allocated_bytes = 0, allocated_ptrs = 0;

    H2Opus_Real *ws_base = (H2Opus_Real *)h2opus_ws->getData(hw);

    getHgemvVectorTree(workspace.xhat, v_level_data, num_vectors, ws_base, stream, hw);
    getHgemvVectorTree(workspace.yhat, u_level_data, num_vectors, ws_base + workspace.xhat.allocated_entries, stream,
                       hw);

    // Align the pointer to integers
    allocated_bytes = (workspace.xhat.allocated_entries + workspace.yhat.allocated_entries) * sizeof(H2Opus_Real);
    allocated_bytes += (allocated_bytes % sizeof(int));

    // Allocate integer arrays for batch gemm operations
    size_t max_gemm_ops = hgemv_getMaxGemmOps(u_level_data, v_level_data, hnode_level_data, offdiag_hnode_level_data);

    // Allocate marshal data for the low rank gemms
    int *dim_base_ptr = (int *)((unsigned char *)h2opus_ws->getData(hw) + allocated_bytes);
    H2Opus_Real **pointers_base_ptr = (H2Opus_Real **)h2opus_ws->getPtrs(hw);

    setGemmMarshalledData(workspace.low_rank_gemms, dim_base_ptr, pointers_base_ptr, max_gemm_ops);

    allocated_bytes += 6 * max_gemm_ops * sizeof(int);
    allocated_ptrs += 3 * max_gemm_ops * sizeof(H2Opus_Real *);

    // Advance the pointers
    dim_base_ptr += 6 * max_gemm_ops;
    pointers_base_ptr += 3 * max_gemm_ops;

    // Allocate marshal data for the dense gemms
    setGemmMarshalledData(workspace.dense_gemms, dim_base_ptr, pointers_base_ptr, dense_nodes);

    allocated_bytes += 6 * dense_nodes * sizeof(int);
    allocated_ptrs += 3 * dense_nodes * sizeof(H2Opus_Real *);

    ws_allocated.setBytes(allocated_bytes, allocated_ptrs, hw);
}

H2OpusWorkspaceState hgemv_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                                     HNodeTreeLevelData &hnode_level_data, int dense_nodes, int coupling_nodes,
                                     int num_vectors, int hw)
{
    return hgemv_workspace(u_level_data, v_level_data, hnode_level_data, NULL, dense_nodes, coupling_nodes, num_vectors,
                           hw);
}

void hgemv_get_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                         HNodeTreeLevelData &hnode_level_data, int dense_nodes, int coupling_nodes, int num_vectors,
                         int hw, H2OpusWorkspaceState &ws_allocated, HgemvWorkspace &workspace,
                         h2opusComputeStream_t stream, h2opusWorkspace_t h2opus_ws)
{
    hgemv_get_workspace(u_level_data, v_level_data, hnode_level_data, NULL, dense_nodes, coupling_nodes, num_vectors,
                        hw, ws_allocated, workspace, stream, h2opus_ws);
}

//////////////////////////////////////////////
// Templates
//////////////////////////////////////////////
template <int hw> H2OpusWorkspaceState hgemv_workspace_template(THMatrix<hw> &hmatrix, int trans, int num_vectors)
{
    // Don't use the transpose code if the matrix is symmetric
    if (trans == H2Opus_Trans && hmatrix.sym)
        trans = H2Opus_NoTrans;

    TBasisTree<hw> &u_basis_tree = (trans == H2Opus_Trans ? hmatrix.v_basis_tree : hmatrix.u_basis_tree);
    TBasisTree<hw> &v_basis_tree = (hmatrix.sym || trans == H2Opus_Trans ? u_basis_tree : hmatrix.v_basis_tree);

    BasisTreeLevelData &u_level_data = u_basis_tree.level_data;
    BasisTreeLevelData &v_level_data = v_basis_tree.level_data;

    int dense_nodes = hmatrix.hnodes.num_dense_leaves;
    int coupling_nodes = hmatrix.hnodes.num_rank_leaves;

    return hgemv_workspace(u_level_data, v_level_data, hmatrix.hnodes.level_data, dense_nodes, coupling_nodes,
                           num_vectors, hw);
}

template <int hw>
void hgemv_get_workspace_template(THMatrix<hw> &hmatrix, int trans, int num_vectors, HgemvWorkspace &workspace,
                                  h2opusHandle_t h2opus_handle)
{
    // Don't use the transpose code if the matrix is symmetric
    if (trans == H2Opus_Trans && hmatrix.sym)
        trans = H2Opus_NoTrans;

    TBasisTree<hw> &u_basis_tree = (trans == H2Opus_Trans ? hmatrix.v_basis_tree : hmatrix.u_basis_tree);
    TBasisTree<hw> &v_basis_tree = (hmatrix.sym || trans == H2Opus_Trans ? u_basis_tree : hmatrix.v_basis_tree);

    BasisTreeLevelData &u_level_data = u_basis_tree.level_data;
    BasisTreeLevelData &v_level_data = v_basis_tree.level_data;
    HNodeTreeLevelData &hnode_level_data = hmatrix.hnodes.level_data;

    int dense_nodes = hmatrix.hnodes.num_dense_leaves;
    int coupling_nodes = hmatrix.hnodes.num_rank_leaves;

    H2OpusWorkspaceState dummy_ws;
    h2opusComputeStream_t stream = h2opus_handle->getMainStream();
    h2opusWorkspace_t h2opus_ws = h2opus_handle->getWorkspace();

    hgemv_get_workspace(u_level_data, v_level_data, hnode_level_data, dense_nodes, coupling_nodes, num_vectors, hw,
                        dummy_ws, workspace, stream, h2opus_ws);
}

//////////////////////////////////////////////
// Interface routines
//////////////////////////////////////////////
H2OpusWorkspaceState hgemv_workspace(HMatrix &hmatrix, int trans, int num_vectors)
{
    return hgemv_workspace_template<H2OPUS_HWTYPE_CPU>(hmatrix, trans, num_vectors);
}

void hgemv_get_workspace(HMatrix &hmatrix, int trans, int num_vectors, HgemvWorkspace &workspace,
                         h2opusHandle_t h2opus_handle)
{
    hgemv_get_workspace_template<H2OPUS_HWTYPE_CPU>(hmatrix, trans, num_vectors, workspace, h2opus_handle);
}

#ifdef H2OPUS_USE_GPU
H2OpusWorkspaceState hgemv_workspace(HMatrix_GPU &hmatrix, int trans, int num_vectors)
{
    return hgemv_workspace_template<H2OPUS_HWTYPE_GPU>(hmatrix, trans, num_vectors);
}

void hgemv_get_workspace(HMatrix_GPU &hmatrix, int trans, int num_vectors, HgemvWorkspace &workspace,
                         h2opusHandle_t h2opus_handle)
{
    hgemv_get_workspace_template<H2OPUS_HWTYPE_GPU>(hmatrix, trans, num_vectors, workspace, h2opus_handle);
}
#endif
