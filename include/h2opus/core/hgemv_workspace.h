#ifndef __HGEMV_WORKSPACE_H__
#define __HGEMV_WORKSPACE_H__

#include <h2opus/core/h2opus_handle.h>
#include <h2opus/core/hmatrix.h>
#include <vector>

// Temporary data structure containing the xhat and yhat tree data
struct VectorTree
{
    // Data is stored flattened by level and allocated from workspace
    std::vector<H2Opus_Real *> data;
    unsigned int allocated_entries;
};

struct BatchGemmMarshalledData
{
    // Arrays of gemm opertation dimensions
    int *m_batch, *n_batch, *k_batch, *lda_batch, *ldb_batch, *ldc_batch;

    // A, B, and C ptrs for batch gemms
    H2Opus_Real **A_ptrs, **B_ptrs, **C_ptrs;
};

struct HgemvWorkspace
{
    VectorTree xhat, yhat;

    BatchGemmMarshalledData low_rank_gemms;
    BatchGemmMarshalledData dense_gemms;
};

// More general routine used by other internal h2opus routines
H2OpusWorkspaceState hgemv_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                                     HNodeTreeLevelData &hnode_level_data, int dense_nodes, int coupling_nodes,
                                     int num_vectors, int hw);

H2OpusWorkspaceState hgemv_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                                     HNodeTreeLevelData &hnode_level_data, HNodeTreeLevelData *offdiag_hnode_level_data,
                                     int dense_nodes, int coupling_nodes, int num_vectors, int hw);

void hgemv_get_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                         HNodeTreeLevelData &hnode_level_data, HNodeTreeLevelData *offdiag_hnode_level_data,
                         int dense_nodes, int coupling_nodes, int num_vectors, int hw,
                         H2OpusWorkspaceState &ws_allocated, HgemvWorkspace &workspace, h2opusComputeStream_t stream,
                         h2opusWorkspace_t h2opus_ws);

void hgemv_get_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                         HNodeTreeLevelData &hnode_level_data, int dense_nodes, int coupling_nodes, int num_vectors,
                         int hw, H2OpusWorkspaceState &ws_allocated, HgemvWorkspace &workspace,
                         h2opusComputeStream_t stream, h2opusWorkspace_t h2opus_ws);

void getHgemvVectorTree(VectorTree &vector_tree, BasisTreeLevelData &level_data, int num_vectors, H2Opus_Real *ws_base,
                        h2opusComputeStream_t stream, int hw);

// Convenience routines
void hgemv_get_workspace(HMatrix &hmatrix, int trans, int num_vectors, HgemvWorkspace &workspace,
                         h2opusHandle_t h2opus_handle);
H2OpusWorkspaceState hgemv_workspace(HMatrix &hmatrix, int trans, int num_vectors);

#ifdef H2OPUS_USE_GPU
void hgemv_get_workspace(HMatrix_GPU &hmatrix, int trans, int num_vectors, HgemvWorkspace &workspace,
                         h2opusHandle_t h2opus_handle);
H2OpusWorkspaceState hgemv_workspace(HMatrix_GPU &hmatrix, int trans, int num_vectors);
#endif

#endif
