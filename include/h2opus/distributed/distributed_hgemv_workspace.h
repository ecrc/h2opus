#ifndef __H2OPUS_DISTRIBUTED_HGEMV_WORKSPACE_H__
#define __H2OPUS_DISTRIBUTED_HGEMV_WORKSPACE_H__

#include <h2opus/core/hgemv_workspace.h>
#include <h2opus/distributed/distributed_h2opus_handle.h>
#include <h2opus/distributed/distributed_hgemv_communication.h>
#include <h2opus/distributed/distributed_hmatrix.h>

struct DistributedHgemvWorkspace
{
    HgemvWorkspace branch_workspace;
    HgemvWorkspace top_level_workspace;

    // Temporary scatter buffer for the roots of the yhat tree
    H2Opus_Real *yhat_scatter_root;

    // Marshal data for the offdiagonal buffer block copies
    H2Opus_Real **ptr_A, **ptr_B;
    int *ptr_m, *ptr_n, *ptr_lda, *ptr_ldb;
};

void distributed_hgemv_get_workspace(DistributedHMatrix &dist_hmatrix, int num_vectors,
                                     DistributedHgemvWorkspace &dist_workspace,
                                     distributedH2OpusHandle_t dist_h2opus_handle);
void distributed_hgemv_workspace(DistributedHMatrix &dist_hmatrix, int num_vectors,
                                 H2OpusWorkspaceState &branch_ws_needed, H2OpusWorkspaceState &top_level_ws_needed,
                                 distributedH2OpusHandle_t dist_h2opus_handle);

#ifdef H2OPUS_USE_GPU
void distributed_hgemv_get_workspace(DistributedHMatrix_GPU &dist_hmatrix, int num_vectors,
                                     DistributedHgemvWorkspace &dist_workspace,
                                     distributedH2OpusHandle_t dist_h2opus_handle);

void distributed_hgemv_workspace(DistributedHMatrix_GPU &dist_hmatrix, int num_vectors,
                                 H2OpusWorkspaceState &branch_ws_needed, H2OpusWorkspaceState &top_level_ws_needed,
                                 distributedH2OpusHandle_t dist_h2opus_handle);
#endif

#endif
