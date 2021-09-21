#ifndef __H2OPUS_DISTRIBUTED_HORTHOG_WORKSPACE_H__
#define __H2OPUS_DISTRIBUTED_HORTHOG_WORKSPACE_H__

#include <h2opus/core/horthog_workspace.h>
#include <h2opus/distributed/distributed_h2opus_handle.h>
#include <h2opus/distributed/distributed_hmatrix.h>

struct DistributedHorthogWorkspace
{
    HorthogWorkspace branch_workspace;
    HorthogWorkspace top_level_workspace;

    // Marshal data for the offdiagonal buffer block copies
    H2Opus_Real **ptr_A, **ptr_B;
};

void distributed_horthog_get_workspace(DistributedHMatrix &dist_hmatrix, DistributedHorthogWorkspace &dist_workspace,
                                       distributedH2OpusHandle_t dist_h2opus_handle);
void distributed_horthog_workspace(DistributedHMatrix &dist_hmatrix, H2OpusWorkspaceState &branch_ws_needed,
                                   H2OpusWorkspaceState &top_level_ws_needed,
                                   distributedH2OpusHandle_t dist_h2opus_handle);

#ifdef H2OPUS_USE_GPU
void distributed_horthog_get_workspace(DistributedHMatrix_GPU &dist_hmatrix,
                                       DistributedHorthogWorkspace &dist_workspace,
                                       distributedH2OpusHandle_t dist_h2opus_handle);

void distributed_horthog_workspace(DistributedHMatrix_GPU &dist_hmatrix, H2OpusWorkspaceState &branch_ws_needed,
                                   H2OpusWorkspaceState &top_level_ws_needed,
                                   distributedH2OpusHandle_t dist_h2opus_handle);
#endif

#endif
