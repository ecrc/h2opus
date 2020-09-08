#ifndef __HORTHOG_WORKSPACE_H__
#define __HORTHOG_WORKSPACE_H__

#include <h2opus/core/h2opus_handle.h>
#include <h2opus/core/hmatrix.h>
#include <vector>

#define PROJECTION_MAX_NODES 1000

struct HorthogWorkspace
{
    std::vector<H2Opus_Real *> Tu_hat, Tv_hat;

    // For reallocation
    H2Opus_Real *realloc_buffer;

    // For the projection phase
    H2Opus_Real *TS;
    H2Opus_Real **TS_array, **Tu_array, **Tv_array, **S_array, **S_new_array;

    // For the upsweep phase
    H2Opus_Real *TE_data, *TE_tau;
    H2Opus_Real **ptr_TE, **ptr_T, **ptr_E;

    // The new ranks if they have to change due to wide matrices being orthogonalized
    std::vector<int> u_new_ranks, v_new_ranks;

    bool symmetric;
};

// More general routines used by other internal h2opus routines
H2OpusWorkspaceState horthog_get_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                                           HNodeTreeLevelData &hnode_level_data,
                                           HNodeTreeLevelData *offdiag_hnode_level_data, int max_children,
                                           bool symmetric, HorthogWorkspace &workspace, h2opusHandle_t h2opus_handle,
                                           int hw);

H2OpusWorkspaceState horthog_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                                       HNodeTreeLevelData &hnode_level_data,
                                       HNodeTreeLevelData *offdiag_hnode_level_data, std::vector<int> &u_new_ranks,
                                       std::vector<int> &v_new_ranks, int max_children, bool symmetric, int hw);

void horthog_get_workspace(HMatrix &hmatrix, HorthogWorkspace &workspace, h2opusHandle_t h2opus_handle);
H2OpusWorkspaceState horthog_workspace(HMatrix &hmatrix);

#ifdef H2OPUS_USE_GPU
void horthog_get_workspace(HMatrix_GPU &hmatrix, HorthogWorkspace &workspace, h2opusHandle_t h2opus_handle);
H2OpusWorkspaceState horthog_workspace(HMatrix_GPU &hmatrix);
#endif

#endif
