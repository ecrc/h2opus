#ifndef __HCOMPRESS_WORKSPACE_H__
#define __HCOMPRESS_WORKSPACE_H__

#include <h2opus/core/h2opus_handle.h>
#include <h2opus/core/hmatrix.h>
#include <vector>

////////////////////////////////////////////////////////////////////////
// Workspace data structures for each phase
////////////////////////////////////////////////////////////////////////
// For the Upsweep phase
struct HcompressUpsweepWorkspace
{
    std::vector<int> new_ranks;
    std::vector<H2Opus_Real *> T_hat, Z_hat;

    H2Opus_Real **ptr_A, **ptr_B, **ptr_C, **ptr_D;
    H2Opus_Real *UZ_data, *TE_data, *tau_data;

    int *row_array, *col_array, *ld_array;
    int *ranks_array;
};

// For the optimal basis generation
struct HcompressOptimalBGenWorkspace
{
    H2Opus_Real *stacked_tau_data, *stacked_node_data;
    int *stacked_node_row, *stacked_node_col, *stacked_node_ld;
    H2Opus_Real **stacked_node_ptrs, **stacked_node_tau_ptrs, **ptr_ZE, **ptr_Z, **ptr_E;
    H2Opus_Real **ptr_row_data, **ptr_S;
};

// For the projection phase
struct HcompressProjectionWorkspace
{
    H2Opus_Real *TC;
    H2Opus_Real **ptr_A, **ptr_B, **ptr_C, **ptr_D;
};

// For projecting the top level of the basis tree
struct HcompressProjectTopLevelWorkspace
{
    H2Opus_Real *old_transfer;
    H2Opus_Real **ptr_A, **ptr_B, **ptr_C;
};

struct HcompressWorkspace
{
    HcompressOptimalBGenWorkspace optimal_u_bgen, optimal_v_bgen;
    HcompressUpsweepWorkspace u_upsweep, v_upsweep;
    HcompressProjectTopLevelWorkspace u_top_level, v_top_level;
    HcompressProjectionWorkspace projection;

    bool symmetric;
};

////////////////////////////////////////////////////////////////////////
// Internal routines used by the distributed workspace routines
////////////////////////////////////////////////////////////////////////
H2OpusWorkspaceState hcompress_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                                         HNodeTreeLevelData &hnode_level_data, std::vector<int> &diagonal_max_nodes,
                                         HNodeTreeLevelData *offdiag_hnode_level_data,
                                         std::vector<int> *offdiag_max_nodes, bool symmetric, int hw);

H2OpusWorkspaceState hcompress_get_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                                             HNodeTreeLevelData &hnode_level_data, std::vector<int> &diagonal_max_nodes,
                                             HNodeTreeLevelData *offdiag_hnode_level_data,
                                             std::vector<int> *offdiag_max_nodes, bool symmetric,
                                             HcompressWorkspace &workspace, h2opusHandle_t h2opus_handle, int hw);

H2OpusWorkspaceState hcompress_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                                         HNodeTreeLevelData &hnode_level_data, std::vector<int> &diagonal_max_row_nodes,
                                         std::vector<int> &diagonal_max_col_nodes,
                                         HNodeTreeLevelData *offdiag_hnode_level_data,
                                         std::vector<int> *offdiag_max_row_nodes,
                                         std::vector<int> *offdiag_max_col_nodes, bool symmetric, int hw);

H2OpusWorkspaceState hcompress_get_workspace(BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                                             HNodeTreeLevelData &hnode_level_data,
                                             std::vector<int> &diagonal_max_row_nodes,
                                             std::vector<int> &diagonal_max_col_nodes,
                                             HNodeTreeLevelData *offdiag_hnode_level_data,
                                             std::vector<int> *offdiag_max_row_nodes,
                                             std::vector<int> *offdiag_max_col_nodes, bool symmetric,
                                             HcompressWorkspace *workspace, h2opusHandle_t h2opus_handle, int hw);
////////////////////////////////////////////////////////////////////////
// Main workspace routines
////////////////////////////////////////////////////////////////////////
void hcompress_get_workspace(HMatrix &hmatrix, HcompressWorkspace &workspace, h2opusHandle_t h2opus_handle);
H2OpusWorkspaceState hcompress_workspace(HMatrix &hmatrix);

#ifdef H2OPUS_USE_GPU
void hcompress_get_workspace(HMatrix_GPU &hmatrix, HcompressWorkspace &workspace, h2opusHandle_t h2opus_handle);
H2OpusWorkspaceState hcompress_workspace(HMatrix_GPU &hmatrix);
#endif

#endif
