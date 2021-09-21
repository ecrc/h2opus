#ifndef __HORTHOG_H__
#define __HORTHOG_H__

#include <h2opus/core/hmatrix.h>
#include <h2opus/core/horthog_workspace.h>

//////////////////////////////////////////////////////////////////////
// CPU
//////////////////////////////////////////////////////////////////////
/**
 * @brief Orthogonalizes the basis of the hierarchical matrix and projects its coupling matrices into the new orthogonal
 * basis
 *
 * @param[in, out] hmatrix The hierarchical matrix whose basis will be orthogonalized
 * @param[in] h2opus_handle The H2Opus handle.
 */
void horthog(HMatrix &hmatrix, h2opusHandle_t h2opus_handle);

void horthog_upsweep_level(BasisTree &basis_tree, HorthogWorkspace &workspace, std::vector<H2Opus_Real *> &T_hat,
                           std::vector<int> &new_ranks, int level, h2opusComputeStream_t stream);

void horthog_upsweep_leaves(BasisTree &basis_tree, HorthogWorkspace &workspace, std::vector<H2Opus_Real *> &T_hat,
                            h2opusComputeStream_t stream);

void horthog_upsweep(BasisTree &basis_tree, HorthogWorkspace &workspace, std::vector<H2Opus_Real *> &T_hat,
                     std::vector<int> &new_ranks, h2opusComputeStream_t stream);

void horthog_stitch(BasisTree &basis_tree, HorthogWorkspace &workspace, std::vector<H2Opus_Real *> &T_hat,
                    std::vector<int> &new_ranks, h2opusComputeStream_t stream);

void horthog_project(HNodeTree &hnodes, int start_level, int end_level, BasisTreeLevelData &u_level_data,
                     BasisTreeLevelData &v_level_data, HorthogWorkspace &workspace, h2opusComputeStream_t stream);

void horthog_project_level(HNodeTree &hnodes, int level, size_t u_level_start, size_t v_level_start,
                           H2Opus_Real *Tu_level, H2Opus_Real *Tv_level, int *node_u_index, int *node_v_index,
                           size_t increment, HorthogWorkspace &workspace, h2opusComputeStream_t stream);

//////////////////////////////////////////////////////////////////////
// GPU
//////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU
void horthog(HMatrix_GPU &hmatrix, h2opusHandle_t h2opus_handle);

void horthog_upsweep_level(BasisTree_GPU &basis_tree, HorthogWorkspace &workspace, std::vector<H2Opus_Real *> &T_hat,
                           std::vector<int> &new_ranks, int level, h2opusComputeStream_t stream);

void horthog_upsweep_leaves(BasisTree_GPU &basis_tree, HorthogWorkspace &workspace, std::vector<H2Opus_Real *> &T_hat,
                            h2opusComputeStream_t stream);

void horthog_upsweep(BasisTree_GPU &basis_tree, HorthogWorkspace &workspace, std::vector<H2Opus_Real *> &T_hat,
                     std::vector<int> &new_ranks, h2opusComputeStream_t stream);

void horthog_stitch(BasisTree_GPU &basis_tree, HorthogWorkspace &workspace, std::vector<H2Opus_Real *> &T_hat,
                    std::vector<int> &new_ranks, h2opusComputeStream_t stream);

void horthog_project(HNodeTree_GPU &hnodes, int start_level, int end_level, BasisTreeLevelData &u_level_data,
                     BasisTreeLevelData &v_level_data, HorthogWorkspace &workspace, h2opusComputeStream_t stream);

void horthog_project_level(HNodeTree_GPU &hnodes, int level, size_t u_level_start, size_t v_level_start,
                           H2Opus_Real *Tu_level, H2Opus_Real *Tv_level, int *node_u_index, int *node_v_index,
                           size_t increment, HorthogWorkspace &workspace, h2opusComputeStream_t stream);
#endif

#endif
