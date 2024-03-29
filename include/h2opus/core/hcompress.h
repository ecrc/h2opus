#ifndef __HCOMPRESS_H__
#define __HCOMPRESS_H__

#include <h2opus/core/hcompress_workspace.h>
#include <h2opus/core/hmatrix.h>
#include <h2opus/core/hcompress_weightpacket.h>

//////////////////////////////////////////////////////////////////////
// CPU
//////////////////////////////////////////////////////////////////////
/**
 * @brief Compresses the basis of the hierarchical matrix and projects the coupling matrices into the new compact basis
 *
 * @param[in, out] hmatrix The hierarchical matrix whose basis will be compressed
 * @param[in] eps The absolute truncation threshold
 * @param[in] h2opus_handle The H2Opus handle
 */
void hcompress(HMatrix &hmatrix, H2Opus_Real eps, h2opusHandle_t h2opus_handle);

void hcompress(HMatrix &hmatrix, WeightAccelerationPacket &packet, H2Opus_Real eps, h2opusHandle_t h2opus_handle);

void hcompress_generate_optimal_basis(HNodeTree &hnodes, HNodeTree *offdiagonal_hnodes, BSNPointerDirection direction,
                                      BasisTree &basis_tree, HcompressUpsweepWorkspace &upsweep_workspace,
                                      HcompressOptimalBGenWorkspace &bgen_workspace, int start_level, H2Opus_Real eps,
                                      h2opusComputeStream_t stream);

int hcompress_compressed_basis_leaf_rank(BasisTree &basis_tree, H2Opus_Real eps, HcompressUpsweepWorkspace &workspace,
                                         h2opusComputeStream_t stream);

void hcompress_truncate_basis_leaves(BasisTree &basis_tree, int new_rank, HcompressUpsweepWorkspace &workspace,
                                     h2opusComputeStream_t stream);

void hcompress_truncate_basis_level(BasisTree &basis_tree, int new_rank, int level,
                                    HcompressUpsweepWorkspace &workspace, h2opusComputeStream_t stream);

int hcompress_compressed_basis_level_rank(BasisTree &basis_tree, H2Opus_Real eps, int level,
                                          HcompressUpsweepWorkspace &workspace, h2opusComputeStream_t stream);

void hcompress_project_level(HNodeTree &hnodes, int level, size_t u_level_start, size_t v_level_start,
                             H2Opus_Real *Tu_level, int ld_tu, H2Opus_Real *Tv_level, int ld_tv, int *node_u_index,
                             int *node_v_index, HcompressUpsweepWorkspace &u_upsweep_ws,
                             HcompressUpsweepWorkspace &v_upsweep_ws, HcompressProjectionWorkspace &proj_ws,
                             h2opusComputeStream_t stream);

void hcompress_project(HNodeTree &hnodes, BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                       HcompressUpsweepWorkspace &u_upsweep_ws, HcompressUpsweepWorkspace &v_upsweep_ws,
                       HcompressProjectionWorkspace &proj_ws, h2opusComputeStream_t stream);

//////////////////////////////////////////////////////////////////////
// GPU
//////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU
void hcompress(HMatrix_GPU &hmatrix, H2Opus_Real eps, h2opusHandle_t h2opus_handle);

void hcompress(HMatrix_GPU &hmatrix, WeightAccelerationPacket_GPU &packet, H2Opus_Real eps,
               h2opusHandle_t h2opus_handle);

void hcompress_generate_optimal_basis(HNodeTree_GPU &hnodes, HNodeTree_GPU *offdiagonal_hnodes,
                                      BSNPointerDirection direction, BasisTree_GPU &basis_tree,
                                      HcompressUpsweepWorkspace &upsweep_workspace,
                                      HcompressOptimalBGenWorkspace &bgen_workspace, int start_level, H2Opus_Real eps,
                                      h2opusComputeStream_t stream);

int hcompress_compressed_basis_leaf_rank(BasisTree_GPU &basis_tree, H2Opus_Real eps,
                                         HcompressUpsweepWorkspace &workspace, h2opusComputeStream_t stream);

void hcompress_truncate_basis_leaves(BasisTree_GPU &basis_tree, int new_rank, HcompressUpsweepWorkspace &workspace,
                                     h2opusComputeStream_t stream);

void hcompress_truncate_basis_level(BasisTree_GPU &basis_tree, int new_rank, int level,
                                    HcompressUpsweepWorkspace &workspace, h2opusComputeStream_t stream);

int hcompress_compressed_basis_level_rank(BasisTree_GPU &basis_tree, H2Opus_Real eps, int level,
                                          HcompressUpsweepWorkspace &workspace, h2opusComputeStream_t stream);

void hcompress_project_level(HNodeTree_GPU &hnodes, int level, size_t u_level_start, size_t v_level_start,
                             H2Opus_Real *Tu_level, int ld_tu, H2Opus_Real *Tv_level, int ld_tv, int *node_u_index,
                             int *node_v_index, HcompressUpsweepWorkspace &u_upsweep_ws,
                             HcompressUpsweepWorkspace &v_upsweep_ws, HcompressProjectionWorkspace &proj_ws,
                             h2opusComputeStream_t stream);

void hcompress_project(HNodeTree_GPU &hnodes, BasisTreeLevelData &u_level_data, BasisTreeLevelData &v_level_data,
                       HcompressUpsweepWorkspace &u_upsweep_ws, HcompressUpsweepWorkspace &v_upsweep_ws,
                       HcompressProjectionWorkspace &proj_ws, h2opusComputeStream_t stream);

#endif

#endif
