#ifndef __HMATRIX_LOW_RANK_UPDATE_H__
#define __HMATRIX_LOW_RANK_UPDATE_H__

#include <h2opus/util/thrust_wrappers.h>
#include <h2opus/core/hmatrix.h>
#include <h2opus/core/hlru_structs.h>

//////////////////////////////////////////////////////////////////////
// CPU
//////////////////////////////////////////////////////////////////////
/**
 * @brief Applies a set of local low rank updates to a hierarchical matrix
 *
 * @param[in, out] hmatrix The hierarchical matrix whose basis will be updated
 * @param[in] update The set of low rank updates
 * @param[in] h2opus_handle The H2Opus handle
 */
int hlru_sym(HMatrix &hmatrix, LowRankUpdate &update, h2opusHandle_t handle);

/**
 * @brief Applies a set of dense updates to the dense blocks of a hierarchical matrix
 *
 * @param[in, out] hmatrix The hierarchical matrix whose dense nodes will be updated
 * @param[in] update The set of dense block updates
 * @param[in] h2opus_handle The H2Opus handle
 */
void hlru_dense_block_update(HMatrix &hmatrix, DenseBlockUpdate &update, h2opusHandle_t handle);

/**
 * @brief Applies a symmetric global low rank update USU^T to a hierarchical matrix, where S is a scaled identity matrix
 *
 * @param[in, out] hmatrix The hierarchical matrix whose basis will be updated
 * @param[in] U The factor of the symmetric global low rank update
 * @param[in] ldu Leading dimension of U
 * @param[in] rank The number of columns of U
 * @param[in] s The scale of the identity matrix (S = s * I)
 * @param[in] h2opus_handle The H2Opus handle
 */
void hlru_sym_global(HMatrix &hmatrix, const H2Opus_Real *U, int ldu, int rank, H2Opus_Real s, h2opusHandle_t handle);

//////////////////////////////////////////////////////////////////////
// GPU
//////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU
int hlru_sym(HMatrix_GPU &hmatrix, LowRankUpdate_GPU &update, h2opusHandle_t handle);
void hlru_dense_block_update(HMatrix_GPU &hmatrix, DenseBlockUpdate_GPU &update, h2opusHandle_t handle);
void hlru_sym_global(HMatrix_GPU &hmatrix, const H2Opus_Real *U, int ldu, int rank, H2Opus_Real s, h2opusHandle_t handle);
#endif

#endif
