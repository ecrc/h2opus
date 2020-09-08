#ifndef __HGEMV_H__
#define __HGEMV_H__

#include <h2opus/core/h2opus_handle.h>
#include <h2opus/core/hgemv_workspace.h>
#include <h2opus/core/hmatrix.h>

//////////////////////////////////////////////////////////////////////
// CPU
//////////////////////////////////////////////////////////////////////
/**
 * @brief Performs the operation Y = beta * Y + alpha * op(H) * X
 *
 * @param[in] trans op(H) = H^T if trans == H2Opus_Trans and op(H) = H otherwise
 * @param[in] alpha is the factor that scales the input vector
 * @param[in] hmatrix the hierarchical matrix H
 * @param[in] X the set of input vectors
 * @param[in] ldx Leading dimension of X
 * @param[in] beta is the factor that scales Y
 * @param[out] Y the set of output vectors
 * @param[in] ldy Leading dimension of Y
 * @param[in] num_vectors Number of columns in X and Y
 * @param[in] h2opus_handle The H2Opus handle.
 */
void hgemv(int trans, H2Opus_Real alpha, HMatrix &hmatrix, H2Opus_Real *X, int ldx, H2Opus_Real beta, H2Opus_Real *Y,
           int ldy, int num_vectors, h2opusHandle_t h2opus_handle);

void hgemv_upsweep(H2Opus_Real alpha, BasisTree &basis_tree, H2Opus_Real *X, int ldx, int num_vectors,
                   HgemvWorkspace &workspace, h2opusComputeStream_t stream);

void hgemv_downsweep(BasisTree &basis_tree, H2Opus_Real *Y, int ldy, int num_vectors, HgemvWorkspace &workspace,
                     h2opusComputeStream_t stream);

void hgemv_mult_level(HNodeTree &hnodes, int level, int num_vectors, size_t u_index_offset, size_t v_index_offset,
                      H2Opus_Real *xhat_level, H2Opus_Real *yhat_level, BatchGemmMarshalledData &marshal_data,
                      typename THNodeTree<H2OPUS_HWTYPE_CPU>::HNodeTreeBSNData *bsn_data, int *column_basis_indexes,
                      int *row_basis_indexes, int kblas_trans_mode, h2opusComputeStream_t stream);

void hgemv_mult(int trans, HNodeTree &hnodes, int start_level, int end_level, int num_vectors, BasisTree &u_basis_tree,
                BasisTree &v_basis_tree, HgemvWorkspace &workspace, h2opusComputeStream_t stream);

void hgemv_denseMult(int kblas_trans_mode, H2Opus_Real alpha, HNodeTree &hnodes, H2Opus_Real *X, int ldx,
                     H2Opus_Real beta, H2Opus_Real *Y, int ldy, int num_vectors, int *node_u_start, int *node_v_start,
                     int *node_u_len, int *node_v_len, int *column_basis_indexes, int *row_basis_indexes,
                     typename THNodeTree<H2OPUS_HWTYPE_CPU>::HNodeTreeBSNData *bsn_data, HgemvWorkspace &workspace,
                     h2opusComputeStream_t stream);

void hgemv_denseMult(int trans, H2Opus_Real alpha, HNodeTree &hnodes, H2Opus_Real *X, int ldx, H2Opus_Real beta,
                     H2Opus_Real *Y, int ldy, int num_vectors, BasisTreeLevelData &u_level_data,
                     BasisTreeLevelData &v_level_data, HgemvWorkspace &workspace, h2opusComputeStream_t stream);

void hgemv_upsweep_leaves(H2Opus_Real alpha, BasisTree &basis_tree, H2Opus_Real *X, int ldx, int num_vectors,
                          HgemvWorkspace &workspace, h2opusComputeStream_t stream);

void hgemv_upsweep_level(BasisTree &basis_tree, int level, int num_vectors, HgemvWorkspace &workspace,
                         h2opusComputeStream_t stream);

//////////////////////////////////////////////////////////////////////
// GPU
//////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU
void hgemv(int trans, H2Opus_Real alpha, HMatrix_GPU &hmatrix, H2Opus_Real *X, int ldx, H2Opus_Real beta,
           H2Opus_Real *Y, int ldy, int num_vectors, h2opusHandle_t h2opus_handle);

void hgemv_upsweep(H2Opus_Real alpha, BasisTree_GPU &basis_tree, H2Opus_Real *X, int ldx, int num_vectors,
                   HgemvWorkspace &workspace, h2opusComputeStream_t stream);

void hgemv_downsweep(BasisTree_GPU &basis_tree, H2Opus_Real *Y, int ldy, int num_vectors, HgemvWorkspace &workspace,
                     h2opusComputeStream_t stream);

void hgemv_mult_level(HNodeTree_GPU &hnodes, int level, int num_vectors, size_t u_index_offset, size_t v_index_offset,
                      H2Opus_Real *xhat_level, H2Opus_Real *yhat_level, BatchGemmMarshalledData &marshal_data,
                      typename THNodeTree<H2OPUS_HWTYPE_GPU>::HNodeTreeBSNData *bsn_data, int *column_basis_indexes,
                      int *row_basis_indexes, int kblas_trans_mode, h2opusComputeStream_t stream);

void hgemv_mult(int trans, HNodeTree_GPU &hnodes, int start_level, int end_level, int num_vectors,
                BasisTree_GPU &u_basis_tree, BasisTree_GPU &v_basis_tree, HgemvWorkspace &workspace,
                h2opusComputeStream_t stream);

void hgemv_denseMult(int kblas_trans_mode, H2Opus_Real alpha, HNodeTree_GPU &hnodes, H2Opus_Real *X, int ldx,
                     H2Opus_Real beta, H2Opus_Real *Y, int ldy, int num_vectors, int *node_u_start, int *node_v_start,
                     int *node_u_len, int *node_v_len, int *column_basis_indexes, int *row_basis_indexes,
                     typename THNodeTree<H2OPUS_HWTYPE_GPU>::HNodeTreeBSNData *bsn_data, HgemvWorkspace &workspace,
                     h2opusComputeStream_t stream);

void hgemv_denseMult(int trans, H2Opus_Real alpha, HNodeTree_GPU &hnodes, H2Opus_Real *X, int ldx, H2Opus_Real beta,
                     H2Opus_Real *Y, int ldy, int num_vectors, BasisTreeLevelData &u_level_data,
                     BasisTreeLevelData &v_level_data, HgemvWorkspace &workspace, h2opusComputeStream_t stream);

void hgemv_upsweep_leaves(H2Opus_Real alpha, BasisTree_GPU &basis_tree, H2Opus_Real *X, int ldx, int num_vectors,
                          HgemvWorkspace &workspace, h2opusComputeStream_t stream);

void hgemv_upsweep_level(BasisTree_GPU &basis_tree, int level, int num_vectors, HgemvWorkspace &workspace,
                         h2opusComputeStream_t stream);

#endif

#endif
