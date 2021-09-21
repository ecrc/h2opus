#ifndef __H2OPUS_TLR_TRSM_H__
#define __H2OPUS_TLR_TRSM_H__

#include <h2opus/core/tlr/tlr_defs.h>
#include <h2opus/core/tlr/tlr_struct.h>
#include <h2opus/core/tlr/tlr_trsm_workspace.h>
#include <h2opus/core/tlr/tlr_trsm_marshal.h>
#include <h2opus/core/tlr/tlr_batch.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Left trsm
//////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw, bool transpose, bool lower>
void tlr_trsm_left(T alpha, TTLR_Matrix<T, hw> &A, int num_vectors, T *B, int ldb,
                   H2OpusTLRDenseTrsmWorkspace<T> &workspace, h2opusComputeStream_t stream)
{
    int block_size = A.block_size, n_block = A.n_block, max_rank = A.max_rank;
    T **block_U_ptrs = vec_ptr(A.block_U_ptrs), **block_V_ptrs = vec_ptr(A.block_V_ptrs);
    int *block_ranks = vec_ptr(A.block_ranks);

    int start, end, inc;
    tlr_trsm_get_loop_param<transpose>(n_block, start, end, inc);
    char uplo = (lower ? H2Opus_Lower : H2Opus_Upper);
    char trans = (transpose ? H2Opus_Trans : H2Opus_NoTrans);

    T alpha_mod = alpha;
    T **Bi_ptrs = workspace.lr_Bi_ptrs, **Bj_ptrs = workspace.lr_Bj_ptrs;
    T **VB_ptrs = workspace.lr_VB_ptrs, **U_ptrs = workspace.U_ptrs, **V_ptrs = workspace.V_ptrs;

    int *bi_rows_batch = workspace.lr_bi_rows_batch, *bj_rows_batch = workspace.lr_bj_rows_batch;
    int *num_vec_batch = workspace.lr_num_vec_batch, *bs_batch = workspace.lr_bs_batch;
    int *ldb_batch = workspace.lr_ldb_batch, *max_rank_batch = workspace.lr_max_rank_batch;
    int *rank_batch = workspace.lr_rank_batch;
    int num_blocks = n_block - 1;

    for (int i = start; i != end; i += inc)
    {
        // Diagonal block trsm on B_i
        T *A_ii = A.getDiagonalBlockHostPtr(i);
        T *B_i = B + i * block_size;
        blas_trsm<T, hw>(stream, H2Opus_Left, uplo, trans, H2Opus_NonUnit, block_size, num_vectors, alpha_mod, A_ii,
                         block_size, B_i, ldb);

        // Update B_j for j = start+1:end:inc by setting
        // For a lower TLR matrix:
        //       B_j = B_j - L(j, i) B_i = U(j,i) * V^T(j,i) * B_i if not transpose and
        //       B_j = B_j - L(i, j)^T B_i = V(j,i) * U^T(j,i) * B_i if transpose
        tlr_trsm_dense_marshal_updateB<T, hw, transpose>(i + inc, inc, i, block_size, n_block, block_V_ptrs,
                                                         block_U_ptrs, block_ranks, B, Bi_ptrs, Bj_ptrs, bi_rows_batch,
                                                         bj_rows_batch, U_ptrs, V_ptrs, rank_batch, num_blocks, stream);

        // VB = V^T * B_i
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, rank_batch, num_vec_batch,
                                                       bi_rows_batch, max_rank, num_vectors, block_size, (T)1,
                                                       (const T **)V_ptrs, bs_batch, (const T **)Bi_ptrs, ldb_batch, 0,
                                                       VB_ptrs, max_rank_batch, num_blocks));

        // B_j -= U * VB
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, bj_rows_batch,
                                                       num_vec_batch, rank_batch, block_size, num_vectors, max_rank,
                                                       (T)-1, (const T **)U_ptrs, bs_batch, (const T **)VB_ptrs,
                                                       max_rank_batch, alpha_mod, Bj_ptrs, ldb_batch, num_blocks));

        // The first iteration scaled the input, so we don't need to scale it anymore
        alpha_mod = 1;
        num_blocks--;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Main routine
//////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw>
void tlr_trsm(int side, int transa, T alpha, TTLR_Matrix<T, hw> &A, int num_vectors, T *B, int ldb,
              h2opusHandle_t h2opus_handle)
{
    assert(A.type == H2OpusTLR_LowerTriangular); // || A.type == H2OpusTLR_UpperTriangular);

    H2OpusWorkspaceState ws_needed = tlr_trsm_workspace<T, hw>(side, transa, A, num_vectors, ldb, h2opus_handle);
    H2OpusWorkspaceState ws_allocated = h2opus_handle->getWorkspaceState();
    if (ws_allocated < ws_needed)
        h2opus_handle->setWorkspaceState(ws_needed);

    H2OpusTLRDenseTrsmWorkspace<T> workspace;
    tlr_trsm_get_workspace<T, hw>(side, transa, A, num_vectors, ldb, workspace, h2opus_handle);

    h2opusComputeStream_t main_stream = h2opus_handle->getMainStream();
    // h2opusComputeStream_t low_priority_stream = h2opus_handle->getLowPriorityStream();

    if (side == H2Opus_Left)
    {
        // Solve op(L_ii) B_i = B_i
        if (transa == H2Opus_Trans)
            tlr_trsm_left<T, hw, true, true>(alpha, A, num_vectors, B, ldb, workspace, main_stream);
        else if (transa == H2Opus_NoTrans)
            tlr_trsm_left<T, hw, false, true>(alpha, A, num_vectors, B, ldb, workspace, main_stream);
    }
}

#endif
