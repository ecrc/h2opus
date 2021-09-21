#ifndef __H2OPUS_TLR_GEMV_H__
#define __H2OPUS_TLR_GEMV_H__

#include <h2opus/core/tlr/tlr_defs.h>
#include <h2opus/core/tlr/tlr_struct.h>
#include <h2opus/core/tlr/tlr_gemv_workspace.h>
#include <h2opus/core/tlr/tlr_gemv_marshal.h>
#include <h2opus/core/tlr/tlr_batch.h>

template <class T, int hw>
void tlr_gemv_dense(int trans, T alpha, TTLR_Matrix<T, hw> &A, T *X, int ldx, T beta, T *Y, int ldy, int num_vectors,
                    H2OpusTLRGemvWorkspace<T> &workspace, h2opusComputeStream_t stream)
{
    int n_block = A.n_block;
    int block_size = A.block_size;

    T **block_ptrs = vec_ptr(A.diagonal_block_ptrs);
    T **input_ptrs = workspace.dense_input_ptrs, **output_ptrs = workspace.dense_output_ptrs;
    int *rows_batch = workspace.dense_rows_batch, *col_batch = workspace.dense_cols_batch;
    int *ldx_batch = workspace.dense_ldx_batch, *ldy_batch = workspace.dense_ldy_batch;
    int *bs_batch = workspace.dense_bs_batch, *num_vec_batch = workspace.num_vec_batch;

    tlr_gemv_marshal_dense<T, hw>(block_size, X, ldx, Y, ldy, num_vectors, input_ptrs, output_ptrs, rows_batch,
                                  col_batch, num_vec_batch, bs_batch, ldx_batch, ldy_batch, n_block, stream);

    check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, trans, H2Opus_NoTrans, rows_batch, num_vec_batch, col_batch,
                                                   block_size, num_vectors, block_size, alpha, (const T **)block_ptrs,
                                                   bs_batch, (const T **)input_ptrs, ldx_batch, beta, output_ptrs,
                                                   ldy_batch, n_block));
}

template <class T, bool transpose, int hw>
void tlr_gemv_low_rank(T alpha, TTLR_Matrix<T, hw> &A, T *X, int ldx, int num_vectors,
                       H2OpusTLRGemvWorkspace<T> &workspace, h2opusComputeStream_t stream)
{
    int num_parallel_columns = workspace.num_parallel_columns;
    int n_block = A.n_block;
    int block_size = A.block_size;
    int max_rank = A.max_rank;
    int n = A.getPaddedDim();

    T **block_U_ptrs = vec_ptr(A.block_U_ptrs), **block_V_ptrs = vec_ptr(A.block_V_ptrs);
    int *block_ranks = vec_ptr(A.block_ranks);

    T *VY_base = workspace.lr_VY_base_data, *UZ_base = workspace.lr_UZ_base_data;
    T **U_ptrs = workspace.U_ptrs, **V_ptrs = workspace.V_ptrs;
    T **input_ptrs = workspace.lr_input_ptrs, **VY_ptrs = workspace.lr_VY_ptrs, **UZ_ptrs = workspace.lr_UZ_ptrs;
    int *rows_batch = workspace.lr_rows_batch, *cols_batch = workspace.lr_cols_batch;
    int *ldx_batch = workspace.lr_ldx_batch, *ldy_batch = workspace.lr_ldy_batch;
    int *bs_batch = workspace.lr_bs_batch, *num_vec_batch = workspace.lr_num_vec_batch;
    int *max_rank_batch = workspace.lr_max_rank_batch, *rank_batch = workspace.lr_rank_batch;

    // Fill up the dims that don't changed from one column to the other
    // They form a n_block x num_parallel_columns column major integer matrix
    tlr_gemv_lr_dim_set<hw>(n, block_size, max_rank, ldx, num_vectors, bs_batch, max_rank_batch, ldy_batch, ldx_batch,
                            num_vec_batch, rows_batch, n_block, num_parallel_columns, stream);

    // Clear the output since we're going to accumulate block products
    fillArray(UZ_base, n * num_vectors * num_parallel_columns, 0, stream, hw);

    int j = 0;
    while (j < n_block)
    {
        int block_cols = std::min(n_block - j, num_parallel_columns);
        int blocks = block_cols * n_block;

        // Marshall all necessary dim and pointer data
        tlr_gemv_lr_mult_marshal<T, transpose, hw>(
            n, block_size, max_rank, num_vectors, X, input_ptrs, VY_base, VY_ptrs, UZ_base, UZ_ptrs, block_U_ptrs,
            U_ptrs, block_V_ptrs, V_ptrs, block_ranks, rank_batch, cols_batch, j, n_block, block_cols, stream);

        // If this is a limiting factor for whatever reasons, then it might be worth it to
        // cache them (in case we do multiple multiplications with the same matrix)
        // Also since the max function forces a synchronous copy from the device, this
        // may prevent the dense work from overlapping on the low priority stream
        int subset_max_rank = getMaxElement(rank_batch, blocks, stream, hw);

        // Z = V^T * X
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, rank_batch, num_vec_batch,
                                                       cols_batch, subset_max_rank, num_vectors, block_size, alpha,
                                                       (const T **)V_ptrs, bs_batch, (const T **)input_ptrs, ldx_batch,
                                                       (T)0, VY_ptrs, max_rank_batch, blocks));

        // Y += A * Z = A * V^T * X
        check_kblas_error(
            (H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, rows_batch, num_vec_batch, rank_batch,
                                         block_size, num_vectors, subset_max_rank, alpha, (const T **)U_ptrs, bs_batch,
                                         (const T **)VY_ptrs, max_rank_batch, (T)1, UZ_ptrs, ldy_batch, blocks));
        j += block_cols;
    }
}

template <class T, int hw>
void tlr_gemv_reduce_output(TTLR_Matrix<T, hw> &A, int num_vectors, H2OpusTLRGemvWorkspace<T> &workspace,
                            h2opusComputeStream_t stream)
{
    int npc = workspace.num_parallel_columns;
    int n = A.getPaddedDim();

    // First entry in workspace dense output ptr array contains the pointer to Y
    T **Y_ptr = workspace.dense_output_ptrs;

    // Low rank mult workspace has all the rest
    int *ldy_batch = workspace.lr_ldy_batch, *rows_batch = workspace.lr_ldy_batch;
    int *cols_batch = workspace.lr_num_vec_batch;

    // We need to generate pointers for the low rank buffers. the low rank workspace
    // pointer arrars have enough space for that
    T **buffer_ptrs = workspace.lr_UZ_ptrs;
    generateArrayOfPointers(workspace.lr_UZ_base_data, buffer_ptrs, n, npc, stream, hw);

    TLR_Batch<T, hw>::reduceMatrixBuffers(1, Y_ptr, ldy_batch, rows_batch, cols_batch, (T)1, buffer_ptrs, rows_batch,
                                          npc, n, num_vectors, 1, stream);
}

template <class T, int hw>
void tlr_gemv(int trans, T alpha, TTLR_Matrix<T, hw> &A, T *X, int ldx, T beta, T *Y, int ldy, int num_vectors, int npc,
              h2opusHandle_t h2opus_handle)
{
    // Use non-transpose code if A is symmetric and transpose operation is requested
    if (A.type == H2OpusTLR_Symmetric && trans == H2Opus_Trans)
        trans = H2Opus_NoTrans;

    H2OpusWorkspaceState ws_needed = tlr_gemv_workspace<T, hw>(A, num_vectors, npc, h2opus_handle);
    H2OpusWorkspaceState ws_allocated = h2opus_handle->getWorkspaceState();
    if (ws_allocated < ws_needed)
        h2opus_handle->setWorkspaceState(ws_needed);

    H2OpusTLRGemvWorkspace<T> workspace;
    tlr_gemv_get_workspace(A, num_vectors, npc, workspace, h2opus_handle);

    h2opusComputeStream_t main_stream = h2opus_handle->getMainStream();
    h2opusComputeStream_t low_priority_stream = h2opus_handle->getLowPriorityStream();

    H2OpusEvents &events = h2opus_handle->getEvents();
    events.allocateEvents<hw>(H2OpusDenseEvent, 1);

    // Make sure the low priority stream waits for any work previously
    // submitted on the main stream before launching any work
    events.recordEvent<hw>(H2OpusDenseEvent, 0, main_stream);
    events.streamWaitEvent<hw>(H2OpusDenseEvent, low_priority_stream, 0);

    // Do the dense part first on a low priority stream
    tlr_gemv_dense<T, hw>(trans, alpha, A, X, ldx, beta, Y, ldy, num_vectors, workspace, low_priority_stream);
    events.recordEvent<hw>(H2OpusDenseEvent, 0, low_priority_stream);

    // Do the low rank part in parallel on the main stream
    if (trans == H2Opus_NoTrans)
        tlr_gemv_low_rank<T, false, hw>(alpha, A, X, ldx, num_vectors, workspace, main_stream);
    else
        tlr_gemv_low_rank<T, true, hw>(alpha, A, X, ldx, num_vectors, workspace, main_stream);

    // Make sure the dense matvecs are done on the low priority stream
    events.streamWaitEvent<hw>(H2OpusDenseEvent, main_stream, 0);

    // Reduce the parallel low rank output buffers into Y
    tlr_gemv_reduce_output<T, hw>(A, num_vectors, workspace, main_stream);
}

#endif
