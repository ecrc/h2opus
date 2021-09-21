#ifndef __H2OPUS_TLR_GEMV_WORKSPACE_H__
#define __H2OPUS_TLR_GEMV_WORKSPACE_H__

#include <h2opus/core/h2opus_compute_stream.h>
#include <h2opus/core/h2opus_workspace.h>
#include <h2opus/core/tlr/tlr_defs.h>

template <class T> struct H2OpusTLRGemvWorkspace
{
    // Workspace for the dense part - this will be done in parallel with the
    // low rank parts, so it needs its own workspace
    T **dense_input_ptrs, **dense_output_ptrs;
    int *dense_bs_batch, *dense_rows_batch, *dense_cols_batch;
    int *dense_ldx_batch, *dense_ldy_batch, *num_vec_batch;

    // Workspace for low rank part - allow multiple block columns to be multiplied at the same
    // time to increase parallelism. Final result is reduced into the output after
    // the dense part is done
    int num_parallel_columns;
    T *lr_VY_base_data, *lr_UZ_base_data;
    T **lr_VY_ptrs, **lr_UZ_ptrs, **lr_input_ptrs, **U_ptrs, **V_ptrs;
    int *lr_rank_batch, *lr_max_rank_batch;
    int *lr_bs_batch, *lr_rows_batch, *lr_cols_batch;
    int *lr_ldx_batch, *lr_ldy_batch, *lr_num_vec_batch;
};

template <class T, int hw>
void tlr_gemv_dense_workspace(int n_block, H2OpusTLRGemvWorkspace<T> *workspace, h2opusWorkspace_t h2opus_ws,
                              h2opusComputeStream_t stream)
{
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(dense_bs_batch), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(dense_rows_batch), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(dense_cols_batch), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(dense_ldx_batch), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(dense_ldy_batch), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(num_vec_batch), hw);

    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(dense_input_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(dense_output_ptrs), hw);
}

template <class T, int hw>
void tlr_gemv_lr_workspace(int n, int n_block, int max_rank, int num_vectors, int num_parallel_columns,
                           H2OpusTLRGemvWorkspace<T> *workspace, h2opusWorkspace_t h2opus_ws,
                           h2opusComputeStream_t stream)
{
    int lr_vy_entries = n_block * max_rank * num_vectors * num_parallel_columns;
    int lr_uz_entries = n * num_vectors * num_parallel_columns;
    int dim_entries = n_block * num_parallel_columns;
    int ptr_entries = n_block * num_parallel_columns;

    h2opus_ws->allocateEntries<T>(lr_vy_entries, H2OPUS_TLR_WS_PTR(lr_VY_base_data), hw);
    h2opus_ws->allocateEntries<T>(lr_uz_entries, H2OPUS_TLR_WS_PTR(lr_UZ_base_data), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(lr_rank_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(lr_max_rank_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(lr_bs_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(lr_rows_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(lr_cols_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(lr_ldx_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(lr_ldy_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(lr_num_vec_batch), hw);

    h2opus_ws->allocatePointerEntries<T>(ptr_entries, H2OPUS_TLR_WS_PTR(lr_input_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(ptr_entries, H2OPUS_TLR_WS_PTR(lr_VY_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(ptr_entries, H2OPUS_TLR_WS_PTR(lr_UZ_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(ptr_entries, H2OPUS_TLR_WS_PTR(U_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(ptr_entries, H2OPUS_TLR_WS_PTR(V_ptrs), hw);

    if (workspace)
        workspace->num_parallel_columns = num_parallel_columns;
}

template <class T, int hw>
H2OpusWorkspaceState tlr_gemv_get_workspace(TTLR_Matrix<T, hw> &A, int num_vectors, int num_parallel_columns,
                                            H2OpusTLRGemvWorkspace<T> *workspace, h2opusHandle_t h2opus_handle)
{
    h2opusComputeStream_t stream = h2opus_handle->getMainStream();
    h2opusWorkspace_t h2opus_ws = h2opus_handle->getWorkspace();
    h2opus_ws->resetAllocatorState();

    tlr_gemv_dense_workspace<T, hw>(A.n_block, workspace, h2opus_ws, stream);
    tlr_gemv_lr_workspace<T, hw>(A.getPaddedDim(), A.n_block, A.max_rank, num_vectors, num_parallel_columns, workspace,
                                 h2opus_ws, stream);

    return h2opus_ws->getAllocatorState();
}

template <class T, int hw>
H2OpusWorkspaceState tlr_gemv_get_workspace(TTLR_Matrix<T, hw> &A, int num_vectors, int num_parallel_columns,
                                            H2OpusTLRGemvWorkspace<T> &workspace, h2opusHandle_t h2opus_handle)
{
    return tlr_gemv_get_workspace<T, hw>(A, num_vectors, num_parallel_columns, &workspace, h2opus_handle);
}

template <class T, int hw>
H2OpusWorkspaceState tlr_gemv_workspace(TTLR_Matrix<T, hw> &A, int num_vectors, int num_parallel_columns,
                                        h2opusHandle_t h2opus_handle)
{
    return tlr_gemv_get_workspace<T, hw>(A, num_vectors, num_parallel_columns, NULL, h2opus_handle);
}
#endif
