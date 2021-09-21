#ifndef __H2OPUS_TLR_TRSM_WORKSPACE_H__
#define __H2OPUS_TLR_TRSM_WORKSPACE_H__

#include <h2opus/core/h2opus_compute_stream.h>
#include <h2opus/core/h2opus_workspace.h>
#include <h2opus/core/tlr/tlr_defs.h>

template <class T> struct H2OpusTLRDenseTrsmWorkspace
{
    // Workspace for low rank part
    T *lr_VB_base_data;
    T **lr_VB_ptrs, **lr_Bi_ptrs, **lr_Bj_ptrs, **U_ptrs, **V_ptrs;
    int *lr_max_rank_batch, *lr_bs_batch, *lr_ldb_batch, *lr_num_vec_batch;
    int *lr_rank_batch, *lr_bi_rows_batch, *lr_bj_rows_batch;
};

template <class T, int hw>
void tlr_trsm_lr_workspace(int n_block, int block_size, int max_rank, int num_vectors, int ldb,
                           H2OpusTLRDenseTrsmWorkspace<T> *workspace, h2opusWorkspace_t h2opus_ws,
                           h2opusComputeStream_t stream)
{
    int lr_vb_entries = max_rank * num_vectors;
    int dim_entries = n_block;
    int ptr_entries = n_block;

    h2opus_ws->allocateEntries<T>(lr_vb_entries * n_block, H2OPUS_TLR_WS_PTR(lr_VB_base_data), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(lr_max_rank_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(lr_bs_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(lr_ldb_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(lr_num_vec_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(lr_rank_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(lr_bi_rows_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(lr_bj_rows_batch), hw);

    h2opus_ws->allocatePointerEntries<T>(ptr_entries, H2OPUS_TLR_WS_PTR(lr_VB_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(ptr_entries, H2OPUS_TLR_WS_PTR(lr_Bi_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(ptr_entries, H2OPUS_TLR_WS_PTR(lr_Bj_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(ptr_entries, H2OPUS_TLR_WS_PTR(U_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(ptr_entries, H2OPUS_TLR_WS_PTR(V_ptrs), hw);

    if (workspace)
    {
        // Doesn't change through the algorithm, so set it here
        generateArrayOfPointers(workspace->lr_VB_base_data, workspace->lr_VB_ptrs, lr_vb_entries, n_block, stream, hw);
        fillArray(workspace->lr_max_rank_batch, n_block, max_rank, stream, hw);
        fillArray(workspace->lr_bs_batch, n_block, block_size, stream, hw);
        fillArray(workspace->lr_ldb_batch, n_block, ldb, stream, hw);
        fillArray(workspace->lr_num_vec_batch, n_block, num_vectors, stream, hw);
    }
}

template <class T, int hw>
H2OpusWorkspaceState tlr_trsm_get_workspace(int side, int transa, TTLR_Matrix<T, hw> &A, int num_vectors, int ldb,
                                            H2OpusTLRDenseTrsmWorkspace<T> *workspace, h2opusHandle_t h2opus_handle)
{
    h2opusComputeStream_t stream = h2opus_handle->getMainStream();
    h2opusWorkspace_t h2opus_ws = h2opus_handle->getWorkspace();
    h2opus_ws->resetAllocatorState();

    tlr_trsm_lr_workspace<T, hw>(A.n_block, A.block_size, A.max_rank, num_vectors, ldb, workspace, h2opus_ws, stream);

    return h2opus_ws->getAllocatorState();
}

template <class T, int hw>
H2OpusWorkspaceState tlr_trsm_get_workspace(int side, int transa, TTLR_Matrix<T, hw> &A, int num_vectors, int ldb,
                                            H2OpusTLRDenseTrsmWorkspace<T> &workspace, h2opusHandle_t h2opus_handle)
{
    return tlr_trsm_get_workspace<T, hw>(side, transa, A, num_vectors, ldb, &workspace, h2opus_handle);
}

template <class T, int hw>
H2OpusWorkspaceState tlr_trsm_workspace(int side, int transa, TTLR_Matrix<T, hw> &A, int num_vectors, int ldb,
                                        h2opusHandle_t h2opus_handle)
{
    return tlr_trsm_get_workspace<T, hw>(side, transa, A, num_vectors, ldb, NULL, h2opus_handle);
}
#endif
