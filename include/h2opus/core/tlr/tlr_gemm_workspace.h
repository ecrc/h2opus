#ifndef __H2OPUS_TLR_GEMM_WORKSPACE_H__
#define __H2OPUS_TLR_GEMM_WORKSPACE_H__

#include <h2opus/core/tlr/tlr_ara_util.h>

template <class T> struct H2OpusTLRGemmWorkspace
{
    ////////////////////////////////////
    // Workspace for the dense part
    ////////////////////////////////////
    int *dense_bs_batch, *dense_max_rank_batch, *dense_a_ranks, *dense_b_ranks;
    T **dense_T1_buffers, **dense_T2_buffers;
    T **dense_Ua_ptrs, **dense_Va_ptrs, **dense_Ub_ptrs, **dense_Vb_ptrs;

    ////////////////////////////////////
    // Workspace for the low rank part
    ////////////////////////////////////
    int total_tiles, sample_bs, num_sampling_buffers;
    int *sampling_bs_batch, *sampling_rank_a_batch, *sampling_rank_b_batch, *sampling_max_rank_batch;
    int *sampling_tile_set, *sampling_samples_batch, *small_vectors, *detected_ranks;

    // Marshalled data from the TLR matrices
    T **sampling_Ua_ptrs, **sampling_Ub_ptrs, **sampling_Va_ptrs, **sampling_Vb_ptrs;
    T **sampling_Da_ptrs, **sampling_Db_ptrs;

    // Sampling buffers
    T *base_buffer_output, *base_buffer_input;
    T **sampling_buffer_T1, **sampling_buffer_T2;
    T **sampling_input, **sampling_output, **sampling_output_mod, **sampling_input_mod;

    // Orthogonalization workspace
    H2OpusTLR_ARA_OrthogWorkspace<T> orthog_workspace;
};

template <class T, int hw>
void tlr_gemm_dense_workspace(int n_block, int block_size, int max_rank, H2OpusTLRGemmWorkspace<T> *workspace,
                              h2opusWorkspace_t h2opus_ws, h2opusComputeStream_t stream)
{
    int T1_entries = max_rank * max_rank;
    int T2_entries = block_size * max_rank;
    T *base_buffer_T1, *base_buffer_T2;

    h2opus_ws->allocateEntries<T>(T1_entries * n_block, &base_buffer_T1, hw);
    h2opus_ws->allocateEntries<T>(T2_entries * n_block, &base_buffer_T2, hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(dense_bs_batch), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(dense_max_rank_batch), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(dense_a_ranks), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(dense_b_ranks), hw);

    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(dense_T1_buffers), hw);
    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(dense_T2_buffers), hw);
    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(dense_Ua_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(dense_Va_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(dense_Ub_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(dense_Vb_ptrs), hw);

    if (workspace)
    {
        fillArray(workspace->dense_bs_batch, n_block, block_size, stream, hw);
        fillArray(workspace->dense_max_rank_batch, n_block, max_rank, stream, hw);
        fillArray(base_buffer_T1, T1_entries * n_block, 0, stream, hw);
        fillArray(base_buffer_T2, T2_entries * n_block, 0, stream, hw);

        generateArrayOfPointers(base_buffer_T1, workspace->dense_T1_buffers, T1_entries, n_block, stream, hw);
        generateArrayOfPointers(base_buffer_T2, workspace->dense_T2_buffers, T2_entries, n_block, stream, hw);
    }
}

template <class T, int hw>
void tlr_gemm_sampling_workspace(int n_block, int sample_bs, int num_sampling_buffers, int block_size, int max_rank,
                                 bool symmetric, H2OpusTLRGemmWorkspace<T> *workspace, h2opusWorkspace_t h2opus_ws,
                                 h2opusComputeStream_t stream)
{
    // We have a sequence of four gemms to sample each product
    // output += A_ik * B_kj * R = Ua_ik * Va^T_ik * Ub_kj (Vb^T_kj * R) = Ua_ik * Va^T_ik * (Ub_kj * T1)
    //                           = Ua_ik * (Va^T_ik * T2) = Ua_ik * T3
    // T1 is of size max_rank * max_rank
    // T2 is of size block_size * max_rank
    // T3 is of size max_rank * max_rank
    // Luckily, given the order of the computation and the compatible sizes,
    // this means we can re-use the memory of T1 for T3
    size_t T1_entries = max_rank * max_rank;
    size_t T2_entries = block_size * max_rank;

    // Then we need memory for the random input and the left factors Q of the
    // approximation for each block within a block column
    size_t input_entries = block_size * max_rank;
    size_t output_entries = block_size * max_rank;

    // Total number of tiles that need to be generated
    int total_tiles = (n_block - 1) * n_block;
    if (symmetric)
        total_tiles /= 2;

    T *base_buffer_T1, *base_buffer_T2;

    h2opus_ws->allocateEntries<T>(T1_entries * num_sampling_buffers, &base_buffer_T1, hw);
    h2opus_ws->allocateEntries<T>(T2_entries * num_sampling_buffers, &base_buffer_T2, hw);
    h2opus_ws->allocateEntries<T>(input_entries * num_sampling_buffers, H2OPUS_TLR_WS_PTR(base_buffer_input), hw);
    h2opus_ws->allocateEntries<T>(output_entries * num_sampling_buffers, H2OPUS_TLR_WS_PTR(base_buffer_output), hw);

    h2opus_ws->allocateEntries<int>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_bs_batch), hw);
    h2opus_ws->allocateEntries<int>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_rank_a_batch), hw);
    h2opus_ws->allocateEntries<int>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_rank_b_batch), hw);
    h2opus_ws->allocateEntries<int>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_max_rank_batch), hw);

    h2opus_ws->allocateEntries<int>(total_tiles, H2OPUS_TLR_WS_PTR(sampling_tile_set), hw);
    h2opus_ws->allocateEntries<int>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_samples_batch), hw);
    h2opus_ws->allocateEntries<int>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(small_vectors), hw);
    h2opus_ws->allocateEntries<int>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(detected_ranks), hw);
    // h2opus_ws->allocateEntries<int>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_tile_indices), hw);
    // h2opus_ws->allocateEntries<int>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sub_detected_ranks), hw);
    // h2opus_ws->allocateEntries<int>(1, H2OPUS_TLR_WS_PTR(converged_blocks), hw);

    h2opus_ws->allocatePointerEntries<T>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_buffer_T1), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_buffer_T2), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_Ua_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_Va_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_Ub_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_Vb_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_Da_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_Db_ptrs), hw);

    h2opus_ws->allocatePointerEntries<T>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_input), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_output), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_output_mod), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_buffers, H2OPUS_TLR_WS_PTR(sampling_input_mod), hw);

    if (workspace)
    {
        workspace->num_sampling_buffers = num_sampling_buffers;
        workspace->sample_bs = sample_bs;
        workspace->total_tiles = total_tiles;

        generateArrayOfPointers(base_buffer_T1, workspace->sampling_buffer_T1, T1_entries, num_sampling_buffers, stream,
                                hw);
        generateArrayOfPointers(base_buffer_T2, workspace->sampling_buffer_T2, T2_entries, num_sampling_buffers, stream,
                                hw);

        generateArrayOfPointers(workspace->base_buffer_input, workspace->sampling_input, input_entries,
                                num_sampling_buffers, stream, hw);
        generateArrayOfPointers(workspace->base_buffer_output, workspace->sampling_output, output_entries,
                                num_sampling_buffers, stream, hw);

        // The dimension arrays for block size and max rank don't change either so fill em up here
        fillArray(workspace->sampling_bs_batch, num_sampling_buffers, block_size, stream, hw);
        fillArray(workspace->sampling_max_rank_batch, num_sampling_buffers, max_rank, stream, hw);

        // Make sure the buffers are initialized
        fillArray(base_buffer_T1, T1_entries * num_sampling_buffers, 0, stream, hw);
        fillArray(base_buffer_T2, T2_entries * num_sampling_buffers, 0, stream, hw);
        fillArray(workspace->base_buffer_input, input_entries * num_sampling_buffers, 0, stream, hw);
        fillArray(workspace->base_buffer_output, output_entries * num_sampling_buffers, 0, stream, hw);
    }

    // Allocate workspace for the basis orthogonalization
    tlr_ara_allocate_orthog_workspace<T, hw>(n_block, sample_bs, H2OPUS_TLR_WS_PTR(orthog_workspace), h2opus_ws,
                                             stream);
}

template <class T, int hw>
H2OpusWorkspaceState tlr_gemm_get_workspace(TTLR_Matrix<T, hw> &A, TTLR_Matrix<T, hw> &B, TTLR_Matrix<T, hw> &C,
                                            int sample_bs, int num_sampling_buffers,
                                            H2OpusTLRGemmWorkspace<T> *workspace, h2opusHandle_t h2opus_handle)
{
    h2opusComputeStream_t stream = h2opus_handle->getMainStream();
    h2opusWorkspace_t h2opus_ws = h2opus_handle->getWorkspace();
    h2opus_ws->resetAllocatorState();

    bool symmetric = (C.type == H2OpusTLR_Symmetric);

    tlr_gemm_dense_workspace<T, hw>(A.n_block, A.block_size, A.max_rank, workspace, h2opus_ws, stream);

    tlr_gemm_sampling_workspace<T, hw>(A.n_block, sample_bs, num_sampling_buffers, A.block_size, A.max_rank, symmetric,
                                       workspace, h2opus_ws, stream);

    return h2opus_ws->getAllocatorState();
}

template <class T, int hw>
H2OpusWorkspaceState tlr_gemm_get_workspace(TTLR_Matrix<T, hw> &A, TTLR_Matrix<T, hw> &B, TTLR_Matrix<T, hw> &C,
                                            int sample_bs, int num_sampling_buffers,
                                            H2OpusTLRGemmWorkspace<T> &workspace, h2opusHandle_t h2opus_handle)
{
    return tlr_gemm_get_workspace<T, hw>(A, B, C, sample_bs, num_sampling_buffers, &workspace, h2opus_handle);
}

template <class T, int hw>
H2OpusWorkspaceState tlr_gemm_workspace(TTLR_Matrix<T, hw> &A, TTLR_Matrix<T, hw> &B, TTLR_Matrix<T, hw> &C,
                                        int sample_bs, int num_sampling_buffers, h2opusHandle_t h2opus_handle)
{
    return tlr_gemm_get_workspace<T, hw>(A, B, C, sample_bs, num_sampling_buffers, NULL, h2opus_handle);
}

#endif
