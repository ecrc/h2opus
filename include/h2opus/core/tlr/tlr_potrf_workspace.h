#ifndef __H2OPUS_TLR_POTRF_WORKSPACE_H__
#define __H2OPUS_TLR_POTRF_WORKSPACE_H__

#include <h2opus/core/tlr/tlr_ara_util.h>
#include <type_traits>

template <class T> struct H2OpusTLRPotrfWorkspace
{
    // Dense diagonal block updates
    int num_dense_parallel_buffers;
    T *base_dense_buffer_D;
    T **dense_buffer_T, **dense_buffer_G, **dense_buffer_D, **dense_Di_ptrs;
    int *dense_bs_batch, *dense_rank_batch, *dense_max_rank_batch;
    int *dense_segment_maxes, *dense_segment_maxes_host, *dense_ipiv;
    T **dense_U_ptrs, **dense_V_ptrs;

    // Triangular solve
    int *trsm_m;
    T **trsm_A_ptrs;

    // Low rank update sampling
    int num_sampling_parallel_buffers, sample_bs;
    // Keep the base pointers for T4 and the output handy since T4 is an accumulator
    // and the output should be reset to allow zero columns past the detected ranks
    T *base_buffer_T4, *base_buffer_output, *base_buffer_input;
    T **sampling_buffer_T1, **sampling_buffer_T2, **sampling_buffer_T4;
    T **sampling_input, **sampling_input_mod, **sampling_output, **sampling_output_mod, **samplinge_input_i_ptrs;
    T **sampling_Uij_ptrs, **sampling_Ukj_ptrs, **sampling_Vij_ptrs, **sampling_Vkj_ptrs, **sampling_D_ptrs;
    int *sampling_samples_batch, *sampling_samples_i_batch, *sampling_rank_ij_batch, *sampling_rank_kj_batch,
        *small_vectors;
    int *detected_ranks, *sampling_max_rank_batch, *sampling_bs_batch, *sampling_row_indices, *sub_detected_ranks;

    // Orthogonalization workspace
    H2OpusTLR_ARA_OrthogWorkspace<T> orthog_workspace;

    // Schur compensation
    T *svd_A, *svd_U, *svd_S, *svd_V, *svd_ws, **ptr_svd;

    // This is a single integer that keeps track of the converged blocks during the ARA
    int *converged_blocks;
};

template <class T, int hw>
void tlr_potrf_svd_workspace(int block_size, H2OpusTLRPotrfWorkspace<T> *workspace, h2opusWorkspace_t h2opus_ws,
                             h2opusComputeStream_t stream)
{
    if (hw == H2OPUS_HWTYPE_CPU)
    {
        h2opus_ws->allocateEntries<T>(2 * block_size * block_size, H2OPUS_TLR_WS_PTR(svd_ws), hw);
    }
    else
    {
#ifdef H2OPUS_USE_GPU
        // cusolverDnHandle_t cusolver_handle = stream->getCuSolverHandle();
        // int lwork;
        // if(std::is_same<T, double>::value)
        // {
        //     gpuCusolverErrchk(cusolverDnDgesvd_bufferSize(cusolver_handle, block_size, block_size, &lwork));
        // }
        // else
        // {
        //     gpuCusolverErrchk(cusolverDnSgesvd_bufferSize(cusolver_handle, block_size, block_size, &lwork));
        // }
        //
        // h2opus_ws->allocateEntries<T>(lwork, H2OPUS_TLR_WS_PTR(svd_ws), hw);
        kblas_ara_batch_wsquery<T>(stream->getKblasHandle(), block_size, 1);
        kblasAllocateWorkspace(stream->getKblasHandle());
#endif
    }
}

template <class T, int hw>
void tlr_potrf_sampling_workspace(bool sytrf_ws, int n_block, int block_size, int max_rank, size_t sample_bs,
                                  size_t num_sampling_parallel_buffers, H2OpusTLRPotrfWorkspace<T> *workspace,
                                  h2opusWorkspace_t h2opus_ws, h2opusComputeStream_t stream)
{
    assert(num_sampling_parallel_buffers >= (size_t)n_block);

    // We have a sequence of four gemms to sample each low rank update
    // For each block i in a column k, we have
    // A_ik = A_ik - A_ij * A_kj^T for j = 1:k-1
    // We sample each low rank update using random vectors R on A_ij * A_kj^T
    // and do a reduction to get the sample of A_ik
    // A_ij * A_kj^T * R = U_ij * V_ij^T V_kj (U_kj^T * R) = U_ij * V_ij^T V_kj * T1
    //                   = U_ij * V_ij^T * T2 = U_ij * T3 = T4
    // T1 is of size max_rank * max_rank
    // T2 is of size block_size * max_rank
    // T3 is of size max_rank * max_rank
    // T4 is of size block_size * max_rank
    // Luckily, given the order of the computation and the compatible sizes,
    // this means we can re-use the memory of T1 for T3 and T2 for T4
    // However, we want to accumulate multiple of these operations, so we
    // can't reuse T2 for T4. it will be its own buffer
    size_t T1_entries = max_rank * max_rank;
    size_t T2_entries = block_size * max_rank;
    size_t T4_entries = block_size * max_rank;

    // Then we need memory for the random input and the left factors Q of the
    // approximation for each block within a block column
    size_t input_entries = block_size * max_rank;
    size_t output_entries = block_size * max_rank;

    // Now we need dimension data for each buffer
    size_t dim_entries = std::max(num_sampling_parallel_buffers, (size_t)n_block);

    T *base_buffer_T1, *base_buffer_T2;

    h2opus_ws->allocateEntries<T>(T1_entries * num_sampling_parallel_buffers, &base_buffer_T1, hw);
    h2opus_ws->allocateEntries<T>(T2_entries * num_sampling_parallel_buffers, &base_buffer_T2, hw);
    h2opus_ws->allocateEntries<T>(T4_entries * num_sampling_parallel_buffers, H2OPUS_TLR_WS_PTR(base_buffer_T4), hw);
    h2opus_ws->allocateEntries<T>(input_entries * n_block, H2OPUS_TLR_WS_PTR(base_buffer_input), hw);
    h2opus_ws->allocateEntries<T>(output_entries * n_block, H2OPUS_TLR_WS_PTR(base_buffer_output), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(sampling_bs_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(sampling_rank_ij_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(sampling_rank_kj_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(sampling_max_rank_batch), hw);
    h2opus_ws->allocateEntries<int>(dim_entries, H2OPUS_TLR_WS_PTR(sampling_samples_i_batch), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(sampling_samples_batch), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(small_vectors), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(detected_ranks), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(sampling_row_indices), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(sub_detected_ranks), hw);
    h2opus_ws->allocateEntries<int>(1, H2OPUS_TLR_WS_PTR(converged_blocks), hw);

    h2opus_ws->allocatePointerEntries<T>(num_sampling_parallel_buffers, H2OPUS_TLR_WS_PTR(sampling_buffer_T1), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_parallel_buffers, H2OPUS_TLR_WS_PTR(sampling_buffer_T2), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_parallel_buffers, H2OPUS_TLR_WS_PTR(sampling_buffer_T4), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_parallel_buffers, H2OPUS_TLR_WS_PTR(sampling_Uij_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_parallel_buffers, H2OPUS_TLR_WS_PTR(sampling_Vij_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_parallel_buffers, H2OPUS_TLR_WS_PTR(sampling_Ukj_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_parallel_buffers, H2OPUS_TLR_WS_PTR(sampling_Vkj_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(num_sampling_parallel_buffers, H2OPUS_TLR_WS_PTR(samplinge_input_i_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(sampling_input), hw);
    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(sampling_output), hw);
    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(sampling_output_mod), hw);
    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(sampling_input_mod), hw);

    if (sytrf_ws)
    {
        h2opus_ws->allocatePointerEntries<T>(num_sampling_parallel_buffers, H2OPUS_TLR_WS_PTR(sampling_D_ptrs), hw);
    }

    if (workspace)
    {
        workspace->num_sampling_parallel_buffers = num_sampling_parallel_buffers;
        workspace->sample_bs = sample_bs;

        // The pointers to T1, T2, T4, G, the input and the output
        // never change - they always point to the same location in memory
        // so we can set them here
        generateArrayOfPointers(base_buffer_T1, workspace->sampling_buffer_T1, T1_entries,
                                num_sampling_parallel_buffers, stream, hw);
        generateArrayOfPointers(base_buffer_T2, workspace->sampling_buffer_T2, T2_entries,
                                num_sampling_parallel_buffers, stream, hw);
        generateArrayOfPointers(workspace->base_buffer_T4, workspace->sampling_buffer_T4, T4_entries,
                                num_sampling_parallel_buffers, stream, hw);

        generateArrayOfPointers(workspace->base_buffer_input, workspace->sampling_input, input_entries, n_block, stream,
                                hw);
        generateArrayOfPointers(workspace->base_buffer_output, workspace->sampling_output, output_entries, n_block,
                                stream, hw);

        // The dimension arrays for block size and max rank don't change either so fill em up here
        fillArray(workspace->sampling_bs_batch, num_sampling_parallel_buffers, block_size, stream, hw);
        fillArray(workspace->sampling_max_rank_batch, num_sampling_parallel_buffers, max_rank, stream, hw);

        // Make sure the buffers are initialized
        fillArray(base_buffer_T1, T1_entries * num_sampling_parallel_buffers, 0, stream, hw);
        fillArray(base_buffer_T2, T2_entries * num_sampling_parallel_buffers, 0, stream, hw);
        fillArray(workspace->base_buffer_T4, T4_entries * num_sampling_parallel_buffers, 0, stream, hw);
        fillArray(workspace->base_buffer_input, input_entries * n_block, 0, stream, hw);
        fillArray(workspace->base_buffer_output, output_entries * n_block, 0, stream, hw);
    }

    // Allocate workspace for the basis orthogonalization
    tlr_ara_allocate_orthog_workspace<T, hw>(n_block, sample_bs, H2OPUS_TLR_WS_PTR(orthog_workspace), h2opus_ws,
                                             stream);
}

template <class T, int hw>
void tlr_potrf_trsm_workspace(int n_block, H2OpusTLRPotrfWorkspace<T> *workspace, h2opusWorkspace_t h2opus_ws,
                              h2opusComputeStream_t stream)
{
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(trsm_m), hw);
    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(trsm_A_ptrs), hw);
}

template <class T, int hw>
void tlr_potrf_dense_workspace(bool sytrf_ws, int n_block, int block_size, int max_rank,
                               size_t num_dense_parallel_buffers, H2OpusTLRPotrfWorkspace<T> *workspace,
                               h2opusWorkspace_t h2opus_ws, int *piv, h2opusComputeStream_t stream)
{
    // Dense diagonal block updates
    // For each low rank update of the form R_i R_i^T where R_i = U_i V_i^T is a low rank block
    // We must first compute G_i = V_i^T V_i, then compute T_i = U_i G_i and finally D_i = T_i U_i^T
    // G_i is a k x k matrix where k is the rank, T_i is a bs x k matrix and D_i is a bs x bs matrix
    // We do num_dense_parallel_buffers of these updates in parallel so we allocate as many buffers
    size_t dense_G_block_entries = max_rank * max_rank;
    size_t dense_T_block_entries = block_size * max_rank;
    size_t dense_D_block_entries = block_size * block_size;

    // If we're doing the pivoted version, then we need as many temporary dense blocks as there are rows and columns
    if (piv)
        num_dense_parallel_buffers = n_block;
    else
    {
        // Make sure we don't allocate more buffers than we can actually use
        if ((int)num_dense_parallel_buffers > n_block - 1)
            num_dense_parallel_buffers = n_block - 1;

        // Make sure there are enough buffers for the routine that ensures positive definiteness
        if ((int)num_dense_parallel_buffers < 4)
            num_dense_parallel_buffers = 4;
    }

    T *base_dense_buffer_G, *base_dense_buffer_T;
    h2opus_ws->allocateEntries<T>(dense_G_block_entries * num_dense_parallel_buffers, &base_dense_buffer_G, hw);
    h2opus_ws->allocateEntries<T>(dense_T_block_entries * num_dense_parallel_buffers, &base_dense_buffer_T, hw);
    h2opus_ws->allocateEntries<T>(dense_D_block_entries * num_dense_parallel_buffers,
                                  H2OPUS_TLR_WS_PTR(base_dense_buffer_D), hw);

    h2opus_ws->allocateEntries<T>(dense_D_block_entries, H2OPUS_TLR_WS_PTR(svd_A), hw);
    h2opus_ws->allocateEntries<T>(dense_D_block_entries, H2OPUS_TLR_WS_PTR(svd_U), hw);
    h2opus_ws->allocateEntries<T>(block_size, H2OPUS_TLR_WS_PTR(svd_S), hw);
    h2opus_ws->allocateEntries<T>(dense_D_block_entries, H2OPUS_TLR_WS_PTR(svd_V), hw);

    h2opus_ws->allocateEntries<int>(num_dense_parallel_buffers, H2OPUS_TLR_WS_PTR(dense_bs_batch), hw);
    h2opus_ws->allocateEntries<int>(num_dense_parallel_buffers, H2OPUS_TLR_WS_PTR(dense_max_rank_batch), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(dense_rank_batch), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(dense_segment_maxes), hw);
    h2opus_ws->allocateEntries<int>(block_size, H2OPUS_TLR_WS_PTR(dense_ipiv), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(dense_segment_maxes_host), H2OPUS_HWTYPE_CPU);

    h2opus_ws->allocatePointerEntries<T>(num_dense_parallel_buffers, H2OPUS_TLR_WS_PTR(dense_buffer_G), hw);
    h2opus_ws->allocatePointerEntries<T>(num_dense_parallel_buffers, H2OPUS_TLR_WS_PTR(dense_buffer_T), hw);
    h2opus_ws->allocatePointerEntries<T>(num_dense_parallel_buffers, H2OPUS_TLR_WS_PTR(dense_buffer_D), hw);
    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(dense_U_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(dense_V_ptrs), hw);
    h2opus_ws->allocatePointerEntries<T>(3, H2OPUS_TLR_WS_PTR(ptr_svd), hw);

    if (sytrf_ws)
    {
        h2opus_ws->allocatePointerEntries<T>(num_dense_parallel_buffers, H2OPUS_TLR_WS_PTR(dense_Di_ptrs), hw);
    }

    if (workspace)
    {
        workspace->num_dense_parallel_buffers = num_dense_parallel_buffers;

        // Populate the pointer arrays using the allocated strided memory
        generateArrayOfPointers(base_dense_buffer_G, workspace->dense_buffer_G, dense_G_block_entries,
                                num_dense_parallel_buffers, stream, hw);
        generateArrayOfPointers(base_dense_buffer_T, workspace->dense_buffer_T, dense_T_block_entries,
                                num_dense_parallel_buffers, stream, hw);
        generateArrayOfPointers(workspace->base_dense_buffer_D, workspace->dense_buffer_D, dense_D_block_entries,
                                num_dense_parallel_buffers, stream, hw);

        // The dimension arrays for block size and max rank don't change either so fill em up here
        fillArray(workspace->dense_bs_batch, num_dense_parallel_buffers, block_size, stream, hw);
        fillArray(workspace->dense_max_rank_batch, num_dense_parallel_buffers, max_rank, stream, hw);

        // Make sure the memory is initialized
        fillArray(base_dense_buffer_G, dense_G_block_entries * num_dense_parallel_buffers, 0, stream, hw);
        fillArray(base_dense_buffer_T, dense_T_block_entries * num_dense_parallel_buffers, 0, stream, hw);
        fillArray(workspace->base_dense_buffer_D, dense_D_block_entries * num_dense_parallel_buffers, 0, stream, hw);

        fillArray(workspace->ptr_svd, 1, workspace->svd_A, stream, hw);
        fillArray(workspace->ptr_svd + 1, 1, workspace->svd_U, stream, hw);
        fillArray(workspace->ptr_svd + 2, 1, workspace->svd_V, stream, hw);
        fillArray(workspace->svd_A, dense_D_block_entries, 0, stream, hw);
    }

    tlr_potrf_svd_workspace<T, hw>(block_size, workspace, h2opus_ws, stream);
}

template <class T, int hw>
H2OpusWorkspaceState tlr_potrf_get_workspace(TTLR_Matrix<T, hw> &A, H2OpusTLRPotrfWorkspace<T> *workspace,
                                             bool sytrf_ws, size_t num_dense_parallel_buffers,
                                             size_t num_sampling_parallel_buffers, size_t sample_bs, int *piv,
                                             h2opusHandle_t h2opus_handle)
{
    if (sample_bs > (size_t)A.max_rank)
    {
        // printf("Sample block size cannot be greater than the max rank. Reducing sample_bs to %d\n", A.max_rank);
        sample_bs = A.max_rank;
    }

    h2opusComputeStream_t stream = h2opus_handle->getMainStream();
    h2opusWorkspace_t h2opus_ws = h2opus_handle->getWorkspace();
    h2opus_ws->resetAllocatorState();

    tlr_potrf_dense_workspace<T, hw>(sytrf_ws, A.n_block, A.block_size, A.max_rank, num_dense_parallel_buffers,
                                     workspace, h2opus_ws, piv, stream);

    tlr_potrf_trsm_workspace<T, hw>(A.n_block, workspace, h2opus_ws, stream);

    tlr_potrf_sampling_workspace<T, hw>(sytrf_ws, A.n_block, A.block_size, A.max_rank, sample_bs,
                                        num_sampling_parallel_buffers, workspace, h2opus_ws, stream);

#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
    {
        kblas_ara_trsm_batch_wsquery<H2Opus_Real>(stream->getKblasHandle(), A.n_block);
        kblasAllocateWorkspace(stream->getKblasHandle());
    }
#endif
    H2OpusWorkspaceState ws_state = h2opus_ws->getAllocatorState();
    float total_d_gbytes = (float)(ws_state.d_data_bytes + ws_state.d_ptrs_bytes) / (1024 * 1024 * 1024);
    float total_h_gbytes = (float)(ws_state.h_data_bytes + ws_state.h_ptrs_bytes) / (1024 * 1024 * 1024);
    printf("Total allocated workspace memory = %.3f GB (Host) %.3f GB (Device)\n", total_h_gbytes, total_d_gbytes);

    return ws_state;
}

template <class T, int hw>
H2OpusWorkspaceState tlr_potrf_get_workspace(TTLR_Matrix<T, hw> &A, H2OpusTLRPotrfWorkspace<T> &workspace,
                                             bool sytrf_ws, size_t num_dense_parallel_buffers,
                                             size_t num_sampling_parallel_buffers, size_t sample_bs, int *piv,
                                             h2opusHandle_t h2opus_handle)
{
    return tlr_potrf_get_workspace<T, hw>(A, &workspace, sytrf_ws, num_dense_parallel_buffers,
                                          num_sampling_parallel_buffers, sample_bs, piv, h2opus_handle);
}

template <class T, int hw>
H2OpusWorkspaceState tlr_potrf_workspace(TTLR_Matrix<T, hw> &A, bool sytrf_ws, size_t num_dense_parallel_buffers,
                                         size_t num_sampling_parallel_buffers, size_t sample_bs, int *piv,
                                         h2opusHandle_t h2opus_handle)
{
    return tlr_potrf_get_workspace<T, hw>(A, NULL, sytrf_ws, num_dense_parallel_buffers, num_sampling_parallel_buffers,
                                          sample_bs, piv, h2opus_handle);
}

#endif
