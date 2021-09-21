#ifndef __H2OPUS_TLR_POTRF_H__
#define __H2OPUS_TLR_POTRF_H__

#include <h2opus/core/hara_util.cuh>
#include <h2opus/util/error_approximation.h>
#include <h2opus/util/host_ara.h>

#include <h2opus/core/tlr/tlr_defs.h>
#include <h2opus/core/tlr/tlr_struct.h>
#include <h2opus/core/tlr/tlr_trsm.h>
#include <h2opus/core/tlr/tlr_potrf_workspace.h>
#include <h2opus/core/tlr/tlr_potrf_marshal.h>
#include <h2opus/core/tlr/tlr_potrf_config.h>
#include <h2opus/core/tlr/tlr_batch.h>
#include <h2opus/core/tlr/tlr_potrf_util.h>

#define H2OPUS_TLR_POTRF_USE_MAGMA_FIX
#define H2OPUS_TLR_USE_SCHUR_COMPENSATION

// TODO: Redo workspace to use ara bs instead of max rank
//       Update projection to sample using the ara bs instead of the rank
//       Merge gemms for small ranks
//       GPU kernels for pivot selection
//       Graceful return when non-spd matrix is encountered

////////////////////////////////////////////////////////////////////////////
// Main subroutines
////////////////////////////////////////////////////////////////////////////
// Sample the sum of the low rank updates coming from the block rows of a sub-matrix of A
// i.e. compute output_i = sum_{j=col_start:col_end} A_ij * A_kj^T * input_i for each block in row i
template <class T, int hw, bool transpose>
void tlr_potrf_sample_lru(TTLR_Matrix<T, hw> &A, int k, int *row_indices, int rows, int col_start, int col_end,
                          T **input_ptrs, T **output_ptrs, int *samples_batch, int max_samples,
                          H2OpusTLRPotrfWorkspace<T> &workspace, h2opusComputeStream_t stream)
{
    int block_size = A.block_size;

    // Figure out how many block columns we can process in parallel
    int par_block_cols = workspace.num_sampling_parallel_buffers / rows;
    assert(par_block_cols != 0);

    par_block_cols = std::min(col_end - col_start + 1, par_block_cols);
    if (par_block_cols == 0)
        return;

    int *bs_batch = workspace.sampling_bs_batch;
    int *rank_ij_batch = workspace.sampling_rank_ij_batch;
    int *rank_kj_batch = workspace.sampling_rank_kj_batch;
    int *max_rank_batch = workspace.sampling_max_rank_batch;
    int *samples_i_batch = workspace.sampling_samples_i_batch;

    T **Uij_ptrs = workspace.sampling_Uij_ptrs, **Ukj_ptrs = workspace.sampling_Ukj_ptrs;
    T **Vij_ptrs = workspace.sampling_Vij_ptrs, **Vkj_ptrs = workspace.sampling_Vkj_ptrs;
    T **T1_ptrs = workspace.sampling_buffer_T1, **T2_ptrs = workspace.sampling_buffer_T2;
    // Reuse T1 buffers for T3
    T **T3_ptrs = workspace.sampling_buffer_T1, **T4_ptrs = workspace.sampling_buffer_T4;
    T **input_i_ptrs = workspace.samplinge_input_i_ptrs;

#ifdef H2OPUS_TLR_POTRF_USE_MAGMA_FIX
    TLR_Potrf_Phase_Times<hw>::startPhase(TLR_Potrf_Phase_Types::Clear);
    // if(hw == H2OPUS_HWTYPE_GPU)
    {
        // Clear out the accumulation buffer T4 since Magma doesn't set C = C * beta when the inner dimension of the
        // gemm is zero
        size_t T4_entries = block_size * max_samples;
        int max_par_blocks = par_block_cols * rows;
        fillArray(workspace.base_buffer_T4, T4_entries * max_par_blocks, 0, stream, hw);
        // This is only necessary since we have to manually zero out the data due to the above Magma issue
        generateArrayOfPointers(workspace.base_buffer_T4, workspace.sampling_buffer_T4, T4_entries, max_par_blocks,
                                stream, hw);
    }
    TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::Clear);
#endif

    // Sample multiple block columns at a time when we can
    // Each sample goes into its own buffer, and we do a reduction
    // on the buffers when we are done
    int sampled_block_cols = col_start;
    T T4_beta = 0;

    TLR_Potrf_Phase_Times<hw>::startPhase(transpose ? TLR_Potrf_Phase_Types::Projection
                                                    : TLR_Potrf_Phase_Types::Sample);
    while (sampled_block_cols <= col_end)
    {
        // Sample the low rank updates:
        // T4 += A_ij * A_kj^T * R = U_ij * V_ij^T V_kj (U_kj^T * R) = U_ij * V_ij^T V_kj * T1
        //                         = U_ij * V_ij^T * T2 = U_ij * T3
        int block_columns = std::min(par_block_cols, col_end - sampled_block_cols + 1);
        int sample_block_count = block_columns * rows;

        // Marshal the blocks that need to be sampled in this set of block columns
        tlr_potrf_marshal_lru_sample_range<T, hw, transpose>(
            vec_ptr(A.block_U_ptrs), vec_ptr(A.block_V_ptrs), vec_ptr(A.block_ranks), k, A.n_block, row_indices, rows,
            sampled_block_cols, block_columns, Uij_ptrs, Vij_ptrs, Ukj_ptrs, Vkj_ptrs, input_ptrs, input_i_ptrs,
            samples_batch, samples_i_batch, rank_ij_batch, rank_kj_batch, NULL, 0, NULL, sample_block_count, stream);

        int max_rank_kj = getMaxElement(rank_kj_batch, sample_block_count, stream, hw);
        int max_rank_ij = getMaxElement(rank_ij_batch, sample_block_count, stream, hw);

        // Now that we marshaled the low rank pointers, execute the needed gemms
        // T1 = U_kj^T * R
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, rank_kj_batch,
                                                       samples_i_batch, bs_batch, max_rank_kj, max_samples, block_size,
                                                       (T)1, (const T **)Ukj_ptrs, bs_batch, (const T **)input_i_ptrs,
                                                       bs_batch, 0, T1_ptrs, max_rank_batch, sample_block_count));

        // T2 = V_kj * T1
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, bs_batch,
                                                       samples_i_batch, rank_kj_batch, block_size, max_samples,
                                                       max_rank_kj, (T)1, (const T **)Vkj_ptrs, bs_batch,
                                                       (const T **)T1_ptrs, max_rank_batch, 0, T2_ptrs, bs_batch,
                                                       sample_block_count));

        // T3 = V_ij^T * T2
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, rank_ij_batch,
                                                       samples_i_batch, bs_batch, max_rank_ij, max_samples, block_size,
                                                       (T)1, (const T **)Vij_ptrs, bs_batch, (const T **)T2_ptrs,
                                                       bs_batch, 0, T3_ptrs, max_rank_batch, sample_block_count));

        // T4 += U_ij * T3
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, bs_batch,
                                                       samples_i_batch, rank_ij_batch, block_size, max_samples,
                                                       max_rank_ij, (T)1, (const T **)Uij_ptrs, bs_batch,
                                                       (const T **)T3_ptrs, max_rank_batch, T4_beta, T4_ptrs, bs_batch,
                                                       sample_block_count));

        sampled_block_cols += block_columns;
        T4_beta = 1;
    }
    TLR_Potrf_Phase_Times<hw>::endPhase(transpose ? TLR_Potrf_Phase_Types::Projection : TLR_Potrf_Phase_Types::Sample);

    TLR_Potrf_Phase_Times<hw>::startPhase(TLR_Potrf_Phase_Types::Reduction);

    // Do a reduction on the parallel buffers
    if (par_block_cols > 0)
    {
        TLR_Batch<T, hw>::reduceMatrixBuffers(0, output_ptrs, bs_batch, bs_batch, samples_batch, -1, T4_ptrs, bs_batch,
                                              par_block_cols, block_size, max_samples, rows, stream);
    }
    TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::Reduction);
}

template <class T, int hw, bool transpose>
void tlr_potrf_sample_col(TTLR_Matrix<T, hw> &A, int k, int *row_indices, int rows, T **input_ptrs, T **output_ptrs,
                          int *samples_batch, int max_samples, H2OpusTLRPotrfWorkspace<T> &workspace,
                          h2opusComputeStream_t stream)
{
    TLR_Potrf_Phase_Times<hw>::startPhase(transpose ? TLR_Potrf_Phase_Types::Projection
                                                    : TLR_Potrf_Phase_Types::Sample);

    int block_size = A.block_size;

    // Re-use the lru sampling arrays
    int *bs_batch = workspace.sampling_bs_batch;
    int *max_rank_batch = workspace.sampling_max_rank_batch;
    T **T1_ptrs = workspace.sampling_buffer_T1;

    T **Uik_ptrs = workspace.sampling_Uij_ptrs;
    T **Vik_ptrs = workspace.sampling_Vij_ptrs;
    int *rank_ik_batch = workspace.sampling_rank_ij_batch;

    tlr_potrf_marshal_col_sample_range<T, hw, transpose>(vec_ptr(A.block_U_ptrs), vec_ptr(A.block_V_ptrs),
                                                         vec_ptr(A.block_ranks), k, A.n_block, row_indices, rows,
                                                         Uik_ptrs, Vik_ptrs, rank_ik_batch, stream);

    int max_col_rank = getMaxElement(rank_ik_batch, rows, stream, hw);

    // T1 = V_ik^T * R
    check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, rank_ik_batch, samples_batch,
                                                   bs_batch, max_col_rank, max_samples, block_size, (T)1,
                                                   (const T **)Vik_ptrs, bs_batch, (const T **)input_ptrs, bs_batch, 0,
                                                   T1_ptrs, max_rank_batch, rows));

    // output += U_ik * T1
    check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, bs_batch, samples_batch,
                                                   rank_ik_batch, block_size, max_samples, max_col_rank, (T)1,
                                                   (const T **)Uik_ptrs, bs_batch, (const T **)T1_ptrs, max_rank_batch,
                                                   (T)1, output_ptrs, bs_batch, rows));

    TLR_Potrf_Phase_Times<hw>::endPhase(transpose ? TLR_Potrf_Phase_Types::Projection : TLR_Potrf_Phase_Types::Sample);
}

// Generate an approximation of the low rank updated block column k
// in left looking cholesky
template <class T, int hw>
void tlr_potrf_update_block_column(TTLR_Matrix<T, hw> &A, T eps, int k, H2OpusTLRPotrfWorkspace<T> &workspace,
                                   h2opusComputeStream_t stream, h2opusHandle_t h2opus_handle)
{
    const int r = 10;

    int max_rank = A.max_rank;
    int n_block = A.n_block;
    int rows = n_block - k - 1;
    int block_size = A.block_size;
    int sample_bs = workspace.sample_bs;

    if (k == 0 || rows == 0)
        return;

    int row_index_start = k + 1;

    T **input_ptrs = workspace.sampling_input, **sub_Q_ptrs = workspace.sampling_input_mod;
    T **Q_ptrs = workspace.sampling_output, **sub_Y_ptrs = workspace.sampling_output_mod;
    int *samples_batch = workspace.sampling_samples_batch;
    int *bs_batch = workspace.sampling_bs_batch;
    int *detected_ranks = workspace.detected_ranks;
    int *small_vectors = workspace.small_vectors;

    // Temporary storage for matrix Z, used in the basis orthogonalization
    // Z is a (current_rank x samples) matrix. we can re-use the T1 array which
    // has enough space for max_rank x samples.
    T **Z_ptrs = workspace.sampling_buffer_T1;
    int *ldz_batch = workspace.sampling_max_rank_batch;

#ifdef H2OPUS_TLR_USE_CHOLESKY_QR
    double *R_diag = workspace.orthog_workspace.R_diag;
    typedef double R_prec;
#else
    T *R_diag = workspace.orthog_workspace.hh_R_diag;
    typedef T R_prec;
#endif

    TLR_Potrf_Phase_Times<hw>::startPhase(TLR_Potrf_Phase_Types::Clear);

    ////////////////////////////////////////////////////////////////////////
    // Generate an approximation basis Q_i for the blocks of the column
    ////////////////////////////////////////////////////////////////////////
    // Clear the output
    // size_t output_entries = block_size * max_rank * rows;
    // fillArray(workspace.base_buffer_output, output_entries, 0, stream, hw);
    fillArray(detected_ranks, rows, 0, stream, hw);
    fillArray(small_vectors, rows, 0, stream, hw);
    fillArray(samples_batch, rows, sample_bs, stream, hw);

    TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::Clear);

    int block_rows = (rows < 20 ? rows : rows / 2);
    int *row_indices = workspace.sampling_row_indices;
    int *subset_ranks = workspace.sub_detected_ranks;

    // Set the subset row index array
    generateSequence(row_indices, rows, row_index_start, stream, hw);

    // Sort the row indices by their original rank in descending order
    int *original_ranks = vec_ptr(A.block_ranks) + row_index_start + k * n_block;
    copyArray(original_ranks, subset_ranks, rows, stream, hw);
    sortByKey(subset_ranks, row_indices, rows, true, stream, hw);

    // Clear the ranks
    fillArray(subset_ranks, rows, 0, stream, hw);

    int converged_blocks = 0;
    int *converged_blocks_ptr = workspace.converged_blocks;
    fillArray(converged_blocks_ptr, 1, 0, stream, hw);

    while (converged_blocks < rows)
    {
        int n_rows = std::min(block_rows, rows - converged_blocks);

        // Set the Q and Y pointers based on the current ranks and the selected block rows
        tlr_potrf_set_sample_basis_ptrs<T, hw>(sub_Q_ptrs, sub_Y_ptrs, Q_ptrs, bs_batch, row_indices, subset_ranks,
                                               row_index_start, n_rows, stream);

        TLR_Potrf_Phase_Times<hw>::startPhase(TLR_Potrf_Phase_Types::RandGen);

        // Generate the random gaussian input vectors - each one of size block_size x samples
        check_kblas_error((H2OpusBatched<T, hw>::rand)(stream, h2opus_handle, bs_batch, samples_batch, block_size,
                                                       sample_bs, input_ptrs, bs_batch, n_rows));

        TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::RandGen);

        // Sample the sum of the low rank updates in each block row in the sub-matrix A(k+1:end, 0:k-1)
        tlr_potrf_sample_lru<T, hw, false>(A, k, row_indices, n_rows, 0, k - 1, input_ptrs, sub_Y_ptrs, samples_batch,
                                           sample_bs, workspace, stream);

        // Sample the current block column k and subtract the result from the previously accumulated samples
        // i.e A_ik * R_i - sum_{j=0:k-1} A_ij * A_kj^T * R_i
        tlr_potrf_sample_col<T, hw, false>(A, k, row_indices, n_rows, input_ptrs, sub_Y_ptrs, samples_batch, sample_bs,
                                           workspace, stream);

        TLR_Potrf_Phase_Times<hw>::startPhase(TLR_Potrf_Phase_Types::Orthog);

        int current_max_rank = getMaxElement(subset_ranks, n_rows, stream, hw);

        // Generate orthogonal basis from the samples and check for convergence
        tlr_ara_gen_basis<T, hw>(sub_Q_ptrs, bs_batch, bs_batch, subset_ranks, block_size, current_max_rank, sub_Y_ptrs,
                                 bs_batch, samples_batch, sample_bs, Z_ptrs, ldz_batch, n_rows,
                                 workspace.orthog_workspace, stream);

        TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::Orthog);

        // Count the number of vectors that have a small magnitude
        // also updates the rank, max diagonal and advances the Y_batch pointers
        hara_util_svec_count_batch<T, R_prec, hw>(samples_batch, subset_ranks, small_vectors, R_diag, r, (R_prec)eps,
                                                  sample_bs, sub_Y_ptrs, sub_Q_ptrs, bs_batch, n_rows, stream);

        // See how many samples we should take next, see how many operations converged and update the pointer arrays if
        // necessary
        converged_blocks =
            tlr_ara_check_converged<hw>(samples_batch, row_indices, subset_ranks, small_vectors, converged_blocks_ptr,
                                        detected_ranks, sample_bs, max_rank, r, n_rows, row_index_start, rows, stream);
    }

    ////////////////////////////////////////////////////////////////////////
    // Form B_i = A_ik^T Q_i
    ////////////////////////////////////////////////////////////////////////
    TLR_Potrf_Phase_Times<hw>::startPhase(TLR_Potrf_Phase_Types::Clear);
    // Clear the input
    // size_t input_entries = block_size * A.max_rank * rows;
    // fillArray(workspace.base_buffer_input, input_entries, 0, stream, hw);

    // Reset the row index array
    generateSequence(row_indices, rows, k + 1, stream, hw);

    // Keep track of how many samples were taken
    int *current_samples = small_vectors;
    fillArray(current_samples, rows, 0, stream, hw);

    TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::Clear);

    T **sub_B_ptrs = sub_Y_ptrs;
    int projected_blocks = 0;
    block_rows = rows;
    while (projected_blocks < rows)
    {
        int n_rows = std::min(block_rows, rows - projected_blocks);

        // Figure out how many samples we need per block
        // tlr_determine_projection_samples<T, hw>(
        //     samples_batch, sample_bs, current_samples, detected_ranks,
        //     Q_ptrs, sub_Q_ptrs, bs_batch, input_ptrs, sub_B_ptrs, bs_batch,
        //     n_rows, stream
        // );
        sub_Q_ptrs = Q_ptrs + projected_blocks;
        sub_B_ptrs = input_ptrs + projected_blocks;
        samples_batch = detected_ranks + projected_blocks;
        int max_samples = getMaxElement(samples_batch, n_rows, stream, hw);

        // Sample the sum of the low rank updates in each block row in the sub-matrix A(k+1:end, 0:k-1)
        tlr_potrf_sample_lru<T, hw, true>(A, k, row_indices, n_rows, 0, k - 1, sub_Q_ptrs, sub_B_ptrs, samples_batch,
                                          max_samples, workspace, stream);

        // Sample the current block column k and subtract the result from the previously accumulated samples
        // i.e A_ik * R_i - sum_{j=0:k-1} A_ij * A_kj^T * R_i
        tlr_potrf_sample_col<T, hw, true>(A, k, row_indices, n_rows, sub_Q_ptrs, sub_B_ptrs, samples_batch, max_samples,
                                          workspace, stream);

        row_indices += n_rows;
        projected_blocks += n_rows;
    }

    TLR_Potrf_Phase_Times<hw>::startPhase(TLR_Potrf_Phase_Types::Realloc);

    // Now reallocate the block column to the new ranks
    A.allocateBlockColumn(k, detected_ranks, stream);

    //...and copy the new low rank data
    T **col_block_U_ptrs = vec_ptr(A.block_U_ptrs) + k + 1 + k * n_block;
    T **col_block_V_ptrs = vec_ptr(A.block_V_ptrs) + k + 1 + k * n_block;

    int new_max_rank = getMaxElement(detected_ranks, rows, stream, hw);

    check_kblas_error((H2OpusBatched<T, hw>::copyBlock)(stream, bs_batch, detected_ranks, block_size, new_max_rank,
                                                        col_block_U_ptrs, bs_batch, Q_ptrs, bs_batch, rows));

    check_kblas_error((H2OpusBatched<T, hw>::copyBlock)(stream, bs_batch, detected_ranks, block_size, new_max_rank,
                                                        col_block_V_ptrs, bs_batch, input_ptrs, bs_batch, rows));

    TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::Realloc);
}

template <class T, int hw>
void tlr_potrf_schurcompensation_update(T *D, T *update, int block_size, T eps, T *U_update, T *S_update, T *V_update,
                                        H2OpusTLRPotrfWorkspace<T> &workspace, int *rank_ptr,
                                        h2opusComputeStream_t stream, h2opusHandle_t h2opus_handle)
{
    TLR_Potrf_Phase_Times<hw>::startPhase(TLR_Potrf_Phase_Types::SchurCompensation);
    const int ara_bs = 32, ara_r = 10;

    // thrust::host_vector<T> host_diff(block_size * block_size);
    // copyVector(host_diff, update, block_size * block_size, hw);
    //
    // // Akk = U * S * V
    // std::vector<T> svd_ws(block_size, 0);
    // int ret = lapack_gesvd(block_size, block_size, update, block_size, S_update, U_update, block_size, V_update,
    // block_size, &svd_ws[0]); if(ret != 0)
    //     printf("SVD failed: %d\n", ret);
    //
    // int rank = 0;
    // while(rank < block_size && S_update[rank] > eps)
    //     rank++;
    // printf("Dense update compressed to %d\n", rank);
    //
    // // U = U * S
    // #pragma omp parallel for
    // for(int j = 0; j < rank; j++)
    //     for(int i = 0; i < block_size; i++)
    //         U_update[i + j * block_size] = U_update[i + j * block_size] * S_update[j];
    //
    // // blas_gemm<T, H2OPUS_HWTYPE_CPU>(
    // //   stream, H2Opus_NoTrans, H2Opus_NoTrans, block_size, block_size, rank,
    // //   (T)(-1), U_update, block_size, V_update, block_size, (T)1, vec_ptr(host_diff), block_size
    // // );
    // //
    // // T diff = 0;
    // // for(int i = 0; i < block_size * block_size; i++)
    // //   diff += host_diff[i] * host_diff[i];
    // // printf("Approximation error = %e\n", sqrt(diff));
    //
    // // Akk = Akk - U * V
    // blas_gemm<T, hw>(
    //     stream, H2Opus_NoTrans, H2Opus_NoTrans, block_size, block_size, rank,
    //     (T)(-1), U_update, block_size, V_update, block_size, (T)1, D, block_size
    // );

    int rank;
    // eps = eps * 50 * sqrt(2 / M_PI);

    if (hw == H2OPUS_HWTYPE_CPU)
    {
        // copyArray(update, U_update, block_size * block_size, stream, hw);
        //
        // check_kblas_error((H2OpusBatched<T, hw>::geqp2)(
        //     stream, block_size, block_size, U_update, block_size, block_size * block_size,
        //     S_update, block_size, rank_ptr, eps, 1
        // ) );
        //
        // rank = thrust_get_value<hw>(rank_ptr);
        //
        // check_kblas_error((H2OpusBatched<T, hw>::orgqr)(
        //     stream, block_size, rank, U_update, block_size, block_size * block_size,
        //     S_update, block_size, 1
        // ) );
        // // printf("Dense update compressed to %d\n", rank);
        //
        // blas_gemm<T, hw>(
        //     stream, H2Opus_Trans, H2Opus_NoTrans, rank, block_size, block_size,
        //     (T)1, U_update, block_size, update, block_size, (T)0, V_update, block_size
        // );
        //     // Akk = Akk - U * V
        // blas_gemm<T, hw>(
        //     stream, H2Opus_NoTrans, H2Opus_NoTrans, block_size, block_size, rank,
        //     (T)(-1), U_update, block_size, V_update, block_size, (T)1, D, block_size
        // );

        // Grab workspace
        T *ara_Z = workspace.svd_ws;
        T *ara_R_diag = ara_Z + block_size * block_size;

        rank = h2opus_ara(stream, block_size, block_size, update, block_size, U_update, block_size, V_update,
                          block_size, ara_Z, block_size, ara_R_diag, eps, ara_bs, ara_r, block_size, h2opus_handle);

        // Akk = Akk - U * V'
        // blas_gemm<T, hw>(
        //     stream, H2Opus_NoTrans, H2Opus_Trans, block_size, block_size, rank,
        //     (T)(-1), U_update, block_size, V_update, block_size, (T)1, D, block_size
        // );

        // Compute D = D - update + diag(Sc_diag)
        // Sc_diag = row_sum(abs(S_diff)) (S_diff is symmetric)
        // S_diff = update - U * V'

        // D = D - update
        blas_axpy<T, hw>(stream, block_size * block_size, (T)(-1), update, 1, D, 1);
        if (rank != block_size)
        {
            T *S_diff = workspace.svd_ws;
            h2opus_fbl_lacpy(H2OpusFBLAll, block_size, block_size, update, block_size, ara_Z, block_size);
            blas_gemm<T, hw>(stream, H2Opus_NoTrans, H2Opus_Trans, block_size, block_size, rank, (T)(-1), U_update,
                             block_size, V_update, block_size, (T)1, S_diff, block_size);

            for (int i = 0; i < block_size; i++)
            {
                T scdiag = 0;
                for (int j = 0; j < block_size; j++)
                    scdiag += fabs(S_diff[j + i * block_size]);

                D[i + i * block_size] += scdiag;
            }
        }
    }
    else
    {
#ifdef H2OPUS_USE_GPU
        int *bs_batch = workspace.sampling_bs_batch;

        kblas_ara_batch(stream->getKblasHandle(), bs_batch, bs_batch, workspace.ptr_svd, bs_batch,
                        workspace.ptr_svd + 1, bs_batch, workspace.ptr_svd + 2, bs_batch, rank_ptr, eps, block_size,
                        block_size, block_size, ara_bs, ara_r, h2opus_handle->getKblasRandState(), 0, 1);

        rank = thrust_get_value<hw>(rank_ptr);
        // printf("Dense update compressed to %d\n", rank);

        // Akk = Akk - U * V'
        blas_gemm<T, hw>(stream, H2Opus_NoTrans, H2Opus_Trans, block_size, block_size, rank, (T)(-1), U_update,
                         block_size, V_update, block_size, (T)1, D, block_size);
#endif
    }

    TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::SchurCompensation);
}

template <class T, int hw>
void tlr_potrf_diagonal_update(TTLR_Matrix<T, hw> &A, T sc_eps, int k, H2OpusTLRPotrfWorkspace<T> &workspace,
                               h2opusComputeStream_t stream, h2opusHandle_t h2opus_handle)
{
    // We do multiple updates in parallel to workspace buffers, then accumulate
    // them at the end into the diagonal block
    int block_size = A.block_size;
    int par_dense_updates = workspace.num_dense_parallel_buffers;
    int applied_dense_update = 0;

    int *bs_batch = workspace.dense_bs_batch, *rank_batch = workspace.dense_rank_batch;
    int *max_rank_batch = workspace.dense_max_rank_batch;

    T **ptr_G = workspace.dense_buffer_G, **ptr_T = workspace.dense_buffer_T;
    T **ptr_D = workspace.dense_buffer_D, **ptr_U = workspace.dense_U_ptrs;
    T **ptr_V = workspace.dense_V_ptrs;

    TLR_Potrf_Phase_Times<hw>::startPhase(TLR_Potrf_Phase_Types::DenseUpdate);

#ifdef H2OPUS_TLR_POTRF_USE_MAGMA_FIX
    // Clear out the dense buffers since Magma doesn't set C = C * beta when the inner dimension of the gemm is zero
    if (hw == H2OPUS_HWTYPE_GPU)
    {
        size_t num_dense_buffer_entries = std::min(par_dense_updates, k) * block_size * block_size;
        fillArray(workspace.base_dense_buffer_D, num_dense_buffer_entries, 0, stream, hw);
    }
#endif

    T dense_beta = 0;

    // Marshal the entire block row operations
    tlr_potrf_marshal_dense_updates<T, hw>(vec_ptr(A.block_U_ptrs), vec_ptr(A.block_V_ptrs), k, A.n_block,
                                           vec_ptr(A.block_ranks), ptr_U, ptr_V, rank_batch, NULL, 0, NULL, k, stream);

    // Find out the max rank for the segmented batches of non-uniform gemms
    int *local_seg_maxes = workspace.dense_segment_maxes;
    int num_segments = getSegmentedMaxElements(rank_batch, k, par_dense_updates, local_seg_maxes, stream, hw);
    int *seg_maxes = workspace.dense_segment_maxes_host;
    copyVectorToHost<int>(seg_maxes, local_seg_maxes, num_segments, hw);
    for (int seg = 0; seg < num_segments; seg++)
    {
        int num_updates = std::min(par_dense_updates, k - applied_dense_update);
        int max_subset_rank = seg_maxes[seg];

        // Execute the batches
        // For each low rank update of the form R_i R_i^T where R_i = U_i V_i^T is a low rank block
        // We must first compute G_i = V_i^T V_i, then compute T_i = U_i G_i and finally D_i = T_i U_i^T
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, rank_batch, rank_batch,
                                                       bs_batch, max_subset_rank, max_subset_rank, block_size, (T)1,
                                                       (const T **)ptr_V, bs_batch, (const T **)ptr_V, bs_batch, 0,
                                                       ptr_G, max_rank_batch, num_updates));

        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, bs_batch, rank_batch,
                                                       rank_batch, block_size, max_subset_rank, max_subset_rank, (T)1,
                                                       (const T **)ptr_U, bs_batch, (const T **)ptr_G, max_rank_batch,
                                                       0, ptr_T, bs_batch, num_updates));

        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_Trans, bs_batch, bs_batch,
                                                       rank_batch, block_size, block_size, max_subset_rank, (T)1,
                                                       (const T **)ptr_T, bs_batch, (const T **)ptr_U, bs_batch,
                                                       dense_beta, ptr_D, bs_batch, num_updates));
        applied_dense_update += num_updates;
        rank_batch += num_updates;
        ptr_U += num_updates;
        ptr_V += num_updates;
        dense_beta = 1;
    }

    TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::DenseUpdate);

    // Do a reduction on all buffers to update the diagonal block
    int updates_to_reduce = std::min(k, par_dense_updates);
    if (updates_to_reduce <= 0)
        return;

#ifdef H2OPUS_TLR_USE_SCHUR_COMPENSATION
    if (sc_eps == 0)
    {
#endif
        TLR_Potrf_Phase_Times<hw>::startPhase(TLR_Potrf_Phase_Types::Reduction);
        TLR_Batch<T, hw>::reduceMatrixBuffers(1, vec_ptr(A.diagonal_block_ptrs) + k, bs_batch, bs_batch, bs_batch, -1,
                                              ptr_D, bs_batch, updates_to_reduce, block_size, block_size, 1, stream);
        TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::Reduction);
#ifdef H2OPUS_TLR_USE_SCHUR_COMPENSATION
    }
    else
    {
        T *D_k = workspace.svd_A, *U = workspace.svd_U, *V = workspace.svd_V, *S = workspace.svd_S;
        T **ptr_A = workspace.ptr_svd;

        TLR_Potrf_Phase_Times<hw>::startPhase(TLR_Potrf_Phase_Types::Reduction);

        TLR_Batch<T, hw>::reduceMatrixBuffers(0, ptr_A, bs_batch, bs_batch, bs_batch, 1, ptr_D, bs_batch,
                                              updates_to_reduce, block_size, block_size, 1, stream);

        TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::Reduction);

        T *A_kk = A.getDiagonalBlockHostPtr(k);

        tlr_potrf_schurcompensation_update<T, hw>(A_kk, D_k, block_size, sc_eps, U, S, V, workspace,
                                                  workspace.converged_blocks, stream, h2opus_handle);
    }
#endif
}

template <class T, int hw>
void tlr_potrf_diagonal_factorize(TTLR_Matrix<T, hw> &A, T eps, int k, H2OpusTLRPotrfWorkspace<T> &workspace,
                                  h2opusComputeStream_t stream)
{
    TLR_Potrf_Phase_Times<hw>::startPhase(TLR_Potrf_Phase_Types::Potrf);

    int block_size = A.block_size;
    T *A_kk = A.getDiagonalBlockHostPtr(k);
    int info;

    // Factorize the diagonal block using dense cholesky
#if 1
    lapack_potrf<T, hw>(stream, block_size, A_kk, block_size, &info);
    if (info != 0)
    {
        printf(
            "TLR matrix was semi-definite. Cholesky failed at column %d of block column %d (A = %p lda = %d n = %d)\n",
            info, k, A_kk, block_size, block_size);
        exit(0);
    }
#else
    info = 1;
    // Make a temporary copy of the diagonal block
    std::vector<T> spd_work(block_size * block_size * 4);

    T *A_kk_temp = vec_ptr(spd_work);
    T *LD_temp = A_kk_temp + block_size * block_size;
    T *D_temp = LD_temp + block_size * block_size;
    T *A_temp2 = D_temp + block_size * block_size;
    int *ipiv = workspace.dense_ipiv;

    copyVector(A_kk_temp, hw, A_kk, hw, block_size * block_size);

    lapack_potrf<T, hw>(stream, block_size, A_kk_temp, block_size, &info);
    if (info != 0)
    {
        printf("TLR matrix was semi-definite. Cholesky failed at column %d of block column %d.\n", info, k);
        // Reset A_kk_temp
        copyVector(A_kk_temp, hw, A_kk, hw, block_size * block_size);
        copyVector(A_temp2, hw, A_kk, hw, block_size * block_size);

        // char buffer[256]; sprintf(buffer, "block_%d.bin", k);
        // save_matrix(A_temp2, block_size * block_size, buffer);

        // Make it positive definite
        tlr_util_make_spd<T, hw>(block_size, A_kk_temp, block_size, eps * 10, D_temp, block_size, LD_temp, block_size,
                                 A_kk, block_size, ipiv, stream);

        T diff = 0, norm_A = 0;
        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                norm_A += A_temp2[i + j * block_size] * A_temp2[i + j * block_size];
                diff += (A_temp2[i + j * block_size] - A_kk[i + j * block_size]) *
                        (A_temp2[i + j * block_size] - A_kk[i + j * block_size]);
            }
        }
        printf("SPD Diff = %e | %e\n", sqrt(diff), sqrt(diff / norm_A));

        // Try again - if it fails then all is lost and we abandon ship
        lapack_potrf<T, hw>(stream, block_size, A_kk, block_size, &info);
        if (info != 0)
            printf("Failed to make the diagonal block %d SPD. Failed at column %d\n", k, info);
    }
    else
    {
        // Copy the successfully factorized block back to the TLR matrix
        copyVector(A_kk, hw, A_kk_temp, hw, block_size * block_size);
    }
#endif

    TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::Potrf);
}

template <class T, int hw>
void tlr_potrf_panel_trsm(TTLR_Matrix<T, hw> &A, T eps, int k, H2OpusTLRPotrfWorkspace<T> &workspace,
                          h2opusComputeStream_t stream)
{
    // For j = k+1:end, we have to update each L_jk = A_jk * L_kk^-T = U_jk V_jk^T * L_kk^-T
    // So all we really need to do is update V_jk^T = V_jk^T * L_kk^-T (or V_jk = L_kk^-1 * V_jk)
    // which translate to a triangular solve L_kk X = V_jk
    int batchCount = A.n_block - k - 1;
    int block_size = A.block_size;
    int n_block = A.n_block;

    TLR_Potrf_Phase_Times<hw>::startPhase(TLR_Potrf_Phase_Types::Trsm);

    if (batchCount > 0)
    {
        T *L_kk = A.getDiagonalBlockHostPtr(k);
        int subdiag_index_start = k * n_block + k + 1;

        int *m_batch = workspace.trsm_m, *n_batch = vec_ptr(A.block_ranks) + subdiag_index_start;
        T **A_ptrs = workspace.trsm_A_ptrs, **B_ptrs = vec_ptr(A.block_V_ptrs) + subdiag_index_start;

        fillArray(m_batch, batchCount, block_size, stream, hw);
        fillArray(A_ptrs, batchCount, L_kk, stream, hw);
        int max_n = getMaxElement(n_batch, batchCount, stream, hw);

        check_kblas_error((H2OpusBatched<T, hw>::trsm)(stream, H2Opus_Left, H2Opus_Lower, H2Opus_NoTrans,
                                                       H2Opus_NonUnit, m_batch, n_batch, block_size, max_n, 1, A_ptrs,
                                                       m_batch, B_ptrs, m_batch, batchCount));

#if 0
        // Compress the result using SVD:
        // T2 and T4 from the workspace have enough memory to hold the singular vectors
        T **svd_U_ptrs = workspace.sampling_buffer_T2, **svd_V_ptrs = workspace.sampling_buffer_T4;
        // T1 has enough memory for the singular values
        T **svd_S_ptrs = workspace.sampling_buffer_T1;

        // Copy the old U and V to the sampling input and output buffers of the workspace
        T **old_tlr_U_ptrs = workspace.sampling_input, **old_tlr_V_ptrs = workspace.sampling_output;
        T **col_block_U_ptrs = vec_ptr(A.block_U_ptrs) + subdiag_index_start;
        T **col_block_V_ptrs = B_ptrs;

        int *bs_batch = workspace.sampling_bs_batch;
        int *detected_ranks = workspace.detected_ranks;
        int *current_ranks = n_batch;

        check_kblas_error((H2OpusBatched<T, hw>::copyBlock)(
            stream, bs_batch, current_ranks, block_size, max_n, old_tlr_U_ptrs,
            bs_batch, col_block_U_ptrs, bs_batch, batchCount));

        check_kblas_error((H2OpusBatched<T, hw>::copyBlock)(
            stream, bs_batch, current_ranks, block_size, max_n, old_tlr_V_ptrs,
            bs_batch, col_block_V_ptrs, bs_batch, batchCount));

        // Set V = U_svd S_svd V_svd (V_svd contains the transpose of the right singular vectors)
        check_kblas_error((H2OpusBatched<T, hw>::gesvd)(
            stream, bs_batch, current_ranks, block_size, max_n, old_tlr_V_ptrs, bs_batch,
            svd_U_ptrs, bs_batch, svd_S_ptrs, svd_V_ptrs, bs_batch, eps, detected_ranks,
            batchCount
        ) );

        // printDenseMatrix(old_tlr_V_ptrs[0], bs_batch[0], bs_batch[0], current_ranks[0], 14, "A");
        // printDenseMatrix(svd_U_ptrs[0], bs_batch[0], bs_batch[0], current_ranks[0], 14, "U");
        // printDenseMatrix(svd_S_ptrs[0], bs_batch[0], current_ranks[0], 1, 14, "S");
        // printDenseMatrix(svd_V_ptrs[0], bs_batch[0], current_ranks[0], current_ranks[0], 14, "V");

        // for(int i = 0; i < batchCount; i++)
        //     printf("(%d %d): %d -> %d\n", i + k + 1, k, current_ranks[i], detected_ranks[i]);

        // Update the block column ranks
        A.allocateBlockColumn(k, detected_ranks, stream);
        int max_detected_rank = getMaxElement(detected_ranks, batchCount, stream, hw);

        // Update the pointers and set the new U and V
        col_block_U_ptrs = vec_ptr(A.block_U_ptrs) + subdiag_index_start;
        col_block_V_ptrs = vec_ptr(A.block_V_ptrs) + subdiag_index_start;

        // The new U = U_old * V_svd^T
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_Trans, bs_batch, detected_ranks,
                                                       current_ranks, block_size, max_detected_rank, max_n, (T)1,
                                                       (const T **)old_tlr_U_ptrs, bs_batch, (const T **)svd_V_ptrs, bs_batch,
                                                       0, col_block_U_ptrs, bs_batch, batchCount));

        // The new V = U_svd * S_svd
        check_kblas_error((H2OpusBatched<T, hw>::diagRightMult)(stream, bs_batch, detected_ranks, block_size, max_detected_rank,
                                                            svd_U_ptrs, bs_batch, (const T **)svd_S_ptrs,
                                                            col_block_V_ptrs, bs_batch, batchCount));
#endif
    }

    TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::Trsm);
}

template <class T, int hw>
void tlr_pstrf_pivot_swap(TTLR_Matrix<T, hw> &A, int k, int pivot_index, int *piv,
                          H2OpusTLRPotrfWorkspace<T> &workspace, h2opusComputeStream_t stream)
{
    int n_block = A.n_block;

    if (piv && pivot_index != k)
    {
        // Swap the columns and rows k and pivot_index using only the lower triangular indices
        A.swapDiagonalBlocks(k, pivot_index, stream);
        A.swapBlocks(k, k, 0, n_block, pivot_index, 0, n_block, false, stream);
        if (pivot_index < n_block - 1)
            A.swapBlocks(n_block - pivot_index - 1, pivot_index + 1, k, 1, pivot_index + 1, pivot_index, 1, false,
                         stream);
        A.swapBlocks(pivot_index - k - 1, k + 1, k, 1, pivot_index, k + 1, n_block, true, stream);
        A.transposeBlock(pivot_index, k, stream);

        // Swap the pivot index entries
        swap_vectors(1, piv + k, 1, piv + pivot_index, 1, hw, stream);

        // Swap the dense update pointers
        swap_vectors(1, workspace.dense_buffer_D + k, 1, workspace.dense_buffer_D + pivot_index, 1, hw, stream);
    }
}

template <class T, int hw>
void tlr_pstrf_update_dense_updates(TTLR_Matrix<T, hw> &A, int k, H2OpusTLRPotrfWorkspace<T> &workspace,
                                    h2opusComputeStream_t stream)
{
    int num_updates = A.n_block - k - 1;
    if (num_updates <= 0)
        return;

    TLR_Potrf_Phase_Times<hw>::startPhase(TLR_Potrf_Phase_Types::DenseUpdate);

    int block_size = A.block_size;

    // Marshall the pointers to the update
    int *bs_batch = workspace.dense_bs_batch;
    int *max_rank_batch = workspace.dense_max_rank_batch;

    T **ptr_G = workspace.dense_buffer_G, **ptr_T = workspace.dense_buffer_T;
    T **ptr_D = workspace.dense_buffer_D + k + 1;

    T **ptr_U = vec_ptr(A.block_U_ptrs) + k * A.n_block + k + 1;
    T **ptr_V = vec_ptr(A.block_V_ptrs) + k * A.n_block + k + 1;
    int *rank_batch = vec_ptr(A.block_ranks) + k * A.n_block + k + 1;
    int max_subset_rank = getMaxElement(rank_batch, num_updates, stream, hw);

    // Execute the batches
    // For each low rank update of the form R_i R_i^T where R_i = U_i V_i^T is a low rank block
    // We must first compute G_i = V_i^T V_i, then compute T_i = U_i G_i and finally D_i = T_i U_i^T
    check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, rank_batch, rank_batch,
                                                   bs_batch, max_subset_rank, max_subset_rank, block_size, (T)1,
                                                   (const T **)ptr_V, bs_batch, (const T **)ptr_V, bs_batch, 0, ptr_G,
                                                   max_rank_batch, num_updates));

    check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, bs_batch, rank_batch,
                                                   rank_batch, block_size, max_subset_rank, max_subset_rank, (T)1,
                                                   (const T **)ptr_U, bs_batch, (const T **)ptr_G, max_rank_batch, 0,
                                                   ptr_T, bs_batch, num_updates));

    check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_Trans, bs_batch, bs_batch, rank_batch,
                                                   block_size, block_size, max_subset_rank, (T)1, (const T **)ptr_T,
                                                   bs_batch, (const T **)ptr_U, bs_batch, 1, ptr_D, bs_batch,
                                                   num_updates));

    TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::DenseUpdate);
}

template <class T, int hw>
int tlr_pstrf_select_pivot(TTLR_Matrix<T, hw> &A, int k, H2OpusTLRPotrfWorkspace<T> &workspace,
                           h2opusComputeStream_t stream, h2opusHandle_t h2opus_handle)
{
    int block_size = A.block_size;
    int pivot_index = k;

    struct Compare
    {
        T val;
        int index;
    };
    Compare max;
    max.val = 0;
    max.index = pivot_index;

    TLR_Potrf_Phase_Times<hw>::startPhase(TLR_Potrf_Phase_Types::PivotSelection);

/* Not used so far */
#if 0
#if defined(_OPENMP) && _OPENMP >= 201811
#pragma omp declare reduction(maximum : struct Compare : omp_out = omp_in.val > omp_out.val ? omp_in : omp_out)
#endif
#pragma omp parallel for reduction(maximum : max)
#endif
    for (int i = k; i < A.n_block; i++)
    {
        T *diagonal_block = A.getDiagonalBlockHostPtr(i);
        T *update_block = workspace.dense_buffer_D[i];

        T block_norm = 0;
#pragma omp parallel for reduction(+ : block_norm)
        for (int b = 0; b < block_size * block_size; b++)
            block_norm += (diagonal_block[b] - update_block[b]) * (diagonal_block[b] - update_block[b]);
        block_norm = sqrt(block_norm);

        if (block_norm > max.val)
        {
            max.val = block_norm;
            max.index = i;
        }
    }
    pivot_index = max.index;

    TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::PivotSelection);

    return pivot_index;
}

template <class T, int hw>
void tlr_pstrf_update_diagonal(TTLR_Matrix<T, hw> &A, int k, T eps, H2OpusTLRPotrfWorkspace<T> &workspace,
                               h2opusComputeStream_t stream, h2opusHandle_t h2opus_handle)
{
    int block_size = A.block_size;
#ifndef H2OPUS_TLR_USE_SCHUR_COMPENSATION
    int *bs_batch = workspace.dense_bs_batch;
    TLR_Batch<T, hw>::reduceMatrixBuffers(1, vec_ptr(A.diagonal_block_ptrs) + k, bs_batch, bs_batch, bs_batch, -1,
                                          workspace.dense_buffer_D + k, bs_batch, 1, block_size, block_size, 1, stream);
#else
    T *D = A.getDiagonalBlockHostPtr(k);
    T *update = workspace.dense_buffer_D[k];

    T *U = workspace.svd_U, *V = workspace.svd_V, *S = workspace.svd_S;

    tlr_potrf_schurcompensation_update<T, hw>(D, update, block_size, eps, U, S, V, workspace,
                                              workspace.converged_blocks, stream, h2opus_handle);
#endif
}

template <class T, int hw>
void tlr_potrf(TTLR_Matrix<T, hw> &A, TLR_Potrf_Config<T, hw> &config, int *piv, h2opusHandle_t h2opus_handle)
{
    assert(A.type == H2OpusTLR_Symmetric);

    // grab configuration
    T eps = config.eps;
    T sc_eps = config.sc_eps;
    int ndpb = config.ndpb;
    int nspb = config.nspb;
    int sample_bs = config.sample_bs;

    H2OpusWorkspaceState ws_needed = tlr_potrf_workspace<T, hw>(A, false, ndpb, nspb, sample_bs, piv, h2opus_handle);
    H2OpusWorkspaceState ws_allocated = h2opus_handle->getWorkspaceState();
    if (ws_allocated < ws_needed)
        h2opus_handle->setWorkspaceState(ws_needed);

    H2OpusTLRPotrfWorkspace<T> workspace;
    tlr_potrf_get_workspace(A, workspace, false, ndpb, nspb, sample_bs, piv, h2opus_handle);

    // No point trying to stream the dense part since many kernels have sync points
    h2opusComputeStream_t main_stream = h2opus_handle->getMainStream();
    h2opusComputeStream_t low_priority_stream = main_stream; // h2opus_handle->getLowPriorityStream();

    // Change the type of A to lower triangular so that rank changes will affect the lower
    // triangular half of A
    A.type = H2OpusTLR_LowerTriangular;

    if (piv)
    {
        assert(A.alloc == H2OpusTLRTile);
        generateSequence(piv, A.n_block, 0, main_stream, hw);

        // No GPU implementation yet. still need pivot selection sorted out
        assert(hw == H2OPUS_HWTYPE_CPU);
    }

    // If you're averaging runtimes, make sure to copy the original matrix
    // TTLR_Matrix<T, H2OPUS_HWTYPE_CPU> original_A(A, main_stream);

    TLR_Potrf_Phase_Times<hw>::init();
    float elapsed_time = 0;
    const int nruns = 1;

    PerformanceCounter::clearCounters();

    for (int run = 0; run < nruns; run++)
    {
        // A.copy(original_A, main_stream);

        Timer<hw> timer;
        timer.init();

        timer.start();

        H2OpusEvents &events = h2opus_handle->getEvents();
        events.allocateEvents<hw>(H2OpusDenseEvent, 1);

        // Make sure the low priority stream waits for any work previously
        // submitted on the main stream before launching any work
        events.recordEvent<hw>(H2OpusDenseEvent, 0, main_stream);
        events.streamWaitEvent<hw>(H2OpusDenseEvent, low_priority_stream, 0);

        // Main left looking block cholesky loop
        for (int k = 0; k < A.n_block; k++)
        {
            // Update diagonal block and factorize
            if (piv)
            {
                if (k > 0)
                    tlr_pstrf_update_dense_updates<T, hw>(A, k - 1, workspace, low_priority_stream);

                // Determine a pivot block and swap it with the current block
                if (k < A.n_block - 1)
                {
                    int pivot_index = tlr_pstrf_select_pivot(A, k, workspace, low_priority_stream, h2opus_handle);
                    // int pivot_index = k + rand() % (A.n_block - k);
                    tlr_pstrf_pivot_swap<T, hw>(A, k, pivot_index, piv, workspace, low_priority_stream);
                }

                // Subtract the update dense update to the diagonal block
                if (k > 0)
                    tlr_pstrf_update_diagonal<T, hw>(A, k, sc_eps, workspace, low_priority_stream, h2opus_handle);
            }
            else
                tlr_potrf_diagonal_update<T, hw>(A, sc_eps, k, workspace, low_priority_stream, h2opus_handle);

            tlr_potrf_diagonal_factorize<T, hw>(A, eps, k, workspace, low_priority_stream);

            events.recordEvent<hw>(H2OpusDenseEvent, 0, low_priority_stream);

            // Next, update column k using ARA on the main stream
            tlr_potrf_update_block_column<T, hw>(A, eps, k, workspace, main_stream, h2opus_handle);

            // Make sure the dense updates are done
            events.streamWaitEvent<hw>(H2OpusDenseEvent, main_stream, 0);

            // Finally do a triangular solve on the updated blocks
            tlr_potrf_panel_trsm<T, hw>(A, eps, k, workspace, main_stream);
        }
        elapsed_time += timer.stop();
    }

    // printf("TLR cholesky done in %.4f\n", elapsed_time / nruns);
    const char *phase_names[] = {"Reduction", "Sample", "Projection", "Realloc",   "Orthog",      "Trsm",
                                 "Potrf",     "Clear",  "RandGen",    "SchurComp", "DenseUpdate", "PivotSelect"};
    double total_time = 0;
    for (int i = 0; i < TLR_Potrf_Phase_Types::TLR_Potrf_TotalPhases; i++)
        printf("%12s ", phase_names[i]);
    printf("%12s %12s\n", "Misc", "Total");

    for (int i = 0; i < TLR_Potrf_Phase_Types::TLR_Potrf_TotalPhases; i++)
    {
        total_time += TLR_Potrf_Phase_Times<hw>::phase_times[i] / nruns;
        printf("%12.4f ", TLR_Potrf_Phase_Times<hw>::phase_times[i] / nruns);
    }
    printf("%12.4f %12.4f\n", elapsed_time / nruns - total_time, elapsed_time / nruns);

    // Clear all the uppertriangular halves of the diagonal dense blocks
    H2OpusBatched<T, hw>::setUpperZero(main_stream, A.block_size, A.block_size, vec_ptr(A.diagonal_block_ptrs),
                                       A.block_size, A.n_block);

    // Clear the low ran pointers and ranks for the upper block triangle matrix
    TLR_ClearUpperTriangle<T> ptr_clear(vec_ptr(A.block_U_ptrs), vec_ptr(A.block_V_ptrs), vec_ptr(A.block_ranks),
                                        A.n_block);
    int num_ptrs = A.n_block * A.n_block;
    thrust::for_each(ThrustRuntime<hw>::get(main_stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_ptrs), ptr_clear);

    if (hw == H2OPUS_HWTYPE_CPU)
    {
        double gemm_gops = PerformanceCounter::getOpCount(PerformanceCounter::GEMM);
        double potrf_gops = PerformanceCounter::getOpCount(PerformanceCounter::POTRF);
        double trsm_gops = PerformanceCounter::getOpCount(PerformanceCounter::TRSM);
        PerformanceCounter::clearCounters();

        printf("Total GOPS = %.3f (%.3f gemm, %.3f potrf, %.3f trsm)\n", gemm_gops + potrf_gops + trsm_gops, gemm_gops,
               potrf_gops, trsm_gops);
    }
}

template <class T, int hw>
void tlr_potrf(TTLR_Matrix<T, hw> &A, TLR_Potrf_Config<T, hw> &config, h2opusHandle_t h2opus_handle)
{
    tlr_potrf<T, hw>(A, config, NULL, h2opus_handle);
}

// Set X = (LL^t)^{-1} * X
template <class T, int hw> void tlr_potrs(TTLR_Matrix<T, hw> &L, int nrhs, T *x, int ldx, h2opusHandle_t h2opus_handle)
{
    assert(L.type == H2OpusTLR_LowerTriangular);

    // set X = L^{-1} * X
    tlr_trsm<T, hw>(H2Opus_Left, H2Opus_NoTrans, 1, L, nrhs, x, ldx, h2opus_handle);

    // set X = L^{-T} * X
    tlr_trsm<T, hw>(H2Opus_Left, H2Opus_Trans, 1, L, nrhs, x, ldx, h2opus_handle);
}

#endif
