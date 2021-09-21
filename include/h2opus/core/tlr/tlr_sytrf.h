#ifndef __H2OPUS_TLR_SYTRF_H__
#define __H2OPUS_TLR_SYTRF_H__

#include <h2opus/core/hara_util.cuh>
#include <h2opus/util/error_approximation.h>
#include <h2opus/util/host_ara.h>

#include <h2opus/core/tlr/tlr_defs.h>
#include <h2opus/core/tlr/tlr_struct.h>
#include <h2opus/core/tlr/tlr_potrf_workspace.h>
#include <h2opus/core/tlr/tlr_potrf_marshal.h>
#include <h2opus/core/tlr/tlr_batch.h>

#define H2OPUS_TLR_SYTRF_USE_MAGMA_FIX

// TODO: a lot of this stuff is common with the tlr potrf. A unified ARA interface should be written
//       for all TLR operations. I started it, but didn't have time to finish before I had to skedaddle

////////////////////////////////////////////////////////////////////////////
// Utility
////////////////////////////////////////////////////////////////////////////

// Timing stuff
namespace TLR_Sytrf_Phase_Types
{
enum Phase
{
    Reduction = 0,
    Sample,
    Projection,
    Realloc,
    Orthog,
    Trsm,
    Sytrf,
    Clear,
    RandGen,
    DenseUpdate,
    TLR_Sytrf_TotalPhases
};
};

template <int hw> struct TLR_Sytrf_Phase_Times
{
    static double phase_times[TLR_Sytrf_Phase_Types::TLR_Sytrf_TotalPhases];
    static Timer<hw> timer;
    static int currentPhase;

    static void init()
    {
        timer.init();
        for (int i = 0; i < TLR_Sytrf_Phase_Types::TLR_Sytrf_TotalPhases; i++)
            phase_times[i] = 0;
        currentPhase = -1;
    }

    static void startPhase(TLR_Sytrf_Phase_Types::Phase type)
    {
        currentPhase = type;
        timer.start();
    }

    static void endPhase(TLR_Sytrf_Phase_Types::Phase type)
    {
        assert(currentPhase == type);

        phase_times[type] += timer.stop();
        currentPhase = -1;
    }
};

template <int hw> Timer<hw> TLR_Sytrf_Phase_Times<hw>::timer;

template <int hw> double TLR_Sytrf_Phase_Times<hw>::phase_times[TLR_Sytrf_Phase_Types::TLR_Sytrf_TotalPhases];

template <int hw> int TLR_Sytrf_Phase_Times<hw>::currentPhase;

template <class T, int hw>
void expandTLRBlock(TTLR_Matrix<T, hw> &A, int i, int j, T *block, h2opusComputeStream_t stream)
{
    int bs = A.block_size, nb = A.n_block;

    if (i != j)
    {
        T *U_ij = A.block_U_ptrs[i + j * nb], *V_ij = A.block_V_ptrs[i + j * nb];
        int rank_ij = A.block_ranks[i + j * nb];

        blas_gemm<T, hw>(stream, H2Opus_NoTrans, H2Opus_Trans, bs, bs, rank_ij, 1, U_ij, bs, V_ij, bs, 0, block, bs);
    }
    else
    {
        T *A_ii = A.diagonal_block_ptrs[i];
        for (int j = 0; j < bs * bs; j++)
            block[j] = A_ii[j];
    }
}

// A = A * D
template <class T, int hw> void diagScaleBlock(int m, int n, T *A, int lda, T *D)
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            A[i + j * lda] *= D[j];
}

template <class T> T matrixBlockDifference(int m, int n, T *A, int lda, T *B, int ldb)
{
    T diff = 0;
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            T err = (A[i + j * lda] - B[i + j * ldb]);
            diff += err * err;
        }
    }
    return sqrt(diff);
}

template <class T, int hw>
void expandLRU(TTLR_Matrix<T, hw> &A, T *D, int i, int k, T *update, h2opusComputeStream_t stream)
{
    int bs = A.block_size;
    std::vector<T> Aij(bs * bs, 0), Akj(bs * bs, 0);

    expandTLRBlock<T, hw>(A, i, k, update, stream);

    for (int j = 0; j < k; j++)
    {
        T *D_jj = D + j * bs;

        expandTLRBlock<T, hw>(A, i, j, vec_ptr(Aij), stream);
        expandTLRBlock<T, hw>(A, k, j, vec_ptr(Akj), stream);

        // A_ik = Aik - Aij * D_jj * Akj^t
        diagScaleBlock<T, hw>(bs, bs, vec_ptr(Aij), bs, D_jj);

        blas_gemm<T, hw>(stream, H2Opus_NoTrans, H2Opus_Trans, bs, bs, bs, -1, vec_ptr(Aij), bs, vec_ptr(Akj), bs, 1,
                         update, bs);
    }
}

////////////////////////////////////////////////////////////////////////////
// Main subroutines
////////////////////////////////////////////////////////////////////////////
// Sample the sum of the low rank updates coming from the block rows of a sub-matrix of A
// i.e. compute output_i = sum_{j=col_start:col_end} A_ij * D_jj * A_kj^T * input_i for each block in row i
template <class T, int hw, bool transpose>
void tlr_sytrf_sample_lru(TTLR_Matrix<T, hw> &A, T *D, int k, int *row_indices, int rows, int col_start, int col_end,
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
    T **D_ptrs = workspace.sampling_D_ptrs;

    // Reuse T1 buffers for T3
    T **T3_ptrs = workspace.sampling_buffer_T1, **T4_ptrs = workspace.sampling_buffer_T4;
    T **input_i_ptrs = workspace.samplinge_input_i_ptrs;

#ifdef H2OPUS_TLR_SYTRF_USE_MAGMA_FIX
    TLR_Sytrf_Phase_Times<hw>::startPhase(TLR_Sytrf_Phase_Types::Clear);
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
    TLR_Sytrf_Phase_Times<hw>::endPhase(TLR_Sytrf_Phase_Types::Clear);
#endif

    TLR_Sytrf_Phase_Times<hw>::startPhase(transpose ? TLR_Sytrf_Phase_Types::Projection
                                                    : TLR_Sytrf_Phase_Types::Sample);

    // Sample multiple block columns at a time when we can
    // Each sample goes into its own buffer, and we do a reduction
    // on the buffers when we are done
    int sampled_block_cols = col_start;
    T T4_beta = 0;

    while (sampled_block_cols <= col_end)
    {
        // Sample the low rank updates:
        // T4 += A_ij * D_jj * A_kj^T * R = U_ij * V_ij^T * D_jj * V_kj (U_kj^T * R) = U_ij * V_ij^T * D_jj * (V_kj *
        // T1)
        //                                = U_ij * V_ij^T * (D_jj * T2) = U_ij * V_ij^T * T2 = U_ij * T3
        int block_columns = std::min(par_block_cols, col_end - sampled_block_cols + 1);
        int sample_block_count = block_columns * rows;
        // Marshal the blocks that need to be sampled in this set of block columns
        tlr_potrf_marshal_lru_sample_range<T, hw, transpose>(
            vec_ptr(A.block_U_ptrs), vec_ptr(A.block_V_ptrs), vec_ptr(A.block_ranks), k, A.n_block, row_indices, rows,
            sampled_block_cols, block_columns, Uij_ptrs, Vij_ptrs, Ukj_ptrs, Vkj_ptrs, input_ptrs, input_i_ptrs,
            samples_batch, samples_i_batch, rank_ij_batch, rank_kj_batch, D, block_size, D_ptrs, sample_block_count,
            stream);

        int max_rank_kj = getMaxElement(rank_kj_batch, sample_block_count, stream, hw);
        int max_rank_ij = getMaxElement(rank_ij_batch, sample_block_count, stream, hw);

        // Now that we marshaled the low rank pointers, execute the needed gemms
        // T1 = U_kj^T * R
        // printf("Sampling Ukj\n");
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, rank_kj_batch,
                                                       samples_i_batch, bs_batch, max_rank_kj, max_samples, block_size,
                                                       (T)1, (const T **)Ukj_ptrs, bs_batch, (const T **)input_i_ptrs,
                                                       bs_batch, 0, T1_ptrs, max_rank_batch, sample_block_count));

        // T2 = V_kj * T1
        // printf("Sampling Vkj\n");
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, bs_batch,
                                                       samples_i_batch, rank_kj_batch, block_size, max_samples,
                                                       max_rank_kj, (T)1, (const T **)Vkj_ptrs, bs_batch,
                                                       (const T **)T1_ptrs, max_rank_batch, 0, T2_ptrs, bs_batch,
                                                       sample_block_count));

        // T2 = D_jj * T2
        check_kblas_error((H2OpusBatched<T, hw>::diagLeftMult)(stream, bs_batch, samples_i_batch, block_size,
                                                               max_samples, (const T **)D_ptrs, T2_ptrs, bs_batch,
                                                               T2_ptrs, bs_batch, sample_block_count));

        // T3 = V_ij^T * T2
        // printf("Sampling Vij\n");
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, rank_ij_batch,
                                                       samples_i_batch, bs_batch, max_rank_ij, max_samples, block_size,
                                                       (T)1, (const T **)Vij_ptrs, bs_batch, (const T **)T2_ptrs,
                                                       bs_batch, 0, T3_ptrs, max_rank_batch, sample_block_count));

        // T4 += U_ij * T3
        // printf("Sampling Uij\n");
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, bs_batch,
                                                       samples_i_batch, rank_ij_batch, block_size, max_samples,
                                                       max_rank_ij, (T)1, (const T **)Uij_ptrs, bs_batch,
                                                       (const T **)T3_ptrs, max_rank_batch, T4_beta, T4_ptrs, bs_batch,
                                                       sample_block_count));

        sampled_block_cols += block_columns;
        T4_beta = 1;
    }

    TLR_Sytrf_Phase_Times<hw>::endPhase(transpose ? TLR_Sytrf_Phase_Types::Projection : TLR_Sytrf_Phase_Types::Sample);

    TLR_Sytrf_Phase_Times<hw>::startPhase(TLR_Sytrf_Phase_Types::Reduction);

    // Do a reduction on the parallel buffers
    if (par_block_cols > 0)
    {
        TLR_Batch<T, hw>::reduceMatrixBuffers(0, output_ptrs, bs_batch, bs_batch, samples_batch, -1, T4_ptrs, bs_batch,
                                              par_block_cols, block_size, max_samples, rows, stream);
    }
    TLR_Sytrf_Phase_Times<hw>::endPhase(TLR_Sytrf_Phase_Types::Reduction);
}

template <class T, int hw, bool transpose>
void tlr_sytrf_sample_col(TTLR_Matrix<T, hw> &A, int k, int *row_indices, int rows, T **input_ptrs, T **output_ptrs,
                          int *samples_batch, int max_samples, H2OpusTLRPotrfWorkspace<T> &workspace,
                          h2opusComputeStream_t stream)
{
    TLR_Sytrf_Phase_Times<hw>::startPhase(transpose ? TLR_Sytrf_Phase_Types::Projection
                                                    : TLR_Sytrf_Phase_Types::Sample);

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

    TLR_Sytrf_Phase_Times<hw>::endPhase(transpose ? TLR_Sytrf_Phase_Types::Projection : TLR_Sytrf_Phase_Types::Sample);
}

// Generate an approximation of the low rank updated block column k
// in left looking cholesky
template <class T, int hw>
void tlr_sytrf_update_block_column(TTLR_Matrix<T, hw> &A, T *D, T eps, int k, H2OpusTLRPotrfWorkspace<T> &workspace,
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

    TLR_Sytrf_Phase_Times<hw>::startPhase(TLR_Sytrf_Phase_Types::Clear);

    ////////////////////////////////////////////////////////////////////////
    // Generate an approximation basis Q_i for the blocks of the column
    ////////////////////////////////////////////////////////////////////////
    // Clear the output
    // size_t output_entries = block_size * max_rank * rows;
    // fillArray(workspace.base_buffer_output, output_entries, 0, stream, hw);
    fillArray(detected_ranks, rows, 0, stream, hw);
    fillArray(small_vectors, rows, 0, stream, hw);
    fillArray(samples_batch, rows, sample_bs, stream, hw);

    TLR_Sytrf_Phase_Times<hw>::endPhase(TLR_Sytrf_Phase_Types::Clear);

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

        TLR_Sytrf_Phase_Times<hw>::startPhase(TLR_Sytrf_Phase_Types::RandGen);

        // Generate the random gaussian input vectors - each one of size block_size x samples
        check_kblas_error((H2OpusBatched<T, hw>::rand)(stream, h2opus_handle, bs_batch, samples_batch, block_size,
                                                       sample_bs, input_ptrs, bs_batch, n_rows));

        TLR_Sytrf_Phase_Times<hw>::endPhase(TLR_Sytrf_Phase_Types::RandGen);

        // Sample the sum of the low rank updates in each block row in the sub-matrix A(k+1:end, 0:k-1)
        tlr_sytrf_sample_lru<T, hw, false>(A, D, k, row_indices, n_rows, 0, k - 1, input_ptrs, sub_Y_ptrs,
                                           samples_batch, sample_bs, workspace, stream);

        // Sample the current block column k and subtract the result from the previously accumulated samples
        // i.e A_ik * R_i - sum_{j=0:k-1} A_ij * A_kj^T * R_i
        tlr_potrf_sample_col<T, hw, false>(A, k, row_indices, n_rows, input_ptrs, sub_Y_ptrs, samples_batch, sample_bs,
                                           workspace, stream);

        TLR_Sytrf_Phase_Times<hw>::startPhase(TLR_Sytrf_Phase_Types::Orthog);

        int current_max_rank = getMaxElement(subset_ranks, n_rows, stream, hw);

        // Generate orthogonal basis from the samples and check for convergence
        tlr_ara_gen_basis<T, hw>(sub_Q_ptrs, bs_batch, bs_batch, subset_ranks, block_size, current_max_rank, sub_Y_ptrs,
                                 bs_batch, samples_batch, sample_bs, Z_ptrs, ldz_batch, n_rows,
                                 workspace.orthog_workspace, stream);

        TLR_Sytrf_Phase_Times<hw>::endPhase(TLR_Sytrf_Phase_Types::Orthog);

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
    TLR_Sytrf_Phase_Times<hw>::startPhase(TLR_Sytrf_Phase_Types::Clear);
    // Clear the input
    // size_t input_entries = block_size * A.max_rank * rows;
    // fillArray(workspace.base_buffer_input, input_entries, 0, stream, hw);

    // Reset the row index array
    generateSequence(row_indices, rows, k + 1, stream, hw);

    // Keep track of how many samples were taken
    int *current_samples = small_vectors;
    fillArray(current_samples, rows, 0, stream, hw);

    TLR_Sytrf_Phase_Times<hw>::endPhase(TLR_Sytrf_Phase_Types::Clear);

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
        tlr_sytrf_sample_lru<T, hw, true>(A, D, k, row_indices, n_rows, 0, k - 1, sub_Q_ptrs, sub_B_ptrs, samples_batch,
                                          max_samples, workspace, stream);

        // Sample the current block column k and subtract the result from the previously accumulated samples
        // i.e A_ik * R_i - sum_{j=0:k-1} A_ij * A_kj^T * R_i
        tlr_sytrf_sample_col<T, hw, true>(A, k, row_indices, n_rows, sub_Q_ptrs, sub_B_ptrs, samples_batch, max_samples,
                                          workspace, stream);

        row_indices += n_rows;
        projected_blocks += n_rows;
    }

    TLR_Sytrf_Phase_Times<hw>::startPhase(TLR_Sytrf_Phase_Types::Realloc);

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

    TLR_Sytrf_Phase_Times<hw>::endPhase(TLR_Sytrf_Phase_Types::Realloc);
}

template <class T, int hw>
void tlr_sytrf_diagonal_update(TTLR_Matrix<T, hw> &A, T *D, T eps, int k, H2OpusTLRPotrfWorkspace<T> &workspace,
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
    T **ptr_V = workspace.dense_V_ptrs, **Di_ptrs = workspace.dense_Di_ptrs;

    TLR_Sytrf_Phase_Times<hw>::startPhase(TLR_Sytrf_Phase_Types::DenseUpdate);

#ifdef H2OPUS_TLR_SYTRF_USE_MAGMA_FIX
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
                                           vec_ptr(A.block_ranks), ptr_U, ptr_V, rank_batch, D, block_size, Di_ptrs, k,
                                           stream);

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
        // We must first compute G_i = V_i^T * D_i * V_i, then compute T_i = U_i G_i and finally D_i = T_i U_i^T

        // Use T_i to store D_i * V_i
        check_kblas_error((H2OpusBatched<T, hw>::diagLeftMult)(stream, bs_batch, rank_batch, block_size,
                                                               max_subset_rank, (const T **)Di_ptrs, ptr_V, bs_batch,
                                                               ptr_T, bs_batch, num_updates));

        // Now we can set G = V_i^T D_i * V_i = V_i^T T_i
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, rank_batch, rank_batch,
                                                       bs_batch, max_subset_rank, max_subset_rank, block_size, (T)1,
                                                       (const T **)ptr_V, bs_batch, (const T **)ptr_T, bs_batch, 0,
                                                       ptr_G, max_rank_batch, num_updates));

        // T_i = U_i G_i
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, bs_batch, rank_batch,
                                                       rank_batch, block_size, max_subset_rank, max_subset_rank, (T)1,
                                                       (const T **)ptr_U, bs_batch, (const T **)ptr_G, max_rank_batch,
                                                       0, ptr_T, bs_batch, num_updates));

        // D_i = T_i U_i^T
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_Trans, bs_batch, bs_batch,
                                                       rank_batch, block_size, block_size, max_subset_rank, (T)1,
                                                       (const T **)ptr_T, bs_batch, (const T **)ptr_U, bs_batch,
                                                       dense_beta, ptr_D, bs_batch, num_updates));
        applied_dense_update += num_updates;
        rank_batch += num_updates;
        ptr_U += num_updates;
        ptr_V += num_updates;
        Di_ptrs += num_updates;
        dense_beta = 1;
    }

    TLR_Sytrf_Phase_Times<hw>::endPhase(TLR_Sytrf_Phase_Types::DenseUpdate);

    // std::vector<T> update(block_size * block_size, 0);
    // expandLRU(A, D, k, k, vec_ptr(update), stream);

    // Do a reduction on all buffers to update the diagonal block
    int updates_to_reduce = std::min(k, par_dense_updates);

    TLR_Sytrf_Phase_Times<hw>::startPhase(TLR_Sytrf_Phase_Types::Reduction);
    if (updates_to_reduce > 0)
    {
        TLR_Batch<T, hw>::reduceMatrixBuffers(1, vec_ptr(A.diagonal_block_ptrs) + k, bs_batch, bs_batch, bs_batch, -1,
                                              ptr_D, bs_batch, updates_to_reduce, block_size, block_size, 1, stream);
    }
    TLR_Sytrf_Phase_Times<hw>::endPhase(TLR_Sytrf_Phase_Types::Reduction);

    // printf("Dense update %d difference = %e\n", k, matrixBlockDifference<T>(block_size, block_size, vec_ptr(update),
    // block_size, A.diagonal_block_ptrs[k], block_size));
}

template <class T, int hw>
void tlr_sytrf_diagonal_factorize(TTLR_Matrix<T, hw> &A, T *D, T eps, int k, H2OpusTLRPotrfWorkspace<T> &workspace,
                                  h2opusComputeStream_t stream)
{
    TLR_Sytrf_Phase_Times<hw>::startPhase(TLR_Sytrf_Phase_Types::Sytrf);

    int block_size = A.block_size;
    T *A_kk = A.getDiagonalBlockHostPtr(k);
    T *D_kk = D + k * block_size;
    int info;

    // Factorize the diagonal block using dense unpivoted sytrf
    // std::vector<T> original_A(A_kk, A_kk + block_size * block_size);
    lapack_sytrf_nopiv<T, hw>(stream, block_size, A_kk, block_size, D_kk, &info);

    // T err = 0;
    // std::vector<T> LD(A_kk, A_kk + block_size * block_size);
    // for(int i = 0; i < block_size; i++)
    //     for(int j = i + 1; j < block_size; j++)
    //         LD[i + j * block_size] = 0;
    // std::vector<T> L = LD;
    // for(int i = 0; i < block_size; i++)
    //     for(int j = 0; j < block_size; j++)
    //         LD[i + j * block_size] *= D_kk[j];
    // blas_gemm<T, hw>(
    //     stream, H2Opus_NoTrans, H2Opus_Trans, block_size, block_size, block_size, -1,
    //     vec_ptr(LD), block_size, vec_ptr(L), block_size, 1, vec_ptr(original_A), block_size
    // );
    // for(int i = 0; i < block_size; i++)
    //     for(int j = 0; j < block_size; j++)
    //         err += original_A[i + j * block_size] * original_A[i + j * block_size];
    // err = sqrt(err);
    // printf("Diagonal LDL error %d = %e\n", k, err);

    if (info != 0)
    {
        printf("TLR diagonal block %d sytrf failed with error %d\n", k, info);
        exit(0);
    }

    TLR_Sytrf_Phase_Times<hw>::endPhase(TLR_Sytrf_Phase_Types::Sytrf);
}

template <class T, int hw>
void tlr_sytrf_panel_trsm(TTLR_Matrix<T, hw> &A, T *D, T eps, int k, H2OpusTLRPotrfWorkspace<T> &workspace,
                          h2opusComputeStream_t stream)
{
    // For j = k+1:end, we have to update each L_jk = A_jk * L_kk^-T * D_kk^-1 = U_jk V_jk^T * L_kk^-T * D_kk^-1
    // So all we really need to do is update V_jk^T = V_jk^T * L_kk^-T  * D_kk^-1 (or V_jk = D_kk^-1 * L_kk^-1 * V_jk)
    // which translate to a triangular solve L_kk X = V_jk and then V_jk = D_kk^-1 * V_jk
    int batchCount = A.n_block - k - 1;
    int block_size = A.block_size;
    int n_block = A.n_block;

    TLR_Sytrf_Phase_Times<hw>::startPhase(TLR_Sytrf_Phase_Types::Trsm);

    if (batchCount > 0)
    {
        T *L_kk = A.getDiagonalBlockHostPtr(k);
        T *D_kk = D + k * block_size;
        int subdiag_index_start = k * n_block + k + 1;

        int *m_batch = workspace.trsm_m, *n_batch = vec_ptr(A.block_ranks) + subdiag_index_start;
        T **A_ptrs = workspace.trsm_A_ptrs, **B_ptrs = vec_ptr(A.block_V_ptrs) + subdiag_index_start;

        fillArray(m_batch, batchCount, block_size, stream, hw);
        fillArray(A_ptrs, batchCount, L_kk, stream, hw);
        int max_n = getMaxElement(n_batch, batchCount, stream, hw);

        check_kblas_error((H2OpusBatched<T, hw>::trsm)(stream, H2Opus_Left, H2Opus_Lower, H2Opus_NoTrans,
                                                       H2Opus_NonUnit, m_batch, n_batch, block_size, max_n, 1, A_ptrs,
                                                       m_batch, B_ptrs, m_batch, batchCount));

        // V_jk = D_kk^-1 * V_jk
        // Reuse A_ptrs for D_Ptrs
        fillArray(A_ptrs, batchCount, D_kk, stream, hw);

        check_kblas_error((H2OpusBatched<T, hw>::diagLeftInvMult)(stream, m_batch, n_batch, block_size, max_n,
                                                                  (const T **)A_ptrs, B_ptrs, m_batch, B_ptrs, m_batch,
                                                                  batchCount));
    }

    TLR_Sytrf_Phase_Times<hw>::endPhase(TLR_Sytrf_Phase_Types::Trsm);
}

template <class T, int hw>
void tlr_sytrf(TTLR_Matrix<T, hw> &A, T *D, T eps, int ndpb, int nspb, int sample_bs, h2opusHandle_t h2opus_handle)
{
    assert(A.type == H2OpusTLR_Symmetric);

    H2OpusWorkspaceState ws_needed = tlr_potrf_workspace<T, hw>(A, true, ndpb, nspb, sample_bs, NULL, h2opus_handle);
    H2OpusWorkspaceState ws_allocated = h2opus_handle->getWorkspaceState();
    if (ws_allocated < ws_needed)
        h2opus_handle->setWorkspaceState(ws_needed);

    H2OpusTLRPotrfWorkspace<T> workspace;
    tlr_potrf_get_workspace(A, workspace, true, ndpb, nspb, sample_bs, NULL, h2opus_handle);

    h2opusComputeStream_t main_stream = h2opus_handle->getMainStream();
    h2opusComputeStream_t low_priority_stream = main_stream; // h2opus_handle->getLowPriorityStream();

    // Change the type of A to lower triangular so that rank changes will affect the lower
    // triangular half of A
    A.type = H2OpusTLR_LowerTriangular;

    // TTLR_Matrix<T, H2OPUS_HWTYPE_CPU> original_A(A, main_stream);

    TLR_Sytrf_Phase_Times<hw>::init();
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
            tlr_sytrf_diagonal_update<T, hw>(A, D, eps, k, workspace, low_priority_stream, h2opus_handle);
            tlr_sytrf_diagonal_factorize<T, hw>(A, D, eps, k, workspace, low_priority_stream);

            events.recordEvent<hw>(H2OpusDenseEvent, 0, low_priority_stream);

            // Next, update column k using ARA on the main stream
            tlr_sytrf_update_block_column<T, hw>(A, D, eps, k, workspace, main_stream, h2opus_handle);

            // Make sure the dense updates are done
            events.streamWaitEvent<hw>(H2OpusDenseEvent, main_stream, 0);

            // Finally do a triangular solve on the updated blocks
            tlr_sytrf_panel_trsm<T, hw>(A, D, eps, k, workspace, main_stream);
        }
        elapsed_time += timer.stop();
    }

    // printf("TLR cholesky done in %.4f\n", elapsed_time / nruns);
    const char *phase_names[] = {"Reduction", "Sample", "Projection", "Realloc", "Orthog",
                                 "Trsm",      "Sytrf",  "Clear",      "RandGen", "DenseUpdate"};
    double total_time = 0;
    for (int i = 0; i < TLR_Sytrf_Phase_Types::TLR_Sytrf_TotalPhases; i++)
        printf("%12s ", phase_names[i]);
    printf("%12s %12s\n", "Misc", "Total");

    for (int i = 0; i < TLR_Sytrf_Phase_Types::TLR_Sytrf_TotalPhases; i++)
    {
        total_time += TLR_Sytrf_Phase_Times<hw>::phase_times[i] / nruns;
        printf("%12.4f ", TLR_Sytrf_Phase_Times<hw>::phase_times[i] / nruns);
    }
    printf("%12.4f %12.4f\n", elapsed_time / nruns - total_time, total_time);

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

// Set X = (LDL^t)^{-1} * X
template <class T, int hw>
void tlr_sytrs(TTLR_Matrix<T, hw> &L, T *D, int nrhs, T *x, int ldx, h2opusHandle_t h2opus_handle)
{
    assert(L.type == H2OpusTLR_LowerTriangular);

    // set X = L^{-1} * X
    tlr_trsm<T, hw>(H2Opus_Left, H2Opus_NoTrans, 1, L, nrhs, x, ldx, h2opus_handle);

    // Set X = D^{-1} * X
    int pn = L.getPaddedDim();
    blas_diagLeftInvMult<T, hw>(h2opus_handle->getMainStream(), pn, nrhs, D, x, ldx, x, ldx);

    // set X = L^{-T} * X
    tlr_trsm<T, hw>(H2Opus_Left, H2Opus_Trans, 1, L, nrhs, x, ldx, h2opus_handle);
}

#endif
