#ifndef __H2OPUS_TLR_GEMM_H__
#define __H2OPUS_TLR_GEMM_H__

#include <h2opus/core/hara_util.cuh>

#include <h2opus/core/tlr/tlr_defs.h>
#include <h2opus/core/tlr/tlr_struct.h>
#include <h2opus/core/tlr/tlr_gemm_workspace.h>
#include <h2opus/core/tlr/tlr_gemm_marshal.h>
#include <h2opus/core/tlr/tlr_batch.h>

template <class T, int hw>
void tlr_gemm_diagonal(T alpha, TTLR_Matrix<T, hw> &A, TTLR_Matrix<T, hw> &B, T beta, TTLR_Matrix<T, hw> &C,
                       H2OpusTLRGemmWorkspace<T> &workspace, h2opusComputeStream_t stream)
{
    int block_size = A.block_size;
    int n_block = A.n_block;

    // First add the products of each diagonal block
    T **A_diag = vec_ptr(A.diagonal_block_ptrs), **B_diag = vec_ptr(B.diagonal_block_ptrs),
      **C_diag = vec_ptr(C.diagonal_block_ptrs);
    int *bs_batch = workspace.dense_bs_batch;
    int *max_rank_batch = workspace.dense_max_rank_batch;

    check_kblas_error((H2OpusBatched<T, hw>::gemm)(
        stream, H2Opus_NoTrans, H2Opus_NoTrans, bs_batch, bs_batch, bs_batch, block_size, block_size, block_size, alpha,
        (const T **)A_diag, bs_batch, (const T **)B_diag, bs_batch, beta, C_diag, bs_batch, n_block));

    // Now expand the low rank products in three steps:
    // C_ij += A_ik B_kj = Ua_ik (Va^T_ik Ub_kj) Vb^T_kj = Ua_ik (T1 Vb^T_kj) = Ua_ik T2^T
    T **T1_buffers = workspace.dense_T1_buffers, **T2_buffers = workspace.dense_T2_buffers;
    T **Ua_ptrs = workspace.dense_Ua_ptrs, **Va_ptrs = workspace.dense_Va_ptrs;
    T **Ub_ptrs = workspace.dense_Ub_ptrs, **Vb_ptrs = workspace.dense_Vb_ptrs;
    int *a_ranks = workspace.dense_a_ranks, *b_ranks = workspace.dense_b_ranks;

    for (int k = 0; k < n_block; k++)
    {
        // Marshall all pointers and ranks from A and B
        tlr_gemm_diagonal_lr_marshal<T, hw>(k, n_block, vec_ptr(A.block_U_ptrs), vec_ptr(A.block_V_ptrs),
                                            vec_ptr(B.block_U_ptrs), vec_ptr(B.block_V_ptrs), vec_ptr(A.block_ranks),
                                            vec_ptr(B.block_ranks), Ua_ptrs, Va_ptrs, Ub_ptrs, Vb_ptrs, a_ranks,
                                            b_ranks, stream);

        int max_a_rank = getMaxElement(a_ranks, n_block, stream, hw);
        int max_b_rank = getMaxElement(b_ranks, n_block, stream, hw);

        // T1 = Va^T_ik Ub_kj
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(
            stream, H2Opus_Trans, H2Opus_NoTrans, a_ranks, b_ranks, bs_batch, max_a_rank, max_b_rank, block_size, alpha,
            (const T **)Va_ptrs, bs_batch, (const T **)Ub_ptrs, bs_batch, (T)0, T1_buffers, max_rank_batch, n_block));

        // T2 = Vb_kj T1^T
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_Trans, bs_batch, a_ranks, b_ranks,
                                                       block_size, max_a_rank, max_b_rank, 1, (const T **)Vb_ptrs,
                                                       bs_batch, (const T **)T1_buffers, max_rank_batch, (T)0,
                                                       T2_buffers, bs_batch, n_block));

        // C += Ua_ik T2^T
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(
            stream, H2Opus_NoTrans, H2Opus_Trans, bs_batch, bs_batch, a_ranks, block_size, block_size, max_a_rank, 1,
            (const T **)Ua_ptrs, bs_batch, (const T **)T2_buffers, bs_batch, (T)1, C_diag, bs_batch, n_block));
    }
}

template <class T, int hw, bool transpose_sample>
void tlr_gemm_sample_AB(T alpha, TTLR_Matrix<T, hw> &A, TTLR_Matrix<T, hw> &B, int *tile_indices, int num_tiles,
                        int *tile_samples, int max_samples, T **input_ptrs, T **output_ptrs,
                        H2OpusTLRGemmWorkspace<T> &workspace, h2opusComputeStream_t stream)
{
    int block_size = A.block_size;
    int n_block = A.n_block;

    int *bs_batch = workspace.sampling_bs_batch;
    int *rank_a_batch = workspace.sampling_rank_a_batch;
    int *rank_b_batch = workspace.sampling_rank_b_batch;
    int *max_rank_batch = workspace.sampling_max_rank_batch;

    // Marshalled data from the TLR matrices
    T **Ua_ptrs = workspace.sampling_Ua_ptrs, **Ub_ptrs = workspace.sampling_Ub_ptrs;
    T **Va_ptrs = workspace.sampling_Va_ptrs, **Vb_ptrs = workspace.sampling_Vb_ptrs;
    T **Da_ptrs = workspace.sampling_Da_ptrs, **Db_ptrs = workspace.sampling_Db_ptrs;

    // Sampling buffers from the workspace
    T **T1_ptrs = workspace.sampling_buffer_T1, **T2_ptrs = workspace.sampling_buffer_T2;
    T **T3_ptrs = workspace.sampling_buffer_T1;

    T output_beta = 0;

    // Sample the sum of the products of all low rank blocks
    for (int k = 0; k < n_block; k++)
    {
        // T4 += A_ik * B_kj * R = Ua_ik * Va^T_ik * Ub_kj (Vb^T_kj * R) = Ua_ik * Va^T_ik * (Ub_kj * T1)
        //                       = Ua_ik * (Va^T_ik * T2) = Ua_ik * T3
        // Marshal the blocks data
        tlr_gemm_marshal_sample_tile_product<T, hw, transpose_sample>(
            k, n_block, vec_ptr(A.block_U_ptrs), vec_ptr(A.block_V_ptrs), vec_ptr(B.block_U_ptrs),
            vec_ptr(B.block_V_ptrs), vec_ptr(A.block_ranks), vec_ptr(B.block_ranks), Ua_ptrs, Va_ptrs, Ub_ptrs, Vb_ptrs,
            rank_a_batch, rank_b_batch, tile_indices, sample_buffer_indices, num_tiles, stream);

        int max_rank_a = getMaxElement(rank_a_batch, num_tiles, stream, hw);
        int max_rank_b = getMaxElement(rank_b_batch, num_tiles, stream, hw);

        // Now that we marshaled the low rank pointers, execute the needed gemms
        // T1 = Vb^T_kj * R
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, rank_b_batch, tile_samples,
                                                       bs_batch, max_rank_b, max_samples, block_size, (T)1,
                                                       (const T **)Vb_ptrs, bs_batch, (const T **)input_ptrs, bs_batch,
                                                       0, T1_ptrs, max_rank_batch, num_tiles));

        // T2 = Ub_kj * T1
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, bs_batch, tile_samples,
                                                       rank_b_batch, block_size, max_samples, max_rank_b, (T)1,
                                                       (const T **)Ub_ptrs, bs_batch, (const T **)T1_ptrs,
                                                       max_rank_batch, 0, T2_ptrs, bs_batch, num_tiles));

        // T3 = Va^T_ik * T2
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, rank_a_batch, tile_samples,
                                                       bs_batch, max_rank_a, max_samples, block_size, (T)1,
                                                       (const T **)Va_ptrs, bs_batch, (const T **)T2_ptrs, bs_batch, 0,
                                                       T3_ptrs, max_rank_batch, num_tiles));

        // T4 += Ua_ik T3
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, bs_batch, tile_samples,
                                                       rank_a_batch, block_size, max_samples, max_rank_a, (T)1,
                                                       (const T **)Ua_ptrs, bs_batch, (const T **)T3_ptrs,
                                                       max_rank_batch, output_beta, output_ptrs, bs_batch, num_tiles));

        output_beta = 1;
    }

    // Final step has one instance of a dense block of A multiplying a low rank block
    // of B and vice versa
    tlr_gemm_marshal_sample_tile_dense_product<T, hw, transpose_sample>(
        n_block, vec_ptr(A.diagonal_block_ptrs), vec_ptr(B.diagonal_block_ptrs), vec_ptr(A.block_U_ptrs),
        vec_ptr(A.block_V_ptrs), vec_ptr(B.block_U_ptrs), vec_ptr(B.block_V_ptrs), vec_ptr(A.block_ranks),
        vec_ptr(B.block_ranks), Da_ptrs, Ub_ptrs, Vb_ptrs, Ua_ptrs, Va_ptrs, Db_ptrs, rank_a_batch, rank_b_batch,
        tile_indices, num_tiles, stream);
    int max_rank_a = getMaxElement(rank_a_batch, num_tiles, stream, hw);
    int max_rank_b = getMaxElement(rank_b_batch, num_tiles, stream, hw);

    // TODO: might get better performance if we cache the product of the dense block
    // with the low rank factor (i.e. AU = A_ii * Ub_ij and VB = B_jj^T * Va_ij)
    // and then sample that low rank factor instead

    int trans_dense_A = (transpose_sample ? H2Opus_Trans : H2Opus_NoTrans);
    int trans_dense_B = (transpose_sample ? H2Opus_Trans : H2Opus_NoTrans);

    // First we do A_ii * B_ij * R
    // T4 += A_ii B_ij * R = A_ii * Ub_ij * (Vb^T_ij * R) = A_ii * (Ub_ij * T1) = A_ii * T2
    // T1 = Vb^T_ij * R
    check_kblas_error((H2OpusBatched<T, hw>::gemm)(
        stream, H2Opus_Trans, H2Opus_NoTrans, rank_b_batch, tile_samples, bs_batch, max_rank_b, max_samples, block_size,
        (T)1, (const T **)Vb_ptrs, bs_batch, (const T **)input_ptrs, bs_batch, 0, T1_ptrs, max_rank_batch, num_tiles));
    // T2 = Ub_ij * T1
    check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, bs_batch, tile_samples,
                                                   rank_b_batch, block_size, max_samples, max_rank_b, (T)1,
                                                   (const T **)Ub_ptrs, bs_batch, (const T **)T1_ptrs, max_rank_batch,
                                                   0, T2_ptrs, bs_batch, num_tiles));

    // T4 += A_ii * T2
    check_kblas_error((H2OpusBatched<T, hw>::gemm)(
        stream, trans_dense_A, H2Opus_NoTrans, bs_batch, tile_samples, bs_batch, block_size, max_samples, block_size,
        (T)1, (const T **)Da_ptrs, bs_batch, (const T **)T2_ptrs, bs_batch, (T)1, output_ptrs, bs_batch, num_tiles));

    // Next we do A_ij * B_jj * R
    // T4 += A_ij * (B_jj * R) = Ua_ij * (Va^T_ij * T2) = Ua_ij * T3
    // T2 = B_jj * R
    check_kblas_error((H2OpusBatched<T, hw>::gemm)(
        stream, trans_dense_B, H2Opus_NoTrans, bs_batch, tile_samples, bs_batch, block_size, max_samples, block_size,
        (T)1, (const T **)Db_ptrs, bs_batch, (const T **)input_ptrs, bs_batch, (T)0, T2_ptrs, bs_batch, num_tiles));
    // T3 = Va^T_ij * T2
    check_kblas_error((H2OpusBatched<T, hw>::gemm)(
        stream, H2Opus_Trans, H2Opus_NoTrans, rank_a_batch, tile_samples, bs_batch, max_rank_a, max_samples, block_size,
        (T)1, (const T **)Va_ptrs, bs_batch, (const T **)T2_ptrs, bs_batch, 0, T3_ptrs, max_rank_batch, num_tiles));

    // T4 += Ua_ij * T3
    check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, bs_batch, tile_samples,
                                                   rank_a_batch, block_size, max_samples, max_rank_a, (T)1,
                                                   (const T **)Ua_ptrs, bs_batch, (const T **)T3_ptrs, max_rank_batch,
                                                   (T)1, output_ptrs, bs_batch, num_tiles));
}

template <class T, int hw>
void tlr_gemm_low_rank(T alpha, TTLR_Matrix<T, hw> &A, TTLR_Matrix<T, hw> &B, T beta, TTLR_Matrix<T, hw> &C, T eps,
                       H2OpusTLRGemmWorkspace<T> &workspace, h2opusComputeStream_t stream, h2opusHandle_t h2opus_handle)
{
    const int r = 10;

    int max_rank = A.max_rank;
    int n_block = A.n_block;
    int block_size = A.block_size;
    int sample_bs = workspace.sample_bs;
    int sampling_buffers = workspace.num_sampling_buffers;

    // Generate the total set of tile indices that need to be approximated
    int *total_tile_set = workspace.sampling_tile_set;
    int total_ops = workspace.total_tiles;
    bool symmetric = (C.type == H2OpusTLR_Symmetric);
    tlr_gemm_generate_tile_set<hw>(total_tile_set, symmetric, n_block, stream);

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

    ////////////////////////////////////////////////////////////////////////
    // Generate an approximation basis Q_i for the blocks of the column
    ////////////////////////////////////////////////////////////////////////
    fillArray(detected_ranks, sampling_buffers, 0, stream, hw);
    fillArray(small_vectors, sampling_buffers, 0, stream, hw);
    fillArray(samples_batch, sampling_buffers, sample_bs, stream, hw);

    int total_ops_completed = 0;
    int *converged_ops_ptr = workspace.converged_blocks;
    fillArray(converged_ops_ptr, 1, 0, stream, hw);

    // Initialize working set with the ops from the full set
    int *working_set_op_indices, *working_set_buffer_ids, *working_set_ranks,
        working_set_size = std::min(sampling_buffers, total_ops);
    int *converged_set_op_indices, *converged_set_buffer_ids, *converged_set_ranks, converged_set_size = 0;
    int *available_buffer_ids, available_buffers = 0;

    fillArray(working_set_ranks, working_set_size, 0, stream, hw);
    generateSequence(working_set_buffer_ids, working_set_size, 0, stream, hw);
    copyArray(total_tile_set, working_set_op_indices, working_set_size, stream, hw);

    // How many ops should converge before projecting them
    int projection_threshold = (working_set_size > 32 ? working_set_size / 2 : working_set_size);

    while (total_ops_completed < total_ops)
    {
        // working_set_size = std::min(sampling_buffers, total_ops - total_converged_ops) - converged_subset_ops;

        // Set the Q and Y pointers based on the current ranks and the selected block rows
        tlr_ara_working_set_basis_ptrs<T, hw>(sub_Q_ptrs, sub_Y_ptrs, Q_ptrs, bs_batch, working_set_buffer_ids,
                                              working_set_ranks, working_set_size, stream);

        // Generate the random gaussian input vectors - each one of size block_size x samples
        check_kblas_error((H2OpusBatched<T, hw>::rand)(stream, h2opus_handle, bs_batch, samples_batch, block_size,
                                                       sample_bs, input_ptrs, bs_batch, working_set_size));

        tlr_gemm_sample_AB<T, hw, false>(alpha, A, B, working_set_op_indices, working_set_size, samples_batch,
                                         sample_bs, input_ptrs, sub_Y_ptrs, workspace, stream);

        int current_max_rank = getMaxElement(working_set_ranks, working_set_size, stream, hw);

        // Generate orthogonal basis from the samples and check for convergence
        tlr_ara_gen_basis<T, hw>(sub_Q_ptrs, bs_batch, bs_batch, working_set_ranks, block_size, current_max_rank,
                                 sub_Y_ptrs, bs_batch, samples_batch, sample_bs, Z_ptrs, ldz_batch, num_ops,
                                 workspace.orthog_workspace, stream);

        // Count the number of vectors that have a small magnitude
        // also updates the rank, max diagonal and advances the Y_batch pointers
        hara_util_svec_count_batch<T, R_prec, hw>(samples_batch, working_set_ranks, small_vectors, R_diag, r,
                                                  (R_prec)eps, sample_bs, sub_Y_ptrs, sub_Q_ptrs, bs_batch, num_ops,
                                                  stream);

        // Update converged list
        converged_set_size = tlr_ara_update_converged<hw>(
            working_set_op_indices, working_set_buffer_ids, working_set_ranks, working_set_size,
            converged_set_op_indices, converged_set_buffer_ids, converged_set_ranks, converged_set_size, samples_batch,
            small_vectors, converged_ops_ptr, sample_bs, max_rank, r, stream);

        // Check if I should project now
        if (converged_set_size >= projection_threshold)
        {

            // Move the converged buffer ids over to the available buffers ids
            copyArray(converged_set_buffer_ids, available_buffer_ids + available_buffers, converged_set_size, stream,
                      hw);
            available_buffers += converged_set_size;
            converged_set_size = 0;
        }

        // Check if I should add more
        if (available_buffers > 0)
        {
        }
    }
    /*
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
        int* current_samples = small_vectors;
        fillArray(current_samples, rows, 0, stream, hw);

        TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::Clear);

        T** sub_B_ptrs = sub_Y_ptrs;
        int projected_blocks = 0;
        block_rows = rows;
        while(projected_blocks < rows)
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
            tlr_potrf_sample_lru<T, hw, true>(A, k, row_indices, n_rows, 0, k - 1, sub_Q_ptrs, sub_B_ptrs,
                                                samples_batch, max_samples, workspace, stream);

            // Sample the current block column k and subtract the result from the previously accumulated samples
            // i.e A_ik * R_i - sum_{j=0:k-1} A_ij * A_kj^T * R_i
            tlr_potrf_sample_col<T, hw, true>(A, k, row_indices, n_rows, sub_Q_ptrs, sub_B_ptrs, samples_batch,
       max_samples, workspace, stream);

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

        check_kblas_error((H2OpusBatched<T, hw>::copyBlock)(
            stream, bs_batch, detected_ranks, block_size, new_max_rank, col_block_U_ptrs,
            bs_batch, Q_ptrs, bs_batch, rows));

        check_kblas_error((H2OpusBatched<T, hw>::copyBlock)(
            stream, bs_batch, detected_ranks, block_size, new_max_rank, col_block_V_ptrs,
            bs_batch, input_ptrs, bs_batch, rows));

        TLR_Potrf_Phase_Times<hw>::endPhase(TLR_Potrf_Phase_Types::Realloc);
    */
}

template <class T, int hw>
void tlr_gemm(T alpha, TTLR_Matrix<T, hw> &A, TTLR_Matrix<T, hw> &B, T beta, TTLR_Matrix<T, hw> &C, T eps,
              int sample_bs, int num_sampling_buffers, h2opusHandle_t h2opus_handle)
{
    H2OpusWorkspaceState ws_needed = tlr_gemm_workspace<T, hw>(A, B, C, sample_bs, num_sampling_buffers, h2opus_handle);
    H2OpusWorkspaceState ws_allocated = h2opus_handle->getWorkspaceState();
    if (ws_allocated < ws_needed)
        h2opus_handle->setWorkspaceState(ws_needed);

    H2OpusTLRGemmWorkspace<T> workspace;
    tlr_gemm_get_workspace(A, B, C, sample_bs, num_sampling_buffers, workspace, h2opus_handle);

    h2opusComputeStream_t stream = h2opus_handle->getMainStream();

    tlr_gemm_diagonal(alpha, A, B, beta, C, workspace, stream);

    // std::vector<int> tile_indices(1, 1), tile_samples(1, 4);
    //
    // check_kblas_error((H2OpusBatched<T, hw>::rand)(stream, h2opus_handle, workspace.sampling_bs_batch,
    // vec_ptr(tile_samples), A.block_size,
    //                                             4, workspace.sampling_input, workspace.sampling_bs_batch, 1));
    //
    // printDenseMatrix(workspace.sampling_input[0], A.block_size, A.block_size, tile_samples[0], 8, "R");
    //
    // tlr_gemm_sample_AB<T, hw, false>(
    //     alpha, A, B, vec_ptr(tile_indices), 1, vec_ptr(tile_samples), 4,
    //     workspace.sampling_input, workspace.sampling_output, workspace, stream
    // );
    //
    // printDenseMatrix(workspace.sampling_output[0], A.block_size, A.block_size, tile_samples[0], 8, "Y");
    //
    // tlr_gemm_sample_AB<T, hw, true>(
    //     alpha, A, B, vec_ptr(tile_indices), 1, vec_ptr(tile_samples), 4,
    //     workspace.sampling_output, workspace.sampling_input, workspace, stream
    // );
    //
    // printDenseMatrix(workspace.sampling_input[0], A.block_size, A.block_size, tile_samples[0], 8, "Q");

    tlr_gemm_low_rank<T, hw>(alpha, A, B, beta, C, eps, workspace, stream, h2opus_handle);
}

#endif
