#ifndef __H2OPUS_TLR_ARA_UTIL_H__
#define __H2OPUS_TLR_ARA_UTIL_H__

#include <h2opus/core/h2opus_compute_stream.h>
#include <h2opus/core/h2opus_workspace.h>
#include <h2opus/core/tlr/tlr_defs.h>

/////////////////////////////////////////////////////////////////////////////////////////////////
// Convergence check
/////////////////////////////////////////////////////////////////////////////////////////////////
struct TLR_ARA_ConvergenceChecker
{
  private:
    int *subset_samples, *op_global_indices, *subset_op_ranks, *subset_small_vectors, *converged_ops;
    int *op_ranks, sample_bs, max_rank, r, _subset_ops, total_ops, op_global_offset;

  public:
    TLR_ARA_ConvergenceChecker(int *subset_samples, int *op_global_indices, int *subset_op_ranks,
                               int *subset_small_vectors, int *converged_ops, int *op_ranks, int sample_bs,
                               int max_rank, int r, int subset_ops, int op_global_offset, int total_ops)
    {
        this->subset_samples = subset_samples;
        this->op_global_indices = op_global_indices;
        this->subset_op_ranks = subset_op_ranks;
        this->subset_small_vectors = subset_small_vectors;
        this->converged_ops = converged_ops;
        this->op_ranks = op_ranks;
        this->sample_bs = sample_bs;
        this->max_rank = max_rank;
        this->r = r;
        this->_subset_ops = subset_ops;
        this->op_global_offset = op_global_offset;
        this->total_ops = total_ops;
    }

    __host__ __device__ void operator()(const unsigned int &c) const
    {
        int total_converged_ops = *converged_ops;
        int subset_ops = _subset_ops;

        // Compact the subset
        for (int op_index = 0; op_index < subset_ops; op_index++)
        {
            int max_samples = max_rank - subset_op_ranks[op_index];
            if (subset_small_vectors[op_index] >= r)
                max_samples = 0;

            // The operation converged
            if (max_samples == 0)
            {
                // Grab another operation to work on
                int next_global_index = subset_ops + total_converged_ops;
                int converged_op_index = op_global_indices[op_index];
                int converged_rank = subset_op_ranks[op_index];

                // See if there's anything more outside the current subset
                // Otherwise we're in the final stretch and the subset is all that's left,
                // so start eating that up
                if (next_global_index < total_ops)
                {
                    op_global_indices[op_index] = op_global_indices[next_global_index];
                    op_global_indices[next_global_index] = -1;

                    // Reset the convergence stats in the local subset
                    subset_small_vectors[op_index] = 0;
                    subset_op_ranks[op_index] = 0;
                    subset_samples[op_index] = sample_bs;
                }
                else
                {
                    subset_ops--;
                    op_global_indices[op_index] = op_global_indices[subset_ops];
                    subset_small_vectors[op_index] = subset_small_vectors[subset_ops];
                    subset_op_ranks[op_index] = subset_op_ranks[subset_ops];

                    // Take a step back
                    op_index--;
                }
                total_converged_ops++;

                // Save the detected rank
                op_ranks[converged_op_index - op_global_offset] = converged_rank;
            }
            else
            {
                int s = (max_samples < sample_bs ? max_samples : sample_bs);
                subset_samples[op_index] = s;
            }
        }

        *converged_ops = total_converged_ops;
    }
};

struct TLR_ARA_ConvergenceUpdater
{
  private:
    int *working_set_op_indices, *working_set_buffer_ids, *working_set_ranks, working_set_size;
    int *converged_set_op_indices, *converged_set_buffer_ids, *converged_set_ranks, converged_set_size;
    int *samples_batch, *small_vectors, *converged_ops_ptr, sample_bs, max_rank, r;

  public:
    TLR_ARA_ConvergenceUpdater(int *working_set_op_indices, int *working_set_buffer_ids, int *working_set_ranks,
                               int working_set_size, int *converged_set_op_indices, int *converged_set_buffer_ids,
                               int *converged_set_ranks, int converged_set_size, int *samples_batch, int *small_vectors,
                               int *converged_ops_ptr, int sample_bs, int max_rank, int r)
    {
        this->working_set_op_indices = working_set_op_indices;
        this->working_set_buffer_ids = working_set_buffer_ids;
        this->working_set_ranks = working_set_ranks;
        this->working_set_size = working_set_size;

        this->converged_set_op_indices = converged_set_op_indices;
        this->converged_set_buffer_ids = converged_set_buffer_ids;
        this->converged_set_ranks = converged_set_ranks;
        this->converged_set_size = converged_set_size;

        this->samples_batch = samples_batch;
        this->small_vectors = small_vectors;
        this->converged_ops_ptr = converged_ops_ptr;
        this->sample_bs = sample_bs;
        this->max_rank = max_rank;
        this->r = r;
    }

    __host__ __device__ void operator()(const unsigned int &c) const
    {
        int converged_ops = converged_set_size;
        int ws_size = working_set_size;

        // Compact the subset
        for (int op_index = 0; op_index < ws_size; op_index++)
        {
            int max_samples = max_rank - working_set_ranks[op_index];
            if (small_vectors[op_index] >= r)
                max_samples = 0;

            // The operation converged
            if (max_samples == 0)
            {
                // Push the current operation to the converged list
                converged_set_op_indices[converged_ops] = working_set_op_indices[op_index];
                converged_set_buffer_ids[converged_ops] = working_set_buffer_ids[op_index];
                converged_set_ranks[converged_ops] = working_set_ranks[op_index];
                converged_ops++;

                // Compress the working set by moving this operation to the end of the list
                ws_size--;
                working_set_op_indices[op_index] = working_set_op_indices[ws_size];
                working_set_buffer_ids[op_index] = working_set_buffer_ids[ws_size];
                working_set_ranks[op_index] = working_set_ranks[ws_size];
                small_vectors[op_index] = small_vectors[ws_size];

                // Take a step back
                op_index--;
            }
            else
            {
                // Set the samples for this operation
                int s = (max_samples < sample_bs ? max_samples : sample_bs);
                samples_batch[op_index] = s;
            }
        }

        *converged_ops_ptr = converged_ops;
    }
};

template <int hw>
int tlr_ara_check_converged(int *subset_samples, int *op_global_indices, int *subset_op_ranks,
                            int *subset_small_vectors, int *converged_ops_ptr, int *op_ranks, int sample_bs,
                            int max_rank, int r, int subset_ops, int op_global_offset, int total_ops,
                            h2opusComputeStream_t stream)
{
    TLR_ARA_ConvergenceChecker convergence_checker(subset_samples, op_global_indices, subset_op_ranks,
                                                   subset_small_vectors, converged_ops_ptr, op_ranks, sample_bs,
                                                   max_rank, r, subset_ops, op_global_offset, total_ops);

    // TODO: this should be properly parallelized, but keeping it serial for now since
    // it's typically for a very small number of ops (<100)
    // If the number of ops go up this might need parallelization
    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(1), convergence_checker);

    return thrust_get_value<hw>(converged_ops_ptr);
}

template <int hw>
int tlr_ara_update_converged(int *working_set_op_indices, int *working_set_buffer_ids, int *working_set_ranks,
                             int working_set_size, int *converged_set_op_indices, int *converged_set_buffer_ids,
                             int *converged_set_ranks, int converged_set_size, int *samples_batch, int *small_vectors,
                             int *converged_ops_ptr, int sample_bs, int max_rank, int r, h2opusComputeStream_t stream)
{
    TLR_ARA_ConvergenceUpdater convergence_updater(working_set_op_indices, working_set_buffer_ids, working_set_ranks,
                                                   working_set_size, converged_set_op_indices, converged_set_buffer_ids,
                                                   converged_set_ranks, converged_set_size, samples_batch,
                                                   small_vectors, converged_ops_ptr, sample_bs, max_rank, r);

    // TODO: this should be properly parallelized, but keeping it serial for now since
    // it's typically for a very small number of ops (<100)
    // If the number of ops go up this might need parallelization
    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(1), convergence_updater);

    return thrust_get_value<hw>(converged_ops_ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Setting up operation subset pointers
/////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct TLR_ARA_Basis_Ptr_Setter
{
  private:
    T **sub_Q_ptrs, **sub_Y_ptrs, **Q_ptrs;
    int *ld_batch, *op_global_indices, *subset_op_ranks;
    int op_global_offset;

  public:
    TLR_ARA_Basis_Ptr_Setter(T **sub_Q_ptrs, T **sub_Y_ptrs, T **Q_ptrs, int *ld_batch, int *op_global_indices,
                             int *subset_op_ranks, int op_global_offset)
    {
        this->sub_Q_ptrs = sub_Q_ptrs;
        this->sub_Y_ptrs = sub_Y_ptrs;
        this->Q_ptrs = Q_ptrs;
        this->ld_batch = ld_batch;
        this->op_global_indices = op_global_indices;
        this->subset_op_ranks = subset_op_ranks;
        this->op_global_offset = op_global_offset;
    }

    __host__ __device__ void operator()(const unsigned int &op_index) const
    {
        int op_global_id = op_global_indices[op_index] - op_global_offset;
        sub_Q_ptrs[op_index] = Q_ptrs[op_global_id];
        sub_Y_ptrs[op_index] = sub_Q_ptrs[op_index] + subset_op_ranks[op_index] * ld_batch[op_index];
    }
};

template <class T, int hw>
void tlr_ara_set_sample_basis_ptrs(T **sub_Q_ptrs, T **sub_Y_ptrs, T **Q_ptrs, int *ld_batch, int *op_global_indices,
                                   int *subset_op_ranks, int op_global_offset, int num_ops,
                                   h2opusComputeStream_t stream)
{
    TLR_ARA_Basis_Ptr_Setter<T> basis_ptr_set(sub_Q_ptrs, sub_Y_ptrs, Q_ptrs, ld_batch, op_global_indices,
                                              subset_op_ranks, op_global_offset);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_ops), basis_ptr_set);
}

template <class T> struct TLR_ARA_Working_Set_Basis_Ptr
{
  private:
    T **sub_Q_ptrs, **sub_Y_ptrs, **Q_ptrs;
    int *ld_batch;
    int *working_set_buffer_ids, *working_set_ranks;

  public:
    TLR_ARA_Working_Set_Basis_Ptr(T **sub_Q_ptrs, T **sub_Y_ptrs, T **Q_ptrs, int *ld_batch,
                                  int *working_set_buffer_ids, int *working_set_ranks)
    {
        this->sub_Q_ptrs = sub_Q_ptrs;
        this->sub_Y_ptrs = sub_Y_ptrs;
        this->Q_ptrs = Q_ptrs;
        this->ld_batch = ld_batch;
        this->working_set_buffer_ids = working_set_buffer_ids;
        this->working_set_ranks = working_set_ranks;
    }

    __host__ __device__ void operator()(const unsigned int &op_index) const
    {
        int working_set_buffer_id = working_set_buffer_ids[op_index];
        sub_Q_ptrs[op_index] = Q_ptrs[working_set_buffer_id];
        sub_Y_ptrs[op_index] = sub_Q_ptrs[op_index] + working_set_ranks[op_index] * ld_batch[op_index];
    }
};

template <class T, int hw>
void tlr_ara_working_set_basis_ptrs(T **sub_Q_ptrs, T **sub_Y_ptrs, T **Q_ptrs, int *ld_batch,
                                    int *working_set_buffer_ids, int *working_set_ranks, int working_set_size,
                                    h2opusComputeStream_t stream)
{
    TLR_ARA_Working_Set_Basis_Ptr<T> basis_ptr_func(sub_Q_ptrs, sub_Y_ptrs, Q_ptrs, ld_batch, working_set_buffer_ids,
                                                    working_set_ranks);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(working_set_size), basis_ptr_func);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Basis orthogonalization
/////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct H2OpusTLR_ARA_OrthogWorkspace
{
    // Orthogonalization workspace
    double **G_ptrs, *R_diag;
    int *ldg_batch, *cholqr_block_ranks;
    T **tau_ptrs, *tau_base_data, *hh_R_diag;
};

template <class T, int hw>
void tlr_ara_allocate_orthog_workspace(int n_block, size_t sample_bs, H2OpusTLR_ARA_OrthogWorkspace<T> *workspace,
                                       h2opusWorkspace_t h2opus_ws, h2opusComputeStream_t stream)
{
#ifdef H2OPUS_TLR_USE_CHOLESKY_QR
    // Memory for the Gram matrix used in the cholesky QR
    // G is always double precision to help stabilize the cholesky QR
    size_t G_entries = sample_bs * sample_bs;
    double *base_buffer_G;

    h2opus_ws->allocateEntries<double>(G_entries * n_block, &base_buffer_G, hw);
    h2opus_ws->allocateEntries<double>(sample_bs * n_block, H2OPUS_TLR_WS_PTR(R_diag), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(ldg_batch), hw);
    h2opus_ws->allocateEntries<int>(n_block, H2OPUS_TLR_WS_PTR(cholqr_block_ranks), hw);

    h2opus_ws->allocatePointerEntries<double>(n_block, H2OPUS_TLR_WS_PTR(G_ptrs), hw);
#else
    h2opus_ws->allocateEntries<T>(sample_bs * n_block, H2OPUS_TLR_WS_PTR(tau_base_data), hw);
    h2opus_ws->allocateEntries<T>(sample_bs * n_block, H2OPUS_TLR_WS_PTR(hh_R_diag), hw);

    h2opus_ws->allocatePointerEntries<T>(n_block, H2OPUS_TLR_WS_PTR(tau_ptrs), hw);
#endif

    if (workspace)
    {
#ifdef H2OPUS_TLR_USE_CHOLESKY_QR
        generateArrayOfPointers(base_buffer_G, workspace->G_ptrs, G_entries, n_block, stream, hw);
        fillArray(workspace->ldg_batch, n_block, sample_bs, stream, hw);
        fillArray(base_buffer_G, G_entries * n_block, 0, stream, hw);
        fillArray(workspace->R_diag, sample_bs * n_block, 0, stream, hw);
#else
        generateArrayOfPointers(workspace->tau_base_data, workspace->tau_ptrs, sample_bs, n_block, stream, hw);
        fillArray(workspace->tau_base_data, sample_bs * n_block, 0, stream, hw);
        fillArray(workspace->hh_R_diag, sample_bs * n_block, 0, stream, hw);
#endif
    }
}

// Orthogonalize the output of the sampling process Y w.r.t the current orthogonal basis Q
template <class T, int hw>
void tlr_ara_gen_basis(T **Q_ptrs, int *ldq_batch, int *rows_q_batch, int *cols_q_batch, int max_rows_q, int max_cols_q,
                       T **Y_ptrs, int *ldy_batch, int *cols_y_batch, int max_cols_y, T **Z_ptrs, int *ldz_batch,
                       int batchCount, H2OpusTLR_ARA_OrthogWorkspace<T> &workspace, h2opusComputeStream_t stream)
{
#ifdef H2OPUS_TLR_USE_CHOLESKY_QR
    double **G_ptrs = workspace.G_ptrs;
    double *R_diag = workspace.R_diag;
    int *ldg_batch = workspace.ldg_batch;
    int *cholqr_block_ranks = workspace.cholqr_block_ranks;
#else
    T *R_diag = workspace.hh_R_diag;
    T **tau_ptrs = workspace.tau_ptrs;
#endif

    // Clear R
    fillArray(R_diag, batchCount * max_cols_y, 1, stream, hw);

    // BCGS with one reorthogonalization step
    for (int i = 0; i < 2; i++)
    {
        // Project samples
        // Y = Y - Q * (Q' * Y) = Y - Q * Z
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, cols_q_batch, cols_y_batch,
                                                       rows_q_batch, max_cols_q, max_cols_y, max_rows_q, 1,
                                                       (const T **)Q_ptrs, ldq_batch, (const T **)Y_ptrs, ldy_batch, 0,
                                                       Z_ptrs, ldz_batch, batchCount));

        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, rows_q_batch,
                                                       cols_y_batch, cols_q_batch, max_rows_q, max_cols_y, max_cols_q,
                                                       -1, (const T **)Q_ptrs, ldq_batch, (const T **)Z_ptrs, ldz_batch,
                                                       1, Y_ptrs, ldy_batch, batchCount));
#ifdef H2OPUS_TLR_USE_CHOLESKY_QR
        // Panel orthogonalization using cholesky qr
        // Compute G = Y'*Y in mixed precision
        check_kblas_error((H2OpusBatched<T, hw>::mp_syrk)(stream, rows_q_batch, cols_y_batch, max_rows_q, max_cols_y,
                                                          (const T **)Y_ptrs, ldy_batch, G_ptrs, ldg_batch,
                                                          batchCount));

        // Cholesky on G into Z
        check_kblas_error((H2OpusBatched<T, hw>::mp_fused_potrf)(stream, cols_y_batch, max_cols_y, G_ptrs, ldg_batch,
                                                                 Z_ptrs, ldz_batch, R_diag, cholqr_block_ranks,
                                                                 batchCount));

        // Copy the ranks over to the cols_y_batch in case the rank was less than the number of columns
        copyArray(cholqr_block_ranks, cols_y_batch, batchCount, stream, hw);

        // TRSM to set Y = Y * Z^-1, the orthogonal factor
        check_kblas_error((H2OpusBatched<T, hw>::trsm_ara)(stream, rows_q_batch, cholqr_block_ranks, max_rows_q,
                                                           max_cols_y, Y_ptrs, ldy_batch, Z_ptrs, ldz_batch,
                                                           batchCount));
#else
        // Regular householder QR
        check_kblas_error((H2OpusBatched<T, hw>::geqrf(stream, rows_q_batch, cols_y_batch, max_rows_q, max_cols_y,
                                                       Y_ptrs, ldy_batch, tau_ptrs, batchCount)));

        // Update R
        check_kblas_error((H2OpusBatched<T, hw>::diagMult(stream, rows_q_batch, cols_y_batch, max_rows_q, max_cols_y,
                                                          Y_ptrs, ldy_batch, R_diag, max_cols_y, batchCount)));

        // Expand reflectors into orthogonal block column
        check_kblas_error((H2OpusBatched<T, hw>::orgqr(stream, rows_q_batch, cols_y_batch, max_rows_q, max_cols_y,
                                                       Y_ptrs, ldy_batch, tau_ptrs, batchCount)));
#endif
    }
}

#endif
