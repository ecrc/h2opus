#ifndef __H2OPUS_TLR_POTRF_MARSHAL_H__
#define __H2OPUS_TLR_POTRF_MARSHAL_H__

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/thrust_runtime.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functors
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct TLR_Potrf_DenseUpdate_Functor
{
  private:
    T **block_U_ptrs, **block_V_ptrs, **ptr_U, **ptr_V;
    int col, n_block, *block_ranks, *dense_rank_batch;

    T *D, **D_ptrs;
    int stride_D;

  public:
    TLR_Potrf_DenseUpdate_Functor(T **block_U_ptrs, T **block_V_ptrs, int col, int n_block, int *block_ranks, T **ptr_U,
                                  T **ptr_V, int *dense_rank_batch, T *D, int stride_D, T **D_ptrs)
    {
        this->block_U_ptrs = block_U_ptrs;
        this->block_V_ptrs = block_V_ptrs;
        this->col = col;
        this->n_block = n_block;
        this->block_ranks = block_ranks;

        this->ptr_U = ptr_U;
        this->ptr_V = ptr_V;
        this->dense_rank_batch = dense_rank_batch;

        this->D = D;
        this->stride_D = stride_D;
        this->D_ptrs = D_ptrs;
    }

    __host__ __device__ void operator()(const unsigned int &update_index) const
    {
        int linear_index = col + update_index * n_block;
        ptr_U[update_index] = block_U_ptrs[linear_index];
        ptr_V[update_index] = block_V_ptrs[linear_index];
        dense_rank_batch[update_index] = block_ranks[linear_index];

        if (D)
            D_ptrs[update_index] = D + stride_D * update_index;
    }
};

template <class T, bool transpose> struct TLR_Potrf_LowRankUpdateSample_Functor
{
  private:
    T **block_U_ptrs, **block_V_ptrs, **Uij_ptrs, **Vij_ptrs, **Ukj_ptrs, **Vkj_ptrs;
    T **input_ptrs, **input_i_ptrs;
    int *block_ranks, *samples_batch, *samples_i_batch, *rank_ij_batch, *rank_kj_batch;
    int k, *row_indices, rows, col_start, cols, n_block;
    T *D, **D_ptrs;
    int stride_D;

  public:
    TLR_Potrf_LowRankUpdateSample_Functor(T **block_U_ptrs, T **block_V_ptrs, int *block_ranks, int k, int n_block,
                                          int *row_indices, int rows, int col_start, int cols, T **Uij_ptrs,
                                          T **Vij_ptrs, T **Ukj_ptrs, T **Vkj_ptrs, T **input_ptrs, T **input_i_ptrs,
                                          int *samples_batch, int *samples_i_batch, int *rank_ij_batch,
                                          int *rank_kj_batch, T *D, int stride_D, T **D_ptrs)
    {
        this->block_U_ptrs = block_U_ptrs;
        this->block_V_ptrs = block_V_ptrs;
        this->block_ranks = block_ranks;
        this->k = k;
        this->n_block = n_block;
        this->row_indices = row_indices;
        this->rows = rows;
        this->col_start = col_start;
        this->cols = cols;
        this->Uij_ptrs = Uij_ptrs;
        this->Vij_ptrs = Vij_ptrs;
        this->Ukj_ptrs = Ukj_ptrs;
        this->Vkj_ptrs = Vkj_ptrs;
        this->input_ptrs = input_ptrs;
        this->input_i_ptrs = input_i_ptrs;
        this->samples_batch = samples_batch;
        this->samples_i_batch = samples_i_batch;
        this->rank_ij_batch = rank_ij_batch;
        this->rank_kj_batch = rank_kj_batch;

        this->D = D;
        this->stride_D = stride_D;
        this->D_ptrs = D_ptrs;
    }

    __host__ __device__ void operator()(const unsigned int &block_index) const
    {
        // Determine the global row and column index of the block in the sub-matrix
        int input_row_index = block_index % rows;
        int i = row_indices[input_row_index];
        int j = block_index / rows + col_start;

        // Grab pointer and rank data
        int linear_index_kj = k + j * n_block;
        int linear_index_ij = i + j * n_block;

        // Swap the indices around if we want the transpose operation
        if (transpose)
        {
            int t = linear_index_kj;
            linear_index_kj = linear_index_ij;
            linear_index_ij = t;
        }

        Ukj_ptrs[block_index] = block_U_ptrs[linear_index_kj];
        Vkj_ptrs[block_index] = block_V_ptrs[linear_index_kj];
        rank_kj_batch[block_index] = block_ranks[linear_index_kj];

        Uij_ptrs[block_index] = block_U_ptrs[linear_index_ij];
        Vij_ptrs[block_index] = block_V_ptrs[linear_index_ij];
        rank_ij_batch[block_index] = block_ranks[linear_index_ij];

        // Input pointer is the row index
        input_i_ptrs[block_index] = input_ptrs[input_row_index];

        // Don't bother taking samples if either rank is zero
        int samples = samples_batch[input_row_index];
        if (rank_kj_batch[block_index] == 0 || rank_ij_batch[block_index] == 0)
            samples = 0;
        samples_i_batch[block_index] = samples;

        if (D)
            D_ptrs[block_index] = D + j * stride_D;
    }
};

template <class T> struct TLR_Potrf_Pointer_Advancer
{
  private:
    T **input_ptrs, **output_ptrs;
    int *samples, block_size;

  public:
    TLR_Potrf_Pointer_Advancer(T **input_ptrs, T **output_ptrs, int *samples, int block_size)
    {
        this->input_ptrs = input_ptrs;
        this->output_ptrs = output_ptrs;
        this->samples = samples;
        this->block_size = block_size;
    }

    __host__ __device__ void operator()(const unsigned int &block_index) const
    {
        input_ptrs[block_index] += block_size * samples[block_index];
        output_ptrs[block_index] += block_size * samples[block_index];
    }
};

template <class T> struct TLR_Projection_Sample_Setter
{
  private:
    int *samples_batch, sample_bs, *current_samples, *global_ranks, *op_indices, op_global_offset;
    T **Q_ptrs, **sub_Q_ptrs, **B_ptrs, **sub_B_ptrs;
    int *ldq_batch, *ldb_batch;

  public:
    TLR_Projection_Sample_Setter(int *samples_batch, int sample_bs, int *current_samples, int *global_ranks,
                                 int *op_indices, int op_global_offset, T **Q_ptrs, T **sub_Q_ptrs, int *ldq_batch,
                                 T **B_ptrs, T **sub_B_ptrs, int *ldb_batch)
    {
        this->samples_batch = samples_batch;
        this->sample_bs = sample_bs;
        this->current_samples = current_samples;
        this->global_ranks = global_ranks;
        this->op_indices = op_indices;
        this->op_global_offset = op_global_offset;
        this->Q_ptrs = Q_ptrs;
        this->sub_Q_ptrs = sub_Q_ptrs;
        this->ldq_batch = ldq_batch;
        this->B_ptrs = B_ptrs;
        this->sub_B_ptrs = sub_B_ptrs;
        this->ldb_batch = ldb_batch;
    }

    __host__ __device__ void operator()(const unsigned int &op_index) const
    {
        int samples_taken = current_samples[op_index];
        int global_op_index = op_indices[op_index] - op_global_offset;

        int samples_to_take = 0;
        int needed_samples = global_ranks[global_op_index] - samples_taken;

        if (needed_samples > 0)
        {
            samples_to_take = (needed_samples < sample_bs ? needed_samples : sample_bs);
            samples_taken += samples_to_take;
        }

        samples_batch[op_index] = samples_to_take;
        current_samples[op_index] = samples_taken;
        sub_Q_ptrs[op_index] = Q_ptrs[global_op_index] + samples_taken * ldq_batch[global_op_index];
        sub_B_ptrs[op_index] = B_ptrs[global_op_index] + samples_taken * ldb_batch[global_op_index];
    }
};

template <class T, bool transpose> struct TLR_Potrf_ColumnSample_Functor
{
  private:
    T **block_U_ptrs, **block_V_ptrs, **Uik_ptrs, **Vik_ptrs;
    int *block_ranks, k, n_block, *row_indices, *rank_ik_batch;

  public:
    TLR_Potrf_ColumnSample_Functor(T **block_U_ptrs, T **block_V_ptrs, int *block_ranks, int k, int n_block,
                                   int *row_indices, T **Uik_ptrs, T **Vik_ptrs, int *rank_ik_batch)
    {
        this->block_U_ptrs = block_U_ptrs;
        this->block_V_ptrs = block_V_ptrs;
        this->block_ranks = block_ranks;
        this->k = k;
        this->n_block = n_block;
        this->row_indices = row_indices;
        this->Uik_ptrs = Uik_ptrs;
        this->Vik_ptrs = Vik_ptrs;
        this->rank_ik_batch = rank_ik_batch;
    }

    __host__ __device__ void operator()(const unsigned int &row_index) const
    {
        int i = row_indices[row_index];
        int linear_index = i + k * n_block;

        Uik_ptrs[row_index] = (transpose ? block_V_ptrs[linear_index] : block_U_ptrs[linear_index]);
        Vik_ptrs[row_index] = (transpose ? block_U_ptrs[linear_index] : block_V_ptrs[linear_index]);
        rank_ik_batch[row_index] = block_ranks[linear_index];
    }
};

template <class T> struct TLR_Potrf_Basis_Ptr_Setter
{
  private:
    T **sub_Q_ptrs, **sub_Y_ptrs, **Q_ptrs;
    int *ld_batch, *op_global_indices, *subset_op_ranks;
    int op_global_offset;

  public:
    TLR_Potrf_Basis_Ptr_Setter(T **sub_Q_ptrs, T **sub_Y_ptrs, T **Q_ptrs, int *ld_batch, int *op_global_indices,
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Marshal routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw>
inline void tlr_determine_projection_samples(int *samples_batch, int sample_bs, int *current_samples, int *global_ranks,
                                             int *op_indices, int op_global_offset, T **Q_ptrs, T **sub_Q_ptrs,
                                             int *ldq_batch, T **B_ptrs, T **sub_B_ptrs, int *ldb_batch, int num_blocks,
                                             h2opusComputeStream_t stream)
{
    TLR_Projection_Sample_Setter<T> sample_setter(samples_batch, sample_bs, current_samples, global_ranks, op_indices,
                                                  op_global_offset, Q_ptrs, sub_Q_ptrs, ldq_batch, B_ptrs, sub_B_ptrs,
                                                  ldb_batch);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_blocks), sample_setter);
}

template <class T, int hw>
inline void tlr_potrf_advance_pointers(T **input_ptrs, T **output_ptrs, int *samples, int block_size, int num_blocks,
                                       h2opusComputeStream_t stream)
{
    TLR_Potrf_Pointer_Advancer<T> ptr_advancer(input_ptrs, output_ptrs, samples, block_size);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_blocks), ptr_advancer);
}

template <class T, int hw>
inline void tlr_potrf_marshal_dense_updates(T **block_U_ptrs, T **block_V_ptrs, int col, int n_block, int *block_ranks,
                                            T **ptr_U, T **ptr_V, int *dense_rank_batch, T *D, int stride_D, T **D_ptrs,
                                            int num_updates, h2opusComputeStream_t stream)
{
    TLR_Potrf_DenseUpdate_Functor<T> denseupdate_functor(block_U_ptrs, block_V_ptrs, col, n_block, block_ranks, ptr_U,
                                                         ptr_V, dense_rank_batch, D, stride_D, D_ptrs);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_updates), denseupdate_functor);
}

template <class T, int hw, bool transpose>
inline void tlr_potrf_marshal_lru_sample_range(T **block_U_ptrs, T **block_V_ptrs, int *block_ranks, int k, int n_block,
                                               int *row_indices, int rows, int col_start, int cols, T **Uij_ptrs,
                                               T **Vij_ptrs, T **Ukj_ptrs, T **Vkj_ptrs, T **input_ptrs,
                                               T **input_i_ptrs, int *samples_batch, int *samples_i_batch,
                                               int *rank_ij_batch, int *rank_kj_batch, T *D, int stride_D, T **D_ptrs,
                                               int sample_block_count, h2opusComputeStream_t stream)
{
    TLR_Potrf_LowRankUpdateSample_Functor<T, transpose> lru_sample_functor(
        block_U_ptrs, block_V_ptrs, block_ranks, k, n_block, row_indices, rows, col_start, cols, Uij_ptrs, Vij_ptrs,
        Ukj_ptrs, Vkj_ptrs, input_ptrs, input_i_ptrs, samples_batch, samples_i_batch, rank_ij_batch, rank_kj_batch, D,
        stride_D, D_ptrs);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(sample_block_count), lru_sample_functor);
}

template <class T, int hw, bool transpose>
inline void tlr_potrf_marshal_col_sample_range(T **block_U_ptrs, T **block_V_ptrs, int *block_ranks, int k, int n_block,
                                               int *row_indices, int rows, T **Uik_ptrs, T **Vik_ptrs,
                                               int *rank_ik_batch, h2opusComputeStream_t stream)
{
    TLR_Potrf_ColumnSample_Functor<T, transpose> col_sample_functor(block_U_ptrs, block_V_ptrs, block_ranks, k, n_block,
                                                                    row_indices, Uik_ptrs, Vik_ptrs, rank_ik_batch);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(rows), col_sample_functor);
}

template <class T, int hw>
void tlr_potrf_set_sample_basis_ptrs(T **sub_Q_ptrs, T **sub_Y_ptrs, T **Q_ptrs, int *ld_batch, int *op_global_indices,
                                     int *subset_op_ranks, int op_global_offset, int num_ops,
                                     h2opusComputeStream_t stream)
{
    TLR_Potrf_Basis_Ptr_Setter<T> basis_ptr_set(sub_Q_ptrs, sub_Y_ptrs, Q_ptrs, ld_batch, op_global_indices,
                                                subset_op_ranks, op_global_offset);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_ops), basis_ptr_set);
}

#endif
