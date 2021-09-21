#ifndef __H2OPUS_TLR_GEMV_MARSHAL_H__
#define __H2OPUS_TLR_GEMV_MARSHAL_H__

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/thrust_runtime.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functors
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct TLR_Gemv_Dense_Functor
{
  private:
    int block_size, ldx, ldy, num_vectors, n_block;
    T *X, *Y, **input_ptrs, **output_ptrs;
    int *rows_batch, *cols_batch, *num_vec_batch;
    int *bs_batch, *ldx_batch, *ldy_batch;

  public:
    TLR_Gemv_Dense_Functor(int block_size, T *X, int ldx, T *Y, int ldy, int num_vectors, T **input_ptrs,
                           T **output_ptrs, int *rows_batch, int *cols_batch, int *num_vec_batch, int *bs_batch,
                           int *ldx_batch, int *ldy_batch, int n_block)
    {
        this->block_size = block_size;
        this->X = X;
        this->ldx = ldx;
        this->Y = Y;
        this->ldy = ldy;
        this->num_vectors = num_vectors;
        this->input_ptrs = input_ptrs;
        this->output_ptrs = output_ptrs;
        this->rows_batch = rows_batch;
        this->cols_batch = cols_batch;
        this->num_vec_batch = num_vec_batch;
        this->bs_batch = bs_batch;
        this->ldx_batch = ldx_batch;
        this->ldy_batch = ldy_batch;
        this->n_block = n_block;
    }

    __host__ __device__ void operator()(const unsigned int &block_index) const
    {
        int block_offset = block_index * block_size;

        input_ptrs[block_index] = X + block_offset;
        output_ptrs[block_index] = Y + block_offset;
        rows_batch[block_index] = block_size;
        cols_batch[block_index] = block_size;
        num_vec_batch[block_index] = num_vectors;
        ldx_batch[block_index] = ldx;
        ldy_batch[block_index] = ldy;
        bs_batch[block_index] = block_size;
    }
};

struct TLR_Gemv_LRDim_Functor
{
  private:
    int n, block_size, max_rank, ldx, num_vectors;
    int *bs_batch, *lr_max_rank_batch, *ldy_batch, *ldx_batch, *num_vec_batch;
    int *rows_batch, n_block, num_parallel_columns;

  public:
    TLR_Gemv_LRDim_Functor(int n, int block_size, int max_rank, int ldx, int num_vectors, int *bs_batch,
                           int *lr_max_rank_batch, int *ldy_batch, int *ldx_batch, int *num_vec_batch, int *rows_batch,
                           int n_block, int num_parallel_columns)
    {
        this->n = n;
        this->block_size = block_size;
        this->max_rank = max_rank;
        this->ldx = ldx;
        this->num_vectors = num_vectors;
        this->bs_batch = bs_batch;
        this->lr_max_rank_batch = lr_max_rank_batch;
        this->ldy_batch = ldy_batch;
        this->ldx_batch = ldx_batch;
        this->num_vec_batch = num_vec_batch;
        this->rows_batch = rows_batch;
        this->n_block = n_block;
        this->num_parallel_columns = num_parallel_columns;
    }

    __host__ __device__ void operator()(const unsigned int &block_index) const
    {
        for (int i = 0; i < num_parallel_columns; i++)
        {
            int dim_index = block_index + i * n_block;
            bs_batch[dim_index] = block_size;
            lr_max_rank_batch[dim_index] = max_rank;
            num_vec_batch[dim_index] = num_vectors;
            rows_batch[dim_index] = block_size;
            ldy_batch[dim_index] = n;
            ldx_batch[dim_index] = ldx;
        }
    }
};

template <class T, bool transpose> struct TLR_Gemv_LRMult_Functor
{
  private:
    int n, block_size, max_rank, num_vectors;
    T *X, **input_ptrs, *VY_base_data, **VY_ptrs, *UZ_base_data, **UZ_ptrs;
    T **block_U_ptrs, **U_ptrs, **block_V_ptrs, **V_ptrs;
    int *block_ranks, *rank_batch, *cols_batch, j_start, n_block, num_block_cols;

  public:
    TLR_Gemv_LRMult_Functor(int n, int block_size, int max_rank, int num_vectors, T *X, T **input_ptrs, T *VY_base_data,
                            T **VY_ptrs, T *UZ_base_data, T **UZ_ptrs, T **block_U_ptrs, T **U_ptrs, T **block_V_ptrs,
                            T **V_ptrs, int *block_ranks, int *rank_batch, int *cols_batch, int j_start, int n_block,
                            int num_block_cols)
    {
        this->n = n;
        this->block_size = block_size;
        this->max_rank = max_rank;
        this->num_vectors = num_vectors;
        this->X = X;
        this->input_ptrs = input_ptrs;
        this->VY_base_data = VY_base_data;
        this->VY_ptrs = VY_ptrs;
        this->UZ_base_data = UZ_base_data;
        this->UZ_ptrs = UZ_ptrs;
        this->block_U_ptrs = block_U_ptrs;
        this->U_ptrs = U_ptrs;
        this->block_V_ptrs = block_V_ptrs;
        this->V_ptrs = V_ptrs;
        this->block_ranks = block_ranks;
        this->rank_batch = rank_batch;
        this->cols_batch = cols_batch;
        this->j_start = j_start;
        this->n_block = n_block;
        this->num_block_cols = num_block_cols;
    }

    __host__ __device__ void operator()(const unsigned int &i) const
    {
        for (int j = j_start; j < j_start + num_block_cols; j++)
        {
            int op_index = i + (j - j_start) * n_block;
            int linear_index = (transpose ? j + i * n_block : i + j * n_block);

            rank_batch[op_index] = block_ranks[linear_index];
            cols_batch[op_index] = block_size;
            U_ptrs[op_index] = (transpose ? block_V_ptrs[linear_index] : block_U_ptrs[linear_index]);
            V_ptrs[op_index] = (transpose ? block_U_ptrs[linear_index] : block_V_ptrs[linear_index]);
            VY_ptrs[op_index] = VY_base_data + op_index * max_rank * num_vectors;
            UZ_ptrs[op_index] = UZ_base_data + i * block_size + (j - j_start) * n;
            input_ptrs[op_index] = X + j * block_size;
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Marshal routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw>
inline void tlr_gemv_marshal_dense(int block_size, T *X, int ldx, T *Y, int ldy, int num_vectors, T **input_ptrs,
                                   T **output_ptrs, int *rows_batch, int *cols_batch, int *num_vec_batch, int *bs_batch,
                                   int *ldx_batch, int *ldy_batch, int n_block, h2opusComputeStream_t stream)
{
    TLR_Gemv_Dense_Functor<T> gemv_dense_func(block_size, X, ldx, Y, ldy, num_vectors, input_ptrs, output_ptrs,
                                              rows_batch, cols_batch, num_vec_batch, bs_batch, ldx_batch, ldy_batch,
                                              n_block);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(n_block), gemv_dense_func);
}

template <int hw>
inline void tlr_gemv_lr_dim_set(int n, int block_size, int max_rank, int ldx, int num_vectors, int *bs_batch,
                                int *lr_max_rank_batch, int *ldy_batch, int *ldx_batch, int *num_vec_batch,
                                int *rows_batch, int n_block, int num_parallel_columns, h2opusComputeStream_t stream)
{
    TLR_Gemv_LRDim_Functor lrdim_func(n, block_size, max_rank, ldx, num_vectors, bs_batch, lr_max_rank_batch, ldy_batch,
                                      ldx_batch, num_vec_batch, rows_batch, n_block, num_parallel_columns);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(n_block), lrdim_func);
}

template <class T, bool transpose, int hw>
inline void tlr_gemv_lr_mult_marshal(int n, int block_size, int max_rank, int num_vectors, T *X, T **input_ptrs,
                                     T *VY_base_data, T **VY_ptrs, T *UZ_base_data, T **UZ_ptrs, T **block_U_ptrs,
                                     T **U_ptrs, T **block_V_ptrs, T **V_ptrs, int *block_ranks, int *rank_batch,
                                     int *cols_batch, int j_start, int n_block, int num_block_cols,
                                     h2opusComputeStream_t stream)
{
    TLR_Gemv_LRMult_Functor<T, transpose> lrmult_func(
        n, block_size, max_rank, num_vectors, X, input_ptrs, VY_base_data, VY_ptrs, UZ_base_data, UZ_ptrs, block_U_ptrs,
        U_ptrs, block_V_ptrs, V_ptrs, block_ranks, rank_batch, cols_batch, j_start, n_block, num_block_cols);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(n_block), lrmult_func);
}

#endif
