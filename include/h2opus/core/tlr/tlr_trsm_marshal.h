#ifndef __H2OPUS_TLR_TRSM_MARSHAL_H__
#define __H2OPUS_TLR_TRSM_MARSHAL_H__

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/thrust_runtime.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility routines
//////////////////////////////////////////////////////////////////////////////////////////////////////
template <bool transpose> void tlr_trsm_get_loop_param(int n, int &start, int &end, int &inc);

template <> void tlr_trsm_get_loop_param<false>(int n, int &start, int &end, int &inc)
{
    start = 0;
    end = n;
    inc = 1;
}

template <> void tlr_trsm_get_loop_param<true>(int n, int &start, int &end, int &inc)
{
    start = n - 1;
    end = -1;
    inc = -1;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functors
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, bool transpose> struct TLR_TrsmDense_UpdateB_Functor
{
  private:
    int start, inc, i, block_size, n_block;
    T **block_V_ptrs, **block_U_ptrs, *B, **Bi_ptrs, **Bj_ptrs, **U_ptrs, **V_ptrs;
    int *block_ranks, *bi_rows_batch, *bj_rows_batch, *ranks_batch;

  public:
    TLR_TrsmDense_UpdateB_Functor(int start, int inc, int i, int block_size, int n_block, T **block_V_ptrs,
                                  T **block_U_ptrs, int *block_ranks, T *B, T **Bi_ptrs, T **Bj_ptrs,
                                  int *bi_rows_batch, int *bj_rows_batch, T **U_ptrs, T **V_ptrs, int *ranks_batch)
    {
        this->start = start;
        this->inc = inc;
        this->i = i;
        this->block_size = block_size;
        this->n_block = n_block;
        this->block_V_ptrs = block_V_ptrs;
        this->block_U_ptrs = block_U_ptrs;
        this->block_ranks = block_ranks;
        this->B = B;
        this->Bi_ptrs = Bi_ptrs;
        this->Bj_ptrs = Bj_ptrs;
        this->bi_rows_batch = bi_rows_batch;
        this->bj_rows_batch = bj_rows_batch;
        this->U_ptrs = U_ptrs;
        this->V_ptrs = V_ptrs;
        this->ranks_batch = ranks_batch;
    }

    __host__ __device__ void operator()(const unsigned int &block_index) const
    {
        int j = start + block_index * inc;
        Bj_ptrs[block_index] = B + j * block_size;
        Bi_ptrs[block_index] = B + i * block_size;

        bj_rows_batch[block_index] = block_size;
        bi_rows_batch[block_index] = block_size;

        int linear_index = (transpose ? i + j * n_block : j + i * n_block);
        U_ptrs[block_index] = (transpose ? block_V_ptrs[linear_index] : block_U_ptrs[linear_index]);
        V_ptrs[block_index] = (transpose ? block_U_ptrs[linear_index] : block_V_ptrs[linear_index]);
        ranks_batch[block_index] = block_ranks[linear_index];
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Marshal routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw, bool transpose>
inline void tlr_trsm_dense_marshal_updateB(int start, int inc, int i, int block_size, int n_block, T **block_V_ptrs,
                                           T **block_U_ptrs, int *block_ranks, T *B, T **Bi_ptrs, T **Bj_ptrs,
                                           int *bi_rows_batch, int *bj_rows_batch, T **U_ptrs, T **V_ptrs,
                                           int *ranks_batch, int num_blocks, h2opusComputeStream_t stream)
{
    TLR_TrsmDense_UpdateB_Functor<T, transpose> trsm_dense_updateB(
        start, inc, i, block_size, n_block, block_V_ptrs, block_U_ptrs, block_ranks, B, Bi_ptrs, Bj_ptrs, bi_rows_batch,
        bj_rows_batch, U_ptrs, V_ptrs, ranks_batch);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_blocks), trsm_dense_updateB);
}

#endif
