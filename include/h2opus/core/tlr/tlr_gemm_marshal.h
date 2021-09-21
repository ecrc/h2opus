#ifndef __H2OPUS_TLR_GEMM_MARSHAL_H__
#define __H2OPUS_TLR_GEMM_MARSHAL_H__

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/thrust_runtime.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functors
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct TLR_Gemm_LR_Diagonal_Functor
{
  private:
    int k, n_block;
    T **A_block_U_ptrs, **A_block_V_ptrs, **B_block_U_ptrs, **B_block_V_ptrs;
    int *A_block_ranks, *B_block_ranks;
    T **Ua_ptrs, **Va_ptrs, **Ub_ptrs, **Vb_ptrs;
    int *a_ranks, *b_ranks;

  public:
    TLR_Gemm_LR_Diagonal_Functor(int k, int n_block, T **A_block_U_ptrs, T **A_block_V_ptrs, T **B_block_U_ptrs,
                                 T **B_block_V_ptrs, int *A_block_ranks, int *B_block_ranks, T **Ua_ptrs, T **Va_ptrs,
                                 T **Ub_ptrs, T **Vb_ptrs, int *a_ranks, int *b_ranks)
    {
        this->k = k;
        this->n_block = n_block;
        this->A_block_U_ptrs = A_block_U_ptrs;
        this->A_block_V_ptrs = A_block_V_ptrs;
        this->B_block_U_ptrs = B_block_U_ptrs;
        this->B_block_V_ptrs = B_block_V_ptrs;
        this->A_block_ranks = A_block_ranks;
        this->B_block_ranks = B_block_ranks;
        this->Ua_ptrs = Ua_ptrs;
        this->Va_ptrs = Va_ptrs;
        this->Ub_ptrs = Ub_ptrs;
        this->Vb_ptrs = Vb_ptrs;
        this->a_ranks = a_ranks;
        this->b_ranks = b_ranks;
    }

    __host__ __device__ void operator()(const unsigned int &block_index) const
    {
        int A_index = block_index + k * n_block;
        int B_index = k + block_index * n_block;

        Ua_ptrs[block_index] = A_block_U_ptrs[A_index];
        Va_ptrs[block_index] = A_block_V_ptrs[A_index];
        a_ranks[block_index] = A_block_ranks[A_index];

        Ub_ptrs[block_index] = B_block_U_ptrs[B_index];
        Vb_ptrs[block_index] = B_block_V_ptrs[B_index];
        b_ranks[block_index] = B_block_ranks[B_index];
    }
};

template <class T, bool transpose_sample> struct TLR_Gemm_Sample_Tile_Product_Functor
{
  private:
    int k, n_block;
    T **A_block_U_ptrs, **A_block_V_ptrs, **B_block_U_ptrs, **B_block_V_ptrs;
    int *A_block_ranks, *B_block_ranks;
    T **Ua_ptrs, **Va_ptrs, **Ub_ptrs, **Vb_ptrs;
    int *a_ranks, *b_ranks, *tile_indices;

  public:
    TLR_Gemm_Sample_Tile_Product_Functor(int k, int n_block, T **A_block_U_ptrs, T **A_block_V_ptrs, T **B_block_U_ptrs,
                                         T **B_block_V_ptrs, int *A_block_ranks, int *B_block_ranks, T **Ua_ptrs,
                                         T **Va_ptrs, T **Ub_ptrs, T **Vb_ptrs, int *a_ranks, int *b_ranks,
                                         int *tile_indices)
    {
        this->k = k;
        this->n_block = n_block;
        this->A_block_U_ptrs = A_block_U_ptrs;
        this->A_block_V_ptrs = A_block_V_ptrs;
        this->B_block_U_ptrs = B_block_U_ptrs;
        this->B_block_V_ptrs = B_block_V_ptrs;
        this->A_block_ranks = A_block_ranks;
        this->B_block_ranks = B_block_ranks;
        this->Ua_ptrs = Ua_ptrs;
        this->Va_ptrs = Va_ptrs;
        this->Ub_ptrs = Ub_ptrs;
        this->Vb_ptrs = Vb_ptrs;
        this->a_ranks = a_ranks;
        this->b_ranks = b_ranks;
        this->tile_indices = tile_indices;
    }

    __host__ __device__ void operator()(const unsigned int &i) const
    {
        int tile_index = tile_indices[i];
        int C_row_index = tile_index % n_block;
        int C_col_index = tile_index / n_block;

        int A_index = C_row_index + k * n_block;
        int B_index = k + C_col_index * n_block;

        // Exclude the diagonal block from either matrix
        if (C_row_index == k || C_col_index == k)
        {
            Ua_ptrs[i] = Va_ptrs[i] = NULL;
            Ub_ptrs[i] = Vb_ptrs[i] = NULL;
            a_ranks[i] = b_ranks[i] = 0;
        }
        else
        {
            Ua_ptrs[i] = (transpose_sample ? B_block_V_ptrs[B_index] : A_block_U_ptrs[A_index]);
            Va_ptrs[i] = (transpose_sample ? B_block_U_ptrs[B_index] : A_block_V_ptrs[A_index]);
            a_ranks[i] = (transpose_sample ? B_block_ranks[B_index] : A_block_ranks[A_index]);

            Ub_ptrs[i] = (transpose_sample ? A_block_V_ptrs[A_index] : B_block_U_ptrs[B_index]);
            Vb_ptrs[i] = (transpose_sample ? A_block_U_ptrs[A_index] : B_block_V_ptrs[B_index]);
            b_ranks[i] = (transpose_sample ? A_block_ranks[A_index] : B_block_ranks[B_index]);
        }
    }
};

template <class T, bool transpose_sample> struct TLR_Gemm_Sample_Tile_Dense_Product_Functor
{
  private:
    int k, n_block;
    T **A_diagonal_block_ptrs, **B_diagonal_block_ptrs;
    T **A_block_U_ptrs, **A_block_V_ptrs, **B_block_U_ptrs, **B_block_V_ptrs;
    int *A_block_ranks, *B_block_ranks;
    T **Ua_ptrs, **Va_ptrs, **Ub_ptrs, **Vb_ptrs;
    T **Da_ptrs, **Db_ptrs;
    int *a_ranks, *b_ranks, *tile_indices;

  public:
    TLR_Gemm_Sample_Tile_Dense_Product_Functor(int n_block, T **A_diagonal_block_ptrs, T **B_diagonal_block_ptrs,
                                               T **A_block_U_ptrs, T **A_block_V_ptrs, T **B_block_U_ptrs,
                                               T **B_block_V_ptrs, int *A_block_ranks, int *B_block_ranks, T **Da_ptrs,
                                               T **Ub_ptrs, T **Vb_ptrs, T **Ua_ptrs, T **Va_ptrs, T **Db_ptrs,
                                               int *a_ranks, int *b_ranks, int *tile_indices)
    {
        this->n_block = n_block;
        this->A_diagonal_block_ptrs = A_diagonal_block_ptrs;
        this->B_diagonal_block_ptrs = B_diagonal_block_ptrs;
        this->A_block_U_ptrs = A_block_U_ptrs;
        this->A_block_V_ptrs = A_block_V_ptrs;
        this->B_block_U_ptrs = B_block_U_ptrs;
        this->B_block_V_ptrs = B_block_V_ptrs;
        this->A_block_ranks = A_block_ranks;
        this->B_block_ranks = B_block_ranks;
        this->Da_ptrs = Da_ptrs;
        this->Ub_ptrs = Ub_ptrs;
        this->Vb_ptrs = Vb_ptrs;
        this->Ua_ptrs = Ua_ptrs;
        this->Va_ptrs = Va_ptrs;
        this->Db_ptrs = Db_ptrs;
        this->a_ranks = a_ranks;
        this->b_ranks = b_ranks;
        this->tile_indices = tile_indices;
    }

    __host__ __device__ void operator()(const unsigned int &i) const
    {
        int tile_index = tile_indices[i];

        int C_row_index = tile_index % n_block;
        int C_col_index = tile_index / n_block;

        Da_ptrs[i] = (transpose_sample ? B_diagonal_block_ptrs[C_col_index] : A_diagonal_block_ptrs[C_row_index]);
        Ub_ptrs[i] = (transpose_sample ? A_block_V_ptrs[tile_index] : B_block_U_ptrs[tile_index]);
        Vb_ptrs[i] = (transpose_sample ? A_block_U_ptrs[tile_index] : B_block_V_ptrs[tile_index]);
        b_ranks[i] = (transpose_sample ? A_block_ranks[tile_index] : B_block_ranks[tile_index]);

        Ua_ptrs[i] = (transpose_sample ? B_block_V_ptrs[tile_index] : A_block_U_ptrs[tile_index]);
        Va_ptrs[i] = (transpose_sample ? B_block_U_ptrs[tile_index] : A_block_V_ptrs[tile_index]);
        a_ranks[i] = (transpose_sample ? B_block_ranks[tile_index] : A_block_ranks[tile_index]);
        Db_ptrs[i] = (transpose_sample ? A_diagonal_block_ptrs[C_row_index] : B_diagonal_block_ptrs[C_col_index]);
    }
};

struct TLR_Gemm_Tile_Setter
{
  private:
    int *tile_set, n_block;
    bool symmetric;

  public:
    TLR_Gemm_Tile_Setter(int *tile_set, bool symmetric, int n_block)
    {
        this->tile_set = tile_set;
        this->symmetric = symmetric;
        this->n_block = n_block;
    }

    __host__ __device__ void operator()(const unsigned int &block_index) const
    {
        int i = block_index % n_block;
        int j = block_index / n_block;

        // only consider the lower triangular tiles for a symmetric matrix
        if (symmetric)
        {
            if (i > j)
            {
                int tile_set_index =
                    (n_block * (n_block - 1)) / 2 - ((n_block - j) * (n_block - j - 1)) / 2 + i - j - 1;
                tile_set[tile_set_index] = block_index;
            }
        }
        else
        {
            if (i != j)
            {
                int tile_set_index = block_index - j;
                if (i > j)
                    tile_set_index--;
                tile_set[tile_set_index] = block_index;
            }
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Marshal routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw>
void tlr_gemm_diagonal_lr_marshal(int k, int n_block, T **A_block_U_ptrs, T **A_block_V_ptrs, T **B_block_U_ptrs,
                                  T **B_block_V_ptrs, int *A_block_ranks, int *B_block_ranks, T **Ua_ptrs, T **Va_ptrs,
                                  T **Ub_ptrs, T **Vb_ptrs, int *a_ranks, int *b_ranks, h2opusComputeStream_t stream)
{
    TLR_Gemm_LR_Diagonal_Functor<T> gemm_diagonal_lr_func(k, n_block, A_block_U_ptrs, A_block_V_ptrs, B_block_U_ptrs,
                                                          B_block_V_ptrs, A_block_ranks, B_block_ranks, Ua_ptrs,
                                                          Va_ptrs, Ub_ptrs, Vb_ptrs, a_ranks, b_ranks);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(n_block), gemm_diagonal_lr_func);
}

template <class T, int hw, bool transpose_sample>
void tlr_gemm_marshal_sample_tile_product(int k, int n_block, T **A_block_U_ptrs, T **A_block_V_ptrs,
                                          T **B_block_U_ptrs, T **B_block_V_ptrs, int *A_block_ranks,
                                          int *B_block_ranks, T **Ua_ptrs, T **Va_ptrs, T **Ub_ptrs, T **Vb_ptrs,
                                          int *a_ranks, int *b_ranks, int *tile_indices, int num_tiles,
                                          h2opusComputeStream_t stream)
{
    TLR_Gemm_Sample_Tile_Product_Functor<T, transpose_sample> tile_product_sampler(
        k, n_block, A_block_U_ptrs, A_block_V_ptrs, B_block_U_ptrs, B_block_V_ptrs, A_block_ranks, B_block_ranks,
        Ua_ptrs, Va_ptrs, Ub_ptrs, Vb_ptrs, a_ranks, b_ranks, tile_indices);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_tiles), tile_product_sampler);
}

template <class T, int hw, bool transpose_sample>
void tlr_gemm_marshal_sample_tile_dense_product(int n_block, T **A_diagonal_block_ptrs, T **B_diagonal_block_ptrs,
                                                T **A_block_U_ptrs, T **A_block_V_ptrs, T **B_block_U_ptrs,
                                                T **B_block_V_ptrs, int *A_block_ranks, int *B_block_ranks, T **Da_ptrs,
                                                T **Ub_ptrs, T **Vb_ptrs, T **Ua_ptrs, T **Va_ptrs, T **Db_ptrs,
                                                int *a_ranks, int *b_ranks, int *tile_indices, int num_tiles,
                                                h2opusComputeStream_t stream)
{
    TLR_Gemm_Sample_Tile_Dense_Product_Functor<T, transpose_sample> tile_dense_product_sampler(
        n_block, A_diagonal_block_ptrs, B_diagonal_block_ptrs, A_block_U_ptrs, A_block_V_ptrs, B_block_U_ptrs,
        B_block_V_ptrs, A_block_ranks, B_block_ranks, Da_ptrs, Ub_ptrs, Vb_ptrs, Ua_ptrs, Va_ptrs, Db_ptrs, a_ranks,
        b_ranks, tile_indices);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_tiles), tile_dense_product_sampler);
}

template <int hw>
void tlr_gemm_generate_tile_set(int *tile_set, bool symmetric, int n_block, h2opusComputeStream_t stream)
{
    TLR_Gemm_Tile_Setter tile_setter(tile_set, symmetric, n_block);
    int blocks = n_block * n_block;

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(blocks), tile_setter);
}

#endif
