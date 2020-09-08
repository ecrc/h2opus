#ifndef __HARA_UTIL_H__
#define __HARA_UTIL_H__

#include <h2opus/core/thrust_runtime.h>
#include <h2opus/util/thrust_wrappers.h>
#include <thrust/logical.h>

//////////////////////////////////////////////////////////////////////////////////////////
// Functors
//////////////////////////////////////////////////////////////////////////////////////////
struct HARA_SetSample_Functor
{
    int *op_samples, *small_vectors;
    int *op_ranks, *op_rows, *op_cols;
    int samples, r;

    HARA_SetSample_Functor(int *op_samples, int *small_vectors, int samples, int r, int *op_ranks, int *op_rows,
                           int *op_cols)
    {
        this->op_samples = op_samples;
        this->small_vectors = small_vectors;
        this->r = r;
        this->samples = samples;

        this->op_ranks = op_ranks;
        this->op_rows = op_rows;
        this->op_cols = op_cols;
    }

    __host__ __device__ void operator()(const unsigned int &op_index) const
    {
        int op_m = op_rows[op_index], op_n = op_cols[op_index];
        int max_rank = (op_m < op_n ? op_m : op_n);

        int max_samples = max_rank - op_ranks[op_index];
        if (small_vectors[op_index] >= r)
            max_samples = 0;

        int s = (max_samples < samples ? max_samples : samples);
        op_samples[op_index] = s;
    }
};

struct HARA_ZeroSample_Predicate
{
    __host__ __device__ bool operator()(const int &x)
    {
        return x == 0;
    }
};

template <class T> struct HARA_SvecCount_Functor
{
    T **U_ptrs, **Y_ptrs;
    double *diag_R, tol;
    int r, BS, *ldu_batch;
    int *block_ranks, *node_ranks, *small_vectors;

    HARA_SvecCount_Functor(int *block_ranks, int *node_ranks, int *small_vectors, double *diag_R, int r, double tol,
                           int BS, T **Y_ptrs, T **U_ptrs, int *ldu_batch)
    {
        this->block_ranks = block_ranks;
        this->node_ranks = node_ranks;
        this->small_vectors = small_vectors;
        this->diag_R = diag_R;
        this->r = r;
        this->tol = tol;
        this->BS = BS;

        this->Y_ptrs = Y_ptrs;
        this->U_ptrs = U_ptrs;
        this->ldu_batch = ldu_batch;
    }

    __host__ __device__ void operator()(const unsigned int &op_index) const
    {
        int block_rank = block_ranks[op_index];
        int small_vecs_op = small_vectors[op_index];

        // if this block has rank 0, the operation has already converged
        if (block_rank == 0)
        {
            small_vectors[op_index] = r;
            return;
        }

        double *diag_op = diag_R + op_index * BS;
        int ldu = ldu_batch[op_index];
        int current_rank = node_ranks[op_index];

        // Check how many consecutive small vectors we have
        for (int i = 0; i < block_rank; i++)
        {
            current_rank++;
            if (diag_op[i] < tol)
            {
                small_vecs_op++;
                if (small_vecs_op == r)
                {
                    current_rank -= r;
                    break;
                }
            }
            else
                small_vecs_op = 0;
        }

        // If the cholesky for this block determined that it is rank deficient
        // then it's likely that this operation has converged and we found the
        // exact rank
        if (block_rank != BS)
            small_vecs_op = r;

        // Update the rank and small vector counts
        node_ranks[op_index] = current_rank;
        small_vectors[op_index] = small_vecs_op;
        Y_ptrs[op_index] = U_ptrs[op_index] + current_rank * ldu;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// Utility routines
//////////////////////////////////////////////////////////////////////////////////////////
template <int hw>
inline int hara_util_set_batch_samples(int *op_samples, int *small_vectors, int samples, int r, int *op_ranks,
                                       int *op_rows, int *op_cols, int num_ops, h2opusComputeStream_t stream)
{
    HARA_SetSample_Functor set_sample_functor(op_samples, small_vectors, samples, r, op_ranks, op_rows, op_cols);

    int converged = 0;

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_ops), set_sample_functor);

    bool all_zero =
        thrust::all_of(ThrustRuntime<hw>::get(stream), op_samples, op_samples + num_ops, HARA_ZeroSample_Predicate());

    if (all_zero)
        converged = 1;

    return converged;
}

template <class T, int hw>
inline void hara_util_svec_count_batch(int *block_ranks, int *node_ranks, int *small_vectors, double *diag_R, int r,
                                       double tol, int BS, T **Y_ptrs, T **U_ptrs, int *ldu_batch, int num_ops,
                                       h2opusComputeStream_t stream)
{
    HARA_SvecCount_Functor<T> svec_count_functor(block_ranks, node_ranks, small_vectors, diag_R, r, tol, BS, Y_ptrs,
                                                 U_ptrs, ldu_batch);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_ops), svec_count_functor);
}

#endif