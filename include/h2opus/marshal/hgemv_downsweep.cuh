#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/thrust_runtime.h>

#include <thrust/for_each.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Thrust unary functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct Downsweep_Functor
{
    T **u_ptrs, **x_ptrs, **y_ptrs;
    T *utrans, *yhat_parent, *yhat_child;

    int child_rank, parent_rank;
    int child_index_offset, parent_index_offset;
    int num_vectors;
    int *parent;

    Downsweep_Functor(T **u_ptrs, T **x_ptrs, T **y_ptrs, T *utrans, T *yhat_parent, T *yhat_child, int child_rank,
                      int parent_rank, int child_index_offset, int parent_index_offset, int *parent, int num_vectors)
    {
        this->u_ptrs = u_ptrs;
        this->x_ptrs = x_ptrs;
        this->y_ptrs = y_ptrs;

        this->utrans = utrans;
        this->yhat_parent = yhat_parent;
        this->yhat_child = yhat_child;

        this->child_rank = child_rank;
        this->parent_rank = parent_rank;
        this->child_index_offset = child_index_offset;
        this->parent_index_offset = parent_index_offset;
        this->parent = parent;

        this->num_vectors = num_vectors;
    }

    __host__ __device__ void operator()(const size_t &node_index)
    {
        int parent_index = parent[child_index_offset + node_index];

        if (parent_index != H2OPUS_EMPTY_NODE)
        {
            size_t parent_level_index = parent_index - parent_index_offset;

            u_ptrs[node_index] = utrans + node_index * child_rank * parent_rank;
            x_ptrs[node_index] = yhat_parent + parent_level_index * parent_rank * num_vectors;
            y_ptrs[node_index] = yhat_child + node_index * child_rank * num_vectors;
        }
        else
            u_ptrs[node_index] = x_ptrs[node_index] = y_ptrs[node_index] = NULL;
    }
};

template <class T> struct Downsweep_Leaves_Functor
{
    T *u_basis, *yhat_leaf_base, *Y;
    T **A_ptrs, **B_ptrs, **C_ptrs;
    int *m_batch, *n_batch, *k_batch, *lda_batch, *ldb_batch, *ldc_batch;
    int *node_start, *node_len;
    int ldy, num_vectors, leaf_size, leaf_rank, node_offset;
    size_t num_leaves;

    Downsweep_Leaves_Functor(T *u_basis, int leaf_size, int leaf_rank, T **A_ptrs, T *yhat_leaf_base, T **B_ptrs, T *Y,
                             int ldy, int num_vectors, T **C_ptrs, int node_offset, int *node_start, int *node_len,
                             int *m_batch, int *n_batch, int *k_batch, int *lda_batch, int *ldb_batch, int *ldc_batch,
                             size_t num_leaves)
    {
        this->u_basis = u_basis;
        this->yhat_leaf_base = yhat_leaf_base;
        this->Y = Y;

        this->A_ptrs = A_ptrs;
        this->B_ptrs = B_ptrs;
        this->C_ptrs = C_ptrs;

        this->m_batch = m_batch;
        this->n_batch = n_batch;
        this->k_batch = k_batch;
        this->lda_batch = lda_batch;
        this->ldb_batch = ldb_batch;
        this->ldc_batch = ldc_batch;

        this->node_start = node_start;
        this->node_len = node_len;

        this->ldy = ldy;
        this->num_vectors = num_vectors;
        this->leaf_size = leaf_size;
        this->leaf_rank = leaf_rank;
        this->node_offset = node_offset;
        this->num_leaves = num_leaves;
    }

    __host__ __device__ void operator()(const size_t &leaf_index)
    {
        int node_index = node_offset + leaf_index;
        int leaf_rows = node_len[node_index];
        int leaf_start = node_start[node_index];

        A_ptrs[leaf_index] = u_basis + leaf_index * leaf_size * leaf_rank;
        B_ptrs[leaf_index] = yhat_leaf_base + leaf_index * leaf_rank * num_vectors;
        C_ptrs[leaf_index] = Y + leaf_start;

        m_batch[leaf_index] = leaf_rows;
        n_batch[leaf_index] = num_vectors;
        k_batch[leaf_index] = leaf_rank;

        lda_batch[leaf_index] = leaf_size;
        ldb_batch[leaf_index] = leaf_rank;
        ldc_batch[leaf_index] = ldy;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Driver routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw>
inline void hgemv_downsweep_batch_marshal(T **A_ptrs, T **B_ptrs, T **C_ptrs, size_t num_children, size_t num_parents,
                                          int num_vectors, T *utrans, T *Yhat_parent, T *Yhat_child, int child_rank,
                                          int parent_rank, int child_index_offset, int parent_index_offset, int *parent,
                                          h2opusComputeStream_t stream)
{
    Downsweep_Functor<T> downsweep_functor(A_ptrs, B_ptrs, C_ptrs, utrans, Yhat_parent, Yhat_child, child_rank,
                                           parent_rank, child_index_offset, parent_index_offset, parent, num_vectors);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(num_children), downsweep_functor);
}

template <class T, int hw>
inline void hgemv_downsweep_leaves_batch_marshal(T *u_basis, int leaf_size, int leaf_rank, T **A_ptrs,
                                                 T *yhat_leaf_base, T **B_ptrs, T *Y, int ldy, int num_vectors,
                                                 T **C_ptrs, int node_offset, int *node_start, int *node_len,
                                                 int *m_batch, int *n_batch, int *k_batch, int *lda_batch,
                                                 int *ldb_batch, int *ldc_batch, size_t num_leaves,
                                                 h2opusComputeStream_t stream)
{
    Downsweep_Leaves_Functor<T> downsweep_leaves_functor(
        u_basis, leaf_size, leaf_rank, A_ptrs, yhat_leaf_base, B_ptrs, Y, ldy, num_vectors, C_ptrs, node_offset,
        node_start, node_len, m_batch, n_batch, k_batch, lda_batch, ldb_batch, ldc_batch, num_leaves);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(num_leaves), downsweep_leaves_functor);
}
