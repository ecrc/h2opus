#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/thrust_runtime.h>

#include <thrust/for_each.h>
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Thrust unary functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct Upsweep_Functor
{
    T **v_ptrs, **x_ptrs, **y_ptrs;
    T *vtrans, *xhat_child, *xhat_parent;
    int num_vectors, parent_rank, child_rank, max_children;
    size_t num_parents;
    int parent_index_offset, child_index_offset;
    int *head, *next;

    Upsweep_Functor(T **v_ptrs, T **x_ptrs, T **y_ptrs, T *vtrans, T *xhat_child, T *xhat_parent, size_t num_parents,
                    int parent_rank, int child_rank, int parent_index_offset, int child_index_offset, int *head,
                    int *next, int max_children, int num_vectors)
    {
        this->v_ptrs = v_ptrs;
        this->x_ptrs = x_ptrs;
        this->y_ptrs = y_ptrs;

        this->vtrans = vtrans;
        this->xhat_child = xhat_child;
        this->xhat_parent = xhat_parent;

        this->num_parents = num_parents;
        this->parent_rank = parent_rank;
        this->child_rank = child_rank;
        this->max_children = max_children;
        this->parent_index_offset = parent_index_offset;
        this->child_index_offset = child_index_offset;

        this->head = head;
        this->next = next;

        this->num_vectors = num_vectors;
    }

    __host__ __device__ void operator()(const size_t &node_index)
    {
        int child_index = head[parent_index_offset + node_index];
        int child_count = 0;

        while (child_index != H2OPUS_EMPTY_NODE && child_count < max_children)
        {
            size_t child_level_index = child_index - child_index_offset;
            size_t ptr_index = child_count * num_parents + node_index;

            v_ptrs[ptr_index] = vtrans + child_level_index * child_rank * parent_rank;
            x_ptrs[ptr_index] = xhat_child + child_level_index * child_rank * num_vectors;
            y_ptrs[ptr_index] = xhat_parent + node_index * parent_rank * num_vectors;

            child_index = next[child_index];
            child_count++;
        }

        // Make sure the remaining pointers are set to NULL for nodes that don't
        // have max_children children
        for (int i = child_count; i < max_children; i++)
        {
            size_t ptr_index = i * num_parents + node_index;
            v_ptrs[ptr_index] = x_ptrs[ptr_index] = y_ptrs[ptr_index] = NULL;
        }
    }
};

template <class T> struct Upsweep_Leaves_Functor
{
    T *v_basis, *xhat_leaf_base, *X;
    T **A_ptrs, **B_ptrs, **C_ptrs;
    int *m_batch, *n_batch, *k_batch, *lda_batch, *ldb_batch, *ldc_batch;
    int *node_start, *node_len;
    int ldx, num_vectors, leaf_size, leaf_rank, node_offset;
    size_t num_leaves;

    Upsweep_Leaves_Functor(T *v_basis, int leaf_size, int leaf_rank, T **A_ptrs, T *X, int ldx, int num_vectors,
                           T **B_ptrs, T *xhat_leaf_base, T **C_ptrs, int node_offset, int *node_start, int *node_len,
                           int *m_batch, int *n_batch, int *k_batch, int *lda_batch, int *ldb_batch, int *ldc_batch,
                           size_t num_leaves)
    {
        this->v_basis = v_basis;
        this->xhat_leaf_base = xhat_leaf_base;
        this->X = X;

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

        this->ldx = ldx;
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

        A_ptrs[leaf_index] = v_basis + leaf_index * leaf_size * leaf_rank;
        B_ptrs[leaf_index] = X + leaf_start;
        C_ptrs[leaf_index] = xhat_leaf_base + leaf_index * leaf_rank * num_vectors;

        m_batch[leaf_index] = leaf_rank;
        n_batch[leaf_index] = num_vectors;
        k_batch[leaf_index] = leaf_rows;

        lda_batch[leaf_index] = leaf_size;
        ldb_batch[leaf_index] = ldx;
        ldc_batch[leaf_index] = leaf_rank;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Driver routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw>
inline void hgemv_upsweep_batch_marshal(T **A_ptrs, T **B_ptrs, T **C_ptrs, int max_children, size_t num_parents,
                                        size_t num_children, int num_vectors, T *vtrans, T *xhat_child, T *xhat_parent,
                                        int parent_rank, int child_rank, int parent_index_offset,
                                        int child_index_offset, int *head, int *next, h2opusComputeStream_t stream)
{
    Upsweep_Functor<T> upsweep_functor(A_ptrs, B_ptrs, C_ptrs, vtrans, xhat_child, xhat_parent, num_parents,
                                       parent_rank, child_rank, parent_index_offset, child_index_offset, head, next,
                                       max_children, num_vectors);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(num_parents), upsweep_functor);
}

template <class T, int hw>
inline void hgemv_upsweep_leaves_batch_marshal(T *v_basis, int leaf_size, int leaf_rank, T **A_ptrs, T *X, int ldx,
                                               int num_vectors, T **B_ptrs, T *xhat_leaf_base, T **C_ptrs,
                                               int node_offset, int *node_start, int *node_len, int *m_batch,
                                               int *n_batch, int *k_batch, int *lda_batch, int *ldb_batch,
                                               int *ldc_batch, size_t num_leaves, h2opusComputeStream_t stream)
{
    Upsweep_Leaves_Functor<T> upsweep_leaves_functor(v_basis, leaf_size, leaf_rank, A_ptrs, X, ldx, num_vectors, B_ptrs,
                                                     xhat_leaf_base, C_ptrs, node_offset, node_start, node_len, m_batch,
                                                     n_batch, k_batch, lda_batch, ldb_batch, ldc_batch, num_leaves);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(num_leaves), upsweep_leaves_functor);
}
