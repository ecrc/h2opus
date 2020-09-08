#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/thrust_runtime.h>

#include <thrust/for_each.h>
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Thrust unary function
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct Mult_Functor
{
    int *node_u_index, *node_v_index, *leaf_tree_index, *leaf_batch_index;
    int node_rank, node_offset, u_index_offset, v_index_offset;
    int num_vectors;
    T *input_base, *node_base, *ouput_base;
    T **A_ptrs, **B_ptrs, **C_ptrs;

    Mult_Functor(int *node_u_index, int *node_v_index, int *leaf_tree_index, int *leaf_batch_index, int node_offset,
                 int u_index_offset, int v_index_offset, T *input_base, T *node_base, int node_rank, int num_vectors,
                 T *ouput_base, T **A_ptrs, T **B_ptrs, T **C_ptrs)
    {
        this->node_u_index = node_u_index;
        this->node_v_index = node_v_index;
        this->leaf_tree_index = leaf_tree_index;
        this->leaf_batch_index = leaf_batch_index;

        this->node_rank = node_rank;
        this->num_vectors = num_vectors;
        this->node_offset = node_offset;
        this->u_index_offset = u_index_offset;
        this->v_index_offset = v_index_offset;

        this->input_base = input_base;
        this->node_base = node_base;
        this->ouput_base = ouput_base;

        this->A_ptrs = A_ptrs;
        this->B_ptrs = B_ptrs;
        this->C_ptrs = C_ptrs;
    }

    __host__ __device__ void operator()(const size_t &leaf_index)
    {
        size_t tree_index = leaf_tree_index[node_offset + leaf_index];
        size_t u_index = node_u_index[tree_index] - u_index_offset;
        size_t v_index = node_v_index[tree_index] - v_index_offset;

        size_t batch_index = leaf_batch_index[leaf_index];

        A_ptrs[batch_index] = node_base + leaf_index * node_rank * node_rank;
        B_ptrs[batch_index] = input_base + v_index * node_rank * num_vectors;
        C_ptrs[batch_index] = ouput_base + u_index * node_rank * num_vectors;
    }
};

template <class T> struct Dense_Mult_Functor
{
    int *node_u_index, *node_v_index, *node_u_start, *node_v_start, *node_u_len, *node_v_len;
    int *leaf_tree_index, *leaf_batch_index;
    int node_size, num_vectors, node_offset;
    int ld_input, ld_output;
    T *input_base, *node_base, *ouput_base;
    T **A_ptrs, **B_ptrs, **C_ptrs;
    int *m_ptrs, *n_ptrs, *k_ptrs, *lda_ptrs, *ldb_ptrs, *ldc_ptrs;

    Dense_Mult_Functor(int *node_u_index, int *node_v_index, int *node_u_start, int *node_v_start, int *node_u_len,
                       int *node_v_len, int *leaf_tree_index, int *leaf_batch_index, int node_offset, T *input_base,
                       int ld_input, T *node_base, int node_size, T *ouput_base, int ld_output, int num_vectors,
                       T **A_ptrs, T **B_ptrs, T **C_ptrs, int *m_ptrs, int *n_ptrs, int *k_ptrs, int *lda_ptrs,
                       int *ldb_ptrs, int *ldc_ptrs)
    {
        this->node_u_index = node_u_index;
        this->node_v_index = node_v_index;
        this->node_u_start = node_u_start;
        this->node_v_start = node_v_start;
        this->node_u_len = node_u_len;
        this->node_v_len = node_v_len;

        this->leaf_tree_index = leaf_tree_index;
        this->leaf_batch_index = leaf_batch_index;

        this->node_size = node_size;
        this->node_offset = node_offset;
        this->ld_output = ld_output;
        this->ld_input = ld_input;
        this->num_vectors = num_vectors;

        this->input_base = input_base;
        this->node_base = node_base;
        this->ouput_base = ouput_base;

        this->A_ptrs = A_ptrs;
        this->B_ptrs = B_ptrs;
        this->C_ptrs = C_ptrs;
        this->m_ptrs = m_ptrs;
        this->n_ptrs = n_ptrs;
        this->k_ptrs = k_ptrs;
        this->lda_ptrs = lda_ptrs;
        this->ldb_ptrs = ldb_ptrs;
        this->ldc_ptrs = ldc_ptrs;
    }

    __host__ __device__ void operator()(const size_t &leaf_index)
    {
        size_t tree_index = leaf_tree_index[node_offset + leaf_index];
        size_t u_index = node_u_index[tree_index];
        size_t v_index = node_v_index[tree_index];

        size_t batch_index = leaf_batch_index[leaf_index];

        A_ptrs[batch_index] = node_base + leaf_index * node_size * node_size;
        B_ptrs[batch_index] = input_base + node_v_start[v_index];
        C_ptrs[batch_index] = ouput_base + node_u_start[u_index];

        m_ptrs[batch_index] = node_u_len[u_index];
        n_ptrs[batch_index] = num_vectors;
        k_ptrs[batch_index] = node_v_len[v_index];

        lda_ptrs[batch_index] = node_size;
        ldb_ptrs[batch_index] = ld_input;
        ldc_ptrs[batch_index] = ld_output;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Driver routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw>
inline void hgemv_mult_batch_marshal(int *node_u_index, int *node_v_index, int *leaf_tree_index, int *leaf_batch_index,
                                     int node_offset, int u_index_offset, int v_index_offset, T *input_base,
                                     T *node_base, int node_size, int num_vectors, T *ouput_base, T **A_ptrs,
                                     T **B_ptrs, T **C_ptrs, size_t leaf_nodes, h2opusComputeStream_t stream)
{
    Mult_Functor<T> mult_functor(node_u_index, node_v_index, leaf_tree_index, leaf_batch_index, node_offset,
                                 u_index_offset, v_index_offset, input_base, node_base, node_size, num_vectors,
                                 ouput_base, A_ptrs, B_ptrs, C_ptrs);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(leaf_nodes), mult_functor);
}

template <class T, int hw>
inline void hgemv_dense_mult_batch_marshal(int *node_u_index, int *node_v_index, int *node_u_start, int *node_v_start,
                                           int *node_u_len, int *node_v_len, int *leaf_tree_index,
                                           int *leaf_batch_index, int node_offset, T *input_base, int ld_input,
                                           T *node_base, int node_size, T *ouput_base, int ld_output, int num_vectors,
                                           T **A_ptrs, T **B_ptrs, T **C_ptrs, int *m_ptrs, int *n_ptrs, int *k_ptrs,
                                           int *lda_ptrs, int *ldb_ptrs, int *ldc_ptrs, size_t leaf_nodes,
                                           h2opusComputeStream_t stream)
{
    Dense_Mult_Functor<T> dense_mult_functor(node_u_index, node_v_index, node_u_start, node_v_start, node_u_len,
                                             node_v_len, leaf_tree_index, leaf_batch_index, node_offset, input_base,
                                             ld_input, node_base, node_size, ouput_base, ld_output, num_vectors, A_ptrs,
                                             B_ptrs, C_ptrs, m_ptrs, n_ptrs, k_ptrs, lda_ptrs, ldb_ptrs, ldc_ptrs);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(leaf_nodes), dense_mult_functor);
}
