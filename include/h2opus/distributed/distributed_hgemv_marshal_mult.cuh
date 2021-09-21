#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/thrust_runtime.h>

#include <thrust/for_each.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Thrust unary function
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct Dist_OffDiagonal_Input_Buffer
{
    T *input_base, *buffer_base;
    int ld_input, num_vectors;
    int *dense_send_node_offsets, *send_process_nodes;
    int *basis_node_start, *basis_node_len;
    int level_start_index;

    T **input_ptrs, **buffer_ptrs;
    int *m_ptrs, *n_ptrs, *ld_input_ptrs, *ld_buffer_ptrs;

    Dist_OffDiagonal_Input_Buffer(T *input_base, int ld_input, T *buffer_base, int num_vectors,
                                  int *dense_send_node_offsets, int level_start_index, int *send_process_nodes,
                                  int *basis_node_start, int *basis_node_len, T **input_ptrs, T **buffer_ptrs,
                                  int *m_ptrs, int *n_ptrs, int *ld_input_ptrs, int *ld_buffer_ptrs)
    {
        this->input_base = input_base;
        this->ld_input = ld_input;
        this->buffer_base = buffer_base;
        this->num_vectors = num_vectors;
        this->dense_send_node_offsets = dense_send_node_offsets;
        this->level_start_index = level_start_index;
        this->send_process_nodes = send_process_nodes;
        this->basis_node_start = basis_node_start;
        this->basis_node_len = basis_node_len;
        this->input_ptrs = input_ptrs;
        this->buffer_ptrs = buffer_ptrs;
        this->m_ptrs = m_ptrs;
        this->n_ptrs = n_ptrs;
        this->ld_input_ptrs = ld_input_ptrs;
        this->ld_buffer_ptrs = ld_buffer_ptrs;
    }

    __host__ __device__ void operator()(const size_t &node_buffer_index)
    {
        int basis_node_index = level_start_index + send_process_nodes[node_buffer_index];
        int node_buffer_offset = dense_send_node_offsets[node_buffer_index];

        int node_start = basis_node_start[basis_node_index];
        int node_len = basis_node_len[basis_node_index];

        input_ptrs[node_buffer_index] = input_base + node_start;
        buffer_ptrs[node_buffer_index] = buffer_base + node_buffer_offset * num_vectors;

        m_ptrs[node_buffer_index] = node_len;
        n_ptrs[node_buffer_index] = num_vectors;
        ld_input_ptrs[node_buffer_index] = ld_input;
        ld_buffer_ptrs[node_buffer_index] = node_len;
    }
};

template <class T> struct Dist_OffDiagonal_DenseMult
{
    T *node_base, *input_base, *output_base;
    int node_size, num_vectors, ld_output, node_offset;
    int *leaf_tree_index, *node_u_index, *compressed_node_v_index, *leaf_batch_index;
    int *node_u_start, *node_u_len, *compressed_node_offset;
    T **A_ptrs, **B_ptrs, **C_ptrs;
    int *m_ptrs, *n_ptrs, *k_ptrs, *lda_ptrs, *ldb_ptrs, *ldc_ptrs;

    Dist_OffDiagonal_DenseMult(T *node_base, int node_size, T *input_base, int num_vectors, T *output_base,
                               int ld_output, int *leaf_tree_index, int *node_u_index, int *compressed_node_v_index,
                               int *leaf_batch_index, int *node_u_start, int *node_u_len, int *compressed_node_offset,
                               int node_offset, T **A_ptrs, T **B_ptrs, T **C_ptrs, int *m_ptrs, int *n_ptrs,
                               int *k_ptrs, int *lda_ptrs, int *ldb_ptrs, int *ldc_ptrs)
    {
        this->node_base = node_base;
        this->node_size = node_size;
        this->input_base = input_base;
        this->num_vectors = num_vectors;
        this->output_base = output_base;
        this->ld_output = ld_output;
        this->leaf_tree_index = leaf_tree_index;
        this->node_u_index = node_u_index;
        this->compressed_node_v_index = compressed_node_v_index;
        this->leaf_batch_index = leaf_batch_index;
        this->node_u_start = node_u_start;
        this->node_u_len = node_u_len;
        this->compressed_node_offset = compressed_node_offset;
        this->node_offset = node_offset;

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
        int tree_index = leaf_tree_index[node_offset + leaf_index];
        int u_index = node_u_index[tree_index];
        int v_index = compressed_node_v_index[tree_index];

        size_t batch_index = leaf_batch_index[leaf_index];

        int node_v_offset = compressed_node_offset[v_index];
        int node_v_size = compressed_node_offset[v_index + 1] - node_v_offset;
        int node_u_offset = node_u_start[u_index];

        A_ptrs[batch_index] = node_base + leaf_index * node_size * node_size;
        C_ptrs[batch_index] = output_base + node_u_offset;
        // The input is strided by node instead of by column like the output
        B_ptrs[batch_index] = input_base + node_v_offset * num_vectors;

        m_ptrs[batch_index] = node_u_len[u_index];
        n_ptrs[batch_index] = num_vectors;
        k_ptrs[batch_index] = node_v_size;

        lda_ptrs[batch_index] = node_size;
        ldb_ptrs[batch_index] = node_v_size;
        ldc_ptrs[batch_index] = ld_output;
    }
};

template <class T> struct Dist_OffDiagonal_Xhat_Buffer
{
    T *xhat_level, *buffer_base;
    int level_rank, num_vectors;

    int *send_process_nodes;
    T **xhat_ptrs, **buffer_ptrs;

    Dist_OffDiagonal_Xhat_Buffer(T *xhat_level, T *buffer_base, int level_rank, int num_vectors,
                                 int *send_process_nodes, T **xhat_ptrs, T **buffer_ptrs)
    {
        this->xhat_level = xhat_level;
        this->buffer_base = buffer_base;
        this->level_rank = level_rank;
        this->num_vectors = num_vectors;
        this->send_process_nodes = send_process_nodes;
        this->xhat_ptrs = xhat_ptrs;
        this->buffer_ptrs = buffer_ptrs;
    }

    __host__ __device__ void operator()(const size_t &node_buffer_index)
    {
        int node_index = send_process_nodes[node_buffer_index];

        xhat_ptrs[node_buffer_index] = xhat_level + node_index * level_rank * num_vectors;
        buffer_ptrs[node_buffer_index] = buffer_base + node_buffer_index * level_rank * num_vectors;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Driver routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw>
inline void distributed_hgemv_offdiag_buffer_input_marshal(
    T *input_base, int ld_input, T *buffer_base, int num_vectors, int *dense_send_node_offsets, int level_start_index,
    int *send_process_nodes, int *basis_node_start, int *basis_node_len, T **input_ptrs, T **buffer_ptrs, int *m_ptrs,
    int *n_ptrs, int *ld_input_ptrs, int *ld_buffer_ptrs, size_t num_nodes, h2opusComputeStream_t stream)
{
    Dist_OffDiagonal_Input_Buffer<T> offdiag_input_buffer(
        input_base, ld_input, buffer_base, num_vectors, dense_send_node_offsets, level_start_index, send_process_nodes,
        basis_node_start, basis_node_len, input_ptrs, buffer_ptrs, m_ptrs, n_ptrs, ld_input_ptrs, ld_buffer_ptrs);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(num_nodes), offdiag_input_buffer);
}

template <class T, int hw>
inline void distributed_hgemv_offdiag_dense_mult_marshal(
    T *node_base, int node_size, T *input_base, int num_vectors, T *output_base, int ld_output, int *leaf_tree_index,
    int *node_u_index, int *compressed_node_v_index, int *leaf_batch_index, int *node_u_start, int *node_u_len,
    int *compressed_node_offset, int node_offset, T **A_ptrs, T **B_ptrs, T **C_ptrs, int *m_ptrs, int *n_ptrs,
    int *k_ptrs, int *lda_ptrs, int *ldb_ptrs, int *ldc_ptrs, size_t num_nodes, h2opusComputeStream_t stream)
{
    Dist_OffDiagonal_DenseMult<T> off_diagonal_dense_mult(
        node_base, node_size, input_base, num_vectors, output_base, ld_output, leaf_tree_index, node_u_index,
        compressed_node_v_index, leaf_batch_index, node_u_start, node_u_len, compressed_node_offset, node_offset,
        A_ptrs, B_ptrs, C_ptrs, m_ptrs, n_ptrs, k_ptrs, lda_ptrs, ldb_ptrs, ldc_ptrs);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(num_nodes), off_diagonal_dense_mult);
}

template <class T, int hw>
inline void distributed_hgemv_offdiag_coupling_buffer_input_marshal(T *xhat_level, T *buffer_base, int level_rank,
                                                                    int num_vectors, int *send_process_nodes,
                                                                    T **xhat_ptrs, T **buffer_ptrs, size_t num_nodes,
                                                                    h2opusComputeStream_t stream)
{
    Dist_OffDiagonal_Xhat_Buffer<T> offdiagonal_xhat_buffer(xhat_level, buffer_base, level_rank, num_vectors,
                                                            send_process_nodes, xhat_ptrs, buffer_ptrs);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(num_nodes), offdiagonal_xhat_buffer);
}
