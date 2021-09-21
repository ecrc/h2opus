#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/thrust_runtime.h>

#include <thrust/for_each.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Thrust unary function
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct Dist_Horthog_OffDiagonal_Projection_Buffer
{
    T *T_hat_level, *buffer_base;
    int proj_rows, proj_cols;

    int *send_process_nodes;
    T **Tu_hat_ptrs, **buffer_ptrs;

    Dist_Horthog_OffDiagonal_Projection_Buffer(T *T_hat_level, T *buffer_base, int proj_rows, int proj_cols,
                                               int *send_process_nodes, T **Tu_hat_ptrs, T **buffer_ptrs)
    {
        this->T_hat_level = T_hat_level;
        this->buffer_base = buffer_base;
        this->proj_rows = proj_rows;
        this->proj_cols = proj_cols;
        this->send_process_nodes = send_process_nodes;
        this->Tu_hat_ptrs = Tu_hat_ptrs;
        this->buffer_ptrs = buffer_ptrs;
    }

    __host__ __device__ void operator()(const size_t &node_buffer_index)
    {
        int node_index = send_process_nodes[node_buffer_index];

        Tu_hat_ptrs[node_buffer_index] = T_hat_level + node_index * proj_rows * proj_cols;
        buffer_ptrs[node_buffer_index] = buffer_base + node_buffer_index * proj_rows * proj_cols;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Driver routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw>
inline void distributed_horthog_offdiag_projection_buffer_marshal(T *T_hat_level, T *buffer_base, int proj_rows,
                                                                  int proj_cols, int *send_process_nodes,
                                                                  T **Tu_hat_ptrs, T **buffer_ptrs, size_t num_nodes,
                                                                  h2opusComputeStream_t stream)
{
    Dist_Horthog_OffDiagonal_Projection_Buffer<T> offdiagonal_buffer(T_hat_level, buffer_base, proj_rows, proj_cols,
                                                                     send_process_nodes, Tu_hat_ptrs, buffer_ptrs);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(num_nodes), offdiagonal_buffer);
}
