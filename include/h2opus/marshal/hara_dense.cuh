#include <h2opus/core/thrust_runtime.h>
#include <h2opus/util/morton.h>

#include <thrust/for_each.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Thrust functors
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct HARA_WeakAdm_Dense_Update_Input_Functor
{
    T *input, **input_ptrs;
    int *rows_array, *cols_array;
    int *hnode_indexes, *node_u_index, *node_v_index;
    int *node_u_len, *node_v_len, *node_v_start;
    int update_cols;

    HARA_WeakAdm_Dense_Update_Input_Functor(T *input, T **input_ptrs, int *rows_array, int *cols_array,
                                            int *hnode_indexes, int *node_u_index, int *node_v_index, int *node_u_len,
                                            int *node_v_len, int *node_v_start, int update_cols)
    {
        this->input = input;
        this->input_ptrs = input_ptrs;
        this->rows_array = rows_array;
        this->cols_array = cols_array;

        this->hnode_indexes = hnode_indexes;
        this->node_u_index = node_u_index;
        this->node_v_index = node_v_index;
        this->node_u_len = node_u_len;
        this->node_v_len = node_v_len;
        this->node_v_start = node_v_start;
        this->update_cols = update_cols;
    }

    __host__ __device__ void operator()(const unsigned int &update_index) const
    {
        int hnode_index = hnode_indexes[update_index];
        int u_index = node_u_index[hnode_index];
        int v_index = node_v_index[hnode_index];

        rows_array[update_index] = node_u_len[u_index];
        // The generated matrices should be node_u_len x node_v_len
        // but we extend it to the full columns of the update, so that the
        // batch transpose can fill in the zeros of the input vectors for us
        cols_array[update_index] = update_cols; // node_v_len[v_index];

        input_ptrs[update_index] = input + node_v_start[v_index];
    }
};

template <class T> struct HARA_WeakAdm_Dense_Update_Output_Functor
{
    T *output, **output_ptrs;
    int *hnode_indexes, *node_u_index, *node_u_start;

    HARA_WeakAdm_Dense_Update_Output_Functor(T *output, T **output_ptrs, int *hnode_indexes, int *node_u_index,
                                             int *node_u_start)
    {
        this->output = output;
        this->output_ptrs = output_ptrs;
        this->hnode_indexes = hnode_indexes;
        this->node_u_index = node_u_index;
        this->node_u_start = node_u_start;
    }

    __host__ __device__ void operator()(const unsigned int &update_index) const
    {
        int hnode_index = hnode_indexes[update_index];
        int u_index = node_u_index[hnode_index];

        output_ptrs[update_index] = output + node_u_start[u_index];
    }
};

struct HARA_WeakAdm_Dense_Update_Morton_Functor
{
    int *update_morton_indexes;

    HARA_WeakAdm_Dense_Update_Morton_Functor(int *update_morton_indexes)
    {
        this->update_morton_indexes = update_morton_indexes;
    }

    __host__ __device__ void operator()(const unsigned int &update_index) const
    {
        update_morton_indexes[update_index] = morton_encode(update_index, update_index);
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Driver routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw>
inline void hara_weak_admissibility_dense_update_input_marshal_batch(T *input, T **input_ptrs, int *rows_array,
                                                                     int *cols_array, int *hnode_indexes,
                                                                     int *node_u_index, int *node_v_index,
                                                                     int *node_u_len, int *node_v_len,
                                                                     int *node_v_start, int update_cols,
                                                                     int num_updates, h2opusComputeStream_t stream)
{
    HARA_WeakAdm_Dense_Update_Input_Functor<T> dense_update_input_functor(
        input, input_ptrs, rows_array, cols_array, hnode_indexes, node_u_index, node_v_index, node_u_len, node_v_len,
        node_v_start, update_cols);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_updates), dense_update_input_functor);
}

template <class T, int hw>
inline void hara_weak_admissibility_dense_update_output_marshal_batch(T *output, T **output_ptrs, int *hnode_indexes,
                                                                      int *node_u_index, int *node_u_start,
                                                                      int num_updates, h2opusComputeStream_t stream)
{
    HARA_WeakAdm_Dense_Update_Output_Functor<T> dense_update_output_functor(output, output_ptrs, hnode_indexes,
                                                                            node_u_index, node_u_start);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_updates), dense_update_output_functor);
}
template <int hw>
void hara_weak_admissibility_dense_update_hnode_index_marshal_batch(int *update_morton_indexes,
                                                                    int *update_hnode_indexes,
                                                                    int *node_morton_level_index, int hnode_start_index,
                                                                    int num_hnodes, int num_updates,
                                                                    h2opusComputeStream_t stream)
{
    HARA_WeakAdm_Dense_Update_Morton_Functor dense_update_morton_functor(update_morton_indexes);

    HARA_LowerBound_To_HNode_Functor lb_to_hnode_functor(update_morton_indexes, update_hnode_indexes,
                                                         node_morton_level_index, hnode_start_index, num_hnodes);

    int hnode_end_index = hnode_start_index + num_hnodes;

    // First generate list of search morton indexes
    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_updates), dense_update_morton_functor);

    // Find the
    thrust::lower_bound(ThrustRuntime<hw>::get(stream), node_morton_level_index + hnode_start_index,
                        node_morton_level_index + hnode_end_index, update_morton_indexes,
                        update_morton_indexes + num_updates, update_hnode_indexes);

    // Transform the lower bound indexes to hnode indexes
    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_updates), lb_to_hnode_functor);
}
