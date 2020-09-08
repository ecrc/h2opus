#include <h2opus/core/thrust_runtime.h>
#include <h2opus/util/morton.h>

#include <thrust/for_each.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Thrust functors
//////////////////////////////////////////////////////////////////////////////////////////////////////////
struct HARA_WeakAdmMorton_Functor
{
    int *morton_indexes, level_nodes;

    HARA_WeakAdmMorton_Functor(int *morton_indexes, int level_nodes)
    {
        this->morton_indexes = morton_indexes;
        this->level_nodes = level_nodes;
    }

    __host__ __device__ void operator()(const unsigned int &update_index) const
    {
        int i = 2 * update_index;
        int j = i + 1;

        morton_indexes[update_index] = morton_encode(j, i);
        morton_indexes[update_index + level_nodes] = morton_encode(i, j);
    }
};

template <class T> struct HARA_LRU_Ptr_Functor
{
    T *U_base, *V_base;
    T **U_ptrs, **V_ptrs;
    int *input_block_sizes, *output_block_sizes;
    int ldu, ldv;
    int *hnode_indexes, *node_u_index, *node_v_index;
    int *node_u_start, *node_v_start, *node_u_len, *node_v_len;
    int num_updates;

    HARA_LRU_Ptr_Functor(T *U_base, int ldu, T *V_base, int ldv, T **U_ptrs, T **V_ptrs, int *input_block_sizes,
                         int *output_block_sizes, int *hnode_indexes, int *node_u_index, int *node_v_index,
                         int *node_u_start, int *node_v_start, int *node_u_len, int *node_v_len, int num_updates)
    {
        this->U_base = U_base;
        this->ldu = ldu;
        this->V_base = V_base;
        this->ldv = ldv;
        this->U_ptrs = U_ptrs;
        this->V_ptrs = V_ptrs;

        this->input_block_sizes = input_block_sizes;
        this->output_block_sizes = output_block_sizes;

        this->hnode_indexes = hnode_indexes;
        this->node_u_index = node_u_index;
        this->node_v_index = node_v_index;
        this->node_u_start = node_u_start;
        this->node_v_start = node_v_start;
        this->node_u_len = node_u_len;
        this->node_v_len = node_v_len;

        this->num_updates = num_updates;
    }

    __host__ __device__ void operator()(const unsigned int &update_index) const
    {
        int hnode_index = hnode_indexes[update_index];
        int u_index = node_u_index[hnode_index];
        int v_index = node_v_index[hnode_index];
        int u_start = node_u_start[u_index];
        int v_start = node_v_start[v_index];

        output_block_sizes[update_index] = node_u_len[u_index];
        input_block_sizes[update_index] = node_v_len[v_index];

        U_ptrs[update_index] = U_base + u_start;
        V_ptrs[update_index] = V_base + v_start;

        U_ptrs[update_index + num_updates] = V_base + v_start;
        V_ptrs[update_index + num_updates] = U_base + u_start;
    }
};

template <class T> struct HARA_WeakAdm_RandInput_Functor
{
    T *input, **ptrs;
    int *update_hnode_indexes, *node_v_index, *node_v_start;

    HARA_WeakAdm_RandInput_Functor(T *input, T **ptrs, int *update_hnode_indexes, int *node_v_index, int *node_v_start)
    {
        this->input = input;
        this->ptrs = ptrs;

        this->update_hnode_indexes = update_hnode_indexes;
        this->node_v_index = node_v_index;
        this->node_v_start = node_v_start;
    }

    __host__ __device__ void operator()(const unsigned int &update_index) const
    {
        int hnode_index = update_hnode_indexes[update_index];
        int v_index = node_v_index[hnode_index];
        int node_start = node_v_start[v_index];

        ptrs[update_index] = input + node_start;
    }
};

template <class T> struct HARA_WeakAdm_ClearOutput_Functor
{
    T *output, **ptrs;
    int *update_hnode_indexes, *node_v_index, *node_v_start;
    int *node_v_len, *output_block_rows;

    HARA_WeakAdm_ClearOutput_Functor(T *output, T **ptrs, int *update_hnode_indexes, int *node_v_index,
                                     int *node_v_start, int *node_v_len, int *output_block_rows)
    {
        this->output = output;
        this->ptrs = ptrs;

        this->update_hnode_indexes = update_hnode_indexes;
        this->node_v_index = node_v_index;
        this->node_v_start = node_v_start;
        this->node_v_len = node_v_len;
        this->output_block_rows = output_block_rows;
    }

    __host__ __device__ void operator()(const unsigned int &update_index) const
    {
        int hnode_index = update_hnode_indexes[update_index];
        int v_index = node_v_index[hnode_index];
        int node_start = node_v_start[v_index];
        int node_len = node_v_len[v_index];

        ptrs[update_index] = output + node_start;
        output_block_rows[update_index] = node_len;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// H2OPUS weak admissibility routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <int hw>
inline void hara_util_generate_weak_admissibility_morton_indexes(int *update_morton_indexes, int num_updates,
                                                                 h2opusComputeStream_t stream)
{
    // Mirror the updates so we divide by two
    num_updates /= 2;

    HARA_WeakAdmMorton_Functor morton_functor(update_morton_indexes, num_updates);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_updates), morton_functor);
}

template <int hw>
inline void hara_util_generate_weak_admissibility_hnode_indexes(int *update_morton_indexes, int *hnode_indexes,
                                                                int *hnode_morton_indexes, int num_updates,
                                                                int hnode_level_start, int hnode_level_size,
                                                                h2opusComputeStream_t stream)
{
    HARA_LowerBound_To_HNode_Functor lb_to_hnode_functor(update_morton_indexes, hnode_indexes, hnode_morton_indexes,
                                                         hnode_level_start, hnode_level_size);

    int hnode_level_end = hnode_level_start + hnode_level_size;

    thrust::lower_bound(ThrustRuntime<hw>::get(stream), hnode_morton_indexes + hnode_level_start,
                        hnode_morton_indexes + hnode_level_end, update_morton_indexes,
                        update_morton_indexes + num_updates, hnode_indexes);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_updates), lb_to_hnode_functor);
}

template <class T, int hw>
inline void hara_util_generate_weak_admissibility_update_ptrs(T *U_base, int ldu, T *V_base, int ldv, T **U_ptrs,
                                                              T **V_ptrs, int *input_block_sizes,
                                                              int *output_block_sizes, int *hnode_indexes,
                                                              int *node_u_index, int *node_v_index, int *node_u_start,
                                                              int *node_v_start, int *node_u_len, int *node_v_len,
                                                              int num_updates, h2opusComputeStream_t stream)
{
    // Mirror the updates so we divide by two
    num_updates /= 2;

    HARA_LRU_Ptr_Functor<T> lru_ptr_functor(U_base, ldu, V_base, ldv, U_ptrs, V_ptrs, input_block_sizes,
                                            output_block_sizes, hnode_indexes, node_u_index, node_v_index, node_u_start,
                                            node_v_start, node_u_len, node_v_len, num_updates);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_updates), lru_ptr_functor);
}

template <class T, int hw>
inline void hara_weak_admissibility_update_marshal_batch(T *sampled_U, int ldu, T *sampled_V, int ldv, T **U_ptrs,
                                                         T **V_ptrs, int *input_block_sizes, int *output_block_sizes,
                                                         int *update_morton_indexes, int num_updates,
                                                         int *hnode_indexes, int *node_morton_level_index,
                                                         int hnode_level_start, int hnode_level_size, int *node_u_index,
                                                         int *node_v_index, int *node_u_start, int *node_v_start,
                                                         int *node_u_len, int *node_v_len, h2opusComputeStream_t stream)
{
    // Generate morton indexes for a weak admissibility sampling structure
    hara_util_generate_weak_admissibility_morton_indexes<hw>(update_morton_indexes, num_updates, stream);

    // Grab the hnode indexes in the approximation stucture
    hara_util_generate_weak_admissibility_hnode_indexes<hw>(update_morton_indexes, hnode_indexes,
                                                            node_morton_level_index, num_updates, hnode_level_start,
                                                            hnode_level_size, stream);

    // Finally, assign pointers to the low rank updates based on the cluster data
    // of the basis trees
    hara_util_generate_weak_admissibility_update_ptrs<T, hw>(
        sampled_U, ldu, sampled_V, ldv, U_ptrs, V_ptrs, input_block_sizes, output_block_sizes, hnode_indexes,
        node_u_index, node_v_index, node_u_start, node_v_start, node_u_len, node_v_len, num_updates, stream);
}

template <class T, int hw>
inline void hara_weak_admissibility_random_input_marshal_batch(T *input, T **ptrs, int *update_hnode_indexes,
                                                               int *node_v_index, int *node_v_start, int num_updates,
                                                               h2opusComputeStream_t stream)
{
    // The updates are mirrored, so we only need to generate input
    // for the first half of the updates
    num_updates /= 2;

    HARA_WeakAdm_RandInput_Functor<T> rand_input_functor(input, ptrs, update_hnode_indexes, node_v_index, node_v_start);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_updates), rand_input_functor);
}

template <class T, int hw>
inline void hara_weak_admissibility_clear_output_marshal_batch(T *output, T **ptrs, int *update_hnode_indexes,
                                                               int *node_v_index, int *node_v_start, int *node_v_len,
                                                               int *output_block_rows, int num_updates,
                                                               h2opusComputeStream_t stream)
{
    // The updates are mirrored, so we only need to clear output
    // for the first half of the updates
    num_updates /= 2;

    HARA_WeakAdm_ClearOutput_Functor<T> clear_output_functor(output, ptrs, update_hnode_indexes, node_v_index,
                                                             node_v_start, node_v_len, output_block_rows);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_updates), clear_output_functor);
}
