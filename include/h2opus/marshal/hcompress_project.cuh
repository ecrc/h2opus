#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/thrust_runtime.h>

#include <thrust/for_each.h>
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Thrust unary function
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct HcompressProject_Functor
{
    T *basis_tree_data;
    int increment, tree_node_offset, coupling_start; //, coupling_end;
    int *leaf_node_index, *basis_tree_nodes;
    T **array_of_tree_nodes;

    HcompressProject_Functor(int *leaf_node_index, int *basis_tree_nodes, T *basis_tree_data, T **array_of_tree_nodes,
                             int increment, int coupling_start, int tree_node_offset)
    {
        this->basis_tree_data = basis_tree_data;
        this->array_of_tree_nodes = array_of_tree_nodes;
        this->increment = increment;
        this->leaf_node_index = leaf_node_index;
        this->basis_tree_nodes = basis_tree_nodes;
        this->tree_node_offset = tree_node_offset;
        this->coupling_start = coupling_start;
    }

    __host__ __device__ void operator()(const unsigned int &thread_id) const
    {
        int coupling_leaf_index = coupling_start + thread_id;
        int node_index = basis_tree_nodes[leaf_node_index[coupling_leaf_index]] - tree_node_offset;

        array_of_tree_nodes[thread_id] = basis_tree_data + node_index * increment;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Driver routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw>
inline void hcompress_project_batch_marshal(int *leaf_node_index, int *basis_tree_nodes, T *basis_tree_data,
                                            T **array_of_tree_nodes, int increment, int coupling_start,
                                            int tree_node_offset, int num_arrays, h2opusComputeStream_t stream)
{
    HcompressProject_Functor<T> project_functor(leaf_node_index, basis_tree_nodes, basis_tree_data, array_of_tree_nodes,
                                                increment, coupling_start, tree_node_offset);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_arrays), project_functor);
}
