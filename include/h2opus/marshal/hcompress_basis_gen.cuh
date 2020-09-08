#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/thrust_runtime.h>

#include <thrust/for_each.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Thrust unary function
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct Hcompress_Parent_Weight_Functor
{
    T **ptr_ZE, **ptr_Z, **ptr_E;
    T *ZE, *Z, *E;
    int *parent, parent_start, level_start, level_offset;
    int ld_ZE, parent_rank, level_rank;

    Hcompress_Parent_Weight_Functor(T **ptr_ZE, T *ZE, T **ptr_Z, T *Z, T **ptr_E, T *E, int *parent, int parent_start,
                                    int level_start, int level_offset, int ld_ZE, int parent_rank, int level_rank)
    {
        this->ptr_ZE = ptr_ZE;
        this->ptr_Z = ptr_Z;
        this->ptr_E = ptr_E;
        this->ZE = ZE;
        this->Z = Z;
        this->E = E;

        this->parent = parent;
        this->parent_start = parent_start;
        this->level_start = level_start;
        this->level_offset = level_offset;
        this->ld_ZE = ld_ZE;
        this->parent_rank = parent_rank;
        this->level_rank = level_rank;
    }

    __host__ __device__ void operator()(const unsigned int &op_index) const
    {
        int node_index = op_index + level_offset;
        int parent_index = parent[level_start + node_index];
        int parent_level_index = parent_index - parent_start;

        if (parent_index == H2OPUS_EMPTY_NODE || parent_level_index < 0)
            ptr_ZE[op_index] = ptr_E[op_index] = ptr_Z[op_index] = NULL;
        else
        {
            ptr_ZE[op_index] = ZE + ld_ZE * level_rank * op_index;
            ptr_E[op_index] = E + parent_rank * level_rank * node_index;
            ptr_Z[op_index] = Z + parent_rank * parent_rank * parent_level_index;
        }
    }
};

template <class T> struct Hcompress_Stack_Coupling_Functor
{
    T **ptr_ZE, *ZE, **ptr_S, *S, *offdiag_S;
    int *coupling_ptrs, *coupling_indexes, *offdiag_coupling_ptrs, *offdiag_coupling_indexes;
    int *stacked_rows, *stacked_cols;
    int parent_rank, level_rank, ld_ZE, row_start_index, coupling_start, offdiag_coupling_start;
    int total_coupling;
    int *coupling_node_to_leaf, *offdiag_coupling_node_to_leaf;
    int coupling_level_start, offdiag_coupling_level_start;

    Hcompress_Stack_Coupling_Functor(T **ptr_ZE, T *ZE, T **ptr_S, T *S, T *offdiag_S, int *coupling_ptrs,
                                     int *coupling_indexes, int *offdiag_coupling_ptrs, int *offdiag_coupling_indexes,
                                     int *coupling_node_to_leaf, int coupling_level_start,
                                     int *offdiag_coupling_node_to_leaf, int offdiag_coupling_level_start,
                                     int *stacked_rows, int *stacked_cols, int parent_rank, int level_rank, int ld_ZE,
                                     int row_start_index, int coupling_start, int total_coupling,
                                     int offdiag_coupling_start)
    {
        this->ptr_ZE = ptr_ZE;
        this->ptr_S = ptr_S;
        this->ZE = ZE;
        this->S = S;
        this->offdiag_S = offdiag_S;

        this->coupling_ptrs = coupling_ptrs;
        this->coupling_indexes = coupling_indexes;
        this->offdiag_coupling_ptrs = offdiag_coupling_ptrs;
        this->offdiag_coupling_indexes = offdiag_coupling_indexes;
        this->coupling_node_to_leaf = coupling_node_to_leaf;
        this->coupling_level_start = coupling_level_start;
        this->offdiag_coupling_node_to_leaf = offdiag_coupling_node_to_leaf;
        this->offdiag_coupling_level_start = offdiag_coupling_level_start;

        this->stacked_rows = stacked_rows;
        this->stacked_cols = stacked_cols;
        this->parent_rank = parent_rank;
        this->level_rank = level_rank;
        this->ld_ZE = ld_ZE;
        this->row_start_index = row_start_index;
        this->coupling_start = coupling_start;
        this->total_coupling = total_coupling;
        this->offdiag_coupling_start = offdiag_coupling_start;
    }

    __host__ __device__ void operator()(const unsigned int &op_index) const
    {
        // Offset the node by the weighted parent contribution
        T *ZE_node = ZE + op_index * ld_ZE * level_rank + parent_rank;

        int row_index = op_index + row_start_index;
        int b_start = coupling_ptrs[row_index];
        int b_end = coupling_ptrs[row_index + 1];

        int offdiag_b_start = 0, offdiag_b_end = 0;
        if (offdiag_S)
        {
            offdiag_b_start = offdiag_coupling_ptrs[row_index];
            offdiag_b_end = offdiag_coupling_ptrs[row_index + 1];
        }
        int total_blocks = (b_end - b_start) + (offdiag_b_end - offdiag_b_start);

        stacked_rows[op_index] = parent_rank + total_blocks * level_rank;
        stacked_cols[op_index] = level_rank;

        for (int block = b_start; block < b_end; block++)
        {
            int coupling_node_index = coupling_indexes[block];
            int leaf_index = coupling_node_to_leaf[coupling_node_index];

            ptr_ZE[block - coupling_start] = ZE_node + (block - b_start) * level_rank;
            ptr_S[block - coupling_start] = S + (leaf_index - coupling_level_start) * level_rank * level_rank;
        }

        int block_offset = b_end - b_start;
        int ptr_offset = total_coupling - offdiag_coupling_start;

        for (int block = offdiag_b_start; block < offdiag_b_end; block++)
        {
            int coupling_node_index = offdiag_coupling_indexes[block];
            int leaf_index = offdiag_coupling_node_to_leaf[coupling_node_index];

            ptr_ZE[ptr_offset + block] = ZE_node + (block_offset + block - offdiag_b_start) * level_rank;
            ptr_S[ptr_offset + block] =
                offdiag_S + (leaf_index - offdiag_coupling_level_start) * level_rank * level_rank;
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Driver routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw>
inline void hcompress_parent_weight_batch_marshal(T **ptr_ZE, T *ZE, T **ptr_Z, T *Z, T **ptr_E, T *E, int *parent,
                                                  int parent_start, int level_start, int level_offset, int ld_ZE,
                                                  int parent_rank, int level_rank, int batch_size,
                                                  h2opusComputeStream_t stream)
{
    Hcompress_Parent_Weight_Functor<T> parent_weight_functor(ptr_ZE, ZE, ptr_Z, Z, ptr_E, E, parent, parent_start,
                                                             level_start, level_offset, ld_ZE, parent_rank, level_rank);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(batch_size), parent_weight_functor);
}

template <class T, int hw>
inline void hcompress_stack_coupling_data_batch_marshal(
    T **ptr_ZE, T *ZE, T **ptr_S, T *S, int *coupling_ptrs, int *coupling_indexes, T *offdiag_S,
    int *offdiag_coupling_ptrs, int *offdiag_coupling_indexes, int *coupling_node_to_leaf, int coupling_level_start,
    int *offdiag_coupling_node_to_leaf, int offdiag_coupling_level_start, int *stacked_rows, int *stacked_cols,
    int parent_rank, int level_rank, int ld_ZE, int row_start_index, int coupling_start, int coupling_end,
    int offdiag_coupling_start, int offdiag_coupling_end, int batch_size, h2opusComputeStream_t stream)
{
    Hcompress_Stack_Coupling_Functor<T> stack_coupling_functor(
        ptr_ZE, ZE, ptr_S, S, offdiag_S, coupling_ptrs, coupling_indexes, offdiag_coupling_ptrs,
        offdiag_coupling_indexes, coupling_node_to_leaf, coupling_level_start, offdiag_coupling_node_to_leaf,
        offdiag_coupling_level_start, stacked_rows, stacked_cols, parent_rank, level_rank, ld_ZE, row_start_index,
        coupling_start, coupling_end - coupling_start, offdiag_coupling_start);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(batch_size), stack_coupling_functor);
}
