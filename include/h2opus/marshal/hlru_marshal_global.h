#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/h2opus_handle.h>
#include <h2opus/core/thrust_runtime.h>

#include <thrust/for_each.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Thrust functors
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct HLRU_Global_Offset_Pointer_Array
{
    T **ptrs;
    int ld, offset_rows, offset_cols;

    HLRU_Global_Offset_Pointer_Array(T **ptrs, int ld, int rows, int cols)
    {
        this->ptrs = ptrs;
        this->ld = ld;
        this->offset_rows = rows;
        this->offset_cols = cols;
    }

    __host__ __device__ void operator()(int index)
    {
        ptrs[index] += offset_rows + offset_cols * ld;
    }
};

template <class T> struct HLRU_Global_BasisLeaf_Functor
{
    T** original_basis_ptrs;
    T*  update_base_ptr;
    T** updated_basis_ptrs;
    T** update_ptrs;
    int *update_rows, *update_cols;
    int *node_start, *node_size;
    int leaf_start, leaf_ld, leaf_rank, update_rank;

    HLRU_Global_BasisLeaf_Functor(T **original_basis_ptrs, const T *update_base_ptr, T **updated_basis_ptrs, T **update_ptrs,
                                  int *update_rows, int *update_cols, int *node_start, int *node_size, int leaf_start,
                                  int leaf_ld, int leaf_rank, int update_rank)
    {
        this->original_basis_ptrs = original_basis_ptrs;
        this->update_base_ptr = (T*)update_base_ptr;

        this->updated_basis_ptrs = updated_basis_ptrs;
        this->update_ptrs = update_ptrs;
        this->update_rows = update_rows;
        this->update_cols = update_cols;

        this->node_start = node_start;
        this->node_size = node_size;

        this->leaf_start = leaf_start;
        this->leaf_ld = leaf_ld;
        this->leaf_rank = leaf_rank;
        this->update_rank = update_rank;
    }

    __host__ __device__ void operator()(int leaf_index)
    {
        int tree_index = leaf_index + leaf_start;
        int update_offset = node_start[tree_index];
        int node_rows = node_size[tree_index];

        update_rows[leaf_index] = node_rows;
        update_cols[leaf_index] = update_rank;

        update_ptrs[leaf_index] = update_base_ptr + update_offset;
        updated_basis_ptrs[leaf_index] = original_basis_ptrs[leaf_index] + leaf_rank * leaf_ld;
    }
};

template <class T> struct HLRU_Global_Dense_Update_Functor
{
  private:
    const T *U, *V;
    T **U_ptrs, **V_ptrs;
    int *dense_leaf_tree_index;
    int *node_u_index, *node_v_index;
    int *node_u_len, *node_v_len;
    int *node_u_start, *node_v_start;
    int *rows_array, *cols_array, *ranks_array;
    int *ldm_array, *ldu_array, *ldv_array;
    int ldm, ldu, ldv, rank;

  public:
    HLRU_Global_Dense_Update_Functor(int ldm, const T *U, int ldu, const T *V, int ldv, int rank, T **U_ptrs, T **V_ptrs,
                                     int *rows_array, int *cols_array, int *ranks_array, int *ldm_array, int *ldu_array,
                                     int *ldv_array, int *dense_leaf_tree_index, int *node_u_index, int *node_v_index,
                                     int *node_u_start, int *node_v_start, int *node_u_len, int *node_v_len)
    {
        this->U = U;
        this->ldu = ldu;
        this->V = V;
        this->ldv = ldv;
        this->rank = rank;
        this->ldm = ldm;

        this->U_ptrs = U_ptrs;
        this->V_ptrs = V_ptrs;

        this->rows_array = rows_array;
        this->cols_array = cols_array;
        this->ranks_array = ranks_array;

        this->ldm_array = ldm_array;
        this->ldu_array = ldu_array;
        this->ldv_array = ldv_array;

        this->dense_leaf_tree_index = dense_leaf_tree_index;
        this->node_u_index = node_u_index;
        this->node_v_index = node_v_index;

        this->node_u_start = node_u_start;
        this->node_v_start = node_v_start;

        this->node_u_len = node_u_len;
        this->node_v_len = node_v_len;
    }

    __host__ __device__ void operator()(const unsigned int &leaf_index) const
    {
        int tree_index = dense_leaf_tree_index[leaf_index];
        int u_index = node_u_index[tree_index];
        int v_index = node_v_index[tree_index];

        U_ptrs[leaf_index] = (T*)(U + node_u_start[u_index]);
        V_ptrs[leaf_index] = (T*)(V + node_v_start[v_index]);

        rows_array[leaf_index] = node_u_len[u_index];
        cols_array[leaf_index] = node_v_len[v_index];
        ranks_array[leaf_index] = rank;

        ldm_array[leaf_index] = ldm;
        ldu_array[leaf_index] = ldu;
        ldv_array[leaf_index] = ldv;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Global Update drivers
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Advance pointers for coupling matrices to allow adding the identity block
template <class T, int hw>
void hlru_offset_pointer_array(T **ptrs, int ld, int rows, int cols, int num_ptrs, h2opusComputeStream_t stream)
{
    HLRU_Global_Offset_Pointer_Array<T> offset_pointers(ptrs, ld, rows, cols);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_ptrs), offset_pointers);
}

template <class T, int hw>
void hlru_global_basis_leaf_marshal_batch(T **original_basis_ptrs, const T *update_base_ptr, T **updated_basis_ptrs,
                                          T **update_ptrs, int *update_rows, int *update_cols, int *node_start,
                                          int *node_size, int leaf_start, int leaf_ld, int leaf_rank, int update_rank,
                                          int num_leaves, h2opusComputeStream_t stream)
{
    HLRU_Global_BasisLeaf_Functor<T> basis_leaf_functor(original_basis_ptrs, update_base_ptr, updated_basis_ptrs,
                                                        update_ptrs, update_rows, update_cols, node_start, node_size,
                                                        leaf_start, leaf_ld, leaf_rank, update_rank);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_leaves), basis_leaf_functor);
}

template <class T, int hw>
void hlru_dense_update_global_marshal_batch(int ldm, const T *U, int ldu, const T *V, int ldv, int rank, T **U_ptrs, T **V_ptrs,
                                            int *rows_array, int *cols_array, int *ranks_array, int *ldm_array,
                                            int *ldu_array, int *ldv_array, int *dense_leaf_tree_index,
                                            int *node_u_index, int *node_v_index, int *node_u_start, int *node_v_start,
                                            int *node_u_len, int *node_v_len, int num_leaves,
                                            h2opusComputeStream_t stream)
{
    HLRU_Global_Dense_Update_Functor<T> dense_upadte_functor(
        ldm, U, ldu, V, ldv, rank, U_ptrs, V_ptrs, rows_array, cols_array, ranks_array, ldm_array, ldu_array, ldv_array,
        dense_leaf_tree_index, node_u_index, node_v_index, node_u_start, node_v_start, node_u_len, node_v_len);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_leaves), dense_upadte_functor);
}
