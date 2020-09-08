#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/thrust_runtime.h>

#include <thrust/for_each.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Basis Update Functors
//////////////////////////////////////////////////////////////////////////////////////////////////////////
struct HLRU_HNode_FlagInit_Functor
{
    int *hnode_flags, *updated_hnodes;

    HLRU_HNode_FlagInit_Functor(int *hnode_flags, int *updated_hnodes)
    {
        this->hnode_flags = hnode_flags;
        this->updated_hnodes = updated_hnodes;
    }

    __host__ __device__ void operator()(const unsigned int &update_index) const
    {
        int hnode_index = updated_hnodes[update_index];
        hnode_flags[hnode_index] = update_index;
    }
};

struct HLRU_HNode_Downsweep_Functor
{
    int *hnode_flags, *parent;

    HLRU_HNode_Downsweep_Functor(int *hnode_flags, int *parent)
    {
        this->hnode_flags = hnode_flags;
        this->parent = parent;
    }

    __host__ __device__ void operator()(const unsigned int &hnode_index) const
    {
        int parent_hnode_index = parent[hnode_index];
        int hnode_flag = (parent_hnode_index != H2OPUS_EMPTY_NODE ? hnode_flags[parent_hnode_index] : -1);
        hnode_flags[hnode_index] = hnode_flag;
    }
};

template <class T> struct HLRU_Coupling_Flag_Functor
{
  private:
    T **coupling_matrices, **flagged_ptrs;
    int *hnode_flags, *leaf_to_node;
    int old_rank, new_rank, node_offset;

  public:
    HLRU_Coupling_Flag_Functor(T **coupling_matrices, T **flagged_ptrs, int *hnode_flags, int *leaf_to_node,
                               int old_rank, int new_rank, int node_offset)
    {
        this->coupling_matrices = coupling_matrices;
        this->flagged_ptrs = flagged_ptrs;
        this->hnode_flags = hnode_flags;
        this->leaf_to_node = leaf_to_node;
        this->node_offset = node_offset;
        this->old_rank = old_rank;
        this->new_rank = new_rank;
    }

    __host__ __device__ void operator()(const unsigned int &leaf_index) const
    {
        int hnode_index = leaf_to_node[node_offset + leaf_index];
        int update_index = hnode_flags[hnode_index];

        T *cptr = (update_index == -1 ? NULL : coupling_matrices[leaf_index] + old_rank + old_rank * new_rank);

        flagged_ptrs[leaf_index] = cptr;
    }
};

template <class T> struct HLRU_Dense_Update_Functor
{
  private:
    T **M, **U, **V;
    T **flagged_m_ptrs, **flagged_u_ptrs, **flagged_v_ptrs;
    int *dense_leaf_tree_index, *hnode_update_index;
    int *node_u_index, *node_v_index, *basis_update_row, *basis_update_col;
    int *node_u_len, *node_v_len;
    int *rows_array, *cols_array, *ranks_array;
    int *ldm_array, *ldu_array, *ldv_array;
    int ldm, ldu, ldv, rank;

  public:
    HLRU_Dense_Update_Functor(T **M, int ldm, T **U, int ldu, T **V, int ldv, int rank, T **flagged_m_ptrs,
                              T **flagged_u_ptrs, T **flagged_v_ptrs, int *rows_array, int *cols_array,
                              int *ranks_array, int *ldm_array, int *ldu_array, int *ldv_array,
                              int *dense_leaf_tree_index, int *node_u_index, int *node_v_index, int *node_u_len,
                              int *node_v_len, int *basis_update_row, int *basis_update_col, int *hnode_update_index)
    {
        this->M = M;
        this->ldm = ldm;
        this->U = U;
        this->ldu = ldu;
        this->V = V;
        this->ldv = ldv;
        this->rank = rank;

        this->flagged_m_ptrs = flagged_m_ptrs;
        this->flagged_u_ptrs = flagged_u_ptrs;
        this->flagged_v_ptrs = flagged_v_ptrs;

        this->rows_array = rows_array;
        this->cols_array = cols_array;
        this->ranks_array = ranks_array;

        this->ldm_array = ldm_array;
        this->ldu_array = ldu_array;
        this->ldv_array = ldv_array;

        this->dense_leaf_tree_index = dense_leaf_tree_index;
        this->node_u_index = node_u_index;
        this->node_v_index = node_v_index;

        this->node_u_len = node_u_len;
        this->node_v_len = node_v_len;

        this->basis_update_row = basis_update_row;
        this->basis_update_col = basis_update_col;
        this->hnode_update_index = hnode_update_index;
    }

    __host__ __device__ void operator()(const unsigned int &leaf_index) const
    {
        int tree_index = dense_leaf_tree_index[leaf_index];
        int update_index = hnode_update_index[tree_index];

        T *m_ptr = NULL, *u_ptr = NULL, *v_ptr = NULL;
        int rows = 0, cols = 0;

        if (update_index != -1)
        {
            int u_index = node_u_index[tree_index];
            int v_index = node_v_index[tree_index];
            int row_start = basis_update_row[u_index];
            int col_start = basis_update_col[v_index];

            rows = node_u_len[u_index];
            cols = node_v_len[v_index];

            m_ptr = M[leaf_index];
            u_ptr = U[update_index] + row_start;
            v_ptr = V[update_index] + col_start;
        }

        rows_array[leaf_index] = rows;
        cols_array[leaf_index] = cols;
        ranks_array[leaf_index] = rank;

        ldm_array[leaf_index] = ldm;
        ldu_array[leaf_index] = ldu;
        ldv_array[leaf_index] = ldv;

        flagged_m_ptrs[leaf_index] = m_ptr;
        flagged_u_ptrs[leaf_index] = u_ptr;
        flagged_v_ptrs[leaf_index] = v_ptr;
    }
};

template <class T> struct HLRU_DenseBlock_Update_Functor
{
    T **dense_ptrs, *dense_mem;
    int *hnode_indexes, *hnode_type, *hnode_to_leaf;
    int *rows_array, *cols_array;
    int *node_u_index, *node_v_index, *node_u_len, *node_v_len;
    int dense_stride;

    HLRU_DenseBlock_Update_Functor(T **dense_ptrs, T *dense_mem, int *rows_array, int *cols_array, int *node_u_index,
                                   int *node_v_index, int *node_u_len, int *node_v_len, int *hnode_indexes,
                                   int *hnode_type, int *hnode_to_leaf, int dense_stride)
    {
        this->dense_ptrs = dense_ptrs;
        this->dense_mem = dense_mem;

        this->rows_array = rows_array;
        this->cols_array = cols_array;

        this->node_u_index = node_u_index;
        this->node_v_index = node_v_index;
        this->node_u_len = node_u_len;
        this->node_v_len = node_v_len;

        this->hnode_indexes = hnode_indexes;
        this->hnode_type = hnode_type;
        this->hnode_to_leaf = hnode_to_leaf;

        this->dense_stride = dense_stride;
    }

    __host__ __device__ void operator()(const unsigned int &update_index) const
    {
        int hnode_index = hnode_indexes[update_index];
        assert(hnode_type[hnode_index] == HMATRIX_DENSE_MATRIX);

        int hnode_dense_index = hnode_to_leaf[hnode_index];
        int u_index = node_u_index[hnode_index];
        int v_index = node_v_index[hnode_index];

        dense_ptrs[update_index] = dense_mem + hnode_dense_index * dense_stride;

        rows_array[update_index] = node_u_len[u_index];
        cols_array[update_index] = node_v_len[v_index];
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Driver routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw>
inline int hlru_flagged_coupling_marshal_batch(T **coupling_matrices, T **flagged_ptrs, int *hnode_flags,
                                               int *leaf_to_node, int old_rank, int new_rank, int node_offset,
                                               int num_nodes, h2opusComputeStream_t stream)
{
    if (num_nodes == 0)
        return 0;

    HLRU_Coupling_Flag_Functor<T> coupling_flag_functor(coupling_matrices, flagged_ptrs, hnode_flags, leaf_to_node,
                                                        old_rank, new_rank, node_offset);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_nodes), coupling_flag_functor);

    return num_nodes;

    // TODO: For some reason remove_if randomly fails to remove all null pointers
    // Remove the NULL pointers
    // T** new_end = thrust::remove_if(ThrustRuntime<hw>::get(stream), flagged_ptrs, flagged_ptrs + num_nodes,
    // null_pred()); return new_end - flagged_ptrs;
}

template <class T, int hw>
inline int hlru_dense_update_marshal_batch(T **M, int ldm, T **U, int ldu, T **V, int ldv, int rank, T **flagged_m_ptrs,
                                           T **flagged_u_ptrs, T **flagged_v_ptrs, int *rows_array, int *cols_array,
                                           int *ranks_array, int *ldm_array, int *ldu_array, int *ldv_array,
                                           int *dense_leaf_tree_index, int *node_u_index, int *node_v_index,
                                           int *node_u_len, int *node_v_len, int *basis_update_row,
                                           int *basis_update_col, int *hnode_update_index, int num_dense_leaves,
                                           h2opusComputeStream_t stream)
{
    // typedef thrust::zip_iterator< thrust::tuple<
    //	T**, T**, T**, int*, int*, int*, int*, int*, int*
    //> > ZipIterator;

    HLRU_Dense_Update_Functor<T> dense_update_functor(
        M, ldm, U, ldu, V, ldv, rank, flagged_m_ptrs, flagged_u_ptrs, flagged_v_ptrs, rows_array, cols_array,
        ranks_array, ldm_array, ldu_array, ldv_array, dense_leaf_tree_index, node_u_index, node_v_index, node_u_len,
        node_v_len, basis_update_row, basis_update_col, hnode_update_index);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_dense_leaves), dense_update_functor);

    return num_dense_leaves;

    // TODO: For some reason remove_if randomly fails to remove all null pointers
    // Remove the NULL pointers
    // ZipIterator zip_start = thrust::make_zip_iterator(thrust::make_tuple(
    // 	flagged_m_ptrs, flagged_u_ptrs, flagged_v_ptrs, rows_array, cols_array, ranks_array, ldm_array, ldu_array,
    // ldv_array
    // ));
    // ZipIterator new_end = thrust::remove_if(ThrustRuntime<hw>::get(stream), zip_start, zip_start + num_dense_leaves,
    // tuple_null_pred()); return new_end - zip_start;
}

template <class T, int hw>
inline void hlru_dense_block_update_marshal_batch(T **dense_ptrs, T *dense_mem, int *rows_array, int *cols_array,
                                                  int *node_u_index, int *node_v_index, int *node_u_len,
                                                  int *node_v_len, int *hnode_indexes, int *hnode_type,
                                                  int *hnode_to_leaf, int dense_stride, int num_nodes,
                                                  h2opusComputeStream_t stream)
{
    HLRU_DenseBlock_Update_Functor<T> dense_block_update_functor(
        dense_ptrs, dense_mem, rows_array, cols_array, node_u_index, node_v_index, node_u_len, node_v_len,
        hnode_indexes, hnode_type, hnode_to_leaf, dense_stride);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_nodes), dense_block_update_functor);
}

template <int hw>
inline void hlru_init_hnode_update(int *hnode_flags, int *updated_hnodes, int num_updates, h2opusComputeStream_t stream)
{
    HLRU_HNode_FlagInit_Functor hnode_flaginit_functor(hnode_flags, updated_hnodes);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_updates), hnode_flaginit_functor);
}

template <int hw>
inline void hlru_downsweep_hnode_update(int *hnode_flags, int *parent, int node_start, int node_end,
                                        h2opusComputeStream_t stream)
{
    HLRU_HNode_Downsweep_Functor hnode_downsweep_functor(hnode_flags, parent);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(node_start),
                     thrust::counting_iterator<int>(node_end), hnode_downsweep_functor);
}
