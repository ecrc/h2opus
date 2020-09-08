#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/thrust_runtime.h>

#include <thrust/for_each.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Basis Update Functors
//////////////////////////////////////////////////////////////////////////////////////////////////////////
struct HLRU_BasisMap_Functor
{
    int *update_index, *update_row, *hnode_basis_index, *updated_hnodes;

    HLRU_BasisMap_Functor(int *update_index, int *update_row, int *hnode_basis_index, int *updated_hnodes)
    {
        this->update_index = update_index;
        this->update_row = update_row;
        this->hnode_basis_index = hnode_basis_index;
        this->updated_hnodes = updated_hnodes;
    }

    __host__ __device__ void operator()(const unsigned int &update_id) const
    {
        int hnode_index = updated_hnodes[update_id];
        int basis_index = hnode_basis_index[hnode_index];

        update_index[basis_index] = update_id;
        update_row[basis_index] = 0;
    }
};

struct HLRU_BasisDownsweep_Functor
{
    int *parent, *node_start, *update_index, *update_row;

    HLRU_BasisDownsweep_Functor(int *parent, int *node_start, int *update_index, int *update_row)
    {
        this->parent = parent;
        this->node_start = node_start;
        this->update_index = update_index;
        this->update_row = update_row;
    }

    __host__ __device__ void operator()(const unsigned int &node_index) const
    {
        int parent_index = parent[node_index];
        if (parent_index != H2OPUS_EMPTY_NODE)
        {
            int parent_offset = node_start[node_index] - node_start[parent_index];
            update_index[node_index] = update_index[parent_index];
            update_row[node_index] = update_row[parent_index] + parent_offset;
        }
        else
        {
            update_index[node_index] = update_row[node_index] = -1;
        }
    }
};

template <class T> struct HLRU_FlaggedBasisLeaf_Functor
{
  private:
    T **new_leaves, **updates;
    T **flagged_leaves, **flagged_updates;
    int *rows_array, *cols_array;
    int *update_index, *update_row, *node_len;
    int leaf_ld, leaf_rank, update_rank, leaf_start;

  public:
    HLRU_FlaggedBasisLeaf_Functor(T **new_leaves, T **updates, T **flagged_leaves, T **flagged_updates, int *rows_array,
                                  int *cols_array, int *update_index, int *update_row, int *node_len, int leaf_ld,
                                  int leaf_rank, int update_rank, int leaf_start)
    {
        this->new_leaves = new_leaves;
        this->updates = updates;
        this->flagged_leaves = flagged_leaves;
        this->flagged_updates = flagged_updates;

        this->rows_array = rows_array;
        this->cols_array = cols_array;
        this->update_index = update_index;
        this->update_row = update_row;
        this->node_len = node_len;

        this->leaf_ld = leaf_ld;
        this->leaf_rank = leaf_rank;
        this->update_rank = update_rank;
        this->leaf_start = leaf_start;
    }

    __host__ __device__ void operator()(const unsigned int &leaf_index) const
    {
        int node_index = leaf_start + leaf_index;
        int update_id = update_index[node_index];

        if (update_id != -1)
        {
            // The offset into the update for this leaf
            int update_row_start = update_row[node_index];

            flagged_leaves[leaf_index] = new_leaves[leaf_index] + leaf_rank * leaf_ld;
            flagged_updates[leaf_index] = updates[update_id] + update_row_start;
            rows_array[leaf_index] = node_len[node_index];
            cols_array[leaf_index] = update_rank;
        }
        else
        {
            flagged_leaves[leaf_index] = NULL;
            flagged_updates[leaf_index] = NULL;
        }
    }
};

template <class T> struct HLRU_FlaggedTransfer_Functor
{
  private:
    T **transfer_matrices, **flagged_transfer;
    int *update_index;
    int node_offset, matrix_offset;

  public:
    HLRU_FlaggedTransfer_Functor(T **transfer_matrices, T **flagged_transfer, int *update_index, int node_offset,
                                 int matrix_offset)
    {
        this->transfer_matrices = transfer_matrices;
        this->flagged_transfer = flagged_transfer;
        this->update_index = update_index;
        this->node_offset = node_offset;
        this->matrix_offset = matrix_offset;
    }

    __host__ __device__ void operator()(const unsigned int &node_index) const
    {
        int update_id = update_index[node_offset + node_index];
        T *flagged_ptr = NULL;
        if (update_id != -1)
            flagged_ptr = transfer_matrices[node_index] + matrix_offset;
        flagged_transfer[node_index] = flagged_ptr;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Driver routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw>
inline int hlru_flagged_basis_marshal_batch(T **new_leaves, T **updates, T **flagged_leaves, T **flagged_updates,
                                            int *rows_array, int *cols_array, int *update_index, int *update_row,
                                            int *node_len, int leaf_ld, int leaf_rank, int update_rank, int leaf_start,
                                            int num_leaves, h2opusComputeStream_t stream)
{
    if (num_leaves == 0)
        return 0;

    // typedef thrust::zip_iterator< thrust::tuple<T**, T**, int*, int* > > ZipIterator;

    HLRU_FlaggedBasisLeaf_Functor<T> flagged_basis_leaf_functor(new_leaves, updates, flagged_leaves, flagged_updates,
                                                                rows_array, cols_array, update_index, update_row,
                                                                node_len, leaf_ld, leaf_rank, update_rank, leaf_start);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_leaves), flagged_basis_leaf_functor);

    return num_leaves;

    // TODO: For some reason remove_if randomly fails to remove all null pointers
    // Remove the NULL pointers
    // ZipIterator zip_start = thrust::make_zip_iterator(thrust::make_tuple(flagged_leaves, flagged_updates, rows_array,
    // cols_array)); ZipIterator new_end = thrust::remove_if(ThrustRuntime<hw>::get(stream), zip_start, zip_start +
    // num_leaves, tuple_null_pred()); return new_end - zip_start;
}

template <class T, int hw>
inline int hlru_flagged_transfer_marshal_batch(T **transfer_matrices, T **flagged_transfer, int *update_index,
                                               int node_offset, int matrix_offset, int num_nodes,
                                               h2opusComputeStream_t stream)
{
    if (num_nodes == 0)
        return 0;

    HLRU_FlaggedTransfer_Functor<T> flagged_transfer_functor(transfer_matrices, flagged_transfer, update_index,
                                                             node_offset, matrix_offset);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_nodes), flagged_transfer_functor);

    return num_nodes;

    // TODO: For some reason remove_if randomly fails to remove all null pointers
    // Remove the NULL pointers
    // T** new_end = thrust::remove_if(ThrustRuntime<hw>::get(stream), flagged_transfer, flagged_transfer + num_nodes,
    // null_pred()); return new_end - flagged_transfer;
}

template <int hw>
inline void hlru_init_basis_update(int *update_index, int *update_row, int *hnode_basis_index, int *updated_hnodes,
                                   int num_updates, h2opusComputeStream_t stream)
{
    HLRU_BasisMap_Functor basis_map_functor(update_index, update_row, hnode_basis_index, updated_hnodes);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_updates), basis_map_functor);
}

template <int hw>
inline void hlru_downsweep_basis_update(int *parent, int *node_start, int *update_index, int *update_row,
                                        int level_start, int level_size, h2opusComputeStream_t stream)
{
    HLRU_BasisDownsweep_Functor basis_downsweep_functor(parent, node_start, update_index, update_row);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(level_start),
                     thrust::counting_iterator<int>(level_start + level_size), basis_downsweep_functor);
}
