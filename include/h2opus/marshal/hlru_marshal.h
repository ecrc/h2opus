#ifndef __HMATRIX_LRU_GPU_UTIL_CUH__
#define __HMATRIX_LRU_GPU_UTIL_CUH__

#include <h2opus/core/h2opus_handle.h>

//////////////////////////////////////////////////
// Common predicates for removing null operations from
// batches of marshaled data
//////////////////////////////////////////////////
struct tuple_null_pred
{
    template <typename Tuple> __host__ __device__ bool operator()(Tuple t)
    {
        return thrust::get<0>(t) == NULL;
    }
};

struct null_pred
{
    template <typename PointerType> __host__ __device__ bool operator()(PointerType t)
    {
        return t == NULL;
    }
};

//////////////////////////////////////////////////
// Local updates
//////////////////////////////////////////////////

// Basis trees
#include <h2opus/marshal/hlru_basis.cuh>

// Hnode tree
#include <h2opus/marshal/hlru_hnodes.cuh>

#endif
