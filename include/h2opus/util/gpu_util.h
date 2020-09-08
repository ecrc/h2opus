#ifndef __GPU_UTIL_H__
#define __GPU_UTIL_H__

#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/util/gpu_err_check.h>

#define WARP_SIZE 32
#define MAX_OPS_PER_BATCH 65535

__device__ __host__ inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

inline __host__ __device__ int getOperationDim(int *dim_array, int op_id)
{
    return dim_array[op_id];
}
inline __host__ __device__ int getOperationDim(int dim, int op_id)
{
    return dim;
}
template <class T> inline __host__ __device__ T *getOperationPtr(T *array, int op_id, int stride)
{
    return array + op_id * stride;
}
template <class T> inline __host__ __device__ T *getOperationPtr(T **array, int op_id, int stride)
{
    return array[op_id];
}

inline __host__ __device__ int *advanceOperationDim(int *dim_array, int op_id)
{
    return dim_array + op_id;
}
inline __host__ __device__ int advanceOperationDim(int dim, int op_id)
{
    return dim;
}
template <class T> inline __host__ __device__ T **advanceOperationPtr(T **array, int op_id, int stride)
{
    return array + op_id;
}
template <class T> inline __host__ __device__ T *advanceOperationPtr(T *array, int op_id, int stride)
{
    return array + op_id * stride;
}

template <class T>
__inline__ __device__ T blockAllReduceSum(T val, int warp_id, int warp_tid, T *temp_storage, int blocksize)
{
    const int warps = blocksize / WARP_SIZE;

// First do a reduction within each warp
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);

    if (warps > 1)
    {
        if (warp_tid == 0)
            temp_storage[warp_id] = val;
        __syncthreads();

        T final_sum = 0;
#pragma unroll
        for (int i = 0; i < warps; i++)
            final_sum += temp_storage[i];

        // Is this sync necessary? I think so since if we call the routine again
        // while one warp is already on a second reduction, the temp values will
        // be overwritten before another warp has a chance to tally the results
        __syncthreads();
        return final_sum;
    }
    else
        return val;
}

template <class T, int BLOCKSIZE>
__inline__ __device__ T blockAllReduceSum(T val, int warp_tid, int warp_id, volatile T *temp_storage)
{
    const int warps = BLOCKSIZE / WARP_SIZE;

// First do a reduction within each warp
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);

    if (warps > 1)
    {
        if (warp_tid == 0)
            temp_storage[warp_id] = val;
        __syncthreads();

        T final_sum = 0;
#pragma unroll
        for (int i = 0; i < warps; i++)
            final_sum += temp_storage[i];

        // Is this sync necessary? I think so since if we call the routine again
        // while one warp is already on a second reduction, the temp values will
        // be overwritten before another warp has a chance to tally the results
        __syncthreads();
        return final_sum;
    }
    else
        return val;
}

template <class T> __inline__ __device__ T warpReduceSum(T val)
{
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

template <class T> __inline__ __device__ T warpAllReduceSum(T val)
{
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__inline__ int upper_power_of_two(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}
#endif
