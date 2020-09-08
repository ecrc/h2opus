#ifndef __H2OPUS_GPU_ERR_CHECK_H__
#define __H2OPUS_GPU_ERR_CHECK_H__

#include <stdio.h>

#ifdef H2OPUS_USE_GPU

#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <kblas.h>
#include <stdio.h>

#define gpuErrchk(ans)                                                                                                 \
    {                                                                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);                                                                          \
    }
__device__ __host__ inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
        printf("GPUassert: %s(%d) %s %d\n", cudaGetErrorString(code), (int)code, file, line);
}

#define check_kblas_error(ans)                                                                                         \
    {                                                                                                                  \
        gpuKblasAssert((ans), __FILE__, __LINE__);                                                                     \
    }
void gpuKblasAssert(int code, const char *file, int line);
inline void gpuKblasAssert(int code, const char *file, int line)
{
    if (code != 1)
    {
        printf("gpuKblasAssert: %s %s %d\n", kblasGetErrorString(code), file, line);
        // exit(-1);
    }
}

__device__ __host__ inline const char *h2opus_cublasGetErrorString(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "out of memory";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "architecture mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "memory mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "internal error";
    default:
        return "unknown error code";
    }
}

__device__ __host__ inline const char *h2opus_cusparseGetErrorString(cusparseStatus_t error)
{
    switch (error)
    {
    case CUSPARSE_STATUS_SUCCESS:
        return "success";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "not initialized";
    case CUSPARSE_STATUS_ALLOC_FAILED:
        return "out of memory";
    case CUSPARSE_STATUS_INVALID_VALUE:
        return "invalid value";
    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "architecture mismatch";
    case CUSPARSE_STATUS_MAPPING_ERROR:
        return "memory mapping error";
    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "execution failed";
    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "internal error";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "matrix type not supported";
    default:
        return "unknown error code";
    }
}

#define gpuCublasErrchk(ans)                                                                                           \
    {                                                                                                                  \
        gpuCublasAssert((ans), __FILE__, __LINE__);                                                                    \
    }
__device__ __host__ inline void gpuCublasAssert(cublasStatus_t code, const char *file, int line)
{
    if (code != CUBLAS_STATUS_SUCCESS)
        printf("GPUassert: %s %s %d\n", h2opus_cublasGetErrorString(code), file, line);
}

#define gpuCusparseErrchk(ans)                                                                                         \
    {                                                                                                                  \
        gpuCusparseAssert((ans), __FILE__, __LINE__);                                                                  \
    }
__device__ __host__ inline void gpuCusparseAssert(cusparseStatus_t code, const char *file, int line)
{
    if (code != CUSPARSE_STATUS_SUCCESS)
        printf("GPUassert: %s %s %d\n", h2opus_cusparseGetErrorString(code), file, line);
}

#else

#define check_kblas_error(ans)                                                                                         \
    {                                                                                                                  \
        kblasAssert((ans), __FILE__, __LINE__);                                                                        \
    }
void kblasAssert(int code, const char *file, int line);
inline void kblasAssert(int code, const char *file, int line)
{
    if (code != 1)
    {
        printf("kblasAssert: %d %s %d\n", code, file, line);
        // exit(-1);
    }
}
#endif
#endif
