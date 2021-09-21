#ifndef __H2OPUS_GPU_ERR_CHECK_H__
#define __H2OPUS_GPU_ERR_CHECK_H__

#include <stdio.h>

#ifdef H2OPUS_USE_GPU

#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <kblas.h>
#include <stdio.h>

#define gpuErrchk(ans)                                                                                                 \
    {                                                                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);                                                                          \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line)
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

inline const char *h2opus_cublasGetErrorString(cublasStatus_t error)
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

inline const char *h2opus_cusparseGetErrorString(cusparseStatus_t error)
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
inline void gpuCublasAssert(cublasStatus_t code, const char *file, int line)
{
    if (code != CUBLAS_STATUS_SUCCESS)
        printf("cuBlasAssert: %s %s %d\n", h2opus_cublasGetErrorString(code), file, line);
}

#define gpuCusparseErrchk(ans)                                                                                         \
    {                                                                                                                  \
        gpuCusparseAssert((ans), __FILE__, __LINE__);                                                                  \
    }
inline void gpuCusparseAssert(cusparseStatus_t code, const char *file, int line)
{
    if (code != CUSPARSE_STATUS_SUCCESS)
        printf("cuSparseAssert: %s %s %d\n", h2opus_cusparseGetErrorString(code), file, line);
}

#define gpuCusolverErrchk(ans)                                                                                         \
    {                                                                                                                  \
        gpuCusolverAssert((ans), __FILE__, __LINE__);                                                                  \
    }
inline void gpuCusolverAssert(cusolverStatus_t code, const char *file, int line)
{
    if (code != CUSOLVER_STATUS_SUCCESS)
        printf("cuSolverAssert: %d %s %d\n", code, file, line);
}

inline void checkDriverError(CUresult result, const char *file, unsigned line)
{
    if (result != CUDA_SUCCESS)
    {
        const char *error_string = NULL;
        cuGetErrorString(result, &error_string);
        printf("Driver error %d at line %d of file %s: %s\n", (int)result, line, file, error_string);
        exit(-1);
    }
}

#define gpuDriverErrchk(x) checkDriverError(x, __FILE__, __LINE__);

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
