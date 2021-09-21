#ifndef __BLAS_WRAPPERS_H__
#define __BLAS_WRAPPERS_H__

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/h2opus_compute_stream.h>

#ifdef H2OPUS_PROFILING_ENABLED
#include <h2opus/util/perf_counter.h>
#endif

// n = norm(x, 2)
template <class T, int hw> inline T blas_norm2(h2opusComputeStream_t stream, int n, T *x, int incx);

// dp = x' * y
template <class T, int hw>
inline T blas_dot_product(h2opusComputeStream_t stream, int n, T *x, int incx, T *y, int incy);

// X = alpha * X
template <class T, int hw> inline void blas_scal(h2opusComputeStream_t stream, int n, T alpha, T *x, int incx);

// Y = alpha * X + Y
template <class T, int hw>
inline void blas_axpy(h2opusComputeStream_t stream, int n, T alpha, const T *x, int incx, T *y, int incy);

// GEMM
template <class T, int hw>
inline void blas_gemm(h2opusComputeStream_t stream, char transa, char transb, int m, int n, int k, T alpha, const T *A,
                      int lda, const T *B, int ldb, T beta, T *C, int ldc);

// CHOLESKY
template <class T, int hw> inline int lapack_potrf(h2opusComputeStream_t stream, int n, T *A, int lda, int *info);

// LDL
template <class T, int hw>
inline int lapack_sytrf_nopiv(h2opusComputeStream_t stream, int n, T *A, int lda, T *D, int *info);

// GEQRF
template <class T, int hw> inline int lapack_geqrf(h2opusComputeStream_t stream, int m, int n, T *a, int lda, T *tau);

// ORGQR
template <class T, int hw>
inline int lapack_orgqr(h2opusComputeStream_t stream, int m, int n, int k, T *a, int lda, T *tau);

// TRSM
template <class T, int hw>
inline int blas_trsm(h2opusComputeStream_t stream, char side, char uplo, char trans, char diag, int m, int n, T alpha,
                     T *A, int lda, T *B, int ldb);

template <class T, int hw>
inline int blas_diagLeftInvMult(h2opusComputeStream_t stream, int m, int n, const T *D, T *A, int lda, T *C, int ldc);

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU
///////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU

#include <cublas_v2.h>
#include <cuda_runtime.h>
#ifndef H2OPUS_USE_MAGMA_POTRF
#include <kblas_potrf.h>
#endif

template <>
inline double blas_norm2<double, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int n, double *x, int incx)
{
    double norm2;
    cublasDnrm2(stream->getCublasHandle(), n, x, incx, &norm2);
    return norm2;
}

template <> inline float blas_norm2<float, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int n, float *x, int incx)
{
    float norm2;
    cublasSnrm2(stream->getCublasHandle(), n, x, incx, &norm2);
    return norm2;
}

template <>
inline double blas_dot_product<double, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int n, double *x, int incx,
                                                          double *y, int incy)
{
    double dp;
    cublasDdot(stream->getCublasHandle(), n, x, incx, y, incy, &dp);
    return dp;
}

template <>
inline float blas_dot_product<float, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int n, float *x, int incx,
                                                        float *y, int incy)
{
    float dp;
    cublasSdot(stream->getCublasHandle(), n, x, incx, y, incy, &dp);
    return dp;
}

template <>
inline void blas_scal<double, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int n, double alpha, double *x, int incx)
{
    cublasDscal(stream->getCublasHandle(), n, &alpha, x, incx);
}

template <>
inline void blas_scal<float, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int n, float alpha, float *x, int incx)
{
    cublasSscal(stream->getCublasHandle(), n, &alpha, x, incx);
}

template <>
inline void blas_axpy<double, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int n, double alpha, const double *x,
                                                 int incx, double *y, int incy)
{
    cublasDaxpy(stream->getCublasHandle(), n, &alpha, x, incx, y, incy);
}

template <>
inline void blas_axpy<float, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int n, float alpha, const float *x,
                                                int incx, float *y, int incy)
{
    cublasSaxpy(stream->getCublasHandle(), n, &alpha, x, incx, y, incy);
}

template <>
inline void blas_gemm<float, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, char transa, char transb, int m, int n,
                                                int k, float alpha, const float *A, int lda, const float *B, int ldb,
                                                float beta, float *C, int ldc)
{
    cublasOperation_t cublas_transA = (transa == H2Opus_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);
    cublasOperation_t cublas_transB = (transb == H2Opus_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);

    cublasSgemm(stream->getCublasHandle(), cublas_transA, cublas_transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C,
                ldc);
}

template <>
inline void blas_gemm<double, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, char transa, char transb, int m, int n,
                                                 int k, double alpha, const double *A, int lda, const double *B,
                                                 int ldb, double beta, double *C, int ldc)
{
    cublasOperation_t cublas_transA = (transa == H2Opus_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);
    cublasOperation_t cublas_transB = (transb == H2Opus_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);

    cublasDgemm(stream->getCublasHandle(), cublas_transA, cublas_transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C,
                ldc);
}

template <>
inline int lapack_potrf<float, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int n, float *A, int lda, int *info)
{
#ifndef H2OPUS_USE_MAGMA_POTRF
    kblas_potrf(stream->getKblasHandle(), KBLAS_Lower, n, A, lda, NULL);
    *info = 0;
#else
    magma_spotrf_native(MagmaLower, n, A, lda, info);
#endif
    return 1;
}

template <>
inline int lapack_potrf<double, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int n, double *A, int lda, int *info)
{
#ifndef H2OPUS_USE_MAGMA_POTRF
    kblas_potrf(stream->getKblasHandle(), KBLAS_Lower, n, A, lda, NULL);
    *info = 0;
#else
    magma_dpotrf_native(MagmaLower, n, A, lda, info);
#endif
    return 1;
}

// LDL
template <>
inline int lapack_sytrf_nopiv<float, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int n, float *A, int lda,
                                                        float *D, int *info)
{
#ifdef H2OPUS_PROFILING_ENABLED
    double operation_gops = (double)(H2OPUS_POTRF_OP_COUNT(n)) * 1e-9;
    PerformanceCounter::addOpCount(PerformanceCounter::POTRF, operation_gops);
#endif

    assert(false && "GPU lapack_sytrf_nopiv wrapper not implemented");
    return 1;
}

template <>
inline int lapack_sytrf_nopiv<double, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int n, double *A, int lda,
                                                         double *D, int *info)
{
#ifdef H2OPUS_PROFILING_ENABLED
    double operation_gops = (double)(H2OPUS_POTRF_OP_COUNT(n)) * 1e-9;
    PerformanceCounter::addOpCount(PerformanceCounter::POTRF, operation_gops);
#endif

    assert(false && "GPU lapack_sytrf_nopiv wrapper not implemented");
    return 1;
}

template <>
inline int blas_trsm<float, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, char side, char uplo, char trans,
                                               char diag, int m, int n, float alpha, float *A, int lda, float *B,
                                               int ldb)

{
    cublasSideMode_t cublas_side = (side == H2Opus_Left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
    cublasFillMode_t cublas_uplo = (uplo == H2Opus_Upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER);
    cublasOperation_t cublas_transA = (trans == H2Opus_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);
    cublasDiagType_t cublas_diag = (diag == H2Opus_Unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT);

    gpuCublasErrchk(kblasStrsm(stream->getCublasHandle(), cublas_side, cublas_uplo, cublas_transA, cublas_diag, m, n,
                               &alpha, A, lda, B, ldb));
    return 1;
}

template <>
inline int blas_trsm<double, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, char side, char uplo, char trans,
                                                char diag, int m, int n, double alpha, double *A, int lda, double *B,
                                                int ldb)

{
    cublasSideMode_t cublas_side = (side == H2Opus_Left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
    cublasFillMode_t cublas_uplo = (uplo == H2Opus_Upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER);
    cublasOperation_t cublas_transA = (trans == H2Opus_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);
    cublasDiagType_t cublas_diag = (diag == H2Opus_Unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT);

    gpuCublasErrchk(kblasDtrsm(stream->getCublasHandle(), cublas_side, cublas_uplo, cublas_transA, cublas_diag, m, n,
                               &alpha, A, lda, B, ldb));
    return 1;
}

template <>
inline int blas_diagLeftInvMult<float, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int m, int n, const float *D,
                                                          float *A, int lda, float *C, int ldc)
{
    assert(false && "GPU blas_diagLeftInvMult kernel not implemented");
    return 1;
}

template <>
inline int blas_diagLeftInvMult<double, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int m, int n, const double *D,
                                                           double *A, int lda, double *C, int ldc)
{
    assert(false && "GPU blas_diagLeftInvMult kernel not implemented");
    return 1;
}

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU
///////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef H2OPUS_BATCH_BLAS_THREADS
#define H2OPUS_BATCH_BLAS_THREADS 1
#endif

#ifdef H2OPUS_USE_LIBXSMM
#include <libxsmm.h>
#endif

#ifdef H2OPUS_USE_BLIS
typedef int32_t dim_t;
extern "C" dim_t bli_thread_get_num_threads(void);
extern "C" void bli_thread_set_ways(dim_t, dim_t, dim_t, dim_t, dim_t);
extern "C" void bli_thread_set_num_threads(dim_t);
#endif

#ifdef H2OPUS_USE_MKL
#ifndef MKL_INT
#define MKL_INT int
#endif
#include <mkl_service.h>
#include <mkl_trans.h>
#include <mkl_vsl.h>
#endif

#ifdef H2OPUS_USE_ESSL
#include <h2opus/util/esslrngwrap.h>
#endif

#ifdef H2OPUS_USE_NEC
#include <asl.h>
#endif

#ifdef H2OPUS_USE_AMDRNG
#include <h2opus/util/amdrngwrap.h>
#endif

#include <h2opus/util/h2opusfblaslapack.h>

// Use specific number of threads when calling blas in batched mode
// cannot be nested, would need push/pop
#ifdef H2OPUS_USE_MKL
#define H2OPUS_BEGIN_BATCH_BLAS()                                                                                      \
    {                                                                                                                  \
        int _old_threads = mkl_domain_get_max_threads(MKL_DOMAIN_BLAS);                                                \
        mkl_domain_set_num_threads(H2OPUS_BATCH_BLAS_THREADS, MKL_DOMAIN_BLAS);
#define H2OPUS_END_BATCH_BLAS()                                                                                        \
    mkl_domain_set_num_threads(_old_threads, MKL_DOMAIN_BLAS);                                                         \
    }
#elif defined(H2OPUS_USE_BLIS)
#define H2OPUS_BEGIN_BATCH_BLAS()                                                                                      \
    {                                                                                                                  \
        dim_t _old_threads = bli_thread_get_num_threads();                                                             \
        bli_thread_set_num_threads(H2OPUS_BATCH_BLAS_THREADS);
#define H2OPUS_END_BATCH_BLAS()                                                                                        \
    bli_thread_set_num_threads(_old_threads);                                                                          \
    }
#else
// just to make sure matching calls are issued
#define H2OPUS_BEGIN_BATCH_BLAS() {
#define H2OPUS_END_BATCH_BLAS() }
#endif

// Dense BLAS used for single execution
template <>
inline double blas_norm2<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, double *x, int incx)
{
    return h2opus_fbl_dnrm2(n, x, incx);
}

template <> inline float blas_norm2<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, float *x, int incx)
{
    return h2opus_fbl_snrm2(n, x, incx);
}

template <>
inline double blas_dot_product<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, double *x, int incx,
                                                          double *y, int incy)
{
    return h2opus_fbl_ddot(n, x, incx, y, incy);
}

template <>
inline float blas_dot_product<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, float *x, int incx,
                                                        float *y, int incy)
{
    return h2opus_fbl_sdot(n, x, incx, y, incy);
}

template <>
inline void blas_scal<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, double alpha, double *x, int incx)
{
    h2opus_fbl_dscal(n, alpha, x, incx);
}

template <>
inline void blas_scal<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, float alpha, float *x, int incx)
{
    h2opus_fbl_sscal(n, alpha, x, incx);
}

template <>
inline void blas_axpy<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, double alpha, const double *x,
                                                 int incx, double *y, int incy)
{
    h2opus_fbl_daxpy(n, alpha, x, incx, y, incy);
}

template <>
inline void blas_axpy<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, float alpha, const float *x,
                                                int incx, float *y, int incy)
{
    h2opus_fbl_saxpy(n, alpha, x, incx, y, incy);
}

template <>
inline void blas_gemm<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, char transa, char transb, int m, int n,
                                                int k, float alpha, const float *A, int lda, const float *B, int ldb,
                                                float beta, float *C, int ldc)
{
#ifdef H2OPUS_PROFILING_ENABLED
    double operation_gops = (double)(H2OPUS_GEMM_OP_COUNT(m, n, k)) * 1e-9;
    PerformanceCounter::addOpCount(PerformanceCounter::GEMM, operation_gops);
#endif

    H2OPUS_FBL_TRANSPOSE h2opus_fbl_transA = (transa == H2Opus_Trans ? H2OpusFBLTrans : H2OpusFBLNoTrans);
    H2OPUS_FBL_TRANSPOSE h2opus_fbl_transB = (transb == H2Opus_Trans ? H2OpusFBLTrans : H2OpusFBLNoTrans);

    h2opus_fbl_sgemm(h2opus_fbl_transA, h2opus_fbl_transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline void blas_gemm<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, char transa, char transb, int m, int n,
                                                 int k, double alpha, const double *A, int lda, const double *B,
                                                 int ldb, double beta, double *C, int ldc)
{
#ifdef H2OPUS_PROFILING_ENABLED
    double operation_gops = (double)(H2OPUS_GEMM_OP_COUNT(m, n, k)) * 1e-9;
    PerformanceCounter::addOpCount(PerformanceCounter::GEMM, operation_gops);
#endif

    H2OPUS_FBL_TRANSPOSE h2opus_fbl_transA = (transa == H2Opus_Trans ? H2OpusFBLTrans : H2OpusFBLNoTrans);
    H2OPUS_FBL_TRANSPOSE h2opus_fbl_transB = (transb == H2Opus_Trans ? H2OpusFBLTrans : H2OpusFBLNoTrans);

    h2opus_fbl_dgemm(h2opus_fbl_transA, h2opus_fbl_transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline int blas_trsm<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, char side, char uplo, char trans,
                                               char diag, int m, int n, float alpha, float *A, int lda, float *B,
                                               int ldb)

{
    H2OPUS_FBL_SIDE h2opus_fbl_side = (side == H2Opus_Left ? H2OpusFBLLeft : H2OpusFBLRight);
    H2OPUS_FBL_UPLO h2opus_fbl_uplo = (uplo == H2Opus_Upper ? H2OpusFBLUpper : H2OpusFBLLower);
    H2OPUS_FBL_TRANSPOSE h2opus_fbl_transA = (trans == H2Opus_Trans ? H2OpusFBLTrans : H2OpusFBLNoTrans);
    H2OPUS_FBL_DIAG h2opus_fbl_diag = (diag == H2Opus_Unit ? H2OpusFBLUnit : H2OpusFBLNonUnit);

    h2opus_fbl_strsm(h2opus_fbl_side, h2opus_fbl_uplo, h2opus_fbl_transA, h2opus_fbl_diag, m, n, alpha, A, lda, B, ldb);

    return 1;
}

template <>
inline int blas_trsm<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, char side, char uplo, char trans,
                                                char diag, int m, int n, double alpha, double *A, int lda, double *B,
                                                int ldb)

{
    H2OPUS_FBL_SIDE h2opus_fbl_side = (side == H2Opus_Left ? H2OpusFBLLeft : H2OpusFBLRight);
    H2OPUS_FBL_UPLO h2opus_fbl_uplo = (uplo == H2Opus_Upper ? H2OpusFBLUpper : H2OpusFBLLower);
    H2OPUS_FBL_TRANSPOSE h2opus_fbl_transA = (trans == H2Opus_Trans ? H2OpusFBLTrans : H2OpusFBLNoTrans);
    H2OPUS_FBL_DIAG h2opus_fbl_diag = (diag == H2Opus_Unit ? H2OpusFBLUnit : H2OpusFBLNonUnit);

    h2opus_fbl_dtrsm(h2opus_fbl_side, h2opus_fbl_uplo, h2opus_fbl_transA, h2opus_fbl_diag, m, n, alpha, A, lda, B, ldb);

    return 1;
}

template <>
inline int lapack_potrf<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, float *A, int lda, int *info)
{
#ifdef H2OPUS_PROFILING_ENABLED
    double operation_gops = (double)(H2OPUS_POTRF_OP_COUNT(n)) * 1e-9;
    PerformanceCounter::addOpCount(PerformanceCounter::POTRF, operation_gops);
#endif

    *info = h2opus_fbl_spotrf(H2OpusFBLLower, n, A, lda);

    return 1;
}

template <>
inline int lapack_potrf<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, double *A, int lda, int *info)
{
#ifdef H2OPUS_PROFILING_ENABLED
    double operation_gops = (double)(H2OPUS_POTRF_OP_COUNT(n)) * 1e-9;
    PerformanceCounter::addOpCount(PerformanceCounter::POTRF, operation_gops);
#endif

    *info = h2opus_fbl_dpotrf(H2OpusFBLLower, n, A, lda);

    return 1;
}

// LDL
template <>
inline int lapack_sytrf_nopiv<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, float *A, int lda,
                                                        float *D, int *info)
{
#ifdef H2OPUS_PROFILING_ENABLED
    double operation_gops = (double)(H2OPUS_POTRF_OP_COUNT(n)) * 1e-9;
    PerformanceCounter::addOpCount(PerformanceCounter::POTRF, operation_gops);
#endif

#ifndef H2OPUS_USE_GPU
    assert(false && "sytrf needs MAGMA");
#else
    magma_ssytrf_nopiv_cpu(MagmaLower, n, 32, A, lda, info);

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        D[i] = A[i + i * lda];
        A[i + i * lda] = 1;
    }
#endif
    return 1;
}

template <>
inline int lapack_sytrf_nopiv<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, double *A, int lda,
                                                         double *D, int *info)
{
#ifdef H2OPUS_PROFILING_ENABLED
    double operation_gops = (double)(H2OPUS_POTRF_OP_COUNT(n)) * 1e-9;
    PerformanceCounter::addOpCount(PerformanceCounter::POTRF, operation_gops);
#endif

#ifndef H2OPUS_USE_GPU
    assert(false && "sytrf needs MAGMA");
#else
    magma_dsytrf_nopiv_cpu(MagmaLower, n, 32, A, lda, info);

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        D[i] = A[i + i * lda];
        A[i + i * lda] = 1;
    }
#endif
    return 1;
}

// ORGQR
template <>
inline int lapack_orgqr<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int m, int n, int k, double *a,
                                                   int lda, double *tau)
{
    int info = h2opus_fbl_dorgqr(m, n, k, a, lda, tau, stream->getFBLWorkspace());
    return info;
}

template <>
inline int lapack_orgqr<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int m, int n, int k, float *a, int lda,
                                                  float *tau)
{
    int info = h2opus_fbl_sorgqr(m, n, k, a, lda, tau, stream->getFBLWorkspace());
    return info;
}

template <>
inline int lapack_geqrf<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int m, int n, float *a, int lda,
                                                  float *tau)
{
    int info = h2opus_fbl_sgeqrf(m, n, a, lda, tau, stream->getFBLWorkspace());
    return info;
}

template <>
inline int lapack_geqrf<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int m, int n, double *a, int lda,
                                                   double *tau)
{
    int info = h2opus_fbl_dgeqrf(m, n, a, lda, tau, stream->getFBLWorkspace());
    return info;
}

template <>
inline int blas_diagLeftInvMult<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int m, int n, const float *D,
                                                          float *A, int lda, float *C, int ldc)
{
    for (int j = 0; j < n; j++)
    {
#pragma omp parallel for
        for (int i = 0; i < m; i++)
            C[i] = A[i] / D[i];

        C += ldc;
        A += lda;
    }
    return 1;
}

template <>
inline int blas_diagLeftInvMult<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int m, int n, const double *D,
                                                           double *A, int lda, double *C, int ldc)
{
    for (int j = 0; j < n; j++)
    {
#pragma omp parallel for
        for (int i = 0; i < m; i++)
            C[i] = A[i] / D[i];

        C += ldc;
        A += lda;
    }
    return 1;
}

// Dense BLAS used in the batch execution
inline void h2opus_fbl_gemm(enum H2OPUS_FBL_TRANSPOSE transa, enum H2OPUS_FBL_TRANSPOSE transb, const int m,
                            const int n, const int k, const float alpha, const float *a, const int lda, const float *b,
                            const int ldb, const float beta, float *c, const int ldc)
{
    h2opus_fbl_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void h2opus_fbl_gemm(enum H2OPUS_FBL_TRANSPOSE transa, enum H2OPUS_FBL_TRANSPOSE transb, const int m,
                            const int n, const int k, const double alpha, const double *a, const int lda,
                            const double *b, const int ldb, const double beta, double *c, const int ldc)
{
    h2opus_fbl_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void h2opus_fbl_trmm(enum H2OPUS_FBL_SIDE Side, enum H2OPUS_FBL_UPLO Uplo, enum H2OPUS_FBL_TRANSPOSE TransA,
                            enum H2OPUS_FBL_DIAG Diag, const int M, const int N, const float alpha, const float *A,
                            const int lda, float *B, const int ldb)
{
    h2opus_fbl_strmm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

inline void h2opus_fbl_trmm(enum H2OPUS_FBL_SIDE Side, enum H2OPUS_FBL_UPLO Uplo, enum H2OPUS_FBL_TRANSPOSE TransA,
                            enum H2OPUS_FBL_DIAG Diag, const int M, const int N, const double alpha, const double *A,
                            const int lda, double *B, const int ldb)
{
    h2opus_fbl_dtrmm(Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

inline void h2opus_fbl_syrk(enum H2OPUS_FBL_UPLO uplo, enum H2OPUS_FBL_TRANSPOSE trans, const int n, const int k,
                            const float alpha, const float *a, const int lda, const float beta, float *c, const int ldc)
{
    h2opus_fbl_ssyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

inline void h2opus_fbl_syrk(enum H2OPUS_FBL_UPLO uplo, enum H2OPUS_FBL_TRANSPOSE trans, const int n, const int k,
                            const double alpha, const double *a, const int lda, const double beta, double *c,
                            const int ldc)
{
    h2opus_fbl_dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

inline void h2opus_fbl_axpy(int n, float alpha, const float *x, int incx, float *y, int incy)
{
    h2opus_fbl_saxpy(n, alpha, x, incx, y, incy);
}

inline void h2opus_fbl_axpy(int n, double alpha, const double *x, int incx, double *y, int incy)
{
    h2opus_fbl_daxpy(n, alpha, x, incx, y, incy);
}

inline void h2opus_fbl_gemv(enum H2OPUS_FBL_TRANSPOSE trans, const int m, const int n, const float alpha,
                            const float *a, const int lda, const float *x, const float beta, float *y)
{
    h2opus_fbl_sgemv(trans, m, n, alpha, a, lda, x, 1, beta, y, 1);
}

inline void h2opus_fbl_gemv(enum H2OPUS_FBL_TRANSPOSE trans, const int m, const int n, const double alpha,
                            const double *a, const int lda, const double *x, const double beta, double *y)
{
    h2opus_fbl_dgemv(trans, m, n, alpha, a, lda, x, 1, beta, y, 1);
}

inline int h2opus_fbl_gesvd(int m, int n, float *a, int lda, float *s, float *superb, h2opus_fbl_ctx *ctx)
{
    return h2opus_fbl_sgesvd(H2OpusFBLJobO, H2OpusFBLJobN, m, n, a, lda, s, NULL, m, NULL, n, superb, ctx);
}

inline int h2opus_fbl_gesvd(int m, int n, double *a, int lda, double *s, double *superb, h2opus_fbl_ctx *ctx)
{
    return h2opus_fbl_dgesvd(H2OpusFBLJobO, H2OpusFBLJobN, m, n, a, lda, s, NULL, m, NULL, n, superb, ctx);
}

inline int h2opus_fbl_gesvd(int m, int n, float *a, int lda, float *s, float *u, int ldu, float *vt, int ldvt,
                            float *superb, h2opus_fbl_ctx *ctx)
{
    return h2opus_fbl_sgesvd(H2OpusFBLJobS, H2OpusFBLJobS, m, n, a, lda, s, u, ldu, vt, ldvt, superb, ctx);
}

inline int h2opus_fbl_gesvd(int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt,
                            double *superb, h2opus_fbl_ctx *ctx)
{
    return h2opus_fbl_dgesvd(H2OpusFBLJobS, H2OpusFBLJobS, m, n, a, lda, s, u, ldu, vt, ldvt, superb, ctx);
}

inline int h2opus_fbl_potrf(int n, float *A, int lda)
{
    return h2opus_fbl_spotrf(H2OpusFBLLower, n, A, lda);
}

inline int h2opus_fbl_potrf(int n, double *A, int lda)
{
    return h2opus_fbl_dpotrf(H2OpusFBLLower, n, A, lda);
}

inline int h2opus_fbl_geqrf(int m, int n, float *a, int lda, float *tau, h2opus_fbl_ctx *ctx)
{
    return h2opus_fbl_sgeqrf(m, n, a, lda, tau, ctx);
}

inline int h2opus_fbl_geqrf(int m, int n, double *a, int lda, double *tau, h2opus_fbl_ctx *ctx)
{
    return h2opus_fbl_dgeqrf(m, n, a, lda, tau, ctx);
}

inline int h2opus_fbl_orgqr(int m, int n, int k, float *a, int lda, float *tau, h2opus_fbl_ctx *ctx)
{
    return h2opus_fbl_sorgqr(m, n, k, a, lda, tau, ctx);
}

inline int h2opus_fbl_orgqr(int m, int n, int k, double *a, int lda, double *tau, h2opus_fbl_ctx *ctx)
{
    return h2opus_fbl_dorgqr(m, n, k, a, lda, tau, ctx);
}

inline int h2opus_fbl_lacpy(enum H2OPUS_FBL_UPLO uplo, int m, int n, float *a, int lda, float *b, int ldb)
{
    return h2opus_fbl_slacpy(uplo, m, n, a, lda, b, ldb);
}

inline int h2opus_fbl_lacpy(enum H2OPUS_FBL_UPLO uplo, int m, int n, double *a, int lda, double *b, int ldb)
{
    return h2opus_fbl_dlacpy(uplo, m, n, a, lda, b, ldb);
}

inline int h2opus_fbl_geqp3(int m, int n, float *a, int lda, int *jpvt, float *tau, h2opus_fbl_ctx *ctx)
{
    return h2opus_fbl_sgeqp3(m, n, a, lda, jpvt, tau, ctx);
}

inline int h2opus_fbl_geqp3(int m, int n, double *a, int lda, int *jpvt, double *tau, h2opus_fbl_ctx *ctx)
{
    return h2opus_fbl_dgeqp3(m, n, a, lda, jpvt, tau, ctx);
}

inline void h2opus_fbl_trsm(enum H2OPUS_FBL_SIDE side, enum H2OPUS_FBL_UPLO uplo, enum H2OPUS_FBL_TRANSPOSE transa,
                            enum H2OPUS_FBL_DIAG diag, int m, int n, float alpha, const float *a, int lda, float *b,
                            int ldb)
{
    h2opus_fbl_strsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

inline void h2opus_fbl_trsm(enum H2OPUS_FBL_SIDE side, enum H2OPUS_FBL_UPLO uplo, enum H2OPUS_FBL_TRANSPOSE transa,
                            enum H2OPUS_FBL_DIAG diag, int m, int n, double alpha, const double *a, int lda, double *b,
                            int ldb)
{
    h2opus_fbl_dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

inline void blas_mp_syrk(int m, int n, const float *A, int lda, double *B, int ldb)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double dp = 0;
            for (int k = 0; k < m; k++)
                dp += (double)A[k + i * lda] * (double)A[k + j * lda];
            B[i + j * ldb] = dp;
        }
    }
}

inline void blas_mp_syrk(int m, int n, const double *A, int lda, double *B, int ldb)
{
    // Use double precision gemm for mp syrk for now
    h2opus_fbl_gemm(H2OpusFBLTrans, H2OpusFBLNoTrans, n, n, m, 1, A, lda, A, lda, 0, B, ldb);
}

inline void blas_transpose(int m, int n, float *A, int lda, float *At, int ldat)
{
#ifdef H2OPUS_USE_MKL
    mkl_somatcopy('C', 'T', m, n, 1, A, lda, At, ldat);
#else
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++)
            At[c + r * ldat] = A[r + c * lda];
#endif
}

inline void blas_transpose(int m, int n, double *A, int lda, double *At, int ldat)
{
#ifdef H2OPUS_USE_MKL
    mkl_domatcopy('C', 'T', m, n, 1, A, lda, At, ldat);
#else
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++)
            At[c + r * ldat] = A[r + c * lda];
#endif
}

#ifdef H2OPUS_USE_MKL
inline int vRngGaussian(int type, VSLStreamStatePtr stream, int num_gens, float *rng_data, float a, float s)
{
    return vsRngGaussian(type, stream, num_gens, rng_data, a, s);
}

inline int vRngGaussian(int type, VSLStreamStatePtr stream, int num_gens, double *rng_data, double a, double s)
{
    return vdRngGaussian(type, stream, num_gens, rng_data, a, s);
}
#endif

#ifdef H2OPUS_USE_NEC
inline asl_error_t asl_rand_gen(asl_random_t hnd, asl_int_t num, float *val)
{
    return asl_random_generate_s(hnd, num, val);
}

inline asl_error_t asl_rand_gen(asl_random_t hnd, asl_int_t num, double *val)
{
    return asl_random_generate_d(hnd, num, val);
}
#endif

#ifdef H2OPUS_USE_AMDRNG
inline rng_int_t rng_rand_gen(rng_state_t &rng, rng_int_t num, float *val)
{
    rng_int_t info;
    sranduniform(num, 0, 1, rng.fstate, val, &info);
    return info;
}

inline rng_int_t rng_rand_gen(rng_state_t &rng, rng_int_t num, double *val)
{
    rng_int_t info;
    dranduniform(num, 0, 1, rng.dstate, val, &info);
    return info;
}
#endif

#ifdef H2OPUS_USE_ESSL
inline void essl_rand_gen(essl_rndstate_t &rng, _ESVINT num, float *val)
{
    return surng(num, 0, 1, val, rng.istate, rng.listate);
}

inline void essl_rand_gen(essl_rndstate_t &rng, _ESVINT num, double *val)
{
    return durng(num, 0, 1, val, rng.istate, rng.listate);
}
#endif

#endif
