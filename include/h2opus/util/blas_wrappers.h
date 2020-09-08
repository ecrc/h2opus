#ifndef __BLAS_WRAPPERS_H__
#define __BLAS_WRAPPERS_H__

#include <h2opus/core/h2opus_compute_stream.h>
#include <h2opus/core/h2opus_defs.h>

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
inline void blas_gemm(h2opusComputeStream_t stream, int transa, int transb, int m, int n, int k, T alpha, const T *A,
                      int lda, const T *B, int ldb, T beta, T *C, int ldc);

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU
///////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU

#include <cublas_v2.h>
#include <cuda_runtime.h>

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
inline void blas_gemm<float, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int transa, int transb, int m, int n,
                                                int k, float alpha, const float *A, int lda, const float *B, int ldb,
                                                float beta, float *C, int ldc)
{
    cublasSgemm(stream->getCublasHandle(), (transa == 1 ? CUBLAS_OP_T : CUBLAS_OP_N),
                (transb == 1 ? CUBLAS_OP_T : CUBLAS_OP_N), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <>
inline void blas_gemm<double, H2OPUS_HWTYPE_GPU>(h2opusComputeStream_t stream, int transa, int transb, int m, int n,
                                                 int k, double alpha, const double *A, int lda, const double *B,
                                                 int ldb, double beta, double *C, int ldc)
{
    cublasDgemm(stream->getCublasHandle(), (transa == 1 ? CUBLAS_OP_T : CUBLAS_OP_N),
                (transb == 1 ? CUBLAS_OP_T : CUBLAS_OP_N), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU
///////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_MKL
#include <mkl.h>
#else
/* Fix for OpenBLAS */
#ifndef LAPACK_COMPLEX_CUSTOM
#include <complex>
#define H2OPUS_UNDEF_OPENBLAS
#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#endif
#include <cblas.h>
#include <lapacke.h>
#ifdef H2OPUS_UNDEF_OPENBLAS
#undef LAPACK_COMPLEX_CUSTOM
#undef lapack_complex_float
#undef lapack_complex_double
#undef H2OPUS_UNDEF_OPENBLAS
#endif

#endif
// Dense BLAS used for single execution
template <>
inline double blas_norm2<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, double *x, int incx)
{
    return cblas_dnrm2(n, x, incx);
}

template <> inline float blas_norm2<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, float *x, int incx)
{
    return cblas_snrm2(n, x, incx);
}

template <>
inline double blas_dot_product<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, double *x, int incx,
                                                          double *y, int incy)
{
    return cblas_ddot(n, x, incx, y, incy);
}

template <>
inline float blas_dot_product<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, float *x, int incx,
                                                        float *y, int incy)
{
    return cblas_sdot(n, x, incx, y, incy);
}

template <>
inline void blas_scal<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, double alpha, double *x, int incx)
{
    cblas_dscal(n, alpha, x, incx);
}

template <>
inline void blas_scal<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, float alpha, float *x, int incx)
{
    cblas_sscal(n, alpha, x, incx);
}

template <>
inline void blas_axpy<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, double alpha, const double *x,
                                                 int incx, double *y, int incy)
{
    cblas_daxpy(n, alpha, x, incx, y, incy);
}

template <>
inline void blas_axpy<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int n, float alpha, const float *x,
                                                int incx, float *y, int incy)
{
    cblas_saxpy(n, alpha, x, incx, y, incy);
}

template <>
inline void blas_gemm<float, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int transa, int transb, int m, int n,
                                                int k, float alpha, const float *A, int lda, const float *B, int ldb,
                                                float beta, float *C, int ldc)
{
    cblas_sgemm(CblasColMajor, transa == 1 ? CblasTrans : CblasNoTrans, transb == 1 ? CblasTrans : CblasNoTrans, m, n,
                k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline void blas_gemm<double, H2OPUS_HWTYPE_CPU>(h2opusComputeStream_t stream, int transa, int transb, int m, int n,
                                                 int k, double alpha, const double *A, int lda, const double *B,
                                                 int ldb, double beta, double *C, int ldc)
{
    cblas_dgemm(CblasColMajor, transa == 1 ? CblasTrans : CblasNoTrans, transb == 1 ? CblasTrans : CblasNoTrans, m, n,
                k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Dense BLAS used in the batch execution
inline void cblas_gemm(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const int m, const int n,
                       const int k, const float alpha, const float *a, const int lda, const float *b, const int ldb,
                       const float beta, float *c, const int ldc)
{
    cblas_sgemm(CblasColMajor, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void cblas_gemm(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const int m, const int n,
                       const int k, const double alpha, const double *a, const int lda, const double *b, const int ldb,
                       const double beta, double *c, const int ldc)
{
    cblas_dgemm(CblasColMajor, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void cblas_gemv(const CBLAS_TRANSPOSE trans, const int m, const int n, const float alpha, const float *a,
                       const int lda, const float *x, const float beta, float *y)
{
    cblas_sgemv(CblasColMajor, trans, m, n, alpha, a, lda, x, 1, beta, y, 1);
}

inline void cblas_gemv(const CBLAS_TRANSPOSE trans, const int m, const int n, const double alpha, const double *a,
                       const int lda, const double *x, const double beta, double *y)
{
    cblas_dgemv(CblasColMajor, trans, m, n, alpha, a, lda, x, 1, beta, y, 1);
}

inline int lapack_gesvd(int m, int n, float *a, int lda, float *s, float *superb)
{
    return LAPACKE_sgesvd(CblasColMajor, 'O', 'N', m, n, a, lda, s, NULL, m, NULL, n, superb);
}

inline int lapack_gesvd(int m, int n, double *a, int lda, double *s, double *superb)
{
    return LAPACKE_dgesvd(CblasColMajor, 'O', 'N', m, n, a, lda, s, NULL, m, NULL, n, superb);
}

inline int lapack_geqrf(int m, int n, float *a, int lda, float *tau)
{
    return LAPACKE_sgeqrf(CblasColMajor, m, n, a, lda, tau);
}

inline int lapack_geqrf(int m, int n, double *a, int lda, double *tau)
{
    return LAPACKE_dgeqrf(CblasColMajor, m, n, a, lda, tau);
}

inline int lapack_orgqr(int m, int n, int k, float *a, int lda, float *tau)
{
    return LAPACKE_sorgqr(CblasColMajor, m, n, k, a, lda, tau);
}

inline int lapack_orgqr(int m, int n, int k, double *a, int lda, double *tau)
{
    return LAPACKE_dorgqr(CblasColMajor, m, n, k, a, lda, tau);
}

inline int lapack_lacpy(char uplo, int m, int n, float *a, int lda, float *b, int ldb)
{
    return LAPACKE_slacpy(CblasColMajor, uplo, m, n, a, lda, b, ldb);
}

inline int lapack_lacpy(char uplo, int m, int n, double *a, int lda, double *b, int ldb)
{
    return LAPACKE_dlacpy(CblasColMajor, uplo, m, n, a, lda, b, ldb);
}

inline int lapack_geqp3(int m, int n, float *a, int lda, int *jpvt, float *tau)
{
    return LAPACKE_sgeqp3(CblasColMajor, m, n, a, lda, jpvt, tau);
}

inline int lapack_geqp3(int m, int n, double *a, int lda, int *jpvt, double *tau)
{
    return LAPACKE_dgeqp3(CblasColMajor, m, n, a, lda, jpvt, tau);
}

inline int lapack_geqpf(int m, int n, float *a, int lda, int *jpvt, float *tau)
{
    return LAPACKE_sgeqpf(CblasColMajor, m, n, a, lda, jpvt, tau);
}

inline int lapack_geqpf(int m, int n, double *a, int lda, int *jpvt, double *tau)
{
    return LAPACKE_dgeqpf(CblasColMajor, m, n, a, lda, jpvt, tau);
}

inline void cblas_trsm(const CBLAS_SIDE side, const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE transa,
                       const CBLAS_DIAG diag, int m, int n, float alpha, const float *a, int lda, float *b, int ldb)
{
    cblas_strsm(CblasColMajor, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

inline void cblas_trsm(const CBLAS_SIDE side, const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE transa,
                       const CBLAS_DIAG diag, int m, int n, double alpha, const double *a, int lda, double *b, int ldb)
{
    cblas_dtrsm(CblasColMajor, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
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

#endif
