#ifndef __H2OPUS_BATCH_WRAPPERS_GPU__
#define __H2OPUS_BATCH_WRAPPERS_GPU__

#include <h2opusconf.h>

#ifdef H2OPUS_USE_GPU

#include <h2opus/util/blas_wrappers.h>
#include <h2opus/util/batch_block_set.h>
#include <h2opus/util/batch_matrix_add.h>
#include <h2opus/util/debug_routines.h>
#include <h2opus/util/gemv_batch.h>
#include <h2opus/util/batch_potrf_rd.h>

#include <kblas_defs.h>
#include <batch_ara.h>
#include <batch_block_copy.h>
#include <batch_geqp.h>
#include <batch_qr.h>
#include <batch_rand.h>
#include <batch_transpose.h>

//////////////////////////////////////////
// Some magma wrappers
//////////////////////////////////////////
inline void magmablas_gemv_batched(magma_trans_t trans, int m, int n, float alpha, float **A, int lda, float **x,
                                   int incx, float beta, float **y, int incy, int batchCount, magma_queue_t queue)
{
    magmablas_sgemv_batched(trans, m, n, alpha, A, lda, x, incx, beta, y, incy, batchCount, queue);
}

inline void magmablas_gemv_batched(magma_trans_t trans, int m, int n, float alpha, double **A, int lda, double **x,
                                   int incx, double beta, double **y, int incy, int batchCount, magma_queue_t queue)
{
    magmablas_dgemv_batched(trans, m, n, alpha, A, lda, x, incx, beta, y, incy, batchCount, queue);
}

inline int magma_geqrf_batched(int m, int n, float **A_ptrs, int lda, float **tau_ptrs, int *info, int batchCount,
                               magma_queue_t queue)
{
    return magma_sgeqrf_batched(m, n, A_ptrs, lda, tau_ptrs, info, batchCount, queue);
}

inline int magma_geqrf_batched(int m, int n, double **A_ptrs, int lda, double **tau_ptrs, int *info, int batchCount,
                               magma_queue_t queue)
{
    return magma_dgeqrf_batched(m, n, A_ptrs, lda, tau_ptrs, info, batchCount, queue);
}

//////////////////////////////////////////
// GPU wrappers
//////////////////////////////////////////
template <class T> struct H2OpusBatched<T, H2OPUS_HWTYPE_GPU>
{
    // GEMMS
    static inline int gemm(h2opusComputeStream_t stream, char transA, char transB, const int m, const int n,
                           const int k, const T alpha, const T **A, int lda, const T **B, int ldb, const T beta, T **C,
                           int ldc, int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0 || k == 0)
            return 1;
        transA = (transA == H2Opus_Trans ? KBLAS_Trans : KBLAS_NoTrans);
        transB = (transB == H2Opus_Trans ? KBLAS_Trans : KBLAS_NoTrans);

#ifdef H2OPUS_PROFILING_ENABLED
        double batch_gops = H2OPUS_GEMM_OP_COUNT(m, n, k) * 1e-9 * batchCount;
        PerformanceCounter::addOpCount(PerformanceCounter::GEMM, batch_gops);
#endif
        // See if we should use gemv instead of gemm
        if (transB == H2Opus_NoTrans && n == 1)
        {
            cudaStream_t cuda_stream = stream->getCudaStream();
            if (transA == KBLAS_Trans)
                gemv_batch('T', k, m, alpha, (T **)A, lda, (T **)B, beta, C, batchCount, cuda_stream);
            else
                gemv_batch('N', m, k, alpha, (T **)A, lda, (T **)B, beta, C, batchCount, cuda_stream);

            return 1;
        }
        else
        {
            int batch_increment = H2OPUS_MAX_OPS_PER_BATCH;
            int batch_start = 0;
            int success = 1;
            while (batch_start != batchCount)
            {
                int batch_size = std::min(batch_increment, batchCount - batch_start);

                int retval = kblas_gemm_batch(stream->getKblasHandle(), transA, transB, m, n, k, alpha, A + batch_start,
                                              lda, B + batch_start, ldb, beta, C + batch_start, ldc, batch_size);
                if (retval != 1)
                    success = 0;

                batch_start += batch_size;
            }
            return success;
        }
    }

    static inline int gemm(h2opusComputeStream_t stream, char transA, char transB, int *m, int *n, int *k, int max_m,
                           int max_n, int max_k, const T alpha, const T **A, int *lda, const T **B, int *ldb,
                           const T beta, T **C, int *ldc, int batchCount)
    {
        if (batchCount == 0)
            return 1;
        transA = (transA == H2Opus_Trans ? KBLAS_Trans : KBLAS_NoTrans);
        transB = (transB == H2Opus_Trans ? KBLAS_Trans : KBLAS_NoTrans);

#ifdef H2OPUS_PROFILING_ENABLED
        // cannot sum, device resident memory
        PerformanceCounter::addOpCount(PerformanceCounter::GEMM,
                                       H2OPUS_GEMM_OP_COUNT(max_m, max_n, max_k) * batchCount * 1e-9);
#endif
        // TODO: vbatch gemv without having to do incx and incy arrays
        return kblas_gemm_batch(stream->getKblasHandle(), transA, transB, m, n, k, max_m, max_n, max_k, alpha, A, lda,
                                B, ldb, beta, C, ldc, batchCount);
    }

    // Syrk
    static inline int syrk(h2opusComputeStream_t stream, char uplo, char trans, int m, int n, const T alpha,
                           const T **A, int lda, const T beta, T **B, int ldb, int batchCount)
    {
        if (batchCount == 0)
            return 1;
        trans = (trans == H2Opus_Trans ? KBLAS_Trans : KBLAS_NoTrans);
        uplo = (uplo == H2Opus_Upper ? KBLAS_Upper : KBLAS_Lower);

#ifdef H2OPUS_PROFILING_ENABLED
        double batch_gops = H2OPUS_SYRK_OP_COUNT(m, n) * 1e-9 * batchCount;
        PerformanceCounter::addOpCount(PerformanceCounter::GEMM, batch_gops);
#endif
        return kblas_syrk_batch(stream->getKblasHandle(), uplo, trans, m, n, alpha, A, lda, beta, B, ldb, batchCount);
    }

    // Mixed precision Syrk
    static inline int mp_syrk(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, const T **A_ptrs,
                              int *lda, double **B_ptrs, int *ldb, int batchCount)
    {
        if (batchCount == 0)
            return 1;

        return kblas_ara_mp_syrk_batch(stream->getKblasHandle(), m, n, max_m, max_n, A_ptrs, lda, B_ptrs, ldb,
                                       batchCount);
    }

    // Regular cholesky
    static inline int potrf(h2opusComputeStream_t stream, int m, double **A_ptrs, int lda, int batchCount)
    {
        // not yet implemented
        assert(false);
        return 0;
    }

    static inline int potrf(h2opusComputeStream_t stream, int *m, int max_m, double **A_ptrs, int *lda, int batchCount)
    {
        // not yet implemented
        assert(false);
        return 0;
    }

    // Mixed precision Cholesky
    static inline int mp_fused_potrf(h2opusComputeStream_t stream, int *m, int max_m, double **A_ptrs, int *lda,
                                     T **R_ptrs, int *ldr, double *R_diag, int *block_ranks, int batchCount)
    {
        if (batchCount == 0)
            return 1;

        return kblas_ara_fused_potrf_batch(m, A_ptrs, lda, R_ptrs, ldr, R_diag, max_m, block_ranks, batchCount,
                                           stream->getCudaStream());
    }

    // Rank deficient Cholesky
    static inline int potrf_rd(h2opusComputeStream_t stream, int m, T **A_ptrs, int lda, int batchCount)
    {
        potrf_rd_batch(m, A_ptrs, lda, batchCount, stream);
        return 1;
    }

    // TRSM
    static inline int trsm_ara(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **B_ptrs, int *ldb,
                               T **A_ptrs, int *lda, int batchCount)
    {
        return kblas_ara_trsm_batch(stream->getKblasHandle(), B_ptrs, ldb, A_ptrs, lda, m, n, batchCount, max_m, max_n);
    }

    static inline int trsm(h2opusComputeStream_t stream, char side, char uplo, char trans, char diag, int *m, int *n,
                           int max_m, int max_n, T alpha, T **A_ptrs, int *lda, T **B_ptrs, int *ldb, int batchCount)
    {
        char kblas_side = (side == H2Opus_Left ? KBLAS_Left : KBLAS_Right);
        char kblas_uplo = (uplo == H2Opus_Upper ? KBLAS_Upper : KBLAS_Lower);
        char kblas_transA = (trans == H2Opus_Trans ? KBLAS_Trans : KBLAS_NoTrans);
        char kblas_diag = (diag == H2Opus_Unit ? KBLAS_Unit : KBLAS_NonUnit);

        return kblas_trsm_batch(stream->getKblasHandle(), kblas_side, kblas_uplo, kblas_transA, kblas_diag, m, n, max_m,
                                max_n, alpha, A_ptrs, lda, B_ptrs, ldb, batchCount);
    }
    // QR
    static inline int geqrf(h2opusComputeStream_t stream, int m, int n, T *A_strided, int lda, int stride_a,
                            T *tau_strided, int stride_tau, int batchCount)
    {
#ifdef H2OPUS_PROFILING_ENABLED
        PerformanceCounter::addOpCount(PerformanceCounter::QR, H2OPUS_QR_OP_COUNT(m, n) * 1e-9 * batchCount);
#endif
        return kblas_geqrf_batch(stream->getKblasHandle(), m, n, A_strided, lda, stride_a, tau_strided, stride_tau,
                                 batchCount);
    }

    static inline int orgqr(h2opusComputeStream_t stream, int m, int n, T *A_strided, int lda, int stride_a,
                            T *tau_strided, int stride_tau, int batchCount)
    {
#ifdef H2OPUS_PROFILING_ENABLED
        PerformanceCounter::addOpCount(PerformanceCounter::QR, H2OPUS_QR_OP_COUNT(m, n) * 1e-9 * batchCount);
#endif
        return kblas_orgqr_batch(stream->getKblasHandle(), m, n, A_strided, lda, stride_a, tau_strided, stride_tau,
                                 batchCount);
    }

    static inline int orgqr(h2opusComputeStream_t stream, int m, int n, T **A_ptrs, int lda, T **tau_ptrs,
                            int batchCount)
    {
#ifdef H2OPUS_PROFILING_ENABLED
        PerformanceCounter::addOpCount(PerformanceCounter::QR, H2OPUS_QR_OP_COUNT(m, n) * 1e-9 * batchCount);
#endif
        return kblas_orgqr_batch(stream->getKblasHandle(), m, n, A_ptrs, lda, tau_ptrs, batchCount);
    }

    static inline int tsqrf(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs, int *lda,
                            T **tau_ptrs, int batchCount)
    {
#ifdef H2OPUS_PROFILING_ENABLED
        // cannot sum, device resident memory
        PerformanceCounter::addOpCount(PerformanceCounter::QR, H2OPUS_QR_OP_COUNT(max_m, max_n) * batchCount * 1e-9);
#endif
        return kblas_tsqrf_vbatch(stream->getKblasHandle(), m, n, max_m, max_n, A_ptrs, lda, tau_ptrs, batchCount);
    }

    static inline int geqp2(h2opusComputeStream_t stream, int m, int n, T *A_strided, int lda, int stride_a,
                            T *tau_strided, int stride_tau, int *ranks, T eps, int batchCount)
    {
        if (m == 0 || n == 0)
        {
            fillArray(ranks, batchCount, 0, stream, H2OPUS_HWTYPE_GPU);
            return 1;
        }

        if (m <= 512)
        {
#ifdef H2OPUS_PROFILING_ENABLED
            PerformanceCounter::addOpCount(PerformanceCounter::QR, H2OPUS_QR_OP_COUNT(m, n) * 1e-9 * batchCount);
#endif
            return kblas_geqp2_batch(stream->getKblasHandle(), m, n, A_strided, lda, stride_a, tau_strided, stride_tau,
                                     NULL, 0, ranks, eps, batchCount);
        }
        else
        {
            thrust::host_vector<T> A_strided_host = copyGPUArray(A_strided, batchCount * stride_a);
            thrust::host_vector<T> tau_strided_host(batchCount * stride_tau, 0);
            thrust::host_vector<int> ranks_host(batchCount);
            H2OpusBatched<H2Opus_Real, H2OPUS_HWTYPE_CPU>::geqp2(stream, m, n, vec_ptr(A_strided_host), lda, stride_a,
                                                                 vec_ptr(tau_strided_host), stride_tau,
                                                                 vec_ptr(ranks_host), eps, batchCount);
            cudaMemcpy(A_strided, vec_ptr(A_strided_host), A_strided_host.size() * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(tau_strided, vec_ptr(tau_strided_host), tau_strided_host.size() * sizeof(T),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(ranks, vec_ptr(ranks_host), ranks_host.size() * sizeof(int), cudaMemcpyHostToDevice);

            return 1;
        }
    }

    // SVD
    static inline int gesvd(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs, int *lda,
                            T **U_ptrs, int *ldu, T **S_ptrs, T **V_ptrs, int *ldv, T eps, int *ranks, int batchCount)
    {
        assert(false);
        return 0;
    }

    // Copies
    static inline int copy_upper(h2opusComputeStream_t stream, int m, int n, T *A_strided, int lda, int stride_a,
                                 T *R_strided, int ldr, int stride_r, int batchCount)
    {
        if (m == 0 || n == 0)
            return 1;

        return kblas_copy_upper_batch(stream->getKblasHandle(), m, n, A_strided, lda, stride_a, R_strided, ldr,
                                      stride_r, batchCount);
    }

    static inline int copyBlock(h2opusComputeStream_t stream, int m, int n, T **dest_ptrs, int dest_row_offset,
                                int dest_col_offset, int ld_dest, T **src_ptrs, int src_row_offset, int src_col_offset,
                                int ld_src, int batchCount)
    {
        if (m == 0 || n == 0)
            return 1;

        return kblas_copyBlock_batch(stream->getKblasHandle(), m, n, dest_ptrs, dest_row_offset, dest_col_offset,
                                     ld_dest, src_ptrs, src_row_offset, src_col_offset, ld_src, batchCount);
    }

    static inline int copyBlock(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **dest_ptrs,
                                int *ld_dest, T **src_ptrs, int *ld_src, int batchCount)
    {
        if (max_m == 0 || max_n == 0)
            return 1;

        return kblas_copyBlock_vbatch(stream->getKblasHandle(), m, n, max_m, max_n, dest_ptrs, ld_dest, src_ptrs,
                                      ld_src, batchCount);
    }

    static inline int copyBlock(h2opusComputeStream_t stream, int m, int n, T *dest_strided, int dest_row_offset,
                                int dest_col_offset, int ld_dest, int dest_stride, T *src_strided, int src_row_offset,
                                int src_col_offset, int ld_src, int src_stride, int batchCount)
    {
        if (m == 0 || n == 0)
            return 1;

        return kblas_copyBlock_batch(stream->getKblasHandle(), m, n, dest_strided, dest_row_offset, dest_col_offset,
                                     ld_dest, dest_stride, src_strided, src_row_offset, src_col_offset, ld_src,
                                     src_stride, batchCount);
    }

    // Transpose
    static inline int transpose(h2opusComputeStream_t stream, int m, int n, T **A_ptrs, int lda, T **At_ptrs, int ldat,
                                int batchCount)
    {
        if (m == 0 || n == 0)
            return 1;

        return kblas_transpose_batch(stream->getKblasHandle(), m, n, A_ptrs, lda, At_ptrs, ldat, batchCount);
    }

    static inline int transpose(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs,
                                int *lda, T **At_ptrs, int *ldat, int batchCount)
    {
        if (max_m == 0 || max_n == 0)
            return 1;

        return kblas_transpose_vbatch(stream->getKblasHandle(), m, n, max_m, max_n, A_ptrs, lda, At_ptrs, ldat,
                                      batchCount);
    }

    // Matrix addition
    // C = alpha * A + beta * B
    static inline void add_matrix(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T alpha,
                                  T **A_ptrs, int *lda, T beta, T **B_ptrs, int *ldb, T **C_ptrs, int *ldc,
                                  int batchCount)
    {
        if (max_m == 0 || max_n == 0)
            return;

        batchMatrixAdd(m, n, max_m, max_n, alpha, A_ptrs, lda, beta, B_ptrs, ldb, C_ptrs, ldc, batchCount,
                       stream->getCudaStream());
    }

    // Set block to identity
    static inline void setIdentity(h2opusComputeStream_t stream, int m, int n, T **A_ptrs, int lda, int batchCount)
    {
        if (m == 0 || n == 0)
            return;

        batchBlockSetIdentity(stream->getCudaStream(), m, n, A_ptrs, lda, batchCount);
    }

    static inline void setDiagonal(h2opusComputeStream_t stream, int m, int n, T **A_ptrs, int lda, T value,
                                   int batchCount)
    {
        if (m == 0 || n == 0)
            return;

        batchBlockSetDiagonal(stream->getCudaStream(), m, n, A_ptrs, lda, value, batchCount);
    }

    static inline void setIdentity(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs,
                                   int *lda, int batchCount)
    {
        if (max_m == 0 || max_n == 0)
            return;

        batchBlockSetIdentity(stream->getCudaStream(), m, n, max_m, max_n, A_ptrs, lda, batchCount);
    }

    static inline void setDiagonal(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs,
                                   int *lda, T value, int batchCount)
    {
        if (max_m == 0 || max_n == 0)
            return;

        batchBlockSetIdentity(stream->getCudaStream(), m, n, max_m, max_n, A_ptrs, lda, value, batchCount);
    }

    // Set block to zero
    static inline void setZero(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs, int *lda,
                               int batchCount)
    {
        if (max_m == 0 || max_n == 0)
            return;

        batchBlockSetZero(stream->getCudaStream(), m, n, max_m, max_n, A_ptrs, lda, batchCount);
    }

    // Set upper triangular half to zero
    static inline int setUpperZero(h2opusComputeStream_t stream, int m, int n, T **A_ptrs, int lda, int batchCount)
    {
        if (m == 0 || n == 0)
            return 1;

        batchBlockSetUpperZero(stream->getCudaStream(), m, n, A_ptrs, lda, batchCount);
        return 1;
    }
    // Batch random number generation
    static inline int rand(h2opusComputeStream_t stream, h2opusHandle_t handle, int *m, int *n, int max_m, int max_n,
                           T **A_ptrs, int *lda, int batchCount)
    {
        if (max_m == 0 || max_n == 0)
            return 1;

        return kblas_rand_batch(stream->getKblasHandle(), m, n, A_ptrs, lda, max_m, handle->getKblasRandState(),
                                batchCount);
    }

    static inline int rand(h2opusComputeStream_t stream, h2opusHandle_t handle, int m, int n, T *A_strided, int lda,
                           int stride_a, int batchCount)
    {
        if (m == 0 || n == 0)
            return 1;

        return kblas_rand_batch(stream->getKblasHandle(), m, n, A_strided, lda, stride_a, handle->getKblasRandState(),
                                batchCount);
    }
};

#endif
#endif
