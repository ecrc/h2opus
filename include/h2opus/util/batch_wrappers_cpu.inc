#ifndef __H2OPUS_BATCH_WRAPPERS_CPU__
#define __H2OPUS_BATCH_WRAPPERS_CPU__

#include <h2opus/util/blas_wrappers.h>

//////////////////////////////////////////
// CPU wrappers
//////////////////////////////////////////
#ifndef CPU_BATCH_LD
#define CPU_BATCH_LD(ld) ((ld) == 0 ? 1 : (ld))
#endif

#ifndef _OPENMP
#define omp_get_thread_num() -1
#else
#include <omp.h>
#endif

template <class T> struct H2OpusBatched<T, H2OPUS_HWTYPE_CPU>
{
    // Experimental interface to library entry points for batched gemm

    static inline void gemv_batch(h2opusComputeStream_t stream, const H2OPUS_FBL_TRANSPOSE transa, int m, int n,
                                  const float alpha, const float **A, int lda, const float **B, const float beta,
                                  float **C, int batchCount)
    {
        H2OPUS_BEGIN_BATCH_BLAS()
#if defined(H2OPUS_USE_BATCHED_BLAS) && defined(H2OPUS_USE_MKL)
        int group_count = 1;
        int group_sizes = batchCount;
        char ctransa = (transa == H2OpusFBLTrans) ? 'T' : 'N';
        h2opus_fbl_sgemv_batch(&ctransa, &m, &n, &alpha, A, &lda, B, &group_count, &beta, C, &group_count, &group_count,
                               &group_sizes);
#else
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
            h2opus_fbl_gemv(transa, m, n, alpha, A[i], lda, B[i], beta, C[i]);
#endif
        H2OPUS_END_BATCH_BLAS()
    }

    static inline void gemv_batch(h2opusComputeStream_t stream, const H2OPUS_FBL_TRANSPOSE transa, int m, int n,
                                  const double alpha, const double **A, int lda, const double **B, const double beta,
                                  double **C, int batchCount)
    {
        H2OPUS_BEGIN_BATCH_BLAS()
#if defined(H2OPUS_USE_BATCHED_BLAS) && defined(H2OPUS_USE_MKL)
        int group_count = 1;
        int group_sizes = batchCount;
        char ctransa = (transa == H2OpusFBLTrans) ? 'T' : 'N';
        h2opus_fbl_dgemv_batch(&ctransa, &m, &n, &alpha, A, &lda, B, &group_count, &beta, C, &group_count, &group_count,
                               &group_sizes);
#else
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
            h2opus_fbl_gemv(transa, m, n, alpha, A[i], lda, B[i], beta, C[i]);
#endif
        H2OPUS_END_BATCH_BLAS()
    }

    static inline void gemv_batch(h2opusComputeStream_t stream, const H2OPUS_FBL_TRANSPOSE transa, const int *m,
                                  const int *n, const float alpha, const float **A, int *lda, const float **B,
                                  const float beta, float **C, int batchCount)
    {
        H2OPUS_BEGIN_BATCH_BLAS()
#if defined(H2OPUS_USE_BATCHED_BLAS) && defined(H2OPUS_USE_MKL)
        const H2OPUS_FBL_TRANSPOSE transb = transa;
        int *ldb = NULL, *ldc = NULL;
        H2OPUS_BATCH_GROUP_BEGIN(float)
        h2opus_fbl_sgemv_batch(ta_array, m, n, alpha_array, A, lda, B, group_sizes, beta_array, C, group_sizes,
                               &group_count, group_sizes);
        H2OPUS_BATCH_GROUP_END()
#else
#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
            h2opus_fbl_gemv(transa, m[i], n[i], alpha, A[i], CPU_BATCH_LD(lda[i]), B[i], beta, C[i]);
#endif
        H2OPUS_END_BATCH_BLAS()
    }

    static inline void gemv_batch(h2opusComputeStream_t stream, const H2OPUS_FBL_TRANSPOSE transa, const int *m,
                                  const int *n, const double alpha, const double **A, int *lda, const double **B,
                                  const double beta, double **C, int batchCount)
    {
        H2OPUS_BEGIN_BATCH_BLAS()
#if defined(H2OPUS_USE_BATCHED_BLAS) && defined(H2OPUS_USE_MKL)
        const H2OPUS_FBL_TRANSPOSE transb = transa;
        int *ldb = NULL, *ldc = NULL;
        H2OPUS_BATCH_GROUP_BEGIN(double)
        h2opus_fbl_dgemv_batch(ta_array, m, n, alpha_array, A, lda, B, group_sizes, beta_array, C, group_sizes,
                               &group_count, group_sizes);
        H2OPUS_BATCH_GROUP_END()
#else
#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
            h2opus_fbl_gemv(transa, m[i], n[i], alpha, A[i], CPU_BATCH_LD(lda[i]), B[i], beta, C[i]);
#endif
        H2OPUS_END_BATCH_BLAS()
    }

    static inline void gemm_batch(h2opusComputeStream_t stream, const H2OPUS_FBL_TRANSPOSE transa,
                                  const H2OPUS_FBL_TRANSPOSE transb, int m, int n, int k, const float alpha,
                                  const float **A, int lda, const float **B, int ldb, const float beta, float **C,
                                  int ldc, int batchCount)
    {
        H2OPUS_BEGIN_BATCH_BLAS()
#if defined(H2OPUS_USE_BATCHED_BLAS)
        int group_count = 1;
        int group_sizes = batchCount;
        char ctransa = (transa == H2OpusFBLTrans) ? 'T' : 'N';
        char ctransb = (transb == H2OpusFBLTrans) ? 'T' : 'N';
#if defined(H2OPUS_USE_LIBXSMM)
        libxsmm_sgemm_batch_omp(&ctransa, &ctransb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, &group_count,
                                &group_sizes);
#else
        h2opus_fbl_sgemm_batch(&ctransa, &ctransb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, &group_count,
                               &group_sizes);
#endif

#else
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
            h2opus_fbl_sgemm(transa, transb, m, n, k, alpha, A[i], lda, B[i], ldb, beta, C[i], ldc);
#endif
        H2OPUS_END_BATCH_BLAS()
    }

    static inline void gemm_batch(h2opusComputeStream_t stream, const H2OPUS_FBL_TRANSPOSE transa,
                                  const H2OPUS_FBL_TRANSPOSE transb, int m, int n, int k, const double alpha,
                                  const double **A, int lda, const double **B, int ldb, const double beta, double **C,
                                  int ldc, int batchCount)
    {
        H2OPUS_BEGIN_BATCH_BLAS()
#if defined(H2OPUS_USE_BATCHED_BLAS)
        int group_count = 1;
        int group_sizes = batchCount;
        char ctransa = (transa == H2OpusFBLTrans) ? 'T' : 'N';
        char ctransb = (transb == H2OpusFBLTrans) ? 'T' : 'N';
#if defined(H2OPUS_USE_LIBXSMM)
        libxsmm_dgemm_batch_omp(&ctransa, &ctransb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, &group_count,
                                &group_sizes);
#else
        h2opus_fbl_dgemm_batch(&ctransa, &ctransb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, &group_count,
                               &group_sizes);
#endif

#else
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
            h2opus_fbl_dgemm(transa, transb, m, n, k, alpha, A[i], lda, B[i], ldb, beta, C[i], ldc);
#endif
        H2OPUS_END_BATCH_BLAS()
    }

    static inline void gemm_batch(h2opusComputeStream_t stream, const H2OPUS_FBL_TRANSPOSE transa,
                                  const H2OPUS_FBL_TRANSPOSE transb, const int *m, const int *n, const int *k,
                                  const float alpha, const float **A, int *lda, const float **B, int *ldb,
                                  const float beta, float **C, int *ldc, int batchCount)
    {
        H2OPUS_BEGIN_BATCH_BLAS()
#if defined(H2OPUS_USE_BATCHED_BLAS)
        H2OPUS_BATCH_GROUP_BEGIN(float)
#if defined(H2OPUS_USE_LIBXSMM)
        // TODO: omp segfaults
        libxsmm_sgemm_batch(ta_array, tb_array, m, n, k, alpha_array, A, lda, B, ldb, beta_array, C, ldc, &group_count,
                            group_sizes);
#else
        h2opus_fbl_sgemm_batch(ta_array, tb_array, m, n, k, alpha_array, A, lda, B, ldb, beta_array, C, ldc,
                               &group_count, group_sizes);
#endif
        H2OPUS_BATCH_GROUP_END()
#else
#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
            h2opus_fbl_sgemm(transa, transb, m[i], n[i], k[i], alpha, A[i], CPU_BATCH_LD(lda[i]), B[i],
                             CPU_BATCH_LD(ldb[i]), beta, C[i], CPU_BATCH_LD(ldc[i]));
#endif
        H2OPUS_END_BATCH_BLAS()
    }

    static inline void gemm_batch(h2opusComputeStream_t stream, const H2OPUS_FBL_TRANSPOSE transa,
                                  const H2OPUS_FBL_TRANSPOSE transb, const int *m, const int *n, const int *k,
                                  const double alpha, const double **A, int *lda, const double **B, int *ldb,
                                  const double beta, double **C, int *ldc, int batchCount)
    {
        H2OPUS_BEGIN_BATCH_BLAS()
#if defined(H2OPUS_USE_BATCHED_BLAS)
        H2OPUS_BATCH_GROUP_BEGIN(double)
#if defined(H2OPUS_USE_LIBXSMM)
        // TODO: omp segfaults
        libxsmm_dgemm_batch(ta_array, tb_array, m, n, k, alpha_array, A, lda, B, ldb, beta_array, C, ldc, &group_count,
                            group_sizes);
#else
        h2opus_fbl_dgemm_batch(ta_array, tb_array, m, n, k, alpha_array, A, lda, B, ldb, beta_array, C, ldc,
                               &group_count, group_sizes);
#endif
        H2OPUS_BATCH_GROUP_END()
#else
#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
            h2opus_fbl_dgemm(transa, transb, m[i], n[i], k[i], alpha, A[i], CPU_BATCH_LD(lda[i]), B[i],
                             CPU_BATCH_LD(ldb[i]), beta, C[i], CPU_BATCH_LD(ldc[i]));
#endif
        H2OPUS_END_BATCH_BLAS()
    }

    // GEMMS
    static inline int gemm(h2opusComputeStream_t stream, char transA, char transB, const int m, const int n,
                           const int k, const T alpha, const T **A, int lda, const T **B, int ldb, const T beta, T **C,
                           int ldc, int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0)
            return 1;

        lda = CPU_BATCH_LD(lda);
        ldb = CPU_BATCH_LD(ldb);
        ldc = CPU_BATCH_LD(ldc);

#ifdef H2OPUS_PROFILING_ENABLED
        double batch_gops = H2OPUS_GEMM_OP_COUNT(m, n, k) * 1e-9 * batchCount;
        PerformanceCounter::addOpCount(PerformanceCounter::GEMM, batch_gops);
#endif
        H2OPUS_FBL_TRANSPOSE h2opus_fbl_transA = (transA == H2Opus_Trans ? H2OpusFBLTrans : H2OpusFBLNoTrans);
        H2OPUS_FBL_TRANSPOSE h2opus_fbl_transB = (transB == H2Opus_Trans ? H2OpusFBLTrans : H2OpusFBLNoTrans);

        // See if we should use gemv instead of gemm
        if (transB == H2Opus_NoTrans && n == 1)
        {
            int mm = (transA == H2Opus_Trans) ? k : m;
            int kk = (transA == H2Opus_Trans) ? m : k;
            gemv_batch(stream, h2opus_fbl_transA, mm, kk, alpha, A, lda, B, beta, C, batchCount);
        }
        else
        {
            gemm_batch(stream, h2opus_fbl_transA, h2opus_fbl_transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
                       batchCount);
        }
        return 1;
    }

    static inline int gemm(h2opusComputeStream_t stream, char transA, char transB, int *m, int *n, int *k, int max_m,
                           int max_n, int max_k, const T alpha, const T **A, int *lda, const T **B, int *ldb,
                           const T beta, T **C, int *ldc, int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0 || max_k == 0)
            return 1;

#ifdef H2OPUS_PROFILING_ENABLED
        double batch_gops = 0;
        for (int i = 0; i < batchCount; i++)
            batch_gops += H2OPUS_GEMM_OP_COUNT(m[i], n[i], k[i]);
        PerformanceCounter::addOpCount(PerformanceCounter::GEMM, batch_gops * 1e-9);
#endif
        H2OPUS_FBL_TRANSPOSE h2opus_fbl_transA = (transA == H2Opus_Trans ? H2OpusFBLTrans : H2OpusFBLNoTrans);
        H2OPUS_FBL_TRANSPOSE h2opus_fbl_transB = (transB == H2Opus_Trans ? H2OpusFBLTrans : H2OpusFBLNoTrans);

        // See if we should use gemv instead of gemm
        if (transB == H2Opus_NoTrans && max_n == 1)
        {
            int *mm = (transA == H2Opus_Trans) ? k : m;
            int *kk = (transA == H2Opus_Trans) ? m : k;
            gemv_batch(stream, h2opus_fbl_transA, mm, kk, alpha, A, lda, B, beta, C, batchCount);
        }
        else
        {
            gemm_batch(stream, h2opus_fbl_transA, h2opus_fbl_transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
                       batchCount);
        }
        return 1;
    }

    // Diagonal multiplication
    // C = A * diag(D)
    // A and C can be the same
    static inline int diagRightMult(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A, int *lda,
                                    const T **D, T **C, int *ldc, int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return 1;

#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            const T *D_op = D[op];
            T *C_op = C[op], *A_op = A[op];
            int lda_op = lda[op], ldc_op = ldc[op], m_op = m[op], n_op = n[op];

            for (int j = 0; j < n_op; j++)
            {
                for (int i = 0; i < m_op; i++)
                    C_op[i] = A_op[i] * D_op[j];

                C_op += ldc_op;
                A_op += lda_op;
            }
        }

        return 1;
    }

    // C = diag(D) * A
    // A and C can be the same
    static inline int diagLeftMult(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, const T **D,
                                   T **A, int *lda, T **C, int *ldc, int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return 1;

#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            const T *D_op = D[op];
            T *C_op = C[op], *A_op = A[op];
            int lda_op = lda[op], ldc_op = ldc[op], m_op = m[op], n_op = n[op];

            for (int j = 0; j < n_op; j++)
            {
                for (int i = 0; i < m_op; i++)
                    C_op[i] = A_op[i] * D_op[i];

                C_op += ldc_op;
                A_op += lda_op;
            }
        }

        return 1;
    }

    // C = inv(diag(D)) * A
    // A and C can be the same
    static inline int diagLeftInvMult(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, const T **D,
                                      T **A, int *lda, T **C, int *ldc, int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return 1;

#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            const T *D_op = D[op];
            T *C_op = C[op], *A_op = A[op];
            int lda_op = lda[op], ldc_op = ldc[op], m_op = m[op], n_op = n[op];

            for (int j = 0; j < n_op; j++)
            {
                for (int i = 0; i < m_op; i++)
                    C_op[i] = A_op[i] / D_op[i];

                C_op += ldc_op;
                A_op += lda_op;
            }
        }

        return 1;
    }

    // Syrk
    static inline int syrk(h2opusComputeStream_t stream, char uplo, char trans, int m, int n, const T alpha,
                           const T **A, int lda, const T beta, T **B, int ldb, int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0)
            return 1;

        H2OPUS_FBL_UPLO h2opus_fbl_uplo = (uplo == H2Opus_Upper ? H2OpusFBLUpper : H2OpusFBLLower);
        H2OPUS_FBL_TRANSPOSE h2opus_fbl_trans = (trans == H2Opus_Trans ? H2OpusFBLTrans : H2OpusFBLNoTrans);

#ifdef H2OPUS_PROFILING_ENABLED
        double batch_gops = H2OPUS_SYRK_OP_COUNT(m, n) * 1e-9 * batchCount;
        PerformanceCounter::addOpCount(PerformanceCounter::GEMM, batch_gops);
#endif
        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            const T *A_op = A[op];
            T *B_op = B[op];
            h2opus_fbl_syrk(h2opus_fbl_uplo, h2opus_fbl_trans, m, n, alpha, A_op, lda, beta, B_op, ldb);
        }
        H2OPUS_END_BATCH_BLAS()

        return 1;
    }

    // Mixed precision Syrk
    static inline int mp_syrk(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, const T **A_ptrs,
                              int *lda, double **B_ptrs, int *ldb, int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return 1;

#ifdef H2OPUS_PROFILING_ENABLED
        double batch_gops = 0;
        for (int i = 0; i < batchCount; i++)
            batch_gops += H2OPUS_GEMM_OP_COUNT(n[i], n[i], m[i]);
        PerformanceCounter::addOpCount(PerformanceCounter::GEMM, batch_gops * 1e-9);
#endif

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            const T *A_op = A_ptrs[op];
            double *B_op = B_ptrs[op];

            int m_op = m[op], n_op = n[op];
            int lda_op = lda[op], ldb_op = ldb[op];

            if (!A_op || !B_op || lda_op == 0 || ldb_op == 0)
                continue;

            blas_mp_syrk(m_op, n_op, A_op, lda_op, B_op, ldb_op);
        }
        H2OPUS_END_BATCH_BLAS()

        return 1;
    }

    // Regular cholesky
    static inline int potrf(h2opusComputeStream_t stream, int m, T **A_ptrs, int lda, int batchCount)
    {
        if (batchCount == 0 || m == 0)
            return 1;

#ifdef H2OPUS_PROFILING_ENABLED
        double batch_gops = H2OPUS_POTRF_OP_COUNT(m) * 1e-9 * batchCount;
        PerformanceCounter::addOpCount(PerformanceCounter::POTRF, batch_gops);
#endif

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            double *A_op = A_ptrs[op];
            if (!A_op)
                continue;
            H2OPUS_FBL_CHK_CALL(h2opus_fbl_potrf(m, A_op, lda));
        }
        H2OPUS_END_BATCH_BLAS()

        return 1;
    }

    static inline int potrf(h2opusComputeStream_t stream, int *m, int max_m, T **A_ptrs, int *lda, int batchCount)
    {
        if (batchCount == 0 || max_m == 0)
            return 1;

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            double *A_op = A_ptrs[op];
            if (!A_op)
                continue;
            int m_op = m[op];
            int lda_op = lda[op];
            H2OPUS_FBL_CHK_CALL(h2opus_fbl_potrf(m_op, A_op, lda_op));
        }
        H2OPUS_END_BATCH_BLAS()
    }

    // Mixed precision Cholesky
    static inline int mp_fused_potrf(h2opusComputeStream_t stream, int *m, int max_m, double **A_ptrs, int *lda,
                                     T **R_ptrs, int *ldr, double *R_diag, int *block_ranks, int batchCount)
    {
        if (batchCount == 0 || max_m == 0)
            return 1;

#ifdef H2OPUS_PROFILING_ENABLED
        double batch_gops = 0;
        for (int i = 0; i < batchCount; i++)
            batch_gops += H2OPUS_POTRF_OP_COUNT(m[i]);
        PerformanceCounter::addOpCount(PerformanceCounter::POTRF, batch_gops * 1e-9);
#endif

#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            double *A_op = A_ptrs[op];
            double *R_diag_op = R_diag + op * max_m;
            T *R_op = R_ptrs[op];

            if (!A_op || !R_op)
                continue;

            int m_op = m[op];
            int lda_op = lda[op], ldr_op = ldr[op];

            double max_diag = 0;
            for (int i = 0; i < m_op; i++)
                if (max_diag < A_op[i + i * lda_op])
                    max_diag = A_op[i + i * lda_op];
            double tol = max_diag * H2OpusEpsilon<double>::eps;

            int k = 0;
            for (; k < m_op; k++)
            {
                if (A_op[k + k * lda_op] <= tol)
                    break;

                A_op[k + k * lda_op] = sqrt(A_op[k + k * lda_op]);
                R_diag_op[k] *= A_op[k + k * lda_op];

                for (int i = k + 1; i < m_op; i++)
                    A_op[i + k * lda_op] /= A_op[k + k * lda_op];

                for (int j = k + 1; j < m_op; j++)
                    for (int i = j; i < m_op; i++)
                        A_op[i + j * lda_op] -= A_op[j + k * lda_op] * A_op[i + k * lda_op];
            }

            block_ranks[op] = k;
            for (int i = 0; i < m_op; i++)
                for (int j = 0; j < m_op; j++)
                    R_op[i + j * ldr_op] = (i < k && j < k && j >= i ? A_op[j + i * lda_op] : 0);

            for (int i = k; i < m_op; i++)
                R_diag_op[i] = 0;
        }

        return 1;
    }

    // Rank deficient Cholesky
    static inline int potrf_rd(h2opusComputeStream_t stream, int m, T **A_ptrs, int lda, int batchCount)
    {
        if (batchCount == 0 || m == 0)
            return 1;

#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            T *A_op = A_ptrs[op];
            if (!A_op)
                continue;

            T max_diag = 0;
            for (int i = 0; i < m; i++)
                if (max_diag < A_op[i + i * lda])
                    max_diag = A_op[i + i * lda];
            T tol = max_diag * H2OpusEpsilon<T>::eps;

            for (int k = 0; k < m; k++)
            {
                if (A_op[k + k * lda] <= tol)
                {
                    for (int i = k; i < m; i++)
                        A_op[i + k * lda] = 0;
                }
                else
                {
                    A_op[k + k * lda] = sqrt(A_op[k + k * lda]);
                    for (int i = k + 1; i < m; i++)
                        A_op[i + k * lda] /= A_op[k + k * lda];
                }

                for (int j = k + 1; j < m; j++)
                    for (int i = j; i < m; i++)
                        A_op[i + j * lda] -= A_op[j + k * lda] * A_op[i + k * lda];
            }

            // Clear upper triangle
            for (int k = 0; k < m; k++)
                for (int j = 0; j < k; j++)
                    A_op[j + k * lda] = 0;

            // Transpose
            // for (int k = 0; k < m; k++)
            //     for(int j = 0; j < k; j++)
            //         std::swap(A_op[k + j * lda], A_op[j + k * lda]);
        }

        return 1;
    }

    // TRSM
    static inline int trsm_ara(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **B_ptrs, int *ldb,
                               T **A_ptrs, int *lda, int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return 1;

#ifdef H2OPUS_PROFILING_ENABLED
        double batch_gops = 0;
        for (int i = 0; i < batchCount; i++)
            batch_gops += H2OPUS_TRSM_OP_COUNT(H2Opus_Right, m[i], n[i]);
        PerformanceCounter::addOpCount(PerformanceCounter::TRSM, batch_gops * 1e-9);
#endif

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
        {
            if (A_ptrs[i] && B_ptrs[i] && m[i] > 0 && n[i] > 0)
            {
                h2opus_fbl_trsm(H2OpusFBLRight, H2OpusFBLUpper, H2OpusFBLNoTrans, H2OpusFBLNonUnit, m[i], n[i], 1,
                                A_ptrs[i], lda[i], B_ptrs[i], ldb[i]);
            }
        }
        H2OPUS_END_BATCH_BLAS()

        return 1;
    }

    static inline int trsm(h2opusComputeStream_t stream, char side, char uplo, char trans, char diag, int *m, int *n,
                           int max_m, int max_n, T alpha, T **A_ptrs, int *lda, T **B_ptrs, int *ldb, int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return 1;

#ifdef H2OPUS_PROFILING_ENABLED
        double batch_gops = 0;
        for (int i = 0; i < batchCount; i++)
            batch_gops += H2OPUS_TRSM_OP_COUNT(side, m[i], n[i]);
        PerformanceCounter::addOpCount(PerformanceCounter::TRSM, batch_gops * 1e-9);
#endif
        H2OPUS_FBL_SIDE h2opus_fbl_side = (side == H2Opus_Left ? H2OpusFBLLeft : H2OpusFBLRight);
        H2OPUS_FBL_UPLO h2opus_fbl_uplo = (uplo == H2Opus_Upper ? H2OpusFBLUpper : H2OpusFBLLower);
        H2OPUS_FBL_TRANSPOSE h2opus_fbl_transA = (trans == H2Opus_Trans ? H2OpusFBLTrans : H2OpusFBLNoTrans);
        H2OPUS_FBL_DIAG h2opus_fbl_diag = (diag == H2Opus_Unit ? H2OpusFBLUnit : H2OpusFBLNonUnit);

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
        {
            if (A_ptrs[i] && B_ptrs[i] && m[i] > 0 && n[i] > 0)
            {
                h2opus_fbl_trsm(h2opus_fbl_side, h2opus_fbl_uplo, h2opus_fbl_transA, h2opus_fbl_diag, m[i], n[i], alpha,
                                A_ptrs[i], lda[i], B_ptrs[i], ldb[i]);
            }
        }
        H2OPUS_END_BATCH_BLAS()

        return 1;
    }

    // QR
    static inline int geqrf(h2opusComputeStream_t stream, int m, int n, T *A_strided, int lda, int stride_a,
                            T *tau_strided, int stride_tau, int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0)
            return 1;

#ifdef H2OPUS_PROFILING_ENABLED
        PerformanceCounter::addOpCount(PerformanceCounter::QR, H2OPUS_QR_OP_COUNT(m, n) * 1e-9 * batchCount);
#endif
        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
        {
            T *A_op = A_strided + i * stride_a;
            T *tau_op = tau_strided + i * stride_tau;
            H2OPUS_FBL_CHK_CALL(
                h2opus_fbl_geqrf(m, n, A_op, lda, tau_op, stream->getFBLWorkspace(omp_get_thread_num())));
        }
        H2OPUS_END_BATCH_BLAS()
        return 1;
    }

    static inline int geqrf(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs, int *lda,
                            T **tau_ptrs, int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return 1;

#ifdef H2OPUS_PROFILING_ENABLED
        double batch_gops = 0;
        for (int i = 0; i < batchCount; i++)
            batch_gops += H2OPUS_QR_OP_COUNT(m[i], n[i]);
        PerformanceCounter::addOpCount(PerformanceCounter::QR, batch_gops * 1e-9);
#endif
        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
        {
            T *A_op = A_ptrs[i];
            T *tau_op = tau_ptrs[i];
            int m_op = m[i], n_op = n[i], lda_op = lda[i];

            if (!A_op || !tau_op)
                continue;

            H2OPUS_FBL_CHK_CALL(
                h2opus_fbl_geqrf(m_op, n_op, A_op, lda_op, tau_op, stream->getFBLWorkspace(omp_get_thread_num())));
        }
        H2OPUS_END_BATCH_BLAS()
        return 1;
    }

    static inline int orgqr(h2opusComputeStream_t stream, int m, int n, T *A_strided, int lda, int stride_a,
                            T *tau_strided, int stride_tau, int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0)
            return 1;

#ifdef H2OPUS_PROFILING_ENABLED
        PerformanceCounter::addOpCount(PerformanceCounter::QR, H2OPUS_QR_OP_COUNT(m, n) * 1e-9 * batchCount);
#endif
        int rank = (m < n ? m : n);

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
        {
            T *A_op = A_strided + i * stride_a;
            T *tau_op = tau_strided + i * stride_tau;
            H2OPUS_FBL_CHK_CALL(
                h2opus_fbl_orgqr(m, rank, rank, A_op, lda, tau_op, stream->getFBLWorkspace(omp_get_thread_num())));
        }
        H2OPUS_END_BATCH_BLAS()
        return 1;
    }

    static inline int orgqr(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs, int *lda,
                            T **tau_ptrs, int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return 1;

#ifdef H2OPUS_PROFILING_ENABLED
        double batch_gops = 0;
        for (int i = 0; i < batchCount; i++)
            batch_gops += H2OPUS_QR_OP_COUNT(m[i], n[i]);
        PerformanceCounter::addOpCount(PerformanceCounter::QR, batch_gops * 1e-9);
#endif

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
        {
            T *A_op = A_ptrs[i];
            T *tau_op = tau_ptrs[i];
            int m_op = m[i], n_op = n[i], lda_op = lda[i];

            if (!A_op || !tau_op)
                continue;

            int rank = (m_op < n_op ? m_op : n_op);
            H2OPUS_FBL_CHK_CALL(h2opus_fbl_orgqr(m_op, rank, rank, A_op, lda_op, tau_op,
                                                 stream->getFBLWorkspace(omp_get_thread_num())));
        }
        H2OPUS_END_BATCH_BLAS()
        return 1;
    }

    static inline int orgqr(h2opusComputeStream_t stream, int m, int n, T **A_ptrs, int lda, T **tau_ptrs,
                            int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0)
            return 1;

#ifdef H2OPUS_PROFILING_ENABLED
        PerformanceCounter::addOpCount(PerformanceCounter::QR, H2OPUS_QR_OP_COUNT(m, n) * 1e-9 * batchCount);
#endif

        int rank = (m < n ? m : n);

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
        {
            T *A_op = A_ptrs[i];
            T *tau_op = tau_ptrs[i];
            if (!A_op || !tau_op)
                continue;

            H2OPUS_FBL_CHK_CALL(
                h2opus_fbl_orgqr(m, rank, rank, A_op, lda, tau_op, stream->getFBLWorkspace(omp_get_thread_num())));
        }
        H2OPUS_END_BATCH_BLAS()
        return 1;
    }

    static inline int tsqrf(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs, int *lda,
                            T **tau_ptrs, int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return 1;

#ifdef H2OPUS_PROFILING_ENABLED
        double batch_gops = 0;
        for (int i = 0; i < batchCount; i++)
            batch_gops += H2OPUS_QR_OP_COUNT(m[i], n[i]);
        PerformanceCounter::addOpCount(PerformanceCounter::QR, batch_gops * 1e-9);
#endif
        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
        {
            if (m[i] > 0 && n[i] > 0 && A_ptrs[i] && tau_ptrs[i])
            {
                H2OPUS_FBL_CHK_CALL(h2opus_fbl_geqrf(m[i], n[i], A_ptrs[i], lda[i], tau_ptrs[i],
                                                     stream->getFBLWorkspace(omp_get_thread_num())));
            }
        }
        H2OPUS_END_BATCH_BLAS()
        return 1;
    }

    static inline int geqp2(h2opusComputeStream_t stream, int m, int n, T *A_strided, int lda, int stride_a,
                            T *tau_strided, int stride_tau, int *ranks, T eps, int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0)
        {
            for (int k = 0; k < batchCount; k++)
                ranks[k] = 0;
            return 1;
        }

#ifdef H2OPUS_PROFILING_ENABLED
        PerformanceCounter::addOpCount(PerformanceCounter::QR, H2OPUS_QR_OP_COUNT(m, n) * 1e-9 * batchCount);
#endif
        int max_rank = (m < n ? m : n);

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        {
            int *jpvt = new int[n];
#ifdef H2OPUS_USE_FLAME
            T zeroval = std::numeric_limits<T>::min();
#else
            T zeroval = 0;
#endif
#pragma omp for
            for (int i = 0; i < batchCount; i++)
            {
                T *A_op = A_strided + i * stride_a;
                T *tau_op = tau_strided + i * stride_tau;
                for (int j = 0; j < n; j++)
                    jpvt[j] = 0;

                H2OPUS_FBL_CHK_CALL(
                    h2opus_fbl_geqp3(m, n, A_op, lda, jpvt, tau_op, stream->getFBLWorkspace(omp_get_thread_num())));

                // Assumes the diagonals are ordered, but lapack does not specify it!
                // There are issues with libflame (AMD)
                int k = 0;
                while (k < max_rank && fabs(A_op[k + k * lda]) * sqrt(n - k) > eps)
                    k++;
                ranks[i] = k;
                // There are issues with libflame (AMD): do not use 0.0, it generates nans
                for (; k < max_rank; k++)
                    tau_op[k] = zeroval;
            }
            delete[] jpvt;
        }
        H2OPUS_END_BATCH_BLAS()
        return 1;
    }

    // SVD
    static inline int gesvd(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs, int *lda,
                            T **U_ptrs, int *ldu, T **S_ptrs, T **V_ptrs, int *ldv, T eps, int *ranks, int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return 1;

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        {
            T *superb = new T[std::min(max_m, max_n)];

#pragma omp for schedule(runtime)
            for (int i = 0; i < batchCount; i++)
            {
                T *A_op = A_ptrs[i], *U_op = U_ptrs[i], *S_op = S_ptrs[i], *V_op = V_ptrs[i];
                int m_op = m[i], n_op = n[i], lda_op = lda[i], ldu_op = ldu[i], ldv_op = ldv[i];

                H2OPUS_FBL_CHK_CALL(h2opus_fbl_gesvd(m_op, n_op, A_op, lda_op, S_op, U_op, ldu_op, V_op, ldv_op, superb,
                                                     stream->getFBLWorkspace(omp_get_thread_num())));

                int max_rank = std::min(m_op, n_op);
                int k = 0;
                while (k < max_rank && S_op[k] > eps)
                    k++;
                ranks[i] = k;
            }

            delete[] superb;
        }
        H2OPUS_END_BATCH_BLAS()
        return 1;
    }

    static inline int diagMult(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs, int *lda,
                               T *R_diag, int stride_R, int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return 1;

#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
        {
            T *A_op = A_ptrs[i];
            T *R_op = R_diag + i * stride_R;
            int lda_op = lda[i], m_op = m[i], n_op = n[i];

            if (!A_op || m_op == 0 || n_op == 0)
                continue;

            int rank = (m_op < n_op ? m_op : n_op);
            for (int j = 0; j < rank; j++)
                R_op[j] *= fabs(A_op[j + j * lda_op]);
        }
        return 1;
    }

    // Copies
    static inline int copy_upper(h2opusComputeStream_t stream, int m, int n, T *A_strided, int lda, int stride_a,
                                 T *R_strided, int ldr, int stride_r, int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0)
            return 1;

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
        {
            T *A_op = A_strided + i * stride_a;
            T *R_op = R_strided + i * stride_r;

            H2OPUS_FBL_CHK_CALL(
                h2opus_fbl_lacpy(H2OpusFBLUpper, m, n, A_op, CPU_BATCH_LD(lda), R_op, CPU_BATCH_LD(ldr)));
        }
        H2OPUS_END_BATCH_BLAS()
        return 1;
    }

    static inline int copyBlock(h2opusComputeStream_t stream, int m, int n, T **dest_ptrs, int dest_row_offset,
                                int dest_col_offset, int ld_dest, T **src_ptrs, int src_row_offset, int src_col_offset,
                                int ld_src, int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0)
            return 1;

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
        {
            if (src_ptrs[i] && dest_ptrs[i])
            {
                T *A_op = src_ptrs[i] + src_row_offset + src_col_offset * ld_src;
                T *B_op = dest_ptrs[i] + dest_row_offset + dest_col_offset * ld_dest;

                H2OPUS_FBL_CHK_CALL(
                    h2opus_fbl_lacpy(H2OpusFBLAll, m, n, A_op, CPU_BATCH_LD(ld_src), B_op, CPU_BATCH_LD(ld_dest)));
            }
        }
        H2OPUS_END_BATCH_BLAS()
        return 1;
    }

    static inline int copyBlock(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **dest_ptrs,
                                int *ld_dest, T **src_ptrs, int *ld_src, int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return 1;

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
        {
            int lda = ld_src[i], ldb = ld_dest[i];
            T *A_op = src_ptrs[i], *B_op = dest_ptrs[i];
            int m_op = m[i], n_op = n[i];

            if (A_op && B_op && m_op > 0 && n_op > 0)
            {
                H2OPUS_FBL_CHK_CALL(
                    h2opus_fbl_lacpy(H2OpusFBLAll, m_op, n_op, A_op, CPU_BATCH_LD(lda), B_op, CPU_BATCH_LD(ldb)));
            }
        }
        H2OPUS_END_BATCH_BLAS()
        return 1;
    }

    static inline int copyBlock(h2opusComputeStream_t stream, int m, int n, T *dest_strided, int dest_row_offset,
                                int dest_col_offset, int ld_dest, int dest_stride, T *src_strided, int src_row_offset,
                                int src_col_offset, int ld_src, int src_stride, int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0)
            return 1;

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
        {
            T *A_op = src_strided + src_stride * i + src_row_offset + src_col_offset * ld_src;
            T *B_op = dest_strided + dest_stride * i + dest_row_offset + dest_col_offset * ld_dest;

            H2OPUS_FBL_CHK_CALL(
                h2opus_fbl_lacpy(H2OpusFBLAll, m, n, A_op, CPU_BATCH_LD(ld_src), B_op, CPU_BATCH_LD(ld_dest)));
        }
        H2OPUS_END_BATCH_BLAS()
        return 1;
    }

    // Transpose
    static inline int transpose(h2opusComputeStream_t stream, int m, int n, T **A_ptrs, int lda, T **At_ptrs, int ldat,
                                int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0)
            return 1;

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
        {
            if (A_ptrs[i] && At_ptrs[i])
                blas_transpose(m, n, A_ptrs[i], lda, At_ptrs[i], ldat);
        }
        H2OPUS_END_BATCH_BLAS()
        return 1;
    }

    static inline int transpose(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs,
                                int *lda, T **At_ptrs, int *ldat, int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0)
            return 1;

        H2OPUS_BEGIN_BATCH_BLAS()
#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int i = 0; i < batchCount; i++)
        {
            if (m[i] > 0 && n[i] > 0 && A_ptrs[i] && At_ptrs[i])
                blas_transpose(m[i], n[i], A_ptrs[i], lda[i], At_ptrs[i], ldat[i]);
        }
        H2OPUS_END_BATCH_BLAS()
        return 1;
    }

    // Matrix addition
    // C = alpha * A + beta * B
    static inline void add_matrix(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T alpha,
                                  T **A_ptrs, int *lda, T beta, T **B_ptrs, int *ldb, T **C_ptrs, int *ldc,
                                  int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return;

#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            T *A_op = A_ptrs[op], *B_op = B_ptrs[op], *C_op = C_ptrs[op];
            int m_op = m[op], n_op = n[op];
            int lda_op = lda[op], ldb_op = ldb[op], ldc_op = ldc[op];

            if (A_op && B_op && C_op && m_op > 0 && n_op > 0)
            {
                for (int j = 0; j < n_op; j++)
                    for (int i = 0; i < m_op; i++)
                        C_op[i + j * ldc_op] = alpha * A_op[i + j * lda_op] + beta * B_op[i + j * ldb_op];
            }
        }
    }

    // Set block to identity
    static inline void setIdentity(h2opusComputeStream_t stream, int m, int n, T **A_ptrs, int lda, int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0)
            return;

#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            T *A_op = A_ptrs[op];
            if (A_op)
            {
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < m; i++)
                        A_op[i + j * lda] = (i == j ? (T)1 : (T)0);
            }
        }
    }

    static inline void setDiagonal(h2opusComputeStream_t stream, int m, int n, T **A_ptrs, int lda, T value,
                                   int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0)
            return;

#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            T *A_op = A_ptrs[op];
            if (A_op)
            {
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < m; i++)
                        A_op[i + j * lda] = (i == j ? value : (T)0);
            }
        }
    }

    static inline void setIdentity(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs,
                                   int *lda, int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return;

#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            T *A_op = A_ptrs[op];
            if (A_op)
            {
                int m_op = m[op];
                int n_op = n[op];
                int lda_op = lda[op];

                for (int j = 0; j < n_op; j++)
                    for (int i = 0; i < m_op; i++)
                        A_op[i + j * lda_op] = (i == j ? (T)1 : (T)0);
            }
        }
    }

    static inline void setDiagonal(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs,
                                   int *lda, T value, int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return;

#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            T *A_op = A_ptrs[op];
            if (A_op)
            {
                int m_op = m[op];
                int n_op = n[op];
                int lda_op = lda[op];

                for (int j = 0; j < n_op; j++)
                    for (int i = 0; i < m_op; i++)
                        A_op[i + j * lda_op] = (i == j ? value : (T)0);
            }
        }
    }

    // Set block to zero
    static inline void setZero(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs, int *lda,
                               int batchCount)
    {
        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return;

#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            T *A_op = A_ptrs[op];
            if (A_op)
            {
                int m_op = m[op];
                int n_op = n[op];
                int lda_op = lda[op];

                for (int j = 0; j < n_op; j++)
                    for (int i = 0; i < m_op; i++)
                        A_op[i + j * lda_op] = 0;
            }
        }
    }

    // Set upper triangular half to zero
    static inline int setUpperZero(h2opusComputeStream_t stream, int m, int n, T **A_ptrs, int lda, int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0)
            return 1;

#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
        for (int op = 0; op < batchCount; op++)
        {
            T *A_op = A_ptrs[op];
            if (A_op)
            {
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < j; i++)
                        A_op[i + j * lda] = 0;
            }
        }

        return 1;
    }

    // Batch random number generation
    static inline int rand(h2opusComputeStream_t stream, h2opusHandle_t handle, int *m, int *n, int max_m, int max_n,
                           T **A_ptrs, int *lda, int batchCount)
    {
        const int minBatchcount = 10;

        if (batchCount == 0 || max_m == 0 || max_n == 0)
            return 1;

        std::vector<H2OpusHostRandState> &rand_state = handle->getHostRandState();
        int num_states = (int)rand_state.size();

        if (batchCount >= minBatchcount)
        {
#ifdef _OPENMP
            int op_inc = std::min(stream->getMaxOmpThreads(), num_states);
#else
            int op_inc = num_states;
#endif
            for (int op_start = 0; op_start < batchCount; op_start += op_inc)
            {
                int ops_left = std::min(op_inc, batchCount - op_start);

#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), batchCount))
                for (int op = 0; op < ops_left; op++)
                {
                    int op_id = op_start + op;
                    T *A_op = A_ptrs[op_id];

                    H2OpusHostRandState &state = rand_state[op];

                    if (A_op)
                    {
                        int m_op = m[op_id];
                        int n_op = n[op_id];
                        int lda_op = lda[op_id];

#ifdef H2OPUS_USE_MKL
                        for (int j = 0; j < n_op; j++)
                            vRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state, m_op, A_op + j * lda_op, 0, 1);
#elif defined(H2OPUS_USE_NEC)
                        for (int j = 0; j < n_op; j++)
                            check_asl_error(asl_rand_gen(state, m_op, A_op + j * lda_op));
#elif defined(H2OPUS_USE_AMDRNG)
                        for (int j = 0; j < n_op; j++)
                            check_rng_error(rng_rand_gen(state, m_op, A_op + j * lda_op));
#elif defined(H2OPUS_USE_ESSL)
                        for (int j = 0; j < n_op; j++)
                            essl_rand_gen(state, m_op, A_op + j * lda_op);

#else
                        std::normal_distribution<T> dist;
                        for (int j = 0; j < n_op; j++)
                        {
                            T *A_col = A_op + j * lda_op;
                            for (int i = 0; i < m_op; i++)
                                A_col[i] = dist(state);
                        }
#endif
                    }
                }
            }
        }
        else
        {
            int block_size = std::min(num_states, max_n);
            int block_cols = (max_n + block_size - 1) / block_size;

            for (int op = 0; op < batchCount; op++)
            {
                T *A_op = A_ptrs[op];
                if (A_op)
                {
                    int m_op = m[op];
                    int n_op = n[op];
                    int lda_op = lda[op];

                    for (int jb = 0; jb < block_cols; jb++)
                    {
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), block_size))
                        for (int k = 0; k < block_size; k++)
                        {
                            int j = k + jb * block_size;
                            if (j < n_op)
                            {
                                H2OpusHostRandState &state = rand_state[k];
                                T *A_col = A_op + j * lda_op;
#ifdef H2OPUS_USE_MKL
                                vRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state, m_op, A_col, 0, 1);
#elif defined(H2OPUS_USE_NEC)
                                check_asl_error(asl_rand_gen(state, m_op, A_col));
#elif defined(H2OPUS_USE_AMDRNG)
                                check_rng_error(rng_rand_gen(state, m_op, A_col));
#elif defined(H2OPUS_USE_ESSL)
                                essl_rand_gen(state, m_op, A_col);
#else
                                std::normal_distribution<T> dist;
                                for (int i = 0; i < m_op; i++)
                                    A_col[i] = dist(state);
#endif
                            }
                        }
                    }
                }
            }
        }

        return 1;
    }

    static inline int rand(h2opusComputeStream_t stream, h2opusHandle_t handle, int m, int n, T *A_strided, int lda,
                           int stride_a, int batchCount)
    {
        const int minBatchcount = 10;

        if (batchCount == 0 || m == 0 || n == 0)
            return 1;

        std::vector<H2OpusHostRandState> &rand_state = handle->getHostRandState();
        int num_states = (int)rand_state.size();

        if (batchCount >= minBatchcount)
        {
#ifdef _OPENMP
            int op_inc = std::min(stream->getMaxOmpThreads(), num_states);
#else
            int op_inc = num_states;
#endif
            for (int op_start = 0; op_start < batchCount; op_start += op_inc)
            {
                int ops_left = std::min(op_inc, batchCount - op_start);

#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), ops_left))
                for (int op = 0; op < ops_left; op++)
                {
                    int op_id = op_start + op;
                    T *A_op = A_strided + op_id * stride_a;

                    H2OpusHostRandState &state = rand_state[op];

#ifdef H2OPUS_USE_MKL
                    for (int j = 0; j < n; j++)
                        vRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state, m, A_op + j * lda, 0, 1);
#elif defined(H2OPUS_USE_NEC)
                    for (int j = 0; j < n; j++)
                        check_asl_error(asl_rand_gen(state, m, A_op + j * lda));
#elif defined(H2OPUS_USE_AMDRNG)
                    for (int j = 0; j < n; j++)
                        check_rng_error(rng_rand_gen(state, m, A_op + j * lda));
#elif defined(H2OPUS_USE_ESSL)
                    for (int j = 0; j < n; j++)
                        essl_rand_gen(state, m, A_op + j * lda);
#else
                    std::normal_distribution<T> dist;
                    for (int j = 0; j < n; j++)
                    {
                        T *A_col = A_op + j * lda;
                        for (int i = 0; i < m; i++)
                            A_col[i] = dist(state);
                    }
#endif
                }
            }
        }
        else
        {
            int block_size = std::min(num_states, n);
            int block_cols = (n + block_size - 1) / block_size;

            for (int op = 0; op < batchCount; op++)
            {
                T *A_op = A_strided + op * stride_a;

                for (int jb = 0; jb < block_cols; jb++)
                {
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), block_size))
                    for (int k = 0; k < block_size; k++)
                    {
                        int j = k + jb * block_size;
                        if (j < n)
                        {
                            H2OpusHostRandState &state = rand_state[k];
                            T *A_col = A_op + j * lda;
#ifdef H2OPUS_USE_MKL
                            vRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state, m, A_col, 0, 1);
#elif defined(H2OPUS_USE_NEC)
                            check_asl_error(asl_rand_gen(state, m, A_col));
#elif defined(H2OPUS_USE_AMDRNG)
                            check_rng_error(rng_rand_gen(state, m, A_col));
#elif defined(H2OPUS_USE_ESSL)
                            essl_rand_gen(state, m, A_col);
#else
                            std::normal_distribution<T> dist;
                            for (int i = 0; i < m; i++)
                                A_col[i] = dist(state);
#endif
                        }
                    }
                }
            }
        }

        return 1;
    }
};

#undef CPU_BATCH_LD
#ifndef _OPENMP
#undef omp_get_thread_num
#endif
#endif
