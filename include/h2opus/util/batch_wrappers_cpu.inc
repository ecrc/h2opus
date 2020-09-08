#ifndef __H2OPUS_BATCH_WRAPPERS_CPU__
#define __H2OPUS_BATCH_WRAPPERS_CPU__

//////////////////////////////////////////
// CPU wrappers
//////////////////////////////////////////
template <class T> struct H2OpusBatched<T, H2OPUS_HWTYPE_CPU>
{
    // GEMMS
    static inline int gemm(h2opusComputeStream_t stream, char transA, char transB, const int m, const int n, const int k,
                           const T alpha, const T **A, int lda, const T **B, int ldb, const T beta, T **C, int ldc,
                           int batchCount)
    {
        if (batchCount == 0 || m == 0 || n == 0 || k == 0)
            return 1;

#ifdef H2OPUS_PROFILING_ENABLED
        double batch_gops = (double)(H2OPUS_GEMM_OP_COUNT(m, n, k) * 1e-9) * batchCount;
        PerformanceCounter::addOpCount(PerformanceCounter::GEMM, batch_gops);
#endif
        CBLAS_TRANSPOSE cblas_transA = (transA == H2Opus_Trans ? CblasTrans : CblasNoTrans);
        CBLAS_TRANSPOSE cblas_transB = (transB == H2Opus_Trans ? CblasTrans : CblasNoTrans);

        // See if we should use gemv instead of gemm
        if (transB == H2Opus_NoTrans && n == 1)
        {
            if (transA == H2Opus_Trans)
            {
#pragma omp parallel for
                for (int i = 0; i < batchCount; i++)
                    if (A[i] && B[i] && C[i])
                        cblas_gemv(cblas_transA, k, m, alpha, A[i], lda, B[i], beta, C[i]);
            }
            else
            {
#pragma omp parallel for
                for (int i = 0; i < batchCount; i++)
                    if (A[i] && B[i] && C[i])
                        cblas_gemv(cblas_transA, m, k, alpha, A[i], lda, B[i], beta, C[i]);
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < batchCount; i++)
                if (A[i] && B[i] && C[i])
                    cblas_gemm(cblas_transA, cblas_transB, m, n, k, alpha, A[i], lda, B[i], ldb, beta, C[i], ldc);
        }
        return 1;
    }

    static inline int gemm(h2opusComputeStream_t stream, char transA, char transB, int *m, int *n, int *k, int max_m,
                           int max_n, int max_k, const T alpha, const T **A, int *lda, const T **B, int *ldb,
                           const T beta, T **C, int *ldc, int batchCount)
    {
        if (batchCount == 0)
            return 1;

#ifdef H2OPUS_PROFILING_ENABLED
        double batch_gops = (double)(H2OPUS_GEMM_OP_COUNT(max_m, max_n, max_k) * 1e-9) * batchCount;
        PerformanceCounter::addOpCount(PerformanceCounter::GEMM, batch_gops);
#endif
        CBLAS_TRANSPOSE cblas_transA = (transA == H2Opus_Trans ? CblasTrans : CblasNoTrans);
        CBLAS_TRANSPOSE cblas_transB = (transB == H2Opus_Trans ? CblasTrans : CblasNoTrans);

        // See if we should use gemv instead of gemm
        if (transB == H2Opus_NoTrans && max_n == 1)
        {
            if (transA == H2Opus_Trans)
            {
#pragma omp parallel for
                for (int i = 0; i < batchCount; i++)
                    if (A[i] && B[i] && C[i] && k[i] > 0 && m[i] > 0 && n[i] > 0)
                        cblas_gemv(cblas_transA, k[i], m[i], alpha, A[i], lda[i], B[i], beta, C[i]);
            }
            else
            {
#pragma omp parallel for
                for (int i = 0; i < batchCount; i++)
                    if (A[i] && B[i] && C[i] && k[i] > 0 && m[i] > 0 && n[i] > 0)
                        cblas_gemv(cblas_transA, m[i], k[i], alpha, A[i], lda[i], B[i], beta, C[i]);
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < batchCount; i++)
                if (A[i] && B[i] && C[i] && k[i] > 0 && m[i] > 0 && n[i] > 0)
                    cblas_gemm(cblas_transA, cblas_transB, m[i], n[i], k[i], alpha, A[i], lda[i], B[i], ldb[i], beta,
                               C[i], ldc[i]);
        }
        return 1;
    }

    // Mixed precision Syrk
    static inline int mp_syrk(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, const T **A_ptrs,
                              int *lda, double **B_ptrs, int *ldb, int batchCount)
    {
        if (batchCount == 0)
            return 1;

#pragma omp parallel for
        for (int op = 0; op < batchCount; op++)
        {
            const T *A_op = A_ptrs[op];
            double *B_op = B_ptrs[op];

            if (!A_op || !B_op)
                continue;

            int m_op = m[op], n_op = n[op];
            int lda_op = lda[op], ldb_op = ldb[op];

            for (int i = 0; i < n_op; i++)
            {
                for (int j = 0; j < n_op; j++)
                {
                    double dp = 0;
                    for (int k = 0; k < m_op; k++)
                        dp += (double)A_op[k + i * lda_op] * (double)A_op[k + j * lda_op];
                    B_op[i + j * ldb_op] = dp;
                }
            }
        }

        return 1;
    }

    // Mixed precision Cholesky
    static inline int mp_fused_potrf(h2opusComputeStream_t stream, int *m, int max_m, double **A_ptrs, int *lda,
                                     T **R_ptrs, int *ldr, double *R_diag, int *block_ranks, int batchCount)
    {
        if (batchCount == 0)
            return 1;

#pragma omp parallel for
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

    // TRSM
    static inline int trsm(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **B_ptrs, int *ldb,
                           T **A_ptrs, int *lda, int batchCount)
    {
#pragma omp parallel for
        for (int i = 0; i < batchCount; i++)
        {
            if (A_ptrs[i] && B_ptrs[i] && m[i] > 0 && n[i] > 0)
            {
                cblas_trsm(CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, m[i], n[i], 1, A_ptrs[i], lda[i],
                           B_ptrs[i], ldb[i]);
            }
        }

        return 1;
    }

    // QR
    static inline int geqrf(h2opusComputeStream_t stream, int m, int n, T *A_strided, int lda, int stride_a,
                            T *tau_strided, int stride_tau, int batchCount)
    {
#ifdef H2OPUS_PROFILING_ENABLED
        double qr_gflops = (double)((2 * m * n * n - 2.0 / 3.0 * n * n * n) * 1e-9) * batchCount;
        PerformanceCounter::addOpCount(PerformanceCounter::QR, qr_gflops);
#endif
#pragma omp parallel for
        for (int i = 0; i < batchCount; i++)
        {
            T *A_op = A_strided + i * stride_a;
            T *tau_op = tau_strided + i * stride_tau;

            int ret_val = lapack_geqrf(m, n, A_op, lda, tau_op);
            if (ret_val != 0)
                printf("GEQRF failure (%d) on operation %d: %d %d %d %p\n", ret_val, i, m, n, lda, A_op);
        }
        return 1;
    }

    static inline int orgqr(h2opusComputeStream_t stream, int m, int n, T *A_strided, int lda, int stride_a,
                            T *tau_strided, int stride_tau, int batchCount)
    {
#ifdef H2OPUS_PROFILING_ENABLED
        double qr_gflops = (double)((2 * m * n * n - 2.0 / 3.0 * n * n * n) * 1e-9) * batchCount;
        PerformanceCounter::addOpCount(PerformanceCounter::QR, qr_gflops);
#endif
        int rank = (m < n ? m : n);

#pragma omp parallel for
        for (int i = 0; i < batchCount; i++)
        {
            T *A_op = A_strided + i * stride_a;
            T *tau_op = tau_strided + i * stride_tau;
            int ret_val = lapack_orgqr(m, rank, rank, A_op, lda, tau_op);
            if (ret_val != 0)
                printf("ORGQR failure (%d) on operation %d: %d %d %d %d %p\n", ret_val, i, m, n, rank, lda, A_op);
        }
        return 1;
    }

    static inline int orgqr(h2opusComputeStream_t stream, int m, int n, T **A_ptrs, int lda, T **tau_ptrs, int batchCount)
    {
#ifdef H2OPUS_PROFILING_ENABLED
        double qr_gflops = (double)((2 * m * n * n - 2.0 / 3.0 * n * n * n) * 1e-9) * batchCount;
        PerformanceCounter::addOpCount(PerformanceCounter::QR, qr_gflops);
#endif
        if (m == 0 || n == 0)
            return 1;

        int rank = (m < n ? m : n);

#pragma omp parallel for
        for (int i = 0; i < batchCount; i++)
        {
            T *A_op = A_ptrs[i];
            T *tau_op = tau_ptrs[i];
            if (!A_op || !tau_op)
                continue;

            int ret_val = lapack_orgqr(m, rank, rank, A_op, lda, tau_op);
            if (ret_val != 0)
                printf("ORGQR failure (%d) on operation %d: %d %d %d %d %p\n", ret_val, i, m, n, rank, lda, A_op);
        }
        return 1;
    }

    static inline int tsqrf(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs, int *lda,
                            T **tau_ptrs, int batchCount)
    {
#ifdef H2OPUS_PROFILING_ENABLED
        double qr_gflops =
            (double)((2 * max_m * max_n * max_n - 2.0 / 3.0 * max_n * max_n * max_n) * 1e-9) * batchCount;
        PerformanceCounter::addOpCount(PerformanceCounter::QR, qr_gflops);
#endif
#pragma omp parallel for
        for (int i = 0; i < batchCount; i++)
        {
            if (m[i] > 0 && n[i] > 0 && A_ptrs[i] && tau_ptrs[i])
            {
                int ret_val = lapack_geqrf(m[i], n[i], A_ptrs[i], lda[i], tau_ptrs[i]);
                if (ret_val != 0)
                    printf("TSQRF failure (%d) on operation %d: %d %d %d %p\n", ret_val, i, m[i], n[i], lda[i],
                           A_ptrs[i]);
            }
        }
        return 1;
    }

    static inline int geqp2(h2opusComputeStream_t stream, int m, int n, T *A_strided, int lda, int stride_a,
                            T *tau_strided, int stride_tau, int *ranks, T eps, int batchCount)
    {
#ifdef H2OPUS_PROFILING_ENABLED
        double qr_gflops = (double)((2 * m * n * n - 2.0 / 3.0 * n * n * n) * 1e-9) * batchCount;
        PerformanceCounter::addOpCount(PerformanceCounter::QR, qr_gflops);
#endif
        if (m == 0 || n == 0)
            return 1;

#pragma omp parallel for
        for (int i = 0; i < batchCount; i++)
        {
            T *A_op = A_strided + i * stride_a;
            T *tau_op = tau_strided + i * stride_tau;
            std::vector<int> jpvt(n, 0);

            // geqp2(m, n, A_op, lda, tau_op, &jpvt[0], ranks[i], eps);

            int ret_val = lapack_geqp3(m, n, A_op, lda, &jpvt[0], tau_op);
            int k = 0;
            while (k < n && fabs(A_op[k + k * lda]) * sqrt(n - k) > eps)
                k++;
            ranks[i] = k;
            for (k = ranks[i]; k < n; k++)
                tau_op[k] = 0;

            if (ret_val != 0)
                printf("GEQP3 failure (%d) on operation %d: %d %d %d %p\n", ret_val, i, m, n, lda, A_op);
        }
        return 1;
    }

    // Copies
    static inline int copy_upper(h2opusComputeStream_t stream, int m, int n, T *A_strided, int lda, int stride_a,
                                 T *R_strided, int ldr, int stride_r, int batchCount)
    {
        if (m == 0 || n == 0)
            return 1;

#pragma omp parallel for
        for (int i = 0; i < batchCount; i++)
        {
            T *A_op = A_strided + i * stride_a;
            T *R_op = R_strided + i * stride_r;

            int ret_val = lapack_lacpy('U', m, n, A_op, lda, R_op, ldr);
            if (ret_val != 0)
                printf("LACPY failure (%d) on operation %d\n", ret_val, i);
        }
        return 1;
    }

    static inline int copyBlock(h2opusComputeStream_t stream, int m, int n, T **dest_ptrs, int dest_row_offset,
                                int dest_col_offset, int ld_dest, T **src_ptrs, int src_row_offset, int src_col_offset,
                                int ld_src, int batchCount)
    {
        if (m == 0 || n == 0)
            return 1;

#pragma omp parallel for
        for (int i = 0; i < batchCount; i++)
        {
            if (src_ptrs[i] && dest_ptrs[i])
            {
                T *A_op = src_ptrs[i] + src_row_offset + src_col_offset * ld_src;
                T *B_op = dest_ptrs[i] + dest_row_offset + dest_col_offset * ld_dest;

                int ret_val = lapack_lacpy('A', m, n, A_op, ld_src, B_op, ld_dest);
                if (ret_val != 0)
                    printf("LACPY failure (%d) on operation %d\n", ret_val, i);
            }
        }
        return 1;
    }

    static inline int copyBlock(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **dest_ptrs,
                                int *ld_dest, T **src_ptrs, int *ld_src, int batchCount)
    {
        if (max_m == 0 || max_n == 0)
            return 1;

#pragma omp parallel for
        for (int i = 0; i < batchCount; i++)
        {
            int lda = ld_src[i], ldb = ld_dest[i];
            T *A_op = src_ptrs[i], *B_op = dest_ptrs[i];
            int m_op = m[i], n_op = n[i];

            if (A_op && B_op && m_op > 0 && n_op > 0)
            {
                int ret_val = lapack_lacpy('A', m_op, n_op, A_op, lda, B_op, ldb);
                if (ret_val != 0)
                    printf("LACPY failure (%d) on operation %d\n", ret_val, i);
            }
        }
        return 1;
    }

    static inline int copyBlock(h2opusComputeStream_t stream, int m, int n, T *dest_strided, int dest_row_offset,
                                int dest_col_offset, int ld_dest, int dest_stride, T *src_strided, int src_row_offset,
                                int src_col_offset, int ld_src, int src_stride, int batchCount)
    {
        if (m == 0 || n == 0)
            return 1;

#pragma omp parallel for
        for (int i = 0; i < batchCount; i++)
        {
            T *A_op = src_strided + src_stride * i + src_row_offset + src_col_offset * ld_src;
            T *B_op = dest_strided + dest_stride * i + dest_row_offset + dest_col_offset * ld_dest;

            int ret_val = lapack_lacpy('A', m, n, A_op, ld_src, B_op, ld_dest);
            if (ret_val != 0)
                printf("LACPY failure (%d) on operation %d\n", ret_val, i);
        }
        return 1;
    }

    // Transpose
    static inline int transpose(h2opusComputeStream_t stream, int m, int n, T **A_ptrs, int lda, T **At_ptrs, int ldat,
                                int batchCount)
    {
        if (m == 0 || n == 0)
            return 1;

#pragma omp parallel for
        for (int i = 0; i < batchCount; i++)
        {
            if (A_ptrs[i] && At_ptrs[i])
                blas_transpose(m, n, A_ptrs[i], lda, At_ptrs[i], ldat);
        }
        return 1;
    }

    static inline int transpose(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T **A_ptrs, int *lda,
                                T **At_ptrs, int *ldat, int batchCount)
    {
        if (max_m == 0 || max_n == 0)
            return 1;

#pragma omp parallel for
        for (int i = 0; i < batchCount; i++)
        {
            if (m[i] > 0 && n[i] > 0 && A_ptrs[i] && At_ptrs[i])
                blas_transpose(m[i], n[i], A_ptrs[i], lda[i], At_ptrs[i], ldat[i]);
        }
        return 1;
    }

    // Matrix addition
    // C = alpha * A + beta * B
    static inline void add_matrix(h2opusComputeStream_t stream, int *m, int *n, int max_m, int max_n, T alpha, T **A_ptrs,
                                  int *lda, T beta, T **B_ptrs, int *ldb, T **C_ptrs, int *ldc, int batchCount)
    {
        if (max_m == 0 || max_n == 0)
            return;

#pragma omp parallel for
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
        if (m == 0 || n == 0)
            return;

#pragma omp parallel for
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
        if (m == 0 || n == 0)
            return;

#pragma omp parallel for
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
        if (max_m == 0 || max_n == 0)
            return;

#pragma omp parallel for
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
        if (max_m == 0 || max_n == 0)
            return;

#pragma omp parallel for
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
        if (max_m == 0 || max_n == 0)
            return;

#pragma omp parallel for
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

    // Batch random number generation
    static inline int rand(h2opusComputeStream_t stream, h2opusHandle_t handle, int *m, int *n, int max_m, int max_n,
                           T **A_ptrs, int *lda, int batchCount)
    {
        if (max_m == 0 || max_n == 0)
            return 1;

        std::vector<thrust::minstd_rand> &rand_state = handle->getHostRandState();
        int num_states = (int)rand_state.size();
        int block_size = std::min(num_states, max_n);

        int block_cols = max_n / block_size;
        if (max_n % block_size != 0)
            block_cols++;

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
#pragma omp parallel for
                    for (int k = 0; k < block_size; k++)
                    {
                        int j = k + jb * block_size;
                        if (j < n_op)
                        {
                            thrust::minstd_rand &state = rand_state[k];
                            generateRandomColumn(A_op + j * lda_op, m_op, state);
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
        if (m == 0 || n == 0)
            return 1;

        std::vector<thrust::minstd_rand> &rand_state = handle->getHostRandState();
        int num_states = (int)rand_state.size();
        int block_size = std::min(num_states, n);

        int block_cols = n / block_size;
        if (n % block_size != 0)
            block_cols++;

        for (int op = 0; op < batchCount; op++)
        {
            T *A_op = A_strided + op * stride_a;

            for (int jb = 0; jb < block_cols; jb++)
            {
#pragma omp parallel for
                for (int k = 0; k < block_size; k++)
                {
                    int j = k + jb * block_size;
                    if (j < n)
                    {
                        thrust::minstd_rand &state = rand_state[k];
                        generateRandomColumn(A_op + j * lda, m, state);
                    }
                }
            }
        }
        return 1;
    }
};

#endif
