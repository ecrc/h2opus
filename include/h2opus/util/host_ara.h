#ifndef __H2OPUS_HOST_ARA_H__
#define __H2OPUS_HOST_ARA_H__

#include <h2opus/util/blas_wrappers.h>

template <class T> void h2opus_rand_matrix(T *omega, int ld, int m, int n, h2opusHandle_t handle)
{
    std::vector<H2OpusHostRandState> &rand_state = handle->getHostRandState();
    int num_states = (int)rand_state.size();
    int block_size = std::min(num_states, n);
    int block_cols = (n + block_size - 1) / block_size;

    for (int jb = 0; jb < block_cols; jb++)
    {
#pragma omp parallel for
        for (int k = 0; k < block_size; k++)
        {
            int j = k + jb * block_size;
            if (j < n)
            {
                H2OpusHostRandState &state = rand_state[k];
                T *omega_col = omega + j * ld;
#ifdef H2OPUS_USE_MKL
                vRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state, m, omega_col, 0, 1);
#elif defined(H2OPUS_USE_NEC)
                check_asl_error(asl_rand_gen(state, m, omega_col));
#elif defined(H2OPUS_USE_AMDRNG)
                check_rng_error(rng_rand_gen(state, m, omega_col));
#elif defined(H2OPUS_USE_ESSL)
                essl_rand_gen(state, m, omega_col);
#else
                std::normal_distribution<T> dist;
                for (int i = 0; i < m; i++)
                    omega_col[i] = dist(state);
#endif
            }
        }
    }
}

template <class T>
int h2opus_ara(h2opusComputeStream_t stream, int m, int n, T *M, int ldm, T *Q, int ldq, T *B, int ldb, T *Z, int ldz,
               T *R_diag, T eps, int bs, int r, int max_rank, h2opusHandle_t handle)
{
    max_rank = std::min(std::min(m, n), max_rank);
    bs = std::min(bs, max_rank);

    // tau should be of size bs, and Z is of size max_rank * max_rank,
    // so tau should fit easily in Z
    T *tau_hh = Z;

    int rank = 0;
    int small_vectors = 0;

    while (rank < max_rank)
    {
        // generate random input vectors. use B as omega
        T *omega = B;
        h2opus_rand_matrix(omega, ldb, n, bs, handle);

        // Sample: Y = M * omega
        T *Y = Q + rank * ldq;
        blas_gemm<T, H2OPUS_HWTYPE_CPU>(stream, H2Opus_NoTrans, H2Opus_NoTrans, m, bs, n, 1, M, ldm, omega, ldb, 0, Y,
                                        ldq);

        for (int i = 0; i < bs; i++)
            R_diag[i] = 1;

        // BCGS with one reorthogonalization step
        for (int i = 0; i < 2; i++)
        {
            // Project samples
            // Y = Y - Q * (Q' * Y) = Y - Q * Z
            if (rank != 0)
            {
                blas_gemm<T, H2OPUS_HWTYPE_CPU>(stream, H2Opus_Trans, H2Opus_NoTrans, rank, bs, m, 1, Q, ldq, Y, ldq, 0,
                                                Z, ldz);
                blas_gemm<T, H2OPUS_HWTYPE_CPU>(stream, H2Opus_NoTrans, H2Opus_NoTrans, m, bs, rank, 1, Q, ldq, Z, ldz,
                                                -1, Y, ldq);
            }

            // Regular householder QR
            lapack_geqrf<T, H2OPUS_HWTYPE_CPU>(stream, m, bs, Y, ldq, tau_hh);

            // Update R
            for (int j = 0; j < bs; j++)
                R_diag[j] *= fabs(Y[j + j * ldq]);

            // Expand reflectors into orthogonal block column
            lapack_orgqr<T, H2OPUS_HWTYPE_CPU>(stream, m, bs, bs, Y, ldq, tau_hh);
        }

        // Check for convergence
        int wearedone = 0;
        for (int i = 0; i < bs; i++)
        {
            rank++;
            if (R_diag[i] < eps)
            {
                small_vectors++;
                if (small_vectors == r)
                {
                    rank -= r;
                    wearedone = 1;
                    break;
                }
            }
            else
                small_vectors = 0;
        }

        if (wearedone)
            break;
    }

    // Projection: B = M' * Q
    blas_gemm<T, H2OPUS_HWTYPE_CPU>(stream, H2Opus_Trans, H2Opus_NoTrans, n, rank, m, 1, M, ldm, Q, ldq, 0, B, ldb);

    return rank;
}

#endif
