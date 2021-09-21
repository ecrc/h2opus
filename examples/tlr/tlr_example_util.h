#ifndef __TLR_EXAMPLE_UTIL_H__
#define __TLR_EXAMPLE_UTIL_H__

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <functional>

#include <h2opus/util/error_approximation.h>
#include <h2opus/core/tlr/tlr_gemv.h>
#include <h2opus/core/tlr/tlr_trsm.h>

template <class T, int hw> class TLRSampler : public HMatrixSampler
{
  private:
    TTLR_Matrix<T, hw> &A;
    h2opusHandle_t h2opus_handle;

  public:
    TLRSampler(TTLR_Matrix<T, hw> &tlr_matrix, h2opusHandle_t h2opus_handle) : A(tlr_matrix)
    {
        this->h2opus_handle = h2opus_handle;
    }

    void sample(T *input, T *output, int samples)
    {
        tlr_gemv<T, hw>(H2Opus_NoTrans, 1, A, input, A.getPaddedDim(), 0, output, A.getPaddedDim(), samples, 2,
                        h2opus_handle);
    }
};

template <class T, int hw> class TLRCholErrorSampler : public HMatrixSampler
{
  private:
    typedef typename VectorContainer<hw, T>::type RealVector;

    TTLR_Matrix<T, hw> &A, &L;
    h2opusHandle_t h2opus_handle;
    RealVector temp_buffer, perm_input, perm_output;
    int *piv;

  public:
    TLRCholErrorSampler(TTLR_Matrix<T, hw> &tlr_matrix, TTLR_Matrix<T, hw> &chol_factor, int *piv,
                        h2opusHandle_t h2opus_handle)
        : A(tlr_matrix), L(chol_factor)
    {
        this->piv = piv;
        this->h2opus_handle = h2opus_handle;
    }

    void sample(T *input, T *output, int samples)
    {
        h2opusComputeStream_t stream = h2opus_handle->getMainStream();

        if (temp_buffer.size() < (size_t)A.getPaddedDim() * samples)
        {
            temp_buffer.resize(A.getPaddedDim() * samples);
            perm_input.resize(A.getPaddedDim() * samples);
            perm_output.resize(A.getPaddedDim() * samples);
        }

        permute_vectors(input, vec_ptr(perm_input), A.getPaddedDim(), 1, piv, 0, hw, stream);

        // t = L' * input
        tlr_gemv<T, hw>(H2Opus_Trans, 1, L, vec_ptr(perm_input), L.getPaddedDim(), 0, vec_ptr(temp_buffer),
                        L.getPaddedDim(), samples, 2, h2opus_handle);

        // output = L * t
        tlr_gemv<T, hw>(H2Opus_NoTrans, 1, L, vec_ptr(temp_buffer), L.getPaddedDim(), 0, vec_ptr(perm_output),
                        L.getPaddedDim(), samples, 2, h2opus_handle);

        permute_vectors(vec_ptr(perm_output), output, A.getPaddedDim(), 1, piv, 1, hw, stream);

        // output = output - A * input = L * L' * input - A * input
        tlr_gemv<T, hw>(H2Opus_NoTrans, 1, A, input, A.getPaddedDim(), -1, output, A.getPaddedDim(), samples, 2,
                        h2opus_handle);
    }
};

template <class T, int hw> class TLRInverseNormSampler : public HMatrixSampler
{
  private:
    typedef typename VectorContainer<hw, T>::type RealVector;

    TTLR_Matrix<T, hw> &L;
    h2opusHandle_t h2opus_handle;
    RealVector temp_buffer;
    int *piv;

  public:
    TLRInverseNormSampler(TTLR_Matrix<T, hw> &chol_factor, int *piv, h2opusHandle_t h2opus_handle) : L(chol_factor)
    {
        this->piv = piv;
        this->h2opus_handle = h2opus_handle;
    }

    void sample(T *input, T *output, int samples)
    {
        h2opusComputeStream_t stream = h2opus_handle->getMainStream();

        if (temp_buffer.size() < (size_t)L.getPaddedDim() * samples)
            temp_buffer.resize(L.getPaddedDim() * samples);

        permute_vectors(input, vec_ptr(temp_buffer), L.getPaddedDim(), 1, piv, 0, hw, stream);

        // set t = L^{-1} * t
        tlr_trsm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(H2Opus_Left, H2Opus_NoTrans, 1, L, 1, vec_ptr(temp_buffer),
                                                 L.getPaddedDim(), h2opus_handle);

        // set t = L^{-T} * t
        tlr_trsm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(H2Opus_Left, H2Opus_Trans, 1, L, 1, vec_ptr(temp_buffer),
                                                 L.getPaddedDim(), h2opus_handle);

        permute_vectors(vec_ptr(temp_buffer), output, L.getPaddedDim(), 1, piv, 1, hw, stream);
    }
};

template <class T, int hw> class TLRLDLErrorSampler : public HMatrixSampler
{
  private:
    typedef typename VectorContainer<hw, T>::type RealVector;

    TTLR_Matrix<T, hw> &A, &L;
    T *D;
    h2opusHandle_t h2opus_handle;
    RealVector temp_buffer;

  public:
    TLRLDLErrorSampler(TTLR_Matrix<T, hw> &tlr_matrix, TTLR_Matrix<T, hw> &ldl_factor, T *D,
                       h2opusHandle_t h2opus_handle)
        : A(tlr_matrix), L(ldl_factor)
    {
        this->D = D;
        this->h2opus_handle = h2opus_handle;
    }

    void sample(T *input, T *output, int samples)
    {
        if (temp_buffer.size() < (size_t)A.getPaddedDim() * samples)
            temp_buffer.resize(A.getPaddedDim() * samples);

        // t = L' * input
        tlr_gemv<T, hw>(H2Opus_Trans, 1, L, input, L.getPaddedDim(), 0, vec_ptr(temp_buffer), L.getPaddedDim(), samples,
                        2, h2opus_handle);

        for (int i = 0; i < A.getPaddedDim(); i++)
            temp_buffer[i] *= D[i];

        // output = L * t
        tlr_gemv<T, hw>(H2Opus_NoTrans, 1, L, vec_ptr(temp_buffer), L.getPaddedDim(), 0, output, L.getPaddedDim(),
                        samples, 2, h2opus_handle);

        // output = output - A * input = L * L' * input - A * input
        tlr_gemv<T, hw>(H2Opus_NoTrans, 1, A, input, A.getPaddedDim(), -1, output, A.getPaddedDim(), samples, 2,
                        h2opus_handle);
    }
};

template <class T, int hw> T tlr_norm(TTLR_Matrix<T, hw> &A, h2opusHandle_t h2opus_handle)
{
    TLRSampler<T, hw> sampler(A, h2opus_handle);
    return sampler_norm<T, hw>(&sampler, A.getPaddedDim(), 20, h2opus_handle);
}

template <class T, int hw> T tlr_inverse_norm(TTLR_Matrix<T, hw> &L, int *piv, h2opusHandle_t h2opus_handle)
{
    TLRInverseNormSampler<T, hw> sampler(L, piv, h2opus_handle);
    return sampler_norm<T, hw>(&sampler, L.getPaddedDim(), 20, h2opus_handle);
}

template <class T, int hw>
T tlr_chol_error_norm(TTLR_Matrix<T, hw> &A, TTLR_Matrix<T, hw> &L, int *piv, h2opusHandle_t h2opus_handle)
{
    TLRCholErrorSampler<T, hw> sampler(A, L, piv, h2opus_handle);
    return sampler_norm<T, hw>(&sampler, A.getPaddedDim(), 20, h2opus_handle);
}

template <class T, int hw>
T tlr_ldl_error_norm(TTLR_Matrix<T, hw> &A, TTLR_Matrix<T, hw> &L, T *D, h2opusHandle_t h2opus_handle)
{
    TLRLDLErrorSampler<T, hw> sampler(A, L, D, h2opus_handle);
    return sampler_norm<T, hw>(&sampler, A.getPaddedDim(), 20, h2opus_handle);
}

template <class T> void remapInputData(PointCloud<T> &cloud, int *index_map, T *remapped_data)
{
    int dim = cloud.getDimension();
    int n = cloud.getDataSetSize();
    for (int i = 0; i < n; i++)
        for (int d = 0; d < dim; d++)
            remapped_data[i + d * n] = cloud.getDataPoint(index_map[i], d);
}

template <class T> void addScaledIdentitytoTLR(TTLR_Matrix<T, H2OPUS_HWTYPE_CPU> &A, T scale)
{
    for (int b = 0; b < A.n_block; b++)
    {
        int index_start = b * A.block_size;
        T *A_d = A.diagonal_block_ptrs[b];
        for (int i = 0; i < A.block_size; i++)
            if (i + index_start < A.n)
                A_d[i + i * A.block_size] += scale;
    }
}

template <class T> void expandTLRMatrix(TTLR_Matrix<T, H2OPUS_HWTYPE_CPU> &A, T *M)
{
    int n = A.getPaddedDim(); // A.n

    // Copy diagonal blocks
    for (int b = 0; b < A.n_block; b++)
    {
        int index_start = b * A.block_size;
        for (int i = 0; i < A.block_size; i++)
        {
            for (int j = 0; j < A.block_size; j++)
            {
                // if(i + index_start < A.n && j + index_start < A.n)
                M[i + index_start + (j + index_start) * n] = A.diagonal_block_ptrs[b][i + j * A.block_size];
            }
        }
    }

    // Expand low rank blocks
    for (int cb = 0; cb < A.n_block; cb++)
    {
        for (int rb = 0; rb < A.n_block; rb++)
        {
            T *U = A.block_U_ptrs[rb + cb * A.n_block];
            T *V = A.block_V_ptrs[rb + cb * A.n_block];
            int rank = A.block_ranks[rb + cb * A.n_block];

            if (rank == 0 || !U || !V)
                continue;

            T *Mb = M + rb * A.block_size + cb * A.block_size * n;
            // int rows = std::min(A.n - rb * A.block_size, A.block_size), cols = std::min(A.n - cb * A.block_size,
            // A.block_size);
            int rows = A.block_size, cols = A.block_size;
            h2opus_fbl_gemm(H2OpusFBLNoTrans, H2OpusFBLTrans, rows, cols, rank, 1, U, A.block_size, V, A.block_size, 0,
                            Mb, n);

            // printf("Rank (%d %d) = %d\n", rb, cb, rank);
        }
    }
}

template <class T> void expandTLRMatrix(TTLR_Matrix<T, H2OPUS_HWTYPE_GPU> &A, T *M)
{
    // TODO: Fix for padded N
    assert(false);
    // Copy diagonal blocks
    for (int b = 0; b < A.n_block; b++)
    {
        thrust::host_vector<T> diagonal_block =
            copyGPUBlock(vec_ptr(A.diagonal_block_ptrs), b, A.block_size, A.block_size, A.block_size);
        int index_start = b * A.block_size;
        for (int i = 0; i < A.block_size; i++)
            for (int j = 0; j < A.block_size; j++)
                if (i + index_start < A.n && j + index_start < A.n)
                    M[i + index_start + (j + index_start) * A.n] = diagonal_block[i + j * A.block_size];
    }

    // Expand low rank blocks
    thrust::host_vector<int> block_ranks(A.n_block * A.n_block);
    copyVector(block_ranks, vec_ptr(A.block_ranks), A.block_ranks.size(), H2OPUS_HWTYPE_GPU);

    for (int cb = 0; cb < A.n_block; cb++)
    {
        for (int rb = 0; rb < A.n_block; rb++)
        {
            int rank = block_ranks[rb + cb * A.n_block];

            if (rank == 0)
                continue;

            thrust::host_vector<T> U_block =
                copyGPUBlock(vec_ptr(A.block_U_ptrs), rb + cb * A.n_block, A.block_size, A.block_size, rank);
            thrust::host_vector<T> V_block =
                copyGPUBlock(vec_ptr(A.block_V_ptrs), rb + cb * A.n_block, A.block_size, A.block_size, rank);

            T *U = vec_ptr(U_block);
            T *V = vec_ptr(V_block);

            T *Mb = M + rb * A.block_size + cb * A.block_size * A.n;
            int rows = std::min(A.n - rb * A.block_size, A.block_size),
                cols = std::min(A.n - cb * A.block_size, A.block_size);
            h2opus_fbl_gemm(H2OpusFBLNoTrans, H2OpusFBLTrans, rows, cols, rank, 1, U, A.block_size, V, A.block_size, 0,
                            Mb, A.n);

            // printf("Rank (%d %d) = %d\n", rb, cb, rank);
        }
    }
}

template <class T, class TLRMatGen> void gen_dense_matrix(T *M, size_t n, TLRMatGen &mat_gen)
{
    for (size_t j = 0; j < n; j++)
    {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++)
            M[i + j * n] = mat_gen(i, j);
    }
}

template <class T, class TLRMatGen> T abs_tlr_error(T *M, int n, TLRMatGen &mat_gen)
{
    T err = 0;
    for (int j = 0; j < n; j++)
    {
        for (int i = j; i < n; i++)
        {
            T entry_diff = M[i + j * n] - mat_gen(i, j);
            err += entry_diff * entry_diff;
        }
    }
    return sqrt(err);
}

template <class T, int hw> void print_statistics(TTLR_Matrix<T, hw> &A, bool output_ranks = false)
{
    // Memory usage
    double total_gbytes = A.memoryUsage() * (1.0 / (1024 * 1024 * 1024));
    double dense_gbytes = A.denseMemoryUsage() * (1.0 / (1024 * 1024 * 1024));
    printf("Memory consumption = %.4f GB\n", total_gbytes);
    printf("Dense memory consumption = %.4f GB\n", dense_gbytes);
    printf("Low rank memory consumption = %.4f GB\n", total_gbytes - dense_gbytes);

    thrust::host_vector<int> block_ranks(A.n_block * A.n_block);
    copyVector(block_ranks, vec_ptr(A.block_ranks), A.block_ranks.size(), hw);

    // Average rank
    int nb = A.n_block;
    int total_tiles = (nb * nb - nb) / 2;

    size_t rank_sum = 0;
    for (int i = 0; i < nb; i++)
        for (int j = 0; j < i; j++)
            rank_sum += block_ranks[i + j * nb];

    printf("Average rank = %.3f\n", (double)rank_sum / total_tiles);

    // Ranks
    if (output_ranks)
    {
        printf("Ranks = [\n");
        for (int i = 0; i < A.n_block; i++)
        {
            for (int j = 0; j < A.n_block; j++)
                printf("%3d ", block_ranks[i + j * A.n_block]);
            printf(";\n");
        }
        printf("];\n");
    }
}

template <class T, int hw> void print_ranks(TTLR_Matrix<T, hw> &A, const char *filename_base, int gx, int gy, int gz)
{
    std::ostringstream out;
    out << filename_base << "_" << gx << "x" << gy;
    if (gz > 1)
        out << "x" << gz;
    out << ".txt";

    FILE *fp = fopen(out.str().c_str(), "w");
    if (!fp)
    {
        printf("Failed to open file %s for writing\n", out.str().c_str());
        return;
    }

    int nb = A.n_block;

    thrust::host_vector<int> block_ranks(nb * nb);
    copyVector(block_ranks, vec_ptr(A.block_ranks), A.block_ranks.size(), hw);

    thrust::host_vector<int> lower_ranks((nb * nb - nb) / 2);
    int lr_index = 0;
    for (int i = 0; i < nb; i++)
        for (int j = 0; j < i; j++)
            lower_ranks[lr_index++] = block_ranks[i + j * nb];

    std::sort(lower_ranks.begin(), lower_ranks.end(), std::greater<int>());

    // Ranks
    for (int i = 0; i < (int)lower_ranks.size(); i++)
        fprintf(fp, "%d\n", lower_ranks[i]);

    fclose(fp);
}

#endif
