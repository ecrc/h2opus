#ifndef __DISTRIBUTED_ERROR_APPROXIMATION_H__
#define __DISTRIBUTED_ERROR_APPROXIMATION_H__

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/util/error_approximation.h>
#include <h2opus/distributed/distributed_hgemv.h>

#include <h2opus/util/batch_wrappers.h>
#include <h2opus/util/blas_wrappers.h>
#include <h2opus/util/thrust_wrappers.h>

template <class H2Opus_Real, int hw>
inline H2Opus_Real distributed_normalize_vector(h2opusComputeStream_t stream, H2Opus_Real *x, int n, MPI_Comm comm)
{
    H2Opus_Real subvec_norm2 = blas_norm2<H2Opus_Real, hw>(stream, n, x, 1);
    subvec_norm2 = subvec_norm2 * subvec_norm2;

    H2Opus_Real norm;
    MPI_Allreduce(&subvec_norm2, &norm, 1, H2OPUS_MPI_REAL, MPI_SUM, comm);
    norm = sqrt(norm);

    blas_scal<H2Opus_Real, hw>(stream, n, (H2Opus_Real)1.0 / norm, x, 1);
    return norm;
}

template <class H2Opus_Real, int hw> inline void distributed_random_vector(h2opusHandle_t handle, H2Opus_Real *x, int n)
{
    H2OpusBatched<H2Opus_Real, hw>::rand(handle->getMainStream(), handle, n, 1, x, n, n, 1);
}

template <class H2Opus_Real, int hw>
inline H2Opus_Real distributed_hmatrix_norm(TDistributedHMatrix<hw> &dist_hmatrix, int samples,
                                            distributedH2OpusHandle_t dist_h2opus_handle)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    h2opusComputeStream_t h2opus_stream = dist_h2opus_handle->handle->getMainStream();
    MPI_Comm comm = dist_h2opus_handle->comm;

    int branch_n = dist_hmatrix.basis_tree.basis_branch.index_map.size();
    H2Opus_Real norm = 0;
    RealVector vec_x1(branch_n), vec_x2(branch_n);

    H2Opus_Real *x1 = vec_ptr(vec_x1), *x2 = vec_ptr(vec_x2);

    random_vector<H2Opus_Real, hw>(dist_h2opus_handle->handle, x1, branch_n);
    distributed_normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, branch_n, comm);

    H2Opus_Real prev_norm = 0;

    for (int i = 0; i < samples; i++)
    {
        distributed_hgemv(1, dist_hmatrix, x1, branch_n, 0, x2, branch_n, 1, dist_h2opus_handle);
        distributed_hgemv(1, dist_hmatrix, x2, branch_n, 0, x1, branch_n, 1, dist_h2opus_handle);

        norm = sqrt(distributed_normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, branch_n, comm));

        if (norm != 0 && abs(prev_norm - norm) / norm < H2OPUS_NORM_APPROX_THRESHOLD)
            break;
        prev_norm = norm;
    }
    return norm;
}

template <class H2Opus_Real, int hw>
inline H2Opus_Real sampler_norm(HMatrixSampler *sampler, int n, int samples,
                                distributedH2OpusHandle_t dist_h2opus_handle)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    h2opusComputeStream_t stream = dist_h2opus_handle->handle->getMainStream();
    MPI_Comm comm = dist_h2opus_handle->comm;

    H2Opus_Real norm = 0;
    RealVector vec_x1, vec_x2;
    vec_x1.resize(n);
    vec_x2.resize(n);

    H2Opus_Real *x1 = vec_ptr(vec_x1), *x2 = vec_ptr(vec_x2);

    random_vector<H2Opus_Real, hw>(dist_h2opus_handle->handle, x1, n);
    distributed_normalize_vector<H2Opus_Real, hw>(stream, x1, n, comm);

    H2Opus_Real prev_norm = 0;

    for (int i = 0; i < samples; i++)
    {
        sampler->sample(x1, x2, 1);
        sampler->sample(x2, x1, 1);

        norm = sqrt(distributed_normalize_vector<H2Opus_Real, hw>(stream, x1, n, comm));
        if (norm != 0 && abs(prev_norm - norm) / norm < H2OPUS_NORM_APPROX_THRESHOLD)
            break;
        prev_norm = norm;
    }
    return norm;
}

#endif
