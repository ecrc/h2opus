#ifndef __ERROR_APPROXIMATION_H__
#define __ERROR_APPROXIMATION_H__

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/hgemv.h>
#include <h2opus/core/hmatrix_sampler.h>

#include <h2opus/util/batch_wrappers.h>
#include <h2opus/util/blas_wrappers.h>
#include <h2opus/util/thrust_wrappers.h>

#define H2OPUS_NORM_APPROX_THRESHOLD 1e-3

template <class H2Opus_Real, int hw>
inline H2Opus_Real normalize_vector(h2opusComputeStream_t stream, H2Opus_Real *x, int n)
{
    H2Opus_Real norm = blas_norm2<H2Opus_Real, hw>(stream, n, x, 1);
    blas_scal<H2Opus_Real, hw>(stream, n, (H2Opus_Real)1.0 / norm, x, 1);
    return norm;
}

template <class H2Opus_Real, int hw> inline void random_vector(h2opusHandle_t handle, H2Opus_Real *x, int n)
{
    H2OpusBatched<H2Opus_Real, hw>::rand(handle->getMainStream(), handle, n, 1, x, n, n, 1);
}

template <class H2Opus_Real, int hw>
inline H2Opus_Real sampler_norm(HMatrixSampler *sampler, int n, int samples, h2opusHandle_t h2opus_handle)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    h2opusComputeStream_t stream = h2opus_handle->getMainStream();

    H2Opus_Real norm = 0;
    RealVector vec_x1, vec_x2;
    resizeThrustArray(vec_x1, n);
    resizeThrustArray(vec_x2, n);

    H2Opus_Real *x1 = vec_ptr(vec_x1), *x2 = vec_ptr(vec_x2);

    random_vector<H2Opus_Real, hw>(h2opus_handle, x1, n);
    normalize_vector<H2Opus_Real, hw>(stream, x1, n);

    H2Opus_Real prev_norm = 0;

    for (int i = 0; i < samples; i++)
    {
        sampler->sample(x1, x2, 1);
        sampler->sample(x2, x1, 1);

        norm = sqrt(normalize_vector<H2Opus_Real, hw>(stream, x1, n));
        if (norm != 0 && abs(prev_norm - norm) / norm < H2OPUS_NORM_APPROX_THRESHOLD)
            break;
        prev_norm = norm;
    }
    return norm;
}

template <class H2Opus_Real, int hw>
inline H2Opus_Real sampler_1_norm(HMatrixSampler *sampler, int n, int samples, h2opusHandle_t h2opus_handle)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    h2opusComputeStream_t h2opus_stream = h2opus_handle->getMainStream();

    RealVector vec_x, vec_y, vec_e, vec_z;

    resizeThrustArray(vec_x, n);
    resizeThrustArray(vec_y, n);
    resizeThrustArray(vec_e, n);
    resizeThrustArray(vec_z, n);

    H2Opus_Real *x = vec_ptr(vec_x), *y = vec_ptr(vec_y);
    H2Opus_Real *e = vec_ptr(vec_e), *z = vec_ptr(vec_z);

    fillArray(x, n, (H2Opus_Real)1.0 / n, h2opus_stream, hw);

    H2Opus_Real norm = 0;

    for (int i = 0; i < samples; i++)
    {
        // y = A * x
        sampler->sample(x, y, 1);
        // e = sign(y);
        signVector(y, e, n, h2opus_stream, hw);
        // z = A * e
        sampler->sample(e, z, 1);

        // [inf_norm, j] = argmax(abs(z))
        H2Opus_Real inf_norm_z;
        size_t argmax_j;
        argMaxAbsElement(z, n, h2opus_stream, hw, inf_norm_z, argmax_j);

        // Convergence test: inf_norm <= z'*x1
        H2Opus_Real dp = blas_dot_product<H2Opus_Real, hw>(h2opus_stream, n, z, 1, x, 1);
        if (inf_norm_z <= dp)
        {
            norm = reduceAbsSum(y, n, h2opus_stream, hw);
            break;
        }

        // x = e_j (all zeros except x(j) = 1
        standardBasisVector(x, n, argmax_j, h2opus_stream, hw);
    }

    return norm;
}

template <class H2Opus_Real, int hw>
inline H2Opus_Real hmatrix_norm(THMatrix<hw> &hmatrix, int samples, h2opusHandle_t h2opus_handle)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    h2opusComputeStream_t h2opus_stream = h2opus_handle->getMainStream();

    int n = hmatrix.n;
    H2Opus_Real norm = 0;
    RealVector vec_x1, vec_x2;
    resizeThrustArray(vec_x1, n);
    resizeThrustArray(vec_x2, n);

    H2Opus_Real *x1 = vec_ptr(vec_x1), *x2 = vec_ptr(vec_x2);

    random_vector<H2Opus_Real, hw>(h2opus_handle, x1, n);
    normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n);

    H2Opus_Real prev_norm = 0;

    for (int i = 0; i < samples; i++)
    {
        hgemv(H2Opus_NoTrans, 1, hmatrix, x1, n, 0, x2, n, 1, h2opus_handle);
        hgemv(H2Opus_NoTrans, 1, hmatrix, x2, n, 0, x1, n, 1, h2opus_handle);

        norm = sqrt(normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n));

        if (norm != 0 && abs(prev_norm - norm) / norm < H2OPUS_NORM_APPROX_THRESHOLD)
            break;
        prev_norm = norm;
    }
    return norm;
}

template <class H2Opus_Real, int hw>
inline H2Opus_Real sampler_difference(HMatrixSampler *sampler, THMatrix<hw> &hmatrix, int samples,
                                      h2opusHandle_t h2opus_handle)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    h2opusComputeStream_t h2opus_stream = h2opus_handle->getMainStream();

    H2Opus_Real norm = 0;
    int n = hmatrix.n;

    RealVector vec_x1, vec_x2;
    resizeThrustArray(vec_x1, n);
    resizeThrustArray(vec_x2, n);

    H2Opus_Real *x1 = vec_ptr(vec_x1), *x2 = vec_ptr(vec_x2);

    random_vector<H2Opus_Real, hw>(h2opus_handle, x1, n);
    normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n);

    H2Opus_Real prev_norm = 0;

    for (int i = 0; i < samples; i++)
    {
        sampler->sample(x1, x2, 1);
        hgemv(H2Opus_NoTrans, 1, hmatrix, x1, n, -1, x2, n, 1, h2opus_handle);

        sampler->sample(x2, x1, 1);
        hgemv(H2Opus_NoTrans, 1, hmatrix, x2, n, -1, x1, n, 1, h2opus_handle);

        norm = sqrt(normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n));
        if (norm != 0 && abs(prev_norm - norm) / norm < H2OPUS_NORM_APPROX_THRESHOLD)
            break;
        prev_norm = norm;
    }
    return norm;
}

template <class H2Opus_Real, int hw>
inline H2Opus_Real inverse_error(THMatrix<hw> &hmatrix, THMatrix<hw> &inverse, int samples,
                                 h2opusHandle_t h2opus_handle)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    h2opusComputeStream_t h2opus_stream = h2opus_handle->getMainStream();

    H2Opus_Real norm = 0;
    int n = hmatrix.n;

    RealVector vec_x1, vec_x2, vec_x3;
    resizeThrustArray(vec_x1, n);
    resizeThrustArray(vec_x2, n);
    resizeThrustArray(vec_x3, n);

    H2Opus_Real *x1 = vec_ptr(vec_x1), *x2 = vec_ptr(vec_x2), *x3 = vec_ptr(vec_x3);

    random_vector<H2Opus_Real, hw>(h2opus_handle, x1, n);
    normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n);

    H2Opus_Real prev_norm = 0;

    for (int i = 0; i < samples; i++)
    {
        hgemv(H2Opus_NoTrans, 1, hmatrix, x1, n, 0, x2, n, 1, h2opus_handle);
        hgemv(H2Opus_NoTrans, 1, inverse, x2, n, 0, x3, n, 1, h2opus_handle);
        blas_axpy<H2Opus_Real, hw>(h2opus_stream, n, -1, x3, 1, x1, 1);

        hgemv(H2Opus_NoTrans, 1, hmatrix, x1, n, 0, x2, n, 1, h2opus_handle);
        hgemv(H2Opus_NoTrans, 1, inverse, x2, n, 0, x3, n, 1, h2opus_handle);
        blas_axpy<H2Opus_Real, hw>(h2opus_stream, n, -1, x3, 1, x1, 1);

        norm = sqrt(normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n));
        if (norm != 0 && abs(prev_norm - norm) / norm < H2OPUS_NORM_APPROX_THRESHOLD)
            break;
        prev_norm = norm;
    }
    return norm;
}

template <class H2Opus_Real, int hw>
inline H2Opus_Real pseudo_inverse_error(THMatrix<hw> &hmatrix, THMatrix<hw> &pseudo_inverse, int samples,
                                        h2opusHandle_t h2opus_handle)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    h2opusComputeStream_t h2opus_stream = h2opus_handle->getMainStream();

    H2Opus_Real norm = 0;
    int n = hmatrix.n;

    RealVector vec_x1, vec_x2, vec_x3;
    resizeThrustArray(vec_x1, n);
    resizeThrustArray(vec_x2, n);
    resizeThrustArray(vec_x3, n);

    H2Opus_Real *x1 = vec_ptr(vec_x1), *x2 = vec_ptr(vec_x2);
    H2Opus_Real *x3 = vec_ptr(vec_x3);

    random_vector<H2Opus_Real, hw>(h2opus_handle, x1, n);
    normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n);

    H2Opus_Real prev_norm = 0;

    // X * (I - A * X)
    for (int i = 0; i < samples; i++)
    {
        // x2 = X * (I - A * X) * x1
        hgemv(H2Opus_NoTrans, 1, pseudo_inverse, x1, n, 0, x2, n, 1, h2opus_handle);
        hgemv(H2Opus_NoTrans, 1, hmatrix, x2, n, 0, x3, n, 1, h2opus_handle);
        blas_axpy<H2Opus_Real, hw>(h2opus_stream, n, -1, x3, 1, x1, 1);
        hgemv(H2Opus_NoTrans, 1, pseudo_inverse, x1, n, 0, x2, n, 1, h2opus_handle);

        // x1 = X * (I - A * X) * x2
        hgemv(H2Opus_NoTrans, 1, pseudo_inverse, x2, n, 0, x3, n, 1, h2opus_handle);
        hgemv(H2Opus_NoTrans, 1, hmatrix, x3, n, 0, x1, n, 1, h2opus_handle);
        blas_axpy<H2Opus_Real, hw>(h2opus_stream, n, -1, x1, 1, x2, 1);
        hgemv(H2Opus_NoTrans, 1, pseudo_inverse, x2, n, 0, x1, n, 1, h2opus_handle);

        norm = sqrt(normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n));
        if (norm != 0 && abs(prev_norm - norm) / norm < H2OPUS_NORM_APPROX_THRESHOLD)
            break;
        prev_norm = norm;
    }
    return norm;
}

template <class H2Opus_Real, int hw>
inline H2Opus_Real pseudo_inverse_error(HMatrixSampler *sampler, THMatrix<hw> &pseudo_inverse, int samples,
                                        h2opusHandle_t h2opus_handle)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    h2opusComputeStream_t h2opus_stream = h2opus_handle->getMainStream();

    H2Opus_Real norm = 0;
    int n = pseudo_inverse.n;

    RealVector vec_x1, vec_x2, vec_x3;
    resizeThrustArray(vec_x1, n);
    resizeThrustArray(vec_x2, n);
    resizeThrustArray(vec_x3, n);

    H2Opus_Real *x1 = vec_ptr(vec_x1), *x2 = vec_ptr(vec_x2);
    H2Opus_Real *x3 = vec_ptr(vec_x3);

    random_vector<H2Opus_Real, hw>(h2opus_handle, x1, n);
    normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n);

    H2Opus_Real prev_norm = 0;

    // X * (I - A * X)
    for (int i = 0; i < samples; i++)
    {
        // x2 = X * (I - A * X) * x1
        hgemv(H2Opus_NoTrans, 1, pseudo_inverse, x1, n, 0, x2, n, 1, h2opus_handle);
        sampler->sample(x2, x3, 1);
        blas_axpy<H2Opus_Real, hw>(h2opus_stream, n, -1, x3, 1, x1, 1);
        hgemv(H2Opus_NoTrans, 1, pseudo_inverse, x1, n, 0, x2, n, 1, h2opus_handle);

        // x1 = X * (I - A * X) * x2
        hgemv(H2Opus_NoTrans, 1, pseudo_inverse, x2, n, 0, x3, n, 1, h2opus_handle);
        sampler->sample(x3, x1, 1);
        blas_axpy<H2Opus_Real, hw>(h2opus_stream, n, -1, x1, 1, x2, 1);
        hgemv(H2Opus_NoTrans, 1, pseudo_inverse, x2, n, 0, x1, n, 1, h2opus_handle);

        norm = sqrt(normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n));
        if (norm != 0 && abs(prev_norm - norm) / norm < H2OPUS_NORM_APPROX_THRESHOLD)
            break;
        prev_norm = norm;
    }
    return norm;
}

template <class H2Opus_Real, int hw>
inline H2Opus_Real product_norm(THMatrix<hw> &hmatrix, THMatrix<hw> &inverse, int samples, h2opusHandle_t h2opus_handle)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    h2opusComputeStream_t h2opus_stream = h2opus_handle->getMainStream();

    H2Opus_Real norm = 0;
    int n = inverse.n;

    RealVector vec_x1, vec_x2;
    resizeThrustArray(vec_x1, n);
    resizeThrustArray(vec_x2, n);

    H2Opus_Real *x1 = vec_ptr(vec_x1), *x2 = vec_ptr(vec_x2);

    random_vector<H2Opus_Real, hw>(h2opus_handle, x1, n);
    normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n);

    H2Opus_Real prev_norm = 0;

    for (int i = 0; i < samples; i++)
    {
        hgemv(H2Opus_NoTrans, 1, hmatrix, x1, n, 0, x2, n, 1, h2opus_handle);
        hgemv(H2Opus_NoTrans, 1, inverse, x2, n, 0, x1, n, 1, h2opus_handle);

        hgemv(H2Opus_NoTrans, 1, hmatrix, x1, n, 0, x2, n, 1, h2opus_handle);
        hgemv(H2Opus_NoTrans, 1, inverse, x2, n, 0, x1, n, 1, h2opus_handle);

        norm = sqrt(normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n));
        if (norm != 0 && abs(prev_norm - norm) / norm < H2OPUS_NORM_APPROX_THRESHOLD)
            break;
        prev_norm = norm;
    }
    return norm;
}

template <class H2Opus_Real, int hw>
inline H2Opus_Real product_norm(HMatrixSampler *sampler, THMatrix<hw> &inverse, int samples,
                                h2opusHandle_t h2opus_handle)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    h2opusComputeStream_t h2opus_stream = h2opus_handle->getMainStream();

    H2Opus_Real norm = 0;
    int n = inverse.n;

    RealVector vec_x1, vec_x2;
    resizeThrustArray(vec_x1, n);
    resizeThrustArray(vec_x2, n);

    H2Opus_Real *x1 = vec_ptr(vec_x1), *x2 = vec_ptr(vec_x2);

    random_vector<H2Opus_Real, hw>(h2opus_handle, x1, n);
    normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n);

    H2Opus_Real prev_norm = 0;

    for (int i = 0; i < samples; i++)
    {
        sampler->sample(x1, x2, 1);
        hgemv(H2Opus_NoTrans, 1, inverse, x2, n, 0, x1, n, 1, h2opus_handle);

        sampler->sample(x1, x2, 1);
        hgemv(H2Opus_NoTrans, 1, inverse, x2, n, 0, x1, n, 1, h2opus_handle);

        norm = sqrt(normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n));
        if (norm != 0 && abs(prev_norm - norm) / norm < H2OPUS_NORM_APPROX_THRESHOLD)
            break;
        prev_norm = norm;
    }
    return norm;
}

template <class H2Opus_Real, int hw>
inline H2Opus_Real inverse_error(HMatrixSampler *sampler, THMatrix<hw> &inverse, int samples,
                                 h2opusHandle_t h2opus_handle)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    h2opusComputeStream_t h2opus_stream = h2opus_handle->getMainStream();

    H2Opus_Real norm = 0;
    int n = inverse.n;

    RealVector vec_x1, vec_x2, vec_x3;
    resizeThrustArray(vec_x1, n);
    resizeThrustArray(vec_x2, n);
    resizeThrustArray(vec_x3, n);

    H2Opus_Real *x1 = vec_ptr(vec_x1), *x2 = vec_ptr(vec_x2), *x3 = vec_ptr(vec_x3);

    random_vector<H2Opus_Real, hw>(h2opus_handle, x1, n);
    normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n);

    H2Opus_Real prev_norm = 0;

    for (int i = 0; i < samples; i++)
    {
        sampler->sample(x1, x2, 1);
        hgemv(H2Opus_NoTrans, 1, inverse, x2, n, 0, x3, n, 1, h2opus_handle);
        blas_axpy<H2Opus_Real, hw>(h2opus_stream, n, -1, x3, 1, x1, 1);

        sampler->sample(x1, x2, 1);
        hgemv(H2Opus_NoTrans, 1, inverse, x2, n, 0, x3, n, 1, h2opus_handle);
        blas_axpy<H2Opus_Real, hw>(h2opus_stream, n, -1, x3, 1, x1, 1);

        norm = sqrt(normalize_vector<H2Opus_Real, hw>(h2opus_stream, x1, n));
        if (norm != 0 && abs(prev_norm - norm) / norm < H2OPUS_NORM_APPROX_THRESHOLD)
            break;
        prev_norm = norm;
    }
    return norm;
}

#endif
