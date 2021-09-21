#ifndef __H2OPUS_VECTOR_OPS_H__
#define __H2OPUS_VECTOR_OPS_H__

#include <vector>
#include <thrust/host_vector.h>
#include <h2opus/core/h2opus_handle.h>

// TODO: A lot of redundant crap in here. Desperately needs a cleanup

#ifdef H2OPUS_USE_GPU
#include <h2opus/util/device_vector.h>

template <class T> void copyVector(H2OpusDeviceVector<T> &dest, const H2OpusDeviceVector<T> &src)
{
    dest.resize(src.size());
    gpuErrchk(cudaMemcpy(dest.data(), src.data(), src.size() * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <class T> void copyVector(H2OpusDeviceVector<T> &dest, const T *src, size_t num_elements, int src_hw)
{
    dest.resize(num_elements);
    cudaMemcpyKind kind = (src_hw == H2OPUS_HWTYPE_CPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
    gpuErrchk(cudaMemcpy(dest.data(), src, num_elements * sizeof(T), kind));
}

template <class T> void copyVector(H2OpusDeviceVector<T> &dest, const std::vector<T> &src)
{
    dest.resize(src.size());
    gpuErrchk(cudaMemcpy(dest.data(), src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T> void copyVector(std::vector<T> &dest, const H2OpusDeviceVector<T> &src)
{
    dest.resize(src.size());
    gpuErrchk(cudaMemcpy(dest.data(), src.data(), src.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

template <class T> void copyVector(H2OpusDeviceVector<T> &dest, const thrust::host_vector<T> &src)
{
    dest.resize(src.size());
    gpuErrchk(cudaMemcpy(dest.data(), src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T> void copyVector(thrust::host_vector<T> &dest, const H2OpusDeviceVector<T> &src)
{
    dest.resize(src.size());
    gpuErrchk(cudaMemcpy(dest.data(), src.data(), src.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

template <class T> void initVector(H2OpusDeviceVector<T> &v, T val, h2opusComputeStream_t stream)
{
    fillArray(vec_ptr(v), v.size(), val, stream, H2OPUS_HWTYPE_GPU);
}
#endif

template <class T> void copyVector(std::vector<T> &dest, const std::vector<T> &src)
{
    dest = src;
}

template <class T> void copyVector(thrust::host_vector<T> &dest, const thrust::host_vector<T> &src)
{
    dest = src;
}

template <class T> void copyVector(std::vector<T> &dest, const thrust::host_vector<T> &src)
{
    dest.resize(src.size());
    memcpy(dest.data(), src.data(), src.size() * sizeof(T));
}

template <class T> void copyVector(std::vector<T> &dest, const T *src, size_t num_elements, int src_hw)
{
    dest.resize(num_elements);
#ifdef H2OPUS_USE_GPU
    cudaMemcpyKind kind = (src_hw == H2OPUS_HWTYPE_CPU ? cudaMemcpyHostToHost : cudaMemcpyDeviceToHost);
    gpuErrchk(cudaMemcpy(dest.data(), src, num_elements * sizeof(T), kind));
#else
    assert(src_hw == H2OPUS_HWTYPE_CPU);
    memcpy(dest.data(), src, num_elements * sizeof(T));
#endif
}

template <class T> void copyVectorToHost(T *host_dest, const T *src, size_t num_elements, int src_hw)
{
#ifdef H2OPUS_USE_GPU
    cudaMemcpyKind kind = (src_hw == H2OPUS_HWTYPE_CPU ? cudaMemcpyHostToHost : cudaMemcpyDeviceToHost);
    gpuErrchk(cudaMemcpy(host_dest, src, num_elements * sizeof(T), kind));
#else
    assert(src_hw == H2OPUS_HWTYPE_CPU);
    memcpy(host_dest, src, num_elements * sizeof(T));
#endif
}

template <class T> void copyVector(T *dest, int dest_hw, const T *src, int src_hw, size_t num_elements)
{
#ifdef H2OPUS_USE_GPU
    cudaMemcpyKind kind;
    if (dest_hw == H2OPUS_HWTYPE_GPU)
        kind = (src_hw == H2OPUS_HWTYPE_CPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
    else
        kind = (src_hw == H2OPUS_HWTYPE_CPU ? cudaMemcpyHostToHost : cudaMemcpyDeviceToHost);
    gpuErrchk(cudaMemcpy(dest, src, num_elements * sizeof(T), kind));
#else
    assert(src_hw == H2OPUS_HWTYPE_CPU && dest_hw == H2OPUS_HWTYPE_CPU);
    memcpy(dest, src, num_elements * sizeof(T));
#endif
}

template <class T> void copyVector(thrust::host_vector<T> &dest, const T *src, size_t num_elements, int src_hw)
{
    dest.resize(num_elements);
#ifdef H2OPUS_USE_GPU
    cudaMemcpyKind kind = (src_hw == H2OPUS_HWTYPE_CPU ? cudaMemcpyHostToHost : cudaMemcpyDeviceToHost);
    gpuErrchk(cudaMemcpy(dest.data(), src, num_elements * sizeof(T), kind));
#else
    assert(src_hw == H2OPUS_HWTYPE_CPU);
    memcpy(dest.data(), src, num_elements * sizeof(T));
#endif
}

template <class T> void initVector(thrust::host_vector<T> &v, T val, h2opusComputeStream_t stream)
{
    fillArray(vec_ptr(v), v.size(), val, stream, H2OPUS_HWTYPE_CPU);
}

template <class T> void initVector(std::vector<T> &v, T val, h2opusComputeStream_t stream)
{
    fillArray(vec_ptr(v), v.size(), val, stream, H2OPUS_HWTYPE_CPU);
}

template <class T, int hw> T *allocateVector(size_t num_elements)
{
    if (num_elements == 0)
        return NULL;

#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
    {
        T *ptr;
        gpuErrchk(cudaMalloc(&ptr, num_elements * sizeof(T)));
        return ptr;
    }
    else
#endif
    {
        T *ptr = (T *)malloc(num_elements * sizeof(T));
        assert(ptr);
        return ptr;
    }
}

template <class T, int hw> void freeVector(T *ptr)
{
    if (ptr == NULL)
        return;

#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
    {
        gpuErrchk(cudaFree(ptr));
    }
    else
#endif
    {
        free(ptr);
    }
}

#endif
