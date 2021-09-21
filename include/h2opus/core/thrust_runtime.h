#ifndef __H2OPUS_THRUST_RUNTIME_H__
#define __H2OPUS_THRUST_RUNTIME_H__

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/h2opus_handle.h>
#include <h2opus/core/h2opus_compute_stream.h>

#ifdef H2OPUS_USE_TBB
#include <thrust/system/tbb/execution_policy.h>
#else
#ifdef _OPENMP
#include <thrust/system/omp/execution_policy.h>
#else
#include <thrust/system/cpp/execution_policy.h>
#endif
#endif

template <int hw> struct ThrustRuntime;

template <> struct ThrustRuntime<H2OPUS_HWTYPE_CPU>
{
#ifdef H2OPUS_USE_TBB
    typedef const thrust::system::tbb::detail::par_t runtime;
    static inline runtime get(h2opusComputeStream_t stream)
    {
        return thrust::system::tbb::par;
    }
#else
#ifdef _OPENMP
    typedef const thrust::system::omp::detail::par_t runtime;
    static inline runtime get(h2opusComputeStream_t stream)
    {
        return thrust::system::omp::par;
    }
#else
    typedef const thrust::system::cpp::detail::par_t runtime;
    static inline runtime get(h2opusComputeStream_t stream)
    {
        return thrust::system::cpp::par;
    }
#endif
#endif
};

#ifdef H2OPUS_USE_GPU
#include <thrust/system/cuda/execution_policy.h>
template <> struct ThrustRuntime<H2OPUS_HWTYPE_GPU>
{
    // typedef const thrust::system::cuda::detail::execute_on_stream runtime;
    typedef const thrust::cuda_cub::execute_on_stream runtime;
    static inline runtime get(h2opusComputeStream_t stream)
    {
        return thrust::system::cuda::par.on((stream ? stream->getCudaStream() : 0));
    }
};
#endif

#endif
