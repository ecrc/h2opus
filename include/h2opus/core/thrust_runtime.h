#ifndef __H2OPUS_THRUST_RUNTIME_H__
#define __H2OPUS_THRUST_RUNTIME_H__

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/h2opus_handle.h>

#include <thrust/system/cpp/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>

template <int hw> struct ThrustRuntime;

template <> struct ThrustRuntime<H2OPUS_HWTYPE_CPU>
{
    typedef const thrust::system::omp::detail::par_t runtime;
    static runtime get(h2opusComputeStream_t stream)
    {
        return thrust::system::omp::par;
    }
};

#ifdef H2OPUS_USE_GPU
#include <thrust/system/cuda/execution_policy.h>
template <> struct ThrustRuntime<H2OPUS_HWTYPE_GPU>
{
    // typedef const thrust::system::cuda::detail::execute_on_stream runtime;
    typedef const thrust::cuda_cub::execute_on_stream runtime;
    static runtime get(h2opusComputeStream_t stream)
    {
        return thrust::system::cuda::par.on(stream->getCudaStream());
    }
};
#endif

#endif
