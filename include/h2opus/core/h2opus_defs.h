#ifndef __H2OPUS_DEFS_H__
#define __H2OPUS_DEFS_H__

#include <h2opusconf.h>

#ifndef H2OPUS_USE_GPU
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#endif

#include <thrust/host_vector.h>
#include <vector>

#ifdef H2OPUS_DOUBLE_PRECISION
typedef double H2Opus_Real;
#else
typedef float H2Opus_Real;
#endif

#define H2OPUS_HWTYPE_CPU 0
#define H2OPUS_HWTYPE_GPU 1

#define H2Opus_Trans 'T'
#define H2Opus_NoTrans 'N'

#define H2OPUS_EMPTY_NODE -1

// Some thrust magic to get things running on the CPU and GPU with the same code
template <int hw, typename T> struct VectorContainer;

#ifndef H2OPUS_USE_GPU
template <typename T> struct VectorContainer<H2OPUS_HWTYPE_CPU, T>
{
    typedef std::vector<T> type;
};
#else
#include <thrust/device_vector.h>
template <typename T> struct VectorContainer<H2OPUS_HWTYPE_CPU, T>
{
    typedef thrust::host_vector<T> type;
};
template <typename T> struct VectorContainer<H2OPUS_HWTYPE_GPU, T>
{
    typedef thrust::device_vector<T> type;
};
#endif

template <class Vector> struct TTreeContainer
{
    typedef std::vector<Vector> type;
};
template <class Vector> struct VectorArray
{
    typedef std::vector<Vector> type;
};
#define vec_ptr(v) thrust::raw_pointer_cast((v.data()))

#endif
