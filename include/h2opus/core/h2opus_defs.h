#ifndef __H2OPUS_DEFS_H__
#define __H2OPUS_DEFS_H__

#include <h2opusconf.h>

#ifndef H2OPUS_USE_GPU
#ifdef H2OPUS_USE_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_TBB
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#elif defined(_OPENMP)
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_OMP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP
#else
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#endif
#endif

#include <thrust/host_vector.h>
#include <vector>

#ifdef H2OPUS_USE_DOUBLE_PRECISION
typedef double H2Opus_Real;
#elif defined(H2OPUS_USE_SINGLE_PRECISION)
typedef float H2Opus_Real;
#endif

#define H2OPUS_HWTYPE_CPU 0
#define H2OPUS_HWTYPE_GPU 1

#define H2Opus_Trans 'T'
#define H2Opus_NoTrans 'N'
#define H2Opus_Lower 'L'
#define H2Opus_Upper 'U'
#define H2Opus_Left 'L'
#define H2Opus_Right 'R'
#define H2Opus_Unit 'U'
#define H2Opus_NonUnit 'N'
#define H2Opus_Symm 'S'
#define H2Opus_NonSymm 'N'

// Some thrust magic to get things running on the CPU and GPU with the same code
template <int hw, typename T> struct VectorContainer;

#ifndef H2OPUS_USE_GPU
template <typename T> struct VectorContainer<H2OPUS_HWTYPE_CPU, T>
{
    typedef thrust::host_vector<T> type;
};
#else
#include <thrust/device_vector.h>
#include <h2opus/util/device_vector.h>
template <typename T> struct VectorContainer<H2OPUS_HWTYPE_CPU, T>
{
    typedef thrust::host_vector<T> type;
};
template <typename T> struct VectorContainer<H2OPUS_HWTYPE_GPU, T>
{
    typedef H2OpusDeviceVector<T> type;
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

// Node types
#define H2OPUS_EMPTY_NODE -1
#define H2OPUS_HMATRIX_RANK_MATRIX 0
#define H2OPUS_HMATRIX_DENSE_MATRIX 1
#define H2OPUS_HMATRIX_INNER_NODE 2

#define H2OPUS_COMPRESSION_BASIS_GEN_MAX_NODES 2000
// #define H2OPUS_HCOMPRESS_USE_CHOLESKY_QR

#endif
