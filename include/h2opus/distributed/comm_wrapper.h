#ifndef __H2OPUS_COMM_WRAPPER_H__
#define __H2OPUS_COMM_WRAPPER_H__

// Prevent CXX namespace to pollute libraries using H2OPUS
#if !defined(MPICH_SKIP_MPICXX)
#define H2OPUS_UNDEF_SKIP_MPICH 1
#define MPICH_SKIP_MPICXX 1
#endif
#if !defined(OMPI_SKIP_MPICXX)
#define H2OPUS_UNDEF_SKIP_OMPI 1
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>
#if defined(H2OPUS_UNDEF_SKIP_MPICH)
#undef OMPI_MPICH_MPICXX
#endif
#if defined(H2OPUS_UNDEF_SKIP_OMPI)
#undef OMPI_SKIP_MPICXX
#endif

#define mpiErrchk(cmd)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        int e = cmd;                                                                                                   \
        if (e != MPI_SUCCESS)                                                                                          \
        {                                                                                                              \
            int len;                                                                                                   \
            char errstring[MPI_MAX_ERROR_STRING];                                                                      \
            MPI_Error_string(e, errstring, &len);                                                                      \
            fprintf(stderr, "MPI error at %s:%d '%d'\n%s\n", __FILE__, __LINE__, e, errstring);                        \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define H2OPUS_COMM_INT 0
#define H2OPUS_COMM_FLOAT 1
#define H2OPUS_COMM_DOUBLE 2

#ifdef H2OPUS_USE_DOUBLE_PRECISION
#define H2OPUS_COMM_REAL H2OPUS_COMM_DOUBLE
#define H2OPUS_MPI_REAL MPI_DOUBLE
#elif defined(H2OPUS_USE_SINGLE_PRECISION)
#define H2OPUS_COMM_REAL H2OPUS_COMM_FLOAT
#define H2OPUS_MPI_REAL MPI_FLOAT
#endif

#define H2OPUS_MPI_INT MPI_INT

// template<int hw> struct H2OpusCommDataType;
// template<int hw, int type> struct H2OpusCommDataTypeSelect;
//
// template<int hw>
// void h2opus_allGather(void* send_data, void* recv_data, size_t send_count, typename H2OpusCommDataType<hw>::type
// data_type, distributedH2OpusHandle_t dist_h2opus_handle);
//
// // CPU
// template<> struct H2OpusCommDataType<H2OPUS_HWTYPE_CPU>{ typedef MPI_Datatype type; };
// template<> struct H2OpusCommDataTypeSelect<H2OPUS_HWTYPE_CPU, H2OPUS_COMM_INT>  {static constexpr MPI_Datatype val =
// MPI_INT; }; template<> struct H2OpusCommDataTypeSelect<H2OPUS_HWTYPE_CPU, H2OPUS_COMM_FLOAT>  {static constexpr
// MPI_Datatype val = MPI_FLOAT; }; template<> struct H2OpusCommDataTypeSelect<H2OPUS_HWTYPE_CPU, H2OPUS_COMM_DOUBLE>
// {static constexpr MPI_Datatype val = MPI_DOUBLE; };
//
// template<>
// void h2opus_allGather<H2OPUS_HWTYPE_CPU>(void* send_data, void* recv_data, size_t send_count, typename
// H2OpusCommDataType<H2OPUS_HWTYPE_CPU>::type data_type, distributedH2OpusHandle_t dist_h2opus_handle) {
// MPI_Allgather(send_data, send_count, data_type, recv_data, send_count, data_type, MPI_COMM_WORLD);  }
//
//
// // GPU
// #ifdef H2OPUS_USE_GPU
//
// #include <nccl.h>
// template<> struct H2OpusCommDataType<H2OPUS_HWTYPE_GPU>{ typedef ncclDataType_t type; };
// template<> struct H2OpusCommDataTypeSelect<H2OPUS_HWTYPE_GPU, H2OPUS_COMM_INT>  {static constexpr ncclDataType_t val
// = ncclInt; }; template<> struct H2OpusCommDataTypeSelect<H2OPUS_HWTYPE_GPU, H2OPUS_COMM_FLOAT>  {static constexpr
// ncclDataType_t val = ncclFloat; }; template<> struct H2OpusCommDataTypeSelect<H2OPUS_HWTYPE_GPU, H2OPUS_COMM_DOUBLE>
// {static constexpr ncclDataType_t val = ncclDouble; };
//
// template<>
// void h2opus_allGather<H2OPUS_HWTYPE_GPU>(void* send_data, void* recv_data, size_t send_count, typename
// H2OpusCommDataType<H2OPUS_HWTYPE_GPU>::type data_type, distributedH2OpusHandle_t dist_h2opus_handle) {
// ncclAllGather(send_data, recv_data, send_count, data_type, dist_h2opus_handle->nccl_comm,
// dist_h2opus_handle->handle->getKblasStream()); }
//
// #endif

#endif
