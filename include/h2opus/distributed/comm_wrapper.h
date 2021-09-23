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

#endif
