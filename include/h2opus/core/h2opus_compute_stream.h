#ifndef __H2OPUS_COMPUTE_STREAM_H__
#define __H2OPUS_COMPUTE_STREAM_H__

#include <h2opusconf.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef H2OPUS_USE_GPU
#include <cublas_v2.h>
#include <kblas.h>
#include <magma.h>
#include <h2opus/util/gpu_err_check.h>
#endif
#include <h2opus/util/h2opusfblaslapack.h>

#include <vector>

struct H2OpusComputeStream
{
#ifdef H2OPUS_USE_GPU
    kblasHandle_t kblas_handle;
    magma_queue_t magma_queue;
    cudaStream_t stream;
    cusolverDnHandle_t cusolver_handle;
#endif
    int current_device;
    int maxompthreads;
    std::vector<h2opus_fbl_ctx *> fblws;

    H2OpusComputeStream()
    {
#ifdef H2OPUS_USE_GPU
        kblas_handle = NULL;
        magma_queue = NULL;
        cusolver_handle = NULL;
        stream = 0;
#endif
        current_device = 0;
        maxompthreads = 1;
    }

    ~H2OpusComputeStream()
    {
        cleanup();
    }

    void cleanup()
    {
#ifdef H2OPUS_USE_GPU
        if (magma_queue)
        {
            magma_queue_destroy(magma_queue);
            magma_queue = NULL;
        }

        if (kblas_handle)
        {
            kblasFreeWorkspace(kblas_handle);
            kblasDestroy(&kblas_handle);
            kblas_handle = NULL;
        }

        if (cusolver_handle)
        {
            gpuCusolverErrchk(cusolverDnDestroy(cusolver_handle));
            cusolver_handle = NULL;
        }

        if (stream)
        {
            gpuErrchk(cudaStreamDestroy(stream));
            stream = 0;
        }
#endif
        for (size_t i = 0; i < fblws.size(); i++)
        {
            free(fblws[i]->w);
            delete fblws[i];
        }
        fblws.resize(0);
        current_device = 0;
        maxompthreads = 1;
    }

#ifdef H2OPUS_USE_GPU
    magma_queue_t getMagmaQueue()
    {
        return magma_queue;
    }

    cublasHandle_t getCublasHandle()
    {
        return kblasGetCublasHandle(kblas_handle);
    }

    kblasHandle_t getKblasHandle()
    {
        return kblas_handle;
    }

    cusolverDnHandle_t getCuSolverHandle()
    {
        return cusolver_handle;
    }

    cudaStream_t getCudaStream()
    {
        return stream;
    }
#endif

    void init(int current_device, bool low_priority, bool default_stream, int nthreads = -1)
    {
        cleanup();

        this->current_device = current_device;

#ifdef H2OPUS_USE_GPU
        if (low_priority)
        {
            int least_priority, highest_priority;
            int pri_support;
            gpuErrchk(cudaDeviceGetAttribute(&pri_support, cudaDevAttrStreamPrioritiesSupported, 0));

            if (pri_support)
            {
                gpuErrchk(cudaDeviceGetStreamPriorityRange(&least_priority, &highest_priority));
                gpuErrchk(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, least_priority));
            }
        }
        else
        {
            if (!default_stream)
                gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }

        kblasCreate(&kblas_handle);
        kblasSetStream(kblas_handle, stream);

        // Initialize cusolver handle
        gpuCusolverErrchk(cusolverDnCreate(&cusolver_handle));
        gpuCusolverErrchk(cusolverDnSetStream(cusolver_handle, stream));

        // Enable magma for better gemm performance and non-uniform gemms
        kblasEnableMagma(kblas_handle);
        kblas_gemm_batch_nonuniform_wsquery(kblas_handle);
        kblasAllocateWorkspace(kblas_handle);

        // Can't get magma queue from kblas, so just make a new one
        // running on the same stream
        magma_queue_create_from_cuda(current_device, stream, kblasGetCublasHandle(kblas_handle), NULL, &magma_queue);
#endif
        if (nthreads < 0)
        {
#ifdef _OPENMP
            maxompthreads = omp_get_max_threads();
#endif
        }

        /* workspace reuse for lapack calls */
        for (int i = 0; i < maxompthreads + 1; i++)
        {
            h2opus_fbl_ctx *ctx = new h2opus_fbl_ctx;
            ctx->s = 0;
            ctx->w = NULL;
            fblws.push_back(ctx);
        }
    }

    int getMaxOmpThreads()
    {
        return maxompthreads;
    }

    void setMaxOmpThreads(int n)
    {
        /* workspace reuse for lapack calls */
        for (int i = maxompthreads; i < n; i++)
        {
            h2opus_fbl_ctx *ctx = new h2opus_fbl_ctx;
            ctx->s = 0;
            ctx->w = NULL;
            fblws.push_back(ctx);
        }
        maxompthreads = n;
    }

    h2opus_fbl_ctx *getFBLWorkspace(int i = -1)
    {
        return fblws.at(i + 1);
    }
};

typedef struct H2OpusComputeStream *h2opusComputeStream_t;

#endif
