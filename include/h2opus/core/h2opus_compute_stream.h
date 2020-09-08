#ifndef __H2OPUS_COMPUTE_STREAM_H__
#define __H2OPUS_COMPUTE_STREAM_H__

#include <h2opusconf.h>

#ifdef H2OPUS_USE_GPU

#include <cublas_v2.h>
#include <kblas.h>
#include <magma.h>

#include <h2opus/util/gpu_err_check.h>

struct H2OpusComputeStream
{
    kblasHandle_t kblas_handle;
    magma_queue_t magma_queue;
    cudaStream_t stream;
    int current_device;

    H2OpusComputeStream()
    {
        kblas_handle = NULL;
        magma_queue = NULL;
        stream = 0;
        current_device = 0;
    }

    ~H2OpusComputeStream()
    {
        cleanup();
    }

    void cleanup()
    {
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

        if (stream)
        {
            gpuErrchk(cudaStreamDestroy(stream));
            stream = 0;
        }
        current_device = 0;
    }

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

    cudaStream_t getCudaStream()
    {
        return stream;
    }

    void init(int current_device, bool low_priority, bool default_stream)
    {
        cleanup();

        this->current_device = current_device;

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
            else
            {
                printf("No priority support!\n");
            }
        }
        else
        {
            if (!default_stream)
                gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }

        kblasCreate(&kblas_handle);
        kblasSetStream(kblas_handle, stream);

        // Enable magma for better gemm performance and non-uniform gemms
        kblasEnableMagma(kblas_handle);
        kblas_gemm_batch_nonuniform_wsquery(kblas_handle);
        kblasAllocateWorkspace(kblas_handle);

        // Can't get magma queue from kblas, so just make a new one
        // running on the same stream
        magma_queue_create_from_cuda(current_device, stream, kblasGetCublasHandle(kblas_handle), NULL, &magma_queue);
    }
};

#else
struct H2OpusComputeStream
{
};
#endif

typedef struct H2OpusComputeStream *h2opusComputeStream_t;

#endif
