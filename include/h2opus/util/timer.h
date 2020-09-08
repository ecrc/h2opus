#ifndef __TIMER_H__
#define __TIMER_H__

#include <h2opus/core/h2opus_defs.h>
#include <chrono>

template <int hw> class Timer;

template <> class Timer<H2OPUS_HWTYPE_CPU>
{
  private:
    std::chrono::duration<float> elapsed_time;

    std::chrono::time_point<std::chrono::system_clock> start_time, stop_time;

  public:
    Timer()
    {
        init();
    }

    ~Timer()
    {
    }

    void init()
    {
        start_time = std::chrono::system_clock::now();
        stop_time = start_time;
    }

    void destroy()
    {
    }

    void start()
    {
        start_time = std::chrono::system_clock::now();
    }

    float stop()
    {
        stop_time = std::chrono::system_clock::now();
        elapsed_time = stop_time - start_time;

        return elapsed_time.count();
    }

    float elapsedSec()
    {
        return elapsed_time.count();
    }
};

#ifdef H2OPUS_USE_GPU
#include <h2opus/util/gpu_err_check.h>

template <> class Timer<H2OPUS_HWTYPE_GPU>
{
  private:
    cudaEvent_t start_event, stop_event;
    float elapsed_sec;

  public:
    Timer()
    {
        init();
    }

    ~Timer()
    {
    }

    void init()
    {
#pragma omp critical(create_timer)
        {
            gpuErrchk(cudaEventCreate(&start_event));
            gpuErrchk(cudaEventCreate(&stop_event));
            elapsed_sec = 0;
        }
    }

    void destroy()
    {
#pragma omp critical(delete_timer)
        {
            gpuErrchk(cudaEventDestroy(start_event));
            gpuErrchk(cudaEventDestroy(stop_event));
        }
    }

    void start(cudaStream_t stream = 0)
    {
        gpuErrchk(cudaEventRecord(start_event, stream));
    }

    float stop(cudaStream_t stream = 0)
    {
        gpuErrchk(cudaEventRecord(stop_event, stream));
        gpuErrchk(cudaEventSynchronize(stop_event));

        float time_since_last_start;
        gpuErrchk(cudaEventElapsedTime(&time_since_last_start, start_event, stop_event));
        elapsed_sec = (time_since_last_start * 0.001);

        return elapsed_sec;
    }

    float elapsedSec()
    {
        return elapsed_sec;
    }
};
#endif

#endif
