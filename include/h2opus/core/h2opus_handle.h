#ifndef __H2OPUS_HANDLE_H__
#define __H2OPUS_HANDLE_H__

#include <h2opusconf.h>

#include <h2opus/core/h2opus_compute_stream.h>
#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/h2opus_workspace.h>

#include <thrust/random.h>

#define H2OPUS_HOST_RAND_STATES 64

#ifdef H2OPUS_USE_GPU
typedef struct KBlasRandState *kblasRandState_t;
#endif

struct H2OpusHandle
{
  private:
    H2OpusWorkspace allocated_ws;
    h2opusComputeStream_t main_stream, secondary_stream, low_priority_stream;
    std::vector<thrust::minstd_rand> host_rand_state;

    void init();

#ifdef H2OPUS_USE_GPU
    kblasRandState_t rand_state;
#endif
  public:
    H2OpusHandle();
    ~H2OpusHandle();

    h2opusComputeStream_t getMainStream()
    {
        return main_stream;
    }
    h2opusComputeStream_t getSecondaryStream()
    {
        return secondary_stream;
    }
    h2opusComputeStream_t getLowPriorityStream()
    {
        return low_priority_stream;
    }
    h2opusWorkspace_t getWorkspace()
    {
        return &allocated_ws;
    }
    std::vector<thrust::minstd_rand> &getHostRandState();
    H2OpusWorkspaceState getWorkspaceState();
    void setWorkspaceState(H2OpusWorkspaceState &ws_state);
    void *getPtrs(int hw);
    void *getData(int hw);

#ifdef H2OPUS_USE_GPU
    void setKblasRandState(kblasRandState_t rand_state);
    kblasRandState_t getKblasRandState();
#endif
};

typedef H2OpusHandle *h2opusHandle_t;
void h2opusCreateHandle(h2opusHandle_t *h2opus_handle);
void h2opusDestroyHandle(h2opusHandle_t h2opus_handle);

#endif
