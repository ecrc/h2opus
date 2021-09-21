#ifndef __H2OPUS_HANDLE_H__
#define __H2OPUS_HANDLE_H__

#include <h2opusconf.h>

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/h2opus_compute_stream.h>
#include <h2opus/core/h2opus_workspace.h>
#include <h2opus/core/events.h>

#define H2OPUS_HOST_RAND_STATES 64

#include <random>
#ifdef H2OPUS_USE_MKL
#ifndef MKL_INT
#define MKL_INT int
#endif
#include <mkl_vsl.h>
typedef VSLStreamStatePtr H2OpusHostRandState;
#elif defined(H2OPUS_USE_NEC)
#include <asl.h>
typedef asl_random_t H2OpusHostRandState;
#define check_asl_error(f)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        asl_error_t e = (f);                                                                                           \
        if (e != ASL_ERROR_OK)                                                                                         \
            fprintf(stderr, "ASL error %d in %s at %s:%d\n", e, __func__, __LINE__, __FILE__);                         \
    } while (0)
#elif defined(H2OPUS_USE_AMDRNG)
/* rng.h define blas, we need a small subset of
 * rng functionality, we declare it ourselves */
#include <h2opus/util/amdrngwrap.h>
typedef rng_state_t H2OpusHostRandState;
#elif defined(H2OPUS_USE_ESSL)
#include <h2opus/util/esslrngwrap.h>
typedef essl_rndstate_t H2OpusHostRandState;
#else
typedef std::mt19937 H2OpusHostRandState;
#endif

#ifdef H2OPUS_USE_GPU
typedef struct KBlasRandState *kblasRandState_t;
#endif

struct H2OpusHandle
{
  private:
    H2OpusWorkspace allocated_ws;
    h2opusComputeStream_t main_stream, secondary_stream, low_priority_stream;
    std::vector<H2OpusHostRandState> host_rand_state;
    H2OpusEvents events;

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
    H2OpusEvents &getEvents()
    {
        return events;
    }

    std::vector<H2OpusHostRandState> &getHostRandState();

    H2OpusWorkspaceState getWorkspaceState();
    void setWorkspaceState(H2OpusWorkspaceState &ws_state);
    void *getPtrs(int hw);
    void *getData(int hw);
    void setRandSeed(unsigned int seed, int hw);

#ifdef H2OPUS_USE_GPU
    void setKblasRandState(kblasRandState_t rand_state);
    kblasRandState_t getKblasRandState();
#endif
};

typedef H2OpusHandle *h2opusHandle_t;
void h2opusCreateHandle(h2opusHandle_t *h2opus_handle);
void h2opusDestroyHandle(h2opusHandle_t h2opus_handle);

#endif
