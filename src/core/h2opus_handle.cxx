#include <h2opus/core/h2opus_handle.h>
#include <h2opus/core/h2opus_defs.h>
#include <h2opus/util/gpu_err_check.h>

H2OpusHandle::H2OpusHandle()
{
    init();
}

H2OpusHandle::~H2OpusHandle()
{
#ifdef H2OPUS_USE_GPU
    kblasDestroyRandState(rand_state);
#endif
    delete main_stream;
    delete secondary_stream;
    delete low_priority_stream;
}

void H2OpusHandle::init()
{
    host_rand_state.resize(H2OPUS_HOST_RAND_STATES);
    for (int i = 0; i < H2OPUS_HOST_RAND_STATES; i++)
        host_rand_state[i] = thrust::minstd_rand(i + 1);

    main_stream = new H2OpusComputeStream();
    secondary_stream = new H2OpusComputeStream();
    low_priority_stream = new H2OpusComputeStream();
}

#ifdef H2OPUS_USE_GPU
void H2OpusHandle::setKblasRandState(kblasRandState_t rand_state)
{
    this->rand_state = rand_state;
}

kblasRandState_t H2OpusHandle::getKblasRandState()
{
    return rand_state;
}
#endif

std::vector<thrust::minstd_rand> &H2OpusHandle::getHostRandState()
{
    return host_rand_state;
}

H2OpusWorkspaceState H2OpusHandle::getWorkspaceState()
{
    return allocated_ws.getWorkspaceState();
}

void H2OpusHandle::setWorkspaceState(H2OpusWorkspaceState &ws_state)
{
    allocated_ws.setWorkspaceState(ws_state);
}

void *H2OpusHandle::getPtrs(int hw)
{
    return allocated_ws.getPtrs(hw);
}

void *H2OpusHandle::getData(int hw)
{
    return allocated_ws.getData(hw);
}

void h2opusCreateHandle(h2opusHandle_t *h2opus_handle)
{
    *h2opus_handle = new H2OpusHandle();
#ifdef H2OPUS_USE_GPU
    int current_device = 0;

// Initalize cuda and magma
#ifdef H2OPUS_DOUBLE_PRECISION
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif
    magma_init();
    gpuErrchk(cudaGetDevice(&current_device));

    // Create the main and low priority stream
    (*h2opus_handle)->getMainStream()->init(current_device, false, true);
    (*h2opus_handle)->getSecondaryStream()->init(current_device, false, false);
    (*h2opus_handle)->getLowPriorityStream()->init(current_device, true, false);

    // Initalize the random state
    kblasRandState_t rand_state;
    kblasHandle_t mainHandle = (*h2opus_handle)->getMainStream()->getKblasHandle();
    kblasInitRandState(mainHandle, &rand_state, 1 << 15, 0);
    (*h2opus_handle)->setKblasRandState(rand_state);
#endif
}

void h2opusDestroyHandle(h2opusHandle_t h2opus_handle)
{
    delete h2opus_handle;
}
