#include <h2opus/core/h2opus_handle.h>
#include <h2opus/core/h2opus_defs.h>
#include <h2opus/util/gpu_err_check.h>

#ifdef H2OPUS_USE_LIBXSMM
extern "C" void libxsmm_init(void);
#endif
#ifdef H2OPUS_USE_FLAME
extern "C" void FLA_Init(void);
#endif
#ifdef H2OPUS_USE_BLIS
extern "C" void bli_init(void);
#endif

void InitBLASStatic()
{
    h2opus_fbl_init();
#ifdef H2OPUS_USE_BLIS
    bli_init();
#endif
#ifdef H2OPUS_USE_FLAME
    FLA_Init();
#endif
#ifdef H2OPUS_USE_LIBXSMM
    libxsmm_init();
#endif
    { /* make sure the mach parameters are safe */
        double a[16] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.};
        float af[16] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.};
        double t[4];
        float tf[4];
        int j[4] = {0, 0, 0, 0};
        int jf[4] = {0, 0, 0, 0};
        h2opus_fbl_sgeqp3(4, 4, af, 4, jf, tf, NULL);
        h2opus_fbl_dgeqp3(4, 4, a, 4, j, t, NULL);
    }
    return;
}

// TODO
// Singleton?
// Proper shutdown of library resources

H2OpusHandle::H2OpusHandle()
{
    init();
}

H2OpusHandle::~H2OpusHandle()
{
#ifdef H2OPUS_USE_GPU
    kblasDestroyRandState(rand_state);
#endif

    for (size_t i = 0; i < host_rand_state.size(); i++)
    {
#ifdef H2OPUS_USE_MKL
        vslDeleteStream(&host_rand_state[i]);
#elif defined(H2OPUS_USE_NEC)
        check_asl_error(asl_random_destroy(host_rand_state[i]));
#elif defined(H2OPUS_USE_AMDRNG)
        check_rng_error(rng_random_destroy(&host_rand_state[i]));
#elif defined(H2OPUS_USE_ESSL)
        essl_random_destroy(&host_rand_state[i]);
#endif
    }

    delete main_stream;
    delete secondary_stream;
    delete low_priority_stream;
}

void H2OpusHandle::init()
{
    host_rand_state.resize(H2OPUS_HOST_RAND_STATES);

    InitBLASStatic();

#ifdef H2OPUS_USE_MKL
    for (size_t i = 0; i < host_rand_state.size(); i++)
        vslNewStream(&host_rand_state[i], VSL_BRNG_SFMT19937, i + 1);
#elif defined(H2OPUS_USE_NEC)
    asl_library_initialize();
    for (size_t i = 0; i < host_rand_state.size(); i++)
    {
        asl_uint32_t seed = i + 1;
        check_asl_error(asl_random_create(&host_rand_state[i], ASL_RANDOMMETHOD_MT19937));
        check_asl_error(asl_random_distribute_normal(host_rand_state[i], 0, 1));
        check_asl_error(asl_random_initialize(host_rand_state[i], 1, &seed));
    }
#elif defined(H2OPUS_USE_AMDRNG)
    for (size_t i = 0; i < host_rand_state.size(); i++)
    {
        rng_int_t seed = i + 1;
        check_rng_error(rng_random_create(&host_rand_state[i]));
        check_rng_error(rng_random_seed(host_rand_state[i], seed));
    }
#elif defined(H2OPUS_USE_ESSL)
    for (size_t i = 0; i < host_rand_state.size(); i++)
    {
        essl_random_create(&host_rand_state[i]);
        essl_random_seed(host_rand_state[i], (_ESVINT)(i + 1));
    }
#else
    for (size_t i = 0; i < host_rand_state.size(); i++)
        host_rand_state[i] = H2OpusHostRandState(i + 1);
#endif

    main_stream = new H2OpusComputeStream();
    secondary_stream = new H2OpusComputeStream();
    low_priority_stream = new H2OpusComputeStream();

#ifdef H2OPUS_USE_GPU
    rand_state = NULL;
#endif
}

void H2OpusHandle::setRandSeed(unsigned int seed, int hw)
{
    if (hw == H2OPUS_HWTYPE_CPU)
    {
        std::mt19937 seed_gen(seed);
#ifdef H2OPUS_USE_MKL
        for (size_t i = 0; i < host_rand_state.size(); i++)
        {
            vslDeleteStream(&host_rand_state[i]);
            vslNewStream(&host_rand_state[i], VSL_BRNG_SFMT19937, seed_gen());
        }
#elif defined(H2OPUS_USE_NEC)
        for (size_t i = 0; i < host_rand_state.size(); i++)
        {
            asl_uint32_t aseed = seed_gen();
            check_asl_error(asl_random_initialize(host_rand_state[i], 1, &aseed));
        }
#elif defined(H2OPUS_USE_AMDRNG)
        for (size_t i = 0; i < host_rand_state.size(); i++)
        {
            check_rng_error(rng_random_seed(host_rand_state[i], seed_gen()));
        }
#elif defined(H2OPUS_USE_ESSL)
        for (size_t i = 0; i < host_rand_state.size(); i++)
        {
            essl_random_seed(host_rand_state[i], seed_gen());
        }
#else
        for (size_t i = 0; i < host_rand_state.size(); i++)
            host_rand_state[i].seed(seed_gen());
#endif
    }
    else
    {
#ifdef H2OPUS_USE_GPU
        kblasDestroyRandState(rand_state);
        kblasHandle_t mainHandle = main_stream->getKblasHandle();
        kblasInitRandState(mainHandle, &rand_state, 1 << 15, seed);
#endif
    }
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

std::vector<H2OpusHostRandState> &H2OpusHandle::getHostRandState()
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

    int current_device = 0;

#ifdef H2OPUS_USE_GPU
    // Initialize cuda and magma (cudaSetDevice must have been called)
#ifdef H2OPUS_USE_DOUBLE_PRECISION
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif
    magma_init();
    gpuErrchk(cudaGetDevice(&current_device));
#endif

    // Create the streams
    (*h2opus_handle)->getMainStream()->init(current_device, false, true);
    (*h2opus_handle)->getSecondaryStream()->init(current_device, false, false, 1);
    (*h2opus_handle)->getLowPriorityStream()->init(current_device, true, false, 1);

    // Initialize the host random state
    (*h2opus_handle)->setRandSeed(0, H2OPUS_HWTYPE_CPU);
#ifdef H2OPUS_USE_GPU
    // Initialize the device random state
    (*h2opus_handle)->setRandSeed(0, H2OPUS_HWTYPE_GPU);
#endif
}

void h2opusDestroyHandle(h2opusHandle_t h2opus_handle)
{
    delete h2opus_handle;
}
