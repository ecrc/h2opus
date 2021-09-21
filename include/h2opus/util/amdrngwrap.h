#ifndef __H2OPUS_AMD_RNG_WRAP_H__
#define __H2OPUS_AMD_RNG_WRAP_H__

#ifdef __cplusplus
extern "C"
{
#endif

    typedef int rng_int_t;
    typedef int rng_strlen_t;
    /* hardcoded for Mersenne-Twister */
    typedef struct
    {
        rng_int_t fstate[633];
        rng_int_t dstate[633];
    } rng_state_t;
    void drandinitialize(rng_int_t genid, rng_int_t subid, rng_int_t *seed, rng_int_t *lseed, rng_int_t *state,
                         rng_int_t *lstate, rng_int_t *info);
    void srandinitialize(rng_int_t genid, rng_int_t subid, rng_int_t *seed, rng_int_t *lseed, rng_int_t *state,
                         rng_int_t *lstate, rng_int_t *info);
    void sranduniform(rng_int_t n, float a, float b, rng_int_t *state, float *x, rng_int_t *info);
    void dranduniform(rng_int_t n, double a, double b, rng_int_t *state, double *x, rng_int_t *info);

#define check_rng_error(f)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        rng_int_t e = (f);                                                                                             \
        if (e)                                                                                                         \
            fprintf(stderr, "RNG error %d in %s at %s:%d\n", e, __func__, __FILE__, __LINE__);                         \
    } while (0)

    inline rng_int_t rng_random_create(rng_state_t *rng)
    {
        return 0;
    }

    inline rng_int_t rng_random_seed(rng_state_t &rng, rng_int_t seed)
    {
        rng_int_t lseed = 1, lstate = 633;
        rng_int_t dinfo, sinfo;
        if (seed < 0)
            seed = -seed;
        drandinitialize(3, 0, &seed, &lseed, rng.dstate, &lstate, &dinfo);
        srandinitialize(3, 0, &seed, &lseed, rng.fstate, &lstate, &sinfo);
        return dinfo || sinfo;
    }

    inline rng_int_t rng_random_destroy(rng_state_t *rng)
    {
        return 0;
    }

#ifdef __cplusplus
}
#endif

#endif
