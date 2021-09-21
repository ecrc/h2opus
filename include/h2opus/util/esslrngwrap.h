#ifndef __H2OPUS_ESSLRNGWRAP_H__
#define __H2OPUS_ESSLRNGWRAP_H__

#include <h2opusconf.h>
#include <essl.h>

typedef struct
{
    _ESVINT *istate;
    _ESVINT listate;
} essl_rndstate_t;

inline void essl_random_create(essl_rndstate_t *rng)
{
    _ESVINT iseed[4] = {198712, 88172, 33, 74294};
    /* awful pass by reference for listate when compiling with c++! */
    rng->listate = -1;
#ifdef H2OPUS_USE_DOUBLE_PRECISION
    initrng(2, 0, iseed, 4, rng->istate, rng->listate);
#elif defined(H2OPUS_USE_SINGLE_PRECISION)
    initrng(1, 0, iseed, 4, rng->istate, rng->listate);
#endif
    rng->istate = (_ESVINT *)malloc(rng->listate * sizeof(_ESVINT));
}

inline void essl_random_destroy(essl_rndstate_t *rng)
{
    free(rng->istate);
}

inline void essl_random_seed(essl_rndstate_t rng, _ESVINT seed)
{
    /* awful pass by reference for listate when compiling with c++! */
#ifdef H2OPUS_USE_DOUBLE_PRECISION
    initrng(2, 0, &seed, 1, rng.istate, rng.listate);
#elif defined(H2OPUS_USE_SINGLE_PRECISION)
    initrng(1, 0, &seed, 1, rng.istate, rng.listate);
#endif
}

#endif
