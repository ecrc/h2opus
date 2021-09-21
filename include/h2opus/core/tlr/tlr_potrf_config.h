#ifndef __H2OPUS_TLR_POTRF_CONFIG_H__
#define __H2OPUS_TLR_POTRF_CONFIG_H__

#include <h2opus/core/tlr/tlr_struct.h>

// TLR POTRF Parameter configuration
template <class T, int hw> struct TLR_Potrf_Config
{
    T eps, sc_eps;
    int ndpb, nspb, sample_bs;

    TLR_Potrf_Config(TTLR_Matrix<T, hw> &A)
    {
        this->eps = 1e-6;
        // Schur compensation is off by default
        this->sc_eps = 0;
        this->ndpb = 20;
        this->nspb = (3 * A.n_block + 1) / 2;
        this->sample_bs = 32;
    }

    TLR_Potrf_Config<T, hw> &tolerance(T eps)
    {
        this->eps = eps;
        return *this;
    }

    TLR_Potrf_Config<T, hw> &schur_tolerance(T sc_eps)
    {
        this->sc_eps = sc_eps;
        return *this;
    }

    TLR_Potrf_Config<T, hw> &densePBuffers(int ndpb)
    {
        this->ndpb = ndpb;
        return *this;
    }

    TLR_Potrf_Config<T, hw> &samplingPBuffers(int nspb)
    {
        this->nspb = nspb;
        return *this;
    }

    TLR_Potrf_Config<T, hw> &samplingBlockSize(int sample_bs)
    {
        this->sample_bs = sample_bs;
        return *this;
    }
};

#endif
