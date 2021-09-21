#ifndef __H2OPUS_TLR_POTRF_UTIL_H__
#define __H2OPUS_TLR_POTRF_UTIL_H__

////////////////////////////////////////////////////////////////////////////
// Utility
////////////////////////////////////////////////////////////////////////////

// Timing stuff
namespace TLR_Potrf_Phase_Types
{
enum Phase
{
    Reduction = 0,
    Sample,
    Projection,
    Realloc,
    Orthog,
    Trsm,
    Potrf,
    Clear,
    RandGen,
    SchurCompensation,
    DenseUpdate,
    PivotSelection,
    TLR_Potrf_TotalPhases
};
};

template <int hw> struct TLR_Potrf_Phase_Times
{
    static double phase_times[TLR_Potrf_Phase_Types::TLR_Potrf_TotalPhases];
    static Timer<hw> timer;
    static int currentPhase;

    static void init()
    {
        timer.init();
        for (int i = 0; i < TLR_Potrf_Phase_Types::TLR_Potrf_TotalPhases; i++)
            phase_times[i] = 0;
        currentPhase = -1;
    }

    static void startPhase(TLR_Potrf_Phase_Types::Phase type)
    {
        currentPhase = type;
        timer.start();
    }

    static void endPhase(TLR_Potrf_Phase_Types::Phase type)
    {
        assert(currentPhase == type);

        phase_times[type] += timer.stop();
        currentPhase = -1;
    }
};

template <int hw> Timer<hw> TLR_Potrf_Phase_Times<hw>::timer;

template <int hw> double TLR_Potrf_Phase_Times<hw>::phase_times[TLR_Potrf_Phase_Types::TLR_Potrf_TotalPhases];

template <int hw> int TLR_Potrf_Phase_Times<hw>::currentPhase;

template <class T> inline void tlr_util_eig_2x2_sym(T a11, T a12, T a22, T &c, T &s, T &d1, T &d2)
{
    T tau = (a22 - a11) / (2 * a12);
    T t = (T)(tau < 0 ? -1 : 1) / (tau + sqrt(1 + tau * tau));
    c = (T)1 / sqrt(1 + t * t);
    s = c * t;
    d1 = a11 - t * a12;
    d2 = a22 + t * a12;
}

template <class T, int hw> class PotrfDiagonalBlockSampler : public HMatrixSampler
{
  private:
    T *D, *LLt;
    int k, block_size, ld_d, ld_llt;
    h2opusComputeStream_t stream;

  public:
    PotrfDiagonalBlockSampler(T *D, int ld_d, int block_size, int k, T *LLt, int ld_llt, h2opusComputeStream_t stream)
    {
        this->D = D;
        this->ld_d = ld_d;
        this->block_size = block_size;
        this->k = k;
        this->LLt = LLt;
        this->ld_llt = ld_llt;
        this->stream = stream;
    }

    void sample(T *input, T *output, int samples)
    {
        // output = (D - LLt) * input = D * input - LLt * input

        T d_beta = 0;

        // output = -LLt * input
        if (k != 0)
        {
            blas_gemm<T, hw>(stream, H2Opus_NoTrans, H2Opus_NoTrans, block_size, samples, block_size, -1, LLt, ld_llt,
                             input, block_size, 0, output, block_size);
            d_beta = 1;
        }
        // output += D * input
        blas_gemm<T, hw>(stream, H2Opus_NoTrans, H2Opus_NoTrans, block_size, samples, block_size, 1, D, ld_d, input,
                         block_size, d_beta, output, block_size);
    }
};

#endif
