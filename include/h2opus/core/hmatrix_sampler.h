#ifndef __HMATRIX_SAMPLER_H__
#define __HMATRIX_SAMPLER_H__

class HMatrixSampler
{
  public:
    int nsamples = 0;
    virtual void sample(H2Opus_Real *input, H2Opus_Real *output, int samples) = 0;
    virtual ~HMatrixSampler()
    {
    }
};

#endif
