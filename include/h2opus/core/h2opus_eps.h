#ifndef __H2OPUS_EPS_H__
#define __H2OPUS_EPS_H__

template <class T> struct H2OpusEpsilon;
template <> struct H2OpusEpsilon<float>
{
    static constexpr float eps = 1.1920928955078125e-07;
};
template <> struct H2OpusEpsilon<double>
{
    static constexpr double eps = 2.2204460492503131e-16;
};

#endif
