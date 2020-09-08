#ifndef _HCOMPRESS_WEIGHTPACKET_H__
#define _HCOMPRESS_WEIGHTPACKET_H__

template <int hw> struct TWeightAccelerationPacket
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

    RealVector Z_level;
    int rank, level;
    int debug_data;

    TWeightAccelerationPacket()
    {
        this->debug_data = 0;
        this->rank = 0;
        this->level = 0;
    };

    TWeightAccelerationPacket(int level, int level_rank, int level_nodes)
    {
        this->level = level;
        this->rank = level_rank;
        this->debug_data = 0;
    }
};

typedef TWeightAccelerationPacket<H2OPUS_HWTYPE_CPU> WeightAccelerationPacket;

#ifdef H2OPUS_USE_GPU
typedef TWeightAccelerationPacket<H2OPUS_HWTYPE_GPU> WeightAccelerationPacket_GPU;
#endif

#endif
