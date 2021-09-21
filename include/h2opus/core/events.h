#ifndef __H2OPUS_EVENTS_H__
#define __H2OPUS_EVENTS_H__

#include <h2opus/util/gpu_err_check.h>

enum H2OpusEventType
{
    H2OpusUpsweepEvent = 0,
    H2OpusDenseEvent,
    H2OpusCouplingEvent,
    H2OpusDownsweepEvent,
    H2OpusCommunicationEvent,
    H2OpusBufferDownEvent,
    H2OpusBufferUpEvent,
    H2OpusTotalEvents
};

struct H2OpusEvents
{
#ifdef H2OPUS_USE_GPU
  private:
    std::vector<cudaEvent_t> events[H2OpusTotalEvents];
#endif
  public:
    template <int hw> void recordEvent(H2OpusEventType type, size_t event_index, h2opusComputeStream_t stream);
    template <int hw> void streamWaitEvent(H2OpusEventType type, h2opusComputeStream_t stream, size_t event_index);
    template <int hw> void synchEvent(H2OpusEventType type, size_t event_index);
    template <int hw> void allocateEvents(H2OpusEventType type, size_t num_events);
    template <int hw> void clear();
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU definitions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <> inline void H2OpusEvents::allocateEvents<H2OPUS_HWTYPE_CPU>(H2OpusEventType type, size_t num_events)
{
}

template <>
inline void H2OpusEvents::recordEvent<H2OPUS_HWTYPE_CPU>(H2OpusEventType type, size_t event_index,
                                                         h2opusComputeStream_t stream)
{
}

template <>
inline void H2OpusEvents::streamWaitEvent<H2OPUS_HWTYPE_CPU>(H2OpusEventType type, h2opusComputeStream_t stream,
                                                             size_t event_index)
{
}

template <> inline void H2OpusEvents::synchEvent<H2OPUS_HWTYPE_CPU>(H2OpusEventType type, size_t event_index)
{
}

template <> inline void H2OpusEvents::clear<H2OPUS_HWTYPE_CPU>()
{
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU definitions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU
template <> inline void H2OpusEvents::allocateEvents<H2OPUS_HWTYPE_GPU>(H2OpusEventType type, size_t num_events)
{
    std::vector<cudaEvent_t> &event_array = events[type];
    if (event_array.size() != num_events)
    {
        // Clear out old events
        for (size_t i = 0; i < event_array.size(); i++)
            gpuErrchk(cudaEventDestroy(event_array[i]));
        // Make new ones
        event_array.resize(num_events);
        for (size_t i = 0; i < event_array.size(); i++)
            gpuErrchk(cudaEventCreateWithFlags(&event_array[i], cudaEventDisableTiming));
    }
}

template <>
inline void H2OpusEvents::recordEvent<H2OPUS_HWTYPE_GPU>(H2OpusEventType type, size_t event_index,
                                                         h2opusComputeStream_t stream)
{
    assert(event_index < events[type].size());
    gpuErrchk(cudaEventRecord(events[type][event_index], stream->getCudaStream()));
}

template <>
inline void H2OpusEvents::streamWaitEvent<H2OPUS_HWTYPE_GPU>(H2OpusEventType type, h2opusComputeStream_t stream,
                                                             size_t event_index)
{
    assert(event_index < events[type].size());
    gpuErrchk(cudaStreamWaitEvent(stream->getCudaStream(), events[type][event_index], 0));
}

template <> inline void H2OpusEvents::synchEvent<H2OPUS_HWTYPE_GPU>(H2OpusEventType type, size_t event_index)
{
    assert(event_index < events[type].size());
    gpuErrchk(cudaEventSynchronize(events[type][event_index]));
}

template <> inline void H2OpusEvents::clear<H2OPUS_HWTYPE_GPU>()
{
    for (int j = 0; j < H2OpusTotalEvents; j++)
    {
        std::vector<cudaEvent_t> &event_array = events[j];

        // Clear out old events
        for (size_t i = 0; i < event_array.size(); i++)
            gpuErrchk(cudaEventDestroy(event_array[i]));

        event_array.clear();
    }
}

#endif

#endif
