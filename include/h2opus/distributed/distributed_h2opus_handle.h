#ifndef __DISTRIBUTED_H2OPUS_HANDLE_H__
#define __DISTRIBUTED_H2OPUS_HANDLE_H__

#include <h2opusconf.h>
#include <h2opus/core/h2opus_handle.h>
#include <h2opus/distributed/distributed_comm_buffer.h>

// Prevent CXX namespace to pollute libraries using H2OPUS
#if !defined(MPICH_SKIP_MPICXX)
#define H2OPUS_UNDEF_SKIP_MPICH 1
#define MPICH_SKIP_MPICXX 1
#endif
#if !defined(OMPI_SKIP_MPICXX)
#define H2OPUS_UNDEF_SKIP_OMPI 1
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>
#if defined(H2OPUS_UNDEF_SKIP_MPICH)
#undef OMPI_MPICH_MPICXX
#endif
#if defined(H2OPUS_UNDEF_SKIP_OMPI)
#undef OMPI_SKIP_MPICXX
#endif

struct DistributedH2OpusHandle
{
  public:
    h2opusHandle_t handle;
    h2opusHandle_t top_level_handle;
    MPI_Comm ocomm;
    MPI_Comm comm;
    MPI_Comm commscatter;
    MPI_Comm commgather;
    bool active;
    int orank, rank, local_rank, num_ranks;
    int th_provided;
    bool usethreads[2];

  private:
    int mpitag, mpitagub;

    // Communication buffers
    std::vector<TDistributedSendRecvBuffer<H2OPUS_HWTYPE_CPU>> host_buffers;
    TDistributedTransferBuffer<H2OPUS_HWTYPE_CPU> host_gather_buffer, host_scatter_buffer;
#ifdef H2OPUS_USE_GPU
    std::vector<TDistributedSendRecvBuffer<H2OPUS_HWTYPE_GPU>> device_buffers;
    TDistributedTransferBuffer<H2OPUS_HWTYPE_GPU> device_gather_buffer, device_scatter_buffer;
#endif
  public:
    template <int hw> size_t getNumBuffers();
    template <int hw> void allocateBuffers(size_t);
    template <int hw> TDistributedSendRecvBuffer<hw> &getSendRecvBuffer(size_t);
    template <int hw> TDistributedTransferBuffer<hw> &getGatherBuffer();
    template <int hw> TDistributedTransferBuffer<hw> &getScatterBuffer();

    DistributedH2OpusHandle(MPI_Comm comm = MPI_COMM_WORLD);
    void setRandSeed(unsigned int seed, int hw)
    {
        handle->setRandSeed(seed, hw);
    }
    int getNewTag();
    template <int hw> void setUseThreads(bool = true);
    template <int hw> bool getUseThreads();
    ~DistributedH2OpusHandle();
};

typedef DistributedH2OpusHandle *distributedH2OpusHandle_t;

void h2opusCreateDistributedHandle(distributedH2OpusHandle_t *h2opus_handle, bool select_local_rank = false);
void h2opusCreateDistributedHandleComm(distributedH2OpusHandle_t *h2opus_handle, MPI_Comm comm,
                                       bool select_local_rank = false);
void h2opusDestroyDistributedHandle(distributedH2OpusHandle_t h2opus_handle);

// CPU definitions
template <> inline size_t DistributedH2OpusHandle::getNumBuffers<H2OPUS_HWTYPE_CPU>()
{
    return host_buffers.size();
}

template <> inline void DistributedH2OpusHandle::allocateBuffers<H2OPUS_HWTYPE_CPU>(size_t buffers)
{
    if (host_buffers.size() < buffers)
        host_buffers.resize(buffers);
}

template <>
inline TDistributedSendRecvBuffer<H2OPUS_HWTYPE_CPU> &DistributedH2OpusHandle::getSendRecvBuffer<H2OPUS_HWTYPE_CPU>(
    size_t index)
{
    return host_buffers[index];
}

template <>
inline TDistributedTransferBuffer<H2OPUS_HWTYPE_CPU> &DistributedH2OpusHandle::getGatherBuffer<H2OPUS_HWTYPE_CPU>()
{
    return host_gather_buffer;
}

template <>
inline TDistributedTransferBuffer<H2OPUS_HWTYPE_CPU> &DistributedH2OpusHandle::getScatterBuffer<H2OPUS_HWTYPE_CPU>()
{
    return host_scatter_buffer;
}

template <> inline void DistributedH2OpusHandle::setUseThreads<H2OPUS_HWTYPE_CPU>(bool use)
{
    this->usethreads[H2OPUS_HWTYPE_CPU] = use;
}

template <> inline bool DistributedH2OpusHandle::getUseThreads<H2OPUS_HWTYPE_CPU>()
{
    return this->usethreads[H2OPUS_HWTYPE_CPU];
}

#ifdef H2OPUS_USE_GPU
// GPU definitions
template <> inline size_t DistributedH2OpusHandle::getNumBuffers<H2OPUS_HWTYPE_GPU>()
{
    return device_buffers.size();
}

template <> inline void DistributedH2OpusHandle::allocateBuffers<H2OPUS_HWTYPE_GPU>(size_t buffers)
{
    if (device_buffers.size() < buffers)
        device_buffers.resize(buffers);
}

template <>
inline TDistributedSendRecvBuffer<H2OPUS_HWTYPE_GPU> &DistributedH2OpusHandle::getSendRecvBuffer<H2OPUS_HWTYPE_GPU>(
    size_t index)
{
    return device_buffers[index];
}

template <>
inline TDistributedTransferBuffer<H2OPUS_HWTYPE_GPU> &DistributedH2OpusHandle::getGatherBuffer<H2OPUS_HWTYPE_GPU>()
{
    return device_gather_buffer;
}

template <>
inline TDistributedTransferBuffer<H2OPUS_HWTYPE_GPU> &DistributedH2OpusHandle::getScatterBuffer<H2OPUS_HWTYPE_GPU>()
{
    return device_scatter_buffer;
}

template <> inline void DistributedH2OpusHandle::setUseThreads<H2OPUS_HWTYPE_GPU>(bool use)
{
    this->usethreads[H2OPUS_HWTYPE_GPU] = use;
}

template <> inline bool DistributedH2OpusHandle::getUseThreads<H2OPUS_HWTYPE_GPU>()
{
    return this->usethreads[H2OPUS_HWTYPE_GPU];
}
#endif

#endif
