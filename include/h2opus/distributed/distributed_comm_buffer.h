#ifndef __DISTRIBUTED_COMM_BUFFER_H__
#define __DISTRIBUTED_COMM_BUFFER_H__

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/distributed/comm_wrapper.h>

// #define __H2OPUS_USE_MANAGED_BUFFERS__

template <int hw> struct TDistributedTransferBuffer
{
  private:
    void *host_down_buffer, *host_up_buffer;
    void *transfer_down_buffer, *transfer_up_buffer;

    size_t allocated_down_buffer, allocated_up_buffer;
    size_t needed_down_buffer, needed_up_buffer;

    MPI_Comm comm;

    int mpi_elements;
    MPI_Datatype transfer_type;

    MPI_Request request;

  public:
    TDistributedTransferBuffer()
    {
        host_down_buffer = host_up_buffer = NULL;
        allocated_down_buffer = allocated_up_buffer = 0;
        needed_down_buffer = needed_up_buffer = 0;

        comm = MPI_COMM_NULL;
        mpi_elements = 0;
        transfer_type = H2OPUS_MPI_REAL;
    }

    void setBufferSizes(size_t down_buffer, size_t up_buffer);
    void prepareDownBuffer(void *down_buffer, h2opusComputeStream_t stream);
    void prepareUpBuffer(void *up_buffer, h2opusComputeStream_t stream);

    void setTransferBuffers(void *transfer_down_buffer, void *transfer_up_buffer)
    {
        this->transfer_down_buffer = transfer_down_buffer;
        this->transfer_up_buffer = transfer_up_buffer;
    }

    void setTransferElements(int elements, MPI_Datatype type)
    {
        this->mpi_elements = elements;
        this->transfer_type = type;
    }

    void setComm(MPI_Comm comm)
    {
        this->comm = comm;
    }
    MPI_Comm getComm()
    {
        return comm;
    }

    int getElements()
    {
        return mpi_elements;
    }
    MPI_Datatype getType()
    {
        return transfer_type;
    }

    void *getDownBuffer();
    void *getUpBuffer();

    MPI_Request *getRequest()
    {
        return &request;
    }

    void freeBuffer()
    {
        if (host_up_buffer)
            free(host_up_buffer);
        if (host_down_buffer)
            free(host_down_buffer);
    }
};

template <int hw> struct TDistributedSendRecvBuffer
{
  private:
    // Auxiliary buffers on the host for nonblocking sends on GPUs pre-pascal
    void *aux_send_buffer, *aux_recv_buffer;
    // Main buffers on either host or device (depending on template parameter hw)
    void *send_buffer, *recv_buffer;
    size_t allocated_send_bytes, allocated_recv_bytes;
    size_t needed_send_bytes, needed_recv_bytes;

    // Communicator and tag
    MPI_Comm comm;
    int tag;

    // The send and receive unblocking requests
    std::vector<MPI_Request> requests;
    size_t num_sends, num_recvs;

    void cleanUp();

  public:
    void setComm(MPI_Comm comm)
    {
        this->comm = comm;
    }

    MPI_Comm getComm()
    {
        return comm;
    }

    void setTag(int tag)
    {
        this->tag = tag;
    }

    int getTag()
    {
        return tag;
    }

    void setRequests(size_t num_sends, size_t num_recvs)
    {
        this->num_sends = num_sends;
        this->num_recvs = num_recvs;
        requests.resize(num_sends + num_recvs);
    }

    size_t getNumRequests()
    {
        return requests.size();
    }

    MPI_Request *getRequests()
    {
        return getSendRequests();
    }

    MPI_Request *getSendRequests()
    {
        assert(requests.size() == num_sends + num_recvs);
        if (num_sends == 0)
            return NULL;
        return vec_ptr(requests);
    }

    MPI_Request *getRecvRequests()
    {
        assert(requests.size() == num_sends + num_recvs);
        if (num_recvs == 0)
            return NULL;
        return vec_ptr(requests) + num_sends;
    }

    void allocateBuffers(size_t send_bytes, size_t recv_bytes);
    void prepareSendBuffer(h2opusComputeStream_t stream);
    void prepareRecvBuffer(h2opusComputeStream_t stream);

    // These are the buffers that the device defined by hw fills in
    void *getSendAccumulationBuffer()
    {
        return send_buffer;
    }
    void *getRecvAccumulationBuffer()
    {
        return recv_buffer;
    }

    // These are the buffers that will be passed to the nonblocking MPI comms
    void *getRecvBuffer();
    void *getSendBuffer();

    TDistributedSendRecvBuffer()
    {
        comm = MPI_COMM_NULL;
        tag = 0;
        aux_send_buffer = aux_recv_buffer = NULL;
        send_buffer = recv_buffer = NULL;
        allocated_send_bytes = allocated_recv_bytes = 0;
    }

    void freeBuffer()
    {
        cleanUp();
    }
};

///////////////////////////////////////////////
// CPU
///////////////////////////////////////////////

// SendRecv
template <>
inline void TDistributedSendRecvBuffer<H2OPUS_HWTYPE_CPU>::allocateBuffers(size_t send_bytes, size_t recv_bytes)
{
    if (send_bytes > allocated_send_bytes)
    {
        if (send_buffer)
            free(send_buffer);
        send_buffer = malloc(send_bytes);
        assert(send_buffer);
        allocated_send_bytes = send_bytes;
    }

    if (recv_bytes > allocated_recv_bytes)
    {
        if (recv_buffer)
            free(recv_buffer);
        recv_buffer = malloc(recv_bytes);
        assert(recv_buffer);
        allocated_recv_bytes = recv_bytes;
    }

    needed_send_bytes = send_bytes;
    needed_recv_bytes = recv_bytes;
}

template <> inline void TDistributedSendRecvBuffer<H2OPUS_HWTYPE_CPU>::prepareSendBuffer(h2opusComputeStream_t stream)
{
}

template <> inline void TDistributedSendRecvBuffer<H2OPUS_HWTYPE_CPU>::prepareRecvBuffer(h2opusComputeStream_t stream)
{
}

template <> inline void *TDistributedSendRecvBuffer<H2OPUS_HWTYPE_CPU>::getSendBuffer()
{
    return send_buffer;
}

template <> inline void *TDistributedSendRecvBuffer<H2OPUS_HWTYPE_CPU>::getRecvBuffer()
{
    return recv_buffer;
}

template <> inline void TDistributedSendRecvBuffer<H2OPUS_HWTYPE_CPU>::cleanUp()
{
    if (send_buffer)
        free(send_buffer);
    if (recv_buffer)
        free(recv_buffer);
}

// Transfer buffer (for gathers and scatters)
template <>
inline void TDistributedTransferBuffer<H2OPUS_HWTYPE_CPU>::setBufferSizes(size_t down_buffer, size_t up_buffer)
{
}

template <>
inline void TDistributedTransferBuffer<H2OPUS_HWTYPE_CPU>::prepareDownBuffer(void *down_buffer,
                                                                             h2opusComputeStream_t stream)
{
}

template <>
inline void TDistributedTransferBuffer<H2OPUS_HWTYPE_CPU>::prepareUpBuffer(void *up_buffer,
                                                                           h2opusComputeStream_t stream)
{
}

template <> inline void *TDistributedTransferBuffer<H2OPUS_HWTYPE_CPU>::getDownBuffer()
{
    return transfer_down_buffer;
}

template <> inline void *TDistributedTransferBuffer<H2OPUS_HWTYPE_CPU>::getUpBuffer()
{
    return transfer_up_buffer;
}

///////////////////////////////////////////////
// GPU
///////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU
// SendRecv
template <>
inline void TDistributedSendRecvBuffer<H2OPUS_HWTYPE_GPU>::allocateBuffers(size_t send_bytes, size_t recv_bytes)
{
    if (send_bytes > allocated_send_bytes)
    {
#ifndef H2OPUS_USE_CUDA_AWARE_MPI
        if (aux_send_buffer)
            gpuErrchk(cudaFreeHost(aux_send_buffer));
        gpuErrchk(cudaMallocHost(&aux_send_buffer, send_bytes));
#endif

        if (send_buffer)
            gpuErrchk(cudaFree(send_buffer));
#ifdef __H2OPUS_USE_MANAGED_BUFFERS__
        gpuErrchk(cudaMallocManaged(&send_buffer, send_bytes));
#else
        gpuErrchk(cudaMalloc(&send_buffer, send_bytes));
#endif

        allocated_send_bytes = send_bytes;
    }

    if (recv_bytes > allocated_recv_bytes)
    {
#ifndef H2OPUS_USE_CUDA_AWARE_MPI
        if (aux_recv_buffer)
            gpuErrchk(cudaFreeHost(aux_recv_buffer));
        gpuErrchk(cudaMallocHost(&aux_recv_buffer, recv_bytes));
        assert(aux_recv_buffer);
#endif

        if (recv_buffer)
            gpuErrchk(cudaFree(recv_buffer));
#ifdef __H2OPUS_USE_MANAGED_BUFFERS__
        gpuErrchk(cudaMallocManaged(&recv_buffer, recv_bytes));
#else
        gpuErrchk(cudaMalloc(&recv_buffer, recv_bytes));
#endif

        allocated_recv_bytes = recv_bytes;
    }

    needed_send_bytes = send_bytes;
    needed_recv_bytes = recv_bytes;
}

template <> inline void TDistributedSendRecvBuffer<H2OPUS_HWTYPE_GPU>::prepareSendBuffer(h2opusComputeStream_t stream)
{
#ifndef H2OPUS_USE_CUDA_AWARE_MPI
    if (needed_send_bytes != 0)
    {
        cudaStream_t cuda_stream = stream->getCudaStream();

        gpuErrchk(
            cudaMemcpyAsync(aux_send_buffer, send_buffer, needed_send_bytes, cudaMemcpyDeviceToHost, cuda_stream));
    }
#endif
}

template <> inline void TDistributedSendRecvBuffer<H2OPUS_HWTYPE_GPU>::prepareRecvBuffer(h2opusComputeStream_t stream)
{
#ifndef H2OPUS_USE_CUDA_AWARE_MPI
    if (needed_recv_bytes != 0)
    {
        cudaStream_t cuda_stream = stream->getCudaStream();

        gpuErrchk(
            cudaMemcpyAsync(recv_buffer, aux_recv_buffer, needed_recv_bytes, cudaMemcpyHostToDevice, cuda_stream));
    }
#endif
}

template <> inline void *TDistributedSendRecvBuffer<H2OPUS_HWTYPE_GPU>::getSendBuffer()
{
#ifndef H2OPUS_USE_CUDA_AWARE_MPI
    return aux_send_buffer;
#else
    return send_buffer;
#endif
}

template <> inline void *TDistributedSendRecvBuffer<H2OPUS_HWTYPE_GPU>::getRecvBuffer()
{
#ifndef H2OPUS_USE_CUDA_AWARE_MPI
    return aux_recv_buffer;
#else
    return recv_buffer;
#endif
}

template <> inline void TDistributedSendRecvBuffer<H2OPUS_HWTYPE_GPU>::cleanUp()
{
    if (send_buffer)
        gpuErrchk(cudaFree(send_buffer));
    if (recv_buffer)
        gpuErrchk(cudaFree(recv_buffer));
#ifndef H2OPUS_USE_CUDA_AWARE_MPI
    if (aux_send_buffer)
        gpuErrchk(cudaFreeHost(aux_send_buffer));
    if (aux_recv_buffer)
        gpuErrchk(cudaFreeHost(aux_recv_buffer));
#endif
}

// Transfer buffer (for gathers and scatters)
template <>
inline void TDistributedTransferBuffer<H2OPUS_HWTYPE_GPU>::setBufferSizes(size_t down_buffer, size_t up_buffer)
{
#ifndef H2OPUS_USE_CUDA_AWARE_MPI
    if (allocated_down_buffer < down_buffer)
    {
        free(host_down_buffer);
        host_down_buffer = malloc(down_buffer);
        allocated_down_buffer = down_buffer;
    }

    if (allocated_up_buffer < up_buffer)
    {
        free(host_up_buffer);
        host_up_buffer = malloc(up_buffer);
        allocated_up_buffer = up_buffer;
    }

    needed_down_buffer = down_buffer;
    needed_up_buffer = up_buffer;
#endif
}

template <>
inline void TDistributedTransferBuffer<H2OPUS_HWTYPE_GPU>::prepareDownBuffer(void *down_buffer,
                                                                             h2opusComputeStream_t stream)
{
#ifndef H2OPUS_USE_CUDA_AWARE_MPI
    if (needed_down_buffer != 0)
    {
        cudaStream_t cuda_stream = stream->getCudaStream();

        gpuErrchk(
            cudaMemcpyAsync(host_down_buffer, down_buffer, needed_down_buffer, cudaMemcpyDeviceToHost, cuda_stream));
    }
#endif
}

template <>
inline void TDistributedTransferBuffer<H2OPUS_HWTYPE_GPU>::prepareUpBuffer(void *up_buffer,
                                                                           h2opusComputeStream_t stream)
{
#ifndef H2OPUS_USE_CUDA_AWARE_MPI
    if (needed_up_buffer != 0)
    {
        cudaStream_t cuda_stream = stream->getCudaStream();

        gpuErrchk(cudaMemcpyAsync(up_buffer, host_up_buffer, needed_up_buffer, cudaMemcpyHostToDevice, cuda_stream));
    }
#endif
}

template <> inline void *TDistributedTransferBuffer<H2OPUS_HWTYPE_GPU>::getDownBuffer()
{
#ifndef H2OPUS_USE_CUDA_AWARE_MPI
    return host_down_buffer;
#else
    return transfer_down_buffer;
#endif
}

template <> inline void *TDistributedTransferBuffer<H2OPUS_HWTYPE_GPU>::getUpBuffer()
{
#ifndef H2OPUS_USE_CUDA_AWARE_MPI
    return host_up_buffer;
#else
    return transfer_up_buffer;
#endif
}
#endif

#endif
