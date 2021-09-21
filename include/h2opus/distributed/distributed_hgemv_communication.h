#ifndef __DISTRIBUTED_HGEMV_COMMUNICATION_H__
#define __DISTRIBUTED_HGEMV_COMMUNICATION_H__

#include <h2opus/distributed/distributed_common.h>
#include <h2opus/distributed/distributed_comm_buffer.h>
#include <h2opus/distributed/distributed_hmatrix.h>

#define H2OPUS_DIST_HGEMV_GATHER_EVENT 0
#define H2OPUS_DIST_HGEMV_SCATTER_EVENT 1
#define H2OPUS_DIST_HGEMV_DENSE_EXCHANGE_EVENT 2
#define H2OPUS_DIST_HGEMV_TOTAL_EVENTS 3

///////////////////////////////////////////////////////////////////////////////////
// Helper structs
///////////////////////////////////////////////////////////////////////////////////
template <int hw> struct HgemvLowRankExchangeHelper
{
    TDistributedCompressedBSNData<hw> *bsn_data;
    TDistributedSendRecvBuffer<hw> *comm_buffer;
    int level_rank, num_vectors;

    HgemvLowRankExchangeHelper(){};

    void init(TDistributedCompressedBSNData<hw> *bsn_data, TDistributedSendRecvBuffer<hw> *comm_buffer, int level_rank,
              int num_vectors)
    {
        this->bsn_data = bsn_data;
        this->comm_buffer = comm_buffer;
        this->level_rank = level_rank;
        this->num_vectors = num_vectors;
    }
};

template <int hw> struct HgemvDenseExchangeHelper
{
    TDistributedCompressedBSNData<hw> *bsn_data;
    TDistributedSendRecvBuffer<hw> *comm_buffer;
    int num_vectors;
    int *dense_send_sizes, *dense_receive_sizes;

    HgemvDenseExchangeHelper(){};

    void init(TDistributedCompressedBSNData<hw> *bsn_data, TDistributedSendRecvBuffer<hw> *comm_buffer, int num_vectors,
              int *dense_send_sizes, int *dense_receive_sizes)
    {
        this->bsn_data = bsn_data;
        this->comm_buffer = comm_buffer;
        this->num_vectors = num_vectors;
        this->dense_send_sizes = dense_send_sizes;
        this->dense_receive_sizes = dense_receive_sizes;
    }
};

///////////////////////////////////////////////////////////////////////////////////
// Handling requests
///////////////////////////////////////////////////////////////////////////////////
template <int hw> inline void hgemvExecuteLowRankExchange(HgemvLowRankExchangeHelper<hw> *exchange_helper)
{
    TDistributedCompressedBSNData<hw> *bsn_data = exchange_helper->bsn_data;
    TDistributedSendRecvBuffer<hw> *comm_buffer = exchange_helper->comm_buffer;
    int level_rank = exchange_helper->level_rank;
    int num_vectors = exchange_helper->num_vectors;

    int buffer_pos;
    MPI_Comm comm = comm_buffer->getComm();
    int mpitag = comm_buffer->getTag();

    // Receives
    buffer_pos = 0;
    H2Opus_Real *recv_buffer = (H2Opus_Real *)comm_buffer->getRecvBuffer();
    MPI_Request *recv_requests = comm_buffer->getRecvRequests();
    for (size_t i = 0; i < bsn_data->receive_process_ids.size(); i++)
    {
        int process_id = bsn_data->receive_process_ids[i];
        int node_start = bsn_data->receive_process_node_ptrs[i];
        int node_end = bsn_data->receive_process_node_ptrs[i + 1];
        int recv_size = (node_end - node_start) * level_rank * num_vectors;

        mpiErrchk(MPI_Irecv(recv_buffer + buffer_pos, recv_size, H2OPUS_MPI_REAL, process_id, mpitag, comm,
                            recv_requests + i));

        buffer_pos += recv_size;
    }

    // Sends
    buffer_pos = 0;
    H2Opus_Real *send_buffer = (H2Opus_Real *)comm_buffer->getSendBuffer();
    MPI_Request *send_requests = comm_buffer->getSendRequests();
    for (size_t i = 0; i < bsn_data->send_process_ids.size(); i++)
    {
        int process_id = bsn_data->send_process_ids[i];
        int node_start = bsn_data->send_process_node_ptrs[i];
        int node_end = bsn_data->send_process_node_ptrs[i + 1];
        int send_size = (node_end - node_start) * level_rank * num_vectors;

        mpiErrchk(MPI_Isend(send_buffer + buffer_pos, send_size, H2OPUS_MPI_REAL, process_id, mpitag, comm,
                            send_requests + i));

        buffer_pos += send_size;
    }
}

template <int hw> inline void hgemvExecuteDenseExchange(HgemvDenseExchangeHelper<hw> *exchange_helper)
{
    TDistributedCompressedBSNData<hw> *bsn_data = exchange_helper->bsn_data;
    TDistributedSendRecvBuffer<hw> *comm_buffer = exchange_helper->comm_buffer;
    int num_vectors = exchange_helper->num_vectors;
    int *dense_send_sizes = exchange_helper->dense_send_sizes;
    int *dense_receive_sizes = exchange_helper->dense_receive_sizes;

    int buffer_pos;
    MPI_Comm comm = comm_buffer->getComm();
    int mpitag = comm_buffer->getTag();

    // Receives
    buffer_pos = 0;
    H2Opus_Real *recv_buffer = (H2Opus_Real *)comm_buffer->getRecvBuffer();
    MPI_Request *recv_requests = comm_buffer->getRecvRequests();
    for (size_t i = 0; i < bsn_data->receive_process_ids.size(); i++)
    {
        int process_id = bsn_data->receive_process_ids[i];
        int recv_size = dense_receive_sizes[i] * num_vectors;

        mpiErrchk(MPI_Irecv(recv_buffer + buffer_pos, recv_size, H2OPUS_MPI_REAL, process_id, mpitag, comm,
                            recv_requests + i));

        buffer_pos += recv_size;
    }

    // Sends
    buffer_pos = 0;
    H2Opus_Real *send_buffer = (H2Opus_Real *)comm_buffer->getSendBuffer();
    MPI_Request *send_requests = comm_buffer->getSendRequests();
    for (size_t i = 0; i < bsn_data->send_process_ids.size(); i++)
    {
        int process_id = bsn_data->send_process_ids[i];
        int send_size = dense_send_sizes[i] * num_vectors;

        mpiErrchk(MPI_Isend(send_buffer + buffer_pos, send_size, H2OPUS_MPI_REAL, process_id, mpitag, comm,
                            send_requests + i));

        buffer_pos += send_size;
    }
}

template <int hw> inline void hgemvExecuteLowRankWait(HgemvLowRankExchangeHelper<hw> *exchange_helper)
{
    TDistributedSendRecvBuffer<hw> *comm_buffer = exchange_helper->comm_buffer;

    mpiErrchk(MPI_Waitall(comm_buffer->getNumRequests(), comm_buffer->getRequests(), MPI_STATUSES_IGNORE));
}

template <int hw> inline void hgemvExecuteDenseWait(HgemvDenseExchangeHelper<hw> *exchange_helper)
{
    TDistributedSendRecvBuffer<hw> *comm_buffer = exchange_helper->comm_buffer;

    mpiErrchk(MPI_Waitall(comm_buffer->getNumRequests(), comm_buffer->getRequests(), MPI_STATUSES_IGNORE));
}

template <int hw>
inline void h2opusExecuteBlockingLowRankExchange(std::vector<HgemvLowRankExchangeHelper<hw>> &low_rank_exchange_helpers,
                                                 H2OpusEvents &events, int local_rank, h2opusComputeStream_t stream)
{
    // This may be run inside a thread, so make sure the device is set
    h2opusSetDevice<hw>(local_rank);

    int depth = (int)low_rank_exchange_helpers.size();

    for (int level = depth - 1; level >= 1; level--)
    {
        events.synchEvent<hw>(H2OpusBufferDownEvent, level);
        hgemvExecuteLowRankExchange(&low_rank_exchange_helpers[level]);
        TDistributedSendRecvBuffer<hw> *comm_buffer = low_rank_exchange_helpers[level].comm_buffer;
        hgemvExecuteLowRankWait(&low_rank_exchange_helpers[level]);
        comm_buffer->prepareRecvBuffer(stream);
        events.recordEvent<hw>(H2OpusBufferUpEvent, level, stream);
    }
}

template <int hw>
inline void h2opusExecuteDenseExchange(HgemvDenseExchangeHelper<hw> *exchange_helper, H2OpusEvents &events,
                                       int local_rank, h2opusComputeStream_t stream, int phase)
{
    // This may be run inside a thread, so make sure the device is set
    h2opusSetDevice<hw>(local_rank);
    if (phase <= 0)
    {
        events.synchEvent<hw>(H2OpusCommunicationEvent, H2OPUS_DIST_HGEMV_DENSE_EXCHANGE_EVENT);

        hgemvExecuteDenseExchange(exchange_helper);
    }
    if (phase != 0)
    {
        hgemvExecuteDenseWait(exchange_helper);
        exchange_helper->comm_buffer->prepareRecvBuffer(stream);
    }
}

#endif
