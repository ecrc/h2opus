#ifndef __DISTRIBUTED_HORTHOG_COMMUNICATION_H__
#define __DISTRIBUTED_HORTHOG_COMMUNICATION_H__

#include <h2opus/distributed/distributed_common.h>
#include <h2opus/distributed/distributed_comm_buffer.h>
#include <h2opus/distributed/distributed_hmatrix.h>

#define H2OPUS_DIST_HCOMPRESS_GATHER_EVENT 0
#define H2OPUS_DIST_HCOMPRESS_SCATTER_EVENT 1
#define H2OPUS_DIST_HCOMPRESS_TOTAL_EVENTS 2

///////////////////////////////////////////////////////////////////////////////////
// Helper structs
///////////////////////////////////////////////////////////////////////////////////
template <int hw> struct HcompressLowRankExchangeHelper
{
    TDistributedCompressedBSNData<hw> *bsn_data;
    TDistributedSendRecvBuffer<hw> *comm_buffer;
    int rows, cols;

    HcompressLowRankExchangeHelper(){};

    void init(TDistributedCompressedBSNData<hw> *bsn_data, TDistributedSendRecvBuffer<hw> *comm_buffer)
    {
        this->bsn_data = bsn_data;
        this->comm_buffer = comm_buffer;
    }

    void setExchangeDataSize(int rows, int cols)
    {
        this->rows = rows;
        this->cols = cols;
    }
};

///////////////////////////////////////////////////////////////////////////////////
// Handling requests
///////////////////////////////////////////////////////////////////////////////////
template <int hw> inline void hcompressExecuteLowRankExchange(HcompressLowRankExchangeHelper<hw> *exchange_helper)
{
    TDistributedCompressedBSNData<hw> *bsn_data = exchange_helper->bsn_data;
    TDistributedSendRecvBuffer<hw> *comm_buffer = exchange_helper->comm_buffer;
    int level_rows = exchange_helper->rows;
    int level_cols = exchange_helper->cols;

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
        int recv_size = (node_end - node_start) * level_rows * level_cols;

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
        int send_size = (node_end - node_start) * level_rows * level_cols;

        mpiErrchk(MPI_Isend(send_buffer + buffer_pos, send_size, H2OPUS_MPI_REAL, process_id, mpitag, comm,
                            send_requests + i));

        buffer_pos += send_size;
    }
}

template <int hw> inline void hcompressExecuteLowRankWait(HcompressLowRankExchangeHelper<hw> *exchange_helper)
{
    TDistributedSendRecvBuffer<hw> *comm_buffer = exchange_helper->comm_buffer;

    mpiErrchk(MPI_Waitall(comm_buffer->getNumRequests(), comm_buffer->getRequests(), MPI_STATUSES_IGNORE));
}

template <int hw>
inline void hcompressExecuteBlockingLowRankExchange(
    std::vector<HcompressLowRankExchangeHelper<hw>> &low_rank_exchange_helpers, H2OpusEvents &events, int local_rank,
    h2opusComputeStream_t stream)
{
    // This may be run inside a thread, so make sure the device is set
    h2opusSetDevice<hw>(local_rank);

    int depth = (int)low_rank_exchange_helpers.size();

    for (int level = depth - 1; level >= 0; level--)
    {
        events.synchEvent<hw>(H2OpusBufferDownEvent, level);
        hcompressExecuteLowRankExchange(&low_rank_exchange_helpers[level]);
        TDistributedSendRecvBuffer<hw> *comm_buffer = low_rank_exchange_helpers[level].comm_buffer;
        hcompressExecuteLowRankWait(&low_rank_exchange_helpers[level]);
        comm_buffer->prepareRecvBuffer(stream);
        events.recordEvent<hw>(H2OpusBufferUpEvent, level, stream);
    }
}

#endif
