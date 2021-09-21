#include <h2opus/core/hgemv.h>
#include <h2opus/marshal/hgemv_marshal.h>

#include <h2opus/distributed/comm_wrapper.h>
#include <h2opus/distributed/distributed_hgemv.h>
#include <h2opus/distributed/distributed_hgemv_communication.h>
#include <h2opus/distributed/distributed_hgemv_marshal_mult.cuh>

#include <h2opus/core/hgemv.h>
#include <h2opus/util/batch_wrappers.h>
#include <h2opus/util/blas_wrappers.h>
#include <h2opus/util/debug_routines.h>
#include <h2opus/util/gpu_err_check.h>
#include <h2opus/util/thrust_wrappers.h>
#include <h2opus/util/timer.h>

#include <thread>
#include <mutex>
#include <condition_variable>

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Template routines
/////////////////////////////////////////////////////////////////////////////////////////////////////////
template <int hw>
void distributed_hgemv_mult_prepare_offDiag_template(TBasisTree<hw> &basis_branch,
                                                     TDistributedCompresseBasisTree<hw> &v_basis_tree,
                                                     DistributedHgemvWorkspace &dist_workspace,
                                                     TDistributedSendRecvBuffer<hw> *comm_buffer, int level,
                                                     int num_vectors, h2opusComputeStream_t stream)
{
    if (level == 0)
        return;

    TDistributedCompressedBSNData<hw> &bsn_data = v_basis_tree.coupling_compressed_bsn_data[level];

    size_t num_nodes = bsn_data.send_process_nodes.size();

    H2Opus_Real **xhat_ptrs = dist_workspace.ptr_A, **buffer_ptrs = dist_workspace.ptr_B;
    int level_rank = basis_branch.getLevelRank(level);

    // Marshall all the block data
    H2Opus_Real *send_buffer = (H2Opus_Real *)comm_buffer->getSendAccumulationBuffer();
    H2Opus_Real *xhat_level = dist_workspace.branch_workspace.xhat.data[level];

    distributed_hgemv_offdiag_coupling_buffer_input_marshal<H2Opus_Real, hw>(
        xhat_level, send_buffer, level_rank, num_vectors, vec_ptr(bsn_data.send_process_nodes), xhat_ptrs, buffer_ptrs,
        num_nodes, stream);

    // Copy the nodes into the buffer using the marshalled data
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copyBlock)(stream, level_rank, num_vectors, buffer_ptrs, 0, 0,
                                                                  level_rank, xhat_ptrs, 0, 0, level_rank, num_nodes));

    comm_buffer->prepareSendBuffer(stream);
}

template <int hw>
void distributed_hgemv_low_rank_exchange(TBasisTree<hw> &basis_branch, TDistributedCompresseBasisTree<hw> &v_basis_tree,
                                         DistributedHgemvWorkspace &dist_workspace, H2OpusEvents &events,
                                         std::vector<HgemvLowRankExchangeHelper<hw>> &low_rank_exchange_helpers,
                                         std::mutex &upsweep_mutex, std::condition_variable &upsweep_cv,
                                         std::vector<int> &level_processed, int local_rank, int num_vectors,
                                         h2opusComputeStream_t stream, int phase)
{
    // This may be run inside a thread, so make sure the device is set
    h2opusSetDevice<hw>(local_rank);
    int num_levels = basis_branch.depth;

    if (phase <= 0)
    {
        for (int level = num_levels - 1; level >= 1; level--)
        {
            // Make sure the level has been processed
            {
                std::unique_lock<std::mutex> lock(upsweep_mutex);
                upsweep_cv.wait(lock, [&level_processed, &level] { return level_processed[level] == 1; });
            }

            // Prepare the local offdiagonal for communication
            events.streamWaitEvent<hw>(H2OpusUpsweepEvent, stream, level);
            TDistributedSendRecvBuffer<hw> *comm_buffer = low_rank_exchange_helpers[level].comm_buffer;
            distributed_hgemv_mult_prepare_offDiag_template<hw>(basis_branch, v_basis_tree, dist_workspace, comm_buffer,
                                                                level, num_vectors, stream);
            events.recordEvent<hw>(H2OpusBufferDownEvent, level, stream);
            events.synchEvent<hw>(H2OpusBufferDownEvent, level);

            // Send and receive all data
            hgemvExecuteLowRankExchange(&low_rank_exchange_helpers[level]);
        }
    }
    if (phase != 0)
    {
        for (int level = num_levels - 1; level >= 1; level--)
        {
            TDistributedSendRecvBuffer<hw> *comm_buffer = low_rank_exchange_helpers[level].comm_buffer;
            hgemvExecuteLowRankWait(&low_rank_exchange_helpers[level]);

            // Send the received offdiagonal data to the device
            comm_buffer->prepareRecvBuffer(stream);
            events.recordEvent<hw>(H2OpusBufferUpEvent, level, stream);
        }
    }
}

template <int hw>
void distributed_hgemv_upsweep_template(H2Opus_Real alpha, TBasisTree<hw> &basis_branch, H2Opus_Real *X, int ldx,
                                        int num_vectors, DistributedHgemvWorkspace &dist_workspace,
                                        H2OpusEvents &events, std::mutex &upsweep_mutex,
                                        std::condition_variable &upsweep_cv, std::vector<int> &level_processed,
                                        h2opusComputeStream_t stream, int rank)
{
    //////////////////////////////////////////////////////////////////////////////////////////
    // Do the upsweep on each branch
    //////////////////////////////////////////////////////////////////////////////////////////
    int num_levels = basis_branch.depth;

#ifdef H2OPUS_PROFILING_ENABLED
    Timer<hw> timer;
    timer.init();
    PerformanceCounter::clearCounters();
    timer.start();
#endif

    hgemv_upsweep_leaves(alpha, basis_branch, X, ldx, num_vectors, dist_workspace.branch_workspace, stream);
    events.recordEvent<hw>(H2OpusUpsweepEvent, num_levels - 1, stream);

#ifdef H2OPUS_PROFILING_ENABLED
    HLibProfile::cumRun(HLibProfile::HGEMV_UPSWEEP, PerformanceCounter::getOpCount(PerformanceCounter::GEMM),
                        timer.stop());
#endif

    // Notify the low rank exchange thread that the level has been processed
    {
        std::lock_guard<std::mutex> lock(upsweep_mutex);
        level_processed[num_levels - 1] = 1;
    }
    upsweep_cv.notify_one();

    for (int level = num_levels - 2; level >= 0; level--)
    {
        // Upsweep for this level
        // Executed on the main stream
#ifdef H2OPUS_PROFILING_ENABLED
        PerformanceCounter::clearCounters();
        timer.start();
#endif
        hgemv_upsweep_level(basis_branch, level, num_vectors, dist_workspace.branch_workspace, stream);
        events.recordEvent<hw>(H2OpusUpsweepEvent, level, stream);

#ifdef H2OPUS_PROFILING_ENABLED
        HLibProfile::cumRun(HLibProfile::HGEMV_UPSWEEP, PerformanceCounter::getOpCount(PerformanceCounter::GEMM),
                            timer.stop());
#endif
        // Notify the low rank exchange thread that the level has been processed
        {
            std::lock_guard<std::mutex> lock(upsweep_mutex);
            level_processed[level] = 1;
        }
        upsweep_cv.notify_one();
    }
}

template <int hw>
void distributed_hgemv_process_toplevel_template(H2Opus_Real alpha, TDistributedHMatrix<hw> &dist_hmatrix,
                                                 H2Opus_Real beta, int num_vectors,
                                                 DistributedHgemvWorkspace &dist_workspace, H2OpusEvents &events,
                                                 TDistributedTransferBuffer<hw> &gather_buffer,
                                                 TDistributedTransferBuffer<hw> &scatter_buffer,
                                                 h2opusComputeStream_t stream, int rank, int local_rank)
{
    h2opusSetDevice<hw>(local_rank);

    H2Opus_Real *yhat_leaf_level = NULL;
    TBasisTree<hw> &basis_branch = dist_hmatrix.basis_tree.basis_branch;
    BasisTreeLevelData &branch_level_data = basis_branch.level_data;

    //////////////////////////////////////////////////////////////////////////////////////////
    // Gather the root nodes of the all branches into the bottom level of the root branch
    //////////////////////////////////////////////////////////////////////////////////////////
    events.streamWaitEvent<hw>(H2OpusUpsweepEvent, stream, 0);

    H2Opus_Real *xhat_branch_root = dist_workspace.branch_workspace.xhat.data[0];
    int top_level_rank = branch_level_data.getLevelRank(0);
    size_t top_elements = top_level_rank * num_vectors;
    H2Opus_Real *xhat_leaf_level = NULL;

    if (rank == 0)
        xhat_leaf_level = dist_workspace.top_level_workspace.xhat.data[dist_hmatrix.basis_tree.top_level.depth - 1];

    gather_buffer.prepareDownBuffer(xhat_branch_root, stream);
    gather_buffer.setTransferBuffers(xhat_branch_root, xhat_leaf_level);
    gather_buffer.setTransferElements(top_elements, H2OPUS_MPI_REAL);

    events.recordEvent<hw>(H2OpusCommunicationEvent, H2OPUS_DIST_HGEMV_GATHER_EVENT, stream);
    events.synchEvent<hw>(H2OpusCommunicationEvent, H2OPUS_DIST_HGEMV_GATHER_EVENT);

    mpiErrchk(MPI_Gather(gather_buffer.getDownBuffer(), gather_buffer.getElements(), gather_buffer.getType(),
                         gather_buffer.getUpBuffer(), gather_buffer.getElements(), gather_buffer.getType(), 0,
                         gather_buffer.getComm()));

    if (rank == 0)
    {
        //////////////////////////////////////////////////////////////////////////////////////////
        // Upsweep
        //////////////////////////////////////////////////////////////////////////////////////////
        // Prepare the upsweep buffer
        int leaf_level = dist_hmatrix.basis_tree.top_level.depth - 1;
        H2Opus_Real *xhat_leaf_level = dist_workspace.top_level_workspace.xhat.data[leaf_level];
        gather_buffer.prepareUpBuffer(xhat_leaf_level, stream);

#ifdef H2OPUS_PROFILING_ENABLED
        Timer<hw> timer;
        timer.init();
        PerformanceCounter::clearCounters();
        timer.start();
#endif
        // Now we can finish sweeping up the tree
        hgemv_upsweep(alpha, dist_hmatrix.basis_tree.top_level, NULL, 0, num_vectors,
                      dist_workspace.top_level_workspace, stream);

#ifdef H2OPUS_PROFILING_ENABLED
        HLibProfile::cumRun(HLibProfile::HGEMV_UPSWEEP, PerformanceCounter::getOpCount(PerformanceCounter::GEMM),
                            timer.stop());
#endif
        //////////////////////////////////////////////////////////////////////////////////////////
        // Mult
        //////////////////////////////////////////////////////////////////////////////////////////

#ifdef H2OPUS_PROFILING_ENABLED
        PerformanceCounter::clearCounters();
        timer.start();
#endif

        hgemv_mult(H2Opus_NoTrans, dist_hmatrix.hnodes.top_level, 0, leaf_level, num_vectors,
                   dist_hmatrix.basis_tree.top_level, dist_hmatrix.basis_tree.top_level,
                   dist_workspace.top_level_workspace, stream);

#ifdef H2OPUS_PROFILING_ENABLED
        HLibProfile::cumRun(HLibProfile::HGEMV_MULT, PerformanceCounter::getOpCount(PerformanceCounter::GEMM),
                            timer.stop());
#endif

        //////////////////////////////////////////////////////////////////////////////////////////
        // Downsweep
        //////////////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_PROFILING_ENABLED
        PerformanceCounter::clearCounters();
        timer.start();
#endif

        hgemv_downsweep(dist_hmatrix.basis_tree.top_level, NULL, 0, num_vectors, dist_workspace.top_level_workspace,
                        stream);

#ifdef H2OPUS_PROFILING_ENABLED
        HLibProfile::cumRun(HLibProfile::HGEMV_DOWNSWEEP, PerformanceCounter::getOpCount(PerformanceCounter::GEMM),
                            timer.stop());
#endif

        yhat_leaf_level = dist_workspace.top_level_workspace.yhat.data[leaf_level];
    }

    // Scatter the leaf level of yhat
    // H2Opus_Real* yhat_scatter_root = dist_workspace.yhat_scatter_root;
    H2Opus_Real *yhat_scatter_root = dist_workspace.branch_workspace.yhat.data[0];
    scatter_buffer.prepareDownBuffer(yhat_leaf_level, stream);
    scatter_buffer.setTransferBuffers(yhat_leaf_level, yhat_scatter_root);
    scatter_buffer.setTransferElements(top_elements, H2OPUS_MPI_REAL);

    events.recordEvent<hw>(H2OpusCommunicationEvent, H2OPUS_DIST_HGEMV_SCATTER_EVENT, stream);
    events.synchEvent<hw>(H2OpusCommunicationEvent, H2OPUS_DIST_HGEMV_SCATTER_EVENT);

    mpiErrchk(MPI_Scatter(scatter_buffer.getDownBuffer(), scatter_buffer.getElements(), scatter_buffer.getType(),
                          scatter_buffer.getUpBuffer(), scatter_buffer.getElements(), scatter_buffer.getType(), 0,
                          scatter_buffer.getComm()));
}

template <int hw>
void distributed_hgemv_downsweep_template(TDistributedBasisTree<hw> &basis_tree,
                                          TDistributedTransferBuffer<hw> &scatter_buffer, H2Opus_Real *Y, int ldy,
                                          int num_vectors, DistributedHgemvWorkspace &dist_workspace,
                                          h2opusComputeStream_t stream)
{
    TBasisTree<hw> &basis_branch = basis_tree.basis_branch;
    H2Opus_Real *yhat_scatter_root = dist_workspace.branch_workspace.yhat.data[0];
    scatter_buffer.prepareUpBuffer(yhat_scatter_root, stream);

    //////////////////////////////////////////////////////////////////////////////////////////
    // Do the downsweep on each branch
    //////////////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_PROFILING_ENABLED
    Timer<hw> timer;
    timer.init();
    PerformanceCounter::clearCounters();
    timer.start();
#endif

    hgemv_downsweep(basis_branch, Y, ldy, num_vectors, dist_workspace.branch_workspace, stream);

#ifdef H2OPUS_PROFILING_ENABLED
    HLibProfile::cumRun(HLibProfile::HGEMV_DOWNSWEEP, PerformanceCounter::getOpCount(PerformanceCounter::GEMM),
                        timer.stop());
#endif
}

template <int hw>
void distributed_hgemv_mult_diagonal_template(TDistributedHNodeTree<hw> &dist_hnodes, int num_vectors,
                                              TDistributedBasisTree<hw> &basis_tree,
                                              DistributedHgemvWorkspace &dist_workspace, h2opusComputeStream_t stream)
{
    // Mult on the diagonal block
    // skip the root level since that was handled by the root branch
#ifdef H2OPUS_PROFILING_ENABLED
    Timer<hw> timer;
    timer.init();
    PerformanceCounter::clearCounters();
    timer.start();
#endif

    hgemv_mult(H2Opus_NoTrans, dist_hnodes.diagonal_block, 1, dist_hnodes.diagonal_block.depth - 1, num_vectors,
               basis_tree.basis_branch, basis_tree.basis_branch, dist_workspace.branch_workspace, stream);

#ifdef H2OPUS_PROFILING_ENABLED
    HLibProfile::cumRun(HLibProfile::HGEMV_MULT, PerformanceCounter::getOpCount(PerformanceCounter::GEMM),
                        timer.stop());
#endif
}

template <int hw>
void distributed_hgemv_denseMult_prepare_offDiag_template(H2Opus_Real *X, int ldx, int num_vectors,
                                                          TDistributedBasisTree<hw> &u_basis_tree,
                                                          TDistributedCompresseBasisTree<hw> &v_basis_tree,
                                                          DistributedHgemvWorkspace &dist_workspace,
                                                          HgemvDenseExchangeHelper<hw> &dense_exchange_helper,
                                                          H2OpusEvents &events, h2opusComputeStream_t stream)
{

    TDistributedCompressedBSNData<hw> &bsn_data = v_basis_tree.dense_compressed_bsn_data;
    TBasisTree<hw> &u_branch = u_basis_tree.basis_branch;
    TDistributedSendRecvBuffer<hw> *comm_buffer = dense_exchange_helper.comm_buffer;

    size_t num_nodes = bsn_data.send_process_nodes.size();

    H2Opus_Real **input_ptrs = dist_workspace.ptr_A, **buffer_ptrs = dist_workspace.ptr_B;
    int *ptr_m = dist_workspace.ptr_m, *ptr_n = dist_workspace.ptr_n;
    int *ld_input_ptrs = dist_workspace.ptr_lda, *ld_buffer_ptrs = dist_workspace.ptr_ldb;
    int level_start_index = u_branch.getLevelStart(u_branch.depth - 1);

    // Marshall all the block data
    H2Opus_Real *send_buffer = (H2Opus_Real *)comm_buffer->getSendAccumulationBuffer();

    distributed_hgemv_offdiag_buffer_input_marshal<H2Opus_Real, hw>(
        X, ldx, send_buffer, num_vectors, vec_ptr(v_basis_tree.dense_send_node_offsets), level_start_index,
        vec_ptr(bsn_data.send_process_nodes), vec_ptr(u_branch.node_start), vec_ptr(u_branch.node_len), input_ptrs,
        buffer_ptrs, ptr_m, ptr_n, ld_input_ptrs, ld_buffer_ptrs, num_nodes, stream);

    // Copy the nodes into the buffer using the marshalled data
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copyBlock)(stream, ptr_m, ptr_n, u_branch.leaf_size, num_vectors,
                                                                  buffer_ptrs, ld_buffer_ptrs, input_ptrs,
                                                                  ld_input_ptrs, num_nodes));

    comm_buffer->prepareSendBuffer(stream);
    // h2opusAddDenseExchangeCallbackToStream(&dense_exchange_helper, stream);
    events.recordEvent<hw>(H2OpusCommunicationEvent, H2OPUS_DIST_HGEMV_DENSE_EXCHANGE_EVENT, stream);
}

// Mostly a copy paste from the single process code due to the different input data layout
// Need to generalize the single process code to handle different types of layouts
template <int hw>
void distributed_hgemv_denseMult_offDiag(H2Opus_Real alpha, TDistributedHNodeTree<hw> &dist_hnodes,
                                         H2Opus_Real *X_compressed, H2Opus_Real beta, H2Opus_Real *Y, int ldy,
                                         int num_vectors, TDistributedBasisTree<hw> &u_basis_tree,
                                         TDistributedCompresseBasisTree<hw> &v_basis_tree,
                                         DistributedHgemvWorkspace &dist_workspace, h2opusComputeStream_t stream)
{
    typedef typename THNodeTree<hw>::HNodeTreeBSNData BSNData;

    // Marshalled data
    BatchGemmMarshalledData &marshal_data = dist_workspace.branch_workspace.dense_gemms;

    H2Opus_Real **A_ptrs = marshal_data.A_ptrs;
    H2Opus_Real **B_ptrs = marshal_data.B_ptrs;
    H2Opus_Real **C_ptrs = marshal_data.C_ptrs;

    int *m_ptr = marshal_data.m_batch, *n_ptr = marshal_data.n_batch, *k_ptr = marshal_data.k_batch;
    int *lda_ptr = marshal_data.lda_batch, *ldb_ptr = marshal_data.ldb_batch, *ldc_ptr = marshal_data.ldc_batch;

    // Basis data
    TBasisTree<hw> &u_branch = u_basis_tree.basis_branch;

    int *node_u_start = vec_ptr(u_branch.node_start);
    int *node_u_len = vec_ptr(u_branch.node_len);

    // Hnode data
    THNodeTree<hw> &hnodes = dist_hnodes.off_diagonal_blocks;
    int *compressed_node_v_index = vec_ptr(dist_hnodes.compressed_v_index);
    int *compressed_node_offset = vec_ptr(v_basis_tree.dense_receive_node_offsets);
    BSNData *bsn_data = &(hnodes.bsn_row_data);
    int dense_node_offset = 0;
    int node_size = hnodes.leaf_size;

    distributed_hgemv_offdiag_dense_mult_marshal<H2Opus_Real, hw>(
        vec_ptr(hnodes.dense_leaf_mem), node_size, X_compressed, num_vectors, Y, ldy,
        vec_ptr(hnodes.dense_leaf_tree_index), vec_ptr(hnodes.node_u_index), compressed_node_v_index,
        vec_ptr(bsn_data->dense_batch_indexes), node_u_start, node_u_len, compressed_node_offset, dense_node_offset,
        A_ptrs, B_ptrs, C_ptrs, m_ptr, n_ptr, k_ptr, lda_ptr, ldb_ptr, ldc_ptr, hnodes.num_dense_leaves, stream);

    // Execute the marshalled operations
    std::vector<int> &dense_batch_ptr = bsn_data->dense_batch_ptr;
    int num_batches = (int)dense_batch_ptr.size() - 1;
    int kblas_trans_mode = H2Opus_NoTrans;

#ifdef H2OPUS_PROFILING_ENABLED
    Timer<hw> timer;
    timer.init();
    PerformanceCounter::clearCounters();
    timer.start();
#endif
    for (int batch_id = 0; batch_id < num_batches; batch_id++)
    {
        H2Opus_Real **A_batch = A_ptrs + dense_batch_ptr[batch_id];
        H2Opus_Real **B_batch = B_ptrs + dense_batch_ptr[batch_id];
        H2Opus_Real **C_batch = C_ptrs + dense_batch_ptr[batch_id];

        int *m_batch = m_ptr + dense_batch_ptr[batch_id];
        int *n_batch = n_ptr + dense_batch_ptr[batch_id];
        int *k_batch = k_ptr + dense_batch_ptr[batch_id];

        int *lda_batch = lda_ptr + dense_batch_ptr[batch_id];
        int *ldb_batch = ldb_ptr + dense_batch_ptr[batch_id];
        int *ldc_batch = ldc_ptr + dense_batch_ptr[batch_id];

        int batch_size = dense_batch_ptr[batch_id + 1] - dense_batch_ptr[batch_id];
        H2Opus_Real batch_beta = 1;

        check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(stream, kblas_trans_mode, H2Opus_NoTrans, m_batch,
                                                                 n_batch, k_batch, node_size, num_vectors, node_size,
                                                                 alpha, (const H2Opus_Real **)A_batch, lda_batch,
                                                                 (const H2Opus_Real **)B_batch, ldb_batch, batch_beta,
                                                                 C_batch, ldc_batch, batch_size));
    }
#ifdef H2OPUS_PROFILING_ENABLED
    HLibProfile::cumRun(HLibProfile::HGEMV_DENSE, PerformanceCounter::getOpCount(PerformanceCounter::GEMM),
                        timer.stop());
#endif
}

template <int hw>
void distributed_hgemv_mult_offDiag(TDistributedHNodeTree<hw> &dist_hnodes, int num_vectors,
                                    TDistributedBasisTree<hw> &u_basis_tree,
                                    std::vector<HgemvLowRankExchangeHelper<hw>> &low_rank_exchange_helpers,
                                    DistributedHgemvWorkspace &dist_workspace, int level, h2opusComputeStream_t stream)
{
    size_t u_basis_offset = u_basis_tree.basis_branch.level_data.getLevelStart(level);

    // Grab compressed offdiagonal xhat level and local full yhat level
    H2Opus_Real *xhat_compressed_level =
        (H2Opus_Real *)low_rank_exchange_helpers[level].comm_buffer->getRecvAccumulationBuffer();
    H2Opus_Real *yhat_level = dist_workspace.branch_workspace.yhat.data[level];

    THNodeTree<hw> &hnodes = dist_hnodes.off_diagonal_blocks;

#ifdef H2OPUS_PROFILING_ENABLED
    Timer<hw> timer;
    timer.init();
    PerformanceCounter::clearCounters();
    timer.start();
#endif

    hgemv_mult_level(hnodes, level, num_vectors, u_basis_offset, 0, xhat_compressed_level, yhat_level,
                     dist_workspace.branch_workspace.low_rank_gemms, &(hnodes.bsn_row_data),
                     vec_ptr(dist_hnodes.compressed_v_index), vec_ptr(hnodes.node_u_index), H2Opus_NoTrans, stream);

#ifdef H2OPUS_PROFILING_ENABLED
    HLibProfile::cumRun(HLibProfile::HGEMV_MULT, PerformanceCounter::getOpCount(PerformanceCounter::GEMM),
                        timer.stop());
#endif
}

template <int hw>
void distributed_hgemv_denseMult_diagonal_template(H2Opus_Real alpha, TDistributedHNodeTree<hw> &dist_hnodes,
                                                   H2Opus_Real *X, int ldx, H2Opus_Real beta, H2Opus_Real *Y, int ldy,
                                                   int num_vectors, TDistributedBasisTree<hw> &u_basis_tree,
                                                   TDistributedCompresseBasisTree<hw> &v_basis_tree,
                                                   DistributedHgemvWorkspace &dist_workspace,
                                                   h2opusComputeStream_t stream)
{
    //////////////////////////////////////////////////////////////////////////////////////////
    // Process the diagonal block
    //////////////////////////////////////////////////////////////////////////////////////////
    THNodeTree<hw> &hnodes = dist_hnodes.diagonal_block;
    TBasisTree<hw> &u_branch = u_basis_tree.basis_branch;
    HgemvWorkspace &branch_workspace = dist_workspace.branch_workspace;

    int *column_basis_indexes = vec_ptr(hnodes.node_v_index);
    int *row_basis_indexes = vec_ptr(hnodes.node_u_index);
    int kblas_trans_mode = H2Opus_NoTrans;
    int *node_u_start = vec_ptr(u_branch.node_start);
    int *node_u_len = vec_ptr(u_branch.node_len);

#ifdef H2OPUS_PROFILING_ENABLED
    Timer<hw> timer;
    timer.init();
    PerformanceCounter::clearCounters();
    timer.start();
#endif

    hgemv_denseMult(kblas_trans_mode, alpha, hnodes, X, ldx, beta, Y, ldy, num_vectors, node_u_start, node_u_start,
                    node_u_len, node_u_len, column_basis_indexes, row_basis_indexes, &(hnodes.bsn_row_data),
                    branch_workspace, stream);

#ifdef H2OPUS_PROFILING_ENABLED
    HLibProfile::cumRun(HLibProfile::HGEMV_DENSE, PerformanceCounter::getOpCount(PerformanceCounter::GEMM),
                        timer.stop());
#endif
}

template <int hw>
void distributed_hgemv_prepare_comm_buffers_template(TDistributedHMatrix<hw> &dist_hmatrix, int num_vectors,
                                                     distributedH2OpusHandle_t dist_h2opus_handle)
{
    TDistributedCompresseBasisTree<hw> &v_basis_tree = dist_hmatrix.compressed_basis_tree_data;
    TBasisTree<hw> &u_branch = dist_hmatrix.basis_tree.basis_branch;

    // Make sure we have enough buffers to accommodate the depth of this hierarchical matrix
    dist_h2opus_handle->allocateBuffers<hw>(u_branch.depth + 1);

    TDistributedSendRecvBuffer<hw> &dense_comm_buffer = dist_h2opus_handle->getSendRecvBuffer<hw>(0);
    TDistributedTransferBuffer<hw> &gather_buffer = dist_h2opus_handle->getGatherBuffer<hw>();
    TDistributedTransferBuffer<hw> &scatter_buffer = dist_h2opus_handle->getScatterBuffer<hw>();

    // Dense buffers
    dense_comm_buffer.allocateBuffers(v_basis_tree.dense_send_total_sum * num_vectors * sizeof(H2Opus_Real),
                                      v_basis_tree.dense_receive_total_sum * num_vectors * sizeof(H2Opus_Real));

    dense_comm_buffer.setRequests(v_basis_tree.dense_compressed_bsn_data.send_process_ids.size(),
                                  v_basis_tree.dense_compressed_bsn_data.receive_process_ids.size());
    dense_comm_buffer.setComm(dist_h2opus_handle->comm);
    dense_comm_buffer.setTag(dist_h2opus_handle->getNewTag());

    // Upsweep gather buffer
    int top_level_rank = u_branch.level_data.getLevelRank(0);
    size_t top_elements = top_level_rank * num_vectors;
    size_t branch_elements = 0;
    if (dist_h2opus_handle->rank == 0)
        branch_elements = top_elements * dist_h2opus_handle->num_ranks;

    gather_buffer.setBufferSizes(top_elements * sizeof(H2Opus_Real), branch_elements * sizeof(H2Opus_Real));
    gather_buffer.setComm(dist_h2opus_handle->commgather);

    // Downsweep scatter buffer
    branch_elements = top_level_rank * num_vectors;
    top_elements = 0;
    if (dist_h2opus_handle->rank == 0)
        top_elements = branch_elements * dist_h2opus_handle->num_ranks;

    scatter_buffer.setBufferSizes(top_elements * sizeof(H2Opus_Real), branch_elements * sizeof(H2Opus_Real));
    scatter_buffer.setComm(dist_h2opus_handle->commscatter);

    // Low rank buffers
    int depth = u_branch.depth;
    for (int level = 1; level < depth; level++)
    {
        TDistributedSendRecvBuffer<hw> &comm_buffer = dist_h2opus_handle->getSendRecvBuffer<hw>(level + 1);
        TDistributedCompressedBSNData<hw> &bsn_data = v_basis_tree.coupling_compressed_bsn_data[level];

        int level_rank = u_branch.level_data.getLevelRank(level);
        size_t nodes_to_send = bsn_data.send_process_nodes.size();
        size_t node_to_receive = bsn_data.receive_process_nodes.size();

        comm_buffer.allocateBuffers(nodes_to_send * level_rank * num_vectors * sizeof(H2Opus_Real),
                                    node_to_receive * level_rank * num_vectors * sizeof(H2Opus_Real));

        comm_buffer.setRequests(bsn_data.send_process_ids.size(), bsn_data.receive_process_ids.size());
        comm_buffer.setComm(dist_h2opus_handle->comm);
        comm_buffer.setTag(dist_h2opus_handle->getNewTag());
    }
}

template <int hw>
void distributed_hgemv_template(H2Opus_Real alpha, TDistributedHMatrix<hw> &dist_hmatrix, H2Opus_Real *X, int ldx,
                                H2Opus_Real beta, H2Opus_Real *Y, int ldy, int num_vectors,
                                distributedH2OpusHandle_t dist_h2opus_handle)
{
#ifdef H2OPUS_PROFILING_ENABLED
    HLibProfile::addRun(HLibProfile::HGEMV_MULT, 0, 0);
    HLibProfile::addRun(HLibProfile::HGEMV_UPSWEEP, 0, 0);
    HLibProfile::addRun(HLibProfile::HGEMV_DOWNSWEEP, 0, 0);
    HLibProfile::addRun(HLibProfile::HGEMV_DENSE, 0, 0);
#endif

    if (!dist_h2opus_handle->active)
        return;

    h2opusHandle_t h2opus_handle = dist_h2opus_handle->handle;
    h2opusHandle_t top_level_handle = dist_h2opus_handle->top_level_handle;

    H2OpusEvents &events = h2opus_handle->getEvents();
    h2opusComputeStream_t main_stream = h2opus_handle->getMainStream();
    h2opusComputeStream_t secondary_stream = h2opus_handle->getSecondaryStream();
    h2opusComputeStream_t low_priority_stream = h2opus_handle->getLowPriorityStream();

    //////////////////////////////////////////////////////////////////////////////////////////
    // Workspace allocation
    //////////////////////////////////////////////////////////////////////////////////////////
    H2OpusWorkspaceState ws_needed, top_level_ws_needed;
    distributed_hgemv_workspace(dist_hmatrix, num_vectors, ws_needed, top_level_ws_needed, dist_h2opus_handle);

    H2OpusWorkspaceState ws_allocated = h2opus_handle->getWorkspaceState();
    H2OpusWorkspaceState top_level_ws_allocated = top_level_handle->getWorkspaceState();

    if (ws_allocated < ws_needed)
        h2opus_handle->setWorkspaceState(ws_needed);

    if (top_level_ws_allocated < top_level_ws_needed)
        top_level_handle->setWorkspaceState(top_level_ws_needed);

    DistributedHgemvWorkspace dist_workspace;
    distributed_hgemv_get_workspace(dist_hmatrix, num_vectors, dist_workspace, dist_h2opus_handle);

    // Prepare all communication buffers
    distributed_hgemv_prepare_comm_buffers_template<hw>(dist_hmatrix, num_vectors, dist_h2opus_handle);

    // Allocate events
    TBasisTree<hw> &basis_branch = dist_hmatrix.basis_tree.basis_branch;
    int branch_depth = basis_branch.depth;
    events.allocateEvents<hw>(H2OpusUpsweepEvent, branch_depth);
    events.allocateEvents<hw>(H2OpusBufferUpEvent, branch_depth);
    events.allocateEvents<hw>(H2OpusBufferDownEvent, branch_depth);
    events.allocateEvents<hw>(H2OpusDenseEvent, 1);
    events.allocateEvents<hw>(H2OpusCommunicationEvent, H2OPUS_DIST_HGEMV_TOTAL_EVENTS);

    if (X == NULL || ldx == 0 || Y == NULL || ldy == 0 || num_vectors == 0)
        return;

    TDistributedTransferBuffer<hw> &gather_buffer = dist_h2opus_handle->getGatherBuffer<hw>();
    TDistributedTransferBuffer<hw> &scatter_buffer = dist_h2opus_handle->getScatterBuffer<hw>();
    TDistributedSendRecvBuffer<hw> &dense_comm_buffer = dist_h2opus_handle->getSendRecvBuffer<hw>(0);

    //////////////////////////////////////////////////////////////////////////////////////////
    // Helper structs for the streaming callbacks
    //////////////////////////////////////////////////////////////////////////////////////////
    std::vector<HgemvLowRankExchangeHelper<hw>> low_rank_exchange_helpers(branch_depth);
    HgemvDenseExchangeHelper<hw> dense_exchange_helper;

    TDistributedCompresseBasisTree<hw> &compressed_tree = dist_hmatrix.compressed_basis_tree_data;
    for (int level = 1; level < branch_depth; level++)
    {
        TDistributedCompressedBSNData<hw> &bsn_data = compressed_tree.coupling_compressed_bsn_data[level];
        TDistributedSendRecvBuffer<hw> &comm_buffer = dist_h2opus_handle->getSendRecvBuffer<hw>(level + 1);
        int level_rank = basis_branch.level_data.getLevelRank(level);

        low_rank_exchange_helpers[level].init(&bsn_data, &comm_buffer, level_rank, num_vectors);
    }

    dense_exchange_helper.init(&compressed_tree.dense_compressed_bsn_data, &dense_comm_buffer, num_vectors,
                               vec_ptr(compressed_tree.dense_send_sizes), vec_ptr(compressed_tree.dense_receive_sizes));

    //////////////////////////////////////////////////////////////////////////////////////////
    // Accumulate all the vector data from X that needs to be sent to other processes into a buffer
    // Stream: Low priority
    //////////////////////////////////////////////////////////////////////////////////////////
    distributed_hgemv_denseMult_prepare_offDiag_template<hw>(X, ldx, num_vectors, dist_hmatrix.basis_tree,
                                                             compressed_tree, dist_workspace, dense_exchange_helper,
                                                             events, low_priority_stream);

    // See if we can use customized threaded communications
    bool usethreads = dist_h2opus_handle->getUseThreads<hw>();

    //////////////////////////////////////////////////////////////////////////////////////////
    // Branch upsweep
    // Stream: Main
    // Stream: Secondary for transfers
    //////////////////////////////////////////////////////////////////////////////////////////
    // Mutex and condition variable for the levels of the upsweep
    std::mutex upsweep_mutex;
    std::condition_variable upsweep_cv;
    std::vector<int> level_processed(branch_depth, 0);
    std::thread *low_rank_exchange_thread = nullptr, *dense_exchange_thread = nullptr, *root_branch_thread = nullptr;

    // Initiate the transfers using threads
    if (usethreads)
    {
        low_rank_exchange_thread =
            new std::thread(distributed_hgemv_low_rank_exchange<hw>, std::ref(basis_branch), std::ref(compressed_tree),
                            std::ref(dist_workspace), std::ref(events), std::ref(low_rank_exchange_helpers),
                            std::ref(upsweep_mutex), std::ref(upsweep_cv), std::ref(level_processed),
                            dist_h2opus_handle->local_rank, num_vectors, secondary_stream, -1);
    }
    distributed_hgemv_upsweep_template<hw>(alpha, basis_branch, X, ldx, num_vectors, dist_workspace, events,
                                           upsweep_mutex, upsweep_cv, level_processed, main_stream,
                                           dist_h2opus_handle->rank);

    if (usethreads)
    {
        dense_exchange_thread =
            new std::thread(h2opusExecuteDenseExchange<hw>, &dense_exchange_helper, std::ref(events),
                            dist_h2opus_handle->local_rank, low_priority_stream, -1);

        root_branch_thread = new std::thread(
            distributed_hgemv_process_toplevel_template<hw>, alpha, std::ref(dist_hmatrix), beta, num_vectors,
            std::ref(dist_workspace), std::ref(events), std::ref(gather_buffer), std::ref(scatter_buffer),
            secondary_stream, dist_h2opus_handle->rank, dist_h2opus_handle->local_rank);
    }
    else
    {
        distributed_hgemv_low_rank_exchange(basis_branch, compressed_tree, dist_workspace, events,
                                            low_rank_exchange_helpers, upsweep_mutex, upsweep_cv, level_processed,
                                            dist_h2opus_handle->local_rank, num_vectors, secondary_stream, 0);
        h2opusExecuteDenseExchange(&dense_exchange_helper, events, dist_h2opus_handle->local_rank, low_priority_stream,
                                   0);
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Dense Diagonal multiplication phase
    // Stream: Low priority
    //////////////////////////////////////////////////////////////////////////////////////////
    distributed_hgemv_denseMult_diagonal_template<hw>(alpha, dist_hmatrix.hnodes, X, ldx, beta, Y, ldy, num_vectors,
                                                      dist_hmatrix.basis_tree, compressed_tree, dist_workspace,
                                                      low_priority_stream);

    //////////////////////////////////////////////////////////////////////////////////////////
    // Low rank Diagonal multiplication phase
    // Stream: Main
    //////////////////////////////////////////////////////////////////////////////////////////
    distributed_hgemv_mult_diagonal_template<hw>(dist_hmatrix.hnodes, num_vectors, dist_hmatrix.basis_tree,
                                                 dist_workspace, main_stream);

    //////////////////////////////////////////////////////////////////////////////////////////
    // Process the offdiagonal dense blocks
    // Stream: Low Priority
    //////////////////////////////////////////////////////////////////////////////////////////
    if (usethreads)
        dense_exchange_thread->join();
    else
    {
        h2opusExecuteDenseExchange(&dense_exchange_helper, events, dist_h2opus_handle->local_rank, low_priority_stream,
                                   1);
    }
    H2Opus_Real *recv_buffer = (H2Opus_Real *)dense_comm_buffer.getRecvAccumulationBuffer();

    distributed_hgemv_denseMult_offDiag<hw>(alpha, dist_hmatrix.hnodes, recv_buffer, beta, Y, ldy, num_vectors,
                                            dist_hmatrix.basis_tree, compressed_tree, dist_workspace,
                                            low_priority_stream);

    events.recordEvent<hw>(H2OpusDenseEvent, 0, low_priority_stream);

    //////////////////////////////////////////////////////////////////////////////////////////
    // Process the offdiagonal low rank blocks
    // Stream: Main
    //////////////////////////////////////////////////////////////////////////////////////////
    if (usethreads)
        low_rank_exchange_thread->join();
    else
    {
        distributed_hgemv_low_rank_exchange(basis_branch, compressed_tree, dist_workspace, events,
                                            low_rank_exchange_helpers, upsweep_mutex, upsweep_cv, level_processed,
                                            dist_h2opus_handle->local_rank, num_vectors, secondary_stream, 1);
    }
    // Skip the root level since that was handled by the root branch
    for (int level = branch_depth - 1; level >= 1; level--)
    {
        events.streamWaitEvent<hw>(H2OpusBufferUpEvent, main_stream, level);

        distributed_hgemv_mult_offDiag<hw>(dist_hmatrix.hnodes, num_vectors, dist_hmatrix.basis_tree,
                                           low_rank_exchange_helpers, dist_workspace, level, main_stream);
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Downsweep
    // Stream: Main
    //////////////////////////////////////////////////////////////////////////////////////////
    if (usethreads)
        root_branch_thread->join();
    else
    {
        // We do a blocking gather/scatter here
        distributed_hgemv_process_toplevel_template(alpha, dist_hmatrix, beta, num_vectors, dist_workspace, events,
                                                    gather_buffer, scatter_buffer, secondary_stream,
                                                    dist_h2opus_handle->rank, dist_h2opus_handle->local_rank);
    }
    events.streamWaitEvent<hw>(H2OpusDenseEvent, main_stream, 0);

    distributed_hgemv_downsweep_template(dist_hmatrix.basis_tree, scatter_buffer, Y, ldy, num_vectors, dist_workspace,
                                         main_stream);

    delete dense_exchange_thread;
    delete low_rank_exchange_thread;
    delete root_branch_thread;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Interface routines
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU
/////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU
void distributed_hgemv(H2Opus_Real alpha, DistributedHMatrix_GPU &dist_hmatrix, H2Opus_Real *X, int ldx,
                       H2Opus_Real beta, H2Opus_Real *Y, int ldy, int num_vectors,
                       distributedH2OpusHandle_t dist_h2opus_handle)
{
    distributed_hgemv_template<H2OPUS_HWTYPE_GPU>(alpha, dist_hmatrix, X, ldx, beta, Y, ldy, num_vectors,
                                                  dist_h2opus_handle);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void distributed_hgemv(H2Opus_Real alpha, DistributedHMatrix &dist_hmatrix, H2Opus_Real *X, int ldx, H2Opus_Real beta,
                       H2Opus_Real *Y, int ldy, int num_vectors, distributedH2OpusHandle_t dist_h2opus_handle)
{
    distributed_hgemv_template<H2OPUS_HWTYPE_CPU>(alpha, dist_hmatrix, X, ldx, beta, Y, ldy, num_vectors,
                                                  dist_h2opus_handle);
}
