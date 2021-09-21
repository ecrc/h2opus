#include <h2opus/core/hcompress.h>
#include <h2opus/marshal/hcompress_marshal.h>

#include <h2opus/distributed/comm_wrapper.h>
#include <h2opus/distributed/distributed_hcompress.h>
#include <h2opus/distributed/distributed_hcompress_communication.h>
#include <h2opus/distributed/distributed_hcompress_marshal.cuh>

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
void distributed_hcompress_project_offdiagonal_template(
    TDistributedHNodeTree<hw> &dist_hnodes, TDistributedBasisTree<hw> &u_basis_tree,
    std::vector<HcompressLowRankExchangeHelper<hw>> &low_rank_exchange_helpers,
    DistributedHcompressWorkspace &dist_workspace, H2OpusEvents &events, h2opusComputeStream_t stream)
{
    THNodeTree<hw> &hnodes = dist_hnodes.off_diagonal_blocks;
    TBasisTree<hw> &basis_tree = u_basis_tree.basis_branch;
    HcompressWorkspace &workspace = dist_workspace.branch_workspace;

    int num_levels = hnodes.depth;
    int top_level = 1; // basis_tree.level_data.nested_root_level;

    // Now go through the levels of the tree and compute the projection
    // of the coupling matrices into the new basis
    for (int level = num_levels - 1; level >= top_level; level--)
    {
        size_t u_level_start = basis_tree.level_data.getLevelStart(level);
        size_t v_level_start = 0;
        H2Opus_Real *Tu_level = workspace.u_upsweep.T_hat[level];
        H2Opus_Real *Tv_level =
            (H2Opus_Real *)low_rank_exchange_helpers[level].comm_buffer->getRecvAccumulationBuffer();
        int *node_u_index = vec_ptr(hnodes.node_u_index);
        int *node_v_index = vec_ptr(dist_hnodes.compressed_v_index);

        int ld_tu = basis_tree.level_data.getLevelRank(level);
        int ld_tv = workspace.u_upsweep.new_ranks[level];

        events.streamWaitEvent<hw>(H2OpusBufferUpEvent, stream, level);

#ifdef H2OPUS_PROFILING_ENABLED
        Timer<hw> timer;
        timer.init();
        PerformanceCounter::clearCounters();
        timer.start();
#endif

        hcompress_project_level(hnodes, level, u_level_start, v_level_start, Tu_level, ld_tu, Tv_level, ld_tv,
                                node_u_index, node_v_index, workspace.u_upsweep, workspace.u_upsweep,
                                workspace.projection, stream);

#ifdef H2OPUS_PROFILING_ENABLED
        HLibProfile::cumRun(HLibProfile::HCOMPRESS_PROJECTION, PerformanceCounter::getOpCount(), timer.stop());
#endif
    }
}

template <int hw>
void distributed_hcompress_project_diagonal_template(TDistributedHNodeTree<hw> &dist_hnodes,
                                                     TDistributedBasisTree<hw> &basis_tree,
                                                     DistributedHcompressWorkspace &dist_workspace,
                                                     h2opusComputeStream_t stream)
{
    // Project the diagonal block
    BasisTreeLevelData &basis_level_data = basis_tree.basis_branch.level_data;
    HcompressWorkspace &workspace = dist_workspace.branch_workspace;

#ifdef H2OPUS_PROFILING_ENABLED
    Timer<hw> timer;
    timer.init();
    PerformanceCounter::clearCounters();
    timer.start();
#endif

    hcompress_project(dist_hnodes.diagonal_block, basis_level_data, basis_level_data, workspace.u_upsweep,
                      workspace.u_upsweep, workspace.projection, stream);

#ifdef H2OPUS_PROFILING_ENABLED
    HLibProfile::cumRun(HLibProfile::HCOMPRESS_PROJECTION, PerformanceCounter::getOpCount(), timer.stop());
#endif
}

template <int hw>
void distributed_hcompress_prepare_offDiag_template(TBasisTree<hw> &basis_branch,
                                                    TDistributedCompresseBasisTree<hw> &v_basis_tree,
                                                    DistributedHcompressWorkspace &dist_workspace,
                                                    TDistributedSendRecvBuffer<hw> *comm_buffer, int level,
                                                    h2opusComputeStream_t stream)
{
    TDistributedCompressedBSNData<hw> &bsn_data = v_basis_tree.coupling_compressed_bsn_data[level];

    size_t num_nodes = bsn_data.send_process_nodes.size();

    H2Opus_Real **T_hat_ptrs = dist_workspace.ptr_A, **buffer_ptrs = dist_workspace.ptr_B;

    int old_rank = basis_branch.level_data.getLevelRank(level);
    int new_rank = dist_workspace.branch_workspace.u_upsweep.new_ranks[level];

    // Marshall all the block data
    H2Opus_Real *send_buffer = (H2Opus_Real *)comm_buffer->getSendAccumulationBuffer();
    H2Opus_Real *T_hat_level = dist_workspace.branch_workspace.u_upsweep.T_hat[level];

    distributed_hcompress_offdiag_projection_buffer_marshal<H2Opus_Real, hw>(
        T_hat_level, send_buffer, new_rank, old_rank, vec_ptr(bsn_data.send_process_nodes), T_hat_ptrs, buffer_ptrs,
        num_nodes, stream);

    // Copy the nodes into the buffer using the marshalled data
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copyBlock)(stream, new_rank, old_rank, buffer_ptrs, 0, 0,
                                                                  new_rank, T_hat_ptrs, 0, 0, old_rank, num_nodes));

    comm_buffer->prepareSendBuffer(stream);
}

template <int hw>
void distributed_hcompress_process_toplevel_template(TDistributedHMatrix<hw> &dist_hmatrix, H2Opus_Real eps,
                                                     DistributedHcompressWorkspace &dist_workspace,
                                                     H2OpusEvents &events,
                                                     TDistributedTransferBuffer<hw> &gather_buffer,
                                                     h2opusComputeStream_t stream, int rank, int local_rank)
{
    h2opusSetDevice<hw>(local_rank);

    TBasisTree<hw> &basis_branch = dist_hmatrix.basis_tree.basis_branch;
    HcompressWorkspace &branch_workspace = dist_workspace.branch_workspace;

    //////////////////////////////////////////////////////////////////////////////////////////
    // Gather the root nodes of the all branches into the bottom level of the root branch
    //////////////////////////////////////////////////////////////////////////////////////////
    H2Opus_Real *T_hat_branch_root = branch_workspace.u_upsweep.T_hat[0];
    int level_rank = basis_branch.getLevelRank(0);
    size_t top_elements = level_rank * level_rank;
    H2Opus_Real *T_hat_leaf_level = NULL;

    if (rank == 0)
    {
        int leaf_level = dist_hmatrix.basis_tree.top_level.depth - 1;
        T_hat_leaf_level = dist_workspace.top_level_workspace.u_upsweep.T_hat[leaf_level];
    }

    gather_buffer.prepareDownBuffer(T_hat_branch_root, stream);
    gather_buffer.setTransferBuffers(T_hat_branch_root, T_hat_leaf_level);
    gather_buffer.setTransferElements(top_elements, H2OPUS_MPI_REAL);

    events.recordEvent<hw>(H2OpusCommunicationEvent, H2OPUS_DIST_HCOMPRESS_GATHER_EVENT, stream);
    events.synchEvent<hw>(H2OpusCommunicationEvent, H2OPUS_DIST_HCOMPRESS_GATHER_EVENT);

    mpiErrchk(MPI_Gather(gather_buffer.getDownBuffer(), gather_buffer.getElements(), gather_buffer.getType(),
                         gather_buffer.getUpBuffer(), gather_buffer.getElements(), gather_buffer.getType(), 0,
                         gather_buffer.getComm()));

    if (rank == 0)
    {
        //////////////////////////////////////////////////////////////////////////////////////////
        // Upsweep
        //////////////////////////////////////////////////////////////////////////////////////////
        // Prepare the upsweep buffer
        TBasisTree<hw> &top_level_basis = dist_hmatrix.basis_tree.top_level;
        int num_levels = top_level_basis.depth;
        HcompressWorkspace &top_workspace = dist_workspace.top_level_workspace;
        H2Opus_Real *T_hat_leaf_level = top_workspace.u_upsweep.T_hat[num_levels - 1];
        gather_buffer.prepareUpBuffer(T_hat_leaf_level, stream);

        // Set the rank from the branch
        top_workspace.u_upsweep.new_ranks[num_levels - 1] = branch_workspace.u_upsweep.new_ranks[0];

#ifdef H2OPUS_PROFILING_ENABLED
        Timer<hw> timer;
        timer.init();
        PerformanceCounter::clearCounters();
        timer.start();
#endif

        // Now we can finish sweeping up the tree
        for (int level = num_levels - 2; level >= 0; level--)
        {
            int local_level_rank =
                hcompress_compressed_basis_level_rank(top_level_basis, eps, level, top_workspace.u_upsweep, stream);

            hcompress_truncate_basis_level(top_level_basis, local_level_rank, level, top_workspace.u_upsweep, stream);
        }

#ifdef H2OPUS_PROFILING_ENABLED
        HLibProfile::cumRun(HLibProfile::HCOMPRESS_TRUNCATE_BASIS, PerformanceCounter::getOpCount(), timer.stop());
        PerformanceCounter::clearCounters();
        timer.start();
#endif

        // TODO WAJIH: Stitch is missing?
        //////////////////////////////////////////////////////////////////////////////////////////
        // Projection
        //////////////////////////////////////////////////////////////////////////////////////////
        hcompress_project(dist_hmatrix.hnodes.top_level, top_level_basis.level_data, top_level_basis.level_data,
                          top_workspace.u_upsweep, top_workspace.u_upsweep, top_workspace.projection, stream);

#ifdef H2OPUS_PROFILING_ENABLED
        HLibProfile::cumRun(HLibProfile::HCOMPRESS_PROJECTION, PerformanceCounter::getOpCount(), timer.stop());
#endif
    }
}

template <int hw>
void distributed_hcompress_low_rank_exchange(TBasisTree<hw> &basis_branch,
                                             TDistributedCompresseBasisTree<hw> &v_basis_tree,
                                             DistributedHcompressWorkspace &dist_workspace, H2OpusEvents &events,
                                             std::vector<HcompressLowRankExchangeHelper<hw>> &low_rank_exchange_helpers,
                                             std::mutex &upsweep_mutex, std::condition_variable &upsweep_cv,
                                             std::vector<int> &level_processed, int local_rank,
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
            distributed_hcompress_prepare_offDiag_template<hw>(basis_branch, v_basis_tree, dist_workspace, comm_buffer,
                                                               level, stream);
            events.recordEvent<hw>(H2OpusBufferDownEvent, level, stream);
            events.synchEvent<hw>(H2OpusBufferDownEvent, level);

            int old_rank = basis_branch.level_data.getLevelRank(level);
            int new_rank = dist_workspace.branch_workspace.u_upsweep.new_ranks[level];
            low_rank_exchange_helpers[level].setExchangeDataSize(new_rank, old_rank);

            // Send and receive all data
            hcompressExecuteLowRankExchange(&low_rank_exchange_helpers[level]);
        }
    }
    if (phase != 0)
    {
        for (int level = num_levels - 1; level >= 1; level--)
        {
            TDistributedSendRecvBuffer<hw> *comm_buffer = low_rank_exchange_helpers[level].comm_buffer;
            hcompressExecuteLowRankWait(&low_rank_exchange_helpers[level]);

            // Send the received offdiagonal data to the device
            comm_buffer->prepareRecvBuffer(stream);
            events.recordEvent<hw>(H2OpusBufferUpEvent, level, stream);
        }
    }
}

template <int hw>
void distributed_hcompress_upsweep_branch(TBasisTree<hw> &basis_tree, H2Opus_Real eps,
                                          HcompressWorkspace &branch_workspace, H2OpusEvents &events,
                                          std::mutex &upsweep_mutex, std::condition_variable &upsweep_cv,
                                          std::vector<int> &level_processed, MPI_Comm comm,
                                          h2opusComputeStream_t stream)
{
    int num_levels = basis_tree.depth;
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Compress the leaves - since we use a uniform rank per level
    // we have to determine the max rank using an allreduce before
    // we can truncate. When we move to non-uniform ranks, the allreduce
    // can be removed

#ifdef H2OPUS_PROFILING_ENABLED
    Timer<hw> timer;
    timer.init();
    timer.start();

    PerformanceCounter::clearCounters();
#endif

    int local_leaf_rank = hcompress_compressed_basis_leaf_rank(basis_tree, eps, branch_workspace.u_upsweep, stream);

#ifdef H2OPUS_PROFILING_ENABLED
    HLibProfile::cumRun(HLibProfile::HCOMPRESS_TRUNCATE_BASIS, PerformanceCounter::getOpCount(), timer.stop());
#endif

    // TODO WAJIH: Can we merge the allreduce with those below?
    int max_leaf_rank;
    mpiErrchk(MPI_Allreduce(&local_leaf_rank, &max_leaf_rank, 1, H2OPUS_MPI_INT, MPI_MAX, comm));

#ifdef H2OPUS_PROFILING_ENABLED
    timer.start();

    PerformanceCounter::clearCounters();
#endif
    hcompress_truncate_basis_leaves(basis_tree, max_leaf_rank, branch_workspace.u_upsweep, stream);

    events.recordEvent<hw>(H2OpusUpsweepEvent, num_levels - 1, stream);

#ifdef H2OPUS_PROFILING_ENABLED
    HLibProfile::cumRun(HLibProfile::HCOMPRESS_TRUNCATE_BASIS, PerformanceCounter::getOpCount(), timer.stop());
#endif

    {
        std::lock_guard<std::mutex> lock(upsweep_mutex);
        level_processed[num_levels - 1] = 1;
    }
    upsweep_cv.notify_one();

    // Now sweep up the tree
    for (int level = num_levels - 2; level >= 0; level--)
    {
#ifdef H2OPUS_PROFILING_ENABLED
        PerformanceCounter::clearCounters();
        timer.start();
#endif
        int local_level_rank =
            hcompress_compressed_basis_level_rank(basis_tree, eps, level, branch_workspace.u_upsweep, stream);

#ifdef H2OPUS_PROFILING_ENABLED
        double cumo = PerformanceCounter::getOpCount();
        PerformanceCounter::clearCounters();
        double cumt = timer.stop();
#endif

        int max_level_rank;
        mpiErrchk(MPI_Allreduce(&local_level_rank, &max_level_rank, 1, H2OPUS_MPI_INT, MPI_MAX, comm));

#ifdef H2OPUS_PROFILING_ENABLED
        timer.start();
#endif

        hcompress_truncate_basis_level(basis_tree, max_level_rank, level, branch_workspace.u_upsweep, stream);
        events.recordEvent<hw>(H2OpusUpsweepEvent, level, stream);

#ifdef H2OPUS_PROFILING_ENABLED
        cumt += timer.stop();
        cumo += PerformanceCounter::getOpCount();
        HLibProfile::cumRun(HLibProfile::HCOMPRESS_TRUNCATE_BASIS, cumo, cumt);
#endif

        {
            std::lock_guard<std::mutex> lock(upsweep_mutex);
            level_processed[level] = 1;
        }
        upsweep_cv.notify_one();
    }
}

template <int hw>
void distributed_hcompress_downsweep_branch(TDistributedHMatrix<hw> &dist_hmatrix,
                                            DistributedHcompressWorkspace &dist_workspace, H2OpusEvents &events,
                                            H2Opus_Real eps, h2opusComputeStream_t stream)
{
    TBasisTree<hw> &basis_branch = dist_hmatrix.basis_tree.basis_branch;
    HcompressWorkspace &workspace = dist_workspace.branch_workspace;

#ifdef H2OPUS_PROFILING_ENABLED
    Timer<hw> timer;
    timer.init();
    timer.start();

    PerformanceCounter::clearCounters();
#endif

    hcompress_generate_optimal_basis(dist_hmatrix.hnodes.diagonal_block, &(dist_hmatrix.hnodes.off_diagonal_blocks),
                                     BSN_DIRECTION_ROW, basis_branch, workspace.u_upsweep, workspace.optimal_u_bgen, 0,
                                     eps, stream);

#ifdef H2OPUS_PROFILING_ENABLED
    HLibProfile::cumRun(HLibProfile::HCOMPRESS_BASIS_GEN, PerformanceCounter::getOpCount(), timer.stop());
#endif
}

template <int hw>
void distributed_hcompress_downsweep_top_level(TDistributedHMatrix<hw> &dist_hmatrix,
                                               DistributedHcompressWorkspace &dist_workspace, int proc_rank,
                                               TDistributedTransferBuffer<hw> &scatter_buffer, H2OpusEvents &events,
                                               H2Opus_Real eps, h2opusComputeStream_t stream)
{
    H2Opus_Real *Zhat_leaf_level = NULL;
    int branch_root_rank = dist_hmatrix.basis_tree.basis_branch.level_data.getLevelRank(0);
    size_t top_elements = branch_root_rank * branch_root_rank;

    // Sweep down the root branch
    if (proc_rank == 0)
    {
        TBasisTree<hw> &basis_top_level = dist_hmatrix.basis_tree.top_level;
        HcompressWorkspace &workspace = dist_workspace.top_level_workspace;

        // Clear out the root level of the weight tree
        H2Opus_Real *Zhat_root = workspace.u_upsweep.Z_hat[0];
        int root_rank = basis_top_level.level_data.getLevelRank(0);
        fillArray(Zhat_root, root_rank * root_rank, 0, stream, hw);

#ifdef H2OPUS_PROFILING_ENABLED
        Timer<hw> timer;
        timer.init();
        timer.start();

        PerformanceCounter::clearCounters();
#endif
        hcompress_generate_optimal_basis(dist_hmatrix.hnodes.top_level, NULL, BSN_DIRECTION_ROW, basis_top_level,
                                         workspace.u_upsweep, workspace.optimal_u_bgen, 0, eps, stream);

#ifdef H2OPUS_PROFILING_ENABLED
        HLibProfile::cumRun(HLibProfile::HCOMPRESS_BASIS_GEN, PerformanceCounter::getOpCount(), timer.stop());
#endif
        Zhat_leaf_level = workspace.u_upsweep.Z_hat[basis_top_level.depth - 1];
        // dumpMatrixTreeContainer(basis_top_level.level_data, workspace.u_upsweep.Z_hat, 4, hw);
    }

    // Scatter the leaf level of the weight tree to all processes
    HcompressWorkspace &branch_workspace = dist_workspace.branch_workspace;
    H2Opus_Real *Zhat_scatter_root = branch_workspace.u_upsweep.Z_hat[0];

    scatter_buffer.prepareDownBuffer(Zhat_leaf_level, stream);
    scatter_buffer.setTransferBuffers(Zhat_leaf_level, Zhat_scatter_root);
    scatter_buffer.setTransferElements(top_elements, H2OPUS_MPI_REAL);

    events.recordEvent<hw>(H2OpusCommunicationEvent, H2OPUS_DIST_HCOMPRESS_SCATTER_EVENT, stream);
    events.synchEvent<hw>(H2OpusCommunicationEvent, H2OPUS_DIST_HCOMPRESS_SCATTER_EVENT);

    mpiErrchk(MPI_Scatter(scatter_buffer.getDownBuffer(), scatter_buffer.getElements(), scatter_buffer.getType(),
                          scatter_buffer.getUpBuffer(), scatter_buffer.getElements(), scatter_buffer.getType(), 0,
                          scatter_buffer.getComm()));

    scatter_buffer.prepareUpBuffer(Zhat_scatter_root, stream);
}

template <int hw>
void distributed_hcompress_prepare_comm_buffers_template(TDistributedHMatrix<hw> &dist_hmatrix,
                                                         DistributedHcompressWorkspace &dist_workspace,
                                                         distributedH2OpusHandle_t dist_h2opus_handle)
{
    TDistributedCompresseBasisTree<hw> &v_basis_tree = dist_hmatrix.compressed_basis_tree_data;
    TBasisTree<hw> &u_branch = dist_hmatrix.basis_tree.basis_branch;

    // Make sure we have enough buffers to accommodate the depth of this hierarchical matrix
    dist_h2opus_handle->allocateBuffers<hw>(u_branch.depth + 1);

    // TDistributedSendRecvBuffer<hw> &dense_comm_buffer = dist_h2opus_handle->getSendRecvBuffer<hw>(0);
    TDistributedTransferBuffer<hw> &gather_buffer = dist_h2opus_handle->getGatherBuffer<hw>();
    TDistributedTransferBuffer<hw> &scatter_buffer = dist_h2opus_handle->getScatterBuffer<hw>();

    // Upsweep gather buffer -  we don't know what the compressed rank is going to be
    // so just allocate based on the current rank
    int top_rank = u_branch.level_data.getLevelRank(0);
    size_t top_elements = top_rank * top_rank;
    size_t branch_elements = 0;
    if (dist_h2opus_handle->rank == 0)
        branch_elements = top_elements * dist_h2opus_handle->num_ranks;

    gather_buffer.setBufferSizes(top_elements * sizeof(H2Opus_Real), branch_elements * sizeof(H2Opus_Real));
    gather_buffer.setComm(dist_h2opus_handle->commgather);

    // Downsweep scatter buffer
    branch_elements = top_elements;
    top_elements = 0;
    if (dist_h2opus_handle->rank == 0)
        top_elements = branch_elements * dist_h2opus_handle->num_ranks;

    scatter_buffer.setBufferSizes(top_elements * sizeof(H2Opus_Real), branch_elements * sizeof(H2Opus_Real));
    scatter_buffer.setComm(dist_h2opus_handle->commscatter);

    // Low rank buffers
    int depth = u_branch.depth;
    for (int level = 0; level < depth; level++)
    {
        TDistributedSendRecvBuffer<hw> &comm_buffer = dist_h2opus_handle->getSendRecvBuffer<hw>(level + 1);
        TDistributedCompressedBSNData<hw> &bsn_data = v_basis_tree.coupling_compressed_bsn_data[level];

        int level_rank = u_branch.level_data.getLevelRank(level);

        size_t level_elements = level_rank * level_rank;
        size_t nodes_to_send = bsn_data.send_process_nodes.size();
        size_t node_to_receive = bsn_data.receive_process_nodes.size();

        comm_buffer.allocateBuffers(nodes_to_send * level_elements * sizeof(H2Opus_Real),
                                    node_to_receive * level_elements * sizeof(H2Opus_Real));

        comm_buffer.setRequests(bsn_data.send_process_ids.size(), bsn_data.receive_process_ids.size());
        comm_buffer.setComm(dist_h2opus_handle->comm);
        comm_buffer.setTag(dist_h2opus_handle->getNewTag());
    }
}

template <int hw>
void distributed_hcompress_template(TDistributedHMatrix<hw> &dist_hmatrix, H2Opus_Real eps,
                                    distributedH2OpusHandle_t dist_h2opus_handle)
{
    // If we are profiling, initialize run here and partially sum performances
#ifdef H2OPUS_PROFILING_ENABLED
    HLibProfile::addRun(HLibProfile::HCOMPRESS_STITCH, 0, 0);
    HLibProfile::addRun(HLibProfile::HCOMPRESS_PROJECTION, 0, 0);
    HLibProfile::addRun(HLibProfile::HCOMPRESS_TRUNCATE_BASIS, 0, 0);
    HLibProfile::addRun(HLibProfile::HCOMPRESS_BASIS_GEN, 0, 0);
#endif

    if (!dist_h2opus_handle->active)
        return;

    h2opusHandle_t h2opus_handle = dist_h2opus_handle->handle;
    h2opusHandle_t top_level_handle = dist_h2opus_handle->top_level_handle;

    H2OpusEvents &events = h2opus_handle->getEvents();
    h2opusComputeStream_t main_stream = h2opus_handle->getMainStream();
    h2opusComputeStream_t secondary_stream = h2opus_handle->getSecondaryStream();

    //////////////////////////////////////////////////////////////////////////////////////////
    // Workspace allocation
    //////////////////////////////////////////////////////////////////////////////////////////
    H2OpusWorkspaceState ws_needed, top_level_ws_needed;
    distributed_hcompress_workspace(dist_hmatrix, ws_needed, top_level_ws_needed, dist_h2opus_handle);

    H2OpusWorkspaceState ws_allocated = h2opus_handle->getWorkspaceState();
    H2OpusWorkspaceState top_level_ws_allocated = top_level_handle->getWorkspaceState();

    if (ws_allocated < ws_needed)
        h2opus_handle->setWorkspaceState(ws_needed);

    if (top_level_ws_allocated < top_level_ws_needed)
        top_level_handle->setWorkspaceState(top_level_ws_needed);

    DistributedHcompressWorkspace dist_workspace;
    distributed_hcompress_get_workspace(dist_hmatrix, dist_workspace, dist_h2opus_handle);

    // Prepare all communication buffers
    distributed_hcompress_prepare_comm_buffers_template<hw>(dist_hmatrix, dist_workspace, dist_h2opus_handle);

    TDistributedTransferBuffer<hw> &gather_buffer = dist_h2opus_handle->getGatherBuffer<hw>();
    TDistributedTransferBuffer<hw> &scatter_buffer = dist_h2opus_handle->getScatterBuffer<hw>();

    // Allocate events
    TBasisTree<hw> &basis_branch = dist_hmatrix.basis_tree.basis_branch;
    int branch_depth = basis_branch.depth;
    events.allocateEvents<hw>(H2OpusUpsweepEvent, branch_depth);
    events.allocateEvents<hw>(H2OpusBufferUpEvent, branch_depth);
    events.allocateEvents<hw>(H2OpusBufferDownEvent, branch_depth);
    events.allocateEvents<hw>(H2OpusDownsweepEvent, branch_depth);
    events.allocateEvents<hw>(H2OpusCommunicationEvent, H2OPUS_DIST_HCOMPRESS_TOTAL_EVENTS);

    //////////////////////////////////////////////////////////////////////////////////////////
    // Helper structs for the streaming callbacks
    //////////////////////////////////////////////////////////////////////////////////////////
    std::vector<HcompressLowRankExchangeHelper<hw>> low_rank_exchange_helpers(branch_depth);
    TDistributedCompresseBasisTree<hw> &compressed_tree = dist_hmatrix.compressed_basis_tree_data;

    for (int level = 0; level < branch_depth; level++)
    {
        TDistributedCompressedBSNData<hw> &bsn_data = compressed_tree.coupling_compressed_bsn_data[level];
        TDistributedSendRecvBuffer<hw> &comm_buffer = dist_h2opus_handle->getSendRecvBuffer<hw>(level + 1);
        low_rank_exchange_helpers[level].init(&bsn_data, &comm_buffer);
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Top level downsweep on the master process - nothing can proceed until this is done
    // Stream: Main
    //////////////////////////////////////////////////////////////////////////////////////////
    distributed_hcompress_downsweep_top_level<hw>(dist_hmatrix, dist_workspace, dist_h2opus_handle->rank,
                                                  scatter_buffer, events, eps, main_stream);

    //////////////////////////////////////////////////////////////////////////////////////////
    // Branch downsweep in parallel
    // Stream: Main
    //////////////////////////////////////////////////////////////////////////////////////////
    distributed_hcompress_downsweep_branch<hw>(dist_hmatrix, dist_workspace, events, eps, main_stream);

    // See if we can use customized threaded communications
    bool usethreads = dist_h2opus_handle->getUseThreads<hw>();

    //////////////////////////////////////////////////////////////////////////////////////////
    // Branch upsweep in parallel
    // Stream: Main
    //////////////////////////////////////////////////////////////////////////////////////////
    // Mutex and condition variable for the levels of the upsweep
    std::mutex upsweep_mutex;
    std::condition_variable upsweep_cv;
    std::vector<int> level_processed(branch_depth, 0);
    std::thread *low_rank_exchange_thread = nullptr, *root_branch_thread = nullptr;

    if (usethreads)
    {
        low_rank_exchange_thread = new std::thread(
            distributed_hcompress_low_rank_exchange<hw>, std::ref(basis_branch), std::ref(compressed_tree),
            std::ref(dist_workspace), std::ref(events), std::ref(low_rank_exchange_helpers), std::ref(upsweep_mutex),
            std::ref(upsweep_cv), std::ref(level_processed), dist_h2opus_handle->local_rank, secondary_stream, -1);
    }
    distributed_hcompress_upsweep_branch<hw>(basis_branch, eps, dist_workspace.branch_workspace, events, upsweep_mutex,
                                             upsweep_cv, level_processed, dist_h2opus_handle->comm, main_stream);

    if (usethreads)
    {
        root_branch_thread =
            new std::thread(distributed_hcompress_process_toplevel_template<hw>, std::ref(dist_hmatrix), eps,
                            std::ref(dist_workspace), std::ref(events), std::ref(gather_buffer), main_stream,
                            dist_h2opus_handle->rank, dist_h2opus_handle->local_rank);
    }
    else
    {
        distributed_hcompress_low_rank_exchange(basis_branch, compressed_tree, dist_workspace, events,
                                                low_rank_exchange_helpers, upsweep_mutex, upsweep_cv, level_processed,
                                                dist_h2opus_handle->local_rank, secondary_stream, 0);
    }
    // Project the diagonal block into the new compact basis
    // Stream: main
    distributed_hcompress_project_diagonal_template<hw>(dist_hmatrix.hnodes, dist_hmatrix.basis_tree, dist_workspace,
                                                        main_stream);

    // Project the offdiagonal blocks
    // Stream: main
    if (usethreads)
        low_rank_exchange_thread->join();
    else
    {
        distributed_hcompress_low_rank_exchange(basis_branch, compressed_tree, dist_workspace, events,
                                                low_rank_exchange_helpers, upsweep_mutex, upsweep_cv, level_processed,
                                                dist_h2opus_handle->local_rank, secondary_stream, 1);
    }
    distributed_hcompress_project_offdiagonal_template<hw>(
        dist_hmatrix.hnodes, dist_hmatrix.basis_tree, low_rank_exchange_helpers, dist_workspace, events, main_stream);

    if (usethreads)
        root_branch_thread->join();
    else
    {
        // We do a blocking gather here
        distributed_hcompress_process_toplevel_template(dist_hmatrix, eps, dist_workspace, events, gather_buffer,
                                                        main_stream, dist_h2opus_handle->rank,
                                                        dist_h2opus_handle->local_rank);
    }

    // Update the rank level data
    basis_branch.level_data.setLevelRanks(vec_ptr(dist_workspace.branch_workspace.u_upsweep.new_ranks));

    dist_hmatrix.hnodes.diagonal_block.level_data.setRankFromBasis(basis_branch.level_data, 0);

    dist_hmatrix.hnodes.off_diagonal_blocks.level_data.setRankFromBasis(basis_branch.level_data, 0);

    if (dist_h2opus_handle->rank == 0)
    {
        dist_hmatrix.basis_tree.top_level.level_data.setLevelRanks(
            vec_ptr(dist_workspace.top_level_workspace.u_upsweep.new_ranks));

        dist_hmatrix.hnodes.top_level.level_data.setRankFromBasis(dist_hmatrix.basis_tree.top_level.level_data, 0);
    }
    delete low_rank_exchange_thread;
    delete root_branch_thread;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Interface routines
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU
/////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU
void distributed_hcompress(DistributedHMatrix_GPU &dist_hmatrix, H2Opus_Real eps,
                           distributedH2OpusHandle_t dist_h2opus_handle)
{
    distributed_hcompress_template<H2OPUS_HWTYPE_GPU>(dist_hmatrix, eps, dist_h2opus_handle);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void distributed_hcompress(DistributedHMatrix &dist_hmatrix, H2Opus_Real eps,
                           distributedH2OpusHandle_t dist_h2opus_handle)
{
    distributed_hcompress_template<H2OPUS_HWTYPE_CPU>(dist_hmatrix, eps, dist_h2opus_handle);
}
