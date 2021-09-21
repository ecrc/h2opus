#include <h2opus/core/hgemv.h>
#include <h2opus/marshal/hgemv_marshal.h>

#include <h2opus/util/perf_counter.h>
#include <h2opus/util/timer.h>

#include <h2opus/util/batch_wrappers.h>
#include <h2opus/util/debug_routines.h>
#include <h2opus/util/gpu_err_check.h>
#include <h2opus/util/thrust_wrappers.h>

// #define H2OPUS_DISABLE_STREAMED_HGEMV

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Template routines
/////////////////////////////////////////////////////////////////////////////////////////////////////////
template <int hw>
void hgemv_upsweep_leaves_template(H2Opus_Real alpha, TBasisTree<hw> &basis_tree, H2Opus_Real *X, int ldx,
                                   int num_vectors, HgemvWorkspace &workspace, h2opusComputeStream_t stream)
{
    size_t num_leaves = basis_tree.basis_leaves;
    BasisTreeLevelData &level_data = basis_tree.level_data;
    VectorTree &xhat = workspace.xhat;
    int num_levels = basis_tree.depth;

    if (num_leaves == 0)
        return;

    // Allocate pointers so we can marshal batch arguments
    BatchGemmMarshalledData &marshal_data = workspace.low_rank_gemms;
    H2Opus_Real **A_ptrs = marshal_data.A_ptrs;
    H2Opus_Real **B_ptrs = marshal_data.B_ptrs;
    H2Opus_Real **C_ptrs = marshal_data.C_ptrs;

    int *m_batch = marshal_data.m_batch, *n_batch = marshal_data.n_batch, *k_batch = marshal_data.k_batch;
    int *lda_batch = marshal_data.lda_batch, *ldb_batch = marshal_data.ldb_batch, *ldc_batch = marshal_data.ldc_batch;

    H2Opus_Real *v_basis = basis_tree.getBasisLeafData();
    int leaf_rank = level_data.getLevelRank(num_levels - 1);
    int leaf_size = basis_tree.leaf_size;

    H2Opus_Real *xhat_leaf_base = xhat.data[num_levels - 1];
    size_t leaf_level_start = level_data.getLevelStart(num_levels - 1);

    hgemv_upsweep_leaves_batch_marshal<H2Opus_Real, hw>(
        v_basis, leaf_size, leaf_rank, A_ptrs, X, ldx, num_vectors, B_ptrs, xhat_leaf_base, C_ptrs, leaf_level_start,
        vec_ptr(basis_tree.node_start), vec_ptr(basis_tree.node_len), m_batch, n_batch, k_batch, lda_batch, ldb_batch,
        ldc_batch, num_leaves, stream);

    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, m_batch, n_batch,
                                                             k_batch, leaf_rank, num_vectors, leaf_size, alpha,
                                                             (const H2Opus_Real **)A_ptrs, lda_batch,
                                                             (const H2Opus_Real **)B_ptrs, ldb_batch, (H2Opus_Real)0,
                                                             C_ptrs, ldc_batch, num_leaves));
}

template <int hw>
void hgemv_upsweep_level_template(TBasisTree<hw> &basis_tree, int level, int num_vectors, HgemvWorkspace &workspace,
                                  h2opusComputeStream_t stream)
{
    BasisTreeLevelData &level_data = basis_tree.level_data;
    VectorTree &xhat = workspace.xhat;
    int max_children = basis_tree.max_children;

    // Allocate pointers so we can marshal batch arguments
    BatchGemmMarshalledData &marshal_data = workspace.low_rank_gemms;
    H2Opus_Real **A_ptrs = marshal_data.A_ptrs;
    H2Opus_Real **B_ptrs = marshal_data.B_ptrs;
    H2Opus_Real **C_ptrs = marshal_data.C_ptrs;

    H2Opus_Real *xhat_child_level = xhat.data[level + 1];
    H2Opus_Real *xhat_parent_level = xhat.data[level];

    int child_rank = level_data.getLevelRank(level + 1);
    int parent_rank = level_data.getLevelRank(level);

    size_t num_children = level_data.getLevelSize(level + 1);
    size_t num_parents = level_data.getLevelSize(level);

    size_t child_start = level_data.getLevelStart(level + 1);
    size_t parent_start = level_data.getLevelStart(level);

    // Get the basis data pointer for this level of the tree
    H2Opus_Real *v_child_level = basis_tree.getTransLevelData(level + 1);

    hgemv_upsweep_batch_marshal<H2Opus_Real, hw>(A_ptrs, B_ptrs, C_ptrs, max_children, num_parents, num_children,
                                                 num_vectors, v_child_level, xhat_child_level, xhat_parent_level,
                                                 parent_rank, child_rank, parent_start, child_start,
                                                 basis_tree.head_ptr(), basis_tree.next_ptr(), stream);

    // Go through the children one by one and perform the batch gemv
    // the first operation will have beta = 0, and the rest will have
    // beta = 1 so that we can accumulate the results
    for (int i = 0; i < max_children; i++)
    {
        H2Opus_Real beta = (i == 0 ? 0 : 1);
        H2Opus_Real alpha = 1;

        check_kblas_error(
            (H2OpusBatched<H2Opus_Real, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, parent_rank, num_vectors,
                                                   child_rank, alpha, (const H2Opus_Real **)A_ptrs + i * num_parents,
                                                   child_rank, (const H2Opus_Real **)B_ptrs + i * num_parents,
                                                   child_rank, beta, C_ptrs + i * num_parents, parent_rank,
                                                   num_parents));
    }
}

template <int hw>
void hgemv_upsweep_template(H2Opus_Real alpha, TBasisTree<hw> &basis_tree, H2Opus_Real *X, int ldx, int num_vectors,
                            HgemvWorkspace &workspace, h2opusComputeStream_t stream)
{
    // Handle the leaves
    hgemv_upsweep_leaves_template<hw>(alpha, basis_tree, X, ldx, num_vectors, workspace, stream);

    // Sweep up the tree
    int num_levels = basis_tree.depth;
    for (int level = num_levels - 2; level >= 0; level--)
        hgemv_upsweep_level_template<hw>(basis_tree, level, num_vectors, workspace, stream);
}

template <int hw>
void hgemv_downsweep_leaves_template(TBasisTree<hw> &basis_tree, H2Opus_Real *Y, int ldy, int num_vectors,
                                     HgemvWorkspace &workspace, h2opusComputeStream_t stream)
{
    BasisTreeLevelData &level_data = basis_tree.level_data;
    VectorTree &yhat = workspace.yhat;

    int num_levels = basis_tree.depth;

    BatchGemmMarshalledData &marshal_data = workspace.low_rank_gemms;
    H2Opus_Real **A_ptrs = marshal_data.A_ptrs;
    H2Opus_Real **B_ptrs = marshal_data.B_ptrs;
    H2Opus_Real **C_ptrs = marshal_data.C_ptrs;

    int *m_batch = marshal_data.m_batch, *n_batch = marshal_data.n_batch, *k_batch = marshal_data.k_batch;
    int *lda_batch = marshal_data.lda_batch, *ldb_batch = marshal_data.ldb_batch, *ldc_batch = marshal_data.ldc_batch;

    size_t num_leaves = basis_tree.basis_leaves;
    if (num_leaves != 0)
    {
        H2Opus_Real *yhat_leaf_base = yhat.data[num_levels - 1];
        H2Opus_Real *u_basis = basis_tree.getBasisLeafData();

        int leaf_rank = level_data.getLevelRank(num_levels - 1);
        int leaf_size = basis_tree.leaf_size;

        size_t leaf_level_start = level_data.getLevelStart(num_levels - 1);

        hgemv_downsweep_leaves_batch_marshal<H2Opus_Real, hw>(
            u_basis, leaf_size, leaf_rank, A_ptrs, yhat_leaf_base, B_ptrs, Y, ldy, num_vectors, C_ptrs,
            leaf_level_start, vec_ptr(basis_tree.node_start), vec_ptr(basis_tree.node_len), m_batch, n_batch, k_batch,
            lda_batch, ldb_batch, ldc_batch, num_leaves, stream);

        check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, m_batch,
                                                                 n_batch, k_batch, leaf_size, num_vectors, leaf_rank,
                                                                 (H2Opus_Real)1, (const H2Opus_Real **)A_ptrs,
                                                                 lda_batch, (const H2Opus_Real **)B_ptrs, ldb_batch,
                                                                 (H2Opus_Real)1, C_ptrs, ldc_batch, num_leaves));
    }
}

template <int hw>
void hgemv_downsweep_level_template(TBasisTree<hw> &basis_tree, int num_vectors, int level, HgemvWorkspace &workspace,
                                    h2opusComputeStream_t stream)
{
    BasisTreeLevelData &level_data = basis_tree.level_data;
    VectorTree &yhat = workspace.yhat;

    // Allocate pointers so we can generate batch arguments
    BatchGemmMarshalledData &marshal_data = workspace.low_rank_gemms;
    H2Opus_Real **A_ptrs = marshal_data.A_ptrs;
    H2Opus_Real **B_ptrs = marshal_data.B_ptrs;
    H2Opus_Real **C_ptrs = marshal_data.C_ptrs;

    H2Opus_Real *yhat_child_level = yhat.data[level];
    H2Opus_Real *yhat_parent_level = yhat.data[level - 1];

    int child_rank = level_data.getLevelRank(level);
    int parent_rank = level_data.getLevelRank(level - 1);

    size_t num_children = level_data.getLevelSize(level);
    size_t num_parents = level_data.getLevelSize(level - 1);

    size_t child_start = level_data.getLevelStart(level);
    size_t parent_start = level_data.getLevelStart(level - 1);

    // Get the basis data pointer for this level of the tree
    H2Opus_Real *u_trans_level = basis_tree.getTransLevelData(level);

    hgemv_downsweep_batch_marshal<H2Opus_Real, hw>(
        A_ptrs, B_ptrs, C_ptrs, num_children, num_parents, num_vectors, u_trans_level, yhat_parent_level,
        yhat_child_level, child_rank, parent_rank, child_start, parent_start, basis_tree.parent_ptr(), stream);

    H2Opus_Real beta = 1, alpha = 1;

    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, child_rank,
                                                             num_vectors, parent_rank, alpha,
                                                             (const H2Opus_Real **)A_ptrs, child_rank,
                                                             (const H2Opus_Real **)B_ptrs, parent_rank, beta, C_ptrs,
                                                             child_rank, num_children));
}

template <int hw>
void hgemv_downsweep_template(TBasisTree<hw> &basis_tree, H2Opus_Real *Y, int ldy, int num_vectors,
                              HgemvWorkspace &workspace, h2opusComputeStream_t stream)
{
    int num_levels = basis_tree.depth;

    //////////////////////////////////////////////////////////
    // Sweep down the tree
    //////////////////////////////////////////////////////////
    for (int level = 1; level < num_levels; level++)
        hgemv_downsweep_level_template<hw>(basis_tree, num_vectors, level, workspace, stream);

    //////////////////////////////////////////////////////////
    // Handle the leaves
    //////////////////////////////////////////////////////////
    hgemv_downsweep_leaves_template<hw>(basis_tree, Y, ldy, num_vectors, workspace, stream);
}

template <int hw>
void hgemv_mult_level_template(THNodeTree<hw> &hnodes, int level, int num_vectors, size_t u_index_offset,
                               size_t v_index_offset, H2Opus_Real *xhat_level, H2Opus_Real *yhat_level,
                               BatchGemmMarshalledData &marshal_data,
                               typename THNodeTree<hw>::HNodeTreeBSNData *bsn_data, int *column_basis_indexes,
                               int *row_basis_indexes, int hblas_trans_mode, h2opusComputeStream_t stream)
{
    HNodeTreeLevelData &level_data = hnodes.level_data;

    size_t num_nodes = level_data.getCouplingLevelSize(level);
    size_t node_offset = level_data.getCouplingLevelStart(level);
    int node_size = level_data.getLevelRank(level);
    int block_rows = hnodes.getLevelBSRRows(level);

    if (block_rows == 0 || node_size == 0 || num_nodes == 0)
        return;

    H2Opus_Real **A_ptrs = marshal_data.A_ptrs;
    H2Opus_Real **B_ptrs = marshal_data.B_ptrs;
    H2Opus_Real **C_ptrs = marshal_data.C_ptrs;

    hgemv_mult_batch_marshal<H2Opus_Real, hw>(
        row_basis_indexes, column_basis_indexes, vec_ptr(hnodes.rank_leaf_tree_index),
        vec_ptr(bsn_data->coupling_batch_indexes[level]), node_offset, u_index_offset, v_index_offset, xhat_level,
        vec_ptr(hnodes.rank_leaf_mem[level]), node_size, num_vectors, yhat_level, A_ptrs, B_ptrs, C_ptrs, num_nodes,
        stream);

    std::vector<int> &coupling_batch_ptr = bsn_data->coupling_batch_ptr[level];
    int num_batches = (int)coupling_batch_ptr.size() - 1;
    for (int batch_id = 0; batch_id < num_batches; batch_id++)
    {
        H2Opus_Real **A_batch = A_ptrs + coupling_batch_ptr[batch_id];
        H2Opus_Real **B_batch = B_ptrs + coupling_batch_ptr[batch_id];
        H2Opus_Real **C_batch = C_ptrs + coupling_batch_ptr[batch_id];

        int batch_size = coupling_batch_ptr[batch_id + 1] - coupling_batch_ptr[batch_id];

        check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(stream, hblas_trans_mode, H2Opus_NoTrans, node_size,
                                                                 num_vectors, node_size, (H2Opus_Real)1,
                                                                 (const H2Opus_Real **)A_batch, node_size,
                                                                 (const H2Opus_Real **)B_batch, node_size,
                                                                 (H2Opus_Real)1, C_batch, node_size, batch_size));
    }
}

template <int hw>
void hgemv_mult_template(int trans, THNodeTree<hw> &hnodes, int start_level, int end_level, int num_vectors,
                         TBasisTree<hw> &u_basis_tree, TBasisTree<hw> &v_basis_tree, HgemvWorkspace &workspace,
                         h2opusComputeStream_t stream)
{
    typedef typename THNodeTree<hw>::HNodeTreeBSNData BSNData;
    bool transpose = (trans == H2Opus_Trans);

    BasisTreeLevelData &u_level_data = (transpose ? v_basis_tree.level_data : u_basis_tree.level_data);
    BasisTreeLevelData &v_level_data = (transpose ? u_basis_tree.level_data : v_basis_tree.level_data);

    HNodeTreeLevelData &level_data = hnodes.level_data;
    VectorTree &xhat = workspace.xhat;
    VectorTree &yhat = workspace.yhat;

    int num_levels = level_data.depth;
    assert(num_levels > end_level);

    BatchGemmMarshalledData &marshal_data = workspace.low_rank_gemms;
    BSNData &bsn_data = (transpose ? hnodes.bsn_col_data : hnodes.bsn_row_data);
    int *column_basis_indexes = (transpose ? vec_ptr(hnodes.node_u_index) : vec_ptr(hnodes.node_v_index));
    int *row_basis_indexes = (transpose ? vec_ptr(hnodes.node_v_index) : vec_ptr(hnodes.node_u_index));

    for (int level = end_level; level >= start_level; level--)
    {
        size_t u_index_offset = u_level_data.getLevelStart(level);
        size_t v_index_offset = v_level_data.getLevelStart(level);

        H2Opus_Real *X = xhat.data[level];
        H2Opus_Real *Y = yhat.data[level];

        hgemv_mult_level_template<hw>(hnodes, level, num_vectors, u_index_offset, v_index_offset, X, Y, marshal_data,
                                      &bsn_data, column_basis_indexes, row_basis_indexes, trans, stream);
    }
}

template <int hw>
void hgemv_denseMult_template(int hblas_trans_mode, H2Opus_Real alpha, THNodeTree<hw> &hnodes, H2Opus_Real *X, int ldx,
                              H2Opus_Real beta, H2Opus_Real *Y, int ldy, int num_vectors, int *node_u_start,
                              int *node_v_start, int *node_u_len, int *node_v_len, int *column_basis_indexes,
                              int *row_basis_indexes, typename THNodeTree<hw>::HNodeTreeBSNData *bsn_data,
                              HgemvWorkspace &workspace, h2opusComputeStream_t stream)
{
    size_t num_nodes = hnodes.num_dense_leaves;
    int node_size = hnodes.leaf_size;
    size_t node_offset = 0;

    BatchGemmMarshalledData &marshal_data = workspace.dense_gemms;

    H2Opus_Real **A_ptrs = marshal_data.A_ptrs;
    H2Opus_Real **B_ptrs = marshal_data.B_ptrs;
    H2Opus_Real **C_ptrs = marshal_data.C_ptrs;

    int *m_ptr = marshal_data.m_batch, *n_ptr = marshal_data.n_batch, *k_ptr = marshal_data.k_batch;
    int *lda_ptr = marshal_data.lda_batch, *ldb_ptr = marshal_data.ldb_batch, *ldc_ptr = marshal_data.ldc_batch;

    // Marshall all the matrix data from the block sparse node data structure
    // in several batches
    hgemv_dense_mult_batch_marshal<H2Opus_Real, hw>(
        row_basis_indexes, column_basis_indexes, node_u_start, node_v_start, node_u_len, node_v_len,
        vec_ptr(hnodes.dense_leaf_tree_index), vec_ptr(bsn_data->dense_batch_indexes), node_offset, X, ldx,
        vec_ptr(hnodes.dense_leaf_mem), node_size, Y, ldy, num_vectors, A_ptrs, B_ptrs, C_ptrs, m_ptr, n_ptr, k_ptr,
        lda_ptr, ldb_ptr, ldc_ptr, num_nodes, stream);

    std::vector<int> &dense_batch_ptr = bsn_data->dense_batch_ptr;
    int num_batches = (int)dense_batch_ptr.size() - 1;

    // Execute each batch on a low priority stream if possible
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
        H2Opus_Real batch_beta = (batch_id == 0 ? beta : 1);

        check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(stream, hblas_trans_mode, H2Opus_NoTrans, m_batch,
                                                                 n_batch, k_batch, node_size, num_vectors, node_size,
                                                                 alpha, (const H2Opus_Real **)A_batch, lda_batch,
                                                                 (const H2Opus_Real **)B_batch, ldb_batch, batch_beta,
                                                                 C_batch, ldc_batch, batch_size));
    }
}

template <int hw>
void hgemv_denseMult_template(int trans, H2Opus_Real alpha, THNodeTree<hw> &hnodes, H2Opus_Real *X, int ldx,
                              H2Opus_Real beta, H2Opus_Real *Y, int ldy, int num_vectors, TBasisTree<hw> &u_basis_tree,
                              TBasisTree<hw> &v_basis_tree, HgemvWorkspace &workspace, h2opusComputeStream_t stream)
{
    typedef typename THNodeTree<hw>::HNodeTreeBSNData BSNData;

    int *node_u_start, *node_v_start, *node_u_len, *node_v_len;
    bool transpose = (trans == H2Opus_Trans);

    int *column_basis_indexes, *row_basis_indexes;
    BSNData *bsn_data;

    if (transpose)
    {
        node_u_start = vec_ptr(v_basis_tree.node_start);
        node_u_len = vec_ptr(v_basis_tree.node_len);
        node_v_start = vec_ptr(u_basis_tree.node_start);
        node_v_len = vec_ptr(u_basis_tree.node_len);

        bsn_data = &(hnodes.bsn_col_data);
        column_basis_indexes = vec_ptr(hnodes.node_u_index);
        row_basis_indexes = vec_ptr(hnodes.node_v_index);
    }
    else
    {
        node_u_start = vec_ptr(u_basis_tree.node_start);
        node_u_len = vec_ptr(u_basis_tree.node_len);
        node_v_start = vec_ptr(v_basis_tree.node_start);
        node_v_len = vec_ptr(v_basis_tree.node_len);

        bsn_data = &(hnodes.bsn_row_data);
        column_basis_indexes = vec_ptr(hnodes.node_v_index);
        row_basis_indexes = vec_ptr(hnodes.node_u_index);
    }

    hgemv_denseMult_template<hw>(trans, alpha, hnodes, X, ldx, beta, Y, ldy, num_vectors, node_u_start, node_v_start,
                                 node_u_len, node_v_len, column_basis_indexes, row_basis_indexes, bsn_data, workspace,
                                 stream);
}

template <int hw>
void hgemv_template(int trans, H2Opus_Real alpha, THMatrix<hw> &hmatrix, H2Opus_Real *X, int ldx, H2Opus_Real beta,
                    H2Opus_Real *Y, int ldy, int num_vectors, h2opusHandle_t h2opus_handle)
{
    // Don't use the transpose code if the matrix is symmetric
    if (trans == H2Opus_Trans && hmatrix.sym)
        trans = H2Opus_NoTrans;

    H2OpusEvents &events = h2opus_handle->getEvents();
    TBasisTree<hw> &u_basis_tree = hmatrix.u_basis_tree;
    TBasisTree<hw> &v_basis_tree = (hmatrix.sym ? hmatrix.u_basis_tree : hmatrix.v_basis_tree);

    H2OpusWorkspaceState ws_needed = hgemv_workspace(hmatrix, trans, num_vectors);
    H2OpusWorkspaceState ws_allocated = h2opus_handle->getWorkspaceState();

    if (ws_allocated < ws_needed)
    {
        // printf("Insufficient workspace for hgemv...allocating...\n");
        h2opus_handle->setWorkspaceState(ws_needed);
    }

    h2opusComputeStream_t main_stream = h2opus_handle->getMainStream();
#ifndef H2OPUS_DISABLE_STREAMED_HGEMV
    h2opusComputeStream_t low_priority_stream = h2opus_handle->getLowPriorityStream();
    events.allocateEvents<hw>(H2OpusDenseEvent, 1);
#else
    h2opusComputeStream_t low_priority_stream = main_stream;
#endif

    HgemvWorkspace workspace;
    hgemv_get_workspace(hmatrix, trans, num_vectors, workspace, h2opus_handle);

#ifdef H2OPUS_PROFILING_ENABLED
    Timer<hw> timer;
    timer.init();
    timer.start();
#endif

    // Dense multiplication phsae
    hgemv_denseMult_template<hw>(trans, alpha, hmatrix.hnodes, X, ldx, beta, Y, ldy, num_vectors, u_basis_tree,
                                 v_basis_tree, workspace, low_priority_stream);
#ifndef H2OPUS_DISABLE_STREAMED_HGEMV
    events.recordEvent<hw>(H2OpusDenseEvent, 0, low_priority_stream);
#endif

#ifdef H2OPUS_PROFILING_ENABLED
    double dense_timer = timer.stop();
    double dense_gops = PerformanceCounter::getOpCount(PerformanceCounter::GEMM);
    PerformanceCounter::clearCounters();
    HLibProfile::addRun(HLibProfile::HGEMV_DENSE, dense_gops, dense_timer);

    timer.start();
#endif

    // Upsweep phase
    if (trans == H2Opus_Trans)
        hgemv_upsweep_template<hw>(alpha, u_basis_tree, X, ldx, num_vectors, workspace, main_stream);
    else
        hgemv_upsweep_template<hw>(alpha, v_basis_tree, X, ldx, num_vectors, workspace, main_stream);

        // dumpHgemvTreeContainer(u_basis_tree.level_data, workspace.xhat.data, num_vectors, 4);

#ifdef H2OPUS_PROFILING_ENABLED
    double upsweep_timer = timer.stop();
    double upsweep_gops = PerformanceCounter::getOpCount(PerformanceCounter::GEMM);
    PerformanceCounter::clearCounters();
    HLibProfile::addRun(HLibProfile::HGEMV_UPSWEEP, upsweep_gops, upsweep_timer);

    timer.start();
#endif

    // Mult phase
    hgemv_mult_template<hw>(trans, hmatrix.hnodes, 0, hmatrix.hnodes.depth - 1, num_vectors, u_basis_tree, v_basis_tree,
                            workspace, main_stream);

    // dumpHgemvTreeContainer(v_basis_tree.level_data, workspace.yhat.data, num_vectors, 4, hw);

#ifdef H2OPUS_PROFILING_ENABLED
    double mult_timer = timer.stop();
    double mult_gops = PerformanceCounter::getOpCount(PerformanceCounter::GEMM);
    PerformanceCounter::clearCounters();
    HLibProfile::addRun(HLibProfile::HGEMV_MULT, mult_gops, mult_timer);

    timer.start();
#endif

#ifndef H2OPUS_DISABLE_STREAMED_HGEMV
    events.streamWaitEvent<hw>(H2OpusDenseEvent, main_stream, 0);
#endif
    // Downsweep phase
    if (trans == H2Opus_Trans)
        hgemv_downsweep_template<hw>(v_basis_tree, Y, ldy, num_vectors, workspace, main_stream);
    else
        hgemv_downsweep_template<hw>(u_basis_tree, Y, ldy, num_vectors, workspace, main_stream);

        // dumpHgemvTreeContainer(v_basis_tree.level_data, workspace.yhat.data, num_vectors, 4, hw);

#ifdef H2OPUS_PROFILING_ENABLED
    double downsweep_timer = timer.stop();
    double downsweep_gops = PerformanceCounter::getOpCount(PerformanceCounter::GEMM);
    PerformanceCounter::clearCounters();
    HLibProfile::addRun(HLibProfile::HGEMV_DOWNSWEEP, downsweep_gops, downsweep_timer);

    timer.destroy();
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Interface routines
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU
/////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU
void hgemv_upsweep(H2Opus_Real alpha, BasisTree_GPU &basis_tree, H2Opus_Real *X, int ldx, int num_vectors,
                   HgemvWorkspace &workspace, h2opusComputeStream_t stream)
{
    hgemv_upsweep_template<H2OPUS_HWTYPE_GPU>(alpha, basis_tree, X, ldx, num_vectors, workspace, stream);
}

void hgemv_downsweep(BasisTree_GPU &basis_tree, H2Opus_Real *Y, int ldy, int num_vectors, HgemvWorkspace &workspace,
                     h2opusComputeStream_t stream)
{
    hgemv_downsweep_template<H2OPUS_HWTYPE_GPU>(basis_tree, Y, ldy, num_vectors, workspace, stream);
}

void hgemv_denseMult(int hblas_trans_mode, H2Opus_Real alpha, HNodeTree_GPU &hnodes, H2Opus_Real *X, int ldx,
                     H2Opus_Real beta, H2Opus_Real *Y, int ldy, int num_vectors, int *node_u_start, int *node_v_start,
                     int *node_u_len, int *node_v_len, int *column_basis_indexes, int *row_basis_indexes,
                     typename THNodeTree<H2OPUS_HWTYPE_GPU>::HNodeTreeBSNData *bsn_data, HgemvWorkspace &workspace,
                     h2opusComputeStream_t stream)
{
    hgemv_denseMult_template<H2OPUS_HWTYPE_GPU>(hblas_trans_mode, alpha, hnodes, X, ldx, beta, Y, ldy, num_vectors,
                                                node_u_start, node_v_start, node_u_len, node_v_len,
                                                column_basis_indexes, row_basis_indexes, bsn_data, workspace, stream);
}

void hgemv_mult_level(HNodeTree_GPU &hnodes, int level, int num_vectors, size_t u_index_offset, size_t v_index_offset,
                      H2Opus_Real *xhat_level, H2Opus_Real *yhat_level, BatchGemmMarshalledData &marshal_data,
                      typename THNodeTree<H2OPUS_HWTYPE_GPU>::HNodeTreeBSNData *bsn_data, int *column_basis_indexes,
                      int *row_basis_indexes, int hblas_trans_mode, h2opusComputeStream_t stream)
{
    hgemv_mult_level_template<H2OPUS_HWTYPE_GPU>(hnodes, level, num_vectors, u_index_offset, v_index_offset, xhat_level,
                                                 yhat_level, marshal_data, bsn_data, column_basis_indexes,
                                                 row_basis_indexes, hblas_trans_mode, stream);
}

void hgemv_downsweep_leaves(BasisTree_GPU &basis_tree, H2Opus_Real *Y, int ldy, int num_vectors,
                            HgemvWorkspace &workspace, h2opusComputeStream_t stream)
{
    hgemv_downsweep_leaves_template<H2OPUS_HWTYPE_GPU>(basis_tree, Y, ldy, num_vectors, workspace, stream);
}

void hgemv_downsweep_level(BasisTree_GPU &basis_tree, int num_vectors, int level, HgemvWorkspace &workspace,
                           h2opusComputeStream_t stream)
{
    hgemv_downsweep_level_template<H2OPUS_HWTYPE_GPU>(basis_tree, num_vectors, level, workspace, stream);
}

void hgemv_upsweep_leaves(H2Opus_Real alpha, BasisTree_GPU &basis_tree, H2Opus_Real *X, int ldx, int num_vectors,
                          HgemvWorkspace &workspace, h2opusComputeStream_t stream)
{
    hgemv_upsweep_leaves_template<H2OPUS_HWTYPE_GPU>(alpha, basis_tree, X, ldx, num_vectors, workspace, stream);
}

void hgemv_upsweep_level(BasisTree_GPU &basis_tree, int level, int num_vectors, HgemvWorkspace &workspace,
                         h2opusComputeStream_t stream)
{
    hgemv_upsweep_level_template<H2OPUS_HWTYPE_GPU>(basis_tree, level, num_vectors, workspace, stream);
}

void hgemv_mult(int trans, HNodeTree_GPU &hnodes, int start_level, int end_level, int num_vectors,
                BasisTree_GPU &u_basis_tree, BasisTree_GPU &v_basis_tree, HgemvWorkspace &workspace,
                h2opusComputeStream_t stream)
{
    hgemv_mult_template<H2OPUS_HWTYPE_GPU>(trans, hnodes, start_level, end_level, num_vectors, u_basis_tree,
                                           v_basis_tree, workspace, stream);
}

void hgemv_denseMult(int trans, H2Opus_Real alpha, HNodeTree_GPU &hnodes, H2Opus_Real *X, int ldx, H2Opus_Real beta,
                     H2Opus_Real *Y, int ldy, int num_vectors, BasisTree_GPU &u_basis_tree, BasisTree_GPU &v_basis_tree,
                     HgemvWorkspace &workspace, h2opusComputeStream_t stream)
{
    hgemv_denseMult_template<H2OPUS_HWTYPE_GPU>(trans, alpha, hnodes, X, ldx, beta, Y, ldy, num_vectors, u_basis_tree,
                                                v_basis_tree, workspace, stream);
}

void hgemv(int trans, H2Opus_Real alpha, HMatrix_GPU &hmatrix, H2Opus_Real *X, int ldx, H2Opus_Real beta,
           H2Opus_Real *Y, int ldy, int num_vectors, h2opusHandle_t h2opus_handle)
{
    hgemv_template<H2OPUS_HWTYPE_GPU>(trans, alpha, hmatrix, X, ldx, beta, Y, ldy, num_vectors, h2opus_handle);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void hgemv_upsweep(H2Opus_Real alpha, BasisTree &basis_tree, H2Opus_Real *X, int ldx, int num_vectors,
                   HgemvWorkspace &workspace, h2opusComputeStream_t stream)
{
    hgemv_upsweep_template<H2OPUS_HWTYPE_CPU>(alpha, basis_tree, X, ldx, num_vectors, workspace, stream);
}

void hgemv_downsweep(BasisTree &basis_tree, H2Opus_Real *Y, int ldy, int num_vectors, HgemvWorkspace &workspace,
                     h2opusComputeStream_t stream)
{
    hgemv_downsweep_template<H2OPUS_HWTYPE_CPU>(basis_tree, Y, ldy, num_vectors, workspace, stream);
}

void hgemv_denseMult(int hblas_trans_mode, H2Opus_Real alpha, HNodeTree &hnodes, H2Opus_Real *X, int ldx,
                     H2Opus_Real beta, H2Opus_Real *Y, int ldy, int num_vectors, int *node_u_start, int *node_v_start,
                     int *node_u_len, int *node_v_len, int *column_basis_indexes, int *row_basis_indexes,
                     typename THNodeTree<H2OPUS_HWTYPE_CPU>::HNodeTreeBSNData *bsn_data, HgemvWorkspace &workspace,
                     h2opusComputeStream_t stream)
{
    hgemv_denseMult_template<H2OPUS_HWTYPE_CPU>(hblas_trans_mode, alpha, hnodes, X, ldx, beta, Y, ldy, num_vectors,
                                                node_u_start, node_v_start, node_u_len, node_v_len,
                                                column_basis_indexes, row_basis_indexes, bsn_data, workspace, stream);
}

void hgemv_mult_level(HNodeTree &hnodes, int level, int num_vectors, size_t u_index_offset, size_t v_index_offset,
                      H2Opus_Real *xhat_level, H2Opus_Real *yhat_level, BatchGemmMarshalledData &marshal_data,
                      typename THNodeTree<H2OPUS_HWTYPE_CPU>::HNodeTreeBSNData *bsn_data, int *column_basis_indexes,
                      int *row_basis_indexes, int hblas_trans_mode, h2opusComputeStream_t stream)
{
    hgemv_mult_level_template<H2OPUS_HWTYPE_CPU>(hnodes, level, num_vectors, u_index_offset, v_index_offset, xhat_level,
                                                 yhat_level, marshal_data, bsn_data, column_basis_indexes,
                                                 row_basis_indexes, hblas_trans_mode, stream);
}

void hgemv_downsweep_leaves(BasisTree &basis_tree, H2Opus_Real *Y, int ldy, int num_vectors, HgemvWorkspace &workspace,
                            h2opusComputeStream_t stream)
{
    hgemv_downsweep_leaves_template<H2OPUS_HWTYPE_CPU>(basis_tree, Y, ldy, num_vectors, workspace, stream);
}

void hgemv_downsweep_level(BasisTree &basis_tree, int num_vectors, int level, HgemvWorkspace &workspace,
                           h2opusComputeStream_t stream)
{
    hgemv_downsweep_level_template<H2OPUS_HWTYPE_CPU>(basis_tree, num_vectors, level, workspace, stream);
}

void hgemv_upsweep_leaves(H2Opus_Real alpha, BasisTree &basis_tree, H2Opus_Real *X, int ldx, int num_vectors,
                          HgemvWorkspace &workspace, h2opusComputeStream_t stream)
{
    hgemv_upsweep_leaves_template<H2OPUS_HWTYPE_CPU>(alpha, basis_tree, X, ldx, num_vectors, workspace, stream);
}

void hgemv_upsweep_level(BasisTree &basis_tree, int level, int num_vectors, HgemvWorkspace &workspace,
                         h2opusComputeStream_t stream)
{
    hgemv_upsweep_level_template<H2OPUS_HWTYPE_CPU>(basis_tree, level, num_vectors, workspace, stream);
}

void hgemv_mult(int trans, HNodeTree &hnodes, int start_level, int end_level, int num_vectors, BasisTree &u_basis_tree,
                BasisTree &v_basis_tree, HgemvWorkspace &workspace, h2opusComputeStream_t stream)
{
    hgemv_mult_template<H2OPUS_HWTYPE_CPU>(trans, hnodes, start_level, end_level, num_vectors, u_basis_tree,
                                           v_basis_tree, workspace, stream);
}

void hgemv_denseMult(int trans, H2Opus_Real alpha, HNodeTree &hnodes, H2Opus_Real *X, int ldx, H2Opus_Real beta,
                     H2Opus_Real *Y, int ldy, int num_vectors, BasisTree &u_basis_tree, BasisTree &v_basis_tree,
                     HgemvWorkspace &workspace, h2opusComputeStream_t stream)
{
    hgemv_denseMult_template<H2OPUS_HWTYPE_CPU>(trans, alpha, hnodes, X, ldx, beta, Y, ldy, num_vectors, u_basis_tree,
                                                v_basis_tree, workspace, stream);
}

void hgemv(int trans, H2Opus_Real alpha, HMatrix &hmatrix, H2Opus_Real *X, int ldx, H2Opus_Real beta, H2Opus_Real *Y,
           int ldy, int num_vectors, h2opusHandle_t h2opus_handle)
{
    hgemv_template<H2OPUS_HWTYPE_CPU>(trans, alpha, hmatrix, X, ldx, beta, Y, ldy, num_vectors, h2opus_handle);
}
