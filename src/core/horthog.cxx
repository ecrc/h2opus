#include <h2opus/core/horthog.h>
#include <h2opus/marshal/horthog_marshal.h>

#include <h2opus/util/perf_counter.h>
#include <h2opus/util/timer.h>

#include <h2opus/util/batch_wrappers.h>
#include <h2opus/util/debug_routines.h>
#include <h2opus/util/gpu_err_check.h>
#include <h2opus/util/thrust_wrappers.h>

template <int hw>
void horthog_project_level_template(THNodeTree<hw> &hnodes, int level, size_t u_level_start, size_t v_level_start,
                                    H2Opus_Real *Tu_level, H2Opus_Real *Tv_level, int *node_u_index, int *node_v_index,
                                    size_t increment, HorthogWorkspace &workspace, h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

    std::vector<int> &new_ranks = workspace.u_new_ranks;
    HNodeTreeLevelData &hnode_level_data = hnodes.level_data;

    H2Opus_Real *TS = workspace.TS, *realloc_buffer = workspace.realloc_buffer;
    H2Opus_Real **TS_array = workspace.TS_array, **Tu_array = workspace.Tu_array, **Tv_array = workspace.Tv_array;
    H2Opus_Real **S_array = workspace.S_array, **S_new_array = workspace.S_new_array;

    H2Opus_Real alpha = 1, beta = 0;

    size_t level_nodes = hnode_level_data.getCouplingLevelSize(level);
    int level_rank = hnode_level_data.getLevelRank(level);
    int level_new_rank = new_ranks[level];

    if (level_nodes == 0)
        return;

    H2Opus_Real *S_level = vec_ptr(hnodes.rank_leaf_mem[level]);
    H2Opus_Real *S_new_level = S_level;

    H2Opus_Real **S_new_array_ptr = S_array;

    // Check if we have to reallocte the level coupling node memory
    if (level_rank != level_new_rank)
    {
        copyArray(S_level, realloc_buffer, hnodes.rank_leaf_mem[level].size(), stream, hw);
        S_level = realloc_buffer;

        hnodes.rank_leaf_mem[level] = RealVector(level_nodes * level_new_rank * level_new_rank, 0);
        S_new_level = vec_ptr(hnodes.rank_leaf_mem[level]);

        generateArrayOfPointers(S_new_level, S_new_array, level_new_rank * level_new_rank, level_nodes, stream, hw);
        S_new_array_ptr = S_new_array;
    }

    // Generate an array of pointers so that we can use the cublas batch gemm routines
    generateArrayOfPointers(TS, TS_array, level_new_rank * level_rank, increment, stream, hw);
    generateArrayOfPointers(S_level, S_array, level_rank * level_rank, level_nodes, stream, hw);

    size_t coupling_start = hnode_level_data.getCouplingLevelStart(level);

    horthog_project_batch_marshal<H2Opus_Real, hw>(vec_ptr(hnodes.rank_leaf_tree_index), node_u_index, Tu_level,
                                                   Tu_array, level_new_rank * level_rank, coupling_start, u_level_start,
                                                   level_nodes, stream);

    horthog_project_batch_marshal<H2Opus_Real, hw>(vec_ptr(hnodes.rank_leaf_tree_index), node_v_index, Tv_level,
                                                   Tv_array, level_new_rank * level_rank, coupling_start, v_level_start,
                                                   level_nodes, stream);

    size_t start_index = 0;
    while (start_index != level_nodes)
    {
        size_t batch_size = std::min(increment, level_nodes - start_index);

        // First calculate TS_{ts} = Tu_{t} S_{ts}
        check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
            stream, H2Opus_NoTrans, H2Opus_NoTrans, level_new_rank, level_rank, level_rank, alpha,
            (const H2Opus_Real **)Tu_array + start_index, level_new_rank, (const H2Opus_Real **)S_array + start_index,
            level_rank, beta, TS_array, level_new_rank, batch_size));

        // Now calculate S_{ts} = TS_{ts} * Pv_{t}^t
        check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
            stream, H2Opus_NoTrans, H2Opus_Trans, level_new_rank, level_new_rank, level_rank, alpha,
            (const H2Opus_Real **)TS_array, level_new_rank, (const H2Opus_Real **)Tv_array + start_index,
            level_new_rank, beta, S_new_array_ptr + start_index, level_new_rank, batch_size));

        start_index += batch_size;
    }
}

template <int hw>
void horthog_project_template(THNodeTree<hw> &hnodes, int start_level, int end_level, BasisTreeLevelData &u_level_data,
                              BasisTreeLevelData &v_level_data, HorthogWorkspace &workspace,
                              h2opusComputeStream_t stream)
{
    assert(u_level_data.nested_root_level == v_level_data.nested_root_level);

    HNodeTreeLevelData &hnode_level_data = hnodes.level_data;

    std::vector<H2Opus_Real *> &Tu_hat = workspace.Tu_hat;
    std::vector<H2Opus_Real *> &Tv_hat = (workspace.symmetric ? workspace.Tu_hat : workspace.Tv_hat);

    size_t max_nodes = hnode_level_data.getMaxLevelCouplingNodes();
    size_t increment = std::min((size_t)PROJECTION_MAX_NODES, max_nodes);

    // Now go through the levels of the tree and compute the projection
    // of the coupling matrices into the new basis
    for (int level = end_level; level >= start_level; level--)
    {
        size_t u_level_start = u_level_data.getLevelStart(level);
        size_t v_level_start = v_level_data.getLevelStart(level);
        H2Opus_Real *Tu_level = Tu_hat[level];
        H2Opus_Real *Tv_level = Tv_hat[level];

        horthog_project_level_template<hw>(hnodes, level, u_level_start, v_level_start, Tu_level, Tv_level,
                                           vec_ptr(hnodes.node_u_index), vec_ptr(hnodes.node_v_index), increment,
                                           workspace, stream);
    }
}

template <int hw>
void horthog_upsweep_level_template(TBasisTree<hw> &basis_tree, HorthogWorkspace &workspace,
                                    std::vector<H2Opus_Real *> &T_hat, std::vector<int> &new_ranks, int level,
                                    h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    BasisTreeLevelData &level_data = basis_tree.level_data;

    // Temporary memory for the products of the projection and transfer matrices
    int max_children = basis_tree.max_children;
    H2Opus_Real *TE_data = workspace.TE_data, *tau = workspace.TE_tau, *realloc_buffer = workspace.realloc_buffer;
    H2Opus_Real **ptr_TE = workspace.ptr_TE, **ptr_T = workspace.ptr_T, **ptr_E = workspace.ptr_E;

    int child_rank = level_data.getLevelRank(level + 1);
    int level_rank = level_data.getLevelRank(level);
    int child_new_rank = new_ranks[level + 1];

    size_t child_level_start = level_data.getLevelStart(level + 1);
    size_t level_start = level_data.getLevelStart(level);
    size_t num_children = level_data.getLevelSize(level + 1);
    size_t num_nodes = level_data.getLevelSize(level);

    if (child_rank == 0 || level_rank == 0)
        return;

    int te_rows = max_children * child_new_rank;
    int level_new_rank = level_rank;

    H2Opus_Real *child_transfer_data = vec_ptr(basis_tree.trans_mem[level + 1]);
    H2Opus_Real *T_hat_child_level = T_hat[level + 1];
    H2Opus_Real *T_hat_level = T_hat[level];

    // See if the rank of this level needs to be reduced
    if (child_new_rank != child_rank || te_rows < level_rank)
    {
        copyArray(child_transfer_data, realloc_buffer, basis_tree.trans_mem[level + 1].size(), stream, hw);
        child_transfer_data = realloc_buffer;
        level_new_rank = std::min(te_rows, level_rank);

        assert(level_new_rank == new_ranks[level]);

        basis_tree.trans_mem[level + 1] = RealVector(num_children * child_new_rank * level_new_rank, 0);
    }

    ////////////////////////////////////////////////////////////////
    // Form TE = [T_c1 E_c1; T_c2 E_c2]
    ////////////////////////////////////////////////////////////////
    // Marshal upsweep pointers
    horthog_upsweep_batch_marshal<H2Opus_Real, hw>(
        T_hat_child_level, child_transfer_data, TE_data, ptr_T, ptr_E, ptr_TE, child_new_rank, child_rank, level_rank,
        child_level_start, level_start, max_children, basis_tree.head_ptr(), basis_tree.next_ptr(), num_nodes, stream);

    // Now execute the batch gemm
    H2Opus_Real alpha = 1, beta = 0;

    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, child_new_rank,
                                                             level_rank, child_rank, alpha, (const H2Opus_Real **)ptr_T,
                                                             child_new_rank, (const H2Opus_Real **)ptr_E, child_rank,
                                                             beta, ptr_TE, te_rows, num_children));

    ////////////////////////////////////////////////////////////////
    // Orthogonalize TE using QR, save the R matrices in the next level of T matrices,
    // unpack Q into TE then copy the submatrices back into the transfer matrices
    ////////////////////////////////////////////////////////////////
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::geqrf)(stream, te_rows, level_rank, TE_data, te_rows,
                                                              te_rows * level_rank, tau, level_rank, num_nodes));

    // Save the R factors so that we can unpack the full Q
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copy_upper)(stream, te_rows, level_rank, TE_data, te_rows,
                                                                   te_rows * level_rank, T_hat_level, level_new_rank,
                                                                   level_new_rank * level_rank, num_nodes));

    // Unpack the househoulder vectors
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::orgqr)(stream, te_rows, level_rank, TE_data, te_rows,
                                                              te_rows * level_rank, tau, level_rank, num_nodes));

    // Now copy the blocks of the Q factor into the transfer matrices
    // First marshal the pointers...
    H2Opus_Real *new_child_transfer = vec_ptr(basis_tree.trans_mem[level + 1]);
    horthog_copyBlock_marshal_batch<H2Opus_Real, hw>(
        new_child_transfer, TE_data, ptr_E, ptr_TE, child_new_rank, level_rank, level_new_rank, child_level_start,
        level_start, max_children, basis_tree.head_ptr(), basis_tree.next_ptr(), num_nodes, stream);
    // ...and then copy
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copyBlock)(stream, child_new_rank, level_new_rank, ptr_E, 0, 0,
                                                                  child_new_rank, ptr_TE, 0, 0, te_rows, num_children));
}

template <int hw>
void horthog_upsweep_template(TBasisTree<hw> &basis_tree, HorthogWorkspace &workspace,
                              std::vector<H2Opus_Real *> &T_hat, std::vector<int> &new_ranks,
                              h2opusComputeStream_t stream)
{
    // Sweep up the tree
    int num_levels = basis_tree.depth;
    int top_level = basis_tree.level_data.nested_root_level;

    for (int level = num_levels - 2; level >= top_level; level--)
    {
        horthog_upsweep_level_template<hw>(basis_tree, workspace, T_hat, new_ranks, level, stream);
    }
}

template <int hw>
void horthog_upsweep_leaves_template(TBasisTree<hw> &basis_tree, HorthogWorkspace &workspace,
                                     std::vector<H2Opus_Real *> &T_hat, h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

    BasisTreeLevelData &level_data = basis_tree.level_data;

    size_t num_leaves = level_data.basis_leaves;
    int leaf_rows = level_data.leaf_size;
    int leaf_rank = level_data.getLevelRank(basis_tree.depth - 1);
    int leaf_level = level_data.depth - 1;

    if (leaf_rank == 0)
        return;

    H2Opus_Real *basis_leaves = vec_ptr(basis_tree.basis_mem);
    H2Opus_Real *realloc_buffer = workspace.realloc_buffer;
    H2Opus_Real *tau = workspace.TE_tau;

    // Check if we need to reduce the rank of the leaves
    int leaf_new_rank = leaf_rank;
    if (leaf_rank > leaf_rows)
    {
        copyArray(basis_leaves, realloc_buffer, basis_tree.basis_mem.size(), stream, hw);
        basis_leaves = realloc_buffer;
        leaf_new_rank = leaf_rows;
        basis_tree.basis_mem = RealVector(num_leaves * leaf_rows * leaf_rows, 0);
    }

    // QR for the leaves - leaves are overwritten with the househoulder vectors
    // and the upper triangular R
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::geqrf)(stream, leaf_rows, leaf_rank, basis_leaves, leaf_rows,
                                                              leaf_rows * leaf_rank, tau, leaf_rank, num_leaves));

    // Save the R factors from the leaves so that we can unpack the full Q
    H2Opus_Real *T_hat_leaf = T_hat[leaf_level];

    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copy_upper)(
        stream, leaf_rows, leaf_rank, basis_leaves, leaf_rows, leaf_rows * leaf_rank, T_hat_leaf, leaf_new_rank,
        leaf_new_rank * leaf_rank, num_leaves));
    // Now unpack the Q factor from the stored househoulder vectors
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::orgqr)(stream, leaf_rows, leaf_rank, basis_leaves, leaf_rows,
                                                              leaf_rows * leaf_rank, tau, leaf_rank, num_leaves));

    // Copy over the truncated leaves if necessary
    if (leaf_rank > leaf_rows)
    {
        H2Opus_Real *truncated_leaves = vec_ptr(basis_tree.basis_mem);

        check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::copyBlock)(
            stream, leaf_rows, leaf_new_rank, truncated_leaves, 0, 0, leaf_rows, leaf_rows * leaf_new_rank,
            basis_leaves, 0, 0, leaf_rows, leaf_rows * leaf_rank, num_leaves));
    }
}

template <int hw>
void horthog_stitch_template(TBasisTree<hw> &basis_tree, HorthogWorkspace &workspace, std::vector<H2Opus_Real *> &T_hat,
                             std::vector<int> &new_ranks, h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

    BasisTreeLevelData &level_data = basis_tree.level_data;

    int top_level = level_data.nested_root_level;

    if (top_level == 0)
        return;

    size_t num_nodes = level_data.getLevelSize(top_level);
    int level_rank = level_data.getLevelRank(top_level);
    int level_new_rank = new_ranks[top_level];
    int parent_rank = level_data.getLevelRank(top_level - 1);

    H2Opus_Real *T_hat_level = T_hat[top_level];

    // Copy the old level
    H2Opus_Real *old_level = workspace.realloc_buffer;
    copyArray(vec_ptr(basis_tree.trans_mem[top_level]), old_level, basis_tree.trans_mem[top_level].size(), stream, hw);

    // Check if we need to resize the projected level
    if (level_new_rank < level_rank)
        basis_tree.trans_mem[top_level] = RealVector(num_nodes * level_new_rank * parent_rank, 0);

    H2Opus_Real *projected_level = vec_ptr(basis_tree.trans_mem[top_level]);

    H2Opus_Real **ptr_T = workspace.ptr_T, **ptr_E = workspace.ptr_E, **ptr_TE = workspace.ptr_TE;
    generateArrayOfPointers(T_hat_level, ptr_T, level_new_rank * level_rank, num_nodes, stream, hw);
    generateArrayOfPointers(old_level, ptr_E, level_rank * parent_rank, num_nodes, stream, hw);
    generateArrayOfPointers(projected_level, ptr_TE, level_new_rank * parent_rank, num_nodes, stream, hw);

    H2Opus_Real alpha = 1, beta = 0;

    // E = T * E
    check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
        stream, H2Opus_NoTrans, H2Opus_NoTrans, level_new_rank, parent_rank, level_rank, alpha,
        (const H2Opus_Real **)ptr_T, level_new_rank, (const H2Opus_Real **)ptr_E, level_rank, beta, ptr_TE,
        level_new_rank, num_nodes));

    // Copy over the ranks above the current top level
    for (int i = top_level - 1; i >= 0; i--)
        new_ranks[i] = level_data.getLevelRank(i);
}

template <int hw> void horthog_template(THMatrix<hw> &hmatrix, h2opusHandle_t h2opus_handle)
{
    H2OpusWorkspaceState ws_needed = horthog_workspace(hmatrix);
    H2OpusWorkspaceState ws_allocated = h2opus_handle->getWorkspaceState();

    if (ws_allocated < ws_needed)
    {
        // printf("Insufficient workspace for horthog...\n");
        h2opus_handle->setWorkspaceState(ws_needed);
    }

    h2opusComputeStream_t main_stream = h2opus_handle->getMainStream();
    HorthogWorkspace workspace;
    horthog_get_workspace(hmatrix, workspace, h2opus_handle);

    TBasisTree<hw> &u_basis_tree = hmatrix.u_basis_tree;
    TBasisTree<hw> &v_basis_tree = (hmatrix.sym ? u_basis_tree : hmatrix.v_basis_tree);

    BasisTreeLevelData &u_level_data = u_basis_tree.level_data;
    BasisTreeLevelData &v_level_data = v_basis_tree.level_data;
    HNodeTreeLevelData &hnode_level_data = hmatrix.hnodes.level_data;

    ////////////////////////////////////////////////////
    // Orthogonalize leaves
    ////////////////////////////////////////////////////
#ifdef H2OPUS_PROFILING_ENABLED
    Timer<hw> timer;
    timer.init();
    PerformanceCounter::clearCounters();

    timer.start();
#endif
    horthog_upsweep_leaves_template<hw>(hmatrix.u_basis_tree, workspace, workspace.Tu_hat, main_stream);
    if (!hmatrix.sym)
        horthog_upsweep_leaves_template<hw>(hmatrix.v_basis_tree, workspace, workspace.Tv_hat, main_stream);
#ifdef H2OPUS_PROFILING_ENABLED
    double leaf_timer = timer.stop();
    double leaf_gops = PerformanceCounter::getOpCount(PerformanceCounter::QR);
    HLibProfile::addRun(HLibProfile::HORTHOG_BASIS_LEAVES, leaf_gops, leaf_timer);
    PerformanceCounter::clearCounters();
#endif
    ////////////////////////////////////////////////////

    ////////////////////////////////////////////////////
    // Sweep up the tree orthogonalizing inner nodes
    ////////////////////////////////////////////////////
#ifdef H2OPUS_PROFILING_ENABLED
    timer.start();
#endif
    horthog_upsweep_template<hw>(hmatrix.u_basis_tree, workspace, workspace.Tu_hat, workspace.u_new_ranks, main_stream);
    if (!hmatrix.sym)
        horthog_upsweep_template<hw>(hmatrix.v_basis_tree, workspace, workspace.Tv_hat, workspace.v_new_ranks,
                                     main_stream);
#ifdef H2OPUS_PROFILING_ENABLED
    double upsweep_timer = timer.stop();
    double upsweep_gops = PerformanceCounter::getOpCount(PerformanceCounter::GEMM) +
                          PerformanceCounter::getOpCount(PerformanceCounter::QR);
    HLibProfile::addRun(HLibProfile::HORTHOG_UPSWEEP, upsweep_gops, upsweep_timer);
    PerformanceCounter::clearCounters();
#endif
    ////////////////////////////////////////////////////

    ////////////////////////////////////////////////////
    // Project the top level transfer nodes if necessary
    ////////////////////////////////////////////////////
#ifdef H2OPUS_PROFILING_ENABLED
    timer.start();
#endif
    horthog_stitch_template<hw>(hmatrix.u_basis_tree, workspace, workspace.Tu_hat, workspace.u_new_ranks, main_stream);
    if (!hmatrix.sym)
        horthog_stitch_template<hw>(hmatrix.v_basis_tree, workspace, workspace.Tv_hat, workspace.v_new_ranks,
                                    main_stream);

#ifdef H2OPUS_PROFILING_ENABLED
    double stitch_timer = timer.stop();
    double stitch_gops = PerformanceCounter::getOpCount(PerformanceCounter::GEMM);
    HLibProfile::addRun(HLibProfile::HORTHOG_STITCH, stitch_gops, stitch_timer);
    PerformanceCounter::clearCounters();
#endif
    ////////////////////////////////////////////////////

    ////////////////////////////////////////////////////
    // Project the coupling nodes into the new orthogonal basis
    ////////////////////////////////////////////////////
#ifdef H2OPUS_PROFILING_ENABLED
    timer.start();
#endif

    horthog_project_template<hw>(hmatrix.hnodes, u_level_data.nested_root_level, u_level_data.depth - 1, u_level_data,
                                 v_level_data, workspace, main_stream);

#ifdef H2OPUS_PROFILING_ENABLED
    double proj_timer = timer.stop();
    double proj_gops = PerformanceCounter::getOpCount(PerformanceCounter::GEMM);
    HLibProfile::addRun(HLibProfile::HORTHOG_PROJECTION, proj_gops, proj_timer);
    PerformanceCounter::clearCounters();
#endif
    ////////////////////////////////////////////////////

    // dumpMatrixTreeContainer(u_level_data, workspace.Tu_hat, 5, hw);

    ////////////////////////////////////////////////////
    // Update the rank meta data

    // printf("Level nodes\tOld_rank\tNew_rank\n");
    // for(int i = u_level_data.depth - 1; i >= 0; i--)
    // 	printf("%d\t\t%d\t\t%d\n", hnode_level_data.getCouplingLevelSize(i), hnode_level_data.getLevelRank(i),
    // workspace.u_new_ranks[i]); printf("\n");

    u_level_data.setLevelRanks(vec_ptr(workspace.u_new_ranks));
    hnode_level_data.setRankFromBasis(u_level_data, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Interface routines
////////////////////////////////////////////////////////////////////////////////////////////////////////
void horthog(HMatrix &hmatrix, h2opusHandle_t h2opus_handle)
{
    horthog_template<H2OPUS_HWTYPE_CPU>(hmatrix, h2opus_handle);
}

void horthog_upsweep_level(BasisTree &basis_tree, HorthogWorkspace &workspace, std::vector<H2Opus_Real *> &T_hat,
                           std::vector<int> &new_ranks, int level, h2opusComputeStream_t stream)
{
    horthog_upsweep_level_template<H2OPUS_HWTYPE_CPU>(basis_tree, workspace, T_hat, new_ranks, level, stream);
}

void horthog_upsweep_leaves(BasisTree &basis_tree, HorthogWorkspace &workspace, std::vector<H2Opus_Real *> &T_hat,
                            h2opusComputeStream_t stream)
{
    horthog_upsweep_leaves_template<H2OPUS_HWTYPE_CPU>(basis_tree, workspace, T_hat, stream);
}

void horthog_upsweep(BasisTree &basis_tree, HorthogWorkspace &workspace, std::vector<H2Opus_Real *> &T_hat,
                     std::vector<int> &new_ranks, h2opusComputeStream_t stream)
{
    horthog_upsweep_template<H2OPUS_HWTYPE_CPU>(basis_tree, workspace, T_hat, new_ranks, stream);
}

void horthog_project(HNodeTree &hnodes, int start_level, int end_level, BasisTreeLevelData &u_level_data,
                     BasisTreeLevelData &v_level_data, HorthogWorkspace &workspace, h2opusComputeStream_t stream)
{
    horthog_project_template<H2OPUS_HWTYPE_CPU>(hnodes, start_level, end_level, u_level_data, v_level_data, workspace,
                                                stream);
}

void horthog_project_level(HNodeTree &hnodes, int level, size_t u_level_start, size_t v_level_start,
                           H2Opus_Real *Tu_level, H2Opus_Real *Tv_level, int *node_u_index, int *node_v_index,
                           size_t increment, HorthogWorkspace &workspace, h2opusComputeStream_t stream)
{
    horthog_project_level_template<H2OPUS_HWTYPE_CPU>(hnodes, level, u_level_start, v_level_start, Tu_level, Tv_level,
                                                      node_u_index, node_v_index, increment, workspace, stream);
}

#ifdef H2OPUS_USE_GPU
void horthog(HMatrix_GPU &hmatrix, h2opusHandle_t h2opus_handle)
{
    horthog_template<H2OPUS_HWTYPE_GPU>(hmatrix, h2opus_handle);
}

void horthog_upsweep_level(BasisTree_GPU &basis_tree, HorthogWorkspace &workspace, std::vector<H2Opus_Real *> &T_hat,
                           std::vector<int> &new_ranks, int level, h2opusComputeStream_t stream)
{
    horthog_upsweep_level_template<H2OPUS_HWTYPE_GPU>(basis_tree, workspace, T_hat, new_ranks, level, stream);
}

void horthog_upsweep_leaves(BasisTree_GPU &basis_tree, HorthogWorkspace &workspace, std::vector<H2Opus_Real *> &T_hat,
                            h2opusComputeStream_t stream)
{
    horthog_upsweep_leaves_template<H2OPUS_HWTYPE_GPU>(basis_tree, workspace, T_hat, stream);
}

void horthog_upsweep(BasisTree_GPU &basis_tree, HorthogWorkspace &workspace, std::vector<H2Opus_Real *> &T_hat,
                     std::vector<int> &new_ranks, h2opusComputeStream_t stream)
{
    horthog_upsweep_template<H2OPUS_HWTYPE_GPU>(basis_tree, workspace, T_hat, new_ranks, stream);
}

void horthog_project(HNodeTree_GPU &hnodes, int start_level, int end_level, BasisTreeLevelData &u_level_data,
                     BasisTreeLevelData &v_level_data, HorthogWorkspace &workspace, h2opusComputeStream_t stream)
{
    horthog_project_template<H2OPUS_HWTYPE_GPU>(hnodes, start_level, end_level, u_level_data, v_level_data, workspace,
                                                stream);
}

void horthog_project_level(HNodeTree_GPU &hnodes, int level, size_t u_level_start, size_t v_level_start,
                           H2Opus_Real *Tu_level, H2Opus_Real *Tv_level, int *node_u_index, int *node_v_index,
                           size_t increment, HorthogWorkspace &workspace, h2opusComputeStream_t stream)
{
    horthog_project_level_template<H2OPUS_HWTYPE_GPU>(hnodes, level, u_level_start, v_level_start, Tu_level, Tv_level,
                                                      node_u_index, node_v_index, increment, workspace, stream);
}

#endif
