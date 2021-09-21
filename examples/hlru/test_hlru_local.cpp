#include <stdio.h>
#include <h2opus.h>

#include "../common/example_problem.h"
#include "../common/example_util.h"
#include "../common/hmatrix_samplers.h"

#include <h2opus/util/boxentrygen.h>

// This example applies the local low rank updates to a full dense matrix
void applyDenseLRU(HMatrix &hmatrix, H2Opus_Real *M, int n, LowRankUpdate &low_rank_update)
{
    assert(hmatrix.sym);

    int num_updates = low_rank_update.hnode_indexes.size();
    int rank = low_rank_update.total_rank;

    double alpha = 1, beta = 0;

    BasisTree &u_basis_tree = hmatrix.u_basis_tree;
    BasisTree &v_basis_tree = hmatrix.u_basis_tree;

    for (int update_id = 0; update_id < num_updates; update_id++)
    {
        int hnode_index = low_rank_update.hnode_indexes[update_id];
        int u_index = hmatrix.hnodes.node_u_index[hnode_index];
        int v_index = hmatrix.hnodes.node_v_index[hnode_index];
        int u_start = u_basis_tree.node_start[u_index];
        int v_start = v_basis_tree.node_start[v_index];
        int rows = u_basis_tree.node_len[u_index];
        int cols = v_basis_tree.node_len[v_index];

        H2Opus_Real *M_block = M + u_start + v_start * n;
        H2Opus_Real *U = low_rank_update.U[update_id];
        H2Opus_Real *V = low_rank_update.V[update_id];

        h2opus_fbl_gemm(H2OpusFBLNoTrans, H2OpusFBLTrans, rows, cols, rank, alpha, U, low_rank_update.ldu, V,
                        low_rank_update.ldv, beta, M_block, n);
    }
}

void generateLRU(HMatrix &hmatrix, H2Opus_Real *U, H2Opus_Real *V, int n, int level, int rank,
                 LowRankUpdate &low_rank_update)
{
    assert(hmatrix.sym);

    int level_nodes = 1 << level;
    low_rank_update.init(n, level, rank, level_nodes);
    low_rank_update.setRank(rank);

    // Half since it's a symmetric matrix
    level_nodes /= 2;

    randomData(U, n * rank, rand());
    randomData(V, n * rank, rand());

    int i = 0, j = 1;

    BasisTree &u_basis_tree = hmatrix.u_basis_tree;
    BasisTree &v_basis_tree = hmatrix.u_basis_tree;

    for (int node = 0; node < level_nodes; node++)
    {
        int hnode_index = hmatrix.hnodes.getHNodeIndex(level, i, j);
        int hnode_index_sym = hmatrix.hnodes.getHNodeIndex(level, j, i);

        int u_index = hmatrix.hnodes.node_u_index[hnode_index];
        int v_index = hmatrix.hnodes.node_v_index[hnode_index];
        int u_start = u_basis_tree.node_start[u_index];
        int v_start = v_basis_tree.node_start[v_index];

        low_rank_update.U[node] = U + u_start;
        low_rank_update.V[node] = V + v_start;
        low_rank_update.hnode_indexes[node] = hnode_index;

        low_rank_update.U[node + level_nodes] = low_rank_update.V[node];
        low_rank_update.V[node + level_nodes] = low_rank_update.U[node];
        low_rank_update.hnode_indexes[node + level_nodes] = hnode_index_sym;

        i += 2;
        j += 2;
    }
}

#ifdef H2OPUS_USE_GPU
void copyLRU(LowRankUpdate &lru, LowRankUpdate_GPU &gpu_lru, H2Opus_Real *U, H2Opus_Real *V, H2Opus_Real *d_U,
             H2Opus_Real *d_V, int n, int rank)
{
    gpu_lru.init(n, lru.level, lru.max_rank, lru.num_updates);
    gpu_lru.setRank(lru.total_rank);
    gpu_lru.hnode_indexes = lru.hnode_indexes;

    thrust::host_vector<H2Opus_Real *> host_U(lru.num_updates);
    thrust::host_vector<H2Opus_Real *> host_V(lru.num_updates);

    int num_updates = lru.num_updates / 2;
    for (int i = 0; i < num_updates; i++)
    {
        int u_diff = lru.U[i] - U;
        int v_diff = lru.V[i] - V;

        host_U[i] = d_U + u_diff;
        host_V[i] = d_V + v_diff;

        host_U[i + num_updates] = host_V[i];
        host_V[i + num_updates] = host_U[i];
    }

    gpu_lru.U = host_U;
    gpu_lru.V = host_V;
}
#endif

int main(int argc, char **argv)
{
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Argument parsing
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    H2OpusArgParser arg_parser;
    arg_parser.setArgs(argc, argv);

    int grid_x = arg_parser.option<int>("gx", "grid_x", "Grid points in the X direction", 32);
    int grid_y = arg_parser.option<int>("gy", "grid_y", "Grid points in the Y direction", 32);
    int grid_z = arg_parser.option<int>("gz", "grid_z", "Grid points in the Z direction", 1);
    int leaf_size = arg_parser.option<int>("m", "leaf_size", "Leaf size in the KD-tree", 64);
    int rank = arg_parser.option<int>("r", "rank", "Number of columns in the low rank update", 64);
    int level = arg_parser.option<int>("l", "level", "Level of the matrix tree that the update is applied to", 1);
    H2Opus_Real eta = arg_parser.option<H2Opus_Real>("e", "eta", "Admissibility parameter eta", DEFAULT_ETA);
    bool check_lru_err = arg_parser.flag("c", "check_lru_err", "Check the low rank update error", true);

    bool output_eps = arg_parser.flag("o", "output_eps", "Output structure of the matrix as an eps file", false);
    bool print_help = arg_parser.flag("h", "help", "This message", false);

    if (!arg_parser.valid() || print_help)
    {
        arg_parser.printUsage();
        return 0;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Geometry generation
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    size_t n = grid_x * grid_y * grid_z;
    printf("N = %zu\n", n);

    if (n >= 32768)
        printf("Warning: This example generates the full dense matrix of size %zu x %zu\n", n, n);

    // Create point cloud
    int dim = (grid_z == 1 ? 2 : 3);
    PointCloud<H2Opus_Real> pt_cloud(dim, n);
    if (grid_z > 1)
        generate3DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, grid_z, 0, 1, 0, 1, 0, 1);
    else
        generate2DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, 0, 1, 0, 1);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Matrix construction
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Setup hmatrix construction parameters:
    // Create a functor that can generate the matrix entries from two points
    FunctionGen<H2Opus_Real> func_gen(dim);
    // Create an entry gen struct from the functor. Currently only supports chebyshev interpolation on the CPU
    BoxEntryGen<H2Opus_Real, H2OPUS_HWTYPE_CPU, FunctionGen<H2Opus_Real>> entry_gen(func_gen);

    // Create the admissibility condition using the eta parameter
    // Decreasing eta refines the matrix tree and increasing it coarsens the tree
    H2OpusBoxCenterAdmissibility admissibility(eta);

    // Build the hmatrix structure. Currently only symmetric matrices are fully supported
    HMatrix hmatrix(n, true);
    buildHMatrixStructure(hmatrix, &pt_cloud, leaf_size, admissibility);
    HMatrix zero_hmatrix = hmatrix;

    if (level <= 0 || level >= hmatrix.u_basis_tree.depth)
    {
        printf("Invalid level %d\nValid range: [%d, %d]", level, 1, hmatrix.u_basis_tree.depth - 1);
        return 0;
    }

    if (output_eps)
        outputEps(hmatrix, "structure.eps");

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Local HLRU
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create h2opus handle
    h2opusHandle_t h2opus_handle;
    h2opusCreateHandle(&h2opus_handle);

    thrust::host_vector<H2Opus_Real> M(n * n), U(n * rank), V(n * rank);
    LowRankUpdate low_rank_update;

    generateLRU(hmatrix, vec_ptr(U), vec_ptr(V), n, level, rank, low_rank_update);
    applyDenseLRU(hmatrix, vec_ptr(M), n, low_rank_update);

    // Apply the low rank updates
    int rank_per_update = 32;
    low_rank_update.setRankPerUpdate(rank_per_update);

#ifdef H2OPUS_USE_GPU
    LowRankUpdate_GPU gpu_low_rank_update;
    thrust::device_vector<H2Opus_Real> d_M = M, d_U = U, d_V = V;
    copyLRU(low_rank_update, gpu_low_rank_update, vec_ptr(U), vec_ptr(V), vec_ptr(d_U), vec_ptr(d_V), n, rank);
    gpu_low_rank_update.setRankPerUpdate(rank_per_update);
#endif

    int done = H2OPUS_LRU_NOT_DONE;
    while (done != H2OPUS_LRU_DONE)
        done = hlru_sym(hmatrix, low_rank_update, h2opus_handle);

    // Make sure the low rank update was applied properly
    DenseSampler<H2OPUS_HWTYPE_CPU> sampler(vec_ptr(M), n, n, 0, NULL, h2opus_handle);
    H2Opus_Real lru_norm = sampler_norm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(&sampler, n, 40, h2opus_handle);

    // Check the difference between the low rank update and the updated hmatrix
    if (check_lru_err)
    {
        H2Opus_Real lru_err = sampler_difference<H2Opus_Real, H2OPUS_HWTYPE_CPU>(&sampler, hmatrix, 40, h2opus_handle);
        printf("CPU local low rank update difference = %e\n", lru_err / lru_norm);
    }

#ifdef H2OPUS_USE_GPU
    // Copy the hmatrix over to the GPU
    HMatrix_GPU gpu_h = zero_hmatrix;

    done = H2OPUS_LRU_NOT_DONE;
    while (done != H2OPUS_LRU_DONE)
        done = hlru_sym(gpu_h, gpu_low_rank_update, h2opus_handle);

    // Make sure the low rank update was applied properly
    DenseSampler<H2OPUS_HWTYPE_GPU> sampler_gpu(vec_ptr(d_M), n, n, 0, NULL, h2opus_handle);

    // Check the difference between the low rank update and the updated hmatrix
    if (check_lru_err)
    {
        H2Opus_Real lru_err =
            sampler_difference<H2Opus_Real, H2OPUS_HWTYPE_GPU>(&sampler_gpu, gpu_h, 40, h2opus_handle);
        printf("GPU local low rank update difference = %e\n", lru_err / lru_norm);
    }
#endif

    // Clean up
    h2opusDestroyHandle(h2opus_handle);

    return 0;
}
