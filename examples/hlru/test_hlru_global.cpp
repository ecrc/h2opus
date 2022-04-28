#include <stdio.h>
#include <h2opus.h>

#include "../common/example_problem.h"
#include "../common/example_util.h"
#include "../common/hmatrix_samplers.h"

#include <h2opus/util/boxentrygen.h>

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
    H2Opus_Real eta = arg_parser.option<H2Opus_Real>("e", "eta", "Admissibility parameter eta", DEFAULT_ETA);
    bool matunsym = arg_parser.flag("matunsym", "matunsym", "Unsymmetric structure of initial matrix", false);
    bool lrunsym = arg_parser.flag("lrsunsym", "lrunsym", "Unsymmetric low rank update", false);

    bool dump = arg_parser.flag("d", "dump", "Dump hmatrix structure", false);
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
    // Create point cloud
    PointCloud<H2Opus_Real> pt_cloud;
    if (grid_z > 1)
        generate3DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, grid_z, 0, 1, 0, 1, 0, 1);
    else if (grid_y > 1)
        generate2DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, 0, 1, 0, 1);
    else
        generate1DGrid<H2Opus_Real>(pt_cloud, grid_x, 0, 1);
    int dim = pt_cloud.getDimension();
    size_t n = pt_cloud.getDataSetSize();
    printf("N = %d\n", (int)n);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Matrix construction
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Setup hmatrix construction parameters:
    // Create a functor that can generate the matrix entries from two points
    FunctionGen<H2Opus_Real> func_gen(dim);
    // Create an entry gen struct from the functor. Currently only supports chebyshev interpolation on the CPU
    BoxEntryGen<H2Opus_Real, H2OPUS_HWTYPE_CPU, FunctionGen<H2Opus_Real>> entry_gen(func_gen);

    // DiagGen<H2Opus_Real> func_gen(dim);
    // BoxEntryGen<H2Opus_Real, H2OPUS_HWTYPE_CPU, DiagGen<H2Opus_Real>> entry_gen(func_gen);

    // Create the admissibility condition using the eta parameter
    // Decreasing eta refines the matrix tree and increasing it coarsens the tree
    H2OpusBoxCenterAdmissibility admissibility(eta);

    // Build the hmatrix structure
    HMatrix hmatrix(n, !matunsym);
    buildHMatrixStructure(hmatrix, &pt_cloud, leaf_size, admissibility);
    HMatrix zero_hmatrix = hmatrix;

    if (output_eps)
        outputEps(hmatrix, "structure.eps");

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Global HLRU
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create h2opus handle
    h2opusHandle_t h2opus_handle;
    h2opusCreateHandle(&h2opus_handle);

    // Dump hmatrix structure
    if (dump)
    {
        printf("Initial HMatrix\n");
        dumpHMatrix(hmatrix, 4, NULL);
    }

    // Generate a random low rank update
    int ldu = n + 4;
    int ldv = n + 7;
    thrust::host_vector<H2Opus_Real> U(ldu * rank);
    thrust::host_vector<H2Opus_Real> V(ldv * rank);
    random_vector<H2Opus_Real, H2OPUS_HWTYPE_CPU>(h2opus_handle, vec_ptr(U), ldu * rank);
    random_vector<H2Opus_Real, H2OPUS_HWTYPE_CPU>(h2opus_handle, vec_ptr(V), ldv * rank);
    if (!lrunsym) // symmetric low-rank update
    {
        V = U;
        ldv = ldu;
    }

    // We can apply the low rank update all at once, or a few vectors at a time
    // If we're compressing the matrix after each low rank update, we consume less
    // memory at the expense of increased runtime (due to a greater number of compressions)
    int applied_rank = 0, rank_per_update = 32;
    LowRankSampler<H2OPUS_HWTYPE_CPU> sampler(vec_ptr(U), ldu, vec_ptr(V), ldv, n, rank, h2opus_handle);
    H2Opus_Real lru_norm = sampler_norm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(&sampler, n, 40, h2opus_handle);

    while (applied_rank < rank)
    {
        int rank_to_apply = std::min(rank_per_update, rank - applied_rank);
        H2Opus_Real *U_update = vec_ptr(U) + applied_rank * ldu;
        H2Opus_Real *V_update = vec_ptr(V) + applied_rank * ldv;

        // apply n times up to 1.0 as scaling factor
        for (int napp = 0; napp < 8; napp++)
        {
            hlru_global(hmatrix, U_update, ldu, V_update, ldv, rank_to_apply, 0.125, h2opus_handle);
        }

        applied_rank += rank_to_apply;
    }

    // Check the difference between the low rank update and the updated hmatrix
    if (check_lru_err)
    {
        H2Opus_Real lru_err = sampler_difference<H2Opus_Real, H2OPUS_HWTYPE_CPU>(&sampler, hmatrix, 40, h2opus_handle);
        printf("CPU Global update difference = %e\n", lru_err / lru_norm);
    }

    // Dump hmatrix structure
    if (dump)
    {
        printf("HMatrix after low-rank update\n");
        dumpHMatrix(hmatrix, 4, NULL);
    }

#ifdef H2OPUS_USE_GPU
    // Copy the hmatrix over to the GPU
    HMatrix_GPU gpu_h = zero_hmatrix;
    thrust::device_vector<H2Opus_Real> d_U = U;
    thrust::device_vector<H2Opus_Real> d_V = V;
    LowRankSampler<H2OPUS_HWTYPE_GPU> sampler_gpu(vec_ptr(d_U), ldu, vec_ptr(d_V), ldv, n, rank, h2opus_handle);

    applied_rank = 0;
    while (applied_rank < rank)
    {
        int rank_to_apply = std::min(rank_per_update, rank - applied_rank);
        H2Opus_Real *U_update = vec_ptr(d_U) + applied_rank * ldu;
        H2Opus_Real *V_update = vec_ptr(d_V) + applied_rank * ldv;

        // apply n times up to 1.0 as scaling factor
        for (int napp = 0; napp < 8; napp++)
        {
            hlru_global(gpu_h, U_update, ldu, V_update, ldv, rank_to_apply, 0.125, h2opus_handle);
        }

        applied_rank += rank_to_apply;
    }
    // Check the difference between the low rank update and the updated hmatrix
    if (check_lru_err)
    {
        H2Opus_Real lru_err =
            sampler_difference<H2Opus_Real, H2OPUS_HWTYPE_GPU>(&sampler_gpu, gpu_h, 40, h2opus_handle);
        printf("GPU Global update difference = %e\n", lru_err / lru_norm);
    }
#endif

    // Clean up
    h2opusDestroyHandle(h2opus_handle);

    return 0;
}
