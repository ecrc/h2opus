#include <stdio.h>
#include <h2opus.h>

#include "../common/example_problem.h"
#include "../common/example_util.h"
#include "../common/hmatrix_samplers.h"
#include <h2opus/util/boxentrygen.h>

template <int hw>
void test_construction(HMatrixSampler *sampler, THMatrix<hw> &hmatrix, int max_samples, H2Opus_Real trunc_eps,
                       const char *label, h2opusHandle_t h2opus_handle)
{
    int n = hmatrix.n;
    H2Opus_Real approx_norm = sampler_norm<H2Opus_Real, hw>(sampler, n, 10, h2opus_handle);
    H2Opus_Real abs_trunc_tol = trunc_eps * approx_norm;
    printf("%s approximate norm = %e, abs_tol = %e\n", label, approx_norm, abs_trunc_tol);

    hara(sampler, hmatrix, max_samples, 10, abs_trunc_tol, 32, h2opus_handle);
    H2Opus_Real approx_construnction_error =
        sampler_difference<H2Opus_Real, hw>(sampler, hmatrix, 40, h2opus_handle) / approx_norm;
    printf("%s %s construction error = %e\n", (hw == H2OPUS_HWTYPE_CPU ? "CPU" : "GPU"), label,
           approx_construnction_error);
}

template <int hw>
void RunTest(THMatrix<hw> &hmatrix, THMatrix<hw> &zero_hmatrix, h2opusHandle_t h2opus_handle, H2Opus_Real trunc_eps,
             int max_samples)
{
    THMatrix<hw> constructed_hmatrix = zero_hmatrix;

    printf("\n-----------------------------------------------\n");
    printf("%s results\n\n", hw == H2OPUS_HWTYPE_GPU ? "GPU matrix" : "CPU matrix");

    // Reconstruction matrix via hara
    SimpleHMatrixSampler<hw> reconstruct_sampler(hmatrix, h2opus_handle);
    test_construction<hw>(&reconstruct_sampler, constructed_hmatrix, max_samples, trunc_eps, "Matrix", h2opus_handle);

    // Clear out matrix data
    constructed_hmatrix = zero_hmatrix;

    // Squaring
    SquareSampler<hw> square_sampler(reconstruct_sampler, h2opus_handle);
    test_construction<hw>(&square_sampler, constructed_hmatrix, max_samples, trunc_eps, "Square", h2opus_handle);

    return;
}

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
    int cheb_grid_pts = arg_parser.option<int>(
        "k", "cheb_grid_pts", "Number of grid points in each dimension for Chebyshev interpolation (rank = k^d)", 8);
    int max_samples =
        arg_parser.option<int>("s", "max_samples", "Max number of samples to take for each level of the h2opus", 128);
    H2Opus_Real eta = arg_parser.option<H2Opus_Real>("e", "eta", "Admissibility parameter eta", DEFAULT_ETA);
    H2Opus_Real trunc_eps = arg_parser.option<H2Opus_Real>(
        "te", "trunc_eps", "Relative truncation error threshold for the construction", 1e-4);
    bool output_eps = arg_parser.flag("o", "output_eps", "Output structure of the matrix as an eps file", false);
    bool dump = arg_parser.flag("d", "dump", "Dump hmatrix structure", false);
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
    printf("N = %d\n", (int)n);
    // Create point cloud
    int dim = (grid_z == 1 ? (grid_y == 1 ? 1 : 2) : 3);
    PointCloud<H2Opus_Real> pt_cloud;
    if (dim == 3)
        generate3DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, grid_z, 0, 1, 0, 1, 0, 1);
    else if (dim == 2)
        generate2DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, 0, 1, 0, 1);
    else
        generate1DGrid<H2Opus_Real>(pt_cloud, grid_x, 0, 1);

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

    // Build the hmatrix.
    // Currently only symmetric matrices are fully supported when constructing from matvecs
    HMatrix hmatrix(n, true), constructed_hmatrix(n, true);
    buildHMatrix(hmatrix, &pt_cloud, admissibility, entry_gen, leaf_size, cheb_grid_pts);
    buildHMatrixStructure(constructed_hmatrix, &pt_cloud, leaf_size, admissibility);
    HMatrix zero_hmatrix = constructed_hmatrix;

    // Dump hmatrix structure
    if (dump)
    {
        printf("Initial HMatrix\n");
        dumpHMatrix(constructed_hmatrix, 2, NULL);
    }

    if (output_eps)
        outputEps(hmatrix, "structure.eps");

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // H2OPUS
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create h2opus handle
    h2opusHandle_t h2opus_handle;
    h2opusCreateHandle(&h2opus_handle);

    /* Run CPU tests */
    RunTest(hmatrix, zero_hmatrix, h2opus_handle, trunc_eps, max_samples);

#ifdef H2OPUS_USE_GPU
    HMatrix_GPU gpu_hmatrix = hmatrix, gpu_zero_hmatrix = zero_hmatrix;
    RunTest(gpu_hmatrix, gpu_zero_hmatrix, h2opus_handle, trunc_eps, max_samples);
#endif

    h2opusDestroyHandle(h2opus_handle);

    return 0;
}
