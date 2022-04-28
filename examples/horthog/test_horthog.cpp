#include <stdio.h>
#include <h2opus.h>

#include "../common/hmatrix_samplers.h"
#include "../common/example_problem.h"
#include "../common/example_util.h"
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
    int cheb_grid_pts = arg_parser.option<int>(
        "k", "cheb_grid_pts", "Number of grid points in each dimension for Chebyshev interpolation (rank = k^d)", 8);
    H2Opus_Real eta = arg_parser.option<H2Opus_Real>("e", "eta", "Admissibility parameter eta", DEFAULT_ETA);
    bool output_eps = arg_parser.flag("o", "output_eps", "Output structure of the matrix as an eps file", false);
    bool check_orthog_err = arg_parser.flag("c", "check_orthog_err", "Check the orthogonalization error", false);
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
    else
        generate2DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, 0, 1, 0, 1);
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

    // Create the admissibility condition using the eta parameter
    // Decreasing eta refines the matrix tree and increasing it coarsens the tree
    H2OpusBoxCenterAdmissibility admissibility(eta);

    // Build the hmatrix. Currently only symmetric matrices are fully supported
    HMatrix hmatrix(n, true);
    buildHMatrix(hmatrix, &pt_cloud, admissibility, entry_gen, leaf_size, cheb_grid_pts);

    if (output_eps)
        outputEps(hmatrix, "structure.eps");

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Horthog
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create h2opus handle
    h2opusHandle_t h2opus_handle;
    h2opusCreateHandle(&h2opus_handle);

    const int runs = 10;

    // Pre-allocate workspace. This can be safely skipped if performance is not a concern
    H2OpusWorkspaceState ws_needed = horthog_workspace(hmatrix);
    h2opus_handle->setWorkspaceState(ws_needed);

    // CPU runs
    HMatrix orthog_hmatrix = hmatrix;

    HLibProfile::clear();
    for (int i = 0; i < runs; i++)
    {
        orthog_hmatrix = hmatrix;
        horthog(orthog_hmatrix, h2opus_handle);
    }

    // Performance data
    double leaf_gops, leaf_time, leaf_perf, leaf_dev;
    double upsweep_gops, upsweep_time, upsweep_perf, upsweep_dev;
    double coupling_gops, coupling_time, coupling_perf, coupling_dev;
    double total_gops, total_time, total_perf, total_dev;

    // Make sure the basis is orthogonal
    printf("CPU Basis orthogonality: %e\n", getBasisOrthogonality(orthog_hmatrix.u_basis_tree, false));

    // Approximate the 2-norm of the difference of the matrices
    if (check_orthog_err)
    {
        SimpleHMatrixSampler<H2OPUS_HWTYPE_CPU> sampler(&hmatrix, h2opus_handle);
        H2Opus_Real approx_norm = sampler_norm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(&sampler, n, 10, h2opus_handle);
        H2Opus_Real abs_diff =
            sampler_difference<H2Opus_Real, H2OPUS_HWTYPE_CPU>(&sampler, orthog_hmatrix, 40, h2opus_handle);
        printf("CPU Orthog Error = %e\n", abs_diff / approx_norm);
    }

    // Gather peroformance metrics of each phase of the computation
    HLibProfile::getPhasePerformance(HLibProfile::HORTHOG_BASIS_LEAVES, leaf_gops, leaf_time, leaf_perf, leaf_dev);
    HLibProfile::getPhasePerformance(HLibProfile::HORTHOG_UPSWEEP, upsweep_gops, upsweep_time, upsweep_perf,
                                     upsweep_dev);
    HLibProfile::getPhasePerformance(HLibProfile::HORTHOG_PROJECTION, coupling_gops, coupling_time, coupling_perf,
                                     coupling_dev);
    HLibProfile::getHorthogPerf(total_gops, total_time, total_perf, total_dev);

    printf("CPU Orthog Leaves: %f Gflop in %f at %f Gflops (%.3f dev)\n", leaf_gops, leaf_time, leaf_perf, leaf_dev);
    printf("CPU Orthog Upsweep: %f Gflop in %f at %f Gflops (%.3f dev)\n", upsweep_gops, upsweep_time, upsweep_perf,
           upsweep_dev);
    printf("CPU Orthog Projection: %f Gflop in %f at %f Gflops (%.3f dev)\n", coupling_gops, coupling_time,
           coupling_perf, coupling_dev);
    printf("CPU Total execution time: %f s at %f (Gflop/s) (%.3f dev)\n", total_time, total_perf, total_dev);

#ifdef H2OPUS_USE_GPU
    HMatrix_GPU gpu_h = hmatrix;

    // Pre-allocate workspace. This can be safely skipped if performance is not a concern
    ws_needed = horthog_workspace(gpu_h);
    h2opus_handle->setWorkspaceState(ws_needed);

    HLibProfile::clear();
    for (int i = 0; i < runs; i++)
    {
        gpu_h = hmatrix;
        horthog(gpu_h, h2opus_handle);
    }

    // Make sure the basis is orthogonal
    HMatrix gpu_orthog_hmatrix = gpu_h;
    printf("GPU Basis orthogonality: %e\n", getBasisOrthogonality(gpu_orthog_hmatrix.u_basis_tree, false));

    // Approximate the 2-norm of the difference of the matrices
    if (check_orthog_err)
    {
        HMatrix_GPU gpu_original_hmatrix = hmatrix;
        SimpleHMatrixSampler<H2OPUS_HWTYPE_GPU> gpu_sampler(&gpu_original_hmatrix, h2opus_handle);
        H2Opus_Real approx_norm = sampler_norm<H2Opus_Real, H2OPUS_HWTYPE_GPU>(&gpu_sampler, n, 10, h2opus_handle);
        H2Opus_Real abs_diff =
            sampler_difference<H2Opus_Real, H2OPUS_HWTYPE_GPU>(&gpu_sampler, gpu_h, 40, h2opus_handle);
        printf("GPU Orthog Error = %e\n", abs_diff / approx_norm);
    }

    // Gather peroformance metrics of each phase of the computation
    HLibProfile::getPhasePerformance(HLibProfile::HORTHOG_BASIS_LEAVES, leaf_gops, leaf_time, leaf_perf, leaf_dev);
    HLibProfile::getPhasePerformance(HLibProfile::HORTHOG_UPSWEEP, upsweep_gops, upsweep_time, upsweep_perf,
                                     upsweep_dev);
    HLibProfile::getPhasePerformance(HLibProfile::HORTHOG_PROJECTION, coupling_gops, coupling_time, coupling_perf,
                                     coupling_dev);
    HLibProfile::getHorthogPerf(total_gops, total_time, total_perf, total_dev);

    printf("GPU Orthog Leaves: %f Gflop in %f at %f Gflops (%.3f dev)\n", leaf_gops, leaf_time, leaf_perf, leaf_dev);
    printf("GPU Orthog Upsweep: %f Gflop in %f at %f Gflops (%.3f dev)\n", upsweep_gops, upsweep_time, upsweep_perf,
           upsweep_dev);
    printf("GPU Orthog Projection: %f Gflop in %f at %f Gflops (%.3f dev)\n", coupling_gops, coupling_time,
           coupling_perf, coupling_dev);
    printf("GPU Total execution time: %f s at %f (Gflop/s) (%.3f dev)\n", total_time, total_perf, total_dev);
#endif

    h2opusDestroyHandle(h2opus_handle);
    return 0;
}
