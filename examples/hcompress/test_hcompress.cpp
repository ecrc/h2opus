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
    H2Opus_Real trunc_eps =
        arg_parser.option<H2Opus_Real>("te", "trunc_eps", "Relative truncation error threshold", 1e-4);
    bool output_eps = arg_parser.flag("o", "output_eps", "Output structure of the matrix as an eps file", false);
    bool check_compress_err = arg_parser.flag("c", "check_compress_err", "Check the compression error", false);
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
    HMatrix original_hmatrix = hmatrix;

    if (output_eps)
        outputEps(hmatrix, "structure.eps");

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Hcompress
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create h2opus handle
    h2opusHandle_t h2opus_handle;
    h2opusCreateHandle(&h2opus_handle);

    // Pre-allocate workspace. This can be safely skipped if performance is not a concern
    H2OpusWorkspaceState ws_needed = horthog_workspace(hmatrix);
    h2opus_handle->setWorkspaceState(ws_needed);
    ws_needed = hcompress_workspace(hmatrix);
    h2opus_handle->setWorkspaceState(ws_needed);

    // Compression uses an absolute norm, so we first approximate the norm of the matrix
    H2Opus_Real approx_hmatrix_norm = hmatrix_norm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(original_hmatrix, 10, h2opus_handle);
    trunc_eps = trunc_eps * approx_hmatrix_norm;

    // Performance data
    double gather_gflops, project_gflops, trunc_gflops;
    double gather_time, project_time, trunc_time;
    double gather_perf, project_perf, trunc_perf;
    double gather_dev, project_dev, trunc_dev;

    double total_gops, total_time, total_perf, total_dev;

    // CPU runs
    const int runs = 10;

    HLibProfile::clear();
    for (int i = 0; i < runs; i++)
    {
        hmatrix = original_hmatrix;
        horthog(hmatrix, h2opus_handle);
        hcompress(hmatrix, trunc_eps, h2opus_handle);
    }

    HLibProfile::getPhasePerformance(HLibProfile::HCOMPRESS_BASIS_GEN, gather_gflops, gather_time, gather_perf,
                                     gather_dev);
    HLibProfile::getPhasePerformance(HLibProfile::HCOMPRESS_TRUNCATE_BASIS, trunc_gflops, trunc_time, trunc_perf,
                                     trunc_dev);
    HLibProfile::getPhasePerformance(HLibProfile::HCOMPRESS_PROJECTION, project_gflops, project_time, project_perf,
                                     project_dev);
    HLibProfile::getHcompressPerf(total_gops, total_time, total_perf, total_dev);

    printf("Compression BasisGen:   %.5f s at %.2f GFLOP/s (%.2f of total)\n", gather_time, gather_perf,
           gather_time / total_time * 100);
    printf("Compression Truncation: %.5f s at %.2f GFLOP/s (%.2f of total)\n", trunc_time, trunc_perf,
           trunc_time / total_time * 100);
    printf("Compression Projection: %.5f s at %.2f GFLOP/s (%.2f of total)\n", project_time, project_perf,
           project_time / total_time * 100);
    printf("Total compression time: %.5f s at %.2f GFLOP/s\n", total_time, total_perf);

    if (check_compress_err)
    {
        SimpleHMatrixSampler<H2OPUS_HWTYPE_CPU> sampler(&original_hmatrix, h2opus_handle);
        H2Opus_Real abs_diff = sampler_difference<H2Opus_Real, H2OPUS_HWTYPE_CPU>(&sampler, hmatrix, 40, h2opus_handle);
        printf("CPU Compression Error = %e\n", abs_diff / approx_hmatrix_norm);
    }

#ifdef H2OPUS_USE_GPU
    // Copy the hmatrix over to the GPU
    HMatrix_GPU gpu_h = original_hmatrix;

    // Pre-allocate workspace. This can be safely skipped if performance is not a concern
    ws_needed = horthog_workspace(gpu_h);
    h2opus_handle->setWorkspaceState(ws_needed);
    ws_needed = hcompress_workspace(gpu_h);
    h2opus_handle->setWorkspaceState(ws_needed);

    HLibProfile::clear();
    // Orthogonalize and compress
    for (int i = 0; i < runs; i++)
    {
        gpu_h = original_hmatrix;
        horthog(gpu_h, h2opus_handle);
        hcompress(gpu_h, trunc_eps, h2opus_handle);
    }

    HLibProfile::getPhasePerformance(HLibProfile::HCOMPRESS_BASIS_GEN, gather_gflops, gather_time, gather_perf,
                                     gather_dev);
    HLibProfile::getPhasePerformance(HLibProfile::HCOMPRESS_TRUNCATE_BASIS, trunc_gflops, trunc_time, trunc_perf,
                                     trunc_dev);
    HLibProfile::getPhasePerformance(HLibProfile::HCOMPRESS_PROJECTION, project_gflops, project_time, project_perf,
                                     project_dev);
    HLibProfile::getHcompressPerf(total_gops, total_time, total_perf, total_dev);

    printf("Compression BasisGen:   %.5f s at %.2f GFLOP/s (%.2f of total)\n", gather_time, gather_perf,
           gather_time / total_time * 100);
    printf("Compression Truncation: %.5f s at %.2f GFLOP/s (%.2f of total)\n", trunc_time, trunc_perf,
           trunc_time / total_time * 100);
    printf("Compression Projection: %.5f s at %.2f GFLOP/s (%.2f of total)\n", project_time, project_perf,
           project_time / total_time * 100);
    printf("Total compression time: %.5f s at %.2f GFLOP/s\n", total_time, total_perf);

    if (check_compress_err)
    {
        HMatrix_GPU gpu_original_hmatrix = original_hmatrix;
        SimpleHMatrixSampler<H2OPUS_HWTYPE_GPU> gpu_sampler(&gpu_original_hmatrix, h2opus_handle);
        H2Opus_Real abs_diff =
            sampler_difference<H2Opus_Real, H2OPUS_HWTYPE_GPU>(&gpu_sampler, gpu_h, 40, h2opus_handle);
        printf("GPU Compression Error = %e\n", abs_diff / approx_hmatrix_norm);
    }

#endif
    // Compute post-compression stats
    printf("Basis orthogonality: %e\n", getBasisOrthogonality(hmatrix.u_basis_tree, false));
    printf("Dense memory consumption: %.3f GB\n", original_hmatrix.getDenseMemoryUsage());
    printf("Low rank Memory consumption before truncation: %.3f GB\n", original_hmatrix.getLowRankMemoryUsage());
    printf("Low rank Memory consumption after truncation: %.3f GB\n", hmatrix.getLowRankMemoryUsage());

    h2opusDestroyHandle(h2opus_handle);
    return 0;
}
