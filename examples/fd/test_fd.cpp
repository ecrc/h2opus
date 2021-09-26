#include <h2opus.h>
#include <h2opus/util/timer.h>
#include "fd_core.h"
#include "../common/example_util.h"
#include "../common/hmatrix_samplers.h"
#include <h2opus/util/boxentrygen.h>
#include <cstdio>

#define DEFAULT_ETA 1.0

template <int hw>
void RunPerf(HMatrix &hmatrix, h2opusHandle_t h2opus_handle, H2Opus_Real trunc_eps, int nruns, bool check_compress_err,
             std::vector<double> &summary)
{
    printf("\nRunning %s tests\n", hw ? "GPU" : "CPU");

    // Compression uses an absolute norm, so we first approximate the norm of the matrix
    h2opus_handle->setRandSeed(123456, H2OPUS_HWTYPE_CPU);
    H2Opus_Real approx_hmatrix_norm = hmatrix_norm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(hmatrix, 20, h2opus_handle);

    // Pre-allocate workspace. This can be safely skipped if performance is not a concern
    H2OpusWorkspaceState ws_needed = horthog_workspace(hmatrix);
    h2opus_handle->setWorkspaceState(ws_needed);
    ws_needed = hcompress_workspace(hmatrix);
    h2opus_handle->setWorkspaceState(ws_needed);

    // Compress first time and average over multiple runs
    // GPU matrices can be instantiated with a constructor taking a host matrix
    THMatrix<hw> compressedhmatrix(hmatrix);
    horthog(compressedhmatrix, h2opus_handle);
    hcompress(compressedhmatrix, trunc_eps * approx_hmatrix_norm, h2opus_handle);
    printf("  Hmatrix memory usage after compression: %g (dense) %g (low rank) %g GB (total)\n",
           compressedhmatrix.getDenseMemoryUsage(), compressedhmatrix.getLowRankMemoryUsage(),
           compressedhmatrix.getDenseMemoryUsage() + compressedhmatrix.getLowRankMemoryUsage());
    summary.push_back(hmatrix.getDenseMemoryUsage());
    summary.push_back(hmatrix.getLowRankMemoryUsage());
    summary.push_back(compressedhmatrix.getLowRankMemoryUsage());

    H2Opus_Real approx_compressedhmatrix_norm = hmatrix_norm<H2Opus_Real, hw>(compressedhmatrix, 20, h2opus_handle);
    printf("  Original matrix 2-norm %g\n", approx_hmatrix_norm);
    printf("  Compressed Hmatrix 2-norm %g\n", approx_compressedhmatrix_norm);

    // Performance data
    // Average over multiple runs
    HLibProfile::clear();
    double gather_gops, project_gops, trunc_gops;
    double gather_time, project_time, trunc_time;
    double gather_perf, project_perf, trunc_perf;
    double gather_dev, project_dev, trunc_dev;
    double total_gops, total_time, total_perf, total_dev;
    double leaf_gops, leaf_time, leaf_perf, leaf_dev;
    double upsweep_gops, upsweep_time, upsweep_perf, upsweep_dev;
    double coupling_gops, coupling_time, coupling_perf, coupling_dev;
    double total_gops_o, total_time_o, total_perf_o, total_dev_o;

    for (int i = 0; i < nruns; i++)
    {
        // GPU matrices can be copied from host
        compressedhmatrix = hmatrix;
        horthog(compressedhmatrix, h2opus_handle);
        hcompress(compressedhmatrix, trunc_eps * approx_hmatrix_norm, h2opus_handle);
    }

    HLibProfile::getPhasePerformance(HLibProfile::HORTHOG_BASIS_LEAVES, leaf_gops, leaf_time, leaf_perf, leaf_dev);
    HLibProfile::getPhasePerformance(HLibProfile::HORTHOG_UPSWEEP, upsweep_gops, upsweep_time, upsweep_perf,
                                     upsweep_dev);
    HLibProfile::getPhasePerformance(HLibProfile::HORTHOG_PROJECTION, coupling_gops, coupling_time, coupling_perf,
                                     coupling_dev);
    HLibProfile::getHorthogPerf(total_gops_o, total_time_o, total_perf_o, total_dev_o);
    HLibProfile::getPhasePerformance(HLibProfile::HCOMPRESS_BASIS_GEN, gather_gops, gather_time, gather_perf,
                                     gather_dev);
    HLibProfile::getPhasePerformance(HLibProfile::HCOMPRESS_TRUNCATE_BASIS, trunc_gops, trunc_time, trunc_perf,
                                     trunc_dev);
    HLibProfile::getPhasePerformance(HLibProfile::HCOMPRESS_PROJECTION, project_gops, project_time, project_perf,
                                     project_dev);
    HLibProfile::getHcompressPerf(total_gops, total_time, total_perf, total_dev);

    printf("  --------------------------- Performances --------------------------------\n");
    printf("  Orthog Leaves:          %.5f s at %.2f GFLOP/s (GFLOPs log %g) (%.2f of total)\n", leaf_time, leaf_perf,
           leaf_gops, leaf_time / total_time_o * 100);
    printf("  Orthog Upsweep:         %.5f s at %.2f GFLOP/s (GFLOPs log %g) (%.2f of total)\n", upsweep_time,
           upsweep_perf, upsweep_gops, upsweep_time / total_time_o * 100);
    printf("  Orthog Projection:      %.5f s at %.2f GFLOP/s (GFLOPs log %g) (%.2f of total)\n", coupling_time,
           coupling_perf, coupling_gops, coupling_time / total_time_o * 100);
    printf("  Compression BasisGen:   %.5f s at %.2f GFLOP/s (GFLOPs log %g) (%.2f of total)\n", gather_time,
           gather_perf, gather_gops, gather_time / total_time * 100);
    printf("  Compression Truncation: %.5f s at %.2f GFLOP/s (GFLOPs log %g) (%.2f of total)\n", trunc_time, trunc_perf,
           trunc_gops, trunc_time / total_time * 100);
    printf("  Compression Projection: %.5f s at %.2f GFLOP/s (GFLOPs log %g) (%.2f of total)\n", project_time,
           project_perf, project_gops, project_time / total_time * 100);
    printf("  Total orthogonalization time: %.5f s at %.2f GFLOP/s (GFLOPs log %g)\n", total_time_o, total_perf_o,
           total_gops_o);
    printf("  Total compression       time: %.5f s at %.2f GFLOP/s (GFLOPs log %g)\n", total_time, total_perf,
           total_gops);

    summary.push_back(total_time_o);
    summary.push_back(total_gops_o);
    summary.push_back(total_time);
    summary.push_back(total_gops);
    summary.push_back(total_time_o + total_time);
    summary.push_back(total_gops_o + total_gops);

    // Check compression error by approximating 2-norm of the difference
    if (check_compress_err)
    {
        SimpleHMatrixSampler<hw> sampler(&hmatrix, h2opus_handle);
        H2Opus_Real abs_diff = sampler_difference<H2Opus_Real, hw>(&sampler, compressedhmatrix, 20, h2opus_handle);
        printf("  Compression Error = %e\n", abs_diff / approx_hmatrix_norm);
    }
}

int main(int argc, char **argv)
{
    // Argument parsing
    H2OpusArgParser arg_parser;
    arg_parser.setArgs(argc, argv);

    int grid_x = arg_parser.option<int>("gx", "grid_x", "Grid points in the X direction", 32);
    int leaf_size = arg_parser.option<int>("m", "leaf_size", "Leaf size in the KD-tree", 32);
    int cheb_grid_pts = arg_parser.option<int>(
        "k", "cheb_grid_pts", "Number of grid points in each dimension for Chebyshev interpolation (rank = k^d)", 8);
    H2Opus_Real eta = arg_parser.option<H2Opus_Real>("e", "eta", "Admissibility parameter eta", DEFAULT_ETA);
    int num_vectors = arg_parser.option<int>("nv", "num_vectors", "Number of vectors the matrix multiplies", 1);
    H2Opus_Real trunc_eps =
        arg_parser.option<H2Opus_Real>("te", "trunc_eps", "Relative truncation error threshold", 1e-4);
    int nruns = arg_parser.option<int>("n", "nruns", "Number of runs to perform", 10);
    int dim = arg_parser.option<int>("dim", "dim", "The geometrical dimension", 2);
    bool output_eps = arg_parser.flag("o", "output_eps", "Output structure of the matrix as an eps file", false);
    bool check_approx_err = arg_parser.flag("ca", "check_approx_err", "Check the approximation error", false);
    bool check_compress_err = arg_parser.flag("c", "check_approx_err", "Check the compression error", false);
    bool summary = arg_parser.flag("summary", "summary", "Print brief summary", false);
    bool print_help = arg_parser.flag("h", "help", "This message", false);

    if (!arg_parser.valid() || print_help)
    {
        arg_parser.printUsage();
        return 0;
    }

    // Geometry
    PointCloud<H2Opus_Real> pt_cloud;
    pt_cloud.generateGrid(dim, grid_x, -1.0 + 2.0 / (grid_x + 1), 1.0 - 2.0 / (grid_x + 1));
    size_t n = pt_cloud.getDataSetSize();
    printf("N = %d\n", (int)n);

    // Generate random input for the hgemv
    thrust::host_vector<H2Opus_Real> x(n * num_vectors, 1), y(n * num_vectors, 0);
    for (size_t i = 0; i < x.size(); i++)
        x[i] = (H2Opus_Real)rand() / (RAND_MAX + 1.0);

    // Create a functor that can generate the matrix entries from two points
    FDGen<H2Opus_Real> func_gen(dim, pt_cloud.h);

    // Create an entry gen struct from the functor. Currently only supports chebyshev interpolation on the CPU
    BoxEntryGen<H2Opus_Real, H2OPUS_HWTYPE_CPU, FDGen<H2Opus_Real>> entry_gen(func_gen);

    // Create the admissibility condition using the eta parameter
    // Decreasing eta refines the matrix tree and increasing it coarsens the tree
    H2OpusBoxCenterAdmissibility admissibility(eta);

    // Build the hmatrix
    HMatrix hmatrix(n, true);
    Timer<H2OPUS_HWTYPE_CPU> timer;
    timer.init();
    timer.start();
    buildHMatrix(hmatrix, &pt_cloud, admissibility, entry_gen, leaf_size, cheb_grid_pts);
    printf("Hmatrix constructed in %fs\n", timer.stop());
    printf("Hmatrix memory usage before compression: %g (dense) %g (low rank) %g GB (total)\n",
           hmatrix.getDenseMemoryUsage(), hmatrix.getLowRankMemoryUsage(),
           hmatrix.getDenseMemoryUsage() + hmatrix.getLowRankMemoryUsage());

    if (output_eps)
        outputEps(hmatrix, "structure.eps");

    // Create h2opus handle
    h2opusHandle_t h2opus_handle;
    h2opusCreateHandle(&h2opus_handle);

    // Check the approximation error by testing a small portion p of the dense matrix vector product
    MatGen<H2Opus_Real> mat_gen(func_gen, pt_cloud, &hmatrix.u_basis_tree.index_map[0]);
    H2Opus_Real p = 0.01;
    if (check_approx_err)
    {
        H2Opus_Real max_approx_err = 0;

        fillArray(vec_ptr(y), n * num_vectors, 0, h2opus_handle->getMainStream(), H2OPUS_HWTYPE_CPU);
        hgemv(H2Opus_NoTrans, 1.0, hmatrix, vec_ptr(x), n, 0.0, vec_ptr(y), n, num_vectors, h2opus_handle);
        for (int i = 0; i < num_vectors; i++)
        {
            H2Opus_Real *y1 = vec_ptr(y) + i * n;
            H2Opus_Real *x1 = vec_ptr(x) + i * n;

            H2Opus_Real approx_err = getHgemvApproximationError(n, mat_gen, p, y1, x1);
            max_approx_err = std::max(max_approx_err, approx_err);
        }
        printf("CPU Max approx error = %e\n", max_approx_err);
    }

    // Run performance tests
    std::vector<double> summaryv;
    RunPerf<H2OPUS_HWTYPE_CPU>(hmatrix, h2opus_handle, trunc_eps, nruns, check_compress_err, summaryv);
#ifdef H2OPUS_USE_GPU
    RunPerf<H2OPUS_HWTYPE_GPU>(hmatrix, h2opus_handle, trunc_eps, nruns, check_compress_err, summaryv);
#else
    for (int i = 0; i < 9; i++)
        summaryv.push_back(0);
#endif
    if (summary)
    {
        printf(
            "=============================================================================================== SUMMARY "
            "===============================================================================================\n");
        printf("%d\t%ld\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1."
               "6g\t%1.6g\n",
               1, n, summaryv[0], summaryv[1], summaryv[2], summaryv[3], summaryv[4], summaryv[12], summaryv[13],
               summaryv[5], summaryv[6], summaryv[14], summaryv[15], summaryv[7], summaryv[8], summaryv[16],
               summaryv[17]);
        printf("======================================================================================================="
               "================================================================================================\n");
    }
    // Clean up
    h2opusDestroyHandle(h2opus_handle);
    return 0;
}
