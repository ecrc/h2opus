#include <h2opusconf.h>
#if defined(H2OPUS_USE_MPI)
#include <h2opus.h>
#include <h2opus/distributed/distributed_h2opus_handle.h>
#include <h2opus/distributed/distributed_geometric_construction.h>
#include <h2opus/distributed/distributed_hgemv.h>
#include <h2opus/distributed/distributed_horthog.h>
#include <h2opus/distributed/distributed_hcompress.h>
#include <h2opus/distributed/distributed_error_approximation.h>
#include "fd_core.h"
#include "../common/example_util.h"
#include "../common/hmatrix_samplers.h"
#include <h2opus/util/boxentrygen.h>

#define DEFAULT_ETA 1.0

void Sync(MPI_Comm comm, int hw)
{
    mpiErrchk(MPI_Barrier(comm));
#ifdef H2OPUS_USE_GPU
    if (hw)
        cudaDeviceSynchronize();
#else
    (void)(hw);
#endif
}

#include <cstdio>

// metric / second
int reducePerf(double &gflops, double &t, MPI_Comm comm)
{
    int size;

    gflops *= t;
    mpiErrchk(MPI_Comm_size(comm, &size));
    mpiErrchk(MPI_Allreduce(MPI_IN_PLACE, &gflops, 1, MPI_DOUBLE_PRECISION, MPI_SUM, comm));
    mpiErrchk(MPI_Allreduce(MPI_IN_PLACE, &t, 1, MPI_DOUBLE_PRECISION, MPI_SUM, comm));
    gflops /= t;
    t /= size;
    return 0;
}

// absolute metric
int reducePerf(double &gflops, MPI_Comm comm)
{
    int size;
    mpiErrchk(MPI_Comm_size(comm, &size));
    double dummy = 1;
    gflops *= size;
    return reducePerf(gflops, dummy, comm);
}

template <int hw>
void RunPerf(DistributedHMatrix &hmatrix, distributedH2OpusHandle_t dist_h2opus_handle, H2Opus_Real trunc_eps,
             int nruns, bool check_compress_err, std::vector<double> &summary)
{
    int proc_rank = dist_h2opus_handle->orank;
    MPI_Comm comm = dist_h2opus_handle->ocomm;
    if (!proc_rank)
        printf("\nRunning %s tests\n", hw ? "GPU" : "CPU");

    // Compression uses an absolute norm, so we first approximate the norm of the matrix
    dist_h2opus_handle->setRandSeed(dist_h2opus_handle->rank, H2OPUS_HWTYPE_CPU);
    H2Opus_Real approx_hmatrix_norm =
        distributed_hmatrix_norm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(hmatrix, 20, dist_h2opus_handle);

    // Compress first time and average over multiple runs
    // GPU matrices can be instantiated with a constructor taking a host matrix
    dist_h2opus_handle->setRandSeed(dist_h2opus_handle->rank, hw);
    TDistributedHMatrix<hw> compressedhmatrix(hmatrix);
    distributed_horthog(compressedhmatrix, dist_h2opus_handle);
    distributed_hcompress(compressedhmatrix, trunc_eps * approx_hmatrix_norm, dist_h2opus_handle);

    double mem[4];
    mem[0] = compressedhmatrix.getLocalDenseMemoryUsage();
    mem[1] = compressedhmatrix.getLocalLowRankMemoryUsage();
    mem[2] = compressedhmatrix.getLocalMemoryUsage();
    mem[3] = hmatrix.getLocalLowRankMemoryUsage();
    mpiErrchk(MPI_Allreduce(MPI_IN_PLACE, mem, 4, MPI_DOUBLE_PRECISION, MPI_SUM, comm));
    if (!proc_rank)
    {
        printf("  Hmatrix memory usage  after compression: %g (dense) %g (low rank) %g GB (total)\n", mem[0], mem[1],
               mem[2]);
    }
    summary.push_back(mem[0]);
    summary.push_back(mem[3]);
    summary.push_back(mem[1]);

    H2Opus_Real approx_compressedhmatrix_norm =
        distributed_hmatrix_norm<H2Opus_Real, hw>(compressedhmatrix, 20, dist_h2opus_handle);
    if (!proc_rank)
        printf("  Original matrix 2-norm %g\n", approx_hmatrix_norm);
    if (!proc_rank)
        printf("  Compressed Hmatrix 2-norm %g\n", approx_compressedhmatrix_norm);

    // Performance data
    // Average over multiple runs
    // We also log user times, since the logging support does not consider MPI calls
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
    double timer_counter[2] = {0, 0}, run_time;
    for (int i = 0; i < nruns; i++)
    {
        // GPU matrices can be copied from host
        compressedhmatrix = hmatrix;
        Sync(comm, hw);
        run_time = 0;
        run_time -= MPI_Wtime();
        distributed_horthog(compressedhmatrix, dist_h2opus_handle);
        Sync(comm, hw);
        run_time += MPI_Wtime();
        timer_counter[0] += run_time;

        run_time = 0;
        run_time -= MPI_Wtime();
        distributed_hcompress(compressedhmatrix, trunc_eps * approx_hmatrix_norm, dist_h2opus_handle);
        Sync(comm, hw);
        run_time += MPI_Wtime();
        timer_counter[1] += run_time;
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

    // Average local performances
    reducePerf(leaf_perf, leaf_time, comm);
    reducePerf(upsweep_perf, upsweep_time, comm);
    reducePerf(coupling_perf, coupling_time, comm);
    reducePerf(total_perf_o, total_time_o, comm);
    reducePerf(gather_perf, gather_time, comm);
    reducePerf(trunc_perf, trunc_time, comm);
    reducePerf(project_perf, project_time, comm);
    reducePerf(total_perf, total_time, comm);
    reducePerf(total_gops_o, comm);
    reducePerf(total_gops, comm);
    reducePerf(leaf_gops, comm);
    reducePerf(upsweep_gops, comm);
    reducePerf(coupling_gops, comm);
    reducePerf(gather_gops, comm);
    reducePerf(trunc_gops, comm);
    reducePerf(project_gops, comm);

    if (!proc_rank)
    {
        printf("  --------------------------- Performances --------------------------------\n");
        printf("  Orthog Leaves:          %.5f s at %.2f GFLOP/s (GFLOPs log %g) (%.2f of total)\n", leaf_time,
               leaf_perf, leaf_gops, leaf_time / total_time_o * 100);
        printf("  Orthog Upsweep:         %.5f s at %.2f GFLOP/s (GFLOPs log %g) (%.2f of total)\n", upsweep_time,
               upsweep_perf, upsweep_gops, upsweep_time / total_time_o * 100);
        printf("  Orthog Projection:      %.5f s at %.2f GFLOP/s (GFLOPs log %g) (%.2f of total)\n", coupling_time,
               coupling_perf, coupling_gops, coupling_time / total_time_o * 100);
        printf("  Compression BasisGen:   %.5f s at %.2f GFLOP/s (GFLOPs log %g) (%.2f of total)\n", gather_time,
               gather_perf, gather_gops, gather_time / total_time * 100);
        printf("  Compression Truncation: %.5f s at %.2f GFLOP/s (GFLOPs log %g) (%.2f of total)\n", trunc_time,
               trunc_perf, trunc_gops, trunc_time / total_time * 100);
        printf("  Compression Projection: %.5f s at %.2f GFLOP/s (GFLOPs log %g) (%.2f of total)\n", project_time,
               project_perf, project_gops, project_time / total_time * 100);
        printf("  Total orthogonalization time: %.5f s, log time %.5f at %.2f GFLOP/s, (GFLOPs log %g)\n",
               timer_counter[0] / nruns, total_time_o, total_perf_o, total_gops_o);
        printf("  Total compression       time: %.5f s, log time %.5f at %.2f GFLOP/s, (GFLOPs log %g)\n",
               timer_counter[1] / nruns, total_time, total_perf, total_gops);
    }

    summary.push_back(timer_counter[0] / nruns);
    summary.push_back(total_gops_o);
    summary.push_back(timer_counter[1] / nruns);
    summary.push_back(total_gops);
    summary.push_back((timer_counter[0] + timer_counter[1]) / nruns);
    summary.push_back(total_gops_o + total_gops);

    // Check compression error by approximating 2-norm of the difference
    if (check_compress_err)
    {
        int dist_ld = hmatrix.basis_tree.basis_branch.index_map.size();
        DistributedHMatrixDiffSampler<hw> sampler(&hmatrix, &compressedhmatrix, dist_h2opus_handle);
        H2Opus_Real abs_diff = sampler_norm<H2Opus_Real, hw>(&sampler, dist_ld, 20, dist_h2opus_handle);
        if (!proc_rank)
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
    H2Opus_Real trunc_eps =
        arg_parser.option<H2Opus_Real>("te", "trunc_eps", "Relative truncation error threshold", 1e-4);
    int nruns = arg_parser.option<int>("n", "nruns", "Number of runs to perform", 10);
    int dim = arg_parser.option<int>("dim", "dim", "The geometrical dimension", 2);
    bool check_compress_err = arg_parser.flag("c", "check_approx_err", "Check the compression error", false);
    bool summary = arg_parser.flag("summary", "summary", "Print brief summary", false);
    bool mpithreaded = arg_parser.flag("mpith", "mpithreaded", "Issue MPI calls from threads", false);
    bool print_help = arg_parser.flag("h", "help", "This message", false);

    if (!arg_parser.valid() || print_help)
    {
        initMPI(argc, argv, false);
        int rank;
        mpiErrchk(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        if (rank == 0)
            arg_parser.printUsage();
        return 0;
    }

    initMPI(argc, argv, mpithreaded);
    int rank, size;
    mpiErrchk(MPI_Comm_size(MPI_COMM_WORLD, &size));
    mpiErrchk(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    if (size == 1)
    {
        printf("Not for uniprocessor runs\n");
        return 0;
    }

    distributedH2OpusHandle_t dist_h2opus_handle;
    h2opusCreateDistributedHandle(&dist_h2opus_handle);
    MPI_Comm comm = dist_h2opus_handle->ocomm;
    if (!rank)
        printf("Using threaded MPI for CPU runs? %d\n", dist_h2opus_handle->getUseThreads<H2OPUS_HWTYPE_CPU>());
#ifdef H2OPUS_USE_GPU
    if (!rank)
        printf("Using threaded MPI for GPU runs? %d\n", dist_h2opus_handle->getUseThreads<H2OPUS_HWTYPE_GPU>());
#endif

    // Geometry
    PointCloud<H2Opus_Real> pt_cloud;
    pt_cloud.generateGrid(dim, grid_x, -1.0 + 2.0 / (grid_x + 1), 1.0 - 2.0 / (grid_x + 1));
    size_t n = pt_cloud.getDataSetSize();
    if (!rank)
    {
        printf("N = %d\n", (int)n);
    }

    // Create a functor that can generate the matrix entries from two points
    FDGen<H2Opus_Real> func_gen(dim, pt_cloud.h);

    // Create an entry gen struct from the functor. Currently only supports chebyshev interpolation on the CPU
    BoxEntryGen<H2Opus_Real, H2OPUS_HWTYPE_CPU, FDGen<H2Opus_Real>> entry_gen(func_gen);

    // Create the admissibility condition using the eta parameter
    // Decreasing eta refines the matrix tree and increasing it coarsens the tree
    H2OpusBoxCenterAdmissibility admissibility(eta);

    // Build the hmatrix. Currently only symmetric matrices are fully supported
    double run_time;

    MPI_Barrier(comm);
    run_time = 0;
    run_time -= MPI_Wtime();
    DistributedHMatrix hmatrix(n);
    buildDistributedHMatrix(hmatrix, &pt_cloud, admissibility, entry_gen, leaf_size, cheb_grid_pts, dist_h2opus_handle);
    MPI_Barrier(comm);
    run_time += MPI_Wtime();
    if (!rank)
        printf("Hmatrix constructed in %fs\n", run_time);

    double mem[3];
    mem[0] = hmatrix.getLocalDenseMemoryUsage();
    mem[1] = hmatrix.getLocalLowRankMemoryUsage();
    mem[2] = hmatrix.getLocalMemoryUsage();
    mpiErrchk(MPI_Allreduce(MPI_IN_PLACE, mem, 3, MPI_DOUBLE_PRECISION, MPI_SUM, comm));
    if (!rank)
    {
        printf("Hmatrix memory usage before compression: %g (dense) %g (low rank) %g GB (total)\n", mem[0], mem[1],
               mem[2]);
    }

    // Run performance tests
    std::vector<double> summaryv;
    RunPerf<H2OPUS_HWTYPE_CPU>(hmatrix, dist_h2opus_handle, trunc_eps, nruns, check_compress_err, summaryv);
#ifdef H2OPUS_USE_GPU
    RunPerf<H2OPUS_HWTYPE_GPU>(hmatrix, dist_h2opus_handle, trunc_eps, nruns, check_compress_err, summaryv);
#else
    for (int i = 0; i < 9; i++)
        summaryv.push_back(0);
#endif

    if (summary && !rank)
    {
        printf(
            "=============================================================================================== SUMMARY "
            "===============================================================================================\n");
        printf("%d\t%ld\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%1."
               "6g\t%1.6g\n",
               dist_h2opus_handle->num_ranks, n, summaryv[0], summaryv[1], summaryv[2], summaryv[3], summaryv[4],
               summaryv[12], summaryv[13], summaryv[5], summaryv[6], summaryv[14], summaryv[15], summaryv[7],
               summaryv[8], summaryv[16], summaryv[17]);
        printf("======================================================================================================="
               "================================================================================================\n");
    }
    // Clean up
    h2opusDestroyDistributedHandle(dist_h2opus_handle);
    mpiErrchk(MPI_Finalize());

    return 0;
}
#else
int main(int argc, char **argv)
{
    return 0;
}
#endif
