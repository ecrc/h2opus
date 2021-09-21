#include <h2opusconf.h>
#if defined(H2OPUS_USE_MPI)
#include <h2opus.h>

#include <h2opus/util/boxentrygen.h>
#include <h2opus/distributed/distributed_h2opus_handle.h>
#include <h2opus/distributed/distributed_geometric_construction.h>
#include <h2opus/distributed/distributed_hgemv.h>
#include <h2opus/distributed/distributed_horthog.h>

#include "../common/example_problem.h"
#include "../common/example_util.h"
#include <stdio.h>

H2Opus_Real dist_vec_diff(H2Opus_Real *x1, H2Opus_Real *x2, int n, MPI_Comm comm)
{
    H2Opus_Real norm_x = 0, diff = 0;
    for (int i = 0; i < n; i++)
    {
        H2Opus_Real entry_diff = x1[i] - x2[i];
        diff += entry_diff * entry_diff;
        norm_x += x1[i] * x1[i];
    }

    H2Opus_Real total_norm, total_diff;
    mpiErrchk(MPI_Allreduce(&norm_x, &total_norm, 1, H2OPUS_MPI_REAL, MPI_SUM, comm));
    mpiErrchk(MPI_Allreduce(&diff, &total_diff, 1, H2OPUS_MPI_REAL, MPI_SUM, comm));

    return sqrt(total_diff / total_norm);
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
    H2Opus_Real eta = arg_parser.option<H2Opus_Real>("e", "eta", "Admissibility parameter eta", DEFAULT_ETA);
    bool check_orthog_err = arg_parser.flag("c", "check_orthog_err", "Check the orthogonalization error", false);
    int nruns = arg_parser.option<int>("n", "nruns", "Number of runs to perform", 10);
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
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Setting up the comm and distributed handle
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    distributedH2OpusHandle_t dist_h2opus_handle;
    h2opusCreateDistributedHandle(&dist_h2opus_handle, true);
    /* parallel computations are performed on a power-of-two size communicator
       This example tests it explicitly. In practice, there is no need to
       invoke the distributed API from active processes only */
    if (dist_h2opus_handle->active)
    {
        int proc_rank = dist_h2opus_handle->rank;
        MPI_Comm comm = dist_h2opus_handle->comm;
        if (!proc_rank)
            printf("Using threaded MPI for CPU runs? %d\n", dist_h2opus_handle->getUseThreads<H2OPUS_HWTYPE_CPU>());
#ifdef H2OPUS_USE_GPU
        if (!proc_rank)
            printf("Using threaded MPI for GPU runs? %d\n", dist_h2opus_handle->getUseThreads<H2OPUS_HWTYPE_GPU>());
#endif

        /////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Geometry generation
        /////////////////////////////////////////////////////////////////////////////////////////////////////////
        size_t n = grid_x * grid_y * grid_z;
        if (!proc_rank)
            printf("N = %d\n", (int)n);
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

        double run_time;

        MPI_Barrier(comm);
        run_time = 0;
        run_time -= MPI_Wtime();
        DistributedHMatrix dist_hmatrix(n);
        buildDistributedHMatrix(dist_hmatrix, &pt_cloud, admissibility, entry_gen, leaf_size, cheb_grid_pts,
                                dist_h2opus_handle);
        MPI_Barrier(comm);
        run_time += MPI_Wtime();
        if (!proc_rank)
            printf("Hmatrix constructed in %fs\n", run_time);

        // H2OPUS parallel distribution: grab local sizes
        int dist_ld = dist_hmatrix.basis_tree.basis_branch.index_map.size();
        thrust::host_vector<H2Opus_Real> dist_y_orthog(dist_ld, 0), dist_y(dist_ld, 0), dist_x(dist_ld);
        randomData(vec_ptr(dist_x), dist_x.size(), dist_h2opus_handle->rank + 1);

#ifdef H2OPUS_USE_GPU
        thrust::device_vector<H2Opus_Real> dist_y_gpu = dist_y, dist_x_gpu = dist_x;
#endif
        distributed_hgemv(1, dist_hmatrix, vec_ptr(dist_x), dist_ld, 0, vec_ptr(dist_y), dist_ld, 1,
                          dist_h2opus_handle);

        double timer_counter_cpu = 0;
#ifdef H2OPUS_USE_GPU
        double timer_counter_gpu = 0;
#endif

        for (int run = 0; run < nruns; run++)
        {
            DistributedHMatrix dist_orthog_hmatrix = dist_hmatrix;

            MPI_Barrier(comm);
            run_time = 0;
            run_time -= MPI_Wtime();
            distributed_horthog(dist_orthog_hmatrix, dist_h2opus_handle);
            MPI_Barrier(comm);
            run_time += MPI_Wtime();
            if (!proc_rank)
                printf("Horthog Run %d: CPU time %fs\n", run, run_time);
            if (run)
                timer_counter_cpu += run_time;

            if (check_orthog_err)
            {
                distributed_hgemv(1, dist_orthog_hmatrix, vec_ptr(dist_x), dist_ld, 0, vec_ptr(dist_y_orthog), dist_ld,
                                  1, dist_h2opus_handle);

                H2Opus_Real horthog_diff = dist_vec_diff(vec_ptr(dist_y), vec_ptr(dist_y_orthog), dist_ld, comm);
                if (!proc_rank)
                    printf("  CPU horthog error = %e\n", horthog_diff);
            }
#ifdef H2OPUS_USE_GPU
            PerformanceCounter::clearCounters();

            DistributedHMatrix_GPU dist_hmatrix_gpu = dist_hmatrix;

            cudaDeviceSynchronize();
            MPI_Barrier(comm);
            run_time = 0;
            run_time -= MPI_Wtime();
            distributed_horthog(dist_hmatrix_gpu, dist_h2opus_handle);
            cudaDeviceSynchronize();
            MPI_Barrier(comm);
            run_time += MPI_Wtime();
            if (!proc_rank)
                printf("Horthog Run %d: GPU time %fs\n", run, run_time);
            if (run)
                timer_counter_gpu += run_time;

            if (check_orthog_err)
            {
                distributed_hgemv(1, dist_hmatrix_gpu, vec_ptr(dist_x_gpu), dist_ld, 0, vec_ptr(dist_y_gpu), dist_ld, 1,
                                  dist_h2opus_handle);
                cudaDeviceSynchronize();

                thrust::host_vector<H2Opus_Real> gpu_result = dist_y_gpu;
                H2Opus_Real horthog_diff = dist_vec_diff(vec_ptr(dist_y), vec_ptr(gpu_result), dist_ld, comm);
                if (!proc_rank)
                    printf("  GPU horthog error = %e\n", horthog_diff);
            }
#endif
        }

        if (!proc_rank && nruns > 1)
        {
            double avg_time = timer_counter_cpu / (nruns - 1);
            printf("Average CPU runtime = %.3f ms\n", 1e3 * avg_time);
        }

#ifdef H2OPUS_USE_GPU
        if (!proc_rank && nruns > 1)
        {
            double avg_time = timer_counter_gpu / (nruns - 1);
            printf("Average GPU runtime = %.3f ms\n", 1e3 * avg_time);
        }
#endif
    }
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
