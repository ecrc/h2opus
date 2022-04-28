#include <h2opusconf.h>
#if defined(H2OPUS_USE_MPI)
#include <h2opus.h>

#include <h2opus/util/boxentrygen.h>
#include <h2opus/distributed/distributed_h2opus_handle.h>
#include <h2opus/distributed/distributed_geometric_construction.h>
#include <h2opus/distributed/distributed_hgemv.h>

#include "../common/example_problem.h"
#include "../common/example_util.h"
#include <stdio.h>

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
    int max_num_vectors = arg_parser.option<int>("nv", "max_num_vectors", "Number of vectors the matrix multiplies", 1);
    bool check_results = arg_parser.flag("c", "check_results", "Check the approximation error", false);
    bool print_results = arg_parser.flag("p", "print_results", "Print input/output vectors to stdout", false);
#ifdef H2OPUS_USE_GPU
    int nruns = arg_parser.option<int>("n", "nruns", "Number of runs to perform", 10);
#endif
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

    int start_vector = 1;
    if (max_num_vectors < 1)
    {
        max_num_vectors = std::max(1, -max_num_vectors);
        start_vector = max_num_vectors;
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
        int num_proc = dist_h2opus_handle->num_ranks;
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
        // Create point cloud
        PointCloud<H2Opus_Real> pt_cloud;
        if (grid_z > 1)
            generate3DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, grid_z, 0, 1, 0, 1, 0, 1);
        else
            generate2DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, 0, 1, 0, 1);
        int dim = pt_cloud.getDimension();
        size_t n = pt_cloud.getDataSetSize();
        if (!proc_rank)
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

        // H2OPUS parallel distribution: grab sizes and use them to gather the results later
        int index_map_offset = dist_hmatrix.basis_tree.basis_branch.index_map_offset;
        std::vector<int> index_map_offsets(num_proc + 1);
        std::vector<int> vec_chunk_sizes(num_proc);
        mpiErrchk(MPI_Allgather(&index_map_offset, 1, MPI_INT, &index_map_offsets[0], 1, MPI_INT, comm));
        index_map_offsets[num_proc] = n;

        int dist_ld = index_map_offsets[proc_rank + 1] - index_map_offsets[proc_rank];
        for (int i = 0; i < num_proc; i++)
            vec_chunk_sizes[i] = index_map_offsets[i + 1] - index_map_offsets[i];

        thrust::host_vector<H2Opus_Real> dist_y(dist_ld * max_num_vectors, 0), dist_x(dist_ld * max_num_vectors);
        randomData(vec_ptr(dist_x), dist_x.size(), dist_h2opus_handle->rank + 1);

#ifdef H2OPUS_USE_GPU
        DistributedHMatrix_GPU dist_hmatrix_gpu = dist_hmatrix;
        thrust::device_vector<H2Opus_Real> dist_y_gpu = dist_y, dist_x_gpu = dist_x;
#endif

        // Full hmatrix if we're checking results
        HMatrix full_hmatrix(n, true);
        if (!proc_rank && check_results)
        {
            buildHMatrix(full_hmatrix, &pt_cloud, admissibility, entry_gen, leaf_size, cheb_grid_pts);
            outputEps(full_hmatrix, "structure.eps");
        }

        for (int num_vectors = start_vector; num_vectors <= max_num_vectors; num_vectors *= 2)
        {
            distributed_hgemv(1, dist_hmatrix, vec_ptr(dist_x), dist_ld, 0, vec_ptr(dist_y), dist_ld, num_vectors,
                              dist_h2opus_handle);
            if (print_results)
            {
                printf("CPU x\n");
                printThrustVector(dist_x);
                printf("CPU y\n");
                printThrustVector(dist_y);
            }
#ifdef H2OPUS_USE_GPU
            cudaDeviceSynchronize();
            MPI_Barrier(comm);

            double timer_counter = 0;
            double hgemm_ops = 0;
            PerformanceCounter::clearCounters();

            for (int run = 0; run < nruns; run++)
            {
                double run_time = 0;
                run_time -= MPI_Wtime();

                distributed_hgemv(1, dist_hmatrix_gpu, vec_ptr(dist_x_gpu), dist_ld, 0, vec_ptr(dist_y_gpu), dist_ld,
                                  num_vectors, dist_h2opus_handle);
                hgemm_ops = PerformanceCounter::getOpCount(PerformanceCounter::GEMM);
                PerformanceCounter::clearCounters();
                cudaDeviceSynchronize();
                MPI_Barrier(comm);
                run_time += MPI_Wtime();

                printf("Rank %d Vectors %d Run %d: %f GFLOP in %fs at %f GFLOP/s\n", proc_rank, num_vectors, run,
                       hgemm_ops, run_time, hgemm_ops / run_time);
                HLibProfile::clear();

                if (print_results)
                {
                    printf("GPU x\n");
                    printThrustVector(dist_x_gpu);
                    printf("GPU y\n");
                    printThrustVector(dist_y_gpu);
                }

                if (run != 0)
                    timer_counter += run_time;
            }

            double total_ops = 0;
            MPI_Reduce(&hgemm_ops, &total_ops, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, comm);

            if (!proc_rank && nruns > 1)
            {
                double avg_time = timer_counter / (nruns - 1);
                printf("Vectors = %d: Average runtime = %.3f ms for %.3f GFLOP at %.3f GFLOP/s\n", num_vectors,
                       1e3 * avg_time, total_ops, total_ops / avg_time);
            }

#endif

            if (check_results)
            {
                // check GPU for GPU runs
#ifdef H2OPUS_USE_GPU
                dist_y = dist_y_gpu;
#endif
                // Assemble the full X and Y
                thrust::host_vector<H2Opus_Real> full_y(n * num_vectors, 0), full_x(n * num_vectors, 0);

                for (int i = 0; i < num_vectors; i++)
                {
                    H2Opus_Real *local_x_col = vec_ptr(dist_x) + i * dist_ld;
                    H2Opus_Real *local_y_col = vec_ptr(dist_y) + i * dist_ld;
                    H2Opus_Real *full_x_col = vec_ptr(full_x) + i * n;
                    H2Opus_Real *full_y_col = vec_ptr(full_y) + i * n;

                    mpiErrchk(MPI_Gatherv(local_y_col, dist_ld, H2OPUS_MPI_REAL, full_y_col, vec_ptr(vec_chunk_sizes),
                                          vec_ptr(index_map_offsets), H2OPUS_MPI_REAL, 0, comm));

                    mpiErrchk(MPI_Gatherv(local_x_col, dist_ld, H2OPUS_MPI_REAL, full_x_col, vec_ptr(vec_chunk_sizes),
                                          vec_ptr(index_map_offsets), H2OPUS_MPI_REAL, 0, comm));
                }

                if (!proc_rank)
                {
                    if (print_results)
                    {
                        printf("FULL CPU x\n");
                        printThrustVector(full_x);
                        printf("FULL CPU y\n");
                        printThrustVector(full_y);
                    }
                    h2opusHandle_t h2opus_handle = dist_h2opus_handle->handle;
                    thrust::host_vector<H2Opus_Real> y(n * num_vectors, 0);
                    hgemv(H2Opus_NoTrans, 1, full_hmatrix, vec_ptr(full_x), n, 0, vec_ptr(y), n, num_vectors,
                          h2opus_handle);
                    printf("Vec diff = %e\n", vec_diff(vec_ptr(full_y), vec_ptr(y), n * num_vectors));
                }
            }
        }
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
