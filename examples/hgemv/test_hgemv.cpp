#include <stdio.h>
#include <h2opus.h>

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
    int num_vectors = arg_parser.option<int>("nv", "num_vectors", "Number of vectors the matrix multiplies", 1);
    int nruns = arg_parser.option<int>("n", "nruns", "Number of runs to perform", 10);
    bool output_eps = arg_parser.flag("o", "output_eps", "Output structure of the matrix as an eps file", false);
    bool check_approx_err = arg_parser.flag("c", "check_approx_err", "Check the approximation error", false);
    bool print_results = arg_parser.flag("p", "print_results", "Print input/output vectors to stdout", false);
    bool print_help = arg_parser.flag("h", "help", "This message", false);

    if (!arg_parser.valid() || print_help)
    {
        arg_parser.printUsage();
        return 0;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Geometry and input vector generation
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    size_t n = grid_x * grid_y * grid_z;
    printf("N = %d\n", (int)n);
    // Create point cloud
    int dim = (grid_z == 1 ? 2 : 3);
    PointCloud<H2Opus_Real> pt_cloud(dim, n);
    if (grid_z > 1)
        generate3DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, grid_z, 0, 1, 0, 1, 0, 1);
    else
        generate2DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, 0, 1, 0, 1);

    // Generate random input for the hgemv
    thrust::host_vector<H2Opus_Real> x(n * num_vectors, 1), y(n * num_vectors, 0);
    for (size_t i = 0; i < x.size(); i++)
        x[i] = (H2Opus_Real)rand() / RAND_MAX;

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

    // Functor to generate the matrix entries so we can test the approximation error
    MatGen<H2Opus_Real> mat_gen(func_gen, pt_cloud, &hmatrix.u_basis_tree.index_map[0]);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Hgemv
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create h2opus handle
    h2opusHandle_t h2opus_handle;
    h2opusCreateHandle(&h2opus_handle);

    // Pre-allocate workspace. This can be safely skipped if performance is not a concern
    H2OpusWorkspaceState ws_needed_cpu = hgemv_workspace(hmatrix, H2Opus_NoTrans, num_vectors);
    h2opus_handle->setWorkspaceState(ws_needed_cpu);

    // Performance data
    double total_gops, total_time, total_perf, total_dev;

    HLibProfile::clear();

    // Average over multiple runs
    H2Opus_Real alpha = 1, beta = 1;

    // CPU execution
    for (int i = 0; i < nruns; i++)
    {
        fillArray(vec_ptr(y), n * num_vectors, 0, h2opus_handle->getMainStream(), H2OPUS_HWTYPE_CPU);
        hgemv(H2Opus_NoTrans, alpha, hmatrix, vec_ptr(x), n, beta, vec_ptr(y), n, num_vectors, h2opus_handle);
        if (print_results)
        {
            printf("CPU x\n");
            printThrustVector(x);
            printf("CPU y\n");
            printThrustVector(y);
        }
    }
    HLibProfile::getHgemvPerf(total_gops, total_time, total_perf, total_dev);
    printf("CPU Total execution time: %f s at %f (Gflop/s) (%.3f dev)\n", total_time, total_perf, total_dev);
    HLibProfile::clear();

    // Check the approximation error by testing a small portion p of the dense matrix vector product
    H2Opus_Real p = 0.01;
    if (check_approx_err)
    {
        H2Opus_Real max_approx_err = 0;

        for (int i = 0; i < num_vectors; i++)
        {
            H2Opus_Real *y1 = vec_ptr(y) + i * n;
            H2Opus_Real *x1 = vec_ptr(x) + i * n;

            H2Opus_Real approx_err = getHgemvApproximationError(n, mat_gen, p, y1, x1);
            max_approx_err = std::max(max_approx_err, approx_err);
        }
        printf("CPU Max approx error = %e\n", max_approx_err);
    }

#ifdef H2OPUS_USE_GPU
    // Test hgemv on the GPU and compare with the CPU results
    thrust::device_vector<H2Opus_Real> gpu_x = x, gpu_y;
    gpu_y.resize(n * num_vectors);

    // Copy the hmatrix over to the GPU
    HMatrix_GPU gpu_h = hmatrix;

    // Set the workspace in the handle for host and gpu
    H2OpusWorkspaceState ws_needed_gpu = hgemv_workspace(gpu_h, H2Opus_NoTrans, num_vectors);
    h2opus_handle->setWorkspaceState(ws_needed_gpu);

    // GPU execution
    for (int i = 0; i < nruns; i++)
    {
        fillArray(vec_ptr(gpu_y), n * num_vectors, 0, h2opus_handle->getMainStream(), H2OPUS_HWTYPE_GPU);
        hgemv(H2Opus_NoTrans, alpha, gpu_h, vec_ptr(gpu_x), n, beta, vec_ptr(gpu_y), n, num_vectors, h2opus_handle);
        printf("GPU x\n");
        printThrustVector(gpu_x);
        printf("GPU y\n");
        printThrustVector(gpu_y);
    }
    HLibProfile::getHgemvPerf(total_gops, total_time, total_perf, total_dev);
    printf("GPU Total execution time: %f s at %f (Gflop/s) (%.3f dev)\n", total_time, total_perf, total_dev);
    HLibProfile::clear();

    // Copy gpu results and compare with CPU
    thrust::host_vector<H2Opus_Real> gpu_results = gpu_y;

    H2Opus_Real max_diff = 0, gpu_max_approx_err = 0;

    for (int i = 0; i < num_vectors; i++)
    {
        H2Opus_Real *y1 = vec_ptr(y) + i * n;
        H2Opus_Real *y2 = vec_ptr(gpu_results) + i * n;
        H2Opus_Real *x1 = vec_ptr(x) + i * n;

        max_diff = std::max(max_diff, vec_diff(y1, y2, n));

        if (check_approx_err)
        {
            H2Opus_Real approx_err = getHgemvApproximationError(n, mat_gen, 0.01, y2, x1);
            gpu_max_approx_err = std::max(gpu_max_approx_err, approx_err);
        }
    }
    if (check_approx_err)
        printf("GPU Max approx error = %e\n", gpu_max_approx_err);

    printf("Max CPU-GPU difference = %e\n", max_diff);

#endif

    // Clean up
    h2opusDestroyHandle(h2opus_handle);

    return 0;
}
