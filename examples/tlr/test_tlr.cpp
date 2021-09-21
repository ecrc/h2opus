#include <h2opus.h>
#include <h2opus/core/tlr/tlr_struct.h>
#include <h2opus/core/tlr/tlr_construct.h>
#include <h2opus/core/tlr/tlr_potrf.h>
#include <h2opus/core/tlr/tlr_sytrf.h>
#include <h2opus/core/tlr/tlr_gemv.h>

#include "../common/example_util.h"
#include "tlr_example.h"
#include "tlr_example_util.h"

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
    int block_size = arg_parser.option<int>("m", "block_size", "Block size of the TLR_Matrix", 512);
    int ara_bs = arg_parser.option<int>("abs", "ara_bsize", "Block size used in the ARA of each tile", 32);
    int ndpb = arg_parser.option<int>("ndpb", "n_dense_buffers", "Number of dense parallel buffers", 20);
    H2Opus_Real nspb = arg_parser.option<H2Opus_Real>(
        "nspb", "n_sampling_buffers", "Number of sampling parallel buffers (factor, must be >= 1)", 1.5);

    H2Opus_Real s = arg_parser.option<H2Opus_Real>("s", "identity_scale",
                                                   "Scale of the identity matrix added to the TLR matrix", 0);
    H2Opus_Real eps = arg_parser.option<H2Opus_Real>("e", "eps", "Absolute truncation tolerance", 1e-6);
    H2Opus_Real chol_eps =
        arg_parser.option<H2Opus_Real>("ce", "chol_eps", "Absolute truncation tolerance for the TLR Cholesky", 1e-6);
    H2Opus_Real sc_eps = arg_parser.option<H2Opus_Real>("sce", "sc_eps", "Schur Comp Eps", 0);

    bool printranks = arg_parser.flag("printranks", "pr", "Print ranks in file", false);
    bool printstats = arg_parser.flag("printstats", "ps", "Print statistics", false);
    bool print_help = arg_parser.flag("h", "help", "This message", false);

    if (!arg_parser.valid() || print_help)
    {
        arg_parser.printUsage();
        exit(0);
    }

    printf("Tile Size = %d. ARA Block size = %d\n", block_size, ara_bs);
    printf("Construction threshold = %e. Cholesky threshold = %e. Diagonal shift s = %e\n", eps, chol_eps, s);

    // Create h2opus handle
    h2opusHandle_t h2opus_handle;
    h2opusCreateHandle(&h2opus_handle);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Geometry and generation
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

    // generateRandomSphere<H2Opus_Real>(pt_cloud, n, 2, 1234);
    // generate3DRandomPointCloud<H2Opus_Real>(pt_cloud, n, 0, 1);

    // Construct node reordering
    TH2OpusKDTree<H2Opus_Real, H2OPUS_HWTYPE_CPU> kdtree(&pt_cloud, block_size);
    kdtree.buildKDtreeMedianUniformSplit();
    int *index_map = kdtree.getIndexMap();

    // Functor to sample matrix entries in original ordering
    TLRMatGen<H2Opus_Real> mat_gen(pt_cloud, index_map);

    // Create the TLR matrix object
    h2opusComputeStream_t stream = h2opus_handle->getMainStream();
    TTLR_Matrix<H2Opus_Real, H2OPUS_HWTYPE_CPU> tlr_matrix(n, H2OpusTLR_Symmetric, block_size, index_map, H2OpusTLRTile,
                                                           stream);
    tlr_matrix.max_rank = block_size / 4;

    // Construct the TLR matrix
    Timer<H2OPUS_HWTYPE_CPU> timer;
    timer.init();

    timer.start();
    construct_tlr_matrix<H2Opus_Real, TLRMatGen<H2Opus_Real>, H2OPUS_HWTYPE_CPU>(tlr_matrix, mat_gen, eps, stream);
    double cpu_construct_time = timer.stop();
    printf("CPU TLR matrix constructed in %.3fs\n", cpu_construct_time);

    H2Opus_Real norm = tlr_norm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(tlr_matrix, h2opus_handle);
    printf("CPU TLR Norm = %e\n", norm);

    int pn = tlr_matrix.getPaddedDim();

    // std::vector<H2Opus_Real> M(pn * pn, 0);
    // expandTLRMatrix(tlr_matrix, vec_ptr(M));
    // save_matrix(vec_ptr(M), pn * pn, "M.bin");
    // printDenseMatrix(vec_ptr(M), pn, pn, pn, 8, "A");

    // save original matrix (potrf/sytrf overwrite the matrix)
    TTLR_Matrix<H2Opus_Real, H2OPUS_HWTYPE_CPU> original_matrix(tlr_matrix, stream);

#ifdef H2OPUS_USE_GPU
    // Copy to the GPU before they are altered by the factorization
    TTLR_Matrix<H2Opus_Real, H2OPUS_HWTYPE_GPU> gpu_tlr_matrix(tlr_matrix, stream);
#endif

    if (printranks)
        print_ranks(tlr_matrix, "results/original_ranks", grid_x, grid_y, grid_z);
    if (printstats)
        print_statistics(tlr_matrix);

    // std::vector<H2Opus_Real> M(pn * pn, 0);
    // expandTLRMatrix(tlr_matrix, vec_ptr(M));
    // printDenseMatrix(vec_ptr(M), pn, pn, pn, 4, "M");
    //
    // std::vector<H2Opus_Real> M_d(n * n, 0);
    // gen_dense_matrix(vec_ptr(M_d), n, mat_gen);
    // printDenseMatrix(vec_ptr(M_d), n, n, n, 4, "M_d");

    std::vector<H2Opus_Real> x(pn, 0), px(pn, 0), y1(pn, 0), y(pn, 0), py(pn, 0), y_A(pn, 0);
    std::vector<H2Opus_Real> D(pn, 1);
    randomData(vec_ptr(x), n);

    // set y_A = A * x
    double total_gemv_time = 0;
    const int nruns = 10;
    for (int i = 0; i < nruns; i++)
    {
        timer.start();
        tlr_gemv<H2Opus_Real, H2OPUS_HWTYPE_CPU>(H2Opus_Trans, 1, tlr_matrix, vec_ptr(x), pn, 0, vec_ptr(y_A), pn, 1, 2,
                                                 h2opus_handle);
        double cpu_gemv_time = timer.stop();
        if (i != 0)
            total_gemv_time += cpu_gemv_time;
        // printf("CPU GEMV time %.3fs\n", cpu_gemv_time);
    }
    printf("CPU GEMV Avg time %.3fs\n", total_gemv_time / (nruns - 1));

    std::vector<int> piv(tlr_matrix.n_block, 0);
    generateSequence(vec_ptr(piv), tlr_matrix.n_block, 0, stream, H2OPUS_HWTYPE_CPU);

    // std::vector<H2Opus_Real> M(n * n, 0);
    // expandTLRMatrix(tlr_matrix, vec_ptr(M));
    // printDenseMatrix(vec_ptr(M), n, n, n, 8, "M");

    // Factor A = L * L'
    TLR_Potrf_Config<H2Opus_Real, H2OPUS_HWTYPE_CPU> config(tlr_matrix);
    config.tolerance(chol_eps);
    config.schur_tolerance(sc_eps);
    config.samplingBlockSize(ara_bs);
    config.densePBuffers(ndpb);
    config.samplingPBuffers(nspb * tlr_matrix.n_block);

    timer.start();
    tlr_potrf<H2Opus_Real, H2OPUS_HWTYPE_CPU>(
        // tlr_matrix, config, vec_ptr(piv), h2opus_handle
        tlr_matrix, config, h2opus_handle);
    double cpu_potrf_time = timer.stop();
    printf("CPU TLR matrix factored in %.3fs\n", cpu_potrf_time);

    // tlr_sytrf<H2Opus_Real, H2OPUS_HWTYPE_CPU>(
    //     tlr_matrix, vec_ptr(D), chol_eps, 20, 1.5 * tlr_matrix.n_block, ara_bs, h2opus_handle
    // );

    std::vector<int> piv_full(pn, 0);
    for (int i = 0; i < tlr_matrix.n_block; i++)
        for (int j = 0; j < tlr_matrix.block_size; j++)
            piv_full[i * tlr_matrix.block_size + j] = piv[i] * tlr_matrix.block_size + j;

    H2Opus_Real chol_err = tlr_chol_error_norm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(original_matrix, tlr_matrix,
                                                                               vec_ptr(piv_full), h2opus_handle);
    printf("Cholesky error norm = %e (abs = %e)\n", chol_err / norm, chol_err);

    H2Opus_Real inv_norm =
        tlr_inverse_norm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(tlr_matrix, vec_ptr(piv_full), h2opus_handle);
    printf("TLR Inverse Norm = %e\n", inv_norm);
    printf("Approximate condition number = %e\n", inv_norm * norm);

    H2Opus_Real ldl_err =
        tlr_ldl_error_norm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(original_matrix, tlr_matrix, vec_ptr(D), h2opus_handle);
    printf("LDL error norm = %e (abs = %e)\n", ldl_err / norm, ldl_err);

    double total_trsm_time = 0;
    for (int i = 0; i < nruns; i++)
    {
        px = x;
        timer.start();
        tlr_potrs<H2Opus_Real, H2OPUS_HWTYPE_CPU>(tlr_matrix, 1, vec_ptr(px), pn, h2opus_handle);
        double cpu_trsm_time = timer.stop();
        if (i != 0)
            total_trsm_time += cpu_trsm_time;
        // printf("CPU TRSM time %.3fs\n", cpu_trsm_time);
    }
    printf("CPU TRSM Avg time %.3fs\n", total_trsm_time / (nruns - 1));

    if (printranks)
        print_ranks(tlr_matrix, "results/factored_ranks", grid_x, grid_y, grid_z);
    if (printstats)
        print_statistics(tlr_matrix);

    // Get y = L * L' * x = L * y1
    permute_vectors(vec_ptr(x), vec_ptr(px), pn, 1, vec_ptr(piv_full), 0, H2OPUS_HWTYPE_CPU, stream);

    tlr_gemv<H2Opus_Real, H2OPUS_HWTYPE_CPU>(H2Opus_Trans, 1, tlr_matrix, vec_ptr(px), pn, 0, vec_ptr(y1), pn, 1, 2,
                                             h2opus_handle);

    for (int i = 0; i < pn; i++)
        if (D[i] <= 0)
            printf("Negative value D[%d] = %e\n", i, D[i]);

    for (int i = 0; i < pn; i++)
        y1[i] *= D[i];

    tlr_gemv<H2Opus_Real, H2OPUS_HWTYPE_CPU>(H2Opus_NoTrans, 1, tlr_matrix, vec_ptr(y1), pn, 0, vec_ptr(y), pn, 1, 2,
                                             h2opus_handle);

    permute_vectors(vec_ptr(y), vec_ptr(py), pn, 1, vec_ptr(piv_full), 1, H2OPUS_HWTYPE_CPU, stream);

    printf("CPU Vec diff = %e\n", vec_diff(vec_ptr(py), vec_ptr(y_A), n));

#ifdef H2OPUS_USE_GPU
    H2Opus_Real gpunorm = tlr_norm<H2Opus_Real, H2OPUS_HWTYPE_GPU>(gpu_tlr_matrix, h2opus_handle);
    printf("GPU TLR Norm = %e\n", gpunorm);

    thrust::device_vector<H2Opus_Real> d_x = x, d_px, d_y1(pn, 0), d_y(pn, 0), d_y_A(pn, 0);

    Timer<H2OPUS_HWTYPE_GPU> gpu_timer;
    gpu_timer.init();

    // set y_A = A * x
    total_gemv_time = 0;
    for (int i = 0; i < nruns; i++)
    {
        gpu_timer.start();
        tlr_gemv<H2Opus_Real, H2OPUS_HWTYPE_GPU>(H2Opus_Trans, 1, gpu_tlr_matrix, vec_ptr(d_x), pn, 0, vec_ptr(d_y_A),
                                                 pn, 1, 2, h2opus_handle);
        double gpu_gemv_time = gpu_timer.stop();
        if (i != 0)
            total_gemv_time += gpu_gemv_time;
        // printf("GPU GEMV time %.3fs\n", gpu_gemv_time);
    }
    printf("GPU GEMV Avg time %.3fs\n", total_gemv_time / (nruns - 1));

    // Factor A = L * L'

    TLR_Potrf_Config<H2Opus_Real, H2OPUS_HWTYPE_GPU> gpu_config(gpu_tlr_matrix);
    gpu_config.tolerance(chol_eps).schur_tolerance(sc_eps).samplingBlockSize(ara_bs);

    timer.start();
    tlr_potrf<H2Opus_Real, H2OPUS_HWTYPE_GPU>(gpu_tlr_matrix, gpu_config, h2opus_handle);
    double gpu_potrf_time = timer.stop();
    printf("GPU TLR matrix factored in %.3fs\n", gpu_potrf_time);

    total_trsm_time = 0;
    for (int i = 0; i < nruns; i++)
    {
        d_px = x;
        gpu_timer.start();
        tlr_potrs<H2Opus_Real, H2OPUS_HWTYPE_GPU>(gpu_tlr_matrix, 1, vec_ptr(d_px), pn, h2opus_handle);
        double gpu_trsm_time = gpu_timer.stop();
        if (i != 0)
            total_trsm_time += gpu_trsm_time;
        // printf("GPU TRSM time %.3fs\n", gpu_trsm_time);
    }
    printf("GPU TRSM Avg time %.3fs\n", total_trsm_time / (nruns - 1));

    // Get y = L * L' * x = L * y1
    tlr_gemv<H2Opus_Real, H2OPUS_HWTYPE_GPU>(H2Opus_Trans, 1, gpu_tlr_matrix, vec_ptr(d_x), pn, 0, vec_ptr(d_y1), pn, 1,
                                             2, h2opus_handle);
    tlr_gemv<H2Opus_Real, H2OPUS_HWTYPE_GPU>(H2Opus_NoTrans, 1, gpu_tlr_matrix, vec_ptr(d_y1), pn, 0, vec_ptr(d_y), pn,
                                             1, 2, h2opus_handle);
    thrust::host_vector<H2Opus_Real> gpu_result = d_y_A, gpu_result2 = d_y;
    printf("GPU Vec diff = %e\n", vec_diff(vec_ptr(gpu_result), vec_ptr(gpu_result2), pn));
#endif

    h2opusDestroyHandle(h2opus_handle);

    return 0;
}
