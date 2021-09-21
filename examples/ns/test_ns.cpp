#include <h2opus.h>
#include <h2opus/util/boxentrygen.h>
#include "../common/example_util.h"
#include "../common/example_problem.h"
#include "../common/hmatrix_samplers.h"
#include "fractional_diffusion.h"

#define APPROX_NORM_MAX_SAMPLES 40

template <int hw>
void cgp(BasicHMatrixSampler *A, THMatrix<hw> &M, H2Opus_Real *x, H2Opus_Real *b, int max_it, H2Opus_Real tol, int &it,
         H2Opus_Real &error, h2opusHandle_t h2opus_handle)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

    h2opusComputeStream_t main_stream = h2opus_handle->getMainStream();
    int n = A->getMatrixDim();

    RealVector r, z, p, q;
    r.resize(n);

    fillArray(vec_ptr(r), n, 0, h2opus_handle->getMainStream(), hw);
    z = p = q = r;

    H2Opus_Real bnorm2 = blas_norm2<H2Opus_Real, hw>(main_stream, n, b, 1);
    if (bnorm2 == 0)
        bnorm2 = 1;

    // r = b - A * x
    A->sample(x, vec_ptr(r), 1);
    blas_scal<H2Opus_Real, hw>(main_stream, n, -1, vec_ptr(r), 1);
    blas_axpy<H2Opus_Real, hw>(main_stream, n, 1, b, 1, vec_ptr(r), 1);

    H2Opus_Real rnorm2 = blas_norm2<H2Opus_Real, hw>(main_stream, n, vec_ptr(r), 1);
    error = rnorm2 / bnorm2;
    it = 0;
    printf("CG iteration %5d error %e\n", it, error);

    H2Opus_Real prev_rho = 0;
    while (error >= tol && it < max_it)
    {
        it++;
        // z = M * r
        hgemv(H2Opus_NoTrans, 1, M, vec_ptr(r), n, 0, vec_ptr(z), n, 1, h2opus_handle);
        // copyArray(vec_ptr(r), vec_ptr(z), n, main_stream, hw);

        // rho = (r'*z)
        H2Opus_Real rho = blas_dot_product<H2Opus_Real, hw>(main_stream, n, vec_ptr(r), 1, vec_ptr(z), 1);

        if (it > 1)
        {
            H2Opus_Real beta = rho / prev_rho;
            // p = z + beta * p;
            blas_scal<H2Opus_Real, hw>(main_stream, n, beta, vec_ptr(p), 1);
            blas_axpy<H2Opus_Real, hw>(main_stream, n, 1, vec_ptr(z), 1, vec_ptr(p), 1);
        }
        else
            copyVector(p, z);

        // q = A*p;
        A->sample(vec_ptr(p), vec_ptr(q), 1);

        // alpha = rho / (p'*q);
        H2Opus_Real alpha = rho / blas_dot_product<H2Opus_Real, hw>(main_stream, n, vec_ptr(p), 1, vec_ptr(q), 1);

        // x = x + alpha * p;
        blas_axpy<H2Opus_Real, hw>(main_stream, n, alpha, vec_ptr(p), 1, x, 1);

        // r = r - alpha*q;
        blas_axpy<H2Opus_Real, hw>(main_stream, n, -alpha, vec_ptr(q), 1, vec_ptr(r), 1);

        // error = norm(r) / bnrm2;
        rnorm2 = blas_norm2<H2Opus_Real, hw>(main_stream, n, vec_ptr(r), 1);
        error = rnorm2 / bnorm2;

        printf("CG iteration %5d error %e\n", it, error);

        prev_rho = rho;
    }

    if (it == max_it)
        printf("CGP: NMax iterations reached.\n");
    else
        printf("CG converged in %d iterations with error %e\n", it, error);
}

template <int hw, class NS_Sampler>
inline void newtonSchultz(NS_Sampler &sampler, THMatrix<hw> &X_prev, int max_samples, H2Opus_Real construction_eps_min,
                          H2Opus_Real construction_eps_max, THMatrix<hw> &inverse, h2opusHandle_t h2opus_handle,
                          THMatrix<hw> &zero_hmatrix, int opt_it)
{
    int it_max = 10;
    int it = 0;
    H2Opus_Real convergence_threshold = 1e-2; // construction_eps_max;

    if (opt_it == 0)
        sampler.setPreviousIterate(NULL, 0);
    else
        sampler.setPreviousIterate(&X_prev, 0);

    H2Opus_Real err = 1;
    H2Opus_Real prod_norm = 0;

    // Main Newton Schultz loop
    H2Opus_Real construction_eps = construction_eps_min;

    while (it < it_max && err > convergence_threshold)
    {
        // if(it > 1 && construction_eps > construction_eps_max)
        if ((err < 1e-1 || prod_norm > 1) && construction_eps > construction_eps_max)
        {
            printf("Threshold reached for loose truncation. Swapping to tighter truncation\n");
            construction_eps /= 10;
            if (construction_eps < construction_eps_max)
                construction_eps = construction_eps_max;
        }
        //////////////////////////////////////////////////////////////////////
        // Construct X_curr
        THMatrix<hw> X_curr = zero_hmatrix;

        H2Opus_Real sampler_approx_norm =
            sampler_norm<H2Opus_Real, hw>(&sampler, X_curr.n, APPROX_NORM_MAX_SAMPLES, h2opus_handle);
        H2Opus_Real abs_err = construction_eps * sampler_approx_norm;

        hara(&sampler, X_curr, max_samples, 10, abs_err, 32, h2opus_handle);

        it++;

        //////////////////////////////////////////////////////////////////////
        // Check for covergences
        err = inverse_error<H2Opus_Real, hw>(sampler.getA(), X_curr, APPROX_NORM_MAX_SAMPLES, h2opus_handle);
        H2Opus_Real construction_err =
            sampler_difference<H2Opus_Real, hw>(&sampler, X_curr, APPROX_NORM_MAX_SAMPLES, h2opus_handle) /
            sampler_approx_norm;

        prod_norm = product_norm<H2Opus_Real, hw>(sampler.getA(), X_curr, 10, h2opus_handle);

        printf("Iteration %d: ||X_curr * A|| = %e\n", it, prod_norm);
        printf("Iteration %d: ||X_curr * A - I|| = %e\n", it, err);
        printf("Construction error = %e | Sampler norm = %e | Absolute threshold = %e\n", construction_err,
               sampler_approx_norm, abs_err);

        //////////////////////////////////////////////////////////////////////
        // Update iterate
        X_prev = X_curr;
        sampler.setPreviousIterate(&X_prev, it);
    }

    inverse = X_prev;
    if (it == it_max)
        printf("Max iterations reached\n");
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
    int leaf_size = arg_parser.option<int>("m", "leaf_size", "Leaf size in the KD-tree", 64);
    int cheb_grid_pts = arg_parser.option<int>(
        "k", "cheb_grid_pts", "Number of grid points in each dimension for Chebyshev interpolation (rank = k^d)", 12);
    int max_samples =
        arg_parser.option<int>("s", "max_samples", "Max number of samples to take for each level of the h2opus", 512);

    H2Opus_Real D = arg_parser.option<H2Opus_Real>("d", "grid_D", "Grid is defined over the interval [-D, D]^d", 10);
    H2Opus_Real eta =
        arg_parser.option<H2Opus_Real>("e", "eta", "Admissibility parameter eta for the original hmatrix", 0.8);
    H2Opus_Real construct_eta = arg_parser.option<H2Opus_Real>(
        "ce", "construct_eta", "Admissibility parameter eta for the constructed preconditioner", 0.8);
    H2Opus_Real alpha =
        arg_parser.option<H2Opus_Real>("a", "alpha", "The order alpha of the fractional Laplacian operator", 1.5);

    H2Opus_Real trunc_eps = arg_parser.option<H2Opus_Real>(
        "te", "trunc_eps", "Relative truncation error threshold for the original hmatrix", 1e-6);
    H2Opus_Real ns_it_eps = arg_parser.option<H2Opus_Real>(
        "ne", "ns_it_eps", "Relative truncation error threshold for the NS iterates", 1e-6);
    H2Opus_Real sigma = arg_parser.option<H2Opus_Real>("si", "sigma", "Scaling factor sigma for (A + sigma * I)", 1e-4);

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
    size_t n = grid_x * grid_y;
    printf("N = %d\n", (int)n);
    // Create point cloud
    int dim = (grid_y == 1 ? 1 : 2);
    PointCloud<H2Opus_Real> pt_cloud(dim, n);
    if (dim == 1)
        generate1DGrid<H2Opus_Real>(pt_cloud, grid_x, -D, D);
    else
        generate2DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, -D, D, -D, D);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Matrix construction
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Setup hmatrix construction parameters:
    // Create a functor that can generate the matrix entries from two points
    H2Opus_Real h = (2 * D) / (grid_x - 1);
    FractionalDiffusionFunction<H2Opus_Real> fd_gen(h, 2 * h, alpha, dim);

    // Create an entry gen struct from the functor. Currently only supports chebyshev interpolation on the CPU
    BoxEntryGen<H2Opus_Real, H2OPUS_HWTYPE_CPU, FractionalDiffusionFunction<H2Opus_Real>> entry_gen(fd_gen);

    // Create the admissibility condition using the eta parameter
    // Decreasing eta refines the matrix tree and increasing it coarsens the tree
    H2OpusBoxCenterAdmissibility admissibility(eta), admissibility_construct(construct_eta);

    // Build the hmatrix. Currently only symmetric matrices are fully supported
    HMatrix hmatrix(n, true), zero_hmatrix(n, true);
    buildHMatrix(hmatrix, &pt_cloud, admissibility, entry_gen, leaf_size, cheb_grid_pts);
    buildHMatrixStructure(zero_hmatrix, &pt_cloud, leaf_size, admissibility_construct);
    HMatrix X_prev = zero_hmatrix, approx_inverse = zero_hmatrix;

    if (output_eps)
        outputEps(hmatrix, "structure.eps");

    // Create h2opus handle
    h2opusHandle_t h2opus_handle;
    h2opusCreateHandle(&h2opus_handle);

    // Compress to a relative threshold, so we need an approximate norm of the matrix
    H2Opus_Real approx_norm =
        hmatrix_norm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(hmatrix, APPROX_NORM_MAX_SAMPLES, h2opus_handle);
    printf("Approximate norm of the original FD matrix = %e\n", approx_norm);
    H2Opus_Real abs_trunc_tol = trunc_eps * approx_norm;

    // Compress the FD matrix - this reduces memory consumption and speeds up operations
    horthog(hmatrix, h2opus_handle);
    hcompress(hmatrix, abs_trunc_tol, h2opus_handle);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Preconditioner construction
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create a sampler for the FD matrix. Since we want to produce an approximate inverse, improving the condition
    // of the matrix by adding a small multiple of the identity will help NS converge better
    // A smaller value produces a better preconditioner, but has an increased chance of failing to converge
    // as the condition of the matrix worsens
    SimpleHMatrixSampler<H2OPUS_HWTYPE_CPU> cpu_hmatrix_sampler(&hmatrix, h2opus_handle);
    cpu_hmatrix_sampler.setScaleA(-1);
    cpu_hmatrix_sampler.setIdentityScale(sigma);

    HighOrderInversionSampler<H2OPUS_HWTYPE_CPU> cpu_ns_sampler(&cpu_hmatrix_sampler, 16, h2opus_handle);

    // Use an adaptive threshold for the construction of the NS iterates, starting at
    // eps_min and tightening to eps_max as we get closer to convergence
    H2Opus_Real eps_min = ns_it_eps * 10, eps_max = ns_it_eps;

    newtonSchultz(cpu_ns_sampler, X_prev, max_samples, eps_min, eps_max, approx_inverse, h2opus_handle, zero_hmatrix,
                  0);

    thrust::host_vector<H2Opus_Real> x0(n, 0), b(n, 1);

    // Remove from the sampler the multiple of the identity that we added to improve conditioning
    cpu_hmatrix_sampler.setIdentityScale(0);

    int it;
    H2Opus_Real error;

    // Solve Ax = b using preconditioned CG
    cgp<H2OPUS_HWTYPE_CPU>(&cpu_hmatrix_sampler, approx_inverse, vec_ptr(x0), vec_ptr(b), 300, 1e-9, it, error,
                           h2opus_handle);

#ifdef H2OPUS_USE_GPU
    HMatrix_GPU gpu_hmatrix = hmatrix;
    HMatrix_GPU gpu_zero_hmatrix = zero_hmatrix;
    HMatrix_GPU gpu_X_prev = gpu_zero_hmatrix, gpu_approx_inverse = gpu_zero_hmatrix;

    SimpleHMatrixSampler<H2OPUS_HWTYPE_GPU> gpu_hmatrix_sampler(&gpu_hmatrix, h2opus_handle);
    gpu_hmatrix_sampler.setScaleA(-1);
    gpu_hmatrix_sampler.setIdentityScale(sigma);

    HighOrderInversionSampler<H2OPUS_HWTYPE_GPU> gpu_ns_sampler(&gpu_hmatrix_sampler, 16, h2opus_handle);

    newtonSchultz(gpu_ns_sampler, gpu_X_prev, max_samples, eps_min, eps_max, gpu_approx_inverse, h2opus_handle,
                  gpu_zero_hmatrix, 0);

    thrust::device_vector<H2Opus_Real> gpu_x0 = thrust::host_vector<H2Opus_Real>(n, 0);
    thrust::device_vector<H2Opus_Real> gpu_b = thrust::host_vector<H2Opus_Real>(n, 1);

    // Remove from the sampler the multiple of the identity that we added to improve conditioning
    gpu_hmatrix_sampler.setIdentityScale(0);

    // Solve Ax = b using preconditioned CG
    cgp<H2OPUS_HWTYPE_GPU>(&gpu_hmatrix_sampler, gpu_approx_inverse, vec_ptr(gpu_x0), vec_ptr(gpu_b), 300, 1e-9, it,
                           error, h2opus_handle);
#endif

    return 0;
}
