#include <h2opus.h>
#include <h2opus_ng.h>
#include <h2opus/util/boxentrygen.h>
#include "../common/example_util.h"
#include "../common/example_problem.h"
#include "../common/hmatrix_samplers.h"
//#include "fractional_diffusion.h"
#include "spatial_statistics.h"
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>

#define APPROX_NORM_MAX_SAMPLES 40

typedef struct {
    double *x; ///< Values in X dimension.
    double *y; ///< Values in Y dimension.
    double *z; ///< Values in Z dimension.
} location;

static double calculateDistance( location* l1, location* l2, int l1_index,
        int l2_index, int distance_metric, int z_flag) {

    double z1, z2;
    double x1=l1->x[l1_index];
    double y1=l1->y[l1_index];
    double x2=l2->x[l2_index];
    double y2=l2->y[l2_index];
    if(l1->z == NULL || l2->z == NULL || z_flag == 1)
    {
        if(distance_metric == 1)
            std::cout<<"Great Circle (GC) distance is not supported here"<<"\n";
        return  sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
    }
    else
    {
        if(distance_metric == 1)
        {
            printf("Great Circle (GC) distance is only valid for 2d\n");
            exit(0);
        }
        z1 = l1->z[l1_index];
        z2 = l2->z[l2_index];
        return  sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2) +  pow((z2 - z1), 2));
    }
}

double uniform_distribution(double rangeLow, double rangeHigh)
    //! Generate uniform distribution between rangeLow , rangeHigh
{
    // unsigned int *seed = &exageostat_seed;
    double myRand = (double) rand() / (double) (1.0 + RAND_MAX);
    double range = rangeHigh - rangeLow;
    double myRand_scaled = (myRand * range) + rangeLow;
    return myRand_scaled;
}

static uint32_t Part1By1(uint32_t x)
    // Spread lower bits of input
{
    x &= 0x0000ffff;
    // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x <<  8)) & 0x00ff00ff;
    // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x <<  4)) & 0x0f0f0f0f;
    // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x <<  2)) & 0x33333333;
    // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x <<  1)) & 0x55555555;
    // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return x;
}

static uint32_t EncodeMorton2(uint32_t x, uint32_t y)
    // Encode two inputs into one
{
    return (Part1By1(y) << 1) + Part1By1(x);
}

static uint32_t Compact1By1(uint32_t x)
    // Collect every second bit into lower part of input
{
    x &= 0x55555555;
    // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x = (x ^ (x >>  1)) & 0x33333333;
    // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >>  2)) & 0x0f0f0f0f;
    // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >>  4)) & 0x00ff00ff;
    // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >>  8)) & 0x0000ffff;
    // x = ---- ---- ---- ---- fedc ba98 7654 3210
    return x;
}

static uint32_t DecodeMorton2X(uint32_t code)
    // Decode first input
{
    return Compact1By1(code >> 0);
}

static uint32_t DecodeMorton2Y(uint32_t code)
    // Decode second input
{
    return Compact1By1(code >> 1);
}

static int compare_uint32(const void *a, const void *b)
    //! Compare two uint32_t
{
    uint32_t _a = *(uint32_t *)a;
    uint32_t _b = *(uint32_t *)b;
    if(_a < _b) return -1;
    if(_a == _b) return 0;
    return 1;
}

static void zsort_locations(int n, location * locations)
    //! Sort in Morton order (input points must be in [0;1]x[0;1] square])
{
    // Some sorting, required by spatial statistics code
    int i;
    uint16_t x, y;
    uint32_t z[n];
    // Encode data into vector z
    for(i = 0; i < n; i++)
    {
        x = (uint16_t)(locations->x[i]*(double)UINT16_MAX +.5);
        y = (uint16_t)(locations->y[i]*(double)UINT16_MAX +.5);
        //printf("%f %f -> %u %u\n", points[i], points[i+n], x, y);
        z[i] = EncodeMorton2(x, y);
    }
    // Sort vector z
    qsort(z, n, sizeof(uint32_t), compare_uint32);
    // Decode data from vector z
    for(i = 0; i < n; i++)
    {
        x = DecodeMorton2X(z[i]);
        y = DecodeMorton2Y(z[i]);
        locations->x[i] = (double)x/(double)UINT16_MAX;
        locations->y[i] = (double)y/(double)UINT16_MAX;
        //printf("%lu (%u %u) -> %f %f\n", z[i], x, y, points[i], points[i+n]);
    }
}

location* GenerateXYLoc(int n, int seed)
    //! Generate XY location for exact computation (MORSE)
{
    //initalization
    int i = 0 ,index = 0, j = 0;
    // unsigned int *seed = &exageostat_seed;
    srand(seed);
    location* locations = (location *) malloc( sizeof(location*));
    //Allocate memory
    locations->x        = (double *) malloc(n * sizeof(double));
    locations->y        = (double *) malloc(n * sizeof(double));
    locations->z		= NULL;
    // if(strcmp(locs_file, "") == 0)
    // {

    int sqrtn = ceil(sqrt(n));

    //Check if the input is square number or not
    //if(pow(sqrtn,2) != n)
    //{
    //printf("n=%d, Please use a perfect square number to generate a valid synthetic dataset.....\n\n", n);
    //exit(0);
    //}

    int *grid = (int *) calloc((int)sqrtn, sizeof(int));

    for(i = 0; i < sqrtn; i++)
    {
        grid[i] = i+1;
    }

    for(i = 0; i < sqrtn && index < n; i++)
        for(j = 0; j < sqrtn && index < n; j++){
            locations->x[index] = (grid[i]-0.5+uniform_distribution(-0.4, 0.4))/sqrtn;
            locations->y[index] = (grid[j]-0.5+uniform_distribution(-0.4, 0.4))/sqrtn;
            //printf("%f, %f\n", locations->x[index], locations->y[index]);
            index++;
        }
    free(grid);
    zsort_locations(n, locations);
    return locations;
}

void core_ng_dcmg (double *A, int m, int n,
        int m0, int n0, location  *l1,
        location *l2, double *localtheta, int distance_metric) {

    int i, j;
    int i0 = m0;
    int j0 = n0;
    //double x0, y0, z0;
    double expr = 0.0;
    double con = 0.0;
    double sigma_square = 1;//localtheta[0];// * localtheta[0];

    con = pow(2,(localtheta[1]-1)) * tgamma(localtheta[1]);
    con = 1.0/con;
    con = sigma_square * con;

    for (i = 0; i < m; i++) {
        j0 = n0;
        for (j = 0; j < n; j++) {
	    expr = 4 * sqrt(2*localtheta[1]) * (calculateDistance(l1, l2, i0, j0, distance_metric, 0)/localtheta[0]);
            if(expr == 0)
                A[i + j * m] = sigma_square /*+ 1e-4*/;
            else
                A[i + j * m] = con*pow(expr, localtheta[1])
                    * gsl_sf_bessel_Knu(localtheta[1], expr); // Matern Function
            j0++;
        }
        i0++;
    }
}

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
    double start, end;

    location* l;
    double phi, nu;
    int seed;
    phi = 0.5;
    nu = 1;
    seed = 1;

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

    l =  GenerateXYLoc( n,  seed);
    zsort_locations(n, l);

    // Create point cloud
    int dim = (grid_y == 1 ? 1 : 2);

    PointCloud<H2Opus_Real> pt_cloud(dim, n);
    
    int i;
    for(i=0;i<n;i++)
    {	
	    pt_cloud.pts[0][i] = l->x[i];
	    pt_cloud.pts[1][i] = l->y[i];
    }

    //if (dim == 1)
    //    generate1DGrid<H2Opus_Real>(pt_cloud, grid_x, -D, D);
    //else
    //    generate2DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, -D, D, -D, D);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Matrix construction
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Setup hmatrix construction parameters:
    // Create a functor that can generate the matrix entries from two points

    start = omp_get_wtime();
    H2Opus_Real h = (2 * D) / (grid_x - 1);
    Spatial_Statistics<H2Opus_Real> fd_gen(phi, nu, dim);

    // Create an entry gen struct from the functor. Currently only supports chebyshev interpolation on the CPU
    BoxEntryGen<H2Opus_Real, H2OPUS_HWTYPE_CPU, Spatial_Statistics<H2Opus_Real>> entry_gen(fd_gen);

    // Create the admissibility condition using the eta parameter
    // Decreasing eta refines the matrix tree and increasing it coarsens the tree
    H2OpusBoxCenterAdmissibility admissibility(eta), admissibility_construct(construct_eta);

    // Build the hmatrix. Currently only symmetric matrices are fully supported
    HMatrix hmatrix(n, true), zero_hmatrix(n, true);
    buildHMatrix(hmatrix, &pt_cloud, admissibility, entry_gen, leaf_size, cheb_grid_pts);
    buildHMatrixStructure(zero_hmatrix, &pt_cloud, leaf_size, admissibility_construct);
    HMatrix X_prev = zero_hmatrix, approx_inverse = zero_hmatrix;
    end = omp_get_wtime();
    std::cout << "Time to create hmatrix: " << (end-start) << std::endl;
    exit(0);

    start = omp_get_wtime();
    if (output_eps)
        outputEps(hmatrix, "structure.eps");

    // Create h2opus handle
    h2opusHandle_t h2opus_handle;
    h2opusCreateHandle(&h2opus_handle);
    end = omp_get_wtime();
    std::cout << "Time to create handle: " << (end-start) << std::endl;

    // Compress to a relative threshold, so we need an approximate norm of the matrix
    start = omp_get_wtime();
    H2Opus_Real approx_norm =
        hmatrix_norm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(hmatrix, APPROX_NORM_MAX_SAMPLES, h2opus_handle);
    end = omp_get_wtime();
    printf("Approximate norm of the original FD matrix = %e\n", approx_norm);
    H2Opus_Real abs_trunc_tol = trunc_eps * approx_norm;
    std::cout << "Time to compute approximate norm of the matrix: " << (end-start) << std::endl;

    // Compress the FD matrix - this reduces memory consumption and speeds up operations
    start = omp_get_wtime();
    horthog(hmatrix, h2opus_handle);
    end = omp_get_wtime();
    std::cout << "Time for horthog function: " << (end-start) << std::endl;
    start = omp_get_wtime();
    hcompress(hmatrix, abs_trunc_tol, h2opus_handle);
    end = omp_get_wtime();
    std::cout << "Time to compress FD matrix: " << (end-start) << std::endl;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Preconditioner construction
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create a sampler for the FD matrix. Since we want to produce an approximate inverse, improving the condition
    // of the matrix by adding a small multiple of the identity will help NS converge better
    // A smaller value produces a better preconditioner, but has an increased chance of failing to converge
    // as the condition of the matrix worsens
    start = omp_get_wtime();
    SimpleHMatrixSampler<H2OPUS_HWTYPE_CPU> cpu_hmatrix_sampler(&hmatrix, h2opus_handle);
    end = omp_get_wtime();
    std::cout << "Time for simple hmatrix sampler: " << (end-start) << std::endl;

    cpu_hmatrix_sampler.setScaleA(-1);
    cpu_hmatrix_sampler.setIdentityScale(sigma);

    start = omp_get_wtime();
    HighOrderInversionSampler<H2OPUS_HWTYPE_CPU> cpu_ns_sampler(&cpu_hmatrix_sampler, 16, h2opus_handle);
    end = omp_get_wtime();
    std::cout << "Time for HighOrderInversion sampler: " << (end-start) << std::endl;

    // Use an adaptive threshold for the construction of the NS iterates, starting at
    // eps_min and tightening to eps_max as we get closer to convergence
    H2Opus_Real eps_min = ns_it_eps * 10, eps_max = ns_it_eps;

    start = omp_get_wtime();
    newtonSchultz(cpu_ns_sampler, X_prev, max_samples, eps_min, eps_max, approx_inverse, h2opus_handle, zero_hmatrix,
                  0);
    end = omp_get_wtime();
    std::cout << "Time for newtonSchultz: " << (end-start) << std::endl;

    start = omp_get_wtime();
    thrust::host_vector<H2Opus_Real> x0(n, 0), b(n, 1);

    // Remove from the sampler the multiple of the identity that we added to improve conditioning
    cpu_hmatrix_sampler.setIdentityScale(0);
    end = omp_get_wtime();
    std::cout << "Time: " << (end-start) << std::endl;

    int it;

    H2Opus_Real error;

    // Solve Ax = b using preconditioned CG
    start  = omp_get_wtime();
    cgp<H2OPUS_HWTYPE_CPU>(&cpu_hmatrix_sampler, approx_inverse, vec_ptr(x0), vec_ptr(b), 300, 1e-9, it, error,
                           h2opus_handle);
    end    = omp_get_wtime();
    std::cout << "Time to solve:" << (end-start) << std::endl;
    exit(0);

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
