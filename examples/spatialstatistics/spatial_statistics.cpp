#include <petsc.h>
#include <h2opus.h>
#include "spatial_statistics.h"
#include "../common/example_problem.h"

#define DEFAULT_ETA 1.0

// PETSc entry point to functor evaluation
static PetscScalar petsc_kernel(PetscInt sdim, PetscReal x[], PetscReal y[], void *ctx)
{
    Spatial_Statistics<H2Opus_Real> *kgen = static_cast<Spatial_Statistics<H2Opus_Real> *>(ctx);
    return (*kgen)(x, y);
}

int main(int argc, char **argv)
{
    PetscErrorCode ierr;

#if defined(PETSC_HAVE_MPI_INIT_THREAD)
    PETSC_MPI_THREAD_REQUIRED = MPI_THREAD_MULTIPLE;
#endif
    ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    if (ierr) return ierr;

    // Argument parsing
    PetscInt num_points = 1024;
    PetscBool dump_points = PETSC_FALSE;
    PetscReal phi = 0.5, nu = 1.0;
    PetscInt m = 64, cheb_grid_pts = 8;
    PetscReal eta = DEFAULT_ETA, trunc_eps = 0.0;
    PetscBool forcecpu = PETSC_FALSE;
    PetscBool native = PETSC_TRUE;
    PetscBool summary = PETSC_TRUE;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "FD2D solver", "");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-n", "Number of random points", __FILE__, num_points, &num_points, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-dump_points", "Dump points", __FILE__, dump_points, &dump_points, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-phi", "Phi in kernel", __FILE__, phi, &phi, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-nu", "Nu in kernel", __FILE__, nu, &nu, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-m", "Leaf size in the KD-tree", __FILE__, m, &m, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-k", "Number of grid points in each dimension for Chebyshev interpolation (rank = k^d)",
                           __FILE__, cheb_grid_pts, &cheb_grid_pts, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eta", "Admissibility parameter eta", __FILE__, eta, &eta, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-te", "Relative truncation error threshold (0.0 for no compression)", __FILE__, trunc_eps, &trunc_eps, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-native", "Perform solve in native mode", __FILE__, native, &native, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-summary", "Report summary", __FILE__, summary, &summary, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-forcecpu", "Force computation to run on the CPU", __FILE__, forcecpu, &forcecpu, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    // Profiling support
    PetscLogStage astage, cstage, pstage, sstage;
    ierr = PetscLogStageRegister("Assembly", &astage);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Compression", &cstage);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Setup", &pstage);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Solve", &sstage);CHKERRQ(ierr);
    double stime[] = {0, 0, 0, 0};

    // Geometry
    PointCloud<H2Opus_Real> pt_cloud;
    generateRandomSphereSurface(pt_cloud, num_points);
    size_t n = pt_cloud.getDataSetSize();
    int dim = pt_cloud.getDimension();
    ierr = PetscPrintf(PETSC_COMM_WORLD, "N = %ld\n", n);CHKERRQ(ierr);
    if (dump_points) pt_cloud.dump();

    // Create a functor that can generate the matrix entries from two points
    Spatial_Statistics<H2Opus_Real> kgen(phi,nu,dim);

    // Create the H2 matrix
    if (summary) { stime[0] = MPI_Wtime(); }
    ierr = PetscLogStagePush(astage);CHKERRQ(ierr);
    Mat A;
    std::vector<PetscScalar> coords = pt_cloud.getCoords();
    ierr = MatCreateH2OpusFromKernel(PETSC_COMM_WORLD,                   // the MPI communicator associated to the matrix
                                     PETSC_DECIDE, PETSC_DECIDE, n, n,   // local and global sizes
                                     dim, coords.data(), PETSC_FALSE,    // point coordinates
                                     petsc_kernel, &kgen,                // kernel
                                     eta, m, cheb_grid_pts,              // construction parameters (can be also selected at runtime from PETSc)
                                     &A);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(A,"cov_");CHKERRQ(ierr);
    ierr = MatBindToCPU(A,forcecpu);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    if (summary) { stime[0] = MPI_Wtime() - stime[0]; }

    // Compress the H2 matrix
    if (trunc_eps > 0.0) {
      PetscReal norm2;
      ierr = MatNorm(A,NORM_2,&norm2);CHKERRQ(ierr);

      if (summary) { stime[1] = MPI_Wtime(); }
      ierr = PetscLogStagePush(cstage);CHKERRQ(ierr);
      ierr = MatH2OpusCompress(A,trunc_eps * norm2);CHKERRQ(ierr);
      ierr = PetscLogStagePop();CHKERRQ(ierr);
      if (summary) { stime[1] = MPI_Wtime() - stime[1]; }
      ierr = MatViewFromOptions(A,NULL,"-mat_view");CHKERRQ(ierr);

      // Print norms
      PetscReal cnorm2;
      ierr = MatNorm(A, NORM_2, &cnorm2);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  Original matrix 2-norm %g\n", norm2);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  Compressed Hmatrix 2-norm %g\n", cnorm2);CHKERRQ(ierr);
    }

    // Setup the solver
    KSP ksp;
    if (summary) { stime[2] = MPI_Wtime(); }
    ierr = PetscLogStagePush(pstage);CHKERRQ(ierr);
    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    if (summary) { stime[2] = MPI_Wtime() - stime[2]; }

    // Solve linear system
    Vec x, b;
    PetscInt its;
    if (summary) { stime[3] = MPI_Wtime(); }
    ierr = MatH2OpusSetNativeMult(A, native);CHKERRQ(ierr);
    ierr = PetscLogStagePush(sstage);CHKERRQ(ierr);
    ierr = MatCreateVecs(A, &x, &b);CHKERRQ(ierr);
    ierr = VecSet(b, 1.0);CHKERRQ(ierr);
    ierr = KSPSolve(ksp, b, x);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp, &its);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    if (summary) { stime[3] = MPI_Wtime() - stime[3]; }

    // Write a brief summary
    if (summary) {
      PetscMPIInt size;
      ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"========================================= SUMMARY =========================================\n");CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%d\t%ld\t%1.6g\t%1.6g\t%1.6g\t%1.6g\t%D\n",size,n,stime[0],stime[1],stime[2],stime[3],its);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"===========================================================================================\n");CHKERRQ(ierr);
    }

    // Cleanup
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
