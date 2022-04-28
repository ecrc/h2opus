#include <slepc.h>
#include <h2opus.h>
#include "../common/example_problem.h"

#define DEFAULT_ETA 1.0

// PETSc entry point to functor evaluation
static PetscScalar petsc_kernel(PetscInt sdim, PetscReal x[], PetscReal y[], void *ctx)
{
    FunctionGen<H2Opus_Real> *fgen = static_cast<FunctionGen<H2Opus_Real> *>(ctx);
    return (*fgen)(x, y);
}

int main(int argc, char **argv)
{
    PetscErrorCode ierr;

#if defined(PETSC_HAVE_MPI_INIT_THREAD)
    PETSC_MPI_THREAD_REQUIRED = MPI_THREAD_MULTIPLE;
#endif
    ierr = SlepcInitialize(&argc, &argv, NULL, NULL);
    if (ierr) return ierr;

    // Argument parsing
    PetscInt dim = 2;
    PetscInt grid_x = 32, grid_y = 32, grid_z = 1, m = 32, cheb_grid_pts = 4;
    PetscReal eta = DEFAULT_ETA;
    PetscBool native = PETSC_TRUE;
    PetscBool summary = PETSC_FALSE;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "FD2D solver", "");CHKERRQ(ierr);
    ierr = PetscOptionsRangeInt("-dim", "The geometrical dimension", __FILE__, dim, &dim, NULL,1,3);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-gx", "Grid points in the X direction", __FILE__, grid_x, &grid_x, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-gy", "Grid points in the Y direction", __FILE__, grid_y, &grid_y, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-gz", "Grid points in the Z direction", __FILE__, grid_z, &grid_z, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-m", "Leaf size in the KD-tree", __FILE__, m, &m, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-k", "Number of grid points in each dimension for Chebyshev interpolation (rank = k^d)",
                           __FILE__, cheb_grid_pts, &cheb_grid_pts, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eta", "Admissibility parameter eta", __FILE__, eta, &eta, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-native", "Perform solve in native mode", __FILE__, native, &native, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-summary", "Report summary", __FILE__, summary, &summary, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    // Profiling support
    PetscLogStage astage, sstage;
    ierr = PetscLogStageRegister("Assembly", &astage);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Solve", &sstage);CHKERRQ(ierr);

    // Geometry
    PointCloud<H2Opus_Real> pt_cloud;
    if (dim == 3)
        generate3DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, grid_z, 0, 1, 0, 1, 0, 1);
    else
        generate2DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, 0, 1, 0, 1);
    size_t n = pt_cloud.getDataSetSize();
    ierr = PetscPrintf(PETSC_COMM_WORLD, "N = %ld\n", n);CHKERRQ(ierr);

    // Create a functor that can generate the matrix entries from two points
    FunctionGen<H2Opus_Real> func_gen(dim);

    // Construct A
    Mat A;
    std::vector<PetscScalar> coords = pt_cloud.getCoords();
    ierr = PetscLogStagePush(astage);CHKERRQ(ierr);
    ierr = MatCreateH2OpusFromKernel(PETSC_COMM_WORLD,                          // the communicator associated to the matrix
                                     PETSC_DECIDE, PETSC_DECIDE, n, n,        // local and global sizes
                                     dim,  coords.data(), PETSC_FALSE,       // point coordinates
                                     petsc_kernel, &func_gen,                  // kernel
                                     eta, m, cheb_grid_pts,                     // construction parameters (can be also selected at runtime from PETSc)
                                     &A);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatH2OpusSetNativeMult(A,native);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);

    // View the matrix
    PetscBool viewexpl = PETSC_FALSE;
    ierr = MatViewFromOptions(A,NULL,"-A_view");CHKERRQ(ierr);
    ierr = PetscOptionsHasName(NULL,NULL,"-A_view_explicit",&viewexpl);CHKERRQ(ierr);
    if (viewexpl) { // Fill a dense operator with the result of A matvecs
      Mat Ae;
      ierr = MatComputeOperator(A,MATDENSE,&Ae);CHKERRQ(ierr);
      ierr = MatViewFromOptions(Ae,NULL,"-A_view_explicit");CHKERRQ(ierr);
      ierr = MatDestroy(&Ae);CHKERRQ(ierr);
    }

    // Setup eigenvalue solver
    EPS eps;
    ierr = PetscLogStagePush(sstage);CHKERRQ(ierr);
    ierr = EPSCreate(PetscObjectComm((PetscObject)A),&eps);CHKERRQ(ierr);
    ierr = EPSSetOperators(eps,A,NULL);CHKERRQ(ierr);
    ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);
    ierr = EPSSolve(eps);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);

    // Cleanup
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = EPSDestroy(&eps);CHKERRQ(ierr);
    ierr = SlepcFinalize();
    return ierr;
}
