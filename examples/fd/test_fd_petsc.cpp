#include <petsc.h>
#include <h2opus.h>
#include "fd_core.h"

#if defined(PETSC_HAVE_H2OPUS)
#define DEFAULT_ETA 1.0

// PETSc entry point to functor evaluation
static PetscScalar petsc_kernel(PetscInt sdim, PetscReal x[], PetscReal y[], void *ctx)
{
    FDGen<H2Opus_Real> *fdgen = static_cast<FDGen<H2Opus_Real> *>(ctx);
    return (*fdgen)(x, y);
}

// A + S + D
typedef struct {
  Mat A;
  Mat S;
  Vec D;
} ApSpDctx;

PetscErrorCode MatMult_ApSpD(Mat M, Vec x, Vec y)
{
  ApSpDctx       *ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(M,&ctx);CHKERRQ(ierr);
  ierr = VecPointwiseMult(y,ctx->D,x);CHKERRQ(ierr);
  ierr = MatMultAdd(ctx->S,x,y,y);CHKERRQ(ierr);
  ierr = MatMultAdd(ctx->A,x,y,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_ApSpD(Mat M)
{
  ApSpDctx       *ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(M,&ctx);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->S);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->A);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->D);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode BuildA(MPI_Comm comm, PointCloud<H2Opus_Real>& pt_cloud, FDGen<H2Opus_Real> &fdgen, PetscReal eta, PetscInt leafsize, PetscInt cheb_grid_pts, Mat *An)
{
    PetscErrorCode ierr;

    PetscInt n = pt_cloud.getDataSetSize();
    PetscInt dim = pt_cloud.getDimension();

    PetscFunctionBeginUser;
    // Force running on the CPU
    PetscBool forcecpu = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL,"-forcecpu",&forcecpu,NULL);CHKERRQ(ierr);

    // Create the PETSc matrix
    std::vector<PetscScalar> coords = pt_cloud.getCoordsV();
    Mat A;
    ierr = MatCreateH2OpusFromKernel(comm,                             // the communicator associated to the matrix
                                     PETSC_DECIDE, PETSC_DECIDE, n, n, // local and global sizes
                                     dim, coords.data(), PETSC_FALSE,  // point coordinates
                                     petsc_kernel, &fdgen,             // kernel
                                     eta, leafsize, cheb_grid_pts,     // construction parameters (can be also selected at runtime from PETSc)
                                     &A);CHKERRQ(ierr);

    ierr = MatBindToCPU(A,forcecpu);CHKERRQ(ierr);

    // Flag the matrix as symmetric and set from options
    // To see the allowed customizations, run with -help and grep for inner_mat_h2opus
    ierr = MatSetOptionsPrefix(A,"inner_");CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);

    // Build the hmatrix
    // If PETSC and H2OPUS are both configured with CUDA support
    // the matrix operator will be applied on the device
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    *An  = A;
    PetscFunctionReturn(0);
}

static PetscErrorCode BuildS(Mat A, PointCloud<H2Opus_Real>& pt_cloud, FDGen<H2Opus_Real> &fdgen, Mat *Sn)
{
    Mat            S;
    PetscInt       nz = 5, n, N, rst, ren, i;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&S);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&n,NULL);CHKERRQ(ierr);
    ierr = MatGetSize(A,&N,NULL);CHKERRQ(ierr);
    ierr = MatSetSizes(S,n,n,N,N);CHKERRQ(ierr);
    ierr = MatSetType(S,MATAIJ);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(S,"S_");CHKERRQ(ierr);
    ierr = MatSetOption(S,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(S);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(S,nz,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(S,nz,NULL,nz,NULL);CHKERRQ(ierr);

    // Assemble S
    SMat<H2Opus_Real> Sgen(&fdgen,&pt_cloud);
    ierr = MatGetOwnershipRange(S,&rst,&ren);CHKERRQ(ierr);
    for (i = rst; i < ren; i++) {
      PetscInt    ridx[5];
      PetscScalar rval[5];

      Sgen.generate_row(i,nz,ridx,rval);CHKERRQ(ierr);
      ierr = MatSetValues(S,1,&i,nz,ridx,rval,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    *Sn  = S;
    PetscFunctionReturn(0);
}

static PetscErrorCode BuildD(Mat A, PointCloud<H2Opus_Real>& pt_cloud, FDGen<H2Opus_Real> &fdgen, PetscReal eta, PetscInt leafsize, PetscInt cheb_grid_pts, Vec *Dn)
{
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    // Assemble an H2 matrix on the outer grid
    OuterGrid<H2Opus_Real> *opt_cloud = pt_cloud.ogrid;
    size_t no = opt_cloud->getDataSetSize();
    PetscInt dim = opt_cloud->getDimension();

    // Create the PETSc matrix
    Mat Ao;
    std::vector<PetscScalar> coords = opt_cloud->getCoordsV();
    ierr = MatCreateH2OpusFromKernel(PetscObjectComm((PetscObject)A),    // the MPI communicator associated to the matrix
                                     PETSC_DECIDE, PETSC_DECIDE, no, no, // local and global sizes
                                     dim,  coords.data(), PETSC_FALSE,   // point coordinates
                                     petsc_kernel, &fdgen,               // kernel
                                     eta, leafsize, cheb_grid_pts,       // construction parameters (can be also selected at runtime from PETSc)
                                     &Ao);CHKERRQ(ierr);

    // Always bind this larger matrix to run on the CPU boundedness
    ierr = MatBindToCPU(Ao,PETSC_TRUE);CHKERRQ(ierr);

    // Flag the matrix as symmetric and set from options
    // to see the allowed customizations, run with -help and grep for outer_mat_h2opus
    ierr = MatSetOptionsPrefix(Ao,"outer_");CHKERRQ(ierr);
    ierr = MatSetOption(Ao,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(Ao,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(Ao);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(Ao,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Ao,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    // Now compute diagonal term for inner matrix
    Vec x, y;
    ierr = MatCreateVecs(Ao,&x,&y);CHKERRQ(ierr);
    ierr = VecSet(x,-1.0);CHKERRQ(ierr);
    ierr = MatMult(Ao,x,y);CHKERRQ(ierr);
    ierr = VecViewFromOptions(y,NULL,"-y_view");CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);

    // Setup PetscSF to map from outergrid ordering to innergrid natural (point cloud) ordering
    PetscInt ist,ien;
    ierr = MatGetOwnershipRange(A,&ist,&ien);CHKERRQ(ierr);
    std::vector<PetscInt> intv;
    opt_cloud->get_indices_interior(intv);
    PetscInt c = 0;
    for (size_t i = 0; i < intv.size(); i++) {
      if (i < (size_t)ist || i >= (size_t)ien) continue;
      intv[c++] = intv[i];
    }
    PetscSF sf;
    PetscLayout rmapo;
    ierr = MatGetLayouts(Ao,&rmapo,NULL);CHKERRQ(ierr);
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)A),&sf);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(sf,rmapo,c,NULL,PETSC_OWN_POINTER,intv.data());CHKERRQ(ierr);

    // Communicate diagonal
    Vec D;
    PetscScalar *yy,*dd;
    ierr = MatCreateVecs(A,&D,NULL);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y,(const PetscScalar**)&yy);CHKERRQ(ierr);
    ierr = VecGetArrayWrite(D,&dd);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sf,MPIU_SCALAR,yy,dd,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_SCALAR,yy,dd,MPI_REPLACE);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y,(const PetscScalar**)&yy);CHKERRQ(ierr);
    ierr = VecRestoreArrayWrite(D,&dd);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
    ierr = MatDestroy(&Ao);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);

    // Check diagonal computation
    PetscBool check_diag = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL,"-check_diag",&check_diag,NULL);CHKERRQ(ierr);
    if (check_diag) {
      PetscReal err;

      ierr = VecDuplicate(D,&y);CHKERRQ(ierr);
      ierr = VecGetArrayRead(D,(const PetscScalar**)&dd);CHKERRQ(ierr);
      ierr = VecGetArrayWrite(y,&yy);CHKERRQ(ierr);
      for (int i = 0; i < ien-ist; i++) {
         PetscReal xx[3];
         xx[0] = pt_cloud.getDataPoint(i+ist,0);
         xx[1] = pt_cloud.getDataPoint(i+ist,1);
         xx[2] = 0;
         yy[i] = fdgen.compute_diagonal(&pt_cloud,xx); // N^2 algorithm!
      }
      ierr = VecRestoreArrayRead(D,(const PetscScalar**)&dd);CHKERRQ(ierr);
      ierr = VecRestoreArrayWrite(y,&yy);CHKERRQ(ierr);
      ierr = VecViewFromOptions(D,NULL,"-D_view");CHKERRQ(ierr);
      ierr = VecViewFromOptions(y,NULL,"-D_exact_view");CHKERRQ(ierr);
      // Uncomment to use exact diagonal
      //ierr = VecCopy(y,D);CHKERRQ(ierr);
      ierr = VecAXPY(y,-1.0,D);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(y,y,D);CHKERRQ(ierr);
      ierr = VecNorm(y,NORM_INFINITY,&err);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)D),"Error diag %g\n",err);CHKERRQ(ierr);
      ierr = VecDestroy(&y);CHKERRQ(ierr);
    }
    *Dn  = D;
    PetscFunctionReturn(0);
}

static PetscErrorCode BuildOperators(Mat A, Mat S, Vec D, Mat *Mn, Mat *Pn)
{
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    // The final operator layout will depend on whether or not we flag
    // A to use native ordering outside of this function
    IS indexmap = NULL;
    PetscBool hord;
    ierr = MatH2OpusGetNativeMult(A,&hord);CHKERRQ(ierr);
    if (hord) {
      ierr = MatH2OpusGetIndexMap(A,&indexmap);CHKERRQ(ierr);
    }

    // Permute S if needed
    Mat Sp = NULL;
    if (hord) {
      ierr = MatPermute(S,indexmap,indexmap,&Sp);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)S);CHKERRQ(ierr);
      Sp   = S;
    }
    ierr = MatSetOption(Sp,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);

    // Copy S to the GPU if needed
    // If the H2 matrix is bound to the CPU, do not convert to CUSPARSE the S matrix
#if defined(PETSC_HAVE_CUDA)
    PetscBool cpu;
    ierr = MatBoundToCPU(A,&cpu);CHKERRQ(ierr);
    if (!cpu) {
      ierr = MatConvert(Sp,MATAIJCUSPARSE,MAT_INPLACE_MATRIX,&Sp);CHKERRQ(ierr);
    }
#endif

    // Permute D if needed
    Vec Dp = NULL;
    if (hord) {
      // Allows to map between native and application (pointcloud) ordering
      ierr = MatH2OpusMapVec(A,PETSC_FALSE,D,&Dp);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)D);CHKERRQ(ierr);
      Dp   = D;
    }

    // A + S + D as a shell matrix (only matvec action is needed)
    ApSpDctx *ctx;
    ierr = PetscNew(&ctx);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    ctx->A = A;
    ctx->S = Sp;
    ctx->D = Dp;

    PetscInt n,N;
    ierr = MatGetLocalSize(A,&n,NULL);CHKERRQ(ierr);
    ierr = MatGetSize(A,&N,NULL);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)A),n,n,N,N,ctx,Mn);CHKERRQ(ierr);
    ierr = MatShellSetOperation(*Mn,MATOP_MULT,(void (*)(void))MatMult_ApSpD);CHKERRQ(ierr);
    ierr = MatShellSetOperation(*Mn,MATOP_DESTROY,(void (*)(void))MatDestroy_ApSpD);CHKERRQ(ierr);

    VecType vtype;
    ierr = MatGetVecType(A,&vtype);CHKERRQ(ierr);
    ierr = MatShellSetVecType(*Mn,vtype);CHKERRQ(ierr);

    // Preconditioning matrix
    ierr = PetscObjectReference((PetscObject)Sp);CHKERRQ(ierr);
    *Pn  = Sp;
    PetscFunctionReturn(0);
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
    PetscInt dim = 2;
    PetscInt grid_x = 32, m = 32, cheb_grid_pts = 8;
    PetscReal eta = DEFAULT_ETA, trunc_eps = 1.e-4;
    PetscBool native = PETSC_FALSE;
    PetscBool summary = PETSC_FALSE;
    PetscOptionsBegin(PETSC_COMM_WORLD, "", "FD2D solver", "");
    ierr = PetscOptionsRangeInt("-dim", "The geometrical dimension", __FILE__, dim, &dim, NULL,2,3);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-gx", "Grid points in the X direction", __FILE__, grid_x, &grid_x, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-m", "Leaf size in the KD-tree", __FILE__, m, &m, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-k", "Number of grid points in each dimension for Chebyshev interpolation (rank = k^d)",
                           __FILE__, cheb_grid_pts, &cheb_grid_pts, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eta", "Admissibility parameter eta", __FILE__, eta, &eta, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-te", "Relative truncation error threshold", __FILE__, trunc_eps, &trunc_eps, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-native", "Perform solve in native mode", __FILE__, native, &native, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-summary", "Report summary", __FILE__, summary, &summary, NULL);CHKERRQ(ierr);
    PetscOptionsEnd();

    // Profiling support
    PetscLogStage astage, cstage, pstage, sstage;
    ierr = PetscLogStageRegister("Assembly", &astage);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Compression", &cstage);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Setup", &pstage);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Solve", &sstage);CHKERRQ(ierr);
    double stime[3] = {0, 0, 0};

    // Geometry
    PointCloud<H2Opus_Real> pt_cloud;
    pt_cloud.generateGrid(dim, grid_x, -1.0 + 2.0 / (grid_x + 1), 1.0 - 2.0 / (grid_x + 1));
    size_t n = pt_cloud.getDataSetSize();
    ierr = PetscPrintf(PETSC_COMM_WORLD, "N = %ld\n", n);CHKERRQ(ierr);
    if (dim == 3) { ierr = PetscPrintf(PETSC_COMM_WORLD,"=========== Warning: Only matrix compression will be performed =============\n");CHKERRQ(ierr); }

    // Create a functor that can generate the matrix entries from two points
    FDGen<H2Opus_Real> fdgen(dim,pt_cloud.h);

    // Construct A+S+D
    Mat A,S,M,P;
    Vec D;
    if (summary) { stime[0] = MPI_Wtime(); }
    ierr = PetscLogStagePush(astage);CHKERRQ(ierr);
    ierr = BuildA(PETSC_COMM_WORLD,pt_cloud,fdgen,eta,m,cheb_grid_pts,&A);CHKERRQ(ierr);
    ierr = BuildS(A,pt_cloud,fdgen,&S);CHKERRQ(ierr);
    ierr = BuildD(A,pt_cloud,fdgen,eta,m,cheb_grid_pts,&D);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);

    // Can compute 2-norm of the matrix (NORM_1 and NORM_INFINITY are also supported)
    PetscReal norm2;
    ierr = MatNorm(A,NORM_2,&norm2);CHKERRQ(ierr);

    // Compress the H2 matrix
    ierr = PetscLogStagePush(cstage);CHKERRQ(ierr);
    ierr = MatH2OpusCompress(A,trunc_eps * norm2);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);

    // Can print basic info (memory usage, structure in eps format etc)
    ierr = MatViewFromOptions(A,NULL,"-mat_view");CHKERRQ(ierr);

    // Print norms
    PetscReal cnorm2;
    ierr = MatNorm(A, NORM_2, &cnorm2);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  Original matrix 2-norm %g\n", norm2);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  Compressed Hmatrix 2-norm %g\n", cnorm2);CHKERRQ(ierr);

    // Setup final operators for the solver
    ierr = MatH2OpusSetNativeMult(A,native);CHKERRQ(ierr);
    ierr = BuildOperators(A,S,D,&M,&P);CHKERRQ(ierr);
    ierr = MatDestroy(&S);CHKERRQ(ierr);
    ierr = VecDestroy(&D);CHKERRQ(ierr);
    if (summary) { stime[0] = MPI_Wtime() - stime[0]; }

    // Setup the solver
    KSP ksp;
    if (summary) { stime[1] = MPI_Wtime(); }
    ierr = PetscLogStagePush(pstage);CHKERRQ(ierr);
    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,M,P);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    if (summary) { stime[1] = MPI_Wtime() - stime[1]; }

    // Solve linear system
    Vec x, b;
    PetscInt its;
    if (summary) { stime[2] = MPI_Wtime(); }
    ierr = PetscLogStagePush(sstage);CHKERRQ(ierr);
    ierr = MatCreateVecs(M, &x, &b);CHKERRQ(ierr);
    ierr = VecSet(b, 1.0);CHKERRQ(ierr);
    ierr = KSPSolve(ksp, b, x);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    if (summary) { stime[2] = MPI_Wtime() - stime[2]; }

    if (native) {
      Vec xn;

      ierr = MatH2OpusMapVec(A,PETSC_TRUE,x,&xn);CHKERRQ(ierr);
      ierr = VecDestroy(&x);CHKERRQ(ierr);
      x = xn;
    }

    // Write a brief summary
    if (summary) {
      PetscMPIInt size;
      ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"========================================= SUMMARY =========================================\n");CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%d\t%ld\t%1.6g\t%1.6g\t%1.6g\t%d\n",size,n,stime[0],stime[1],stime[2],(int)its);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"===========================================================================================\n");CHKERRQ(ierr);
    }

    // Cleanup
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    ierr = MatDestroy(&M);CHKERRQ(ierr);
    ierr = MatDestroy(&P);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
#else
#error "This example requires PETSc compiled with H2OPUS support. Reconfigure PETSc with --download-h2opus or --with-h2opus-lib=... --with-h2opus-include=..."
#endif
