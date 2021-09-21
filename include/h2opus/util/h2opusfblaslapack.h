#ifndef __H2OPUS_FBLASLAPACK_H
#define __H2OPUS_FBLASLAPACK_H

/* a tiny wrapper on top of FORTRAN BLAS/LAPACK calls */

#include <stdlib.h>
#include <stddef.h>

/* FORTRAN name mangling */

#ifndef H2OPUS_FMANGLE
#if defined(H2OPUS_FMANGLE_ADD)
#define H2OPUS_FMANGLE(lcname, UCNAME) lcname##_
#elif defined(H2OPUS_FMANGLE_UPPER)
#define H2OPUS_FMANGLE(lcname, UCNAME) UCNAME
#elif defined(H2OPUS_FMANGLE_NOCHANGE)
#define H2OPUS_FMANGLE(lcname, UCNAME) lcname
#else /* If not specified, use ADD */
#define H2OPUS_FMANGLE(lcname, UCNAME) lcname##_
#endif
#endif

/* Context based reuse of workspace */

typedef struct
{
    size_t s; /* allocated size */
    void *w;  /* allocated workspace */
} h2opus_fbl_ctx;

#define H2OPUS_FBL_CHK_CALL(f)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        int e = (f);                                                                                                   \
        if (e)                                                                                                         \
            fprintf(stderr, "BLAS/LAPACK error %d in %s at %s:%d\n", e, __func__, __FILE__, __LINE__);                 \
    } while (0)

#define H2OPUS_FBL_BEGIN()                                                                                             \
    {                                                                                                                  \
        int fbl_info = 0;

#define H2OPUS_FBL_END()                                                                                               \
    return fbl_info;                                                                                                   \
    }

#define H2OPUS_FBL_LWORK_BEGIN(c)                                                                                      \
    {                                                                                                                  \
        int fbl_info = 0;                                                                                              \
        int fbl_lwork = -1;                                                                                            \
        double fbl_lwork_dummy;                                                                                        \
        void *fbl_work = &fbl_lwork_dummy;                                                                             \
        void *fbl_work_c = (c) ? (c)->w : NULL;                                                                        \
        size_t fbl_work_s = (c) ? (c)->s : 0;

#define H2OPUS_FBL_LWORK_ALLOC(s)                                                                                      \
    if (fbl_info)                                                                                                      \
    {                                                                                                                  \
        fprintf(stderr, "H2OPUS_FBL_ALLOC error %d in %s at %s:%d\n", fbl_info, __func__, __FILE__, __LINE__);         \
        return fbl_info;                                                                                               \
    }                                                                                                                  \
    fbl_lwork = (int)(*((s *)fbl_work));                                                                               \
    fbl_work = fbl_work_c;                                                                                             \
    if (fbl_work_s < (sizeof(s)) * (size_t)fbl_lwork)                                                                  \
    {                                                                                                                  \
        free(fbl_work_c);                                                                                              \
        fbl_work_s = (sizeof(s)) * (size_t)fbl_lwork;                                                                  \
        fbl_work = malloc(fbl_work_s);                                                                                 \
    }

#define H2OPUS_FBL_LWORK_END(c)                                                                                        \
    if (!(c))                                                                                                          \
    {                                                                                                                  \
        free(fbl_work);                                                                                                \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        c->w = fbl_work;                                                                                               \
        c->s = fbl_work_s;                                                                                             \
    }                                                                                                                  \
    return fbl_info;                                                                                                   \
    }

#ifdef __cplusplus
extern "C"
{
#endif

    /* Enums (CBLAS style) */
    enum H2OPUS_FBL_TRANSPOSE
    {
        H2OpusFBLNoTrans = 111,
        H2OpusFBLTrans = 112,
        H2OpusFBLConjTrans = 113
    };

    enum H2OPUS_FBL_UPLO
    {
        H2OpusFBLUpper = 121,
        H2OpusFBLLower = 122,
        H2OpusFBLAll = 123 /* Not in CBLAS */
    };

    enum H2OPUS_FBL_DIAG
    {
        H2OpusFBLNonUnit = 131,
        H2OpusFBLUnit = 132
    };

    enum H2OPUS_FBL_SIDE
    {
        H2OpusFBLLeft = 141,
        H2OpusFBLRight = 142
    };

    enum H2OPUS_FBL_JOB /* Not in CBLAS */
    {
        H2OpusFBLJobA = 151,
        H2OpusFBLJobS = 152,
        H2OpusFBLJobO = 153,
        H2OpusFBLJobN = 154
    };

    extern float H2OPUS_FMANGLE(snrm2, SNRM2)(int *, const float *, int *);
    inline float h2opus_fbl_snrm2(int N, const float *X, int incX)
    {
        return H2OPUS_FMANGLE(snrm2, SNRM2)(&N, X, &incX);
    }

    extern double H2OPUS_FMANGLE(dnrm2, DNRM2)(int *, const double *, int *);
    inline double h2opus_fbl_dnrm2(int N, const double *X, int incX)
    {
        return H2OPUS_FMANGLE(dnrm2, DNRM2)(&N, X, &incX);
    }

    extern void H2OPUS_FMANGLE(sscal, SSCAL)(int *, float *, float *, int *);
    inline void h2opus_fbl_sscal(int N, float alpha, float *X, int incX)
    {
        return H2OPUS_FMANGLE(sscal, SSCAL)(&N, &alpha, X, &incX);
    }

    extern void H2OPUS_FMANGLE(dscal, DSCAL)(int *, double *, double *, int *);
    inline void h2opus_fbl_dscal(int N, double alpha, double *X, int incX)
    {
        return H2OPUS_FMANGLE(dscal, DSCAL)(&N, &alpha, X, &incX);
    }

    extern void H2OPUS_FMANGLE(saxpy, SAXPY)(int *, float *, const float *, int *, float *, int *);
    inline void h2opus_fbl_saxpy(int N, float alpha, const float *X, int incX, float *Y, int incY)
    {
        return H2OPUS_FMANGLE(saxpy, SAXPY)(&N, &alpha, X, &incX, Y, &incY);
    }

    extern void H2OPUS_FMANGLE(daxpy, DAXPY)(int *, double *, const double *, int *, double *, int *);
    inline void h2opus_fbl_daxpy(int N, double alpha, const double *X, int incX, double *Y, int incY)
    {
        return H2OPUS_FMANGLE(daxpy, DAXPY)(&N, &alpha, X, &incX, Y, &incY);
    }

    extern float H2OPUS_FMANGLE(sdot, SDOT)(int *, const float *, int *, const float *, int *);
    inline float h2opus_fbl_sdot(int N, const float *X, int incX, const float *Y, int incY)
    {
        float x = H2OPUS_FMANGLE(sdot, SDOT)(&N, X, &incX, Y, &incY);
        return x;
    }

    extern double H2OPUS_FMANGLE(ddot, DDOT)(int *, const double *, int *, const double *, int *);
    inline double h2opus_fbl_ddot(int N, const double *X, int incX, const double *Y, int incY)
    {
        double x = H2OPUS_FMANGLE(ddot, DDOT)(&N, X, &incX, Y, &incY);
        return x;
    }

    extern void H2OPUS_FMANGLE(sgemv, SGEMV)(const char *, int *, int *, float *, const float *, int *, const float *,
                                             int *, float *, float *, int *);
    inline void h2opus_fbl_sgemv(enum H2OPUS_FBL_TRANSPOSE TransA, int M, int N, float alpha, const float *A, int lda,
                                 const float *X, int incX, float beta, float *Y, int incY)
    {
        return H2OPUS_FMANGLE(sgemv, SGEMV)(TransA == H2OpusFBLNoTrans ? "N" : "T", &M, &N, &alpha, A, &lda, X, &incX,
                                            &beta, Y, &incY);
    }

    extern void H2OPUS_FMANGLE(dgemv, DGEMV)(const char *, int *, int *, double *, const double *, int *,
                                             const double *, int *, double *, double *, int *);
    inline void h2opus_fbl_dgemv(enum H2OPUS_FBL_TRANSPOSE TransA, int M, int N, double alpha, const double *A, int lda,
                                 const double *X, int incX, double beta, double *Y, int incY)
    {
        return H2OPUS_FMANGLE(dgemv, DGEMV)(TransA == H2OpusFBLNoTrans ? "N" : "T", &M, &N, &alpha, A, &lda, X, &incX,
                                            &beta, Y, &incY);
    }

    extern void H2OPUS_FMANGLE(sgemm, SGEMM)(const char *, const char *, int *, int *, int *, float *, const float *,
                                             int *, const float *, int *, float *, float *, int *);
    inline void h2opus_fbl_sgemm(enum H2OPUS_FBL_TRANSPOSE TransA, enum H2OPUS_FBL_TRANSPOSE TransB, int M, int N,
                                 int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                                 float *C, int ldc)
    {
        return H2OPUS_FMANGLE(sgemm, SGEMM)(TransA == H2OpusFBLNoTrans ? "N" : "T",
                                            TransB == H2OpusFBLNoTrans ? "N" : "T", &M, &N, &K, &alpha, A, &lda, B,
                                            &ldb, &beta, C, &ldc);
    }

    extern void H2OPUS_FMANGLE(dgemm, DGEMM)(const char *, const char *, int *, int *, int *, double *, const double *,
                                             int *, const double *, int *, double *, double *, int *);
    inline void h2opus_fbl_dgemm(enum H2OPUS_FBL_TRANSPOSE TransA, enum H2OPUS_FBL_TRANSPOSE TransB, int M, int N,
                                 int K, double alpha, const double *A, int lda, const double *B, int ldb, double beta,
                                 double *C, int ldc)
    {
        return H2OPUS_FMANGLE(dgemm, DGEMM)(TransA == H2OpusFBLNoTrans ? "N" : "T",
                                            TransB == H2OpusFBLNoTrans ? "N" : "T", &M, &N, &K, &alpha, A, &lda, B,
                                            &ldb, &beta, C, &ldc);
    }

    extern void H2OPUS_FMANGLE(strsm, STRSM)(const char *, const char *, const char *, const char *, int *, int *,
                                             float *, const float *, int *, float *, int *);
    inline void h2opus_fbl_strsm(enum H2OPUS_FBL_SIDE Side, enum H2OPUS_FBL_UPLO Uplo, enum H2OPUS_FBL_TRANSPOSE TransA,
                                 enum H2OPUS_FBL_DIAG Diag, int M, int N, float alpha, const float *A, int lda,
                                 float *B, int ldb)
    {
        return H2OPUS_FMANGLE(strsm, STRSM)(Side == H2OpusFBLLeft ? "L" : "R", Uplo == H2OpusFBLUpper ? "U" : "L",
                                            TransA == H2OpusFBLNoTrans ? "N" : "T",
                                            Diag == H2OpusFBLNonUnit ? "N" : "U", &M, &N, &alpha, A, &lda, B, &ldb);
    }

    extern void H2OPUS_FMANGLE(dtrsm, DTRSM)(const char *, const char *, const char *, const char *, int *, int *,
                                             double *, const double *, int *, double *, int *);
    inline void h2opus_fbl_dtrsm(enum H2OPUS_FBL_SIDE Side, enum H2OPUS_FBL_UPLO Uplo, enum H2OPUS_FBL_TRANSPOSE TransA,
                                 enum H2OPUS_FBL_DIAG Diag, int M, int N, double alpha, const double *A, int lda,
                                 double *B, int ldb)
    {
        return H2OPUS_FMANGLE(dtrsm, DTRSM)(Side == H2OpusFBLLeft ? "L" : "R", Uplo == H2OpusFBLUpper ? "U" : "L",
                                            TransA == H2OpusFBLNoTrans ? "N" : "T",
                                            Diag == H2OpusFBLNonUnit ? "N" : "U", &M, &N, &alpha, A, &lda, B, &ldb);
    }

    extern void H2OPUS_FMANGLE(strmm, STRMM)(const char *, const char *, const char *, const char *, int *, int *,
                                             float *, const float *, int *, float *, int *);
    inline void h2opus_fbl_strmm(enum H2OPUS_FBL_SIDE Side, enum H2OPUS_FBL_UPLO Uplo, enum H2OPUS_FBL_TRANSPOSE TransA,
                                 enum H2OPUS_FBL_DIAG Diag, int M, int N, float alpha, const float *A, int lda,
                                 float *B, int ldb)
    {
        return H2OPUS_FMANGLE(strmm, STRMM)(Side == H2OpusFBLLeft ? "L" : "R", Uplo == H2OpusFBLUpper ? "U" : "L",
                                            TransA == H2OpusFBLNoTrans ? "N" : "T",
                                            Diag == H2OpusFBLNonUnit ? "N" : "U", &M, &N, &alpha, A, &lda, B, &ldb);
    }

    extern void H2OPUS_FMANGLE(dtrmm, DTRMM)(const char *, const char *, const char *, const char *, int *, int *,
                                             double *, const double *, int *, double *, int *);
    inline void h2opus_fbl_dtrmm(enum H2OPUS_FBL_SIDE Side, enum H2OPUS_FBL_UPLO Uplo, enum H2OPUS_FBL_TRANSPOSE TransA,
                                 enum H2OPUS_FBL_DIAG Diag, int M, int N, double alpha, const double *A, int lda,
                                 double *B, int ldb)
    {
        return H2OPUS_FMANGLE(dtrmm, DTRMM)(Side == H2OpusFBLLeft ? "L" : "R", Uplo == H2OpusFBLUpper ? "U" : "L",
                                            TransA == H2OpusFBLNoTrans ? "N" : "T",
                                            Diag == H2OpusFBLNonUnit ? "N" : "U", &M, &N, &alpha, A, &lda, B, &ldb);
    }

    extern void H2OPUS_FMANGLE(ssyrk, SSYRK)(const char *, const char *, int *, int *, float *, const float *, int *,
                                             float *, float *, int *);
    inline void h2opus_fbl_ssyrk(enum H2OPUS_FBL_UPLO Uplo, enum H2OPUS_FBL_TRANSPOSE Trans, int N, int K, float alpha,
                                 const float *A, int lda, float beta, float *C, int ldc)
    {
        return H2OPUS_FMANGLE(ssyrk, SSYRK)(Uplo == H2OpusFBLUpper ? "U" : "L", Trans == H2OpusFBLNoTrans ? "N" : "T",
                                            &N, &K, &alpha, A, &lda, &beta, C, &ldc);
    }

    extern void H2OPUS_FMANGLE(dsyrk, DSYRK)(const char *, const char *, int *, int *, double *, const double *, int *,
                                             double *, double *, int *);
    inline void h2opus_fbl_dsyrk(enum H2OPUS_FBL_UPLO Uplo, enum H2OPUS_FBL_TRANSPOSE Trans, int N, int K, double alpha,
                                 const double *A, int lda, double beta, double *C, int ldc)
    {
        return H2OPUS_FMANGLE(dsyrk, DSYRK)(Uplo == H2OpusFBLUpper ? "U" : "L", Trans == H2OpusFBLNoTrans ? "N" : "T",
                                            &N, &K, &alpha, A, &lda, &beta, C, &ldc);
    }

    /* returns useless int to match lapacke style */
    extern int H2OPUS_FMANGLE(slacpy, SLACPY)(const char *, int *, int *, const float *, int *, float *, int *);
    inline int h2opus_fbl_slacpy(enum H2OPUS_FBL_UPLO Uplo, int m, int n, const float *a, int lda, float *b, int ldb)
    {
        H2OPUS_FMANGLE(slacpy, SLACPY)
        (Uplo == H2OpusFBLUpper ? "U" : (Uplo == H2OpusFBLLower ? "L" : "A"), &m, &n, a, &lda, b, &ldb);
        return 0;
    }

    extern int H2OPUS_FMANGLE(dlacpy, DLACPY)(const char *, int *, int *, const double *, int *, double *, int *);
    inline int h2opus_fbl_dlacpy(enum H2OPUS_FBL_UPLO Uplo, int m, int n, const double *a, int lda, double *b, int ldb)
    {
        H2OPUS_FMANGLE(dlacpy, DLACPY)
        (Uplo == H2OpusFBLUpper ? "U" : (Uplo == H2OpusFBLLower ? "L" : "A"), &m, &n, a, &lda, b, &ldb);
        return 0;
    }

    extern int H2OPUS_FMANGLE(sorgqr, SORGQR)(int *, int *, int *, float *, int *, const float *, float *, int *,
                                              int *);
    inline int h2opus_fbl_sorgqr(int m, int n, int k, float *a, int lda, const float *tau, h2opus_fbl_ctx *ctx)
    {
        H2OPUS_FBL_LWORK_BEGIN(ctx)
        H2OPUS_FMANGLE(sorgqr, SORGQR)(&m, &n, &k, a, &lda, tau, (float *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_ALLOC(float)
        H2OPUS_FMANGLE(sorgqr, SORGQR)(&m, &n, &k, a, &lda, tau, (float *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_END(ctx)
    }

    extern int H2OPUS_FMANGLE(dorgqr, DORGQR)(int *, int *, int *, double *, int *, const double *, double *, int *,
                                              int *);
    inline int h2opus_fbl_dorgqr(int m, int n, int k, double *a, int lda, const double *tau, h2opus_fbl_ctx *ctx)
    {
        H2OPUS_FBL_LWORK_BEGIN(ctx)
        H2OPUS_FMANGLE(dorgqr, DORGQR)(&m, &n, &k, a, &lda, tau, (double *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_ALLOC(double)
        H2OPUS_FMANGLE(dorgqr, DORGQR)(&m, &n, &k, a, &lda, tau, (double *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_END(ctx)
    }

    extern int H2OPUS_FMANGLE(spotrf, SPOTRF)(const char *, int *, float *, int *, int *);
    inline int h2opus_fbl_spotrf(enum H2OPUS_FBL_UPLO Uplo, int n, float *a, int lda)
    {
        H2OPUS_FBL_BEGIN()
        H2OPUS_FMANGLE(spotrf, SPOTRF)(Uplo == H2OpusFBLUpper ? "U" : "L", &n, a, &lda, &fbl_info);
        H2OPUS_FBL_END()
    }

    extern int H2OPUS_FMANGLE(dpotrf, DPOTRF)(const char *, int *, double *, int *, int *);
    inline int h2opus_fbl_dpotrf(enum H2OPUS_FBL_UPLO Uplo, int n, double *a, int lda)
    {
        H2OPUS_FBL_BEGIN()
        H2OPUS_FMANGLE(dpotrf, DPOTRF)(Uplo == H2OpusFBLUpper ? "U" : "L", &n, a, &lda, &fbl_info);
        H2OPUS_FBL_END()
    }

    extern int H2OPUS_FMANGLE(sgeqrf, SGEQRF)(int *, int *, float *, int *, float *, float *, int *, int *);
    inline int h2opus_fbl_sgeqrf(int m, int n, float *a, int lda, float *tau, h2opus_fbl_ctx *ctx)
    {
        H2OPUS_FBL_LWORK_BEGIN(ctx)
        H2OPUS_FMANGLE(sgeqrf, SGEQRF)(&m, &n, a, &lda, tau, (float *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_ALLOC(float)
        H2OPUS_FMANGLE(sgeqrf, SGEQRF)(&m, &n, a, &lda, tau, (float *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_END(ctx)
    }

    extern int H2OPUS_FMANGLE(dgeqrf, DGEQRF)(int *, int *, double *, int *, double *, double *, int *, int *);
    inline int h2opus_fbl_dgeqrf(int m, int n, double *a, int lda, double *tau, h2opus_fbl_ctx *ctx)
    {
        H2OPUS_FBL_LWORK_BEGIN(ctx)
        H2OPUS_FMANGLE(dgeqrf, DGEQRF)(&m, &n, a, &lda, tau, (double *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_ALLOC(double)
        H2OPUS_FMANGLE(dgeqrf, DGEQRF)(&m, &n, a, &lda, tau, (double *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_END(ctx)
    }

    extern int H2OPUS_FMANGLE(sgesvd, SGESVD)(const char *, const char *, int *, int *, float *, int *, float *,
                                              float *, int *, float *, int *, float *, float *, int *, int *);
    inline int h2opus_fbl_sgesvd(enum H2OPUS_FBL_JOB jobu, enum H2OPUS_FBL_JOB jobvt, int m, int n, float *a, int lda,
                                 float *s, float *u, int ldu, float *vt, int ldvt, float *superb, h2opus_fbl_ctx *ctx)
    {
        const char *JOBS = "ASON";
        char *cjobu = (char *)(JOBS + ((ptrdiff_t)jobu - (ptrdiff_t)H2OpusFBLJobA));
        char *cjobvt = (char *)(JOBS + ((ptrdiff_t)jobvt - (ptrdiff_t)H2OpusFBLJobA));
        H2OPUS_FBL_LWORK_BEGIN(ctx)
        H2OPUS_FMANGLE(sgesvd, SGESVD)
        (cjobu, cjobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, superb, (float *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_ALLOC(float)
        H2OPUS_FMANGLE(sgesvd, SGESVD)
        (cjobu, cjobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, superb, (float *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_END(ctx)
    }

    extern int H2OPUS_FMANGLE(dgesvd, DGESVD)(const char *, const char *, int *, int *, double *, int *, double *,
                                              double *, int *, double *, int *, double *, double *, int *, int *);
    inline int h2opus_fbl_dgesvd(enum H2OPUS_FBL_JOB jobu, enum H2OPUS_FBL_JOB jobvt, int m, int n, double *a, int lda,
                                 double *s, double *u, int ldu, double *vt, int ldvt, double *superb,
                                 h2opus_fbl_ctx *ctx)
    {
        const char *JOBS = "ASON";
        char *cjobu = (char *)(JOBS + ((ptrdiff_t)jobu - (ptrdiff_t)H2OpusFBLJobA));
        char *cjobvt = (char *)(JOBS + ((ptrdiff_t)jobvt - (ptrdiff_t)H2OpusFBLJobA));
        H2OPUS_FBL_LWORK_BEGIN(ctx)
        H2OPUS_FMANGLE(dgesvd, DGESVD)
        (cjobu, cjobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, superb, (double *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_ALLOC(double)
        H2OPUS_FMANGLE(dgesvd, DGESVD)
        (cjobu, cjobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, superb, (double *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_END(ctx)
    }

    extern int H2OPUS_FMANGLE(sgeqp3, SGEQP3)(int *, int *, float *, int *, int *, float *, float *, int *, int *);
    inline int h2opus_fbl_sgeqp3(int m, int n, float *a, int lda, int *jpvt, float *tau, h2opus_fbl_ctx *ctx)
    {
        H2OPUS_FBL_LWORK_BEGIN(ctx)
        H2OPUS_FMANGLE(sgeqp3, SGEQP3)(&m, &n, a, &lda, jpvt, tau, (float *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_ALLOC(float)
        H2OPUS_FMANGLE(sgeqp3, SGEQP3)(&m, &n, a, &lda, jpvt, tau, (float *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_END(ctx)
    }

    extern int H2OPUS_FMANGLE(dgeqp3, DGEQP3)(int *, int *, double *, int *, int *, double *, double *, int *, int *);
    inline int h2opus_fbl_dgeqp3(int m, int n, double *a, int lda, int *jpvt, double *tau, h2opus_fbl_ctx *ctx)
    {
        H2OPUS_FBL_LWORK_BEGIN(ctx)
        H2OPUS_FMANGLE(dgeqp3, DGEQP3)(&m, &n, a, &lda, jpvt, tau, (double *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_ALLOC(double)
        H2OPUS_FMANGLE(dgeqp3, DGEQP3)(&m, &n, a, &lda, jpvt, tau, (double *)fbl_work, &fbl_lwork, &fbl_info);
        H2OPUS_FBL_LWORK_END(ctx)
    }

    /* ?lamch are not thread-safe, make sure mach values are initialized */
    extern float H2OPUS_FMANGLE(slamch, SLAMCH)(const char *);
    extern float H2OPUS_FMANGLE(dlamch, DLAMCH)(const char *);
    inline void h2opus_fbl_init()
    {
        char sstr[] = "ESBPNRMULO";
        for (int i = 0; i < 10; i++)
        {
            (void)(H2OPUS_FMANGLE(slamch, SLAMCH)(sstr + i));
            (void)(H2OPUS_FMANGLE(dlamch, DLAMCH)(sstr + i));
        }
        return;
    }

    /* batch blas */
#ifndef CPU_BATCH_LD
#define CPU_BATCH_LD(ld) ((ld) == 0 ? 1 : (ld))
#endif

#ifndef H2OPUS_STATIC_GEMM_BUF
#define H2OPUS_STATIC_GEMM_BUF 1024
#endif

#define H2OPUS_BATCH_GROUP_BEGIN(T)                                                                                    \
    {                                                                                                                  \
        int group_count = batchCount;                                                                                  \
        int *group_sizes, *gw = NULL, gs[H2OPUS_STATIC_GEMM_BUF];                                                      \
        char *ta_array, *tb_array, *taw = NULL, *tbw = NULL, tas[H2OPUS_STATIC_GEMM_BUF], tbs[H2OPUS_STATIC_GEMM_BUF]; \
        T *alpha_array, *beta_array, *aw = NULL, *bw = NULL, as[H2OPUS_STATIC_GEMM_BUF], bs[H2OPUS_STATIC_GEMM_BUF];   \
                                                                                                                       \
        if (batchCount > H2OPUS_STATIC_GEMM_BUF)                                                                       \
        {                                                                                                              \
            group_sizes = gw = (int *)malloc(sizeof(int) * batchCount);                                                \
            ta_array = taw = (char *)malloc(sizeof(char) * batchCount);                                                \
            tb_array = tbw = (char *)malloc(sizeof(char) * batchCount);                                                \
            alpha_array = aw = (T *)malloc(sizeof(T) * batchCount);                                                    \
            beta_array = bw = (T *)malloc(sizeof(T) * batchCount);                                                     \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            group_sizes = gs;                                                                                          \
            ta_array = tas;                                                                                            \
            tb_array = tbs;                                                                                            \
            alpha_array = as;                                                                                          \
            beta_array = bs;                                                                                           \
        }                                                                                                              \
                                                                                                                       \
        for (int i = 0; i < batchCount; i++)                                                                           \
        {                                                                                                              \
            ta_array[i] = (transa == H2OpusFBLTrans) ? 'T' : 'N';                                                      \
            tb_array[i] = (transb == H2OpusFBLTrans) ? 'T' : 'N';                                                      \
            alpha_array[i] = alpha;                                                                                    \
            beta_array[i] = beta;                                                                                      \
            group_sizes[i] = 1;                                                                                        \
            lda[i] = CPU_BATCH_LD(lda[i]);                                                                             \
            if (ldb)                                                                                                   \
                ldb[i] = CPU_BATCH_LD(ldb[i]);                                                                         \
            if (ldc)                                                                                                   \
                ldc[i] = CPU_BATCH_LD(ldc[i]);                                                                         \
        }

#define H2OPUS_BATCH_GROUP_END()                                                                                       \
    if (batchCount > H2OPUS_STATIC_GEMM_BUF)                                                                           \
    {                                                                                                                  \
        free(taw);                                                                                                     \
        free(tbw);                                                                                                     \
        free(aw);                                                                                                      \
        free(bw);                                                                                                      \
        free(gw);                                                                                                      \
    }                                                                                                                  \
    }

    /* Mangle Batch blas TODO: add others */
    extern void H2OPUS_FMANGLE(sgemv_batch, SGEMV_BATCH)(const char *, const int *, const int *, const float *,
                                                         const float **, const int *, const float **, const int *,
                                                         const float *, float **, const int *, const int *,
                                                         const int *);
    inline void h2opus_fbl_sgemv_batch(const char *trans, const int *m, const int *n, const float *alpha,
                                       const float **a, const int *lda, const float **x, const int *incx,
                                       const float *beta, float **y, const int *incy, const int *group_count,
                                       const int *group_size)
    {
        return H2OPUS_FMANGLE(sgemv_batch, SGEMV_BATCH)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy, group_count,
                                                        group_size);
    }

    extern void H2OPUS_FMANGLE(dgemv_batch, DGEMV_BATCH)(const char *, const int *, const int *, const double *,
                                                         const double **, const int *, const double **, const int *,
                                                         const double *, double **, const int *, const int *,
                                                         const int *);
    inline void h2opus_fbl_dgemv_batch(const char *trans, const int *m, const int *n, const double *alpha,
                                       const double **a, const int *lda, const double **x, const int *incx,
                                       const double *beta, double **y, const int *incy, const int *group_count,
                                       const int *group_size)
    {
        return H2OPUS_FMANGLE(dgemv_batch, DGEMV_BATCH)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy, group_count,
                                                        group_size);
    }

    extern void H2OPUS_FMANGLE(sgemm_batch, SGEMM_BATCH)(const char *, const char *, const int *, const int *,
                                                         const int *, const float *, const float **, const int *,
                                                         const float **, const int *, const float *, float **,
                                                         const int *, const int *, const int *);
    inline void h2opus_fbl_sgemm_batch(const char *transa_array, const char *transb_array, const int *m_array,
                                       const int *n_array, const int *k_array, const float *alpha_array,
                                       const float **a_array, const int *lda_array, const float **b_array,
                                       const int *ldb_array, const float *beta_array, float **c_array,
                                       const int *ldc_array, const int *group_count, const int *group_size)
    {
        return H2OPUS_FMANGLE(sgemm_batch, SGEMM_BATCH)(transa_array, transb_array, m_array, n_array, k_array,
                                                        alpha_array, a_array, lda_array, b_array, ldb_array, beta_array,
                                                        c_array, ldc_array, group_count, group_size);
    }

    extern void H2OPUS_FMANGLE(dgemm_batch, DGEMM_BATCH)(const char *, const char *, const int *, const int *,
                                                         const int *, const double *, const double **, const int *,
                                                         const double **, const int *, const double *, double **,
                                                         const int *, const int *, const int *);
    inline void h2opus_fbl_dgemm_batch(const char *transa_array, const char *transb_array, const int *m_array,
                                       const int *n_array, const int *k_array, const double *alpha_array,
                                       const double **a_array, const int *lda_array, const double **b_array,
                                       const int *ldb_array, const double *beta_array, double **c_array,
                                       const int *ldc_array, const int *group_count, const int *group_size)
    {
        return H2OPUS_FMANGLE(dgemm_batch, DGEMM_BATCH)(transa_array, transb_array, m_array, n_array, k_array,
                                                        alpha_array, a_array, lda_array, b_array, ldb_array, beta_array,
                                                        c_array, ldc_array, group_count, group_size);
    }

#ifdef __cplusplus
}
#endif

#endif
