#ifndef __SIMPLE_WRAPPERS_H__
#define __SIMPLE_WRAPPERS_H__

inline float norm(float *x, int n)
{
    return cblas_snrm2(n, x, 1);
}
inline double norm(double *x, int n)
{
    return cblas_dnrm2(n, x, 1);
}

inline float dot_product(float *x, float *y, int n)
{
    return cblas_sdot(n, x, 1, y, 1);
}
inline double dot_product(double *x, double *y, int n)
{
    return cblas_ddot(n, x, 1, y, 1);
}

inline void copy_vector(float *dest, float *src, int n)
{
    cblas_scopy(n, src, 1, dest, 1);
}
inline void copy_vector(double *dest, double *src, int n)
{
    cblas_dcopy(n, src, 1, dest, 1);
}
inline void cblas_axpy(int n, float alpha, float *x, int incx, float *y, int incy)
{
    cblas_saxpy(n, alpha, x, incx, y, incy);
}

inline void cblas_axpy(int n, double alpha, double *x, int incx, double *y, int incy)
{
    cblas_daxpy(n, alpha, x, incx, y, incy);
}

inline void cblas_scal(int n, float alpha, float *x, int incx)
{
    cblas_sscal(n, alpha, x, incx);
}

inline void cblas_scal(int n, double alpha, double *x, int incx)
{
    cblas_dscal(n, alpha, x, incx);
}

#endif
