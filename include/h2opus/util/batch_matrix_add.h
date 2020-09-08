#ifndef __BATCH_MATRIX_ADD_H__
#define __BATCH_MATRIX_ADD_H__

#include <cublas_v2.h>

// C = alpha * A + beta * B

// Array of pointers interface
void batchMatrixAdd(int rows, int cols, float alpha, float **A_ptrs, int lda, float beta, float **B_ptrs, int ldb,
                    float **C_ptrs, int ldc, int ops, cudaStream_t stream);

void batchMatrixAdd(int rows, int cols, double alpha, double **A_ptrs, int lda, double beta, double **B_ptrs, int ldb,
                    double **C_ptrs, int ldc, int ops, cudaStream_t stream);

void batchMatrixAdd(int *rows_batch, int *cols_batch, int max_rows, int max_cols, float alpha, float **A_ptrs,
                    int *lda_batch, float beta, float **B_ptrs, int *ldb_batch, float **C_ptrs, int *ldc_batch, int ops,
                    cudaStream_t stream);

void batchMatrixAdd(int *rows_batch, int *cols_batch, int max_rows, int max_cols, double alpha, double **A_ptrs,
                    int *lda_batch, double beta, double **B_ptrs, int *ldb_batch, double **C_ptrs, int *ldc_batch,
                    int ops, cudaStream_t stream);

// Strided interface
void batchMatrixAdd(int rows, int cols, float alpha, float *A_strided, int lda, int stride_a, float beta,
                    float *B_strided, int ldb, int stride_b, float *C_strided, int ldc, int stride_c, int ops,
                    cudaStream_t stream);

void batchMatrixAdd(int rows, int cols, double alpha, double *A_strided, int lda, int stride_a, double beta,
                    double *B_strided, int ldb, int stride_b, double *C_strided, int ldc, int stride_c, int ops,
                    cudaStream_t stream);

#endif
