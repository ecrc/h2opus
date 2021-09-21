#ifndef __BATCH_BLOCK_SET_H__
#define __BATCH_BLOCK_SET_H__

#include <cublas_v2.h>

void batchBlockSetDiagonal(cudaStream_t stream, int rows, int cols, float **block_ptrs, int ld, float value, int ops);
void batchBlockSetDiagonal(cudaStream_t stream, int rows, int cols, double **block_ptrs, int ld, double value, int ops);
void batchBlockSetDiagonal(cudaStream_t stream, int *rows_batch, int *cols_batch, int max_rows, int max_cols,
                           float **block_ptrs, int *ld_batch, float value, int ops);
void batchBlockSetDiagonal(cudaStream_t stream, int *rows_batch, int *cols_batch, int max_rows, int max_cols,
                           double **block_ptrs, int *ld_batch, double value, int ops);

void batchBlockSetIdentity(cudaStream_t stream, int rows, int cols, float **block_ptrs, int ld, int ops);
void batchBlockSetIdentity(cudaStream_t stream, int rows, int cols, double **block_ptrs, int ld, int ops);
void batchBlockSetIdentity(cudaStream_t stream, int *rows_batch, int *cols_batch, int max_rows, int max_cols,
                           float **block_ptrs, int *ld_batch, int ops);
void batchBlockSetIdentity(cudaStream_t stream, int *rows_batch, int *cols_batch, int max_rows, int max_cols,
                           double **block_ptrs, int *ld_batch, int ops);

void batchBlockSetZero(cudaStream_t stream, int rows, int cols, float **block_ptrs, int ld, int ops);
void batchBlockSetZero(cudaStream_t stream, int rows, int cols, double **block_ptrs, int ld, int ops);
void batchBlockSetZero(cudaStream_t stream, int *rows_batch, int *cols_batch, int max_rows, int max_cols,
                       float **block_ptrs, int *ld_batch, int ops);
void batchBlockSetZero(cudaStream_t stream, int *rows_batch, int *cols_batch, int max_rows, int max_cols,
                       double **block_ptrs, int *ld_batch, int ops);

void batchBlockSetUpperZero(cudaStream_t stream, int m, int n, float **A_ptrs, int lda, int batchCount);
void batchBlockSetUpperZero(cudaStream_t stream, int m, int n, double **A_ptrs, int lda, int batchCount);

#endif
