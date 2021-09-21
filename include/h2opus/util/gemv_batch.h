#ifndef __GEMV_BATCH_H__
#define __GEMV_BATCH_H__

// Strided matrix data
void gemv_batch(char transpose, int m, int n, double alpha, double *A_batch, int lda, double *x_batch, double beta,
                double *y_batch, int num_ops, cudaStream_t stream = 0);
void gemv_batch(char transpose, int m, int n, float alpha, float *A_batch, int lda, float *x_batch, float beta,
                float *y_batch, int num_ops, cudaStream_t stream = 0);

// Array of pointers
void gemv_batch(char transpose, int m, int n, double alpha, double **A_batch, int lda, double **x_batch, double beta,
                double **y_batch, int num_ops, cudaStream_t stream = 0);
void gemv_batch(char transpose, int m, int n, float alpha, float **A_batch, int lda, float **x_batch, float beta,
                float **y_batch, int num_ops, cudaStream_t stream = 0);

#endif
