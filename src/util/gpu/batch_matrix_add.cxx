#include <h2opus/util/batch_matrix_add.h>
#include <h2opus/util/gpu_util.h>
#include <algorithm>

template <class T, class T_ptr, class DimType>
__global__ void batchMatrixAdd_kernel(DimType rows_batch, DimType cols_batch, T alpha, T_ptr A_batch, DimType lda_batch,
                                      int stride_a, T beta, T_ptr B_batch, DimType ldb_batch, int stride_b,
                                      T_ptr C_batch, DimType ldc_batch, int stride_c, int ops)
{
    int op_id = blockIdx.z;
    if (op_id >= ops)
        return;

    T *A_matrix = getOperationPtr<T>(A_batch, op_id, stride_a);
    T *B_matrix = getOperationPtr<T>(B_batch, op_id, stride_b);
    T *C_matrix = getOperationPtr<T>(C_batch, op_id, stride_c);

    if (!A_matrix || !B_matrix || !C_matrix)
        return;

    int thread_row = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_col = blockIdx.y * blockDim.y + threadIdx.y;

    int rows = getOperationDim(rows_batch, op_id);
    int cols = getOperationDim(cols_batch, op_id);

    if (thread_row >= rows || thread_col >= cols)
        return;

    int lda = getOperationDim(lda_batch, op_id);
    int ldb = getOperationDim(ldb_batch, op_id);
    int ldc = getOperationDim(ldc_batch, op_id);

    C_matrix[thread_row + thread_col * ldc] =
        alpha * A_matrix[thread_row + thread_col * lda] + beta * B_matrix[thread_row + thread_col * ldb];
}

template <class T, class T_ptr, class DimType>
void batchMatrixAdd_template(DimType rows_batch, DimType cols_batch, int max_rows, int max_cols, T alpha, T_ptr A_batch,
                             DimType lda_batch, int stride_a, T beta, T_ptr B_batch, DimType ldb_batch, int stride_b,
                             T_ptr C_batch, DimType ldc_batch, int stride_c, int batchCount, cudaStream_t stream)
{
    if (batchCount == 0 || max_rows == 0 || max_cols == 0)
        return;

    int max_thread_x = 64;
    int max_thread_y = 8;

    int thread_x = (max_rows < max_thread_x ? max_rows : max_thread_x);
    int thread_y = (max_cols < max_thread_y ? max_cols : max_thread_y);

    int grid_x = iDivUp(max_rows, thread_x);
    int grid_y = iDivUp(max_cols, thread_y);

    dim3 dimBlock(thread_x, thread_y);

    int batch_increment = MAX_OPS_PER_BATCH;
    int batch_start = 0;
    while (batch_start != batchCount)
    {
        int batch_size = std::min(batch_increment, batchCount - batch_start);
        dim3 dimGrid(grid_x, grid_y, batch_size);

        DimType rows_sub_batch = advanceOperationDim(rows_batch, batch_start);
        DimType cols_sub_batch = advanceOperationDim(cols_batch, batch_start);
        DimType lda_sub_batch = advanceOperationDim(lda_batch, batch_start);
        DimType ldb_sub_batch = advanceOperationDim(ldb_batch, batch_start);
        DimType ldc_sub_batch = advanceOperationDim(ldc_batch, batch_start);

        T_ptr A_sub_batch = advanceOperationPtr(A_batch, batch_start, stride_a);
        T_ptr B_sub_batch = advanceOperationPtr(B_batch, batch_start, stride_b);
        T_ptr C_sub_batch = advanceOperationPtr(C_batch, batch_start, stride_c);

        batchMatrixAdd_kernel<T><<<dimGrid, dimBlock, 0, stream>>>(
            rows_sub_batch, cols_sub_batch, alpha, A_sub_batch, lda_sub_batch, stride_a, beta, B_sub_batch,
            ldb_sub_batch, stride_b, C_sub_batch, ldc_sub_batch, stride_c, batch_size);

        gpuErrchk(cudaGetLastError());

        batch_start += batch_size;
    }
}

// Array of pointers interface
void batchMatrixAdd(int rows, int cols, float alpha, float **A_ptrs, int lda, float beta, float **B_ptrs, int ldb,
                    float **C_ptrs, int ldc, int ops, cudaStream_t stream)
{
    batchMatrixAdd_template<float, float **, int>(rows, cols, rows, cols, alpha, A_ptrs, lda, 0, beta, B_ptrs, ldb, 0,
                                                  C_ptrs, ldc, 0, ops, stream);
}

void batchMatrixAdd(int rows, int cols, double alpha, double **A_ptrs, int lda, double beta, double **B_ptrs, int ldb,
                    double **C_ptrs, int ldc, int ops, cudaStream_t stream)
{
    batchMatrixAdd_template<double, double **, int>(rows, cols, rows, cols, alpha, A_ptrs, lda, 0, beta, B_ptrs, ldb, 0,
                                                    C_ptrs, ldc, 0, ops, stream);
}

void batchMatrixAdd(int *rows_batch, int *cols_batch, int max_rows, int max_cols, float alpha, float **A_ptrs,
                    int *lda_batch, float beta, float **B_ptrs, int *ldb_batch, float **C_ptrs, int *ldc_batch, int ops,
                    cudaStream_t stream)
{
    batchMatrixAdd_template<float, float **, int *>(rows_batch, cols_batch, max_rows, max_cols, alpha, A_ptrs,
                                                    lda_batch, 0, beta, B_ptrs, ldb_batch, 0, C_ptrs, ldc_batch, 0, ops,
                                                    stream);
}

void batchMatrixAdd(int *rows_batch, int *cols_batch, int max_rows, int max_cols, double alpha, double **A_ptrs,
                    int *lda_batch, double beta, double **B_ptrs, int *ldb_batch, double **C_ptrs, int *ldc_batch,
                    int ops, cudaStream_t stream)
{
    batchMatrixAdd_template<double, double **, int *>(rows_batch, cols_batch, max_rows, max_cols, alpha, A_ptrs,
                                                      lda_batch, 0, beta, B_ptrs, ldb_batch, 0, C_ptrs, ldc_batch, 0,
                                                      ops, stream);
}

// Strided interface
void batchMatrixAdd(int rows, int cols, float alpha, float *A_strided, int lda, int stride_a, float beta,
                    float *B_strided, int ldb, int stride_b, float *C_strided, int ldc, int stride_c, int ops,
                    cudaStream_t stream)
{
    batchMatrixAdd_template<float, float *, int>(rows, cols, rows, cols, alpha, A_strided, lda, stride_a, beta,
                                                 B_strided, ldb, stride_b, C_strided, ldc, stride_c, ops, stream);
}

void batchMatrixAdd(int rows, int cols, double alpha, double *A_strided, int lda, int stride_a, double beta,
                    double *B_strided, int ldb, int stride_b, double *C_strided, int ldc, int stride_c, int ops,
                    cudaStream_t stream)
{
    batchMatrixAdd_template<double, double *, int>(rows, cols, rows, cols, alpha, A_strided, lda, stride_a, beta,
                                                   B_strided, ldb, stride_b, C_strided, ldc, stride_c, ops, stream);
}
