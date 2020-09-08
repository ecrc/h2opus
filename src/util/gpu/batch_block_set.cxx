#include <h2opus/util/batch_block_set.h>
#include <h2opus/util/gpu_util.h>
#include <algorithm>

#define COLS_PER_THREAD 8
#define MAX_THREAD_Y 8

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA kernels
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, class DimType>
__global__ void batchBlockSetDiagonalKernel(DimType rows_batch, DimType cols_batch, T **block_ptrs, DimType ld_batch,
                                            T value, int ops)
{
    int op_id = blockIdx.z;
    if (op_id >= ops)
        return;

    T *block_ptr = block_ptrs[op_id];

    if (block_ptr == NULL)
        return;

    int rows = getOperationDim(rows_batch, op_id);
    int cols = getOperationDim(cols_batch, op_id);
    int ld = getOperationDim(ld_batch, op_id);

    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
    int col_index = (blockDim.y * blockIdx.y + threadIdx.y) * COLS_PER_THREAD;

    if (row_index >= rows || col_index >= cols)
        return;

    block_ptr += row_index + col_index * ld;

#pragma unroll
    for (int j = 0; j < COLS_PER_THREAD; j++)
    {
        if (j + col_index < cols)
        {
            T entry = (row_index == j + col_index ? value : 0);
            block_ptr[j * ld] = entry;
        }
    }
}

template <class T, class DimType>
__global__ void batchBlockSetZeroKernel(DimType rows_batch, DimType cols_batch, T **block_ptrs, DimType ld_batch,
                                        int ops)
{
    int op_id = blockIdx.z;
    if (op_id >= ops)
        return;

    T *block_ptr = block_ptrs[op_id];

    if (block_ptr == NULL)
        return;

    // Grab the current operation pointer and dimension data
    int rows = getOperationDim(rows_batch, op_id);
    int cols = getOperationDim(cols_batch, op_id);
    int ld = getOperationDim(ld_batch, op_id);

    // Advance the row pointer according to the block index
    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
    int col_index = (blockDim.y * blockIdx.y + threadIdx.y) * COLS_PER_THREAD;

    if (row_index >= rows || col_index >= cols)
        return;

    block_ptr += row_index + col_index * ld;

#pragma unroll
    for (int j = 0; j < COLS_PER_THREAD; j++)
        if (j + col_index < cols)
            block_ptr[j * ld] = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Templates
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, class DimType>
void batchBlockSetDiagonalT(cudaStream_t stream, DimType rows_batch, DimType cols_batch, int max_rows, int max_cols,
                            T **block_ptrs, DimType ld_batch, T value, int batchCount)
{
    if (batchCount <= 0 || max_cols <= 0 || max_rows <= 0)
        return;

    int thread_x = WARP_SIZE, thread_y = std::min(MAX_THREAD_Y, iDivUp(max_cols, COLS_PER_THREAD));
    int grid_x = iDivUp(max_rows, thread_x), grid_y = iDivUp(max_cols, thread_y * COLS_PER_THREAD);

    dim3 dimBlock(thread_x, thread_y, 1);

    int batch_increment = MAX_OPS_PER_BATCH;
    int batch_start = 0;
    while (batch_start != batchCount)
    {
        int batch_size = std::min(batch_increment, batchCount - batch_start);
        dim3 dimGrid(grid_x, grid_y, batch_size);

        DimType rows_sub_batch = advanceOperationDim(rows_batch, batch_start);
        DimType cols_sub_batch = advanceOperationDim(cols_batch, batch_start);
        DimType ld_sub_batch = advanceOperationDim(ld_batch, batch_start);
        T **block_sub_ptrs = block_ptrs + batch_start;

        batchBlockSetDiagonalKernel<T><<<dimGrid, dimBlock, 0, stream>>>(rows_sub_batch, cols_sub_batch, block_sub_ptrs,
                                                                         ld_sub_batch, value, batch_size);

        gpuErrchk(cudaGetLastError());

        batch_start += batch_size;
    }
}

template <class T, class DimType>
void batchBlockSetZeroT(cudaStream_t stream, DimType rows_batch, DimType cols_batch, int max_rows, int max_cols,
                        T **block_ptrs, DimType ld_batch, int batchCount)
{
    if (batchCount <= 0 || max_cols <= 0 || max_rows <= 0)
        return;

    int thread_x = WARP_SIZE, thread_y = std::min(MAX_THREAD_Y, iDivUp(max_cols, COLS_PER_THREAD));
    int grid_x = iDivUp(max_rows, thread_x), grid_y = iDivUp(max_cols, thread_y * COLS_PER_THREAD);

    dim3 dimBlock(thread_x, thread_y, 1);

    int batch_increment = MAX_OPS_PER_BATCH;
    int batch_start = 0;
    while (batch_start != batchCount)
    {
        int batch_size = std::min(batch_increment, batchCount - batch_start);
        dim3 dimGrid(grid_x, grid_y, batch_size);

        DimType rows_sub_batch = advanceOperationDim(rows_batch, batch_start);
        DimType cols_sub_batch = advanceOperationDim(cols_batch, batch_start);
        DimType ld_sub_batch = advanceOperationDim(ld_batch, batch_start);
        T **block_sub_ptrs = block_ptrs + batch_start;

        batchBlockSetZeroKernel<T, DimType><<<dimGrid, dimBlock, 0, stream>>>(rows_sub_batch, cols_sub_batch,
                                                                              block_sub_ptrs, ld_sub_batch, batch_size);

        gpuErrchk(cudaGetLastError());

        batch_start += batch_size;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Interface routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void batchBlockSetDiagonal(cudaStream_t stream, int rows, int cols, float **block_ptrs, int ld, float value, int ops)
{
    batchBlockSetDiagonalT<float, int>(stream, rows, cols, rows, cols, block_ptrs, ld, value, ops);
}

void batchBlockSetDiagonal(cudaStream_t stream, int rows, int cols, double **block_ptrs, int ld, double value, int ops)
{
    batchBlockSetDiagonalT<double, int>(stream, rows, cols, rows, cols, block_ptrs, ld, value, ops);
}

void batchBlockSetDiagonal(cudaStream_t stream, int *rows_batch, int *cols_batch, int max_rows, int max_cols,
                           float **block_ptrs, int *ld_batch, float value, int ops)
{
    batchBlockSetDiagonalT<float, int *>(stream, rows_batch, cols_batch, max_rows, max_cols, block_ptrs, ld_batch,
                                         value, ops);
}

void batchBlockSetDiagonal(cudaStream_t stream, int *rows_batch, int *cols_batch, int max_rows, int max_cols,
                           double **block_ptrs, int *ld_batch, double value, int ops)
{
    batchBlockSetDiagonalT<double, int *>(stream, rows_batch, cols_batch, max_rows, max_cols, block_ptrs, ld_batch,
                                          value, ops);
}

void batchBlockSetIdentity(cudaStream_t stream, int rows, int cols, float **block_ptrs, int ld, int ops)
{
    batchBlockSetDiagonalT<float, int>(stream, rows, cols, rows, cols, block_ptrs, ld, 1, ops);
}

void batchBlockSetIdentity(cudaStream_t stream, int rows, int cols, double **block_ptrs, int ld, int ops)
{
    batchBlockSetDiagonalT<double, int>(stream, rows, cols, rows, cols, block_ptrs, ld, 1, ops);
}

void batchBlockSetIdentity(cudaStream_t stream, int *rows_batch, int *cols_batch, int max_rows, int max_cols,
                           float **block_ptrs, int *ld_batch, int ops)
{
    batchBlockSetDiagonalT<float, int *>(stream, rows_batch, cols_batch, max_rows, max_cols, block_ptrs, ld_batch, 1,
                                         ops);
}

void batchBlockSetIdentity(cudaStream_t stream, int *rows_batch, int *cols_batch, int max_rows, int max_cols,
                           double **block_ptrs, int *ld_batch, int ops)
{
    batchBlockSetDiagonalT<double, int *>(stream, rows_batch, cols_batch, max_rows, max_cols, block_ptrs, ld_batch, 1,
                                          ops);
}

void batchBlockSetZero(cudaStream_t stream, int rows, int cols, float **block_ptrs, int ld, int ops)
{
    batchBlockSetZeroT<float, int>(stream, rows, cols, rows, cols, block_ptrs, ld, ops);
}

void batchBlockSetZero(cudaStream_t stream, int rows, int cols, double **block_ptrs, int ld, int ops)
{
    batchBlockSetZeroT<double, int>(stream, rows, cols, rows, cols, block_ptrs, ld, ops);
}

void batchBlockSetZero(cudaStream_t stream, int *rows_batch, int *cols_batch, int max_rows, int max_cols,
                       float **block_ptrs, int *ld_batch, int ops)
{
    batchBlockSetZeroT<float, int *>(stream, rows_batch, cols_batch, max_rows, max_cols, block_ptrs, ld_batch, ops);
}

void batchBlockSetZero(cudaStream_t stream, int *rows_batch, int *cols_batch, int max_rows, int max_cols,
                       double **block_ptrs, int *ld_batch, int ops)
{
    batchBlockSetZeroT<double, int *>(stream, rows_batch, cols_batch, max_rows, max_cols, block_ptrs, ld_batch, ops);
}
