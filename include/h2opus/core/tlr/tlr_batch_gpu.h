#ifndef __H2OPUS_TLR_BATCH_GPU_H__
#define __H2OPUS_TLR_BATCH_GPU_H__

#ifdef H2OPUS_USE_GPU

#include <h2opus/util/gpu_util.h>

#define COLS_PER_THREAD 8
#define MAX_THREAD_Y 8
#define REDUCTION_PAR_BLOCKS_MAX_SIZE 8192
#define REDUCTION_MIN_BLOCKS_PER_THREAD 4
#define REDUCTION_MIN_BUFFERS 8

template <class T, class FunctionGen>
__global__ void generateDenseBlocksKernel(T **block_ptrs, int block_size, int block_row_start, int block_column,
                                          int blockCount, int n, FunctionGen func_gen)
{
    // extern __shared__ char sdata[];

    int op_id = blockIdx.z;
    T *A = block_ptrs[op_id];
    // T *pt_data = func_gen.getData();
    // int dim = func_gen.getDim();
    int block = op_id + block_row_start;

    if (A == NULL || block == block_column)
        return;

    int bx = blockDim.x;
    int by = blockDim.y * COLS_PER_THREAD;
    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;

    // Grab the current operation pointer and dimension data
    int row_start = block * block_size;
    int col_start = (block_column == H2OPUS_TLR_BLOCK_GEN_DIAGONAL ? row_start : block_column * block_size);

    // Advance the row pointer according to the block index
    int row_offset = bx * blockIdx.x + lane_id;
    int col_offset = (blockDim.y * blockIdx.y + warp_id) * COLS_PER_THREAD;

    A += row_offset + col_offset * block_size;

    row_start += row_offset;
    col_start += col_offset;

    // Grab point data for this thread
    // T* pt_i = (T*)sdata;
    // T* pt_j = pt_i + bx * dim;

    // Load the row points
    // if(row_start < n && warp_id == 0)
    //     for(int d = 0; d < dim; d++)
    //         pt_i[lane_id + d * bx] = (pt_data ? pt_data[row_start + d * n] : 0);
    //
    // // Load the column points
    // if(warp_id == 0)
    // {
    //     int load_loops = (by + bx - 1) / bx;
    //     if(pt_data)
    //         pt_data += col_start;
    //
    //     for(int i = 0; i < load_loops; i++)
    //     {
    //         int pt_index = i * bx + lane_id;
    //         if(pt_index + col_start < n && pt_index < by)
    //             for(int d = 0; d < dim; d++)
    //                 pt_j[pt_index + d * by] = (pt_data ? pt_data[pt_index + d * n] : 0);
    //     }
    // }
    //
    // __syncthreads();

    if (row_offset >= block_size || col_offset >= block_size)
        return;

#pragma unroll
    for (int j = 0; j < COLS_PER_THREAD; j++)
    {
        T val = 0;
        if (j + col_start < n && row_start < n)
            val = func_gen(row_start, j + col_start); // (pt_i + lane_id, bx, pt_j + j + warp_id * COLS_PER_THREAD, by,
                                                      // row_start, j + col_start);
        else
            val = (row_start == j + col_start ? 1 : 0);

        if (j + col_offset < block_size)
            A[j * block_size] = val;
    }
}

// Going to serially reduce (num_reduce_buffers - reduction_par_blocks) for each of the num_buffers rows
// into the first reduction_par_blocks blocks of the row in parallel (i.e. parallel per row
// and parallel on reduction_par_blocks blocks)
template <class T>
__global__ void reduceMatrixBuffersParallelKernel(int *rows_batch, int *cols_batch, T **buffer_ptrs, int *ldb_batch,
                                                  int reduction_par_blocks, int buffers_per_block,
                                                  int num_reduce_buffers, int num_buffers)
{
    int buffer_row_id = blockIdx.z % num_buffers;
    int dest_col_id = blockIdx.z / num_buffers;

    // Get dims
    int rows = rows_batch[buffer_row_id];
    int cols = cols_batch[buffer_row_id];

    // Figure out which sub-block this thread block is handling
    int row_offset = blockDim.x * blockIdx.x + threadIdx.x;
    int col_offset = (blockDim.y * blockIdx.y + threadIdx.y) * COLS_PER_THREAD;

    if (row_offset >= rows || col_offset >= cols)
        return;

    int dest_buffer_index = buffer_row_id + dest_col_id * num_buffers;
    int ld_dest = ldb_batch[dest_buffer_index];
    T *dest_ptr = buffer_ptrs[dest_buffer_index];
    dest_ptr += row_offset + col_offset * ld_dest;

    // Accumulate the intermediate sums in registers
    T dest_data[COLS_PER_THREAD];
#pragma unroll
    for (int j = 0; j < COLS_PER_THREAD; j++)
        dest_data[j] = 0;

    // Reduce the buffers serially
    for (int b = 0; b < buffers_per_block; b++)
    {
        int src_col_id = (b + 1) * reduction_par_blocks + dest_col_id;
        if (src_col_id >= num_reduce_buffers)
            break;

        int src_buffer_index = buffer_row_id + src_col_id * num_buffers;
        int ld_src = ldb_batch[src_buffer_index];
        T *src_ptr = buffer_ptrs[src_buffer_index];
        src_ptr += row_offset + col_offset * ld_src;

#pragma unroll
        for (int j = 0; j < COLS_PER_THREAD; j++)
            if (j + col_offset < cols)
                dest_data[j] += src_ptr[j * ld_src];
    }

// Accumulate into global memory
#pragma unroll
    for (int j = 0; j < COLS_PER_THREAD; j++)
        if (j + col_offset < cols)
            dest_ptr[j * ld_dest] += dest_data[j];
}

// Simple serial reduction
template <class T>
__global__ void reduceMatrixBuffersKernel(T beta, T **dest_ptrs, int *ldd_batch, int *rows_batch, int *cols_batch,
                                          T alpha, T **buffer_ptrs, int *ldb_batch, int num_reduce_buffers,
                                          int num_buffers)
{
    int buffer_row_id = blockIdx.z;

    // Get dims
    int rows = rows_batch[buffer_row_id];
    int cols = cols_batch[buffer_row_id];

    // Figure out which sub-block this thread block is handling
    int row_offset = blockDim.x * blockIdx.x + threadIdx.x;
    int col_offset = (blockDim.y * blockIdx.y + threadIdx.y) * COLS_PER_THREAD;

    if (row_offset >= rows || col_offset >= cols)
        return;

    int ld_dest = ldd_batch[buffer_row_id];
    T *dest_ptr = dest_ptrs[buffer_row_id];
    dest_ptr += row_offset + col_offset * ld_dest;

    // Accumulate the intermediate sums in registers
    T dest_data[COLS_PER_THREAD];
#pragma unroll
    for (int j = 0; j < COLS_PER_THREAD; j++)
        dest_data[j] = 0;

    // Reduce the buffers serially
    for (int b = 0; b < num_reduce_buffers; b++)
    {
        int src_buffer_index = buffer_row_id + b * num_buffers;
        int ld_src = ldb_batch[src_buffer_index];
        T *src_ptr = buffer_ptrs[src_buffer_index];
        src_ptr += row_offset + col_offset * ld_src;

#pragma unroll
        for (int j = 0; j < COLS_PER_THREAD; j++)
            if (j + col_offset < cols)
                dest_data[j] += src_ptr[j * ld_src];
    }

// Accumulate into global memory
#pragma unroll
    for (int j = 0; j < COLS_PER_THREAD; j++)
        if (j + col_offset < cols)
            dest_ptr[j * ld_dest] = beta * dest_ptr[j * ld_dest] + alpha * dest_data[j];
}

// GPU
template <class T> struct TLR_Batch<T, H2OPUS_HWTYPE_GPU>
{
    // If block_column == H2OPUS_TLR_BLOCK_GEN_DIAGONAL, then we are generating
    // the diagonal blocks. otherwise we generate the block column of the
    // matrix excluding the diagonal block
    template <class FunctionGen>
    static inline void generateDenseBlocks(T **block_ptrs, int block_size, int block_row_start, int block_column,
                                           int blockCount, int n, FunctionGen &func_gen, h2opusComputeStream_t stream)
    {
        int thread_x = WARP_SIZE, thread_y = std::min(MAX_THREAD_Y, iDivUp(block_size, COLS_PER_THREAD));
        int grid_x = iDivUp(block_size, thread_x), grid_y = iDivUp(block_size, thread_y * COLS_PER_THREAD);

        size_t smem_per_block = 0; //(thread_x + thread_y * COLS_PER_THREAD) * func_gen.getDim() * sizeof(T);

        dim3 dimBlock(thread_x, thread_y, 1);
        dim3 dimGrid(grid_x, grid_y, blockCount);

        generateDenseBlocksKernel<T, FunctionGen><<<dimGrid, dimBlock, smem_per_block, stream->getCudaStream()>>>(
            block_ptrs, block_size, block_row_start, block_column, blockCount, n, func_gen);

        gpuErrchk(cudaGetLastError());
    }

    // buffer_ptrs is a [num_buffers x num_reduce_buffers] column major matrix of pointers
    // reduce the rows of buffers into dest_ptrs i.e.
    // dest[i] = beta * dest[i] + alpha * sum_{j = 1:num_reduce_buffers} buffer_ptrs[i + j * num_buffers]
    static inline void reduceMatrixBuffers(T beta, T **dest_ptrs, int *ldd_batch, int *rows_batch, int *cols_batch,
                                           T alpha, T **buffer_ptrs, int *ldb_batch, int num_reduce_buffers,
                                           int max_rows, int max_cols, int num_buffers, h2opusComputeStream_t stream)
    {
        int thread_x = WARP_SIZE, thread_y = std::min(MAX_THREAD_Y, iDivUp(max_cols, COLS_PER_THREAD));
        int grid_x = iDivUp(max_rows, thread_x), grid_y = iDivUp(max_cols, thread_y * COLS_PER_THREAD);

        int reduction_par_blocks = iDivUp(REDUCTION_PAR_BLOCKS_MAX_SIZE, max_rows);
        int buffers_per_block = iDivUp(num_reduce_buffers - reduction_par_blocks, reduction_par_blocks);
        dim3 dimBlock(thread_x, thread_y, 1);

        // Only do a parallel reduce on the blocks if there is not enough work to do
        if (num_buffers < REDUCTION_MIN_BUFFERS && buffers_per_block > REDUCTION_MIN_BLOCKS_PER_THREAD)
        {
            int grid_z = num_buffers * reduction_par_blocks;
            dim3 dimGrid(grid_x, grid_y, grid_z);

            reduceMatrixBuffersParallelKernel<T><<<dimGrid, dimBlock, 0, stream->getCudaStream()>>>(
                rows_batch, cols_batch, buffer_ptrs, ldb_batch, reduction_par_blocks, buffers_per_block,
                num_reduce_buffers, num_buffers);

            gpuErrchk(cudaGetLastError());

            num_reduce_buffers = reduction_par_blocks;
        }

        // Finish the accumulation with a serial reduce
        dim3 dimGrid(grid_x, grid_y, num_buffers);

        reduceMatrixBuffersKernel<T><<<dimGrid, dimBlock, 0, stream->getCudaStream()>>>(
            beta, dest_ptrs, ldd_batch, rows_batch, cols_batch, alpha, buffer_ptrs, ldb_batch, num_reduce_buffers,
            num_buffers);

        gpuErrchk(cudaGetLastError());
    }
};

#undef COLS_PER_THREAD
#undef MAX_THREAD_Y
#undef REDUCTION_PAR_BLOCKS_MAX_SIZE
#undef REDUCTION_MIN_BLOCKS_PER_THREAD
#undef REDUCTION_MIN_BUFFERS

#endif

#endif
