#ifndef __H2OPUS_TLR_BATCH_CPU_H__
#define __H2OPUS_TLR_BATCH_CPU_H__

// CPU
template <class T> struct TLR_Batch<T, H2OPUS_HWTYPE_CPU>
{
    // If block_column == H2OPUS_TLR_BLOCK_GEN_DIAGONAL, then we are generating
    // the diagonal blocks. otherwise we generate the block column of the
    // matrix excluding the diagonal block
    template <class FunctionGen>
    static inline void generateDenseBlocks(T **block_ptrs, int block_size, int block_row_start, int block_column,
                                           int blockCount, int n, FunctionGen &func_gen, h2opusComputeStream_t stream)
    {
        // T* pt_data = func_gen.getData();
        // int dim = func_gen.getDim();

#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), blockCount))
        for (int b = 0; b < blockCount; b++)
        {
            T *A = block_ptrs[b];
            int block = b + block_row_start;

            if (block == block_column)
                continue;

            int row_start = block * block_size;
            int col_start = (block_column == H2OPUS_TLR_BLOCK_GEN_DIAGONAL ? row_start : block_column * block_size);

            int fill = 0;
            int iup = n - row_start < block_size ? (fill = 1, n - row_start) : block_size;
            int jup = n - col_start < block_size ? (fill = 1, n - col_start) : block_size;
            for (int j = 0; j < jup; j++)
                for (int i = 0; i < iup; i++)
                    A[i + j * block_size] = func_gen(i + row_start, j + col_start);

            if (fill)
            {
                for (int j = jup; j < block_size; j++)
                    for (int i = iup; i < block_size; i++)
                        A[i + j * block_size] = (i == j ? 1 : 0);
            }
        }
    }

    // buffer_ptrs is a [num_buffers x num_reduce_buffers] column major matrix of pointers
    // reduce the rows of buffers into dest_ptrs i.e.
    // dest[i] = beta * dest[i] + alpha * sum_{j = 1:num_reduce_buffers} buffer_ptrs[i + j * num_buffers]
    static inline void reduceMatrixBuffers(T beta, T **dest_ptrs, int *ldd_batch, int *rows_batch, int *cols_batch,
                                           T alpha, T **buffer_ptrs, int *ldb_batch, int num_reduce_buffers,
                                           int max_rows, int max_cols, int num_buffers, h2opusComputeStream_t stream)
    {
        const int minBuffers = 1;

        if (num_buffers >= minBuffers)
        {
#pragma omp parallel for schedule(runtime) num_threads(std::min(stream->getMaxOmpThreads(), num_buffers))
            for (int bi = 0; bi < num_buffers; bi++)
            {
                const int rows = rows_batch[bi];
                const int cols = cols_batch[bi];
                const int ldd = ldd_batch[bi];
                T *dest = dest_ptrs[bi];

                if (beta != 1)
                {
                    for (int j = 0; j < cols; j++)
                        for (int i = 0; i < rows; i++)
                            dest[i + j * ldd] *= beta;
                }

                for (int bj = 0; bj < num_reduce_buffers; bj++)
                {
                    const int src_index = bi + bj * num_buffers;
                    const T *src = buffer_ptrs[src_index];
                    const int ldb = ldb_batch[src_index];

                    if (ldb != rows || ldd != rows)
                    {
                        for (int j = 0; j < cols; j++)
                        {
                            for (int i = 0; i < rows; i++)
                                dest[i + j * ldd] += alpha * src[i + j * ldb];
                        }
                    }
                    else
                    {
                        h2opus_fbl_axpy(rows * cols, alpha, src, 1, dest, 1);
                    }
                }
            }
        }
        else
        {
            for (int bi = 0; bi < num_buffers; bi++)
            {
                int rows = rows_batch[bi], cols = cols_batch[bi];
                int ldd = ldd_batch[bi];
                T *dest = dest_ptrs[bi];

                if (beta != 1)
                {
#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), rows *cols))
                    for (int i = 0; i < rows * cols; i++)
                    {
                        int row = i % rows, col = i / rows;
                        dest[row + col * ldd] *= beta;
                    }
                }

                for (int bj = 0; bj < num_reduce_buffers; bj++)
                {
                    int src_index = bi + bj * num_buffers;
                    T *src = buffer_ptrs[src_index];
                    int ldb = ldb_batch[src_index];

#pragma omp parallel for schedule(static) num_threads(std::min(stream->getMaxOmpThreads(), rows *cols))
                    for (int i = 0; i < rows * cols; i++)
                    {
                        int row = i % rows, col = i / rows;
                        int index_d = row + col * ldd, index_b = row + col * ldb;
                        dest[index_d] += alpha * src[index_b];
                    }
                }
            }
        }
    }
};

#endif
