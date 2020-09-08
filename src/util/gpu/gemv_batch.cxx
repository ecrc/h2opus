#include <h2opus/util/gemv_batch.h>
#include <h2opus/util/gpu_util.h>

#define GEMV_LOAD(x) __ldg(&(x))

template <class T, class T_ptr, int row_blocks>
__global__ void gemvt_batch(int m, int n, T alpha, T_ptr A_batch, T_ptr x_batch, T beta, T_ptr y_batch, int num_ops,
                            int ops_per_block)
{
    int lane_id = threadIdx.x % WARP_SIZE;
    int block_col = threadIdx.y;
    int op_id = blockIdx.x * ops_per_block + threadIdx.x / WARP_SIZE;

    if (op_id >= num_ops)
        return;

    int cols_per_thread = iDivUp(n, blockDim.y);
    int col_start = cols_per_thread * block_col;
    if (col_start + cols_per_thread > n)
        cols_per_thread = n - col_start;

    T *A = getOperationPtr<T>(A_batch, op_id, m * n);
    T *x = getOperationPtr<T>(x_batch, op_id, m);
    T *y = getOperationPtr<T>(y_batch, op_id, n);

    A += m * col_start;
    y += col_start;

    T x_data[row_blocks];
#pragma unroll
    for (int i = 0; i < row_blocks; i++)
        x_data[i] = alpha * (i * WARP_SIZE + lane_id < m ? GEMV_LOAD(x[i * WARP_SIZE + lane_id]) : 0);

    // Loop over the column blocks
    for (int col = 0; col < cols_per_thread; col++)
    {
        T product = 0;
// Loop over the row blocks
#pragma unroll
        for (int i = 0; i < row_blocks; i++)
        {
            T data = i;
            if (i * WARP_SIZE + lane_id < m)
                data = GEMV_LOAD(A[i * WARP_SIZE + lane_id]);
            product += data * x_data[i];
        }
        product = warpReduceSum(product);

        if (lane_id == 0)
            y[col] = beta * y[col] + product;
        A += m;
    }
}

template <class T, class T_ptr>
__global__ void gemvn_batch(int m, int n, T alpha, T_ptr A_batch, T_ptr x_batch, T beta, T_ptr y_batch, int num_ops,
                            int ops_per_block)
{
    extern __shared__ char sdata[];

    int thread_id = threadIdx.x % m;
    int local_op_id = threadIdx.x / m;
    int op_id = blockIdx.x * ops_per_block + local_op_id;

    if (op_id >= num_ops)
        return;

    volatile T *vec_data = (T *)(sdata) + n * local_op_id;

    T *A = getOperationPtr<T>(A_batch, op_id, m * n);
    T *x = getOperationPtr<T>(x_batch, op_id, n);
    T *y = getOperationPtr<T>(y_batch, op_id, m);

    // First load in the vector from global memory
    int c = thread_id;
    while (c < n)
    {
        vec_data[c] = alpha * GEMV_LOAD(x[c]);
        c += m;
    }
    __syncthreads();

    // Now get the dot product of the row with the vector
    T dotp = 0;
    T *A_ptr = A + thread_id;
    //#pragma unroll
    for (int i = 0; i < n; i++)
    {
        dotp += GEMV_LOAD(*A_ptr) * vec_data[i];
        A_ptr += m;
    }

    // Now flush the results to global memory
    y[thread_id] = beta * y[thread_id] + dotp;
}

template <class T, class T_ptr>
void gemv_batch(char transpose, int m, int n, T alpha, T_ptr A_batch, T_ptr x_batch, T beta, T_ptr y_batch, int num_ops,
                cudaStream_t stream)
{
    if (m <= 0 || n <= 0 || num_ops <= 0)
        return;

    if (transpose == 'T')
    {
        int ops_per_block = 4;
        int row_blocks = iDivUp(m, WARP_SIZE);

        int warps_per_op = 1;
        if (num_ops < 1000 && num_ops > 100)
        {
            ops_per_block = 1;
            warps_per_op = (4 < n ? 4 : n);
        }
        else if (num_ops <= 100)
        {
            ops_per_block = 1;
            warps_per_op = (16 < n ? 16 : n);
        }

        dim3 dimBlock(WARP_SIZE * ops_per_block, warps_per_op);
        dim3 dimGrid(iDivUp(num_ops, ops_per_block));

        switch (row_blocks)
        {
        case 1:
            gemvt_batch<T, T_ptr, 1><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                       num_ops, ops_per_block);
            break;
        case 2:
            gemvt_batch<T, T_ptr, 2><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                       num_ops, ops_per_block);
            break;
        case 3:
            gemvt_batch<T, T_ptr, 3><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                       num_ops, ops_per_block);
            break;
        case 4:
            gemvt_batch<T, T_ptr, 4><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                       num_ops, ops_per_block);
            break;
        case 5:
            gemvt_batch<T, T_ptr, 5><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                       num_ops, ops_per_block);
            break;
        case 6:
            gemvt_batch<T, T_ptr, 6><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                       num_ops, ops_per_block);
            break;
        case 7:
            gemvt_batch<T, T_ptr, 7><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                       num_ops, ops_per_block);
            break;
        case 8:
            gemvt_batch<T, T_ptr, 8><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                       num_ops, ops_per_block);
            break;
        case 9:
            gemvt_batch<T, T_ptr, 9><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                       num_ops, ops_per_block);
            break;
        case 10:
            gemvt_batch<T, T_ptr, 10><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                        num_ops, ops_per_block);
            break;
        case 11:
            gemvt_batch<T, T_ptr, 11><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                        num_ops, ops_per_block);
            break;
        case 12:
            gemvt_batch<T, T_ptr, 12><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                        num_ops, ops_per_block);
            break;
        case 13:
            gemvt_batch<T, T_ptr, 13><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                        num_ops, ops_per_block);
            break;
        case 14:
            gemvt_batch<T, T_ptr, 14><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                        num_ops, ops_per_block);
            break;
        case 15:
            gemvt_batch<T, T_ptr, 15><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                        num_ops, ops_per_block);
            break;
        case 16:
            gemvt_batch<T, T_ptr, 16><<<dimGrid, dimBlock, 0, stream>>>(m, n, alpha, A_batch, x_batch, beta, y_batch,
                                                                        num_ops, ops_per_block);
            break;
        default:
            printf("GEMVT batch: Invalid row count %d\n", m);
        }

        gpuErrchk(cudaGetLastError());
    }
    else
    {
        int ops_per_block = 4;
        if (ops_per_block * m > 1024)
            ops_per_block = 1024 / m;

        size_t smem_per_block = ops_per_block * n * sizeof(T);
        dim3 dimBlock(m * ops_per_block);
        dim3 dimGrid(iDivUp(num_ops, ops_per_block));

        gemvn_batch<T, T_ptr><<<dimGrid, dimBlock, smem_per_block, stream>>>(m, n, alpha, A_batch, x_batch, beta,
                                                                             y_batch, num_ops, ops_per_block);

        gpuErrchk(cudaGetLastError());
    }
}

void gemv_batch(char transpose, int m, int n, float alpha, float **A_batch, float **x_batch, float beta,
                float **y_batch, int num_ops, cudaStream_t stream)
{
    gemv_batch<float, float **>(transpose, m, n, alpha, A_batch, x_batch, beta, y_batch, num_ops, stream);
}

void gemv_batch(char transpose, int m, int n, double alpha, double **A_batch, double **x_batch, double beta,
                double **y_batch, int num_ops, cudaStream_t stream)
{
    gemv_batch<double, double **>(transpose, m, n, alpha, A_batch, x_batch, beta, y_batch, num_ops, stream);
}

void gemv_batch(char transpose, int m, int n, float alpha, float *A_batch, float *x_batch, float beta, float *y_batch,
                int num_ops, cudaStream_t stream)
{
    gemv_batch<float, float *>(transpose, m, n, alpha, A_batch, x_batch, beta, y_batch, num_ops, stream);
}

void gemv_batch(char transpose, int m, int n, double alpha, double *A_batch, double *x_batch, double beta,
                double *y_batch, int num_ops, cudaStream_t stream)
{
    gemv_batch<double, double *>(transpose, m, n, alpha, A_batch, x_batch, beta, y_batch, num_ops, stream);
}
