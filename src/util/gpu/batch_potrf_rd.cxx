#include <h2opus/core/h2opus_eps.h>
#include <h2opus/util/batch_matrix_add.h>
#include <h2opus/util/gpu_util.h>
#include <algorithm>

#define POTRF_RD_BS 32
#define POTRF_RD_CLEAR_R_CPT 8

/////////////////////////////////////////////////////////////////////////////
// Utility functions
/////////////////////////////////////////////////////////////////////////////
template <class T, int N> __device__ __forceinline__ T warp_max(T a)
{
#pragma unroll
    for (int mask = N / 2; mask > 0; mask /= 2)
    {
        T b = __shfl_xor_sync(0xFFFFFFFF, a, mask);
        if (b > a)
            a = b;
    }
    return a;
}

inline void magmablas_gemm_batched_core(magma_trans_t transA, magma_trans_t transB, int m, int n, int k, double alpha,
                                        double **dA_array, int Ai, int Aj, int ldda, double **dB_array, int Bi, int Bj,
                                        int lddb, double beta, double **dC_array, int Ci, int Cj, int lddc,
                                        int batchCount, magma_queue_t queue)
{
    magmablas_dgemm_batched_core(transA, transB, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta,
                                 dC_array, Ci, Cj, lddc, batchCount, queue);
}

inline void magmablas_gemm_batched_core(magma_trans_t transA, magma_trans_t transB, int m, int n, int k, float alpha,
                                        float **dA_array, int Ai, int Aj, int ldda, float **dB_array, int Bi, int Bj,
                                        int lddb, float beta, float **dC_array, int Ci, int Cj, int lddc,
                                        int batchCount, magma_queue_t queue)
{
    magmablas_sgemm_batched_core(transA, transB, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta,
                                 dC_array, Ci, Cj, lddc, batchCount, queue);
}

template <class T> __global__ void potrf_rd_batch_clear_R_kernel(int dim, T **A_ptrs, int lda)
{
    int op_id = blockIdx.z;

    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
    int col_index = (blockDim.y * blockIdx.y + threadIdx.y) * POTRF_RD_CLEAR_R_CPT;

    if (row_index >= dim || col_index >= dim)
        return;

    T *A = A_ptrs[op_id];
    A += row_index + col_index * lda;

#pragma unroll
    for (int j = 0; j < POTRF_RD_CLEAR_R_CPT; j++)
        if (j + col_index < dim && row_index < j + col_index)
            A[j * lda] = 0;
}

template <class T> void potrf_rd_batch_clear_R(int dim, T **A_ptrs, int lda, int num_ops, h2opusComputeStream_t stream)
{
    int max_thread_y = 8;
    int thread_x = WARP_SIZE, thread_y = std::min(max_thread_y, iDivUp(dim, POTRF_RD_CLEAR_R_CPT));
    int grid_x = iDivUp(dim, thread_x), grid_y = iDivUp(dim, thread_y * POTRF_RD_CLEAR_R_CPT);

    dim3 dimBlock(thread_x, thread_y, 1);
    dim3 dimGrid(grid_x, grid_y, num_ops);

    potrf_rd_batch_clear_R_kernel<T><<<dimGrid, dimBlock, 0, stream->getCudaStream()>>>(dim, A_ptrs, lda);

    gpuErrchk(cudaGetLastError());
}

/////////////////////////////////////////////////////////////////////////////
// Diagonal block cholesky
// Supports BS 16, 32
/////////////////////////////////////////////////////////////////////////////
template <class T, int BS>
__global__ void potrf_rd_batch_kernel(int dim, T **A_batch, int lda, int row_offset, int col_offset, int num_ops)
{
    extern __shared__ char sdata[];

    int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    int op_id = thread_index / BS;

    if (op_id >= num_ops)
        return;

    // Get the local thread data within the block
    int local_op = threadIdx.x / BS;
    int tid = threadIdx.x % BS;
    int warp_id = threadIdx.y;
    int num_warps = blockDim.y;
    int cpt = iDivUp(dim, num_warps);
    int shared_ld = WARP_SIZE;

    T *A = A_batch[op_id] + row_offset + col_offset * lda;
    volatile T *shared_A = (T *)sdata + BS * local_op;

    // Load the matrix into shared memory
    if (tid < dim)
    {
        for (int i = 0; i < cpt; i++)
        {
            int col_index = warp_id + i * num_warps;
            if (col_index < dim)
                shared_A[tid + col_index * shared_ld] = A[tid + col_index * lda];
        }
    }
    __syncthreads();

    // Get the max of the diagonal and scale it by eps to get the tolerance for rank deficiency
    T tol = warp_max<T, BS>(tid < dim ? shared_A[tid + tid * shared_ld] : 0) * H2OpusEpsilon<T>::eps;
    __syncthreads();

    for (int k = 0; k < dim; k++)
    {
        // If the pivot is less than the tolerance, zero it and its column out
        T pivot = shared_A[k + k * shared_ld];

        if (warp_id == 0)
        {
            if (pivot <= tol)
                shared_A[tid + k * shared_ld] = 0;
            else
            {
                // Compute the diagonal
                if (tid == k)
                    shared_A[k + k * shared_ld] = sqrt(pivot);
                // Update the column
                if (tid > k)
                    shared_A[tid + k * shared_ld] /= shared_A[k + k * shared_ld];
            }
        }
        __syncthreads();

        // Update the trailing submatrix
        if (tid > k)
        {
            T A_ik = shared_A[tid + k * shared_ld];
            cpt = iDivUp(dim - k - 1, num_warps);

            for (int j = 0; j < cpt; j++)
            {
                int col_index = k + 1 + warp_id + j * num_warps;
                if (col_index < dim)
                    shared_A[tid + col_index * shared_ld] -= shared_A[col_index + k * shared_ld] * A_ik;
            }
        }
        __syncthreads();
    }

    // Flush results to global memory
    cpt = iDivUp(dim, num_warps);

    if (tid < dim)
    {
        for (int i = 0; i < cpt; i++)
        {
            int col_index = warp_id + i * num_warps;
            if (col_index < dim)
            {
                // Zero out upper triangular section
                T val = (tid >= col_index ? shared_A[tid + col_index * shared_ld] : 0);
                A[tid + col_index * lda] = val;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
// 16x16 or 32x32 potrf kernel
/////////////////////////////////////////////////////////////////////////////
template <class T>
void potrf_rd_batch_template(int dim, T **A_batch, int lda, int row_offset, int col_offset, int num_ops,
                             h2opusComputeStream_t stream)
{
    assert(dim <= WARP_SIZE);

    int bs = upper_power_of_two(dim);
    int ops_per_block = WARP_SIZE / bs;

    if (ops_per_block > num_ops)
        ops_per_block = num_ops;
    int warps_per_op = 8 * bs / WARP_SIZE;
    if (warps_per_op < 1)
        warps_per_op = 1;

    int threads_per_block = ops_per_block * bs;
    int smem_per_block = ops_per_block * (WARP_SIZE * bs) * sizeof(T);
    int thread_blocks = iDivUp(num_ops, ops_per_block);

    dim3 dimBlock(threads_per_block, warps_per_op);
    dim3 dimGrid(thread_blocks, 1);

    switch (bs)
    {
    case 1:
        potrf_rd_batch_kernel<T, 1><<<dimGrid, dimBlock, smem_per_block, stream->getCudaStream()>>>(
            dim, A_batch, lda, row_offset, col_offset, num_ops);
        break;
    case 2:
        potrf_rd_batch_kernel<T, 2><<<dimGrid, dimBlock, smem_per_block, stream->getCudaStream()>>>(
            dim, A_batch, lda, row_offset, col_offset, num_ops);
        break;
    case 4:
        potrf_rd_batch_kernel<T, 4><<<dimGrid, dimBlock, smem_per_block, stream->getCudaStream()>>>(
            dim, A_batch, lda, row_offset, col_offset, num_ops);
        break;
    case 8:
        potrf_rd_batch_kernel<T, 8><<<dimGrid, dimBlock, smem_per_block, stream->getCudaStream()>>>(
            dim, A_batch, lda, row_offset, col_offset, num_ops);
        break;
    case 16:
        potrf_rd_batch_kernel<T, 16><<<dimGrid, dimBlock, smem_per_block, stream->getCudaStream()>>>(
            dim, A_batch, lda, row_offset, col_offset, num_ops);
        break;
    case 32:
        potrf_rd_batch_kernel<T, 32><<<dimGrid, dimBlock, smem_per_block, stream->getCudaStream()>>>(
            dim, A_batch, lda, row_offset, col_offset, num_ops);
        break;
    default:
        printf("potrf_rd_batch_template: Invalid block size %d!\n", bs);
        break;
    }

    gpuErrchk(cudaGetLastError());
}

/////////////////////////////////////////////////////////////////////////////
// 16x16 trsm kernel
/////////////////////////////////////////////////////////////////////////////
// Rows and cols are the total number of rows and columns involved in the trsm
// not the rows and cols of A
// col_offset is the offset into the columns of A
// row_offset_A is the offset into the rows of A
// L = A(row_offset_A:row_offset_A+cols, col_offset:col_offset+cols) is assumed to be lower triangular
// row_offset_B is the offset into the rows of A for the columns that are being solved for
// B = A(row_offset_B:row_offset_A+rows, col_offset:col_offset+cols)
// solves B = B * L^-T
template <class T, int BS>
__global__ void trsm_rd_leaf_kernel(int rows, int cols, T **A_batch, int lda, int col_offset, int row_offset_A,
                                    int row_offset_B, int num_ops)
{
    int op_id = blockDim.y * blockIdx.y + threadIdx.y;
    if (op_id >= num_ops)
        return;

    int B_row_offset = blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    int row_index = B_row_offset + tid;

    // fetch all operation data
    T *A = A_batch[op_id] + row_offset_A + col_offset * lda;
    T *B = A_batch[op_id] + row_offset_B + col_offset * lda + B_row_offset;

    // Store the data for B in registers and the data for A in shared memory
    T B_reg[BS];
    extern __shared__ char sdata[];
    T *A_shared = (T *)sdata + threadIdx.y * BS * BS;

    if (tid < cols)
        for (int i = 0; i < cols; i++)
            A_shared[tid + i * BS] = A[tid + i * lda];
    __syncthreads();

    if (row_index >= rows)
        return;

#pragma unroll
    for (int i = 0; i < BS; i++)
        B_reg[i] = (i < cols ? B[row_index + i * lda] : 0);

// Now do the triangular solve
#pragma unroll
    for (int j = 0; j < BS; j++)
    {
        if (j >= cols || A_shared[j + j * BS] == 0)
        {
            B_reg[j] = 0;
            continue;
        }

#pragma unroll
        for (int k = 0; k < j; k++)
            B_reg[j] -= A_shared[j + k * BS] * B_reg[k];

        B_reg[j] /= A_shared[j + j * BS];
    }

// Flush data back to global memory
#pragma unroll
    for (int i = 0; i < BS; i++)
        if (i < cols)
            B[row_index + i * lda] = B_reg[i];
}

template <class T>
inline void trsm_rd_leaf_batch_template(int rows, int cols, T **A_batch, int lda, int col_offset, int row_offset_A,
                                        int row_offset_B, int num_ops, h2opusComputeStream_t stream)
{
    // TODO: need some serious optimization for subwarp computations
    const int max_block_threads = 128;

    int bs = upper_power_of_two(cols);

    int block_thread_x = max_block_threads;
    int block_thread_y = 1;

    if (block_thread_x > rows)
    {
        block_thread_x = rows;
        block_thread_y = max_block_threads / rows;

        // Don't go overboard with the number of ops per block
        // Since we might not have enough shared memory
        if (block_thread_y > 16)
            block_thread_y = 16;
    }

    // Make sure we have enough threads to load the triangular block
    if (block_thread_x < cols)
        block_thread_x = cols;

    int grid_x = iDivUp(rows, block_thread_x);
    int grid_y = iDivUp(num_ops, block_thread_y);

    dim3 dimBlock(block_thread_x, block_thread_y);
    dim3 dimGrid(grid_x, grid_y);

    int smem_per_block = block_thread_y * bs * bs * sizeof(T);

    switch (bs)
    {
    case 1:
        trsm_rd_leaf_kernel<T, 1><<<dimGrid, dimBlock, smem_per_block, stream->getCudaStream()>>>(
            rows, cols, A_batch, lda, col_offset, row_offset_A, row_offset_B, num_ops);
        break;
    case 2:
        trsm_rd_leaf_kernel<T, 2><<<dimGrid, dimBlock, smem_per_block, stream->getCudaStream()>>>(
            rows, cols, A_batch, lda, col_offset, row_offset_A, row_offset_B, num_ops);
        break;
    case 4:
        trsm_rd_leaf_kernel<T, 4><<<dimGrid, dimBlock, smem_per_block, stream->getCudaStream()>>>(
            rows, cols, A_batch, lda, col_offset, row_offset_A, row_offset_B, num_ops);
        break;
    case 8:
        trsm_rd_leaf_kernel<T, 8><<<dimGrid, dimBlock, smem_per_block, stream->getCudaStream()>>>(
            rows, cols, A_batch, lda, col_offset, row_offset_A, row_offset_B, num_ops);
        break;
    case 16:
        trsm_rd_leaf_kernel<T, 16><<<dimGrid, dimBlock, smem_per_block, stream->getCudaStream()>>>(
            rows, cols, A_batch, lda, col_offset, row_offset_A, row_offset_B, num_ops);
        break;
    default:
        printf("trsm_rd_leaf_batch_template: Invalid block size %d!\n", bs);
        break;
    }
    gpuErrchk(cudaGetLastError());
}

template <class T>
inline void trsm_rd_batch_template(int rows, int cols, T **A_batch, int lda, int col_offset, int row_offset_A,
                                   int row_offset_B, int num_ops, h2opusComputeStream_t stream)
{
    if (rows == 0 || cols == 0)
        return;

    const int POTRF_RD_TRSM_LEAF_SIZE = 16;

    if (cols <= POTRF_RD_TRSM_LEAF_SIZE)
        trsm_rd_leaf_batch_template<T>(rows, cols, A_batch, lda, col_offset, row_offset_A, row_offset_B, num_ops,
                                       stream);
    else
    {
        int n1 = upper_power_of_two(cols) / 2;
        int n2 = cols - n1;

        // First trsm of the top diagonal block: B1 = B1 * A11^{-T}
        trsm_rd_batch_template(rows, n1, A_batch, lda, col_offset, row_offset_A, row_offset_B, num_ops, stream);

        // Update the right block of B with a gemm
        // B2 = B2 - B1 * A12^T
        magma_queue_t queue = stream->getMagmaQueue();
        magmablas_gemm_batched_core(MagmaNoTrans, MagmaTrans, rows, n2, n1, (T)(-1), A_batch, row_offset_B, col_offset,
                                    lda, A_batch, row_offset_A + n1, col_offset, lda, (T)1, A_batch, row_offset_B,
                                    col_offset + n1, lda, num_ops, queue);

        // Final trsm of the lower diagonal block: B2 = B2 * A22^{-T}
        trsm_rd_batch_template(rows, n2, A_batch, lda, col_offset + n1, row_offset_A + n1, row_offset_B, num_ops,
                               stream);
    }
}

template <class T>
inline void potrf_rd_batch_template(int dim, T **A_batch, int lda, int num_ops, h2opusComputeStream_t stream)
{
    const int BS = POTRF_RD_BS;
    int k = 0;

    while (k < dim)
    {
        int bs = std::min(BS, dim - k);

        // Diagonal block potrf
        potrf_rd_batch_template<T>(bs, A_batch, lda, k, k, num_ops, stream);

        // Trsm on block column k
        int B_rows = dim - k - bs;
        if (B_rows > 0)
        {
            trsm_rd_batch_template<T>(B_rows, bs, A_batch, lda, k, k, k + bs, num_ops, stream);

            // Syrk on trailing sub-matrix (using gemms for now since syrk offset routines are not exposed)
            magma_queue_t queue = stream->getMagmaQueue();
            magmablas_gemm_batched_core(MagmaNoTrans, MagmaTrans, B_rows, B_rows, bs, (T)(-1), A_batch, k + bs, k, lda,
                                        A_batch, k + bs, k, lda, (T)1, A_batch, k + bs, k + bs, lda, num_ops, queue);
        }
        k += bs;
    }

    potrf_rd_batch_clear_R<T>(dim, A_batch, lda, num_ops, stream);
}

void potrf_rd_batch(int dim, double **A_batch, int lda, int num_ops, h2opusComputeStream_t stream)
{
    potrf_rd_batch_template<double>(dim, A_batch, lda, num_ops, stream);
}

void potrf_rd_batch(int dim, float **A_batch, int lda, int num_ops, h2opusComputeStream_t stream)
{
    potrf_rd_batch_template<float>(dim, A_batch, lda, num_ops, stream);
}
