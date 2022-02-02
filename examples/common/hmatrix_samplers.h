#ifndef __HMATRIX_SAMPLERS_H__
#define __HMATRIX_SAMPLERS_H__

#include <h2opus.h>

#ifdef H2OPUS_USE_GPU
#include <cusparse.h>
#endif

#ifdef H2OPUS_USE_MKL
#include <mkl_spblas.h>
#endif

#define ONE_NORM_EST_MAX_SAMPLES 20

/////////////////////////////////////////////////////////////////////////////
// Samplers for various matrix formats
/////////////////////////////////////////////////////////////////////////////
class BasicHMatrixSampler : public HMatrixSampler
{
  public:
    virtual int getMatrixDim() = 0;
    virtual void sample(H2Opus_Real *, H2Opus_Real *, int) = 0;
    virtual H2Opus_Real get1Norm() = 0;
    H2Opus_Real *compute() // XXX Only for host
    {
        int n = getMatrixDim();
        size_t mem = (size_t)n * (size_t)n * sizeof(H2Opus_Real);
        H2Opus_Real *o = (H2Opus_Real *)malloc(mem);
        thrust::host_vector<H2Opus_Real> I(n, 0.0);
        for (int i = 0; i < n; i++)
        {
            I[i] = 1.0;
            if (i)
                I[i - 1] = 0.0;
            sample(vec_ptr(I), o + n * i, 1);
        }
        return o;
    }
};

template <int hw> class LowRankSampler : public BasicHMatrixSampler
{
  private:
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

    RealVector temp_buffer;
    H2Opus_Real *U, *V;
    int n, rank, ldu, ldv;
    h2opusHandle_t h2opus_handle;
    h2opusComputeStream_t main_stream;
    H2Opus_Real one_norm_A;

  public:
    LowRankSampler(H2Opus_Real *U, H2Opus_Real *V, int n, int rank, h2opusHandle_t h2opus_handle)
    {
        this->U = U;
        this->V = V;
        this->n = n;
        this->ldu = n;
        this->ldv = n;
        this->rank = rank;
        this->h2opus_handle = h2opus_handle;
        this->main_stream = h2opus_handle->getMainStream();

        this->one_norm_A = sampler_1_norm<H2Opus_Real, hw>(this, n, ONE_NORM_EST_MAX_SAMPLES, h2opus_handle);
    }

    LowRankSampler(H2Opus_Real *U, int ldu, H2Opus_Real *V, int ldv, int n, int rank, h2opusHandle_t h2opus_handle)
    {
        this->U = U;
        this->V = V;
        this->n = n;
        this->ldu = ldu;
        this->ldv = ldv;
        this->rank = rank;
        this->h2opus_handle = h2opus_handle;
        this->main_stream = h2opus_handle->getMainStream();

        this->one_norm_A = sampler_1_norm<H2Opus_Real, hw>(this, n, ONE_NORM_EST_MAX_SAMPLES, h2opus_handle);
    }

    void sample(H2Opus_Real *input, H2Opus_Real *output, int samples)
    {
        if (temp_buffer.size() < (size_t)n * samples)
            temp_buffer.resize(n * samples);

        blas_gemm<H2Opus_Real, hw>(main_stream, H2Opus_Trans, H2Opus_NoTrans, rank, samples, n, 1, V, ldv, input, n, 0,
                                   vec_ptr(temp_buffer), n);

        blas_gemm<H2Opus_Real, hw>(main_stream, H2Opus_NoTrans, H2Opus_NoTrans, n, samples, rank, 1, U, ldu,
                                   vec_ptr(temp_buffer), n, 0, output, n);
    }

    int getMatrixDim()
    {
        return n;
    }

    H2Opus_Real get1Norm()
    {
        return one_norm_A;
    }
};

template <int hw> class DenseSampler : public BasicHMatrixSampler
{
  private:
    typedef typename VectorContainer<hw, int>::type IntVector;
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

    H2Opus_Real *M;
    int ldm, n;
    H2Opus_Real alpha, one_norm_A;

    IntVector index_map;
    RealVector temp_buffer_A, temp_buffer_B;

    h2opusHandle_t h2opus_handle;
    h2opusComputeStream_t main_stream;

  public:
    // Index map is assumed to be on the CPU
    DenseSampler(H2Opus_Real *M, int ldm, int n, H2Opus_Real alpha, int *index_map, h2opusHandle_t h2opus_handle)
    {
        this->M = M;
        this->ldm = ldm;
        this->n = n;
        this->alpha = alpha;

        if (index_map)
            copyVector(this->index_map, index_map, n, H2OPUS_HWTYPE_CPU);

        this->h2opus_handle = h2opus_handle;
        this->main_stream = h2opus_handle->getMainStream();
        this->one_norm_A = sampler_1_norm<H2Opus_Real, hw>(this, n, ONE_NORM_EST_MAX_SAMPLES, h2opus_handle);
    }

    void sample(H2Opus_Real *input, H2Opus_Real *output, int samples)
    {
        if (temp_buffer_A.size() < (size_t)n * samples)
            temp_buffer_A.resize(n * samples);
        if (temp_buffer_B.size() < (size_t)n * samples)
            temp_buffer_B.resize(n * samples);

        // output = P*(M + alpha * I)*P^t*input

        // ta = P^t*input
        H2Opus_Real *permuted_input = input;
        H2Opus_Real *permuted_output = output;

        if (index_map.size() != 0)
        {
            permute_vectors(input, vec_ptr(temp_buffer_A), n, samples, vec_ptr(index_map), 1, hw, main_stream);
            permuted_input = vec_ptr(temp_buffer_A);
            permuted_output = vec_ptr(temp_buffer_B);
        }

        // tb = M * ta
        blas_gemm<H2Opus_Real, hw>(main_stream, H2Opus_NoTrans, H2Opus_NoTrans, n, samples, n, 1, M, ldm,
                                   permuted_input, n, 0, permuted_output, n);

        // tb = ta * alpha + tb
        if (alpha != 0)
        {
            blas_axpy<H2Opus_Real, hw>(main_stream, n * samples, alpha, permuted_input, 1, permuted_output, 1);
        }

        // output = P * tb
        if (index_map.size() != 0)
            permute_vectors(permuted_output, output, n, samples, vec_ptr(index_map), 0, hw, main_stream);
    }

    int getMatrixDim()
    {
        return n;
    }

    H2Opus_Real get1Norm()
    {
        return one_norm_A;
    }
};

template <int hw> class SimpleHMatrixSampler : public BasicHMatrixSampler
{
  private:
    typedef typename VectorContainer<hw, int>::type IntVector;
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

    THMatrix<hw> ownedA;
    THMatrix<hw> *A;
    int n;
    h2opusHandle_t h2opus_handle;
    h2opusComputeStream_t main_stream;

    H2Opus_Real one_norm_A, scale_A, identity_scale;

  public:
    SimpleHMatrixSampler(THMatrix<hw> *A, h2opusHandle_t h2opus_handle) : ownedA(0)
    {
        this->A = A;
        this->n = A->n;
        this->h2opus_handle = h2opus_handle;
        this->scale_A = 1;
        this->identity_scale = 0;

        this->one_norm_A = sampler_1_norm<H2Opus_Real, hw>(this, n, ONE_NORM_EST_MAX_SAMPLES, h2opus_handle);
        this->main_stream = h2opus_handle->getMainStream();
    }

    template <int otherhw> SimpleHMatrixSampler(THMatrix<otherhw> *A, h2opusHandle_t h2opus_handle) : ownedA(*A)
    {
        this->A = &this->ownedA;
        this->n = A->n;
        this->h2opus_handle = h2opus_handle;
        this->scale_A = 1;
        this->identity_scale = 0;

        this->one_norm_A = sampler_1_norm<H2Opus_Real, hw>(this, n, ONE_NORM_EST_MAX_SAMPLES, h2opus_handle);
        this->main_stream = h2opus_handle->getMainStream();
    }

    void sample(H2Opus_Real *input, H2Opus_Real *output, int samples)
    {
        hgemv(H2Opus_NoTrans, scale_A, *A, input, n, 0, output, n, samples, h2opus_handle);

        if (identity_scale != 0)
            blas_axpy<H2Opus_Real, hw>(main_stream, n * samples, identity_scale, input, 1, output, 1);
    }

    int getMatrixDim()
    {
        return n;
    }

    void setScaleA(H2Opus_Real scale_A)
    {
        this->scale_A = scale_A;
    }

    void setIdentityScale(H2Opus_Real identity_scale)
    {
        this->identity_scale = identity_scale;
    }

    H2Opus_Real get1Norm()
    {
        return fabs(scale_A) * one_norm_A;
    }
};

// CUSPARSE API is way too different from the MKL one, so we'll create separate samplers
// and use template specialization instead
template <int hw> class SparseSampler;

/*
template<>
class SparseSampler<H2OPUS_HWTYPE_GPU> : public BasicHMatrixSampler {
private:
    typedef typename VectorContainer<H2OPUS_HWTYPE_GPU, int>::type IntVector;
    typedef typename VectorContainer<H2OPUS_HWTYPE_GPU, H2Opus_Real>::type RealVector;

    h2opusHandle_t h2opus_handle;

    int n, nnz;
    RealVector csrValA;
    IntVector csrRowPtrA, csrColIndA;
    H2Opus_Real one_norm_A;

    RealVector input_buffer, output_buffer;
    IntVector index_map;

    cusparseHandle_t handle;
    cudaStream_t stream;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t mat_input, mat_output;
    cudaDataType cdt;
    bool initializedA, initializedB;

    void* csrmm_worskpace;
    size_t csrmm_ws_buffer_size;

    void set_csrmm_ws(size_t bufferSize)
    {
        if(bufferSize != csrmm_ws_buffer_size)
        {
            csrmm_ws_buffer_size = bufferSize;
            gpuErrchk( cudaFree(csrmm_worskpace) );
            gpuErrchk( cudaMalloc(&csrmm_worskpace, csrmm_ws_buffer_size) );
        }
    }
public:
    SparseSampler(int n, int *index_map, h2opusHandle_t h2opus_handle)
    {
        this->n = n;
        this->index_map = IntVector(index_map, index_map + n);
        this->h2opus_handle = h2opus_handle;
        this->stream = h2opus_handle->getKblasStream();

#ifdef H2OPUS_USE_DOUBLE_PRECISION
        this->cdt = CUDA_R_64F;
#elif defined(H2OPUS_USE_SINGLE_PRECISION)
        this->cdt = CUDA_R_32F;
#endif
        initializedA = initializedB = false;
        csrmm_worskpace = NULL;
        csrmm_ws_buffer_size = 0;
    }

    ~SparseSampler()
    {
        gpuCusparseErrchk( cusparseDestroy(handle) );

        if(initializedA)
            gpuCusparseErrchk( cusparseDestroySpMat(matA) );

        if(initializedB)
        {
            gpuCusparseErrchk( cusparseDestroyDnMat(mat_input) );
            gpuCusparseErrchk( cusparseDestroyDnMat(mat_output) );
        }

        if(csrmm_worskpace && csrmm_ws_buffer_size != 0)
            gpuErrchk( cudaFree(csrmm_worskpace) );
    }

    void setSparseData(int* ia, int* ja, H2Opus_Real* val, int nnz)
    {
        if(initializedA)
            gpuCusparseErrchk( cusparseDestroySpMat(matA) );

        initializedA = true;

        this->nnz  = nnz;
        csrValA    = RealVector(val, val + nnz);
        csrRowPtrA = IntVector(ia,  ia  + n+1);
        csrColIndA = IntVector(ja,  ja  + nnz);

        gpuCusparseErrchk( cusparseCreateCsr(
            &matA, n, n, nnz, vec_ptr(csrRowPtrA), vec_ptr(csrColIndA), vec_ptr(csrValA),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cdt
        ) );

        this->one_norm_A = sampler_1_norm<H2Opus_Real, H2OPUS_HWTYPE_GPU>(this, n, ONE_NORM_EST_MAX_SAMPLES,
h2opus_handle);
    }

    void verifyBufferSizes(int samples)
    {
        if (input_buffer.size()  < n * samples) input_buffer.resize(n * samples);
        if (output_buffer.size() < n * samples) output_buffer.resize(n * samples);
    }

    void sample(H2Opus_Real *input, H2Opus_Real *output, int samples)
    {
        verifyBufferSizes(samples);

        // output = P*A*P^t*input
        permute_vectors(input, vec_ptr(input_buffer), n, samples, vec_ptr(index_map), 1, H2OPUS_HWTYPE_GPU, stream);

        // Initialize all the cusparse stuff
        if(initializedB)
        {
            gpuCusparseErrchk( cusparseDestroyDnMat(mat_input) );
            gpuCusparseErrchk( cusparseDestroyDnMat(mat_output) );
        }

        initializedB = true;

        gpuCusparseErrchk( cusparseCreateDnMat(
            &mat_input, n, samples, n, vec_ptr(input_buffer), cdt, CUSPARSE_ORDER_COL
        ) );

        gpuCusparseErrchk( cusparseCreateDnMat(
            &mat_output, n, samples, n, vec_ptr(output_buffer), cdt, CUSPARSE_ORDER_COL
        ) );

        H2Opus_Real alpha = 1, beta = 0;

        // Figure out if we need to resize the workspace buffer
        size_t bufferSize;

        gpuCusparseErrchk( cusparseSpMM_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, mat_input &beta, mat_output, cdt, CUSPARSE_CSRMM_ALG1, &bufferSize
        ) );

        set_csrmm_ws(bufferSize);

        // execute the sparseMM
        gpuCusparseErrchk( cusparseSpMM(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, mat_input &beta, mat_output, cdt, CUSPARSE_CSRMM_ALG1,
            csrmm_worskpace
        ) );

        permute_vectors(vec_ptr(output_buffer), output, n, samples, vec_ptr(index_map), 0, H2OPUS_HWTYPE_GPU, stream);
    }

    int getMatrixDim()
    {
        return n;
    }

    H2Opus_Real get1Norm()
    {
        return one_norm_A;
    }
};
*/

#ifdef H2OPUS_USE_GPU

template <> class SparseSampler<H2OPUS_HWTYPE_GPU> : public BasicHMatrixSampler
{
#ifdef H2OPUS_USE_DOUBLE_PRECISION
#define cusparse_csrmv cusparseDcsrmv
#define cusparse_csrmm cusparseDcsrmm
#elif defined(H2OPUS_USE_SINGLE_PRECISION)
#define cusparse_csrmv cusparseScsrmv
#define cusparse_csrmm cusparseScsrmm
#endif
  private:
    typedef typename VectorContainer<H2OPUS_HWTYPE_GPU, int>::type IntVector;
    typedef typename VectorContainer<H2OPUS_HWTYPE_GPU, H2Opus_Real>::type RealVector;

    h2opusHandle_t h2opus_handle;

    int n, nnz;
    RealVector csrValA;
    IntVector csrRowPtrA, csrColIndA;

    H2Opus_Real one_norm_A;
    cusparseMatDescr_t descrA;

    RealVector input_buffer, output_buffer;
    IntVector index_map;

    cusparseHandle_t handle;
    h2opusComputeStream_t main_stream;

  public:
    // Index map is assumed to be on the CPU
    SparseSampler(int n, int *index_map, h2opusHandle_t h2opus_handle)
    {
        this->n = n;
        copyVector(this->index_map, index_map, n, H2OPUS_HWTYPE_CPU);
        this->h2opus_handle = h2opus_handle;
        this->main_stream = h2opus_handle->getMainStream();

        gpuCusparseErrchk(cusparseCreate(&handle));
        gpuCusparseErrchk(cusparseCreateMatDescr(&descrA));
        gpuCusparseErrchk(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        gpuCusparseErrchk(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    }

    ~SparseSampler()
    {
        gpuCusparseErrchk(cusparseDestroy(handle));
        gpuCusparseErrchk(cusparseDestroyMatDescr(descrA));
    }

    // Assumes data is on the CPU
    void setSparseData(int *ia, int *ja, H2Opus_Real *val, int nnz)
    {
        this->nnz = nnz;
        copyVector(csrValA, val, nnz, H2OPUS_HWTYPE_CPU);
        copyVector(csrRowPtrA, ia, n + 1, H2OPUS_HWTYPE_CPU);
        copyVector(csrColIndA, ja, nnz, H2OPUS_HWTYPE_CPU);

        this->one_norm_A =
            sampler_1_norm<H2Opus_Real, H2OPUS_HWTYPE_GPU>(this, n, ONE_NORM_EST_MAX_SAMPLES, h2opus_handle);
    }

    void verifyBufferSizes(int samples)
    {
        if (input_buffer.size() < (size_t)n * samples)
            input_buffer.resize(n * samples);
        if (output_buffer.size() < (size_t)n * samples)
            output_buffer.resize(n * samples);
    }

    void sample(H2Opus_Real *input, H2Opus_Real *output, int samples)
    {
        verifyBufferSizes(samples);

        // output = P*A*P^t*input
        permute_vectors(input, vec_ptr(input_buffer), n, samples, vec_ptr(index_map), 1, H2OPUS_HWTYPE_GPU,
                        main_stream);

        // I give up! cusparse changes API too often
        // TODO: Fix this to work with current version of cusparse

        // H2Opus_Real alpha = 1, beta = 0;

        // cusparse_csrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, samples, n, nnz, &alpha, descrA,
        // vec_ptr(csrValA),
        //                vec_ptr(csrRowPtrA), vec_ptr(csrColIndA), vec_ptr(input_buffer), n, &beta,
        //                vec_ptr(output_buffer), n);

        permute_vectors(vec_ptr(output_buffer), output, n, samples, vec_ptr(index_map), 0, H2OPUS_HWTYPE_GPU,
                        main_stream);
    }

    int getMatrixDim()
    {
        return n;
    }

    H2Opus_Real get1Norm()
    {
        return one_norm_A;
    }
};
#endif

#ifdef H2OPUS_USE_MKL

template <> class SparseSampler<H2OPUS_HWTYPE_CPU> : public BasicHMatrixSampler
{
#ifdef H2OPUS_USE_DOUBLE_PRECISION
#define mkl_sparse_create_csr mkl_sparse_d_create_csr
#define mkl_sparse_mm mkl_sparse_d_mm
#elif defined(H2OPUS_USE_SINGLE_PRECISION)
#define mkl_sparse_create_csr mkl_sparse_s_create_csr
#define mkl_sparse_mm mkl_sparse_s_mm
#endif
  private:
    typedef typename VectorContainer<H2OPUS_HWTYPE_CPU, int>::type IntVector;
    typedef typename VectorContainer<H2OPUS_HWTYPE_CPU, H2Opus_Real>::type RealVector;

    h2opusHandle_t h2opus_handle;
    h2opusComputeStream_t main_stream;

    int n, nnz;
    RealVector csrValA;
    IntVector csrRowPtrA, csrColIndA;
    H2Opus_Real one_norm_A;

    sparse_matrix_t matA;
    matrix_descr descrA;
    bool initialized;

    RealVector input_buffer, output_buffer;
    IntVector index_map;

    void destroySparseMat()
    {
        if (initialized)
        {
            sparse_status_t result = mkl_sparse_destroy(matA);

            if (result != SPARSE_STATUS_SUCCESS)
                printf("Error destroying sparse structure: %d\n", result);
        }
    }

  public:
    // Index map is assumed to be on the CPU
    SparseSampler(int n, int *index_map, h2opusHandle_t h2opus_handle)
    {
        this->n = n;
        copyVector(this->index_map, index_map, n, H2OPUS_HWTYPE_CPU);
        this->h2opus_handle = h2opus_handle;
        this->main_stream = h2opus_handle->getMainStream();

        this->descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

        initialized = false;
    }

    ~SparseSampler()
    {
        destroySparseMat();
    }

    void setSparseData(int *ia, int *ja, H2Opus_Real *val, int nnz)
    {
        destroySparseMat();
        initialized = true;

        this->nnz = nnz;
        copyVector(csrValA, val, nnz, H2OPUS_HWTYPE_CPU);
        copyVector(csrRowPtrA, ia, n + 1, H2OPUS_HWTYPE_CPU);
        copyVector(csrColIndA, ja, nnz, H2OPUS_HWTYPE_CPU);

        sparse_status_t result = mkl_sparse_create_csr(&matA, SPARSE_INDEX_BASE_ZERO, n, n, vec_ptr(csrRowPtrA),
                                                       vec_ptr(csrRowPtrA) + 1, vec_ptr(csrColIndA), vec_ptr(csrValA));

        if (result != SPARSE_STATUS_SUCCESS)
            printf("Error creating sparse structure: %d\n", result);

        this->one_norm_A =
            sampler_1_norm<H2Opus_Real, H2OPUS_HWTYPE_CPU>(this, n, ONE_NORM_EST_MAX_SAMPLES, h2opus_handle);
    }

    void verifyBufferSizes(int samples)
    {
        if (input_buffer.size() < (size_t)n * samples)
            input_buffer.resize(n * samples);
        if (output_buffer.size() < (size_t)n * samples)
            output_buffer.resize(n * samples);
    }

    void sample(H2Opus_Real *input, H2Opus_Real *output, int samples)
    {
        verifyBufferSizes(samples);

        // output = P*A*P^t*input
        permute_vectors(input, vec_ptr(input_buffer), n, samples, vec_ptr(index_map), 1, H2OPUS_HWTYPE_CPU,
                        main_stream);

        H2Opus_Real alpha = 1, beta = 0;

        sparse_status_t result =
            mkl_sparse_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, matA, descrA, SPARSE_LAYOUT_COLUMN_MAJOR,
                          vec_ptr(input_buffer), samples, n, beta, vec_ptr(output_buffer), n);

        if (result != SPARSE_STATUS_SUCCESS)
            printf("Error performing sparse MM: %d\n", result);

        permute_vectors(vec_ptr(output_buffer), output, n, samples, vec_ptr(index_map), 0, H2OPUS_HWTYPE_CPU,
                        main_stream);
    }

    int getMatrixDim()
    {
        return n;
    }

    H2Opus_Real get1Norm()
    {
        return one_norm_A;
    }
};
#endif

/////////////////////////////////////////////////////////////////////////////
// Various Newton Schulz samplers
/////////////////////////////////////////////////////////////////////////////
template <int hw> class HighOrderInversionSampler : public HMatrixSampler
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

  private:
    THMatrix<hw> *prev_iterate;
    BasicHMatrixSampler *A;

    int n, ns_it, order;

    h2opusHandle_t h2opus_handle;
    h2opusComputeStream_t main_stream;

    RealVector temp_bufferA, temp_bufferB;
    RealVector S_buffer, R_buffer;

  public:
    HighOrderInversionSampler(BasicHMatrixSampler *A, int order, h2opusHandle_t h2opus_handle)
    {
        this->A = A;
        this->n = A->getMatrixDim();
        this->order = order;
        this->h2opus_handle = h2opus_handle;
        this->main_stream = h2opus_handle->getMainStream();
        this->prev_iterate = NULL;
        this->ns_it = 0;
    }

    BasicHMatrixSampler *getA()
    {
        return A;
    }

    void setPreviousIterate(THMatrix<hw> *hmatrix, int ns_it)
    {
        this->prev_iterate = hmatrix;
        this->ns_it = ns_it;
    }

    void setOrder(int order)
    {
        assert(order >= 2);
        this->order = order;
    }

    void verifyBufferSizes(int samples)
    {
        size_t buffer_size = n * samples;

        if (temp_bufferA.size() < buffer_size)
            temp_bufferA.resize(buffer_size);
        if (temp_bufferB.size() < buffer_size)
            temp_bufferB.resize(buffer_size);
        if (S_buffer.size() < buffer_size)
            S_buffer.resize(buffer_size);
        if (R_buffer.size() < buffer_size)
            R_buffer.resize(buffer_size);
    }

    void sample(H2Opus_Real *input, H2Opus_Real *output, int samples)
    {
        verifyBufferSizes(samples);

        // Symmetric matrix so 1-norm = inf-norm
        H2Opus_Real inf_norm_A = A->get1Norm();

        ///////////////////////////////////////////////////////////////////////
        // X * input = X_prev * sum_{i=0...p-1}{(I - P * A *P^t * X_prev)^i} * input
        //           = X_prev * sum_{i=0...p-1}{R^i} * input
        ///////////////////////////////////////////////////////////////////////
        // The R_buffer will hold R^i * input with R = (I - P * A * P^t * X_prev)
        // Accumulate into S_buffer, starting with i = 0 (the identity)

        copyArray(input, vec_ptr(R_buffer), n * samples, main_stream, hw);
        copyArray(input, vec_ptr(S_buffer), n * samples, main_stream, hw);

        H2Opus_Real X0_scaling_factor = (ns_it == 0 ? (H2Opus_Real)1.0 / inf_norm_A : 1);

        // Now we begin the accumulation loop by update R_buffer and accumulating into S_buffer
        for (int i = 1; i < order; i++)
        {
            ///////////////////////////////////////////////////////////////////////
            // Update R_buffer = R * R_buffer = R * R^{i-1} * input = R^i * input
            ///////////////////////////////////////////////////////////////////////
            // ta = X_prev * R_buffer
            if (prev_iterate)
            {
                // Use the previous iterate if it was provided as an hmatrix
                hgemv(H2Opus_NoTrans, X0_scaling_factor, *prev_iterate, vec_ptr(R_buffer), n, 0, vec_ptr(temp_bufferA),
                      n, samples, h2opus_handle);
            }
            else
            {
                // Otherwise use the scaled identity
                copyArray(vec_ptr(R_buffer), vec_ptr(temp_bufferA), n * samples, main_stream, hw);
                blas_scal<H2Opus_Real, hw>(main_stream, n * samples, X0_scaling_factor, vec_ptr(temp_bufferA), 1);
            }

            // tb = A * ta
            A->sample(vec_ptr(temp_bufferA), vec_ptr(temp_bufferB), samples);

            // R_buffer = R_buffer - tb = (I - P * A * P^t * X_prev) * R_buffer
            H2Opus_Real alpha_axpy = -1;
            blas_axpy<H2Opus_Real, hw>(main_stream, n * samples, alpha_axpy, vec_ptr(temp_bufferB), 1,
                                       vec_ptr(R_buffer), 1);

            // Accumulate R_buffer into S_buffer: S += R^i * input
            alpha_axpy = 1;
            blas_axpy<H2Opus_Real, hw>(main_stream, n * samples, alpha_axpy, vec_ptr(R_buffer), 1, vec_ptr(S_buffer),
                                       1);
        }

        // output = X_prev * S
        if (prev_iterate)
        {
            // Use the previous iterate if it was provided as an hmatrix
            hgemv(H2Opus_NoTrans, X0_scaling_factor, *prev_iterate, vec_ptr(S_buffer), n, 0, output, n, samples,
                  h2opus_handle);
        }
        else
        {
            // Otherwise use the scaled identity
            copyArray(vec_ptr(S_buffer), output, n * samples, main_stream, hw);
            blas_scal<H2Opus_Real, hw>(main_stream, n * samples, X0_scaling_factor, output, 1);
        }
    }
};

template <int hw> class NewtonSchultzSampler : public HMatrixSampler
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

  private:
    BasicHMatrixSampler *A;
    THMatrix<hw> *prev_iterate;

    int n, ns_it;
    h2opusHandle_t h2opus_handle;
    h2opusComputeStream_t main_stream;

    RealVector temp_bufferA, temp_bufferB;

  public:
    NewtonSchultzSampler(BasicHMatrixSampler *A, h2opusHandle_t h2opus_handle)
    {
        this->A = A;
        this->n = A->getMatrixDim();

        this->prev_iterate = NULL;
        this->h2opus_handle = h2opus_handle;
        this->ns_it = 0;
        this->main_stream = h2opus_handle->getMainStream();
    }

    void setPreviousIterate(THMatrix<hw> *hmatrix, int iteration)
    {
        this->prev_iterate = hmatrix;
        this->ns_it = iteration;
    }

    void verifyBufferSizes(int samples)
    {
        if (temp_bufferA.size() < n * samples)
            temp_bufferA.resize(n * samples);
        if (temp_bufferB.size() < n * samples)
            temp_bufferB.resize(n * samples);
    }

    BasicHMatrixSampler *getA()
    {
        return A;
    }

    void sample(H2Opus_Real *input, H2Opus_Real *output, int samples)
    {
        verifyBufferSizes(samples);

        // Symmetric matrix so 1-norm = inf-norm
        H2Opus_Real inf_norm_A = A->get1Norm();

        //////////////////////////////////////////////////////////////////
        // X * input = X_prev * (2 * I - A * X_prev) * input;
        //////////////////////////////////////////////////////////////////
        // ta = X_prev * input
        if (ns_it == 0)
        {
            copyArray(input, vec_ptr(temp_bufferA), n * samples, main_stream, hw);
            H2Opus_Real alpha_scal = (H2Opus_Real)1.0 / inf_norm_A;
            blas_scal<H2Opus_Real, hw>(main_stream, n * samples, alpha_scal, vec_ptr(temp_bufferA), 1);
        }
        else
            hgemv(H2Opus_NoTrans, 1, *prev_iterate, input, n, 0, vec_ptr(temp_bufferA), n, samples, h2opus_handle);

        // tb = A * ta
        A->sample(vec_ptr(temp_bufferA), vec_ptr(temp_bufferB), samples);

        // tb = 2 * input - tb = 2 * input - A * X_prev * input
        //    = (2 * I - A * X_prev) * input
        H2Opus_Real alpha_scal = -1, alpha_axpy = 2;

        for (int i = 0; i < samples; i++)
        {
            H2Opus_Real *y = vec_ptr(temp_bufferB) + i * n;
            H2Opus_Real *x = input + i * n;

            blas_scal<H2Opus_Real, hw>(main_stream, n, alpha_scal, y, 1);
            blas_axpy<H2Opus_Real, hw>(main_stream, n, alpha_axpy, x, 1, y, 1);
        }

        // output = X_prev * tb = X_prev * (2 * I - P * A * P^t * X_prev) * input
        if (ns_it == 0)
        {
            H2Opus_Real alpha_scal = (H2Opus_Real)1.0 / inf_norm_A;
            blas_scal<H2Opus_Real, hw>(main_stream, n * samples, alpha_scal, vec_ptr(temp_bufferB), 1);
            copyArray(vec_ptr(temp_bufferB), output, n * samples, main_stream, hw);
        }
        else
            hgemv(H2Opus_NoTrans, 1, *prev_iterate, vec_ptr(temp_bufferB), n, 0, output, n, samples, h2opus_handle);
    }
};

template <int hw> class UnrolledNewtonSchultzSampler : public HMatrixSampler
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

  private:
    THMatrix<hw> *prev_iterate;
    BasicHMatrixSampler *A;

    int n;

    h2opusHandle_t h2opus_handle;
    h2opusComputeStream_t main_stream;

    std::vector<long> coef;
    RealVector temp_bufferA, temp_bufferB;
    RealVector S_buffer, R_buffer;

    void setCoefficients(int unrolls)
    {
        int num_coef = 1 << unrolls;
        coef.resize(num_coef + 1);
        coef[0] = -1;
        int k = 1;

        for (int i = 1; i < num_coef + 1; i++)
        {
            coef[i] = k * abs(coef[i - 1]) * (num_coef - i + 1) / i;
            k *= -1;
        }
    }

  public:
    UnrolledNewtonSchultzSampler(BasicHMatrixSampler *A, int unrolls, h2opusHandle_t h2opus_handle)
    {
        this->A = A;
        this->n = A->getMatrixDim();
        this->h2opus_handle = h2opus_handle;
        this->main_stream = h2opus_handle->getMainStream();

        setCoefficients(unrolls);
    }

    void setPreviousIterate(THMatrix<hw> *hmatrix, int ns_it)
    {
        this->prev_iterate = hmatrix;
    }

    void verifyBufferSizes(int samples)
    {
        size_t buffer_size = n * samples;

        if (temp_bufferA.size() < buffer_size)
            temp_bufferA.resize(buffer_size);
        if (temp_bufferB.size() < buffer_size)
            temp_bufferB.resize(buffer_size);
        if (S_buffer.size() < buffer_size)
            S_buffer.resize(buffer_size);
        if (R_buffer.size() < buffer_size)
            R_buffer.resize(buffer_size);
    }

    void sample(H2Opus_Real *input, H2Opus_Real *output, int samples)
    {
        verifyBufferSizes(samples);

        // Symmetric matrix so 1-norm = inf-norm
        H2Opus_Real inf_norm_A = A->get1Norm();

        ///////////////////////////////////////////////////////////////////////
        // X * input = X_prev * sum_{i=0...n}{coef[i] * (A * X_prev)^i} * input
        ///////////////////////////////////////////////////////////////////////
        // The R_buffer will hold R^i * input with R = (A * X_prev)
        // Accumulate into S_buffer, starting with i = 0 (the identity)
        copyArray(input, vec_ptr(R_buffer), n * samples, main_stream, hw);
        copyArray(input, vec_ptr(S_buffer), n * samples, main_stream, hw);
        blas_scal<H2Opus_Real, hw>(main_stream, n * samples, (H2Opus_Real)coef[1], vec_ptr(S_buffer), 1);

        H2Opus_Real X0_scaling_factor = (H2Opus_Real)1.0 / inf_norm_A;

        // Now we begin the accumulation loop by update R_buffer and accumulating into S_buffer
        for (int i = 2; i < coef.size(); i++)
        {
            // Update R_buffer = R * R_buffer = A * X_prev * (A * X_prev)^{i-1} * input) = R^i * input
            hgemv(H2Opus_NoTrans, X0_scaling_factor, *prev_iterate, vec_ptr(R_buffer), n, 0, vec_ptr(temp_bufferA), n,
                  samples, h2opus_handle);

            A->sample(vec_ptr(temp_bufferA), vec_ptr(R_buffer), samples);

            // Accumulate coef[i] * R_buffer into S_buffer: S += coef[i] * R^i * input
            blas_axpy<H2Opus_Real, hw>(main_stream, n * samples, (H2Opus_Real)coef[i], vec_ptr(R_buffer), 1,
                                       vec_ptr(S_buffer), 1);
        }

        // output = X_prev * S
        hgemv(H2Opus_NoTrans, X0_scaling_factor, *prev_iterate, vec_ptr(S_buffer), n, 0, output, n, samples,
              h2opus_handle);
    }

    BasicHMatrixSampler *getA()
    {
        return A;
    }
};

template <int hw> class PinvNewtonSchultzSamplerPhaseA : public HMatrixSampler
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

  private:
    BasicHMatrixSampler *A;
    THMatrix<hw> *Xk;

    int n;
    H2Opus_Real alpha, x_scale;
    h2opusHandle_t h2opus_handle;
    h2opusComputeStream_t main_stream;

    RealVector temp_bufferA, temp_bufferB;

  public:
    PinvNewtonSchultzSamplerPhaseA(BasicHMatrixSampler *A, h2opusHandle_t h2opus_handle)
    {
        this->A = A;
        this->n = A->getMatrixDim();
        this->Xk = NULL;

        this->h2opus_handle = h2opus_handle;
        this->main_stream = h2opus_handle->getMainStream();
    }

    void setIterate(THMatrix<hw> *hmatrix, H2Opus_Real scale)
    {
        this->Xk = hmatrix;
        this->x_scale = scale;
    }

    void verifyBufferSizes(int samples)
    {
        if (temp_bufferA.size() < n * samples)
            temp_bufferA.resize(n * samples);
        if (temp_bufferB.size() < n * samples)
            temp_bufferB.resize(n * samples);
    }

    void setAlpha(H2Opus_Real alpha)
    {
        this->alpha = alpha;
    }

    void sample(H2Opus_Real *input, H2Opus_Real *output, int samples)
    {
        verifyBufferSizes(samples);

        //////////////////////////////////////////////////////////////////
        // X * input = X_prev * (2 * I - A * X_prev) * input;
        //////////////////////////////////////////////////////////////////
        // ta = X_prev * input = x_scale * Xk * input
        if (Xk)
            hgemv(H2Opus_NoTrans, x_scale, *Xk, input, n, 0, vec_ptr(temp_bufferA), n, samples, h2opus_handle);
        else
        {
            copyArray(input, vec_ptr(temp_bufferB), n * samples, main_stream, hw);
            blas_scal<H2Opus_Real, hw>(main_stream, n * samples, x_scale, vec_ptr(temp_bufferB), 1);

            A->sample(vec_ptr(temp_bufferB), vec_ptr(temp_bufferA), samples);
        }

        // tb = A * ta
        A->sample(vec_ptr(temp_bufferA), vec_ptr(temp_bufferB), samples);

        // tb = 2 * input - tb = 2 * input - A * X_prev * input
        //    = (2 * I - A * X_prev) * input
        H2Opus_Real alpha_scal = -1, alpha_axpy = 2;

        for (int i = 0; i < samples; i++)
        {
            H2Opus_Real *y = vec_ptr(temp_bufferB) + i * n;
            H2Opus_Real *x = input + i * n;

            blas_scal<H2Opus_Real, hw>(main_stream, n, alpha_scal, y, 1);
            blas_axpy<H2Opus_Real, hw>(main_stream, n, alpha_axpy, x, 1, y, 1);
        }

        // output = X_prev * tb = X_prev * (2 * I - P * A * P^t * X_prev) * input
        if (Xk)
            hgemv(H2Opus_NoTrans, x_scale * alpha, *Xk, vec_ptr(temp_bufferB), n, 0, output, n, samples, h2opus_handle);
        else
        {
            copyArray(vec_ptr(temp_bufferB), vec_ptr(temp_bufferA), n * samples, main_stream, hw);
            blas_scal<H2Opus_Real, hw>(main_stream, n * samples, x_scale, vec_ptr(temp_bufferA), 1);

            A->sample(vec_ptr(temp_bufferA), output, samples);
        }
    }
};

template <int hw> class PinvNewtonSchultzSamplerPhaseB : public HMatrixSampler
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

  private:
    BasicHMatrixSampler *A;
    THMatrix<hw> *Xk;

    int n;
    H2Opus_Real x_scale;
    H2Opus_Real coef[3];

    h2opusHandle_t h2opus_handle;
    h2opusComputeStream_t main_stream;

    RealVector temp_buffer;
    RealVector S_buffer, R_buffer;

  public:
    PinvNewtonSchultzSamplerPhaseB(BasicHMatrixSampler *A, h2opusHandle_t h2opus_handle)
    {
        this->A = A;
        this->n = A->getMatrixDim();
        this->Xk = NULL;

        this->h2opus_handle = h2opus_handle;
        this->main_stream = h2opus_handle->getMainStream();

        // X = X * ((A*X)^3 - 4 * (A*X)^2 + 4*A*X)
        coef[0] = 4;
        coef[1] = -4;
        coef[2] = 1;
    }

    void setIterate(THMatrix<hw> *hmatrix, H2Opus_Real scale)
    {
        this->Xk = hmatrix;
        this->x_scale = scale;
    }

    void verifyBufferSizes(int samples)
    {
        size_t buffer_size = n * samples;

        if (temp_buffer.size() < buffer_size)
            temp_buffer.resize(buffer_size);
        if (S_buffer.size() < buffer_size)
            S_buffer.resize(buffer_size);
        if (R_buffer.size() < buffer_size)
            R_buffer.resize(buffer_size);
    }

    void sample(H2Opus_Real *input, H2Opus_Real *output, int samples)
    {
        verifyBufferSizes(samples);

        ///////////////////////////////////////////////////////////////////////
        // X * input = X_prev * sum_{i=0...2}{coef[i] * (A * Xk)^{i+1}} * input
        ///////////////////////////////////////////////////////////////////////
        // The R_buffer will hold R^i * input with R^i = (A * Xk)^{i+1}
        // Accumulate into S_buffer
        fillArray(vec_ptr(S_buffer), n * samples, 0, main_stream, hw);
        copyArray(input, vec_ptr(R_buffer), n * samples, main_stream, hw);

        for (int i = 0; i < 3; i++)
        {
            // ta = X_prev * R = x_scale * Xk * R
            hgemv(H2Opus_NoTrans, x_scale, *Xk, vec_ptr(R_buffer), n, 0, vec_ptr(temp_buffer), n, samples,
                  h2opus_handle);

            // R = A * ta = A * X_prev * R = (A * Xk)^{i+1} * input
            A->sample(vec_ptr(temp_buffer), vec_ptr(R_buffer), samples);

            // S = S + coef[i] * R = S + coef[i] * (A * Xk)^{i+1} * input
            blas_axpy<H2Opus_Real, hw>(main_stream, n * samples, coef[i], vec_ptr(R_buffer), 1, vec_ptr(S_buffer), 1);
        }

        // output = X_prev * S = X_prev * sum_{i=0...2}{coef[i] * (A * Xk)^{i+1}} * input
        hgemv(H2Opus_NoTrans, x_scale, *Xk, vec_ptr(S_buffer), n, 0, output, n, samples, h2opus_handle);
    }
};

#ifdef H2OPUS_USE_MPI

#include <h2opus/distributed/distributed_hgemv.h>

template <int hw> class DistributedHMatrixDiffSampler : public HMatrixSampler
{
  private:
    typedef typename VectorContainer<hw, int>::type IntVector;
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;

    TDistributedHMatrix<hw> ownedA;
    TDistributedHMatrix<hw> *A, *B;
    distributedH2OpusHandle_t dist_h2opus_handle;
    int n;

  public:
    DistributedHMatrixDiffSampler(TDistributedHMatrix<hw> *A, TDistributedHMatrix<hw> *B,
                                  distributedH2OpusHandle_t dist_h2opus_handle)
        : ownedA(0)
    {
        this->A = A;
        this->B = B;
        this->dist_h2opus_handle = dist_h2opus_handle;
        assert(A->basis_tree.basis_branch.index_map.size() == B->basis_tree.basis_branch.index_map.size());
        this->n = A->basis_tree.basis_branch.index_map.size();
    }

    template <int otherhw>
    DistributedHMatrixDiffSampler(TDistributedHMatrix<otherhw> *A, TDistributedHMatrix<hw> *B,
                                  distributedH2OpusHandle_t dist_h2opus_handle)
        : ownedA(*A)
    {
        this->A = &this->ownedA;
        this->B = B;
        this->dist_h2opus_handle = dist_h2opus_handle;
        assert(A->basis_tree.basis_branch.index_map.size() == B->basis_tree.basis_branch.index_map.size());
        this->n = A->basis_tree.basis_branch.index_map.size();
    }

    void sample(H2Opus_Real *input, H2Opus_Real *output, int samples)
    {
        distributed_hgemv(1, *A, input, n, 0, output, n, samples, dist_h2opus_handle);
        distributed_hgemv(-1, *B, input, n, 1, output, n, samples, dist_h2opus_handle);
    }
};
#endif

#endif
