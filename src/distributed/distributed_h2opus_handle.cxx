#include <h2opus/distributed/distributed_h2opus_handle.h>
#include <h2opus/distributed/comm_wrapper.h>
#ifdef H2OPUS_USE_GPU
#include <h2opus/util/gpu_err_check.h>
#endif
#include <string.h>
#include <unistd.h>

static inline void getHostName(char *hostname, int maxlen)
{
    gethostname(hostname, maxlen);
    hostname[maxlen - 1] = 0;
    long int len = strlen(hostname);
    for (long int i = 0; i < len; i++)
    {
        if (i == len - 1)
            continue;
        if (hostname[i] == '.')
        {
            hostname[i] = 0;
            return;
        }
    }
}

DistributedH2OpusHandle::DistributedH2OpusHandle(MPI_Comm comm)
{
    /* needs power-of-two size */
    int size;
    mpiErrchk(MPI_Comm_dup(comm, &this->ocomm));
    mpiErrchk(MPI_Comm_size(this->ocomm, &size));
    mpiErrchk(MPI_Comm_rank(this->ocomm, &this->orank));
    int size2 = std::exp2((int)std::log2(size));
    this->active = true;
    if (size2 < size)
    {
        int color = this->orank < size2 ? 0 : 1;
        mpiErrchk(MPI_Comm_split(this->ocomm, color, this->orank, &this->comm));
        this->active = color ? false : true;
    }
    else
    {
        mpiErrchk(MPI_Comm_dup(this->ocomm, &this->comm));
    }
    mpiErrchk(MPI_Comm_dup(this->comm, &this->commscatter));
    mpiErrchk(MPI_Comm_dup(this->comm, &this->commgather));
    mpiErrchk(MPI_Comm_rank(this->comm, &this->rank));
    mpiErrchk(MPI_Comm_size(this->comm, &this->num_ranks));
    mpiErrchk(MPI_Query_thread(&this->th_provided));
    int isset, *mpitagub;
    mpiErrchk(MPI_Comm_get_attr(this->comm, MPI_TAG_UB, &mpitagub, &isset));
    if (!isset)
        this->mpitagub = 32767; // https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf page 335
    else
        this->mpitagub = *mpitagub;
    this->mpitag = 0;
    this->handle = NULL;
    this->top_level_handle = NULL;
}

int DistributedH2OpusHandle::getNewTag()
{
    if (this->mpitag == this->mpitagub)
        this->mpitag = 0;
    return this->mpitag++;
}

DistributedH2OpusHandle::~DistributedH2OpusHandle()
{
    // Free the comm buffers
    for (int i = 0; i < (int)host_buffers.size(); i++)
        host_buffers[i].freeBuffer();
    host_gather_buffer.freeBuffer();
    host_scatter_buffer.freeBuffer();
#ifdef H2OPUS_USE_GPU
    for (int i = 0; i < (int)device_buffers.size(); i++)
        device_buffers[i].freeBuffer();
    device_gather_buffer.freeBuffer();
    device_scatter_buffer.freeBuffer();
#endif

    mpiErrchk(MPI_Comm_free(&this->ocomm));
    mpiErrchk(MPI_Comm_free(&this->comm));
    mpiErrchk(MPI_Comm_free(&this->commscatter));
    mpiErrchk(MPI_Comm_free(&this->commgather));
    h2opusDestroyHandle(this->handle);
    h2opusDestroyHandle(this->top_level_handle);
}

void h2opusCreateDistributedHandle(distributedH2OpusHandle_t *h2opus_handle, bool select_local_rank)
{
    h2opusCreateDistributedHandleComm(h2opus_handle, MPI_COMM_WORLD, select_local_rank);
}

void h2opusCreateDistributedHandleComm(distributedH2OpusHandle_t *h2opus_handle, MPI_Comm comm, bool select_local_rank)
{
    distributedH2OpusHandle_t dist_h2opus_handle = new DistributedH2OpusHandle(comm);

    char hostname[1024];
    getHostName(hostname, 1024);

    dist_h2opus_handle->local_rank = 0;
#ifdef H2OPUS_USE_GPU
    // Determine which GPU on the same host we should use if not already set by the user
    if (dist_h2opus_handle->active)
    {
        int localrank = 0;
        if (select_local_rank)
        {
            int devCount = 0;
            gpuErrchk(cudaGetDeviceCount(&devCount));
            gpuErrchk(cudaSetDevice(dist_h2opus_handle->rank % devCount));
        }
        gpuErrchk(cudaGetDevice(&dist_h2opus_handle->local_rank));
        if (select_local_rank)
        {
            cudaDeviceProp prop;
            gpuErrchk(cudaGetDeviceProperties(&prop, dist_h2opus_handle->local_rank));

            printf("H2OPUS: Local Rank %2d Pid %6d on %10s device %2d [0x%02x] %s\n", localrank, getpid(), hostname,
                   dist_h2opus_handle->local_rank, prop.pciBusID, prop.name);
        }
    }
#endif

#if defined(H2OPUS_USE_NVOMP) || (defined(_OPENMP) && defined(_CRAYC))
    /* MPI (OpenMPI) is thread-safe, but NVOMP does not allow calling omp_* API from non-OpenMP spawned threads
       Not sure how to detect this at compilation time since omp.h for nvc (at least as of 21.7 release) does not
       contain any special define */
    /* Disable explicit threading also when using CRAY compilers and OpenMP */
    bool mpithm = dist_h2opus_handle->th_provided == MPI_THREAD_MULTIPLE;
    dist_h2opus_handle->setUseThreads<H2OPUS_HWTYPE_CPU>(false);
#else
    bool mpithm = dist_h2opus_handle->th_provided == MPI_THREAD_MULTIPLE;
    dist_h2opus_handle->setUseThreads<H2OPUS_HWTYPE_CPU>(mpithm);
#endif
#ifdef H2OPUS_USE_GPU
    dist_h2opus_handle->setUseThreads<H2OPUS_HWTYPE_GPU>(mpithm);
#endif

    h2opusHandle_t internal_handle, top_handle;
    h2opusCreateHandle(&internal_handle);
    h2opusCreateHandle(&top_handle);

    dist_h2opus_handle->handle = internal_handle;
    dist_h2opus_handle->top_level_handle = top_handle;

    *h2opus_handle = dist_h2opus_handle;
}

void h2opusDestroyDistributedHandle(distributedH2OpusHandle_t h2opus_handle)
{
    delete h2opus_handle;
}
