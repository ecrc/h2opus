#ifndef __H2OPUS_DEVICE_VECTOR_H__
#define __H2OPUS_DEVICE_VECTOR_H__

// TODO: the new cuda async memory and copies make most of this stuff unnecessary.
//       should be simplified in a future release and take stream arguments to
//       be completely async

#include <h2opusconf.h>
#if defined(H2OPUS_USE_GPU)
#include <cuda.h>

#ifndef H2OPUS_USE_CUDA_VMM
#ifdef CUDA_VERSION
#if (CUDA_VERSION == 10020 || CUDA_VERSION > 10020)
#define H2OPUS_USE_CUDA_VMM
#endif
#endif
#endif

#include <h2opus/core/h2opus_handle.h>
#include <h2opus/util/gpu_err_check.h>

template <class T> struct H2OpusDeviceVector
{
  private:
#ifdef H2OPUS_USE_CUDA_VMM
    size_t memory_allocation_size, allocation_granularity;
    CUmemGenericAllocationHandle allocation_handle;
    CUmemAllocationProp allocation_properties;
    CUmemAccessDesc access_description;
    CUdeviceptr device_ptr;
#endif
    T *vector_data;
    size_t allocated_entries;
    size_t entries_capacity;

    void init()
    {
#ifdef H2OPUS_USE_CUDA_VMM
        int device_id;
        gpuErrchk(cudaGetDevice(&device_id));

        allocation_properties = {};
        allocation_properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        allocation_properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        allocation_properties.location.id = device_id;
        access_description.location = allocation_properties.location;
        access_description.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        gpuDriverErrchk(cuMemGetAllocationGranularity(&allocation_granularity, &allocation_properties,
                                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        memory_allocation_size = 0;
        device_ptr = 0;
#endif
        allocated_entries = entries_capacity = 0;
        vector_data = NULL;
    }

    void freeMemory()
    {
#ifdef H2OPUS_USE_CUDA_VMM
        if (device_ptr != 0 && memory_allocation_size != 0)
        {
            gpuErrchk(cudaDeviceSynchronize());
            gpuDriverErrchk(cuMemUnmap(device_ptr, memory_allocation_size));
            gpuDriverErrchk(cuMemAddressFree(device_ptr, memory_allocation_size));
            gpuDriverErrchk(cuMemRelease(allocation_handle));

            device_ptr = 0;
            memory_allocation_size = 0;
        }
#else
        if (vector_data != NULL && entries_capacity != 0)
            gpuErrchk(cudaFree(vector_data));
#endif
        vector_data = NULL;
        allocated_entries = entries_capacity = 0;
    }

    void swap(H2OpusDeviceVector<T> &A)
    {
        std::swap(this->vector_data, A.vector_data);
        std::swap(this->allocated_entries, A.allocated_entries);
        std::swap(this->entries_capacity, A.entries_capacity);
#ifdef H2OPUS_USE_CUDA_VMM
        std::swap(this->memory_allocation_size, A.memory_allocation_size);
        std::swap(this->allocation_granularity, A.allocation_granularity);
        std::swap(this->allocation_handle, A.allocation_handle);
        std::swap(this->allocation_properties, A.allocation_properties);
        std::swap(this->access_description, A.access_description);
        std::swap(this->device_ptr, A.device_ptr);
#endif
    }

    void init(const thrust::host_vector<T> &A)
    {
        size_t num_elements = A.size();
        init();
        resize(num_elements);
        gpuErrchk(cudaMemcpy(vector_data, A.data(), num_elements * sizeof(T), cudaMemcpyHostToDevice));
    }

    void init(const H2OpusDeviceVector &A)
    {
        size_t num_elements = A.size();
        init();
        resize(num_elements);
        gpuErrchk(cudaMemcpy(vector_data, A.vector_data, num_elements * sizeof(T), cudaMemcpyDeviceToDevice));
    }

  public:
    H2OpusDeviceVector()
    {
        init();
    }

    H2OpusDeviceVector(size_t array_size)
    {
        init();
        resize(array_size);
    }

    H2OpusDeviceVector(const H2OpusDeviceVector &A)
    {
        init(A);
    }

    H2OpusDeviceVector(const thrust::host_vector<T> &A)
    {
        init(A);
    }

    H2OpusDeviceVector &operator=(const H2OpusDeviceVector &A)
    {
        init(A);
        return *this;
    }

    H2OpusDeviceVector &operator=(const thrust::host_vector<T> &A)
    {
        init(A);
        return *this;
    }

    ~H2OpusDeviceVector()
    {
        freeMemory();
    }

    size_t size() const
    {
        return allocated_entries;
    }

    void shrink_to_fit()
    {
        H2OpusDeviceVector<T>(*this).swap(*this);
    }

    T *data() const
    {
        return vector_data;
    }

    void clear()
    {
        freeMemory();
    }

    void resize(size_t array_size)
    {
        if (array_size <= entries_capacity)
        {
            allocated_entries = array_size;
            return;
        }

        freeMemory();

        size_t bytes = array_size * sizeof(T);

#ifdef H2OPUS_USE_CUDA_VMM
        // align memory size to granularity
        memory_allocation_size =
            ((bytes + allocation_granularity - 1) / allocation_granularity) * allocation_granularity;

        gpuDriverErrchk(cuMemCreate(&allocation_handle, memory_allocation_size, &allocation_properties, 0));
        gpuDriverErrchk(cuMemAddressReserve(&device_ptr, memory_allocation_size, 0ULL, 0ULL, 0ULL));
        gpuDriverErrchk(cuMemMap(device_ptr, memory_allocation_size, 0ULL, allocation_handle, 0ULL));
        gpuDriverErrchk(cuMemSetAccess(device_ptr, memory_allocation_size, &access_description, 1ULL));

        vector_data = (T *)device_ptr;
#else
        gpuErrchk(cudaMalloc(&vector_data, bytes));
#endif
        allocated_entries = array_size;
        entries_capacity = array_size;
    }
};

#endif

#endif
