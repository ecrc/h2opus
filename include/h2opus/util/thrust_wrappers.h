#ifndef __THRUST_WRAPPERS_H__
#define __THRUST_WRAPPERS_H__

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/h2opus_handle.h>
#include <h2opus/core/thrust_runtime.h>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Some array utility functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void exclusiveScan(int *array, size_t num_entries, int *result, int init, h2opusComputeStream_t stream, int hw);
void inclusiveScan(int *array, size_t num_entries, int *result, h2opusComputeStream_t stream, int hw);

// Fill array with a value
void fillArray(float *array, size_t num_entries, float val, h2opusComputeStream_t stream, int hw);
void fillArray(double *array, size_t num_entries, double val, h2opusComputeStream_t stream, int hw);
void fillArray(int *array, size_t num_entries, int val, h2opusComputeStream_t stream, int hw);

// Min and max elements of arrays
int getMaxElement(int *a, size_t elements, h2opusComputeStream_t stream, int hw);
int getMinElement(int *a, size_t elements, h2opusComputeStream_t stream, int hw);
float getMaxElement(float *a, size_t elements, h2opusComputeStream_t stream, int hw);
double getMaxElement(double *a, size_t elements, h2opusComputeStream_t stream, int hw);
float getMaxAbsElement(float *a, size_t elements, h2opusComputeStream_t stream, int hw);
double getMaxAbsElement(double *a, size_t elements, h2opusComputeStream_t stream, int hw);

// Argmax (returns max element and its position in the array)
void argMaxAbsElement(float *a, size_t elements, h2opusComputeStream_t stream, int hw, float &max, size_t &j);
void argMaxAbsElement(double *a, size_t elements, h2opusComputeStream_t stream, int hw, double &max, size_t &j);

// Sum reduction
double reduceSum(double *a, size_t elements, h2opusComputeStream_t stream, int hw);
float reduceSum(float *a, size_t elements, h2opusComputeStream_t stream, int hw);
double reduceAbsSum(double *a, size_t elements, h2opusComputeStream_t stream, int hw);
float reduceAbsSum(float *a, size_t elements, h2opusComputeStream_t stream, int hw);

void copyArray(float *input, float *output, size_t num_entries, h2opusComputeStream_t stream, int hw);
void copyArray(double *input, double *output, size_t num_entries, h2opusComputeStream_t stream, int hw);
void copyArray(float **input, float **output, size_t num_entries, h2opusComputeStream_t stream, int hw);
void copyArray(double **input, double **output, size_t num_entries, h2opusComputeStream_t stream, int hw);
void copyArray(int *input, int *output, size_t num_entries, h2opusComputeStream_t stream, int hw);

// Sign of a vector (-1 if < 0, 1 if >= 0
void signVector(float *v, float *e, size_t n, h2opusComputeStream_t stream, int hw);
void signVector(double *v, double *e, size_t n, h2opusComputeStream_t stream, int hw);

// Set v to a standard basis vector e_j
void standardBasisVector(float *v, size_t n, size_t j, h2opusComputeStream_t stream, int hw);
void standardBasisVector(double *v, size_t n, size_t j, h2opusComputeStream_t stream, int hw);

// A[i] = std::min(V - A[i], 0)
void getRemainingElements(int *a, int v, size_t elements, h2opusComputeStream_t stream, int hw);

// host only
void generateRandomColumn(double *A, size_t n, thrust::minstd_rand &state);
void generateRandomColumn(float *A, size_t n, thrust::minstd_rand &state);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generating array of pointers from either a strided array or another array of pointers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void generateArrayOfPointers(double *original_array, double **array_of_arrays, int stride, int offset,
                             size_t num_arrays, h2opusComputeStream_t stream, int hw);
void generateArrayOfPointers(double *original_array, double **array_of_arrays, int stride, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw);
void generateArrayOfPointers(double **original_array, double **array_of_arrays, int stride, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw);
void generateArrayOfPointers(double **original_array, double **array_of_arrays, int stride, int offset,
                             size_t num_arrays, h2opusComputeStream_t stream, int hw);

void generateArrayOfPointers(float *original_array, float **array_of_arrays, int stride, int offset, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw);
void generateArrayOfPointers(float *original_array, float **array_of_arrays, int stride, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw);
void generateArrayOfPointers(float **original_array, float **array_of_arrays, int stride, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw);
void generateArrayOfPointers(float **original_array, float **array_of_arrays, int stride, int offset, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw);

void generateArrayOfPointers(int *original_array, int **array_of_arrays, int stride, int offset, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw);
void generateArrayOfPointers(int *original_array, int **array_of_arrays, int stride, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw);
void generateArrayOfPointers(int **original_array, int **array_of_arrays, int stride, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw);
void generateArrayOfPointers(int **original_array, int **array_of_arrays, int stride, int offset, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Permuting vectors using an index map
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void permute_vectors(float *original, float *permuted, int n, int num_vectors, int *index_map, int reverse, int hw,
                     h2opusComputeStream_t stream);
void permute_vectors(double *original, double *permuted, int n, int num_vectors, int *index_map, int reverse, int hw,
                     h2opusComputeStream_t stream);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Getting around thrust's annoying inability to simply resize vectors without
// using nvcc as the compiler when device vectors are involved
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU
void resizeThrustArray(thrust::device_vector<float> &array, size_t new_size);
void resizeThrustArray(thrust::device_vector<double> &array, size_t new_size);
void resizeThrustArray(thrust::device_vector<int> &array, size_t new_size);
void resizeThrustArray(thrust::device_vector<float *> &array, size_t new_size);
void resizeThrustArray(thrust::device_vector<double *> &array, size_t new_size);

void resizeThrustArray(typename TTreeContainer<thrust::device_vector<float>>::type &array, size_t new_size);
void resizeThrustArray(typename TTreeContainer<thrust::device_vector<double>>::type &array, size_t new_size);
void resizeThrustArray(typename TTreeContainer<thrust::device_vector<int>>::type &array, size_t new_size);

void copyThrustArray(thrust::device_vector<double> &dest, const thrust::device_vector<double> &src);
void copyThrustArray(thrust::device_vector<float> &dest, const thrust::device_vector<float> &src);
void copyThrustArray(thrust::device_vector<int> &dest, const thrust::device_vector<int> &src);

void copyThrustArray(thrust::host_vector<double> &dest, const thrust::device_vector<double> &src);
void copyThrustArray(thrust::host_vector<float> &dest, const thrust::device_vector<float> &src);
void copyThrustArray(thrust::host_vector<int> &dest, const thrust::device_vector<int> &src);

void copyThrustArray(thrust::device_vector<double> &dest, const thrust::host_vector<double> &src);
void copyThrustArray(thrust::device_vector<float> &dest, const thrust::host_vector<float> &src);
void copyThrustArray(thrust::device_vector<int> &dest, const thrust::host_vector<int> &src);
#endif

void resizeThrustArray(thrust::host_vector<float> &array, size_t new_size);
void resizeThrustArray(thrust::host_vector<double> &array, size_t new_size);
void resizeThrustArray(thrust::host_vector<int> &array, size_t new_size);
void resizeThrustArray(thrust::host_vector<float *> &array, size_t new_size);
void resizeThrustArray(thrust::host_vector<double *> &array, size_t new_size);

void resizeThrustArray(typename TTreeContainer<thrust::host_vector<float>>::type &array, size_t new_size);
void resizeThrustArray(typename TTreeContainer<thrust::host_vector<double>>::type &array, size_t new_size);
void resizeThrustArray(typename TTreeContainer<thrust::host_vector<int>>::type &array, size_t new_size);

void copyThrustArray(thrust::host_vector<double> &dest, const thrust::host_vector<double> &src);
void copyThrustArray(thrust::host_vector<float> &dest, const thrust::host_vector<float> &src);
void copyThrustArray(thrust::host_vector<int> &dest, const thrust::host_vector<int> &src);

template <class T> void copyThrustArray(std::vector<T> &dest, const std::vector<T> &src)
{
    dest = src;
}
template <class T> void resizeThrustArray(std::vector<T> &array, size_t new_size)
{
    array.resize(new_size);
}
template <class T> void resizeThrustArray(typename TTreeContainer<thrust::host_vector<T>>::type &array, size_t new_size)
{
    array.resize(new_size);
}

#endif
