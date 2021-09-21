#ifndef __THRUST_WRAPPERS_H__
#define __THRUST_WRAPPERS_H__

#include <h2opus/core/thrust_runtime.h>
#include <thrust/random.h>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Some array utility functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int exclusiveScan(const int *array, size_t num_entries, int *result, int init, h2opusComputeStream_t stream, int hw);
int inclusiveScan(const int *array, size_t num_entries, int *result, h2opusComputeStream_t stream, int hw);

// Fill array with a value
void fillArray(float *array, size_t num_entries, float val, h2opusComputeStream_t stream, int hw);
void fillArray(double *array, size_t num_entries, double val, h2opusComputeStream_t stream, int hw);
void fillArray(int *array, size_t num_entries, int val, h2opusComputeStream_t stream, int hw);
void fillArray(float **array, size_t num_entries, float *val, h2opusComputeStream_t stream, int hw);
void fillArray(double **array, size_t num_entries, double *val, h2opusComputeStream_t stream, int hw);

// Generate a sequence
void generateSequence(int *array, size_t num_entries, int start_val, h2opusComputeStream_t stream, int hw);

// Min and max elements of arrays
int getMaxElement(int *a, size_t elements, h2opusComputeStream_t stream, int hw);
int getMinElement(int *a, size_t elements, h2opusComputeStream_t stream, int hw);
float getMaxElement(float *a, size_t elements, h2opusComputeStream_t stream, int hw);
double getMaxElement(double *a, size_t elements, h2opusComputeStream_t stream, int hw);
float getMaxAbsElement(float *a, size_t elements, h2opusComputeStream_t stream, int hw);
double getMaxAbsElement(double *a, size_t elements, h2opusComputeStream_t stream, int hw);

// Segmented reductions
int getSegmentedMaxElements(int *a, size_t elements, size_t seg_size, int *seg_maxes, h2opusComputeStream_t stream,
                            int hw);

// Sort by key
void sortByKey(int *keys, int *values, size_t elements, bool ascending, h2opusComputeStream_t stream, int hw);

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
// Swap two vectors
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void swap_vectors(int n, float *x, int incx, float *y, int incy, int hw, h2opusComputeStream_t stream);
void swap_vectors(int n, double *x, int incx, double *y, int incy, int hw, h2opusComputeStream_t stream);
void swap_vectors(int n, float **x, int incx, float **y, int incy, int hw, h2opusComputeStream_t stream);
void swap_vectors(int n, double **x, int incx, double **y, int incy, int hw, h2opusComputeStream_t stream);
void swap_vectors(int n, int *x, int incx, int *y, int incy, int hw, h2opusComputeStream_t stream);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// inline functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <int hw> inline int thrust_get_value(int *ptr);

template <> inline int thrust_get_value<H2OPUS_HWTYPE_CPU>(int *ptr)
{
    return *ptr;
}

#ifdef H2OPUS_USE_GPU
template <> inline int thrust_get_value<H2OPUS_HWTYPE_GPU>(int *ptr)
{
    return *thrust::device_ptr<int>(ptr);
}
#endif

#endif
