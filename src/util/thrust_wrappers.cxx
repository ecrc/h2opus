#include <h2opus/util/thrust_wrappers.h>

#include <thrust/host_vector.h>
#ifdef H2OPUS_USE_GPU
#include <thrust/device_vector.h>
#endif

#include <iostream>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform_scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Some array utility functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int exclusiveScan(const int *array, size_t num_entries, int *result, int init, h2opusComputeStream_t stream, int hw)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
    {
        const thrust::device_ptr<const int> start(array);
        thrust::device_ptr<int> res(result);
        thrust::device_ptr<int> end_ptr = thrust::exclusive_scan(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), start,
                                                                 start + num_entries, res, init);
        return *(end_ptr - 1);
    }
    else
#endif
    {
        int *end_ptr = thrust::exclusive_scan(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), array, array + num_entries,
                                              result, init);
        return *(end_ptr - 1);
    }
}

int inclusiveScan(const int *array, size_t num_entries, int *result, h2opusComputeStream_t stream, int hw)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
    {
        const thrust::device_ptr<const int> start(array);
        thrust::device_ptr<int> res(result);

        thrust::device_ptr<int> end_ptr =
            thrust::inclusive_scan(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), start, start + num_entries, res);
        return *(end_ptr - 1);
    }
    else
#endif
    {
        int *end_ptr =
            thrust::inclusive_scan(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), array, array + num_entries, result);
        return *(end_ptr - 1);
    }
}

void sortByKey(int *keys, int *values, size_t elements, bool ascending, h2opusComputeStream_t stream, int hw)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
    {
        thrust::sort_by_key(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), keys, keys + elements, values,
                            (ascending ? thrust::greater<int>() : thrust::greater<int>()));
    }
    else
#endif
    {
        thrust::sort_by_key(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), keys, keys + elements, values,
                            (ascending ? thrust::greater<int>() : thrust::greater<int>()));
    }
}

struct which_segment : thrust::unary_function<int, int>
{
    int seg_size;

    __host__ __device__ which_segment(int seg_size)
    {
        this->seg_size = seg_size;
    }

    inline __host__ __device__ int operator()(int idx) const
    {
        return idx / seg_size;
    }
};

int getSegmentedMaxElements(int *a, size_t elements, size_t seg_size, int *seg_maxes, h2opusComputeStream_t stream,
                            int hw)
{
    thrust::counting_iterator<int> c_first(0);
    thrust::transform_iterator<which_segment, thrust::counting_iterator<int>> t_first(c_first, which_segment(seg_size));

    thrust::equal_to<int> binary_pred;
    thrust::maximum<int> binary_op;

#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
    {
        auto end = thrust::reduce_by_key(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), t_first, t_first + elements, a,
                                         thrust::make_discard_iterator(), seg_maxes, binary_pred, binary_op);

        return end.second - seg_maxes;
    }
    else
#endif
    {
        auto end = thrust::reduce_by_key(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), t_first, t_first + elements, a,
                                         thrust::make_discard_iterator(), seg_maxes, binary_pred, binary_op);

        return end.second - seg_maxes;
    }
}

template <typename T> struct absolute_value : public thrust::unary_function<T, T>
{
    inline __host__ __device__ T operator()(const T &x) const
    {
        return (x < 0 ? -x : x);
    }
};

// assumes entries of a are all positive
template <class T> inline T getMinElementT(T *a, size_t elements, h2opusComputeStream_t stream, int hw)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
        return thrust::reduce(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), a, a + elements, (T)0,
                              thrust::minimum<T>());
    else
#endif
        return thrust::reduce(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), a, a + elements, (T)0,
                              thrust::minimum<T>());
}

// assumes entries of a are all positive
template <class T> inline T getMaxElementT(T *a, size_t elements, h2opusComputeStream_t stream, int hw)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
        return thrust::reduce(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), a, a + elements, (T)0,
                              thrust::maximum<T>());
    else
#endif
    {
        return thrust::reduce(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), a, a + elements, (T)0,
                              thrust::maximum<T>());
    }
}

template <class T> inline T getMaxAbsElementT(T *a, size_t elements, h2opusComputeStream_t stream, int hw)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
        return thrust::transform_reduce(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), a, a + elements,
                                        absolute_value<T>(), (T)0, thrust::maximum<T>());
    else
#endif
        return thrust::transform_reduce(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), a, a + elements,
                                        absolute_value<T>(), (T)0, thrust::maximum<T>());
}

int getMaxElement(int *a, size_t elements, h2opusComputeStream_t stream, int hw)
{
    return getMaxElementT<int>(a, elements, stream, hw);
}

int getMinElement(int *a, size_t elements, h2opusComputeStream_t stream, int hw)
{
    return getMinElementT<int>(a, elements, stream, hw);
}

float getMaxElement(float *a, size_t elements, h2opusComputeStream_t stream, int hw)
{
    return getMaxElementT<float>(a, elements, stream, hw);
}

double getMaxElement(double *a, size_t elements, h2opusComputeStream_t stream, int hw)
{
    return getMaxElementT<double>(a, elements, stream, hw);
}

float getMaxAbsElement(float *a, size_t elements, h2opusComputeStream_t stream, int hw)
{
    return getMaxAbsElementT<float>(a, elements, stream, hw);
}

double getMaxAbsElement(double *a, size_t elements, h2opusComputeStream_t stream, int hw)
{
    return getMaxAbsElementT<double>(a, elements, stream, hw);
}

template <class T> struct argmaxabs_reduction_functor
{
    inline __host__ __device__ thrust::tuple<T, size_t> operator()(const thrust::tuple<T, size_t> &lhs,
                                                                   const thrust::tuple<T, size_t> &rhs)
    {
        if (fabs(thrust::get<0>(lhs)) < fabs(thrust::get<0>(rhs)))
            return rhs;
        if (fabs(thrust::get<0>(lhs)) > fabs(thrust::get<0>(rhs)))
            return lhs;
        if (thrust::get<1>(lhs) < thrust::get<1>(rhs))
            return lhs;
        else
            return rhs;
    }
};

template <class T>
inline void argMaxAbsElementT(T *a, size_t elements, h2opusComputeStream_t stream, int hw, T &max, size_t &j)
{
    if (elements == 0)
        return;

    typedef thrust::tuple<T, size_t> ArgMaxTuple;
    ArgMaxTuple initial = ArgMaxTuple(0, 0);
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
    {
        ArgMaxTuple result = thrust::reduce(
            ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream),
            thrust::make_zip_iterator(thrust::make_tuple(a, thrust::counting_iterator<size_t>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(a, thrust::counting_iterator<size_t>(0))) + elements, initial,
            argmaxabs_reduction_functor<T>());

        j = thrust::get<1>(result);
        max = fabs(thrust::get<0>(result));
    }
    else
#endif
    {
        ArgMaxTuple result = thrust::reduce(
            ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream),
            thrust::make_zip_iterator(thrust::make_tuple(a, thrust::counting_iterator<size_t>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(a, thrust::counting_iterator<size_t>(0))) + elements, initial,
            argmaxabs_reduction_functor<T>());

        j = thrust::get<1>(result);
        max = fabs(thrust::get<0>(result));
    }
}

void argMaxAbsElement(float *a, size_t elements, h2opusComputeStream_t stream, int hw, float &max, size_t &j)
{
    argMaxAbsElementT<float>(a, elements, stream, hw, max, j);
}

void argMaxAbsElement(double *a, size_t elements, h2opusComputeStream_t stream, int hw, double &max, size_t &j)
{
    argMaxAbsElementT<double>(a, elements, stream, hw, max, j);
}

template <class T> inline T reduceSumT(T *a, size_t elements, h2opusComputeStream_t stream, int hw)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
        return thrust::reduce(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), a, a + elements);
    else
#endif
        return thrust::reduce(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), a, a + elements);
}

double reduceSum(double *a, size_t elements, h2opusComputeStream_t stream, int hw)
{
    return reduceSumT<double>(a, elements, stream, hw);
}

float reduceSum(float *a, size_t elements, h2opusComputeStream_t stream, int hw)
{
    return reduceSumT<float>(a, elements, stream, hw);
}

template <class T> inline T reduceAbsSumT(T *a, size_t elements, h2opusComputeStream_t stream, int hw)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
        return thrust::transform_reduce(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), a, a + elements,
                                        absolute_value<T>(), (T)0, thrust::plus<T>());
    else
#endif
        return thrust::transform_reduce(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), a, a + elements,
                                        absolute_value<T>(), (T)0, thrust::plus<T>());
}

double reduceAbsSum(double *a, size_t elements, h2opusComputeStream_t stream, int hw)
{
    return reduceAbsSumT<double>(a, elements, stream, hw);
}

float reduceAbsSum(float *a, size_t elements, h2opusComputeStream_t stream, int hw)
{
    return reduceAbsSumT<float>(a, elements, stream, hw);
}

template <class T> inline void fillArrayT(T *array, size_t num_entries, T val, h2opusComputeStream_t stream, int hw)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
        thrust::fill(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), array, array + num_entries, val);
    else
#endif
#ifdef H2OPUS_USE_NEC
#pragma omp parallel for firstprivate(val)
        for (size_t e = 0; e < num_entries; e++)
            array[e] = val;
#else
    thrust::fill(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), array, array + num_entries, val);
#endif
}

void fillArray(float *array, size_t num_entries, float val, h2opusComputeStream_t stream, int hw)
{
    fillArrayT<float>(array, num_entries, val, stream, hw);
}

void fillArray(double *array, size_t num_entries, double val, h2opusComputeStream_t stream, int hw)
{
    fillArrayT<double>(array, num_entries, val, stream, hw);
}

void fillArray(int *array, size_t num_entries, int val, h2opusComputeStream_t stream, int hw)
{
    fillArrayT<int>(array, num_entries, val, stream, hw);
}

void fillArray(float **array, size_t num_entries, float *val, h2opusComputeStream_t stream, int hw)
{
    fillArrayT<float *>(array, num_entries, val, stream, hw);
}

void fillArray(double **array, size_t num_entries, double *val, h2opusComputeStream_t stream, int hw)
{
    fillArrayT<double *>(array, num_entries, val, stream, hw);
}

void generateSequence(int *array, size_t num_entries, int start_val, h2opusComputeStream_t stream, int hw)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
        thrust::sequence(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), array, array + num_entries, start_val);
    else
#endif
#ifdef H2OPUS_USE_NEC
#pragma omp parallel for firstprivate(start_val)
        for (size_t e = 0; e < num_entries; e++)
            array[e] = start_val + e;
#else
    thrust::sequence(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), array, array + num_entries, start_val);
#endif
}

template <class T> inline void copyArrayT(T *input, T *output, size_t num_entries, h2opusComputeStream_t stream, int hw)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
        thrust::copy(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), input, input + num_entries, output);
    else
#endif
#ifdef H2OPUS_USE_NEC
#pragma omp parallel for
        for (size_t e = 0; e < num_entries; e++)
            output[e] = input[e];
#else
    thrust::copy(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), input, input + num_entries, output);
#endif
}

void copyArray(float *input, float *output, size_t num_entries, h2opusComputeStream_t stream, int hw)
{
    copyArrayT<float>(input, output, num_entries, stream, hw);
}

void copyArray(double *input, double *output, size_t num_entries, h2opusComputeStream_t stream, int hw)
{
    copyArrayT<double>(input, output, num_entries, stream, hw);
}

void copyArray(int *input, int *output, size_t num_entries, h2opusComputeStream_t stream, int hw)
{
    copyArrayT<int>(input, output, num_entries, stream, hw);
}

void copyArray(float **input, float **output, size_t num_entries, h2opusComputeStream_t stream, int hw)
{
    copyArrayT<float *>(input, output, num_entries, stream, hw);
}

void copyArray(double **input, double **output, size_t num_entries, h2opusComputeStream_t stream, int hw)
{
    copyArrayT<double *>(input, output, num_entries, stream, hw);
}

template <class T> struct SignVector_Functor
{
    T *v, *e;
    SignVector_Functor(T *v, T *e)
    {
        this->v = v;
        this->e = e;
    }
    inline __host__ __device__ void operator()(size_t i)
    {
        e[i] = (v[i] >= 0 ? 1 : -1);
    }
};

template <class T> void signVectorT(T *v, T *e, size_t n, h2opusComputeStream_t stream, int hw)
{
    SignVector_Functor<T> sign_vector(v, e);
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
    {
        thrust::for_each(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), thrust::counting_iterator<size_t>(0),
                         thrust::counting_iterator<size_t>(n), sign_vector);
    }
    else
#endif
    {
        thrust::for_each(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), thrust::counting_iterator<size_t>(0),
                         thrust::counting_iterator<size_t>(n), sign_vector);
    }
}

void signVector(float *v, float *e, size_t n, h2opusComputeStream_t stream, int hw)
{
    signVectorT<float>(v, e, n, stream, hw);
}

void signVector(double *v, double *e, size_t n, h2opusComputeStream_t stream, int hw)
{
    signVectorT<double>(v, e, n, stream, hw);
}

template <class T> struct StandardBasis_Functor
{
    T *v;
    size_t j;

    StandardBasis_Functor(T *v, size_t j)
    {
        this->v = v;
        this->j = j;
    }
    inline __host__ __device__ void operator()(size_t i)
    {
        v[i] = (i == j ? 1 : 0);
    }
};

template <class T> void standardBasisVectorT(T *v, size_t n, size_t j, h2opusComputeStream_t stream, int hw)
{
    StandardBasis_Functor<T> standard_basis(v, j);
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
    {
        thrust::for_each(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), thrust::counting_iterator<size_t>(0),
                         thrust::counting_iterator<size_t>(n), standard_basis);
    }
    else
#endif
    {
        thrust::for_each(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), thrust::counting_iterator<size_t>(0),
                         thrust::counting_iterator<size_t>(n), standard_basis);
    }
}

void standardBasisVector(float *v, size_t n, size_t j, h2opusComputeStream_t stream, int hw)
{
    standardBasisVectorT<float>(v, n, j, stream, hw);
}

void standardBasisVector(double *v, size_t n, size_t j, h2opusComputeStream_t stream, int hw)
{
    standardBasisVectorT<double>(v, n, j, stream, hw);
}

struct RemElements_Functor
{
    int *a, v;

    RemElements_Functor(int *a, int v)
    {
        this->a = a;
        this->v = v;
    }

    inline __host__ __device__ void operator()(size_t i)
    {
        int val = v - a[i];
        if (val < 0)
            val = 0;

        a[i] = val;
    }
};

void getRemainingElements(int *a, int v, size_t elements, h2opusComputeStream_t stream, int hw)
{
    RemElements_Functor rem_elements(a, v);
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
    {
        thrust::for_each(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), thrust::counting_iterator<size_t>(0),
                         thrust::counting_iterator<size_t>(elements), rem_elements);
    }
    else
#endif
    {
        thrust::for_each(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), thrust::counting_iterator<size_t>(0),
                         thrust::counting_iterator<size_t>(elements), rem_elements);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Permuting vectors using an index map
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int reverse> struct vector_permutor : public thrust::unary_function<int, int>
{
    int *index_map, n, num_vectors;
    T *original, *permuted;

    vector_permutor(int *index_map, T *original, T *permuted, int n, int num_vectors)
    {
        this->index_map = index_map;
        this->original = original;
        this->permuted = permuted;
        this->n = n;
        this->num_vectors = num_vectors;
    }

    inline __host__ __device__ void operator()(int original_index)
    {
        int permuted_index = index_map[original_index];
        int input_index = (reverse ? original_index : permuted_index);
        int output_index = (reverse ? permuted_index : original_index);

        for (int i = 0; i < num_vectors; i++)
        {
            T *x = original + i * n;
            T *px = permuted + i * n;

            px[output_index] = x[input_index];
        }
    }
};

template <class T, int hw>
inline void permute_vectors_template(T *original, T *permuted, int n, int num_vectors, int *index_map, int reverse,
                                     h2opusComputeStream_t stream)
{
#ifdef H2OPUS_USE_NEC
    if (reverse == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < n; i++)
            permuted[i] = original[index_map[i]];
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < n; i++)
            permuted[index_map[i]] = original[i];
    }
#else
    if (reverse == 0)
    {
        thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(n),
                         vector_permutor<T, 0>(index_map, original, permuted, n, num_vectors));
    }
    else
    {
        thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(n),
                         vector_permutor<T, 1>(index_map, original, permuted, n, num_vectors));
    }
#endif
}

void permute_vectors(float *original, float *permuted, int n, int num_vectors, int *index_map, int reverse, int hw,
                     h2opusComputeStream_t stream)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
        permute_vectors_template<float, H2OPUS_HWTYPE_GPU>(original, permuted, n, num_vectors, index_map, reverse,
                                                           stream);
    else
#endif
        permute_vectors_template<float, H2OPUS_HWTYPE_CPU>(original, permuted, n, num_vectors, index_map, reverse,
                                                           stream);
}

void permute_vectors(double *original, double *permuted, int n, int num_vectors, int *index_map, int reverse, int hw,
                     h2opusComputeStream_t stream)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
        permute_vectors_template<double, H2OPUS_HWTYPE_GPU>(original, permuted, n, num_vectors, index_map, reverse,
                                                            stream);
    else
#endif
        permute_vectors_template<double, H2OPUS_HWTYPE_CPU>(original, permuted, n, num_vectors, index_map, reverse,
                                                            stream);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Swap two vectors
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct vector_swapper : public thrust::unary_function<int, int>
{
    int incx, incy;
    T *x, *y;

    vector_swapper(T *x, T *y, int incx, int incy)
    {
        this->x = x;
        this->y = y;
        this->incx = incx;
        this->incy = incy;
    }

    inline __host__ __device__ void operator()(int i)
    {
        int ix = i * incx, iy = i * incy;

        T tmp = x[ix];
        x[ix] = y[iy];
        y[iy] = tmp;
    }
};

template <class T, int hw>
inline void swap_vectors_template(int n, T *x, int incx, T *y, int incy, h2opusComputeStream_t stream)
{
    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(n), vector_swapper<T>(x, y, incx, incy));
}

void swap_vectors(int n, float *x, int incx, float *y, int incy, int hw, h2opusComputeStream_t stream)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
        swap_vectors_template<float, H2OPUS_HWTYPE_GPU>(n, x, incx, y, incy, stream);
    else
#endif
        swap_vectors_template<float, H2OPUS_HWTYPE_CPU>(n, x, incx, y, incy, stream);
}

void swap_vectors(int n, double *x, int incx, double *y, int incy, int hw, h2opusComputeStream_t stream)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
        swap_vectors_template<double, H2OPUS_HWTYPE_GPU>(n, x, incx, y, incy, stream);
    else
#endif
        swap_vectors_template<double, H2OPUS_HWTYPE_CPU>(n, x, incx, y, incy, stream);
}

void swap_vectors(int n, float **x, int incx, float **y, int incy, int hw, h2opusComputeStream_t stream)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
        swap_vectors_template<float *, H2OPUS_HWTYPE_GPU>(n, x, incx, y, incy, stream);
    else
#endif
        swap_vectors_template<float *, H2OPUS_HWTYPE_CPU>(n, x, incx, y, incy, stream);
}

void swap_vectors(int n, double **x, int incx, double **y, int incy, int hw, h2opusComputeStream_t stream)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
        swap_vectors_template<double *, H2OPUS_HWTYPE_GPU>(n, x, incx, y, incy, stream);
    else
#endif
        swap_vectors_template<double *, H2OPUS_HWTYPE_CPU>(n, x, incx, y, incy, stream);
}

void swap_vectors(int n, int *x, int incx, int *y, int incy, int hw, h2opusComputeStream_t stream)
{
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
        swap_vectors_template<int, H2OPUS_HWTYPE_GPU>(n, x, incx, y, incy, stream);
    else
#endif
        swap_vectors_template<int, H2OPUS_HWTYPE_CPU>(n, x, incx, y, incy, stream);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generating array of pointers from either a strided array or another array of pointers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> inline __host__ __device__ T *getOperationPtr(T *array, size_t op_id, int stride)
{
    return array + op_id * stride;
}

template <class T> inline __host__ __device__ T *getOperationPtr(T **array, size_t op_id, int stride)
{
    return array[op_id];
}

template <class T, class T_ptr> struct UnaryAoAAssign
{
    T_ptr original_array;
    int stride, offset;
    T **output;

    UnaryAoAAssign(T **output, T_ptr original_array, int stride, int offset)
    {
        this->output = output;
        this->original_array = original_array;
        this->stride = stride;
        this->offset = offset;
    }

    inline __host__ __device__ void operator()(const size_t &index) const
    {
        output[index] = getOperationPtr<T>(original_array, index, stride) + offset;
    }
};

template <class T, class T_ptr>
inline void generateArrayOfPointersT(T_ptr original_array, T **array_of_arrays, int stride, int offset,
                                     size_t num_arrays, h2opusComputeStream_t stream, int hw)
{
    UnaryAoAAssign<T, T_ptr> aoa_functor(array_of_arrays, original_array, stride, offset);
#ifdef H2OPUS_USE_GPU
    if (hw == H2OPUS_HWTYPE_GPU)
    {
        thrust::for_each(ThrustRuntime<H2OPUS_HWTYPE_GPU>::get(stream), thrust::counting_iterator<size_t>(0),
                         thrust::counting_iterator<size_t>(num_arrays), aoa_functor);
    }
    else
#endif
    {
        thrust::for_each(ThrustRuntime<H2OPUS_HWTYPE_CPU>::get(stream), thrust::counting_iterator<size_t>(0),
                         thrust::counting_iterator<size_t>(num_arrays), aoa_functor);
    }
}

void generateArrayOfPointers(double *original_array, double **array_of_arrays, int stride, int offset,
                             size_t num_arrays, h2opusComputeStream_t stream, int hw)
{
    generateArrayOfPointersT<double, double *>(original_array, array_of_arrays, stride, offset, num_arrays, stream, hw);
}

void generateArrayOfPointers(double **original_array, double **array_of_arrays, int stride, int offset,
                             size_t num_arrays, h2opusComputeStream_t stream, int hw)
{
    generateArrayOfPointersT<double, double **>(original_array, array_of_arrays, stride, offset, num_arrays, stream,
                                                hw);
}

void generateArrayOfPointers(float *original_array, float **array_of_arrays, int stride, int offset, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw)
{
    generateArrayOfPointersT<float, float *>(original_array, array_of_arrays, stride, offset, num_arrays, stream, hw);
}

void generateArrayOfPointers(float **original_array, float **array_of_arrays, int stride, int offset, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw)
{
    generateArrayOfPointersT<float, float **>(original_array, array_of_arrays, stride, offset, num_arrays, stream, hw);
}

void generateArrayOfPointers(int *original_array, int **array_of_arrays, int stride, int offset, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw)
{
    generateArrayOfPointersT<int, int *>(original_array, array_of_arrays, stride, offset, num_arrays, stream, hw);
}

void generateArrayOfPointers(int **original_array, int **array_of_arrays, int stride, int offset, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw)
{
    generateArrayOfPointersT<int, int **>(original_array, array_of_arrays, stride, offset, num_arrays, stream, hw);
}

void generateArrayOfPointers(int **original_array, int **array_of_arrays, int stride, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw)
{
    generateArrayOfPointersT<int, int **>(original_array, array_of_arrays, stride, 0, num_arrays, stream, hw);
}

void generateArrayOfPointers(int *original_array, int **array_of_arrays, int stride, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw)
{
    generateArrayOfPointersT<int, int *>(original_array, array_of_arrays, stride, 0, num_arrays, stream, hw);
}

void generateArrayOfPointers(float **original_array, float **array_of_arrays, int stride, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw)
{
    generateArrayOfPointersT<float, float **>(original_array, array_of_arrays, stride, 0, num_arrays, stream, hw);
}

void generateArrayOfPointers(double *original_array, double **array_of_arrays, int stride, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw)
{
    generateArrayOfPointersT<double, double *>(original_array, array_of_arrays, stride, 0, num_arrays, stream, hw);
}

void generateArrayOfPointers(double **original_array, double **array_of_arrays, int stride, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw)
{
    generateArrayOfPointersT<double, double **>(original_array, array_of_arrays, stride, 0, num_arrays, stream, hw);
}

void generateArrayOfPointers(float *original_array, float **array_of_arrays, int stride, size_t num_arrays,
                             h2opusComputeStream_t stream, int hw)
{
    generateArrayOfPointersT<float, float *>(original_array, array_of_arrays, stride, 0, num_arrays, stream, hw);
}
