Example 1: test_hgemv.cpp
=========================

This example file generates a hierarchical matrix for a spatial covariance matrix in either 2D or 3D, and computes matrix-vector products either with a single vector or multiple vectors. The multiple vector product has higher arithmetic intensity and achieves much higher performance, especially on GPUs.

Matrix Construction
-------------------

The first operation of the code is to generate the matrix. This is done in lines 63-65

```c++
    [....]
    // Build the hmatrix. Currently only symmetric matrices are fully supported 
    HMatrix hmatrix(n, true);
    buildHMatrix(hmatrix, &pt_cloud, admissibility, entry_gen, leaf_size, cheb_grid_pts);
```

An nxn symmetric hierarchical matrix object of type ```HMatrix``` is declared. The function ```buildHMatrix()``` first builds the hierarchical matrix partitioning using a given point cloud and an admissibility condition that are passed in the second and third arguments. The subdivision of the matrix stops when blocks of size leaf_size, passed in the fifth argument, are reached. The nested bases U and V and the coupling matrices S are then generated using an interpolation of the kernel passed in the fourth argument entry_gen. The construction uses a tensor product Chebyshev interpolation polynomial defined in the last argument cheb_grid_pts. The U and V trees and S blocks of the matrix are computed and stored in the ```HMatrix``` passed in the first argument. 

Matrix-vector multiplication
----------------------------

The second operation of the code uses the hierarchical matrix thus generated to perform matrix-vector multiplication. This is done in lines 122-136:

```c++
    // copy x to the GPU, and set size of result 
    thrust::device_vector<H2Opus_Real> gpu_x = x, gpu_y;
    gpu_y.resize(n * num_vectors);

    // Copy the hmatrix over to the GPU
    HMatrix_GPU gpu_h = hmatrix;

    // Set the workspace in the handle for host and gpu
    H2OpusWorkspaceState ws_needed_gpu = hgemv_workspace(gpu_h, H2Opus_NoTrans, num_vectors);
    ara_handle->setWorkspaceState(ws_needed_gpu);

    // GPU execution
    fillArray(vec_ptr(gpu_y), n * num_vectors, 0, h2opus_handle->getMainStream(), H2OPUS_HWTYPE_GPU);
    hgemv(H2Opus_NoTrans, alpha, gpu_h, vec_ptr(gpu_x), n, beta, vec_ptr(gpu_y), n, num_vectors, h2opus_handle);
```

The last line in the snippet above performs the matrix-multiple-vector operation by calling the ```hgemv()``` routine. The parameters of ```hgemv``` are inspired by the Level 2 BLAS xGEMV. The routine computes ```y = alpha A x + beta y``` or ```y = alpha A^T x + beta y```, where ```x``` and ```y``` are GPU-resident vectors or multiple vectors of size ```n```.
