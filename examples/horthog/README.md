Example 2: test_horthog.cpp
=========================

This example file generates a hierarchical matrix for a spatial covariance matrix in either 2D or 3D as in Example 1. The bases of the hierarchical matrix are not orthogonal by construction. The function ```horthog()``` generates appropriate nested and othogonal bases. These bases have the property that at the leaf level U<sup>T</sup> U = I for every leaf node of the basis tree, and that
the sum of the interlevel transfer matrices over the children of every non-leaf node is also the identity matrix, <math>&Sigma;<sub>c</sub> </math> E<sup>T</sup> E = I.

Orthogonalization
-----------------

After the matrix ```hmatrix``` is generated, orthogonalization is performed as follows: 

```c++
    [....]
    // Build an H^2 matrix
    HMatrix hmatrix(n, true);
    buildHMatrix(hmatrix, &pt_cloud, admissibility, entry_gen, leaf_size, cheb_grid_pts);
    HMatrix_GPU gpu_h = hmatrix;

    // Create h2opus handle
    h2opusHandle_t h2opus_handle;
    h2opusCreateHandle(&h2opus_handle);

    // Pre-allocate workspace. This can be safely skipped if performance is not a concern
    ws_needed = horthog_workspace(gpu_h);
    h2opus_handle->setWorkspaceState(ws_needed);

    // Orthogonalization
    horthog(gpu_h, h2opus_handle);

    // Make sure the basis is orthogonal
    HMatrix gpu_orthog_hmatrix = gpu_h;
    printf("GPU Basis orthogonality: %e\n", getBasisOrthogonality(gpu_orthog_hmatrix.u_basis_tree, false));
```

Approximation of the 2-norm of the difference of the matrices expressed in the original non-orthogonal bases and in the orthogonal ones can also be computed to verify that is essentially zero. Performance metrics of each phase of the computation can also be collected as illustrated in the code.
