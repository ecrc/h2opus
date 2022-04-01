// routines to be called from python

#include <h2opus.h>
#include <h2opus/util/boxentrygen.h>

struct PyH2Mat
{
    int n;
    int leaf_size;
    int depth;
    int precision;
    int *index_map;

    int num_dense_leaves;
    int *dense_node_indexes;
    int *u_index;
    int *v_index;
    H2Opus_Real *dense_leaf_mem;

    int *level_rank;
    int *num_lr_leaves;
    int **lr_node_indexes;
    H2Opus_Real **lr_leaf_mem;

    int *trans_dim;
    H2Opus_Real *basis_mem;  // RealVector basis_mem;
    H2Opus_Real **trans_mem; // TreeContainer trans_mem;
};

struct PyH2Mat make_h2mat(HMatrix &hmatrix)
{
    struct PyH2Mat h2mat;
    h2mat.n = hmatrix.n;
    h2mat.leaf_size = hmatrix.hnodes.leaf_size; // from HNodeTree
    h2mat.depth = hmatrix.hnodes.depth;         // from HNodeTree
    h2mat.precision = sizeof(H2Opus_Real);
    h2mat.index_map = hmatrix.u_basis_tree.index_map.data();

    // dense blocks
    h2mat.num_dense_leaves = hmatrix.hnodes.num_dense_leaves;
    h2mat.dense_node_indexes = hmatrix.hnodes.bsn_row_data.dense_node_indexes.data(); // tree indices
    h2mat.u_index = hmatrix.hnodes.node_u_index.data();
    h2mat.v_index = hmatrix.hnodes.node_v_index.data();
    h2mat.dense_leaf_mem = hmatrix.hnodes.dense_leaf_mem.data();

    // low rank blocks
    h2mat.level_rank = hmatrix.hnodes.level_data.level_rank.data();
    h2mat.num_lr_leaves = (int *)malloc(h2mat.depth * sizeof(int));
    h2mat.lr_node_indexes = (int **)malloc(h2mat.depth * sizeof(int *));
    h2mat.lr_leaf_mem = (H2Opus_Real **)malloc(h2mat.depth * sizeof(H2Opus_Real *));
    for (int l = 0; l < h2mat.depth; l++)
    {
        h2mat.num_lr_leaves[l] = hmatrix.hnodes.bsn_row_data.coupling_node_indexes[l].size();
        h2mat.lr_node_indexes[l] = hmatrix.hnodes.bsn_row_data.coupling_node_indexes[l].data();
        h2mat.lr_leaf_mem[l] = hmatrix.hnodes.rank_leaf_mem[l].data();
    }

    // basis tree
    h2mat.trans_dim = hmatrix.u_basis_tree.level_data.trans_dim.data();
    h2mat.basis_mem = hmatrix.u_basis_tree.basis_mem.data();
    h2mat.trans_mem = (H2Opus_Real **)malloc(h2mat.depth * sizeof(H2Opus_Real *));
    for (int l = 0; l < h2mat.depth; l++)
    {
        h2mat.trans_mem[l] = hmatrix.u_basis_tree.trans_mem[l].data();
    }

    return h2mat;
}

// TODO make a gallery?
#include "../../../examples/common/example_problem.h"

extern "C" struct PyH2Mat build_hmatrix(int grid_x, int grid_y, int leaf_size, int cheb_grid_pts, H2Opus_Real eta)
{
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Geometry
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    size_t n = grid_x * grid_y;
    int dim = 2;
    PointCloud<H2Opus_Real> pt_cloud(dim, n);
    generate2DGrid<H2Opus_Real>(pt_cloud, grid_x, grid_y, 0, 1, 0, 1);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Matrix construction
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Setup hmatrix construction parameters:
    // Create a functor that can generate the matrix entries from two points
    FunctionGen<H2Opus_Real> func_gen(dim);
    // Create an entry gen struct from the functor. Currently only supports chebyshev interpolation on the CPU
    BoxEntryGen<H2Opus_Real, H2OPUS_HWTYPE_CPU, FunctionGen<H2Opus_Real>> entry_gen(func_gen);
    // Create the admissibility condition using the eta parameter
    // Decreasing eta refines the matrix tree and increasing it coarsens the tree
    H2OpusBoxCenterAdmissibility admissibility(eta);
    // Build the hmatrix. Currently only symmetric matrices are fully supported
    HMatrix *phmatrix = new HMatrix(n, true);
    HMatrix &hmatrix = *phmatrix;
    buildHMatrix(hmatrix, &pt_cloud, admissibility, entry_gen, leaf_size, cheb_grid_pts);

    return make_h2mat(hmatrix);
}
