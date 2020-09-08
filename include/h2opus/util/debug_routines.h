#ifndef __DEBUG_ROUTINES_H__
#define __DEBUG_ROUTINES_H__

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/hmatrix.h>

void expandHmatrix(HMatrix &hmatrix, H2Opus_Real *matrix, int dense = 1, int level = -1);
void dumpHMatrix(HMatrix &hmatrix, int digits);
void dumpBasisTree(BasisTree &basis_tree, int digits, const char *label);
void dumpTransferLevel(BasisTree &basis_tree, int digits, int level);
void dumpCouplingMatrices(HMatrix &hmatrix, int digits);
void dumpTikzMatrixIndices(HMatrix &hmatrix);
void expandBasis(BasisTree &basis_tree, int node_index, H2Opus_Real *matrices, int depth, int n, int max_rank);
void outputEps(HMatrix &h, const char *filename, int level = -1);
void printMatrixTreeStructure(HMatrix &hmatrix);
H2Opus_Real truncationError(HMatrix &h_orthog, HMatrix &h_trunc);
H2Opus_Real frobeniusHNorm(HMatrix &h_orthog);
H2Opus_Real frobeniusHNorm(HMatrix &h_orthog, int dense, int low_rank);
H2Opus_Real getBasisOrthogonality(BasisTree &basis_tree, bool print_per_level = false);

void dumpMatrixTreeContainer(BasisTreeLevelData &level_data, std::vector<H2Opus_Real *> &Zhat, int digits,
                             int hw = H2OPUS_HWTYPE_CPU);
void dumpHgemvTreeContainer(BasisTreeLevelData &level_data, std::vector<H2Opus_Real *> &Z_hat, int num_vectors,
                            int digits, int hw = H2OPUS_HWTYPE_CPU);
void printDenseMatrix(float *matrix, int ldm, int m, int n, int digits, const char *name, int hw = H2OPUS_HWTYPE_CPU);
void printDenseMatrix(double *matrix, int ldm, int m, int n, int digits, const char *name, int hw = H2OPUS_HWTYPE_CPU);

#ifdef H2OPUS_USE_GPU
thrust::host_vector<float> copyGPUBlock(float **data_ptrs, int index, int ld, int rows, int cols);
thrust::host_vector<double> copyGPUBlock(double **data_ptrs, int index, int ld, int rows, int cols);
thrust::host_vector<float> copyGPUBlock(float *data, int ld, int rows, int cols);
thrust::host_vector<double> copyGPUBlock(double *data, int ld, int rows, int cols);

thrust::host_vector<double *> copyGPUArray(double **data, int entries);
thrust::host_vector<float *> copyGPUArray(float **data, int entries);
thrust::host_vector<double> copyGPUArray(double *data, int entries);
thrust::host_vector<float> copyGPUArray(float *data, int entries);
thrust::host_vector<int> copyGPUArray(int *data, int entries);
#endif

H2Opus_Real orthogonality(H2Opus_Real *matrix, int m, int n);
H2Opus_Real orthogonality(H2Opus_Real *matrix, int ld, int m, int n);
H2Opus_Real frobeniusError(H2Opus_Real *m1, H2Opus_Real *m2, int m, int n);
void eye(std::vector<H2Opus_Real> &identity, int m);
void randomData(H2Opus_Real *m, size_t entries);
void randomData(H2Opus_Real *m, size_t entries, int seed);
H2Opus_Real frobeniusNorm(H2Opus_Real *m, int rows, int cols);

// Templated routines
template <class T> inline void avg_and_stdev(T *values, int num_vals, T &avg, T &std_dev)
{
    if (num_vals == 0)
        return;
    if (num_vals == 1)
    {
        avg = values[0];
        std_dev = 0;
    }

    avg = 0;
    // Skip the first run
    for (int i = 1; i < num_vals; i++)
        avg += values[i];
    avg /= (num_vals - 1);
    std_dev = 0;
    for (int i = 1; i < num_vals; i++)
        std_dev += (values[i] - avg) * (values[i] - avg);
    std_dev = sqrt(std_dev / (num_vals - 1));
}

template <typename BBox> inline void printBBox(BBox &box, int Dim, const char *label)
{
    printf("%s = \n", label);
    for (int i = 0; i < Dim; i++)
        printf("[%f, %f] ", box[i].low, box[i].high);
    printf("\n");
}

/*
template<typename KDTreeNodePtr>
void printBBoxLevel(KDTreeNodePtr& node, int print_level, int level, int max_depth, int& current_label)
{
    if(level >= max_depth) return;

    if(print_level == level)
    {
        printf("Box(%d, %d, :) = [", print_level + 1, current_label);
        current_label++;
        for(int i = 0; i < PointCloud<H2Opus_Real>::Dim; i++)
            printf("%f %f ", node->bbox[i].high, node->bbox[i].low);
        printf("];\n");
    }
    else if(node->child1 != NULL && node->child2 != NULL)
    {
        printBBoxLevel(node->child1, print_level, level + 1, max_depth, current_label);
        printBBoxLevel(node->child2, print_level, level + 1, max_depth, current_label);
    }
}

template<typename PointCloud>
void printBBoxHierarchy(PointCloud& pt_cloud)
{
    typedef typename BoxTypeHelper<PointCloud>::my_kd_tree_t KDTree;
    typedef typename KDTree::NodePtr KDTreeNodePtr;
    KDTree tree(PointCloud::Dim, pt_cloud, KDTreeSingleIndexAdaptorParams(min_n));
    tree.buildIndex();

    printf("P=[\n");
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < PointCloud<H2Opus_Real>::Dim; j++)
            printf("%f ", pt_cloud.get_pt(i, j));
        printf(";\n");
    }
    printf("];\n");

    KDTreeNodePtr root_node = tree.getRootNode();
    int depth = tree.getTreeDepth();

    for(int i = 0; i < depth; i++)
    {
        int node_label = 1;
        printBBoxLevel<KDTreeNodePtr>(root_node, i, 0, depth, node_label);
    }
}
*/

#endif
