#ifndef __H2OPUS_GEOMETRIC_CONSTRUCTION_H__
#define __H2OPUS_GEOMETRIC_CONSTRUCTION_H__

#include <h2opus/core/hmatrix.h>
#include <h2opus/util/geometric_admissibility.h>
#include <h2opus/util/kdtree.h>

template <class T, int hw, typename EntryGen>
void generateUBasisTreeEntries(TBasisTree<hw> &basis_tree, TH2OpusKDTree<T, hw> &kdtree, EntryGen &entry_gen,
                               std::vector<int> &level_slices)
{
    int num_levels = basis_tree.depth;
    // Compute the basis leaves and transfer matrices for the basis tree
    int leaf_offset = basis_tree.getLevelStart(num_levels - 1);
    int leaf_slices = level_slices[num_levels - 1];
    int leaf_ld = basis_tree.leaf_size;

#pragma omp parallel for
    for (size_t i = 0; i < (size_t)basis_tree.basis_leaves; i++)
    {
        int node_index = leaf_offset + i;
        int cluster_index = basis_tree.global_cluster_index[node_index];

        H2Opus_Real *leaf_entries = basis_tree.getBasisLeaf(i);
        entry_gen.u_basis_leaf(leaf_entries, leaf_ld, kdtree, cluster_index, leaf_slices);
    }

    // Transfer matrices
    for (int level = num_levels - 1; level >= 0; level--)
    {
        int node_start, node_end;
        basis_tree.getLevelRange(level, node_start, node_end);

        int slices = level_slices[level];
        int parent_slices = (level == 0 ? slices : level_slices[level - 1]);
        int level_rank = basis_tree.level_data.getLevelRank(level);

#pragma omp parallel for
        for (int node_index = node_start; node_index < node_end; node_index++)
        {
            int parent_index = basis_tree.parent[node_index];
            int cluster_index = basis_tree.global_cluster_index[node_index];
            int parent_cluster_index = kdtree.getParent(cluster_index);

            if (parent_index == H2OPUS_EMPTY_NODE && parent_cluster_index == H2OPUS_EMPTY_NODE)
                continue;

            H2Opus_Real *transfer = basis_tree.getTransNode(level, node_index - node_start);
            entry_gen.u_transfer_matrix(transfer, level_rank, kdtree, cluster_index, parent_cluster_index, slices,
                                        parent_slices);
        }
    }
}

template <class T, int hw, typename EntryGen>
void generateVBasisTreeEntries(TBasisTree<hw> &basis_tree, TH2OpusKDTree<T, hw> &kdtree, EntryGen &entry_gen,
                               std::vector<int> &level_slices)
{
    int num_levels = basis_tree.depth;
    // Compute the basis leaves and transfer matrices for the basis tree
    int leaf_offset = basis_tree.getLevelStart(num_levels - 1);
    int leaf_slices = level_slices[num_levels - 1];
    int leaf_ld = basis_tree.leaf_size;

#pragma omp parallel for
    for (size_t i = 0; i < (size_t)basis_tree.basis_leaves; i++)
    {
        int node_index = leaf_offset + i;
        int cluster_index = basis_tree.global_cluster_index[node_index];

        H2Opus_Real *leaf_entries = basis_tree.getBasisLeaf(i);
        entry_gen.v_basis_leaf(leaf_entries, leaf_ld, kdtree, cluster_index, leaf_slices);
    }

    // Transfer matrices
    for (int level = num_levels - 1; level >= 0; level--)
    {
        int node_start, node_end;
        basis_tree.getLevelRange(level, node_start, node_end);

        int slices = level_slices[level];
        int parent_slices = (level == 0 ? slices : level_slices[level - 1]);
        int level_rank = basis_tree.level_data.getLevelRank(level);

#pragma omp parallel for
        for (int node_index = node_start; node_index < node_end; node_index++)
        {
            int parent_index = basis_tree.parent[node_index];
            int cluster_index = basis_tree.global_cluster_index[node_index];
            int parent_cluster_index = kdtree.getParent(cluster_index);

            if (parent_index == H2OPUS_EMPTY_NODE && parent_cluster_index == H2OPUS_EMPTY_NODE)
                continue;

            H2Opus_Real *transfer = basis_tree.getTransNode(level, node_index - node_start);
            entry_gen.v_transfer_matrix(transfer, level_rank, kdtree, cluster_index, parent_cluster_index, slices,
                                        parent_slices);
        }
    }
}

template <class T, int hw, typename EntryGen>
void generateHNodeEntries(THNodeTree<hw> &hnodes, TH2OpusKDTree<T, hw> &u_kdtree, TBasisTree<hw> &u_basis_tree,
                          TH2OpusKDTree<T, hw> &v_kdtree, TBasisTree<hw> &v_basis_tree, EntryGen &entry_gen,
                          std::vector<int> &level_slices)
{
    int num_levels = hnodes.depth;
    // Coupling matrices
    for (int level = num_levels - 1; level >= 0; level--)
    {
        int node_start, node_end;
        hnodes.getCouplingLevelRange(level, node_start, node_end);
        int slices = level_slices[level];
        int level_rank = hnodes.level_data.getLevelRank(level);

#pragma omp parallel for
        for (int node_index = node_start; node_index < node_end; node_index++)
        {
            H2Opus_Real *coupling_matrix = hnodes.getCouplingMatrix(level, node_index - node_start);
            int tree_index = hnodes.rank_leaf_tree_index[node_index];

            int u_index = hnodes.node_u_index[tree_index];
            int v_index = hnodes.node_v_index[tree_index];
            int u_cluster_index = u_basis_tree.global_cluster_index[u_index];
            int v_cluster_index = v_basis_tree.global_cluster_index[v_index];

            entry_gen.coupling_matrix(coupling_matrix, level_rank, u_kdtree, u_cluster_index, v_kdtree, v_cluster_index,
                                      slices);
        }
    }

    // Dense matrices
    int num_dense_leaves = hnodes.num_dense_leaves;
    if (num_dense_leaves == 0)
        return;

    int ld = u_basis_tree.leaf_size;
    assert(v_basis_tree.leaf_size == ld);
    int *u_index_map = &(u_basis_tree.index_map[0]);
    int *v_index_map = &(v_basis_tree.index_map[0]);

#pragma omp parallel for
    for (int leaf = 0; leaf < num_dense_leaves; leaf++)
    {
        H2Opus_Real *dense_leaf = hnodes.getDenseMatrix(leaf);
        int tree_index = hnodes.dense_leaf_tree_index[leaf];

        int u_index = hnodes.node_u_index[tree_index];
        int v_index = hnodes.node_v_index[tree_index];
        int u_cluster_index = u_basis_tree.global_cluster_index[u_index];
        int v_cluster_index = v_basis_tree.global_cluster_index[v_index];

        entry_gen.dense_matrix(dense_leaf, ld, u_kdtree, u_cluster_index, v_kdtree, v_cluster_index);
    }
}

template <class T, int hw, typename EntryGen>
void generateHMatrixEntries(THMatrix<hw> &hmatrix, TH2OpusKDTree<T, hw> &kdtree, EntryGen &entry_gen,
                            std::vector<int> &level_slices)
{
    std::vector<int> level_ranks(level_slices.size(), pow(level_slices[0], kdtree.getDim()));

    int leaf_size = kdtree.getLeafSize();
    hmatrix.u_basis_tree.allocateMatrixData(&level_ranks[0], level_ranks.size(), leaf_size);
    generateUBasisTreeEntries(hmatrix.u_basis_tree, kdtree, entry_gen, level_slices);

    if (!hmatrix.sym)
    {
        hmatrix.v_basis_tree.allocateMatrixData(&level_ranks[0], level_ranks.size(), leaf_size);
        generateVBasisTreeEntries(hmatrix.v_basis_tree, kdtree, entry_gen, level_slices);
    }

    hmatrix.hnodes.allocateMatrixData(hmatrix.u_basis_tree.level_data);

    generateHNodeEntries(hmatrix.hnodes, kdtree, hmatrix.u_basis_tree, kdtree, hmatrix.u_basis_tree, entry_gen,
                         level_slices);
}

template <class T, int hw>
void buildHMatrixStructure(THMatrix<hw> &hmatrix, TH2OpusKDTree<T, hw> &kdtree,
                           TH2OpusAdmissibility<T, hw> &admissibility)
{
    hmatrix.n = kdtree.getDataSet()->getDataSetSize();
    hmatrix.u_basis_tree.generateStructureFromKDTree(kdtree, 0, true, kdtree.getDepth());
    hmatrix.hnodes.determineStructure(kdtree, admissibility, hmatrix.u_basis_tree);

    hmatrix.hnodes.allocateBSRData(hmatrix.u_basis_tree);
    hmatrix.hnodes.allocateBSNData(hmatrix.u_basis_tree);

    if (!hmatrix.sym)
        hmatrix.v_basis_tree.copyStructureData(hmatrix.u_basis_tree);
}

template <class T, int hw>
void buildHMatrixStructure(THMatrix<hw> &hmatrix, H2OpusDataSet<T> *data_set, int leaf_size,
                           TH2OpusAdmissibility<T, hw> &admissibility)
{
    TH2OpusKDTree<T, hw> kdtree(data_set, leaf_size);
    kdtree.buildKDtreeMedianSplit();
    buildHMatrixStructure<T, hw>(hmatrix, kdtree, admissibility);
}

template <class T, int hw, typename EntryGen>
void buildHMatrix(THMatrix<hw> &hmatrix, H2OpusDataSet<T> *data_set, TH2OpusAdmissibility<T, hw> &admissibility,
                  EntryGen &entry_gen, int leaf_size, int slices)
{
    TH2OpusKDTree<T, hw> kdtree(data_set, leaf_size);
    kdtree.buildKDtreeMedianSplit();

    std::vector<int> level_slices(kdtree.getDepth(), slices);
    buildHMatrixStructure(hmatrix, kdtree, admissibility);
    generateHMatrixEntries(hmatrix, kdtree, entry_gen, level_slices);
}

#endif
