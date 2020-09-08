#ifndef __HNODE_TREE_H__
#define __HNODE_TREE_H__

#include <h2opus/core/basis_tree.h>
#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/hnode_tree_data.h>

#include <h2opus/util/geometric_admissibility.h>
#include <h2opus/util/kdtree.h>
#include <h2opus/util/thrust_wrappers.h>

#define HMATRIX_RANK_MATRIX 0
#define HMATRIX_DENSE_MATRIX 1
#define HMATRIX_INNER_NODE 2

template <int hw> struct THNodeTree
{
  public:
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    typedef typename VectorContainer<hw, int>::type IntVector;
    typedef typename TTreeContainer<RealVector>::type TreeContainer;
    typedef THNodeTreeBSNData<IntVector> HNodeTreeBSNData;

  private:
    void determineAdmissibleBlocks(TH2OpusKDTree<H2Opus_Real, hw> &kdtree,
                                   TH2OpusAdmissibility<H2Opus_Real, hw> &admissibility, TBasisTree<hw> &u_basis_tree,
                                   TBasisTree<hw> &v_basis_tree, std::vector<int> &level_counts,
                                   std::vector<int> &rank_level_counts, std::vector<int> &inode_level_counts,
                                   int &dense_leaf_count, int u_node, int v_node, int level, int max_depth);

    void buildAdmissibleBlocks(TH2OpusKDTree<H2Opus_Real, hw> &kdtree,
                               TH2OpusAdmissibility<H2Opus_Real, hw> &admissibility, TBasisTree<hw> &u_basis_tree,
                               TBasisTree<hw> &v_basis_tree, std::vector<int> &level_counts,
                               std::vector<int> &rank_level_counts, std::vector<int> &inode_level_counts,
                               int &dense_leaf_count, int u_node, int v_node, int parent_node, int level,
                               int max_depth);

    void reorderLeafIndexes(int *dense_map, int *rank_map);

    void generateBSNData(int *node_tree_index, int node_offset, int num_nodes, int *basis_indexes,
                         int basis_index_offset, int basis_level_nodes, std::vector<int> &workspace,
                         IntVector &bsn_ptrs, IntVector &bsn_node_indexes, std::vector<int> &bsn_batch_ptr,
                         IntVector &batch_indexes);

  public:
    // All the relevant node data
    IntVector node_u_index, node_v_index;
    IntVector node_morton_level_index;
    IntVector head, next, parent;
    IntVector dense_leaf_tree_index, rank_leaf_tree_index, inner_node_tree_index;
    IntVector node_type, node_to_leaf;

    // Level data
    HNodeTreeLevelData level_data;

    // Matrix data
    RealVector dense_leaf_mem;
    TreeContainer rank_leaf_mem;

    // Node and depth counts
    int num_nodes, depth, leaf_size;
    int num_dense_leaves, num_rank_leaves, num_inodes;

    // BSR data for matvecs
    HNodeTreeBSRData<IntVector> bsr_data;

    // BSN data
    THNodeTreeBSNData<IntVector> bsn_row_data;
    THNodeTreeBSNData<IntVector> bsn_col_data;

    THNodeTree()
    {
        num_nodes = depth = leaf_size = 0;
        num_dense_leaves = num_rank_leaves = num_inodes = 0;
    }

    // Clear only the matrix data, leaving structure data intact
    void clearData()
    {
        // The dense leaf data is never reallocated since their size won't change
        // Clearing them just means setting them to zero
        fillArray(vec_ptr(dense_leaf_mem), dense_leaf_mem.size(), 0, 0, hw);

        for (size_t i = 0; i < rank_leaf_mem.size(); i++)
            rank_leaf_mem[i].clear();

        level_data.clearData();
    }

    template <int other_hw> THNodeTree &operator=(const THNodeTree<other_hw> &h)
    {
        // Scalars
        this->num_nodes = h.num_nodes;
        this->depth = h.depth;
        this->leaf_size = h.leaf_size;
        this->num_dense_leaves = h.num_dense_leaves;
        this->num_rank_leaves = h.num_rank_leaves;
        this->num_inodes = h.num_inodes;

        // Structures
        this->level_data = h.level_data;
        this->bsr_data = h.bsr_data;
        this->bsn_row_data = h.bsn_row_data;
        this->bsn_col_data = h.bsn_col_data;

        // Vectors
        copyThrustArray(this->node_u_index, h.node_u_index);
        copyThrustArray(this->node_v_index, h.node_v_index);
        copyThrustArray(this->node_morton_level_index, h.node_morton_level_index);
        copyThrustArray(this->head, h.head);
        copyThrustArray(this->next, h.next);
        copyThrustArray(this->parent, h.parent);
        copyThrustArray(this->dense_leaf_tree_index, h.dense_leaf_tree_index);
        copyThrustArray(this->rank_leaf_tree_index, h.rank_leaf_tree_index);
        copyThrustArray(this->inner_node_tree_index, h.inner_node_tree_index);
        copyThrustArray(this->node_type, h.node_type);
        copyThrustArray(this->node_to_leaf, h.node_to_leaf);

        copyThrustArray(this->dense_leaf_mem, h.dense_leaf_mem);

        // Have to deep copy this one
        resizeThrustArray(rank_leaf_mem, h.rank_leaf_mem.size());
        for (size_t i = 0; i < h.rank_leaf_mem.size(); i++)
            copyThrustArray(this->rank_leaf_mem[i], h.rank_leaf_mem[i]);

        return *this;
    }

    H2Opus_Real getMemoryUsage()
    {
        return getLowRankMemoryUsage() + getDenseMemoryUsage();
    }

    H2Opus_Real getLowRankMemoryUsage()
    {
        size_t int_entries = node_u_index.size() + node_v_index.size();
        int_entries += head.size() + next.size() + parent.size();
        int_entries += dense_leaf_tree_index.size() + rank_leaf_tree_index.size() + inner_node_tree_index.size();
        int_entries += node_type.size() + node_to_leaf.size();

        size_t real_entries = 0;
        for (int level = 0; level < depth; level++)
            real_entries += rank_leaf_mem[level].size();

        H2Opus_Real bsr_data_mem = bsr_data.getMemoryUsage();

        return ((H2Opus_Real)int_entries * sizeof(int) + (H2Opus_Real)real_entries * sizeof(H2Opus_Real)) * 1e-9 +
               bsr_data_mem;
    }

    H2Opus_Real getDenseMemoryUsage()
    {
        size_t real_entries = dense_leaf_mem.size();
        return (H2Opus_Real)real_entries * sizeof(H2Opus_Real) * 1e-9;
    }

    int getHNodeIndex(int level, int i, int j);

    void allocateNodes(int num_nodes);

    void allocateLevels(int levels);

    void allocateLeaves(int dense, int rank);

    void allocateInnerNodes(int inodes);

    void setLevelPointers(int *level_counts, int *rank_level_counts, int *inode_level_counts);

    void allocateMatrixData(BasisTreeLevelData &basis_level_data, int start_level, int depth);
    void allocateDenseLeafMemory();
    void allocateMatrixData(BasisTreeLevelData &basis_level_data)
    {
        allocateMatrixData(basis_level_data, 0, basis_level_data.depth);
    }

    void allocateBSRData(TBasisTree<hw> &u_basis_tree, TBasisTree<hw> &v_basis_tree, int u_start_level,
                         int v_start_level);
    void allocateBSRData(TBasisTree<hw> &u_basis_tree, TBasisTree<hw> &v_basis_tree, int u_start_level,
                         int v_start_level, int *coupling_index_map, int *dense_index_map);
    void allocateBSRData(TBasisTree<hw> &basis_tree)
    {
        allocateBSRData(basis_tree, basis_tree, 0, 0);
    }

    void allocateBSNData(TBasisTree<hw> &u_basis_tree, TBasisTree<hw> &v_basis_tree, int u_start_level,
                         int v_start_level);
    void allocateBSNData(TBasisTree<hw> &basis_tree)
    {
        allocateBSNData(basis_tree, basis_tree, 0, 0);
    }
    void allocateRowBSNData(TBasisTree<hw> &u_basis_tree, int u_start_level);
    void allocateColumnBSNData(TBasisTree<hw> &v_basis_tree, int v_start_level);

    H2Opus_Real *getCouplingMatrixLevelData(int level);
    H2Opus_Real *getCouplingMatrix(int level, size_t level_index);
    H2Opus_Real *getDenseMatrix(size_t index);

    void getDenseBSRData(int **ia, int **ja, H2Opus_Real **a);
    void getLevelBSRData(int level, int **ia, int **ja, H2Opus_Real **a);
    int getLevelBSRRows(int level);

    int getCouplingLevelStart(int level)
    {
        return level_data.getCouplingLevelStart(level);
    }
    int getCouplingLevelSize(int level)
    {
        return level_data.getCouplingLevelSize(level);
    }

    int getInodeLevelStart(int level)
    {
        return level_data.getInodeLevelStart(level);
    }
    int getInodeLevelSize(int level)
    {
        return level_data.getInodeLevelSize(level);
    }

    int getLevelStart(int level)
    {
        return level_data.getLevelStart(level);
    }
    int getLevelRank(int level)
    {
        return level_data.getLevelRank(level);
    }
    int getLevelSize(int level)
    {
        return level_data.getLevelSize(level);
    }
    int getLargestRank()
    {
        return level_data.getLargestRank();
    }

    void getCouplingLevelRange(int level, int &start, int &end)
    {
        level_data.getCouplingLevelRange(level, start, end);
    }
    void getInodeLevelRange(int level, int &start, int &end)
    {
        level_data.getInodeLevelRange(level, start, end);
    }
    void getLevelRange(int level, int &start, int &end)
    {
        level_data.getLevelRange(level, start, end);
    }

    void determineStructure(TH2OpusKDTree<H2Opus_Real, hw> &kdtree,
                            TH2OpusAdmissibility<H2Opus_Real, hw> &admissibility, TBasisTree<hw> &u_basis_tree,
                            int u_start_level, TBasisTree<hw> &v_basis_tree, int v_start_level, int max_depth,
                            std::vector<int> v_list);

    // Variant that is used for symmetric matrices
    void determineStructure(TH2OpusKDTree<H2Opus_Real, hw> &kdtree,
                            TH2OpusAdmissibility<H2Opus_Real, hw> &admissibility, TBasisTree<hw> &basis_tree);

    // Get the u and v indexes of the innernodes of a level - useful for determining which
    // nodes have to be (or were) subdivided at that level
    void extractLevelUVInodeIndexes(std::vector<int> &u_indexes, std::vector<int> &v_indexes, int level);
};

#include <h2opus/core/hnode_tree.cuh>

typedef THNodeTree<H2OPUS_HWTYPE_CPU> HNodeTree;

#ifdef H2OPUS_USE_GPU
typedef THNodeTree<H2OPUS_HWTYPE_GPU> HNodeTree_GPU;
#endif

#endif
