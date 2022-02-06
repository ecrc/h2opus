#ifndef __BASIS_TREE_H__
#define __BASIS_TREE_H__

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/basis_tree_level_data.h>
#include <h2opus/util/kdtree.h>
#include <h2opus/util/thrust_wrappers.h>
#include <h2opus/util/vector_operations.h>

template <int hw> struct TBasisTree
{
  private:
    // Construction from a KDTree
    void getKDTreeLevelCounts(std::vector<int> &counts, TH2OpusKDTree<H2Opus_Real, hw> &kdtree, int root_node,
                              int level, int max_depth);
    void getKDtreeStructure(TH2OpusKDTree<H2Opus_Real, hw> &kdtree, int kd_node_index, int parent_node, int level,
                            std::vector<int> &level_counts, std::vector<int> &node_tails, int max_depth);

  public:
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    typedef typename VectorContainer<hw, int>::type IntVector;
    typedef typename TTreeContainer<RealVector>::type TreeContainer;

    // Data for the start and length of each node in the full matrix
    IntVector node_start, node_len;

    // Index map for cluster index -> original index
    IntVector index_map;

    // Store the global cluster indices of each node in the tree
    IntVector global_cluster_index;

    // These are indices into the nodes array storing the structure of the tree
    IntVector head, next, parent;

    // Level data for the tree containing offsets and node dimensions by level
    BasisTreeLevelData level_data;

    // Pointers for the basis and transformation matrices
    RealVector basis_mem;
    TreeContainer trans_mem;

    // Some handy dandy metadata
    int leaf_size, depth, basis_leaves, num_nodes, max_children;
    int index_map_offset;

    TBasisTree()
    {
        // Scalars
        depth = num_nodes = basis_leaves = leaf_size = index_map_offset = 0;

        // Binary tree for now
        max_children = 2;
    }

    // Clear only the matrix data, leaving structure data intact
    void clearData()
    {
        basis_mem.clear();
        for (size_t i = 0; i < trans_mem.size(); i++)
            trans_mem[i].clear();

        level_data.clearData();
    }

    template <int other_hw> void copyBasisData(const TBasisTree<other_hw> &b)
    {
        // Make sure the trees are compatible
        assert(this->depth = b.depth);
        assert(this->leaf_size == b.leaf_size);
        assert(this->basis_leaves == b.basis_leaves);
        assert(this->num_nodes == b.num_nodes);

        // Copy matrix data
        copyVector(this->basis_mem, b.basis_mem);

        this->trans_mem.clear();
        trans_mem.resize(b.trans_mem.size());

        for (size_t i = 0; i < b.trans_mem.size(); i++)
            copyVector(this->trans_mem[i], b.trans_mem[i]);

        // Update level data
        this->level_data.copyLevelData(b.level_data);
    }

    template <int other_hw> void copyStructureData(const TBasisTree<other_hw> &b)
    {
        // Scalars
        this->leaf_size = b.leaf_size;
        this->depth = b.depth;
        this->basis_leaves = b.basis_leaves;
        this->num_nodes = b.num_nodes;
        this->max_children = b.max_children;
        this->index_map_offset = b.index_map_offset;

        // Structures
        this->level_data = b.level_data;

        // Vectors
        copyVector(this->node_start, b.node_start);
        copyVector(this->node_len, b.node_len);
        copyVector(this->index_map, b.index_map);
        copyVector(this->global_cluster_index, b.global_cluster_index);

        copyVector(this->head, b.head);
        copyVector(this->next, b.next);
        copyVector(this->parent, b.parent);

        // Allocate the levels
        trans_mem.resize(depth);
    }

    template <int other_hw> TBasisTree &operator=(const TBasisTree<other_hw> &b)
    {
        this->copyStructureData<other_hw>(b);

        copyVector(this->basis_mem, b.basis_mem);
        // Deep copy for this array
        for (int level = 0; level < depth; level++)
            copyVector(this->trans_mem[level], b.trans_mem[level]);

        return *this;
    }

    // Return the memory used by the basis tree in GB
    H2Opus_Real getMemoryUsage()
    {
        size_t int_entries = node_start.size() + node_len.size() + index_map.size();
        int_entries += global_cluster_index.size() + head.size() + next.size() + parent.size();

        size_t real_entries = basis_mem.size();
        for (int level = 0; level < depth; level++)
            real_entries += trans_mem[level].size();

        return ((H2Opus_Real)int_entries * sizeof(int) + (H2Opus_Real)real_entries * sizeof(H2Opus_Real)) * 1e-9;
    }

    // Generate the structure of the cluster tree from a kd-tree
    void generateStructureFromKDTree(TH2OpusKDTree<H2Opus_Real, hw> &kdtree, int root_node, bool copyMap,
                                     int max_depth);

    // Allocation routines
    void allocateNodes(int num_nodes);
    void allocateLevels(int *level_counts, int levels);
    void allocateMatrixData(int *level_ranks, int levels, int leaf_size);

    // Shortcuts to the level data
    void getTransferDims(int level, int &rows, int &cols)
    {
        level_data.getTransferDims(level, rows, cols);
    }
    void getLevelRange(int level, int &start, int &end)
    {
        level_data.getLevelRange(level, start, end);
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

    H2Opus_Real *getXhatNode(TreeContainer &xhat, int level, int level_index)
    {
        return level_data.getXhatNode(xhat, level, level_index);
    }
    H2Opus_Real *getYhatNode(TreeContainer &yhat, int level, int level_index)
    {
        return level_data.getYhatNode(yhat, level, level_index);
    }
    void populateXhat(TreeContainer &xhat)
    {
        level_data.populateXhat(xhat);
    }
    void populateYhat(TreeContainer &yhat)
    {
        level_data.populateYhat(yhat);
    }

    H2Opus_Real *getProjectionLevelData(TreeContainer &proj, int level)
    {
        return level_data.getProjectionLevelData(proj, level);
    }
    void populateProjectionTree(TreeContainer &proj, int start_level = 0)
    {
        level_data.populateProjectionTree(proj, start_level);
    }
    void setVectorTree(TreeContainer &tree, H2Opus_Real val)
    {
        level_data.setVectorTree(tree, val);
    }

    // Getting pointer to leaves and transfer nodes
    H2Opus_Real *getBasisLeafData()
    {
        return vec_ptr(basis_mem);
    }

    H2Opus_Real *getTransLevelData(int level)
    {
        return vec_ptr(trans_mem[level]);
    }

    size_t getBasisTotalEntries()
    {
        return basis_mem.size();
    }

    H2Opus_Real *getBasisLeaf(size_t index)
    {
        H2Opus_Real *leaf_base = vec_ptr(basis_mem);
        size_t entries = leaf_size * getLevelRank(depth - 1) * index;
        assert(entries < basis_mem.size() || (!leaf_base && !entries));

        return leaf_base + entries;
    }

    H2Opus_Real *getTransNode(int level, size_t level_index)
    {
        int rows, cols;
        getTransferDims(level, rows, cols);
        size_t entries = rows * cols * level_index;
        H2Opus_Real *base = vec_ptr(trans_mem[level]);
        assert(entries < trans_mem[level].size());

        return base + entries;
    }

    int *head_ptr()
    {
        return vec_ptr(head);
    }

    int *next_ptr()
    {
        return vec_ptr(next);
    }

    int *parent_ptr()
    {
        return vec_ptr(parent);
    }

    int numChildren(int node)
    {
        int index = head[node];
        int num_children = 0;
        while (index != H2OPUS_EMPTY_NODE)
        {
            num_children++;
            index = next[index];
        }
        return num_children;
    }
};

#include <h2opus/core/basis_tree.cuh>

typedef TBasisTree<H2OPUS_HWTYPE_CPU> BasisTree;

#ifdef H2OPUS_USE_GPU
typedef TBasisTree<H2OPUS_HWTYPE_GPU> BasisTree_GPU;
#endif

#endif
