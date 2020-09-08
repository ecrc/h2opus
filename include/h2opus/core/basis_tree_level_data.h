#ifndef __BASIS_TREE_LEVEL_DATA_H__
#define __BASIS_TREE_LEVEL_DATA_H__

#include <h2opus/core/h2opus_defs.h>
#include <algorithm>
#include <vector>
#include <stdio.h>

// Level data for the tree containing offsets and node dimensions by level
struct BasisTreeLevelData
{
    // Pointers that show the start and end of each level
    std::vector<int> level_ptrs;

    // The transformation matrices dimensions each level would be
    // trans_dim[level+1]xtrans_dim[level] The rank of the level can be deduced
    // from the rows of the transfer matrix (i.e. trans_dim[level+1])
    std::vector<int> trans_dim;

    int depth, leaf_size, basis_leaves, max_children;
    int nested_root_level;

    BasisTreeLevelData()
    {
        depth = leaf_size = basis_leaves = nested_root_level = 0;
        max_children = 2;
    }

    BasisTreeLevelData(const BasisTreeLevelData &b)
    {
        init(b);
    }

    BasisTreeLevelData &operator=(const BasisTreeLevelData &b)
    {
        init(b);
        return *this;
    }

    void clearData()
    {
        thrust::fill(trans_dim.begin(), trans_dim.end(), 0);
    }

    void copyLevelData(const BasisTreeLevelData &level_data)
    {
        // Make sure the level data is compatible
        assert(this->depth == level_data.depth);
        assert(this->leaf_size == level_data.leaf_size);
        assert(this->basis_leaves == level_data.basis_leaves);

        this->trans_dim = level_data.trans_dim;
    }

    int getLevelRank(int level)
    {
        return trans_dim[level + 1];
    }

    void getTransferDims(int level, int &rows, int &cols)
    {
        rows = trans_dim[level + 1];
        cols = trans_dim[level];
    }

    void getLevelRange(int level, int &start, int &end)
    {
        start = getLevelStart(level);
        end = getLevelEnd(level);
    }

    int getLevelStart(int level)
    {
        return level_ptrs[level];
    }

    int getLevelEnd(int level)
    {
        return level_ptrs[level + 1];
    }

    int getLevelSize(int level)
    {
        return getLevelEnd(level) - getLevelStart(level);
    }

    int getLargestRank()
    {
        int max_rank = 0;
        for (int level = 0; level < depth; level++)
            if (getLevelRank(level) > max_rank)
                max_rank = getLevelRank(level);
        return max_rank;
    }

    int getLargestLevel()
    {
        int max_level = 0, max_nodes = 0;
        for (int level = 0; level < depth; level++)
        {
            int num_level_nodes = getLevelSize(level);
            if (num_level_nodes > max_nodes)
            {
                max_nodes = num_level_nodes;
                max_level = level;
            }
        }

        return max_level;
    }

    size_t getLargestLevelSize()
    {
        int max_level = getLargestLevel();
        int level_rows, level_cols;
        getTransferDims(max_level, level_rows, level_cols);

        size_t level_size = getLevelSize(max_level) * level_rows * level_cols;
        return level_size;
    }

    size_t getLargestParentLevel()
    {
        size_t max_level = 0, max_nodes = 0;
        for (int level = 0; level < depth - 1; level++)
        {
            size_t num_level_nodes = getLevelSize(level);
            if (num_level_nodes > max_nodes)
            {
                max_nodes = num_level_nodes;
                max_level = level;
            }
        }

        return max_level;
    }

    // Determines the amount of entries needed if we were to stack
    // max_children amount of transfer nodes (or nodes of the same size)
    // on top of each other
    size_t getLargestChildStackSize(int max_children)
    {
        size_t max_stacked_size = 0;
        for (int level = 0; level < depth - 1; level++)
        {
            int num_level_nodes = getLevelSize(level);
            int child_rows, child_cols;
            getTransferDims(level + 1, child_rows, child_cols);
            size_t stacked_size = num_level_nodes * (max_children * child_rows * child_cols);
            max_stacked_size = std::max(max_stacked_size, stacked_size);
        }

        return max_stacked_size;
    }

    size_t getLargestLevelSizeByRank()
    {
        size_t max_size = 0;
        for (int level = 0; level < depth; level++)
        {
            size_t level_size_by_rank = getLevelSize(level) * getLevelRank(level);
            max_size = std::max(max_size, level_size_by_rank);
        }
        return max_size;
    }

    void allocateLevels(int *level_counts, int levels)
    {
        level_ptrs.resize(levels + 1, 0);
        trans_dim.resize(levels + 1, 0);
        depth = levels;

        thrust::inclusive_scan(level_counts, level_counts + levels, &level_ptrs[1]);
    }

    void setLevelRanks(int *level_ranks)
    {
        // copy the level ranks
        thrust::copy(level_ranks, level_ranks + depth, &trans_dim[1]);
        trans_dim[0] = level_ranks[0];
    }

    template <class TreeContainer> H2Opus_Real *getXhatNode(TreeContainer &xhat, int level, int level_index)
    {
        int entries = getLevelRank(level) * level_index;
        H2Opus_Real *base = vec_ptr(xhat[level]);
        return base + entries;
    }

    template <class TreeContainer> H2Opus_Real *getYhatNode(TreeContainer &yhat, int level, int level_index)
    {
        int entries = getLevelRank(level) * level_index;
        H2Opus_Real *base = vec_ptr(yhat[level]);
        return base + entries;
    }

    template <class TreeContainer> void populateXhat(TreeContainer &xhat)
    {
        xhat.resize(depth);
        for (int level = 0; level < depth; level++)
            xhat[level].resize(getLevelSize(level) * getLevelRank(level));
    }

    template <class TreeContainer> void populateYhat(TreeContainer &yhat)
    {
        yhat.resize(depth);
        for (int level = 0; level < depth; level++)
            yhat[level].resize(getLevelSize(level) * getLevelRank(level));
    }

    template <class TreeContainer> H2Opus_Real *getProjectionLevelData(TreeContainer &proj, int level)
    {
        return vec_ptr(proj[level]);
    }

    template <class TreeContainer> H2Opus_Real *getTreeConatinerLevelData(TreeContainer &proj, int level)
    {
        return vec_ptr(proj[level]);
    }

    void printData()
    {
        printf("Tree depth: %d\n", depth);
        printf("Transfer Dims: \n");
        for (int i = 0; i < depth + 1; i++)
            printf("%d ", trans_dim[i]);
        printf("\nLevel pointers:\n");
        for (int i = 0; i < depth + 1; i++)
            printf("%d ", level_ptrs[i]);
        printf("\n");
    }

    template <class TreeContainer> void populateProjectionTree(TreeContainer &proj, int start_level = 0)
    {
        proj.resize(depth);
        for (int level = start_level; level < depth; level++)
            proj[level].resize(getLevelSize(level) * getLevelRank(level) * getLevelRank(level), 0);
    }

    template <class TreeContainer> void populateTransferTree(TreeContainer &transfer)
    {
        if (transfer.size() != (size_t)depth)
            transfer.resize(depth);
        for (int level = 0; level < depth; level++)
        {
            int level_nodes = getLevelSize(level);
            int level_trans_rows, level_trans_cols;
            getTransferDims(level, level_trans_rows, level_trans_cols);
            size_t level_entries = level_trans_rows * level_trans_cols * level_nodes;
            if (level_entries != 0)
                transfer[level].resize(level_entries, 0);
        }
    }

    template <class TreeContainer> void setVectorTree(TreeContainer &tree, H2Opus_Real val)
    {
        for (int level = 0; level < depth; level++)
            thrust::fill(tree[level].begin(), tree[level].end(), val);
    }

  private:
    void init(const BasisTreeLevelData &b)
    {
        this->level_ptrs = b.level_ptrs;
        this->trans_dim = b.trans_dim;
        this->depth = b.depth;
        this->leaf_size = b.leaf_size;
        this->basis_leaves = b.basis_leaves;
        this->max_children = b.max_children;
        this->nested_root_level = b.nested_root_level;
    }
};

#endif
