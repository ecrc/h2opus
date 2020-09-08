#ifndef __HNODE_TREE_DATA_H__
#define __HNODE_TREE_DATA_H__

#define COMPRESSION_BASIS_GEN_MAX_NODES 2000

struct HNodeTreeLevelData
{
    // Pointers to the node arrays for each level of the tree
    std::vector<int> rank_level_ptrs, inner_node_level_ptrs, level_ptrs;

    // Ranks per level - copied from the basis tree
    std::vector<int> level_rank;

    // Depth for convenience
    int depth, u_tree_start_level, v_tree_start_level;

    HNodeTreeLevelData()
    {
        depth = u_tree_start_level = v_tree_start_level = 0;
    }

    HNodeTreeLevelData(const HNodeTreeLevelData &h)
    {
        init(h);
    }

    HNodeTreeLevelData &operator=(const HNodeTreeLevelData &h)
    {
        init(h);
        return *this;
    }

    void clearData()
    {
        thrust::fill(level_rank.begin(), level_rank.end(), 0);
    }

    void allocateLevels(int levels)
    {
        this->depth = levels;

        rank_level_ptrs.resize(levels + 1, 0);
        inner_node_level_ptrs.resize(levels + 1, 0);
        level_ptrs.resize(levels + 1, 0);
        level_rank.resize(levels);
    }

    void setRankFromBasis(BasisTreeLevelData &level_data, int start_level)
    {
        // Copy over the level ranks from the basis tree
        for (int i = 0; i < depth; i++)
            level_rank[i] = level_data.getLevelRank(start_level + i);
    }

    size_t getMaxCouplingLevelSize()
    {
        size_t max_size = 0;
        for (int level = 0; level < depth; level++)
        {
            size_t num_level_nodes = getCouplingLevelSize(level);
            int level_rank = getLevelRank(level);
            size_t level_size = num_level_nodes * level_rank * level_rank;
            max_size = std::max(max_size, level_size);
        }
        return max_size;
    }

    size_t getMaxLevelCouplingNodes()
    {
        size_t max_nodes = 0;
        for (int level = 0; level < depth; level++)
        {
            size_t num_level_nodes = getCouplingLevelSize(level);
            max_nodes = std::max(max_nodes, num_level_nodes);
        }
        return max_nodes;
    }

    int getLargestRank()
    {
        int max_rank = 0;
        for (int level = 0; level < depth; level++)
            if (getLevelRank(level) > max_rank)
                max_rank = getLevelRank(level);
        return max_rank;
    }

    void setLevelPointers(int *level_counts, int *rank_level_counts, int *inode_level_counts)
    {
        thrust::inclusive_scan(level_counts, level_counts + depth, &level_ptrs[1]);
        thrust::inclusive_scan(rank_level_counts, rank_level_counts + depth, &rank_level_ptrs[1]);
        thrust::inclusive_scan(inode_level_counts, inode_level_counts + depth, &inner_node_level_ptrs[1]);
    }

    void getInodeLevelRange(int level, int &start, int &end)
    {
        start = getInodeLevelStart(level);
        end = getInodeLevelEnd(level);
    }

    int getInodeLevelStart(int level)
    {
        return inner_node_level_ptrs[level];
    }

    int getInodeLevelEnd(int level)
    {
        return inner_node_level_ptrs[level + 1];
    }

    int getInodeLevelSize(int level)
    {
        return getInodeLevelEnd(level) - getInodeLevelStart(level);
    }

    void getCouplingLevelRange(int level, int &start, int &end)
    {
        start = getCouplingLevelStart(level);
        end = getCouplingLevelEnd(level);
    }

    int getCouplingLevelStart(int level)
    {
        return rank_level_ptrs[level];
    }

    int getCouplingLevelEnd(int level)
    {
        return rank_level_ptrs[level + 1];
    }

    int getCouplingLevelSize(int level)
    {
        return getCouplingLevelEnd(level) - getCouplingLevelStart(level);
    }

    int getLevelRank(int level)
    {
        return level_rank[level];
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

  private:
    void init(const HNodeTreeLevelData &h)
    {
        this->rank_level_ptrs = h.rank_level_ptrs;
        this->inner_node_level_ptrs = h.inner_node_level_ptrs;
        this->level_ptrs = h.level_ptrs;
        this->level_rank = h.level_rank;

        this->depth = h.depth;
        this->u_tree_start_level = h.u_tree_start_level;
        this->v_tree_start_level = h.v_tree_start_level;
    }
};

// Block sparse node data
enum BSNPointerDirection
{
    BSN_DIRECTION_COLUMN,
    BSN_DIRECTION_ROW
};

template <class IntVector> struct THNodeTreeBSNData
{
    typedef typename VectorArray<IntVector>::type IntVectorArray;

    // Node and pointers
    IntVector dense_node_indexes, dense_ptrs;
    IntVector dense_batch_indexes;
    std::vector<int> dense_batch_ptr;

    // Node and pointers for the coupling matrices stored by level
    IntVectorArray coupling_node_indexes, coupling_ptrs;
    IntVectorArray coupling_batch_indexes;
    std::vector<std::vector<int>> coupling_batch_ptr;

    // Per level cached node block pointers for the batching of
    // optimal basis generation during compression to avoid
    // copying data down for the GPU version
    std::vector<std::vector<int>> cached_coupling_ptrs;

    // Depth for convenience
    int depth;

    // The max number of blocks in a single row/column at each level
    std::vector<int> max_nodes;

    // Pointer direction (row or column)
    BSNPointerDirection direction;

    THNodeTreeBSNData()
    {
        this->depth = 0;
        this->direction = BSN_DIRECTION_ROW;
    }

    template <class otherIntVector> THNodeTreeBSNData &operator=(const THNodeTreeBSNData<otherIntVector> &h)
    {
        copyThrustArray(this->dense_node_indexes, h.dense_node_indexes);
        copyThrustArray(this->dense_ptrs, h.dense_ptrs);
        copyThrustArray(this->dense_batch_indexes, h.dense_batch_indexes);

        this->dense_batch_ptr = h.dense_batch_ptr;
        this->max_nodes = h.max_nodes;

        // Deep copy for this one
        resizeThrustArray(coupling_node_indexes, h.depth);
        resizeThrustArray(coupling_ptrs, h.depth);
        resizeThrustArray(coupling_batch_indexes, h.depth);
        coupling_batch_ptr.resize(h.depth);

        for (int level = 0; level < h.depth; level++)
        {
            copyThrustArray(this->coupling_node_indexes[level], h.coupling_node_indexes[level]);
            copyThrustArray(this->coupling_ptrs[level], h.coupling_ptrs[level]);
            copyThrustArray(this->coupling_batch_indexes[level], h.coupling_batch_indexes[level]);
            this->coupling_batch_ptr[level] = h.coupling_batch_ptr[level];
        }

        this->cached_coupling_ptrs = h.cached_coupling_ptrs;
        this->depth = h.depth;
        this->direction = h.direction;

        return *this;
    }

    void setCachedCouplingPtrs()
    {
        const int block_size = COMPRESSION_BASIS_GEN_MAX_NODES;

        cached_coupling_ptrs.resize(depth);

        for (int i = 0; i < depth; i++)
        {
            IntVector &level_coupling_ptrs = coupling_ptrs[i];
            std::vector<int> &cached_level_coupling_ptrs = cached_coupling_ptrs[i];

            if (level_coupling_ptrs.size() == 0)
            {
                printf("setCachedCouplingPtrs fatal error\n");
                assert(0);
            }
            int level_clusters = (int)level_coupling_ptrs.size() - 1;
            int level_blocks =
                (level_clusters % block_size == 0 ? level_clusters / block_size : level_clusters / block_size + 1);

            cached_level_coupling_ptrs.resize(level_blocks + 1);
            cached_level_coupling_ptrs[0] = 0;
            for (int b = 0; b < level_blocks + 1; b++)
            {
                int last_row = std::min(b * block_size, level_clusters);
                cached_level_coupling_ptrs[b] = level_coupling_ptrs[last_row];
            }
        }
    }

    void setLevelMaxNodes()
    {
        this->max_nodes.resize(depth, 0);
        for (int level = 0; level < depth; level++)
        {
            IntVector &storage_ptr = coupling_ptrs[level];
            int dofs = (int)(storage_ptr.size()) - 1;
            for (int i = 0; i < dofs; i++)
            {
                int nodes = storage_ptr[i + 1] - storage_ptr[i];
                if (max_nodes[level] < nodes)
                    max_nodes[level] = nodes;
            }
        }
    }

    void setDirection(BSNPointerDirection direction)
    {
        this->direction = direction;
    }

    void allocateLevels(int levels)
    {
        this->depth = levels;
        resizeThrustArray(coupling_ptrs, levels);
        resizeThrustArray(coupling_node_indexes, levels);
        resizeThrustArray(coupling_batch_indexes, levels);
        this->coupling_batch_ptr.resize(levels);
    }

    void allocateBSNDenseData(int dense_nodes, int dense_rows_cols)
    {
        resizeThrustArray(dense_ptrs, dense_rows_cols + 1);
        resizeThrustArray(dense_node_indexes, dense_nodes);
        resizeThrustArray(dense_batch_indexes, dense_nodes);
    }

    void allocateBSNLevelData(int level, int coupling_nodes, int coupling_rows_cols)
    {
        resizeThrustArray(coupling_ptrs[level], coupling_rows_cols + 1);
        resizeThrustArray(coupling_node_indexes[level], coupling_nodes);
        resizeThrustArray(coupling_batch_indexes[level], coupling_nodes);
    }
};

template <class IntVector> struct HNodeTreeBSRData
{
    typedef typename VectorArray<IntVector>::type IntVectorArray;

    // Column and row pointers for the dense blocks
    IntVector dense_row_ptr, dense_col_index;

    // Colum and row pointers for the coupling matrices stored by level
    IntVectorArray rank_row_ptr, rank_col_index;

    // The max number of blocks in a single row at each level
    IntVector max_row_blocks;

    // Depth for convenience
    int depth;

    HNodeTreeBSRData()
    {
        this->depth = 0;
    }

    template <class otherIntVector> HNodeTreeBSRData &operator=(const HNodeTreeBSRData<otherIntVector> &h)
    {
        copyThrustArray(this->dense_row_ptr, h.dense_row_ptr);
        copyThrustArray(this->dense_col_index, h.dense_col_index);

        this->max_row_blocks = h.max_row_blocks;

        // Deep copy for this one
        resizeThrustArray(rank_row_ptr, h.depth);
        resizeThrustArray(rank_col_index, h.depth);
        for (int level = 0; level < h.depth; level++)
        {
            copyThrustArray(this->rank_row_ptr[level], h.rank_row_ptr[level]);
            copyThrustArray(this->rank_col_index[level], h.rank_col_index[level]);
        }

        this->depth = h.depth;

        return *this;
    }

    H2Opus_Real getMemoryUsage()
    {
        size_t int_entries = dense_row_ptr.size() + dense_col_index.size() + max_row_blocks.size();
        for (int i = 0; i < depth; i++)
            int_entries += rank_row_ptr.size() + rank_col_index.size();
        return ((H2Opus_Real)int_entries * sizeof(int)) * 1e-9;
    }

    void setLevelMaxRowBlocks()
    {
        this->max_row_blocks.resize(depth, 0);
        for (int level = 0; level < depth; level++)
        {
            IntVector &row_ptr = rank_row_ptr[level];
            int rows = getLevelBSRRows(level);
            for (int i = 0; i < rows; i++)
            {
                int cols = row_ptr[i + 1] - row_ptr[i];
                if (max_row_blocks[level] < cols)
                    max_row_blocks[level] = cols;
            }
        }
    }

    int getMaxRowBlocks()
    {
        int max_blocks = 0;
        for (int i = 0; i < depth; i++)
            if (max_blocks < max_row_blocks[i])
                max_blocks = max_row_blocks[i];
        return max_blocks;
    }

    void allocateLevels(int levels)
    {
        this->depth = levels;
        this->rank_row_ptr.resize(levels);
        this->rank_col_index.resize(levels);
    }

    void getLevelBSRData(int level, int **ia, int **ja)
    {
        *ia = vec_ptr(rank_row_ptr[level]);
        *ja = vec_ptr(rank_col_index[level]);
    }

    void getDenseBSRData(int **ia, int **ja)
    {
        *ia = vec_ptr(dense_row_ptr);
        *ja = vec_ptr(dense_col_index);
    }

    int getLevelBSRRows(int level)
    {
        return (int)rank_row_ptr[level].size() - 1;
    }

    int getDenseBSRRows()
    {
        return (int)dense_row_ptr.size() - 1;
    }

    void allocateBSRDenseData(int dense_nodes, int dense_rows)
    {
        // the row pointer should be of size (rows + 1): this is the same as the number
        // of row basis leaf nodes which we can get from the basis tree
        resizeThrustArray(dense_row_ptr, dense_rows + 1);

        // the column pointer should be the same size as the number of dense leaves
        resizeThrustArray(dense_col_index, dense_nodes);
    }

    void allocateBSRLevelData(int level, int coupling_nodes, int coupling_rows)
    {
        resizeThrustArray(rank_row_ptr[level], coupling_rows + 1);
        resizeThrustArray(rank_col_index[level], coupling_nodes);
    }

    void printLevelData(int level)
    {
        printf("Rank Level %d Row pointers: \n", level);
        for (int i = 0; i < rank_row_ptr[level].size(); i++)
            printf("%d ", rank_row_ptr[level][i]);
        printf("\n");
        printf("Rank Level %d Col indexes : \n", level);
        for (int i = 0; i < rank_col_index[level].size(); i++)
            printf("%d ", rank_col_index[level][i]);
        printf("\n");
    }

    void printDenseData()
    {
        printf("Dense Row pointers: \n");
        for (int i = 0; i < dense_row_ptr.size(); i++)
            printf("%d ", dense_row_ptr[i]);
        printf("\n");
        printf("Dense Col indexes : \n");
        for (int i = 0; i < dense_col_index.size(); i++)
            printf("%d ", dense_col_index[i]);
        printf("\n");
    }

    void print()
    {
        printDenseData();
        for (int i = 0; i < depth; i++)
            printLevelData(i);
    }
};

#endif
