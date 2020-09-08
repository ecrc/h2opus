template <int hw> void TBasisTree<hw>::allocateNodes(int num_nodes)
{
    head.resize(num_nodes, H2OPUS_EMPTY_NODE);
    next.resize(num_nodes, H2OPUS_EMPTY_NODE);
    parent.resize(num_nodes, H2OPUS_EMPTY_NODE);
    global_cluster_index.resize(num_nodes, H2OPUS_EMPTY_NODE);

    node_start.resize(num_nodes, 0);
    node_len.resize(num_nodes, 0);
    this->num_nodes = num_nodes;
}

template <int hw> void TBasisTree<hw>::allocateLevels(int *level_counts, int levels)
{
    level_data.allocateLevels(level_counts, levels);
    trans_mem.resize(levels);

    depth = levels;
    // Don't allocate any leaves if there is no index_map
    basis_leaves = (index_map.size() == 0 ? 0 : level_counts[levels - 1]);
    level_data.basis_leaves = basis_leaves;
    // printf("%d basis leaves\n", basis_leaves);

    allocateNodes(level_data.level_ptrs[depth]);
}

template <int hw> void TBasisTree<hw>::allocateMatrixData(int *level_ranks, int levels, int leaf_size)
{
    this->leaf_size = leaf_size;
    this->depth = levels;
    level_data.setLevelRanks(level_ranks);
    level_data.leaf_size = leaf_size;

    // First the basis leaves
    size_t basis_entries = leaf_size * getLevelRank(levels - 1) * basis_leaves;
    if (basis_entries != 0)
        basis_mem.resize(basis_entries, 0);
    level_data.populateTransferTree(trans_mem);
}

template <int hw>
void TBasisTree<hw>::getKDTreeLevelCounts(std::vector<int> &counts, TH2OpusKDTree<H2Opus_Real, hw> &kdtree,
                                          int root_node, int level, int max_depth)
{
    if (level >= max_depth)
        return;

    counts[level]++;
    int child = kdtree.getHeadChild(root_node);
    while (child != H2OPUS_EMPTY_NODE)
    {
        getKDTreeLevelCounts(counts, kdtree, child, level + 1, max_depth);
        child = kdtree.getNextChild(child);
    }
}

template <int hw>
void TBasisTree<hw>::getKDtreeStructure(TH2OpusKDTree<H2Opus_Real, hw> &kdtree, int kd_node_index, int parent_node,
                                        int level, std::vector<int> &level_counts, std::vector<int> &node_tails,
                                        int max_depth)
{
    if (level >= max_depth)
        return;

    int node_index = getLevelStart(level) + level_counts[level];
    level_counts[level]++;

    // Set node properties
    int node_begin, node_end;
    kdtree.getNodeLimits(kd_node_index, node_begin, node_end);

    node_start[node_index] = node_begin - index_map_offset;
    node_len[node_index] = node_end - node_begin;
    global_cluster_index[node_index] = kd_node_index;

    // Set the node parent, head, and next pointers
    parent[node_index] = parent_node;

    if (parent_node != H2OPUS_EMPTY_NODE)
    {
        if (head[parent_node] == H2OPUS_EMPTY_NODE)
        {
            head[parent_node] = node_index;
            node_tails[parent_node] = node_index;
        }
        else
        {
            next[node_tails[parent_node]] = node_index;
            node_tails[parent_node] = node_index;
        }
    }

    int child = kdtree.getHeadChild(kd_node_index);
    while (child != H2OPUS_EMPTY_NODE)
    {
        getKDtreeStructure(kdtree, child, node_index, level + 1, level_counts, node_tails, max_depth);
        child = kdtree.getNextChild(child);
    }
}

template <int hw>
void TBasisTree<hw>::generateStructureFromKDTree(TH2OpusKDTree<H2Opus_Real, hw> &kdtree, int root_node, bool copyMap,
                                                 int max_depth)
{
    // Grab all meta data from the kdtree
    int node_begin, node_end;
    kdtree.getNodeLimits(root_node, node_begin, node_end);

    this->leaf_size = kdtree.getLeafSize();
    this->depth = max_depth;
    this->index_map_offset = node_begin;

    level_data.max_children = kdtree.getMaxChildren();
    level_data.leaf_size = leaf_size;

    // Copy over the index map
    int num_points = node_end - node_begin;
    int *root_index_map = kdtree.getIndexMap() + node_begin;
    if (copyMap)
        index_map = IntVector(root_index_map, root_index_map + num_points);
    else
        index_map.clear();

    // Get level counts
    std::vector<int> level_counts(depth, 0);
    getKDTreeLevelCounts(level_counts, kdtree, root_node, 0, depth);
    allocateLevels(&level_counts[0], depth);

    // Temporarily keep track of the last node added to the child list
    // of a node to make node insertions faster
    std::vector<int> node_tails(num_nodes, H2OPUS_EMPTY_NODE);

    // Reset the level counter so we can use it to keep track of indexes within a level
    std::fill(level_counts.begin(), level_counts.end(), 0);
    getKDtreeStructure(kdtree, root_node, H2OPUS_EMPTY_NODE, 0, level_counts, node_tails, depth);
}
