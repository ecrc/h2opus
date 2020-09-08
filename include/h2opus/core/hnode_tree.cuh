#include <h2opus/util/morton.h>
#include <thrust/binary_search.h>

template <int hw> void THNodeTree<hw>::allocateNodes(int num_nodes)
{
    this->num_nodes = num_nodes;

    head.resize(num_nodes, H2OPUS_EMPTY_NODE);
    next.resize(num_nodes, H2OPUS_EMPTY_NODE);
    parent.resize(num_nodes, H2OPUS_EMPTY_NODE);
    node_u_index.resize(num_nodes, H2OPUS_EMPTY_NODE);
    node_v_index.resize(num_nodes, H2OPUS_EMPTY_NODE);
    node_morton_level_index.resize(num_nodes, H2OPUS_EMPTY_NODE);
    node_to_leaf.resize(num_nodes, H2OPUS_EMPTY_NODE);
    node_type.resize(num_nodes, HMATRIX_INNER_NODE);
}

template <int hw> void THNodeTree<hw>::allocateLeaves(int dense, int rank)
{
    this->num_dense_leaves = dense;
    this->num_rank_leaves = rank;

    dense_leaf_tree_index.resize(dense);
    rank_leaf_tree_index.resize(rank);
}

template <int hw> void THNodeTree<hw>::allocateInnerNodes(int inodes)
{
    this->num_inodes = inodes;
    inner_node_tree_index.resize(inodes);
}

template <int hw> void THNodeTree<hw>::allocateLevels(int levels)
{
    this->depth = levels;
    rank_leaf_mem.resize(levels);
    level_data.allocateLevels(levels);
    bsr_data.allocateLevels(levels);
    bsn_row_data.allocateLevels(levels);
    bsn_col_data.allocateLevels(levels);
}

template <int hw>
void THNodeTree<hw>::setLevelPointers(int *level_counts, int *rank_level_counts, int *inode_level_counts)
{
    level_data.setLevelPointers(level_counts, rank_level_counts, inode_level_counts);
}

template <int hw>
void THNodeTree<hw>::allocateMatrixData(BasisTreeLevelData &basis_level_data, int start_level, int depth)
{
    assert(this->depth = depth);

    this->level_data.setRankFromBasis(basis_level_data, start_level);

    // Allocate the low rank matrix data for each level
    for (int level = 0; level < depth; level++)
    {
        size_t num_level_nodes = level_data.getCouplingLevelSize(level + start_level);
        size_t level_rank = level_data.getLevelRank(level + start_level);
        size_t num_elements = num_level_nodes * level_rank * level_rank;

        rank_leaf_mem[level].resize(num_elements, 0);
    }
}

template <int hw> H2Opus_Real *THNodeTree<hw>::getCouplingMatrixLevelData(int level)
{
    return vec_ptr(rank_leaf_mem[level]);
}

template <int hw> H2Opus_Real *THNodeTree<hw>::getCouplingMatrix(int level, size_t level_index)
{
    int level_rank = level_data.getLevelRank(level);
    size_t level_offset = (size_t)(level_rank * level_rank) * level_index;
    assert(level_offset + level_rank * level_rank <= rank_leaf_mem[level].size());

    return vec_ptr(rank_leaf_mem[level]) + level_offset;
}

template <int hw> H2Opus_Real *THNodeTree<hw>::getDenseMatrix(size_t index)
{
    size_t offset = index * (size_t)(leaf_size * leaf_size);
    assert(offset < dense_leaf_mem.size());
    return vec_ptr(dense_leaf_mem) + offset;
}

template <int hw> void THNodeTree<hw>::getLevelBSRData(int level, int **ia, int **ja, H2Opus_Real **a)
{
    *a = vec_ptr(rank_leaf_mem[level]);
    bsr_data.getLevelBSRData(level, ia, ja);
}

template <int hw> void THNodeTree<hw>::getDenseBSRData(int **ia, int **ja, H2Opus_Real **a)
{
    *a = vec_ptr(dense_leaf_mem);
    bsr_data.getDenseBSRData(ia, ja);
}

template <int hw> int THNodeTree<hw>::getHNodeIndex(int level, int i, int j)
{
    int *search_start = vec_ptr(node_morton_level_index) + getLevelStart(level);
    int *search_end = search_start + getLevelSize(level);
    int key = morton_encode(j, i);

    int *position = thrust::lower_bound(search_start, search_end, key);
    if (position == search_end || *position != key)
        return -1;
    else
        return (position - search_start) + getLevelStart(level);
}

template <int hw>
void THNodeTree<hw>::determineAdmissibleBlocks(TH2OpusKDTree<H2Opus_Real, hw> &kdtree,
                                               TH2OpusAdmissibility<H2Opus_Real, hw> &admissibility,
                                               TBasisTree<hw> &u_basis_tree, TBasisTree<hw> &v_basis_tree,
                                               std::vector<int> &level_counts, std::vector<int> &rank_level_counts,
                                               std::vector<int> &inode_level_counts, int &dense_leaf_count, int u_node,
                                               int v_node, int level, int max_depth)
{
    if (level >= max_depth)
        return;

    level_counts[level]++;

    int u_cluster_index = u_basis_tree.global_cluster_index[u_node];
    int v_cluster_index = v_basis_tree.global_cluster_index[v_node];

    if (admissibility(&kdtree, u_cluster_index, v_cluster_index))
        rank_level_counts[level]++;
    else if (kdtree.isLeaf(u_cluster_index) || kdtree.isLeaf(v_cluster_index))
        dense_leaf_count++;
    else
    {
        inode_level_counts[level]++;
        int u_child = u_basis_tree.head[u_node];
        while (u_child != H2OPUS_EMPTY_NODE)
        {
            int v_child = v_basis_tree.head[v_node];
            while (v_child != H2OPUS_EMPTY_NODE)
            {
                determineAdmissibleBlocks(kdtree, admissibility, u_basis_tree, v_basis_tree, level_counts,
                                          rank_level_counts, inode_level_counts, dense_leaf_count, u_child, v_child,
                                          level + 1, max_depth);
                v_child = v_basis_tree.next[v_child];
            }
            u_child = u_basis_tree.next[u_child];
        }
    }
}

template <int hw>
void THNodeTree<hw>::buildAdmissibleBlocks(TH2OpusKDTree<H2Opus_Real, hw> &kdtree,
                                           TH2OpusAdmissibility<H2Opus_Real, hw> &admissibility,
                                           TBasisTree<hw> &u_basis_tree, TBasisTree<hw> &v_basis_tree,
                                           std::vector<int> &level_counts, std::vector<int> &rank_level_counts,
                                           std::vector<int> &inode_level_counts, int &dense_leaf_count, int u_node,
                                           int v_node, int parent_node, int level, int max_depth)
{
    if (level >= max_depth)
        return;

    int node_index = getLevelStart(level) + level_counts[level];
    level_counts[level]++;

    int u_level_index = u_node - u_basis_tree.getLevelStart(level);
    int v_level_index = v_node - v_basis_tree.getLevelStart(level);
    node_u_index[node_index] = u_node;
    node_v_index[node_index] = v_node;
    node_morton_level_index[node_index] = morton_encode(v_level_index, u_level_index);
    parent[node_index] = parent_node;

    if (parent_node != H2OPUS_EMPTY_NODE)
    {
        if (head[parent_node] == H2OPUS_EMPTY_NODE)
            head[parent_node] = node_index;
        else
        {
            int old_head = head[parent_node];
            head[parent_node] = node_index;
            next[node_index] = old_head;
        }
    }

    // Pass the matrix node defined by the cluster pair (u, v) to the admissibility condition
    int u_cluster_index = u_basis_tree.global_cluster_index[u_node];
    int v_cluster_index = v_basis_tree.global_cluster_index[v_node];

    if (admissibility(&kdtree, u_cluster_index, v_cluster_index))
    {
        int rank_leaf_index = getCouplingLevelStart(level) + rank_level_counts[level];

        node_type[node_index] = HMATRIX_RANK_MATRIX;
        node_to_leaf[node_index] = rank_leaf_index;

        rank_leaf_tree_index[rank_leaf_index] = node_index;
        rank_level_counts[level]++;
    }
    else if (kdtree.isLeaf(u_cluster_index) || kdtree.isLeaf(v_cluster_index))
    {
        node_type[node_index] = HMATRIX_DENSE_MATRIX;
        node_to_leaf[node_index] = dense_leaf_count;

        dense_leaf_tree_index[dense_leaf_count] = node_index;
        dense_leaf_count++;
    }
    else
    {
        int inner_node_index = getInodeLevelStart(level) + inode_level_counts[level];
        inner_node_tree_index[inner_node_index] = node_index;
        inode_level_counts[level]++;

        int u_child = u_basis_tree.head[u_node];
        while (u_child != H2OPUS_EMPTY_NODE)
        {
            int v_child = v_basis_tree.head[v_node];
            while (v_child != H2OPUS_EMPTY_NODE)
            {
                buildAdmissibleBlocks(kdtree, admissibility, u_basis_tree, v_basis_tree, level_counts,
                                      rank_level_counts, inode_level_counts, dense_leaf_count, u_child, v_child,
                                      node_index, level + 1, max_depth);
                v_child = v_basis_tree.next[v_child];
            }
            u_child = u_basis_tree.next[u_child];
        }
    }
}
template <int hw> void THNodeTree<hw>::allocateDenseLeafMemory()
{
    // Allocate the dense data
    size_t entries_per_dense_node = leaf_size * leaf_size;
    size_t num_dense_elements = num_dense_leaves * entries_per_dense_node;
    dense_leaf_mem.resize(num_dense_elements, 0);
}

template <int hw>
void THNodeTree<hw>::determineStructure(TH2OpusKDTree<H2Opus_Real, hw> &kdtree,
                                        TH2OpusAdmissibility<H2Opus_Real, hw> &admissibility,
                                        TBasisTree<hw> &u_basis_tree, int u_start_level, TBasisTree<hw> &v_basis_tree,
                                        int v_start_level, int max_depth, std::vector<int> v_list)
{
    num_dense_leaves = 0;
    depth = u_basis_tree.depth - u_start_level;
    if (max_depth <= depth)
        depth = max_depth;
    level_data.u_tree_start_level = u_start_level;
    level_data.v_tree_start_level = v_start_level;

    leaf_size = u_basis_tree.leaf_size;
    assert((max_depth <= v_basis_tree.depth - v_start_level || depth == v_basis_tree.depth - v_start_level) &&
           leaf_size == v_basis_tree.leaf_size);

    // First a pass to determine all admissible nodes in the matrix
    std::vector<int> rank_level_counts(depth, 0), inode_level_counts(depth, 0);
    std::vector<int> level_counts(depth, 0);

    // Loop over the level and recursively determine admissibilty
    int u_start, u_end, v_start, v_end;
    u_basis_tree.getLevelRange(u_start_level, u_start, u_end);
    v_basis_tree.getLevelRange(v_start_level, v_start, v_end);

    for (int u_node = u_start; u_node < u_end; u_node++)
    {
        for (int v_index = 0; v_index < (int)v_list.size(); v_index++)
        {
            int v_node = v_list[v_index];
            determineAdmissibleBlocks(kdtree, admissibility, u_basis_tree, v_basis_tree, level_counts,
                                      rank_level_counts, inode_level_counts, num_dense_leaves, u_node, v_node, 0,
                                      max_depth);
        }
    }

    // Now allocate the necessary space and level pointers for the amount of admissible/dense/inner
    // nodes that we just determined
    allocateLevels(level_counts.size());
    setLevelPointers(&level_counts[0], &rank_level_counts[0], &inode_level_counts[0]);
    allocateLeaves(num_dense_leaves, level_data.rank_level_ptrs[depth]);
    allocateInnerNodes(level_data.inner_node_level_ptrs[depth]);
    allocateNodes(level_data.level_ptrs[depth]);

    // We can allocate the dense leaf memory as soon as we determine the number of
    // dense leaves so we do that here
    allocateDenseLeafMemory();

    // Reset the level counts so we can use them for node indexing
    std::fill(rank_level_counts.begin(), rank_level_counts.end(), 0);
    std::fill(inode_level_counts.begin(), inode_level_counts.end(), 0);
    std::fill(level_counts.begin(), level_counts.end(), 0);

    // Now we have counts to generate indexes, we can go ahead and generate the structure
    // of the matrix tree
    int dummy_leaf_count = 0;
    for (int u_node = u_start; u_node < u_end; u_node++)
    {
        for (int v_index = 0; v_index < (int)v_list.size(); v_index++)
        {
            int v_node = v_list[v_index];
            buildAdmissibleBlocks(kdtree, admissibility, u_basis_tree, v_basis_tree, level_counts, rank_level_counts,
                                  inode_level_counts, dummy_leaf_count, u_node, v_node, H2OPUS_EMPTY_NODE, 0,
                                  max_depth);
        }
    }
}

template <int hw>
void THNodeTree<hw>::determineStructure(TH2OpusKDTree<H2Opus_Real, hw> &kdtree,
                                        TH2OpusAdmissibility<H2Opus_Real, hw> &admissibility,
                                        TBasisTree<hw> &basis_tree)
{
    std::vector<int> dummy_v(1, 0);
    determineStructure(kdtree, admissibility, basis_tree, 0, basis_tree, 0, basis_tree.depth, dummy_v);
}

template <int hw>
void THNodeTree<hw>::extractLevelUVInodeIndexes(std::vector<int> &u_indexes, std::vector<int> &v_indexes, int level)
{
    int num_inodes = level_data.getInodeLevelSize(level);
    u_indexes.resize(num_inodes);
    v_indexes.resize(num_inodes);

    int inode_start, inode_end;
    level_data.getInodeLevelRange(level, inode_start, inode_end);

    for (int i = inode_start; i < inode_end; i++)
    {
        int inode_tree_index = inner_node_tree_index[i];
        u_indexes[i - inode_start] = node_u_index[inode_tree_index];
        v_indexes[i - inode_start] = node_v_index[inode_tree_index];
    }
}

template <class IntVector> void reorderArray(IntVector &array, int *map)
{
    IntVector temp_copy = array;
    for (size_t i = 0; i < temp_copy.size(); i++)
        array[map[i]] = temp_copy[i];
}

template <int hw> void THNodeTree<hw>::reorderLeafIndexes(int *dense_map, int *coupling_index_map)
{
    if (num_dense_leaves != 0 && dense_map != NULL)
    {
        // First remap the tree to leaf array
        for (int i = 0; i < num_dense_leaves; i++)
        {
            int tree_index = dense_leaf_tree_index[i];
            node_to_leaf[tree_index] = dense_map[i];
        }
        // Now remap the leaf to tree index array
        reorderArray(dense_leaf_tree_index, dense_map);
    }

    if (num_rank_leaves != 0 && coupling_index_map != NULL)
    {
        // Do the same for the low rank nodes
        for (int i = 0; i < num_rank_leaves; i++)
        {
            int tree_index = rank_leaf_tree_index[i];
            node_to_leaf[tree_index] = coupling_index_map[i];
        }
        reorderArray(rank_leaf_tree_index, coupling_index_map);
    }
}

template <int hw> int THNodeTree<hw>::getLevelBSRRows(int level)
{
    return bsr_data.getLevelBSRRows(level);
}

template <int hw>
void THNodeTree<hw>::allocateBSRData(TBasisTree<hw> &u_basis_tree, TBasisTree<hw> &v_basis_tree, int u_start_level,
                                     int v_start_level)
{
    std::vector<int> dense_index_map(num_dense_leaves);
    std::vector<int> coupling_index_map(num_rank_leaves);

    allocateBSRData(u_basis_tree, v_basis_tree, u_start_level, v_start_level, vec_ptr(coupling_index_map),
                    vec_ptr(dense_index_map));
}

template <int hw>
void THNodeTree<hw>::allocateBSNData(TBasisTree<hw> &u_basis_tree, TBasisTree<hw> &v_basis_tree, int u_start_level,
                                     int v_start_level)
{
    allocateRowBSNData(u_basis_tree, u_start_level);
    allocateColumnBSNData(v_basis_tree, v_start_level);
}

template <int hw>
void THNodeTree<hw>::generateBSNData(int *node_tree_index, int node_offset, int num_nodes, int *basis_indexes,
                                     int basis_index_offset, int basis_level_nodes, std::vector<int> &workspace,
                                     IntVector &bsn_ptrs, IntVector &bsn_node_indexes, std::vector<int> &bsn_batch_ptr,
                                     IntVector &batch_indexes)
{
    // Clear workspace
    thrust::fill(workspace.begin(), workspace.begin() + basis_level_nodes + 1, 0);

    // Determine the number of nodes in each row/column
    for (int i = 0; i < num_nodes; i++)
    {
        int tree_index = node_tree_index[node_offset + i];
        int basis_cr_index = basis_indexes[tree_index] - basis_index_offset;
        workspace[basis_cr_index]++;
    }

    // Now do a scan on the row/column counts to generate the row/column pointers
    thrust::exclusive_scan(workspace.begin(), workspace.begin() + basis_level_nodes + 1, bsn_ptrs.begin(), 0);

    // Copy the scanned data back into workspace so we can use it to determine
    // the positions of the node indexes in the node index array
    thrust::copy(bsn_ptrs.begin(), bsn_ptrs.end(), workspace.begin());

    // Now we generate the node indexes per row/column
    for (int i = 0; i < num_nodes; i++)
    {
        int tree_index = node_tree_index[node_offset + i];
        int basis_cr_index = basis_indexes[tree_index] - basis_index_offset;
        // dense_index_map[i] = workspace[basis_cr_index];
        bsn_node_indexes[workspace[basis_cr_index]++] = tree_index;
    }

    // Determine batch indexes and pointers
    int max_nodes = 0;
    for (int basis_node = 0; basis_node < basis_level_nodes; basis_node++)
    {
        int coupling_nodes = bsn_ptrs[basis_node + 1] - bsn_ptrs[basis_node];
        max_nodes = std::max(max_nodes, coupling_nodes);
    }
    bsn_batch_ptr.resize(max_nodes + 1);

    int csum = 0;
    bsn_batch_ptr[0] = 0;
    for (int batch_id = 0; batch_id < max_nodes; batch_id++)
    {
        int batch_size = 0;
        for (int basis_node = 0; basis_node < basis_level_nodes; basis_node++)
        {
            int coupling_nodes = bsn_ptrs[basis_node + 1] - bsn_ptrs[basis_node];
            if (batch_id < coupling_nodes)
            {
                int node_index = bsn_ptrs[basis_node] + batch_id;
                int tree_index = bsn_node_indexes[node_index];
                int leaf_index = node_to_leaf[tree_index];

                batch_indexes[leaf_index - node_offset] = csum + batch_size;
                batch_size++;
            }
        }
        csum += batch_size;
        bsn_batch_ptr[batch_id + 1] = csum;
    }
}

template <int hw> void THNodeTree<hw>::allocateRowBSNData(TBasisTree<hw> &u_basis_tree, int u_start_level)
{
    // Find out how much workspace we need based on the largest number
    // of rows in the tree by level
    int max_rows = u_basis_tree.basis_leaves;
    for (int level = 0; level < depth; level++)
    {
        int level_rows = u_basis_tree.getLevelSize(level + u_start_level);
        max_rows = std::max(level_rows, max_rows);
    }

    std::vector<int> workspace(max_rows + 1, 0);

    //////////////////////////////////////////////////////////////////////////////
    // First the dense pointers
    //////////////////////////////////////////////////////////////////////////////
    // Now we have to populate the row pointers by first calculating the number of non-zeros
    // in each row - this amounts to going through the dense leaves and incrementing a counter
    // for each node that we find in a row
    // The indexes are stored as tree node indexes when they are generated
    // so we have to subtract the tree index to start at 0
    int dense_rows = u_basis_tree.basis_leaves;

    if (dense_rows != 0)
    {
        int u_index_offset = u_basis_tree.getLevelStart(u_basis_tree.depth - 1);
        bsn_row_data.allocateBSNDenseData(num_dense_leaves, dense_rows);

        generateBSNData(vec_ptr(dense_leaf_tree_index), 0, num_dense_leaves, vec_ptr(node_u_index), u_index_offset,
                        dense_rows, workspace, bsn_row_data.dense_ptrs, bsn_row_data.dense_node_indexes,
                        bsn_row_data.dense_batch_ptr, bsn_row_data.dense_batch_indexes);
    }

    //////////////////////////////////////////////////////////////////////////////
    // Now the low rank pointers
    //////////////////////////////////////////////////////////////////////////////
    // Going to have to generate one index set per level in the tree
    for (int level = 0; level < depth; level++)
    {
        int level_num_coupling_nodes = level_data.getCouplingLevelSize(level);
        int level_num_rows = u_basis_tree.getLevelSize(level + u_start_level);
        int level_start = level_data.getCouplingLevelStart(level);
        int u_index_offset = u_basis_tree.getLevelStart(level + u_start_level);

        bsn_row_data.allocateBSNLevelData(level, level_num_coupling_nodes, level_num_rows);

        generateBSNData(vec_ptr(rank_leaf_tree_index), level_start, level_num_coupling_nodes, vec_ptr(node_u_index),
                        u_index_offset, level_num_rows, workspace, bsn_row_data.coupling_ptrs[level],
                        bsn_row_data.coupling_node_indexes[level], bsn_row_data.coupling_batch_ptr[level],
                        bsn_row_data.coupling_batch_indexes[level]);
    }

    bsn_row_data.setLevelMaxNodes();
    bsn_row_data.setDirection(BSN_DIRECTION_ROW);
    bsn_row_data.setCachedCouplingPtrs();
}

template <int hw> void THNodeTree<hw>::allocateColumnBSNData(TBasisTree<hw> &v_basis_tree, int v_start_level)
{
    int max_cols = v_basis_tree.basis_leaves;
    for (int level = 0; level < depth; level++)
    {
        int level_cols = v_basis_tree.getLevelSize(level + v_start_level);
        max_cols = std::max(level_cols, max_cols);
    }

    std::vector<int> workspace(max_cols + 1, 0);

    //////////////////////////////////////////////////////////////////////////////
    // First the dense pointers
    //////////////////////////////////////////////////////////////////////////////
    // Now we have to populate the row pointers by first calculating the number of non-zeros
    // in each row - this amounts to going through the dense leaves and incrementing a counter
    // for each node that we find in a row
    // The indexes are stored as tree node indexes when they are generated
    // so we have to subtract the tree index to start at 0
    int dense_cols = v_basis_tree.basis_leaves;

    if (dense_cols != 0)
    {
        int v_index_offset = v_basis_tree.getLevelStart(v_basis_tree.depth - 1);
        bsn_col_data.allocateBSNDenseData(num_dense_leaves, dense_cols);

        generateBSNData(vec_ptr(dense_leaf_tree_index), 0, num_dense_leaves, vec_ptr(node_v_index), v_index_offset,
                        dense_cols, workspace, bsn_col_data.dense_ptrs, bsn_col_data.dense_node_indexes,
                        bsn_col_data.dense_batch_ptr, bsn_col_data.dense_batch_indexes);
    }

    //////////////////////////////////////////////////////////////////////////////
    // Now the low rank pointers
    //////////////////////////////////////////////////////////////////////////////
    // Going to have to generate one index set per level in the tree
    for (int level = 0; level < depth; level++)
    {
        int level_num_coupling_nodes = level_data.getCouplingLevelSize(level);
        int level_num_cols = v_basis_tree.getLevelSize(level + v_start_level);
        int level_start = level_data.getCouplingLevelStart(level);
        int v_index_offset = v_basis_tree.getLevelStart(level + v_start_level);

        bsn_col_data.allocateBSNLevelData(level, level_num_coupling_nodes, level_num_cols);

        generateBSNData(vec_ptr(rank_leaf_tree_index), level_start, level_num_coupling_nodes, vec_ptr(node_v_index),
                        v_index_offset, level_num_cols, workspace, bsn_col_data.coupling_ptrs[level],
                        bsn_col_data.coupling_node_indexes[level], bsn_col_data.coupling_batch_ptr[level],
                        bsn_col_data.coupling_batch_indexes[level]);
    }

    bsn_col_data.setLevelMaxNodes();
    bsn_col_data.setDirection(BSN_DIRECTION_COLUMN);
    bsn_col_data.setCachedCouplingPtrs();
}

template <int hw>
void THNodeTree<hw>::allocateBSRData(TBasisTree<hw> &u_basis_tree, TBasisTree<hw> &v_basis_tree, int u_start_level,
                                     int v_start_level, int *coupling_index_map, int *dense_index_map)
{
    const int H2OPUS_BSR_INDEX_BASE = 1;

    // Find out how much workspace we need based on the largest number
    // of rows in the tree by level
    int max_rows = u_basis_tree.basis_leaves;
    for (int level = 0; level < depth; level++)
    {
        int level_rows = u_basis_tree.getLevelSize(level + u_start_level);
        if (max_rows < level_rows)
            max_rows = level_rows;
    }
    std::vector<int> workspace(max_rows + 1, 0);

    //////////////////////////////////////////////////////////////////////////////
    // First the dense pointers
    //////////////////////////////////////////////////////////////////////////////
    int dense_rows = u_basis_tree.basis_leaves;
    bsr_data.allocateBSRDenseData(num_dense_leaves, dense_rows);

    // Now we have to populate the row pointers by first calculating the number of non-zeros
    // in each row - this amounts to going through the dense leaves and incrementing a counter
    // for each node that we find in a row
    // The indexes are stored as tree node indexes when they are generated
    // so we have to subtract the tree index to start at 0
    int u_index_offset = u_basis_tree.getLevelStart(u_basis_tree.depth - 1);
    int v_index_offset = v_basis_tree.getLevelStart(v_basis_tree.depth - 1);

    if (dense_rows != 0)
    {
        for (int i = 0; i < num_dense_leaves; i++)
        {
            int tree_index = dense_leaf_tree_index[i];
            int row_index = node_u_index[tree_index] - u_index_offset;
            workspace[row_index]++;
        }

        // Now do a scan on the row counts to generate the row pointers
        thrust::exclusive_scan(workspace.begin(), workspace.end(), bsr_data.dense_row_ptr.begin(),
                               H2OPUS_BSR_INDEX_BASE);

        // Copy the scanned data back into workspace so we can use it to determine
        // the positions of the column indexes in the column pointer array
        thrust::copy(bsr_data.dense_row_ptr.begin(), bsr_data.dense_row_ptr.end(), workspace.begin());

        // Now we generate the column indexes per row and generate an index
        // transformation map that will let us reassign tree indexes
        for (int i = 0; i < num_dense_leaves; i++)
        {
            int tree_index = dense_leaf_tree_index[i];
            int row_index = node_u_index[tree_index] - u_index_offset;
            int col_index = node_v_index[tree_index] - v_index_offset;
            dense_index_map[i] = workspace[row_index] - H2OPUS_BSR_INDEX_BASE;
            bsr_data.dense_col_index[workspace[row_index]++ - H2OPUS_BSR_INDEX_BASE] =
                col_index + H2OPUS_BSR_INDEX_BASE;
        }
    }

    //////////////////////////////////////////////////////////////////////////////
    // Now the low rank pointers
    //////////////////////////////////////////////////////////////////////////////
    // Going to have to generate one index set per level in the tree
    for (int level = 0; level < depth; level++)
    {
        int level_num_rank_nodes = level_data.getCouplingLevelSize(level);
        int level_num_rows = u_basis_tree.getLevelSize(level + u_start_level);
        int level_start = level_data.getCouplingLevelStart(level);

        bsr_data.allocateBSRLevelData(level, level_num_rank_nodes, level_num_rows);

        if (level_num_rank_nodes == 0)
            continue;

        IntVector &level_col_index = bsr_data.rank_col_index[level];
        IntVector &level_row_ptr = bsr_data.rank_row_ptr[level];

        // First build up the row pointers
        int u_index_offset = u_basis_tree.getLevelStart(level + u_start_level);
        int v_index_offset = v_basis_tree.getLevelStart(level + v_start_level);
        std::fill(workspace.begin(), workspace.begin() + level_num_rows + 1, 0);

        for (int node = 0; node < level_num_rank_nodes; node++)
        {
            int tree_index = rank_leaf_tree_index[level_start + node];
            int row_index = node_u_index[tree_index] - u_index_offset;
            workspace[row_index]++;
        }

        // Now do a scan on the row counts to generate the row pointers
        thrust::exclusive_scan(workspace.begin(), workspace.begin() + level_num_rows + 1, level_row_ptr.begin(),
                               H2OPUS_BSR_INDEX_BASE);

        // Copy the scanned data back into workspace so we can use it to determine
        // the positions of the column indexes in the column pointer array
        thrust::copy(level_row_ptr.begin(), level_row_ptr.end(), &workspace[0]);

        // Now we generate the column indexes per row and generate an index
        // transformation map that will let us reassign tree indexes
        for (int node = 0; node < level_num_rank_nodes; node++)
        {
            int tree_index = rank_leaf_tree_index[level_start + node];
            int row_index = node_u_index[tree_index] - u_index_offset;
            int col_index = node_v_index[tree_index] - v_index_offset;

            coupling_index_map[level_start + node] = level_start + workspace[row_index] - H2OPUS_BSR_INDEX_BASE;
            level_col_index[workspace[row_index]++ - H2OPUS_BSR_INDEX_BASE] = col_index + H2OPUS_BSR_INDEX_BASE;
        }
        /*printf("Level %d\n", level);
        printf("Row ptrs: \n");
        for(int i = 0; i < level_row_ptr.size(); i++)
            printf("%d ", level_row_ptr[i]);
        printf("\nCol Index:\n");
        for(int i = 0; i < level_col_index.size(); i++)
            printf("%d ", level_col_index[i]);
        printf("\n");*/
    }

    reorderLeafIndexes(dense_index_map, coupling_index_map);

    bsr_data.setLevelMaxRowBlocks();

    /*printf("Row ptrs: \n");
    for(int i = 0; i < bsr_data.dense_row_ptr.size(); i++)
        printf("%d ", bsr_data.dense_row_ptr[i]);
    printf("\nCol Index:\n");
    for(int i = 0; i < bsr_data.dense_col_index.size(); i++)
        printf("%d ", bsr_data.dense_col_index[i]);
    printf("\n");*/
}
