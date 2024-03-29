#ifndef __H2OPUS_KD_TREE_H__
#define __H2OPUS_KD_TREE_H__

#include <h2opus/core/h2opus_defs.h>
#include <math.h>

template <class T> class H2OpusDataSet
{
  public:
    virtual int getDimension() const = 0;
    virtual size_t getDataSetSize() const = 0;
    virtual T getDataPoint(size_t index, int dimension_index) const = 0;
    virtual ~H2OpusDataSet(){};
};

template <class T, int hw> struct TH2OpusKDTree
{
  private:
    typedef typename VectorContainer<hw, int>::type IntVector;
    typedef typename VectorContainer<hw, T>::type DataVector;
    typedef typename VectorContainer<hw, DataVector>::type DataVectorArray;

    // Dataset pointer
    H2OpusDataSet<T> *data_set;

    // Data per dataset entry
    IntVector index_map;

    // Data per node
    IntVector head, parent, next;
    IntVector node_index_left, node_index_right;
    DataVectorArray bounding_box_low, bounding_box_high;

    std::vector<int> level_ptrs;
    size_t leaf_size, data_set_size;
    int dimension, depth;

    void allocateNodes(size_t num_nodes)
    {
        head.resize(num_nodes, H2OPUS_EMPTY_NODE);
        parent.resize(num_nodes, H2OPUS_EMPTY_NODE);
        next.resize(num_nodes, H2OPUS_EMPTY_NODE);

        node_index_left.resize(num_nodes);
        node_index_right.resize(num_nodes);

        for (int i = 0; i < dimension; i++)
        {
            bounding_box_low[i].resize(num_nodes);
            bounding_box_high[i].resize(num_nodes);
        }
    }

    // Split along the dimension that has the largest span
    int getSplitDimension(DataVector &bbox_low, DataVector &bbox_high)
    {
        T split_span = bbox_high[0] - bbox_low[0];
        int split_dim = 0;

        for (int i = 1; i < dimension; i++)
        {
            T span = bbox_high[i] - bbox_low[i];
            if (span > split_span)
            {
                split_span = span;
                split_dim = i;
            }
        }

        return split_dim;
    }

    int getSplitDimension(int node_index)
    {
        T max_span = bounding_box_high[0][node_index] - bounding_box_low[0][node_index];
        int split_dim = 0;

        for (int i = 1; i < dimension; i++)
        {
            T span = bounding_box_high[i][node_index] - bounding_box_low[i][node_index];
            if (span > max_span)
            {
                max_span = span;
                split_dim = i;
            }
        }
        return split_dim;
    }

    // Assumes left and right have been set before this call
    void computBoundingBox(int node_index)
    {
        int left = node_index_left[node_index];
        int right = node_index_right[node_index];

        DataVector bbox_low(dimension), bbox_high(dimension);
        computBoundingBox(left, right, bbox_low, bbox_high);
        setBoundingBox(node_index, bbox_low, bbox_high);
    }

    void computBoundingBox(int left, int right, DataVector &bbox_low, DataVector &bbox_high)
    {
        assert(left <= right);

        for (int i = 0; i < dimension; i++)
        {
            int k = left;
            T bbox_low_dim, bbox_high_dim;

            T data_set_val = data_set->getDataPoint(index_map[k], i);
            bbox_low_dim = bbox_high_dim = data_set_val;

            for (k = left + 1; k < right; k++)
            {
                data_set_val = data_set->getDataPoint(index_map[k], i);
                bbox_low_dim = std::min(bbox_low_dim, data_set_val);
                bbox_high_dim = std::max(bbox_high_dim, data_set_val);
            }

            bbox_low[i] = bbox_low_dim;
            bbox_high[i] = bbox_high_dim;
        }
    }

    void setBoundingBox(int node_index, DataVector &bbox_low, DataVector &bbox_high)
    {
        for (int i = 0; i < dimension; i++)
        {
            bounding_box_low[i][node_index] = bbox_low[i];
            bounding_box_high[i][node_index] = bbox_high[i];
        }
    }

    struct KDTreeNode
    {
        KDTreeNode(int dim)
        {
            bbox_low.resize(dim);
            bbox_high.resize(dim);
            left_child = right_child = parent = NULL;
        }
        int left, right, level, level_index;
        KDTreeNode *left_child, *right_child, *parent;
        DataVector bbox_low, bbox_high;
    };

    int upper_power_of_two(int v)
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // Median Split
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    void medianDivideAllocate()
    {
        size_t num_points = index_map.size();
        int tree_levels = ceil(log2((double)num_points / leaf_size)) + 1;
        if (tree_levels < 0)
        {
            printf("Leaf size %ld was too large for a data set of size %ld\n", leaf_size, num_points);
            exit(-1);
        }
        int total_nodes = (1 << tree_levels) - 1;

        allocateNodes(total_nodes);

        int node_counter = 0;
        level_ptrs.resize(tree_levels + 1);
        for (int i = 0; i < tree_levels; i++)
        {
            level_ptrs[i] = node_counter;
            node_counter += (1 << i);
        }
        level_ptrs[tree_levels] = node_counter;
        depth = tree_levels;
    }

    // Median sort needs a sort
    struct MedianSplitCompare
    {
        const H2OpusDataSet<T> *data_set;
        int component;

        MedianSplitCompare(const H2OpusDataSet<T> *inputData, int comp) : data_set(inputData), component(comp){};
        bool operator()(int i, int j)
        {
            return data_set->getDataPoint(i, component) < data_set->getDataPoint(j, component);
        }
    };

    void medianDivide(int left, int right, int parent_node, int level, std::vector<int> &node_tails,
                      std::vector<int> &level_nodes_alloc)
    {
        int node_index = level_ptrs[level] + level_nodes_alloc[level];
        level_nodes_alloc[level]++;
        node_index_left[node_index] = left;
        node_index_right[node_index] = right;

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

        computBoundingBox(node_index);

        // Is the subset small enough to make it a leaf?
        if (level != depth - 1) // right - left > leaf_size)
        {
            // Figure out split dimension. In this case we use the dimension with the largest span
            // (i.e. we split along the longest edge of the bounding box)
            int split_dim = getSplitDimension(node_index);

            int *index_map_start = vec_ptr(index_map) + left;
            int count = right - left;
            int split_index = count / 2;
            std::sort(index_map_start, index_map_start + count, MedianSplitCompare(data_set, split_dim));

            medianDivide(left, left + split_index, node_index, level + 1, node_tails, level_nodes_alloc);
            medianDivide(left + split_index, right, node_index, level + 1, node_tails, level_nodes_alloc);
        }
    }

    // Uniform split
    KDTreeNode *medianUniformSplit(int left, int right, KDTreeNode *parent, int level,
                                   std::vector<int> &level_nodes_alloc)
    {
        while ((size_t)level >= level_nodes_alloc.size())
            level_nodes_alloc.push_back(0);

        KDTreeNode *node = new KDTreeNode(dimension);
        node->level_index = level_nodes_alloc[level]++;
        node->level = level;
        node->left = left;
        node->right = right;
        node->parent = parent;

        computBoundingBox(left, right, node->bbox_low, node->bbox_high);

        if (right - left > (int)leaf_size)
        {
            // Find the largest dim of the bounding box to split along
            int split_dim = getSplitDimension(node->bbox_low, node->bbox_high);

            // Sort the points
            int *index_map_start = vec_ptr(index_map) + left;
            int count = right - left;
            std::sort(index_map_start, index_map_start + count, MedianSplitCompare(data_set, split_dim));

            // Split the point set into two such that one half is a multiple of the leaf size
            // and the other half is the remaining points
            int split_index = upper_power_of_two((int)((count + leaf_size - 1) / leaf_size)) / 2;
            split_index = split_index * leaf_size;

            node->left_child = medianUniformSplit(left, left + split_index, node, level + 1, level_nodes_alloc);
            node->right_child = medianUniformSplit(left + split_index, right, node, level + 1, level_nodes_alloc);
        }

        return node;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // Middle split
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    void freeKDTreeNode(KDTreeNode *node)
    {
        if (node == NULL)
            return;

        freeKDTreeNode(node->left_child);
        freeKDTreeNode(node->right_child);

        delete node;
    }

    void flattenKDTreeNode(KDTreeNode *node, int parent_node, int level, std::vector<int> &level_nodes_alloc,
                           std::vector<int> &node_tails)
    {
        if (!node)
            return;

        int node_index = level_ptrs[level] + level_nodes_alloc[level];
        level_nodes_alloc[level]++;
        node_index_left[node_index] = node->left;
        node_index_right[node_index] = node->right;
        setBoundingBox(node_index, node->bbox_low, node->bbox_high);

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

        flattenKDTreeNode(node->left_child, node_index, level + 1, level_nodes_alloc, node_tails);
        flattenKDTreeNode(node->right_child, node_index, level + 1, level_nodes_alloc, node_tails);
    }

    void flattenKDTree(KDTreeNode *root, std::vector<int> &level_nodes_alloc)
    {
        depth = (int)level_nodes_alloc.size();
        level_ptrs.resize(depth + 1, 0);

        for (int i = 0; i < depth; i++)
        {
            level_ptrs[i + 1] = level_ptrs[i] + level_nodes_alloc[i];
            level_nodes_alloc[i] = 0;
        }

        int total_nodes = level_ptrs[depth];
        allocateNodes(total_nodes);
        std::vector<int> node_tails(total_nodes, 0);
        flattenKDTreeNode(root, H2OPUS_EMPTY_NODE, 0, level_nodes_alloc, node_tails);
    }

    KDTreeNode *middleSplit(int left, int right, KDTreeNode *parent, int level, std::vector<int> &level_nodes_alloc)
    {
        while (level >= level_nodes_alloc.size())
            level_nodes_alloc.push_back(0);

        KDTreeNode *node = new KDTreeNode(dimension);
        node->level_index = level_nodes_alloc[level]++;
        node->level = level;
        node->left = left;
        node->right = right;
        node->parent = parent;

        computBoundingBox(left, right, node->bbox_low, node->bbox_high);

        if (right - left > leaf_size)
        {
            // Find the largest dim of the bounding box to split along
            int split_dim = getSplitDimension(node->bbox_low, node->bbox_high);

            // Split down the middle
            T split_val = (node->bbox_low[split_dim] + node->bbox_high[split_dim]) / 2;
            int *indices = vec_ptr(index_map) + left;
            int split_index = planeSplit(indices, right - left, split_dim, split_val);

            node->left_child = middleSplit(left, left + split_index, node, level + 1, level_nodes_alloc);
            node->right_child = middleSplit(left + split_index, right, node, level + 1, level_nodes_alloc);
        }

        return node;
    }

    int planeSplit(int *indices, int count, int split_dim, T split_val)
    {
        int left = 0, right = count - 1;

        while (left < right)
        {
            // Advance until the point at the left index is to the right of the split value
            while (left < count && data_set->getDataPoint(indices[left], split_dim) <= split_val)
                left++;
            // Retreat until the point at the right index is to the left of the split value
            while (right >= 0 && data_set->getDataPoint(indices[right], split_dim) > split_val)
                right--;
            if (left < right)
                std::swap(indices[left], indices[right]);
        }

        return left;
    }

  public:
    TH2OpusKDTree(H2OpusDataSet<T> *data_set, size_t leaf_size)
    {
        assert(data_set != NULL);

        this->data_set = data_set;
        this->data_set_size = data_set->getDataSetSize();
        this->index_map.resize(this->data_set_size);
        for (size_t i = 0; i < index_map.size(); i++)
            this->index_map[i] = i;

        this->leaf_size = leaf_size;
        this->dimension = data_set->getDimension();
        this->bounding_box_low.resize(this->dimension);
        this->bounding_box_high.resize(this->dimension);

        this->level_ptrs.resize(1, 0);
    }

    H2OpusDataSet<T> *getDataSet()
    {
        return data_set;
    }

    int getLeafSize()
    {
        return leaf_size;
    }

    int getDim()
    {
        return dimension;
    }

    int getDepth()
    {
        return depth;
    }

    // binary tree for now
    int getMaxChildren()
    {
        return 2;
    }

    void buildKDtreeMedianSplit()
    {
        medianDivideAllocate();

        std::vector<int> level_nodes_alloc(depth, 0);
        std::vector<int> node_tails(head.size(), 0);
        medianDivide(0, data_set_size, H2OPUS_EMPTY_NODE, 0, node_tails, level_nodes_alloc);
    }

    void buildKDtreeMedianUniformSplit()
    {
        std::vector<int> level_nodes_alloc;
        KDTreeNode *root = medianUniformSplit(0, data_set_size, NULL, 0, level_nodes_alloc);
        flattenKDTree(root, level_nodes_alloc);
        freeKDTreeNode(root);
    }

    void buildKDtreeMiddleSplit()
    {
        std::vector<int> level_nodes_alloc;
        KDTreeNode *root = middleSplit(0, data_set_size, NULL, 0, level_nodes_alloc);
        flattenKDTree(root, level_nodes_alloc);
        freeKDTreeNode(root);
    }

    void getBoundingBoxComponent(size_t node_index, int component, T &low, T &high)
    {
        low = bounding_box_low[component][node_index];
        high = bounding_box_high[component][node_index];
    }

    int getLevelSize(size_t level_index)
    {
        return level_ptrs[level_index + 1] - level_ptrs[level_index];
    }

    int getLevelNodeStartIndex(size_t level_index)
    {
        return level_ptrs[level_index];
    }

    int getLevelNodeEndIndex(size_t level_index)
    {
        return level_ptrs[level_index + 1];
    }

    int getNodeStartIndex(size_t node_index)
    {
        return node_index_left[node_index];
    }

    int getNodeEndIndex(size_t node_index)
    {
        return node_index_right[node_index];
    }

    void getNodeLimits(size_t node_index, int &start, int &end)
    {
        start = node_index_left[node_index];
        end = node_index_right[node_index];
    }

    int getNodeSize(size_t node_index)
    {
        return node_index_right[node_index] - node_index_left[node_index];
    }

    int *getIndexMap()
    {
        return vec_ptr(index_map);
    }

    int getHeadChild(int node)
    {
        return head[node];
    }

    int getNextChild(int node)
    {
        return next[node];
    }

    int getParent(int node)
    {
        return parent[node];
    }

    bool isLeaf(int node)
    {
        return head[node] == H2OPUS_EMPTY_NODE;
    }
};

template <class T, int hw> T h2opusBBoxDiam(TH2OpusKDTree<T, hw> *kdtree, int node_index)
{
    T bbox_low, bbox_high, diam = 0;
    int dim = kdtree->getDim();
    for (int i = 0; i < dim; i++)
    {
        kdtree->getBoundingBoxComponent(node_index, i, bbox_low, bbox_high);
        T dim_diff = bbox_high - bbox_low;
        diam += dim_diff * dim_diff;
    }
    return sqrt(diam);
}

template <class T, int hw> T h2opusBBoxDist(TH2OpusKDTree<T, hw> *kdtree, int node_index_a, int node_index_b)
{
    T bbox_a_low, bbox_a_high, bbox_b_low, bbox_b_high, dist = 0;
    int dim = kdtree->getDim();

    for (int i = 0; i < dim; i++)
    {
        kdtree->getBoundingBoxComponent(node_index_a, i, bbox_a_low, bbox_a_high);
        kdtree->getBoundingBoxComponent(node_index_b, i, bbox_b_low, bbox_b_high);

        if (bbox_a_high < bbox_b_low)
            dist += (bbox_b_low - bbox_a_high) * (bbox_b_low - bbox_a_high);
        else if (bbox_b_high < bbox_a_low)
            dist += (bbox_a_low - bbox_b_high) * (bbox_a_low - bbox_b_high);
    }
    return sqrt(dist);
}

template <class T, int hw> T h2opusBBoxCenterDist(TH2OpusKDTree<T, hw> *kdtree, int node_index_a, int node_index_b)
{
    T bbox_a_low, bbox_a_high, bbox_b_low, bbox_b_high, dist = 0;
    int dim = kdtree->getDim();

    for (int i = 0; i < dim; i++)
    {
        kdtree->getBoundingBoxComponent(node_index_a, i, bbox_a_low, bbox_a_high);
        kdtree->getBoundingBoxComponent(node_index_b, i, bbox_b_low, bbox_b_high);

        T center_diff = 0.5 * ((bbox_a_high + bbox_a_low) - (bbox_b_high + bbox_b_low));
        dist += center_diff * center_diff;
    }
    return sqrt(dist);
}

typedef TH2OpusKDTree<H2Opus_Real, H2OPUS_HWTYPE_CPU> H2OpusKDTree;

#endif
