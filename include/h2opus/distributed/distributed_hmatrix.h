#ifndef __DISTRIBUTED_HMATRIX_H__
#define __DISTRIBUTED_HMATRIX_H__

#include <h2opus/core/hmatrix.h>
#include <h2opus/distributed/distributed_comm_buffer.h>
#include <h2opus/util/dynarray.h>

template <int hw> struct TDistributedCompressedBSNData
{
    typedef typename VectorContainer<hw, int>::type IntVector;

    // Node indexes that this branch expects to receive from other processes
    IntVector receive_process_nodes;
    std::vector<int> receive_process_node_ptrs;
    std::vector<int> receive_process_ids;

    // Node indexes that this branch will send to other processes
    IntVector send_process_nodes;
    std::vector<int> send_process_node_ptrs;
    std::vector<int> send_process_ids;

    TDistributedCompressedBSNData()
    {
    }

    TDistributedCompressedBSNData(const TDistributedCompressedBSNData &h)
    {
        init(h);
    }

    TDistributedCompressedBSNData &operator=(const TDistributedCompressedBSNData &h)
    {
        init(h);
        return *this;
    }

    template <int other_hw> TDistributedCompressedBSNData(const TDistributedCompressedBSNData<other_hw> &h)
    {
        init(h);
    }

    template <int other_hw> TDistributedCompressedBSNData &operator=(const TDistributedCompressedBSNData<other_hw> &h)
    {
        init(h);
        return *this;
    }

    // Return the memory used in GB
    H2Opus_Real getMemoryUsage()
    {
        H2Opus_Real lsize = 0, gb = 1024 * 1024;

        lsize += (receive_process_nodes.size() / sizeof(int)) / gb;
        lsize += (receive_process_node_ptrs.size() / sizeof(int)) / gb;
        lsize += (receive_process_ids.size() / sizeof(int)) / gb;
        lsize += (send_process_nodes.size() / sizeof(int)) / gb;
        lsize += (send_process_node_ptrs.size() / sizeof(int)) / gb;
        lsize += (send_process_ids.size() / sizeof(int)) / gb;

        return lsize;
    }

  private:
    template <int other_hw> void init(const TDistributedCompressedBSNData<other_hw> &h)
    {
        copyVector(this->receive_process_nodes, h.receive_process_nodes);
        copyVector(this->send_process_nodes, h.send_process_nodes);

        this->receive_process_node_ptrs = h.receive_process_node_ptrs;
        this->receive_process_ids = h.receive_process_ids;
        this->send_process_node_ptrs = h.send_process_node_ptrs;
        this->send_process_ids = h.send_process_ids;
    }
};

template <int hw> struct TDistributedCompresseBasisTree
{
    typedef typename VectorContainer<hw, int>::type IntVector;

    // The vnode end indices per level of the branch
    std::vector<int> global_vnode_end_index;

    DynamicArray<TDistributedCompressedBSNData<hw>> coupling_compressed_bsn_data;
    // std::vector<TDistributedSendRecvBuffer<hw>> coupling_comm_buffers;

    // Since dense data is non-uniform we need additional data for each node and process
    // If the low rank nodes ever become non-uniform this could be included in the
    // compressed BSN data structure
    TDistributedCompressedBSNData<hw> dense_compressed_bsn_data;
    // TDistributedSendRecvBuffer<hw> dense_comm_buffer;

    // Transfer buffers for the root branch in the root process (for upsweeps and downsweeps)
    // TDistributedTransferBuffer<hw> gather_buffer, scatter_buffer;

    // Node offsets to indicate where each node will fit in the buffer
    IntVector dense_send_node_offsets, dense_receive_node_offsets;

    // Save how much data needs to be sent per dof for each process
    std::vector<int> dense_send_sizes, dense_receive_sizes;
    int dense_send_total_sum, dense_receive_total_sum;

    TDistributedCompresseBasisTree()
    {
    }

    TDistributedCompresseBasisTree(const TDistributedCompresseBasisTree &h)
    {
        init(h);
    }

    TDistributedCompresseBasisTree &operator=(const TDistributedCompresseBasisTree &h)
    {
        init(h);
        return *this;
    }

    template <int other_hw> TDistributedCompresseBasisTree(const TDistributedCompresseBasisTree<other_hw> &h)
    {
        init(h);
    }

    template <int other_hw> TDistributedCompresseBasisTree &operator=(const TDistributedCompresseBasisTree<other_hw> &h)
    {
        init(h);
        return *this;
    }

    H2Opus_Real getLocalDenseMemoryUsage()
    {
        H2Opus_Real lsize = 0, gb = 1024 * 1024;

        lsize += dense_compressed_bsn_data.getMemoryUsage();
        lsize += (dense_send_node_offsets.size() / sizeof(int)) / gb;
        lsize += (dense_receive_node_offsets.size() / sizeof(int)) / gb;

        return lsize;
    }

    H2Opus_Real getLocalLowRankMemoryUsage()
    {
        H2Opus_Real lsize = 0;

        for (size_t i = 0; i < coupling_compressed_bsn_data.size(); i++)
            lsize += coupling_compressed_bsn_data[i].getMemoryUsage();

        return lsize;
    }

    H2Opus_Real getLocalMemoryUsage()
    {
        return getLocalLowRankMemoryUsage() + getLocalDenseMemoryUsage();
    }

  private:
    template <int other_hw> void init(const TDistributedCompresseBasisTree<other_hw> &h)
    {
        copyVector(this->dense_send_node_offsets, h.dense_send_node_offsets);
        copyVector(this->dense_receive_node_offsets, h.dense_receive_node_offsets);

        this->global_vnode_end_index = h.global_vnode_end_index;
        this->dense_compressed_bsn_data = h.dense_compressed_bsn_data;
        this->dense_send_sizes = h.dense_send_sizes;
        this->dense_receive_sizes = h.dense_receive_sizes;

        this->dense_send_total_sum = h.dense_send_total_sum;
        this->dense_receive_total_sum = h.dense_receive_total_sum;

        this->coupling_compressed_bsn_data.resize(h.coupling_compressed_bsn_data.size());
        for (size_t i = 0; i < coupling_compressed_bsn_data.size(); i++)
            this->coupling_compressed_bsn_data[i] = h.coupling_compressed_bsn_data[i];

        // Only allocate the buffers array - the contents will be set in the calling
        // distributed h2opus function
        // this->coupling_comm_buffers.resize(h.coupling_comm_buffers.size());
    }
};

template <int hw> struct TDistributedBasisTree
{
    // Only populated for the master process
    TBasisTree<hw> top_level;
    TBasisTree<hw> basis_branch;

    TDistributedBasisTree()
    {
    }

    TDistributedBasisTree(const TDistributedBasisTree &h)
    {
        init(h);
    }

    TDistributedBasisTree &operator=(const TDistributedBasisTree &h)
    {
        init(h);
        return *this;
    }

    template <int other_hw> TDistributedBasisTree(const TDistributedBasisTree<other_hw> &h)
    {
        init(h);
    }

    template <int other_hw> TDistributedBasisTree &operator=(const TDistributedBasisTree<other_hw> &h)
    {
        init(h);
        return *this;
    }

    // Return the memory used by the basis tree in GB
    H2Opus_Real getLocalMemoryUsage()
    {
        return top_level.getMemoryUsage() + basis_branch.getMemoryUsage();
    }

  private:
    template <int other_hw> void init(const TDistributedBasisTree<other_hw> &h)
    {
        this->top_level = h.top_level;
        this->basis_branch = h.basis_branch;
    }
};

template <int hw> struct TDistributedHNodeTree
{
    typedef typename VectorContainer<hw, int>::type IntVector;

    // Only populated for the master process
    THNodeTree<hw> top_level;

    THNodeTree<hw> diagonal_block, off_diagonal_blocks;

    // These are the remapped v-indexes of the nodes in the off diagonal blocks
    // corresponding to the compressed basis. Inner nodes will have empty nodes
    // as their index and all node indices are local to their level
    IntVector compressed_v_index;

    TDistributedHNodeTree()
    {
    }

    TDistributedHNodeTree(const TDistributedHNodeTree &h)
    {
        init(h);
    }

    TDistributedHNodeTree &operator=(const TDistributedHNodeTree &h)
    {
        init(h);
        return *this;
    }

    template <int other_hw> TDistributedHNodeTree(const TDistributedHNodeTree<other_hw> &h)
    {
        init(h);
    }

    template <int other_hw> TDistributedHNodeTree &operator=(const TDistributedHNodeTree<other_hw> &h)
    {
        init(h);
        return *this;
    }

    H2Opus_Real getLocalMemoryUsage()
    {
        return top_level.getMemoryUsage() + diagonal_block.getMemoryUsage() + off_diagonal_blocks.getMemoryUsage();
    }

    H2Opus_Real getLocalDenseMemoryUsage()
    {
        return top_level.getDenseMemoryUsage() + diagonal_block.getDenseMemoryUsage() +
               off_diagonal_blocks.getDenseMemoryUsage();
    }

    H2Opus_Real getLocalLowRankMemoryUsage()
    {
        return top_level.getLowRankMemoryUsage() + diagonal_block.getLowRankMemoryUsage() +
               off_diagonal_blocks.getLowRankMemoryUsage();
    }

  private:
    template <int other_hw> void init(const TDistributedHNodeTree<other_hw> &h)
    {
        this->top_level = h.top_level;
        this->diagonal_block = h.diagonal_block;
        this->off_diagonal_blocks = h.off_diagonal_blocks;

        copyVector(this->compressed_v_index, h.compressed_v_index);
    }
};

template <int hw> struct TDistributedHMatrix
{
  public:
    // Assuming a symmetric matrix for now (i.e only one basis tree)
    TDistributedHNodeTree<hw> hnodes;
    TDistributedBasisTree<hw> basis_tree;
    int n;

    // Basis tree node data compressed according to what is needed for various operations
    // that process the off-diagonal portion of the hnode tree
    TDistributedCompresseBasisTree<hw> compressed_basis_tree_data;

    TDistributedHMatrix()
    {
    }

    TDistributedHMatrix(int n)
    {
        this->n = n;
    }

    TDistributedHMatrix(const TDistributedHMatrix &h)
    {
        init(h);
    }

    TDistributedHMatrix &operator=(const TDistributedHMatrix &h)
    {
        init(h);
        return *this;
    }

    template <int other_hw> TDistributedHMatrix(const TDistributedHMatrix<other_hw> &h)
    {
        init(h);
    }

    template <int other_hw> TDistributedHMatrix &operator=(const TDistributedHMatrix<other_hw> &h)
    {
        init(h);
        return *this;
    }

    // Get memory consumption in GB
    H2Opus_Real getLocalMemoryUsage()
    {
        return basis_tree.getLocalMemoryUsage() + compressed_basis_tree_data.getLocalMemoryUsage() +
               hnodes.getLocalMemoryUsage();
    }

    H2Opus_Real getLocalDenseMemoryUsage()
    {
        return hnodes.getLocalDenseMemoryUsage() + compressed_basis_tree_data.getLocalDenseMemoryUsage();
    }

    H2Opus_Real getLocalLowRankMemoryUsage()
    {
        return basis_tree.getLocalMemoryUsage() + compressed_basis_tree_data.getLocalLowRankMemoryUsage() +
               hnodes.getLocalLowRankMemoryUsage();
    }

  private:
    template <int other_hw> void init(const TDistributedHMatrix<other_hw> &h)
    {
        this->n = h.n;
        this->hnodes = h.hnodes;
        this->basis_tree = h.basis_tree;
        this->compressed_basis_tree_data = h.compressed_basis_tree_data;
    }
};

typedef TDistributedHMatrix<H2OPUS_HWTYPE_CPU> DistributedHMatrix;
typedef TDistributedBasisTree<H2OPUS_HWTYPE_CPU> DistributedBasisTree;
typedef TDistributedHNodeTree<H2OPUS_HWTYPE_CPU> DistributedHNodeTree;
typedef TDistributedCompresseBasisTree<H2OPUS_HWTYPE_CPU> DistributedCompresseBasisTree;
typedef TDistributedCompressedBSNData<H2OPUS_HWTYPE_CPU> DistributedCompressedBSNData;

#ifdef H2OPUS_USE_GPU
typedef TDistributedHMatrix<H2OPUS_HWTYPE_GPU> DistributedHMatrix_GPU;
typedef TDistributedBasisTree<H2OPUS_HWTYPE_GPU> DistributedBasisTree_GPU;
typedef TDistributedHNodeTree<H2OPUS_HWTYPE_GPU> DistributedHNodeTree_GPU;
typedef TDistributedCompresseBasisTree<H2OPUS_HWTYPE_GPU> DistributedCompresseBasisTree_GPU;
typedef TDistributedCompressedBSNData<H2OPUS_HWTYPE_GPU> DistributedCompressedBSNData_GPU;
#endif

#endif
