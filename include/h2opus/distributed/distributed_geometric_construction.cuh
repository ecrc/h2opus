#include <h2opus/util/debug_routines.h>
#include <h2opus/util/geometric_admissibility.h>
#include <h2opus/util/geometric_construction.h>
#include <h2opus/util/kdtree.h>

#include <h2opus/distributed/distributed_hmatrix.h>

#include <map>
#include <set>

void getDistributedBranches(H2OpusKDTree &kdtree, int node_index, std::vector<int> &branch_root_nodes, int stop_level,
                            int current_level, int &current_node)
{
    if (current_level == stop_level)
        branch_root_nodes[current_node++] = node_index;
    else
    {
        int child = kdtree.getHeadChild(node_index);
        while (child != H2OPUS_EMPTY_NODE)
        {
            getDistributedBranches(kdtree, child, branch_root_nodes, stop_level, current_level + 1, current_node);
            child = kdtree.getNextChild(child);
        }
    }
}

void getDistributedBranches(H2OpusKDTree &kdtree, int node_index, std::vector<int> &branch_root_nodes)
{
    // Assuming a complete binary tree for now, but a more involved strategy will be needed for more
    // complex trees (probably one that tries to balance branch sizes)
    int num_branches = branch_root_nodes.size();
    int stop_level = (int)log2((float)num_branches);
    int nodes = 0;
    getDistributedBranches(kdtree, node_index, branch_root_nodes, stop_level, 0, nodes);
    assert(nodes == num_branches);
}

void buildDistributedBasisTreeStructure(DistributedBasisTree &basis_tree, H2OpusKDTree &kdtree, int slices,
                                        distributedH2OpusHandle_t handle)
{
    int num_branches = handle->num_ranks;
    int proc_rank = handle->rank;
    int max_leaf_size = kdtree.getLeafSize();
    int max_rank = pow(slices, kdtree.getDim());

    // Now construct the basis tree for the hmatrix
    int full_depth = kdtree.getDepth();
    int top_level_depth = (int)log2((float)num_branches) + 1;
    int branch_depth = full_depth - top_level_depth + 1;

    // Make sure the matrix is large enough to be distributed
    assert(branch_depth >= 1);

    std::vector<int> level_ranks(std::max(branch_depth, top_level_depth), max_rank);
    std::vector<int> branch_root_nodes(num_branches);

    getDistributedBranches(kdtree, 0, branch_root_nodes);

    // Get the top level of the tree for the master process (rank 0)
    // since we don't have any leaves, don't give it the index map
    if (proc_rank == 0)
    {
        basis_tree.top_level.generateStructureFromKDTree(kdtree, 0, false, top_level_depth);
        basis_tree.top_level.allocateMatrixData(&level_ranks[0], top_level_depth, max_leaf_size);
    }

    // Now set the branch for the local process
    basis_tree.basis_branch.generateStructureFromKDTree(kdtree, branch_root_nodes[proc_rank], true, branch_depth);
    basis_tree.basis_branch.allocateMatrixData(&level_ranks[0], branch_depth, max_leaf_size);
}

template <typename EntryGen>
void generateDistributedBasisTreeEntries(DistributedBasisTree &basis_tree, H2OpusKDTree &kdtree, EntryGen &entry_gen,
                                         int slices, distributedH2OpusHandle_t handle)
{
    int proc_rank = handle->rank;

    // Generate the top level entries first for the master process (rank 0)
    // shouldn't produce any basis leaves
    if (proc_rank == 0)
    {
        std::vector<int> top_slices(basis_tree.top_level.depth, slices);
        generateUBasisTreeEntries(basis_tree.top_level, kdtree, entry_gen, top_slices);
    }

    // Now generate the branches for the local process
    std::vector<int> branch_slices(basis_tree.basis_branch.depth, slices);
    generateUBasisTreeEntries(basis_tree.basis_branch, kdtree, entry_gen, branch_slices);
}

void getBranchVIndexes(std::vector<std::vector<int>> &branch_v_indexes, std::vector<int> &diagonal_v_indexes,
                       BasisTree &top_basis, HNodeTree &top_hnode, distributedH2OpusHandle_t handle)
{
    int top_level_depth = top_basis.depth;
    assert(top_level_depth > 1);

    // Get all the leaf inner nodes of the top level
    std::vector<int> inode_u_index, inode_v_index;
    top_hnode.extractLevelUVInodeIndexes(inode_u_index, inode_v_index, top_level_depth - 2);

    // The children of these inner nodes are then tested for admissibilty in the branches
    // so we gather all children nodes and group them by u_index (since we are decomposing
    // by rows)
    int branch_u_offset = top_basis.getLevelStart(top_level_depth - 1);
    for (size_t i = 0; i < inode_u_index.size(); i++)
    {
        int u_child = top_basis.head[inode_u_index[i]];
        while (u_child != H2OPUS_EMPTY_NODE)
        {
            int v_child = top_basis.head[inode_v_index[i]];
            while (v_child != H2OPUS_EMPTY_NODE)
            {
                if (u_child != v_child)
                    branch_v_indexes[u_child - branch_u_offset].push_back(v_child);
                else
                    diagonal_v_indexes[u_child - branch_u_offset] = v_child;
                v_child = top_basis.next[v_child];
            }
            u_child = top_basis.next[u_child];
        }
    }
}

void buildDistributedHNodeTreeStructure(H2OpusKDTree &kdtree, H2OpusAdmissibility &admissibility,
                                        DistributedHNodeTree &hnodes, DistributedBasisTree &basis_tree,
                                        BasisTree &complete_tree, int leaf_size, distributedH2OpusHandle_t handle)
{
    int num_branches = handle->num_ranks;
    int proc_rank = handle->rank;
    MPI_Comm comm = handle->comm;

    std::vector<int> branch_v_indexes_sizes;
    std::vector<int> branch_v_indexes_full;
    std::vector<int> branch_v_indexes_displ;
    std::vector<std::vector<int>> branch_v_indexes;
    std::vector<int> diagonal_v_indexes;
    std::vector<int> local_v_indexes;
    std::vector<int> diagonal_v_index(1);
    int usesend, num_indexes;

    // The basis tree has one extra level for the ghost level
    int top_level_depth = (int)log2((float)num_branches) + 1;

    // Generate the top level hnode structure for the master process (rank 0)
    if (proc_rank == 0)
    {
        std::vector<int> dummy_v(1, 0);
        hnodes.top_level.determineStructure(kdtree, admissibility, basis_tree.top_level, 0, basis_tree.top_level, 0,
                                            top_level_depth, dummy_v);

        hnodes.top_level.allocateMatrixData(basis_tree.top_level.level_data, 0, top_level_depth);
        hnodes.top_level.allocateBSRData(basis_tree.top_level, basis_tree.top_level, 0, 0);
        hnodes.top_level.allocateBSNData(basis_tree.top_level, basis_tree.top_level, 0, 0);

        branch_v_indexes.resize(num_branches);
        diagonal_v_indexes.resize(num_branches);
        branch_v_indexes_sizes.resize(num_branches);
        branch_v_indexes_displ.resize(num_branches);

        getBranchVIndexes(branch_v_indexes, diagonal_v_indexes, basis_tree.top_level, hnodes.top_level, handle);
        size_t cum = 0;
        int icum = 0;
        for (int i = 0; i < num_branches; i++)
        {
            size_t s = branch_v_indexes[i].size();
            branch_v_indexes_sizes[i] = s;
            branch_v_indexes_displ[i] = icum;
            icum += s;
            cum += s;
        }
        // check for overflow
        usesend = (cum != (size_t)icum ? 1 : 0);
        // usesend = 1;
        mpiErrchk(MPI_Bcast(&usesend, 1, MPI_INT, 0, comm));
        if (!usesend)
        {
            branch_v_indexes_full.reserve(cum);
            for (auto const &items : branch_v_indexes)
            {
                branch_v_indexes_full.insert(std::end(branch_v_indexes_full), std::begin(items), std::end(items));
            }
            branch_v_indexes.clear();
        }
        else
        {
            branch_v_indexes_displ.clear();
        }
    }
    else
    {
        mpiErrchk(MPI_Bcast(&usesend, 1, MPI_INT, 0, comm));
    }

    mpiErrchk(MPI_Scatter(vec_ptr(branch_v_indexes_sizes), 1, MPI_INT, &num_indexes, 1, MPI_INT, 0, comm));
    local_v_indexes.resize(num_indexes);

    if (usesend)
    {
        int mpitag = 917539;
        if (proc_rank == 0)
        {
            local_v_indexes = branch_v_indexes[0];
            for (int i = 1; i < num_branches; i++)
            {
                mpiErrchk(MPI_Send(vec_ptr(branch_v_indexes[i]), branch_v_indexes[i].size(), MPI_INT, i, mpitag, comm));
            }
        }
        else
        {
            mpiErrchk(MPI_Recv(vec_ptr(local_v_indexes), num_indexes, MPI_INT, 0, mpitag, comm, MPI_STATUS_IGNORE));
        }
    }
    else
    {
        mpiErrchk(MPI_Scatterv(vec_ptr(branch_v_indexes_full), vec_ptr(branch_v_indexes_sizes),
                               vec_ptr(branch_v_indexes_displ), MPI_INT, vec_ptr(local_v_indexes), num_indexes, MPI_INT,
                               0, comm));
    }
    // mpiErrchk( MPI_Scatter(vec_ptr(diagonal_v_indexes), 1, MPI_INT, vec_ptr(diagonal_v_index), 1, MPI_INT, 0, comm)
    // );

    // Generate the hnode tree diagonal block for the local process
    hnodes.diagonal_block.determineStructure(kdtree, admissibility, basis_tree.basis_branch);

    hnodes.diagonal_block.allocateMatrixData(basis_tree.basis_branch.level_data);
    hnodes.diagonal_block.allocateBSRData(basis_tree.basis_branch);
    hnodes.diagonal_block.allocateBSNData(basis_tree.basis_branch);

    // Generate the hnode tree for the block row excluding the diagonal block for the local process
    hnodes.off_diagonal_blocks.determineStructure(kdtree, admissibility, basis_tree.basis_branch, 0, complete_tree,
                                                  top_level_depth - 1, basis_tree.basis_branch.depth, local_v_indexes);

    hnodes.off_diagonal_blocks.allocateMatrixData(basis_tree.basis_branch.level_data);
    hnodes.off_diagonal_blocks.allocateBSRData(basis_tree.basis_branch, complete_tree, 0, top_level_depth - 1);
    hnodes.off_diagonal_blocks.allocateBSNData(basis_tree.basis_branch, complete_tree, 0, top_level_depth - 1);
}

template <typename EntryGen>
void generateDistributedHNodeTreeEntries(EntryGen &entry_gen, DistributedHNodeTree &hnodes,
                                         DistributedBasisTree &basis_tree, H2OpusKDTree &kdtree,
                                         BasisTree &complete_tree, int slices, distributedH2OpusHandle_t handle)
{
    int proc_rank = handle->rank;

    // Generate the hnode structure for the root subtree for the master process (rank 0)
    if (proc_rank == 0)
    {
        std::vector<int> top_slices(basis_tree.top_level.depth, slices);

        generateHNodeEntries(hnodes.top_level, kdtree, basis_tree.top_level, kdtree, basis_tree.top_level, entry_gen,
                             top_slices);
    }

    // Generate the hnode structure for the branch on the local process
    std::vector<int> branch_slices(basis_tree.basis_branch.depth, slices);

    generateHNodeEntries(hnodes.diagonal_block, kdtree, basis_tree.basis_branch, kdtree, basis_tree.basis_branch,
                         entry_gen, branch_slices);

    generateHNodeEntries(hnodes.off_diagonal_blocks, kdtree, basis_tree.basis_branch, kdtree, complete_tree, entry_gen,
                         branch_slices);
}

void generateCompressedBSNData(DistributedCompressedBSNData &cbsn_data, std::map<int, int> &unique_v_indexes,
                               std::vector<int> &branch_limits, std::vector<int> &comm_receive_node_counts,
                               std::vector<int> &comm_send_node_counts, distributedH2OpusHandle_t handle)
{
    int num_branches = handle->num_ranks;
    MPI_Comm comm = handle->comm;

    // Allocate memory for the node receive lists
    cbsn_data.receive_process_nodes.resize(unique_v_indexes.size());

    // Figure out which index is needed from which process by comparing against the branch limits
    int current_process = 0;
    int current_node = 0;

    for (auto it = unique_v_indexes.begin(); it != unique_v_indexes.end(); it++)
    {
        int v_index = it->first;
        it->second = current_node;

        // Check if we need to advance the process window
        while (current_process < num_branches - 1 && v_index >= branch_limits[current_process + 1])
            current_process++;

        // If this process hasn't been added before, append it to the list of
        // processes and add to the list of node pointers for the process
        if (cbsn_data.receive_process_ids.size() == 0 || cbsn_data.receive_process_ids.back() != current_process)
        {
            cbsn_data.receive_process_node_ptrs.push_back(current_node);
            cbsn_data.receive_process_ids.push_back(current_process);
        }
        comm_receive_node_counts[current_process]++;

        // Add the node to the list of expected received nodes
        cbsn_data.receive_process_nodes[current_node++] = v_index - branch_limits[current_process];
    }
    cbsn_data.receive_process_node_ptrs.push_back(current_node);

    // Do an all to all to get the required node counts from each process
    mpiErrchk(
        MPI_Alltoall(vec_ptr(comm_receive_node_counts), 1, MPI_INT, vec_ptr(comm_send_node_counts), 1, MPI_INT, comm));

    // Figure out the node counts that have to be sent and the processor ids
    // that will be sent to
    int node_counts = 0;
    cbsn_data.send_process_node_ptrs.push_back(node_counts);
    for (int i = 0; i < num_branches; i++)
    {
        if (comm_send_node_counts[i] != 0)
        {
            node_counts += comm_send_node_counts[i];
            cbsn_data.send_process_node_ptrs.push_back(node_counts);
            cbsn_data.send_process_ids.push_back(i);
        }
    }
    cbsn_data.send_process_nodes.resize(node_counts);

    std::vector<MPI_Request> requests(cbsn_data.receive_process_ids.size() + cbsn_data.send_process_ids.size());
    // Grab the nodes that this process should send out
    // We use sends and recvs instead of alltoall since theoretically
    // there should be very few processes that need to interact with each
    // other in the off-diagonal
    int mpitag = handle->getNewTag();
    for (size_t i = 0; i < cbsn_data.receive_process_ids.size(); i++)
    {
        // This process expects to receive the nodes listed in
        // cbsn_data.receive_process_nodes[node_start:node_end] from process[i] so
        // we send that list to process[i] using an Isend
        int process_id = cbsn_data.receive_process_ids[i];
        int node_start = cbsn_data.receive_process_node_ptrs[i];
        int node_end = cbsn_data.receive_process_node_ptrs[i + 1];

        mpiErrchk(MPI_Isend(vec_ptr(cbsn_data.receive_process_nodes) + node_start, node_end - node_start, MPI_INT,
                            process_id, mpitag, comm, &requests[i]));
    }

    int request_index = cbsn_data.receive_process_ids.size();
    for (size_t i = 0; i < cbsn_data.send_process_ids.size(); i++)
    {
        // This process will send the nodes listed in
        // cbsn_data.send_process_nodes[node_start:node_end] to process[i] so
        // we receive that list from process[i] using an Irecv
        int process_id = cbsn_data.send_process_ids[i];
        int node_start = cbsn_data.send_process_node_ptrs[i];
        int node_end = cbsn_data.send_process_node_ptrs[i + 1];

        mpiErrchk(MPI_Irecv(vec_ptr(cbsn_data.send_process_nodes) + node_start, node_end - node_start, MPI_INT,
                            process_id, mpitag, comm, &requests[i + request_index]));
    }

    // Wait for all sends and recvs to end
    mpiErrchk(MPI_Waitall(requests.size(), vec_ptr(requests), MPI_STATUSES_IGNORE));
}

void generateCompressedDenseBSNData(DistributedBasisTree &basis_tree, DistributedCompresseBasisTree &compressed_tree,
                                    distributedH2OpusHandle_t handle)
{
    MPI_Comm comm = handle->comm;
    int mpitag;

    // Determine the dense node compressed offsets
    auto &dense_send_node_offsets = compressed_tree.dense_send_node_offsets;
    auto &dense_receive_node_offsets = compressed_tree.dense_receive_node_offsets;
    auto &branch_level_data = basis_tree.basis_branch.level_data;
    auto &cbsn_data = compressed_tree.dense_compressed_bsn_data;

    dense_send_node_offsets.resize(cbsn_data.send_process_nodes.size() + 1);
    compressed_tree.dense_send_sizes.resize(cbsn_data.send_process_ids.size(), 0);
    dense_receive_node_offsets.resize(cbsn_data.receive_process_nodes.size() + 1);
    compressed_tree.dense_receive_sizes.resize(cbsn_data.receive_process_nodes.size(), 0);

    int dense_vindex_start = branch_level_data.getLevelStart(branch_level_data.depth - 1);

    compressed_tree.dense_send_total_sum = 0;
    compressed_tree.dense_receive_total_sum = 0;

    ////////////////////////////////////////////////////////////////////////////////
    // Local Offset computation
    ////////////////////////////////////////////////////////////////////////////////
    for (size_t i = 0; i < cbsn_data.send_process_ids.size(); i++)
    {
        int node_start = cbsn_data.send_process_node_ptrs[i];
        int node_end = cbsn_data.send_process_node_ptrs[i + 1];

        // These are local offsets within the subset of the buffer
        // that is being sent to each process
        int node_offset = 0;
        for (int node = node_start; node < node_end; node++)
        {
            dense_send_node_offsets[node] = node_offset;

            int vindex = dense_vindex_start + cbsn_data.send_process_nodes[node];
            node_offset += basis_tree.basis_branch.node_len[vindex];
        }
        compressed_tree.dense_send_sizes[i] = node_offset;
        compressed_tree.dense_send_total_sum += node_offset;
    }

    // Send the necessary offsets to the necessary processes
    std::vector<MPI_Request> requests(cbsn_data.receive_process_ids.size() + cbsn_data.send_process_ids.size());

    mpitag = handle->getNewTag();

    for (size_t i = 0; i < cbsn_data.send_process_ids.size(); i++)
    {
        // This process will send the offsets listed in
        // dense_send_node_offsets[node_start:node_end] to process[i]
        int process_id = cbsn_data.send_process_ids[i];
        int node_start = cbsn_data.send_process_node_ptrs[i];
        int node_end = cbsn_data.send_process_node_ptrs[i + 1];

        mpiErrchk(MPI_Isend(vec_ptr(dense_send_node_offsets) + node_start, node_end - node_start, MPI_INT, process_id,
                            mpitag, comm, &requests[i]));
    }

    int request_index = cbsn_data.send_process_ids.size();
    for (size_t i = 0; i < cbsn_data.receive_process_ids.size(); i++)
    {
        // This process will receive the offsets listed in
        // dense_receive_node_offsets[node_start:node_end] to process[i]
        int process_id = cbsn_data.receive_process_ids[i];
        int node_start = cbsn_data.receive_process_node_ptrs[i];
        int node_end = cbsn_data.receive_process_node_ptrs[i + 1];

        mpiErrchk(MPI_Irecv(vec_ptr(dense_receive_node_offsets) + node_start, node_end - node_start, MPI_INT,
                            process_id, mpitag, comm, &requests[i + request_index]));
    }

    // Wait for all sends and recvs to end
    mpiErrchk(MPI_Waitall(requests.size(), vec_ptr(requests), MPI_STATUSES_IGNORE));

    ////////////////////////////////////////////////////////////////////////////////
    // Share process subset sizes
    ////////////////////////////////////////////////////////////////////////////////
    mpitag = handle->getNewTag();

    for (size_t i = 0; i < cbsn_data.send_process_ids.size(); i++)
    {
        int process_id = cbsn_data.send_process_ids[i];

        // Send the total sizes
        mpiErrchk(
            MPI_Isend(&(compressed_tree.dense_send_sizes[i]), 1, MPI_INT, process_id, mpitag, comm, &requests[i]));
    }

    for (size_t i = 0; i < cbsn_data.receive_process_ids.size(); i++)
    {
        int process_id = cbsn_data.receive_process_ids[i];

        // Receive the total sizes
        mpiErrchk(MPI_Irecv(&(compressed_tree.dense_receive_sizes[i]), 1, MPI_INT, process_id, mpitag, comm,
                            &requests[i + request_index]));
    }

    // Wait for all sends and recvs to end
    mpiErrchk(MPI_Waitall(requests.size(), vec_ptr(requests), MPI_STATUSES_IGNORE));

    // Add up all the received sizes
    for (size_t i = 0; i < cbsn_data.receive_process_ids.size(); i++)
        compressed_tree.dense_receive_total_sum += compressed_tree.dense_receive_sizes[i];

    ////////////////////////////////////////////////////////////////////////////////
    // Update the local offsets to the global ones for both sends and recvs
    ////////////////////////////////////////////////////////////////////////////////
    int offset_sum = 0;
    for (size_t i = 1; i < cbsn_data.send_process_ids.size(); i++)
    {
        int node_start = cbsn_data.send_process_node_ptrs[i];
        int node_end = cbsn_data.send_process_node_ptrs[i + 1];

        // Accumulate the offset from the previous process subset
        // then add it to the local offsets
        offset_sum += compressed_tree.dense_send_sizes[i - 1];

        for (int node = node_start; node < node_end; node++)
            dense_send_node_offsets[node] += offset_sum;
    }
    dense_send_node_offsets.back() = compressed_tree.dense_send_total_sum;

    offset_sum = 0;
    for (size_t i = 1; i < cbsn_data.receive_process_ids.size(); i++)
    {
        int node_start = cbsn_data.receive_process_node_ptrs[i];
        int node_end = cbsn_data.receive_process_node_ptrs[i + 1];

        // Accumulate the offset from the previous process subset
        // then add it to the local offsets
        offset_sum += compressed_tree.dense_receive_sizes[i - 1];

        for (int node = node_start; node < node_end; node++)
            dense_receive_node_offsets[node] += offset_sum;
    }
    dense_receive_node_offsets.back() = compressed_tree.dense_receive_total_sum;
}

void generateCompressedBasisTreeStructure(DistributedHMatrix &hmatrix, BasisTree &complete_tree,
                                          distributedH2OpusHandle_t handle)
{
    int num_branches = handle->num_ranks;
    int proc_rank = handle->rank;
    MPI_Comm comm = handle->comm;

    // The basis tree has one extra level for the ghost level
    int top_level_depth = (int)log2((float)num_branches) + 1;

    DistributedBasisTree &basis_tree = hmatrix.basis_tree;
    DistributedHNodeTree &hnodes = hmatrix.hnodes;
    DistributedCompresseBasisTree &compressed_tree = hmatrix.compressed_basis_tree_data;

    compressed_tree.coupling_compressed_bsn_data.resize(hnodes.off_diagonal_blocks.depth);
    // compressed_tree.coupling_comm_buffers.resize(hnodes.off_diagonal_blocks.depth);
    hnodes.compressed_v_index.resize(hnodes.off_diagonal_blocks.num_nodes, H2OPUS_EMPTY_NODE);

    // Figure out the global index end for each level in the branch
    std::vector<int> &branch_global_vend_index = compressed_tree.global_vnode_end_index;
    branch_global_vend_index.resize(basis_tree.basis_branch.depth);
    for (int i = 0; i < basis_tree.basis_branch.depth; i++)
    {
        int level_size = basis_tree.basis_branch.level_data.getLevelSize(i);

        // Presumably all levels have the same size, so I can just consider the start
        // of process p as p * level_size (otherwise do allgather on level_size into level_sizes,
        // then do a prefix sum on level_sizes so that process p's global level offset is level_sizes[p])
        branch_global_vend_index[i] = proc_rank * level_size;

        // Add the complete tree level offset
        branch_global_vend_index[i] += complete_tree.level_data.getLevelStart(i + top_level_depth - 1);
    }

    std::vector<int> branch_limits(num_branches, 0);
    std::vector<int> comm_receive_node_counts(num_branches, 0);
    std::vector<int> comm_send_node_counts(num_branches, 0);

    for (int level = 0; level < hnodes.off_diagonal_blocks.depth; level++)
    {
        int level_start, level_end;
        hnodes.off_diagonal_blocks.getCouplingLevelRange(level, level_start, level_end);

        mpiErrchk(
            MPI_Allgather(&branch_global_vend_index[level], 1, MPI_INT, vec_ptr(branch_limits), 1, MPI_INT, comm));

        std::fill(comm_receive_node_counts.begin(), comm_receive_node_counts.end(), 0);
        std::map<int, int> unique_v_indexes;

        // Insert the unique v indexes of the block into a set
        for (int leaf = level_start; leaf < level_end; leaf++)
        {
            int tree_index = hnodes.off_diagonal_blocks.rank_leaf_tree_index[leaf];
            unique_v_indexes.insert({hnodes.off_diagonal_blocks.node_v_index[tree_index], 0});
        }

        generateCompressedBSNData(compressed_tree.coupling_compressed_bsn_data[level], unique_v_indexes, branch_limits,
                                  comm_receive_node_counts, comm_send_node_counts, handle);

        // Remap the coupling node v-indexes to the compressed indexes
        for (int leaf = level_start; leaf < level_end; leaf++)
        {
            int tree_index = hnodes.off_diagonal_blocks.rank_leaf_tree_index[leaf];
            int v_index = hnodes.off_diagonal_blocks.node_v_index[tree_index];
            hnodes.compressed_v_index[tree_index] = unique_v_indexes[v_index];
        }
    }

    // Now compress the offdiagonal dense block data
    std::map<int, int> unique_v_indexes;
    int num_dense_leaves = hnodes.off_diagonal_blocks.num_dense_leaves;

    mpiErrchk(MPI_Allgather(&branch_global_vend_index[basis_tree.basis_branch.depth - 1], 1, MPI_INT,
                            vec_ptr(branch_limits), 1, MPI_INT, comm));
    std::fill(comm_receive_node_counts.begin(), comm_receive_node_counts.end(), 0);

    for (int leaf = 0; leaf < num_dense_leaves; leaf++)
    {
        int tree_index = hnodes.off_diagonal_blocks.dense_leaf_tree_index[leaf];
        unique_v_indexes.insert({hnodes.off_diagonal_blocks.node_v_index[tree_index], 0});
    }
    generateCompressedBSNData(compressed_tree.dense_compressed_bsn_data, unique_v_indexes, branch_limits,
                              comm_receive_node_counts, comm_send_node_counts, handle);

    // Remap the dense node v-indexes to the compressed indexes
    for (int leaf = 0; leaf < num_dense_leaves; leaf++)
    {
        int tree_index = hnodes.off_diagonal_blocks.dense_leaf_tree_index[leaf];
        int v_index = hnodes.off_diagonal_blocks.node_v_index[tree_index];
        hnodes.compressed_v_index[tree_index] = unique_v_indexes[v_index];
    }

    generateCompressedDenseBSNData(basis_tree, compressed_tree, handle);
}

template <typename EntryGen>
void buildDistributedHMatrix(DistributedHMatrix &hmatrix, H2OpusDataSet<H2Opus_Real> *data_set,
                             H2OpusAdmissibility &admissibility, EntryGen &entry_gen, int leaf_size, int slices,
                             distributedH2OpusHandle_t handle)
{
    if (!handle->active)
        return;
    // Make sure the leaf size makes sense
    hmatrix.n = data_set->getDataSetSize();
    if (leaf_size >= hmatrix.n)
        leaf_size = hmatrix.n;

    // First build a kd-tree for the point cloud
    H2OpusKDTree kdtree(data_set, leaf_size);
    kdtree.buildKDtreeMedianSplit();

    buildDistributedBasisTreeStructure(hmatrix.basis_tree, kdtree, slices, handle);
    generateDistributedBasisTreeEntries<EntryGen>(hmatrix.basis_tree, kdtree, entry_gen, slices, handle);

    // printf("Basis tree done on rank %d\n", handle->rank);

    // Now generate the complete tree structure for the hnodes (for now, since we're distributing by block rows)
    BasisTree complete_tree;
    complete_tree.generateStructureFromKDTree(kdtree, 0, true, kdtree.getDepth());

    // Using the distributed branches and the complete tree, we can construct the block rows
    buildDistributedHNodeTreeStructure(kdtree, admissibility, hmatrix.hnodes, hmatrix.basis_tree, complete_tree,
                                       leaf_size, handle);
    generateDistributedHNodeTreeEntries<EntryGen>(entry_gen, hmatrix.hnodes, hmatrix.basis_tree, kdtree, complete_tree,
                                                  slices, handle);

    // printf("HNode-Tree done on rank %d\n", handle->rank);

    generateCompressedBasisTreeStructure(hmatrix, complete_tree, handle);

    // printf("Conmpressed basis tree done on rank %d\n", handle->rank);
}
