#include <h2opus/core/hara_weak_util.h>
#include <h2opus/core/hara_util.cuh>
#include <h2opus/core/hgemv.h>
#include <h2opus/marshal/hara_marshal.h>

#include <h2opus/util/batch_wrappers.h>
#include <h2opus/util/debug_routines.h>
#include <h2opus/util/gpu_err_check.h>
#include <h2opus/util/thrust_wrappers.h>
#include <h2opus/util/vector_operations.h>

template <int hw>
void sampleLevel(HMatrixSampler *sampler, THMatrix<hw> &hmatrix, int rank, H2Opus_Real *input, H2Opus_Real *output,
                 h2opusHandle_t h2opus_handle)
{
    int n = hmatrix.n;

    // Sample the original matrix A*x
    sampler->sample(input, output, rank);
    sampler->nsamples += rank;

    // compute output = output - A_H^l * x = A * x - A_H^l * x
    hgemv(H2Opus_NoTrans, -1, hmatrix, input, n, 1, output, n, rank, h2opus_handle);
}

template <int hw>
void hara_weak_admissibility_low_rank_update_template(HMatrixSampler *sampler, THMatrix<hw> &hmatrix,
                                                      TLowRankUpdate<hw> &low_rank_update, H2Opus_Real *sampled_U,
                                                      H2Opus_Real *sampled_V, int level, int max_rank, int r,
                                                      H2Opus_Real eps, int BS, h2opusHandle_t h2opus_handle)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    typedef typename VectorContainer<hw, double>::type DoubleVector;
    typedef typename VectorContainer<hw, int>::type IntVector;
    typedef typename VectorContainer<hw, H2Opus_Real *>::type RealPointerArray;
    typedef typename VectorContainer<hw, double *>::type DoublePointerArray;

    h2opusComputeStream_t main_stream = h2opus_handle->getMainStream();

    THNodeTree<hw> &hnodes = hmatrix.hnodes;
    HNodeTreeLevelData &level_data = hnodes.level_data;
    TBasisTree<hw> &u_basis_tree = hmatrix.u_basis_tree;
    TBasisTree<hw> &v_basis_tree = (hmatrix.sym ? hmatrix.u_basis_tree : hmatrix.v_basis_tree);

    int n = hmatrix.n;
    int num_updates = 1 << level;

    // The number of generate nodes is half the number of updates
    int level_nodes = num_updates / 2;

    // Prepare kblas workspace for the trsm
#ifdef H2OPUS_USE_GPU
    kblas_ara_trsm_batch_wsquery<H2Opus_Real>(main_stream->getKblasHandle(), level_nodes);
    kblasAllocateWorkspace(main_stream->getKblasHandle());
#endif
    ///////////////////////////////////////////////////////////////////
    // Low rank update marshaling
    ///////////////////////////////////////////////////////////////////
    low_rank_update.init(n, level, max_rank, num_updates);

    IntVector update_morton_indexes(num_updates);
    IntVector input_block_sizes(level_nodes), output_block_sizes(level_nodes);

    int hnode_level_start = level_data.getLevelStart(level);
    int hnode_level_size = level_data.getLevelSize(level);

    hara_weak_admissibility_update_marshal_batch<H2Opus_Real, hw>(
        sampled_U, low_rank_update.ldu, sampled_V, low_rank_update.ldv, vec_ptr(low_rank_update.U),
        vec_ptr(low_rank_update.V), vec_ptr(input_block_sizes), vec_ptr(output_block_sizes),
        vec_ptr(update_morton_indexes), num_updates, vec_ptr(low_rank_update.hnode_indexes),
        vec_ptr(hnodes.node_morton_level_index), hnode_level_start, hnode_level_size, vec_ptr(hnodes.node_u_index),
        vec_ptr(hnodes.node_v_index), vec_ptr(u_basis_tree.node_start), vec_ptr(v_basis_tree.node_start),
        vec_ptr(u_basis_tree.node_len), vec_ptr(v_basis_tree.node_len), main_stream);

    int max_input_block_size = getMaxElement(vec_ptr(input_block_sizes), level_nodes, main_stream, hw);
    int max_output_block_size = getMaxElement(vec_ptr(output_block_sizes), level_nodes, main_stream, hw);
    int max_node_size = std::max(max_input_block_size, max_output_block_size);

    ///////////////////////////////////////////////////////////////////
    // H2OPUS_Level workspace setup
    ///////////////////////////////////////////////////////////////////
    if (max_rank > max_node_size)
        max_rank = max_node_size;
    if (BS > max_rank)
        BS = max_rank;
    if (BS > max_node_size)
        BS = max_node_size;
    if (BS < 32)
        BS = 16;

    // ARA temporary workspace
    IntVector node_ranks(level_nodes), op_samples(level_nodes);
    IntVector block_ranks(level_nodes), small_vectors(level_nodes);
    IntVector ldu_batch(level_nodes), ldz_batch(level_nodes);

    // Initialize arrays
    initVector(node_ranks, 0, main_stream);
    initVector(op_samples, max_rank, main_stream);
    initVector(block_ranks, 0, main_stream);
    initVector(small_vectors, 0, main_stream);
    initVector(ldu_batch, n, main_stream);
    initVector(ldz_batch, max_rank, main_stream);

    RealVector Z_strided(max_rank * max_rank * level_nodes);
    RealPointerArray Y_ptrs(level_nodes), Z_ptrs(level_nodes);

    // Double precision gram matrix
    DoubleVector R_diag(level_nodes * BS);
    DoubleVector G_strided(level_nodes * BS * BS);
    DoublePointerArray G_ptrs(level_nodes);

    ///////////////////////////////////////////////////////////////////
    // Marshal the random input pointers and dimensions.
    // They are generated just once since we reuse the input vectors
    // and the only thing that changes in the loop is the required
    // number of samples
    ///////////////////////////////////////////////////////////////////
    RealVector random_input(n * BS);
    initVector(random_input, (H2Opus_Real)0, main_stream);

    RealPointerArray input_ptrs(level_nodes);

    hara_weak_admissibility_random_input_marshal_batch<H2Opus_Real, hw>(
        vec_ptr(random_input), vec_ptr(input_ptrs), vec_ptr(low_rank_update.hnode_indexes),
        vec_ptr(hnodes.node_v_index), vec_ptr(v_basis_tree.node_start), num_updates, main_stream);

    // Initialize Z ptrs: strided matrix array with stride max_rank * max_rank
    generateArrayOfPointers(vec_ptr(Z_strided), vec_ptr(Z_ptrs), max_rank * max_rank, level_nodes, main_stream, hw);

    // Initialize G ptrs: strided matrix array with stride BS * BS
    generateArrayOfPointers(vec_ptr(G_strided), vec_ptr(G_ptrs), BS * BS, level_nodes, main_stream, hw);

    // Copy over U to Y so we can advance Y
    copyArray(vec_ptr(low_rank_update.U), vec_ptr(Y_ptrs), level_nodes, main_stream, hw);

    ///////////////////////////////////////////////////////////////////
    // H2OPUS_Level main loop to determine the orthogonal range of the blocks
    ///////////////////////////////////////////////////////////////////
    int rank = 0;
    H2Opus_Real **Q_batch = vec_ptr(low_rank_update.U), **Y_batch = vec_ptr(Y_ptrs);
    H2Opus_Real **Z_batch = vec_ptr(Z_ptrs);

    while (rank < max_rank)
    {
        int samples = std::min(BS, max_rank - rank);

        // Set the op samples to 0 if the operation has converged
        int converged = hara_util_set_batch_samples<hw>(vec_ptr(op_samples), vec_ptr(small_vectors), samples, r,
                                                        vec_ptr(node_ranks), vec_ptr(output_block_sizes),
                                                        vec_ptr(input_block_sizes), level_nodes, main_stream);

        if (converged == 1)
            break;

        // Clear the random input so that if we need to generate non-uniform
        // random blocks, the extra columns are zero
        // fillArray(vec_ptr(random_input), n * samples, 0, main_stream, hw);

        // Generate the random gaussian zero padded input vectors
        check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::rand)(main_stream, h2opus_handle, vec_ptr(input_block_sizes),
                                                                 vec_ptr(op_samples), max_input_block_size, samples,
                                                                 vec_ptr(input_ptrs), vec_ptr(ldu_batch), level_nodes));

        // Get Y = A*x - A^l*x
        H2Opus_Real *sample_U_block = sampled_U + rank * n;
        sampleLevel<hw>(sampler, hmatrix, samples, vec_ptr(random_input), sample_U_block, h2opus_handle);

        // Set diag_R = 1
        fillArray(vec_ptr(R_diag), R_diag.size(), 1, main_stream, hw);

        // BCGS with one reorthogonalization step
        for (int i = 0; i < 4; i++)
        {
            // Project samples
            // Y = Y - Q * (Q' * Y) = Y - Q * Z
            check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
                main_stream, H2Opus_Trans, H2Opus_NoTrans, vec_ptr(node_ranks), vec_ptr(op_samples),
                vec_ptr(output_block_sizes), rank, samples, max_output_block_size, 1, (const H2Opus_Real **)Q_batch,
                vec_ptr(ldu_batch), (const H2Opus_Real **)Y_batch, vec_ptr(ldu_batch), 0, Z_batch, vec_ptr(ldz_batch),
                level_nodes));

            check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::gemm)(
                main_stream, H2Opus_NoTrans, H2Opus_NoTrans, vec_ptr(output_block_sizes), vec_ptr(op_samples),
                vec_ptr(node_ranks), max_output_block_size, samples, rank, -1, (const H2Opus_Real **)Q_batch,
                vec_ptr(ldu_batch), (const H2Opus_Real **)Z_batch, vec_ptr(ldz_batch), 1, Y_batch, vec_ptr(ldu_batch),
                level_nodes));

            // Panel orthogonalization using cholesky qr
            // Compute G = A'*A in mixed precision
            check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::mp_syrk)(
                main_stream, vec_ptr(output_block_sizes), vec_ptr(op_samples), max_output_block_size, samples,
                (const H2Opus_Real **)Y_batch, vec_ptr(ldu_batch), vec_ptr(G_ptrs), vec_ptr(op_samples), level_nodes));

            // Cholesky on G into Z
            check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::mp_fused_potrf)(
                main_stream, vec_ptr(op_samples), BS, vec_ptr(G_ptrs), vec_ptr(op_samples), Z_batch, vec_ptr(ldz_batch),
                vec_ptr(R_diag), vec_ptr(block_ranks), level_nodes));

            // Copy the ranks over to the samples in case the rank was less than the samples
            copyArray(vec_ptr(block_ranks), vec_ptr(op_samples), level_nodes, main_stream, hw);

            // TRSM to set Y = Y * Z^-1, the orthogonal factor
            check_kblas_error((H2OpusBatched<H2Opus_Real, hw>::trsm_ara)(
                main_stream, vec_ptr(output_block_sizes), vec_ptr(block_ranks), max_output_block_size, BS, Y_batch,
                vec_ptr(ldu_batch), Z_batch, vec_ptr(ldz_batch), level_nodes));
        }

        // Count the number of vectors that have a small magnitude
        // also updates the rank, max diagonal and advances the Y_batch pointers
        hara_util_svec_count_batch<H2Opus_Real, double, hw>(
            vec_ptr(op_samples), vec_ptr(node_ranks), vec_ptr(small_vectors), vec_ptr(R_diag), r, (double)eps, BS,
            Y_batch, vec_ptr(low_rank_update.U), vec_ptr(ldu_batch), level_nodes, main_stream);

        // Advance the rank
        rank += samples;
    }

    rank = getMaxElement(vec_ptr(node_ranks), level_nodes, main_stream, hw);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Get the right factor V of the low rank approximation of the blocks
    // V = A' * U
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Zero out the columns for each U_i matrix that weren't orthogonalized (i.e. the columns j such that
    // node_ranks[i] < j < rank)
    int *rows_batch = vec_ptr(output_block_sizes), *cols_batch = vec_ptr(node_ranks);

    getRemainingElements(cols_batch, rank, level_nodes, main_stream, hw);
    int max_remaining_cols = getMaxElement(cols_batch, level_nodes, main_stream, hw);

    H2OpusBatched<H2Opus_Real, hw>::setZero(main_stream, rows_batch, cols_batch, max_node_size, max_remaining_cols,
                                            Y_batch, vec_ptr(ldu_batch), level_nodes);

    // Prepare the matvec input vectors for the projection
    // Reuse the input pointers for the output
    H2Opus_Real **output_ptrs = vec_ptr(input_ptrs);
    int *output_block_rows = vec_ptr(output_block_sizes);

    hara_weak_admissibility_clear_output_marshal_batch<H2Opus_Real, hw>(
        sampled_U, output_ptrs, vec_ptr(low_rank_update.hnode_indexes), vec_ptr(hnodes.node_v_index),
        vec_ptr(v_basis_tree.node_start), vec_ptr(v_basis_tree.node_len), output_block_rows, num_updates, main_stream);

    fillArray(cols_batch, level_nodes, rank, main_stream, hw);
    max_node_size = getMaxElement(output_block_rows, level_nodes, main_stream, hw);

    H2OpusBatched<H2Opus_Real, hw>::setZero(main_stream, output_block_rows, cols_batch, max_node_size, rank,
                                            output_ptrs, vec_ptr(ldu_batch), level_nodes);

    int V_rank = 0;
    while (V_rank < rank)
    {
        int samples = std::min(BS, rank - V_rank);
        H2Opus_Real *sample_U_block = sampled_U + V_rank * n;
        H2Opus_Real *sample_V_block = sampled_V + V_rank * n;

        // Get A*x - A^l-x
        sampleLevel<hw>(sampler, hmatrix, samples, sample_U_block, sample_V_block, h2opus_handle);

        V_rank += samples;
    }

    // printDenseMatrix(sampled_U, n, n, rank, 4, "U");
    // printDenseMatrix(sampled_V, n, n, rank, 4, "V");

    low_rank_update.setRank(rank);
}

template <int hw>
void hara_weak_admissibility_dense_update_template(HMatrixSampler *sampler, THMatrix<hw> &hmatrix, H2Opus_Real *input,
                                                   H2Opus_Real *output, TDenseBlockUpdate<hw> &update,
                                                   h2opusHandle_t h2opus_handle)
{
    // typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    typedef typename VectorContainer<hw, int>::type IntVector;
    typedef typename VectorContainer<hw, H2Opus_Real *>::type RealPointerArray;

    h2opusComputeStream_t main_stream = h2opus_handle->getMainStream();

    THNodeTree<hw> &hnodes = hmatrix.hnodes;
    HNodeTreeLevelData &level_data = hnodes.level_data;
    TBasisTree<hw> &u_basis_tree = hmatrix.u_basis_tree;
    TBasisTree<hw> &v_basis_tree = (hmatrix.sym ? hmatrix.u_basis_tree : hmatrix.v_basis_tree);

    int num_updates = hmatrix.u_basis_tree.basis_leaves;
    int n = hmatrix.n, leaf_size = hmatrix.u_basis_tree.leaf_size;

    RealPointerArray input_ptrs(num_updates);
    update.init(leaf_size, num_updates, n);

    // Search for the hnode indexes corresponding to a weak admissibility structure
    // using the morton indexes
    IntVector morton_indexes(num_updates);
    int leaf_level = hnodes.depth - 1;

    hara_weak_admissibility_dense_update_hnode_index_marshal_batch<hw>(
        vec_ptr(morton_indexes), vec_ptr(update.hnode_indexes), vec_ptr(hnodes.node_morton_level_index),
        level_data.getLevelStart(leaf_level), level_data.getLevelSize(leaf_level), num_updates, main_stream);

    // Generate block pointers to set as the identity
    IntVector rows_array(num_updates), cols_array(num_updates), ld_array(num_updates);

    hara_weak_admissibility_dense_update_input_marshal_batch<H2Opus_Real, hw>(
        input, vec_ptr(input_ptrs), vec_ptr(rows_array), vec_ptr(cols_array), vec_ptr(update.hnode_indexes),
        vec_ptr(hnodes.node_u_index), vec_ptr(hnodes.node_v_index), vec_ptr(u_basis_tree.node_len),
        vec_ptr(v_basis_tree.node_len), vec_ptr(v_basis_tree.node_start), leaf_size, num_updates, main_stream);
    fillArray(vec_ptr(ld_array), num_updates, n, main_stream, hw);

    H2OpusBatched<H2Opus_Real, hw>::setIdentity(main_stream, vec_ptr(rows_array), vec_ptr(cols_array), leaf_size,
                                                leaf_size, vec_ptr(input_ptrs), vec_ptr(ld_array), num_updates);

    // Sample the level using matvecs on the structured input
    // Get output = (A - A^l) * input
    sampleLevel<hw>(sampler, hmatrix, leaf_size, input, output, h2opus_handle);

    // Marshal the output into the update
    hara_weak_admissibility_dense_update_output_marshal_batch<H2Opus_Real, hw>(
        output, vec_ptr(update.M), vec_ptr(update.hnode_indexes), vec_ptr(hnodes.node_u_index),
        vec_ptr(u_basis_tree.node_start), num_updates, main_stream);

    // Symmetricize the generated blocks, since they will often not be exactly symmetric
    // Transpose them into the input (the blocks should be square, so the same pointers are
    // still valid), then add them into the input and average: M = (M + M') / 2

    // The blocks are on the diagonal and should be square
    H2OpusBatched<H2Opus_Real, hw>::transpose(main_stream, vec_ptr(rows_array), vec_ptr(rows_array), leaf_size,
                                              leaf_size, vec_ptr(update.M), vec_ptr(ld_array), vec_ptr(input_ptrs),
                                              vec_ptr(ld_array), num_updates);

    H2OpusBatched<H2Opus_Real, hw>::add_matrix(main_stream, vec_ptr(rows_array), vec_ptr(rows_array), leaf_size,
                                               leaf_size, 0.5, vec_ptr(update.M), vec_ptr(ld_array), 0.5,
                                               vec_ptr(input_ptrs), vec_ptr(ld_array), vec_ptr(update.M),
                                               vec_ptr(ld_array), num_updates);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Interface routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void hara_weak_admissibility_dense_update(HMatrixSampler *sampler, HMatrix &hmatrix, H2Opus_Real *input,
                                          H2Opus_Real *output, DenseBlockUpdate &update, h2opusHandle_t h2opus_handle)
{
    hara_weak_admissibility_dense_update_template<H2OPUS_HWTYPE_CPU>(sampler, hmatrix, input, output, update,
                                                                     h2opus_handle);
}

void hara_weak_admissibility_low_rank_update(HMatrixSampler *sampler, HMatrix &hmatrix, LowRankUpdate &low_rank_update,
                                             H2Opus_Real *sampled_U, H2Opus_Real *sampled_V, int level, int max_rank,
                                             int r, H2Opus_Real eps, int BS, h2opusHandle_t h2opus_handle)
{
    hara_weak_admissibility_low_rank_update_template<H2OPUS_HWTYPE_CPU>(
        sampler, hmatrix, low_rank_update, sampled_U, sampled_V, level, max_rank, r, eps, BS, h2opus_handle);
}

#ifdef H2OPUS_USE_GPU
void hara_weak_admissibility_dense_update(HMatrixSampler *sampler, HMatrix_GPU &hmatrix, H2Opus_Real *input,
                                          H2Opus_Real *output, DenseBlockUpdate_GPU &update,
                                          h2opusHandle_t h2opus_handle)
{
    hara_weak_admissibility_dense_update_template<H2OPUS_HWTYPE_GPU>(sampler, hmatrix, input, output, update,
                                                                     h2opus_handle);
}

void hara_weak_admissibility_low_rank_update(HMatrixSampler *sampler, HMatrix_GPU &hmatrix,
                                             LowRankUpdate_GPU &low_rank_update, H2Opus_Real *sampled_U,
                                             H2Opus_Real *sampled_V, int level, int max_rank, int r, H2Opus_Real eps,
                                             int BS, h2opusHandle_t h2opus_handle)
{
    hara_weak_admissibility_low_rank_update_template<H2OPUS_HWTYPE_GPU>(
        sampler, hmatrix, low_rank_update, sampled_U, sampled_V, level, max_rank, r, eps, BS, h2opus_handle);
}
#endif
