#ifndef __H2OPUS_TLR_CONSTRUCT_H__
#define __H2OPUS_TLR_CONSTRUCT_H__

#include <h2opus/util/batch_wrappers.h>
#include <h2opus/core/tlr/tlr_struct.h>
#include <h2opus/core/tlr/tlr_batch.h>

template <class T, class FunctionGen, int hw>
void construct_tlr_matrix(TTLR_Matrix<T, hw> &A, FunctionGen &func_gen, T eps, h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, T>::type RealVector;
    typedef typename VectorContainer<hw, T *>::type RealPointerArray;
    typedef typename VectorContainer<hw, int>::type IntVector;

    int n = A.n, block_size = A.block_size, n_block = A.n_block;

    // assert(n == func_gen.getDataSetSize());

    bool sym = (A.type == H2OpusTLR_Symmetric);
    bool generate_lower_only = (sym || A.type == H2OpusTLR_LowerTriangular);
    bool generate_triangular = (generate_lower_only || A.type == H2OpusTLR_UpperTriangular);

    // Generate diagonal blocks
    TLR_Batch<T, hw>::template generateDenseBlocks<FunctionGen>(
        vec_ptr(A.diagonal_block_ptrs), block_size, 0, H2OPUS_TLR_BLOCK_GEN_DIAGONAL, n_block, n, func_gen, stream);

    // Generate low rank blocks one block column at a time
    // and then compress them to the tolerance eps
    RealVector temp_blocks, temp_tau, original_blocks;
    RealPointerArray temp_block_ptrs, original_block_ptrs;
    IntVector block_ranks, block_size_array;

    temp_blocks.resize(n_block * block_size * block_size);
    original_blocks.resize(n_block * block_size * block_size);
    temp_tau.resize(n_block * block_size);
    temp_block_ptrs.resize(n_block);
    original_block_ptrs.resize(n_block);
    block_ranks.resize(n_block);
    block_size_array.resize(n_block);

    fillArray(vec_ptr(block_size_array), n_block, block_size, stream, hw);

    generateArrayOfPointers(vec_ptr(temp_blocks), vec_ptr(temp_block_ptrs), block_size * block_size, 0, n_block, stream,
                            hw);

    generateArrayOfPointers(vec_ptr(original_blocks), vec_ptr(original_block_ptrs), block_size * block_size, 0, n_block,
                            stream, hw);

    for (int col = 0; col < n_block; col++)
    {
        int block_row_start = (generate_lower_only ? col + 1 : 0);
        int blockCount = n_block - (generate_triangular ? col + 1 : 0);

        if (blockCount == 0)
            continue;

        TLR_Batch<T, hw>::template generateDenseBlocks<FunctionGen>(
            vec_ptr(temp_block_ptrs), block_size, block_row_start, col, blockCount, n, func_gen, stream);

        // Copy the original blocks since the compression will alter them
        check_kblas_error((H2OpusBatched<T, hw>::copyBlock)(stream, block_size, block_size,
                                                            vec_ptr(original_block_ptrs), 0, 0, block_size,
                                                            vec_ptr(temp_block_ptrs), 0, 0, block_size, blockCount));

        // Clear out tau and the ranks
        fillArray(vec_ptr(temp_tau), blockCount * block_size, 0, stream, hw);
        fillArray(vec_ptr(block_ranks), n_block, 0, stream, hw);

        int *block_ranks_ptr = vec_ptr(block_ranks) + block_row_start;

        // Find the ranks of the blocks
        // TODO: This is slow as heck for large blocks on GPUs so might need to replace by another compression kernel
        // perhaps ARA or ACA or even fixed rank RSVD with rank = block_size / 2 would probably be better lol
        // TODO: the CPU version uses geqp3 which does not guarantee ordered diagonals (Rs)
        // issues with libflame-amd
        check_kblas_error((H2OpusBatched<T, hw>::geqp2)(stream, block_size, block_size, vec_ptr(temp_blocks),
                                                        block_size, block_size * block_size, vec_ptr(temp_tau),
                                                        block_size, block_ranks_ptr, eps, blockCount));

        // Find the max of the ranks since the orgqr routine does not support
        // non-uniform batches
        int max_rank = getMaxElement(block_ranks_ptr, blockCount, stream, hw);

        if (max_rank > A.max_rank)
        {
            printf("Supplied max_rank for A (%d) was insufficient! Using new max %d\n", A.max_rank, max_rank);
            A.max_rank = max_rank;
        }
        // Allocate memory for the low rank blocks in the block column
        // based on the detected ranks
        A.allocateBlockColumn(col, block_ranks_ptr, stream);

        check_kblas_error((H2OpusBatched<T, hw>::orgqr)(stream, block_size, max_rank, vec_ptr(temp_blocks), block_size,
                                                        block_size * block_size, vec_ptr(temp_tau), block_size,
                                                        blockCount));

        // Copy over the orthogonal factors from the temp blocks to the tlr matrix
        T **col_block_U_ptrs = vec_ptr(A.block_U_ptrs) + block_row_start + col * n_block;

        check_kblas_error((H2OpusBatched<T, hw>::copyBlock)(stream, vec_ptr(block_size_array), block_ranks_ptr,
                                                            block_size, max_rank, col_block_U_ptrs,
                                                            vec_ptr(block_size_array), vec_ptr(temp_block_ptrs),
                                                            vec_ptr(block_size_array), blockCount));

        // Set each V factor as the projection of the original dense block into the orthogonal U factor
        // i.e. V' = U' * M for each block M or V = M' * U
        T **col_block_V_ptrs = vec_ptr(A.block_V_ptrs) + block_row_start + col * n_block;

        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, vec_ptr(block_size_array),
                                                       block_ranks_ptr, vec_ptr(block_size_array), block_size, max_rank,
                                                       block_size, (T)1, (const T **)(vec_ptr(original_block_ptrs)),
                                                       vec_ptr(block_size_array), (const T **)col_block_U_ptrs,
                                                       vec_ptr(block_size_array), (T)0, col_block_V_ptrs,
                                                       vec_ptr(block_size_array), blockCount));
    }
}

template <class T, class FunctionGen, int hw>
void construct_spd_tlr_matrix(TTLR_Matrix<T, hw> &A, FunctionGen &func_gen, T eps, h2opusComputeStream_t stream)
{
    typedef typename VectorContainer<hw, T>::type RealVector;
    typedef typename VectorContainer<hw, T *>::type RealPointerArray;
    typedef typename VectorContainer<hw, int>::type IntVector;

    int n = A.n, block_size = A.block_size, n_block = A.n_block;

    // assert(n == func_gen.getDataSetSize());

    bool sym = (A.type == H2OpusTLR_Symmetric);
    bool generate_lower_only = (sym || A.type == H2OpusTLR_LowerTriangular);
    bool generate_triangular = (generate_lower_only || A.type == H2OpusTLR_UpperTriangular);

    // Generate diagonal blocks
    TLR_Batch<T, hw>::template generateDenseBlocks<FunctionGen>(
        vec_ptr(A.diagonal_block_ptrs), block_size, 0, H2OPUS_TLR_BLOCK_GEN_DIAGONAL, n_block, n, func_gen, stream);

    // Generate low rank blocks one block column at a time
    // and then compress them to the tolerance eps
    RealVector temp_blocks, temp_tau, original_blocks;
    RealVector factored_diagonal_blocks, projected_blocks;
    RealPointerArray temp_block_ptrs, original_block_ptrs;
    RealPointerArray factored_diagonal_ptrs, factored_diagonal_ptrs_j, projected_block_ptrs;
    IntVector block_ranks, block_size_array;

    factored_diagonal_blocks.resize(n_block * block_size * block_size);
    projected_blocks.resize(n_block * block_size * block_size);
    temp_blocks.resize(n_block * block_size * block_size);
    original_blocks.resize(n_block * block_size * block_size);

    temp_tau.resize(n_block * block_size);
    factored_diagonal_ptrs.resize(n_block);
    factored_diagonal_ptrs_j.resize(n_block);
    projected_block_ptrs.resize(n_block);
    temp_block_ptrs.resize(n_block);
    original_block_ptrs.resize(n_block);
    block_ranks.resize(n_block);
    block_size_array.resize(n_block);

    fillArray(vec_ptr(block_size_array), n_block, block_size, stream, hw);

    generateArrayOfPointers(vec_ptr(projected_blocks), vec_ptr(projected_block_ptrs), block_size * block_size, 0,
                            n_block, stream, hw);

    generateArrayOfPointers(vec_ptr(temp_blocks), vec_ptr(temp_block_ptrs), block_size * block_size, 0, n_block, stream,
                            hw);

    generateArrayOfPointers(vec_ptr(original_blocks), vec_ptr(original_block_ptrs), block_size * block_size, 0, n_block,
                            stream, hw);

    // Factorize the diagonal blocks
    generateArrayOfPointers(vec_ptr(factored_diagonal_blocks), vec_ptr(factored_diagonal_ptrs), block_size * block_size,
                            0, n_block, stream, hw);
    check_kblas_error((H2OpusBatched<T, hw>::copyBlock)(stream, block_size, block_size, vec_ptr(factored_diagonal_ptrs),
                                                        0, 0, block_size, vec_ptr(A.diagonal_block_ptrs), 0, 0,
                                                        block_size, n_block));

    check_kblas_error(
        (H2OpusBatched<T, hw>::potrf)(stream, block_size, vec_ptr(factored_diagonal_ptrs), block_size, n_block));
    check_kblas_error((H2OpusBatched<T, hw>::setUpperZero)(stream, block_size, block_size,
                                                           vec_ptr(factored_diagonal_ptrs), block_size, n_block));

    for (int col = 0; col < n_block; col++)
    {
        int block_row_start = (generate_lower_only ? col + 1 : 0);
        int blockCount = n_block - (generate_triangular ? col + 1 : 0);

        if (blockCount == 0)
            continue;

        TLR_Batch<T, hw>::template generateDenseBlocks<FunctionGen>(
            vec_ptr(temp_block_ptrs), block_size, block_row_start, col, blockCount, n, func_gen, stream);

        // Scale each block: A(i, j) = L_i^{-1} A(i, j) L_j^{-T}
        T *L_j = vec_ptr(factored_diagonal_blocks) + col * block_size * block_size;
        fillArray(vec_ptr(factored_diagonal_ptrs_j), blockCount, L_j, stream, hw);

        // const int print_digits = 7;
        //
        // if(col == 0)
        // {
        //     printDenseMatrix(A.diagonal_block_ptrs[block_row_start], block_size, block_size, block_size,
        //     print_digits, "Aii", hw); printDenseMatrix(A.diagonal_block_ptrs[col], block_size, block_size,
        //     block_size, print_digits, "Ajj", hw); printDenseMatrix(temp_block_ptrs[0], block_size, block_size,
        //     block_size, print_digits, "Aij", hw);
        //
        //     printDenseMatrix(factored_diagonal_ptrs[block_row_start], block_size, block_size, block_size,
        //     print_digits, "Lii", hw); printDenseMatrix(L_j, block_size, block_size, block_size, print_digits, "Ljj",
        //     hw);
        // }

        check_kblas_error((
            H2OpusBatched<T, hw>::trsm)(stream, H2Opus_Left, H2Opus_Lower, H2Opus_NoTrans, H2Opus_NonUnit,
                                        vec_ptr(block_size_array), vec_ptr(block_size_array), block_size, block_size, 1,
                                        vec_ptr(factored_diagonal_ptrs) + block_row_start, vec_ptr(block_size_array),
                                        vec_ptr(temp_block_ptrs), vec_ptr(block_size_array), blockCount));
        check_kblas_error((H2OpusBatched<T, hw>::trsm)(stream, H2Opus_Right, H2Opus_Lower, H2Opus_Trans, H2Opus_NonUnit,
                                                       vec_ptr(block_size_array), vec_ptr(block_size_array), block_size,
                                                       block_size, 1, vec_ptr(factored_diagonal_ptrs_j),
                                                       vec_ptr(block_size_array), vec_ptr(temp_block_ptrs),
                                                       vec_ptr(block_size_array), blockCount));

        // if(col == 0)
        // {
        //     printDenseMatrix(temp_block_ptrs[0], block_size, block_size, block_size, print_digits, "Aij_2", hw);
        // }

        // Copy the original blocks since the compression will alter them
        check_kblas_error((H2OpusBatched<T, hw>::copyBlock)(stream, block_size, block_size,
                                                            vec_ptr(original_block_ptrs), 0, 0, block_size,
                                                            vec_ptr(temp_block_ptrs), 0, 0, block_size, blockCount));

        // Clear out tau and the ranks
        fillArray(vec_ptr(temp_tau), blockCount * block_size, 0, stream, hw);
        fillArray(vec_ptr(block_ranks), n_block, 0, stream, hw);

        int *block_ranks_ptr = vec_ptr(block_ranks) + block_row_start;

        // Find the ranks of the blocks
        // TODO: This is slow as heck for large blocks on GPUs so might need to replace by another compression kernel
        // perhaps ARA or ACA or even fixed rank RSVD with rank = block_size / 2 would probably be better lol
        check_kblas_error((H2OpusBatched<T, hw>::geqp2)(stream, block_size, block_size, vec_ptr(temp_blocks),
                                                        block_size, block_size * block_size, vec_ptr(temp_tau),
                                                        block_size, block_ranks_ptr, eps, blockCount));

        // Find the max of the ranks since the orgqr routine does not support
        // non-uniform batches
        int max_rank = getMaxElement(block_ranks_ptr, blockCount, stream, hw);

        if (max_rank > A.max_rank)
        {
            printf("Supplied max_rank for A (%d) was insufficient! Using new max %d\n", A.max_rank, max_rank);
            A.max_rank = max_rank;
        }
        // Allocate memory for the low rank blocks in the block column
        // based on the detected ranks
        A.allocateBlockColumn(col, block_ranks_ptr, stream);

        check_kblas_error((H2OpusBatched<T, hw>::orgqr)(stream, block_size, max_rank, vec_ptr(temp_blocks), block_size,
                                                        block_size * block_size, vec_ptr(temp_tau), block_size,
                                                        blockCount));

        // Set each V factor as the projection of the original dense block into the orthogonal U factor
        // i.e. V' = U' * M for each block M or V = M' * U
        check_kblas_error((H2OpusBatched<T, hw>::gemm)(stream, H2Opus_Trans, H2Opus_NoTrans, vec_ptr(block_size_array),
                                                       block_ranks_ptr, vec_ptr(block_size_array), block_size, max_rank,
                                                       block_size, (T)1, (const T **)(vec_ptr(original_block_ptrs)),
                                                       vec_ptr(block_size_array), (const T **)vec_ptr(temp_block_ptrs),
                                                       vec_ptr(block_size_array), (T)0, vec_ptr(projected_block_ptrs),
                                                       vec_ptr(block_size_array), blockCount));

        // U(i, j) = L_i * U(i, j) and V(i, j) = L_j * V(i, j)
        T **col_block_U_ptrs = vec_ptr(A.block_U_ptrs) + block_row_start + col * n_block;

        check_kblas_error(
            (H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, vec_ptr(block_size_array),
                                         block_ranks_ptr, vec_ptr(block_size_array), block_size, max_rank, block_size,
                                         (T)1, (const T **)(vec_ptr(factored_diagonal_ptrs) + block_row_start),
                                         vec_ptr(block_size_array), (const T **)vec_ptr(temp_block_ptrs),
                                         vec_ptr(block_size_array), (T)0, col_block_U_ptrs, vec_ptr(block_size_array),
                                         blockCount));

        T **col_block_V_ptrs = vec_ptr(A.block_V_ptrs) + block_row_start + col * n_block;

        check_kblas_error(
            (H2OpusBatched<T, hw>::gemm)(stream, H2Opus_NoTrans, H2Opus_NoTrans, vec_ptr(block_size_array),
                                         block_ranks_ptr, vec_ptr(block_size_array), block_size, max_rank, block_size,
                                         (T)1, (const T **)(vec_ptr(factored_diagonal_ptrs_j)),
                                         vec_ptr(block_size_array), (const T **)vec_ptr(projected_block_ptrs),
                                         vec_ptr(block_size_array), (T)0, col_block_V_ptrs, vec_ptr(block_size_array),
                                         blockCount));

        // if(col == 0)
        // {
        //     printDenseMatrix(col_block_U_ptrs[0], block_size, block_size, block_ranks_ptr[0], print_digits, "Uij",
        //     hw); printDenseMatrix(col_block_V_ptrs[0], block_size, block_size, block_ranks_ptr[0], print_digits,
        //     "Vij", hw);
        // }
        //
        // exit(0);
    }
}

#endif
