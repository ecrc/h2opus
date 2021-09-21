#ifndef __H2OPUS_TLR_DEFS_H__
#define __H2OPUS_TLR_DEFS_H__

enum H2OpusTLRType
{
    H2OpusTLR_NonSymmetric,
    H2OpusTLR_Symmetric,
    H2OpusTLR_LowerTriangular,
    H2OpusTLR_UpperTriangular
};

enum H2OpusTLRAlloc
{
    H2OpusTLRColumn,
    H2OpusTLRTile
};

#define H2OPUS_TLR_BLOCK_GEN_DIAGONAL -1
#define H2OPUS_TLR_WS_PTR(member) (workspace ? &(workspace->member) : NULL)
#define H2OPUS_TLR_BLOCK_DIM(block_index, block_size, n_block, n)                                                      \
    ((block_index) == ((n_block)-1) ? (n) - (block_index) * (block_size) : block_size)

#define H2OPUS_TLR_USE_CHOLESKY_QR

#if !defined(H2OPUS_TLR_USE_CHOLESKY_QR) && defined(H2OPUS_USE_GPU)
#error Only Cholesky QR is supported on GPUs
#endif

#endif
