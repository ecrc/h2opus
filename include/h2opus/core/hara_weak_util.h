#ifndef __HARA_WEAK_UTIL_H__
#define __HARA_WEAK_UTIL_H__

#include <h2opus/core/h2opus_handle.h>
#include <h2opus/core/hlru.h>
#include <h2opus/core/hmatrix.h>
#include <h2opus/core/hmatrix_sampler.h>

void hara_weak_admissibility_dense_update(HMatrixSampler *sampler, HMatrix &hmatrix, H2Opus_Real *input,
                                          H2Opus_Real *output, DenseBlockUpdate &update, h2opusHandle_t h2opus_handle);

void hara_weak_admissibility_low_rank_update(HMatrixSampler *sampler, HMatrix &hmatrix, LowRankUpdate &low_rank_update,
                                             H2Opus_Real *sampled_U, H2Opus_Real *sampled_V, int level, int max_rank,
                                             int r, H2Opus_Real eps, int BS, h2opusHandle_t h2opus_handle);

#ifdef H2OPUS_USE_GPU
void hara_weak_admissibility_dense_update(HMatrixSampler *sampler, HMatrix_GPU &hmatrix, H2Opus_Real *input,
                                          H2Opus_Real *output, DenseBlockUpdate_GPU &update,
                                          h2opusHandle_t h2opus_handle);

void hara_weak_admissibility_low_rank_update(HMatrixSampler *sampler, HMatrix_GPU &hmatrix,
                                             LowRankUpdate_GPU &low_rank_update, H2Opus_Real *sampled_U,
                                             H2Opus_Real *sampled_V, int level, int max_rank, int r, H2Opus_Real eps,
                                             int BS, h2opusHandle_t h2opus_handle);
#endif

#endif
