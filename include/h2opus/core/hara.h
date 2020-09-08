#ifndef __H2OPUS_H__
#define __H2OPUS_H__

#include <h2opus/core/h2opus_handle.h>
#include <h2opus/core/hmatrix.h>
#include <h2opus/core/hmatrix_sampler.h>

//////////////////////////////////////////////////////////////////////
// CPU
//////////////////////////////////////////////////////////////////////
/**
 * @brief Constructs a hierarchical matrix using a sampler black box to an absolute threshold eps
 *
 * @param[in, out] sampler The black box sampler that defines the HMatrixSampler interface
 * @param[in, out] hmatrix The hierarchical matrix that will be constructed. The data is expected to be cleared out on
 * entry.
 * @param[in] max_rank The maximum number of samples that will be taken at each level of the approximation
 * @param[in] r The number of samples that must satisfy the error threshold before stopping the sampling process. 10 is
 * a good value
 * @param[in] eps The absolute error threshold of the construction
 * @param[in] bs The number of samples taken at the same time. Must be 16 or 32
 * @param[in] h2opus_handle H2Opus handle
 * @param[in] verbose Print out sampling and construction statistics to the standard output
 */
void hara(HMatrixSampler *sampler, HMatrix &hmatrix, int max_rank, int r, H2Opus_Real eps, int bs,
          h2opusHandle_t h2opus_handle, bool verbose = true);

//////////////////////////////////////////////////////////////////////
// GPU
//////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU
void hara(HMatrixSampler *sampler, HMatrix_GPU &hmatrix, int max_rank, int r, H2Opus_Real eps, int bs,
          h2opusHandle_t h2opus_handle, bool verbose = true);
#endif

#endif
