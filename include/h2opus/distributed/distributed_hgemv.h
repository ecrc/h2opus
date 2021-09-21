#ifndef __H2OPUS_DISTRIBUTED_HGEMV_H__
#define __H2OPUS_DISTRIBUTED_HGEMV_H__

#include <h2opus/distributed/distributed_h2opus_handle.h>
#include <h2opus/distributed/distributed_hgemv_workspace.h>
#include <h2opus/distributed/distributed_hmatrix.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU
/////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU
void distributed_hgemv(H2Opus_Real alpha, DistributedHMatrix_GPU &dist_hmatrix, H2Opus_Real *X, int ldx,
                       H2Opus_Real beta, H2Opus_Real *Y, int ldy, int num_vectors,
                       distributedH2OpusHandle_t dist_h2opus_handle);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void distributed_hgemv(H2Opus_Real alpha, DistributedHMatrix &dist_hmatrix, H2Opus_Real *X, int ldx, H2Opus_Real beta,
                       H2Opus_Real *Y, int ldy, int num_vectors, distributedH2OpusHandle_t dist_h2opus_handle);

#endif
