#ifndef __H2OPUS_DISTRIBUTED_HORTHOG_H__
#define __H2OPUS_DISTRIBUTED_HORTHOG_H__

#include <h2opus/distributed/distributed_h2opus_handle.h>
#include <h2opus/distributed/distributed_hmatrix.h>
#include <h2opus/distributed/distributed_horthog_workspace.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU
/////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU
void distributed_horthog(DistributedHMatrix_GPU &dist_hmatrix, distributedH2OpusHandle_t dist_h2opus_handle);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void distributed_horthog(DistributedHMatrix &dist_hmatrix, distributedH2OpusHandle_t dist_h2opus_handle);

#endif
