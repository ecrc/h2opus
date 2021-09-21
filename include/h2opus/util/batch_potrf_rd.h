#ifndef __H2OPUS_POTRF_RD_H__
#define __H2OPUS_POTRF_RD_H__

#include <h2opus/core/h2opus_compute_stream.h>

void potrf_rd_batch(int dim, double **A_batch, int lda, int num_ops, h2opusComputeStream_t stream);

void potrf_rd_batch(int dim, float **A_batch, int lda, int num_ops, h2opusComputeStream_t stream);

#endif
