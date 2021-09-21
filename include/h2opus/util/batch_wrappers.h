#ifndef __BATCH_WRAPPERS_H__
#define __BATCH_WRAPPERS_H__

#include <h2opus/core/h2opus_eps.h>
#include <h2opus/core/h2opus_handle.h>
#include <h2opus/core/h2opus_compute_stream.h>

#define H2OPUS_MAX_OPS_PER_BATCH 65535

#ifdef H2OPUS_PROFILING_ENABLED
#include <h2opus/util/perf_counter.h>
#endif
#include <h2opus/util/debug_routines.h>

template <class T, int hw> struct H2OpusBatched;

#include <h2opus/util/batch_wrappers_cpu.inc>
#include <h2opus/util/batch_wrappers_gpu.inc>

#endif
