# Example configuration using HIPCC (and tuned AOCL for BLAS)

ROCM_DIR ?= $(ROCM_PATH)
CXXCPPFLAGS = $(shell $(ROCM_DIR)/bin/hipconfig --cpp_config)

CXX = $(ROCM_DIR)/bin/hipcc
CXXFLAGS = -fPIC -fopenmp -Wall -O3 -march=native -mtune=native

CXXCPPFLAGS += -DH2OPUS_PROFILING_ENABLED

# This is a shortcut for H2OPUS_USE_BLIS, H2OPUS_USE_FLAME, H2OPUS_USE_AMDRNG
# All the above configurations can be used separately
H2OPUS_USE_AOCL = 1

# BLAS/LAPACK
# We link against the multithreaded version, number of threads can be configured via the
# environment variable BLIS_SET_NUM_THREADS
# BLAS calls within batched routines will be sequential because we use
#   bli_thread_set_num_threads(H2OPUS_BATCH_BLAS_THREADS)
# use CXXCPPFLAGS = -DH2OPUS_BATCH_BLAS_THREADS=n to change the default value of 1
AOCL_DIR ?= /opt/aocl
BLAS_LIBS = -Wl,-rpath,$(AOCL_DIR)/lib -L$(AOCL_DIR)/lib -Wl,-Bstatic -lflame -Wl,-Bdynamic -lblis-mt
LDFLAGS = $(AOCL_DIR)/lib/librng_amd.a $(AOCL_DIR)/lib/libamdsecrng.a
CXXCPPFLAGS += -DH2OPUS_USE_BATCHED_BLAS

# Resolve missing symbols
LDFLAGS += -fopenmp -lamdlibm -lm
