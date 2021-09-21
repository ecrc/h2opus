override H2OPUS_DIR := $(CURDIR)
override LIBH2OPUSVARS := $(H2OPUS_DIR)/lib/h2opus/make.inc
override OBJ_DIR := $(H2OPUS_DIR)/obj

# Default parallel build
all:
	@$(MAKE) -j$(MAKE_NP) lib

.SECONDEXPANSION:
%/.keep :
	@mkdir -p $(@D)
	@touch $@

.PRECIOUS: %/.keep

# Any change in make.inc or lib/h2opus/make.inc will trigger reconfiguration
$(H2OPUS_DIR)/make.inc:
	@cp make.inc.cpu make.inc

$(LIBH2OPUSVARS): $(H2OPUS_DIR)/make.inc | $(OBJ_DIR)/.keep $(H2OPUS_DIR)/lib/h2opus/.keep
	@printf "include $(H2OPUS_DIR)/make.inc" >> $@
	@$(MAKE) -s dumpmakeinc

-include make.inc
-include $(LIBH2OPUSVARS)

# Archiver and shared linker default options
AR_SUFFIX ?= a
AR ?= ar
AR_FLAGS ?= -cr
RANLIB ?= ranlib

SL_SUFFIX ?= so
SL ?= $(CXX)
SL_FLAGS ?= -fPIC -shared
SL_LINK_FLAG ?= -Wl,-rpath,

override LIBH2OPUS_static := $(H2OPUS_DIR)/lib/libh2opus.$(AR_SUFFIX)
override LIBH2OPUS_shared := $(H2OPUS_DIR)/lib/libh2opus.$(SL_SUFFIX)
H2OPUS_INSTALL_DIR ?= $(H2OPUS_DIR)
override LIBH2OPUSI_static := $(H2OPUS_INSTALL_DIR)/lib/libh2opus.$(AR_SUFFIX)
override LIBH2OPUSI_shared := $(H2OPUS_INSTALL_DIR)/lib/libh2opus.$(SL_SUFFIX)
override LIBH2OPUSVARSI := $(H2OPUS_INSTALL_DIR)/lib/h2opus/make.inc

############################################################################
# H2Opus config header file generation
############################################################################
write-confheader-pre    = @printf "\#ifndef __H2OPUSCONF_H__\n\#define __H2OPUSCONF_H__\n" >> $1;
write-confheader-define = @printf "\n\#ifndef H2OPUS_$2\n\#define H2OPUS_$2\n\#endif\n" >> $1;
write-confheader-post   = @printf "\n\#endif\n" >> $1;

config-confheader := $(H2OPUS_DIR)/include/h2opusconf.h
$(config-confheader): $(LIBH2OPUSVARS)
	@rm -rf $@
	@echo "########################################################################"
	@echo "Writing H2OPUS configuration (see make.inc.* examples for customization)"
	$(call write-confheader-pre,$@)
ifeq ($(H2OPUS_USE_GPU),)
	@echo "  With CPU backend"
else
	@echo "  With GPU backend"
	$(call write-confheader-define,$@,USE_GPU)
endif
ifneq ($(H2OPUS_USE_MKL),)
	@echo "  With MKL support"
	$(call write-confheader-define,$@,USE_MKL)
endif
ifneq ($(H2OPUS_USE_ESSL),)
	@echo "  With ESSL support"
	$(call write-confheader-define,$@,USE_ESSL)
endif
ifneq ($(H2OPUS_USE_BLIS),)
	@echo "  With BLIS support"
	$(call write-confheader-define,$@,USE_BLIS)
endif
ifneq ($(H2OPUS_USE_FLAME),)
	@echo "  With FLAME support"
	$(call write-confheader-define,$@,USE_FLAME)
endif
ifneq ($(H2OPUS_USE_NEC),)
	@echo "  With NEC support"
	$(call write-confheader-define,$@,USE_NEC)
endif
ifneq ($(H2OPUS_USE_NVOMP),)
	@echo "  With NVOMP support"
	$(call write-confheader-define,$@,USE_NVOMP)
endif
ifneq ($(H2OPUS_USE_MPI),)
	@echo "  With MPI support"
	$(call write-confheader-define,$@,USE_MPI)
ifneq ($(H2OPUS_USE_GPU),)
ifneq ($(H2OPUS_USE_CUDA_AWARE_MPI),)
	@echo "  With CUDA_AWARE_MPI support"
	$(call write-confheader-define,$@,USE_CUDA_AWARE_MPI)
endif
endif
endif
ifneq ($(H2OPUS_USE_AOCL),)
	@echo "  With AOCL support"
	$(call write-confheader-define,$@,USE_AOCL)
	$(call write-confheader-define,$@,USE_AMDRNG)
	$(call write-confheader-define,$@,USE_BLIS)
	$(call write-confheader-define,$@,USE_FLAME)
endif
ifeq ($(H2OPUS_USE_SINGLE_PRECISION),)
	@echo "  With double precision real numbers"
	$(call write-confheader-define,$@,USE_DOUBLE_PRECISION)
else
	@echo "  With single precision real numbers"
	$(call write-confheader-define,$@,USE_SINGLE_PRECISION)
endif
ifneq ($(H2OPUS_USE_GPU),)
ifneq ($(H2OPUS_USE_MAGMA_POTRF),)
	@echo "  With magma potrf"
	$(call write-confheader-define,$@,USE_MAGMA_POTRF)
else
	@echo "  With kblas potrf"
endif
endif
ifeq ($(H2OPUS_FMANGLE_TYPE),add)
	@echo "  With append underscore for Fortran name mangling"
	$(call write-confheader-define,$@,FMANGLE_ADD)
endif
ifeq ($(H2OPUS_FMANGLE_TYPE),upper)
	@echo "  With to-upper for Fortran name mangling"
	$(call write-confheader-define,$@,FMANGLE_UPPER)
endif
ifeq ($(H2OPUS_FMANGLE_TYPE),nochange)
	@echo "  With no-change Fortran name mangling"
	$(call write-confheader-define,$@,FMANGLE_NOCHANGE)
endif
	@echo "########################################################################"
	$(call write-confheader-post,$@)

# force configuration
config: config-clean
	@$(MAKE) -s $(config-confheader)

clean:
	@rm -rf $(OBJ_DIR)/*
	@rm -f $(H2OPUS_DIR)/lib/libh2opus.*
	@rm -rf h2opus.scanbuild

config-clean : clean
	@rm -f $(LIBH2OPUSVARS)
	@rm -f $(config-confheader)

.PHONY:  config config-clean clean

distclean: config-clean

############################################################################
# Main library compilation
############################################################################
H2OPUS_LIBS := $(KBLAS_LIBS) $(MAGMA_LIBS) $(BLAS_LIBS) $(CUDA_LIBS) $(LDFLAGS)

H2OPUS_SRC_DIRS := src/core src/util
H2OPUS_INC_DIRS := include include/h2opus/core include/h2opus/core/tlr include/h2opus/util include/h2opus/marshal
ifneq ($(H2OPUS_USE_GPU),)
H2OPUS_SRC_DIRS += src/util/gpu
endif
ifneq ($(H2OPUS_USE_MPI),)
H2OPUS_SRC_DIRS += src/distributed
H2OPUS_INC_DIRS += include/h2opus/distributed
endif

H2OPUS_EXAMPLE_DIRS := \
                       examples/common \
                       examples/hgemv \
                       examples/horthog \
                       examples/hcompress \
                       examples/hlru_global \
                       examples/hlru_local \
                       examples/hara \
                       examples/tlr \
                       examples/fd \
                       examples/ns

# GPU support
ifneq ($(H2OPUS_USE_GPU),)
  ifeq ($(findstring ccbin,$(NVCCFLAGS)),)
    NVCCFLAGS += -ccbin $(CXX)
    HCXXFLAGS := -x cu $(NVCCFLAGS) $(H2OPUS_GENCODE_FLAGS) $(addprefix -Xcompiler ,$(CXXFLAGS))
  else # if ccbin is in NVCCFLAGS, assumes flags for the host compiler are passed with NVCCFLAGS
    HCXXFLAGS := -x cu $(NVCCFLAGS) $(H2OPUS_GENCODE_FLAGS)
  endif
  HCXX := $(NVCC)
else
  HCXX := $(CXX)
  HCXXFLAGS := $(CXXFLAGS)
endif

H2OPUS_INCLUDE = -I$(H2OPUS_DIR)/include
H2OPUS_INCLUDEI = -I$(H2OPUS_INSTALL_DIR)/include
ifneq ($(BLAS_INCDIR),)
  INCLUDES += -I$(BLAS_INCDIR)
endif
ifneq ($(THRUST_INCDIR),)
  INCLUDES += -I$(THRUST_INCDIR)
endif
ifneq ($(KBLAS_INCDIR),)
  INCLUDES += -I$(KBLAS_INCDIR)
endif
ifneq ($(MAGMA_INCDIR),)
  INCLUDES += -I$(MAGMA_INCDIR)
endif

H2OPUS_SRC_LIST := $(wildcard $(addsuffix /*.cxx, $(H2OPUS_SRC_DIRS)))
H2OPUS_OBJ_LIST := $(H2OPUS_SRC_LIST:%.cxx=$(OBJ_DIR)/%.o)
H2OPUS_INC_LIST := $(wildcard $(addsuffix /*.h, $(H2OPUS_INC_DIRS))) $(wildcard $(addsuffix /*.cuh, $(H2OPUS_INC_DIRS))) $(wildcard $(addsuffix /*.inc, $(H2OPUS_INC_DIRS)))

H2OPUS_EXAMPLE_SRC_LIST := $(wildcard $(addsuffix /*.cpp, $(H2OPUS_EXAMPLE_DIRS)))
H2OPUS_EXAMPLE_INC_LIST := $(wildcard $(addsuffix /*.h, $(H2OPUS_EXAMPLE_DIRS)))
# Keep PETSc examples formatted PETSc-style
H2OPUS_EXAMPLE_SRC_LIST := $(filter-out examples/fd/test_fd_petsc.cpp,$(H2OPUS_EXAMPLE_SRC_LIST))

H2OPUS_CXX ?= $(HCXX)
H2OPUS_CXXFLAGS ?= $(HCXXFLAGS)
H2OPUS_CXXCPP ?= $(H2OPUS_INCLUDE) $(CXXCPPFLAGS) $(INCLUDES)
H2OPUS_LINKER ?= $(SL)
H2OPUS_FULL_LIBS ?= $(SL_LINK_FLAG)$(H2OPUS_DIR)/lib -L$(H2OPUS_DIR)/lib -lh2opus $(H2OPUS_LIBS)
H2OPUS_FULL_LIBSI ?= $(SL_LINK_FLAG)$(H2OPUS_INSTALL_DIR)/lib -L$(H2OPUS_INSTALL_DIR)/lib -lh2opus $(H2OPUS_LIBS)

$(OBJ_DIR)/%.o: %.cxx $(config-confheader) $(LIBH2OPUSVARS) | $$(@D)/.keep
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(H2OPUS_CXXFLAGS) -c $< -o $@

# Both shared and static libraries by default
LIBH2OPUS := $(LIBH2OPUS_shared) $(LIBH2OPUS_static)
ifneq ($(H2OPUS_DISABLE_STATIC),)
LIBH2OPUS = $(filter-out $(LIBH2OPUS_static),$(LIBH2OPUS))
endif
ifneq ($(H2OPUS_DISABLE_SHARED),)
LIBH2OPUS = $(filter-out $(LIBH2OPUS_shared),$(LIBH2OPUS))
endif

dumpmakeinc:
	@printf "# H2OPUS makefile variables\n" > $(LIBH2OPUSVARS)
	@printf "H2OPUS_CXX = $(H2OPUS_CXX)\n" >> $(LIBH2OPUSVARS)
	@printf "H2OPUS_CXXFLAGS = $(H2OPUS_CXXFLAGS)\n" >> $(LIBH2OPUSVARS)
	@printf "H2OPUS_CXXCPP = $(H2OPUS_CXXCPP)\n" >> $(LIBH2OPUSVARS)
	@printf "H2OPUS_LINKER = $(H2OPUS_LINKER)\n" >> $(LIBH2OPUSVARS)
	@printf "H2OPUS_LIBS = $(H2OPUS_FULL_LIBS)\n" >> $(LIBH2OPUSVARS)

.PHONY: dumpmakeinc

$(LIBH2OPUS): $(H2OPUS_OBJ_LIST) | $$(@D)/.keep

$(LIBH2OPUS_static): $(H2OPUS_OBJ_LIST)
	$(AR) $(AR_FLAGS) $(LIBH2OPUS_static) $(H2OPUS_OBJ_LIST)
	$(RANLIB) $(LIBH2OPUS_static)

$(LIBH2OPUS_shared): $(H2OPUS_OBJ_LIST)
	$(SL) $(SL_FLAGS) -o $(LIBH2OPUS_shared) $(H2OPUS_OBJ_LIST) $(H2OPUS_LIBS)
ifneq ($(DSYMUTIL),)
	$(DSYMUTIL) $@
endif

lib: $(LIBH2OPUS) $(config-confheader)

.PHONY: lib

check: $(LIBH2OPUS)
	@echo "Compiling examples to verify the library"
	@(cd $(H2OPUS_DIR)/examples/hara && make clean && make)
	@(cd $(H2OPUS_DIR)/examples/hgemv && make clean && make)
	@(cd $(H2OPUS_DIR)/examples/hcompress && make clean && make)
	@(cd $(H2OPUS_DIR)/examples/hlru && make clean && make)
	@(cd $(H2OPUS_DIR)/examples/horthog && make clean && make)
	@(cd $(H2OPUS_DIR)/examples/tlr && make clean && make)
	@(cd $(H2OPUS_DIR)/examples/fd && make clean && make)
	@(cd $(H2OPUS_DIR)/examples/ns && make clean && make)

.PHONY: check

install: lib
	mkdir -p $(H2OPUS_INSTALL_DIR)/include
	mkdir -p $(H2OPUS_INSTALL_DIR)/lib/h2opus
	cp -r include/* $(H2OPUS_INSTALL_DIR)/include
	cp $(LIBH2OPUS) $(H2OPUS_INSTALL_DIR)/lib
	@printf "# H2OPUS makefile variables\n" > $(LIBH2OPUSVARSI)
	@printf "H2OPUS_CXX = $(H2OPUS_CXX)\n" >> $(LIBH2OPUSVARSI)
	@printf "H2OPUS_CXXFLAGS = $(H2OPUS_CXXFLAGS)\n" >> $(LIBH2OPUSVARSI)
	@printf "H2OPUS_CXXCPP = $(H2OPUS_CXXCPP)\n" >> $(LIBH2OPUSVARSI)
	@printf "H2OPUS_LINKER = $(H2OPUS_LINKER)\n" >> $(LIBH2OPUSVARSI)
	@printf "H2OPUS_LIBS = $(H2OPUS_FULL_LIBSI)\n" >> $(LIBH2OPUSVARSI)

.PHONY: install

print:
	$(info $($(VAR)))
	@true

# CLANG related support

CLANG_FORMAT ?= clang-format
SCAN_BUILD ?= scan-build
format:
	@$(CLANG_FORMAT) -i $(H2OPUS_SRC_LIST) $(H2OPUS_INC_LIST) $(H2OPUS_EXAMPLE_SRC_LIST) $(H2OPUS_EXAMPLE_INC_LIST)

scanbuild:
	@$(MAKE) distclean
	@$(SCAN_BUILD) -v --force-analyze-debug-code -enable-checker optin.mpi.MPI-Checker -o h2opus.scanbuild sh -c 'export OMPI_CXX=$${CXX}; export MPICH_CXX=$${CXX}; make; make check'
