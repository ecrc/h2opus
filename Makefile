include make.inc
override H2OPUS_DIR := $(CURDIR)
override OBJ_DIR := $(H2OPUS_DIR)/obj
override LIBH2OPUS := $(H2OPUS_DIR)/lib/libh2opus.a
override LIBH2OPUSVARS := $(H2OPUS_DIR)/lib/h2opus/make.inc
H2OPUS_INSTALL_DIR ?= $(H2OPUS_DIR)
override LIBH2OPUSI := $(H2OPUS_INSTALL_DIR)/lib/libh2opus.a
override LIBH2OPUSVARSI := $(H2OPUS_INSTALL_DIR)/lib/h2opus/make.inc

all: config
	@make -j$(H2OPUS_MAKE_NP) lib

############################################################################
# H2Opus config header file generation
############################################################################
write-confheader-pre    = @printf "\#ifndef __H2OPUSCONF_H__\n\#define __H2OPUSCONF_H__\n" >> $1;
write-confheader-define = @printf "\n\#ifndef H2OPUS_$2\n\#define H2OPUS_$2\n\#endif\n" >> $1;
write-confheader-post   = @printf "\n\#endif\n" >> $1;

config-confheader := $(H2OPUS_DIR)/include/h2opusconf.h

$(config-confheader) :
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
ifeq ($(H2OPUS_USE_SINGLE_PRECISION),)
	@echo "  With double precision real numbers"
	$(call write-confheader-define,$@,DOUBLE_PRECISION)
else
	@echo "  With single precision real numbers"
endif
	@echo "########################################################################"
	$(call write-confheader-post,$@)
	@mkdir -p $(OBJ_DIR)

config : $(config-confheader)

clean:
	rm -rf $(OBJ_DIR)/*
	rm -f $(LIBH2OPUS)
	rm -f $(LIBH2OPUSVARS)

config-clean : clean
	rm -f $(config-confheader)

.PHONY:  config config-clean clean

distclean: config-clean

############################################################################
# Main library compilation
############################################################################
H2OPUS_SRC_DIRS := src/core src/util
H2OPUS_INC_DIRS := include include/h2opus/core include/h2opus/util include/h2opus/marshal
ifneq ($(H2OPUS_USE_GPU),)
H2OPUS_SRC_DIRS += src/util/gpu
endif

H2OPUS_EXAMPLE_DIRS := \
                       examples/common \
                       examples/hgemv \
                       examples/horthog \
                       examples/hcompress \
                       examples/hlru_global \
                       examples/hlru_local \
                       examples/hara \
                       examples/fractional_diffusion

ifneq ($(H2OPUS_USE_MKL),)
  CXXFLAGS += -DMKL_INT=int
endif

# GPU support
ifneq ($(H2OPUS_USE_GPU),)
  HCXX := $(NVCC) -ccbin $(CXX)
  HCXXFLAGS := -x cu $(NVCCFLAGS) $(H2OPUS_GENCODE_FLAGS) $(addprefix -Xcompiler ,$(CXXFLAGS))
else
  HCXX := $(CXX)
  HCXXFLAGS := $(CXXFLAGS)
endif

H2OPUS_INCLUDE = -I$(H2OPUS_DIR)/include
H2OPUS_INCLUDEI = -I$(H2OPUS_INSTALL_DIR)/include
# Add the path for the C headers for blas and lapack
ifneq ($(H2OPUS_CBLAS_INCDIR),)
  INCLUDES += -I$(H2OPUS_CBLAS_INCDIR)
endif
ifneq ($(H2OPUS_THRUST_INCDIR),)
  INCLUDES += -I$(H2OPUS_THRUST_INCDIR)
endif
ifneq ($(H2OPUS_KBLAS_INCDIR),)
  INCLUDES += -I$(H2OPUS_KBLAS_INCDIR)
endif
ifneq ($(H2OPUS_MAGMA_INCDIR),)
  INCLUDES += -I$(H2OPUS_MAGMA_INCDIR)
endif

H2OPUS_SRC_LIST := $(wildcard $(addsuffix /*.cxx, $(H2OPUS_SRC_DIRS)))
H2OPUS_OBJ_LIST := $(H2OPUS_SRC_LIST:%.cxx=$(OBJ_DIR)/%.o)
H2OPUS_INC_LIST := $(wildcard $(addsuffix /*.h, $(H2OPUS_INC_DIRS)))

H2OPUS_EXAMPLE_SRC_LIST := $(wildcard $(addsuffix /*.cpp, $(H2OPUS_EXAMPLE_DIRS)))
H2OPUS_EXAMPLE_INC_LIST := $(wildcard $(addsuffix /*.h, $(H2OPUS_EXAMPLE_DIRS)))

.SECONDEXPANSION:
%/.keep :
	@mkdir -p $(@D)
	@touch $@

.PRECIOUS: %/.keep
$(OBJ_DIR)/%.o: %.cxx | $$(@D)/.keep
	$(HCXX) $(CXXCPPFLAGS) $(H2OPUS_INCLUDE) $(INCLUDES) $(HCXXFLAGS) -c $< -o $@

$(LIBH2OPUS): $(H2OPUS_OBJ_LIST) | $$(@D)/.keep

lib: $(LIBH2OPUS) $(H2OPUS_OBJ_LIST)
	ar -cr $(LIBH2OPUS) $(H2OPUS_OBJ_LIST)
	ranlib $(LIBH2OPUS)
	@mkdir -p $(H2OPUS_DIR)/lib/h2opus
	@printf "# H2OPUS makefile variables\n" > $(LIBH2OPUSVARS)
	@printf "H2OPUS_CXX = $(HCXX)\n" >> $(LIBH2OPUSVARS)
	@printf "H2OPUS_CXXFLAGS = $(HCXXFLAGS)\n" >> $(LIBH2OPUSVARS)
	@printf "H2OPUS_CXXCPP = $(CXXCPPFLAGS) $(H2OPUS_INCLUDE) $(INCLUDES)\n" >> $(LIBH2OPUSVARS)
	@printf "H2OPUS_LINKER = $(CXX)\n" >> $(LIBH2OPUSVARS)
	@printf "H2OPUS_LIBS = $(LIBH2OPUS) $(H2OPUS_KBLAS_LIBS) $(H2OPUS_MAGMA_LIBS) $(H2OPUS_CBLAS_LIBS) $(H2OPUS_CUDA_LIBS) $(LDFLAGS)\n" >> $(LIBH2OPUSVARS)

check: lib
	@echo "Compiling examples to verify the library"
	@(cd $(H2OPUS_DIR)/examples/hara && make clean && make)
	@(cd $(H2OPUS_DIR)/examples/hgemv && make clean && make)
	@(cd $(H2OPUS_DIR)/examples/hcompress && make clean && make)
	@(cd $(H2OPUS_DIR)/examples/hlru_global && make clean && make)
	@(cd $(H2OPUS_DIR)/examples/hlru_local && make clean && make)
	@(cd $(H2OPUS_DIR)/examples/horthog && make clean && make)

install: lib
	mkdir -p $(H2OPUS_INSTALL_DIR)/include
	mkdir -p $(H2OPUS_INSTALL_DIR)/lib/h2opus
	cp -r include/* $(H2OPUS_INSTALL_DIR)/include
	cp $(LIBH2OPUS) $(H2OPUS_INSTALL_DIR)/lib
	@printf "# H2OPUS makefile variables\n" > $(LIBH2OPUSVARSI)
	@printf "H2OPUS_CXX = $(HCXX)\n" >> $(LIBH2OPUSVARSI)
	@printf "H2OPUS_CXXFLAGS = $(HCXXFLAGS)\n" >> $(LIBH2OPUSVARSI)
	@printf "H2OPUS_CXXCPP = $(CXXCPPFLAGS) $(H2OPUS_INCLUDEI) $(INCLUDES)\n" >> $(LIBH2OPUSVARSI)
	@printf "H2OPUS_LINKER = $(CXX)\n" >> $(LIBH2OPUSVARSI)
	@printf "H2OPUS_LIBS = $(LIBH2OPUSI) $(H2OPUS_KBLAS_LIBS) $(H2OPUS_MAGMA_LIBS) $(H2OPUS_CBLAS_LIBS) $(H2OPUS_CUDA_LIBS) $(LDFLAGS)\n" >> $(LIBH2OPUSVARSI)

print:
	$(info $($(VAR)))
	@true

format:
	@clang-format -i $(H2OPUS_SRC_LIST) $(H2OPUS_INC_LIST)
	@clang-format -i $(H2OPUS_EXAMPLE_SRC_LIST) $(H2OPUS_EXAMPLE_INC_LIST)
