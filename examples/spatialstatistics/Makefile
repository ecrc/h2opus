include ../../lib/h2opus/make.inc

OBJDIR := obj
OBJ_LIST := spatial_statistics.o
OBJS := $(addprefix $(OBJDIR)/,$(OBJ_LIST))
EXE := spatial_statistics

all: build

build: spatial_statistics

PETSC_INC = -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(PETSC_DIR)/include
PETSC_LIB = -Wl,-rpath,$(PETSC_DIR)/$(PETSC_ARCH)/lib -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc
# this example needs the GSL library
GSL_DIR ?= /usr
GSL_LIBDIR ?= $(GSL_DIR)/lib64
GSL_LDFLAGS = -Wl,-rpath,$(GSL_LIBDIR) -L$(GSL_LIBDIR) -lgsl -lgslcblas

$(OBJDIR)/%.o: %.cpp spatial_statistics.h
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(PETSC_INC) $(H2OPUS_CXXFLAGS) $(CXXFLAGS) -o $@ -c $<

spatial_statistics: $(OBJDIR)/spatial_statistics.o
	$(H2OPUS_LINKER) -o $@ $+ $(PETSC_LIB) $(H2OPUS_LIBS) $(GSL_LDFLAGS) $(LDFLAGS)

clean:
	rm -f $(EXE) $(OBJDIR)/*

clobber: clean
