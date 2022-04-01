include ../../lib/h2opus/make.inc

OBJDIR := obj
OBJ_LIST := test_eig_slepc.o
OBJS := $(addprefix $(OBJDIR)/,$(OBJ_LIST))
EXE := test_eig_slepc

SLEPC_ARCH = $(PETSC_ARCH)
SLEPC_INC = -I$(SLEPC_DIR)/$(SLEPC_ARCH)/include -I$(SLEPC_DIR)/include
SLEPC_LIB = -Wl,-rpath,$(SLEPC_DIR)/$(SLEPC_ARCH)/lib -L$(SLEPC_DIR)/$(SLEPC_ARCH)/lib -lslepc
PETSC_INC = -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(PETSC_DIR)/include
PETSC_LIB = -Wl,-rpath,$(PETSC_DIR)/$(PETSC_ARCH)/lib -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc
$(OBJDIR)/test_eig_slepc.o: test_eig_slepc.cpp
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(SLEPC_INC) $(PETSC_INC) $(H2OPUS_CXXFLAGS) $(CXXFLAGS) -o $@ -c $<

test_eig_slepc: $(OBJDIR)/test_eig_slepc.o
	$(H2OPUS_LINKER) -o $@ $+ $(SLEPC_LIB) $(PETSC_LIB) $(H2OPUS_LIBS) $(LDFLAGS)

clean:
	rm -f $(EXE) $(OBJDIR)/*

clobber: clean
