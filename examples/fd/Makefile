include ../../lib/h2opus/make.inc

OBJDIR := obj
OBJ_LIST := test_fd.o test_fd_dist.o test_fd_petsc.o
OBJS := $(addprefix $(OBJDIR)/,$(OBJ_LIST))
EXE := test_fd test_fd_dist test_fd_petsc

all: build

build: test_fd test_fd_dist

$(OBJDIR)/%.o: %.cpp fd_core.h
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(H2OPUS_CXXFLAGS) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR)/hcubature.o: hcubature.cpp
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(H2OPUS_CXXFLAGS) $(CXXFLAGS) -o $@ -c $<

test_fd: $(OBJDIR)/test_fd.o $(OBJDIR)/hcubature.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)

test_fd_dist: $(OBJDIR)/test_fd_dist.o $(OBJDIR)/hcubature.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)

PETSC_INC = -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(PETSC_DIR)/include
PETSC_LIB = -Wl,-rpath,$(PETSC_DIR)/$(PETSC_ARCH)/lib -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc
$(OBJDIR)/test_fd_petsc.o: test_fd_petsc.cpp fd_core.h
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(PETSC_INC) $(H2OPUS_CXXFLAGS) $(CXXFLAGS) -o $@ -c $<

test_fd_petsc: $(OBJDIR)/test_fd_petsc.o $(OBJDIR)/hcubature.o
	$(H2OPUS_LINKER) -o $@ $+ $(PETSC_LIB) $(H2OPUS_LIBS) $(LDFLAGS)

clean:
	rm -f $(EXE) $(OBJDIR)/*

clobber: clean
