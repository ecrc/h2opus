include ../../lib/h2opus/make.inc

################################################################################
# Regular C files
OBJDIR := obj
OBJ_LIST := test_tlr.o test_tlr_batch.o test_tlr_kdtree.o test_tlr_rng.o test_tlr_trsm.o test_tlr_gemm.o test_tlr_fd.o kernel_regression.o
OBJS := $(addprefix $(OBJDIR)/,$(OBJ_LIST))

################################################################################
# Target rules
all: build

build: test_tlr

LDFLAGS_FD = -lgsl -L$(HOME)/codes/gsl-2.6/.libs
CXXFLAGS_FD = -I$(HOME)/codes/gsl-2.6/

$(OBJDIR)/%.o: %.cpp
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(H2OPUS_CXXFLAGS) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR)/test_tlr_fd.o: test_tlr_fd.cpp
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(H2OPUS_CXXFLAGS) $(CXXFLAGS_FD) $(CXXFLAGS)-o $@ -c $<

test_tlr: $(OBJDIR)/test_tlr.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)
    
test_tlr_trsm: $(OBJDIR)/test_tlr_trsm.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)
    
test_tlr_gemm: $(OBJDIR)/test_tlr_gemm.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)

test_tlr_batch: $(OBJDIR)/test_tlr_batch.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)
    
test_tlr_kdtree: $(OBJDIR)/test_tlr_kdtree.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)
    
test_tlr_rng: $(OBJDIR)/test_tlr_rng.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)
    
test_tlr_fd: $(OBJDIR)/test_tlr_fd.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS_FD) $(LDFLAGS)

kernel_regression: $(OBJDIR)/kernel_regression.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)    

clean:
	rm -f test_tlr test_tlr_batch test_tlr_kdtree test_tlr_rng test_tlr_trsm test_tlr_gemm test_tlr_fd kernel_regression $(OBJS)

clobber: clean
################################################################################
