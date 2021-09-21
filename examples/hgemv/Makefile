include ../../lib/h2opus/make.inc

OBJDIR := obj
OBJ_LIST := test_hgemv.o test_hgemv_dist.o
OBJS := $(addprefix $(OBJDIR)/,$(OBJ_LIST))

# Target rules
all: build

build: test_hgemv test_hgemv_dist

$(OBJDIR)/%.o: %.cpp
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(H2OPUS_CXXFLAGS) $(CXXFLAGS) -o $@ -c $<

test_hgemv: $(OBJDIR)/test_hgemv.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)

test_hgemv_dist: $(OBJDIR)/test_hgemv_dist.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)

clean:
	rm -f test_hgemv test_hgemv_dist $(OBJS)

clobber: clean
