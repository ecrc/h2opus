include ../../lib/h2opus/make.inc

OBJDIR := obj
OBJ_LIST := test_hcompress.o test_hcompress_dist.o
OBJS := $(addprefix $(OBJDIR)/,$(OBJ_LIST))

# Target rules
all: build

build: test_hcompress test_hcompress_dist

$(OBJDIR)/%.o: %.cpp
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(H2OPUS_CXXFLAGS) $(CXXFLAGS) -o $@ -c $<

test_hcompress: $(OBJDIR)/test_hcompress.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)

test_hcompress_dist: $(OBJDIR)/test_hcompress_dist.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)

clean:
	rm -f test_hcompress test_hcompress_dist $(OBJS)

clobber: clean
