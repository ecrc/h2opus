include ../../lib/h2opus/make.inc

OBJDIR := obj
OBJ_LIST := test_horthog.o test_horthog_dist.o
OBJS := $(addprefix $(OBJDIR)/,$(OBJ_LIST))

# Target rules
all: build

build: test_horthog test_horthog_dist

$(OBJDIR)/%.o: %.cpp
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(H2OPUS_CXXFLAGS) $(CXXFLAGS) -o $@ -c $<

test_horthog: $(OBJDIR)/test_horthog.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)

test_horthog_dist: $(OBJDIR)/test_horthog_dist.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)

clean:
	rm -f test_horthog test_horthog_dist $(OBJS)

clobber: clean
