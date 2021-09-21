include ../../lib/h2opus/make.inc

OBJDIR := obj
OBJ_LIST := test_hlru_global.o test_hlru_local.o
OBJS := $(addprefix $(OBJDIR)/,$(OBJ_LIST))

# Target rules
all: build

build: test_hlru_global test_hlru_local

$(OBJDIR)/%.o: %.cpp
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(H2OPUS_CXXFLAGS) $(CXXFLAGS) -o $@ -c $<

test_hlru_local: $(OBJDIR)/test_hlru_local.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)

test_hlru_global: $(OBJDIR)/test_hlru_global.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)

clean:
	rm -f test_hlru_local test_hlru_global $(OBJS)

clobber: clean
