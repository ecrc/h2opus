include ../../lib/h2opus/make.inc

OBJDIR := obj
OBJ_LIST := test_ns.o
OBJS := $(addprefix $(OBJDIR)/,$(OBJ_LIST))

# Target rules
all: build

build: test_ns

# this example needs the GSL library
GSL_DIR ?= /usr
GSL_LIBDIR ?= $(GSL_DIR)/lib64
GSL_LDFLAGS = -Wl,-rpath,$(GSL_LIBDIR) -L$(GSL_LIBDIR) -lgsl -lgslcblas

$(OBJDIR)/%.o: %.cpp
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(H2OPUS_CXXFLAGS) $(CXXFLAGS) -o $@ -c $<

test_ns: $(OBJDIR)/test_ns.o
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(GSL_LDFLAGS)  $(LDFLAGS)

clean:
	rm -f test_ns $(OBJS)

clobber: clean
