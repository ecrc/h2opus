include ../../lib/h2opus/make.inc

OBJDIR := obj
OBJ_LIST := test_tlr.o
OBJS := $(addprefix $(OBJDIR)/,$(OBJ_LIST))

# Target rules
all: build

build: test_tlr

$(OBJDIR)/%.o: %.cpp
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(H2OPUS_CXXFLAGS) $(CXXFLAGS) -o $@ -c $<

test_tlr: $(OBJS)
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)

clean:
	rm -f test_tlr $(OBJS)

clobber: clean
