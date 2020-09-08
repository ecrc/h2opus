include ../../lib/h2opus/make.inc

################################################################################
# Regular C files
OBJDIR := obj
OBJ_LIST := test_hlru_local.o
OBJS := $(addprefix $(OBJDIR)/,$(OBJ_LIST))

################################################################################
# Target rules
all: build

build: test_hlru_local

$(OBJDIR)/%.o: %.cpp
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(H2OPUS_CXXFLAGS) $(CXXFLAGS) -o $@ -c $<

test_hlru_local: $(OBJS)
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)

clean:
	rm -f test_hlru_local $(OBJS)

clobber: clean
################################################################################
