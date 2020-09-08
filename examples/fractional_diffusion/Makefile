include ../../lib/h2opus/make.inc

################################################################################
# Regular C files
OBJDIR := obj
OBJ_LIST := fractional_diffusion.o
OBJS := $(addprefix $(OBJDIR)/,$(OBJ_LIST))

################################################################################
# Target rules
all: build

build: fractional_diffusion

LDFLAGS += -lgsl -L$(HOME)/scratch/codes/gsl/.libs
CXXFLAGS += -I$(HOME)/scratch/codes/gsl/

$(OBJDIR)/%.o: %.cpp
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(H2OPUS_CXXFLAGS) $(CXXFLAGS) -o $@ -c $<

fractional_diffusion: $(OBJS)
	$(H2OPUS_LINKER) -o $@ $+ $(H2OPUS_LIBS) $(LDFLAGS)

clean:
	rm -f fractional_diffusion $(OBJS)

clobber: clean
################################################################################
