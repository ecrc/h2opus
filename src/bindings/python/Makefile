include ../../../lib/h2opus/make.inc

OBJDIR := obj

libpyh2opus.so: $(OBJDIR)/pyh2opus.o
	$(H2OPUS_LINKER) -fPIC -shared $(OBJDIR)/pyh2opus.o -o libpyh2opus.so $(H2OPUS_LIBS)

$(OBJDIR)/%.o: %.cpp
	$(H2OPUS_CXX) $(H2OPUS_CXXCPP) $(H2OPUS_CXXFLAGS) $(CXXFLAGS) -o $@ -c $<

clean:
	rm -f libpyh2opus.so $(OBJDIR)/*

clobber: clean
