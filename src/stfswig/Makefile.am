if BUILD_MODULE
else
if BUILD_PYTHON

pkglib_LTLIBRARIES = libstf.la

$(srcdir)/stfswig_wrap.cxx $(srcdir)/stf.py: $(srcdir)/stfswig.i
	$(SWIG) $(SWIG_PYTHON_OPT) -o $@ $<
	cat $(srcdir)/gccwarn $(srcdir)/stfswig_wrap.cxx > $(srcdir)/stfswig_wrap_tmp.cxx
	mv $(srcdir)/stfswig_wrap_tmp.cxx $(srcdir)/stfswig_wrap.cxx

nodist_libstf_la_SOURCES = $(srcdir)/stfswig_wrap.cxx

libstf_la_SOURCES = $(srcdir)/stfswig.cxx # $(SWIG_SOURCES)
noinst_HEADERS = stfswig.h

INCLUDES = $(LIBNUMPY_INCLUDES)

libstf_la_CPPFLAGS = $(SWIG_PYTHON_CPPFLAGS) -I$(top_srcdir)/src
libstf_la_CXXFLAGS = $(OPT_CXXFLAGS)
libstf_la_LDFLAGS = $(LIBPYTHON_LDFLAGS) $(LIBSTF_LDFLAGS) $(LIBBIOSIG_LDFLAGS)
libstf_la_LIBADD = ../app/libstimfit.la $(WX_LIBS)

EXTRA_DIST = stfswig.i numpy.i ivtools.py mintools.py natools.py minidemo.py charlie.py stf.py mintools.py hdf5tools.py spells.py embedded_init.py embedded_stf.py embedded_mpl.py embedded_ipython.py unittest_stfio.py heka.py extensions.py Doxyfile gccwarn

install-exec-hook:
	cd ${prefix}/lib/stimfit && ln -sf ${STF_PYTHON_LIBNAME} _stf.so
	rm -f ${prefix}/lib/stimfit/*.la
	rm -f ${prefix}/lib/stimfit/*.a
	cp -p $(srcdir)/ivtools.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/mintools.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/natools.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/minidemo.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/charlie.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/hdf5tools.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/stf.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/stfio_plot.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/spells.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/embedded_init.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/embedded_stf.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/embedded_mpl.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/embedded_ipython.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/heka.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/extensions.py ${prefix}/lib/stimfit
	chmod -x ${prefix}/lib/stimfit/*.py

uninstall-hook:
	rm -rf ${prefix}/lib/stimfit

clean_local:
	rm -f $(srcdir)/stfswig_wrap.cxx $(srcdir)/stf.py

endif
endif
