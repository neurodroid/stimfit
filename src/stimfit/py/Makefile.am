if BUILD_PYTHON

pkglib_LTLIBRARIES = libpystf.la

$(srcdir)/pystf_wrap.cxx $(srcdir)/stf.py: $(srcdir)/pystf.i
	$(SWIG) $(SWIG_PYTHON_OPT) -o $@ $<
	cat $(srcdir)/gccwarn $(srcdir)/pystf_wrap.cxx > $(srcdir)/pystf_wrap_tmp.cxx
	mv $(srcdir)/pystf_wrap_tmp.cxx $(srcdir)/pystf_wrap.cxx

nodist_libpystf_la_SOURCES = $(srcdir)/pystf_wrap.cxx

libpystf_la_SOURCES = $(srcdir)/pystf.cxx # $(SWIG_SOURCES)
noinst_HEADERS = pystf.h

INCLUDES = $(LIBNUMPY_INCLUDES)

libpystf_la_CPPFLAGS = $(SWIG_PYTHON_CPPFLAGS) -I$(top_srcdir)/src
libpystf_la_CXXFLAGS = $(OPT_CXXFLAGS) $(WX_CXXFLAGS)
libpystf_la_LDFLAGS = $(LIBPYTHON_LDFLAGS) $(LIBSTF_LDFLAGS) $(LIBBIOSIG_LDFLAGS)
libpystf_la_LIBADD = ../libstimfit.la ../../libstfio/libstfio.la $(WX_LIBS)

EXTRA_DIST = pystf.i numpy.i ivtools.py mintools.py natools.py minidemo.py charlie.py mintools.py hdf5tools.py spells.py embedded_init.py embedded_stf.py embedded_mpl.py embedded_ipython.py heka.py extensions.py gccwarn

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
	cp -p $(srcdir)/spells.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/embedded_init.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/embedded_stf.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/embedded_mpl.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/embedded_ipython.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/heka.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/extensions.py ${prefix}/lib/stimfit
	cp -p $(srcdir)/../../pystfio/stfio_plot.py ${prefix}/lib/stimfit
	chmod -x ${prefix}/lib/stimfit/*.py

uninstall-hook:
	rm -rf ${prefix}/lib/stimfit

clean_local:
	rm -f $(srcdir)/pystf_wrap.cxx $(srcdir)/stf.py

endif
