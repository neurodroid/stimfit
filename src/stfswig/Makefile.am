if BUILD_MODULE

else
if BUILD_PYTHON

lib_LTLIBRARIES = libstf.la

libstf_la_SOURCES = $(srcdir)/stfswig_wrap.cxx $(srcdir)/stfswig.cxx # $(SWIG_SOURCES)
noinst_HEADERS = stfswig.h

$(srcdir)/stfswig_wrap.cxx : $(srcdir)/stfswig.i
	$(SWIG) $(SWIG_PYTHON_OPT) -o $@ $<

INCLUDES = $(LIBNUMPY_INCLUDES)

libstf_la_CPPFLAGS = $(SWIG_PYTHON_CPPFLAGS) -I$(top_srcdir)/src
libstf_la_CXXFLAGS = $(OPT_CXXFLAGS)
libstf_la_LDFLAGS = $(WX_LIBS) $(LIBPYTHON_LDFLAGS)

EXTRA_DIST = stfswig.i ivtools.py mintools.py natools.py minidemo.py charlie.py stf.py mintools.py hdf5tools.py spells.py embedded_init.py embedded_stf.py embedded_ipython.py unittest_stfio.py Doxyfile

install-exec-hook:
	$(MKDIR_P) ${PYTHON_SITE_PKG}/stimfit
	$(LN_S) -f ${prefix}/lib/${STF_PYTHON_LIBNAME} ${PYTHON_SITE_PKG}/stimfit/_stf.so
	cp -p $(srcdir)/ivtools.py ${PYTHON_SITE_PKG}/stimfit
	cp -p $(srcdir)/mintools.py ${PYTHON_SITE_PKG}/stimfit
	cp -p $(srcdir)/natools.py ${PYTHON_SITE_PKG}/stimfit
	cp -p $(srcdir)/minidemo.py ${PYTHON_SITE_PKG}/stimfit
	cp -p $(srcdir)/charlie.py ${PYTHON_SITE_PKG}/stimfit
	cp -p $(srcdir)/hdf5tools.py ${PYTHON_SITE_PKG}/stimfit
	cp -p $(srcdir)/stf.py ${PYTHON_SITE_PKG}/stimfit
	cp -p $(srcdir)/spells.py ${PYTHON_SITE_PKG}/stimfit
	cp -p $(srcdir)/embedded_init.py ${PYTHON_SITE_PKG}/stimfit
	cp -p $(srcdir)/embedded_stf.py ${PYTHON_SITE_PKG}/stimfit
	cp -p $(srcdir)/embedded_ipython.py ${PYTHON_SITE_PKG}/stimfit
	echo stimfit > ${PYTHON_SITE_PKG}/stimfit.pth

endif
endif
