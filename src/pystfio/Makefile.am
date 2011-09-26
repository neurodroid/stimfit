pkglibdir = ${PYTHON_SITE_PKG}/stfio
pkglib_LTLIBRARIES = libpystfio.la

TESTS_ENVIRONMENT = cp ./test.h5 ./ && \
  cp .libs/libpystfio.so ./_stfio.so && \
  $(PYTHON)
TESTS = ./unittest_stfio.py

PYTHON_ADDINCLUDES = $(LIBPYTHON_INCLUDES)
PYTHON_ADDLDFLAGS = $(LIBPYTHON_LDFLAGS)
PYTHON_ADDLIBS = 

nodist_libpystfio_la_SOURCES = ./pystfio_wrap.cxx

libpystfio_la_SOURCES =  ./pystfio.cxx

noinst_HEADERS = ./pystfio.h

INCLUDES = $(LIBNUMPY_INCLUDES) $(PYTHON_ADDINCLUDES)

EXTRA_DIST = ./pystfio.i ./unittest_stfio.py ./test.h5

libpystfio_la_CPPFLAGS = $(SWIG_PYTHON_CPPFLAGS) -I$(top_srcdir)/src
libpystfio_la_CXXFLAGS = $(OPT_CXXFLAGS)
libpystfio_la_LDFLAGS = $(PYTHON_ADDLDFLAGS) $(LIBSTF_LDFLAGS)
libpystfio_la_LIBADD = $(PYTHON_ADDLIBS) ./../libstfio/libstfio.la

install-exec-hook:
	mv ${PYTHON_SITE_PKG}/stfio/${STFIO_PYTHON_LIBNAME} ${PYTHON_SITE_PKG}/stfio/_stfio.so
	rm -f ${PYTHON_SITE_PKG}/stfio/*.la
	rm -f ${PYTHON_SITE_PKG}/stfio/*.a
	cp -p $(top_srcdir)/src/pystfio/stfio_plot.py ${PYTHON_SITE_PKG}/stfio
	cp -p $(top_srcdir)/src/pystfio/stfio.py ${PYTHON_SITE_PKG}/stfio
	chmod -x ${PYTHON_SITE_PKG}/stfio/*
	echo stfio > ${PYTHON_SITE_PKG}/stfio.pth

uninstall-hook:
	rm -f ${PYTHON_SITE_PKG}/stfio/*.so
	rm -f ${PYTHON_SITE_PKG}/stfio.pth

clean-local:
	rm -f $(top_srcdir)/src/pystfio/pystfio_wrap.cxx $(top_srcdir)/src/pystfio/stfio.py
	rm -f test.h5
	rm -f _stfio.so

$(top_srcdir)/src/pystfio/pystfio_wrap.cxx $(top_srcdir)/src/pystfio/stfio.py: $(top_srcdir)/src/pystfio/pystfio.i
	$(SWIG) $(SWIG_PYTHON_OPT) -o $@ $<
	cat $(top_srcdir)/src/pystfio/gccwarn $(top_srcdir)/src/pystfio/pystfio_wrap.cxx > $(top_srcdir)/src/pystfio/pystfio_wrap_tmp.cxx
	mv $(top_srcdir)/src/pystfio/pystfio_wrap_tmp.cxx $(top_srcdir)/src/pystfio/pystfio_wrap.cxx
