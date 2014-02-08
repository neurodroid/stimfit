if ISDARWIN
PYTHON_TARGET_DIR=${PYTHON_DIST_PKG}
else
PYTHON_TARGET_DIR=${PYTHON_PRE_DIST_PKG}
# PYTHON_TARGET_DIR=${PYTHON_DIST_PKG}
endif
PYTHON_DEST_DIR=${DESTDIR}${PYTHON_TARGET_DIR}
pkglibdir = ${PYTHON_TARGET_DIR}/stfio

pkglib_LTLIBRARIES = libpystfio.la

TESTS_ENVIRONMENT = cp ./test.h5 ./ && \
  cp .libs/libpystfio.so ./_stfio.so && \
  $(PYTHON)
TESTS = ./unittest_stfio.py

PYTHON_ADDINCLUDES = $(LIBPYTHON_INCLUDES)
PYTHON_ADDLDFLAGS = $(LIBPYTHON_LDFLAGS)
PYTHON_ADDLIBS = 


$(top_srcdir)/src/pystfio/pystfio_wrap.cxx $(top_srcdir)/src/pystfio/stfio.py: $(top_srcdir)/src/pystfio/pystfio.i
	$(SWIG) $(SWIG_PYTHON_OPT) -o $@ $<
	cat $(top_srcdir)/src/stimfit/py/gccwarn $(top_srcdir)/src/pystfio/pystfio_wrap.cxx > $(top_srcdir)/src/pystfio/pystfio_wrap_tmp.cxx
	mv $(top_srcdir)/src/pystfio/pystfio_wrap_tmp.cxx $(top_srcdir)/src/pystfio/pystfio_wrap.cxx

nodist_libpystfio_la_SOURCES = $(top_srcdir)/src/pystfio/pystfio_wrap.cxx

libpystfio_la_SOURCES =  $(top_srcdir)/src/pystfio/pystfio.cxx

noinst_HEADERS = $(top_srcdir)/src/pystfio/pystfio.h

INCLUDES = $(LIBNUMPY_INCLUDES) $(PYTHON_ADDINCLUDES)

EXTRA_DIST = ./pystfio.i ./unittest_stfio.py ./test.h5

libpystfio_la_CPPFLAGS = $(SWIG_PYTHON_CPPFLAGS) -I$(top_srcdir)/src
libpystfio_la_CXXFLAGS = $(OPT_CXXFLAGS)
libpystfio_la_LDFLAGS = $(PYTHON_ADDLDFLAGS) $(LIBSTF_LDFLAGS)
libpystfio_la_LIBADD = $(PYTHON_ADDLIBS) ./../libstfio/libstfio.la

LTTARGET=${PYTHON_DIST_PKG}/stfio
install-exec-hook:
	mv ${PYTHON_DEST_DIR}/stfio/${STFIO_PYTHON_LIBNAME} ${PYTHON_DEST_DIR}/stfio/_stfio.so
if BUILD_DEBIAN
	chrpath -r $(LTTARGET) ${PYTHON_DEST_DIR}/stfio/_stfio.so
endif
	rm -f ${PYTHON_DEST_DIR}/stfio/*.la
	rm -f ${PYTHON_DEST_DIR}/stfio/*.a
	cp -p $(top_srcdir)/src/pystfio/__init__.py ${PYTHON_DEST_DIR}/stfio
	cp -p $(top_srcdir)/src/pystfio/stfio_plot.py ${PYTHON_DEST_DIR}/stfio
	cp -p $(top_srcdir)/src/pystfio/stfio.py ${PYTHON_DEST_DIR}/stfio
	${PYTHON} -m compileall -l ${PYTHON_DEST_DIR}/stfio
	chmod -x ${PYTHON_DEST_DIR}/stfio/*
#       echo stfio > ${PYTHON_DEST_DIR}/stfio.pth

uninstall-hook:
	rm -f ${PYTHON_DEST_DIR}/stfio/*.so
	rm -f ${PYTHON_DEST_DIR}/stfio.pth
	rm -r ${PYTHON_DEST_DIR}/stfio

clean-local:
	rm -f $(top_srcdir)/src/pystfio/pystfio_wrap.cxx $(top_srcdir)/src/pystfio/stfio.py
	rm -f test.h5
	rm -f _stfio.so
