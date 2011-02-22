if BUILD_MODULE

else
if BUILD_PYTHON

pkglib_LTLIBRARIES = libstf.la

libstf_la_SOURCES = $(srcdir)/stfswig_wrap.cxx $(srcdir)/stfswig.cxx # $(SWIG_SOURCES)
noinst_HEADERS = stfswig.h

$(srcdir)/stfswig_wrap.cxx : $(srcdir)/stfswig.i
	$(SWIG) $(SWIG_PYTHON_OPT) -o $@ $<

INCLUDES = $(LIBNUMPY_INCLUDES)

libstf_la_CPPFLAGS = $(SWIG_PYTHON_CPPFLAGS) -I$(top_srcdir)/src
libstf_la_CXXFLAGS = $(OPT_CXXFLAGS)
libstf_la_LDFLAGS = $(LIBPYTHON_LDFLAGS) $(LIBSTF_LDFLAGS) $(LIBBIOSIG_LDFLAGS)
libstf_la_LIBADD = $(WX_LIBS)  ../../Axb.o ../../stimfit-abfheadr.o ../../stimfit-cfslib.o ../../stimfit-FileIO.o ../../stimfit-Oldheadr.o ../../stimfit-table.o \
 ../../cfs.o ../../stimfit-abfhwave.o ../../stimfit-channel.o ../../stimfit-FileReadCache.o ../../stimfit-parentframe.o \
 ../../stimfit-abflib.o ../../stimfit-childframe.o ../../stimfit-fileUtils.o \
 ../../stimfit-abfutil.o ../../stimfit-copygrid.o ../../stimfit-fitlib.o ../../stimfit-printout.o ../../stimfit-unix.o \
 ../../stimfit-core.o ../../stimfit-fitseldlg.o ../../stimfit-ProtocolReaderABF2.o ../../stimfit-usrdlg.o \
 ../../lmbc.o ../../stimfit-asciilib.o ../../stimfit-csynch.o ../../stimfit-funclib.o ../../stimfit-recording.o ../../stimfit-view.o \
 ../../lmlec.o ../../stimfit-atflib.o ../../stimfit-cursorsdlg.o ../../stimfit-graph.o ../../stimfit-section.o ../../stimfit-zoom.o \
 ../../lm.o ../../stimfit-axatffio32.o ../../stimfit-unopt.o ../../stimfit-doc.o ../../stimfit-app.o ../../stimfit-hdf5lib.o ../../stimfit-SimpleStringCache.o \
 ../../stimfit-abf2headr.o ../../stimfit-axglib.o ../../stimfit-eventdlg.o ../../stimfit-hekalib.o ../../stimfit-smalldlgs.o \
 ../../misc.o ../../stimfit-abferror.o ../../stimfit-AxoGraph_ReadWrite.o ../../stimfit-filedesc.o ../../stimfit-measlib.o ../../stimfit-stfcheckbox.o \
 ../../stimfit-abffiles.o ../../stimfit-byteswap.o ../../stimfit-fileio2.o ../../stimfit-msbincvt.o ../../stimfit-stringUtils.o

EXTRA_DIST = stfswig.i ivtools.py mintools.py natools.py minidemo.py charlie.py stf.py mintools.py hdf5tools.py spells.py embedded_init.py embedded_stf.py embedded_mpl.py embedded_ipython.py unittest_stfio.py heka.py extensions.py Doxyfile

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

endif
endif
