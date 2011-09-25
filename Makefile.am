SUBDIRS = src

if BUILD_MODULE

pkglibdir = ${PYTHON_SITE_PKG}/stfio
pkglib_LTLIBRARIES = libstfio.la

TESTS_ENVIRONMENT = cp $(srcdir)/src/stfswig/test.h5 ./ && \
  cp .libs/libstfio.so ./_stfio.so && \
  $(PYTHON)
TESTS = ./src/stfswig/unittest_stfio.py

PYTHON_ADDINCLUDES = $(LIBPYTHON_INCLUDES)
PYTHON_ADDLDFLAGS = $(LIBPYTHON_LDFLAGS)
PYTHON_ADDLIBS = 

nodist_libstfio_la_SOURCES = ./src/stfswig/stfioswig_wrap.cxx

libstfio_la_SOURCES =  ./src/stfswig/stfioswig.cxx \
	./src/core/channel.cpp ./src/core/section.cpp ./src/core/recording.cpp \
	./src/core/core.cpp \
	./src/core/stfio.cpp \
	./src/core/filelib/cfslib.cpp ./src/core/filelib/cfs.c ./src/core/filelib/hdf5lib.cpp \
	./src/core/filelib/abflib.cpp \
	./src/core/filelib/axon/AxAbfFio32/abffiles.cpp \
	./src/core/filelib/axon/AxAbfFio32/csynch.cpp \
	./src/core/filelib/axon/AxAbfFio32/filedesc.cpp \
	./src/core/filelib/axon/Common/FileReadCache.cpp \
	./src/core/filelib/axon/Common/FileIO.cpp \
	./src/core/filelib/axon/AxAbfFio32/abfheadr.cpp \
	./src/core/filelib/axon/AxAbfFio32/abfhwave.cpp \
	./src/core/filelib/axon/AxAbfFio32/Oldheadr.cpp \
	./src/core/filelib/axon/AxAbfFio32/abfutil.cpp \
	./src/core/filelib/axon/AxAbfFio32/msbincvt.cpp \
	./src/core/filelib/axon/Common/unix.cpp \
	./src/core/filelib/axon/AxAbfFio32/abferror.cpp \
        ./src/core/filelib/axon/AxAtfFio32/axatffio32.cpp \
        ./src/core/filelib/axon/AxAtfFio32/fileio2.cpp \
        ./src/core/filelib/axon2/ProtocolReaderABF2.cpp \
        ./src/core/filelib/axon2/SimpleStringCache.cpp \
	./src/core/filelib/axon2/abf2headr.cpp \
	./src/core/filelib/atflib.cpp ./src/core/filelib/axglib.cpp \
	./src/core/filelib/axg/AxoGraph_ReadWrite.cpp \
	./src/core/filelib/axg/fileUtils.cpp \
	./src/core/filelib/axg/stringUtils.cpp \
	./src/core/filelib/axg/byteswap.cpp \
        ./src/core/filelib/hekalib.cpp \
	./src/core/filelib/igorlib.cpp \
	./src/core/filelib/igor/CrossPlatformFileIO.c \
	./src/core/filelib/igor/WriteWave.c

if WITH_BIOSIG
libstfio_la_SOURCES += ./src/core/filelib/biosiglib.cpp
endif
# 
noinst_HEADERS = ./src/stfswig/stfioswig.h ./src/core/channel.h ./src/core/section.h ./src/core/recording.h ./src/core/core.h \
            ./src/core/stimdefs.h \
            ./src/core/stfio.h \
            ./src/core/filelib/cfslib.h ./src/core/filelib/hdf5lib.h\
            ./src/core/filelib/abflib.h ./src/core/filelib/atflib.h ./src/core/filelib/axglib.h \
            ./src/core/filelib/cfs.h ./src/core/filelib/machine.h \
            ./src/core/filelib/hekalib.h \
            ./src/core/filelib/axon/AxAbfFio32/abffiles.h \
            ./src/core/filelib/axon/AxAbfFio32/csynch.hpp \
            ./src/core/filelib/axon/AxAbfFio32/filedesc.hpp \
            ./src/core/filelib/axon/Common/FileReadCache.hpp \
            ./src/core/filelib/axon/Common/FileIO.hpp \
            ./src/core/filelib/axon/AxAbfFio32/abfheadr.h \
            ./src/core/filelib/axon/AxAbfFio32/oldheadr.h \
            ./src/core/filelib/axon/AxAbfFio32/abfutil.h \
            ./src/core/filelib/axon/AxAbfFio32/msbincvt.h \
            ./src/core/filelib/axon/Common/unix.h \
            ./src/core/filelib/axon/Common/axodefn.h \
            ./src/core/filelib/axon/Common/axodebug.h \
            ./src/core/filelib/axon/Common/wincpp.hpp \
            ./src/core/filelib/axon/AxAbfFio32/AxAbffio32.h \
            ./src/core/filelib/axon/AxAbfFio32/abfoldnx.h \
            ./src/core/filelib/axon/Common/resource.h \
            ./src/core/filelib/axon/AxAtfFio32/axatffio32.h \
            ./src/core/filelib/axon/AxAtfFio32/atfutil.h \
            ./src/core/filelib/axon/AxAtfFio32/atfintl.h \
            ./src/core/filelib/axon/Common/colors.h \
            ./src/core/filelib/axon/Common/adcdac.h \
            ./src/core/filelib/axon/Common/ArrayPtr.hpp \
            ./src/core/filelib/axon/Common/wincpp.hpp \
            ./src/core/filelib/axon2/ProtocolReaderABF2.hpp \
            ./src/core/filelib/axon2/SimpleStringCache.hpp \
            ./src/core/filelib/axon2/ProtocolStructs.h \
	    ./src/core/filelib/axon2/abf2headr.h \
            ./src/core/filelib/axg/AxoGraph_ReadWrite.h \
            ./src/core/filelib/axg/fileUtils.h \
            ./src/core/filelib/axg/stringUtils.h \
            ./src/core/filelib/axg/byteswap.h \
            ./src/core/filelib/axg/longdef.h \
            ./src/core/filelib/igorlib.h \
            ./src/core/filelib/igor/CrossPlatformFileIO.h \
            ./src/core/filelib/igor/IgorBin.h

if WITH_BIOSIG
    noinst_HEADERS += ./src/core/filelib/biosiglib.h
endif

INCLUDES = $(LIBNUMPY_INCLUDES) $(PYTHON_ADDINCLUDES)

EXTRA_DIST = ./src/stfswig/stfioswig.i ./src/stfswig/unittest_stfio.py ./src/stfswig/hdf5tools.py ./src/stfswig/test.h5

libstfio_la_CPPFLAGS = $(SWIG_PYTHON_CPPFLAGS) -I$(top_srcdir)/src
libstfio_la_CXXFLAGS = $(OPT_CXXFLAGS)
libstfio_la_LDFLAGS = $(PYTHON_ADDLDFLAGS) $(LIBSTF_LDFLAGS)
libstfio_la_LIBADD = -lhdf5 -lhdf5_hl $(PYTHON_ADDLIBS)

install-exec-hook:
	mv ${PYTHON_SITE_PKG}/stfio/${STFIO_PYTHON_LIBNAME} ${PYTHON_SITE_PKG}/stfio/_stfio.so
	rm -f ${PYTHON_SITE_PKG}/stfio/*.la
	rm -f ${PYTHON_SITE_PKG}/stfio/*.a
	cp -p $(srcdir)/src/stfswig/hdf5tools.py ${PYTHON_SITE_PKG}/stfio
	cp -p $(srcdir)/src/stfswig/stfio_plot.py ${PYTHON_SITE_PKG}/stfio
	cp -p $(srcdir)/src/stfswig/stfio.py ${PYTHON_SITE_PKG}/stfio
	chmod -x ${PYTHON_SITE_PKG}/stfio/*
	echo stfio > ${PYTHON_SITE_PKG}/stfio.pth

uninstall-hook:
	rm -f ${PYTHON_SITE_PKG}/stfio/_stfio.so
	rm -f ${PYTHON_SITE_PKG}/stfio.pth

clean-local:
	rm -f $(srcdir)/src/stfswig/stfioswig_wrap.cxx $(srcdir)/src/stfswig/stfio.py
	rm -f test.h5
	rm -f _stfio.so

$(srcdir)/src/stfswig/stfioswig_wrap.cxx $(srcdir)/src/stfswig/stfio.py: $(srcdir)/src/stfswig/stfioswig.i
	$(SWIG) $(SWIG_PYTHON_OPT) -o $@ $<
	cat $(srcdir)/src/stfswig/gccwarn $(srcdir)/src/stfswig/stfioswig_wrap.cxx > $(srcdir)/src/stfswig/stfioswig_wrap_tmp.cxx
	mv $(srcdir)/src/stfswig/stfioswig_wrap_tmp.cxx $(srcdir)/src/stfswig/stfioswig_wrap.cxx

else
bin_PROGRAMS = stimfit
check_PROGRAMS = stimfittest
TESTS = ${check_PROGRAMS}
stimfit_SOURCES = src/app/main.cpp

stimfittest_SOURCES = ./src/test/section.cpp ./src/test/channel.cpp ./src/test/recording.cpp ./src/test/measlib.cpp \
            ./src/test/gtest/src/gtest-all.cc ./src/test/gtest/src/gtest_main.cc

noinst_HEADERS = ./src/core/channel.h ./src/core/section.h ./src/core/recording.h ./src/core/core.h \
            ./src/core/stimdefs.h ./src/core/stfio.h ./src/core/fitlib.h ./src/core/spline.h ./src/core/measlib.h \
            ./src/core/filelib/asciilib.h ./src/core/filelib/cfslib.h ./src/core/filelib/hdf5lib.h\
            ./src/core/filelib/abflib.h ./src/core/filelib/atflib.h ./src/core/filelib/axglib.h \
            ./src/core/filelib/cfs.h ./src/core/filelib/machine.h \
            ./src/core/filelib/hekalib.h \
            ./src/core/filelib/axon/AxAbfFio32/abffiles.h \
            ./src/core/filelib/axon/AxAbfFio32/csynch.hpp \
            ./src/core/filelib/axon/AxAbfFio32/filedesc.hpp \
            ./src/core/filelib/axon/Common/FileReadCache.hpp \
            ./src/core/filelib/axon/Common/FileIO.hpp \
            ./src/core/filelib/axon/AxAbfFio32/abfheadr.h \
            ./src/core/filelib/axon/AxAbfFio32/oldheadr.h \
            ./src/core/filelib/axon/AxAbfFio32/abfutil.h \
            ./src/core/filelib/axon/AxAbfFio32/msbincvt.h \
            ./src/core/filelib/axon/Common/unix.h \
            ./src/core/filelib/axon/Common/axodefn.h \
            ./src/core/filelib/axon/Common/axodebug.h \
            ./src/core/filelib/axon/Common/wincpp.hpp \
            ./src/core/filelib/axon/AxAbfFio32/AxAbffio32.h \
            ./src/core/filelib/axon/AxAbfFio32/abfoldnx.h \
            ./src/core/filelib/axon/Common/resource.h \
            ./src/core/filelib/axon/AxAtfFio32/axatffio32.h \
            ./src/core/filelib/axon/AxAtfFio32/atfutil.h \
            ./src/core/filelib/axon/AxAtfFio32/atfintl.h \
            ./src/core/filelib/axon/Common/colors.h \
            ./src/core/filelib/axon/Common/adcdac.h \
            ./src/core/filelib/axon/Common/ArrayPtr.hpp \
            ./src/core/filelib/axon/Common/wincpp.hpp \
            ./src/core/filelib/axon2/ProtocolReaderABF2.hpp \
            ./src/core/filelib/axon2/SimpleStringCache.hpp \
            ./src/core/filelib/axon2/ProtocolStructs.h \
	    ./src/core/filelib/axon2/abf2headr.h \
            ./src/core/filelib/axg/AxoGraph_ReadWrite.h \
            ./src/core/filelib/axg/fileUtils.h \
            ./src/core/filelib/axg/stringUtils.h \
            ./src/core/filelib/axg/byteswap.h \
            ./src/core/filelib/axg/longdef.h \
            ./src/core/filelib/igorlib.h \
            ./src/core/filelib/igor/CrossPlatformFileIO.h \
            ./src/core/filelib/igor/IgorBin.h \
            ./src/core/levmar/lm.h ./src/core/levmar/levmar.h ./src/core/levmar/misc.h ./src/core/levmar/compiler.h \
            ./src/app/app.h ./src/app/stfcheckbox.h \
            ./src/app/copygrid.h ./src/app/graph.h \
            ./src/app/printout.h \
            ./src/app/doc.h ./src/app/parentframe.h ./src/app/childframe.h ./src/app/view.h \
            ./src/app/table.h ./src/app/zoom.h \
            ./src/app/dlgs/cursorsdlg.h ./src/app/dlgs/eventdlg.h \
	    ./src/app/dlgs/fitseldlg.h ./src/app/dlgs/smalldlgs.h \
            ./src/app/funclib/funclib.h \
            ./src/app/usrdlg/usrdlg.h \
            ./src/test/gtest/include/gtest/gtest-death-test.h \
            ./src/test/gtest/include/gtest/gtest-message.h \
            ./src/test/gtest/include/gtest/gtest-param-test.h.pump \
            ./src/test/gtest/include/gtest/gtest-printers.h \
            ./src/test/gtest/include/gtest/gtest-spi.h \
            ./src/test/gtest/include/gtest/gtest-typed-test.h \
            ./src/test/gtest/include/gtest/gtest.h \
            ./src/test/gtest/include/gtest/gtest-param-test.h	\
            ./src/test/gtest/include/gtest/gtest_pred_impl.h \
            ./src/test/gtest/include/gtest/gtest_prod.h \
            ./src/test/gtest/include/gtest/gtest-test-part.h  \
            ./src/test/gtest/include/gtest/internal/gtest-death-test-internal.h \
            ./src/test/gtest/include/gtest/internal/gtest-linked_ptr.h \
            ./src/test/gtest/include/gtest/internal/gtest-param-util.h \
            ./src/test/gtest/include/gtest/internal/gtest-tuple.h \
            ./src/test/gtest/include/gtest/internal/gtest-type-util.h.pump \
            ./src/test/gtest/include/gtest/internal/gtest-filepath.h \
            ./src/test/gtest/include/gtest/internal/gtest-param-util-generated.h \
            ./src/test/gtest/include/gtest/internal/gtest-port.h \
            ./src/test/gtest/include/gtest/internal/gtest-tuple.h.pump \
            ./src/test/gtest/include/gtest/internal/gtest-internal.h \
            ./src/test/gtest/include/gtest/internal/gtest-param-util-generated.h.pump \
            ./src/test/gtest/include/gtest/internal/gtest-string.h \
            ./src/test/gtest/include/gtest/internal/gtest-type-util.h \
            ./src/test/gtest/src/gtest-internal-inl.h

if WITH_BIOSIG
    noinst_HEADERS += ./src/core/filelib/biosiglib.h
endif

EXTRA_DIST = ./src/icons/16-em-down.xpm
EXTRA_DIST+= ./src/icons/16-em-open.xpm
EXTRA_DIST+= ./src/icons/accept.xpm
EXTRA_DIST+= ./src/icons/arrow_down.xpm
EXTRA_DIST+= ./src/icons/arrow_left.xpm
EXTRA_DIST+= ./src/icons/arrow_out.xpm
EXTRA_DIST+= ./src/icons/arrow_right.xpm
EXTRA_DIST+= ./src/icons/arrow_up.xpm
EXTRA_DIST+= ./src/icons/camera.xpm
EXTRA_DIST+= ./src/icons/ch1.xpm
EXTRA_DIST+= ./src/icons/ch2.xpm
EXTRA_DIST+= ./src/icons/cursor.xpm
EXTRA_DIST+= ./src/icons/event.xpm
EXTRA_DIST+= ./src/icons/fit.xpm
EXTRA_DIST+= ./src/icons/fit_lim.xpm
EXTRA_DIST+= ./src/icons/latency_lim.xpm
EXTRA_DIST+= ./src/icons/resultset_first.xpm
EXTRA_DIST+= ./src/icons/resultset_last.xpm
EXTRA_DIST+= ./src/icons/resultset_next.xpm
EXTRA_DIST+= ./src/icons/resultset_previous.xpm
EXTRA_DIST+= ./src/icons/stimfit.png
EXTRA_DIST+= ./src/icons/sum_new.xpm
EXTRA_DIST+= ./src/icons/sum_new_aligned.xpm
EXTRA_DIST+= ./src/icons/table.xpm
EXTRA_DIST+= ./src/icons/zoom.xpm
EXTRA_DIST+= ./src/icons/zoom_in.xpm
EXTRA_DIST+= ./src/icons/zoom_out.xpm
EXTRA_DIST+= ./src/core/levmar/Axb_core.c
EXTRA_DIST+= ./src/core/levmar/lmbc_core.c
EXTRA_DIST+= ./src/core/levmar/lm_core.c
EXTRA_DIST+= ./src/core/levmar/lmlec_core.c
EXTRA_DIST+= ./src/core/levmar/misc_core.c
EXTRA_DIST+= ./src/core/levmar/LICENSE
EXTRA_DIST+= ./src/core/levmar/README.txt
EXTRA_DIST+= ./Doxyfile
EXTRA_DIST+= ./acsite.m4
EXTRA_DIST+= ./autogen.sh
EXTRA_DIST+= ./macosx/scripts/conf_mac_release.sh
EXTRA_DIST+= ./macosx/scripts/change_deps_release.sh
EXTRA_DIST+= ./macosx/app.r
EXTRA_DIST+= ./macosx/stimfit.icns
# EXTRA_DIST+= ./src/stfswig/stfioswig_wrap.cxx 
# EXTRA_DIST+= ./src/stfswig/stfio.py 
EXTRA_DIST+= ./src/stfswig/stfioswig.cxx 
EXTRA_DIST+= ./src/stfswig/stfioswig.i 
EXTRA_DIST+= ./src/stfswig/stfio_plot.py 
EXTRA_DIST+= ./src/stfswig/unittest_stfio.py
EXTRA_DIST+= ./src/stfswig/test.h5
EXTRA_DIST+= ./src/stfswig/stfioswig.h 
# EXTRA_DIST+= ./debian/changelog
EXTRA_DIST+= ./debian/compat
EXTRA_DIST+= ./debian/control
EXTRA_DIST+= ./debian/copyright
EXTRA_DIST+= ./debian/docs
EXTRA_DIST+= ./debian/mkdeb.sh
EXTRA_DIST+= ./debian/python-stfio.files
EXTRA_DIST+= ./debian/python-stfio.install
EXTRA_DIST+= ./debian/rules
EXTRA_DIST+= ./debian/stimfit.1
EXTRA_DIST+= ./debian/stimfit.desktop
EXTRA_DIST+= ./debian/stimfit.files
EXTRA_DIST+= ./debian/stimfit.install
EXTRA_DIST+= ./debian/stimfit.menu
EXTRA_DIST+= ./src/test/gtest/CHANGES
EXTRA_DIST+= ./src/test/gtest/CONTRIBUTORS
EXTRA_DIST+= ./src/test/gtest/COPYING
EXTRA_DIST+= ./src/test/gtest/README
EXTRA_DIST+= ./src/test/gtest/src/gtest.cc
EXTRA_DIST+= ./src/test/gtest/src/gtest-death-test.cc
EXTRA_DIST+= ./src/test/gtest/src/gtest-filepath.cc
EXTRA_DIST+= ./src/test/gtest/src/gtest-port.cc
EXTRA_DIST+= ./src/test/gtest/src/gtest-printers.cc
EXTRA_DIST+= ./src/test/gtest/src/gtest-test-part.cc
EXTRA_DIST+= ./src/test/gtest/src/gtest-typed-test.cc

if BUILD_PYTHON
    PYTHON_ADDINCLUDES = $(LIBPYTHON_INCLUDES)
    PYTHON_ADDLDFLAGS = $(LIBPYTHON_LDFLAGS)
    PYTHON_ADDLIBS = ./src/stfswig/libstf.la
else
    PYTHON_ADDLDFLAGS = 
    PYTHON_ADDLIBS = 
    PYTHON_ADDINCLUDES = 
endif

INCLUDES = $(PYTHON_ADDINCLUDES)

stimfit_CXXFLAGS = $(OPT_CXXFLAGS)
stimfit_LDFLAGS = $(LIBLAPACK_LDFLAGS) $(PYTHON_ADDLDFLAGS) $(LIBSTF_LDFLAGS) $(LIBBIOSIG_LDFLAGS)
stimfit_LDADD = $(WX_LIBS) -lfftw3 -lhdf5 -lhdf5_hl ./src/app/libstimfit.la # $(PYTHON_ADDLIBS) 

stimfittest_CXXFLAGS = $(GT_CXXFLAGS)
stimfittest_CPPFLAGS = ${CPPFLAGS} $(GT_CPPFLAGS) -DSTF_TEST -I$(top_srcdir)/src/test/gtest -I$(top_srcdir)/src/test/gtest/include
stimfittest_LDFLAGS = $(LIBLAPACK_LDFLAGS) $(PYTHON_ADDLDFLAGS) $(GT_LDFLAGS)
stimfittest_LDADD = $(WX_LIBS) $(PYTHON_ADDLIBS) $(GT_LIBS) -lfftw3 -lhdf5 -lhdf5_hl ./src/app/libstimfit.la

if BUILD_DEBIAN
LTTARGET=/usr/lib/stimfit
install-exec-hook:
	$(LIBTOOL) --finish $(LTTARGET)
	chrpath -r $(LTTARGET) $(prefix)/bin/stimfit
	chrpath -r $(LTTARGET) $(prefix)/lib/stimfit/libstf.so
	chrpath -r $(LTTARGET) $(prefix)/lib/stimfit/libstimfit.so
	mkdir -p $(prefix)/share/pixmaps
	mkdir -p $(prefix)/share/applications
	cp $(top_srcdir)/src/icons/stimfit.png $(prefix)/share/pixmaps/stimfit.png
	cp $(top_srcdir)/debian/stimfit.desktop $(prefix)/share/applications/
else
LTTARGET=$(prefix)/lib/stimfit
install-exec-hook:
	$(LIBTOOL) --finish $(LTTARGET)
	chrpath -r $(LTTARGET) $(prefix)/bin/stimfit
	mkdir -p $(prefix)/share/pixmaps
	mkdir -p $(prefix)/share/applications
	cp $(top_srcdir)/src/icons/stimfit.png $(prefix)/share/pixmaps/stimfit.png
	cp $(top_srcdir)/debian/stimfit.desktop $(prefix)/share/applications/
endif

uninstall-hook:
	rm -f $(prefix)/share/pixmaps/stimfit.png
	rm -f $(prefix)/share/applications/stimfit.desktop

# wxMac resource fork/unbundled app
stimfit.app: stimfit
	mkdir -p stimfit.app/Contents
	mkdir -p stimfit.app/Contents/MacOS
	mkdir -p stimfit.app/Contents/Resources
	mkdir -p stimfit.app/Contents/Resources/English.lproj
	cp $(top_srcdir)/stimfit.plist.in stimfit.app/Contents/Info.plist
	echo "APPL????\c" > stimfit.app/Contents/PkgInfo
	rm -f stimfit.app/Contents/MacOS/stimfit$(EXEEXT)
if BUILD_PYTHON
	cp -p -f .libs/stimfit$(EXEEXT) stimfit.app/Contents/MacOS/stimfit$(EXEEXT)
else
	cp -p -f ./stimfit$(EXEEXT) stimfit.app/Contents/MacOS/stimfit$(EXEEXT)
endif
	$(POSTLINK_COMMAND) stimfit.app/Contents/MacOS/stimfit$(EXEEXT) \
	                    $(srcdir)/macosx/app.r
	$(MACSETFILE) -a C stimfit.app/Contents/MacOS/stimfit$(EXEEXT)
	cp -f $(top_srcdir)/macosx/stimfit.icns stimfit.app/Contents/Resources/stimfit.icns
endif
