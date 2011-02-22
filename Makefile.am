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

libstfio_la_SOURCES = ./src/stfswig/stfioswig_wrap.cxx ./src/stfswig/stfioswig.cxx \
	./src/core/channel.cpp ./src/core/section.cpp ./src/core/recording.cpp \
	./src/core/core.cpp \
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
        ./src/core/filelib/hekalib.cpp

if WITH_BIOSIG
libstfio_la_SOURCES += ./src/core/filelib/biosiglib.cpp
endif
# 
noinst_HEADERS = ./src/stfswig/stfioswig.h ./src/core/channel.h ./src/core/section.h ./src/core/recording.h ./src/core/core.h \
            ./src/core/stimdefs.h \
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
            ./src/core/filelib/axg/longdef.h

if WITH_BIOSIG
    noinst_HEADERS += ./src/core/filelib/biosiglib.h
endif

$(srcdir)/src/stfswig/stfioswig_wrap.cxx : ./src/stfswig/stfioswig.i
	$(SWIG) $(SWIG_PYTHON_OPT) -o $@ $<

INCLUDES = $(LIBNUMPY_INCLUDES) $(PYTHON_ADDINCLUDES)

EXTRA_DIST = ./src/stfswig/stfioswig.i ./src/stfswig/stfio.py ./src/stfswig/unittest_stfio.py ./src/stfswig/hdf5tools.py ./src/stfswig/test.h5

libstfio_la_CPPFLAGS = $(SWIG_PYTHON_CPPFLAGS) -I$(top_srcdir)/src
libstfio_la_CXXFLAGS = $(OPT_CXXFLAGS)
libstfio_la_LDFLAGS = $(PYTHON_ADDLDFLAGS) $(LIBSTF_LDFLAGS)
libstfio_la_LIBADD = -lhdf5 -lhdf5_hl $(PYTHON_ADDLIBS)

install-exec-hook:
	mv ${PYTHON_SITE_PKG}/stfio/libstfio.so ${PYTHON_SITE_PKG}/stfio/_stfio.so
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
	rm -f test.h5
	rm -f _stfio.so

else
bin_PROGRAMS = stimfit
check_PROGRAMS = stimfittest
TESTS = ${check_PROGRAMS}
stimfit_SOURCES =  ./src/core/channel.cpp ./src/core/section.cpp ./src/core/recording.cpp \
	    ./src/core/core.cpp ./src/core/fitlib.cpp ./src/core/measlib.cpp ./src/core/filelib/asciilib.cpp \
	    ./src/core/filelib/cfslib.cpp ./src/core/filelib/hdf5lib.cpp ./src/core/filelib/cfs.c \
            ./src/core/filelib/abflib.cpp ./src/core/filelib/atflib.cpp ./src/core/filelib/axglib.cpp \
            ./src/core/filelib/hekalib.cpp \
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
            ./src/core/filelib/axg/AxoGraph_ReadWrite.cpp \
            ./src/core/filelib/axg/fileUtils.cpp \
            ./src/core/filelib/axg/stringUtils.cpp \
            ./src/core/filelib/axg/byteswap.cpp \
            ./src/core/levmar/lm.c ./src/core/levmar/Axb.c ./src/core/levmar/misc.c ./src/core/levmar/lmlec.c ./src/core/levmar/lmbc.c \
            ./src/app/app.cpp ./src/app/unopt.cpp ./src/app/doc.cpp ./src/app/stfcheckbox.cpp ./src/app/copygrid.cpp ./src/app/graph.cpp \
            ./src/app/printout.cpp ./src/app/parentframe.cpp ./src/app/childframe.cpp ./src/app/view.cpp ./src/app/table.cpp ./src/app/zoom.cpp \
            ./src/app/dlgs/cursorsdlg.cpp ./src/app/dlgs/eventdlg.cpp \
	    ./src/app/dlgs/fitseldlg.cpp ./src/app/dlgs/smalldlgs.cpp \
            ./src/app/funclib/funclib.cpp \
            ./src/app/usrdlg/usrdlg.cpp

if WITH_BIOSIG
stimfit_SOURCES += ./src/core/filelib/biosiglib.cpp
endif

stimfittest_SOURCES = ./src/app/app.cpp ./src/test/section.cpp ./src/test/channel.cpp ./src/test/recording.cpp ./src/test/measlib.cpp

noinst_HEADERS = ./src/core/channel.h ./src/core/section.h ./src/core/recording.h ./src/core/core.h \
            ./src/core/stimdefs.h ./src/core/fitlib.h ./src/core/spline.h ./src/core/measlib.h \
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
            ./src/core/levmar/lm.h ./src/core/levmar/misc.h \
            ./src/app/app.h ./src/app/stfcheckbox.h \
            ./src/app/copygrid.h ./src/app/graph.h \
            ./src/app/printout.h \
            ./src/app/doc.h ./src/app/parentframe.h ./src/app/childframe.h ./src/app/view.h \
            ./src/app/table.h ./src/app/zoom.h \
            ./src/app/dlgs/cursorsdlg.h ./src/app/dlgs/eventdlg.h \
	    ./src/app/dlgs/fitseldlg.h ./src/app/dlgs/smalldlgs.h \
            ./src/app/funclib/funclib.h \
            ./src/app/usrdlg/usrdlg.h

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
EXTRA_DIST+= ./src/icons/sum_new.xpm
EXTRA_DIST+= ./src/icons/sum_new_aligned.xpm
EXTRA_DIST+= ./src/icons/table.xpm
EXTRA_DIST+= ./src/icons/zoom.xpm
EXTRA_DIST+= ./src/icons/zoom_in.xpm
EXTRA_DIST+= ./src/icons/zoom_out.xpm
EXTRA_DIST+= ./src/core/levmar/Axb_core.c
EXTRA_DIST+= ./src/core/levmar/lmbc_core.c
EXTRA_DIST+= ./src/core/levmar/lm_core.c
EXTRA_DIST+= ./src/core/levmar/lmdemo.c
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
EXTRA_DIST+= ./macosx/wxmac.icns
EXTRA_DIST+= ./src/stfswig/stfioswig_wrap.cxx 
EXTRA_DIST+= ./src/stfswig/stfioswig.cxx 
EXTRA_DIST+= ./src/stfswig/stfioswig.i 
EXTRA_DIST+= ./src/stfswig/stfio.py 
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
EXTRA_DIST+= ./debian/stimfit.files
EXTRA_DIST+= ./debian/stimfit.install

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
stimfit_LDADD = $(WX_LIBS) $(PYTHON_ADDLIBS) -lfftw3 -lhdf5 -lhdf5_hl
stimfittest_CXXFLAGS = $(GT_CXXFLAGS)
stimfittest_CPPFLAGS = ${CPPFLAGS} $(GT_CPPFLAGS) -DSTF_TEST
stimfittest_LDFLAGS = $(LIBLAPACK_LDFLAGS) $(PYTHON_ADDLDFLAGS) $(GT_LDFLAGS)

# stimfit_OBJECTS won't work here because stimfit-app.o has a main() function
# this is why *.o files are explicitly listed here
stimfittest_LDADD = $(WX_LIBS) $(PYTHON_ADDLIBS) -lgtest_main -lfftw3 -lhdf5 -lhdf5_hl \
 Axb.o stimfit-abfheadr.o stimfit-cfslib.o stimfit-FileIO.o stimfit-Oldheadr.o stimfit-table.o \
 cfs.o stimfit-abfhwave.o stimfit-channel.o stimfit-FileReadCache.o stimfit-parentframe.o \
 stimfit-abflib.o stimfit-childframe.o stimfit-fileUtils.o \
 stimfit-abfutil.o stimfit-copygrid.o stimfit-fitlib.o stimfit-printout.o stimfit-unix.o \
 stimfit-core.o stimfit-fitseldlg.o stimfit-ProtocolReaderABF2.o stimfit-usrdlg.o \
 lmbc.o stimfit-asciilib.o stimfit-csynch.o stimfit-funclib.o stimfit-recording.o stimfit-view.o \
 lmlec.o stimfit-atflib.o stimfit-cursorsdlg.o stimfit-graph.o stimfit-section.o stimfit-zoom.o \
 lm.o stimfit-axatffio32.o stimfit-unopt.o stimfit-doc.o stimfit-hdf5lib.o stimfit-SimpleStringCache.o \
 stimfit-abf2headr.o stimfit-axglib.o stimfit-eventdlg.o stimfit-hekalib.o stimfit-smalldlgs.o \
 misc.o stimfit-abferror.o stimfit-AxoGraph_ReadWrite.o stimfit-filedesc.o stimfit-measlib.o stimfit-stfcheckbox.o \
 stimfit-abffiles.o stimfit-byteswap.o stimfit-fileio2.o stimfit-msbincvt.o stimfit-stringUtils.o

if BUILD_DEBIAN
  LTTARGET=/usr/lib/stimfit
else
  LTTARGET=$(prefix)/lib/stimfit
endif

install-exec-hook:
	$(LIBTOOL) --finish $(LTTARGET)
	chrpath -r $(LTTARGET) $(prefix)/bin/stimfit

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
	cp -f $(top_srcdir)/macosx/wxmac.icns stimfit.app/Contents/Resources/wxmac.icns
endif
