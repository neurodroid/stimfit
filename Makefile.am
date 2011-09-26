SUBDIRS = src

if !BUILD_MODULE
bin_PROGRAMS = stimfit
check_PROGRAMS = stimfittest
TESTS = ${check_PROGRAMS}
stimfit_SOURCES = ./src/stimfit/gui/main.cpp

stimfittest_SOURCES = ./src/stimfit/test/section.cpp ./src/stimfit/test/channel.cpp ./src/stimfit/test/recording.cpp ./src/stimfit/test/measlib.cpp \
            ./src/stimfit/test/gtest/src/gtest-all.cc ./src/stimfit/test/gtest/src/gtest_main.cc

noinst_HEADERS = ./src/libstfio/channel.h ./src/libstfio/section.h ./src/libstfio/recording.h ./src/libstfio/stfio.h \
	./src/libstfio/measure.h ./src/libstfio/zoom.h \
	./src/libstfio/cfs/cfslib.h ./src/libstfio/cfs/cfs.h ./src/libstfio/cfs/machine.h \
	./src/libstfio/hdf5/hdf5lib.h \
	./src/libstfio/heka/hekalib.h \
	./src/libstfio/abf/abflib.h \
	./src/libstfio/abf/axon/AxAbfFio32/abffiles.h \
	./src/libstfio/abf/axon/AxAbfFio32/csynch.hpp \
	./src/libstfio/abf/axon/AxAbfFio32/filedesc.hpp \
	./src/libstfio/abf/axon/Common/FileReadCache.hpp \
	./src/libstfio/abf/axon/Common/FileIO.hpp \
	./src/libstfio/abf/axon/AxAbfFio32/abfheadr.h \
	./src/libstfio/abf/axon/AxAbfFio32/oldheadr.h \
	./src/libstfio/abf/axon/AxAbfFio32/abfutil.h \
	./src/libstfio/abf/axon/AxAbfFio32/msbincvt.h \
	./src/libstfio/abf/axon/Common/unix.h \
	./src/libstfio/abf/axon/Common/axodefn.h \
	./src/libstfio/abf/axon/Common/axodebug.h \
	./src/libstfio/abf/axon/Common/wincpp.hpp \
	./src/libstfio/abf/axon/AxAbfFio32/AxAbffio32.h \
	./src/libstfio/abf/axon/AxAbfFio32/abfoldnx.h \
	./src/libstfio/abf/axon/Common/resource.h \
	./src/libstfio/abf/axon/AxAtfFio32/axatffio32.h \
	./src/libstfio/abf/axon/AxAtfFio32/atfutil.h \
	./src/libstfio/abf/axon/AxAtfFio32/atfintl.h \
	./src/libstfio/abf/axon/Common/colors.h \
	./src/libstfio/abf/axon/Common/adcdac.h \
	./src/libstfio/abf/axon/Common/ArrayPtr.hpp \
	./src/libstfio/abf/axon/Common/wincpp.hpp \
	./src/libstfio/abf/axon2/ProtocolReaderABF2.hpp \
	./src/libstfio/abf/axon2/SimpleStringCache.hpp \
	./src/libstfio/abf/axon2/ProtocolStructs.h \
	./src/libstfio/abf/axon2/abf2headr.h \
	./src/libstfio/atf/atflib.h \
	./src/libstfio/axg/axglib.h \
	./src/libstfio/axg/AxoGraph_ReadWrite.h \
	./src/libstfio/axg/fileUtils.h \
	./src/libstfio/axg/stringUtils.h \
	./src/libstfio/axg/byteswap.h \
	./src/libstfio/axg/longdef.h \
	./src/libstfio/igor/igorlib.h \
	./src/libstfio/igor/CrossPlatformFileIO.h \
	./src/libstfio/igor/IgorBin.h \
	./src/stimfit/stf.h \
	./src/stimfit/math/stfmath.h ./src/stimfit/math/fit.h ./src/stimfit/math/spline.h \
	./src/stimfit/math/levmar/lm.h ./src/stimfit/math/levmar/levmar.h ./src/stimfit/math/levmar/misc.h ./src/stimfit/math/levmar/compiler.h \
	./src/stimfit/gui/app.h ./src/stimfit/gui/stfcheckbox.h \
	./src/stimfit/gui/copygrid.h ./src/stimfit/gui/graph.h \
	./src/stimfit/gui/printout.h \
	./src/stimfit/gui/doc.h ./src/stimfit/gui/parentframe.h ./src/stimfit/gui/childframe.h ./src/stimfit/gui/view.h \
	./src/stimfit/gui/table.h  \
	./src/stimfit/gui/dlgs/cursorsdlg.h ./src/stimfit/gui/dlgs/eventdlg.h \
	./src/stimfit/gui/dlgs/fitseldlg.h ./src/stimfit/gui/dlgs/smalldlgs.h \
	./src/stimfit/math/funclib.h \
	./src/stimfit/gui/usrdlg/usrdlg.h \
	./src/stimfit/test/gtest/include/gtest/gtest-death-test.h \
	./src/stimfit/test/gtest/include/gtest/gtest-message.h \
	./src/stimfit/test/gtest/include/gtest/gtest-param-test.h.pump \
	./src/stimfit/test/gtest/include/gtest/gtest-printers.h \
	./src/stimfit/test/gtest/include/gtest/gtest-spi.h \
	./src/stimfit/test/gtest/include/gtest/gtest-typed-test.h \
	./src/stimfit/test/gtest/include/gtest/gtest.h \
	./src/stimfit/test/gtest/include/gtest/gtest-param-test.h	\
	./src/stimfit/test/gtest/include/gtest/gtest_pred_impl.h \
	./src/stimfit/test/gtest/include/gtest/gtest_prod.h \
	./src/stimfit/test/gtest/include/gtest/gtest-test-part.h  \
	./src/stimfit/test/gtest/include/gtest/internal/gtest-death-test-internal.h \
	./src/stimfit/test/gtest/include/gtest/internal/gtest-linked_ptr.h \
	./src/stimfit/test/gtest/include/gtest/internal/gtest-param-util.h \
	./src/stimfit/test/gtest/include/gtest/internal/gtest-tuple.h \
	./src/stimfit/test/gtest/include/gtest/internal/gtest-type-util.h.pump \
	./src/stimfit/test/gtest/include/gtest/internal/gtest-filepath.h \
	./src/stimfit/test/gtest/include/gtest/internal/gtest-param-util-generated.h \
	./src/stimfit/test/gtest/include/gtest/internal/gtest-port.h \
	./src/stimfit/test/gtest/include/gtest/internal/gtest-tuple.h.pump \
	./src/stimfit/test/gtest/include/gtest/internal/gtest-internal.h \
	./src/stimfit/test/gtest/include/gtest/internal/gtest-param-util-generated.h.pump \
	./src/stimfit/test/gtest/include/gtest/internal/gtest-string.h \
	./src/stimfit/test/gtest/include/gtest/internal/gtest-type-util.h \
	./src/stimfit/test/gtest/src/gtest-internal-inl.h

if WITH_BIOSIG
    noinst_HEADERS += ./src/libstfio/biosig/biosiglib.h
endif

EXTRA_DIST = ./src/stimfit/res/16-em-down.xpm
EXTRA_DIST+= ./src/stimfit/res/16-em-open.xpm
EXTRA_DIST+= ./src/stimfit/res/accept.xpm
EXTRA_DIST+= ./src/stimfit/res/arrow_down.xpm
EXTRA_DIST+= ./src/stimfit/res/arrow_left.xpm
EXTRA_DIST+= ./src/stimfit/res/arrow_out.xpm
EXTRA_DIST+= ./src/stimfit/res/arrow_right.xpm
EXTRA_DIST+= ./src/stimfit/res/arrow_up.xpm
EXTRA_DIST+= ./src/stimfit/res/camera.xpm
EXTRA_DIST+= ./src/stimfit/res/ch1.xpm
EXTRA_DIST+= ./src/stimfit/res/ch2.xpm
EXTRA_DIST+= ./src/stimfit/res/cursor.xpm
EXTRA_DIST+= ./src/stimfit/res/event.xpm
EXTRA_DIST+= ./src/stimfit/res/fit.xpm
EXTRA_DIST+= ./src/stimfit/res/fit_lim.xpm
EXTRA_DIST+= ./src/stimfit/res/latency_lim.xpm
EXTRA_DIST+= ./src/stimfit/res/resultset_first.xpm
EXTRA_DIST+= ./src/stimfit/res/resultset_last.xpm
EXTRA_DIST+= ./src/stimfit/res/resultset_next.xpm
EXTRA_DIST+= ./src/stimfit/res/resultset_previous.xpm
EXTRA_DIST+= ./src/stimfit/res/stimfit.png
EXTRA_DIST+= ./src/stimfit/res/sum_new.xpm
EXTRA_DIST+= ./src/stimfit/res/sum_new_aligned.xpm
EXTRA_DIST+= ./src/stimfit/res/table.xpm
EXTRA_DIST+= ./src/stimfit/res/zoom.xpm
EXTRA_DIST+= ./src/stimfit/res/zoom_in.xpm
EXTRA_DIST+= ./src/stimfit/res/zoom_out.xpm
EXTRA_DIST+= ./src/stimfit/math/levmar/Axb_core.c
EXTRA_DIST+= ./src/stimfit/math/levmar/lmbc_core.c
EXTRA_DIST+= ./src/stimfit/math/levmar/lm_core.c
EXTRA_DIST+= ./src/stimfit/math/levmar/lmlec_core.c
EXTRA_DIST+= ./src/stimfit/math/levmar/misc_core.c
EXTRA_DIST+= ./src/stimfit/math/levmar/LICENSE
EXTRA_DIST+= ./src/stimfit/math/levmar/README.txt
EXTRA_DIST+= ./Doxyfile
EXTRA_DIST+= ./acsite.m4
EXTRA_DIST+= ./autogen.sh
EXTRA_DIST+= ./macosx/scripts/conf_mac_release.sh
EXTRA_DIST+= ./macosx/scripts/change_deps_release.sh
EXTRA_DIST+= ./macosx/app.r
EXTRA_DIST+= ./macosx/stimfit.icns
# EXTRA_DIST+= ./src/pystfio/stfioswig_wrap.cxx 
# EXTRA_DIST+= ./src/pystfio/stfio.py 
EXTRA_DIST+= ./src/pystfio/pystfio.cxx 
EXTRA_DIST+= ./src/pystfio/pystfio.i 
EXTRA_DIST+= ./src/pystfio/stfio_plot.py 
EXTRA_DIST+= ./src/pystfio/unittest_stfio.py
EXTRA_DIST+= ./src/pystfio/test.h5
EXTRA_DIST+= ./src/pystfio/pystfio.h 
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
EXTRA_DIST+= ./src/stimfit/test/gtest/CHANGES
EXTRA_DIST+= ./src/stimfit/test/gtest/CONTRIBUTORS
EXTRA_DIST+= ./src/stimfit/test/gtest/COPYING
EXTRA_DIST+= ./src/stimfit/test/gtest/README
EXTRA_DIST+= ./src/stimfit/test/gtest/src/gtest.cc
EXTRA_DIST+= ./src/stimfit/test/gtest/src/gtest-death-test.cc
EXTRA_DIST+= ./src/stimfit/test/gtest/src/gtest-filepath.cc
EXTRA_DIST+= ./src/stimfit/test/gtest/src/gtest-port.cc
EXTRA_DIST+= ./src/stimfit/test/gtest/src/gtest-printers.cc
EXTRA_DIST+= ./src/stimfit/test/gtest/src/gtest-test-part.cc
EXTRA_DIST+= ./src/stimfit/test/gtest/src/gtest-typed-test.cc

if BUILD_PYTHON
    PYTHON_ADDINCLUDES = $(LIBPYTHON_INCLUDES)
    PYTHON_ADDLDFLAGS = $(LIBPYTHON_LDFLAGS)
    PYTHON_ADDLIBS = ./src/stimfit/py/libpystf.la
else
    PYTHON_ADDLDFLAGS = 
    PYTHON_ADDLIBS = 
    PYTHON_ADDINCLUDES = 
endif

INCLUDES = $(PYTHON_ADDINCLUDES)

stimfit_CXXFLAGS = $(OPT_CXXFLAGS) $(WX_CXXFLAGS)
stimfit_LDFLAGS = $(LIBLAPACK_LDFLAGS) $(PYTHON_ADDLDFLAGS) $(LIBSTF_LDFLAGS) $(LIBBIOSIG_LDFLAGS)
stimfit_LDADD = $(WX_LIBS) -lfftw3 ./src/stimfit/libstimfit.la ./src/libstfio/libstfio.la # $(PYTHON_ADDLIBS) 

stimfittest_CXXFLAGS = $(GT_CXXFLAGS) $(WX_CXXFLAGS)
stimfittest_CPPFLAGS = ${CPPFLAGS} $(GT_CPPFLAGS) -DSTF_TEST -I$(top_srcdir)/src/stimfit/test/gtest -I$(top_srcdir)/src/stimfit/test/gtest/include
stimfittest_LDFLAGS = $(LIBLAPACK_LDFLAGS) $(PYTHON_ADDLDFLAGS) $(GT_LDFLAGS)
stimfittest_LDADD = $(WX_LIBS) $(PYTHON_ADDLIBS) $(GT_LIBS) -lfftw3 ./src/stimfit/gui/libstimfit.la

if BUILD_DEBIAN
LTTARGET=/usr/lib/stimfit
install-exec-hook:
	$(LIBTOOL) --finish $(LTTARGET)
	chrpath -r $(LTTARGET) $(prefix)/bin/stimfit
	chrpath -r $(LTTARGET) $(prefix)/lib/stimfit/libpystf.so
	chrpath -r $(LTTARGET) $(prefix)/lib/stimfit/libstimfit.so
	chrpath -r $(LTTARGET) $(prefix)/lib/stimfit/libstfio.so
	mkdir -p $(prefix)/share/pixmaps
	mkdir -p $(prefix)/share/applications
	cp $(top_srcdir)/src/stimfit/res/stimfit.png $(prefix)/share/pixmaps/stimfit.png
	cp $(top_srcdir)/debian/stimfit.desktop $(prefix)/share/applications/
else
LTTARGET=$(prefix)/lib/stimfit
install-exec-hook:
	$(LIBTOOL) --finish $(LTTARGET)
	chrpath -r $(LTTARGET) $(prefix)/bin/stimfit
	mkdir -p $(prefix)/share/pixmaps
	mkdir -p $(prefix)/share/applications
	cp $(top_srcdir)/src/stimfit/res/stimfit.png $(prefix)/share/pixmaps/stimfit.png
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
