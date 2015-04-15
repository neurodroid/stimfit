SUBDIRS = src

if !BUILD_MODULE
bin_PROGRAMS = stimfit
check_PROGRAMS = stimfittest
TESTS = ${check_PROGRAMS}
stimfit_SOURCES = ./src/stimfit/gui/main.cpp

stimfittest_SOURCES = ./src/test/section.cpp ./src/test/channel.cpp ./src/test/recording.cpp ./src/test/fit.cpp ./src/test/measure.cpp \
            ./src/test/gtest/src/gtest-all.cc ./src/test/gtest/src/gtest_main.cc

noinst_HEADERS = ./src/libstfio/channel.h ./src/libstfio/section.h ./src/libstfio/recording.h ./src/libstfio/stfio.h \
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
	./src/libstfnum/stfnum.h ./src/libstfnum/fit.h ./src/libstfnum/spline.h \
	./src/libstfnum/measure.h \
	./src/libstfnum/levmar/lm.h ./src/libstfnum/levmar/levmar.h \
	./src/libstfnum/levmar/misc.h ./src/libstfnum/levmar/compiler.h \
	./src/libstfnum/funclib.h \
	./src/stimfit/stf.h \
	./src/stimfit/gui/app.h ./src/stimfit/gui/stfcheckbox.h \
	./src/stimfit/gui/copygrid.h ./src/stimfit/gui/graph.h \
	./src/stimfit/gui/printout.h \
	./src/stimfit/gui/doc.h ./src/stimfit/gui/parentframe.h ./src/stimfit/gui/childframe.h ./src/stimfit/gui/view.h \
	./src/stimfit/gui/table.h ./src/stimfit/gui/zoom.h \
	./src/stimfit/gui/dlgs/cursorsdlg.h ./src/stimfit/gui/dlgs/eventdlg.h \
	./src/stimfit/gui/dlgs/fitseldlg.h ./src/stimfit/gui/dlgs/smalldlgs.h \
	./src/stimfit/gui/usrdlg/usrdlg.h \
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

if WITH_BIOSIG2
    noinst_HEADERS += ./src/libstfio/biosig/biosiglib.h
else
if WITH_BIOSIG
    noinst_HEADERS += ./src/libstfio/biosig/biosiglib.h
endif # WITH_BIOSIG
endif # WITH_BIOSIG2

EXTRA_DIST = ./src/stimfit/res/16-em-down.xpm
EXTRA_DIST+= ./src/stimfit/res/16-em-open.xpm
EXTRA_DIST+= ./src/stimfit/res/accept.xpm
EXTRA_DIST+= ./src/stimfit/res/arrow_down.xpm
EXTRA_DIST+= ./src/stimfit/res/arrow_left.xpm
EXTRA_DIST+= ./src/stimfit/res/arrow_out.xpm
EXTRA_DIST+= ./src/stimfit/res/arrow_right.xpm
EXTRA_DIST+= ./src/stimfit/res/arrow_up.xpm
EXTRA_DIST+= ./src/stimfit/res/camera.xpm
EXTRA_DIST+= ./src/stimfit/res/camera_ps.xpm
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
EXTRA_DIST+= ./src/stimfit/res/slope.xpm
EXTRA_DIST+= ./src/stimfit/res/stimfit.png
EXTRA_DIST+= ./src/stimfit/res/stimfit16x16.xpm
EXTRA_DIST+= ./src/stimfit/res/stimfit32x32.xpm
EXTRA_DIST+= ./src/stimfit/res/sum_new.xpm
EXTRA_DIST+= ./src/stimfit/res/sum_new_aligned.xpm
EXTRA_DIST+= ./src/stimfit/res/table.xpm
EXTRA_DIST+= ./src/stimfit/res/zoom.xpm
EXTRA_DIST+= ./src/stimfit/res/zoom_in.xpm
EXTRA_DIST+= ./src/stimfit/res/zoom_out.xpm
EXTRA_DIST+= ./src/libstfnum/levmar/Axb_core.c
EXTRA_DIST+= ./src/libstfnum/levmar/lmbc_core.c
EXTRA_DIST+= ./src/libstfnum/levmar/lm_core.c
EXTRA_DIST+= ./src/libstfnum/levmar/lmlec_core.c
EXTRA_DIST+= ./src/libstfnum/levmar/misc_core.c
EXTRA_DIST+= ./src/libstfnum/levmar/LICENSE
EXTRA_DIST+= ./src/libstfnum/levmar/README.txt
EXTRA_DIST+= ./doc/Doxyfile
EXTRA_DIST+= ./acsite.m4
EXTRA_DIST+= ./autogen.sh
EXTRA_DIST+= ./Makefile.static
EXTRA_DIST+= ./dist/macosx/scripts/conf_mac_release.sh
EXTRA_DIST+= ./dist/macosx/scripts/change_deps_release.sh
EXTRA_DIST+= ./dist/macosx/app.r
EXTRA_DIST+= ./dist/macosx/stimfit.icns
EXTRA_DIST+= ./src/pystfio/__init__.py
# EXTRA_DIST+= ./src/pystfio/stfioswig_wrap.cxx 
# EXTRA_DIST+= ./src/pystfio/stfio.py 
EXTRA_DIST+= ./src/pystfio/pystfio.cxx 
EXTRA_DIST+= ./src/pystfio/pystfio.i 
EXTRA_DIST+= ./src/pystfio/stfio_plot.py 
EXTRA_DIST+= ./src/pystfio/stfio_neo.py
EXTRA_DIST+= ./src/pystfio/unittest_stfio.py
EXTRA_DIST+= ./src/pystfio/test.h5
EXTRA_DIST+= ./src/pystfio/pystfio.h 
# EXTRA_DIST+= ./dist/debian/changelog
EXTRA_DIST+= ./dist/debian/compat
EXTRA_DIST+= ./dist/debian/control
EXTRA_DIST+= ./dist/debian/copyright
EXTRA_DIST+= ./dist/debian/docs
EXTRA_DIST+= ./dist/debian/mkdeb.sh
EXTRA_DIST+= ./dist/debian/python-stfio.files
EXTRA_DIST+= ./dist/debian/python-stfio.install
EXTRA_DIST+= ./dist/debian/rules
EXTRA_DIST+= ./dist/debian/stimfit.1
EXTRA_DIST+= ./dist/debian/stimfit.desktop
EXTRA_DIST+= ./dist/debian/stimfit.files
EXTRA_DIST+= ./dist/debian/stimfit.install
EXTRA_DIST+= ./dist/debian/stimfit.menu
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
    PYTHON_ADDLIBS = ./src/stimfit/py/libpystf.la
else !BUILD_PYTHON
    PYTHON_ADDLDFLAGS = 
    PYTHON_ADDLIBS = 
    PYTHON_ADDINCLUDES = 
endif !BUILD_PYTHON

INCLUDES = $(PYTHON_ADDINCLUDES)

stimfit_CXXFLAGS = $(OPT_CXXFLAGS) $(WX_CXXFLAGS)
stimfit_LDFLAGS = $(LIBLAPACK_LDFLAGS) $(PYTHON_ADDLDFLAGS) $(LIBSTF_LDFLAGS) $(LIBBIOSIG_LDFLAGS)
stimfit_LDADD = $(WX_LIBS) -lfftw3 ./src/stimfit/libstimfit.la ./src/libstfio/libstfio.la ./src/libstfnum/libstfnum.la # $(PYTHON_ADDLIBS) 

stimfittest_CXXFLAGS = $(GT_CXXFLAGS) $(WX_CXXFLAGS)
stimfittest_CPPFLAGS = ${CPPFLAGS} $(GT_CPPFLAGS) -DSTF_TEST -I$(top_srcdir)/src/test/gtest -I$(top_srcdir)/src/test/gtest/include
stimfittest_LDFLAGS = $(LIBLAPACK_LDFLAGS) $(PYTHON_ADDLDFLAGS) $(GT_LDFLAGS)
stimfittest_LDADD = $(WX_LIBS) $(PYTHON_ADDLIBS) $(GT_LIBS) -lfftw3 ./src/stimfit/libstimfit.la ./src/libstfio/libstfio.la ./src/libstfnum/libstfnum.la

if !ISDARWIN
if BUILD_DEBIAN
LTTARGET=/usr/lib/stimfit
install-exec-hook:
	$(LIBTOOL) --finish $(prefix)/lib/stimfit
	chrpath -r $(LTTARGET) $(prefix)/bin/stimfit
	chrpath -r $(LTTARGET) $(prefix)/lib/stimfit/libpystf.so
	chrpath -r $(LTTARGET) $(prefix)/lib/stimfit/libstimfit.so
	chrpath -r $(LTTARGET) $(prefix)/lib/stimfit/libstfio.so
	chrpath -r $(LTTARGET) $(prefix)/lib/stimfit/libstfnum.so
	install -d $(prefix)/share/pixmaps
	install -d $(prefix)/share/applications
	install -m 644 $(top_srcdir)/src/stimfit/res/stimfit16x16.xpm $(prefix)/share/pixmaps/stimfit16x16.xpm
	install -m 644 $(top_srcdir)/src/stimfit/res/stimfit32x32.xpm $(prefix)/share/pixmaps/stimfit32x32.xpm
	install -m 644 $(top_srcdir)/dist/debian/stimfit.desktop $(prefix)/share/applications/
else !BUILD_DEBIAN
LTTARGET=$(prefix)/lib/stimfit
install-exec-hook:
	$(LIBTOOL) --finish $(LTTARGET)
	chrpath -r $(LTTARGET) $(prefix)/bin/stimfit
	install -d $(prefix)/share/pixmaps
	install -d $(prefix)/share/applications
	install -m 644 $(top_srcdir)/src/stimfit/res/stimfit16x16.xpm $(prefix)/share/pixmaps/stimfit16x16.xpm
	install -m 644 $(top_srcdir)/src/stimfit/res/stimfit32x32.xpm $(prefix)/share/pixmaps/stimfit32x32.xpm
	install -m 644 $(top_srcdir)/dist/debian/stimfit.desktop $(prefix)/share/applications/
endif !BUILD_DEBIAN
uninstall-hook:
	rm -f $(prefix)/share/pixmaps/stimfit16x16.xpm
	rm -f $(prefix)/share/pixmaps/stimfit32x32.xpm
	rm -f $(prefix)/share/applications/stimfit.desktop
else ISDARWIN
LTTARGET=$(prefix)/lib/stimfit
# wxMac resource fork/unbundled app
install: stimfit
	mkdir -p ${DESTDIR}/stimfit.app/Contents/MacOS
	mkdir -p ${DESTDIR}/stimfit.app/Contents/Resources
	mkdir -p ${DESTDIR}/stimfit.app/Contents/Resources/English.lproj
	mkdir -p ${DESTDIR}/stimfit.app/Contents/Frameworks/stimfit
	mkdir -p ${DESTDIR}/stimfit.app/Contents/lib/stimfit
	cp -v ./src/stimfit/.libs/libstimfit.dylib ${DESTDIR}/stimfit.app/Contents/lib/stimfit/libstimfit.dylib
	cp -v ./src/libstfio/.libs/libstfio.dylib ${DESTDIR}/stimfit.app/Contents/lib/stimfit/libstfio.dylib
	cp -v ./src/libstfnum/.libs/libstfnum.dylib ${DESTDIR}/stimfit.app/Contents/lib/stimfit/libstfnum.dylib
	cp $(top_srcdir)/dist/macosx/stimfit.plist.in ${DESTDIR}/stimfit.app/Contents/Info.plist
	echo "APPL????\c" > ${DESTDIR}/stimfit.app/Contents/PkgInfo
	rm -f ${DESTDIR}/stimfit.app/Contents/MacOS/stimfit$(EXEEXT)
	cp -p -f .libs/stimfit$(EXEEXT) ${DESTDIR}/stimfit.app/Contents/MacOS/stimfit$(EXEEXT)
if BUILD_PYTHON
	cp -v ./src/stimfit/py/.libs/libpystf.dylib ${DESTDIR}/stimfit.app/Contents/lib/stimfit/libpystf.dylib
	ln -sf ../../lib/stimfit/libpystf.dylib ${DESTDIR}/stimfit.app/Contents/Frameworks/stimfit/_stf.so
	cp -v $(top_srcdir)/src/stimfit/py/*.py ${DESTDIR}/stimfit.app/Contents/Frameworks/stimfit/
	cp -v $(top_srcdir)/src/pystfio/*.py ${DESTDIR}/stimfit.app/Contents/Frameworks/stimfit/
	${PYTHON} -m compileall -l ${DESTDIR}/stimfit.app/Contents/Frameworks/stimfit/
endif BUILD_PYTHON
	$(POSTLINK_COMMAND) ${DESTDIR}/stimfit.app/Contents/MacOS/stimfit$(EXEEXT) \
	                    $(srcdir)/dist/macosx/app.r
	$(MACSETFILE) -a C ${DESTDIR}/stimfit.app/Contents/MacOS/stimfit$(EXEEXT)
	cp -f $(top_srcdir)/dist/macosx/stimfit.icns ${DESTDIR}/stimfit.app/Contents/Resources/stimfit.icns
	install_name_tool -change \
	                  $(LTTARGET)/libstimfit.dylib \
	                  @executable_path/../lib/stimfit/libstimfit.dylib \
	                  ${DESTDIR}/stimfit.app/Contents/MacOS/stimfit
	install_name_tool -change \
	                  $(LTTARGET)/libstfio.dylib \
	                  @executable_path/../lib/stimfit/libstfio.dylib \
	                  ${DESTDIR}/stimfit.app/Contents/MacOS/stimfit
	install_name_tool -change \
	                  $(LTTARGET)/libstfnum.dylib \
	                  @executable_path/../lib/stimfit/libstfnum.dylib \
	                  ${DESTDIR}/stimfit.app/Contents/MacOS/stimfit
	install_name_tool -change \
	                  $(LTTARGET)/libstimfit.dylib \
	                  @executable_path/../lib/stimfit/libstimfit.dylib \
	                  ${DESTDIR}/stimfit.app/Contents/lib/stimfit/libstimfit.dylib
	install_name_tool -change \
	                  $(LTTARGET)/libstfio.dylib \
	                  @executable_path/../lib/stimfit/libstfio.dylib \
	                  ${DESTDIR}/stimfit.app/Contents/lib/stimfit/libstimfit.dylib
	install_name_tool -change \
	                  $(LTTARGET)/libstfio.dylib \
	                  @executable_path/../lib/stimfit/libstfio.dylib \
	                  ${DESTDIR}/stimfit.app/Contents/lib/stimfit/libstfio.dylib
	install_name_tool -change \
	                  $(LTTARGET)/libstfnum.dylib \
	                  @executable_path/../lib/stimfit/libstfnum.dylib \
	                  ${DESTDIR}/stimfit.app/Contents/lib/stimfit/libstimfit.dylib
	install_name_tool -change \
	                  $(LTTARGET)/libstfnum.dylib \
	                  @executable_path/../lib/stimfit/libstfnum.dylib \
	                  ${DESTDIR}/stimfit.app/Contents/lib/stimfit/libstfnum.dylib
if BUILD_PYTHON
	install_name_tool -change \
	                  $(LTTARGET)/libstimfit.dylib \
	                  @executable_path/../lib/stimfit/libstimfit.dylib \
	                  ${DESTDIR}/stimfit.app/Contents/lib/stimfit/libpystf.dylib
	install_name_tool -change \
	                  $(LTTARGET)/libstfio.dylib \
	                  @executable_path/../lib/stimfit/libstfio.dylib \
	                  ${DESTDIR}/stimfit.app/Contents/lib/stimfit/libpystf.dylib
	install_name_tool -change \
	                  $(LTTARGET)/libstfnum.dylib \
	                  @executable_path/../lib/stimfit/libstfnum.dylib \
	                  ${DESTDIR}/stimfit.app/Contents/lib/stimfit/libpystf.dylib
	install_name_tool -change \
	                  $(LTTARGET)/libpystf.dylib \
	                  @executable_path/../lib/stimfit/libpystf.dylib \
	                  ${DESTDIR}/stimfit.app/Contents/lib/stimfit/libpystf.dylib
endif BUILD_PYTHON

endif ISDARWIN

endif  # !BUILD_MODULE
