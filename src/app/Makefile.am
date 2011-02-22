if BUILD_MODULE

else
pkglib_LTLIBRARIES = libstimfit.la

libstimfit_la_SOURCES = ./../core/channel.cpp ./../core/section.cpp ./../core/recording.cpp \
	    ./../core/core.cpp ./../core/fitlib.cpp ./../core/measlib.cpp ./../core/filelib/asciilib.cpp \
	    ./../core/filelib/cfslib.cpp ./../core/filelib/hdf5lib.cpp ./../core/filelib/cfs.c \
            ./../core/filelib/abflib.cpp ./../core/filelib/atflib.cpp ./../core/filelib/axglib.cpp \
            ./../core/filelib/hekalib.cpp \
	    ./../core/filelib/axon/AxAbfFio32/abffiles.cpp \
	    ./../core/filelib/axon/AxAbfFio32/csynch.cpp \
	    ./../core/filelib/axon/AxAbfFio32/filedesc.cpp \
	    ./../core/filelib/axon/Common/FileReadCache.cpp \
	    ./../core/filelib/axon/Common/FileIO.cpp \
	    ./../core/filelib/axon/AxAbfFio32/abfheadr.cpp \
	    ./../core/filelib/axon/AxAbfFio32/abfhwave.cpp \
	    ./../core/filelib/axon/AxAbfFio32/Oldheadr.cpp \
	    ./../core/filelib/axon/AxAbfFio32/abfutil.cpp \
	    ./../core/filelib/axon/AxAbfFio32/msbincvt.cpp \
	    ./../core/filelib/axon/Common/unix.cpp \
	    ./../core/filelib/axon/AxAbfFio32/abferror.cpp \
            ./../core/filelib/axon/AxAtfFio32/axatffio32.cpp \
            ./../core/filelib/axon/AxAtfFio32/fileio2.cpp \
            ./../core/filelib/axon2/ProtocolReaderABF2.cpp \
            ./../core/filelib/axon2/SimpleStringCache.cpp \
	    ./../core/filelib/axon2/abf2headr.cpp \
            ./../core/filelib/axg/AxoGraph_ReadWrite.cpp \
            ./../core/filelib/axg/fileUtils.cpp \
            ./../core/filelib/axg/stringUtils.cpp \
            ./../core/filelib/axg/byteswap.cpp \
            ./../core/levmar/lm.c ./../core/levmar/Axb.c ./../core/levmar/misc.c ./../core/levmar/lmlec.c ./../core/levmar/lmbc.c \
            ./../app/app.cpp ./../app/unopt.cpp ./../app/doc.cpp ./../app/stfcheckbox.cpp ./../app/copygrid.cpp ./../app/graph.cpp \
            ./../app/printout.cpp ./../app/parentframe.cpp ./../app/childframe.cpp ./../app/view.cpp ./../app/table.cpp ./../app/zoom.cpp \
            ./../app/dlgs/cursorsdlg.cpp ./../app/dlgs/eventdlg.cpp \
	    ./../app/dlgs/fitseldlg.cpp ./../app/dlgs/smalldlgs.cpp \
            ./../app/funclib/funclib.cpp \
            ./../app/usrdlg/usrdlg.cpp

if WITH_BIOSIG
libstimfit_la_SOURCES += ./../core/filelib/biosiglib.cpp
endif

libstimfit_la_CXXFLAGS = $(OPT_CXXFLAGS)
libstimfit_la_LDFLAGS = $(LIBPYTHON_LDFLAGS) $(LIBLAPACK_LDFLAGS) $(LIBSTF_LDFLAGS) $(LIBBIOSIG_LDFLAGS)
libstimfit_la_LIBADD = $(WX_LIBS) -lfftw3 -lhdf5 -lhdf5_hl 

# the application source, library search path, and link libraries
if BUILD_PYTHON
    PYTHON_ADDINCLUDES = $(LIBPYTHON_INCLUDES)
else
    PYTHON_ADDINCLUDES = 
endif

INCLUDES = $(PYTHON_ADDINCLUDES)
endif
