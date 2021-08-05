pkglib_LTLIBRARIES = libstimfit.la

libstimfit_la_SOURCES = ./stf.cpp \
            ./gui/app.cpp ./gui/unopt.cpp ./gui/doc.cpp ./gui/copygrid.cpp ./gui/graph.cpp \
            ./gui/printout.cpp ./gui/parentframe.cpp ./gui/childframe.cpp ./gui/view.cpp ./gui/table.cpp ./gui/zoom.cpp \
            ./gui/dlgs/convertdlg.cpp ./gui/dlgs/cursorsdlg.cpp ./gui/dlgs/eventdlg.cpp \
	    ./gui/dlgs/fitseldlg.cpp ./gui/dlgs/smalldlgs.cpp \
            ./gui/usrdlg/usrdlg.cpp

libstimfit_la_CPPFLAGS = 
libstimfit_la_CXXFLAGS = $(OPT_CXXFLAGS) $(WX_CXXFLAGS)
libstimfit_la_LDFLAGS = $(LIBPYTHON_LDFLAGS) $(LIBSTF_LDFLAGS)
libstimfit_la_LIBADD = $(WX_LIBS) ../libstfio/libstfio.la ../libstfnum/libstfnum.la
if WITH_BIOSIGLITE
libstimfit_la_LIBADD += ../libbiosiglite/libbiosiglite.la
endif

# the application source, library search path, and link libraries
if BUILD_PYTHON
    PYTHON_ADDINCLUDES = $(LIBNUMPY_INCLUDES) $(LIBPYTHON_INCLUDES) $(LIBWXPYTHON_INCLUDES)
else
    PYTHON_ADDINCLUDES = 
endif

INCLUDES = $(PYTHON_ADDINCLUDES)
