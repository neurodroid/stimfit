SUBDIRS = py

pkglib_LTLIBRARIES = libstimfit.la

libstimfit_la_SOURCES = ./math/fit.cpp \
            ./levmar/lm.c ./levmar/Axb.c ./levmar/misc.c ./levmar/lmlec.c ./levmar/lmbc.c \
            ./gui/app.cpp ./gui/unopt.cpp ./gui/doc.cpp ./gui/stfcheckbox.cpp ./gui/copygrid.cpp ./gui/graph.cpp \
            ./gui/printout.cpp ./gui/parentframe.cpp ./gui/childframe.cpp ./gui/view.cpp ./gui/table.cpp \
            ./gui/dlgs/cursorsdlg.cpp ./gui/dlgs/eventdlg.cpp \
	    ./gui/dlgs/fitseldlg.cpp ./gui/dlgs/smalldlgs.cpp \
            ./gui/funclib/funclib.cpp \
            ./gui/usrdlg/usrdlg.cpp

libstimfit_la_CXXFLAGS = $(OPT_CXXFLAGS)
libstimfit_la_LDFLAGS = $(LIBPYTHON_LDFLAGS) $(LIBLAPACK_LDFLAGS) $(LIBSTF_LDFLAGS)
libstimfit_la_LIBADD = $(WX_LIBS) -lfftw3

# the application source, library search path, and link libraries
if BUILD_PYTHON
    PYTHON_ADDINCLUDES = $(LIBPYTHON_INCLUDES)
else
    PYTHON_ADDINCLUDES = 
endif

INCLUDES = $(PYTHON_ADDINCLUDES)
