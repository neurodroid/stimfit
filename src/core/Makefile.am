# these are the headers for your project

#########################################################################
# APPLICATION SECTION
#########################################################################
# this is the program that gets installed.  Its name is used for all
# of the other Makefile.am variables

noinst_LTLIBRARIES = libstfcore.la

if BUILD_PYTHON
        PYTHON_ADDSOURCES = # stfpython/stfpython.cpp
else
        PYTHON_ADDSOURCES = 
endif

libstfcore_la_SOURCES =  ./channel.cpp ./section.cpp ./recording.cpp \
	    ./core.cpp ./fitlib.cpp ./filelib/asciilib.cpp \
	    ./filelib/cfslib.cpp ./filelib/cfs.c \
            ./filelib/abflib.cpp ./filelib/atflib.cpp \
	    ./filelib/axon/AxAbfFio32/abffiles.cpp \
	    ./filelib/axon/AxAbfFio32/csynch.cpp \
	    ./filelib/axon/AxAbfFio32/filedesc.cpp \
	    ./filelib/axon/Common/FileReadCache.cpp \
	    ./filelib/axon/Common/FileIO.cpp \
	    ./filelib/axon/AxAbfFio32/abfheadr.cpp \
	    ./filelib/axon/AxAbfFio32/abfhwave.cpp \
	    ./filelib/axon/AxAbfFio32/Oldheadr.cpp \
	    ./filelib/axon/AxAbfFio32/abfutil.cpp \
	    ./filelib/axon/AxAbfFio32/msbincvt.cpp \
	    ./filelib/axon/Common/unix.c \
	    ./filelib/axon/AxAbfFio32/abferror.cpp \
            ./filelib/axon/AxAtfFio32/axatffio32.cpp \
            ./filelib/axon/AxAtfFio32/fileio.cpp $(PYTHON_ADDSOURCES) \
            ./levmar/lm.c ./levmar/Axb.c ./levmar/misc.c ./levmar/lmlec.c ./levmar/lmbc.c 

if BUILD_PYTHON
        PYTHON_ADDHEADERS = # stfpython/stfpython.h
else
        PYTHON_ADDHEADERS = 
endif

noinst_HEADERS =  ./channel.h ./section.h ./recording.h ./core.h \
            ./stimdefs.h ./fitlib.h ./spline.h ./measlib.h \
            ./filelib/asciilib.h ./filelib/cfslib.h \
            ./filelib/abflib.h ./filelib/atflib.h \
            ./filelib/cfs.h ./filelib/machine.h \
            ./filelib/axon/AxAbfFio32/abffiles.h \
            ./filelib/axon/AxAbfFio32/csynch.hpp \
            ./filelib/axon/AxAbfFio32/filedesc.hpp \
            ./filelib/axon/Common/FileReadCache.hpp \
            ./filelib/axon/Common/FileIO.hpp \
            ./filelib/axon/AxAbfFio32/abfheadr.h \
            ./filelib/axon/AxAbfFio32/oldheadr.h \
            ./filelib/axon/AxAbfFio32/abfutil.h \
            ./filelib/axon/AxAbfFio32/msbincvt.h \
            ./filelib/axon/Common/unix.h \
            ./filelib/axon/Common/axodefn.h \
            ./filelib/axon/Common/axodebug.h \
            ./filelib/axon/Common/wincpp.hpp \
            ./filelib/axon/AxAbfFio32/AxAbffio32.h \
            ./filelib/axon/AxAbfFio32/abfoldnx.h \
            ./filelib/axon/Common/resource.h \
            ./filelib/axon/AxAtfFio32/axatffio32.h \
            ./filelib/axon/AxAtfFio32/atfutil.h \
            ./filelib/axon/AxAtfFio32/atfintl.h \
            ./filelib/axon/Common/colors.h \
            ./filelib/axon/Common/adcdac.h \
            ./filelib/axon/Common/ArrayPtr.hpp \
            ./filelib/axon/Common/wincpp.hpp $(PYTHON_ADDHEADERS) \
            ./levmar/lm.h ./levmar/misc.h 


# the application source, library search path, and link libraries
if BUILD_PYTHON
    PYTHON_ADDINCLUDES = $(LIBPYTHON_INCLUDES)
else
    PYTHON_ADDINCLUDES = 
endif

INCLUDES = $(PYTHON_ADDINCLUDES)

