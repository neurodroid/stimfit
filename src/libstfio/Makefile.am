if BUILD_MODULE
if ISDARWIN
PYTHON_TARGET_DIR=${PYTHON_DIST_PKG}
else
PYTHON_TARGET_DIR=${PYTHON_PRE_DIST_PKG}
# PYTHON_TARGET_DIR=${PYTHON_DIST_PKG}
endif
PYTHON_DEST_DIR=${DESTDIR}${PYTHON_TARGET_DIR}
pkglibdir = ${PYTHON_TARGET_DIR}/stfio

endif
pkglib_LTLIBRARIES = libstfio.la

libstfio_la_SOURCES =  ./channel.cpp ./section.cpp ./recording.cpp ./stfio.cpp \
	./cfs/cfslib.cpp ./cfs/cfs.c \
	./hdf5/hdf5lib.cpp \
	./abf/abflib.cpp \
	./abf/axon/AxAbfFio32/abffiles.cpp \
	./abf/axon/AxAbfFio32/csynch.cpp \
	./abf/axon/AxAbfFio32/filedesc.cpp \
	./abf/axon/Common/FileReadCache.cpp \
	./abf/axon/Common/FileIO.cpp \
	./abf/axon/AxAbfFio32/abfheadr.cpp \
	./abf/axon/AxAbfFio32/abfhwave.cpp \
	./abf/axon/AxAbfFio32/Oldheadr.cpp \
	./abf/axon/AxAbfFio32/abfutil.cpp \
	./abf/axon/AxAbfFio32/msbincvt.cpp \
	./abf/axon/Common/unix.cpp \
	./abf/axon/AxAbfFio32/abferror.cpp \
	./abf/axon/AxAtfFio32/axatffio32.cpp \
	./abf/axon/AxAtfFio32/fileio2.cpp \
	./abf/axon2/ProtocolReaderABF2.cpp \
	./abf/axon2/SimpleStringCache.cpp \
	./abf/axon2/abf2headr.cpp \
	./atf/atflib.cpp \
	./igor/igorlib.cpp \
	./igor/CrossPlatformFileIO.c \
	./igor/WriteWave.c \
	./intan/common.cpp \
	./intan/intanlib.cpp \
	./intan/streams.cpp

if !WITH_BIOSIG
libstfio_la_SOURCES += \
	./axg/axglib.cpp \
	./axg/AxoGraph_ReadWrite.cpp \
	./axg/fileUtils.cpp \
	./axg/stringUtils.cpp \
	./axg/byteswap.cpp \
	./heka/hekalib.cpp
endif
if WITH_BIOSIG
libstfio_la_SOURCES += \
	./biosig/biosiglib.cpp
else
if WITH_BIOSIGLITE
libstfio_la_SOURCES += \
	./biosig/biosiglib.cpp
endif
endif

libstfio_la_LDFLAGS =
libstfio_la_LIBADD = $(LIBSTF_LDFLAGS) $(LIBHDF5_LDFLAGS) $(LIBBIOSIG_LDFLAGS)

if ISDARWIN
# don't install anything because it has to go into the app bundle
if !BUILD_MODULE
install:
endif
endif
#. /ascii/asciilib.cpp
