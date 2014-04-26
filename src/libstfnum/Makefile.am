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
pkglib_LTLIBRARIES = libstfnum.la

libstfnum_la_SOURCES =  ./fit.cpp \
            ./levmar/lm.c ./levmar/Axb.c ./levmar/misc.c ./levmar/lmlec.c ./levmar/lmbc.c \
            ./funclib.cpp ./stfnum.cpp ./measure.cpp

libstfnum_la_LDFLAGS = 
libstfnum_la_LIBADD = $(LIBSTF_LDFLAGS)

if ISDARWIN
# don't install anything because it has to go into the app bundle
if !BUILD_MODULE
install:
endif
endif
