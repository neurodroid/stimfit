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

pkglib_LTLIBRARIES = libbiosiglite.la

libbiosiglite_la_SOURCES = \
  ../biosig/biosig4c++/t210/sopen_abf_read.c \
  ../biosig/biosig4c++/t210/sopen_alpha_read.c \
  ../biosig/biosig4c++/t210/sopen_axg_read.c \
  ../biosig/biosig4c++/t210/sopen_cfs_read.c \
  ../biosig/biosig4c++/t210/sopen_heka_read.c \
  ../biosig/biosig4c++/t210/sopen_igor.c \
  ../biosig/biosig4c++/t210/sopen_rhd2000_read.c \
  ../biosig/biosig4c++/t210/sopen_scp_read.c \
  ../biosig/biosig4c++/t210/sopen_matio.c \
  ../biosig/biosig4c++/t210/sopen_biosigdump_read.c \
  ../biosig/biosig4c++/t210/sopen_sqlite.c \
  ../biosig/biosig4c++/t210/sopen_hdf5.c \
  ../biosig/biosig4c++/t210/sopen_cadwell_read.c \
  ../biosig/biosig4c++/t210/scp-decode.cpp \
  ../biosig/biosig4c++/t220/crc4scp.c \
  ../biosig/biosig4c++/t220/sopen_scp_write.c \
  ../biosig/biosig4c++/t230/sopen_hl7aecg.cpp \
  ../biosig/biosig4c++/test0/sandbox.c \
  ../biosig/biosig4c++/biosig.c \
  ../biosig/biosig4c++/biosig2.c \
  ../biosig/biosig4c++/gdftime.c \
  ../biosig/biosig4c++/mdc_ecg_codes.c \
  ../biosig/biosig4c++/physicalunits.c \
  ../biosig/biosig4c++/XMLParser/tinyxml.cpp \
  ../biosig/biosig4c++/XMLParser/tinyxmlparser.cpp \
  ../biosig/biosig4c++/XMLParser/tinyxmlerror.cpp \
  ../biosig/biosig4c++/XMLParser/tinystr.cpp

  # ../biosig/biosig4c++/biosig-network.c

# libbiosiglite_la_CPPFLAGS = -DWITH_BIOSIG -DWITH_ZLIB -DWITH_CHOLMOD -DWITHOUT_NETWORK
libbiosiglite_la_CPPFLAGS = -DWITH_BIOSIG -DWITH_ZLIB -DWITHOUT_NETWORK
libbiosiglite_la_LDFLAGS = 
# libbiosiglite_la_LIBADD = $(LIBSTF_LDFLAGS) -lz -lcholmod
libbiosiglite_la_LIBADD = $(LIBSTF_LDFLAGS) -lz


if ISDARWIN
# don't install anything because it has to go into the app bundle
if !BUILD_MODULE
install:
endif
endif
