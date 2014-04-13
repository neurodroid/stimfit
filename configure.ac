AC_INIT([stimfit],[0.13.17])
AC_CONFIG_SRCDIR(src/stimfit/gui/main.cpp)
AM_INIT_AUTOMAKE([1.11])
AM_SILENT_RULES([yes])
AC_CONFIG_MACRO_DIR([m4])

AC_PROG_CC
AC_PROG_INSTALL

_LT_SET_OPTION([LT_INIT],[dlopen])
AC_DIAGNOSE([obsolete],[AC_LIBTOOL_DLOPEN: Remove this warning and the call to _LT_SET_OPTION when you
put the `dlopen' option into LT_INIT's first parameter.])

LT_INIT
AC_PROG_CPP
AC_PROG_AWK
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_MKDIR_P
AC_PROG_CXX
AC_PROG_LIBTOOL

# BUILDDATE=`date`

# Build a standalone python module
AC_ARG_ENABLE([module], AS_HELP_STRING([--enable-module],[build a standalone python module; implies --enable-python]),[])
AM_CONDITIONAL(BUILD_MODULE, test "$enable_module" = "yes")

# pbuilder debian package build
AC_ARG_ENABLE([debian], AS_HELP_STRING([--enable-debian],[special build for pbuilder]),[])
AM_CONDITIONAL(BUILD_DEBIAN, test "$enable_debian" = "yes")

AC_MSG_CHECKING(for kernel)
case `uname` in 
    Darwin)
        AC_MSG_RESULT(darwin)
        CXXFLAGS="${CXXFLAGS} -fPIC"
        CFLAGS="${CFLAGS} -fPIC"
        STFKERNEL="darwin"
    ;;
    *)  
        # treat everything else (kfreebsd, hurd) as linux
        AC_MSG_RESULT(linux or similar)
        CXXFLAGS="-fPIC"
        CFLAGS="-fPIC"
        STFKERNEL="linux"
    ;;
esac
AM_CONDITIONAL([ISDARWIN], [test $STFKERNEL = "darwin"])

# Checks for python libraries.
AC_ARG_ENABLE([python], 
  AS_HELP_STRING(
    [--enable-python],
    [enable python console (default="yes")]),,
    [enable_python="yes"])

AM_CONDITIONAL(BUILD_PYTHON, test "$enable_python" = "yes")
if (test "$enable_python" = "yes") || (test "$enable_module" = "yes"); then
    AC_PYTHON_DEVEL
    AC_PROG_SWIG(1.3.17)
    SWIG_ENABLE_CXX
    SWIG_PYTHON
    AC_SUBST(SWIG)
    CXXFLAGS="${CXXFLAGS}"
    CFLAGS="${CFLAGS}"
    LIBPYTHON_LDFLAGS=$PYTHON_LDFLAGS 
    LIBPYTHON_INCLUDES=$PYTHON_CPPFLAGS
    LIBNUMPY_INCLUDES=$PYTHON_NUMPY_INCLUDE
else
    LIBPYTHON_LDFLAGS= 
    LIBPYTHON_INCLUDES= 
    LIBNUMPY_INCLUDES= 
fi
AC_SUBST(LIBPYTHON_LDFLAGS)
AC_SUBST(LIBPYTHON_INCLUDES)
AC_SUBST(LIBNUMPY_INCLUDES)

AC_MSG_CHECKING(for kernel)
case ${STFKERNEL} in 
    darwin)
        LIBSTF_LDFLAGS="-avoid-version"
        if test "$enable_module" = "yes" ; then
            CPPFLAGS="${CPPFLAGS} -DMODULE_ONLY"
        else
            CPPFLAGS="${CPPFLAGS}"
        fi
        STF_PYTHON_LIBNAME="libpystf.dylib"
        STFIO_PYTHON_LIBNAME="libpystfio.dylib"
        ARCH_FLAGS=""
        CFLAGS="${CFLAGS} ${ARCH_FLAGS}"
        CXXFLAGS="${CXXFLAGS} ${ARCH_FLAGS}"
        LDFLAGS="${LDFLAGS} ${ARCH_FLAGS}"
        OBJCFLAGS="${OBJCFLAGS} ${ARCH_FLAGS}"
        OBJCXXFLAGS="${OBJCXXFLAGS} ${ARCH_FLAGS}"
    ;;
    *)
        if test "$enable_module" = "yes" ; then
            LIBSTF_LDFLAGS="-avoid-version"
            CPPFLAGS="${CPPFLAGS} -DMODULE_ONLY"
        else
            if test "$enable_debian" = "yes" ; then
              LIBSTF_LDFLAGS="-Wl,-rpath,/usr/lib/stimfit -avoid-version"
              CPPFLAGS="${CPPFLAGS} `dpkg-buildflags --get CPPFLAGS`"
              CFLAGS="${CFLAGS} `dpkg-buildflags --get CFLAGS`"
              CXXFLAGS="${CXXFLAGS} `dpkg-buildflags --get CXXFLAGS`"
              LDFLAGS="${LDFLAGS} `dpkg-buildflags --get LDFLAGS`"
            else
              LIBSTF_LDFLAGS="-Wl,-rpath,${prefix}/lib/stimfit -avoid-version"
            fi
            CPPFLAGS="${CPPFLAGS}"
        fi
        STF_PYTHON_LIBNAME="libpystf.so"
        STFIO_PYTHON_LIBNAME="libpystfio.so"
    ;;
esac
AC_SUBST(LIBSTF_LDFLAGS)
AC_SUBST(STF_PYTHON_LIBNAME)
AC_SUBST(STFIO_PYTHON_LIBNAME)

# Checks for python libraries.
if test "$enable_python" = "yes" ; then
    CPPFLAGS="${CPPFLAGS} -DWITH_PYTHON"
fi

AC_ARG_ENABLE([ipython], AS_HELP_STRING([--enable-ipython],[enable ipython as the default shell (experimental); implies --enable-python]),[])
if test "$enable_ipython" = "yes" ; then
    CPPFLAGS="${CPPFLAGS} -DIPYTHON"
fi

# Build the exotic Stimfit flavour with Slope cursors
AC_ARG_WITH([pslope], AS_HELP_STRING([--with-pslope],[include slope measure cursors]),[])
if test "$with_pslope" = "yes" ; then
    CPPFLAGS="${CPPFLAGS} -DWITH_PSLOPE"
fi

AC_ARG_WITH([biosig], AS_HELP_STRING([--with-biosig],[build with libbiosig support - better tested than --with-biosig2]),[])
AM_CONDITIONAL(WITH_BIOSIG, test "$with_biosig" = "yes")

AC_ARG_WITH([biosig2], AS_HELP_STRING([--with-biosig2],[alternative to --with-biosig - eventually, this will provide better ABI compatibility when upgrading libbiosig; currently it's in a testing state; requires libbiosig2 v1.5.6 or higher]),[])
AM_CONDITIONAL(WITH_BIOSIG2, test "$with_biosig2" = "yes")

if test "$with_biosig2" = "yes" ; then
    CPPFLAGS="${CPPFLAGS} -DWITH_BIOSIG2"
    LIBBIOSIG_LDFLAGS="-lbiosig2 -lcholmod"
elif test "$with_biosig" = "yes" ; then
    CPPFLAGS="${CPPFLAGS} -DWITH_BIOSIG"
    LIBBIOSIG_LDFLAGS="-lbiosig -lcholmod"
fi
AC_SUBST(LIBBIOSIG_LDFLAGS)

AC_ARG_WITH([lapack-lib],
AS_HELP_STRING([--with-lapack-lib=LAPACKLIB],[Provide full path to custom lapack library]),
[
    if test "$withval" != "yes" -a "$withval" != ""; then
        LAPACKLIB=$withval
        LIBLAPACK_LDFLAGS="$LAPACKLIB"
        CPPFLAGS="${CPPFLAGS} -DHAVE_LAPACK" 
    fi
])

# stuff not required for standalone module
if test "$enable_module" != "yes"; then
    # Checks for libraries.
    AC_CHECK_LIB([fftw3], [fftw_malloc], HAVE_FFTW3="yes")
    if test "${HAVE_FFTW3}" != "yes" ; then
        AC_MSG_ERROR([Couldn't find fftw3.])
    fi

    if test "$LAPACKLIB" = ""; then
        if test "$STFKERNEL" = "darwin" ; then
            # System LAPACK
            LIBLAPACK_LDFLAGS="/usr/lib/liblapack.dylib -framework Accelerate"
            CPPFLAGS="${CPPFLAGS} -DHAVE_LAPACK -DHAVE_LAPACK_SUFFIX" 
        else
            AC_CHECK_LIB([lapack], [dgemm_], HAVE_LAPACKX="yes")
            if test "${HAVE_LAPACKX}" != "yes" ; then
                AC_CHECK_LIB([lapack3], [dgemm_], HAVE_LAPACK3="yes")
                if test "${HAVE_LAPACK3}" != "yes" ; then
                    AC_CHECK_LIB([lapack-3], [dgemm_], HAVE_LAPACK_3="yes")
                        if test "${HAVE_LAPACK_3}" != "yes" ; then
                            AC_CHECK_LIB([blas], [dgemm_], HAVE_ATLAS="yes")
                            if test "${HAVE_ATLAS}" != "yes" ; then
                                AC_MSG_ERROR([Couldn't find lapack.])
                            else
                                LIBLAPACK_LDFLAGS="-llapack -lblas"
                                CPPFLAGS="${CPPFLAGS} -DHAVE_LAPACK " 
                            fi
                        else
                            LIBLAPACK_LDFLAGS="-llapack-3"
                            CPPFLAGS="${CPPFLAGS} -DHAVE_LAPACK " 
                        fi
                else
                    LIBLAPACK_LDFLAGS=-llapack3
                    CPPFLAGS="${CPPFLAGS} -DHAVE_LAPACK " 
                fi
            else
                LIBLAPACK_LDFLAGS="-llapack -lblas"
                CPPFLAGS="${CPPFLAGS} -DHAVE_LAPACK " 
            fi
        fi
    fi
    AC_SUBST(LIBLAPACK_LDFLAGS)



    # Optionally enables aui for doc/view architecture
    AC_ARG_ENABLE([aui], AS_HELP_STRING([--enable-aui],[enable AUI for the doc/view architecture (experimental)]),[])
    if test "$enable_aui" = "yes" ; then
        CPPFLAGS="${CPPFLAGS} -DWITH_AUIDOCVIEW"
    fi

    WXCONFIG=wx-config
    AC_ARG_WITH(wx-config,
    [[  --with-wx-config=FILE     Use the given path to wx-config when determining
                                wxWidgets configuration; defaults to "wx-config"]],
    [
        if test "$withval" != "yes" -a "$withval" != ""; then
            WXCONFIG=$withval
        fi
    ])

    wxversion=0

    AC_DEFUN([WXTEST],
    [
            AC_REQUIRE([AC_PROG_AWK])
            AC_MSG_CHECKING([wxWidgets version])
            if wxversion=`$WXCONFIG --version`; then
                    AC_MSG_RESULT([$wxversion])
            else
                    AC_MSG_RESULT([not found])
                    AC_MSG_ERROR([wxWidgets is required. Try --with-wx-config.])
            fi])

    # Call WXTEST func
    WXTEST

    # Verify minimus requires
    AC_MSG_CHECKING(for wxWidgets libraries)
    vers=`echo $wxversion | $AWK 'BEGIN { FS = "."; } { printf "% d", ($1 * 1000 + $2) * 1000 + $3;}'`
    if test -n "$vers" && test "$vers" -ge 2008000; then
            WX_CPPFLAGS="`$WXCONFIG --cppflags`"
            WX_CXXFLAGS="`$WXCONFIG --cxxflags`"
            if test "$STFKERNEL" = "darwin" ; then
                if test "${PY_AC_VERSION:0:1}" -ge 3; then
                    WX_LIBS="`$WXCONFIG --libs base`"
                else
                    WX_LIBS="`$WXCONFIG --libs propgrid,aui,adv,core,net,base`"
                fi
            else
                WX_LIBS="`$WXCONFIG --libs base,core,adv,aui,net`"
            fi
            AC_MSG_RESULT([$WX_LIBS])
    else
            AC_MSG_ERROR([wxWidgets 2.8.0 or newer is required])
    fi

    # CPPFLAGS="$CPPFLAGS $WX_CPPFLAGS"
    # CXXFLAGS="$CXXFLAGS $WX_CXXFLAGS" dnl included in cppflags

    AC_SUBST(WX_LIBS)
    AC_SUBST(WX_CPPFLAGS)
    AC_SUBST(WX_CXXFLAGS)
dnl default value is to (silently) do nothing in the makefile
POSTLINK_COMMAND="@true"
MACSETFILE="@true"

AC_MSG_CHECKING(for wxWidgets platform)
WX_BASENAME="`$WXCONFIG --basename`"
case $WX_BASENAME in 
    *wx_osx*)
        AC_MSG_RESULT($WX_BASENAME)
        AC_CHECK_PROG(REZ, Rez, Rez, /Developer/Tools/Rez)
        POSTLINK_COMMAND="\$(REZ) -d __DARWIN__ -t APPL -o"
        AC_CHECK_PROG(SETFILE, SetFile, SetFile, /Developer/Tools/SetFile)
        MACSETFILE="\$(SETFILE)"
    ;;
    *wx_mac*)
        AC_MSG_RESULT($WX_BASENAME)
        AC_CHECK_PROG(REZ, Rez, Rez, /Developer/Tools/Rez)
        POSTLINK_COMMAND="\$(REZ) -d __DARWIN__ -t APPL -o"
        AC_CHECK_PROG(SETFILE, SetFile, SetFile, /Developer/Tools/SetFile)
        MACSETFILE="\$(SETFILE)"
    ;;
    *) 
        AC_MSG_RESULT(other)
    ;;
esac

AC_SUBST(POSTLINK_COMMAND)
AC_SUBST(MACSETFILE)

fi

# Checks for hdf5 libraries.
AC_ARG_WITH([hdf5-prefix],
AS_HELP_STRING([--with-hdf5-prefix=HDF5_PREFIX],[Provide full path to hdf5 prefix]),
[
    if test "$withval" != "yes" -a "$withval" != ""; then
        HDF5_PREFIX=${withval}
        LDFLAGS="${LDFLAGS} -L${HDF5_PREFIX}/lib"
        CPPFLAGS="${CPPFLAGS} -I${HDF5_PREFIX}/include" 
    fi
])

AC_CHECK_HEADER([hdf5.h], [], [AC_MSG_ERROR([Couldn't find hdf5 header])])
AC_CHECK_LIB([hdf5],[H5Fopen],HAVE_HDF5="yes")
if test "${HAVE_HDF5}" != "yes" ; then
    AC_MSG_ERROR([Couldn't find hdf5 libraries.])
else
    CPPFLAGS="${CPPFLAGS} -DH5_USE_16_API"
    LIBHDF5_LDFLAGS="-lhdf5 -lhdf5_hl"
fi
AC_SUBST(LIBHDF5_LDFLAGS)

AC_ARG_ENABLE([debug],
        AS_HELP_STRING([--enable-debug],[build stimfit in debug mode]),[
])
    if test "$enable_debug" = "yes" ; then
        CPPFLAGS="${CPPFLAGS} -D_STFDEBUG "
        OPT_CXXFLAGS="-O0 -g3"
        CFLAGS="${CFLAGS} -O0 -g3"
    else
        OPT_CXXFLAGS="-O2 -g"
        CFLAGS="${CFLAGS} -O2 -g"
    fi
AC_SUBST(OPT_CXXFLAGS)

# gtest
GT_CPPFLAGS=""
GT_CXXFLAGS=""
GT_LIBS="-lpthread"
GT_LDFLAGS=""

AC_SUBST(GT_CPPFLAGS)
AC_SUBST(GT_CXXFLAGS)
AC_SUBST(GT_LIBS)
AC_SUBST(GT_LDFLAGS)
# end gtest

# CPPFLAGS="${CPPFLAGS} -DSTFDATE='\"${BUILDDATE}\"'"
CXXFLAGS="${CXXFLAGS} -Wall"

AC_CONFIG_HEADERS([stfconf.h])

AC_CONFIG_FILES([Makefile src/Makefile src/libstfio/Makefile src/pystfio/Makefile src/stimfit/Makefile src/stimfit/py/Makefile dist/macosx/stimfit.plist])

AC_OUTPUT
