
AC_DEFUN([AC_PYTHON_DEVEL],[
        #
        # Allow the use of a (user set) custom python version
        #
        AC_ARG_VAR([PYTHON_VERSION],[The installed Python
                version to use, for example '2.3'. This string
                will be appended to the Python interpreter
                canonical name.])

        AC_PATH_PROG([PYTHON],[python[$PYTHON_VERSION]])
        if test -z "$PYTHON"; then
           AC_MSG_ERROR([Cannot find python$PYTHON_VERSION in your system path])
           PYTHON_VERSION=""
        fi

        #
        # Check for a version of Python >= 2.1.0
        #
        AC_MSG_CHECKING([for a version of Python >= '2.1.0'])
        ac_supports_python_ver=`$PYTHON -c "import sys, string; \
                ver = sys.version.split()[[0]]; \
                print(ver >= '2.1.0')"`
        if test "$ac_supports_python_ver" != "True"; then
                if test -z "$PYTHON_NOVERSIONCHECK"; then
                        AC_MSG_RESULT([no])
                        AC_MSG_FAILURE([
This version of the AC@&t@_PYTHON_DEVEL macro
doesn't work properly with versions of Python before
2.1.0. You may need to re-run configure, setting the
variables PYTHON_CPPFLAGS, PYTHON_LDFLAGS, PYTHON_SITE_PKG,
PYTHON_EXTRA_LIBS and PYTHON_EXTRA_LDFLAGS by hand.
Moreover, to disable this check, set PYTHON_NOVERSIONCHECK
to something else than an empty string.
])
                else
                        AC_MSG_RESULT([skip at user request])
                fi
        else
                AC_MSG_RESULT([yes])
        fi

        #
        # if the macro parameter ``version'' is set, honour it
        #
        if test -n "$1"; then
                AC_MSG_CHECKING([for a version of Python $1])
                ac_supports_python_ver=`$PYTHON -c "import sys, string; \
                        ver = string.split(sys.version)[[0]]; \
                        sys.stdout.write(ver + '$1' + '\n')"`
                if test "$ac_supports_python_ver" = "True"; then
                   AC_MSG_RESULT([yes])
                else
                        AC_MSG_RESULT([no])
                        AC_MSG_ERROR([this package requires Python $1.
If you have it installed, but it isn't the default Python
interpreter in your system path, please pass the PYTHON_VERSION
variable to configure. See ``configure --help'' for reference.
])
                        PYTHON_VERSION=""
                fi
        fi

        #
        # Check if you have distutils, else fail
        #
        AC_MSG_CHECKING([for the distutils Python package])
        ac_distutils_result=`$PYTHON -c "import distutils" 2>&1`
        if test -z "$ac_distutils_result"; then
                AC_MSG_RESULT([yes])
        else
                AC_MSG_RESULT([no])
                AC_MSG_ERROR([cannot import Python module "distutils".
Please check your Python installation. The error was:
$ac_distutils_result])
                PYTHON_VERSION=""
        fi

        #
        # Check for Python include path
        #
        AC_MSG_CHECKING([for Python include path])
        if test -z "$PYTHON_CPPFLAGS"; then
                python_path=`$PYTHON -c "import distutils.sysconfig, sys; \
                        sys.stdout.write(distutils.sysconfig.get_python_inc() + '\n');"`
                if test -n "${python_path}"; then
                        python_path="-I$python_path"
                fi
                PYTHON_CPPFLAGS=$python_path
        fi
        AC_MSG_RESULT([$PYTHON_CPPFLAGS])
        AC_SUBST([PYTHON_CPPFLAGS])

        #
        # Check for Python library path
        #
        #
        # Check for Python library path
        #
        AC_MSG_CHECKING([for Python library path])
        if test -z "$PYTHON_LDFLAGS"; then
                # (makes two attempts to ensure we've got a version number
                # from the interpreter)
		py_version=`$PYTHON -c \
"import sys; from distutils.sysconfig import *
if get_config_vars('LDVERSION')[[0]] is None:
    sys.stdout.write(' '.join(get_config_vars('VERSION'))+'\n')
else:
    sys.stdout.write(' '.join(get_config_vars('LDVERSION'))+'\n')"`
                if test "$py_version" == "[None]"; then
                        if test -n "$PYTHON_VERSION"; then
                                py_version=$PYTHON_VERSION
                        else
                                py_version=`$PYTHON -c "import sys; \
                                        sys.stdout.write(sys.version[[:3]] + '\n')"`
                        fi
                fi
                PY_AC_VERSION=$py_version
                PYTHON_LDFLAGS=`$PYTHON -c "import sys; from distutils.sysconfig import *; \
                        sys.stdout.write('-L' + get_config_vars()[['LIBDIR']] + \
                        ' -lpython' + '\n');"`$py_version
        fi
        AC_MSG_RESULT([$PYTHON_LDFLAGS])
        AC_SUBST([PYTHON_LDFLAGS])
        AC_SUBST([PY_AC_VERSION])

        #
        # Check for prefixed site packages
        #
        AC_MSG_CHECKING([for prefixed Python site-packages path])
        if test -z "$PYTHON_SITE_PKG"; then
                PYTHON_SITE_PKG=`$PYTHON -c \
"import sys, distutils.sysconfig; \
acprefix = \"${prefix}\"
if acprefix is \"NONE\": acprefix=\"/usr/local/\"
sys.stdout.write(distutils.sysconfig.get_python_lib(0,1,prefix=acprefix)+'\n');"`
                PYTHON_SITE_PKG="${PYTHON_SITE_PKG}/dist-packages"

        fi
        AC_MSG_RESULT([$PYTHON_SITE_PKG])
        AC_SUBST([PYTHON_SITE_PKG])

        #
        # Check for prefixed dist packages
        #
        AC_MSG_CHECKING([for prefixed Python dist-packages path])
        if test -z "$PYTHON_PRE_DIST_PKG"; then
                PYTHON_PRE_DIST_PKG=`$PYTHON -c \
"import sys, distutils.sysconfig; \
acprefix = \"${prefix}\"
if acprefix is \"NONE\": acprefix=\"/usr/local/\"
sys.stdout.write(distutils.sysconfig.get_python_lib(0,0,prefix=acprefix)+'\n');"`
                PYTHON_PRE_DIST_PKG=${PYTHON_PRE_DIST_PKG}

        fi
        AC_MSG_RESULT([$PYTHON_PRE_DIST_PKG])
        AC_SUBST([PYTHON_PRE_DIST_PKG])

        #
        # Check for unprefixed dist packages path
        #
        AC_MSG_CHECKING([for unprefixed Python dist-packages path])
        if test -z "$PYTHON_DIST_PKG"; then
                PYTHON_DIST_PKG=`$PYTHON -c \
"import sys, distutils.sysconfig; \
sys.stdout.write(distutils.sysconfig.get_python_lib(0,0)+'\n');"`
                PYTHON_DIST_PKG=${PYTHON_DIST_PKG}

        fi
        AC_MSG_RESULT([$PYTHON_DIST_PKG])
        AC_SUBST([PYTHON_DIST_PKG])

        #
        # Check if you have numpy, else fail
        #
        AC_MSG_CHECKING([for numpy])
        ac_numpy_result=`$PYTHON -c "import numpy" 2>&1`
        if test -z "$ac_numpy_result"; then
                AC_MSG_RESULT([yes])
        else
                AC_MSG_RESULT([no])
                AC_MSG_ERROR([cannot import Python module "numpy".
Please check your numpy installation. The error was:
$ac_numpy_result])
                PYTHON_VERSION=""
        fi

        #
        # Check for numpy headers
        #
        AC_MSG_CHECKING([for numpy include path])
        if test -z "$PYTHON_NUMPY_INCLUDE"; then
                PYTHON_NUMPY_INCLUDE=-I`$PYTHON -c "import sys, numpy; \
                        sys.stdout.write(numpy.lib.get_include() + '\n');"`
        fi
        AC_MSG_RESULT([$PYTHON_NUMPY_INCLUDE])
        AC_SUBST([PYTHON_NUMPY_INCLUDE])

        #
        # Check if you have wxPython, else fail
        #
        AC_MSG_CHECKING([for wxPython])
        ac_wxpython_result=`$PYTHON -c "import wx" 2>&1`
        if test -z "$ac_wxpython_result"; then
                AC_MSG_RESULT([yes])
        else
                AC_MSG_RESULT([no])
                AC_MSG_ERROR([cannot import Python module "wxpython".
Please check your wxpython installation. The error was:
$ac_wxpython_result])
                PYTHON_VERSION=""
        fi

        #
        # Check for wxpython headers
        #
        AC_MSG_CHECKING([for wxpython include path])
        if test -z "$PYTHON_WXPYTHON_INCLUDE"; then
                PYTHON_WXPYTHON_INCLUDE=-I`$PYTHON -c "import os, sys, wx; \
                        sys.stdout.write(os.path.join(os.path.dirname(wx.__spec__.origin), 'include') + '\n');"`
        fi
        AC_MSG_RESULT([$PYTHON_WXPYTHON_INCLUDE])
        AC_SUBST([PYTHON_WXPYTHON_INCLUDE])

        #
        # libraries which must be linked in when embedding
        #
        AC_MSG_CHECKING(python extra libraries)
        if test -z "$PYTHON_EXTRA_LIBS"; then
           PYTHON_EXTRA_LIBS=`$PYTHON -c "import sys, distutils.sysconfig; \
                conf = distutils.sysconfig.get_config_var; \
                sys.stdout.write(conf('LOCALMODLIBS') + ' ' + conf('LIBS') + ' ' + '\n')"`
        fi
        AC_MSG_RESULT([$PYTHON_EXTRA_LIBS])
        AC_SUBST(PYTHON_EXTRA_LIBS)

        #
        # linking flags needed when embedding
        #
        AC_MSG_CHECKING(python extra linking flags)
        if test -z "$PYTHON_EXTRA_LDFLAGS"; then
                PYTHON_EXTRA_LDFLAGS=`$PYTHON -c "import sys, distutils.sysconfig; \
                        conf = distutils.sysconfig.get_config_var; \
                        sys.stdout.write(conf('LINKFORSHARED')+'\n')"`
        fi
        AC_MSG_RESULT([$PYTHON_EXTRA_LDFLAGS])
        AC_SUBST(PYTHON_EXTRA_LDFLAGS)

        #
        # final check to see if everything compiles alright
        #
        AC_MSG_CHECKING([consistency of all components of python development environment])
        AC_LANG_PUSH([C])
        # save current global flags
        LIBS="$ac_save_LIBS $PYTHON_LDFLAGS -lm"
        CPPFLAGS="$ac_save_CPPFLAGS $PYTHON_CPPFLAGS"
        AC_TRY_LINK([
                #include <Python.h>
        ],[
                Py_Initialize();
        ],[pythonexists=yes],[pythonexists=no])

        AC_MSG_RESULT([$pythonexists])

        if test ! "$pythonexists" = "yes"; then
           AC_MSG_ERROR([
  Could not link test program to Python. Maybe the main Python library has been
  installed in some non-standard library path. If so, pass it to configure,
  via the LDFLAGS environment variable.
  Example: ./configure LDFLAGS="-L/usr/non-standard-path/python/lib"
  ============================================================================
   ERROR!
   You probably have to install the development version of the Python package
   for your distribution.  The exact name of this package varies among them.
  ============================================================================
           ])
          PYTHON_VERSION=""
        fi
        AC_LANG_POP
        # turn back to default flags
        CPPFLAGS="$ac_save_CPPFLAGS"
        LIBS="$ac_save_LIBS"

        #
        # all done!
        #
])

##### http://autoconf-archive.cryp.to/ac_pkg_swig.html
#
# SYNOPSIS
#
#   AC_PROG_SWIG([major.minor.micro])
#
# DESCRIPTION
#
#   This macro searches for a SWIG installation on your system. If
#   found you should call SWIG via $(SWIG). You can use the optional
#   first argument to check if the version of the available SWIG is
#   greater than or equal to the value of the argument. It should have
#   the format: N[.N[.N]] (N is a number between 0 and 999. Only the
#   first N is mandatory.)
#
#   If the version argument is given (e.g. 1.3.17), AC_PROG_SWIG checks
#   that the swig package is this version number or higher.
#
#   In configure.in, use as:
#
#     AC_PROG_SWIG(1.3.17)
#     SWIG_ENABLE_CXX
#     SWIG_MULTI_MODULE_SUPPORT
#     SWIG_PYTHON
#
# LAST MODIFICATION
#
#   2006-10-22
#
# COPYLEFT
#
#   Copyright (c) 2006 Sebastian Huber <sebastian-huber@web.de>
#   Copyright (c) 2006 Alan W. Irwin <irwin@beluga.phys.uvic.ca>
#   Copyright (c) 2006 Rafael Laboissiere <rafael@laboissiere.net>
#   Copyright (c) 2006 Andrew Collier <colliera@ukzn.ac.za>
#
#   This program is free software; you can redistribute it and/or
#   modify it under the terms of the GNU General Public License as
#   published by the Free Software Foundation; either version 2 of the
#   License, or (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#   General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the Free Software
#   Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
#   02111-1307, USA.
#
#   As a special exception, the respective Autoconf Macro's copyright
#   owner gives unlimited permission to copy, distribute and modify the
#   configure scripts that are the output of Autoconf when processing
#   the Macro. You need not follow the terms of the GNU General Public
#   License when using or distributing such scripts, even though
#   portions of the text of the Macro appear in them. The GNU General
#   Public License (GPL) does govern all other use of the material that
#   constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the
#   Autoconf Macro released by the Autoconf Macro Archive. When you
#   make and distribute a modified version of the Autoconf Macro, you
#   may extend this special exception to the GPL to apply to your
#   modified version as well.

AC_DEFUN([AC_PROG_SWIG],[
        AC_PATH_PROG([SWIG],[swig])
        if test -z "$SWIG" ; then
                AC_MSG_WARN([cannot find 'swig' program. You should look at http://www.swig.org])
                SWIG='echo "Error: SWIG is not installed. You should look at http://www.swig.org" ; false'
        elif test -n "$1" ; then
                AC_MSG_CHECKING([for SWIG version])
                [swig_version=`$SWIG -version 2>&1 | grep 'SWIG Version' | sed 's/.*\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\).*/\1/g'`]
                AC_MSG_RESULT([$swig_version])
                if test -n "$swig_version" ; then
                        # Calculate the required version number components
                        [required=$1]
                        [required_major=`echo $required | sed 's/[^0-9].*//'`]
                        if test -z "$required_major" ; then
                                [required_major=0]
                        fi
                        [required=`echo $required | sed 's/[0-9]*[^0-9]//'`]
                        [required_minor=`echo $required | sed 's/[^0-9].*//'`]
                        if test -z "$required_minor" ; then
                                [required_minor=0]
                        fi
                        [required=`echo $required | sed 's/[0-9]*[^0-9]//'`]
                        [required_patch=`echo $required | sed 's/[^0-9].*//'`]
                        if test -z "$required_patch" ; then
                                [required_patch=0]
                        fi
                        # Calculate the available version number components
                        [available=$swig_version]
                        [available_major=`echo $available | sed 's/[^0-9].*//'`]
                        if test -z "$available_major" ; then
                                [available_major=0]
                        fi
                        [available=`echo $available | sed 's/[0-9]*[^0-9]//'`]
                        [available_minor=`echo $available | sed 's/[^0-9].*//'`]
                        if test -z "$available_minor" ; then
                                [available_minor=0]
                        fi
                        [available=`echo $available | sed 's/[0-9]*[^0-9]//'`]
                        [available_patch=`echo $available | sed 's/[^0-9].*//'`]
                        if test -z "$available_patch" ; then
                                [available_patch=0]
                        fi
                        if test $available_major -ne $required_major \
                                -o $available_minor -ne $required_minor \
                                -o $available_patch -lt $required_patch ; then
                                if test $available_major -lt $required_major ; then
                                AC_MSG_WARN([SWIG version >= $1 is required.  You have $swig_version.  You should look at http://www.swig.org])
                                SWIG='echo "Error: SWIG version >= $1 is required.  You have '"$swig_version"'.  You should look at http://www.swig.org" ; false'
                                fi
                        else
                                AC_MSG_NOTICE([SWIG executable is '$SWIG'])
                                SWIG_LIB=`$SWIG -swiglib`
                                AC_MSG_NOTICE([SWIG library directory is '$SWIG_LIB'])
                        fi
                else
                        AC_MSG_WARN([cannot determine SWIG version])
                        SWIG='echo "Error: Cannot determine SWIG version.  You should look at http://www.swig.org" ; false'
                fi
        fi
        AC_SUBST([SWIG_LIB])
])

AC_DEFUN([SWIG_ENABLE_CXX],[
        AC_REQUIRE([AC_PROG_SWIG])
        AC_REQUIRE([AC_PROG_CXX])
        SWIG="$SWIG -c++"
])

AC_DEFUN([SWIG_PYTHON],[
        AC_REQUIRE([AC_PROG_SWIG])
        AC_REQUIRE([AC_PYTHON_DEVEL])
        test "x$1" != "xno" || swig_shadow=" -noproxy"
        AC_SUBST([SWIG_PYTHON_OPT],[-python$swig_shadow])
        AC_SUBST([SWIG_PYTHON_CPPFLAGS],[$PYTHON_CPPFLAGS])
])
