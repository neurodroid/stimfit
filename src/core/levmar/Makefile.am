#
# Unix/Linux GCC Makefile for Levenberg - Marquardt minimization
# adapted for building into stimfit
# Under windows, use Makefile.vc for MSVC
#

## Libraries that we make from this directory:
lib_LTLIBRARIES = liblevmar.la

# the application source, library search path, and link libraries
libextend_la_SOURCES = lm.c Axb.c misc.c lmlec.c lmbc.c

noinst_HEADERS = lm.h misc.h

INCLUDES = -I../

LAPACKFLAG=-DHAVE_LAPACK # COMMENT THIS LINE IF YOU DO NOT HAVE LAPACK

