if BUILD_MODULE

else
noinst_LIBRARIES = libstimfit.a

libstimfit_a_SOURCES = unopt.cpp

libstimfit_a_CXXFLAGS = $(OPT_CXXFLAGS) # -O0 -g
libstimfit_a_LDFLAGS = $(LIBSTF_LDFLAGS)

# the application source, library search path, and link libraries
if BUILD_PYTHON
    PYTHON_ADDINCLUDES = $(LIBPYTHON_INCLUDES)
else
    PYTHON_ADDINCLUDES = 
endif

INCLUDES = $(PYTHON_ADDINCLUDES)
endif
