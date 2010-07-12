%define DOCSTRING
"The stf module allows to access a running stimfit
application from the embedded python shell."
%enddef

%module(docstring=DOCSTRING) stfio

%{
#define SWIG_FILE_WITH_INIT
#include "stfioswig.h"
%}
%include "numpy.i"
%include "std_string.i"
%init %{
import_array();
%}

%define %apply_numpy_typemaps(TYPE)

%apply (TYPE* ARGOUT_ARRAY1, int DIM1) {(TYPE* outvec, int size)};
%apply (TYPE* IN_ARRAY1, int DIM1) {(TYPE* invec, int size)};
%apply (TYPE* IN_ARRAY2, int DIM1, int DIM2) {(TYPE* inarr, int traces, int size)};

%enddef    /* %apply_numpy_typemaps() macro */

%apply_numpy_typemaps(double)

//--------------------------------------------------------------------
%pythoncode {
import numpy as np
}
//--------------------------------------------------------------------
