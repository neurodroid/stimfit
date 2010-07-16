%define DOCSTRING
"The stfio module provides functions to read/write data from/to
common electrophysiology file formats"
%enddef

%module(docstring=DOCSTRING) stfio

%{
#define SWIG_FILE_WITH_INIT
#include <string>
#include <numpy/arrayobject.h>
    
#include "./../core/recording.h"
#include "./../core/channel.h"
#include "./../core/section.h"
    
#include "stfioswig.h"

#define array_data(a)          (((PyArrayObject *)a)->data)

%}
%include "numpy.i"
%include "std_string.i"
%init %{
import_array();
%}

class Recording {
 public:
    double dt;
    std::string time, date, comment, xunits;

};

class Channel {
 public:
    std::string name, yunits;
    
};

class Section {
};

%extend Recording {
    Channel& __getitem__(std::size_t at) { return $self->operator[](at); }
    int __len__() { return $self->size(); }
}

%extend Channel {
    Section& __getitem__(std::size_t at) { return $self->operator[](at); }
    int __len__() { return $self->size(); }
}

%extend Section {
    double __getitem__(std::size_t at) { return $self->operator[](at); }
    int __len__() { return $self->size(); }
    PyObject* asarray() {
        npy_intp dims[1] = {$self->size()};
        PyObject* np_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        double* gDataP = (double*)array_data(np_array);

        std::copy( &($self->operator[](0)),
                   &($self->operator[]($self->size())),
                   gDataP);
        return np_array;
    };
}

//--------------------------------------------------------------------
%feature("autodoc", 0) _read;
%feature("docstring", "Reads a file and returns a recording object.
      
Arguments:
filename -- file name
ftype    -- File type

Returns:
A recording object.") _read;
bool _read(const std::string& filename, const std::string& ftype, Recording& Data);
//--------------------------------------------------------------------


//--------------------------------------------------------------------
%pythoncode {
import os
        
def read(fname, ftype=None):
    """Reads a file and returns a Recording object.

    Arguments:
    fname  -- file name
    ftype  -- file type
              if type is None (default), it will be guessed from the
              extension.

    Returns:
    A Recording object.
    """
    if ftype is None:
        # Guess file type:
        ext = os.path.splitext(fname)[1]
        if ext==".dat": 
            ftype = "cfs"
        elif ext==".h5":
            ftype = "hdf5"
        elif ext==".abf":
            ftype = "abf"
    rec = Recording()
    if not _read(fname, ftype, rec):
        return None
    return rec
}
//--------------------------------------------------------------------
