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
#include "./../core/core.h"

#include "stfioswig.h"

#define array_data(a)          (((PyArrayObject *)a)->data)

void wrap_array() {
    import_array();
}
    

%}
%include "numpy.i"
%include "std_string.i"
%include "std_vector.i"
%init %{
    import_array();
%}

class Recording {
 public:
    Recording();
    %feature("autodoc", "The sampling interval") dt;
    double dt;
    %feature("autodoc", "File description") file_description;
    %feature("autodoc", "The time of recording") time;
    %feature("autodoc", "The date of recording") date;
    %feature("autodoc", "Comment on the recording") comment;
    %feature("autodoc", "x unit string") xunits;
    std::string file_description, time, date, comment, xunits;

};

class Channel {
 public:
    %feature("autodoc", "Channel name") name;
    %feature("autodoc", "y unit string") yunits;
    std::string name, yunits;
    
};

class Section {
};

%extend Recording {
    Recording(PyObject* ChannelList) {
        if (!PyList_Check(ChannelList)) {
            std::cerr << "Argument is not a list\n";
            return NULL;
        }
        Py_ssize_t listsize = PyList_Size(ChannelList);
        std::vector<Channel> ChannelCpp(listsize);
        
        for (std::size_t i=0; i<listsize; ++i) {
            PyObject* sec0 = PyList_GetItem(ChannelList, i);
            void* argp1;
            int res1 = SWIG_ConvertPtr(sec0, &argp1, SWIGTYPE_p_Channel, 0 | 0 );
            if (!SWIG_IsOK(res1)) {
                std::cerr << "List doesn't consist of channels\n";
                return NULL;
            }
            Channel* arg1 = reinterpret_cast< Channel * >(argp1);

            ChannelCpp[i] = *arg1;
        }

        // Note that array size is fixed by this allocation:
        Recording* rec = new Recording(ChannelCpp);

        return rec;
    }
    ~Recording() {delete $self;}

    Channel& __getitem__(std::size_t at) {
        try {
            return $self->at(at);
        } catch (const std::out_of_range& e) {
            std::cerr << "Channel index out of range\n" << e.what() << std::endl;
        }
    }
    int __len__() { return $self->size(); }

    %feature("autodoc", "Reads a file and returns a Recording object.

    Arguments:
    fname  -- file name
    ftype  -- file type (string). At present, only \"hdf5\" is supported.

    Returns:
    True upon successful completion.") write;
    bool write(const std::string& fname, const std::string& ftype="hdf5") {
        stf::filetype stftype = gettype(ftype);
        return stf::exportFile(fname, stftype, *($self));
    }
}

%extend Channel {
    Channel(PyObject* SectionList) {
        if (!PyList_Check(SectionList)) {
            std::cerr << "Argument is not a list\n";
            return NULL;
        }
        Py_ssize_t listsize = PyList_Size(SectionList);
        std::vector<Section> SectionCpp(listsize);
        
        for (std::size_t i=0; i<listsize; ++i) {
            PyObject* sec0 = PyList_GetItem(SectionList, i);
            void* argp1;
            int res1 = SWIG_ConvertPtr(sec0, &argp1, SWIGTYPE_p_Section, 0 | 0 );
            if (!SWIG_IsOK(res1)) {
                std::cerr << "List doesn't consist of sections\n";
                return NULL;
            }
            Section* arg1 = reinterpret_cast< Section * >(argp1);

            SectionCpp[i] = *arg1;
        }

        // Note that array size is fixed by this allocation:
        Channel *ch = new Channel(SectionCpp);

        return ch;
    }

    ~Channel() {delete $self;}
    
    Section& __getitem__(std::size_t at) {
        try {
            return $self->at(at);
        } catch (const std::out_of_range& e) {
            std::cerr << "Section index out of range\n" << e.what() << std::endl;
        }
    }
    int __len__() { return $self->size(); }
}

%extend Section {

    Section(PyObject* nparray) {
        wrap_array();
        
        npy_intp nplen = PyArray_DIM(nparray, 0);

        // Note that array size is fixed by this allocation:
        Section *sec = new Section(nplen, "");

        double* npptr = (double*)PyArray_DATA(nparray);
        std::copy(&npptr[0], &npptr[nplen], &(sec->get_w()[0]));

        return sec;
    }
    ~Section() {
        delete($self);
    }
    double __getitem__(std::size_t at) {
        try {
            return $self->at(at);
        } catch (const std::out_of_range& e) {
            std::cerr << "Point index out of range\n" << e.what() << std::endl;
        }
    }
    int __len__() { return $self->size(); }

    %feature("autodoc", "Returns the section as a numpy array.") asarray;
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
    ftype  -- file type (string); can be one of:
              "cfs"  - CED filing system
              "hdf5" - HDF5
              "abf"  - Axon binary file
              "atf"  - Axon text file
              "axg"  - Axograph X binary file
              if ftype is None (default), it will be guessed from the
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
        elif ext==".atf":
            ftype = "atf"
        elif ext==".axgd" or ext==".axgx":
            ftype = "axg"
    rec = Recording()
    if not _read(fname, ftype, rec):
        return None
    return rec
        
}
//--------------------------------------------------------------------
