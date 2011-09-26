%define DOCSTRING
"The stfio module provides functions to read/write data from/to
common electrophysiology file formats"
%enddef

%module(docstring=DOCSTRING) stfio

%{
#define SWIG_FILE_WITH_INIT
#include <string>
#include <iostream>
#include <numpy/arrayobject.h>
    
#include "./../libstfio/stfio.h"
#include "./../libstfio/recording.h"
#include "./../libstfio/channel.h"
#include "./../libstfio/section.h"

#include "stfioswig.h"

#define array_data(a)          (((PyArrayObject *)a)->data)

#if PY_MAJOR_VERSION >= 3
int
#else
void
#endif
wrap_array() {
    import_array();
}
    

%}
%include "numpy.i"
%include "std_string.i"
%init %{
    import_array();
%}

class Recording {
 public:
    Recording();
    /* %feature("autodoc", "The sampling interval") dt;
       double dt;
    %feature("autodoc", "File description") file_description;
    %feature("autodoc", "The time of recording") time;
    %feature("autodoc", "The date of recording") date;
    %feature("autodoc", "Comment on the recording") comment;
    %feature("autodoc", "x unit string") xunits;
    std::string file_description, time, date, comment, xunits;
    */
};

class Channel {
 public:
    /*    %feature("autodoc", "Channel name") name;
    %feature("autodoc", "y unit string") yunits;
    std::string name, yunits;*/
    
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

    double dt;
    char* file_description;
    char* time;
    char* date;
    char* comment;
    char* xunits;
    
    Channel& __getitem__(int at) {
        if (at >= 0 && at < $self->size()) {
            return (*($self))[at];
        } else {
            PyErr_SetString(PyExc_IndexError, "Channel index out of range");
            std::cerr << "Channel index " << at << " out of range\n" << std::endl;
            throw std::out_of_range("Channel index out of range");
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
        stfio::filetype stftype = gettype(ftype);
        StdoutProgressInfo progDlg("File import", "Reading file", 100);
        try {
            return stfio::exportFile(fname, stftype, *($self), progDlg);
        } catch (const std::exception& e) {
            std::cerr << "Couldn't write to file:\n"
                      << e.what() << std::endl;
            return false;
        }
    }
}

%{
    double Recording_dt_get(Recording *r) {
        return r->GetXScale();
    }
    void Recording_dt_set(Recording *r, double val) {
        r->SetXScale(val);
    }
    const char* Recording_file_description_get(Recording *r) {
        return r->GetFileDescription().c_str();
    }
    void Recording_file_description_set(Recording *r, char* val) {
        r->SetFileDescription(std::string(val));
    }
    const char* Recording_time_get(Recording *r) {
        return r->GetTime().c_str();
    }
    void Recording_time_set(Recording *r, char* val) {
        r->SetTime(std::string(val));
    }
    const char* Recording_date_get(Recording *r) {
        return r->GetDate().c_str();
    }
    void Recording_date_set(Recording *r, char* val) {
        r->SetDate(std::string(val));
    }
    const char* Recording_xunits_get(Recording *r) {
        return r->GetXUnits().c_str();
    }
    void Recording_xunits_set(Recording *r, char* val) {
        r->SetXUnits(std::string(val));
    }
    const char* Recording_comment_get(Recording *r) {
        return r->GetComment().c_str();
    }
    void Recording_comment_set(Recording *r, char* val) {
        r->SetComment(std::string(val));
    }

%}

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
    
    Section& __getitem__(int at) {
        if (at >= 0 && at < $self->size()) {
            return (*($self))[at];
        } else {
            PyErr_SetString(PyExc_IndexError, "Section index out of range");
            std::cerr << "Section index " << at << " out of range\n" << std::endl;
            throw std::out_of_range("Section index out of range");
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
    double __getitem__(int at) {
        if (at >= 0 && at < $self->size()) {
            return (*($self))[at];
        } else {
            PyErr_SetString(PyExc_IndexError, "Point index out of range");
            std::cerr << "Point index " << at << " out of range\n" << std::endl;
            throw std::out_of_range("Point index out of range");
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

# code added by Jose
class StfIOException(Exception):
    """ raises Exceptions for the Stfio module """
    def __init__(self, error_msg):
        self.msg = error_msg 

    def __str__(self):
        return repr(self.msg)

filetype = {
    '.dat':'cfs',
    '.h5':'hdf5',
    '.abf':'abf',
    '.atf':'atf',
    '.axgd':'axg',
    '.axgx':'axg'}

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
              "heka" - HEKA binary file
              if ftype is None (default), it will be guessed from the
              extension.

    Returns:
    A Recording object.
    """
    if not os.path.exists(fname):
        raise StfIOException('File %s does not exist' % fname)
    
    if ftype is None:
        # Guess file type:
        ext = os.path.splitext(fname)[1]
        try:
            ftype = filetype[ext]
        except KeyError:
            raise StfIOException('Couldn\'t guess file type from extension (%s)' % ext)

    rec = Recording()
    if not _read(fname, ftype, rec):
        raise StfIOException('Error reading file')

    return rec

}
//--------------------------------------------------------------------
