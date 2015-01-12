%define DOCSTRING
"The stfio module provides functions to read/write data from/to
common electrophysiology file formats"
%enddef

%module(docstring=DOCSTRING) stfio

%{
#define SWIG_FILE_WITH_INIT
#include <string>
#include <iostream>
#include <cassert>
#include <ctime>

#include <numpy/arrayobject.h>
#include <datetime.h>

#include "./../libstfio/stfio.h"
#include "./../libstfio/recording.h"
#include "./../libstfio/channel.h"
#include "./../libstfio/section.h"

#include "pystfio.h"

static int myErr = 0; // flag to save error state
%}
%include "../stimfit/py/numpy.i"
%include "std_string.i"
%include "exception.i"
%init %{
    import_array();
    PyDateTime_IMPORT;
%}


%define %apply_numpy_typemaps(TYPE)

%apply (TYPE* IN_ARRAY1, int DIM1) {(TYPE* invec, int size)};
%apply (TYPE* IN_ARRAY1, int DIM1) {(TYPE* data, int size_data)};
%apply (TYPE* IN_ARRAY1, int DIM1) {(TYPE* templ, int size_templ)};

%enddef    /* %apply_numpy_typemaps() macro */

%apply_numpy_typemaps(double)

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
    %feature("autodoc", "The date and time of recording") datetime;
    std::string file_description, time, date, comment, xunits;
    PyObject* datetime;
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

%exception Recording::__getitem__ {
    assert(!myErr);
    $action
    if (myErr) {
        myErr = 0;
        SWIG_exception(SWIG_IndexError, "Index out of bounds");
    }
}

%exception Channel::__getitem__ {
    assert(!myErr);
    $action
    if (myErr) {
        myErr = 0;
        SWIG_exception(SWIG_IndexError, "Index out of bounds");
    }
}

%exception Section::__getitem__ {
    assert(!myErr);
    $action
    if (myErr) {
        myErr = 0;
        SWIG_exception(SWIG_IndexError, "Index out of bounds");
    }
}

%extend Recording {
    Recording(PyObject* ChannelList) :
       dt(1.0),
       file_description(""),
       time(""),
       date(""),
       comment(""),
       xunits("")
    {
        if (!PyList_Check(ChannelList)) {
            std::cerr << "Argument is not a list\n";
            return NULL;
        }
        Py_ssize_t listsize = PyList_Size(ChannelList);
        std::vector<Channel> ChannelCpp(listsize);
        
        for (Py_ssize_t i=0; i<listsize; ++i) {
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
    std::string file_description;
    std::string time;
    std::string date;
    std::string comment;
    std::string xunits;
    PyObject* datetime;
    
    Channel* __getitem__(int at) {
        if (at >= 0 && at < (int)$self->size()) {
            return &(*($self))[at];
        } else {
            myErr = 1;
            return NULL;
        }
    }
    int __len__() { return $self->size(); }

    %feature("autodoc", "Writes a Recording to a file.

    Arguments:
    fname  -- file name
#ifndef TEST_MINIMAL
    ftype  -- file type (string). At present, only \"hdf5\" is supported.
#else
    ftype  -- file type (string). At present, \"hdf5\", \"gdf\", \"cfs\" and \"ibw\" are supported.
#endif // TEST_MINIMAL
    verbose-- Show info while writing

    Returns:
    True upon successful completion.") write;
    bool write(const std::string& fname, const std::string& ftype="hdf5", bool verbose=false) {
        stfio::filetype stftype = gettype(ftype);
        stfio::StdoutProgressInfo progDlg("File export", "Writing file", 100, verbose);
        try {
            return stfio::exportFile(fname, stftype, *($self), progDlg);
        } catch (const std::exception& e) {
            std::cerr << "Couldn't write to file:\n"
                      << e.what() << std::endl;
            return false;
        }
    }

    %pythoncode {
        def aspandas(self):
            import sys
            import numpy as np
            has_pandas = True
            try:
                import pandas as pd
            except ImportError:
                has_pandas = False
            if has_pandas:
                chnames = [ch.name for ch in self]
                channels = np.array([np.concatenate([sec for sec in ch]) for ch in self])
                date_range = pd.date_range(start=self.datetime, periods=channels.shape[1],
                                           freq='%dU' % np.round(self.dt*1e3))
                return pd.DataFrame(channels.transpose(), index=date_range, columns=chnames)
            else:
                sys.stderr.write("Pandas is not available on this system\n")
                return None
    }

}

%{
    double Recording_dt_get(Recording *r) {
        return r->GetXScale();
    }
    void Recording_dt_set(Recording *r, double val) {
        r->SetXScale(val);
    }
    const std::string& Recording_file_description_get(Recording *r) {
        return r->GetFileDescription();
    }
    void Recording_file_description_set(Recording *r, const std::string& val) {
        r->SetFileDescription(val);
    }
    const std::string& Recording_time_get(Recording *r) {
        return r->GetTime();
    }
    void Recording_time_set(Recording *r, const std::string& val) {
        r->SetTime(val);
    }
    const std::string& Recording_date_get(Recording *r) {
        return r->GetDate();
    }
    void Recording_date_set(Recording *r, const std::string& val) {
        r->SetDate(val);
    }
    const std::string& Recording_xunits_get(Recording *r) {
        return r->GetXUnits();
    }
    void Recording_xunits_set(Recording *r, const std::string& val) {
        r->SetXUnits(val);
    }
    const std::string& Recording_comment_get(Recording *r) {
        return r->GetComment();
    }
    void Recording_comment_set(Recording *r, const std::string& val) {
        r->SetComment(val);
    }
    PyObject* Recording_datetime_get(Recording *r) {
        struct tm rec_tm = r->GetDateTime();
        if (rec_tm.tm_hour < 0 || rec_tm.tm_hour >= 24) {
            std::cerr << "Date out of range: hour is " << rec_tm.tm_hour
                      << std::endl;
        }
        return PyDateTime_FromDateAndTime(rec_tm.tm_year+1900, rec_tm.tm_mon+1, rec_tm.tm_mday,
                                          rec_tm.tm_hour, rec_tm.tm_min, rec_tm.tm_sec, 0);
    }
    void Recording_datetime_set(Recording *r, const PyObject* val) {
        if (val != NULL && PyDate_Check(val)) {
            int year = PyDateTime_GET_YEAR(val);
            int month = PyDateTime_GET_MONTH(val);
            int day = PyDateTime_GET_DAY(val);
            int hour = PyDateTime_DATE_GET_HOUR(val);
            int minute = PyDateTime_DATE_GET_MINUTE(val);
            int second = PyDateTime_DATE_GET_SECOND(val);
            r->SetDateTime(year, month, day, hour, minute, second);
        }
    }

%}

%extend Channel {
 Channel(PyObject* SectionList, const std::string& yunits_="") :
    name(""),
    yunits(yunits_)
    {
        if (!PyList_Check(SectionList)) {
            std::cerr << "Argument is not a list\n";
            return NULL;
        }
        Py_ssize_t listsize = PyList_Size(SectionList);
        std::vector<Section> SectionCpp(listsize);
        
        for (Py_ssize_t i=0; i<listsize; ++i) {
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
        ch->SetYUnits(yunits_);

        return ch;
    }

    ~Channel() {delete $self;}

    std::string name;
    std::string yunits;
    
    Section* __getitem__(int at) {
        if (at >= 0 && at < (int)$self->size()) {
            return &(*($self))[at];
        } else {
            myErr = 1;
            return NULL;
        }
    }
    int __len__() { return $self->size(); }
}

%{
    const std::string& Channel_name_get(Channel *c) {
        return c->GetChannelName();
    }
    void Channel_name_set(Channel *c, const std::string& val) {
        c->SetChannelName(val);
    }
    const std::string& Channel_yunits_get(Channel *c) {
        return c->GetYUnits();
    }
    void Channel_yunits_set(Channel *c, const std::string& val) {
        c->SetYUnits(val);
    }

%}

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
        if (at >= 0 && at < (int)$self->size()) {
            return (*($self))[at];
        } else {
            myErr = 1;
            return 0;
        }
    }
    int __len__() { return $self->size(); }

    %feature("autodoc", "Returns the section as a numpy array.") asarray;
    PyObject* asarray() {
        npy_intp dims[1] = {$self->size()};
        PyObject* np_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        double* gDataP = (double*)array_data(np_array);

        std::copy( $self->get().begin(),
                   $self->get().end(),
                   gDataP);
        return np_array;
    };
}

//--------------------------------------------------------------------
%feature("autodoc", 0) _read;
%feature("docstring", "Reads a file and returns a recording object.
      
Arguments:
filename -- file name
#ifndef TEST_MINIMAL
ftype    -- File type
#else
ftype    -- File type (obsolete)
#endif // TEST_MINIMAL
verbose  -- Show info while reading

Returns:
A recording object.") _read;
bool _read(const std::string& filename, const std::string& ftype, bool verbose, Recording& Data);
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) detect_events;
%feature("kwargs") detect_events;
%feature("docstring", "
      
Arguments:
") detect_events;
PyObject* detect_events(double* data, int size_data, double* templ, int size_templ, double dt,
                        const std::string& mode="criterion",
                        bool norm=true, double lowpass=0.5, double highpass=0.0001);
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) peak_detection;
%feature("kwargs") peak_detection;
%feature("docstring", "

Arguments:
") peak_detection;
PyObject* peak_detection(double* invec, int size, double threshold, int min_distance);
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%pythoncode {
import os

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

def read(fname, ftype=None, verbose=False):
    """Reads a file and returns a Recording object.

    Arguments:
    fname  -- file name
#ifndef TEST_MINIMAL
    ftype  -- file type (string); can be one of:
              "cfs"  - CED filing system
              "hdf5" - HDF5
              "abf"  - Axon binary file
              "atf"  - Axon text file
              "axg"  - Axograph X binary file
              "heka" - HEKA binary file
              if ftype is None (default), it will be guessed from the
              extension.
#else
    ftype  -- file type (string) is obsolete.
              in the past it has been used to determine the file type.
              Now an automated file type identification is used, and this
              parameter become obsolete; eventually it will be removed.
#endif // TEST_MINIMAL
    verbose-- Show info while reading file

    Returns:
    A Recording object.
    """
    if not os.path.exists(fname):
        raise StfIOException('File %s does not exist' % fname)

#ifndef TEST_MINIMAL
    if ftype is None:
        ext = os.path.splitext(fname)[1]
        try:
            ftype = filetype[ext]
        except KeyError:
            raise StfIOException('Couldn\'t guess file type from extension (%s)' % ext)
#endif // TEST_MINIMAL

    rec = Recording()
    if not _read(fname, ftype, verbose, rec):
        raise StfIOException('Error reading file')

    if verbose:
        print("")
        
    return rec

}
//--------------------------------------------------------------------
