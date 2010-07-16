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

%apply (TYPE* ARGOUT_ARRAY1, int DIM1) {(TYPE* outvec, int npts)};
%apply (TYPE* IN_ARRAY1, int DIM1) {(TYPE* invec, int npts)};
%apply (TYPE* IN_ARRAY2, int DIM1, int DIM2) {(TYPE* inarr, int nsections, int npts)};
%apply (TYPE* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(TYPE* inarr, int channels, int sections, int npts)};

%enddef    /* %apply_numpy_typemaps() macro */

%apply_numpy_typemaps(double)

//--------------------------------------------------------------------
%feature("autodoc", 0) _open;
%feature("docstring", "Opens a file and returns a recording object.
      
Arguments:
filename -- file name

Returns:
A recording object.") _open;
void _open(const char* filename);
//--------------------------------------------------------------------


//--------------------------------------------------------------------
%pythoncode {
import numpy as np

class Recording():
    def __init__(self, channels, comment, date, time):
        self.channels = channels
        self.comment = comment
        self.date = date
        self.time = time

    def __getitem__( self, i ):
        return self.channels[i]

    def get_list( self ):
        return [ [ s.data for s in c.sections ] for c in self.channels ]

    def __len__( self ):
        return len( self.channels )

class Channel():
    def __init__(self, sections, name):
        self.sections = sections
        self.name = name

    def __len__( self ):
        return len( self.sections )

    def __getitem__( self, i ):
        return self.sections[i]

class Section():
    def __init__(self, data, dt, xunits, yunits):
        self.data = data
        self.dt = dt
        self.xunits = xunits
        self.yunits = yunits

    def __len__( self ):
        return len( self.data )

    def __getitem__( self, i ):
        return self.data[i]

def read(filename, stftype=None):
    """
    Reads a file into a Recording object.
    """

    # read data from channels: 
    channel_list = list()
    for n_c in range(n_channels):

        # required number of zeros:
        if n_sections==1:
            max_log10 = 0
        else:
            max_log10 = int(N.log10(n_sections-1))

        # loop through sections:
        section_list = list()
        for n_s in range(n_sections):
            dt = secdesc_node.col("dt")[0]
            xunits = secdesc_node.col("xunits")[0]
            yunits = secdesc_node.col("yunits")[0]
            data = h5file.getNode( section_node, "data").read()
            section_list.append( Section(data, dt, xunits, yunits) )

        channel_list.append( Channel(section_list, channel_names[n_c]) )

    return Recording( channel_list, comment, date, time )

}
//--------------------------------------------------------------------
