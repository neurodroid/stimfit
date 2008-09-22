# This file was created automatically by SWIG 1.3.29.
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.

"""
The stf module allows to access a running stimfit
application from the embedded python shell.
"""

import _stf
import new
new_instancemethod = new.instancemethod
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'PySwigObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0
del types



def _get_trace_fixedsize(*args):
  """
    _get_trace_fixedsize(outvec, trace, channel)

    Returns a trace as a 1-dimensional NumPy array.
    This returns an array of a given size.
    Don't use this, use get_trace instead.
          
    Arguments:
    size --    Size of the array to be filled.       
    trace --   ZERO-BASED index of the trace within the channel.
               Note that this is one less than what is shown
               in the drop-down box.
               The default value of -1 returns the currently
               displayed trace.
    channel -- ZERO-BASED index of the channel. This is independent
               of whether a channel is active or not.
               The default value of -1 returns the currently
               active channel.
    Returns:
    The trace as a 1D NumPy array.
    """
  return _stf._get_trace_fixedsize(*args)

def new_window(*args):
  """
    new_window(invec)

    Creates a new window showing a
    1D NumPy array.
          
    Arguments:
    invec --   The NumPy array to be shown.
    """
  return _stf.new_window(*args)

def _new_window_gVector(*args):
  """
    _new_window_gVector()

    Creates a new window from the global vector.
    Do not use directly.
    """
  return _stf._new_window_gVector(*args)

def new_window_matrix(*args):
  """
    new_window_matrix(inarr)

    Creates a new window showing a
    2D NumPy array.
          
    Arguments:
    inarr --   The NumPy array to be shown. First dimension
               are the traces, second dimension the sampling
               points within the traces.
    """
  return _stf.new_window_matrix(*args)

def new_window_selected_this(*args):
  """
    new_window_selected_this() -> bool

    Creates a new window showing the
    selected traces of the current file.
    Returns:
    True if successful.
    """
  return _stf.new_window_selected_this(*args)

def new_window_selected_all(*args):
  """
    new_window_selected_all() -> bool

    Creates a new window showing the
    selected traces of all open files.
    Returns:
    True if successful.
    """
  return _stf.new_window_selected_all(*args)

def get_sampling_interval(*args):
  """
    get_sampling_interval() -> double

    Returns the sampling interval.

    Returns:
    The sampling interval.
    """
  return _stf.get_sampling_interval(*args)

def set_sampling_interval(*args):
  """
    set_sampling_interval(si) -> bool

    Sets a new sampling interval.

    Argument:
    si --     The new sampling interval.

    Returns:
    False upon failure.
    """
  return _stf.set_sampling_interval(*args)

def select_all(*args):
  """
    select_all()

    Selects all traces in the current file. Stores 
    the baseline along with the trace index.
    """
  return _stf.select_all(*args)

def unselect_all(*args):
  """
    unselect_all()

    Unselects all previously selected traces in the
    current file.
    """
  return _stf.unselect_all(*args)

def subtract_base(*args):
  """
    subtract_base() -> bool

    Subtracts the baseline from the selected traces
    of the current file, then displays the subtracted
    traces in a new window.

    Returns:
    True if the subtraction was successful, False otherwise.
    """
  return _stf.subtract_base(*args)

def leastsq_param_size(*args):
  """
    leastsq_param_size(fselect) -> int

    Retrieves the number of parameters for a
    function.

    Arguments:
    fselect -- Zero-based index of the function as it appears in the fit
               selection dialog.

    Returns:
    The number of parameters for the function with index fselect, or a 
    negative value upon failure.
    """
  return _stf.leastsq_param_size(*args)

def check_doc(*args):
  """
    check_doc() -> bool

    Checks whether a file is open.

    Returns:
    True if a file is open, False otherwise.
    """
  return _stf.check_doc(*args)

def _gVector_resize(*args):
  """
    _gVector_resize(size)

    Resizes the global vector. Do not use directly.
       
    Arguments:
    size -- New size of the global vector.
    """
  return _stf._gVector_resize(*args)

def _gVector_at(*args):
  """
    _gVector_at(invec, at)

    Sets the valarray at the specified position of
    the global vector. Do not use directly.
    Arguments:
    invec -- The NumPy array to be used.
    at --    The position within the global vector.
    """
  return _stf._gVector_at(*args)

def file_open(*args):
  """
    file_open(filename) -> bool

    Opens a file.
       
    Arguments:
    filename -- The file to be opened. On Windows, use double back-
                slashes ("\\\\") between directories to avoid con-
                version to special characters such as "\\t" or "\\n".
                Example usage in Windows:
                file_open("C:\\\data\\\datafile.dat")
                Example usage in Linux:
                file_open("/home/cs/data/datafile.dat")
                This is surprisingly slow when called from python. 
                Haven't figured out the reason yet.

    Returns:
    True if the file could be opened, False otherwise.
    """
  return _stf.file_open(*args)

def file_save(*args):
  """
    file_save(filename) -> bool

    Saves a file.
       
    Arguments:
    filename -- The file to be saved. On Windows, use double back-
                slashes ("\\\\") between directories to avoid con-
                version to special characters such as "\\t" or "\\n".
                Example usage in Windows:
                file_save("C:\\\data\\\datafile.dat")
                Example usage in Linux:
                file_save("/home/cs/data/datafile.dat")
                This is surprisingly slow when called from python. 
                Haven't figured out the reason yet.

    Returns:
    True if the file could be saved, False otherwise.
    """
  return _stf.file_save(*args)

def close_all(*args):
  """
    close_all() -> bool

    Closes all open files.
       
    Returns:
    True if all files could be closed.
    """
  return _stf.close_all(*args)

def close_this(*args):
  """
    close_this() -> bool

    Closes the currently active file.
       
    Returns:
    True if the file could be closed.
    """
  return _stf.close_this(*args)

def get_base(*args):
  """
    get_base() -> double

    Returns the current baseline value. Uses the 
    currently measured values, i.e. does not update measurements if the 
    peak or base window cursors have changed.
             
    Returns:
    The current baseline.
    """
  return _stf.get_base(*args)

def get_peak(*args):
  """
    get_peak() -> double

    Returns the current peak value, measured from
    zero (!). Uses the currently measured values, i.e. does not update 
    measurements if the peak or base window cursors have changed.
             
    Returns:
    The current peak value, measured from zero (again: !).
    """
  return _stf.get_peak(*args)
peak_index_cb = _stf.peak_index_cb
maxrise_index_cb = _stf.maxrise_index_cb
foot_index_cb = _stf.foot_index_cb
t50left_index_cb = _stf.t50left_index_cb
t50right_index_cb = _stf.t50right_index_cb
set_peak_mean = _stf.set_peak_mean
set_peak_direction = _stf.set_peak_direction
measure = _stf.measure
get_selected_indices = _stf.get_selected_indices
set_trace = _stf.set_trace
get_trace_index = _stf.get_trace_index
set_marker = _stf.set_marker
erase_markers = _stf.erase_markers
import numpy as N

def get_trace(trace = -1, channel = -1):
    """Returns a trace as a 1-dimensional NumPy array.
      
    Arguments:       
    trace --   ZERO-BASED index of the trace within the channel.
               Note that this is one less than what is shown
               in the drop-down box.
               The default value of -1 returns the currently
               displayed trace.
    channel -- ZERO-BASED index of the channel. This is independent
               of whether a channel is active or not.
               The default value of -1 returns the currently
               active channel.
    Returns:
    The trace as a 1D NumPy array.
    """
    return _get_trace_fixedsize(get_size_trace(trace, channel), trace, channel)
    
def new_window_list( array_list ):
    """Creates a new window showing a list of
    1D NumPy arrays. As opposed to new_window_matrix(), this
    has the advantage that the arrays need not have equal sizes.
      
    Arguments:       
    array_list -- A tuple of numpy arrays.
    """

    _gVector_resize( len(array_list) )
    n = 0
    for a in array_list:
        _gVector_at( a, n )
        n = n+1

    _new_window_gVector( )

def cut_traces( pt ):
    """Cuts the selected traces at the sampling point pt,
    and shows the cut traces in a new window.
    Returns True upon success, False upon failure."""
    
    if not get_selected_indices():
        return False
    new_list = list()
    for n in get_selected_indices():
        if not set_trace(n): return False
        
        if pt < get_size_trace():
            new_list.append( get_trace()[:pt] )
            new_list.append( get_trace()[pt:] )
        else:
            print "Cutting point", pt, "is out of range"

    if len(new_list) > 0: new_window_list( new_list )
    
    return True
    
def cut_traces_multi( pt_list ):
    """Cuts the selected traces at the sampling points
    in pt_list and shows the cut traces in a new window.
    Returns True upon success, False upon failure."""
    if not get_selected_indices():
        return False
    new_list = list()
    for n in get_selected_indices():
        if not set_trace(n): return False
        old_pt = 0
        for pt in pt_list:
            if pt < get_size_trace():
                new_list.append( get_trace()[old_pt:pt] )
                old_pt = pt
            else:
                print "Cutting point", pt, "is out of range"
        if len(new_list) > 0: new_list.append( get_trace()[old_pt:] )
    new_window_list( new_list )
    return True



def show_table(*args):
  """
    show_table(dict, caption="Python table") -> bool
    show_table(dict) -> bool

    Shows a python dictionary in a results table.
    The dictionary has to have the form "string" : float

    Arguments:
    dict --    A dictionary with strings as key values and floating point
               numbers as values.
    caption -- An optional caption for the table.

    Returns:
    True if successful.
    """
  return _stf.show_table(*args)

def show_table_dictlist(*args):
  """
    show_table_dictlist(dict, caption="Python table", reverse=True) -> bool
    show_table_dictlist(dict, caption="Python table") -> bool
    show_table_dictlist(dict) -> bool

    Shows a python dictionary in a results table.
    The dictionary has to have the form "string" : list. 

    Arguments:
    dict --    A dictionary with strings as key values and lists of 
               floating point numbers as values.
    caption -- An optional caption for the table.
    reverse -- If True, The table will be filled in column-major order,
               i.e. dictionary keys will become column titles. Setting
               it to False has not been implemented yet.

    Returns:
    True if successful.
    """
  return _stf.show_table_dictlist(*args)

def get_size_trace(*args):
  """
    get_size_trace(trace=-1, channel=-1) -> int
    get_size_trace(trace=-1) -> int
    get_size_trace() -> int

    Retrieves the number of sample points of a trace.
       
    Arguments:
    trace --   ZERO-BASED index of the trace. Default value of -1
               will use the currently displayed trace. Note that
               this is one less than what is displayed in the drop-
               down list.
    channel -- ZERO-BASED index of the channel. Default value of
               -1 will use the current channel.
    Returns:
    The number of sample points.
    """
  return _stf.get_size_trace(*args)

def get_size_channel(*args):
  """
    get_size_channel(channel=-1) -> int
    get_size_channel() -> int

    Retrieves the number of traces in a channel.
    Note that at present, stimfit only supports equal-sized channels, i.e. 
    all channels within a file need to have the same number of traces. The
    channel argument is only for future extensions. 
       
    Arguments:
    channel -- ZERO-BASED index of the channel. Default value of
               -1 will use the current channel. 
    Returns:
    The number traces in a channel.
    """
  return _stf.get_size_channel(*args)

def select_trace(*args):
  """
    select_trace(trace=-1) -> bool
    select_trace() -> bool

    Selects a trace. Checks for out-of-range
    indices and stores the baseline along with the trace index.
       
    Arguments:
    trace --   ZERO-BASED index of the trace. Default value of -1
               will select the currently displayed trace. Note that
               this is one less than what is displayed in the drop-
               down list.
    Returns:
    True if the trace could be selected, False otherwise.
    """
  return _stf.select_trace(*args)

def leastsq(*args):
  """
    leastsq(fselect, refresh=True) -> PyObject
    leastsq(fselect) -> PyObject

    Fits a function to the data between the current
    fit cursors.

    Arguments:
    fselect -- Zero-based index of the function as it appears in the fit
               selection dialog.
    refresh -- To avoid flicker during batch analysis, this may be set to
               False so that the fitted function will not immediately
               be drawn.

    Returns:
    A dictionary with the best-fit parameters and the least-squared
    error, or a null pointer upon failure.
    """
  return _stf.leastsq(*args)
peak_index = _stf.peak_index
maxrise_index = _stf.maxrise_index
foot_index = _stf.foot_index
t50left_index = _stf.t50left_index
t50right_index = _stf.t50right_index
get_fit_start = _stf.get_fit_start
get_fit_end = _stf.get_fit_end
set_fit_start = _stf.set_fit_start
set_fit_end = _stf.set_fit_end
get_peak_start = _stf.get_peak_start
get_peak_end = _stf.get_peak_end
set_peak_start = _stf.set_peak_start
set_peak_end = _stf.set_peak_end
get_base_start = _stf.get_base_start
get_base_end = _stf.get_base_end
set_base_start = _stf.set_base_start
set_base_end = _stf.set_base_end
get_channel_index = _stf.get_channel_index
align_selected = _stf.align_selected

