#if defined(__WXMAC__) || defined(__WXGTK__)
  #pragma GCC diagnostic ignored "-Wwrite-strings"
#endif

%define DOCSTRING
"The stf module allows to access a running stimfit
application from the embedded python shell."
%enddef

%module(docstring=DOCSTRING) stf

%{
#define SWIG_HAVE_SNPRINTF
#define SWIG_FILE_WITH_INIT
#include "pystf.h"
%}
%include "numpy.i"
%include "std_string.i"
%include "std_vector.i"
namespace std {
    %template(vectord) vector<double>;
};

%init %{
    import_array();
    
%}

%define %apply_numpy_typemaps(TYPE)

%apply (TYPE* IN_ARRAY1, int DIM1) {(TYPE* invec, int size)};
%apply (TYPE* IN_ARRAY2, int DIM1, int DIM2) {(TYPE* inarr, int traces, int size)};

%enddef    /* %apply_numpy_typemaps() macro */

%apply_numpy_typemaps(double)

//--------------------------------------------------------------------
%feature("autodoc", 0) get_versionstring;
%feature("docstring",
"Returns the current version of Stimfit.") get_versionstring;
std::string get_versionstring( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_trace;
%feature("kwargs") get_trace;
%feature("docstring", """Returns a trace as a 1-dimensional NumPy array.

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
The trace as a 1D NumPy array.""") get_trace;
PyObject* get_trace(int trace=-1, int channel=-1);
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) new_window;
%feature("docstring", "Creates a new window showing a
1D NumPy array.
      
Arguments:
invec --   The NumPy array to be shown.

Returns:
True upon successful completion, false otherwise.") new_window;
bool new_window( double* invec, int size );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) _new_window_gMatrix;
%feature("docstring", "Creates a new window from the global matrix.
Do not use directly.") _new_window_gMatrix;
bool _new_window_gMatrix( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) new_window_matrix;
%feature("docstring", "Creates a new window showing a
2D NumPy array.
      
Arguments:
inarr --   The NumPy array to be shown. First dimension
           are the traces, second dimension the sampling
           points within the traces.

Returns:
True upon successful completion, false otherwise.") new_window_matrix;
bool new_window_matrix( double* inarr, int traces, int size );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) new_window_selected_this;
%feature("docstring", "Creates a new window showing the
selected traces of the current file.
Returns:
True if successful.") new_window_selected_this;
bool new_window_selected_this( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) new_window_selected_all;
%feature("docstring", "Creates a new window showing the
selected traces of all open files.
Returns:
True if successful.") new_window_selected_all;
bool new_window_selected_all( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) show_table;
%feature("kwargs") show_table;
%feature("docstring", "Shows a python dictionary in a results table.
The dictionary has to have the form \"string\" : float

Arguments:
dict --    A dictionary with strings as key values and floating point
           numbers as values.
caption -- An optional caption for the table.

Returns:
True if successful.") show_table;
bool show_table( PyObject* dict, const char* caption = "Python table" );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) show_table_dictlist;
%feature("kwargs") show_table_dictlist;
%feature("docstring", "Shows a python dictionary in a results table.
The dictionary has to have the form \"string\" : list. 

Arguments:
dict --    A dictionary with strings as key values and lists of 
           floating point numbers as values.
caption -- An optional caption for the table.
reverse -- If True, The table will be filled in column-major order,
           i.e. dictionary keys will become column titles. Setting
           it to False has not been implemented yet.

Returns:
True if successful.") show_table_dictlist;
bool show_table_dictlist( PyObject* dict, const char* caption = "Python table", bool reverse = true );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_size_trace;
%feature("kwargs") get_size_trace;
%feature("docstring", "Retrieves the number of sample points of a trace.
   
Arguments:
trace --   ZERO-BASED index of the trace. Default value of -1
           will use the currently displayed trace. Note that
           this is one less than what is displayed in the drop-
           down list.
channel -- ZERO-BASED index of the channel. Default value of
           -1 will use the current channel.
Returns:
The number of sample points.") get_size_trace;
int get_size_trace( int trace = -1, int channel = -1 );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_size_channel;
%feature("kwargs") get_size_channel;
%feature("docstring", "Retrieves the number of traces in a channel.
Note that at present, stimfit only supports equal-sized channels, i.e. 
all channels within a file need to have the same number of traces. The
channel argument is only for future extensions. 
   
Arguments:
channel -- ZERO-BASED index of the channel. Default value of
           -1 will use the current channel. 
Returns:
The number traces in a channel.") get_size_channel;
int get_size_channel( int channel = -1 );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_maxdecay;
%feature("docstring", "Retrieves the maximal slope of the decay 
between the peak cursors in the current trace. Uses the currently 
measured values, i.e. does not update measurements if the peak window 
cursors have changed.
  
   
Returns:
The maximal slope of the decay, -1.0 upon failure.") get_maxdecay;
double get_maxdecay( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_maxrise;
%feature("docstring", "Retrieves the maximal slope of the rise 
between the peak cursors in the current trace. Uses the currently 
measured values, i.e. does not update measurements if the peak window 
cursors have changed.

   
Returns:
The maximal slope of the rise, -1.0 upon failure.") get_maxrise;
double get_maxrise( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_size_recording;
%feature("docstring", "Retrieves the number of channels in a 
recording.
   
Returns:
The number of channels in a recording.") get_size_recording;
int get_size_recording( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_sampling_interval;
%feature("docstring", "Returns the sampling interval.

Returns:
The sampling interval.") get_sampling_interval;
double get_sampling_interval( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_xunits;
%feature("docstring", "Returns the x units of the specified section.
X units are assumed to be the same for the entire file.

Returns:
The x units as a string.") get_xunits;
const char* get_xunits( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_yunits;
%feature("kwargs") get_yunits;
%feature("docstring", "Returns the y units of the specified trace.
Y units are not allowed to change between traces at present.

Arguments:
trace -- The zero-based index of the trace of interest. If < 0, the
      	   name of the active trace will be returned.
channel -- The zero-based index of the channel of interest. If < 0, the
      	   active channel will be used.

Returns:
The x units as a string.") get_yunits;
const char* get_yunits( int trace = -1, int channel = -1 );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_yunits;
%feature("kwargs") set_yunits;
%feature("docstring", "Sets the y unit string of the specified trace.
Y units are not allowed to change between traces at present.

Arguments:
units --   The new y unit string.
trace --   The zero-based index of the trace of interest. If < 0, the
      	   name of the active trace will be returned.
channel -- The zero-based index of the channel of interest. If < 0, the
      	   active channel will be used.

Returns:
True if successful.") set_yunits;
bool set_yunits( const char* units, int trace = -1, int channel = -1 );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_xunits;
%feature("kwargs") set_xunits;
%feature("docstring", "Sets the x unit string for the entire file.

Arguments:
units --   The new x unit string.

Returns:
True if successful.") set_xunits;
bool set_xunits( const char* units );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_sampling_interval;
%feature("docstring", "Sets a new sampling interval.

Argument:
si --     The new sampling interval.

Returns:
False upon failure.") set_sampling_interval;
bool set_sampling_interval( double si );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) select_trace;
%feature("kwargs") select_trace;
%feature("docstring", "Selects a trace. Checks for out-of-range
indices and stores the baseline along with the trace index.
   
Arguments:
trace --   ZERO-BASED index of the trace. Default value of -1
           will select the currently displayed trace. Note that
           this is one less than what is displayed in the drop-
           down list.
Returns:
True if the trace could be selected, False otherwise.") select_trace;
bool select_trace( int trace = -1 );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) select_all;
%feature("docstring", "Selects all traces in the current file. Stores 
the baseline along with the trace index.") select_all;
void select_all( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) unselect_all;
%feature("docstring", "Unselects all previously selected traces in the
current file.") unselect_all;
void unselect_all( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) subtract_base;
%feature("docstring", "Subtracts the baseline from the selected traces
of the current file, then displays the subtracted
traces in a new window.

Returns:
True if the subtraction was successful, False otherwise.") subtract_base;
bool subtract_base( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) subtract_base;
%feature("docstring", "Subtracts the baseline from the selected traces
of the current file, then displays the subtracted
traces in a new window.

Returns:
True if the subtraction was successful, False otherwise.") subtract_base;
bool subtract_base( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) leastsq;
%feature("kwargs") leastsq;
%feature("docstring", "Fits a function to the data between the current
fit cursors.

Arguments:
fselect -- Zero-based index of the function as it appears in the fit
           selection dialog.
refresh -- To avoid flicker during batch analysis, this may be set to
           False so that the fitted function will not immediately
           be drawn.

Returns:
A dictionary with the best-fit parameters and the least-squared
error, or a null pointer upon failure.") leastsq;
PyObject* leastsq( int fselect, bool refresh = true );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_fit;
%feature("kwargs") get_fit;
%feature("docstring", "Get the waveform resulted from the fitting, if available.

Arguments:
trace -- The zero-based index of the trace of interest. If < 0, the
      	   name of the active trace will be returned.
channel -- The zero-based index of the channel of interest. If < 0, the
      	   active channel will be used.

Returns:
A 2D Numpy array with the x-values of the fit in the first dimension, and the y-values in the second dimension. None if no fit is available.") get_fit;
PyObject* get_fit( int trace = -1, int channel = -1 );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) leastsq_param_size;
%feature("docstring", "Retrieves the number of parameters for a
function.

Arguments:
fselect -- Zero-based index of the function as it appears in the fit
           selection dialog.

Returns:
The number of parameters for the function with index fselect, or a 
negative value upon failure.") leastsq_param_size;
int leastsq_param_size( int fselect );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) check_doc;
%feature("docstring", "Checks whether a file is open.

Arguments:
show_dialog -- True if an error dialog should be shown if no file is
               open.

Returns:
True if a file is open, False otherwise.") check_doc;
bool check_doc( bool show_dialog=true );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_filename;
%feature("docstring",
"Returns the name of the current file.") get_filename;
std::string get_filename( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) _gMatrix_resize;
%feature("docstring", "Resizes the global matrix. Do not use directly.
   
Arguments:
channels -- New number of channels of the global matrix.
sections -- New number of sections of the global matrix.

") _gMatrix_resize;
void _gMatrix_resize( std::size_t channels, std::size_t sections );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) _gMatrix_at;
%feature("docstring", "Sets the valarray at the specified position of
the global matrix. Do not use directly.
Arguments:
invec --   The NumPy array to be used.
channel -- The channel index within the global matrix.
section -- The seciton index within the global matrix.
") _gMatrix_at;
void _gMatrix_at( double* invec, int size, int channel, int section );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) _gNames_resize;
%feature("docstring", "Resizes the global names. Do not use directly.
   
Arguments:
channels -- New number of channels of the global names.

") _gNames_resize;
void _gNames_resize( std::size_t channels );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) _gNames_at;
%feature("docstring", "Sets the channel name of the specifies channel.
Do not use directly.
Arguments:
name --   The new channel name
channel -- The channel index within the global names.
") _gNames_at;
void _gNames_at( const char* name, int channel );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) file_open;
%feature("docstring", "Opens a file.
   
Arguments:
filename -- The file to be opened. On Windows, use double back-
            slashes (\"\\\\\\\\\") between directories to avoid con-
            version to special characters such as \"\\\\t\" or \"\\\\n\".
            Example usage in Windows:
            file_open(\"C:\\\\\\data\\\\\\datafile.dat\")
            Example usage in Linux:
            file_open(\"/home/cs/data/datafile.dat\")
            This is surprisingly slow when called from python. 
            Haven't figured out the reason yet.

Returns:
True if the file could be opened, False otherwise.") file_open;
bool file_open( const char* filename );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) file_save;
%feature("docstring", "Saves a file.
   
Arguments:
filename -- The file to be saved. On Windows, use double back-
            slashes (\"\\\\\\\\\") between directories to avoid con-
            version to special characters such as \"\\\\t\" or \"\\\\n\".
            Example usage in Windows:
            file_save(\"C:\\\\\\data\\\\\\datafile.dat\")
            Example usage in Linux:
            file_save(\"/home/cs/data/datafile.dat\")
            This is surprisingly slow when called from python. 
            Haven't figured out the reason yet.

Returns:
True if the file could be saved, False otherwise.") file_save;
bool file_save( const char* filename );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_recording_time;
%feature("docstring", "Returns the time at which the recording was 
started as a string.") get_recording_time;
const char* get_recording_time( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_recording_date;
%feature("docstring", "Returns the date at which the recording was 
started as a string.") get_recording_date;
const char* get_recording_date( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_recording_comment;
%feature("docstring", "Returns a comment about the recording.
") get_recording_comment;
std::string get_recording_comment( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_recording_date;
%feature("docstring", "Sets a date about the recording.

Argument:
date -- A date string.

Returns:
True upon successful completion.") set_recording_date;
bool set_recording_date( const char* date );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_recording_time;
%feature("docstring", "Sets a time about the recording.

Argument:
time -- A time string.

Returns:
True upon successful completion.") set_recording_time;
bool set_recording_time( const char* time );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_recording_comment;
%feature("docstring", "Sets a comment about the recording.

Argument:
comment -- A comment string.

Returns:
True upon successful completion.") set_recording_comment;
bool set_recording_comment( const char* comment );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_recording_comment;
%feature("docstring", "Sets a comment about the recording.

Argument:
comment -- A comment string.

Returns:
True upon successful completion.") set_recording_comment;
bool set_recording_comment( const char* comment );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) close_all;
%feature("docstring", "Closes all open files.
   
Returns:
True if all files could be closed.") close_all;
bool close_all( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) close_this;
%feature("docstring", "Closes the currently active file.
   
Returns:
True if the file could be closed.") close_this;
bool close_this( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_base;
%feature("kwargs") get_base;
%feature("docstring", "Returns the current baseline value. Uses the 
currently measured values, i.e. does not update measurements if the 
peak or base window cursors have changed.

Arguments:
active -- If True, returns the baseline in the active channel. If False
          returns the baseline within the reference channel.

Returns:
The current baseline.") get_base;
double get_base( bool active = true);
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_peak;
%feature("docstring", "Returns the current peak value, measured from
zero (!). Uses the currently measured values, i.e. does not update 
measurements if the peak or base window cursors have changed.
         
Returns:
The current peak value, measured from zero (again: !).") get_peak;
double get_peak( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
#ifdef WITH_PSLOPE
%feature("autodoc", 0) get_pslope;
%feature("docstring", "Returns the slope, measured from
slope cursor values defined in the cursors settings menu(!).
This option is only available under GNU/Linux. 
         
Returns:
The current slope value.") get_pslope;
double get_pslope( );
#endif
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) peak_index;
%feature("kwargs") peak_index;
%feature("docstring", "Returns the zero-based index of the current
peak position in the specified channel. Uses the currently measured
values, i.e. does not update measurements if the peak window cursors
have changed.
   
Arguments:
active -- If True, returns the current peak index of the active channel.
          Otherwise, returns the current peak index of the reference channel.
          
Returns:
The zero-based index in units of sampling points. May be interpolated
if more than one point is used for the peak calculation. Returns a 
negative value upon failure.") peak_index;
%callback("%s_cb");
double peak_index( bool active = true );
%nocallback;
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) maxrise_index;
%feature("kwargs") maxrise_index;
%feature("docstring", "Returns the zero-based index of the maximal
slope of rise in the specified channel. Uses the currently measured
values, i.e. does not update measurements if the peak window cursors
have changed.
   
Arguments:
active -- If True, returns the current index of the maximal slope of 
          rise within the active channel. Otherwise, returns the 
          current index of the maximal slope of rise within the 
          reference channel.
          
Returns:
The zero-based index of the maximal slope of  rise in units of 
sampling points interpolated between adjacent sampling points.
Returns a negative value upon failure.") maxrise_index;
%callback("%s_cb");
double maxrise_index( bool active = true );
%nocallback;
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) maxdecay_index;
%feature("docstring", "Returns the zero-based index of the maximal
slope of decay in the current channel. Uses the currently measured
values, i.e. does not update measurements if the peak window cursors
have changed. Note that in contrast to maxrise_index, this function 
only works on the active channel.
   
Returns:
The zero-based index of the maximal slope of decay in units of 
sampling points interpolated between adjacent sampling points.
Returns a negative value upon failure.") maxdecay_index;
double maxdecay_index( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) foot_index;
%feature("kwargs") foot_index;
%feature("docstring", "Returns the zero-based index of the foot of 
an event in the active channel. The foot is the intersection of an
interpolated line through the points of 20 and 80% rise with the
baseline. Uses the currently measured values, i.e. does not update 
measurements if the peak or base window cursors have changed.
   
Arguments:
active -- If True, returns the current index of the foot within the 
          active channel. Only implemented for the active channel
          at this time. Will return a negative value and show an 
          error message if active == False.
          
Returns:
The zero-based index of the foot of an event in units of sampling 
points. Interpolates between sampling points.
Returns a negative value upon failure.") foot_index;
%callback("%s_cb");
double foot_index( bool active = true );
%nocallback;
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) t50left_index;
%feature("kwargs") t50left_index;
%feature("docstring", "Returns the zero-based index of the left half-
maximal amplitude of an event in the specified channel. Uses the 
currently measured values, i.e. does not update measurements if the 
peak or base window cursors have changed.
   
Arguments:
active -- If True, returns the current index of the left half-
          maximal amplitude within the active channel. If False, 
          returns the current index of the left half-maximal amplitude
          within the reference channel.
          
Returns:
The zero-based index of the left half-maximal amplitude in units of 
sampling points. Interpolates between sampling points. Returns a 
negative value upon failure.") t50left_index;
%callback("%s_cb");
double t50left_index( bool active = true );
%nocallback;
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) t50right_index;
%feature("kwargs") t50right_index;
%feature("docstring", "Returns the zero-based index of the right half-
maximal amplitude of an event in the active channel. Uses the 
currently measured values, i.e. does not update measurements if the 
peak or base window cursors have changed.
   
Arguments:
active -- If True, returns the current index of the right half-
          maximal amplitude within the active channel. Only 
          implemented for the active channel at this time. Will return 
          a negative value and show an error message if 
          active == False.
          
Returns:
The zero-based index of the right half-maximal amplitude in units of 
sampling points. Interpolates between sampling points. Returns a 
negative value upon failure.") t50right_index;
%callback("%s_cb");
double t50right_index( bool active = true );
%nocallback;
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_halfwidth;
%feature("kwargs") get_halfwidth;
%feature("docstring", "Returns the half-maximal amplitude of an event 
in the specified channel. Uses the currently measured values, i.e. 
does not update measurements if the peak or base window cursors have 
changed. Only implemented for the active channel.
   
Arguments:
active -- If True, returns the current half-maximal amplitude within the 
          active channel. 
          
Returns:
The half-maximal amplitude in units of x-units. Returns a 
negative value upon failure.") get_halfwidth;
%callback("%s_cb");
double get_halfwidth( bool active = true );
%nocallback;
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) rtlow_index;
%feature("kwargs") rtlow_index;
%feature("docstring", "Returns the zero-based index of the lower rise-
time of an event in the specified channel. Uses the currently measured
values, i.e. does not update measurements if the peak or base window
cursors have changed.
   
Arguments:
active -- If True, returns the current index within the active
          channel. If False, returns the current index within the
          reference channel.
          
Returns:
The zero-based index of the lower rise time in units of sampling
points. Interpolates between sampling points. Returns a negative value
upon failure.") rtlow_index;
%callback("%s_cb");
double rtlow_index( bool active = true );
%nocallback;
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) rthigh_index;
%feature("kwargs") rthigh_index;
%feature("docstring", "Returns the zero-based index of the higher
rise time of an event in the specified channel. Uses the currently
measured values, i.e. does not update measurements if the peak or base
window cursors have changed.
   
Arguments:
active -- If True, returns the current index within the active
          channel. If False, returns the current index within the
          reference channel.
          
Returns:
The zero-based index of the higher rise time in units of sampling
points. Interpolates between sampling points. Returns a negative value
upon failure.") rthigh_index;
%callback("%s_cb");
double rthigh_index( bool active = true );
%nocallback;
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_threshold_time;
%feature("kwargs") get_threshold_time;
%feature("docstring", "Returns the crossing value of the threshold 
slope. Note that this value is not updated after changing the AP 
threshold. Call measure() or hit enter to update the cursors.

Arguments:
is_time -- If False (default), returns the zero-based index at which 
           the threshold slope is crossed (e.g in mV). If True,
           returns the time point at which the threshold slope is 
           crossed. A negative number is returned upon failure. 
") get_threshold_time;
double get_threshold_time( bool is_time = false );
//--------------------------------------------------------------------


//--------------------------------------------------------------------
%feature("autodoc", 0) get_latency;
%feature("docstring", "Returns the latency value (in x-units)
determined by the latency cursors set in the cursors settings menu. 
Call measure() or hit enter to update the cursors.

") get_latency;
double get_latency( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_risetime;
%feature("docstring", "Returns the rise time (in x-units)
determined by the linear interpolation between sampling points
(e.g at 20% and 80% of the peak amplitude. )
Call measure() or hit enter to update the cursors.

") get_risetime;
double get_risetime( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_risetime_factor;
%feature("docstring", "Returns the lower proportion factor used
to calculate the rise time 
(e.g 0.2 if we calculate the 20--80% rise time. )

") get_risetime_factor;
double get_risetime_factor( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_risetime_factor;
%feature("kwargs") set_risetime_factor;
%feature("docstring", "Sets the lower proportion factor
to calculate the rise time
(e.g 0.2 if we want to calculate the 20--80% rise time. )
It will update the rise time measurement.

Arguments:
factor --  the low proportion factor to calculate rise time

Returns:
False upon failure (such as factor lower than 0.05 or larger than 0.5)
") set_risetime_factor;
bool set_risetime_factor(double factor);
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_threshold_value;
%feature("docstring", "Returns the value found at the threshold 
slope. Note that this value is not updated after changing the AP 
threshold. Call measure or hit enter to update the threshold.
") get_threshold_value;
double get_threshold_value( );
//--------------------------------------------------------------------


//--------------------------------------------------------------------
%feature("autodoc", 0) get_fit_start;
%feature("kwargs") get_fit_start;
%feature("docstring", "Returns the zero-based index or the time point
of the fit start cursor.

Arguments:
is_time -- If False (default), returns the zero-based index. If True,
           returns the time from the beginning of the trace to the
           cursor position.          
") get_fit_start;
double get_fit_start( bool is_time = false );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_fit_end;
%feature("kwargs") get_fit_end;
%feature("docstring", "Returns the zero-based index or the time point
of the fit end cursor.

Arguments:
is_time -- If False (default), returns the zero-based index. If True,
           returns the time from the beginning of the trace to the
           cursor position.          
") get_fit_end;
double get_fit_end( bool is_time = false );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_fit_start;
%feature("kwargs") set_fit_start;
%feature("docstring", "Sets the fit start cursor to a new position.

Arguments:
pos --     The new cursor position, either in units of sampling points
           if is_time == False (default) or in units of time if
           is_time == True.
is_time -- see above.

Returns:
False upon failure (such as out-of-range).") set_fit_start;
bool set_fit_start( double pos, bool is_time = false );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_fit_end;
%feature("kwargs") set_fit_end;
%feature("docstring", "Sets the fit end cursor to a new position.

Arguments:
pos --     The new cursor position, either in units of sampling points
           if is_time == False (default) or in units of time if
           is_time == True.
is_time -- see above.

Returns:
False upon failure (such as out-of-range).") set_fit_end;
bool set_fit_end( double pos, bool is_time = false );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_peak_start;
%feature("kwargs") get_peak_start;
%feature("docstring", "Returns the zero-based index or the time point
of the peak start cursor.

Arguments:
is_time -- If False (default), returns the zero-based index. If True,
           returns the time from the beginning of the trace to the
           cursor position.          
") get_peak_start;
double get_peak_start( bool is_time = false );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_peak_end;
%feature("kwargs") get_peak_end;
%feature("docstring", "Returns the zero-based index or the time point
of the peak end cursor.

Arguments:
is_time -- If False (default), returns the zero-based index. If True,
           returns the time from the beginning of the trace to the
           cursor position.          
") get_peak_end;
double get_peak_end( bool is_time = false );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_peak_start;
%feature("kwargs") set_peak_start;
%feature("docstring", "Sets the peak start cursor to a new position.
This will NOT update the peak calculation. You have to either call 
measure() or hit enter in the main window to achieve that.

Arguments:
pos --     The new cursor position, either in units of sampling points
           if is_time == False (default) or in units of time if
           is_time == True.
is_time -- see above.

Returns:
False upon failure (such as out-of-range).") set_peak_start;
bool set_peak_start( double pos, bool is_time = false );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_peak_end;
%feature("kwargs") set_peak_end;
%feature("docstring", "Sets the peak end cursor to a new position.
This will NOT update the peak calculation. You have to either call 
measure() or hit enter in the main window to achieve that.

Arguments:
pos --     The new cursor position, either in units of sampling points
           if is_time == False (default) or in units of time if
           is_time == True.
is_time -- see above.

Returns:
False upon failure (such as out-of-range).") set_peak_end;
bool set_peak_end( double pos, bool is_time = false );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_peak_mean;
%feature("docstring", "Sets the number of points used for the peak 
calculation.

Arguments:
pts -- A moving average (aka sliding, boxcar or running average) is 
       used to determine the peak value. Pts specifies the number of
       sampling points used for the moving window.
       Passing a value of -1 will calculate the average of all
       sampling points within the peak window.

Returns:
False upon failure (such as out-of-range).") set_peak_mean;
bool set_peak_mean( int pts );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_peak_mean;
%feature("docstring", "Returns the number of sampling points used for
the peak calculation.

Returns:
0 upon failure (i.e no file opened). -1 means average of all sampling 
points within the peak window.") get_peak_mean;
int get_peak_mean( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_peak_direction;
%feature("docstring", "Sets the direction of the peak detection.

Arguments:
direction -- A string specifying the peak direction. Can be one of:
             \"up\", \"down\" or \"both\"

Returns:
False upon failure.") set_peak_direction;
bool set_peak_direction( const char* direction );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_peak_direction;
%feature("docstring", "Gets the current direction of the detection
for the peak cursors.

Returns:
A string specifying the peak direction. Can be one of:
\"up\", \"down\" or \"both\"") get_peak_direction;
const char* get_peak_direction( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_latency_start;
%feature("kwargs") set_latency_start;
%feature("docstring", "Sets the first latency cursor to a new position.

Arguments:
pos --     The new cursor position, either in units of sampling points
           if is_time == False (default) or in units of time if
           is_time == True.
is_time -- see above.

Returns:
False upon failure (such as out-of-range).") set_latency_start;
bool set_latency_start( double pos, bool is_time = false );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_latency_start_mode;
%feature("docstring", "Sets the mode of the latency start cursor.

Arguments:
direction -- A string specifying the mode for the latency start cursor. Can be one of \"manual\", \"peak\", \"rise\" or \"half\"

Returns:
False upon failure.") set_latency_start_mode;
bool set_latency_start_mode( const char* mode );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_baseline_method;
%feature("docstring", "Gets the method used to compute the baseline

Returns:
A string specifying the method to compute the baseline. Can be one of
\"mean\" or \"median\"") get_baseline_method;
const char* get_baseline_method( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_baseline_method;
%feature("docstring", "Sets the method to compute the baseline.

Arguments:
method -- A string specifying the method to calculate the baseline.
            Can be one of \"mean\" or \"median\"

Returns:
False upon failure.") set_baseline_method;
bool set_baseline_method( const char* method );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_latency_start_mode;
%feature("docstring", "Gets the latency start mode

Returns:
A string specifying the latency start mode. Can be one of
\"manual\", \"peak\", \"rise\" or \"half\"") get_latency_start_mode;
const char* get_latency_start_mode( );
//--------------------------------------------------------------------


//--------------------------------------------------------------------
%feature("autodoc", 0) set_latency_end;
%feature("kwargs") set_latency_end;
%feature("docstring", "Sets the second latency cursor to a new position.

Arguments:
pos --     The new cursor position, either in units of sampling points
           if is_time == False (default) or in units of time if
           is_time == True.
is_time -- see above.

Returns:
False upon failure (such as out-of-range).") set_latency_end;
bool set_latency_end( double pos, bool is_time = false );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_latency_end_mode;
%feature("docstring", "Sets the mode of the latency end cursor.

Arguments:
direction -- A string specifying the mode for the latency end cursor. Can be one of
             \"manual\", \"peak\", \"rise\", \"foot\" or \"half\"

Returns:
False upon failure.") set_latency_end_mode;
bool set_latency_end_mode( const char* mode );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_latency_end_mode;
%feature("docstring", "Gets the latency end mode

Returns:
A string specifying the latency end mode. Can be one of \"manual\",
\"peak\", \"rise\", \"foot\" or \"half\"") get_latency_end_mode;
const char* get_latency_end_mode( );
//--------------------------------------------------------------------


//--------------------------------------------------------------------
%feature("autodoc", 0) get_latency_start;
%feature("kwargs") get_latency_start;
%feature("docstring", "Returns the zero-based index or the time point
of the latency start cursor.

Arguments:
is_time -- If False (default), returns the zero-based index. If True,
           returns the time from the beginning of the trace to the
           cursor position.") get_latency_start;
double get_latency_start( bool is_time = false );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_latency_end;
%feature("kwargs") get_latency_end;
%feature("docstring", "Returns the zero-based index or the time point
of the latency end cursor.

Arguments:
is_time -- If False (default), returns the zero-based index. If True,
           returns the time from the beginning of the trace to the
           cursor position.") get_latency_end;
double get_latency_end( bool is_time = false );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_base_SD;
%feature("docstring", "Returns the standard deviation of the baseline
in the current (active) channel. Uses the currently measured values,
i.e. does not update measurements if the baseline cursors have changed. 

Returns:
0.0 upon failure (i.e no file opened), otherwise, the standard deviation
of the baseline.") get_base_SD;
double get_base_SD( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_base_start;
%feature("kwargs") get_base_start;
%feature("docstring", "Returns the zero-based index or the time point
of the base start cursor.

Arguments:
is_time -- If False (default), returns the zero-based index. If True,
           returns the time from the beginning of the trace to the
           cursor position.") get_base_start;
double get_base_start( bool is_time = false );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_base_end;
%feature("kwargs") get_base_end;
%feature("docstring", "Returns the zero-based index or the time point
of the base end cursor.

Arguments:
is_time -- If False (default), returns the zero-based index. If True,
           returns the time from the beginning of the trace to the
           cursor position.") get_base_end;
double get_base_end( bool is_time = false );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_base_start;
%feature("kwargs") set_base_start;
%feature("docstring", "Sets the base start cursor to a new position.
This will NOT update the baseline calculation. You have to either call 
measure() or hit enter in the main window to achieve that.

Arguments:
pos --     The new cursor position, either in units of sampling points
           if is_time == False (default) or in units of time if
           is_time == True.
is_time -- see above.

Returns:
False upon failure (such as out-of-range).") set_base_start;
bool set_base_start( double pos, bool is_time = false );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_base_end;
%feature("kwargs") set_base_end;
%feature("docstring", "Sets the base end cursor to a new position.
This will NOT update the baseline calculation. You have to either call 
measure() or hit enter in the main window to achieve that.

Arguments:
pos --     The new cursor position, either in units of sampling points
           if is_time == False (default) or in units of time if
           is_time == True.
is_time -- see above.

Returns:
False upon failure (such as out-of-range).") set_base_end;
bool set_base_end( double pos, bool is_time = false );
//--------------------------------------------------------------------


//--------------------------------------------------------------------
%feature("autodoc", 0) set_slope;
%feature("docstring", "Sets the AP threshold to the value given by the
slope and takes it as reference for AP kinetic measurements. Note that 
you have to either call measure() or hit enter to update calculations.

Arguments:
slope --  slope value in mV/ms  

Returns:
False upon failure (such as out-of-range).") set_slope;
bool set_slope( double slope);
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) measure;
%feature("docstring", "Updates all measurements (e.g. peak, baseline, 
latency) according to the current cursor settings. As if you had
pressed \"Enter\" in the main window.
Returns:
False upon failure, True otherwise.") measure;
bool measure( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_selected_indices;
%feature("docstring", "Returns a tuple with the indices (ZERO-BASED) 
of the selected traces.") get_selected_indices;
PyObject* get_selected_indices( );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_trace;
%feature("docstring", "Sets the currently displayed trace to a new
index. Subsequently updates all measurements (e.g. peak, base,
latency, i.e. you don't need to call measure() yourself.)

Arguments:
trace -- The zero-based index of the new trace to be displayed.

Returns:
True upon success, false otherwise (such as out-of-range).") set_trace;
bool set_trace( int trace );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_trace_index;
%feature("docstring", "Returns the ZERO-BASED index of the currently
displayed trace (this is one less than what is shown in the combo box).
") get_trace_index;
int get_trace_index();
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_channel_index;
%feature("kwargs") get_channel_index;
%feature("docstring", "Returns the ZERO-BASED index of the specified
channel.

Arguments:

active -- If True, returns the index of the active (black) channel.
If False, returns the index of the reference (red) channel.
") get_channel_index;
int get_channel_index( bool active = true );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_channel_name;
%feature("kwargs") get_channel_name;
%feature("docstring", "Returns the name of the channel with the 
specified index.

Arguments:

index -- The zero-based index of the channel of interest. If < 0, the
      	 name of the active channel will be returned.

Returns:
the name of the channel with the specified index.") get_channel_name;
const char* get_channel_name( int index = -1 );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_channel;
%feature("docstring", "Sets the currently displayed channel to a new
index. Subsequently updates all measurements (e.g. peak, base,
latency, i.e. you don't need to call measure() yourself.)

Arguments:
channel -- The zero-based index of the new trace to be displayed.

Returns:
True upon success, false otherwise (such as out-of-range).") set_channel;
bool set_channel( int channel);
//--------------------------------------------------------------------


//--------------------------------------------------------------------
%feature("autodoc", 0) set_channel_name;
%feature("kwargs") set_channel_name;
%feature("docstring", "Sets the name of the channel with the 
specified index.

Arguments:
name  -- The new name of the channel.
index -- The zero-based index of the channel of interest. If < 0, the
      	 active channel will be used.

Returns:
True upon success.") set_channel_name;
bool set_channel_name( const char* name, int index = -1 );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) get_trace_name;
%feature("kwargs") get_trace_name;
%feature("docstring", "Returns the name of the trace with the 
specified index.

Arguments:
trace -- The zero-based index of the trace of interest. If < 0, the
      	   name of the active trace will be returned.
channel -- The zero-based index of the channel of interest. If < 0, the
      	   active channel will be used.

Returns:
the name of the trace with the specified index.") get_trace_name;
const char* get_trace_name( int trace = -1, int channel = -1 );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) align_selected;
%feature("kwargs") align_selected;
%feature("docstring", "Aligns the selected traces to the index that is 
returned by the alignment function, and then creates a new window 
showing the aligned traces. This function requires to select the
traces of interest and the presence of a second (i.e reference) channel.
Arguments:       
alignment -- The alignment function to be used. Accepts any function
             returning a valid index within a trace. These are some
             predefined possibilities:
             maxrise_index (default; maximal slope during rising phase)
             peak_index (Peak of an event)
             foot_index (Beginning of an event)
             t50left_index 
             t50right_index (Left/right half-maximal amplitude)
active --    If True, the alignment function will be applied to
             the active channel. If False (default), it will be applied
             to the reference channel.
zeropad --   Not yet implemented:
             If True, missing parts at the beginning or end of a trace 
             will be padded with zeros after the alignment. If False
             (default), traces will be cropped so that all traces have
             equal sizes.
") align_selected;
void align_selected(  double (*alignment)( bool ), bool active = false );

//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) set_marker;
%feature("docstring", "Sets a marker to the specified position in the
current trace.

Arguments:
x -- The horizontal marker position in units of sampling points.
y -- The vertical marker position in measurement units (e.g. mV).

Returns:
False upon failure (such as out-of-range).") set_marker;
bool set_marker( double x, double y );
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) erase_markers;
%feature("docstring", "Erases all markers in the current trace.

Returns:
False upon failure.") erase_marker;
bool erase_markers();
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) plot_xmin;
%feature("docstring", "Returns x value of the left screen border")
plot_xmin;
double plot_xmin();
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) plot_xmax;
%feature("docstring", "Returns x value of the right screen border")
plot_xmax;
double plot_xmax();
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) plot_ymin;
%feature("docstring", "Returns y value of the bottom screen border")
plot_ymin;
double plot_ymin();
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) plot_ymax;
%feature("docstring", "Returns y value of the top screen border")
plot_ymax;
double plot_ymax();
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) plot_y2min;
%feature("docstring", "Returns y value of the bottom screen border for
the reference channel")
plot_y2min;
double plot_y2min();
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) plot_y2max;
%feature("docstring", "Returns y value of the top screen border for
the reference channel")
plot_y2max;
double plot_y2max();
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%feature("autodoc", 0) mpl_panel;
%feature("kwargs") mpl_panel;
%feature("docstring", "Returns a pointer to the parent window")
mpl_panel;
PyObject* mpl_panel(const std::vector<double>& figsize = std::vector<double>(_figsize, _figsize+2));
//--------------------------------------------------------------------

//--------------------------------------------------------------------
%pythoncode {

class StfException(Exception):
    """ raises Exceptions for the Stfio module """
    def __init__(self, error_msg):
        self.msg = error_msg 

    def __str__(self):
        return repr(self.msg)

def new_window_list( array_list ):
    """Creates a new window showing a sequence of
    1D NumPy arrays, or a sequence of a sequence of 1D
    NumPy arrays. As opposed to new_window_matrix(), this
    has the advantage that the arrays need not have equal sizes.
      
    Arguments:       
    array_list -- A sequence (e.g. list or tuple) of numpy arrays, or
                  a sequence of a sequence of numpy arrays.

    Returns:
    True upon successful completion, false otherwise.
    """
    try: 
        it = iter(array_list)
    except TypeError: 
        print( "Argument is not a sequence" )
        return False

    try: 
        it = iter(array_list[0])
    except TypeError: 
        print( "Argument is not a sequence of sequences." )
        print( "You can either pass a sequence of 1D NumPy arrays," )
        print( "Or a sequence of sequences of 1D NumPy arrays." )
        return False
        
    is_3d = True
    try: 
        it = iter(array_list[0][0])
    except TypeError: 
        is_3d = False
        n_channels = 1

    if is_3d:
        n_channels = len(array_list)
        _gMatrix_resize( n_channels, len(array_list[0]) )
        for (n_c, c) in enumerate(array_list):
            for (n_s, s) in enumerate(c):
                _gMatrix_at( s, n_c, n_s )
        
    else:
        _gMatrix_resize( n_channels, len(array_list) )
        for (n, a) in enumerate(array_list):
            _gMatrix_at( a, 0, n )

    return _new_window_gMatrix( )

def stfio_open(stfio_rec):
    """Open an stfio recording object with Stimfit

    Arguments:
    stfio_rec  -- An stfio recording object, e.g. opened with
                  stfio.read("filename")

    Returns:
    True upon successful completion, false otherwise.
    """
    n_channels = len(stfio_rec)
    _gMatrix_resize( n_channels, len(stfio_rec[0]) )
    for n_c, c in enumerate(stfio_rec):
        for n_s, s in enumerate(c):
            _gMatrix_at( c[n_s].asarray(), n_c, n_s )

    succ = _new_window_gMatrix( )
    if not succ:
        return False

    for n_c, c in enumerate(stfio_rec):
        set_channel_name(c.name, n_c)
        set_yunits(c.yunits, n_c)

    set_recording_date(stfio_rec.date)
    set_recording_time(stfio_rec.time)
    set_recording_comment(stfio_rec.comment)
    set_xunits(stfio_rec.xunits)
    set_sampling_interval(stfio_rec.dt)
    measure()

    return True

def cut_traces( pt ):
    """Cuts the selected traces at the sampling point pt,
    and shows the cut traces in a new window.
    Returns True upon success, False upon failure."""
    
    if not get_selected_indices():
        print( "Trace is not selected!" )
        return False
    new_list = list()
    for n in get_selected_indices():
        if not set_trace(n): return False
        
        if pt < get_size_trace():
            new_list.append( get_trace()[:pt] )
            new_list.append( get_trace()[pt:] )
        else:
            print( "Cutting point %d is out of range" %pt )

    if len(new_list) > 0: new_window_list( new_list )
    
    return True
    
def cut_traces_multi( pt_list ):
    """Cuts the selected traces at the sampling points
    in pt_list and shows the cut traces in a new window.
    Returns True upon success, False upon failure."""
    if not get_selected_indices():
        print( "Trace is not selected!" )
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
                print( "Cutting point %d is out of range" %pt )
        if len(new_list) > 0: new_list.append( get_trace()[old_pt:] )
    new_window_list( new_list )
    return True

def template_matching(template, mode="criterion", norm=True, lowpass=0.5, highpass=0.0001):
    import sys
    sys.stderr.write("template_matching is deprecated. Use detect_events instead.\n")
    return detect_events(template, mode, norm, lowpass, highpass)

def detect_events(template, mode="criterion", norm=True, lowpass=0.5, highpass=0.0001):
    if not check_doc():
        return None
    import stfio
    return stfio.detect_events(get_trace(), template, get_sampling_interval(),
                               mode, norm, lowpass, highpass)

def peak_detection(data, threshold, min_distance):
    import stfio
    return stfio.peak_detection(data, threshold, min_distance)

class _cursor_pair(object):
    def __init__(self, get_start, set_start, get_end, set_end, get_value=None, index=None):
        self._get_start = get_start
        self._set_start = set_start
        self._get_end = get_end
        self._set_end = set_end
        self._get_value = get_value
        self._index = index

    def _get_cursors(self, is_time):
        if not check_doc(show_dialog=False):            
            raise StfException("Couldn't find open file")
        return (self._get_start(is_time=is_time), self._get_end(is_time=is_time))

    def _set_cursors(self, cursors, is_time):
        if not check_doc(show_dialog=False):            
            raise StfException("Couldn't find open file")
        try:
            if len(cursors) != 2:
                raise ValueError("cursors has to have length 2 when setting the time value")
        except TypeError:
            raise TypeError("cursors has to be a tuple or list of length 2")
        if cursors[0] is not None:
            self._set_start(cursors[0], is_time=is_time)
        if cursors[1] is not None:
            self._set_end(cursors[1], is_time=is_time)

    @property
    def cursor_time(self):
        return self._get_cursors(True)

    @cursor_time.setter
    def cursor_time(self, cursors):
        self._set_cursors(cursors, True)

    @property
    def cursor_index(self):
        return self._get_cursors(False)

    @cursor_index.setter
    def cursor_index(self, cursors):
        self._set_cursors(cursors, False)

    @property
    def value(self):
        if self._get_value is None:
            raise AttributeError("Missing _get_value function")
        return self._get_value()

    @property
    def index(self):
        if self._index is None:
            raise AttributeError("Missing _index function")
        return self._index()

    
base = _cursor_pair(get_base_start, set_base_start, get_base_end, set_base_end,
                    get_base)
peak = _cursor_pair(get_peak_start, set_peak_start, get_peak_end, set_peak_end,
                    get_peak, peak_index)
fit = _cursor_pair(get_fit_start, set_fit_start, get_fit_end, set_fit_end)
latency = _cursor_pair(
    get_latency_start, set_latency_start, get_latency_end, set_latency_end,
    get_latency)

}
//--------------------------------------------------------------------
