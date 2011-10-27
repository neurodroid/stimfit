:mod:`stf`
==========

:Author: Christoph Schmidt-Hieber (christsc at gmx.de)
:Release: |version|
:Date:  |today|

.. module:: stf
    :synopsis: The stf module allows to access a running stimfit application from the embedded python shell.

The :mod:`stf` module defines the following functions:

.. function:: align_selected(alignment, active=False)

    Aligns the selected traces to the index that is returned by the alignment function, and then creates a new window showing the aligned traces.
        
        **Arguments:**       

        *alignment* -- The alignment function to be used. 
        Accepts any function returning a valid index within a trace. These are some predefined possibilities: maxrise_index (default; maximal slope during rising phase), peak_index (Peak of an event), foot_index (Beginning of an event), t50left_index, t50right_index (Left/right half-maximal amplitude)

        *active* --    If True, the alignment function will be applied to the active channel. If False (default), it will be applied to the inactive channel.

        *zeropad* --   Not yet implemented:If True, missing parts at the beginning or end of a trace  will be padded with zeros after the alignment. If False (default), traces will be cropped so that all traces have equal sizes.

.. function:: check_doc(\*args)
        
    Checks whether a file is open.
        
        **Returns:**

        True if a file is open, False otherwise

.. function:: close_all(\*args)

    Closes all open files.

        **Returns:**

        True if all files could be closed.

.. function:: close_this(\*args)

    Closes the currently active file.
           
        **Returns:**

        True if the file could be closed.

.. function:: cut_traces(pt)

    Cuts the selected traces at the sampling points in pt_list and shows the cut traces in a new window.

        **Returns:**

        True upon success, False upon failure.

.. function:: cut_traces_multi(pt_list)

    Cuts the selected traces at the sampling points in pt_list and show the cut traces in a new window. 

        **Returns:**

        True upons sucess, False upon failure.

.. function:: erase_markers()

    Delete the markes created with :func:`set_marker()`

.. function:: file_open(\*args)

    Opens a file.

        **Arguments:**

        *filename* -- The file to be opened. On Windows, use double back-slashes ("\\") between directories to avoid conversion to special characters such as "\t" or "\n".
        Example usage in Windows:
        
        ::
        
        >>> file_open("C:\\data\\datafile.dat"). 
        
        Example usage in Linux:
        
        ::
        
        >>> file_open("/home/cs/data/datafile.dat").
        
        This is surprisingly slow when called from python. Haven't figured out the reason yet.

        **Returns:**

        True is the file could be opened, False otherwise.

.. function:: file_save(\*args)

    Saves a file.

        **Arguments:**

        *filename* -- The file to be saved. On Windows, use double back-slashes ("\\") between directories to avoid con-version to special characters such as "\t" or "\n".

        Example usage in Windows: 

        ::
        
        >>> file_save("C:\\data\\datafile.dat")
                    
        Example usage in Linux:

        ::

        >>> file_save("/home/cs/data/datafile.dat")
                    
        This is surprisingly slow when called from python. Haven't figured out the reason yet.

        **Returns:**

        True if the file could be saved, False otherwise.

.. function:: foot_index(active=True)

    Returns the zero-based index of the foot of an event in the active channel. The foot is the intersection of an interpolated line through the points of 20 and 80% rise with the baseline. Uses the currently measured values, i.e. does not update measurements if the peak or base window cursors have changed.

        **Arguments:**

        *active* -- If True, returns the current index of the foot within the active channel. Only implemented for the active channelat this time. Will return a negative value and show an error message if *active* == False.

        **Returns:**

        The zero-based index of the foot of an event in units of sampling points. Interpolates between sampling points. Returns a negative value upon failure.


.. function:: get_base(\*args)

    Returns the current baseline value. Uses the currently measured values, i.e. does not update measurements if the peak or base window cursors have changed.
       
        **Returns:** 

        The current baseline.
   
.. function:: get_base_end(is_time=False)

    Returns the zero-based index or the time point of the base end cursor.
        
        **Arguments:**

        *is_time* -- If False (default), returns the zero-based index. If True,returns the time from the beginning of the trace to the cursor position.
    

.. function:: get_base_start(is_time=False)

    Returns the zero-based index or the time point of the base start cursor.
        
        **Arguments:**

        *is_time* -- If False (default), returns the zero-based index. If True,returns the time from the beginning of the trace to the cursor position.
    
.. function:: get_channel_index(active=True)

    Returns the ZERO-BASED index of the specified channel.
        
        **Arguments:**
        
        *active* -- If True, returns the index of the active (black) channel. If False, returns the index of the inactive (red) channel.
    
.. function:: get_channel_name(index=-1)

    Returns the name of the channel with the specified index.
        
        **Arguments:**
        
        *index* -- The zero-based index of the channel of interest. If < 0, the name of the active channel will be returned.
        
        **Returns:**

        the name of the channel with the specified index.
    
.. function:: get_filename(\*args)
    
    Returns the name of the current file.
   
.. function:: get_fit_end(is_time=False)

    Returns the zero-based index or the time point of the fit end cursor.
        
        **Arguments:**

        *is_time* -- If False (default), returns the zero-based index. If True,returns the time from the beginning of the trace to the cursor position.
    
    
.. function:: get_fit_start(is_time=False)

    Returns the zero-based index or the time point of the fit start cursor.
        
        **Arguments:**
        
        *is_time* -- If False (default), returns the zero-based index. If True, returns the time from the beginning of the trace to the cursor position.
    
.. function:: get_latency(\*args)

    Returns the latency value (in x-units) determined by the latency cursors set in the cursors settings menu. Call :func:`measure()` or hit enter to update the cursors.

.. function:: get_maxdecay(\*args)

    Returns the the maximal slope of the decay between the peak cursors. Returns -1.0 upon error. Call :func:`measure()` or hit enter to update the value.

.. function:: get_maxrise(\*args)

    Returns the the maximal slope of the rise between the peak cursors. Returns -1.0 upon error. Call :func:`measure()` or hit enter to update the value.

.. function:: get_risetime(\*args)

    Returns the 20-80% rise time (in x-units) by calculation of the interpolated adjacent sampling points at 20% and 80% of the peak amplitude. Returns -1.0 upon failure. Call :func:`measure()` or hit enter to update the value.

.. function:: get_slope(\*args)

    Returns the slope value using the cursors described in the cursors setting dialog.

        **Returns:**

        The slope value
    
.. function:: get_peak(\*args)

    Returns the current peak value, measured from zero (!). Uses the currently measured values, i.e. does not update measurements if the peak or base window cursors have changed.
                 
        **Returns:**

        The current peak value, measured from zero (again: !).
    
.. function:: get_peak_end(is_time=False)
    
    Returns the zero-based index or the time point jof the peak end cursor.
        
        **Arguments:**

        *is_time* -- If False (default), returns the zero-based index. If True, returns the time from the beginning of the trace to the cursor position.
    
.. function:: get_peak_start(is_time=False)
        
    Returns the zero-based index or the time point of the peak start cursor.
        
        **Arguments:**

        *is_time* -- If False (default), returns the zero-based index. If True, returns the time from the beginning of the trace to the cursor position.
    
    
.. function:: get_recording_comment(\*args)
        
    Returns a comment about the recording.
    
.. function:: get_recording_date(\*args)

    Returns the date at which the recording was started as a string.
    
.. function:: get_recording_time(\*args)

    Returns the time at which the recording was started as a string.

.. function:: get_sampling_interval(\*args)

    Returns the sampling interval.
        
.. function:: get_selected_indices(...)
        
    Returns a tuple with the indices (ZERO-BASED) of the selected traces.

.. function::  get_size_channel(channel=-1) 

    Retrieves the number of traces in a channel.Note that at present, stimfit only supports equal-sized channels, i.e. all channels within a file need to have the same number of traces. The channel argument is only for future extensions. 
           
        **Arguments:**

        *channel* -- ZERO-BASED index of the channel. Default value of -1 will use the current channel. 
        
        **Returns:**

        The number traces in a channel.
    
.. function:: get_size_recording(\*args)

    Retrieves the number of channels in a recording.
           
        **Returns:**

        The number of channels in a recording.
    
.. function:: get_size_trace(trace=-1, channel=-1)

    Retrieves the number of sample points of a trace.
           
        **Arguments:**
        
        *trace* --   ZERO-BASED index of the trace. Default value of -1 will use the currently displayed trace. Note that this is one less than what is displayed in the drop- down list.
        
        *channel* -- ZERO-BASED index of the channel. Default value of -1 will use the current channel.
        
        **Returns:**

        The number of sample points.
    
.. function:: get_trace(trace=-1, channel=-1)
    
    Returns a trace as a 1-dimensional NumPy array.
          
        **Arguments:**      
        
        *trace* --   ZERO-BASED index of the trace within the channel. Note that this is one less than what is shown in the drop-down box. The default value of -1 returns the currently displayed trace.
        
        *channel* -- ZERO-BASED index of the channel. This is independent of whether a channel is active or not. The default value of -1 returns the currently active channel.
        
        **Returns:**

        The trace as a 1D NumPy array.
    
.. function:: get_trace_index(...)

    Returns the ZERO-BASED index of the currently displayed trace (this is one less than what is shown in the combo box).
    
.. function:: get_trace_name(trace=-1, channel=-1)

    Returns the name of the trace with the specified index.
        
        **Arguments:**
        
        *trace* -- The zero-based index of the trace of interest. If < 0, the name of the active trace will be returned.
        
        *channel* -- The zero-based index of the channel of interest. If < 0, the active channel will be used.
        
        **Returns:**

        the name of the trace with the specified index.

.. function:: get_threshold_time(is_time=False) 

    Returns the crossing value of the threshold slope. Note that this value is not update after changing the AP threshold. Call :func:`measure()` or hit enter in the main window to update the cursors.
        
        **Arguments:**
        
        *is_time* -- If false (default), returns the zero-based index at which the threshold slope is crossed. If True, returns the time at which the threshold slope is crossed (e.g. in units of the y-axis). A negative number is returned upon failure.
        
        **Returns:**

        False upon failure (such as out-of-range).

.. function:: get_threshold_value() 

    Returns value found at the threshold slope. Note that this value is not update after changing the AP threshold. Calle :func:`measure()` or hit enter in the main window to update the cursors.
        
        **Returns:**

        False upon failure (such as out-of-range).

    
.. function:: get_xunits(trace=-1, channel=-1) 

    Returns the x units of the specified section. X units are not allowed to change between sections at present, and they are hard-coded to "ms". This function is for future extension.
        
        **Arguments:**
        
        *trace* -- The zero-based index of the trace of interest. If < 0, the name of the active trace will be returned.
        
        *channel* -- The zero-based index of the channel of interest. If < 0, the active channel will be used.
        
        **Returns:**

        The x units as a string.

.. function:: get_yunits(trace=-1, channel=-1) 

    Returns the y units of the specified trace. Y units are not allowed to change between traces at present.
        
        **Arguments:**

        *trace* -- The zero-based index of the trace of interest. If < 0, the name of the active trace will be returned.

        *channel* -- The zero-based index of the channel of interest. If < 0, the active channel will be used.
        
        **Returns:**

        The x units as a string.
    
    
.. function:: leastsq(fselect, refresh=True)

    Fits a function to the data between the current fit cursors.
        
        **Arguments:**

        *fselect* -- Zero-based index of the function as it appears in the fit selection dialog.

        *refresh* -- To avoid flicker during batch analysis, this may be set to False so that the fitted function will not immediately be drawn.
        
        **Returns:**

        A dictionary with the best-fit parameters and the least-squared error, or a null pointer upon failure.
    
.. function:: leastsq_param_size(fselect) 

    Retrieves the number of parameters for a function.
        
        **Arguments:**
        
        *fselect* -- Zero-based index of the function as it appears in the fit selection dialog.
        
        **Returns:**

        The number of parameters for the function with index fselect, or a negative value upon failure.
    
.. function:: maxrise_index(active=True) 

    Returns the zero-based index of the maximal slope of rise in the specified channel. Uses the currently measured values, i.e. does not update measurements if the peak window cursors have changed.
           
        **Arguments:**

        *active*-- If True, returns the current index of the maximal slope of rise within the active channel. Otherwise, returns the current index of the maximal slope of rise within the inactive channel.
                  
        **Returns:**
        
        The zero-based index of the maximal slope of  rise in units of sampling points interpolated between adjacent sampling points. Returns a negative value upon failure.
        
.. function:: maxdecay_index() 

    Returns the zero-based index of the maximal slope of decay in the current channel. Uses the currently measured values, i.e. does not update measurements if the peak window cursors have changed. Note that in contrast to :func:`maxrise_index()`, this function only works on the active channel.

        **Returns:**

        The zero-based index of the maximal slope of decay in units of sampling points interpolated between adjacent sampling points. Returns a negative value upon failure.

.. function:: measure()
    
    Updates all measurements (e.g. peak, baseline, latency) according to the current cursor settings. As if you had pressed **Enter** in the main window.

        **Returns:**

        False upon failure, True otherwise.
   
.. function::  new_window(\*args)

    Creates a new window showing a 1D NumPy array.
              
        **Arguments:**

        *arg* -- The NumPy array to be shown.
    
.. function:: new_window_list(array_list)

    Creates a new window showing a sequence of 1D NumPy arrays, or a sequence of a sequence of 1D NumPy arrays. As opposed to :func:`new_window_matrix()`, this has the advantage that the arrays need not have equal sizes.
          
        **Arguments:**       

        *array_list* -- A sequence (e.g. list or tuple) of numpy arrays, or a sequence of a sequence of numpy arrays.
    
.. function:: new_window_matrix(\*args)

    Creates a new window showing a 2D NumPy array.
              
        **Arguments:**

        *arg* --   The NumPy array to be shown. First dimension are the traces, second dimension the sampling points within the traces.
    
    
.. function:: new_window_selected_all(\*args)
        
    Creates a new window showing the selected traces of all open files.

        **Returns:**

        True if successful.
    
.. function:: new_window_selected_this(\*args)
        
    Creates a new window showing the selected traces of the current file.

        **Returns:**

        True if successful.
        
    
.. function:: peak_index(active=True) 
        
    Returns the zero-based index of the current peak position in the specified channel. Uses the currently measured values, i.e. does not update measurements if the peak window cursors have changed.
           
        **Arguments:**
        
        *active* -- If True, returns the current peak index of the active channel. Otherwise, returns the current peak index of the inactive channel.
                  
        **Returns:**

        The zero-based index in units of sampling points. May be interpolated if more than one point is used for the peak calculation. Returns a negative value upon failure.
        
    
.. function:: select_all(\*args)
        
    Selects all traces in the current file. Stores the baseline along with the trace index.
    
.. function:: select_trace(trace=-1) 

    Selects a trace. Checks for out-of-range indices and stores the baseline along with the trace index.
           
        **Arguments:**

        *trace* --   ZERO-BASED index of the trace. Default value of -1 will select the currently displayed trace. Note that this is one less than what is displayed in the drop-down list.

        **Returns:**
        
        True if the trace could be selected, False otherwise.
    
.. function:: set_base_end(pos, is_time=False) 

    Sets the base end cursor to a new position.This will NOT update the baseline calculation. You have to either call :func:`measure()` or hit enter in the main window to achieve that.
        
        **Arguments:**
        
        *pos* -- The new cursor position, either in units of sampling points if *is_time* == False (default) or in units of time if *is_time* == True.

        *is_time* -- see above.
        
        **Returns:**

        False upon failure (such as out-of-range).
    
.. function:: set_base_start(pos, is_time=False) 

    Sets the base start cursor to a new position.This will NOT update the baseline calculation. You have to either call :func:`measure()` or hit enter in the main window to achieve that.
        
        **Arguments:**

        *pos* --     The new cursor position, either in units of sampling points if *is_time* == False (default) or in units of time if *is_time* == True.

        *is_time* -- see above.
        
        **Returns:**

        False upon failure (such as out-of-range).

.. function:: set_channel(channel)

    Sets the currently displayed channel to a new index. Subsequently updatges all measurements (e.g. peak, base, latency, i.e. you do not have to call :func:`measure()` yourself.)

        **Arguments:**

        *channel*-- The zero-based index of the new trace to be displayed.

        **Returns:**

        True upon sucess, false otherwise (such as out-of-range).
    
.. function:: set_channel_name(name, index=-1)

    Sets the name of the channel with the specified index.
        
        **Arguments:**
        
        *name*  -- The new name of the channel.

        *index* -- The zero-based index of the channel of interest. If < 0, the active channel will be used.
        
        **Returns:**

        True upon success.
    
.. function:: set_fit_end(pos, is_time=False) 

    Sets the fit end cursor to a new position.
        
        **Arguments:**

        *pos* --     The new cursor position, either in units of sampling points if *is_time* == False (default) or in units of time if *is_time* == True.

        *is_time* -- see above.
        
        **Returns:**

        False upon failure (such as out-of-range).
    
.. function:: set_fit_start(pos, is_time=False)

    Sets the fit start cursor to a new position.
        
        **Arguments:**

        *pos* --     The new cursor position, either in units of sampling points if *is_time* == False (default) or in units of time if *is_time* == True.

        *is_time* -- see above.
        
        **Returns:**

        False upon failure (such as out-of-range).
    
.. function:: set_marker(x, y) 

    Sets a marker to the specified position in the current trace.
        
        **Arguments:**

        *x* -- The horizontal marker position in units of sampling points.

        *y* -- The vertical marker position in measurement units (e.g. mV).
        
        **Returns:**
        
        False upon failure (such as out-of-range).
    
.. function:: set_peak_direction(direction)

    Sets the direction of the peak detection.
        
        **Arguments:**

        *direction* -- A string specifying the peak direction. Can be one of: "up", "down" or "both"
        
        **Returns:**

        False upon failure.
    
.. function:: set_peak_end(pos, is_time=False) 

    Sets the peak end cursor to a new position. This will NOT update the peak calculation. You have to either call :func:`measure()` or hit enter in the main window to achieve that.
        
        **Arguments:**

        *pos* --     The new cursor position, either in units of sampling points if *is_time* == False (default) or in units of time if *is_time* == True.
        *is_time* -- see above.
        
        **Returns:**

        False upon failure (such as out-of-range).

    
.. function:: set_peak_mean(pts) 

    Sets the number of points used for the peak calculation.
        
        **Arguments:**
        
        *pts* -- A moving average (aka sliding, boxcar or running average) is used to determine the peak value. Pts specifies the number of sampling points used for the moving window.Passing a value of -1 will calculate the average of all sampling points within the peak window.
        
        **Returns:**

        False upon failure (such as out-of-range).
    
.. function:: get_peak_direction()

    Gets the direction of the peak detection.

        **Returns:**
        
        A string specifying the peak direction. Can be one of: "up", "donw",or "both".

.. function:: get_peak_mean()

    Returns the number of sampling points used for peak calculation.

        **Returns:**
 
        0 upon failure (i.e no file opened). -1 means average of all sampling points.


.. function:: set_peak_start(pos, is_time=False) 
       
    Sets the peak start cursor to a new position. This will NOT update the peak calculation. You have to either call :func:`measure()` or hit enter in the main window to achieve that.
        
        **Arguments:**

        *pos* --     The new cursor position, either in units of sampling points if *is_time* == False (default) or in units of time if *is_time* == True.
        *is_time* -- see above.
        
        **Returns:**

        False upon failure (such as out-of-range).
    
    
.. function:: set_recording_comment(comment) 

    Sets a comment about the recording.
        
        **Argument:**

        *comment* -- A comment string.
        
        **Returns:**

        True upon successful completion.
    
.. function:: set_recording_date(date) 

    Sets a date about the recording.
        
        **Argument:**

        *date* -- A date string.
        
        **Returns:**

        True upon successful completion.
    
.. function:: set_recording_time(time) 

    Sets a time about the recording.
        
        **Argument:**

        *time* -- A time string.
        
        **Returns:**

        True upon successful completion.
    
.. function:: set_sampling_interval(si)

    Sets a new sampling interval.
        
        **Argument:**
        
        *si* -- The new sampling interval.
        
        **Returns:**

        False upon failure.

.. function:: set_slope(slope)

    Sets the AP threshold to the value given by the slope and takes it as reference for AP kinetic measurements. Note that you have to either call :func:`measure()` or hit enter to update calculations.

        **Argument:**

        *slope* --  Slope value in mV/ms

        **Returns:**

        False upon failure (such as out-of-range)

    
.. function:: set_trace(trace)
        
    Sets the currently displayed trace to a new index. Subsequently updates all measurements (e.g. peak, base, latency, i.e. you don't need to call :func:`measure()` yourself.)
        
        **Arguments:**

        *trace* -- The zero-based index of the new trace to be displayed.
        
        **Returns:**
        
        True upon success, false otherwise (such as out-of-range).
    
.. function:: set_xunits(units, trace=-1, channel=-1) 

    Sets the x unit string of the specified section. X units are not allowed to change between sections at present, and they are hard-coded to "ms". This function is for future extension.
        
        
        **Arguments:**

        *units* --   The new x unit string.

        *trace* --   The zero-based index of the trace of interest. If < 0, the name of the active trace will be returned.

        *channel* -- The zero-based index of the channel of interest. If < 0, the active channel will be used.
        
        **Returns:**

        True if successful.
    
.. function:: set_yunits(units, trace=-1, channel=-1) 

    Sets the y unit string of the specified trace. Y units are not allowed to change between traces at present.
        
        **Arguments:**

        *units* --   The new y unit string.

        *trace* --   The zero-based index of the trace of interest. If < 0, the name of the active trace will be returned.

        *channel* -- The zero-based index of the channel of interest. If < 0, the active channel will be used.
        
        **Returns:**
        
        True if successful.
    
.. function:: show_table(dict, caption="Python table")

    Shows a python dictionary in a results table. The dictionary has to have the form "string".
        
        **Arguments:**
        
        *dict* --    A dictionary with strings as key values and floating point numbers as values.

        *caption* -- An optional caption for the table.
        
        **Returns:**

        True if successful.
    
.. function:: show_table_dictlist(dict, caption="Python table", reverse=True)

    Shows a python dictionary in a results table. The dictionary has to have the form "string" : list. 
        
        **Arguments:**
        
        *dict* --    A dictionary with strings as key values and lists of floating point numbers as values.

        *caption* -- An optional caption for the table.

        *reverse* -- If True, The table will be filled in column-major order, i.e. dictionary keys will become column titles. Setting it to False has not been implemented yet.
        
        **Returns:**

        True if successful.
    
.. function:: subtract_base(\*args)

    Subtracts the baseline from the selected traces of the current file, then displays the subtracted traces in a new window.
        
        **Returns:**

        True if the subtraction was successful, False otherwise.
    
.. function:: t50left_index(active=True) 

    Returns the zero-based index of the left half-maximal amplitude of an event in the specified channel. Uses the currently measured values, i.e. does not update measurements if the peak or base window cursors have changed.
           
        **Arguments:**

        *active* -- If True, returns the current index of the left half-maximal amplitude within the active channel. If False,returns the current index of the left half-maximal amplitude within the inactive channel.
                  
        **Returns:**

        The zero-based index of the left half-maximal amplitude in units of sampling points. Interpolates between sampling points. Returns a negative value upon failure.
        
.. function:: t50right_index(active=True) 

    Returns the zero-based index of the right half-maximal amplitude of an event in the active channel. Uses the currently measured values, i.e. does not update measurements if the peak or base window cursors have changed.
           
        **Arguments:**

        *active*-- If True, returns the current index of the right half maximal amplitude within the active channel. Only implemented for the active channel at this time. Will return a negative value and show an error message if *active* == False.
                  
        **Returns:**

        The zero-based index of the right half-maximal amplitude in units of sampling points. Interpolates between sampling points. Returns a negative value upon failure.
        
    
.. function:: unselect_all(\*args)

    Unselects all previously selected traces in the current file.


