*******************************
Calculations on selected traces
*******************************

:Author: Jose Guzman
:Date:  |today|

A widely used feature of ``Stimfit`` is the selection of some traces of interest within a file to make some calculations on them (average, peaks, amplitudes, etc.). The batch-analysis of ``Stimfit`` does precisely that. However, in some cases we can enhance its possibilities writing our custom functions in Python for the selected traces. 

In ``Stimfit``, the indices of selected traces can be easily retrieved using :func:`stf.get_selected_indices()`. This function returns a tuple with the selected indices. 

::

    >>> stf.get_selected_indices()
    >>> (1,2,3)

In this case, we selected the 2nd, 3rd and 4th trace in the file (note the zero-based index!).

The routine described below performs a simple algorithm only on the traces selected previously (either with the menu bar or with typing **S**). I've chosen a very simple calculation (amplitude of the signal) for didactic purposes, but a more complex function can be writen.


=====================
Using selected traces
=====================

In the following function we calculate the amplitude of the signal of the selected traces. One of the arguments of the function (*trace=None*) will select the trace that we want to use to make the calculation. Note that this is an optional argument; by default it will accept the current trace (or sweep) of the file, but if not, you can enter the zero-based index of the traces in the channel. 

The amplitude function will be calculated based on the traces selected by *trace*. Here is the function:


::


    # stimfit python module 
    import stf
        
    def get_amplitude(base, peak, delta, trace=None):
        """ Calculates the amplitude deviation (peak-base) in units of the Y-axis

        Arguments:
        base        -- Starting point (in ms) of the baseline cursor.
        peak        -- Starting point (in ms) of the peak cursor.
        delta       -- Time interval to calculate baseline/find the peak.
        trace       -- Zero-based index of the trace to be processed, if None then current 
                        trace is computed.
        

        Returns:
        A float with the variation of the amplitude. False if  

        Example:
        get_amplitude(980,1005,10,i) returns the variation of the Y unit of the trace i between 
        peak value (10050+10) msec and baseline (980+10) msec 
        """

        # sets the current trace or the one given in trace
        if trace is None:
            sweep = stf.get_trace_index()
        else:
            if typ(trace) != int:
                print "trace argument admits only intergers"
                return False
            sweep = trace 
    

        # set base cursors:
        if not(stf.set_base_start(base,True)): return False # out-of range
        if not(stf.set_base_end(base+delta,True)): return False 

        # set peak cursors:
        if not(stf.set_peak_start(peak,True)): return False # out-of range
        if not(stf.set_peak_end(peak+delta,True)): return False 

        # update measurements
        stf.set_trace(sweep)

        amplitude = stf.get_peak()-stf.get_base() 

        return amplitude

==============
Code commented
==============

*None* is a Python built-in constant. It is used in to represent the absence of a value. Therefore, in our example, when the argument *trace* is empty (its value is *None*) we will simply select the current trace with :func:`stf.get_trace_index()` and store it in the variable **sweep**. If not, the variable **sweep** will take the value taken by *trace*. This iscontroled by the following if-block within the function:

::

    if trace is None:
        sweep = stf.get_trace_index()
    else:
        if type(trace) !=int:
            print "trace argument admits only integers"
            return False
        sweep = trace

An additional if block inside the else instruction allows us to control that trace will be an integer. 

..

    >>> if type(trace) !=int:

If the argument traces is not an integer, the function will be cancell and returns False.

Note that after setting the stf cursors, we update the measurements in the trace whose index is given by the local variable **sweep** with :func:`stf.set_trace()`.

=====
Usage
=====

The function accepts an optional *trace* argument. That means, that we do not need to declare it when using the function. In that case, the function will work on the current trace. For example, if we want to calculate the amplitude between a baseline between (500+10) msec and a peak between 750 and 760 msec on the current trace, we simply enter:

::

    >>> spells.get_amplitude(500,750,10)

To calculate the same amplitude in the trace number 10 (zero-based index is 9) we can type:

::

    >>> spells.get_amplitude(500,750,10,9)

More interesting is to get the amplitude in the selected traces, we can pass the tuple of selected traces to the *trace* argument and thereby calculate the amplitude on our selected traces:

::

    >>> amplitudes_list = [spells.get_amplitude(500,750,10,i) for i in stf.get_selected_indices()]

In this way the tuple of selected indices is passed by the for loop to the function. Next, everything is wrapped in a Python list called amplitudes_list. 

For further analysis in spreadsheet programs (Calc, Gnumeric, Excel or similar), the values can be printed into a table that allows to copy and paste the contents. :func:`stf.show_table` takes a dictionary as its first argument. The dictionary has to be composed of strings as keys and numbers as values. You could use it as follows:

::

    >>> mytable = dict()
    >>> for i in stf.get_selected_indices(): mytable["Trace %.3d" % i] = amplitudes_list[i]
    >>> stf.show_table(mytable)

Note that the dictionary will be sorted alphabetically according to its keys. Therefore, using "%.3d" is used to keep the table in the same order as the traces. If you wanted to print out more than one value for each trace, you could use :func:`stf.show_table_dictlist` that uses a similar syntax, but requires a list of numbers as the values of the dictionary.
