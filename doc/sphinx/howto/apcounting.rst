**************
Event counting
**************

:Author: Jose Guzman
:Date:  |today|

Counting the number of events (for example action potentials) within a time window is a very common task in electrophysiology. In its simplest form, the user would like to know how many spikes occur following the onset of a stimulus (i.e current injection). We can write a simple Python function which automatically performs this calculation with a simple event detection routine. 

.. note::
    Stimfit has built-in functions to count the number of events (i.e action potentials). From the menu, select Analysis -> event detection-> threshold crossing... or alternatively with the Analysis-> Batch analysis->threshold crossing. However, this Python script allows for more flexibility while counting events, such as detecting positive- or negative-going events.
    
==========================
The counter event function
==========================

The following function counts the number of events (e.g action potentials) by detecting upward (up=True) or downward (up=False) threshold-crossings.
::
    
    # load main Stimfit module
    import stf

    def count_events(start, delta, threshold=0, up=True, trace=None, mark=True):
        """
        Counts the number of events (e.g action potentials (AP)) in the current trace.
    
        Arguments:

        start       -- starting time (in ms) to look for events. 
        delta       -- time interval (in ms) to look for events.
        threshold   -- (optional) detection threshold (default = 0).
        up          -- (optional) True (default) will look for upward events, False downwards. 
        trace       -- (optional) zero-based index of the trace in the current channel, 
                       if None, the current trace is selected.
        mark        -- (optional) if True (default), set a mark at the point of threshold crossing                        
        Returns:
        An integer with the number of events.
         
        Examples:
        count_events(500,1000) returns the number of events found between t=500 ms and t=1500 ms 
            above 0 in the current trace and shows a stf marker.
        count_events(500,1000,0,False,-10,i) returns the number of events found below -10 in the
            trace i and shows the corresponding stf markers.
        """

        # sets the current trace or the one given in trace.
        if trace is None:
            sweep = stf.get_trace_index()
        else:
            if type(trace) !=int:
                print "trace argument admits only integers"
                return False
            sweep = trace

        # set the trace described in sweep 
        stf.set_trace(sweep)

        # transform time into sampling points
        dt = stf.get_sampling_interval()

        pstart = int( round(start/dt) )
        pdelta = int( round(delta/dt) )

        # select the section of interest within the trace
        selection = stf.get_trace()[pstart:(pstart+pdelta)]

        # algorithm to detect events
        EventCounter,i = 0,0 # set counter and index to zero

	# choose comparator according to direction:
	if up:
	    comp = lambda a, b: a > b
        else:
            comp = lambda a, b: a < b

        # run the loop
	while i<len(selection):
            if comp(selection[i],threshold):
                EventCounter +=1
		if mark:
		    stf.set_marker(pstart+i, selection[i])
                while comp(selection[i],threshold):
                    i+=1 # skip values until the value is below or above threshold again
            i+=1

        return EventCounter 
                    
==============
Code commented
==============

The traces are selected by the optional argument trace, as explained in :doc:`/howto/amplitudes`. The algorithm to detect action potentials requires some familiarity with Python iterations but it is easy to understand. 

::

    while i<len(selection):
        if comp(selection[i], threshold):
            EventCounter +=1
            while comp(selection[i], threshold): 
                i+=1 # skip values until the value is below or above threshold again
        i+=1

The while loop allows us to move within the indices of the array called selection. We insert an if-block inside to test whether the threshold is crossed at [i]. In this case we will add 1 to the counter (EventCounter +=1) and move to the second while loop. 

::

    while comp(selection[i], threshold): 
        i+=1 # jump until the value is below or above threshold again
    
This second loop is very important, because it moves within the array until the value crosses the threshold again in the other direction, and skips every value until the threshold is crossed again. If we do not write this while there, the if condition will be True for all values after the threshold crossing, and the counter would give us the number of sampling points between threshold crossings (and not the number of events). 

Finally, we move the index one to the next position in the array to look for the next event whenever the position is not larger that the length of the array. Note that preserving the Python indentation is extremely important here. The last i+=1 belongs to the first while condition (while i<len(selection), and allows us to perform the loop appropriately.

.. note::

    Do not try to write while loops in the embedded python console of ``Stimfit`` unless you are very familiar with while loops in Python or in any other language. While loops, if written incorrectly, may run infinite iterations and block the Python terminal of ``Stimfit``. For that reason, it is a good idea to explore while loops in an independent python terminal before using them in ``Stimfit``. 

=====
Usage
=====

As in :doc:`/howto/amplitudes` we can use the function in different ways:

::

    >>> spells.count_events(start=500,delta=1000)

will return the number of events above 0 mV in the current trace/channel between t=500 ms and t=1500 ms, and shows a blue stf marker 

::

    >>> spells.count_events(start=500,delta=1000,threshold=-40,up=False,trace=10,mark=False)

this will look for events below the value -40 but not in the current trace, only in the trace 11 (zero-based index is 10) in the downwards direction. Here a blue marker around the point found bellow the threshold will be shown too. Note that functions with a large number of arguments are difficult to remember. You can always change the order of the arguments if you describe the arguments in the function. For example, the following sentence has the same effect as the one above, but shows a different argument order:

::

    >>> spells.count_events(threshold=-40,start=500,up=False,delta=1000,mark=False,trace=10)

If you want to create a list of events with the events found in a selection of traces, you can simply type:

::

    >>> spikes_list= [spells.count_events(500,1000,0,True,i,False) for i in stf.get_selected_indices()]

this will create a Python list with the number of events (e.g spikes) found between t=500ms and t=1500ms above 0 in the selected traces and no marker will be shown. In the same way as described previously in , you can create a table to copy the results.
::

   >>> mytable = dict()
   >>> for i in stf.get_selected_indices(): mytable["Trace %.3d" %i] = spikes_list[i]
   >>> stf.show_table(mytable)

this creates a table with 2 columns with the trace number a number of spikes found previously. 

Obviously, the function could be extended to return the time points of threshold crossings so that the interspike intervals can be calculated. This is left as an exercise to the reader.

.. note::

    Use the :func:`stf.erase_markers()` to clean the blue markers on the main stf window. If not, every time that you call the routine in the given trace, a series of blue markers showing the crossing points of the different threshold will overlap with each other. Alternatively, you can add :func:`stf.erase_markers()` in the beginning of count_events() to delete any marker presented previously:

    
