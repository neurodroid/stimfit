**************
Event counting
**************

:Author: Jose Guzman
:Date:  |today|

Counting the number of events (for example action potentials) in a fixed time window is a very common task for the electrophysiologist. In the easiest form, the user would like to know how many spikes we can find following the onset of a stimulus (i.e current injection). We can write a simple Python function which automatically perform this calculation with a simple event detection routine. 

.. note::
    Stimfit has one in-build function which provides a way to count the number of events (i.e action potentials). In the menu bar, just select Analysis -> event detection-> threshold crossing... or alternatively with the Analysis-> Batch analysis->threshold crossing.
    
==========================
The counter event function
==========================

The following function counts the number of events (e.g action potentials) in a trace as responses above (updirection=True) or bellow (updirection=False) a threshold (by default 0). The argument updirectionenhances the original ``Stimfit`` event detection function. We can detect events in the possitive and negative direction.
::
    
    # load main Stimfit module
    import stf

    def count_events(start, delta, threshold=0, updirection=True, trace=None):
        """
        Counts the number of events (e.g action potentials (AP)) in the current trace.
    
        Arguments:

        start       -- starting time (in ms) to look for events. 
        delta       -- time interval (in ms) to look for events.
        threshold   -- (optional) detection threshold (default = 0).
        updirection -- (optional) True (default) will look for events upwards, False downwards. 
        trace       -- (optional) zero-based index of the trace in the current channel, 
                    if None, the current trace is selected.
                        
        Returns:
        An integer with the number of events.
         
        Examples:
        count_events(500,1000) returns the number of events found between t=500 ms and t=1500 ms 
            above 0 in the current trace.
        count_events(500,1000,False,-10,i) returns the number of events found bellow -10 in the
            trace i.
        """

        # sets the current trace or the one given in trace.
        if sequence is None:
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
        selection = stf.get_trace()[pstart:(pstart+delta)]

        # algorithm to detect events
        EventCounter,i = 0,0 # set counter and index to zero

        # first test direction, then run the loop
        if updirection is True:
            while i<len(selection):

                if selection[i]>threshold:
                    EventCounter +=1
                    while selection[i]>threshold: i+=1 # jump until the value is bellow the trheshold again
                i+=1

        else:
            while i<len(selection):

                if selection[i]<threshold:
                    EventCounter +=1
                    while selection[i]<threshold: i+=1
                i+=1

        return EventCounter 
                    
==============
Code commented
==============

The traces are selected the optional argument provided by trace. Please, consult the code commented section in :doc:`/howto/amplitudes` if you are not familiar with this construction. The algorithm to detect action potentials requires some familiarity with Python iterations but it is easy to understand. 

::

    if updirection is True: 
        while i<len(selection):

            if selection[i] >threshold:
                EventCounter +=1
                while selection[i]>threshold: i+=1 # jump until the value is bellow the threshold again
            i+=1

The block of code above will be only exectued if the condition is True (by default to look for events in the upward direction). If the condition is False, the alternative code will be executed (and the function will look for events in downward direction).

The while loop inside the if condition allows us to move withtin the indices of the array called selection. We insert a if-block inside to test if the position [i] is above the threshold. In this case we will add 1 to the counter (EventCounter +=1) and move to the second while loop. 

..

    >>> while selection[i]>threshold: i+=1 # jump until the value is bellow the threshold again
    
This second loop is very important, because it moves within the array until the value is bellow the threshold again, and jumps every value above the threshold in the array. If we do not write this while there, the if condition will be True for all the values above the threshold, and the counter would give us the number of sampling points above that threshold (and not the number of events). 

Finally, we move the index one to the next position in the array to look for the next event whenever the position is not larger that the length of the array. Note that preserving the Python indentation is extremely important here. The last i+=1 belongs to the first while condition (while i<len(selection), and allows us to perform the loop apropiately.

.. note::

    Do not try to write while loops in the embedded python console of ``Stimfit`` unless you are very familiar with while loops in Python or in any other language. While loops, if written incorrectly, may run infinite iterations and block the Python terminal of ``Stimfit``. For that reason, it is a good idea to explore while loops in an independent python terminal before using them in ``Stimfit``. 

=====
Usage
=====

As in :doc:`/howto/amplitudes` we can use the function in different ways:

::

    >>> myFile.count_events(500,1000)

will return the number of events above 0 mV in the current trace/channel between t=500 ms and t=1500 ms.

::

    >>> myFile.count_events(500,100,False,-40,10)

this will look for events bellow the value -40 but not in the current trace, only in the trace 9 (zero-based index is 10).

::

    >>> spikes_list= [myFile.count_events(500,1000,True,0,i) for i in stf.get_selected_indices()]

will create a Python list with the number of events (e.g spikes) found between t=500ms and t=1500ms above 0 in the selected traces.

::

   >>> mytable = dict()
   >>> for i in stf.get_selected_indices(): mytable["Trace %.3d" %i] = spikes_list[i]
   >>> stf.show_table(mytable)

this creates a table with 2 columns with the trace number a number of spikes found previously. 
