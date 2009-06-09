**************
Cutting traces
**************

:Author: Jose Guzman
:Date:  |today|

As described in :doc:`/manual/python` chapter of the :doc:`/manual/index`, a very often requested feature of ``Stimfit`` is to cut an original trace to show it in a presentation or publication. This feature, however, has been only integrated into the stf module, and not in the ``Stimfit`` main menubar. With this, ``Stimfit`` preserves its user interface as clear and user-friendly as possible.

We can use the built-in stf function :func:`stf.new_window()` to show a new stf window with the current trace within an interval. For example, 
::

    >>> stf.new_window(stf.get_trace()[1600:3200])

presents a new window with the current trace between the sampling points 1600 and 3200. Remember that :func:`stf.new_window()` takes a 1D-NumPy array as argument. To cut the trace within the desired limits, we have to slice it before with
::

    >>> stf.get_trace()[1600:3200]
    
Note that the index :math:`i` of a sampling point and the corresponding time :math:`t`, measured from the start of the trace, are related as follows:

.. math::

      {\displaystyle i=\frac{t}{\Delta t}}  

where the sampling interval :math:`\Delta t` can be obtained with the following function:
::

    >>> dt = stf.get_sampling_interval()

Then, if our sampling interval (dt) is 0.05 msec, the points selected correspond to 80 and 160 msec respectively. Alternatively, one could have thought about this command:
::

    >>> stf.get_trace()[80/dt:160/dt]

However this will not work.  Slicing requires integers arguments and not floats (both 80/dt and 160/dt are floats). So we have to transform this ratio to integer. Besides that, the float precision of python will play against us here. If we make dt = stf.get_sampling_interval, we will find that dt = 0.05000000074505806 rather than 0.05. Note that if you do not round up (with ceil) before int(80/dt) you will get one sampling point less than expected.   

::

    >>> from math import ceil # we need this module to round the float
    >>> pstart = int(ceil(80/dt)) # we ceil (80/dt=1599.9999) before transforming into integer to get 1600.0
    >>> pend =  int(ceil(160/dt)) # the same, 160/dt=31.9999) we ceil it to 3200.0
    >>> stf.get_trace()[pstart:pend] # now the slicing withing the integer values

This solution, although not ideal, is the only one I could think about. I did not find a more elegant way to do it.

============================
The cutting traces  function
============================

In the chapter devopted to Python (:doc:`/manual/index`)  in  :doc:`/manual/index` you can find a function to cut a given trace within the sampling points. This function is slightly different. As described above, we would take times and not sampling points as argument. After that, we will take list of traces and not a single trace to cut. This function will use :func:`stf.new_window_list()` which takes a list of 1D-Numpy arrays to present a new stf window.

::
    
    # load main Stimfit module
    import stf

    # load ceil to round the sampling points
    from math import ceil

    def cut_sweeps(start, delta, sequence=None):
        """
        Cut the traces selected in sequence and creates a new
        window with them.
    
        Arguments:

        start       -- starting point (in ms) to cut. 
        delta       -- time interval (in ms) to cut

        sequence:   -- list of indices to be cut. If None, every trace in the
                        channel will be cut.
                        
        Returns:
        A new window with the traced cut. 
        
        Examples:
        cut_sweeps(200,300) cut the traces between t=200 ms and t=500 ms within the whole channel.
        cut_sweeps(200,300,range(30,60)) the same as above, but only between traces 30 and 60.
        cut_sweeps(200,300,stf.get_selected_indices()) cut between 200 ms and 500 msec
            only in the selected traces.

        """

        # select every trace in the channel if not selection is given in sequence
        if sequence is None:
            sequence = range(stf.get_size_channel())
        else:
            if type(sequence) != list:
                sequence = list(sequence)
        
        # transform time into sampling points
        dt = stf.get_sampling_interval()

        pstart = int(ceil(start/dt))
        pdelta = int(ceil(delta/dt))

        # creates a destination python list to append the data 
        dlist = [] 

        # creates a sequence of 1D-NumPy arrays
        for i in sequence:
            dlist.append(stf.get_trace(i)[pstart:(pstart+pdelta)])        

        return stf.new_window_list(dlist)

Code commented
**************

We provide some flexibility with the argument *sequence*. By default, we will select every trace in the channel.

::

    if sequence is None:
        sequence = range(stf.get_size_channel())

    else:
        if type(sequence) != list:
            list(sequence)

but if we want to use a python type other than a list (for example a tuple) we have to cast it to a list before. This will allow us to use :func:`stf.get_selected_indices` as argument for the function (remember that :func:`stf.get_selected_indices()` returns a tuple with the indices of the selected traces in a channel).

Finally we add to the list the NumPy arrays whose index is described in the sequence.

::

    for i in sequence:
        dlist.append(stf.get_trace(i)[pstart:(pstart+pdelta)])

and slice the NumPy array as described above.

=====
Usage
=====
In any case, a new stf window with the traces cut will appear

::

    >>> myfile.cut_sweeps(200,300)

will create a new window with all the traces of the channel cut between t=200 ms and t=500 ms.

::

    >>> myfile.cut_sweeps(200,300,range(30,60))

will create a new window with the same selection, but only between the traces 30 and 60.

::

    >>> myfile.cut_sweeps(200,300,stf.get_selected_indices())

will create a new window with the cut traces only if they were previously selected.

