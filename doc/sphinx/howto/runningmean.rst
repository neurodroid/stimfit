************
Running mean
************

:Author: Jose Guzman
:Date:  |today|

The running mean (or running average) is simple way to smooth the data. Given a certain set of points, a running average will create a new set of data points which will be computed by adding a series of averages of different subsets of the full data set.

Given for example a sequence :math:`X` of :math:`n` points, we can create a new set of data points :math:`S` of length :math:`n` by simply taking the average of a subset of :math:`w` points from the original data set for every point :math:`S_i` within the set:

.. math::

    {\displaystyle S_i=\frac{1}{w} \sum^{w+i}_{j=i} X_j }

=========================
The running mean function
=========================

The following Python function calculates the running mean of the current channel. Both trace and channel can be selected as zero-based indices. The width of the running average (refereed to here as binwidth) can be selected. The resulting average will appear in a new `Stimfit <http://www.stimfit.org>`_ window.

::
    
    # load main Stimfit module
    import stf

    # load NumPy for numerical analysis
    import numpy as np 

    def rmean(binwidth, trace=-1,channel=-1):
        """
        Calculates a running mean of a single trace
    
        Arguments:

        binwidth    -- size of the bin in sampling points (pt). 
        Obviously, it should be smaller than the length of the trace.

        trace:  -- ZERO-BASED index of the trace within the channel. 
        Note that this is one less than what is shown in the drop-down box.
        The default value of -1 returns the currently displayed trace.

        channel  -- ZERO-BASED index of the channel. This is independent 
        of whether a channel is active or not. The default value of -1 
        returns the currently active channel.

        Returns: 

        A smoothed traced in a new stf window.

        """
        # loads the current trace of the channel in a 1D Numpy Array
        sweep = stf.get_trace(trace,channel)

        # creates a destination python list to append the data 
        dsweep = np.empty((len(sweep))) 

        # running mean algorithm
        for i in range(len(sweep)):
        
            if (len(sweep)-i) > binwidth:
                # append to list the running mean of `binwidth` values
                # np.mean(sweep) calculates the mean of list
                # sweep[p0:p10] takes the values in the vector between p0 and p10 (zero-based) 
                dsweep[i] = np.mean( sweep[i:(binwidth+i)] )

            else:
	        # use all remaining points for the average:
                dsweep[i] = np.mean( sweep[i:] )
		

        stf.new_window(dsweep)

==============
Code commented
==============

`Stimfit <http://www.stimfit.org>`_ commonly uses the value -1 to set the current trace/Channel. In this function the default argument values are -1 (see the function arguments *trace=-1* and *channel=-1*). 

..

    >>> sweep = stf.get_trace(trace,channel)

:func:`stf.get_trace()` simply imports the **trace** of the **channel** into a 1D-Numpy array that we called sweep. The default values provided by the function are -1. This means that by default, the current trace/channel will be imported.

We create a new stf window with the following 

..

    >>> stf.new_window(dsweep)

where dsweep is the 1D-NumPy array obtained after performing  the running average.

=====
Usage
=====

To perform the running average of 10 sampling points of the current trace, simply type:

::

    >>> spells.rmean(10)

A new window with the running mean will appear.
