************
Running mean
************

Running mean (or running average) is a way to smooth the data. Given a certain set of points, a running average will create a new set of data points which will be computed by adding a series of averages of different subsets of the full data set.

Given for example a sequence :math:`X_i` of :math:`n` points.

.. math::

    {\displaystyle \sum^{i=n}_{i=1} X_i}

we can create a new set of data points :math:`S_i` simply taking the average of a subset of :math:`w` points from the original data set:

.. math::

    {\displaystyle S_i=\frac{1}{w} \sum^{w+i}_{j=i} X_j }

=========================
The running mean function
=========================

The following python function perform the running mean of the current channel. Both trace and channel can be selected as zero-based indices. The width of the running average (refereed here as binwidth) can be selected. The resulting average will appear in a new ``Stimfit`` window.

::
    
    # load main Stimfit module
    import stf

    # load NumPy for numerical analysis
    import numpy as np

    def rmean(binwidth, trace=1,channel=-1):
        """
        perform a running mean of the current trace 
        in the channel to smooth the trace. 
    
        Keyword arguments:

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

        # creates a destiny python list to append the data 
        dsweep = [] 

        # running mean algorithm
        for i in range(len(sweep)):
        
            if ( (len(sweep)-i) > binwidth):
                # append to list the running mean of `binwidth` values
                # np.mean(sweep) makes the mean of list
                # sweep[p0:p10] takes the values in the vector p0 and p10 (zero-based) 
                dsweep.append(np.mean(sweep[i:(binwidth+i)]))

            else:
                break

        stf.new_window(dsweep)


To perform the running average of 10 sampling points of the current trace, simply type:

::

    >>> myfile.rmean(10)
