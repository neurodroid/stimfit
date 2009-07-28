"""
spells.py

Several Python recipes to solve frequently requested tasks with Stimfit.
You can find a complete description of these functions in the 
Stimfit online documentation (http://www.stimfit.org/doc/sphinx/index.html)
Check "The Stimfit Book of Spells" for details.
"""


import numpy as N

# stimfit python module:
import stf

def resistance( base_start, base_end, peak_start, peak_end, amplitude):
    """Calculates the resistance from a series of voltage clamp traces.

    Keyword arguments:
    base_start -- Starting index (zero-based) of the baseline cursors.
    base_end   -- End index (zero-based) of the baseline cursors.
    peak_start -- Starting index (zero-based) of the peak cursors.
    peak_end   -- End index (zero-based) of the peak cursors.
    amplitude  -- Amplitude of the voltage command.

    Returns:
    The resistance.
    """

    if not stf.check_doc():
        print "Couldn't find an open file; aborting now."
        return 0

    #A temporary array to calculate the average:
    set = N.empty( (stf.get_size_channel(), stf.get_size_trace()) )
    for n in range( 0,  stf.get_size_channel() ):
        # Add this trace to set:
        set[n] = stf.get_trace( n )


    # calculate average and create a new section from it:
    stf.new_window( N.average(set,0) )

    # set peak cursors:
    if not stf.set_peak_mean(-1): return 0 # -1 means all points within peak window.
    if not stf.set_peak_start(peak_start): return 0
    if not stf.set_peak_end(peak_end): return 0

    # set base cursors:
    if not stf.set_base_start(base_start): return 0
    if not stf.set_base_end(base_end): return 0

    # measure everything:
    stf.measure()

    # calculate r_seal and return:
    return amplitude / (stf.get_peak()-stf.get_base())

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
    dsweep = N.empty((len(sweep)))

    # running mean algorithm
    for i in range(len(sweep)):

        if (len(sweep)-i) > binwidth:
            # append to list the running mean of `binwidth` values
            # N.mean(sweep) calculates the mean of list
            # sweep[p0:p10] takes the values in the vector between p0 and p10 (zero-based)
            dsweep[i] = N.mean( sweep[i:(binwidth+i)] )

        else:
            # use all remaining points for the average:
            dsweep[i] = N.mean( sweep[i:] )


    stf.new_window(dsweep)

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

def cut_sweeps(start, delta, sequence=None):
    """
    Cuts a sequence of traces and present
    them in a new window.

    Arguments:

    start       -- starting point (in ms) to cut.
    delta       -- time interval (in ms) to cut
    sequence    -- list of indices to be cut. If None, every trace in the
                    channel will be cut.

    Returns:
    A new window with the traced cut.

    Examples:
    cut_sweeps(200,300) cut the traces between t=200 ms and t=500 ms within the whole channel.
    cut_sweeps(200,300,range(30,60)) the same as above, but only between traces 30 and 60.
    cut_sweeps(200,300,stf.get_selected_indices()) cut between 200 ms and 500 ms
        only in the selected traces.

    """

    # select every trace in the channel if not selection is given in sequence
    if sequence is None:
        sequence = range(stf.get_size_channel())

    # transform time into sampling points
    dt = stf.get_sampling_interval()

    pstart = int( round(start/dt) )
    pdelta = int( round(delta/dt) )

    # creates a destination python list
    dlist = [ stf.get_trace(i)[pstart:(pstart+pdelta)] for i in sequence ]

    return stf.new_window_list(dlist)

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
            while i<len(selection) and comp(selection[i],threshold):
                i+=1 # skip values until the value is below or above threshold again
        else:
            i+=1

    return EventCounter

