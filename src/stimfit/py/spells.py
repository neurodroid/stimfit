"""
spells.py

Python recipes to solve frequently requested tasks with Stimfit.
You can find a complete description of these functions in the 
Stimfit online documentation (http://www.stimfit.org/doc/sphinx/index.html)
Check "The Stimfit Book of Spells" for details.
"""


import numpy as np

# stimfit python module:
import stf

import wx # see APFrame class
import wx.grid # see APFrame class

from math import ceil, floor

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
        print('Couldn\'t find an open file; aborting now.')
        return 0

    #A temporary array to calculate the average:
    array = np.empty( (stf.get_size_channel(), stf.get_size_trace()) )
    for n in range( 0,  stf.get_size_channel() ):
        # Add this trace to set:
        array[n] = stf.get_trace( n )


    # calculate average and create a new section from it:
    stf.new_window( np.average(set, 0) )

    # set peak cursors:
    # -1 means all points within peak window.
    if not stf.set_peak_mean(-1): 
        return 0 
    if not stf.set_peak_start(peak_start): 
        return 0
    if not stf.set_peak_end(peak_end): 
        return 0

    # set base cursors:
    if not stf.set_base_start(base_start): 
        return 0
    if not stf.set_base_end(base_end): 
        return 0

    # measure everything:
    stf.measure()

    # calculate r_seal and return:
    return amplitude / (stf.get_peak()-stf.get_base())

def rmean(binwidth, trace=-1, channel=-1):
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
    sweep = stf.get_trace(trace, channel)

    # creates a destination python list to append the data
    dsweep = np.empty((len(sweep)))

    # running mean algorithm
    for i in range(len(sweep)):

        if (len(sweep)-i) > binwidth:
            # append to list the running mean of `binwidth` values
            # np.mean(sweep) calculates the mean of list
            dsweep[i] = np.mean( sweep[i:(binwidth+i)] )

        else:
            # use all remaining points for the average:
            dsweep[i] = np.mean( sweep[i:] )


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
    get_amplitude(980,1005,10,i) returns the variation of the Y unit of the
        trace i between
    peak value (10050+10) msec and baseline (980+10) msec
    """

    # sets the current trace or the one given in trace
    if trace is None:
        sweep = stf.get_trace_index()
    else:
        if type(trace) != int:
            print('trace argument admits only intergers')
            return False
        sweep = trace


    # set base cursors:
    if not(stf.set_base_start(base, True)): 
        return False # out-of range
    if not(stf.set_base_end(base+delta, True)): 
        return False

    # set peak cursors:
    if not(stf.set_peak_start(peak, True)): 
        return False # out-of range
    if not(stf.set_peak_end(peak+delta, True)): 
        return False

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
    cut_sweeps(200,300) cut the traces between t=200 ms and t=500 ms 
        within the whole channel.
    cut_sweeps(200,300,range(30,60)) the same as above, but only between 
        traces 30 and 60.
    cut_sweeps(200,300,stf.get_selected_indices()) cut between 200 ms               and 500 ms only in the selected traces.

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
    up          -- (optional) True (default) will look for upward events,
                    False downwards.
    trace       -- (optional) zero-based index of the trace in the current 
                    channel, if None, the current trace is selected.
    mark        -- (optional) if True (default), set a mark at the point 
                    of threshold crossing
    Returns:
    An integer with the number of events.

    Examples:
    count_events(500,1000) returns the number of events found between t=500
         ms and t=1500 ms above 0 in the current trace and shows a stf 
         marker.
    count_events(500,1000,0,False,-10,i) returns the number of events found
         below -10 in the trace i and shows the corresponding stf markers.
    """

    # sets the current trace or the one given in trace.
    if trace is None:
        sweep = stf.get_trace_index()
    else:
        if type(trace) !=int:
            print('trace argument admits only integers')
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
    event_counter, i = 0, 0 # set counter and index to zero

    # choose comparator according to direction:
    if up:
        comp = lambda a, b: a > b
    else:
        comp = lambda a, b: a < b

    # run the loop
    while i < len(selection):
        if comp(selection[i], threshold):
            event_counter += 1
            if mark:
                stf.set_marker(pstart+i, selection[i])
            while i < len(selection) and comp(selection[i], threshold):
                i += 1 # skip  until value is below/above threshold
        else:
            i += 1

    return event_counter

class Spike(object):
    """ 
    A collection of methods to calculate AP properties
    from threshold (see Stuart et al., 1997). Note that all
    calculations are performed in the active/current channel!!!
    """

    def __init__(self,threshold):
        """ 
        Create a Spike instance with sampling rate and threshold 
        measurements are performed in the current/active channel!!!

        Arguments:
        threshold   -- slope threshold to measure AP kinetics 
        """

        self._thr = threshold
        # set all the necessary AP parameters at construction
        self._updateattributes()

    def _updateattributes(self):
        """
        update base, peak, t50, max_rise and tamplitude 
        """

        self.base = self.get_base() # in Stimfit is baseline
        self.peak = self.get_peak() # in Stimfit peak (from threshold)
        self.t50 = self.get_t50()   # in Stimfit t50
        self.max_rise = self.get_max_rise() # in Stimfit Slope (rise)
        self.thr = self.get_threshold_value() # in Stimit Threshold

        # attributes necessary to calculate latencies
        self.tonset = self.get_threshold_time()
        self.tpeak = self.get_tamplitude()
        self.t50_left = self.get_t50left()
    
    def update(self):
        """ update current trace sampling rate, 
        cursors position and  measurements (peak, baseline & AP kinetics)
        according to the threshold value set at construction or when
        the object is called with a threshold argument.
        """
        # set slope
        stf.set_slope(self._thr) # on stf v0.93 or above

        # update sampling rate
        self._dt = stf.get_sampling_interval() 

        # update cursors and AP kinetics (peak and half-width)
        stf.measure() 
    
    def __call__(self, threshold=None ):
        """ update AP kinetic parameters to a new threshold in the 
        current trace/channel
        threshold (optional)   -- the new threshold value

        Examples :
        dend = Spike(40) # set the spike threshold at 40mV/ms
        dend(20) # now we set the spike threshold at 20mV/ms 

        The AP parameters will be thereby updated in the current 
        trace/channel. This method allow us to use the same object 
        to calculate AP latencies in different traces.
        """
       
        if threshold is not None:
           self._thr = threshold # set a new threshold

        self.update() # update dt and sampling rate
        self._updateattributes()


    def get_base(self):
        """
        Get baseline according to cursor possition in the 
        given current channel/trace

        """

        self.update()

        return stf.get_trace(trace = -1 ,channel = -1)[stf.get_base_start():stf.get_base_end()+1].mean()

    def get_peak(self):
        """ 
        calculate peak measured from threshold in the current trace, 
        (see Stuart et al (1997)
        """

        stf.set_peak_mean(1) # a single point for the peak value
        stf.set_peak_direction("up") # peak direction up

        self.update()
        
        peak = stf.get_peak()-stf.get_threshold_value()  
        return peak

    def get_t50(self):
        """ calculates the half-widht in ms in the current trace"""

        self.update()

        # current t50's difference to calculate half-width (t50)

        return (stf.t50right_index()-stf.t50left_index())*self._dt

    def get_max_rise(self):
        """ 
        maximum rate of rise (dV/dt) of AP in the current trace, 
        which depends on the available Na+ conductance, 
        see Mainen et al, 1995, Schmidt-Hieber et al, 2008 
        """

        self.update()
        pmaxrise = stf.maxrise_index() # in active channel

        trace = stf.get_trace(trace = -1, channel =-1) # current trace

        dV = trace[int(ceil(pmaxrise))]-trace[(int(floor(pmaxrise)))]

        return dV/self._dt

    def get_tamplitude(self):
        """ return the time a the peak in the current trace"""

        #stf.peak_index() does not update cursors!!!
        self.update()

        return stf.peak_index()*self._dt

    def get_t50left(self):
        """ return the time at the half-width """
        self.update()

        return stf.t50left_index()*self._dt

    def show_threshold(self):
        """ return the threshold value (in mV/ms) set at construction
        or when the object was called"""
        return self._thr

    def get_threshold_value(self):
        """ return the value (in y-units) at the threshold """
        self.update() # stf.get_threshold_value does not update
        return stf.get_threshold_value()

    def get_threshold_time(self):
        """ return the value (in x-units) at the threshold """
        self.update()
        return stf.get_threshold_time('True')


# TODO
# how to get the name of the object as string
# Latencies according to Schmidt-Hieber need revision!!!
class APFrame(wx.Frame):

    def __init__(self, soma, dend):
        """ creates a grid and fill it with AP kinetics from
        soma and dendrites 

        Arguments:
        soma_AP   -- Spike object for the soma
        dend_AP   -- Spike object for the dendrite
        see Spike() class for more details
        """
        # first check that both soma and dend are Spike() instances
        if not isinstance(soma,Spike) or not isinstance(dend,Spike):
            print('wrong argument, did you create a Spike object???')
            return 0

        # initialize the wxframe
        wx.Frame.__init__(self, None, \
            title = "AP parameters (from threshold)", size = (740,135)) 

        # wxgrid columns
        self.col = ["Threshold\n (mV/ms)", "Onset\n (ms)", "Onset\n (mV)", "Baseline\n (mV)", "AP Peak\n (mV)", "AP Peak\n (ms)", "Half-width\n (ms)","Vmax\n (mV/ms)"]

        # wxgrid rows
        self.row = ["Soma", "Dend", "latency"]
        
        # Grid
        grid = wx.grid.Grid(self)
        grid.CreateGrid(len(self.row), len(self.col))

        # Set grid labels
        for i in range(len(self.col)):
            grid.SetColLabelValue(i, self.col[i])

        for i in range(len(self.row)):
            grid.SetRowLabelValue(i, self.row[i])
        
        # Create a list with the AP parameters for the dendrite
        somalist =  [soma.show_threshold(), soma.tonset, soma.thr, soma.base, soma.peak, soma.tpeak, soma.t50, soma.max_rise]
               
        # Fill soma values in the grid
        for i in range(len(self.col)):
            grid.SetCellValue(0,i, "%.4f"%somalist[i])

        # Create a list with the AP parameters for the dendrite
        dendlist =  [dend.show_threshold(), dend.tonset, dend.thr, dend.base, dend.peak, dend.tpeak, dend.t50, dend.max_rise]

        # Fill dend values in the grid 
        
        for i in range(len(self.col)):
            grid.SetCellValue(1,i, "%.4f"%dendlist[i])

        # Calculate latencies with different methods
        # onset latency
        grid.SetCellValue(2,1, "%.4f"%(dend.tonset - soma.tonset))  
        # peak latency
        grid.SetCellValue(2,5, "%.4f"%(dend.tpeak - soma.tpeak))
        # half-width latency
        grid.SetCellValue(2,6, "%.4f"%(dend.t50_left -soma.t50_left))

def latency(soma, dend):
    """ 
    Shows a results table with the latencies between the 
    somatic and dendritic  object 
    
    Arguments:
    soma    -- Spike Object of a trace containing the somatic AP 
    dend    -- Spike Object of a trace containing the dendritic AP
    see Spike() class for more details
    """
    frame = APFrame(soma, dend)
    frame.Show()

def count_aps():
    """
    Shows a result table with the number of action potentials (i.e
    events whose potential is above 0 mV) in selected traces.  

    Returns:
    False if document is not open or no trace is selected
    """
    if not stf.check_doc():
        print("Open file first")
        return False
   
    sel_trace = stf.get_selected_indices()
    if not sel_trace: 
        print("Select traces first")
        return False

    mytable = dict()
    for trace in sel_trace:
        tstart = 0
        tend = stf.get_size_trace(trace)*stf.get_sampling_interval()
        threshold = 0
        spikes = count_events(tstart, tend, threshold, True, trace, True)
        mytable["Trace %.3d" %trace] = spikes

    stf.show_table(mytable)

    return True
