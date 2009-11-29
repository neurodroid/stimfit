"""
AP.py
Routines to evaluate basic action potential properties

Mon Nov 16 23:51:03 CET 2009

"""
import wx
import wx.grid


import stf
import numpy as np

from math import ceil, floor

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
        self.update()
    
    def update(self):
        """ update current trace sampling rate, 
        cursors position and  measurementsi (peak, baseline & AP kinetics)
        according to the threshold value set at construction.
        """
        # set slope
        stf.set_slope(self._thr) # on stf v0.93 or above

        # update sampling rate
        self._dt = stf.get_sampling_interval() 

        # update cursors and AP kinetics (peak and half-width)
        stf.measure() 

    def get_base(self):
        """
        Get baseline according to cursor possition in the 
        given current channel/trace

        """

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

    def get_threshold(self):
        """ return the threshold value (in mV/ms) set at construction"""
        return self._thr

    def get_threshold_value(self):
        """ return the value (in y-units) at the threshold """
        self.update() # stf.get_threshold_value does not update
        return stf.get_threshold_value()

    def get_threshold_time(self):
        """ return the value (in x-units) at the threshold """
        self.update()
        return stf.get_threshold_time('True')

    def set_threshold(self, thr):
        """ set a new threshold value 
        Arguments:
        thr     -- the new threshold (mV/ms)
        """

        self._th = thr
        self.update()


class APTrace(Spike):
    def __init__(self, threshold):
        """ creates a static instance of the Spike class. That is
        a "screenshot" of the current AP kinetics for the current trace.
        There should be another way to do that """

        Spike.__init__(self, threshold)
        self.threshold = threshold
        self.threshold_time = self.get_threshold_time()
        self.threshold_value =  self.get_threshold_value()
        self.base = self.get_base()
        self.peak = self.get_peak()
        self.tamplitude = self.get_tamplitude()
        self.t50 = self.get_t50()
        self.max_rise = self.get_max_rise()

    def list(self):
        """ Returns a list with the current parameters of the 
        Spike
        """
        list = []
        list.append(self.threshold)
        list.append(self.threshold_time)
        list.append(self.threshold_value)
        list.append(self.base)
        list.append(self.peak)
        list.append(self.tamplitude)
        list.append(self.t50)
        list.append(self.max_rise)

        return list

class APFrame(wx.Frame, APTrace):

    def __init__(self, th_soma, th_dend):
        """ creates a grid and fill it with AP kinetics from
        soma and dendrites 
        Arguments:
        th_soma     -- AP threshold for the soma
        th_dend     -- AP threshold for the dendrite
        """

        wx.Frame.__init__(self, None, title = "AP parameters (from threshold)", size = (740,135)) 

        
        self.__th_soma = th_soma
        self.__th_dend = th_dend

        self.col = ["Threshold\n (mV/ms)", "Onset\n (ms)", "Onset\n (mV)", "Baseline\n (mV)", "AP Peak\n (mV)", "AP Peak\n (ms)", "Half-width\n (ms)","Vmax\n (mV/ms)"]
        self.row = ["Soma [%d]"%(stf.get_channel_index()), "Dend [%d]"%(self.inactive_channel()), "latency"]

        # Grid
        grid = wx.grid.Grid(self)
        grid.CreateGrid(len(self.row), len(self.col))

        # Set grid labels
        for i in range(len(self.col)):
            grid.SetColLabelValue(i, self.col[i])

        for i in range(len(self.row)):
            grid.SetRowLabelValue(i, self.row[i])
        

        # Calculate the values for the somatic AP
        soma = APTrace(th_soma).list()
        
        # Fill soma values in the grid
        for i in range(len(self.col)):
            grid.SetCellValue(0,i, "%.4f"%soma[i])

        # Swap channel
        self.swap_channels()

        # Calculate the values for the dendritic AP
        dend = APTrace(th_dend).list()

        # Fill dend values in the grid 
        
        for i in range(len(self.col)):
            grid.SetCellValue(1,i, "%.4f"%dend[i])

        # Calculate latencies with different methods
        grid.SetCellValue(2,1, "%.4f"%(dend[1]-soma[1]))
        grid.SetCellValue(2,5, "%.4f"%(dend[5]-soma[5]))
        grid.SetCellValue(2,6, "%.4f"%self.latency())

        self.swap_channels()  # go back to the origin

    def inactive_channel(self):
        """ Returns the number of the inactive channel 
        (i.e returns 0 if the active channel is 1, and 1 if the active 
        channel is 0 )
        """

        return not (stf.get_channel_index())

    def swap_channels(self):
        """ Swap channels (i.e set the inactive channel to active) 
        1 will be 0 and 0 will be 1). False upon failure"""     

        if stf.get_channel_index():
            return stf.set_channel(0) 
        else:
            return stf.set_channel(1) 

    def latency(self):
        """ Calculate the latency according to Schmidt-Hieber, 2008.
        (i.e time difference between the half-maximal amplitudes). 
        In our case we have to set a different threshold for dendrites 
        and soma. Negative values indicate dendritic APs precede 
        somatic APs. Possitive values indicate somatic AP precede dendritic
        AP.
        """

        # time at the half-width in the dendrite
        stf.set_slope(self.__th_dend)
        t50_dend = stf.t50left_index()*stf.get_sampling_interval()
        
        self.swap_channels()

        # time at the half-width in the dendrite
        stf.set_slope(self.__th_soma)
        t50_soma= stf.t50left_index()*stf.get_sampling_interval()

        self.swap_channels()

        return (t50_dend - t50_soma)

def calc(th_soma, th_dend):
    frame = APFrame(th_soma, th_dend)
    frame.Show()
