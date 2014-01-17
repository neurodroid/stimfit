#===========================================================================
# embedded_mpl.py
# 2011.02.05
# Don't modify this file unless you know what you are doing!!!
#===========================================================================

"""
embedded_mpl.py
starting code to embed a matplotlib wx figure into the stf application.

"""

import sys
if 'linux' in sys.platform:
    import wxversion
    try:
        wxversion.select('2.8')
    except:
        pass
import wx
import matplotlib
if sys.version_info[0] < 3:
    matplotlib.use('WXAgg')
    from matplotlib.backends.backend_wxagg import \
        FigureCanvasWxAgg as FigCanvas, \
        NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.mlab as mlab
import numpy as np

import stfio_plot

class MplPanel(wx.Panel):
    """The matplotlib figure"""
    def __init__(self, parent, figsize=(8.0, 6.0)):
        super(MplPanel, self).__init__(parent, -1)
        self.fig = Figure(figsize, dpi=72)
        self.canvas = FigCanvas(self, -1, self.fig)

        # Since we have only one plot, we can use add_axes 
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.axes = self.fig.add_subplot(111)
        
        # Create the navigation toolbar, tied to the canvas
        #
        self.toolbar = NavigationToolbar(self.canvas)
        
        #
        # Layout with box sizers
        #
        
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, wx.EXPAND | wx.BOTTOM | wx.LEFT | wx.RIGHT, 10)
        self.vbox.Add(self.toolbar, 0, wx.EXPAND)
        
        self.SetSizer(self.vbox)


    def plot_screen(self):
        import stf
        
        tsl = []
        try:
            l = stf.get_selected_indices()
            for idx in l:
                tsl.append(stfio_plot.Timeseries(stf.get_trace(idx), 
                                                 yunits = stf.get_yunits(),
                                                 dt = stf.get_sampling_interval(),
                                                 color='0.2'))
                fit = stf.get_fit(idx)
                if fit is not None:
                    self.axes.plot(fit[0], fit[1], color='0.4', alpha=0.5, lw=5.0)
        except:
            pass
        
        tsl.append(stfio_plot.Timeseries(stf.get_trace(), 
                                         yunits = stf.get_yunits(),
                                         dt = stf.get_sampling_interval()))
        if stf.get_size_recording()>1:
            tsl2 = [stfio_plot.Timeseries(stf.get_trace(trace=-1, channel=stf.get_channel_index(False)), 
                                          yunits = stf.get_yunits(trace=-1, channel=stf.get_channel_index(False)),
                                          dt = stf.get_sampling_interval(),
                                          color='r', linestyle='-r')]
            stfio_plot.plot_traces(tsl, traces2=tsl2, ax=self.axes, textcolor2 = 'r',
                                   xmin=stf.plot_xmin(), xmax=stf.plot_xmax(),
                                   ymin=stf.plot_ymin(), ymax=stf.plot_ymax(), 
                                   y2min=stf.plot_y2min(), y2max=stf.plot_y2max())
        else:
            stfio_plot.plot_traces(tsl, ax=self.axes,
                                   xmin=stf.plot_xmin(), xmax=stf.plot_xmax(),
                                   ymin=stf.plot_ymin(), ymax=stf.plot_ymax())
        fit = stf.get_fit()
        if fit is not None:
            self.axes.plot(fit[0], fit[1], color='0.2', alpha=0.5, lw=5.0)

    def plot_spectrum(self):
        import stf
        Pow, freq = mlab.psd(stf.get_trace(), 
                             Fs=(1.0/stf.get_sampling_interval())*1e3,
                             detrend=mlab.detrend_linear)
        self.axes.plot(freq, 10*np.log10(Pow))
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("Power spectral density (dB/Hz)")
