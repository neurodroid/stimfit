#===========================================================================
# embedded_mpl.py
# 2011.02.05
# Don't modify this file unless you know what you are doing!!!
#===========================================================================

"""
embedded_mpl.py
starting code to embed a matplotlib wx figure into the stf application.

"""

import wxversion
wxversion.select('2.8')
import wx
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import \
    FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar

import stfio_plot

class MplPanel(wx.Panel):
    """The matplotlib figure"""
    def __init__(self, parent):
        super(MplPanel, self).__init__(parent, -1)
        self.fig = Figure((8.0, 6.0), dpi=96)
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
                tsl.append(stfio_plot.timeseries(stf.get_trace(idx), 
                                                 yunits = stf.get_yunits(),
                                                 dt = stf.get_sampling_interval(),
                                                 color='0.2'))
        except:
            pass
        
        tsl.append(stfio_plot.timeseries(stf.get_trace(), 
                                         yunits = stf.get_yunits(),
                                         dt = stf.get_sampling_interval()))

        stfio_plot.plot_traces(tsl, self.axes, xmin=stf.plot_xmin(), xmax=stf.plot_xmax(),
                               ymin=stf.plot_ymin(), ymax=stf.plot_ymax())

    def get_axes(self):
