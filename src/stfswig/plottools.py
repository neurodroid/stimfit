"""
A demonstration of creating a matlibplot window from within wx.
A resize only causes a single redraw of the panel.
The WXAgg backend is used as it is quicker.

Edward Abraham, Datamine, April, 2006
(works with wxPython 2.6.1, Matplotlib 0.87 and Python 2.4)
"""

import matplotlib
matplotlib.interactive(True)
#Use the WxAgg back end. The Wx one takes too long to render
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
#used in the particular example
from matplotlib.numerix import arange, sin, cos, pi
import wx

class NoRepaintCanvas(FigureCanvasWxAgg):
    """We subclass FigureCanvasWxAgg, overriding the _onPaint method, so that
    the draw method is only called for the first two paint events. After that,
    the canvas will only be redrawn when it is resized.
    """
    def __init__(self, *args, **kwargs):
        FigureCanvasWxAgg.__init__(self, *args, **kwargs)
        self._drawn = 0

    def _onPaint(self, evt):
        """
        Called when wxPaintEvt is generated
        """
        if not self._isRealized:
            self.realize()
        if self._drawn < 2:
            self.draw(repaint = False)
            self._drawn += 1
        self.gui_repaint(drawDC=wx.PaintDC(self))

class PlotPanel(wx.Panel):
    """
    The PlotPanel has a Figure and a Canvas. OnSize events simply set a 
    flag, and the actually redrawing of the
    figure is triggered by an Idle event.
    """
    def __init__(self, parent, id = -1, color = None,\
        dpi = None, style = wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        wx.Panel.__init__(self, parent, id = id, style = style, **kwargs)
        self.figure = Figure(None, dpi)
        self.canvas = NoRepaintCanvas(self, -1, self.figure)
        self.SetColor(color)
        self.Bind(wx.EVT_IDLE, self._onIdle)
        self.Bind(wx.EVT_SIZE, self._onSize)
        self._resizeflag = True
        self._SetSize()

    def SetColor(self, rgbtuple):
        """Set figure and canvas colours to be the same"""
        if not rgbtuple:
            rgbtuple = wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNFACE).Get()
        col = [c/255.0 for c in rgbtuple]
        self.figure.set_facecolor(col)
        self.figure.set_edgecolor(col)
        self.canvas.SetBackgroundColour(wx.Colour(*rgbtuple))

    def _onSize(self, event):
        self._resizeflag = True

    def _onIdle(self, evt):
        if self._resizeflag:
            self._resizeflag = False
            self._SetSize()
            self.draw()

    def _SetSize(self, pixels = None):
        """
        This method can be called to force the Plot to be a desired size, which defaults to
        the ClientSize of the panel
        """
        if not pixels:
            pixels = self.GetClientSize()
        self.canvas.SetSize(pixels)
        self.figure.set_figsize_inches(pixels[0]/self.figure.get_dpi(),
        pixels[1]/self.figure.get_dpi())

    def draw(self):
        """Where the actual drawing happens"""
        pass

def demo():
    class DemoPlotPanel(PlotPanel):
        """An example plotting panel. The only method that needs 
        overriding is the draw method"""
        def draw(self):
            if not hasattr(self, 'subplot'):
                self.subplot = self.figure.add_subplot(111)
            theta = arange(0, 45*2*pi, 0.02)
            rad = (0.8*theta/(2*pi)+1)
            r = rad*(8 + sin(theta*7+rad/1.8))
            x = r*cos(theta)
            y = r*sin(theta)
            #Now draw it
            self.subplot.plot(x,y, '-r')
            #Set some plot attributes
            # self.subplot.set_title("A polar flower (%s points)"%len(x), fontsize = 12)
            self.subplot.set_title(r'$\alpha_i > \beta_i$')
            self.subplot.set_xlabel("Flower is from  http://www.physics.emory.edu/~weeks/ideas/rose.html", fontsize = 8)
            self.subplot.set_xlim([-400, 400])
            self.subplot.set_ylim([-400, 400])


    #Initialise a frame ...
    frame = wx.Frame(None, -1, 'WxPython and Matplotlib')
    #Make a child plot panel...
    panel = DemoPlotPanel(frame)

    #Put it in a sizer ...   
    sizer = wx.BoxSizer(wx.HORIZONTAL)
    panel.SetSizer(sizer)
    sizer.SetItemMinSize(panel, 300, 300)
    panel.Fit()
    panel._SetSize()
    #And we are done ...    
    frame.Show()
