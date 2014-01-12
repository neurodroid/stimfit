#===========================================================================
# embedded_stf.py
# 2009.09.14
# Don't modify this file unless you know what you are doing!!!
#===========================================================================

"""
embedded_stf.py
starting code to embed wxPython into the stf application.

"""
import sys
if 'win' in sys.platform:
    import wxversion
    wxversion.select('3.0-msw')
import wx
from wx.py import shell

# to access the current versions of Stimfit, NumPy and wxPython
from embedded_init import intro_msg

# test if stf_init was loaded
try:
    import stf_init
except ImportError:
    LOADED = " "
except SyntaxError:
    LOADED = " Syntax error in custom initialization script stf_init.py"
else:
    LOADED = " Successfully loaded custom initializaton script stf_init.py"

class MyPanel(wx.Panel):
    """ The wxPython shell application """
    def __init__(self, parent):
        # super makes the same as wx.Panel.__init__(self, parent, etc..)
        # but prepares for Python 3.0 among other things...
        super(MyPanel, self).__init__(parent, -1, \
            style = wx.BORDER_NONE | wx.MAXIMIZE)

        # the Pycrust shell object
        pycrust = shell.Shell(self,-1, \
            introText = intro_msg() + LOADED)

        # Workaround for http://trac.wxwidgets.org/ticket/15008
        if "darwin" in sys.platform:
            pycrust.autoCallTip = False

        pycrust.push('from embedded_init import *', silent = True)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(pycrust, 1, wx.EXPAND | wx.BOTTOM | wx.LEFT | wx.RIGHT, 10)
        self.SetSizer(sizer)

