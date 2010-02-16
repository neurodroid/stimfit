#===========================================================================
# embedded_ipython.py
# 2009.09.14
# Don't modify this file unless you know what you are doing!!!
#===========================================================================

"""
embedded_ipython.py
starting code to embed wxPython into the stf application.

"""

import wx
from IPython.frontend.wx.wx_frontend import WxController

import IPython.ipapi

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

        # the shell object
        ipython_shell = WxController(self)

        # the ip object  will access the IPython functionality
        ip =  IPython.ipapi.get()

        # Stimfit and NumPy are visible to the interactive sesion.
        ip.ex('from embedded_init import *')

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(ipython_shell, 1, wx.EXPAND | wx.BOTTOM | wx.LEFT | wx.RIGHT, 10)
        self.SetSizer(sizer)



