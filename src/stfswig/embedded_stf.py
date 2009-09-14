#===========================================================================
# embedded_stf.py
# 2009.09.14
# Don't modify this file unless you know what you are doing!!!
#===========================================================================

"""
embedded_stf.py
starting code to embed wxPython into the stf application.

"""

import wx
from wx.py import shell

from numpy.version import version as numpy_version
# by default the stf module is loaded, so we can access stf functions
import stf

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

        version_s = "NumPy %s, wxPython %s" % (numpy_version, wx.version()) 
        intro = '%s, using %s' % (stf.get_versionstring(), version_s)

        # the Pycrust shell object
        pycrust = shell.Shell(self, -1, introText = intro + LOADED)
        pycrust.push("import numpy as N", silent = True)
        pycrust.push("import stf", silent = True)
        pycrust.push("from stf import *", silent = True)
        pycrust.push("try:", silent = True)
        pycrust.push("    from stf_init import *", silent = True)
        pycrust.push("except ImportError:", silent = True)
        pycrust.push("    pass", silent = True)
        pycrust.push("except SyntaxError:", silent = True)
        pycrust.push("    pass", silent = True)
        pycrust.push("else:", silent = True)
        pycrust.push("    pass", silent = True)
        pycrust.push("", silent = True)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(pycrust, 1, wx.EXPAND | wx.BOTTOM | wx.LEFT | wx.RIGHT, 10)
        self.SetSizer(sizer)

