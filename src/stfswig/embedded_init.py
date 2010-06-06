#=========================================================================
# embedded_init.py
# 2009.12.31
# 
# This file simply loads both Numpy and stf modules into the current 
# namespace.
# Additionally, it loads the custom initialization script (stf_init.py)
# It is used by embedded_stf.py and embedded_ipython.py 
# Please, do not modify this file unless you know what you are doing
#
#=========================================================================

import numpy as N 
import stf
from stf import *

try:
    from stf_init import *
except ImportError:
    pass
except SyntaxError:
    pass
else:
    pass

def intro_msg():
    """ this is the starting message of the embedded Python shell.
    Contains the current Stimfit version, together with the NumPy
    and wxPython version.
    """
    # access current versions of wxWidgets and NumPy
    from wx import version as wx_version
    from numpy.version import version as numpy_version


    version_s = "NumPy %s, wxPython %s" % (numpy_version, wx_version()) 
    intro = '%s, using %s' % (stf.get_versionstring(), version_s)

    return intro

class Recording(object):
    """ General class for recording properties """
    def __init__(self):
        """ parameters set at constructions will not be updated """

        self._rectime = stf.get_recording_time()
        self._recdate = stf.get_recording_date()
        self._update()

    def _update(self):
        """ update Recording attributes """
        self._dt = stf.get_sampling_interval()
        self._size = stf.get_size_recording()

    def _get_dt(self):
        """ get sampling interval """
        self._update()
        return self._dt

    def _set_dt(self, value):
        """ set the sampling interval """
        return stf.set_sampling_interval(value)

    # setter and getter for dt
    dt = property(_get_dt, _set_dt )


