#=========================================================================
# embedded_init.py
# 2009.12.31
# 
# This file loads both Numpy and stf modules into the current namespace.
# Additionally, it loads the custom initialization script (stf_init.py)
# and the major stf class (Recording, Channel, Section)
#
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

class Section(object):
    """ This class correspond to a single section 
    (a sweep or trace) in Stimfit """
    def __init__(self, section = None):
        self._update(section)

    def _update(self, section=None):
        """ update trace, dt, xunits and yunits """
        if section is None:
            # take the current trace/channel
            self._array = stf.get_trace()
            self._xunits = stf.get_xunits()
            self._yunits = stf.get_yunits()
        else:
            self._array = stf.get_trace(trace = section)
            self._xunits = stf.get_xunits(trace = section)
            self._yunits = stf.get_yunits(trace = section)

        # dt does not change between traces
        self._dt = stf.get_sampling_interval()

    def __getitem__(self, i)
        """ get the value in position i of the array """
        self._update()
        return self._array[i]

    def __len__(self):
        """ returns the number of samples in the current trace """
        # we should consider to remove get_size_trace()
        self._update()
        return len(self._array)

    def _set_dt(self, value):
        """ set sampling interval """
        return stf.set_sampling_interval(value)

    def _get_baseline(self):
        """ returns baseline according to the baseline cursors position """
        self._update()
        return self._array[stf.get_base_start():stf.get_base_end()].mean()

    def _get_peak(self):
        """ returns the peak amplitude from baseline according to
        the current peak cursors positionh"""

        # calculate peak 'a la Stimfit'
        pselection= self._array[stf.get_peak_start():stf.get_peak_end()]
        peak = N.max(pselection)
        pindex = list(pselection).index(peak)        
        # get the number of points for a peak
        npts = stf.get_peak_mean()
        peakval = pselection[pindex-(npts/2):pindex+(npts/2)].mean()

        return self._get_baseline()-peakval

    # getters and setters
    dt = property(lambda self: stf.get_sampling_interval(), _set_dt)
    baseline = property(_get_base)
    peak = property(_get_peak)


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


