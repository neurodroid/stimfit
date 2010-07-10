#=========================================================================
# embedded_init.py
# 2009.12.31
# 
# This file loads both Numpy and stf modules into the current namespace.
# Additionally, it loads the custom initialization script (stf_init.py)
#
# 2010.06.12
# Major stf classes were added (Recording, Channel, Section)
#
# It is used by embedded_stf.py and embedded_ipython.py 
# Please, do not modify this file unless you know what you are doing
#
#=========================================================================

import numpy as N 
import stf
from stf import *

from os.path import basename

try:
    from stf_init import *
except ImportError:
    # let the user know  stf_init does not work!
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


    version_s = 'NumPy %s, wxPython %s' % (numpy_version, wx_version()) 
    intro = '%s, using %s' % (stf.get_versionstring(), version_s)

    return intro

class Recording(object):
    def __init__(self):
        # TODO
        # 1.- comments, date and time should be updated in a next release
        # 2.- a method to export in hdf5 format should be available
        # self._filename is a state variable and its the real filename
        # of the recording object
        self._filename = basename(stf.get_filename())

    def __set_filename(self, filename):
        """ updates self._filename with the value in filename """
        self._filename = filename

    def show_filename(self):
        """ Returns the filename in the current trace"""
        return basename(stf.get_filename())

    def __repr__(self):
        """ prints current information of the Recording object """
        return "Recording filename: %s"%self._filename 

    # getter and setter
    filename = property(lambda self: self._filename, __set_filename)

class Channel(Recording):
    def __init__(self, channel = None):
        # self._channel is state variable for Channel
        self._channel = ()

        if channel is None:
            # current channel 
            self._channel = (stf.get_channel_index(), \
                stf.get_channel_name())
        else:
            # TODO: catch the exception!!!
            self._channel = (channel, stf.get_channel_name(channel))

        # a dict with channel names
        # Do we need this?
        self._chdict = dict()
        for i in range(stf.get_size_recording()):
            self.__setitem__(i, stf.get_channel_name(i))
            
        # inheritance to handle filename
        super(Channel, self).__init__()
        
    def __len__(self):
        """ returns number of channels in the file """
        return  stf.get_size_recording()

    def __setitem__(self, key, item):
        """ sets name of the channel i in the dictionary """
        self._chdict[key] = item 

    def show_channel(self):
        """ returns a tuple with the current channel index and name"""
        index = stf.get_channel_index()
        name = stf.get_channel_name()
        return (index, name)

    def __repr__(self):
        """ prints current information of the Section object """
        index, name = self._channel
        return "Channel[%d]-%s in %s"%(index, name, self._filename)

    # getter for channel
    channel = property(lambda self: self._channel)

# Follow Liskov substitution principle: 
# 'What work for Recording class should also work for the Section class'
class Section(Channel):
    """ This class correspond to a single section 
    (a sweep or trace) in Stimfit. Note that Section must
    know to which channel and recording belongs! """
    def __init__(self, trace = None, channel=None):
        if trace is None:
            # state variable with current trace
            self._trace = stf.get_trace_index()
        else:
            self._trace = trace

        # double inheritance from Channel and Recording
        # state variables self._filename and self._channel
        # are inherited!!!
        super(Section, self).__init__()

        self.__update(trace)

    def __update(self, trace = None):
        """ updates from Section trace, array, xunits, yunits and dt 
        from Channel
        from Recording filename
        but ONLY if self._trace is different from current trace"""
        
        #if we change the Recording
        if self.filename != self.show_filename(): 
            # update filename
            self.filename = self.show_filename()
        
        if trace is None:
            # take the current trace/channel
            self._trace = stf.get_trace_index()
            self._array = stf.get_trace()
            self._xunits = stf.get_xunits()
            self._yunits = stf.get_yunits()

        else:
            self._trace = trace
            self._array = stf.get_trace(trace = trace)
            self._xunits = stf.get_xunits(trace = trace)
            self._yunits = stf.get_yunits(trace = trace)

        # dt does not change between traces
        self._dt = stf.get_sampling_interval()

    def __len__(self):
        """ returns the number of samples in the current trace """
        # we should consider to remove get_size_trace()
        self._update()
        return len(self._array)

    def __getitem__(self, i):
        """ get the value in position i of the array """
        self._update()
        return self._array[i]

    def __repr__(self):
        """ prints current information of the Section object """
        index, name = self.channel
        return "Trace %d in channel[%d]-%s in %s"\
            %((self._trace+1), index, name, self.filename)

    def __set_dt(self, value):
        """ set sampling interval """
        return stf.set_sampling_interval(value)
    
    def get_baseline(self):
        """ returns baseline according to the baseline cursors position """
        self.__update()
        stf.measure()
        return stf.get_base()
        #return self._array[stf.get_base_start():stf.get_base_end()].mean()

    def get_peak(self):
        """ returns the peak amplitude from baseline according to
        the current peak cursors position"""
        # baseline calls self.measure() allready
        baseline = self.get_baseline()

        return baseline-stf.get_peak()

    # getters and setters
    dt = property(lambda self: stf.get_sampling_interval(), __set_dt)

