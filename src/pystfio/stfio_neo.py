"""
Adapter to represent stfio recordings as neo objects

Based on exampleio.py and axonio.py from neo.io

08 Feb 2014, C. Schmidt-Hieber, University College London

"""

# needed for python 3 compatibility
from __future__ import absolute_import

import sys

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal

import stfio

# I need to subclass BaseIO
class StimfitIO(BaseIO):
    """
    Class for converting an stfio Recording to a neo object

    Usage:
        >>> import stfio
        >>> import stfio.neo
        >>> neo_obj = stfio.neo.StimfitIO("file.abf")
        or
        >>> stfio_obj = stfio.read("file.abf")
        >>> neo_obj = stfio.neo.StimfitIO(rec=stfio_obj)
    """

    is_readable        = True
    is_writable        = False

    supported_objects  = [ Block, Segment, AnalogSignal ]
    readable_objects   = [ Block ]
    writeable_objects  = [ ]

    has_header         = False
    is_streameable     = False

    read_params        = { Block : [ ] }
    write_params       = None

    name               =  'Stimfit'
    extensions         = [ 'abf','dat','axgx','axgd','cfs' ]

    mode = 'file'

    def __init__(self , filename = None, rec=None) :
        """
        Arguments:
            filename : the filename
            rec : an stfio recording
        Note:
            Either the filename or an stfio recording can be provided.

        """
        BaseIO.__init__(self)

        self.filename = filename
        self.stfio_rec = rec
            
    def read_block(self, lazy = False, cascade = True ):

        """
        Return an stfio section as a neo Segment.

        self.filename will be ignored.

        Parameters:
            segment_duration :is the size in secend of the segment.
            num_analogsignal : number of AnalogSignal in this segment
            num_spiketrain : number of SpikeTrain in this segment

        """
        if self.filename is not None:
            self.stfio_rec = stfio.read(self.filename)

        bl = Block()
        bl.description = self.stfio_rec.file_description
        bl.annotate(comment = self.stfio_rec.comment)
        try:
            bl.rec_datetime = self.stfio_rec.datetime
        except:
            bl.rec_datetime = None

        if not cascade:
            return bl

        dt = np.round(self.stfio_rec.dt * 1e-3, 9) * pq.s # ms to s
        sampling_rate = 1.0/dt
        t_start = 0 * pq.s

        # iterate over sections first:
        for j in range(len(self.stfio_rec[0])):
            seg = Segment(index = j)
            length = len(self.stfio_rec[0][j])

            # iterate over channels:
            for i in range(len(self.stfio_rec)):
                name = self.stfio_rec[i].name
                unit = self.stfio_rec[i].yunits
                try:
                    pq.Quantity(1, unit)
                except:
                    unit = ''

                if lazy:
                    signal = [ ] * pq.Quantity(1, unit)
                else:
                    signal = np.array(self.stfio_rec[i][j]) * pq.Quantity(1, unit)
                anaSig = AnalogSignal(signal, sampling_rate=sampling_rate,
                                      t_start=t_start, name=str(name),
                                      channel_index=i)
                if lazy:
                    anaSig.lazy_shape = length
                seg.analogsignals.append( anaSig )

            bl.segments.append(seg)
            t_start += length * dt

        bl.create_many_to_one_relationship()

        return bl

def neo2stfio(neo_obj):
    """Convert neo object to stfio recording.
       Restrictions:
       * Only converts the first block
       * Assumes that the sampling rate is constant throughout segments
       * Assumes that we have the same number of channels throughout segments
       * Assumes that the channel units do not change

       Usage:
       >>> import neo
       >>> neo_obj = neo.io.AxonIO("filename.abf")
       >>> import stfio.neo
       >>> stfio_obj = stfio.neo.neo2stfio(neo_obj)
       >>> assert(stfio_obj[0][0][0] == neo_obj.read()[0].segments[0].analogsignals[0][0])
    """
    
    blocks = neo_obj.read()
    if len(blocks) > 1:
        sys.stderr.write("Warning: Only the first block" +  
                         "of this neo object will be converted\n")

    reference_signal = blocks[0].segments[0].analogsignals

    nchannels = len(reference_signal)

    rec = stfio.Recording([
        stfio.Channel([
            stfio.Section(np.array(seg.analogsignals[nc], dtype=np.float64))
            for seg in blocks[0].segments
        ], reference_signal[nc].units.dimensionality.string)
        for nc in range(nchannels)
    ])
    rec.dt = float(reference_signal[0].sampling_period.rescale('ms'))
    rec.xunits = "ms"

    return rec
