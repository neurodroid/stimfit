"""
Adapter to represent stfio recordings as neo objects

Based on exampleio.py and axonio.py from neo.io

08 Feb 2014, C. Schmidt-Hieber, University College London

"""

# needed for python 3 compatibility
from __future__ import absolute_import

import sys

import numpy as np

import stfio

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
           >>> import stfio
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
