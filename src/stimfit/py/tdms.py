import numpy as np
import stf
import stfio

def tdms_open(fn):
    record = stfio.read_tdms(fn)
    if record is None:
        return None

    return record['data'], record['dt']
