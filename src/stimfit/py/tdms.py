import stf
import stfio

def tdms_open(fn):
    rec = stfio.read_tdms(fn)
    if rec is None:
        return None

    li = [[sec.asarray() for sec in chan] for chan in rec]

    return li, rec.dt
