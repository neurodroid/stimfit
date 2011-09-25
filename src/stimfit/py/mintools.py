"""
2008-04-11, C. Schmidt-Hieber
Some helper functions for least-squares mimization using SciPy
"""

from scipy.optimize import leastsq
import numpy as np
import stf

def fexpbde(p,x):
    tpeak = p[3]*p[1]*np.log(p[3]/p[1])/(p[3]-p[1])
    adjust = 1.0/((1.0-np.exp(-tpeak/p[3]))-(1.0-np.exp(-tpeak/p[1])));
    e1=np.exp((p[0]-x)/p[1]);
    e2=np.exp((p[0]-x)/p[3]);
    
    # normalize the amplitude so that the peak really is the peak:
    ret = adjust*p[2]*e1 - adjust*p[2]*e2 + stf.get_base();
    start_index = 0
    for elem in x:
        if ( elem < p[0] ):
            start_index = start_index+1
        else:
            break
    ret[ 0 : start_index ] = stf.get_base()
    return ret

def leastsq_stf(p,y,lsfunc,x):
    return y - lsfunc(p,x)

def stf_fit( p0, lsfunc ):
    data = stf.get_trace()[ stf.get_fit_start() : stf.get_fit_end() ]
    dt = stf.get_sampling_interval()
    x = np.arange(0, len(data)*dt, dt)

    plsq = leastsq(leastsq_stf, p0, args=(data, lsfunc, x))
    
    return plsq[0]
    
    
