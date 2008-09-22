"""Performs fits as decribed in the manual to create 
preliminary and final templates from minis.dat.
last revision: May 09, 2008
C. Schmidt-Hieber
"""

import stf
def preliminary():
    """Creates a preliminary template"""
    stf.set_peak_start(209900)
    stf.set_peak_end(210500)
    stf.set_fit_start(209900)
    stf.set_fit_end(210400)
    stf.set_peak_mean(3)
    stf.set_base_start(209600)
    stf.set_base_end(209900)
    stf.measure()
    return stf.leastsq(5)

def final():
    """Creates a final template"""
    stf.set_peak_start(100)
    stf.set_peak_end(599)
    stf.set_fit_start(100)
    stf.set_fit_end(599)
    stf.set_peak_mean(3)
    stf.set_base_start(0)
    stf.set_base_end(100)
    stf.measure()
    return stf.leastsq(5)

def batch_cursors():
    """Sets appropriate cursor positions for analysing
    the extracted events."""
    stf.set_peak_start(100)
    stf.set_peak_end(598)
    stf.set_fit_start(120)
    stf.set_fit_end(598)
    stf.set_peak_mean(3)
    stf.set_base_start(0)
    stf.set_base_end(100)
    stf.measure()

