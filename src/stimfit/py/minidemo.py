"""
This script set base, peak and fit cursors to 
perform events detection as decribed in the Stimfit manual [1] 
It ctimeates a preliminary and final templates from a file 'minis.dat.

You can download the file here: http://stimfit.org/tutorial/minis.dat

last revision:  Sep 3 15:45:55 CEST 2018

C. Schmidt-Hieber

[1] https://neurodroid.github.io/stimfit/manual/event_extraction.html

"""
import stf

from wx import MessageBox

if stf.get_filename()[-9:] != 'minis.dat':
    MessageBox('minis.dat is not current file!', 'Warning')

def preliminary():
    """
    Set peak, base and fit cursors around a synaptic event
    and performs a biexponetial fit to create the preliminary template 
    for event detection. 
    """

    stf.base.cursor_index = (209600, 209900)
    stf.peak.cursor_index = (209900, 210500)
    stf.fit.cursor_index  = (209900, 210400)


    stf.measure() # update cursors

    return stf.leastsq(5)

def final():
    """
    Set peak, base and fit cursors around a synaptic event
    and performs a biexponetial fit to create the final template 
    for event detection. 
    """

    stf.base.cursor_index = (000, 100)
    stf.peak.cursor_index = (100, 599)
    stf.fit.cursor_index  = (100, 599)

    stf.set_peak_mean(3)

    stf.measure() # update cursors

    return stf.leastsq(5)

def batch_cursors():
    """
    Set peak, base and fit cursors around a synaptic event
    for the batch analysis of the extracted events.
    """

    stf.base.cursor_index = (000, 100)
    stf.peak.cursor_index = (100, 598)
    stf.fit.cursor_index  = (120, 598)

    stf.set_peak_mean(3)

    stf.measure()

