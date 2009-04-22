"""
Calculates resistances
2008-05-06, C. Schmidt-Hieber
Indices are zero-based!
"""

import numpy as N

# stimfit python module:
import stf

# import iv tools:
import ivtools


def resistance( base_start, base_end, peak_start, peak_end, amplitude ):
    """Calculates the resistance from a series of voltage clamp
    traces. 

    Keyword arguments:
    base_start -- Starting index (zero-based) of the baseline cursors.
    base_end   -- End index (zero-based) of the baseline cursors.
    peak_start -- Starting index (zero-based) of the peak cursors.
    peak_end   -- End index (zero-based) of the peak cursors.
    amplitude  -- Amplitude of the voltage command.
    
    Returns:
    The resistance.
    """
    
    if (stf.check_doc() == False):
        print "Couldn't find an open file; aborting now."
        return 0

    # A temporary array to calculate the average:
    set = N.empty( (stf.get_size_channel(), stf.get_size_trace()) )
    for n in range( 0,  stf.get_size_channel() ):
        # Add this trace to set:
        set[n] = stf.get_trace( n )

    # calculate average and create a new section from it:
    stf.new_window( N.average(set,0) )

    # set peak cursors:
    if ( not(stf.set_peak_mean(-1)) ): return False # -1 means all points within peak window.
    if ( not(stf.set_peak_start(peak_start)) ): return False
    if ( not(stf.set_peak_end(peak_end)) ): return False
    
    # set base cursors:
    if ( not(stf.set_base_start(base_start)) ): return False
    if ( not(stf.set_base_end(base_end)) ): return False
    
    # measure everything:
    stf.measure()
    
    # calculate r_seal and return:
    return amplitude / (stf.get_peak()-stf.get_base())

def r_seal( amplitude=50 ):
    """Calculates the seal resistance from a series of voltage clamp
    traces. 

    Keyword arguments:
    amplitude  -- Amplitude of the voltage command. Defaults to 50 mV.
    
    Returns:
    The seal resistance.
    """
    return resistance( 0, 199, 1050, 1199, amplitude )

def r_in( amplitude=-5 ):
    """Calculates the input resistance from a series of voltage clamp
    traces. 

    Keyword arguments:
    amplitude  -- Amplitude of the voltage command. Defaults to -5 mV.
    
    Returns:
    The input resistance.
    """
    return resistance( 0, 999, 10700, 10999, amplitude )

def glu_iv( pulses = 13, subtract_base=True ):
    """Calculates an iv from a repeated series of fast application and
    voltage pulses. 

    Keyword arguments:
    pulses        -- Number of pulses for the iv.
    subtract_base -- If True (default), baseline will be subtracted.
    
    Returns:
    True if successful.
    """

    # Some ugly definitions for the time being
    # Cursors are in ms here.
    gFitEnd = 330.6 # fit end cursor is variable
    gFSelect  =  0 # Monoexp
    gDictSize =  stf.leastsq_param_size( gFSelect ) + 2 # Parameters, chisqr, peak value
    gBaseStart  = 220.5 # Start and end of the baseline before the control pulse, in ms
    gBaseEnd    = 223.55
    gPeakStart  = 223.55 # Start and end of the peak cursors for the control pulse, in ms
    gPeakEnd = 253.55 
    
    if ( gDictSize < 0 ):
        print "Couldn't retrieve function #", gFSelect, "; aborting now."
        return False        
    
    if ( not(stf.check_doc()) ):
        print "Couldn't find an open file; aborting now."
        return False
    
    # analyse iv, subtract baseline if requested:
    ivtools.analyze_iv( pulses )
    if ( subtract_base == True ):
        if ( not(stf.set_base_start( gBaseStart, True )) ): return False
        if ( not(stf.set_base_end( gBaseEnd, True )) ): return False
        stf.measure()
        stf.select_all()
        stf.subtract_base()
    
    # set cursors:
    if ( not(stf.set_peak_start( gPeakStart, True )) ): return False
    if ( not(stf.set_peak_end( gPeakEnd, True )) ): return False
    if ( not(stf.set_base_start( gBaseStart, True )) ): return False
    if ( not(stf.set_base_end( gBaseEnd, True )) ): return False
    if ( not(stf.set_fit_end( gFitEnd, True )) ): return False
    
    if ( not(stf.set_peak_mean( 3 )) ): return False
    if ( not(stf.set_peak_direction( "both" )) ): return False

    # A list for dictionary keys and values:
    dict_keys = []
    dict_values = N.empty( (gDictSize, stf.get_size_channel()) )
    firstpass = True
    for n in range( 0, stf.get_size_channel() ):
        if ( stf.set_trace( n ) == False ):
            print "Couldn't set a new trace; aborting now."
            return False
        
        print "Analyzing trace ", n+1, " of ", stf.get_size_channel()
        # set the fit window cursors:
        if ( not(stf.set_fit_start( stf.peak_index() )) ): return False
        
        # Least-squares fitting:
        p_dict = stf.leastsq( gFSelect )
        
        if ( p_dict == 0 ):
            print "Couldn't perform a fit; aborting now."
            return False
            
        # Create an empty list:
        tempdict_entry = []
        row = 0
        for k, v in p_dict.iteritems():
            if ( firstpass == True ):
                dict_keys.append( k )
            dict_values[row][n] = v 
            row = row+1
        
        if ( firstpass ):
            dict_keys.append( "Peak amplitude" )
        dict_values[row][n] = stf.get_peak()-stf.get_base()
        
        firstpass = False
    
    retDict = dict()
    # Create the dictionary for the table:
    entry = 0
    for elem in dict_keys:
        retDict[ elem ] = dict_values[entry].tolist()
        entry = entry+1
   
    return stf.show_table_dictlist( retDict )
    