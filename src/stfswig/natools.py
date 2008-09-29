"""
2008-04-11, C. Schmidt-Hieber
Batch analysis of Na IVs
"""

import stf
import numpy as N
from scipy.io import write_array

def dens_batch( nFunc = 0 ):
    """Fits activation and inactivation of 15 iv pulses
    using a biexponential funtion with a delay, creates a
    table showing the results.
    
    Keyword argument:
    nFunc -- Index of function used for fitting. At present,
             10 is the HH gNa function,
             5  is a sum of two exponentials with a delay."""

    # Some ugly definitions for the time being
    gFitStart = 70.56
    gFSelect  =  nFunc    # HH function
    gDictSize =  stf.leastsq_param_size( gFSelect ) + 2 # Parameters, chisqr, peak value
    gBaseStartCtrl  = 69.5 # Start and end of the baseline before the control pulse, in ms
    gBaseEndCtrl    = 70.5
    gPeakStartCtrl  = 70.64 # Start and end of the peak cursors for the control pulse, in ms
    gPeakWindowSize = 0.8
    gFitEnd   =  gFitStart+4.5
    dt = stf.get_sampling_interval()
    if ( gDictSize < 0 ):
        print "Couldn't retrieve function #", gFSelect, "; aborting now."
        return False        
    
    if ( not(stf.check_doc()) ):
        print "Couldn't find an open file; aborting now."
        return False
    
    # set cursors:
    if ( not(stf.set_peak_start( gPeakStartCtrl/dt )) ):
        return False
    if ( not(stf.set_base_start( gBaseStartCtrl/dt )) ):
        return False
    if ( not(stf.set_base_end( gBaseEndCtrl/dt )) ):
        return False
    
    if ( not(stf.set_peak_mean( 3 )) ):
        return False
    if ( not(stf.set_peak_direction( "both" )) ):
        return False

    # A list for dictionary keys and values:
    dict_keys = []
    dict_values = N.empty( (gDictSize, 1) )

    if ( not(stf.set_peak_end( (gPeakStartCtrl + gPeakWindowSize)/dt )) ):
        return False
    stf.measure()

    # set the fit window cursors:
    if ( not(stf.set_fit_start( stf.peak_index()+1 )) ):
        return False
    if ( not(stf.set_fit_end( gFitEnd/dt )) ):
        return False
        
    # Least-squares fitting:
    p_dict = stf.leastsq( gFSelect )
        
    if ( p_dict == 0 ):
        print "Couldn't perform a fit; aborting now."
        return False
            
    # Create an empty list:
    tempdict_entry = []
    row = 0
    for k, v in p_dict.iteritems():
        dict_keys.append( k )
        dict_values[row][0] = v 
        row = row+1
        
    dict_keys.append( "Peak amplitude" )
    dict_values[row][0] = stf.get_peak()-stf.get_base()
        
    retDict = dict()
    # Create the dictionary for the table:
    entry = 0
    for elem in dict_keys:
        retDict[ elem ] = dict_values[entry].tolist()
        entry = entry+1
   
    if not stf.show_table_dictlist( retDict ):
        return False

    # divide by inactivation part:
    trace = stf.get_trace()[gFitStart/dt:gFitEnd/dt]
    l = N.empty( (3, len(trace)) )
    l[1] = trace - stf.get_base()
    t = N.arange(0,len(l[1]))*dt
    l[2] = N.exp(-t/p_dict['Tau_0'])
    l[0] = l[1] / l[2]
    stf.new_window_matrix( l )
    stf.set_base_start(0)
    stf.set_base_end(0)
    stf.set_peak_mean(-1)
    stf.set_peak_start(10)
    stf.set_peak_end(32)
    stf.measure()

def act_batch( nFunc = 5, filename="", lat=60 ):
    """Fits activation and inactivation of 15 iv pulses
    using a biexponential funtion with a delay, creates a
    table showing the results.
    
    Keyword argument:
    nFunc --    Index of function used for fitting. At present,
                10 is the HH gNa function,
                5  is a sum of two exponentials with a delay.
    filename -- If not an empty string, stores the best-fit parameters
                in this file."""

    # Some ugly definitions for the time being
    gFitStart = 70.5 + lat/1000.0 # fit end cursor is variable
    gFSelect  =  nFunc    # HH function
    gDictSize =  stf.leastsq_param_size( gFSelect ) + 2 # Parameters, chisqr, peak value
    gBaseStartCtrl  = 69.5 # Start and end of the baseline before the control pulse, in ms
    gBaseEndCtrl    = 70.5
    gPeakStartCtrl  = 70.64 # Start and end of the peak cursors for the control pulse, in ms
    gPeakWindowSizes = ( 2.5,   2, 1.5, 1,   1, 0.8, 0.8, 0.8, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.4 )  
    gFitDurations   =  (   8,   8,   7, 6, 5.5,   5, 4.5, 3.5, 2.5,   2, 1.5, 1.5, 1.0, 0.8, 0.8 )
    gPulses = len( gFitDurations )    # Number of traces 
    
    if ( gDictSize < 0 ):
        print "Couldn't retrieve function #", gFSelect, "; aborting now."
        return False        
    
    if ( not(stf.check_doc()) ):
        print "Couldn't find an open file; aborting now."
        return False
    
    # set cursors:
    if ( not(stf.set_peak_start( gPeakStartCtrl, True )) ):
        return False
    if ( not(stf.set_peak_end( stf.get_size_trace(0)-1 )) ):
        return False
    if ( not(stf.set_base_start( gBaseStartCtrl, True )) ):
        return False
    if ( not(stf.set_base_end( gBaseEndCtrl, True )) ):
        return False
    
    if ( not(stf.set_peak_mean( 3 )) ):
        return False
    if ( not(stf.set_peak_direction( "both" )) ):
        return False

    firstpass = True
    # A list for dictionary keys and values:
    dict_keys = []
    dict_values = N.empty( (gDictSize, stf.get_size_channel()) )
    if not filename=="":
        ls_file=N.empty((gPulses,stf.leastsq_param_size(nFunc)))
    for n in range( 0, gPulses ):
        if ( stf.set_trace( n ) == False ):
            print "Couldn't set a new trace; aborting now."
            return False
        
        print "Analyzing trace ", n+1, " of ", stf.get_size_channel()
        # set the fit window cursors:
        if ( not(stf.set_peak_end( gPeakStartCtrl + gPeakWindowSizes[n], True )) ):
            return False
        if ( not(stf.set_fit_start( gFitStart, True )) ):
            return False
        if ( not(stf.set_fit_end( gFitStart + gFitDurations[n], True )) ):
            return False
        
        stf.measure()
        
        # Least-squares fitting:
        p_dict = stf.leastsq( gFSelect )
        if not filename=="":
            ls_file[n][0]=p_dict["gprime_na"]
            ls_file[n][1]=p_dict["tau_m"]
            ls_file[n][2]=p_dict["tau_h"]
            ls_file[n][3]=p_dict["offset"]

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
    
    
    if not filename=="":
        write_array(file(filename,'w'), ls_file, precision=15)

    retDict = dict()
    # Create the dictionary for the table:
    entry = 0
    for elem in dict_keys:
        retDict[ elem ] = dict_values[entry].tolist()
        entry = entry+1
    
    return stf.show_table_dictlist( retDict )

def inact_batch():
    """Determines peak amplitudes for inactivation protocol."""

    # Some ugly definitions for the time being
    gDictSize =  1 # Parameters, chisqr, peak value
    gBaseStartCtrl  = 69 # Start and end of the baseline before the control pulse, in ms
    gBaseEndCtrl    = 70
    gPeakStartCtrl  = 70.12 # Start and end of the peak cursors for the control pulse, in ms
    gPeakWindowSize = 0.2
    gPeakEndCtrl    = gPeakStartCtrl + gPeakWindowSize
    gPulses = 11    # Number of traces 
    
    if ( not(stf.check_doc()) ):
        print "Couldn't find an open file; aborting now."
        return False
    
    # set cursors:
    if ( not(stf.set_peak_start( gPeakStartCtrl, True )) ):
        return False
    if ( not(stf.set_peak_end( gPeakEndCtrl, True )) ):
        return False
    if ( not(stf.set_base_start( gBaseStartCtrl, True )) ):
        return False
    if ( not(stf.set_base_end( gBaseEndCtrl, True )) ):
        return False
    
    if ( not(stf.set_peak_mean( 4 )) ):
        return False
    if ( not(stf.set_peak_direction( "both" )) ):
        return False

    # A list for dictionary keys and values:
    dict_keys = [ "Peak amplitude", ]
    dict_values = N.empty( (gDictSize, gPulses) )

    for n in range( 0, gPulses ):
        if ( stf.set_trace( n ) == False ):
            print "Couldn't set a new trace; aborting now."
            return False
        
        print "Analyzing pulse ", n+1, " of ", stf.get_size_channel()

        # Update calculations:
        stf.measure()

        # Store values:
        dict_values[0][n] = stf.get_peak() - stf.get_base()
    
    inactDict = dict()
    # Create the dictionary for the table:
    entry = 0
    for elem in dict_keys:
        inactDict[ elem ] = dict_values[entry].tolist()
        entry = entry+1
   
    return stf.show_table_dictlist( inactDict )

def deact_batch( filename="" ):
    """Fits deactivation time constants: Monoexponential until <=-70,
    biexponential for >-70 mV.

    filename -- If not an empty string, stores the best-fit parameters
                in this file."""

    # Some ugly definitions for the time being
    gNMono = 5   # Monoexponential fits 
    gNBi   = 4   # Biexponential fits
    gFMono = 0   # id of monoexponential function
    gFBi   = 3   # id of biexponential function
    gMonoDictSize =  stf.leastsq_param_size( gFMono ) + 1 # Parameters, chisqr
    gBiDictSize =    stf.leastsq_param_size( gFBi ) + 1   # Parameters, chisqr

    if ( gMonoDictSize < 0 or gBiDictSize < 0 ):
        print "Couldn't retrieve function; aborting now."
        return False        
    
    if ( not(stf.check_doc()) ):
        print "Couldn't find an open file; aborting now."
        return False

    # set the test pulse window cursors:
    if ( not(stf.set_peak_start( 70.84, True )) ):
        return False
    if ( not(stf.set_peak_end( 74.84, True )) ):
        return False

    if ( not(stf.set_base_start( 69.5, True )) ):
        return False
    if ( not(stf.set_base_end( 70.5, True )) ):
        return False
    
    if ( not(stf.set_peak_mean( 1 )) ):
        return False
    if ( not(stf.set_peak_direction( "down" )) ):
        return False

    # Monoexponential loop ---------------------------------------------------
    
    firstpass = True
    # A list for dictionary keys...
    mono_keys = []
    # ... and values:
    mono_values = N.empty( (gMonoDictSize, gNMono) )
    if not filename=="":
        ls_file=N.empty((gNMono,stf.leastsq_param_size(gFMono)))
    
    # Monoexponential fits:
    for n in range( 0, gNMono ):
        if ( stf.set_trace( n ) == False ):
            print "Couldn't set a new trace; aborting now."
            return False
        
        print "Analyzing trace ", n+1, " of ", stf.get_size_channel()
        
        # set the fit window cursors:
        
        # use the index for the start cursor:
        if ( not(stf.set_fit_start( stf.peak_index( True ) )) ):
            return False
        
        # fit 1.5 ms:
        fit_end_time = stf.get_fit_start( True )+1.0
        if ( not(stf.set_fit_end( fit_end_time, True)) ):
            return False
        
        # Least-squares fitting:
        p_dict = stf.leastsq( gFMono )
        if not filename=="":
            ls_file[n][0]=p_dict["Amp_0"]
            ls_file[n][1]=p_dict["Tau_0"]
            ls_file[n][2]=p_dict["Offset"]
        
        if ( p_dict == 0 ):
            print "Couldn't perform a fit; aborting now."
            return False
            
        # Create an empty list:
        tempdict_entry = []
        row = 0
        for k, v in p_dict.iteritems():
            if ( firstpass == True ):
                mono_keys.append( k )
            mono_values[row][n] = v 
            row = row+1
        
        firstpass = False
    
    monoDict = dict()
    # Create the dictionary for the table:
    entry = 0
    for elem in mono_keys:
        monoDict[ elem ] = mono_values[entry].tolist()
        entry = entry+1
   
    if ( not(stf.show_table_dictlist( monoDict )) ):
        return False
    
    # Biexponential loop ---------------------------------------------------
    
    firstpass = True
    # A list for dictionary keys...
    bi_keys = []
    # ... and values:
    bi_values = N.empty( (gBiDictSize, gNBi) )
    
    # Monoexponential fits:
    for n in range( gNMono, gNBi+gNMono ):
        if ( stf.set_trace( n ) == False ):
            print "Couldn't set a new trace; aborting now."
            return False
        
        print "Analyzing trace ", n+1, " of ", stf.get_size_channel()
        
        # set the fit window cursors:
        
        # use the index for the start cursor:
        if ( not(stf.set_fit_start( stf.peak_index( True ) )) ):
            return False
        
        # fit 4 ms:
        fit_end_time = stf.get_fit_start( True )+3.5
        if ( not(stf.set_fit_end( fit_end_time, True)) ):
            return False
        
        # Least-squares fitting:
        p_dict = stf.leastsq( gFBi )
        
        if ( p_dict == 0 ):
            print "Couldn't perform a fit; aborting now."
            return False
            
        # Create an empty list:
        tempdict_entry = []
        row = 0
        for k, v in p_dict.iteritems():
            if ( firstpass == True ):
                bi_keys.append( k )
            bi_values[row][n-gNMono] = v 
            row = row+1
        
        firstpass = False
    
    biDict = dict()
    
    # Create the dictionary for the table:
    entry = 0
    for elem in bi_keys:
        biDict[ elem ] = bi_values[entry].tolist()
        entry = entry+1

    if not filename=="":
        write_array(file(filename,'w'), ls_file, precision=15)
   
    if ( not(stf.show_table_dictlist( biDict )) ):
        return False
    
    return True

def inact_recov_batch():
    """Determines recovery from inactivation."""

    if ( not(stf.check_doc()) ):
        print "Couldn't find an open file; aborting now."
        return False

    # Some ugly definitions for the time being
    gBaseStartCtrl  = 69 # Start and end of the baseline before the control pulse, in ms
    gBaseEndCtrl    = 70
    gPeakStartCtrl  = 70.12 # Start and end of the peak cursors for the control pulse, in ms
    gPeakWindowSize = 0.5
    gPeakEndCtrl    = gPeakStartCtrl + gPeakWindowSize
    gDictSize       = 2 # Control peak amplitude, test peak amplitude
    gDurations      = ( 0, 1, 2, 3, 5, 7, 9, 13, 20, 30, 50, 100 ) # Durations of the pulses
    gPulses         = len( gDurations ) # Number of pulses
    
    # A list for dictionary keys...
    dict_keys = [ "Control amplitude", "Test amplitude" ]
    # ... and values:
    dict_values = N.empty( (gDictSize, gPulses) )

    
    if ( not(stf.set_peak_mean( 3 )) ):
        return False
    if ( not(stf.set_peak_direction( "down" )) ):
        return False

    for n in range( 0, gPulses ):

        if ( stf.set_trace( n ) == False ):
            print "Couldn't set a new trace; aborting now."
            return False
        
        print "Analyzing control pulse ", n+1, " of ", stf.get_size_channel()
        
        # set the control pulse window cursors:
        if ( not(stf.set_peak_start( gPeakStartCtrl, True )) ):
            return False
        if ( not(stf.set_peak_end( gPeakEndCtrl, True )) ):
            return False

        if ( not(stf.set_base_start( gBaseStartCtrl, True )) ):
            return False
        if ( not(stf.set_base_end( gBaseEndCtrl, True )) ):
            return False
        
        # Update calculations:
        stf.measure()
        
        # Store values:
        dict_values[0][n] = stf.get_peak() - stf.get_base()
        
        print "Analyzing test pulse ", n+1, " of ", stf.get_size_channel()
 
        # set the test pulse window cursors:
        if ( not(stf.set_peak_start( gDurations[n]+100.16, True )) ):
            return False
        if ( not(stf.set_peak_end( gDurations[n]+100.16+gPeakWindowSize, True )) ):
            return False

        if ( not(stf.set_base_start( gDurations[n]+100-1, True )) ):
            return False
        if ( not(stf.set_base_end( gDurations[n]+100, True )) ):
            return False

        # Update calculations:
        stf.measure()

        # Store values:
        dict_values[1][n] = stf.get_peak() - stf.get_base()
    
    inactDict = dict()
    # Create the dictionary for the table:
    entry = 0
    for elem in dict_keys:
        inactDict[ elem ] = dict_values[entry].tolist()
        entry = entry+1
   
    return stf.show_table_dictlist( inactDict )

def inact_onset_batch( show_table = True ):
    """Determines onset of inactivation."""

    if ( not(stf.check_doc()) ):
        print "Couldn't find an open file; aborting now."
        return -1

    # Some ugly definitions for the time being
    gPeakWindowSize = 0.5
    gDictSize       = 1 # Control peak amplitude, test peak amplitude
    gDurations      = ( 0, 1, 2, 3, 5, 7, 9, 13, 20, 25, 30 ) # Durations of the pulses
    gPulses         = len( gDurations ) # Number of pulses
    
    # A list for dictionary keys...
    dict_keys = [ "Test amplitude", ]
    # ... and values:
    dict_values = N.empty( (gDictSize, gPulses) )

    if ( not(stf.set_peak_mean( 4 )) ):
        return -1
    if ( not(stf.set_peak_direction( "down" )) ):
        return -1

    for n in range( 0, gPulses ):

        if ( stf.set_trace( n ) == False ):
            print "Couldn't set a new trace; aborting now."
            return -1
        
        print "Analyzing pulse ", n+1, " of ", stf.get_size_channel()
        
        # set the test pulse window cursors:
        if ( not(stf.set_peak_start( gDurations[n]+70.12, True )) ):
            return -1
        if ( not(stf.set_peak_end( gDurations[n]+70.12+gPeakWindowSize, True )) ):
            return -1

        if ( not(stf.set_base_start( gDurations[n]+70-1, True )) ):
            return -1
        if ( not(stf.set_base_end( gDurations[n]+70, True )) ):
            return -1

        # Update calculations:
        stf.measure()

        # Store values:
        dict_values[0][n] = stf.get_peak() - stf.get_base()
    
    inactDict = dict()
    # Create the dictionary for the table:
    entry = 0
    for elem in dict_keys:
        inactDict[ elem ] = dict_values[entry].tolist()
        entry = entry+1
    
    if show_table:
        if not stf.show_table_dictlist( inactDict ):
            return -1

    return dict_values[0]
