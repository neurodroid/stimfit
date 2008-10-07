"""
Some functions to create I-V curves
2008-03-26, C. Schmidt-Hieber
Indices are zero-based!
"""
import numpy as N

# stimfit python module:
import stf

def analyze_iv( pulses, trace_start = 0, factor = 1.0 ):
    """Creates an IV for the currently active channel.

    Keyword arguments:
    pulses --      Number of pulses for the IV.
    trace_start -- ZERO-BASED index of the first trace to be
                   used for the IV. Note that this is one less
                   than what is diplayed in the drop-down box.
    factor --      Multiply result with an optional factor, typically
                   from some external scaling.
    Returns:
    True upon success, False otherwise.
    """
    
    if (stf.check_doc() == False):
        print "Couldn't find an open file; aborting now."
        return False

    if (pulses < 1):
        print "Number of pulses has to be greater or equal 1."
        return False
    
    # create an empty array (will contain random numbers)
    channel = list()
    for m in range(pulses):
        # A temporary array to calculate the average:
        set = N.empty( (int((stf.get_size_channel()-m-1-trace_start)/pulses)+1, \
                        stf.get_size_trace( trace_start+m )) )
        n_set = 0
        for n in range( trace_start+m, stf.get_size_channel(), pulses ):
            # Add this trace to set:
            set[n_set,:] = stf.get_trace( n )
            n_set = n_set+1

        # calculate average and create a new section from it, multiply:
        channel.append( N.average(set, 0) * factor )
        
    stf.new_window_list( channel )
    
    return True

def select_pon( pon_pulses = 8 ):
    """Selects correction-subtracted pulses from FPulse-generated
    files.
    
    Keyword arguments:
    pon_pulses -- Number of p-over-n correction pulses.
                  This is typically 4 (for PoN=5 in the FPulse script)
                  or 8 (for PoN=9).
    Returns:
    True upon success, False otherwise.
    """

    # Zero-based indices! Hence, for P over N = 8, the first corrected
    # trace index is 9.
    for n in range( pon_pulses+1, stf.get_size_channel(), pon_pulses+2 ):
        if ( stf.select_trace( n ) == False ):
            # Unselect everything and break if there was an error:
            stf.unselect_all()
            return False
        
    return True
        
def pon_batch( iv_pulses, pon_pulses = 8, trace_start = 0, subtract_base = False, factor = 1.0 ):
    """Extracts p-over-n corrected traces in FPulse-generated files,
    and then creates an IV for the currently active channel.
    
    Keyword arguments:
    iv_pulses --     Number of pulses for the IV.
    pon_pulses --    Number of p-over-n correction pulses.
                     This is typically 4 (for PoN=5 in the FPulse script)
                     or 8 (for PoN=9).
    trace_start --   ZERO-BASED index of the first trace to be
                     used for the IV. Note that this is one less
                     than what is diplayed in the drop-down box.
    subtract_base -- Set to True if you want to subtract the baseline
                     at the end. You will need to set the baseline cursors
                     in the original file to appropriate positions if
                     you want to do this.
    factor --        Multiply result with an optional factor, typically
                     from some external scaling.
    Returns:
    True upon success, False otherwise.
    """
    
    # Extract corrected pulses:
    if ( select_pon( pon_pulses ) == False ):
        return False
    
    if ( stf.new_window_selected_this( ) == False ):
        return False
    
    # Create IV:
    if ( analyze_iv( iv_pulses, trace_start, factor ) == False ):
        return False
    
    # Subtract base:
    if ( subtract_base == True ):
        stf.select_all( )
        if ( stf.subtract_base( ) == False ):
            return False
        
    return True
