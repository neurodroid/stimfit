*************************
Loading custom text files
*************************

:Authors: Jose Guzman
:Updated: |today|

Very often we want to analyze data in `Stimfit <http://stimfit.org>`_ that is generated from a simulation or stored in non-stardard formats, like for example fluorescence. While `Stimfit <http://stimfit.org>`_ supports a huge variety of file formats, data from more exotic sources can be copied to a text file loaded in with a very simple Python script.

Examples of such cases are NEURON files, which are saved as text files as \*.dat format. Alternatively, users configure custom files formats to, for example, analyze the timecourse of a fluorescent measurement with `Stimfit <https://stimfit.org>`_.  

=========================
Reading NEURON text files
=========================

NEURON allows to save a simulation in a very simple text file format. The file consists of a header with two lines containing the event that was recorded and the number of samples. After that, it is followed by the recording time and the sampled data data in a two-column format. To load such a file, we would like to skip the first two rows, and to load the adquisition time.

::

    import stf
    
    def loadnrn(file):
        """
        Loads a NEURON datafile and opens a new Stimfit window
        with a trace with the default units (e.g ms and mV)

        file    -- (string) file to read
        """

        time, trace = np.loadtxt(fname = file, skiprows = 2, unpack= True )

        dt = time[1] # the second temporal sampling point is the sampling
        stf.new_window( trace )
        stf.set_sampling_interval( dt )

        
You can download an example of such a file `here <http://stimfit.org/doc/EPSP.dat>`_

Note that the argument of the function *loadnrn* is a string containing the exact path and filename of the file that we want to load. For very lengthy paths, it can be convenient to write a small gadget that cares about the proper identification of the file. This is what we propose in the section below.

==========================================
Custom ascii files containing fluorescence 
==========================================

When creating custom text files to be loaded later, it is generally convenient to take into account the folowing advices:

1. Use a custom extension ( generally dat, .text or similar denote files associated with specific applications).
2. Add comments that contains the experimental conditions. 
3. Information about the author and date of file modification can be included in the header inside these comments.

In the example below, a function will load a file with extension \*.GoR that contains fluorescent measurements in time (acquired at 400 Hz). In addition, a small wx gadget will allow us to create a small window to select the file that we want to import. 

::

    import stf

    def loadtxt( freq=400 ):
        """
        Loads and ASCII file with extension *.GoR. 
        This file contains ratiometric fluorescent measurementes 
        (i.e Green over Red fluorescence)
        saved in one column. This function opens a new Stimfit window and 
        sets the x-units to ms and y-units "Delta G over R".
        Arguments:

        freq -- (float) the sampling rate (in Hz) for the acquistion.
                the default value is 400 Hz.
        """

        # wx Widgets to create a file selection window
        fname = wx.FileSelector("Import Ca transients" ,
            default_extension = "Ratiometric" ,
            default_path = "." ,
            wildcard = "Ratiometric fluorescence (*.GoR)|*.GoR" ,
            flags = wx.OPEN | wx.FILE_MUST_EXIST)

        stf.new_window( np.loadtxt( fname ) )
        stf.set_xunits('ms')
        stf.set_yunits('Delta G/R')

        stf.set_sampling_interval(1.0/freq*1000) # acquision in ms 
    
You can download an example file containing fluorescence `here <http://stimfit.org/tutorial/transient.GoR>`_.

==============
Code commented
==============

In both script cases, most of the work is done with *loadtxt* from the *NumPy* package. The first argument that we provide is a string with the name of the file to be loaded. Note that the argument options of *loadtxt* permits a lot of flexibility when loading ASCII files (for example, skiprows = 2 will skip the first two lines). Check the `NumPy documentation <http://docs.scipy.org/doc/>`_ of this function for further details.

To make the experience more interactive, we can load the string of the file with a the GUI that is provided by the wxPython package. This is what we do when we called the wx.FileSelector object in the second example. wxPython is loaded at the beginnig of Stimfit as 'wx' and runs already under the Stimfit application. Remember to create a wxPython application if you want to use wx.FileSelector outside Stimfit.

=====
Usage
=====

To see how both function work import the spells module in the Python console of Stimfit and try the example files. For example: 

::

    import spells

    # to load the file in a give location 
    spells.loadnrn( "/home/myuser/myDownloads/EPSP.dat" )

    # to select a file containing fluorescence with a selection menue
    spells.loadtxt() 
