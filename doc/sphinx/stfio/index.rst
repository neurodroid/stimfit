****************
The stfio module
****************

:Author: Christoph Schmidt-Hieber
:Date:  |today|

The stfio Python module allows to read and write data in common electrophysiology formats without running Stimfit. Build instructions for GNU/Linux can be found in :doc:`/linux_install_guide/moduleonly`.

The central object in the stfio module is called a *Recording*. There are two ways to construct a *Recording*: You can either read it in from a file, or you can build it up from scratch using NumPy arrays.

=============
Reading files
=============

Files can be opened using the *read* function that returns a *Recording* object:

::

    >>> import stfio
    >>> rec = stfio.read("/home/cs/data/test.abf")

*read* takes a filename and optionally a file type (as a string) as an argument. At present, the following types are supported:

+--------+------------------------+
| ftype  | Description            |
+========+========================+
| "cfs"  | CED filing system      |
+--------+------------------------+
| "hdf5" | HDF5                   |  
+--------+------------------------+
| "abf"  | Axon binary file       |
+--------+------------------------+
| "atf"  | Axon text file         |
+--------+------------------------+
| "axg"  | Axograph X binary file |
+--------+------------------------+
| "heka" | HEKA binary file       |
+--------+------------------------+

If the file type is *None* (default), it will be guessed from the file name extension.

A *Recording* has a number of attributes that describe the recording:

::

    >>> print(rec.comment)
    Created with Clampex
    >>> print(rec.date)
    2008/1/18
    >>> print(rec.dt) # sampling interval
    0.1
    >>> print(rec.file_description) # no file description in this case

    >>> print(rec.time)
    15:08:20
    >>> print(rec.xunits)
    ms

A *Recording* consists of one or more *Channel*\s, which in turn are composed of one or more *Section*\s. They can be accessed using indexing operators ([]).

::

    >>> len(rec) # Recording consists of 2 Channels
    2
    >>> len(rec[0]) # First Channel consists of 13 Sections
    13
    >>> len(rec[0][0]) # First Section in first channel contains 146450 data points
    146450
    >>> print(rec[0].name) # channel name
    Current
    >>> print(rec[1].name)
    IN 3
    >>> print(rec[0].yunits) # channel units 
    pA
    >>> print(rec[1].yunits)
    C

The time series in a *Section* can be accessed as a NumPy array:

::

    >>> arr = rec[0][0].asarray()
    >>> type(arr)
    <type 'numpy.ndarray'>
    >>> arr.shape
    (146450,)

Note that the *Section* itself is *not* a NumPy array and therefore needs to be converted as described above before you can do fancy arithmetics:

::
    
    >>> type(rec[0][0])
    <class 'stfio.Section'>
    >>> res = rec[0][0] + 2.0
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: unsupported operand type(s) for +: 'Section' and 'float'
    >>> res = rec[0][0].asarray() + 2.0

====================================
Constructing Recordings from scratch
====================================

*Recording*\s can be assembled from NumPy arrays. Here's a particularly stupid example:

::

    import stfio
    import numpy as np
    
    arr = np.arange(0,500,0.1)

    # construct Sections from arrays:
    seclist = [stfio.Section(arr), stfio.Section(arr)]

    # construct Channels from lists of Sections
    chlist = [stfio.Channel(seclist), stfio.Channel(seclist)]
    # Set channel units
    chlist[0].yunits = "pA" 
    chlist[1].yunits = "mV" 

    # construct a Recording from a list of channels
    rec = stfio.Recording(chlist)
    rec.dt = 0.05 # set sampling interval
    rec.xunits = "ms" # set time units

=============
Writing files
=============

*Recording*\s can be stored to files using the *write* method:

::
    
    >>> import stfio
    >>> rec = stfio.read("/home/cs/data/test.abf")
    >>> rec.write("/home/cs/data/out.h5")

At present, *write* only supports hdf5 files.

