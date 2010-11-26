*********
Downloads
*********

:Author: Jose Guzman
:Date:  |today|

Downloading and unpacking the sources of Stimfit
================================================

You can download the latest development code for Stimfit from the Google code repository. In your home directory ($HOME)

::

    $ hg clone https://stimfit.googlecode.com/hg/ stimfit 

This will grab all the required files into $HOME/stimfit.

Download wxWidgets and wxPython via svn
=======================================

Stimfit depends on wxWidgets 2.9 (unstable) to manage the graphical user interface and on wxPython to manage the embedded Python shell.
The easiest way to build both wxWidgets and wxPython in one go is to get the code for both in one single tarball from the wxPython site:

::

    $ wget http://downloads.sourceforge.net/wxpython/wxPython-src-2.9.1.1.tar.bz2
    $ tar xvfj wxPython-src-2.9.1.1.tar.bz2

Check http://www.wxpython.org/download.php#unstable for updates.

The GTK+ and OpenGL development environment
===========================================

To install the GTK+ development libraries, do as root:

::

    $ apt-get install libgtk2.0-dev

For the OpenGL development libraries, do:

::

    $ apt-get install libgl1-mesa-dev libglu1-mesa-dev 

Additional packages
===================

Some additional dependencies are required to build stimfit. They are briefly described here with the corresponding package name and installation command.

1. **Boost:** developmental C++ libraries to manage shared pointers.

::

      $ apt-get install libboost-dev

2. **Lapack:** A package for using algebra under C++ to solve systems of linear equations. 

::

      $ apt-get install liblapack-dev

3. **fftw:** The C subroutine library for computing discrete Fourier transformations.

::

      $ apt-get install libfftw3-3 libfftw3-dev

4. **NumPy:** To handle multidimensional arrays and perform more complex numerical computations with Python.

::

      $ apt-get install python-numpy

5. **HDF5:** This is the hierarchical Data Format 5 (HDF5) to manage large amount of data.

::

      $ apt-get install libhdf5-serial-dev
