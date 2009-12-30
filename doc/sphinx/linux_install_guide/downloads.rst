*********
Downloads
*********

:Author: Jose Guzman
:Date:  |today|

Downloading and unpacking the sources of Stimfit
================================================

You can download the last version of `Stimfit <http://www.stimfit.org>`_ from the `Stimfit homepage <http://www.stimfit.org>`_ . Simply click on the Download section, which contains different versions of `Stimfit <http://www.stimfit.org>`_ for different systems. In this guide we will download the `Stimfit <http://www.stimfit.org>`_ version |version| for Linux. Just click on the link containing the sources (i.e the stimfit- |version|.tar.gz file) or type the following command in your terminal:

::

    $ wget http://www.stimfit.org/stimfit-version.tar.gz

Where **version** is the current `Stimfit <http://www.stimfit.org>`_ version (|version| in our case). Once the the tar.gz is located into a local directory (for example /usr/local) we can type the following command to uncompress/untar it:

::

    $ tar xvfz stimfit-version.tar.gz

this will create the directory /stimfit-|version|/ in our current directory (for example /usr/local/stimfit-|version|/). 

Download wxWidgets and wxPython via svn
=======================================

`Stimfit <http://www.stimfit.org>`_ strongly depends on the wxWidgets library to manage the graphic user environment and on wxPython to manage the embeded Python shell. For that reason, we should download and build the corresponding packages via svn in our system. 
Since wxWidgets 2.9 has officially been released, there's no need to check it out from svn. You can download the sources from sourceforge and unpack them:

::

    $ wget http://prdownloads.sourceforge.net/wxwindows/wxWidgets-2.9.0.tar.bz2
    $ tar xvfj wxWidgets-2.9.0.tar.bz2
    $ mv wxWidgets-2.9.0 wxWidgets

This will create a directory called wxWidgets with the source code. Unfortunately, wxPython 2.9 has not been released yet, so we need to get the sources from svn:

::

    $ svn checkout -r 62250 http://svn.wxwidgets.org/svn/wx/wxPython/trunk/ wxPython

.. note::

    Do not attempt to download wxPython sources other than the one described above. The GNU/Linux version of `Stimfit <http://www.stimfit.org>`_ is still experimental and strongly relies on this development version of wxPython.


After obtaining the sources via svn we will have a directory called wxPython with the corresponding source code.  http://svn.wxwidgets.org/svn/wx/wxPython/trunk/ is the main development branch for wxPython. Now you will have the following directories: /usr/local/wxPython and /usr/local/wxWidgets.

The GTK+ and OpenGL development environment
===========================================

To install the development libraries of GTK+ just type as root

::

    $ apt-get install libgtk2.0-dev

Now you are going to need OpenGL. The development libraries to get would be: libgl1-mesa-dev & libglu1-mesa-dev. Just use the same command as root with:

::

    $ apt-get install libgl1-mesa-dev libglu1-mesa-dev 

If you are interested in programming graphics, you might want to go ahead and install libsdl-image1.2 and libsdl-image1.2-dev SDL_image will make loading textures for SDL and OpenGL a breeze (in all kinds of formats too), but this last is not necessary for `Stimfit <http://www.stimfit.org>`_.

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

