*********
Downloads
*********

:Author: Jose Guzman
:Date:  |today|

Downloading and unpacking the sources of Stimfit
================================================

You can visit the `Stimfit homepage <http://www.stimfit.org>`_ and click on the menu downloads. There you will find the Linux/Unix link which redirects you to the download zone of the different Stimfit versions. For this guide we will download the `Stimfit <http://www.stimfit.org>`_ version |version| for Linux. Just click on it or type the following in your terminal:

::

    $ wget http://www.stimfit.org/stimfit-version.tar.gz

Where **version** is the current `Stimfit <http://www.stimfit.org>`_ version (|version| in our case). After download the tar.gz file into a local directory (for example /usr/local) and type the following command to uncompress/untar:

::

    $ tar xvfz stimfit-version.tar.gz

this will create the directory /stimfit-|version|/ in your directory, for example /usr/local/stimfit-|version|/. 

Download wxWidgets and wxPython via svn
=======================================

Since wxWidgets 2.9 has officially been released, there's no need to check it out from svn any more. You can download the sources from sourceforge and unpack them:

::

    $ wget http://prdownloads.sourceforge.net/wxwindows/wxWidgets-2.9.0.tar.bz2
    $ tar xvfj wxWidgets-2.9.0.tar.bz2
    $ mv wxWidgets-2.9.0 wxWidgets

This will create a directory called wxWidgets with the source code. Unfortunately, wxPython 2.9 hasn't been released yet, so we still need to get the sources from svn:

::

    $ svn checkout -r 61426 http://svn.wxwidgets.org/svn/wx/wxPython/trunk/ wxPython

This will create a directory called wxPython with the corresponding source code.  http://svn.wxwidgets.org/svn/wx/wxPython/trunk/ is the main development branch for wxPython respectively. After that you will have the following directories: /usr/local/wxPython and /usr/local/wxWidgets.

The GTK+ and OpenGL development environment
===========================================

To install the development libraries of GTK+ just type as root

::

    $ apt-get install libgtk2.0-dev

Now you're going to need OpenGL. The development libraries to get would be: libgl1-mesa-dev & libglu1-mesa-dev. Just use the same command as root with:

::

    $ apt-get install libgl1-mesa-dev libglu1-mesa-dev. 

If you are interested in programming graphics, you might want to go ahead and install libsdl-image1.2 and libsdl-image1.2-dev SDL_image will make loading textures for SDL and OpenGL a breeze (in all kinds of formats too), but this last is not necessary for `Stimfit <http://www.stimfit.org>`_.

Additional packages
===================

These dependencies are required to build stimfit.

1. **Boost:** C++ libraries

::

      $ apt-get install libboost-dev

2. **Lapack:** A package for linear algebra. To install it just type as root:

::

      $ apt-get install liblapack-dev

3. **fftw:** The C subroutine library for computing the discrete Fourier transformations. To install the package type as root:

::

      $ apt-get install libfftw3-3 libfftw3-dev

4. **NumPy:** This package allows easy array computations from the Python shell.

::

      $ apt-get install python-numpy

5. **HDF5:** This is the hierarchical Data Format 5 (HDF5) to manage large amount of data.

::

      $ apt-get install libhdf5-serial-dev

