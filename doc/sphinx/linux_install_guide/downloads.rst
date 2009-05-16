*********
Downloads
*********

Downloading and unpacking the sources of Stimfit
================================================

You can visit the `Stimfit homepage <http://www.stimfit.org>`_ and click on the menu downloads. There you will find the Linux/Unix link which redirects you to the download zone of the different Stimfit verstions. For this guide we will download the version 0.8.19 for Linux. Just click on it or type the following in your terminal:

::

    >>> wget http://www.stimfit.org/stimfit-0.8.19.tar.gz

After download the tar.gz file into a local directory (for example /usr/local) and type the following command to uncompress/untar:

::

    >>> tar xvfz stimfit-0.8.19.tar.gz

this will create the directory /stimfit-0.8.19/ in your directory, for example /usr/local/stimfit-0.8.19/. 

Download wxWidgets and wxPython via svn
=======================================

To get your a copy of the source code from SVN simply run svn checkout -r with the number of the subversion. In our example, we will get revision 59896:

::

    >>> svn checkout -r 59896 http://svn.wxwidgets.org/svn/wx/wxWidgets/trunk wxWidgets

This will create a directory called wxWidgets with the source code. In the same way, we type the following to get wxPython via svn

::

    >>> svn checkout -r 59896 http://svn.wxwidgets.org/svn/wx/wxPython/trunk/ wxPython

This will create a directory called wxPython with the corresponding source code. Both http://svn.wxwidgets.org/svn/wx/wxWidgets/trunk and http://svn.wxwidgets.org/svn/wx/wxPython/trunk/ are the main development branch for wxWidgets and wxPython respectively. After that you will have the following directories: /usr/local/wxPython and /usr/local/wxWidgets.

The GTK+ and OpenGL development environment
===========================================

To install the development libraries of GTK+ just type as root

::

    >>> apt-get install libgtk2.0-dev

Now you're going to need OpenGL. The development libraries to get would be: libgl1-mesa-dev & libglu1-mesa-dev. Just use the same command as root with:

::

    >>> apt-get install libgl1-mesa-dev libglu1-mesa-dev. 

If you are interested in programming graphics, you might want to go ahead and install libsdl-image1.2 and libsdl-image1.2-dev SDL_image will make loading textures for SDL and OpenGL a breeze (in all kinds of formats too), but this last is not necessary for ``Stimfit``.

Additional packages
===================

These dependencies are required to build stimfit.

1. Boost: C++ libraries

::

      >>> apt-get install libboost-dev

2. Lapack: A package for linear algebra. To install it just type as root:

::

      >>> apt-get install liblapack-dev

3. fftw: The C subroutine library for computing the discrete Fourier transformations. To install the package type as root:

::

      >>> apt-get install libfftw3-3 libfftw3-dev

4. NumPy: This package allows easy array computations from the Python shell.

::

      >>> apt-get install python-numpy

5. HDF5 This is the hierarchical Data Format 5 (HDF5) to manage large amount of data.

::

      >>> apt-get install libhdf5-serial-dev

