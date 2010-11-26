*************
Prerequisites
*************

:Author: Jose Guzman
:Date:    |today|

This document describes how to install `Stimfit <http://www.stimfit.org>`_ |version| under GNU/Linux. The installation was tested on a GNU/Debian testing/unstable system, with a 2.6-based kernel and with support for Python 2.5. It should work on other Debian-based systems (e.g Ubuntu) as well. I assume that you have the GNU C compiler (gcc) and the GNU C++ compiler (g++) already installed in your system. Please, check that both versions match. For our installation we will use gcc-4.2.4 and the same version of g++.

============================
What we need before we start
============================

For the impatient, here are all `Stimfit <http://www.stimfit.org>`_ dependencies in just one line:

::

    $ sudo apt-get install build-essential subversion libgtk2.0-dev libgl1-mesa-dev libglu1-mesa-dev libboost-dev liblapack-dev libfftw3-3 libfftw3-dev python-numpy libhdf5-serial-dev python-dev

To install and run `Stimfit <http://www.stimfit.org>`_ propertly under GNU/Linux, we first need the following packages:

* The current `Stimfit <http://www.stimfit.org>`_ version. You can simply visit the download section of the `Stimfit webpage <http://www.stimfit.org/>`_ and download the Version |version| for **GNU/Linux**.
* [wxWidgets]_ and [wxPython]_ 2.9 (unstable): Need to be built from source at present. wxWidgets/wxPython 2.8 won't work.
* Additional dependencies for wxWidgets, such as the development libraries for GTK+ and the Opengl. Both are necessary for the wxWidgets library to work properly.
* Additional dependencies for `Stimfit <http://www.stimfit.org>`_: you will need some additional packages to build and run `Stimfit <http://www.stimfit.org>`_. For example [boost]_, which is used internally by `Stimfit <http://www.stimfig.org>`_ for shared pointers (we need versions 1.33 or later). [Lapack]_ is needed for solving systems of linear equations. [HDF5]_ is needed for binary file handling. Additionaly, you will need [fftw]_ to perform the Fourier transformations (version 3.1 or later). Finally, if you want to use `Stimfit <http://www.stimfit.org>`_ with the embedded Python shell, we will need [NumPy]_, which allows to work with multidimensional arrays and perform complex numerical methods with Python.


.. [wxWidgets] http://www.wxwidgets.org/
.. [wxPython] http://www.wxpython.org/
.. [boost] http://www.boost.org/
.. [Lapack] http://www.netlib.org/lapack/
.. [HDF5] http://www.hdfgroup.org/HDF5/
.. [fftw] http://www.fftw.org/
