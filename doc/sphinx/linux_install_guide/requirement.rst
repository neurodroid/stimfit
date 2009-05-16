*************
Prerequisites
*************

This document describes the installation of ``Stimfit`` on a GNU/Debian testing/unstable system, with a 2.6 kernel, with support for python 2.5. It should work on other Debian-based systems (e.g Ubuntu) as well. I assume that you have the GNU C compiler (gcc) and the GNU C++ compiler (g++) already instelled in your system. Please, check that both versions match. For our installation we will use gcc-4.2.4 and the same version of g++. Any other installation should be carefully accomplished reading the manual of your corresponding GNU/Linux distribution.

To install ``Stimifit`` we first need:

* The current ``Stimfit`` version. You can simply download it `here <http://www.stimfit.org/stimfit-0.8.19.tar.gz>`_
* Developmental libraries of [wxWidgets]_ and [wxPython]_: to avoid problems with printing, we need to build wxWidgets 2.9 from the subversion repository. This can be with the command svn. You can get the subversion using *apt-get install subversion command* as root.It is important that the wxWidgets version dowloaded via svn correspond to the same version of wxPython to assure the compatibility of ``Stimifit`` with the Python interpreter.
* Additional dependencies for wxWidgets, such as the development libraries for GTK+ and the Opengl. Both are necessary for the wxWidgets library to work propertly.
* Additional dependencies for ``Stimfit``: you will need a couple of additional packages to build and run ``Stimfit``. For example [boost]_, which is mainly needed for shared pointers (versions 1.33 or later). [Lapack]_ is needed for solving linear equation systems. [HDF5]_ is needed for binary file handling. Finally, you will need [fftw]_ to accomplish the Fourier transformations (version 3.1 or later) and in the case of Stimfit with Python interpreter the package [NumPy]_, which allows easy handling of matrices arithmetic and numerical operations. 


.. [wxWidgets] http://www.wxwidgets.org/
.. [wxPython] http://www.wxpython.org/
.. [boost] http://www.boost.org/
.. [Lapack] http://www.netlib.org/lapack/
.. [HDF5] http://www.hdfgroup.org/HDF5/
.. [fftw] http://www.fftw.org/



