****************
Building Stimfit
****************

:Author: Jose Guzman, Christoph Schmidt-Hieber
:Date:    |today|

This document describes how to install `Stimfit <http://www.stimfit.org>`_ |version| under GNU/Linux. The installation was tested on a GNU/Debian testing/unstable system, with a 2.6-based kernel and with support for Python 2.5. and Python 2.6. It should work on other Debian-based systems (e.g Ubuntu) as well. I assume that you have the GNU C compiler (gcc) and the GNU C++ compiler (g++) already installed in your system. Please, check that both versions match. For our installation we will use gcc-4.2.4 and the same version of g++.

============================
What we need before we start
============================

For the impatient, here are all `Stimfit <http://www.stimfit.org>`_ build dependencies:

::

    $ sudo apt-get install libboost-dev \
                           python-dev \
                           python-numpy \
                           python-matplotlib \
                           libhdf5-serial-dev \
                           swig \
                           python-wxgtk2.8 \
                           libwxgtk2.8-dev \
                           fftw3-dev \
                           liblapack-dev \
                           chrpath \
                           mercurial \
                           automake \
                           autoconf \
                           libtool \
                           libgtest-dev

This will get you, amongst others:

* [boost]_: C++ library that is mainly used for its shared pointers.
* [Lapack]_: A linear algebra library.
* [fftw]_:  Library for computing Fourier transformations.
* [NumPy]_: To handle multidimensional arrays and perform more complex numerical computations with Python.
* [HDF5]_: This is the hierarchical Data Format 5 (HDF5) to manage large amount of data.

=======================
Optional: wxWidgets 2.9
=======================

* [wxWidgets]_ and [wxPython]_ 2.9 (unstable): If you'd like to live on the bleeding edge and get decent printing support through gtk-print, you can build against wxWidgets 2.9, which in turn needs to be built from source. To get the build dependencies (which are the same as for 2.8), do:

::

    $  sudo apt-get build-dep wxwidgets2.8

Get the source for both wxWidgets and wxPython in a single tarball:

::

    $ wget http://downloads.sourceforge.net/wxpython/wxPython-src-2.9.1.1.tar.bz2
    $ tar xvfj wxPython-src-2.9.1.1.tar.bz2

Check http://www.wxpython.org/download.php#unstable for updates.

From there, follow the build instructions found `here <http://www.wxpython.org/builddoc.php>`_

================================
Download the Stimfit source code
================================

You can download the latest development code for `Stimfit <http://www.stimfit.org>`_ from the `Google code repository <http://code.google.com/p/stimfit/>`_. In your home directory ($HOME)

::

    $ git clone https://code.google.com/p/stimfit $HOME/stimfit

This will grab all the required files into $HOME/stimfit. If you'd like to update at a later point, do:

::

    $ cd $HOME/stimfit
    $ git pull

=============
Build Stimfit
=============

Go to the stimfit directory (in our example $HOME/stimfit) and type:

::

    $ ./autogen.sh

to generate the configure script. Remember that we need Autoconf, Automake and LibTool to use autogen. After that, you can call it with

::

    $ ./configure --enable-python

The **--enable-python** option is absolutely necessary to install `Stimfit <http://www.stimfit.org>`_ since some of the functionality depends on Python. The configure script has some additional options. For example, we may want to use `IPython <http://www.scipy.org>`_  instead of the default embedded python shell with the option **---enable-ipython**  (note that the `IPython <http://www.scipy.org>`_ shell is only available under GNU/Linux and it is still very experimental). 



Finally, after running configure, you can type

::

    $ make -j[N]

where [N] is the number of parallel builds you want to start. And finally:

::

    $ sudo make install
    $ sudo /sbin/ldconfig

.. note::

    If you want to install Stimfit as local user (e.g in ~/.local) with a local version of Python (e.g ~/.local/lib/python2.6) you have to add the following argument to configure
    script:

    $ ./configure --prefix= $HOME/.local PYTHON = $HOME/.local/lib/python2.6 --enable-python

    and after that simply call **make** and **make install** as normal user. The Stimfit executable will be now in $HOME/.local

==========================================
Building Stimfit with BioSig import filter
==========================================

It is recommended to build `Stimfit <http://www.stimfit.org>`_  with the BioSig import filter to read HEKA files or to have the possibility import some other file formats used biomedical signal processing. To do it, follow this instructions:

1. It is first recommended to install libsuitesparse and libz libraries:

::

    sudo apt-get install libsuitesparse-dev libz-dev

2. Download BioSig sources: you can obtain the latest BioSig version in `Biosig downloads <http://biosig.sourceforge.net/download.html>`_ . Choose BioSig for C/C++, libbiosig. Alternatively, you can obtain the latest developmental version from the repository:

::

    svc co https://biosig.svn.sourceforge.net/svnroot/biosig/truck/biosig4c++

3. Compile the sources:

::

    make

4. Install the library:

::

    sudo make install


After that you can enter the option **--with-biosig** in the configure script of `Stimfit <http://www.stimfit.org>`_ and compile as usual.

.. [wxWidgets] http://www.wxwidgets.org
.. [wxPython] http://www.wxpython.org
.. [boost] http://www.boost.org
.. [Lapack] http://www.netlib.org/lapack/
.. [HDF5] http://www.hdfgroup.org/HDF5/
.. [NumPy] http://www.numpy.org
.. [fftw] http://www.fftw.org
