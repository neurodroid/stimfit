****************
Building Stimfit
****************

:Author: Alois Schlögl, Jose Guzman, Christoph Schmidt-Hieber
:Date:    |today|

This document describes how to install `Stimfit <http://www.stimfit.org>`_ |version| under GNU/Linux. The installation was tested on Debian 10 (Buster) and 11 (Bullseye) with support for Python 3.*. It should work on other Debian-based systems (e.g. Ubuntu) as with newer version of Stimfit as well. I assume that you have the GNU C compiler (gcc) and the GNU C++ compiler (g++) and that both versions match.

.. important::

   The historical autotools build described in older Stimfit documentation is
   no longer the supported path for current [`master`](README.md:13). Use the
   CMake workflow from [`BUILDING.md`](BUILDING.md) or the commands below.

============================
What we need before we start
============================

For the impatient, here are all `Stimfit <http://www.stimfit.org>`_ build dependencies :
for Debian13/bookworm (and probably Ubuntu 24.xx) and later

::

    $ sudo apt-get install \
                           python3-dev \
                           python3-numpy \
                           python3-matplotlib \
                           libhdf5-dev \
                           swig \
                           sip-tools \
                           python3-wxgtk4.0 \
                           libwxgtk3.2-dev \
                           wx-common \
                           libfftw3-dev \
                           libbiosig-dev \
                           liblapack-dev \
                           chrpath \
                           git \
                           automake \
                           autoconf \
                           libtool \
                           libgtest-dev


for Debian11/bullseye (and probably Ubuntu 20.x) it was

::

    $ sudo apt-get install \
                           python3-dev \
                           python3-numpy \
                           python3-matplotlib \
                           libhdf5-serial-dev \
                           swig \
                           python3-sip-dev \
                           python3-wxgtk4.0 \
                           libwxgtk3.0-gtk3-dev \
                           wx-common \
                           libfftw3-dev \
                           libbiosig-dev \
                           liblapack-dev \
                           chrpath \
                           git \
                           automake \
                           autoconf \
                           libtool \
                           libgtest-dev


This will get you, amongst others:

* [wxWidgets]_: C++ graphical user interface toolkit (version >= 2.8; tested with 3.2.8, 3.3.1)
* [wxPython]_: GUI toolkit for the Python language.
* [boost]_: C++ library that is mainly used for its shared pointers (only needed when -std=c++17 is not supported)
* [Biosig]_: A library for reading a large number biomedical signal data formats.
* [Lapack]_: A linear algebra library.
* [fftw]_:  Library for computing Fourier transformations.
* [NumPy]_: To handle numerical computations with Python (tested with version >=1.19).
* [HDF5]_: Hierarchical Data Format 5 (HDF5) to manage large amount of data.
* [Matplotlib]_: Plotting library for Python (use version >= 1.5.1)

In addition, install Sphinx and Doxygen if you want to build the documentation.

=======================
Optional: PyEMF
=======================

[PyEMF]_ is needed to export figures to the windows meta file format (WMF/EMF). EMF is a vector graphics format and can be imported in different Office software including LibreOffice. To install it, do:

::

     $ wget http://sourceforge.net/projects/pyemf/files/pyemf/2.0.0/pyemf-2.0.0.tar.gz/download -O pyemf-2.0.0.tar.gz
     $ tar xvf pyemf-2.0.0.tar.gz && cd pyemf-2.0.0
     $ sudo python setup.py install


================================
Download the Stimfit source code
================================

You can download the latest development code for `Stimfit <http://www.stimfit.org>`_ from the `Github code repository <https://github.com/neurodroid/stimfit/>`_. For that, type from your current $HOME directory:

::

    $ git clone https://github.com/neurodroid/stimfit.git

This will grab all the required files into $HOME/stimfit. If you'd like to update at a later point, do:

::

    $ cd $HOME/stimfit
    $ git pull

=============
Build Stimfit
=============

Go to the stimfit directory (in our example $HOME/stimfit) and configure a
dedicated CMake build directory:

::

    $ cd $HOME/stimfit
    $ cmake -S . -B build/linux -G Ninja \
              -DCMAKE_BUILD_TYPE=Release \
              -DSTF_ENABLE_PYTHON=ON \
              -DSTF_WITH_BIOSIG=ON
    $ cmake --build build/linux --parallel
    $ sudo cmake --install build/linux

.. note::

    If you want to install Stimfit as a local user into `~/.local` or target a
    specific Python interpreter, add the relevant CMake cache entries during
    configure, for example:

::

    $ cmake -S . -B build/linux-local -G Ninja \
              -DCMAKE_INSTALL_PREFIX=$HOME/.local \
              -DPython3_EXECUTABLE=$HOME/.local/bin/python3

    Then run `cmake --build` and `cmake --install` as your normal user.

.. _BioSigBuild:

==========================================
Building Stimfit with BioSig import filter
==========================================

We recommend to build `Stimfit <http://www.stimfit.org>`_  with the `BioSig library <http://biosig.sourceforge.net>`_  to import files in from different biomedical disciplines. It is necessary to read files acquired with `HEKA amplifiers <http://www.heka.com>`_ or with `Signal <http://ced.co.uk/products/sigovin>`_ from CED. To do it, follow this instructions:

Install libbiosig-dev through the package manager of your distribution:

::

    sudo apt-get install libbiosig-dev

Alternatively, use the in-tree BioSig provider managed by the current CMake
toolchain. This is the preferred fallback when your distribution package is too
old or unavailable:

::

    $ cd $HOME/stimfit
    $ cmake -S . -B build/linux-biosig-submodule -G Ninja \
              -DCMAKE_BUILD_TYPE=Release \
              -DSTF_WITH_BIOSIG=ON \
              -DSTF_BIOSIG_PROVIDER=SUBMODULE
    $ cmake --build build/linux-biosig-submodule --parallel


Alternatively, if you are maintaining BioSig itself and need a separate local
checkout from its git repository:

::

    sudo apt-get install libsuitesparse-dev libz-dev gawk libdcmtk-dev

    git clone https://git.code.sf.net/p/biosig/code biosig-code
    cd biosig-code

Then build that checkout using the current BioSig project's own documented build
instructions, install it into your desired prefix, and point Stimfit's CMake
configure step at that installation through the appropriate provider settings.

After that you can configure Stimfit with `-DSTF_WITH_BIOSIG=ON` and, when
needed, select the desired provider through `-DSTF_BIOSIG_PROVIDER=...`.

======================
Building documentation
======================

The manual of `Stimfit <http://www.stimfit.org>`_ is published at
https://neurodroid.github.io/stimfit/. For a local build, install Sphinx and
related helpers in a Python 3 environment:

::

    python3 -m pip install -r doc/sphinx/requirements.txt sphinx

To build a local copy call:

::

    sphinx-build -b html doc/sphinx doc/sphinx/.build/html

The HTML documentation will be written to `doc/sphinx/.build/html/index.html`.

Additionally, the source code can be documented with [Doxygen]_. Install Doxygen
and Graphviz first:

::

    sudo apt-get install doxygen graphviz

Enter a directory called **doc** inside Stimfit (e.g $HOME/stimfit/doc) and type:

::

    cd $HOME/stimfit/doc
    doxygen Doxyfile.in

The local documentation of the source code will be in `$HOME/stimfit/doc/doxygen/html`.

.. [wxWidgets] http://www.wxwidgets.org
.. [wxPython] http://www.wxpython.org
.. [Biosig] http://biosig.sourceforge.net
.. [boost] http://www.boost.org
.. [Lapack] http://www.netlib.org/lapack/
.. [HDF5] http://www.hdfgroup.org/HDF5/
.. [NumPy] http://www.numpy.org
.. [PyEMF] http://pyemf.sourceforge.net
.. [fftw] http://www.fftw.org
.. [Doxygen] http://www.doxygen.org
.. [Matplotlib] https://matplotlib.org
