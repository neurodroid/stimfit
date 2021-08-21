****************
Building Stimfit
****************

:Author: Alois Schlögl, Jose Guzman, Christoph Schmidt-Hieber
:Date:    |today|

This document describes how to install `Stimfit <http://www.stimfit.org>`_ |version| under GNU/Linux. The installation was tested on Debian 10 (Buster) and 11 (Bullseye) with support for Python 3.*. It should work on other Debian-based systems (e.g. Ubuntu) as with newer version of Stimfit as well. I assume that you have the GNU C compiler (gcc) and the GNU C++ compiler (g++) and that both versions match.

============================
What we need before we start
============================

For the impatient, here are all `Stimfit <http://www.stimfit.org>`_ build dependencies:

::

    $ sudo apt-get install libboost-dev \
                           python3-dev \
                           python3-numpy \
                           python3-matplotlib \
                           libhdf5-serial-dev \
                           swig \
                           python3-wxgtk4.0 \
                           libwxgtk3.0-dev \
                           wx-common \
                           fftw3-dev \
                           libbiosig-dev \
                           liblapack-dev \
                           chrpath \
                           git \
                           automake \
                           autoconf \
                           libtool \
                           libgtest-dev


This will get you, amongst others:

* [wxWidgets]_: C++ graphical user interface toolkit (version >= 2.8; tested with 3.0.5)
* [wxPython]_: GUI toolkit for the Python language.
* [boost]_: C++ library that is mainly used for its shared pointers (only needed when -std=c++17 is not supported)
* [Biosig]_: A library for reading a large number biomedical signal data formats.
* [Lapack]_: A linear algebra library.
* [fftw]_:  Library for computing Fourier transformations.
* [NumPy]_: To handle numerical computations with Python (tested with version >=1.19).
* [HDF5]_: Hierarchical Data Format 5 (HDF5) to manage large amount of data.
* [Matplotlib]_: Plotting library for Python (use version >= 1.5.1)

In addition, you can install doxygen, python-sphinx (with graphviz and Latex) if you want to build the documentation.

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

Go to the stimfit directory (in our example $HOME/stimfit) and type:

::

    $ cd $HOME/stimfit
    $ ./autogen.sh

to generate the configure script. Remember that we need Autoconf, Automake and LibTool to use autogen. After that, you can call it with

::
    $ ./configure PYTHON_VERSION=3

The **--enable-python** option is activated as a default.


Finally, after running configure, you can type

::

    $ make -j[N]

where [N] is the number of parallel builds you want to start. And finally:

::

    $ sudo make install
    $ sudo /sbin/ldconfig

.. note::

    If you want to install Stimfit as local user (e.g in ~/.local) with a local version of Python (e.g ~/.local/lib/python3.9) you have to add the following argument to configure
    script:

::

    $ PYTHON=$HOME/.local/lib/python3.9 ./configure --prefix=$HOME/.local

and after that call **make** and **make install** as normal user. The Stimfit executable will be now in $HOME/.local

.. _BioSigBuild:

==========================================
Building Stimfit with BioSig import filter
==========================================

We recommend to build `Stimfit <http://www.stimfit.org>`_  with the `BioSig library <http://biosig.sourceforge.net>`_  to import files in from different biomedical disciplines. It is necessary to read files acquired with `HEKA amplifiers <http://www.heka.com>`_ or with `Signal <http://ced.co.uk/products/sigovin>`_ from CED. To do it, follow this instructions:

Install libbiosig-dev through the package manager of your distribution:

::

    sudo apt-get install libbiosig-dev

Alternatively, get the full version of biosig and its build requirements: you can obtain the latest BioSig version in `BioSig downloads <http://biosig.sourceforge.net/download.html>`_ . Choose BioSig for C/C++, libbiosig (2.3.1 or higher is recommended because of improved support for ABF2, ATF, and AXG format).

::

	./configure
	make
	sudo make install


Alternatively, you can obtain the latest developmental version from the git repository:

::

    sudo apt-get install libsuitesparse-dev libz-dev gawk libdcmtk-dev

    git clone https://git.code.sf.net/p/biosig/code biosig-code
    cd biosig-code
    autoconf # needed first time after getting repository
    ./configure
    make
    sudo make install

After that you can enter the option --with-biosig in the configure script of `Stimfit <http://www.stimfit.org>`_ and compile as usual.

======================
Building documentation
======================

The manual of `Stimfit <http://www.stimfit.org>`_ including the documentation is accessible on-line in http://www.stimfit.org/doc/sphix/. To have your local copy, you will need to install sphinx version 1.7 or older:

::

    sudo apt-get install python-sphinx

To build a local copy call:

::

    sphinx-build $HOME/Stimfit/doc/sphinx/ <destination> # destination folder could be $HOME/tmp/stf/doc/

The html documentation will be located in <destination>/index.html

Additionally, the source code is documented with [Doxygen]_ and is also accessible on-line in http://www.stimfit.org/doc/doxygen/html/. If you want to have a local copy of the documentation, you will need to install the doxygen and gravphvix:

::

    sudo apt-get install doxygen gravphvix

Enter a directory called **doc** inside Stimfit (e.g $HOME/stimfit/doc) and type:

::

    cd $HOME/stimfit/doc
    doxygen DoxyFile

The local documentation of the source code will be in $HOME/stimfit/doc/doxygen/html

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
