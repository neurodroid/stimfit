*************
Prerequisites
*************

:Author: Christoph Schmidt-Hieber
:Date:    |today|

This document describes how to build a 64-bit version of `Stimfit <http://www.stimfit.org>`_ on Windows. I strongly recommend sticking to the suggested directory names. If for any reason you'd like to use different target directories, you'll have to update all the property sheets (Config.vsprops) in the Visual Studio solution.

Getting all the prerequisites takes about an hour, but only needs to be completed once. Building the full solution takes about 3 minutes.

=======================
Visual C++ Express 2008
=======================

The official Windows version of Python 2.7 was built with Visual Studio 2008. We therefore have to use Visual C++ 2008 so that we link against the same C runtime library. Luckily, there's a free version called Visual C++ 2008 Express that you can get directly from `here <http://go.microsoft.com/?linkid=7729279>`_

64-bit
------
Visual C++ Express 2008 will not build 64-bit targets out of the box. This capability has to be added by installing the Windows SDK and making some registry edits. Follow the instructions here to do this: :doc:`VCExpress64bitsetup`. Note that if you have the full rather than express version of Visual C++, this should not be necessary.

You also need 64-bit versions of all the required libraries.

We have dropped support for 32-bit with version 0.14.

=========
Libraries
=========

HDF5
----
Get the HDF5 libraries from `here <http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.13/bin/windows/>`_. Use hdf5-1.8.13-win64-VS2008-shared.zip. Extract the zip file, and then install to a folder called "hdf5" in your home directory (e.g. C:\\Users\\username) using the extracted executable.

Boost
-----
Get the `Boost C++ libraries <http://www.boost.org>`_. Move the extracted folder to your home directory and rename to "boost". If you used the zip file, you might have to move the first folder (called something like boost_1_54_0, or whatever the current version is) one directory up and rename it to "boost". At any rate, you should check that you have the boost header files (\*.hpp) sitting in C:\\Users\\username\\boost\\boost\\\*.hpp. 

Python
------
Download and install the 64-bit version of `Python 2.7 <http://www.python.org>`_. Make sure to install it for all users so that it ends up in C:\\Python27

PyEMF
-----
Get PyEMF 2.0.0 from `here <http://sourceforge.net/projects/pyemf/files/latest/download?source=files>`_. Install to your home directory(_not_ to C:\\Python27\\*!) to pyemf_2.0.0. Rename the folder to "pyemf-2.0.0".

NumPy
-----
Install NumPy from `Chris Gohlke's repository <http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy>`_. Use the win64 (amd64) wheel for Python 2.7. Install by executing ``pip install *.whl``.

Matplotlib
----------
Install matplotlib from `Chris Gohlke's repository <http://www.lfd.uci.edu/~gohlke/pythonlibs/#matplotlib>`_. Use the win64 (amd64) wheel for Python 2.7. Install by executing ``pip install *.whl``.

wxWidgets
---------
Get the prebuilt wxWidgets 3.0 libraries from the wxPython site `here <http://www.wxpython.org/download.php#unstable>`_. Choose the "64-bit binaries for MSVC 9" in the "Windows Development Files" section. Install to your home directory. Rename the folder to "wx".

wxPython
--------
Get wxPython 3.0 from `here <http://www.wxpython.org/download.php#unstable>`_. Choose the 64-bit version for Python 2.7 in the "Windows Binaries" section. Install to your home directory(_not_ to C:\\Python27\\*!). Rename the folder to "wxPython".

FFTW
----
The latest fftw development files are avaible `here <http://www.stimfit.org/libs>`_. Download the zip file and extract to a folder called "fftw" in your home directory. Make sure that the files are at the top level of this fftw folder. These development files were prepared from the latest 64-bit fftw dlls from `here <http://fftw.org/install/windows.html>`_.

BLAS/LAPACK
-----------
Get the precompiled 64-bit BLAS/LAPACK libraries from `here <http://www.stimfit.org/libs>`_. They will be moved into a library folder within the stimfit tree later. They were obtained `from the official LAPACK site <http://www.netlib.org/clapack/LIB_WINDOWS/prebuilt_libraries_windows.html>`_ using the "nowrap" versions.

libbiosig
---------
Get the 64-bit biosig development files from `here <http://www.stimfit.org/libs>`_. They were prepared using `mxe <http://mxe.cc/>`_ and converted with Visual Studio's lib tool. Extract to a folder called "biosig" in your home directory. Make sure that the files and folders are directly in the top-level biosig folder.

===========
Build Tools
===========

SWIG
----
`Cygwin <http://www.cygwin.com>`_ has SWIG in its repositories. Otherwise, you can download it from `here <http://www.swig.org>`_. At any rate, make sure that the binary is located in C:\\cygwin64\\bin\\swig.exe.

git
---
`Cygwin <http://www.cygwin.com>`_ has git in its repositories. Otherwise, you can download it from `here <http://www.git-scm.org>`_.

nsis
----
Get nsis from `here <http://nsis.sourceforge.net/Download>`_.
