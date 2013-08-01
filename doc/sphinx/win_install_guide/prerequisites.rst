*************
Prerequisites
*************

:Author: Christoph Schmidt-Hieber
:Date:    |today|

This document describes how to build `Stimfit <http://www.stimfit.org>`_ version |version| on Windows. I strongly recommend sticking to the suggested directory names. If for any reason you'd like to use different target directories, you'll have to update all the property sheets (Config.vsprops) in the Visual Studio solution.

Getting all the prerequisites takes about an hour, but only needs to be completed once. Building the full solution takes about 3 minutes.

=======================
Visual C++ Express 2008
=======================

The official Windows version of Python 2.7 was built with Visual Studio 2008. We therefore have to use Visual C++ 2008 so that we link against the same C runtime library. Luckily, there's a free version called Visual C++ 2008 Express that you can get directly from `here <http://go.microsoft.com/?linkid=7729279>`_


=========
Libraries
=========

HDF5
----
Get the HDF5 libraries from `here <http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.9/bin/windows/>`_. Use HDF5189-win32-vs9-shared.zip. There are no VS2008-prebuilt binaries for HDF5 >= 1.8.10 unfortunately. Extract the zip file, and then install to a folder called "hdf5" in your home directory (e.g. C:\\Users\\username) using the extracted executable.

Boost
-----
Get the `Boost C++ libraries <http://www.boost.org>`_. Move the extracted folder to your home directory and rename to "boost". If you used the zip file, you might have to move the first folder (called something like boost_1_54_0, or whatever the current version is) one directory up and rename it to "boost". At any rate, you should check that you have the boost header files (\*.hpp) sitting in C:\\Users\\username\\boost\\boost\\\*.hpp. 

Python
------
Download and install `Python 2.7 <http://www.python.org>`_. Make sure to install it for all users so that it ends up in C:\\Python27

NumPy
-----
Install NumPy from `here <http://sourceforge.net/projects/numpy/files/NumPy/>`_. Use the win32 "superpack" for Python 2.7.

Matplotlib
----------
Install Matplotlib from `here <http://matplotlib.org/downloads.html>`_. Use the win32 version for Python 2.7.

wxWidgets
---------
Get the prebuilt wxWidgets 2.9 libraries from the wxPython site `here <http://www.wxpython.org/download.php#unstable>`_. Choose the "32-bit binaries for MSVC 9" in the "Windows Development Files" section. Install to your home directory. Rename the folder to "wx".

wxPython
--------
Get wxPython 2.9 from `here <http://www.wxpython.org/download.php#unstable>`_. Choose the 32-bit version for Python 2.7 in the "Windows Binaries" section. Install to your home directory(_not_ to C:\\Python27\\*!). Rename the folder to "wxPython".

FFTW
----
Get the latest 32-bit fftw dlls from `here <http://fftw.org/install/windows.html>`_. Extract to a folder called "fftw" in your home directory. As instructed on the fftw install page, open the Visual Studio Command Prompt that is in the Visual Studio Tools section of the Program menu. cd into the fftw folder and type:

``C:\Users\username\fftw> lib /def:libfftw3-3.def``

``C:\Users\username\fftw> lib /def:libfftw3f-3.def``

``C:\Users\username\fftw> lib /def:libfftw3l-3.def``

===========
Build Tools
===========

SWIG
----
`Cygwin <http://www.cygwin.com>`_ has SWIG in its repositories. Otherwise, you can download it from `here <http://www.swig.org>`_. At any rate, make sure that the binary is located in C:\\cygwin\\bin\\swig.exe.

git
---
`Cygwin <http://www.cygwin.com>`_ has git in its repositories. Otherwise, you can download it from `here <http://www.git-scm.org>`_.

nsis
----
Get nsis from `here <http://nsis.sourceforge.net/Download>`_.
