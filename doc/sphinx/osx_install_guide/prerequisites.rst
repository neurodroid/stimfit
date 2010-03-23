*************
Prerequisites
*************

:Author: Christoph Schmidt-Hieber
:Date:    |today|

This document describes how to build `Stimfit <http://www.stimfit.org>`_ version |version| on OS X. The installation was tested on OS X Tiger (10.4, powerpc) Leopard (10.5, i386) and Snow Leopard (10.6, x86_64).

=====
Xcode
=====

You'll need `Apple's Xcode Developer Tools <http://developer.apple.com/tools/xcode/>`_ (version 3.2.1 or later for Snow Leopard, 3.1.4 or later for Leopard, or 2.5 for Tiger), found at the `Apple Developer Connection site <http://connect.apple.com/>`_ or on your Mac OS X installation CDs/DVD. Ensure that the optional components for command line development are installed ("Unix Development" in the Xcode 3.x installer).

========
MacPorts
========

I recommend using `MacPorts <http://www.macports.org>`_ to get all the build dependencies. `Fink <http://finkproject.org>`_ might work as well, but I haven't tested it.

Once you've installed MacPorts, you can use it to build some required libraries:

::

    $ sudo port install subversion mercurial boost fftw-3 python26 py26-numpy hdf5-18 swig swig-python

On OS X 10.4, you might have to do

::

    $ sudo port install atlas py26-numpy +no_gcc43

if you get failures while building NumPy.

MacPorts will build everything from the sources, so it might be a good idea to run this overnight if you have a slower machine.

============
dylibbundler
============

`dylibbundler <http://macdylibbundler.sourceforge.net/>`_ is required to create an application bundle. The download link is well hidden on the sourceforge site, so here's a direct link: 

http://downloads.sourceforge.net/macdylibbundler/dylibbundler0.3.1.zip

Follow the installation instructions that come with the package.

=========
wxWidgets
=========

Unfortunately, `wxWidgets <http://www.wxWidgets.org>`_ and `wxPython <http://www.wxPython.org>`_ need to be built from source to get the latest development version.

You can get the source from their subversion repositories:

::

    $ svn checkout http://svn.wxwidgets.org/svn/wx/wxWidgets/trunk wxWidgets -r63598
    $ svn checkout http://svn.wxwidgets.org/svn/wx/wxPython/trunk wxPython -r63598

It's a good idea to configure and build outside of the source tree:

::

    $ cd wxWidgets
    $ mkdir bld
    $ cd bld

To build 32bit libraries of wxWidgets on Snow Leopard that can be used on previous OS X versions as well, I use the following bash script:

::

    #! /bin/bash

    arch_flags="-arch i386 -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5"

    ../configure CC=gcc-4.0 CXX=g++-4.0 LD=g++-4.0 \
          CFLAGS="$arch_flags" CXXFLAGS="$arch_flags" \
          LDFLAGS="$arch_flags" OBJCFLAGS="$arch_flags" \
          OBJCXXFLAGS="$arch_flags" \
          --prefix=~/wxbin \
          --with-osx_carbon \
          --with-opengl \
          --enable-geometry \
          --enable-graphics_ctx \
          --enable-sound \
          --with-sdl \
          --enable-mediactrl \
          --enable-std_string

This will build wxWidgets against the Carbon API, which is the recommended way because the Cocoa version is still very unstable. If you definitely want 64bit libraries, you could try:

::

    ../configure \
          --prefix=~/wxbin \
          --with-osx_cocoa \
          --with-opengl \
          --enable-geometry \
          --enable-graphics_ctx \
          --enable-sound \
          --with-sdl \
          --enable-mediactrl \
          --enable-std_string

See the `OS X build instructions <http://wiki.wxwidgets.org/Development:_wxMac#Building_under_10.6_Snow_Leopard>`_ for further details, but note that they refer to wxWidgets 2.8.

On OS X 10.4, I use:

::

    $ ../configure --prefix=~/wxbin \
            --with-osx_carbon 
            --with-opengl \
            --enable-geometry \
            --enable-graphics_ctx \
            --enable-sound \
            --with-sdl \
            --enable-mediactrl \
            --enable-std_string

Once you've successfully completed the configuration, you can build the libraries:

::

    $ make -jN # where N is the number of parallel builds
    $ make install

Finally, you should set the WXWIN environment variable your wxWidgets path, for instance by adding this line to your ~/.profile 

::

    export WXWIN=/Users/LOGIN/wxWidgets # where LOGIN is your login name

========
wxPython
========

In the wxPython directory, you can build and install using:

::

    $ python setup.py build_ext --inplace WXPORT=osx_carbon WX_CONFIG=~/wxbin/bin/wx-config
    $ python setup.py install WXPORT=osx_carbon WX_CONFIG=~/wxbin/bin/wx-config

Replace osx_carbon with osx_cocoa if you've built the 64bit version of wxWidgets.
I prefer to install this locally to avoid interfering with any system-wide wx installations.

