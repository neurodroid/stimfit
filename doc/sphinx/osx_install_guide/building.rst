****************
Building Stimfit
****************

:Author: Christoph Schmidt-Hieber
:Date:  |today|

=========
Configure
=========

Check out the latest source code from the Google code repository:

::

    $ hg clone https://stimfit.googlecode.com/hg/ stimfit

Generate the build system:

::

    $ cd stimfit
    $ ./autogen.sh

It's a good idea to configure and build outside of the source tree:

::

    $ mkdir bld
    $ cd bld

To configure a 32bit build on Snow Leopard, I use the following bash script:

::

    #! /bin/bash

    arch_flags="-arch i386 -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5"
    LOGIN=cs

    ../configure --enable-python --with-wx-config=~/wxbin/bin/wx-config \
        CXX="/usr/bin/g++-4.0" CC="/usr/bin/gcc-4.0" LD="/usr/bin/g++-4.0" \
        CPPFLAGS="-DH5_USE_16_API" CFLAGS="$arch_flags" CXXFLAGS="$arch_flags" \
        LDFLAGS="$arch_flags" \
        -I/Users/$LOGIN/wxPython/include -I/opt/local/include \
        -headerpad_max_install_names \
        -L/Users/$LOGIN/wxbin/lib -L/opt/local/lib -L/usr/lib

For a 64 bit build, you will need a 64bit version of wxWidgets built against the Cocoa API in the previous step. You can then configure as follows:

::

    #! /bin/bash

    LOGIN=cs

    ../configure --enable-python --with-wx-config=~/wxbin/bin/wx-config \
        CPPFLAGS="-DH5_USE_16_API" \
        -I/Users/$LOGIN/wxPython/include -I/opt/local/include \
        -headerpad_max_install_names \
        -L/Users/$LOGIN/wxbin/lib -L/opt/local/lib -L/usr/lib

On OS X 10.4, you can use

::
    
    LOGIN=cs
    $ ../configure --enable-python \
          --with-wx-config=/Users/$LOGIN/wxWidgets/bld/wx-config \
          CXXFLAGS=-I/Users/$LOGIN/wxPython/include \
          -I/opt/local/include \
          LDFLAGS=-headerpad_max_install_names \
          -L/Users/$LOGIN/wxWidgets/bld/lib -L/opt/local/lib -L/usr/lib \
          -lsz -lz PYTHON=/opt/local/bin/python

=================
Build and install
=================

You can then build using

::

    $ make -jN # where N is the number of parallel builds

Creating an application bundle can be done using the following script:

::

    #! /bin/bash

    LOGIN=cs

    sudo chown -R $LOGIN:staff stimfit.app

    WX_CONFIG=/Users/$LOGIN/wxbin/bin/wx-config
    WXPY_DIR=/Users/$LOGIN/wxPython
    WXPY_INSTALL_DIR=/Users/$LOGIN/wxPython/dummy-install/lib/python2.6/site-packages

    make stimfit.app
    mkdir -p ./stimfit.app/Contents/Frameworks/stimfit

    cp -R ${WXPY_INSTALL_DIR}/wx* ./stimfit.app/Contents/Frameworks/
    rsync -rtuvl `${WX_CONFIG} --exec-prefix`/lib/libwx*.dylib ./stimfit.app/Contents/libs/
    sudo cp -v ./src/stfswig/.libs/libstf.0.dylib /usr/local/lib/libstf.0.dylib
    rm -f ./stimfit.app/Contents/Frameworks/stimfit/_stf.so
    cp -v ./src/stfswig/.libs/libstf.0.dylib ./stimfit.app/Contents/Frameworks/stimfit/_stf.so
    rm -f ./stimfit.app/Contents/libs/libstf.0.dylib
    dylibbundler -of -b -x ./stimfit.app/Contents/MacOS/stimfit -d ./stimfit.app/Contents/libs/

    find ./stimfit.app  -name "*.dylib" -exec dylibbundler -of -b -x '{}' -d ./stimfit.app/Contents/libs/ \;
    find ./stimfit.app  -name "*.so" -exec dylibbundler -of -b -x '{}' -d ./stimfit.app/Contents/libs/ \;
    sudo rm /usr/local/lib/*stf*

    cp -v ../../src/stfswig/*.py ./stimfit.app/Contents/Frameworks/stimfit/

========================
Installing with MacPorts
========================

Download and install MacPorts (http://www.macports.org), and then run the following command to install git

::

    sudo port install git-core

Get the stimfit source from the git repository

::

    git clone https://code.google.com/p/stimfit/
    
Edit the MacPorts sources configuration file (/opt/local/etc/macports/sources.conf) and place the following line before the one that reads "rsync://rsync.macports.org/release/tarballs/ports.tar [default]" (change the path to the stimfit directory accordingly).

::

    file:///${STIMFITDIR}/stimfit/macosx/macports
    
NOTE: using the root of your account as opposed to a subdirectory (ie, Documents or Downloads folders) may prevent permissions access errors when building.

Next, go to the Stimfit macports directory

::

    cd ${STIMFITDIR}/macosx/macports
    
Add the local ports file to MacPorts by running the following command at this location

::

    sudo portindex
    
When finished, you can now build Stimfit in MacPorts

::

    sudo port install stimfit
    
MacPorts will download and install various dependencies, and then attempt to build Stimfit from source.
