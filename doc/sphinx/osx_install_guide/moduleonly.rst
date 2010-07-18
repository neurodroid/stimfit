*******************************
Building the Python module only
*******************************

:Author: Christoph Schmidt-Hieber
:Date:    |today|

Building only the standalone Python file i/o module is fairly straightforward. First, you need XCode and MacPorts as described in :doc:`/osx_install_guide/prerequisites`, with some modifications to the MacPorts libraries:

The following selection of MacPorts libraries should be sufficient since we're not building the full program:

::

    $ sudo port install mercurial boost python26 py26-numpy hdf5-18 swig swig-python autoconf automake libtool

On OS X 10.4, you might have to do

::

    $ sudo port install atlas py26-numpy +no_gcc43

We don't need all the other stuff (dylibbundler, wxWidgets etc.) for the standalone module.

Next, get the stimfit source code:

::

    $ cd $HOME
    $ hg clone https://stimfit.googlecode.com/hg/ stimfit 

This will grab the source code into $HOME/stimfit, where $HOME is your home directory.

Next, you need to generate the build system:

::

    $ cd $HOME/stimfit
    $ ./autogen.sh

Now you can configure. I strongly recommend building in a separate directory.

::

    $ cd $HOME/stimfit
    $ mkdir build
    $ cd build
    $ mkdir module
    $ cd module
    $ ../../configure --enable-module CXXFLAGS="-I/opt/local/include" LDFLAGS="-L/opt/local/lib" PYTHON=/opt/local/bin/python2.6

Then, build and install:

::

    $ make -j 4 # where 4 refers to the number of parallel build processes
    $ sudo make install

Finally, run python to test the module, as described in :doc:`/stfio/index`.


