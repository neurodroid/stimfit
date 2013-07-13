*******************************
Building the Python module only
*******************************

:Author: Christoph Schmidt-Hieber
:Date:    |today|

Building only the standalone Python file i/o module is fairly straightforward. First, you need a couple of libraries:

::

    $ sudo apt-get install build-essential mercurial libboost-dev python-dev python-numpy libhdf5-serial-dev swig

Then, you need the `Stimfit <http://www.stimfit.org>`_ source code:

::

    $ cd $HOME
    $ git clone https://code.google.com/p/stimfit stfio

This will download the code to a directory called *stfio*.

Next, you need to generate the build system:

::

    $ cd $HOME/stfio
    $ ./autogen.sh

Now you can configure. I strongly recommend building in a separate directory.

::

    $ cd $HOME/stfio
    $ mkdir build
    $ cd build
    $ mkdir module
    $ cd module
    $ ../../configure --enable-module

Then, build and install:

::

    $ make -j 4 # where 4 refers to the number of parallel build processes
    $ sudo make install

Finally, run python to test the module, as described in :doc:`/stfio/index`.
