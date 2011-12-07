*******************************
Building the Python module only
*******************************

:Author: Christoph Schmidt-Hieber
:Date:    |today|

Building only the standalone Python file i/o module is fairly straightforward. First, you need a couple of libraries:

::

    $ sudo apt-get install build-essential mercurial libboost-dev python-dev python-numpy libhdf5-serial-dev swig

Then, you need the Stimfit source code:

::

    $ cd $HOME
    $ git clone https://code.google/com/p/stimfit stimfiy

This will download the code to a directory called *stimfit*.

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
    $ ../../configure --enable-module

Then, build and install:

::

    $ make -j 4 # where 4 refers to the number of parallel build processes
    $ sudo make install

Finally, run python to test the module, as described in :doc:`/stfio/index`.
