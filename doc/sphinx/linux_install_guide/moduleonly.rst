*******************************
Building the Python module only
*******************************

:Author: Christoph Schmidt-Hieber
:Date:    |today|

Building only the standalong Python file i/o module is fairly straightforward. First, you need a couple of libraries:

::

    $ sudo apt-get install build-essential mercurial libboost-dev python-dev python-numpy libhdf5-serial-dev

Then, you need the stimfit source code:

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
    $ ../../configure --enable-module

Then, build and install:

::

    $ make -j 4 # where 4 refers to the number of parallel build processes
    $ sudo make install

Finally, run python to test the module:

::

    >>> import stfio
    >>> print dir(stfio)




