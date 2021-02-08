*******************************
Building the Python module only
*******************************

:Author: Christoph Schmidt-Hieber
:Date:    |today|

Building only the standalone Python file i/o module is fairly straightforward. First, you need a couple of libraries:

::

    $ sudo apt-get install build-essential git libboost-dev python-dev python-numpy libhdf5-serial-dev swig

Then, you need the `Stimfit <http://www.stimfit.org>`_ source code:

::

    $ cd $HOME
    $ git clone https://github.com/neurodroid/stimfit.git

It will download the code to a directory called *stimfit*.

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

We recommend to use BioSig to read extra biomedical fileformats (see :ref:`BioSigBuild`) :

::

    $ ../../configure --enable-module --with-biosiglite

===================================================
Building stfio for non-default Python distributions
===================================================

To install the *stfio* module in distributions such as **Anaconda Python**, use the argument *--prefix=* to specify the path where the Python distribution is installed. For example, to install *stfio* for Anaconda Python 2.7 use:

::

    $ ../../configure --enable-module --prefix=$HOME/anaconda/

If using virtual environment, try something like this:

::

    $ ../../configure --enable-module --prefix=$HOME/anaconda/envs/py36

Other Python versions are also possible. For example, to install the module in your local Python version, you could use:

::

    $ ../../configure --enable-module --prefix=$HOME/.local/lib/python2.7

Then, build and install:

::

    $ make -j 4 # where 4 refers to the number of parallel build processes
    $ sudo make install

Finally, run python to test the module, as described in :doc:`/stfio/index`.


==========================
Building stfio using cmake
==========================

__Beta__

In addition to dependencies listed about, you need to install the followings.

1. cmake version 3.12 or higher.
2. python-wheel. Use `apt get install python3-wheel` or `python3 -m pip install wheel --user`.

Build instructions.

::
    
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make -j4

You should have a `whl` file in the `build` directory. You can install it using
`pip`

::

    $ python3 -m pip install stimfit*.whl
