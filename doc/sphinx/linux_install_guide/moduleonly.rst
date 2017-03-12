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

Remember to add the argument *--with-biosig* to the configure script 
if you want to have extra biomedical fileformats for stfio. 

If you want to install the *stfio* module for a non-default Python distribution, such as **Anaconda Python**, use the argument *--prefix=* to specify the installation path as the *site-packages* folder of your favorite Python distribution. The following example will install *stfio* for Anaconda Python 2.7:

::

    $ ../../configure --enable-module --prefix=$HOME/anaconda/lib/python2.7/site-packages

Then, build and install:

::

    $ make -j 4 # where 4 refers to the number of parallel build processes
    $ sudo make install

Finally, run python to test the module, as described in :doc:`/stfio/index`.
