*************************************
Building the stfio Python Module Only
*************************************

:Author: Yueqi Wang
:Date:    |today|

This documentation describes how to install the standalone Python file i/o module for Mac OS.

For details on how to use the *stfio* module, see :doc:`/stfio/index`.

Installing stfio for MacPorts Python
====================================
Download and install MacPorts from `here <http://www.macports.org>`_.

::

  $ sudo port install py27-stfio py34-stfio
  
Using this method, the stfio module can only be imported using MacPorts Python.

  
Installing stfio for non-MacPorts Python distributions
======================================================
Note: The officially-supported *stfio* module is not currently available for non-MacPorts Python distributions. You can either link the MacPorts *stfio* to your favorite Python distribution or build the module from source. 

Linking the MacPorts stfio module to your Python PATH
-----------------------------------------------------
Note: This works most of the times, but is not always recommended. 

First, install the *stfio* module using MacPorts as described above. Then find the path where *stfio* is installed 

::

  $ port content py27-stfio
  
The install path will look something like this: */opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/stfio/*

There are two ways of adding the MacPorts stfio module to the Python PATH of your non-MacPorts Python distribution:

1) open *$HOME/.bash_profile*, add the following line to the file, and save the file:

::

    export PATHONPATH=$PYTHONPATH:/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/stfio/
  
- Then, update the bash profile:

::

    $ source $HOME/.bash_profile
  
2) Alternatively, you can soft link the MacPorts *stfio* module folder to the *site-packages* folder of your favorite Python distribution. The following example will link MacPorts *stfio* to Anaconda Python 2.7:

:: 

  $ ln -s /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/stfio/ $HOME/anaconda/lib/python2.7/site-packages/
  
Finally, run python to test the module, as described in :doc:`/stfio/index`.



Building the stfio module from source
-------------------------------------

Install `Homebrew <https://brew.sh/>`_

::

  $ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Install dependencies using Homebrew

::

  $ brew update
  $ brew install boost
  $ brew install boost-python
  $ brew install autoconf
  $ brew install automake
  $ brew install libtool
  $ brew install fftw
  $ brew install homebrew/science/hdf5
  $ brew install biosig
  
Now download the stimfit source code:

::

  $ cd $HOME
  $ git clone https://github.com/neurodroid/stimfit.git

Next, Generate the build system and configure:

::

  $ cd stimfit
  $ ./autogen.sh
  $ mkdir build
  $ mkdir build/module
  $ cd build/module
  $ ../../configure --enable-module --with-biosig

If you want to install the stfio module for a non-default Python distribution, such as **Anaconda Python**, use the argument *--prefix=* to specify the installation path as where your favorite Python distribution is installed. The following example will install stfio for Anaconda Python 2.7:

::
  
  $ ../../configure --enable-module --prefix=$HOME/anaconda/
 
If using virtual environment, try something like this: 

::

  $ ../../configure --enable-module --prefix=$HOME/anaconda/envs/py36

Then, build and install:

::

  $ make -j 4
  $ make install

Finally, run python to test the module, as described in :doc:`/stfio/index`.

