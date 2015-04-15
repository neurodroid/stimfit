****************
Building Stimfit
****************

:Author: Christoph Schmidt-Hieber
:Date:  |today|

========================
Installing with MacPorts
========================

Download and install MacPorts from `here <http://www.macports.org>`_.

If you just want to install Stimfit and/or the stfio module, do

::

    sudo port install stimfit py27-stfio py34-stfio

If you want to build stimfit from source, you'll first need to install git:

::

    sudo port install git-core

Get the `Stimfit <http://www.stimfit.org>`_ source from the git repository

::

    git clone https://code.google.com/p/stimfit/
    
Edit the MacPorts sources configuration file (/opt/local/etc/macports/sources.conf) and place the following line before the one that reads 

``rsync://rsync.macports.org/release/tarballs/ports.tar [default]`` 

(change the path to the `Stimfit <http://www.stimfit.org>`_ directory accordingly).

``file:///${STIMFITDIR}/stimfit/macosx/macports``
    
.. note::

    using the root of your account as opposed to a subdirectory (ie, Documents or Downloads folders) may prevent permissions access errors when building.


Next, go to the `Stimfit <http://www.stimfit.org>`_ macports directory

::

    cd ${STIMFITDIR}/macosx/macports
    
Add the local ports file to MacPorts by running the following command at this location

::

    sudo portindex
    

When finished, you can now build `Stimfit <http://www.stimfit.org>`_ in MacPorts

::

    sudo port install stimfit
    
MacPorts will download and install various dependencies, and then attempt to build `Stimfit <http://www.stimfit.org>`_ from source.
