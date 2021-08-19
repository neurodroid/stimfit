.. test documentation master file, created by
   sphinx-quick start on Tue May 12 19:47:15 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OS X Build Guide
================

:Author: Christoph Schmidt-Hieber
:Release: |version|
:Date:    |today|

If you just want to install Stimfit and/or the stfio module, download and install MacPorts from `here <http://www.macports.org>`_, then type into a terminal:

::

    sudo port install stimfit py27-stfio py34-stfio

Building `Stimfit <http://www.stimfit.org>`_ from scratch requires you to install `Xcode <http://developer.apple.com/tools/xcode/>`_, `MacPorts <http://www.macports.org/>`_ and a couple of libraries. Note that this may take several hours. There is experimental support for Stimfit in `Homebrew <http://brew.sh>`_, but currently it does not install a full version of Stimfit.

Contents:

.. toctree::
   :maxdepth: 2

   building
   moduleonly
