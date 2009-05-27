.. test documentation master file, created by
   sphinx-quick start on Tue May 12 19:47:15 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The Stimfit book of spells
==========================


:Author: Jose Guzman
:Release: |version|
:Date:  |today|

This document collects answers to some questions like "How do I make ... in ``Stimfit`` with python?". Though much of the material can be easily found in the ``Stimfit`` manual, the examples provided here are a good way for the casual user to start using Python in ``Stimfit``. 

It assumes a basic knowledge of the embedded Python shell of ``Stimfit``. Some Python knowledge and a substantial profiency in ``Stimfit`` are recommendable. Please note that this is not a Python manual, but a way to use Python for some basic analysis tasks provided with ``Stimfit``. For a detailed Python manual, we encourage the user to visit the official Python documentation on the [Python-website]_ and going through the ``Stimfit`` manual.

In this document we assume that functions will be saved in a file called myfile.py (where myfile can be any custom made), and imported in the embedded Python shell with:

::

    >>> import myfile
    
For example, a function called rmean() would be saved in **myfile.py** and after importing, called in this way:

::

    >>> myfile.rmean(10)



Contents:

.. toctree::
   :maxdepth: 2

   amplitudes
   resistances
   runningmean
