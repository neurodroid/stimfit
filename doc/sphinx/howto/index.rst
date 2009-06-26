.. test documentation master file, created by
   sphinx-quick start on Tue May 12 19:47:15 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The Stimfit book of spells
==========================


:Author: Jose Guzman
:Release: |version|
:Date:  |today|

This document collects answers to some questions like "How do I make ... in ``Stimfit`` with python?". Though much of the material can be easily found in the :doc:`/manual/index`, the examples provided here are a good way for the casual user to start using Python in ``Stimfit``. 

It assumes a basic knowledge of the embedded Python shell of ``Stimfit``. Some Python knowledge and a substantial proficiency in ``Stimfit`` are recommendable. Please note that this is not a Python manual, but a way to use Python for some basic analysis tasks provided with ``Stimfit``. For a detailed Python manual, we encourage the user to visit the official Python documentation on the [Python-website]_ and going through the :doc:`/manual/index`.

In this document we assume that functions will be saved in a file called myfile.py (where myfile can be any custom made), and imported in the embedded Python shell with:

::

    >>> import myFile
    
For example, a function called rmean() would be saved in **myFile.py** and after importing, called in this way:

::

    >>> myFile.rmean(10)

Finally, the contents of this document are organized with increased level of complexity, assuming some of the last chapters concepts and topics described in the first chapters. Thus, we encourage the newcomer to follow the order exposed bellow or to visit the section **code commented** in previous chapters. 

Contents:

.. toctree::
   :maxdepth: 2
    
   resistances
   runningmean
   amplitudes
   cuttingtraces
   apcounting
