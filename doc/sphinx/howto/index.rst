.. test documentation master file, created by
   sphinx-quick start on Tue May 12 19:47:15 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Howto Python in Stimfit 
=======================

:Release: |version|
:Date:    |today|

This document collects the answers to some questions like "How do I make ... in ``Stimfit`` with python?". Though much of the material can be easily found in the ``Stimfit`` manual, the examples provided here are a good way for the casual user to start using Python in ``Stimfit``. 

It assumes a basic knowledge of the embeded python shell of ``Stimfit``. A previous running Python knowledge and a substantial profiency in ``Stimfit`` are recommendable. Please, note that this is not a Python manual, but a way to use python for some basic analysis tasks with provided with ``Stimfit``. For a detailed Python manual, we encourage the user to visit official Python documentation in the [Python-website]_ and going through the ``Stimfit`` manual.

In this document we assumed that the given function of interested will be saved in a file called myfile.py, and imported in the embeded python shell with:

::

    >>> import myfile
    
For example, a function called rmean() would be saved in myfile.py and after importing, called in this way:

::

    >>> myfile.rmean(10)



Contents:

.. toctree::
   :maxdepth: 2

   resistances
   runningmean
