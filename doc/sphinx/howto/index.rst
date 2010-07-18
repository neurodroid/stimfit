.. test documentation master file, created by
   sphinx-quick start on Tue May 12 19:47:15 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The Stimfit book of spells
==========================


:Author: Jose Guzman
:Release: |version|
:Date:  |today|

This document collects answers to some questions like "How do I make ... in `Stimfit <http://www.stimfit.org>`_ with python?". Though much of the material can be easily found in the :doc:`/manual/index`, the examples provided here are a good way for the casual user to start using Python in `Stimfit <http://www.stimfit.org>`_. 

It assumes a basic knowledge of the embedded Python shell of `Stimfit <http://www.stimfit.org>`_. Some Python knowledge and a substantial proficiency in `Stimfit <http://www.stimfit.org>`_ are recommendable. Please note that this is not a Python manual, but a way to use Python for some basic analysis tasks provided with `Stimfit <http://www.stimfit.org>`_. For a detailed Python manual, we encourage the user to visit the official Python documentation on the [Python-website]_ and to read carefully the :doc:`/manual/index`.

The functions described along this document are available in your current `Stimfit <http://www.stimfit.org>`_ version. To make use of them you have simply to type the following line in the `Stimfit <http://www.stimfit.org>`_ embedded Python shell:

::

    >>> import spells 
    
After that, functions can be called with the dot notation (i.e just typing **spells** before the function) For example, if we want to call the function rmean() we  would simply do it in this way:

::

    >>> spells.rmean(10)

Finally, the contents of this document are organized with increased level of complexity, assuming some of the last chapters concepts and topics described in the first chapters. Thus, we encourage the newcomer to follow the order exposed below or to visit the section **code commented** in previous chapters. 

Contents:

.. toctree::
   :maxdepth: 2
    
   resistances
   runningmean
   amplitudes
   cuttingtraces
   apcounting
   introclass
   latencies
