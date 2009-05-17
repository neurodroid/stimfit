.. test documentation master file, created by
   sphinx-quickstart on Tue May 12 19:47:15 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GNU/Linux Installation Guide
============================

:Release: |version|
:Date:    |today|

``Stimfit`` is an excellent programm for the analysis of electrophysiological signals. It waq developed by Christoph Schmidt-Hieber in the Physiology department at the University of Freiburg in Germany. Originally, ``Stimifit`` was written in Pascal by Peter Jonas and used to analyze excitatory currents evoked by extracellular electrical stimulation (Jonas et al., 1993). After that, a new ``Stimifit`` was rewritten in C/C++ and allowed a large amount of new functions, for example the analysis of miniature events or the measure of action potential latencies (Schmidt-Hieber et al., 2008). In the las version of ``Stimfit``, a Python interpreter was embedded, so that everybody can create and define its own routines and adapt them to its particular experiments. Very interestingly, the new ``Stimfit`` was developed with a platform independent GUI library [wxWidgets]_, and therefore can be run on GNU/Linux machines. However the GNU/Linux installation is not trivial.


Contents:

.. toctree::
   :maxdepth: 2

   requirement
   downloads
   building
   updates

