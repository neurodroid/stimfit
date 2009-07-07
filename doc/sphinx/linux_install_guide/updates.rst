**********************************
Recommendations for system updates
**********************************

:Author: Jose Guzman
:Date:    |today|

If you plan to upgrade/update your system after the installation of `Stimfit <http://www.stimfit.org>`_, you should be sure that the versions of libraries of wxpython and wxwidgets remains the same. If not, you will find the message "Segmentation fault" after running `Stimfit <http://www.stimfit.org>`_ in your terminal.

::

    >>> echo package hold|dpkg --set-selections

To take a package off Hold

::

    >>> echo package install|dpkg --set-selections

To list packages on hold:

::

    >>> dpkg --get-selections | grep hold

I collect here a list of debian packages that you should hold 

    * python-wxgtk2.8
    * python-wxgtk2.6
