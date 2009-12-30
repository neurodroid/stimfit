**********************************
Recommendations for system updates
**********************************

:Author: Jose Guzman
:Date:    |today|

If you plan to upgrade/update your system after the installation of `Stimfit <http://www.stimfit.org>`_, you should be sure that the versions of libraries of wxPython and wxWidgets are allways the same. If not, you will run into troubles (for example, you can get the message "Segmentation fault" after running `Stimfit <http://www.stimfit.org>`_ in your terminal).

We can hold some packages in our Debian-based system with the package manager command dpkg. In this way, you will be sure that updating your system (for example with apt-get upgrade) will not affect your `Stimfit <http://www.stimfit.org>`_ installation. To hold a package you can type: 

::

    >>> echo package hold|dpkg --set-selections

where *package* is the package name to hold.
If you want to restore the update of a package (to take a package off Hold), just type:

::

    >>> echo package install|dpkg --set-selections

To see a list of packages on hold, we use:

::

    >>> dpkg --get-selections | grep hold

====================================
Packages which should not be updated
====================================

I collected a list of packages that you should hold in your system. This avoids troubles with `Stimfit <http://wwww.stimfit.org>`_. If you find any other packages with are non-compatible with `Stimfit <http://www.stimfit.org>`_, please report it to `Stimfit <http://www.stimfit.org>`_ developers and users in the *Stimfit users mail list*

    * python-wxgtk2.8
    * python-wxgtk2.6
