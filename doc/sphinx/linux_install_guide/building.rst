****************************************
Building wxWidgets, wxPython and Stimfit
****************************************

Building wxWidgets and wxPython
===============================

:Author: Jose Guzman
:Date:  |today|

To build wxWidgets and wxPython, follow the build instructions found `here <http://www.wxpython.org/builddoc.php>`_

Building Stimfit
=================

Go to the stimfit directory (in our example $HOME/stimfit) and type:

::

    $ ./autogen.sh

to generate the configure script. Remember that we need Autoconf, Automake and LibTool to use autogen. After that, you can call it with

::

    $ ./configure --enable-python

The configure script has some additional options. For example, we may want to use `IPython <http://www.scipy.org>`_  instead of the default embedded python shell with the option **---enable-ipython**  (note that the `IPython <http://www.scipy.org>`_ shell is only available under GNU/Linux and it is still very experimental). 

Finally, after running configure, you can type

::

    $ make 

and finally type as root

::

    $ sudo make install
