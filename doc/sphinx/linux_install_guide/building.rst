****************************************
Building wxWidgets, wxPython and Stimfit
****************************************

Building wxWidgets
==================

:Author: Jose Guzman
:Date:  |today|

Once we have unpacked the wxWidgets sources in $HOME/wxWidgets, we can create build directory called build2.9 to compile the files: 

::

    $ cd $HOME/wxWidgets
    $ mkdir build2.9 


Inside the build2.9 directory you can run the configure script with the following options to create the Makefiles: 

::

    $ ../configure --with-gtkprint --without-gnomeprint --with-opengl --enable-calendar --enable-graphics_ctx

We type *--with-gtkprint --without-gnomeprint* because we need latest version of wxWidgets (which requires itself gtk) to print, and not gnome. Note that you call the script configure from $HOME/wxWidgets but the make file will be created in $HOME/wxWidgets/build2.9/

.. note::
    If you find the following error:

        Could not run GTK+ test program, checking why...
        The test program failed to compile or link. See the file config.log for the
        exact error that occurred. This usually means GTK+ is incorrectly installed.
        configure: error:
        The development files for GTK+ were not found. For GTK+ 2, please
        ensure that pkg-config is in the path and that gtk+-2.0.pc is
        installed. For GTK+ 1.2 please check that gtk-config is in the path,
        and that the version is 1.2.3 or above. Also check that the
        libraries returned by 'pkg-config gtk+-2.0 --libs' or 'gtk-config
        --libs' are in the LD_LIBRARY_PATH or equivalent.

        This is because to compile GTK+ applications (like /wxWidgets) we need the '''pkg-config''' tool properly configured. Pkg-config will give us the necessary libraries and include files that we need when we compile GTK+ applications. We simply need to set the path to the directory where file gtk+-2.0.pc is found (generally /usr/lib/pkgconfig/). This file contains the instructions to use the libraries and headers dependencies of GTK+

        $  export PKG_CONFIG_PATH=/usr/lib/pkgconfig/

        You can check if the variable was just typing

        $ echo $PKG_CONFIG_PATH

        which will give /usr/lib/pkgconfig/. 
        
If everything was OK, you will see the following message after running configure. 

::

    Configured wxWidgets 2.9.0 for i686-pc-linux-gnu
        
Now we just type make and after that, run make install as root. All this inside $HOME/wxWidgets/build2.9/

::

    $ make 
    $ sudo make install 

Building wxPython
=================

Now we will build xwPython. In the same directory where we downloaded the sources for wxPython ($HOME/wxPython) you  just type:

::

    $ python setup.py build_ext --inplace

You will need first to have installed the python development libraries in your system (if not just type as root *apt-get install python-dev*). You will need the same version of gcc and g++ (in our example both versions are 4.2.4). After that just as root type:

::

    $ python setup.py install

With that, you have built and installed wxWidgets and wxPython. We now only need to build Stimfit.

Building Stimfit
=================

Go to the stimfit directory (in our example $HOME/stimfit) and type:

::

    $ ./autogen.sh

to generate the configure script. After that, you can call it with

::

    $ ./configure --enable-python

The configure script has some additional options. For example, we may want to use `IPython <http://www.scipy.org>`_  in stead of the default embedded python shell with the option **---enable-ipython**  (note that the `IPython <http://www.scipy.org>`_ shell is only available under GNU/Linux and it is still very experimental). 

Finally, after running configure , you can type

::

    $ make 

and finally type as root

::

    $ sudo make install
