****************************************
Building wxWidgets, wxPython and Stimfit
****************************************

Building wxWidgets
==================

.. sectionauthor:: Jose Guzman <>

Assuming that we created a directory in /usr/local/wxWidgets to build the sources, we will create a build directory inside
::

    >>> cd /usr/local/wxWidgets
    >>> mkdir build2.9 

After that enter the build2.9 directory and run the configure script to create the make file as follows: 
::

    >>> ./configure --with-gtkprint --without-gnomeprint --with-opengl --enable-calendar --enable-graphics_ctx

We type --with-gtkprint --without-gnomeprint because we need latest version of wxWidgets (which requires itself gtk) to print. Note that you call the script configure from /usr/local/wxWidgets but the make file will be created in /usr/local/wxWidgets/build2.9/

.. note::
    If you find the following error:

    >>>  *** Could not run GTK+ test program, checking why...
        *** The test program failed to compile or link. See the file config.log for the
        *** exact error that occurred. This usually means GTK+ is incorrectly installed.
        configure: error:
        The development files for GTK+ were not found. For GTK+ 2, please
        ensure that pkg-config is in the path and that gtk+-2.0.pc is
        installed. For GTK+ 1.2 please check that gtk-config is in the path,
        and that the version is 1.2.3 or above. Also check that the
        libraries returned by 'pkg-config gtk+-2.0 --libs' or 'gtk-config
        --libs' are in the LD_LIBRARY_PATH or equivalent.

        This is because to compile GTK+ applications (like /wxWidgets) we need the '''pkg-config''' tool properly configured. Pkg-config will give us the necessary libraries and include files that we need when we compile GTK+ applications. We simply need to set the path to the directory where file gtk+-2.0.pc is found (generally /usr/lib/pkgconfig/). This file contains the instructions to use the libraries and headers dependencies of GTK+

        >>>  export PKG_CONFIG_PATH=/usr/lib/pkgconfig/

        You can check if the variable was just typing

        >>> echo $PKG_CONFIG_PATH

        which will give /usr/lib/pkgconfig/. 
        
After this command you should see something like this: 

::

    >>> Configured wxWidgets 2.9.0 for `i686-pc-linux-gnu
        
Now we just type make and after that as root make install. All this inside /usr/local/wxWidgets/build2.9/

::

    >>> make 
    >>> make install 

Building wxPython
=================

Now we will do the same to build xwPython. In the same directory where we downloaded the sources for wxPython (/usr/local/wxPython) just type:

::

    >>> python setup.py build_ext --inplace

Remember that you need first to have installed the python development libraries in your system (if not just type as root apt-get install python-dev) and that the versions of gcc and g++ should be the same (in our example both versions are 4.2.4). After that just as root type:

::

    >>> python setup.py install

With that, you have built and installed wxWidgets and wxPython. We now only need to build Stimfit.

Building Stimfit
=================

Go to the directory where you unpacked your version of Stimfit (in our example /usr/local/stimfit-0.8.19/ and type:

::

    >>> ./configure --enable-python

After that, and if everything went Ok, just type

::

    >>> make 

and finally type as root

::

    >>> make install.
