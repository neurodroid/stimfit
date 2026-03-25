*******************************
Building the Python module only
*******************************

:Author: Christoph Schmidt-Hieber
:Date:    |today|

The standalone [`stfio`](doc/sphinx/stfio/index.rst) module is built from the
same CMake source tree as the full application, but without the GUI target.
The old autotools path is no longer supported on current [`master`](README.md:13).

On Debian-based systems, install the core build requirements first:

::

    $ sudo apt-get install build-essential cmake ninja-build git \
                           python3-dev python3-numpy swig \
                           libhdf5-dev libfftw3-dev liblapack-dev

If you want BioSig-backed import support in [`stfio`](doc/sphinx/stfio/index.rst),
also install the distribution package:

::

    $ sudo apt-get install libbiosig-dev

Then clone the repository:

::

    $ cd $HOME
    $ git clone https://github.com/neurodroid/stimfit.git

That will create a directory called *stimfit*. Configure a dedicated module
build directory from the repository root:

::

    $ cd $HOME/stimfit
    $ cmake -S . -B build/module -G Ninja \
              -DSTF_BUILD_MODULE=ON \
              -DSTF_ENABLE_PYTHON=ON \
              -DSTF_BUILD_TESTS=OFF \
              -DSTF_BUILD_NUMERIC_TESTS=OFF \
              -DSTF_WITH_BIOSIG=ON

If your system already provides `libbiosig-dev`, the default provider selection
is usually sufficient. If you need to force the in-tree BioSig source on a
non-packaged platform, you can additionally pass `-DSTF_BIOSIG_PROVIDER=SUBMODULE`.

Build the module:

::

    $ cmake --build build/module

To stage an install into a local prefix for inspection:

::

    $ cmake --install build/module --prefix $HOME/.local

===================================================
Building stfio for non-default Python distributions
===================================================

To target a non-default Python 3 interpreter, point CMake at the interpreter
you want it to use during configuration. For example, with a virtual
environment:

::

    $ cmake -S . -B build/module-venv -G Ninja \
              -DSTF_BUILD_MODULE=ON \
              -DSTF_ENABLE_PYTHON=ON \
              -DPython3_EXECUTABLE=$HOME/venvs/stimfit/bin/python

Then build and install into that environment or another target prefix:

::

    $ cmake --build build/module-venv
    $ cmake --install build/module-venv --prefix $HOME/.local

Finally, run python to test the module, as described in :doc:`/stfio/index`.
