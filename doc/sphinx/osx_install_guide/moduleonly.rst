*************************************
Building the stfio Python Module Only
*************************************

:Author: Yueqi Wang
:Date:    |today|

This documentation describes how to install the standalone Python file i/o module for Mac OS.

For details on how to use the *stfio* module, see :doc:`/stfio/index`.

Installing stfio with current MacPorts packages
===============================================

The legacy standalone [`py-stfio`](dist/macosx/macports/python/py-stfio/Portfile.in)
port still exists for reference, but it predates the current CMake provider
model and old Python-only instructions in this document are no longer current.

For maintained macOS packaging, prefer the Stimfit MacPorts port defined in
[`dist/macosx/macports/science/stimfit/Portfile.in`](dist/macosx/macports/science/stimfit/Portfile.in),
which supports current Python 3 variants and can enable the embedded Python
shell in the app bundle.

  
Building the stfio module from source
=====================================

For current source builds, use the same repository checkout as the full
application and configure a dedicated CMake build tree for the module target.

Install MacPorts build dependencies first:

::

  $ sudo port -N selfupdate
  $ sudo port -N install cmake ninja swig-python hdf5 fftw-3 git python313 py313-numpy

Then clone the repository and configure the module build:

::

  $ git clone https://github.com/neurodroid/stimfit.git
  $ cd stimfit
  $ cmake -S . -B build/macos-module -G Ninja \
            -DSTF_BUILD_MODULE=ON \
            -DSTF_ENABLE_PYTHON=ON \
            -DSTF_BUILD_TESTS=OFF \
            -DSTF_BUILD_NUMERIC_TESTS=OFF \
            -DPython3_EXECUTABLE=/opt/local/bin/python3.13

Build and stage the install:

::

  $ cmake --build build/macos-module
  $ cmake --install build/macos-module --prefix $HOME/.local

If you need BioSig-backed import support, install the appropriate library and
reconfigure with `-DSTF_WITH_BIOSIG=ON`. If you specifically need the in-tree
provider on a maintainer machine, add `-DSTF_BIOSIG_PROVIDER=SUBMODULE`.

Using another Python 3 environment
----------------------------------

To build against a non-MacPorts interpreter such as a virtual environment,
replace `-DPython3_EXECUTABLE=/opt/local/bin/python3.13` with the path to the
desired interpreter.

For example:

::

  $ cmake -S . -B build/macos-module-venv -G Ninja \
            -DSTF_BUILD_MODULE=ON \
            -DSTF_ENABLE_PYTHON=ON \
            -DPython3_EXECUTABLE=$HOME/venvs/stimfit/bin/python

Finally, run python to test the module, as described in :doc:`/stfio/index`.
