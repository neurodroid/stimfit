****************
Building Stimfit
****************

:Author: Christoph Schmidt-Hieber
:Date:  |today|

Current macOS source builds use the repository helper script
[`build_macos_cmake.sh`](build_macos_cmake.sh), which drives the CMake app
bundle workflow used by current development.

=============================
Building with MacPorts tools
=============================

Install MacPorts from `macports.org <https://www.macports.org>`_, then install
the build tools and libraries used by the current CMake path:

::

    sudo port -N selfupdate
    sudo port -N install cmake ninja pkgconfig fftw-3 hdf5 wxWidgets-3.2 git
    sudo port select --set wxWidgets wxWidgets-3.2

If you want the embedded Python build, also install a supported Python variant
and its matching wxPython package. The active MacPorts port definitions in
[`dist/macosx/macports/science/stimfit/Portfile.in`](dist/macosx/macports/science/stimfit/Portfile.in)
currently support Python 3.10 through 3.14, with Python 3.13 as the default
variant.

Clone the repository and build from the repository root:

::

    git clone https://github.com/neurodroid/stimfit.git
    cd stimfit
    ./build_macos_cmake.sh

For a Python-enabled bundle build:

::

    ./build_macos_cmake.sh --with-python

The script configures a dedicated CMake build tree, installs the app bundle into
`build/macos-app*/install`, and verifies that `stimfit.app` was produced.

=================
MacPorts packages
=================

Maintainers updating MacPorts packages should use the active port sources under
[`dist/macosx/macports/`](dist/macosx/macports/) rather than the historical
instructions that modified `sources.conf` manually. The CMake-based Stimfit port
is defined in [`dist/macosx/macports/science/stimfit/Portfile.in`](dist/macosx/macports/science/stimfit/Portfile.in).

====================
Homebrew status note
====================

The old Homebrew tap instructions are no longer maintained here. For current
source work on macOS, prefer MacPorts plus [`build_macos_cmake.sh`](build_macos_cmake.sh).

