****************
Building Stimfit
****************

:Author: Christoph Schmidt-Hieber
:Date:  |today|

Current Windows source builds use the CMake-based MSVC workflow rather than the
historical Visual Studio 2008 solution files.

===================
Get the source code
===================

Clone the repository into a local working directory:

::

    PS C:\> git clone https://github.com/neurodroid/stimfit.git
    PS C:\> cd stimfit

=============
Build Stimfit
=============

Run the supported helper script from PowerShell:

::

    PS C:\stimfit> .\build_windows_msvc.ps1

For a Python-enabled build:

::

    PS C:\stimfit> .\build_windows_msvc.ps1 -WithPython

The script prepares the current `vcpkg` dependency layout, selects the matching
CMake preset, and runs configure, build, and install steps.

==================
Create an installer
==================

The CMake build can also generate distributable packages with CPack. Use one of
the packaging modes below after installing Inno Setup 6 when you want a native
installer:

::

    PS C:\stimfit> .\build_windows_msvc.ps1 -PackageGenerator INNOSETUP
    PS C:\stimfit> .\build_windows_msvc.ps1 -PackageGenerator ZIP

This replaces the old NSIS-based packaging instructions for current releases.
