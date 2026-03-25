*************
Prerequisites
*************

:Author: Christoph Schmidt-Hieber
:Date:    |today|

This document describes the prerequisites for current 64-bit Windows builds of `Stimfit <http://www.stimfit.org>`_ using Visual Studio, CMake presets, `vcpkg`, and optional CPack packaging.

.. important::

   The Visual C++ Express 2008 and Python 2.7 workflow described in older
   Stimfit documentation is legacy material. Current Windows builds on
   [`master`](README.md:13) use Visual Studio 2022, CMake presets, `vcpkg`, and
   CPack as described in [`README.md`](README.md:105).

For current work, prefer this modern toolchain:

* Visual Studio 2022 build tools
* PowerShell
* `vcpkg`
* optional Inno Setup 6 for installer generation

The repository helper script [`build_windows_msvc.ps1`](build_windows_msvc.ps1)
drives the supported Windows flow and mirrors the active CI setup in
[`ci.yml`](.github/workflows/ci.yml).

Getting all the prerequisites takes some time, but only needs to be completed once per machine.

=================================
Visual Studio and packaging tools
=================================

Install Visual Studio 2022 build tools or a full Visual Studio 2022 release
with C++ support. If you want to generate installer packages, also install
Inno Setup 6 for the `INNOSETUP` CPack generator used by the current workflow.

=========
Libraries
=========

Current third-party C and C++ dependencies are resolved through `vcpkg` by
[`build_windows_msvc.ps1`](build_windows_msvc.ps1) and the repository's CMake
presets. You do not need to download the old hand-curated dependency ZIP files
used by the Visual Studio 2008 workflow.

If you are building with embedded Python enabled, install a matching Python 3
interpreter and ensure it is discoverable when running
[`build_windows_msvc.ps1`](build_windows_msvc.ps1).

===========
Build Tools
===========

SWIG
----
If you are building outside the helper script, install a current SWIG binary for
Windows and make sure it is available on `PATH`. The repository helper script
and CI workflow assume a standard command-line installation rather than the old
Cygwin-specific layout.

git
---
Install a current Git for Windows release from `git-scm.com <https://git-scm.com>`_.

nsis
----
The legacy NSIS packaging path is no longer the supported release route.
Prefer Inno Setup 6 with the `INNOSETUP` CPack generator used by the current
Windows workflow.
