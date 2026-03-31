# Stimfit

Documentation is available [here](https://neurodroid.github.io/stimfit).

## Introduction

Stimfit is a free, fast and simple program for viewing and analyzing electrophysiological data. It's currently available for GNU/Linux, Mac OS X and Windows. The standard version of Stimfit features an embedded Python shell that allows you to extend the program functionality by using numerical libraries such as [NumPy](http://numpy.scipy.org) and [SciPy](http://www.scipy.org). A standalone Python module for file i/o that doesn't depend on the graphical user interface is also available. The "lite" version of Stimfit comes without an embedded Python shell. Stimfit-lite is more lite-weight, easier to build and install. Stimfit is using the Import filters of [Biosig](https://biosig.sourceforge.net/) which supports reading of over 50 different dataformats.

## Branch model

The repository uses two upstream lines and two Debian packaging lines:

- `master`: primary development branch for the modern CMake-based toolchain
- `0.16`: legacy-maintenance branch for the historical 0.16/autotools line
- `debian/sid`: Debian unstable packaging branch tracking `master`
- `debian/sid-0.16`: Debian packaging branch tracking `0.16`

Contributor and maintainer workflow details are documented in [BRANCHES.md](BRANCHES.md).


## List of references 

In [this link](https://neurodroid.github.io/stimfit/references/index.html) you can find a list of publications that used Stimfit for analysis. We'd appreciate if you could cite the following publication when you use Stimfit for your research:

Guzman SJ, Schlögl A, Schmidt-Hieber C (2014) Stimfit: quantifying electrophysiological data with Python. *Front Neuroinform* [doi: 10.3389/fninf.2014.00016](http://www.frontiersin.org/Journal/10.3389/fninf.2014.00016/abstract)

## Installation and source builds

For current source builds from this repository, use the CMake-based helper
scripts documented in [`BUILDING.md`](BUILDING.md).

### Binary packages

- Debian-based systems may provide `stimfit` and `python-stfio` packages through their repositories.
- Release artifacts for supported platforms are published on [GitHub Releases](https://github.com/neurodroid/stimfit/releases).

### Python package status

- A modern [`pyproject.toml`](pyproject.toml) based `pip` build path is being introduced for the standalone `stfio` Python module.
- This targets use inside a user's own Python environment and is separate from the full Stimfit GUI application.
- `pip install stimfit` for the full GUI application is not yet a supported distribution path.

### Source builds

- GNU/Linux build guide: <https://neurodroid.github.io/stimfit/linux_install_guide/index.html>
- macOS build guide: <https://neurodroid.github.io/stimfit/osx_install_guide/index.html>
- Windows build guide: <https://neurodroid.github.io/stimfit/win_install_guide/index.html>

The supported repository entry points are:

- [`build_linux_cmake.sh`](build_linux_cmake.sh)
- [`build_macos_cmake.sh`](build_macos_cmake.sh)
- [`build_windows_msvc.ps1`](build_windows_msvc.ps1)

Windows packaging is performed with CMake and CPack through [`build_windows_msvc.ps1`](build_windows_msvc.ps1) as described in [`BUILDING.md`](BUILDING.md).


## Important links

* [Online documentation](https://neurodroid.github.io/stimfit)
* [User mailing list](http://groups.google.com/group/stimfit)
* [Downloads](https://github.com/neurodroid/stimfit/wiki/Downloads)


## Source code structure

| Directory       | Description |
| --------------- | ----------- |
|./src/libstfio   | File i/o library for common electrophysiology formats |
|./src/libstfnum  | Mathematical operations for measurements and fittings |
|./src/pystfio    | Python wrapper around libstfio |
|./src/stimfit    | Stimfit program |
|./src/stimfit/py | stf module that gets imported into the embedded Python shell |

libstfio is a private library that won't be installed system-wide. You may therefore end up with two copies of libstfio.so: One in the private stimfit library directory (/usr/lib/stimfit/ or similar), the other one in the Python site packages path for pystfio. libstfio may turn into a system-wide library in the future.

The historical [`setup.py.in`](setup.py.in) is retained only as legacy reference material and is not the supported packaging entry point for current releases.

## Build system migration status

An initial CMake bootstrap layer is available to support migration from Autotools.
See [`CMAKE_MIGRATION.md`](CMAKE_MIGRATION.md) for details.
