# Stimfit

Documentation is available [here](https://neurodroid.github.io/stimfit).

## Introduction

Stimfit is a free, fast and simple program for viewing and analyzing electrophysiological data. It's currently available for GNU/Linux, Mac OS X and Windows. The standard version of Stimfit features an embedded Python shell that allows you to extend the program functionality by using numerical libraries such as [NumPy](http://numpy.scipy.org) and [SciPy](http://www.scipy.org). A standalone Python module for file i/o that doesn't depend on the graphical user interface is also available. The "lite" version of Stimfit comes without an embedded Python shell. Stimfit-lite is more lite-weight, easier to build and install. Stimfit is using the Import filters of [Biosig](https://biosig.sourceforge.net/) which supports reading of over 50 different dataformats.


## List of references 

In [this link](https://neurodroid.github.io/stimfit/references/index.html) you can find a list of publications that used Stimfit for analysis. We'd appreciate if you could cite the following publication when you use Stimfit for your research:

Guzman SJ, Schlögl A, Schmidt-Hieber C (2014) Stimfit: quantifying electrophysiological data with Python. *Front Neuroinform* [doi: 10.3389/fninf.2014.00016](http://www.frontiersin.org/Journal/10.3389/fninf.2014.00016/abstract)

## Installation

### GNU/Linux

#### Debian-based systems (incl Ubuntu, and on WSL2)
 you can get Stimfit and the stfio module from the standard repositories:

```
$ sudo apt-get install stimfit python-stfio
```

This approach works also on WSL2 of the most recent version of Windows10 (build: 10.0.19045.2546 ). Stimfit is also available through a number of [distros](https://repology.org/project/stimfit/packages)

### MacOSX:

#### MacPorts
* Stimfit for OS X is available through [MacPorts](http://www.macports.org). After [installation of MacPorts](https://www.macports.org/install.php), run

```
$ sudo port install stimfit py27-stfio
```
#### Homebrew: stimfit (lite, no embedded Python)
* Stimfit-lite (w/o python)  can be also installed through [HomeBrew](https://brew.sh). Afer installing homebrew, run

```
$ brew install schloegl/biosig/stimfit
```

### MS-Windows:


There are several options to install Stimfit on Windows. Each has its own  advantages (+) and disadvantages (-):

#### The traditional version of [Stimfit v0.15.8-beta1](https://github.com/neurodroid/stimfit/releases/download/v0.15.8windows/Stimfit-x64-0.15.8BETA1-bundle.exe)
including the python-stfio module, is available from [releases](https://github.com/neurodroid/stimfit/releases/). It is quite dated, there are a number of known issues with import filters.

```
 - import filter not up-to-date [issues: 93, 95, 97]
   affected formats: ABF2, ATF, AXG, CNT, EDF+, HEKA, IBW, MFER, RHD/RDS
 - python2 only, python2 has reached end-of-live, (issue 88)
 - warning from windows defender (issue 98 [3])
 + works with all versions of Windows7 and later.
```

#### [Stimfit lite v0.16.2](https://github.com/neurodroid/stimfit/releases/tag/v0.16.2macports)

The lite-version of Stimfit (w/o embedded Python) is available as part of the Biosig-tools. Download, unzip and copy ../bin/stimfit.exe to your desktop.

```
 + import filters are up-to-date (chances are this would address the issues: 93, 95, 97 [3]).
 - no embedded python (CLI, printing not available)
 - warning from windows defender (issue 98 [3])
 + works with all versions of Windows7 and later.
```


#### Stimfit through WSL2
With the most recent version of Windows10 (build: 10.0.19045.2546 ), Stimfit for Linux can be installed through WSL2. e.g. when using Ubuntu in WSL2 (see GNU/Linux above).

```
 + import filters are up-to-date
 + python3 embedded
 + no warnings from windows defender, checksum of download is checked by "apt"
 - requires most recent Windows10 (version 10.0.19045.2546) with WSL2; certain functionality (e.g. copy/paste, file access) might require a non-standard workflow.
```


#### From Source
* [Build guide for GNU/Linux](https://neurodroid.github.io/stimfit/linux_install_guide/index.html)

* [Build guides for MacOSX](https://neurodroid.github.io/stimfit/osx_install_guide/index.html)

* [Build guides for Windows](https://neurodroid.github.io/stimfit/win_install_guide/index.html)

* [Cross-compiling Stimfit with MXE on GNU/Linux for Windows](https://github.com/schloegl/mxe)


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
