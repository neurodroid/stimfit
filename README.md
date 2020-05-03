[![Build Status](https://travis-ci.org/dilawar/stimfit.svg?branch=master)](https://travis-ci.org/dilawar/stimfit)

# Stimfit

Documentation is available [here](https://neurodroid.github.io/stimfit).

## Introduction

Stimfit is a free, fast and simple program for viewing and analyzing electrophysiological data. It's currently available for GNU/Linux, Mac OS X and Windows. It features an embedded Python shell that allows you to extend the program functionality by using numerical libraries such as [NumPy](http://numpy.scipy.org) and [SciPy](http://www.scipy.org). A standalone Python module for file i/o that doesn't depend on the graphical user interface is also available.

## List of references 

In [this link](https://neurodroid.github.io/stimfit/references/index.html) you can find a list of publications that used Stimfit for analysis. We'd appreciate if you could cite the following publication when you use Stimfit for your research:

Guzman SJ, Schlögl A, Schmidt-Hieber C (2014) Stimfit: quantifying electrophysiological data with Python. *Front Neuroinform* [doi: 10.3389/fninf.2014.00016](http://www.frontiersin.org/Journal/10.3389/fninf.2014.00016/abstract)

## Installation

#### Windows

The Windows version, including the python-stfio module, is available [here](https://github.com/neurodroid/stimfit/releases).

#### OS X

Stimfit for OS X is available through [MacPorts](http://www.macports.org). After [installation of MacPorts](https://www.macports.org/install.php), run
```
$ sudo port install stimfit py27-stfio
```

#### GNU/Linux

On Debian and Ubuntu systems, you can get Stimfit and the stfio module from the standard repositories:
```
$ sudo apt-get install stimfit python-stfio
```
More recent versions can be found on [the Debian Neuroscience Repository](http://neuro.debian.net/index.html), and there's [a bleeding-edge ppa for Ubuntu](https://launchpad.net/~christsc-gmx/+archive/neuropy) on launchpad.

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
