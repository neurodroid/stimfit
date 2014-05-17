# Stimfit

Issue tracker and Wiki are still on [Google Code](https://code.google.com/p/stimfit).

## Introduction

Stimfit is a free, fast and simple program for viewing and analyzing electrophysiological data. It's currently available for GNU/Linux, Mac OS X and Windows. It features an embedded Python shell that allows you to extend the program functionality by using numerical libraries such as [NumPy](http://numpy.scipy.org) and [SciPy](http://www.scipy.org). A standalone Python module for file i/o that doesn't depend on the graphical user interface is also available.

## List of references 

In [this link](http://www.stimfit.org/doc/sphinx/references/index.html) you can find a list of publications that used Stimfit for analysis. We'd appreciate if you could cite the following publication when you use Stimfit for your research:

Guzman SJ, Schlögl A, Schmidt-Hieber C (2014) Stimfit: quantifying electrophysiological data with Python. *Front Neuroinform* [doi: 10.3389/fninf.2014.00016](http://www.frontiersin.org/Journal/10.3389/fninf.2014.00016/abstract)

## Important links

* [Online documentation](http://www.stimfit.org/doc/sphinx/index.html)
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
