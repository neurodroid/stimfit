![Stimfit logo](https://neurodroid.github.io/stimfit/_static/stimfit_128.png)

Stimfit is a free, fast and simple program for viewing and analyzing electrophysiological data. It's currently available for GNU/Linux, macOS and Windows. It features an embedded Python shell that allows you to extend the program functionality by using numerical libraries such as [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/). A standalone Python module for file I/O that doesn't depend on the graphical user interface is also available.

> This wiki page is maintained from the main documentation workflow. The canonical documentation lives at [neurodroid.github.io/stimfit](https://neurodroid.github.io/stimfit/).

We'd appreciate if you could cite the following publication when you use Stimfit for your research:

Guzman SJ, Schlögl A, Schmidt-Hieber C (2014) Stimfit: quantifying electrophysiological data with Python. *Front Neuroinform* [doi:10.3389/fninf.2014.00016](https://doi.org/10.3389/fninf.2014.00016)

## Supported file types

- Read/write: CFS binary data, HDF5 files, Axon text files, ASCII files
- Read-only: Axon binary files versions 1 and 2 (`*.abf`), Axograph files (`*.axgd`, `*.axgx`), HEKA files (`*.dat`)
- Write-only: Igor binary waves (`*.ibw`)

Support for other file types may be implemented upon request.

## Installing and building

### Windows

Windows builds and packaging are based on Visual Studio 2022, CMake presets, `vcpkg`, and CPack. Current source-build instructions are documented here:

- [Windows build guide](https://neurodroid.github.io/stimfit/win_install_guide/index.html)
- [Repository build overview](https://github.com/neurodroid/stimfit/blob/master/BUILDING.md)

Release artifacts are published on the GitHub releases page:

- [Stimfit releases](https://github.com/neurodroid/stimfit/releases)

### macOS

Stimfit on macOS is currently built from source with the CMake toolchain and MacPorts-oriented dependencies. Current instructions are documented here:

- [macOS build guide](https://neurodroid.github.io/stimfit/osx_install_guide/index.html)
- [Repository build overview](https://github.com/neurodroid/stimfit/blob/master/BUILDING.md)

### Debian and Ubuntu systems

You can get Stimfit and the `stfio` module from standard repositories on Debian-based systems:

```bash
sudo apt-get install stimfit python-stfio
```

For current source builds and dependency guidance, see:

- [GNU/Linux build guide](https://neurodroid.github.io/stimfit/linux_install_guide/index.html)
- [Repository build overview](https://github.com/neurodroid/stimfit/blob/master/BUILDING.md)

## Build from source

The source code lives in the main repository:

- [neurodroid/stimfit](https://github.com/neurodroid/stimfit)

Current build instructions are maintained in:

- [Published documentation site](https://neurodroid.github.io/stimfit/)
- [BUILDING.md](https://github.com/neurodroid/stimfit/blob/master/BUILDING.md)

## Community and links

- [Online documentation](https://neurodroid.github.io/stimfit/)
- [Downloads and releases](https://github.com/neurodroid/stimfit/releases)
- [Issues](https://github.com/neurodroid/stimfit/issues)
- [Source repository](https://github.com/neurodroid/stimfit)
- [User mailing list / Google group](https://groups.google.com/g/stimfit)
