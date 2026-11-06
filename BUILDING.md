# Building Stimfit

Stimfit now uses the CMake-based build flow on all supported platforms. The legacy autotools bootstrap path has been deprecated, and [`autogen.sh`](autogen.sh) now stops with a deprecation message instead of generating build files.

## Supported entry points

Use the platform helper script that matches your environment:

- GNU/Linux: [`build_linux_cmake.sh`](build_linux_cmake.sh)
- macOS: [`build_macos_cmake.sh`](build_macos_cmake.sh)
- Windows with Visual Studio/MSVC: [`build_windows_msvc.ps1`](build_windows_msvc.ps1)

All platform helpers include the embedded Python shell by default. On Windows, use
`-WithoutPython` to opt out; on Linux and macOS, use `--without-python`.

Windows and macOS use the Biosig submodule pinned to upstream release **4.0.0**
(`v4.0.0`). Linux prefers system Biosig, without imposing the submodule release
version on installed libraries. Explicit Linux submodule builds use the same pin:
`./build_linux_cmake.sh --biosig-provider SUBMODULE`.

The shared pin is maintained in `cmake/StimfitBiosigVersion.cmake` and the
`src/biosig` submodule. Use upstream release tags when updating it; the library
API/ABI version and the old upstream CMake version fields are separate values.
Submodule builds apply a shared compatibility fix for the 4.0.0 ABF2 string cache,
which can otherwise index past the string table when directory and SSCH counts differ. Generated/staged
sources receive this fix; the upstream release checkout stays pristine.

These scripts are thin wrappers around CMake presets and cache options. They standardize the repository's preferred configure, build, install, and packaging steps so local builds follow the same structure as current CI and migration work.

Python integration requires Python 3 and wxPython Phoenix. Python 2 build and
packaging workflows have been removed. The standalone `stfio` package requires
Python 3.9 or newer, as specified in `pyproject.toml`.

## Python package builds (`stfio`)

The standalone `stfio` module can now be targeted through [`pyproject.toml`](pyproject.toml) using a CMake-backed `pip` build flow. This is intended for the Stimfit-independent file I/O module only, not the full Stimfit GUI application.

Example local build commands:

```bash
python3 -m pip install .
python3 -m pip wheel .
```

Use a Python interpreter that provides development headers for CMake discovery. On macOS, the system or Xcode-provided Python may be insufficient for native builds; a framework or package-manager Python is typically required.

This packaging path forces [`STF_BUILD_MODULE`](cmake/StimfitOptions.cmake:3) and uses a wheel-oriented install layout for the contents of [`src/pystfio`](src/pystfio).

Full Stimfit application builds should still follow the platform-specific CMake helper scripts described below.

## GNU/Linux

For a full Debian/Ubuntu GUI build with the current CMake flow, install at least:

```bash
sudo apt-get update
sudo apt-get install -y \
  cmake \
  ninja-build \
  g++ \
  pkg-config \
  swig \
  libbiosig-dev \
  libfftw3-dev \
  libhdf5-dev \
  libwxgtk3.2-dev \
  liblapack-dev \
  libblas-dev \
  libopenblas-dev \
  python3-dev \
  python3-numpy
```

If you want embedded Python support in the GUI build, make sure the selected `python3`
interpreter also provides development headers and that the environment includes the
required runtime Python packages such as `wxPython`, `IPython`, and `numpy`.

Run [`build_linux_cmake.sh`](build_linux_cmake.sh) from the repository root:

```bash
./build_linux_cmake.sh
```

Direct CMake preset entry points:

```bash
cmake --preset linux-ninja-python
cmake --build --preset linux-ninja-python-build
```

No-Python variant:

```bash
cmake --preset linux-ninja
cmake --build --preset linux-ninja-build
```

Useful variants:

```bash
./build_linux_cmake.sh --without-python
./build_linux_cmake.sh --install
./build_linux_cmake.sh --package-generator TGZ
```

Explicit no-Python, no-Biosig fallback build:

```bash
cmake -S . -B build/linux-fallback -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTF_ENABLE_PYTHON=OFF \
  -DSTF_WITH_BIOSIG=OFF
cmake --build build/linux-fallback
```

What the script does:

- configures a CMake build tree under `build/linux-*`
- enables or disables embedded Python support
- prefers system Biosig (`STF_WITH_BIOSIG=ON`, `STF_BIOSIG_PROVIDER=SYSTEM`); use `--biosig-provider SUBMODULE` to build the pinned release explicitly
- builds with [`cmake`](CMakeLists.txt)
- optionally installs into a local prefix under `build/linux-*/install`
- optionally runs `cpack` to create distributable artifacts

The script also tries to locate a usable Python interpreter with development headers before enabling Python support. On Debian/Ubuntu, system Biosig detection now checks `pkg-config` first (`libbiosig`/`biosig`) and falls back to direct library/header probing if no pkg-config metadata is available.

## macOS

Run [`build_macos_cmake.sh`](build_macos_cmake.sh) from the repository root:

```bash
./build_macos_cmake.sh
```

Direct CMake preset entry points:

```bash
cmake --preset macos-ninja-app-python
cmake --build --preset macos-ninja-app-python-stimfit
```

No-Python variant:

```bash
cmake --preset macos-ninja-app
cmake --build --preset macos-ninja-app-stimfit
```

Python-enabled app bundle build:

```bash
./build_macos_cmake.sh --with-python
```

What the script does:

- configures a dedicated macOS CMake build tree under `build/macos-app*`
- enables creation of a `.app` bundle through `STF_MACOS_APP_BUNDLE=ON`
- uses the Biosig submodule-backed CMake path (`STF_BIOSIG_PROVIDER=SUBMODULE`)
- locates a compatible MacPorts Python and, when available, a matching `wx-config`
- builds and installs the bundle into `build/macos-app*/install`
- verifies that `stimfit.app` was produced successfully

This is the preferred path for modern macOS builds because it centralizes bundle layout and runtime-path handling in the CMake toolchain.

The macOS helper prefers Ninja when it is available on `PATH` and falls back to
Unix Makefiles otherwise. Makefiles builds use a separate default build directory
with a `-makefiles` suffix (for example, `build/macos-app-makefiles`) and install
under that directory. You can also select Makefiles explicitly with
`./build_macos_cmake.sh --generator "Unix Makefiles"`. An explicit `--build-dir`
must use the same generator as any existing CMake cache in that directory.

After an Xcode update, the helper checks whether the SDK path in an existing
CMake cache still exists. If it is missing, the helper stops with an explanation
and recovery arguments. Use a fresh build and install directory, or move the
existing build directory aside and rerun the helper.

## Windows (MSVC)

Run [`build_windows_msvc.ps1`](build_windows_msvc.ps1) from a PowerShell session:

```powershell
./build_windows_msvc.ps1
```

Direct CMake preset entry points:

```powershell
./prepare_wxpython_windows.ps1
cmake --preset vs2022-vcpkg-wx-hdf5-python314-biosig-patched
cmake --build --preset vs2022-release-all-python314-biosig-patched
```

The default build includes the embedded Python shell with wxPython 4.3.2 and
wxWidgets 3.3.4. The helper prepares missing wxPython dependencies automatically.
Explicit no-Python build:

```powershell
./build_windows_msvc.ps1 -WithoutPython
```

Packaging examples:

```powershell
./build_windows_msvc.ps1 -PackageGenerator INNOSETUP
./build_windows_msvc.ps1 -PackageGenerator ZIP
```

What the script does:

- prepares `vcpkg` dependency locations from environment variables or Visual Studio defaults
- installs the repository's required Windows dependencies through the custom triplet in [`cmake/triplets/x64-windows-ci-release.cmake`](cmake/triplets/x64-windows-ci-release.cmake)
- prepares wxPython 4.3.2 and its matching native libraries if they are missing, using the same Python 3.14 interpreter as CMake
- selects the appropriate Visual Studio configure preset from [`CMakePresets.json`](CMakePresets.json)
- runs CMake configure, build, and install steps for the chosen preset
- optionally runs `cpack` to generate either an Inno Setup installer or a ZIP package

This script mirrors the repository's active Windows CI-oriented workflow more closely than invoking raw commands manually.

The submodule providers validate the Biosig release on all platforms and check
the upstream release tag when available. Old `STF_BIOSIG_EXPECTED_TAG` CMake cache
entries are superseded by the shared release pin. Initialize the submodule with
`git submodule update --init --recursive` before configuring.

### Default Windows wxPython setup

The default Python-enabled Windows preset uses wxPython 4.3.2 and its bundled
wxWidgets 3.3.4 with 64-bit Python 3.14. Matching native development libraries
and the Python wheel live in the repository's ignored `build/wxpython-4.3.2/`
directory. The normal build helper prepares them automatically on the first
build. To prepare them in advance, run `./prepare_wxpython_windows.ps1`.

The versioned `vs2022-python314-wxpython432` preset uses the same dependencies
with separate build and install directories:

```powershell
./prepare_wxpython_windows.ps1
cmake --preset vs2022-python314-wxpython432
cmake --build --preset vs2022-python314-wxpython432-release --parallel 4
cmake --install build/windows-wxpython432 --config Release
./cmake/tests/check-windows-snapshot.ps1 -InstallRoot build/windows-wxpython432/install -CheckDirectory build/windows-wxpython432/checks -ExpectedVersion (Get-Content VERSION) -ExpectedWxPythonVersion 4.3.2
```

Use the same `VCPKG_ROOT` and `VCPKG_INSTALLED_DIR` environment variables as
the normal Windows build. The preparation script discovers Python through
`py -3.14`; pass `-PythonExecutable <path>` to choose it explicitly, and pass
the same path as `-DPython3_EXECUTABLE=<path>` when configuring if necessary.
The selected interpreter must already contain the other runtime packages
listed below and `setuptools` for native compiler discovery.

Preparation checks the source archive's SHA-256 against PyPI metadata,
installs the wheel with `pip --target`, and builds only wxWidgets from that
same wxPython release. It then copies the wheel's wxWidgets DLLs into the
native library directory, so both installed DLL locations use identical
binaries. Stimfit's native headers, import libraries, wxPython C API headers,
and runtime packages must move together; updating only the Python wheel can
break the embedded shell.

The embedded interpreter leaves wxWidgets cleanup to the native application.
On Windows with wxWidgets 3.3+, Python finalization runs after native cleanup
has released its Python callbacks. This also avoids wxPython's Python-exit
hook tearing down the application inside wxWidgets' shutdown sequence.

The native build uses nmake by default. For a parallel build, download
[Qt's jom](https://download.qt.io/official_releases/jom/) and pass
`-JomExecutable <path-to-jom.exe> -Jobs 8`. The script handles the shared PDB
and precompiled-header conflicts in the upstream parallel makefile.

wxPython 4.3.x tracks the API/ABI-unstable wxWidgets 3.3.x series, so the Windows
presets pin wxPython 4.3.2 and use its matching native libraries.
See the [wxPython release notes](https://wxpython.org/pages/changes/).

### Windows Python refresh behavior

The Python-enabled Windows preset bundles `numpy`, `wxPython`, and Matplotlib
with its runtime dependencies so the snapshot toolbar button works on a clean
computer. Install Matplotlib into the Python interpreter used for the build
before configuring:

```powershell
py -3.14 -m pip install matplotlib 'python-dateutil<3'
```

The bundled module selection supports the python-dateutil 2.x dependency
set, including `six`; retain the version constraint when installing wheels.
Alternatively, install those wheels into a separate directory with
`pip install --target <directory> matplotlib 'python-dateutil<3'` and append that directory to
`STF_WINDOWS_PYTHON_EXTRA_PATHS`, preserving the prepared wxPython wheel directory
used by the preset. `STF_WINDOWS_PYTHON_SITE_PACKAGES` contains Python import names;
`PIL`, `dateutil`, and `fontTools` are provided by the Pillow, python-dateutil,
and fonttools distributions. Configuration fails if a selected module is
missing. Each module's companion DLL directories and distribution metadata
(including wheel licenses) are copied from the same source directory.

For Python-enabled Windows installs, the CMake cache option `STF_WINDOWS_PYTHON_FULL_REFRESH` controls how stdlib and `stf-site-packages` are synchronized:

- `OFF` (default): fast mode. Install skips repeated deep per-file up-to-date checks by using a signature marker and only refreshes when relevant source roots/package selection change.
- `ON`: full refresh mode. Restores the traditional per-file CMake install behavior for stdlib and selected site-packages.

Examples:

```powershell
# Default (fast mode)
cmake --preset vs2022-vcpkg-wx-hdf5-python314-biosig-patched

# Opt-in full refresh/check mode
cmake --preset vs2022-vcpkg-wx-hdf5-python314-biosig-patched -DSTF_WINDOWS_PYTHON_FULL_REFRESH=ON
```

Python runtime DLL handling is unchanged by this option.

The package-selection regression can also run without building Stimfit:

```powershell
cmake -P cmake/tests/python_packages.cmake
```

After installing a Python-enabled Windows build, check the snapshot toolbar
against the installed runtime:

```powershell
./cmake/tests/check-windows-snapshot.ps1 -InstallRoot <install-directory> -CheckDirectory <results-directory> -ExpectedVersion (Get-Content VERSION)
```

This opens the installed application with external Python paths cleared,
creates snapshots of calibrated one- and two-channel recordings through the
toolbar, and checks the rendered PNGs. The test restores the startup hook and
environment after closing its application instance.

## Manual CMake usage

If you need a custom local workflow, you can invoke CMake directly. The repository root is still the source directory:

```bash
cmake -S . -B build/custom
cmake --build build/custom
```

For platform-specific defaults, use the helper scripts above as the authoritative examples.
