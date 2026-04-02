# Building Stimfit

Stimfit now uses the CMake-based build flow on all supported platforms. The legacy autotools bootstrap path has been deprecated, and [`autogen.sh`](autogen.sh) now stops with a deprecation message instead of generating build files.

## Supported entry points

Use the platform helper script that matches your environment:

- GNU/Linux: [`build_linux_cmake.sh`](build_linux_cmake.sh)
- macOS: [`build_macos_cmake.sh`](build_macos_cmake.sh)
- Windows with Visual Studio/MSVC: [`build_windows_msvc.ps1`](build_windows_msvc.ps1)

These scripts are thin wrappers around CMake presets and cache options. They standardize the repository's preferred configure, build, install, and packaging steps so local builds follow the same structure as current CI and migration work.

## Python package builds (`stfio`)

The standalone `stfio` module can now be targeted through [`pyproject.toml`](pyproject.toml) using a CMake-backed `pip` build flow. This is intended for the Stimfit-independent file I/O module only, not the full Stimfit GUI application.

Example local build commands:

```bash
python -m pip install .
python -m pip wheel .
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
- prefers the system Biosig integration (`STF_WITH_BIOSIG=ON`, `STF_BIOSIG_PROVIDER=SYSTEM`)
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

## Windows (MSVC)

Run [`build_windows_msvc.ps1`](build_windows_msvc.ps1) from a PowerShell session:

```powershell
./build_windows_msvc.ps1
```

Direct CMake preset entry points:

```powershell
cmake --preset vs2022-vcpkg-wx-hdf5-biosig-patched
cmake --build --preset vs2022-release-all-biosig-patched
```

Python-enabled build:

```powershell
./build_windows_msvc.ps1 -WithPython
```

Packaging examples:

```powershell
./build_windows_msvc.ps1 -PackageGenerator INNOSETUP
./build_windows_msvc.ps1 -PackageGenerator ZIP
```

What the script does:

- prepares `vcpkg` dependency locations from environment variables or Visual Studio defaults
- installs the repository's required Windows dependencies through the custom triplet in [`cmake/triplets/x64-windows-ci-release.cmake`](cmake/triplets/x64-windows-ci-release.cmake)
- selects the appropriate Visual Studio configure preset from [`CMakePresets.json`](CMakePresets.json)
- runs CMake configure, build, and install steps for the chosen preset
- optionally runs `cpack` to generate either an Inno Setup installer or a ZIP package

This script mirrors the repository's active Windows CI-oriented workflow more closely than invoking raw commands manually.

For the patched-submodule biosig provider on Windows, Stimfit now requires
`src/biosig` to match the pinned upstream tag `v3.9.3` exactly. Configure fails
fast when the submodule HEAD differs from that tag's commit.

### Windows Python refresh behavior

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

## Manual CMake usage

If you need a custom local workflow, you can invoke CMake directly. The repository root is still the source directory:

```bash
cmake -S . -B build/custom
cmake --build build/custom
```

For platform-specific defaults, use the helper scripts above as the authoritative examples.
