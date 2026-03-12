# CMake Migration Bootstrap

This repository now includes a **functional CMake toolchain baseline** mirroring the Autotools chain in [`configure.ac`](configure.ac) and the core [`Makefile.am`](Makefile.am) files.

## What was added

- Top-level [`CMakeLists.txt`](CMakeLists.txt)
- Shared CMake module directory [`cmake/`](cmake)
  - [`cmake/StimfitOptions.cmake`](cmake/StimfitOptions.cmake)
  - [`cmake/StimfitDependencies.cmake`](cmake/StimfitDependencies.cmake)
  - [`cmake/StimfitToolchain.cmake`](cmake/StimfitToolchain.cmake)
  - [`cmake/StimfitMigration.cmake`](cmake/StimfitMigration.cmake)
- Generated config header template [`cmake/stfconf.h.in`](cmake/stfconf.h.in)
- Source tree entrypoint [`src/CMakeLists.txt`](src/CMakeLists.txt)
- Per-component scaffold `CMakeLists.txt` files:
  - [`src/libstfio/CMakeLists.txt`](src/libstfio/CMakeLists.txt)
  - [`src/libstfnum/CMakeLists.txt`](src/libstfnum/CMakeLists.txt)
  - [`src/libbiosiglite/CMakeLists.txt`](src/libbiosiglite/CMakeLists.txt)
  - [`src/pystfio/CMakeLists.txt`](src/pystfio/CMakeLists.txt)
  - [`src/stimfit/CMakeLists.txt`](src/stimfit/CMakeLists.txt)
  - [`src/stimfit/py/CMakeLists.txt`](src/stimfit/py/CMakeLists.txt)

## Current scope

The CMake tree now defines real targets for the main migration path:

- Libraries:
  - `stfnum` from [`src/libstfnum/CMakeLists.txt`](src/libstfnum/CMakeLists.txt)
  - `stfio` from [`src/libstfio/CMakeLists.txt`](src/libstfio/CMakeLists.txt)
  - `biosiglite` from [`src/libbiosiglite/CMakeLists.txt`](src/libbiosiglite/CMakeLists.txt) (when enabled)
  - `stimfit_core` from [`src/stimfit/CMakeLists.txt`](src/stimfit/CMakeLists.txt)
- Executables:
  - `stimfit`
  - `stimfittest` (with `ctest` registration)
- Python modules (when enabled):
  - `_stfio` from [`src/pystfio/CMakeLists.txt`](src/pystfio/CMakeLists.txt)
  - `pystf` from [`src/stimfit/py/CMakeLists.txt`](src/stimfit/py/CMakeLists.txt)

## Option mapping from Autotools

The following CMake options mirror high-level Autotools switches from [`configure.ac`](configure.ac):

- `STF_BUILD_MODULE` ⇔ `--enable-module`
- `STF_ENABLE_PYTHON` ⇔ `--enable-python`
- `STF_ENABLE_IPYTHON` ⇔ `--enable-ipython`
- `STF_ENABLE_PSLOPE` ⇔ `--enable-pslope`
- `STF_ENABLE_AUI` ⇔ `--enable-aui`
- `STF_BUILD_DEBIAN` ⇔ `--enable-debian`
- `STF_WITH_BIOSIG` / `STF_WITH_BIOSIGLITE` ⇔ biosig selection flags
- `STF_HDF5_PREFIX` ⇔ `--with-hdf5-prefix`

Compatibility behavior already included:

- Enabling `STF_BUILD_MODULE` forces `STF_ENABLE_PYTHON=ON`
- Enabling `STF_WITH_BIOSIGLITE` disables external `STF_WITH_BIOSIG`

## How to configure

```bash
cmake -S . -B build/cmake-bootstrap
```

On Windows/cmd with Scoop MinGW + Ninja:

```bat
set "PATH=C:\Users\C02380\scoop\apps\mingw\current\bin;C:\Users\C02380\scoop\apps\ninja\current;%PATH%"
"C:\Users\C02380\scoop\shims\cmake.exe" -S . -B build\cmake-migrated -G Ninja -DCMAKE_C_COMPILER=C:/Users/C02380/scoop/apps/mingw/current/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/Users/C02380/scoop/apps/mingw/current/bin/g++.exe
```

You should see a migration summary printed by [`stf_print_configuration_summary()`](cmake/StimfitMigration.cmake:3).

## External dependencies currently required

The CMake chain follows Autotools behavior and expects development libraries for:

- HDF5 (mandatory; or provide [`STF_HDF5_PREFIX`](cmake/StimfitOptions.cmake))
- FFTW3
- LAPACK/BLAS (or OpenBLAS)
- wxWidgets (non-module builds)
- Python3 + SWIG (+ NumPy headers) when Python features are enabled

## Next migration steps

1. Add missing install/post-install parity (`chrpath`, desktop/icons handling, macOS bundle behavior).
2. Tighten dependency mapping (wx variants, Python/wxPython nuances, optional fallbacks).
3. Complete full feature parity for all platform-specific options and packaging scripts.
4. Keep Autotools active until all required CI/build workflows pass under CMake.

