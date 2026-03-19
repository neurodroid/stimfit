# Building Stimfit

Stimfit now uses the CMake-based build flow on all supported platforms. The legacy autotools bootstrap path has been deprecated, and [`autogen.sh`](autogen.sh) now stops with a deprecation message instead of generating build files.

## Supported entry points

Use the platform helper script that matches your environment:

- GNU/Linux: [`build_linux_cmake.sh`](build_linux_cmake.sh)
- macOS: [`build_macos_cmake.sh`](build_macos_cmake.sh)
- Windows with Visual Studio/MSVC: [`build_windows_msvc.ps1`](build_windows_msvc.ps1)

These scripts are thin wrappers around CMake presets and cache options. They standardize the repository's preferred configure, build, install, and packaging steps so local builds follow the same structure as current CI and migration work.

## GNU/Linux

Run [`build_linux_cmake.sh`](build_linux_cmake.sh) from the repository root:

```bash
./build_linux_cmake.sh
```

Useful variants:

```bash
./build_linux_cmake.sh --without-python
./build_linux_cmake.sh --install
./build_linux_cmake.sh --package-generator TGZ
```

What the script does:

- configures a CMake build tree under `build/linux-*`
- enables or disables embedded Python support
- prefers the system Biosig integration (`STF_WITH_BIOSIG=ON`)
- builds with [`cmake`](CMakeLists.txt)
- optionally installs into a local prefix under `build/linux-*/install`
- optionally runs `cpack` to create distributable artifacts

The script also tries to locate a usable Python interpreter with development headers before enabling Python support.

## macOS

Run [`build_macos_cmake.sh`](build_macos_cmake.sh) from the repository root:

```bash
./build_macos_cmake.sh
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

## Manual CMake usage

If you need a custom local workflow, you can invoke CMake directly. The repository root is still the source directory:

```bash
cmake -S . -B build/custom
cmake --build build/custom
```

For platform-specific defaults, use the helper scripts above as the authoritative examples.
