# Stimfit macOS CMake Port Plan

## Scope agreed

- Deliver **build + install parity** on macOS using CMake.
- Include compile, link, install, and loader path behavior.
- Exclude legacy app bundle and DMG workflows from deprecated scripts in [`dist/macosx/scripts`](dist/macosx/scripts).
- Keep existing Linux GCC and Windows MSVC CMake behavior stable.
- Treat [`dist/macosx/macports/python/py-stfio`](dist/macosx/macports/python/py-stfio) as retired and out of scope.

## Current-state summary

Autotools macOS behavior comes from [`configure.ac`](configure.ac) and [`Makefile.am`](Makefile.am):

- Darwin sets PIC and platform-specific library naming via `STF_PYTHON_LIBNAME` and `STFIO_PYTHON_LIBNAME`.
- Non-module builds compile wx GUI executable and shared libs.
- Legacy install recipe for macOS used manual copy and extensive `install_name_tool -change` rewrite in [`Makefile.am`](Makefile.am:283).

Current CMake baseline in [`CMakeLists.txt`](CMakeLists.txt) and per-component files already covers:

- Core targets: `stfio`, `stfnum`, optional `biosiglite`, `stimfit_core`, `stimfit`, optional `pystf`, optional `_stfio`.
- Linux RPATH handling via [`CMakeLists.txt`](CMakeLists.txt:83).
- Windows runtime collection and packaging logic via [`CMakeLists.txt`](CMakeLists.txt:35).

macOS gap: no explicit, unified CMake policy for install-name and runtime lookup parity equivalent to historical loader path fixups.

## Design principles

1. Prefer modern CMake target properties over shell post-processing.
2. Keep macOS logic isolated behind `APPLE` conditionals and new opt-in cache variables where needed.
3. Do not alter default Linux or Windows behavior paths.
4. Align install layout with active MacPorts expectations from [`dist/macosx/macports/science/stimfit/Portfile`](dist/macosx/macports/science/stimfit/Portfile).

## Planned CMake changes

### 1. Add macOS runtime path policy module

Create [`cmake/StimfitMacOS.cmake`](cmake/StimfitMacOS.cmake) with:

- Global defaults under `if(APPLE)`:
  - `CMAKE_MACOSX_RPATH ON`
  - `CMAKE_INSTALL_NAME_DIR "@rpath"`
- Helper function `stf_apply_macos_runtime_policy(target)` to set:
  - `BUILD_WITH_INSTALL_RPATH` OFF
  - `INSTALL_RPATH "@loader_path/../lib/stimfit;@loader_path"` for executables and modules as appropriate
  - `INSTALL_NAME_DIR "@rpath"` for shared libs/modules
- Optional toggle for strict post-install validation command using `otool -L` (validation only).

### 2. Wire module into top-level build

In [`CMakeLists.txt`](CMakeLists.txt):

- Include new module after existing includes.
- Keep existing Linux `CMAKE_INSTALL_RPATH` block untouched.
- Add guarded macOS block to define canonical install subdir variables, reusing `${CMAKE_INSTALL_LIBDIR}/stimfit` layout already used by targets.

### 3. Apply policy to all relevant targets

In these files, call `stf_apply_macos_runtime_policy()` after target creation:

- [`src/libstfio/CMakeLists.txt`](src/libstfio/CMakeLists.txt)
- [`src/libstfnum/CMakeLists.txt`](src/libstfnum/CMakeLists.txt)
- [`src/libbiosiglite/CMakeLists.txt`](src/libbiosiglite/CMakeLists.txt)
- [`src/stimfit/CMakeLists.txt`](src/stimfit/CMakeLists.txt)
- [`src/pystfio/CMakeLists.txt`](src/pystfio/CMakeLists.txt)
- [`src/stimfit/py/CMakeLists.txt`](src/stimfit/py/CMakeLists.txt)
- top-level executable `stimfit` in [`CMakeLists.txt`](CMakeLists.txt:100)
- test executable `stimfittest` in [`CMakeLists.txt`](CMakeLists.txt:391)

### 4. Ensure Python module loader-path parity on macOS

For `pystf` and `_stfio` module targets:

- Ensure install name and rpath allow resolving `libstimfit`, `libstfio`, `libstfnum`, and optional `libbiosiglite` from installed tree.
- Keep current install destinations unchanged to avoid Linux and Windows churn.

### 5. Migrate active MacPorts Stimfit port to CMake immediately

Update active MacPorts files:

- [`dist/macosx/macports/science/stimfit/Portfile.in`](dist/macosx/macports/science/stimfit/Portfile.in)
- regenerated [`dist/macosx/macports/science/stimfit/Portfile`](dist/macosx/macports/science/stimfit/Portfile)

Planned MacPorts migration details:

- Switch from Autotools-driven configure arguments to CMake flow by defining:
  - `configure.cmd` as CMake
  - `build.cmd` as CMake build invocation
  - `destroot.cmd` as CMake install invocation
- Pass equivalent options from variants into CMake cache arguments, including:
  - `-DSTF_ENABLE_PYTHON` toggled by Python variants
  - `-DSTF_WITH_BIOSIGLITE=ON` default behavior
  - `-DSTF_WITH_BIOSIG=OFF` unless explicit biosig variant is selected
  - `-DSTF_BUILD_MODULE=OFF` for stimfit application port
- Keep wx integration by forwarding selected `wx-config` path or equivalent include/link hints to CMake through explicit cache variables where needed.
- Preserve existing MacPorts dependency model and Python variant logic while changing only build backend semantics.

Out-of-scope remains unchanged:

- No migration work for retired [`dist/macosx/macports/python/py-stfio`](dist/macosx/macports/python/py-stfio).
- No reliance on deprecated scripts in [`dist/macosx/scripts`](dist/macosx/scripts).

## Regression safety strategy

- All new logic wrapped in `if(APPLE)` or invoked by function that no-ops on non-Apple.
- No changes to Windows runtime dependency installer sections in [`CMakeLists.txt`](CMakeLists.txt:35).
- No changes to Linux RPATH logic in [`CMakeLists.txt`](CMakeLists.txt:83).
- No changes to dependency selection semantics in [`cmake/StimfitDependencies.cmake`](cmake/StimfitDependencies.cmake).

## Acceptance checks for Code mode

### macOS

1. Configure non-module:
   - `cmake -S . -B build/macos -G Ninja -DSTF_WITH_BIOSIGLITE=ON`
2. Build:
   - `cmake --build build/macos`
3. Install to staging prefix:
   - `cmake --install build/macos --prefix build/macos/install`
4. Validate loader references:
   - `otool -L build/macos/install/bin/stimfit`
   - `otool -L build/macos/install/lib/stimfit/libstimfit.dylib`
   - `otool -L <installed pystf/_stfio module path>`
5. Validate expected `@rpath` and `@loader_path` usage and absence of broken absolute build-tree paths.

### Linux GCC smoke

- Reconfigure existing Linux preset and ensure successful configure/build without changed option behavior.

### Windows MSVC smoke

- Reconfigure existing Windows preset and ensure configure step does not regress option/dependency resolution.

## Ordered implementation sequence for Code mode

1. Add [`cmake/StimfitMacOS.cmake`](cmake/StimfitMacOS.cmake).
2. Include it in [`CMakeLists.txt`](CMakeLists.txt) and add macOS-scoped defaults.
3. Apply runtime policy calls in component `CMakeLists.txt` files.
4. Apply runtime policy calls for top-level executables/tests.
5. Run configure/build/install validation on macOS.
6. Run Linux and Windows configuration smoke checks.
7. Document resulting behavior in [`CMAKE_MIGRATION.md`](CMAKE_MIGRATION.md).
