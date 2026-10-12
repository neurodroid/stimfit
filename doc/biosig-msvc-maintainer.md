# Maintaining MSVC biosig patches downstream

This repository keeps MSVC compatibility for biosig as a downstream patch queue.
Upstream biosig does not accept MSVC support, so patches are applied only to a
throwaway working copy outside the submodule tree.

Windows patching is pinned to BioSig 3.9.8, upstream release commit
`1563ee153331954f217b30bf3fd520e123d563da` (there is no upstream 3.9.8 tag).
Configuration fails fast when `src/biosig` is not exactly at that commit.
The Windows vcpkg manifest includes libiconv for channel-label decoding.
After changing the pin, use a fresh build directory or clear the previous pin
with `cmake --preset vs2022-vcpkg-wx-hdf5-biosig-patched -U STF_BIOSIG_EXPECTED_TAG`.

## Files involved

- Patch queue: `cmake/patches/biosig-msvc/*.patch`
- Helper script: `cmake/PrepareBiosigMSVC.cmake`
- Stimfit preset using patched biosig:
  - Configure preset: `vs2022-vcpkg-wx-hdf5-biosig-patched`
  - Build preset: `vs2022-release-stimfit-biosig-patched`

## One-time build flow on Windows

Quick wrapper (runs prepare + configure + build):

```powershell
./cmake/build-stimfit-with-patched-biosig.ps1
```

To skip patch preparation and reuse existing prepared biosig artifacts:

```powershell
./cmake/build-stimfit-with-patched-biosig.ps1 -SkipPrepare
```

1. Prepare patched biosig and build `biosig2shared`:

```powershell
cmake "-DCMAKE_PREFIX_PATH=$env:VCPKG_INSTALLED_DIR/x64-windows-ci-release" -P cmake/PrepareBiosigMSVC.cmake
```

2. Build Stimfit against the external patched biosig artifacts:

```powershell
cmake --preset vs2022-vcpkg-wx-hdf5-biosig-patched
cmake --build --preset vs2022-release-stimfit-biosig-patched
```

The helper script prints and writes the resolved paths to:

- `build/biosig-msvc-build/biosig-msvc-paths.cmake`

## Refreshing the patch queue

When updating the pinned biosig version, refresh patches from your local biosig
branch that contains MSVC fixes.

1. Checkout/update `src/biosig` to the pinned release commit (currently `1563ee153331954f217b30bf3fd520e123d563da`).
2. Rebase/update your biosig patch branch against that same tag/commit.
3. Export patch queue into this repository:

```powershell
git -C src/biosig format-patch 1563ee153331954f217b30bf3fd520e123d563da..feature/msvc2022-libbiosig -o cmake/patches/biosig-msvc
```

4. Re-run helper script to verify patches still apply and build:

```powershell
cmake "-DCMAKE_PREFIX_PATH=$env:VCPKG_INSTALLED_DIR/x64-windows-ci-release" -P cmake/PrepareBiosigMSVC.cmake
```

If patch apply fails, resolve in the biosig patch branch, regenerate patch files,
and retry.

## Script options

The helper script accepts cache variables when invoked with `-D`:

- `STF_BIOSIG_SOURCE_DIR`: source repo (default `src/biosig`)
- `STF_BIOSIG_WORK_DIR`: patched working copy path
- `STF_BIOSIG_BUILD_DIR`: out-of-tree build path
- `STF_BIOSIG_PATCH_DIR`: patch queue directory
- `STF_BIOSIG_EXPECTED_TAG`: required biosig tag or commit for patching (default `1563ee153331954f217b30bf3fd520e123d563da`, BioSig 3.9.8)
- `CMAKE_PREFIX_PATH`: dependency prefix containing libiconv
- `STF_BIOSIG_GENERATOR`: CMake generator (default VS 2022)
- `STF_BIOSIG_ARCH`: architecture (default `x64`)
- `STF_BIOSIG_CONFIG`: build config (default `Release`)
- `STF_BIOSIG_TARGETS`: targets to build (default `biosig2shared`)
- `STF_BIOSIG_CLEAN`: clean work/build dirs before prepare (default `ON`)
