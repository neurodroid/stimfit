# Maintaining MSVC biosig patches downstream

This repository keeps MSVC compatibility for biosig as a downstream patch queue.
Upstream biosig does not accept MSVC support, so patches are applied only to a
throwaway working copy outside the submodule tree.

Windows patching is pinned to a specific upstream biosig tag (`v3.9.5`).
Configuration now fails fast when `src/biosig` is not exactly at that tag's
commit.

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
cmake -P cmake/PrepareBiosigMSVC.cmake
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

1. Checkout/update `src/biosig` to the pinned tag commit (currently `v3.9.5`).
2. Rebase/update your biosig patch branch against that same tag/commit.
3. Export patch queue into this repository:

```powershell
git -C src/biosig format-patch v3.9.5..feature/msvc2022-libbiosig -o cmake/patches/biosig-msvc
```

4. Re-run helper script to verify patches still apply and build:

```powershell
cmake -P cmake/PrepareBiosigMSVC.cmake
```

If patch apply fails, resolve in the biosig patch branch, regenerate patch files,
and retry.

## Script options

The helper script accepts cache variables when invoked with `-D`:

- `STF_BIOSIG_SOURCE_DIR`: source repo (default `src/biosig`)
- `STF_BIOSIG_WORK_DIR`: patched working copy path
- `STF_BIOSIG_BUILD_DIR`: out-of-tree build path
- `STF_BIOSIG_PATCH_DIR`: patch queue directory
- `STF_BIOSIG_EXPECTED_TAG`: required biosig tag for patching (default `v3.9.5`)
- `STF_BIOSIG_GENERATOR`: CMake generator (default VS 2022)
- `STF_BIOSIG_ARCH`: architecture (default `x64`)
- `STF_BIOSIG_CONFIG`: build config (default `Release`)
- `STF_BIOSIG_TARGETS`: targets to build (default `biosig2shared`)
- `STF_BIOSIG_CLEAN`: clean work/build dirs before prepare (default `ON`)
