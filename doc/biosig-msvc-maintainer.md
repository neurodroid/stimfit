# Maintaining MSVC biosig patches downstream

This repository keeps MSVC compatibility for biosig as a downstream patch queue.
Upstream biosig does not accept MSVC support, so patches are applied only to a
throwaway working copy outside the submodule tree.

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

When upstream `src/biosig` changes, refresh patches from your local biosig branch
that contains MSVC fixes.

1. Rebase/update your biosig patch branch against upstream `master`.
2. Export patch queue into this repository:

```powershell
git -C src/biosig format-patch master..feature/msvc2022-libbiosig -o cmake/patches/biosig-msvc
```

3. Re-run helper script to verify patches still apply and build:

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
- `STF_BIOSIG_BASE_REF`: base ref before applying patches (default `master`)
- `STF_BIOSIG_GENERATOR`: CMake generator (default VS 2022)
- `STF_BIOSIG_ARCH`: architecture (default `x64`)
- `STF_BIOSIG_CONFIG`: build config (default `Release`)
- `STF_BIOSIG_TARGETS`: targets to build (default `biosig2shared`)
- `STF_BIOSIG_CLEAN`: clean work/build dirs before prepare (default `ON`)
