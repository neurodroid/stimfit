# Pip installation feasibility for Stimfit

## Current state summary

- The historical packaging entry point in [`setup.py.in`](setup.py.in) is tightly coupled to the old distutils and NumPy-distutils world, hardcodes source lists, probes system libraries manually, and does not reflect the active CMake build graph.
- The repository now builds through [`CMakeLists.txt`](CMakeLists.txt) and related CMake modules, with the Python pieces split between the standalone [`src/pystfio/CMakeLists.txt`](src/pystfio/CMakeLists.txt) module path and the full GUI application path in [`src/stimfit/CMakeLists.txt`](src/stimfit/CMakeLists.txt) plus [`src/stimfit/py/CMakeLists.txt`](src/stimfit/py/CMakeLists.txt).
- The build already distinguishes two product shapes:
  - [`STF_BUILD_MODULE`](cmake/StimfitOptions.cmake:3) for standalone Python module builds
  - full native application builds when [`STF_BUILD_MODULE`](cmake/StimfitOptions.cmake:3) is `OFF`
- The current presets show that full Stimfit depends on a platform-specific native dependency stack, especially [`wxWidgets`](cmake/StimfitDependencies.cmake:621), HDF5, FFTW, LAPACK, SWIG, and BIOSIG, with extra coupling to [`wxPython`](cmake/StimfitDependencies.cmake:701) when embedded Python is enabled.

## Feasibility assessment

### 1. Pip-installing the full Stimfit application

Feasible only in a limited technical sense, and not yet as a robust cross-platform user story.

Why:

- A `pip install .` flow can build native extension modules and even drive CMake through a backend such as scikit-build-core.
- However, the full Stimfit product is not just a Python extension. It is a GUI executable with private native libraries, platform-specific install layouts, bundle handling on macOS, Windows runtime copying rules, and dependency discovery logic that currently assumes a CMake install tree rather than a wheel layout.
- Full Stimfit with embedded Python additionally requires that the C++ wx runtime match the wx runtime used by [`wxPython`](cmake/StimfitDependencies.cmake:534), which is explicitly called out as ABI-sensitive in [`cmake/StimfitDependencies.cmake`](cmake/StimfitDependencies.cmake).
- On macOS, the preferred output is a `.app` bundle via [`STF_MACOS_APP_BUNDLE`](cmake/StimfitOptions.cmake:7), which does not naturally map to standard wheel installation semantics.
- On Windows, the current approach copies Python runtime pieces and selected site-packages into the install tree through options such as [`STF_WINDOWS_COPY_PYTHON_SITE_PACKAGES`](cmake/StimfitOptions.cmake:14), which is the opposite of the normal pip model where the environment owns Python packages.

Conclusion:

- Full pip installation of Stimfit is not impossible, but it is currently a packaging redesign project, not a thin wrapper around the existing build.
- It is much more realistic as a long-term effort for selected targets than as an immediate general replacement for the current platform installers and app-bundle flows.

### 2. Pip-installing the standalone Python file I/O module

Feasible and much closer to the repository's current architecture.

Why:

- [`src/pystfio/CMakeLists.txt`](src/pystfio/CMakeLists.txt) already builds a standalone [`_stfio`](src/pystfio/CMakeLists.txt:22) extension plus Python package files.
- [`src/CMakeLists.txt`](src/CMakeLists.txt:11) already routes standalone module builds through [`STF_BUILD_MODULE`](cmake/StimfitOptions.cmake:3).
- The install destination is already Python-aware through [`STF_PYTHON_PLATLIB`](cmake/StimfitDependencies.cmake:663) and the package directory layout under [`src/pystfio`](src/pystfio).

Conclusion:

- The cleanest path toward letting users use Stimfit functionality inside their own Python environment is to package the standalone `stfio` module first.
- That path directly addresses the user-value proposition in the task while avoiding the hardest GUI-app packaging problems.

## Main blockers

### Python-module blockers

1. No modern packaging metadata
   - There is no [`pyproject.toml`](pyproject.toml) yet.
   - The old [`setup.py.in`](setup.py.in) is obsolete and should not be revived as-is.

2. CMake is not yet shaped as a wheel backend contract
   - There is no documented wheel-oriented configure preset or backend entry path.
   - Install destinations are CMake-install centric, not explicitly wheel-staging centric.

3. Native dependency policy is unresolved for wheels
   - Need a decision on whether Linux wheels target manylinux, musllinux, or source-only sdists.
   - Need a policy for HDF5, FFTW, LAPACK, and BIOSIG linkage in wheels.

4. SWIG-generated sources add build-time requirements
   - This is manageable, but must be declared cleanly in packaging metadata and CI.

### Full-application blockers

1. Wheel semantics do not match app semantics
   - [`stimfit`](CMakeLists.txt:128) is installed as a native executable or `.app` bundle, not as a Python importable package.

2. Embedded Python creates two-way dependency coupling
   - The app embeds Python while also depending on Python packages like `numpy`, `wx`, and optionally `IPython` as discovered in [`cmake/StimfitDependencies.cmake`](cmake/StimfitDependencies.cmake:653).
   - A pip-installed app inside an arbitrary environment would be sensitive to that environment's package versions and ABI combinations.

3. wxWidgets and wxPython ABI matching
   - The repository already documents a hard runtime constraint between C++ wxWidgets and [`wxPython`](cmake/StimfitDependencies.cmake:534).
   - This is a major blocker for reliable wheel-based distribution of the full GUI app.

4. Platform-specific install transformations
   - macOS `.app` bundling, Windows runtime DLL copying, and Linux RPATH/install layout logic are all designed around install trees, not wheel contents.

## Candidate strategies

### Strategy A: Full Stimfit wheel now

Not recommended.

- Would require redefining the product as a pip-installed GUI application with bundled native assets.
- High risk across macOS and Windows.
- Likely to produce fragile user outcomes, especially around wx and embedded Python.

### Strategy B: `stfio` wheel first, full app remains installer-based

Recommended.

- Introduce modern Python packaging for the standalone `stfio` module only.
- Keep full Stimfit using the current CMake plus installer or bundle approach.
- Explicitly document that `pip install stimfit` is not yet the supported distribution path for the GUI app.
- This delivers the main practical benefit: users can access Stimfit file I/O from their own Python ecosystem.

### Strategy C: Hybrid future path

Potential longer-term direction after Strategy B.

- Package `stfio` first.
- Refactor the GUI app so that embedding Python becomes optional or is replaced by a clearer plugin boundary.
- Evaluate whether a Python launcher plus native helper libraries could make a pip-installed GUI variant workable on one platform at a time.
- Treat macOS app-bundle distribution and Windows installer distribution as still-primary even if a developer-oriented pip path emerges later.

## Recommended plan for implementation mode

1. Establish packaging scope
   - Define the initial deliverable as a wheel and sdist for `stfio`, not the full GUI app.
   - Mark full-app pip installation as exploratory and unsupported for the first iteration.

2. Add modern packaging metadata
   - Create [`pyproject.toml`](pyproject.toml) using a CMake-aware backend such as scikit-build-core.
   - Surface project metadata from [`VERSION`](VERSION) and align package naming.

3. Create a wheel-focused CMake entry path
   - Ensure a wheel build can force [`STF_BUILD_MODULE`](cmake/StimfitOptions.cmake:3) `ON`.
   - Keep the build isolated from GUI-only targets in [`src/stimfit/CMakeLists.txt`](src/stimfit/CMakeLists.txt).

4. Define dependency policy
   - Decide which native libraries are vendored, dynamically linked, or required from the host.
   - Start with a conservative support matrix if needed, such as sdist-first plus platform wheels only where dependencies are controllable.

5. Remove legacy packaging confusion
   - Deprecate or delete [`setup.py.in`](setup.py.in) after the replacement path is in place.
   - Update [`README.md`](README.md) and [`BUILDING.md`](BUILDING.md) with separate guidance for wheel builds versus full-app builds.

6. Add validation
   - CI should build the standalone package, install it into a fresh virtual environment, and verify `import stfio` plus a minimal smoke test.

## Suggested handoff todo list for implementation

- Add modern packaging metadata for the standalone `stfio` Python package
- Introduce a wheel-oriented CMake configuration that builds only the standalone module path
- Map CMake install outputs to wheel-compatible package layout
- Decide and document native dependency handling for wheel and sdist builds
- Add CI jobs for build and import smoke tests in isolated Python environments
- Retire or clearly mark [`setup.py.in`](setup.py.in) as obsolete
- Update user documentation to distinguish `stfio` pip installs from full Stimfit application builds

## Explicit non-goals for the first implementation pass

- Do not promise `pip install stimfit` for the full GUI application yet
- Do not try to replace macOS `.app` bundling or Windows installer flows in the same change set
- Do not preserve the historical distutils-based logic from [`setup.py.in`](setup.py.in) beyond extracting useful source/dependency knowledge
