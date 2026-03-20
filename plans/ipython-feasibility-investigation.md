# IPython feasibility investigation for the embedded Python shell

## Scope

Investigate whether Stimfit can realistically use modern IPython for the embedded Python shell, given the current codebase and surviving legacy integration hooks.

## Summary conclusion

A direct revival of the historical IPython integration does not look viable.

The repository still carries remnants of an old IPython path, but the implementation depends on APIs and files that no longer exist in the current tree. The current embedded shell architecture has already shifted to a non-IPython approach built around [`embedded_shell_modern.py`](../src/stimfit/py/embedded_shell_modern.py) and, optionally, the older [`embedded_stf.py`](../src/stimfit/py/embedded_stf.py).

The most likely outcome of a full implementation attempt would be one of these:

- treat legacy IPython support as obsolete and remove or deprecate its stale build and packaging hooks
- or design a substantially new architecture that embeds a modern Jupyter or IPython frontend rather than reviving the old in-process wx integration

## Evidence inventory

### Remaining legacy IPython hooks

- Build option [`STF_ENABLE_IPYTHON`](../cmake/StimfitOptions.cmake) still exists and is described as equivalent to legacy [`--enable-ipython`](../CMAKE_MIGRATION.md)
- Toolchain logic still converts that option into compile definition [`IPYTHON`](../cmake/StimfitToolchain.cmake:15)
- Historical Autotools support also still shows the old [`--enable-ipython`](../dist/macosx/scripts/configure.10.4.in:81) flag
- Window creation in [`parentframe.cpp`](../src/stimfit/gui/parentframe.cpp) still branches on [`#ifdef IPYTHON`](../src/stimfit/gui/parentframe.cpp:346) and tries to import [`embedded_ipython`](../src/stimfit/gui/parentframe.cpp:347)
- Import-module handling in [`unopt.cpp`](../src/stimfit/gui/unopt.cpp) still branches on [`#ifdef IPYTHON`](../src/stimfit/gui/unopt.cpp:369)
- Windows packaging still tries to ship [`embedded_ipython.py`](../dist/windows/nsis/installer.nsi.in:234)
- Comments in [`embedded_init.py`](../src/stimfit/py/embedded_init.py:11) still mention [`embedded_ipython.py`](../src/stimfit/py/embedded_init.py:11)
- Historical references in [`ChangeLog.old`](../ChangeLog.old) show that IPython support was once an active feature

### Missing or inconsistent pieces

- The source tree no longer contains [`embedded_ipython.py`](../src/stimfit/py/CMakeLists.txt)
- The Python install list in [`src/stimfit/py/CMakeLists.txt`](../src/stimfit/py/CMakeLists.txt) includes [`embedded_stf.py`](../src/stimfit/py/embedded_stf.py) and [`embedded_shell_modern.py`](../src/stimfit/py/embedded_shell_modern.py), but not [`embedded_ipython.py`](../src/stimfit/py/CMakeLists.txt)
- Current build artifacts likewise install [`embedded_shell_modern.py`](../build/macos-app/install_manifest.txt) rather than any IPython-specific shell module

## Legacy assumptions versus current architecture

### Historical IPython path

The legacy path assumes all of the following:

1. Stimfit can import a module named [`embedded_ipython`](../src/stimfit/gui/parentframe.cpp:347)
2. That module provides [`MyPanel`](../src/stimfit/gui/parentframe.cpp:354) as a wx widget
3. The embedded shell exposes an object reachable through [`IPython.ipapi.get()`](../src/stimfit/gui/unopt.cpp:372)
4. Stimfit can execute user imports by calling [`ip.ex`](../src/stimfit/gui/unopt.cpp:380) inside the live IPython session

That model is tightly coupled to a very old IPython API surface.

### Current shell model

The current codebase instead supports two non-IPython shell backends:

- [`embedded_stf.py`](../src/stimfit/py/embedded_stf.py), the older wx shell based on [`wx.py.shell.Shell`](../src/stimfit/py/embedded_stf.py:18)
- [`embedded_shell_modern.py`](../src/stimfit/py/embedded_shell_modern.py), the current default, which prefers [`wx.py.shell`](../src/stimfit/py/embedded_shell_modern.py:15) and falls back to a minimal in-process [`code.InteractiveConsole`](../src/stimfit/py/embedded_shell_modern.py:72)

Current backend selection is controlled by [`STF_PY_SHELL_BACKEND`](../cmake/StimfitOptions.cmake) and the compile-time macro selection in [`parentframe.cpp`](../src/stimfit/gui/parentframe.cpp:281).

### Import workflow comparison

For the non-IPython shells, Stimfit imports user modules by constructing Python source and executing it with [`PyRun_SimpleString`](../src/stimfit/gui/unopt.cpp:401). That path only assumes a normal interpreter namespace.

For the legacy IPython path, Stimfit instead imports through [`ip.ex`](../src/stimfit/gui/unopt.cpp:380), which assumes a shell-specific control object obtained from the deprecated [`IPython.ipapi`](../src/stimfit/gui/unopt.cpp:371) module.

This means the old IPython integration is not just another frontend widget. It changes the import control path and assumes shell-specific execution semantics.

## Feasibility assessment for modern IPython

### API blockers

The strongest blocker is reliance on removed or obsolete IPython APIs:

- [`IPython.ipapi`](../src/stimfit/gui/unopt.cpp:371)
- [`IPython.ipapi.get()`](../src/stimfit/gui/unopt.cpp:372)
- shell execution through [`ip.ex`](../src/stimfit/gui/unopt.cpp:380)

Even if a replacement frontend existed, these calls would need a redesign.

### UI and embedding blockers

The old design assumes an in-process wx panel that behaves like the legacy shell. Modern IPython development moved toward terminal integration, Qt consoles, kernels, and Jupyter frontends rather than a simple wx-embedded shell object with the same control API.

By contrast, the currently supported path uses [`wx.py.shell`](../src/stimfit/py/embedded_shell_modern.py:15) directly or falls back to [`InteractiveConsole`](../src/stimfit/py/embedded_shell_modern.py:72), both of which match Stimfit’s existing embedding model much more naturally.

### Packaging blockers

The stale Windows installer entry for [`embedded_ipython.py`](../dist/windows/nsis/installer.nsi.in:234) shows that packaging metadata has already drifted away from reality. Reintroducing modern IPython would likely add substantially more runtime complexity across macOS, Windows, and Linux than the current shell backends require.

### Maintenance blockers

The codebase already has a clear modernization path centered on [`STF_PY_SHELL_BACKEND`](../cmake/StimfitOptions.cmake) and [`embedded_shell_modern.py`](../src/stimfit/py/embedded_shell_modern.py). Reopening IPython support would create a second advanced shell architecture with different assumptions, dependencies, and failure modes.

## Go or no-go recommendation

### Recommendation

No-go for direct revival of the historical IPython integration.

### Rationale

- missing source file [`embedded_ipython.py`](../src/stimfit/py/CMakeLists.txt)
- obsolete API dependency on [`IPython.ipapi.get()`](../src/stimfit/gui/unopt.cpp:372)
- stale packaging reference in [`installer.nsi.in`](../dist/windows/nsis/installer.nsi.in:234)
- current supported architecture already solved the embedded-shell problem without IPython through [`embedded_shell_modern.py`](../src/stimfit/py/embedded_shell_modern.py)

## Recommended next actions

1. Confirm through a brief implementation-free review that no modern IPython wx embedding API exists that preserves the old in-process model
2. If that review confirms the current evidence, deprecate or remove the stale IPython hooks:
   - [`STF_ENABLE_IPYTHON`](../cmake/StimfitOptions.cmake)
   - compile definition wiring in [`StimfitToolchain.cmake`](../cmake/StimfitToolchain.cmake:15)
   - dead guarded branches in [`parentframe.cpp`](../src/stimfit/gui/parentframe.cpp:346) and [`unopt.cpp`](../src/stimfit/gui/unopt.cpp:369)
   - stale packaging entry in [`installer.nsi.in`](../dist/windows/nsis/installer.nsi.in:234)
   - stale comments mentioning [`embedded_ipython.py`](../src/stimfit/py/embedded_init.py:11)
3. Keep the supported embedded shell strategy centered on [`embedded_shell_modern.py`](../src/stimfit/py/embedded_shell_modern.py)

## Decision statement

Based on the repository state, there is no credible path to simply turn on IPython for the embedded shell again. Any future IPython use would require a new design effort, not restoration of the legacy integration.
