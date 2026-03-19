#!/bin/sh

cat >&2 <<'MSG'
autotools support has been deprecated in this repository.

Do not run ./autogen.sh for current builds.
Use the CMake build instructions in BUILDING.md instead.

Primary entry points:
  ./build_linux_cmake.sh
  ./build_macos_cmake.sh
  ./build_windows_msvc.ps1
MSG

exit 1
