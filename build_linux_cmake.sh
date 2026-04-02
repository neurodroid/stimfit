#!/bin/bash
set -euo pipefail

# Configure, build, install, and optionally package Stimfit on GNU/Linux using CMake.
#
# Usage:
#   ./build_linux_cmake.sh
#   ./build_linux_cmake.sh --without-python
#   ./build_linux_cmake.sh --with-python --install-prefix build/linux-python/install
#   ./build_linux_cmake.sh --install
#   ./build_linux_cmake.sh --package-generator TGZ
#
# Optional environment overrides:
#   PYTHON_EXECUTABLE=/usr/bin/python3.12

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cmake/StimfitPresetHelpers.sh
source "${SCRIPT_DIR}/cmake/StimfitPresetHelpers.sh"

BUILD_DIR="build/linux-default"
INSTALL_PREFIX="build/linux-default/install"
GENERATOR="Ninja"
BUILD_TYPE="Release"
WITH_PYTHON=1
DO_INSTALL=0
PACKAGE_GENERATOR=""
DEFAULT_SYSTEM_PYTHON="/usr/bin/python3"

pick_python_for_cmake() {
  local requested="$1"
  local candidates=(
    "$requested"
    /usr/bin/python3.14
    /usr/bin/python3.13
    /usr/bin/python3.12
    /usr/bin/python3.11
    /usr/bin/python3.10
    /usr/bin/python3
  )

  for py in "${candidates[@]}"; do
    [[ -x "$py" ]] || continue
    if "$py" - <<'PY' >/dev/null 2>&1
import sysconfig
include_dir = sysconfig.get_config_var('INCLUDEPY')
lib_dir = sysconfig.get_config_var('LIBDIR')
ld_library = sysconfig.get_config_var('LDLIBRARY')
ok = bool(include_dir) and (bool(lib_dir) or bool(ld_library))
raise SystemExit(0 if ok else 1)
PY
    then
      echo "$py"
      return 0
    fi
  done

  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-python)
      WITH_PYTHON=1
      BUILD_DIR="build/linux-python"
      INSTALL_PREFIX="build/linux-python/install"
      shift
      ;;
    --without-python)
      WITH_PYTHON=0
      BUILD_DIR="build/linux-lite"
      INSTALL_PREFIX="build/linux-lite/install"
      shift
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --install-prefix)
      INSTALL_PREFIX="$2"
      shift 2
      ;;
    --generator)
      GENERATOR="$2"
      shift 2
      ;;
    --build-type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    --package-generator)
      PACKAGE_GENERATOR="$2"
      shift 2
      ;;
    --install)
      DO_INSTALL=1
      shift
      ;;
    --no-install)
      DO_INSTALL=0
      shift
      ;;
    -h|--help)
      sed -n '1,45p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

stf_select_presets \
  "$WITH_PYTHON" \
  "linux-ninja-python" \
  "linux-ninja" \
  "linux-ninja-python-build" \
  "linux-ninja-build"

CONFIGURE_PRESET="$STF_CONFIGURE_PRESET"
BUILD_PRESET="$STF_BUILD_PRESET"

if [[ "$WITH_PYTHON" -eq 1 ]]; then
  PRESET_BUILD_DIR="build/linux-default"
else
  PRESET_BUILD_DIR="build/linux-lite"
fi

cmake_configure_args=(
  --preset "$CONFIGURE_PRESET"
  -B "$BUILD_DIR"
  -G "$GENERATOR"
  -USTF_USE_BIOSIG_SUBMODULE
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  "-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}"
  "-DSTF_BIOSIG_PROVIDER=SYSTEM"
)

if [[ "$WITH_PYTHON" -eq 1 ]]; then
  PYTHON_EXECUTABLE_GUESS="${PYTHON_EXECUTABLE:-$DEFAULT_SYSTEM_PYTHON}"
  PYTHON_EXECUTABLE_GUESS="$(pick_python_for_cmake "$PYTHON_EXECUTABLE_GUESS")" || {
    echo "ERROR: Could not find a usable Python with development files (tried 3.14..3.10 and python3)." >&2
    exit 1
  }

  if [[ ! -x "${PYTHON_EXECUTABLE_GUESS}" ]]; then
    echo "ERROR: Python executable not found or not executable: ${PYTHON_EXECUTABLE_GUESS}" >&2
    exit 1
  fi

  cmake_configure_args+=( "-DPython3_EXECUTABLE=${PYTHON_EXECUTABLE_GUESS}" )
  echo "==> Python executable: ${PYTHON_EXECUTABLE_GUESS}"
fi

stf_print_preset_selection "$BUILD_DIR"

echo "==> Configuring"
cmake "${cmake_configure_args[@]}"

echo "==> Building"
stf_build_with_optional_preset "$BUILD_DIR" "$PRESET_BUILD_DIR" "$BUILD_PRESET"

if [[ "$DO_INSTALL" -eq 1 ]]; then
  if [[ -e "$INSTALL_PREFIX" ]]; then
    echo "==> Removing existing install prefix: $INSTALL_PREFIX"
    rm -rf "$INSTALL_PREFIX"
  fi

  echo "==> Installing"
  cmake --install "$BUILD_DIR" --prefix "$INSTALL_PREFIX"
else
  echo "==> Skipping install (default). Use --install to run install step."
fi

if [[ -n "$PACKAGE_GENERATOR" ]]; then
  PACKAGE_CONFIG="${BUILD_DIR}/CPackConfig.cmake"
  if [[ ! -f "$PACKAGE_CONFIG" ]]; then
    echo "ERROR: CPack configuration not found at $PACKAGE_CONFIG" >&2
    exit 1
  fi

  echo "==> Packaging"
  cpack --config "$PACKAGE_CONFIG" -G "$PACKAGE_GENERATOR"
fi

if [[ "$DO_INSTALL" -eq 1 ]]; then
  INSTALLED_BINARY="${INSTALL_PREFIX}/bin/stimfit"
  if [[ -x "$INSTALLED_BINARY" ]]; then
    echo "==> Install complete"
    echo "==> Binary: ${INSTALLED_BINARY}"
  else
    echo "ERROR: Expected installed binary not found at ${INSTALLED_BINARY}" >&2
    exit 1
  fi
else
  echo "==> Build complete"
  echo "==> Build directory: ${BUILD_DIR}"
fi
