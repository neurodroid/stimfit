#!/bin/bash
set -euo pipefail

# Configure, build, and install Stimfit on macOS using CMake,
# including production of a macOS .app bundle.
#
# Usage:
#   ./build_macos_cmake.sh
#   ./build_macos_cmake.sh --with-python
#   ./build_macos_cmake.sh --without-python
#   ./build_macos_cmake.sh --with-python --install-prefix build/macos-app-py/install
#
# Optional environment overrides:
#   PYTHON_EXECUTABLE=/opt/local/bin/python3.14

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cmake/StimfitPresetHelpers.sh
source "${SCRIPT_DIR}/cmake/StimfitPresetHelpers.sh"

BUILD_DIR="build/macos-app"
INSTALL_PREFIX="build/macos-app/install"
GENERATOR="Ninja"
WITH_PYTHON=1
DEFAULT_MACPORTS_PYTHON="/opt/local/bin/python3.14"

pick_python_for_cmake() {
  local requested="$1"
  local candidates=(
    "$requested"
    /opt/local/bin/python3.13
    /opt/local/bin/python3.12
    /opt/local/bin/python3.11
    /opt/local/bin/python3.10
  )

  for py in "${candidates[@]}"; do
    [[ -x "$py" ]] || continue
    if "$py" - <<'PY' >/dev/null 2>&1
import sysconfig
ok = bool(sysconfig.get_config_var('INCLUDEPY')) and bool(sysconfig.get_config_var('LIBDIR'))
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
      BUILD_DIR="build/macos-app-py"
      INSTALL_PREFIX="build/macos-app-py/install"
      shift
      ;;
    --without-python)
      WITH_PYTHON=0
      BUILD_DIR="build/macos-app-nopython"
      INSTALL_PREFIX="build/macos-app-nopython/install"
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
    -h|--help)
      sed -n '1,35p' "$0"
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
  "macos-ninja-app-python" \
  "macos-ninja-app" \
  "macos-ninja-app-python-stimfit" \
  "macos-ninja-app-stimfit"

CONFIGURE_PRESET="$STF_CONFIGURE_PRESET"
BUILD_PRESET="$STF_BUILD_PRESET"

if [[ "$WITH_PYTHON" -eq 1 ]]; then
  PRESET_BUILD_DIR="build/macos-app"
else
  PRESET_BUILD_DIR="build/macos-app-nopython"
fi

cmake_configure_args=(
  --preset "$CONFIGURE_PRESET"
  -B "$BUILD_DIR"
  -G "$GENERATOR"
  -USTF_USE_BIOSIG_SUBMODULE
  "-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}"
)

if [[ "$WITH_PYTHON" -eq 1 ]]; then
  PYTHON_EXECUTABLE_GUESS="${PYTHON_EXECUTABLE:-$DEFAULT_MACPORTS_PYTHON}"
  PYTHON_EXECUTABLE_GUESS="$(pick_python_for_cmake "$PYTHON_EXECUTABLE_GUESS")" || {
    echo "ERROR: Could not find a usable MacPorts Python with development files (tried 3.14..3.10)." >&2
    exit 1
  }

  if [[ ! -x "${PYTHON_EXECUTABLE_GUESS}" ]]; then
    echo "ERROR: Python executable not found or not executable: ${PYTHON_EXECUTABLE_GUESS}" >&2
    exit 1
  fi

  PYTHON_BIN_DIR="$(dirname "${PYTHON_EXECUTABLE_GUESS}")"
  WX_CONFIG_CANDIDATE="${PYTHON_BIN_DIR}/wx-config"
  if [[ ! -x "${WX_CONFIG_CANDIDATE}" ]]; then
    PYTHON_FRAMEWORK_WX_CONFIG="$(${PYTHON_EXECUTABLE_GUESS} -c 'import pathlib, sys; print((pathlib.Path(sys.base_prefix) / "bin" / "wx-config").as_posix())')"
    if [[ -x "${PYTHON_FRAMEWORK_WX_CONFIG}" ]]; then
      WX_CONFIG_CANDIDATE="${PYTHON_FRAMEWORK_WX_CONFIG}"
    fi
  fi
  if [[ -x "${WX_CONFIG_CANDIDATE}" ]]; then
    cmake_configure_args+=( "-DwxWidgets_CONFIG_EXECUTABLE=${WX_CONFIG_CANDIDATE}" )
  fi

  cmake_configure_args+=( "-DPython3_EXECUTABLE=${PYTHON_EXECUTABLE_GUESS}" )

  echo "==> Python executable: ${PYTHON_EXECUTABLE_GUESS}"
fi

stf_print_preset_selection "$BUILD_DIR"

echo "==> Configuring"
cmake "${cmake_configure_args[@]}"

echo "==> Building"
stf_build_with_optional_preset "$BUILD_DIR" "$PRESET_BUILD_DIR" "$BUILD_PRESET"

if [[ -e "$INSTALL_PREFIX" ]]; then
  echo "==> Removing existing install prefix: $INSTALL_PREFIX"
  rm -rf "$INSTALL_PREFIX"
fi

echo "==> Installing"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_PREFIX"

APP_BUNDLE="${INSTALL_PREFIX}/stimfit.app"

if [[ -d "$APP_BUNDLE" ]]; then
  echo "==> App bundle created: $APP_BUNDLE"
  echo "==> Binary: ${APP_BUNDLE}/Contents/MacOS/stimfit"
else
  echo "ERROR: Expected app bundle not found at $APP_BUNDLE" >&2
  exit 1
fi
