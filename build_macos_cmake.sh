#!/bin/bash
set -euo pipefail

# Configure, build, and install Stimfit on macOS using CMake,
# including production of a macOS .app bundle.
#
# Usage:
#   ./build_macos_cmake.sh
#   ./build_macos_cmake.sh --with-python
#   ./build_macos_cmake.sh --with-python --install-prefix build/macos-app-py/install
#
# Optional environment overrides:
#   PYTHON_EXECUTABLE=/opt/local/bin/python3.14

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

cmake_args=(
  -S .
  -B "$BUILD_DIR"
  -G "$GENERATOR"
  -DSTF_MACOS_APP_BUNDLE=ON
  -DSTF_WITH_BIOSIGLITE=ON
)

if [[ "$WITH_PYTHON" -eq 1 ]]; then
  cmake_args+=( -DSTF_ENABLE_PYTHON=ON )

  PYTHON_EXECUTABLE_GUESS="${PYTHON_EXECUTABLE:-$DEFAULT_MACPORTS_PYTHON}"
  PYTHON_EXECUTABLE_GUESS="$(pick_python_for_cmake "$PYTHON_EXECUTABLE_GUESS")" || {
    echo "ERROR: Could not find a usable MacPorts Python with development files (tried 3.14..3.10)." >&2
    exit 1
  }
  cmake_args+=( "-DPython3_EXECUTABLE=${PYTHON_EXECUTABLE_GUESS}" )

  if [[ ! -x "${PYTHON_EXECUTABLE_GUESS}" ]]; then
    echo "ERROR: Python executable not found or not executable: ${PYTHON_EXECUTABLE_GUESS}" >&2
    exit 1
  fi

  echo "==> Python executable: ${PYTHON_EXECUTABLE_GUESS}"
else
  cmake_args+=( -DSTF_ENABLE_PYTHON=OFF )
fi

echo "==> Configuring"
cmake "${cmake_args[@]}"

echo "==> Building"
cmake --build "$BUILD_DIR"

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
