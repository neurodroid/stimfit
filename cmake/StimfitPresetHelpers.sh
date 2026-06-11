#!/bin/bash

stf_select_presets() {
  local with_python="$1"
  local python_configure_preset="$2"
  local no_python_configure_preset="$3"
  local python_build_preset="${4:-}"
  local no_python_build_preset="${5:-}"

  if [[ "$with_python" -eq 1 ]]; then
    STF_CONFIGURE_PRESET="$python_configure_preset"
    STF_BUILD_PRESET="$python_build_preset"
  else
    STF_CONFIGURE_PRESET="$no_python_configure_preset"
    STF_BUILD_PRESET="$no_python_build_preset"
  fi
}

stf_print_preset_selection() {
  local build_dir="$1"

  echo "==> Configure preset: ${STF_CONFIGURE_PRESET}"
  if [[ -n "${STF_BUILD_PRESET:-}" ]]; then
    echo "==> Build preset: ${STF_BUILD_PRESET}"
  fi
  echo "==> Build directory: ${build_dir}"
}

stf_build_with_optional_preset() {
  local build_dir="$1"
  local default_build_dir="$2"
  local build_preset="${3:-}"

  if [[ -n "$build_preset" && "$build_dir" == "$default_build_dir" ]]; then
    cmake --build --preset "$build_preset"
  else
    cmake --build "$build_dir"
  fi
}
