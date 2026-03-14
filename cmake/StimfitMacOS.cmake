include_guard(GLOBAL)

if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(CMAKE_INSTALL_NAME_DIR "@rpath")
endif()

function(stf_apply_macos_runtime_policy target_name)
  if(NOT APPLE)
    return()
  endif()

  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "stf_apply_macos_runtime_policy: target '${target_name}' does not exist")
  endif()

  get_target_property(_stf_target_type ${target_name} TYPE)
  if(NOT _stf_target_type)
    unset(_stf_target_type)
    return()
  endif()

  set(_stf_install_rpath "@loader_path")

  if(_stf_target_type STREQUAL "EXECUTABLE")
    get_target_property(_stf_is_macos_bundle ${target_name} MACOSX_BUNDLE)
    if(_stf_is_macos_bundle)
      list(APPEND _stf_install_rpath "@loader_path/../lib/stimfit")
    else()
      list(APPEND _stf_install_rpath "@loader_path/../lib/stimfit")
    endif()
  else()
    list(APPEND _stf_install_rpath "@loader_path/../lib/stimfit")
  endif()

  if(ARGN)
    foreach(_stf_extra_rpath IN LISTS ARGN)
      if(NOT "${_stf_extra_rpath}" STREQUAL "")
        list(APPEND _stf_install_rpath "${_stf_extra_rpath}")
      endif()
    endforeach()
  endif()
  list(REMOVE_DUPLICATES _stf_install_rpath)

  # On macOS, CMake otherwise emits install-time `install_name_tool -add_rpath`
  # commands for these targets. Re-running `cmake --install` against an existing
  # prefix then fails once the installed binary already carries the same LC_RPATH
  # entries. Building with the final install rpath avoids that extra install-time
  # mutation and makes installs idempotent.
  set_property(TARGET ${target_name} PROPERTY BUILD_WITH_INSTALL_RPATH ON)
  set_property(TARGET ${target_name} PROPERTY INSTALL_RPATH "${_stf_install_rpath}")

  if(_stf_target_type STREQUAL "SHARED_LIBRARY" OR _stf_target_type STREQUAL "MODULE_LIBRARY")
    set_property(TARGET ${target_name} PROPERTY INSTALL_NAME_DIR "@rpath")
  endif()

  unset(_stf_extra_rpath)
  unset(_stf_is_macos_bundle)
  unset(_stf_install_rpath)
  unset(_stf_target_type)
endfunction()
