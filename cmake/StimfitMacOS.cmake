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

  set(_stf_install_rpath
    "@loader_path"
    "@loader_path/../lib/stimfit"
  )

  if(_stf_target_type STREQUAL "EXECUTABLE")
    get_target_property(_stf_is_macos_bundle ${target_name} MACOSX_BUNDLE)
    if(_stf_is_macos_bundle)
      list(APPEND _stf_install_rpath "@loader_path/../../../../lib/stimfit")
    endif()
  endif()

  if(DEFINED CMAKE_INSTALL_PREFIX AND NOT "${CMAKE_INSTALL_PREFIX}" STREQUAL "")
    list(APPEND _stf_install_rpath "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/stimfit")
  endif()

  if(ARGN)
    foreach(_stf_extra_rpath IN LISTS ARGN)
      if(NOT "${_stf_extra_rpath}" STREQUAL "")
        list(APPEND _stf_install_rpath "${_stf_extra_rpath}")
      endif()
    endforeach()
  endif()
  list(REMOVE_DUPLICATES _stf_install_rpath)

  set_property(TARGET ${target_name} PROPERTY BUILD_WITH_INSTALL_RPATH OFF)
  set_property(TARGET ${target_name} PROPERTY INSTALL_RPATH "${_stf_install_rpath}")

  if(_stf_target_type STREQUAL "SHARED_LIBRARY" OR _stf_target_type STREQUAL "MODULE_LIBRARY")
    set_property(TARGET ${target_name} PROPERTY INSTALL_NAME_DIR "@rpath")
  endif()

  unset(_stf_extra_rpath)
  unset(_stf_is_macos_bundle)
  unset(_stf_install_rpath)
  unset(_stf_target_type)
endfunction()
