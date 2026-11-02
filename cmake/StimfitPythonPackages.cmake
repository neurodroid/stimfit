include_guard(GLOBAL)

# Resolve importable modules together with the wheel's DLL directory, version
# metadata, and licenses. Keep each module and its companions from one root so
# an extra site-packages directory cannot mix two builds of the same wheel.
function(stf_collect_python_package_sources output)
  cmake_parse_arguments(PARSE_ARGV 1 _stf "" "" "PACKAGES;ROOTS")
  list(REMOVE_DUPLICATES _stf_ROOTS)
  set(_stf_sources)
  set(_stf_missing)
  foreach(_stf_pkg IN LISTS _stf_PACKAGES)
    set(_stf_found FALSE)
    foreach(_stf_root IN LISTS _stf_ROOTS)
      if(_stf_root STREQUAL "")
        continue()
      endif()
      set(_stf_module_sources)
      if(IS_DIRECTORY "${_stf_root}/${_stf_pkg}")
        list(APPEND _stf_module_sources "${_stf_root}/${_stf_pkg}")
      elseif(EXISTS "${_stf_root}/${_stf_pkg}.py")
        list(APPEND _stf_module_sources "${_stf_root}/${_stf_pkg}.py")
      else()
        file(GLOB _stf_module_sources
          "${_stf_root}/${_stf_pkg}.pyd"
          "${_stf_root}/${_stf_pkg}.*.pyd")
      endif()
      if(NOT _stf_module_sources)
        continue()
      endif()

      # Wheel distribution names can differ from their Python import names.
      string(TOLOWER "${_stf_pkg}" _stf_distribution)
      if(_stf_pkg STREQUAL "PIL")
        set(_stf_distribution pillow)
      elseif(_stf_pkg STREQUAL "dateutil")
        set(_stf_distribution python_dateutil)
      elseif(_stf_pkg STREQUAL "wx")
        set(_stf_distribution wxpython)
      endif()
      foreach(_stf_libs IN ITEMS "${_stf_pkg}.libs" "${_stf_distribution}.libs")
        if(IS_DIRECTORY "${_stf_root}/${_stf_libs}")
          list(APPEND _stf_module_sources "${_stf_root}/${_stf_libs}")
        endif()
      endforeach()
      if(_stf_pkg STREQUAL "matplotlib")
        # Matplotlib exposes these additional top-level imports from its wheel.
        foreach(_stf_companion IN ITEMS mpl_toolkits pylab.py)
          if(EXISTS "${_stf_root}/${_stf_companion}")
            list(APPEND _stf_module_sources "${_stf_root}/${_stf_companion}")
          endif()
        endforeach()
      endif()
      file(GLOB _stf_metadata
        "${_stf_root}/${_stf_distribution}-*.dist-info"
        "${_stf_root}/${_stf_distribution}-*.egg-info")
      list(APPEND _stf_sources ${_stf_module_sources} ${_stf_metadata})
      set(_stf_found TRUE)
      break()
    endforeach()
    if(NOT _stf_found)
      list(APPEND _stf_missing "${_stf_pkg}")
    endif()
  endforeach()
  if(_stf_missing)
    message(FATAL_ERROR
      "Required Windows Python modules were not found: ${_stf_missing}. "
      "Searched: ${_stf_ROOTS}. Install the corresponding packages for "
      "${Python3_EXECUTABLE}, or add their site-packages directory to "
      "STF_WINDOWS_PYTHON_EXTRA_PATHS.")
  endif()
  list(REMOVE_DUPLICATES _stf_sources)
  set(${output} "${_stf_sources}" PARENT_SCOPE)
endfunction()
