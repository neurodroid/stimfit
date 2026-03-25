include_guard(GLOBAL)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_definitions($<$<CONFIG:Debug>:_STFDEBUG>)

if(STF_ENABLE_PYTHON)
  add_compile_definitions(WITH_PYTHON)
endif()

if(STF_ENABLE_PYTHON)
  if(STF_PY_SHELL_BACKEND STREQUAL "LEGACY")
    add_compile_definitions(STF_PY_SHELL_BACKEND_LEGACY)
  elseif(STF_PY_SHELL_BACKEND STREQUAL "JUPYTER")
    add_compile_definitions(STF_PY_SHELL_BACKEND_JUPYTER)
  else()
    add_compile_definitions(STF_PY_SHELL_BACKEND_MODERN)
  endif()
endif()

if(STF_ENABLE_PSLOPE)
  add_compile_definitions(WITH_PSLOPE)
endif()

if(STF_ENABLE_AUI)
  add_compile_definitions(WITH_AUIDOCVIEW)
endif()

if(STF_BUILD_MODULE)
  add_compile_definitions(MODULE_ONLY)
endif()

if(STF_WITH_BIOSIG)
  add_compile_definitions(WITH_BIOSIG)
endif()

add_compile_definitions(H5_USE_16_API)

if(HAVE_STRPTIME_H)
  add_compile_definitions(HAVE_STRPTIME_H=1)
endif()

add_compile_definitions(PACKAGE_VERSION=\"${PROJECT_VERSION}\")

add_library(stimfit_config INTERFACE)
target_include_directories(stimfit_config INTERFACE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_BINARY_DIR}
)

target_compile_options(stimfit_config INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4 /Zc:__cplusplus>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<CONFIG:Debug>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O0 -g3>
)

if(STF_BUILD_DEBIAN)
  if(DEFINED ENV{CPPFLAGS})
    separate_arguments(_debian_cppflags NATIVE_COMMAND "$ENV{CPPFLAGS}")
    target_compile_options(stimfit_config INTERFACE ${_debian_cppflags})
  endif()
  if(DEFINED ENV{CFLAGS})
    separate_arguments(_debian_cflags NATIVE_COMMAND "$ENV{CFLAGS}")
    target_compile_options(stimfit_config INTERFACE ${_debian_cflags})
  endif()
  if(DEFINED ENV{CXXFLAGS})
    separate_arguments(_debian_cxxflags NATIVE_COMMAND "$ENV{CXXFLAGS}")
    target_compile_options(stimfit_config INTERFACE ${_debian_cxxflags})
  endif()
  if(DEFINED ENV{LDFLAGS})
    separate_arguments(_debian_ldflags NATIVE_COMMAND "$ENV{LDFLAGS}")
    target_link_options(stimfit_config INTERFACE ${_debian_ldflags})
  endif()
endif()

if(STF_ENABLE_PYTHON AND NOT STF_BUILD_MODULE AND TARGET stimfit::python AND Python3_Interpreter_FOUND)
  stf_get_python_bootstrap(_stf_python_bootstrap)

  set(_stf_wxpython_include_dir "${STF_WXPYTHON_INCLUDE_DIR}")
  if(_stf_wxpython_include_dir)
    if(NOT EXISTS "${_stf_wxpython_include_dir}/wxPython/wxpy_api.h")
      message(FATAL_ERROR "STF_WXPYTHON_INCLUDE_DIR='${_stf_wxpython_include_dir}' does not contain wxPython/wxpy_api.h")
    endif()
  else()
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "${_stf_python_bootstrap}import os, wx; print(os.path.join(os.path.dirname(wx.__spec__.origin), 'include'))"
      OUTPUT_VARIABLE STF_WXPYTHON_INCLUDE_FROM_WX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    file(TO_CMAKE_PATH "${STF_WXPYTHON_INCLUDE_FROM_WX}" STF_WXPYTHON_INCLUDE_FROM_WX)

    set(_stf_wxpython_include_hints "")
    if(STF_WXPYTHON_INCLUDE_FROM_WX)
      list(APPEND _stf_wxpython_include_hints "${STF_WXPYTHON_INCLUDE_FROM_WX}")
    endif()
    if(STF_WINDOWS_PYTHON_EXTRA_PATHS)
      list(APPEND _stf_wxpython_include_hints ${STF_WINDOWS_PYTHON_EXTRA_PATHS})
    endif()
    if(DEFINED STF_WXPYTHON_DIR AND NOT "${STF_WXPYTHON_DIR}" STREQUAL "")
      list(APPEND _stf_wxpython_include_hints "${STF_WXPYTHON_DIR}")
    endif()
    if(DEFINED STF_PYTHON_PLATLIB AND NOT "${STF_PYTHON_PLATLIB}" STREQUAL "")
      list(APPEND _stf_wxpython_include_hints "${STF_PYTHON_PLATLIB}")
    endif()
    if(DEFINED STF_PYTHON_PURELIB AND NOT "${STF_PYTHON_PURELIB}" STREQUAL "")
      list(APPEND _stf_wxpython_include_hints "${STF_PYTHON_PURELIB}")
    endif()

    foreach(_stf_wx_hint IN LISTS _stf_wxpython_include_hints)
      if(EXISTS "${_stf_wx_hint}/wxPython/wxpy_api.h")
        set(_stf_wxpython_include_dir "${_stf_wx_hint}")
        break()
      endif()
    endforeach()

    if(NOT _stf_wxpython_include_dir)
      find_path(_stf_wxpython_include_dir
        NAMES wxPython/wxpy_api.h
        HINTS ${_stf_wxpython_include_hints}
        PATH_SUFFIXES wx/include include
      )
    endif()
  endif()

  if(_stf_wxpython_include_dir)
    set(STF_WXPYTHON_INCLUDE_DIR "${_stf_wxpython_include_dir}" CACHE PATH "Path containing wxPython/wxpy_api.h (e.g. <Phoenix>/wx/include)" FORCE)
    target_include_directories(stimfit::python INTERFACE "${STF_WXPYTHON_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "Could not locate wxPython/wxpy_api.h. Set STF_WXPYTHON_INCLUDE_DIR (for Phoenix, typically <Phoenix>/wx/include).")
  endif()

  unset(_stf_wxpython_include_dir)
  unset(_stf_wxpython_include_hints)
  unset(_stf_wx_hint)
  unset(STF_WXPYTHON_INCLUDE_FROM_WX)
  unset(_stf_python_bootstrap)
endif()
