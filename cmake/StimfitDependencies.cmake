include_guard(GLOBAL)

include(CheckSymbolExists)

find_package(Threads REQUIRED)

find_package(PkgConfig QUIET)

if(STF_HDF5_PREFIX)
  list(PREPEND CMAKE_PREFIX_PATH "${STF_HDF5_PREFIX}")
  list(PREPEND CMAKE_LIBRARY_PATH "${STF_HDF5_PREFIX}/lib")
  list(PREPEND CMAKE_INCLUDE_PATH "${STF_HDF5_PREFIX}/include")
endif()

# Force FindHDF5 module mode and prevent it from delegating to an HDF5
# config package, because this vendor HDF5 export only ships RelWithDebInfo
# imported targets and breaks multi-config generators (VS MinSizeRel/Debug).
set(HDF5_NO_FIND_PACKAGE_CONFIG_FILE TRUE)
find_package(HDF5 MODULE COMPONENTS C HL QUIET)
unset(HDF5_NO_FIND_PACKAGE_CONFIG_FILE)
if(HDF5_FOUND)
  add_library(stimfit::hdf5 INTERFACE IMPORTED)
  target_include_directories(stimfit::hdf5 INTERFACE ${HDF5_INCLUDE_DIRS})
  if(HDF5_DEFINITIONS)
    target_compile_definitions(stimfit::hdf5 INTERFACE ${HDF5_DEFINITIONS})
  endif()
  if(HDF5_C_DEFINITIONS)
    target_compile_definitions(stimfit::hdf5 INTERFACE ${HDF5_C_DEFINITIONS})
  endif()
  set(_stimfit_hdf5_libs ${HDF5_LIBRARIES})
  if(HDF5_HL_LIBRARIES)
    list(APPEND _stimfit_hdf5_libs ${HDF5_HL_LIBRARIES})
  endif()
  if(HDF5_C_HL_LIBRARIES)
    list(APPEND _stimfit_hdf5_libs ${HDF5_C_HL_LIBRARIES})
  endif()
  list(REMOVE_DUPLICATES _stimfit_hdf5_libs)
  target_link_libraries(stimfit::hdf5 INTERFACE ${_stimfit_hdf5_libs})
  unset(_stimfit_hdf5_libs)
else()
  find_library(HDF5_LIBRARY NAMES hdf5)
  find_library(HDF5_HL_LIBRARY NAMES hdf5_hl)
  if(HDF5_LIBRARY AND HDF5_HL_LIBRARY)
    add_library(stimfit::hdf5 INTERFACE IMPORTED)
    target_link_libraries(stimfit::hdf5 INTERFACE ${HDF5_LIBRARY} ${HDF5_HL_LIBRARY})
    if(STF_HDF5_PREFIX)
      target_include_directories(stimfit::hdf5 INTERFACE ${STF_HDF5_PREFIX}/include)
    endif()
  else()
    message(FATAL_ERROR "HDF5 not found. Install HDF5 development files or set STF_HDF5_PREFIX to your HDF5 prefix.")
  endif()
endif()

check_symbol_exists(strptime "time.h" HAVE_STRPTIME_H)

if(STF_WITH_BIOSIG AND NOT STF_WITH_BIOSIGLITE)
  set(STF_BIOSIG_USE_SUBMODULE OFF)
  if(STF_USE_BIOSIG_SUBMODULE AND EXISTS "${CMAKE_SOURCE_DIR}/src/biosig/biosig4c++/CMakeLists.txt")
    set(STF_BIOSIG_USE_SUBMODULE ON)
  endif()

  if(NOT STF_BIOSIG_USE_SUBMODULE)
    find_path(BIOSIG_INCLUDE_DIR NAMES biosig.h)
    find_library(BIOSIG_LIBRARY NAMES biosig)
    if(BIOSIG_LIBRARY)
      add_library(stimfit::biosig UNKNOWN IMPORTED)
      set_target_properties(stimfit::biosig PROPERTIES IMPORTED_LOCATION "${BIOSIG_LIBRARY}")
      if(BIOSIG_INCLUDE_DIR)
        target_include_directories(stimfit::biosig INTERFACE "${BIOSIG_INCLUDE_DIR}")
      else()
        get_filename_component(_biosig_lib_dir "${BIOSIG_LIBRARY}" DIRECTORY)
        set(_biosig_include_candidates
          "${_biosig_lib_dir}"
          "${_biosig_lib_dir}/include"
          "${_biosig_lib_dir}/../include"
        )
        foreach(_biosig_inc IN LISTS _biosig_include_candidates)
          if(EXISTS "${_biosig_inc}/biosig.h")
            target_include_directories(stimfit::biosig INTERFACE "${_biosig_inc}")
            break()
          endif()
        endforeach()
        unset(_biosig_inc)
        unset(_biosig_include_candidates)
        unset(_biosig_lib_dir)
      endif()
    else()
      message(WARNING "STF_WITH_BIOSIG is ON but src/biosig was unavailable and external libbiosig was not found; turning it OFF")
      set(STF_WITH_BIOSIG OFF CACHE BOOL "Use external libbiosig if available" FORCE)
    endif()
  endif()
endif()

find_library(FFTW3_LIBRARY NAMES fftw3 libfftw3-3 fftw3-3 REQUIRED)
add_library(stimfit::fftw3 UNKNOWN IMPORTED)
set_target_properties(stimfit::fftw3 PROPERTIES IMPORTED_LOCATION "${FFTW3_LIBRARY}")

# Try to locate fftw3.h for custom/local installs (e.g. ~/libs/fftw3).
get_filename_component(_fftw3_lib_dir "${FFTW3_LIBRARY}" DIRECTORY)
set(_fftw3_include_candidates
  "${_fftw3_lib_dir}"
  "${_fftw3_lib_dir}/include"
  "${_fftw3_lib_dir}/../include"
)
foreach(_fftw3_inc IN LISTS _fftw3_include_candidates)
  if(EXISTS "${_fftw3_inc}/fftw3.h")
    target_include_directories(stimfit::fftw3 INTERFACE "${_fftw3_inc}")
    break()
  endif()
endforeach()
unset(_fftw3_inc)
unset(_fftw3_include_candidates)
unset(_fftw3_lib_dir)

find_library(OPENBLAS_LIBRARY NAMES openblas)
if(OPENBLAS_LIBRARY)
  add_library(stimfit::lapack UNKNOWN IMPORTED)
  set_target_properties(stimfit::lapack PROPERTIES IMPORTED_LOCATION "${OPENBLAS_LIBRARY}")
  target_compile_definitions(stimfit::lapack INTERFACE WITH_OPENBLAS HAVE_LAPACK)
else()
  find_package(LAPACK QUIET)
  if(LAPACK_FOUND)
    add_library(stimfit::lapack INTERFACE IMPORTED)
    target_link_libraries(stimfit::lapack INTERFACE ${LAPACK_LIBRARIES})
    target_compile_definitions(stimfit::lapack INTERFACE HAVE_LAPACK)
  else()
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
      add_library(stimfit::lapack INTERFACE IMPORTED)
      target_link_libraries(stimfit::lapack INTERFACE ${BLAS_LIBRARIES})
      target_compile_definitions(stimfit::lapack INTERFACE HAVE_LAPACK)
    else()
      find_library(LAPACK_LIBRARY NAMES lapack lapack3 lapack-3 REQUIRED)
      add_library(stimfit::lapack UNKNOWN IMPORTED)
      set_target_properties(stimfit::lapack PROPERTIES IMPORTED_LOCATION "${LAPACK_LIBRARY}")
      target_compile_definitions(stimfit::lapack INTERFACE HAVE_LAPACK)
    endif()
  endif()
endif()

if(NOT STF_BUILD_MODULE)
  find_package(wxWidgets REQUIRED COMPONENTS base core adv aui net)
  add_library(stimfit::wx INTERFACE IMPORTED)
  target_include_directories(stimfit::wx INTERFACE ${wxWidgets_INCLUDE_DIRS})
  target_link_libraries(stimfit::wx INTERFACE ${wxWidgets_LIBRARIES})
  target_compile_definitions(stimfit::wx INTERFACE ${wxWidgets_DEFINITIONS})
else()
  add_library(stimfit::wx INTERFACE IMPORTED)
endif()

if(STF_ENABLE_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  add_library(stimfit::python INTERFACE IMPORTED)
  target_include_directories(stimfit::python INTERFACE ${Python3_INCLUDE_DIRS})
  target_link_libraries(stimfit::python INTERFACE ${Python3_LIBRARIES})

  set(_stf_python_bootstrap "")
  if(WIN32)
    set(_stf_python_extra_paths "${STF_WINDOWS_PYTHON_EXTRA_PATHS}")
    list(FILTER _stf_python_extra_paths EXCLUDE REGEX "^$")
    if(_stf_python_extra_paths)
      foreach(_stf_path IN LISTS _stf_python_extra_paths)
        string(REPLACE "\\" "\\\\" _stf_path_escaped "${_stf_path}")
        set(_stf_python_bootstrap "${_stf_python_bootstrap}import sys; sys.path.insert(0, r'${_stf_path_escaped}'); ")
      endforeach()
    endif()
  endif()

  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "${_stf_python_bootstrap}import sysconfig; print(sysconfig.get_paths()['platlib'])"
      OUTPUT_VARIABLE STF_PYTHON_PLATLIB
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    file(TO_CMAKE_PATH "${STF_PYTHON_PLATLIB}" STF_PYTHON_PLATLIB)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "${_stf_python_bootstrap}import sysconfig; print(sysconfig.get_paths()['purelib'])"
      OUTPUT_VARIABLE STF_PYTHON_PURELIB
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    file(TO_CMAKE_PATH "${STF_PYTHON_PURELIB}" STF_PYTHON_PURELIB)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "${_stf_python_bootstrap}import sysconfig; print(sysconfig.get_path('stdlib'))"
      OUTPUT_VARIABLE STF_PYTHON_STDLIB
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    file(TO_CMAKE_PATH "${STF_PYTHON_STDLIB}" STF_PYTHON_STDLIB)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "${_stf_python_bootstrap}import pathlib, sys; print((pathlib.Path(sys.base_prefix) / 'DLLs').as_posix())"
      OUTPUT_VARIABLE STF_PYTHON_DLLS_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    file(TO_CMAKE_PATH "${STF_PYTHON_DLLS_DIR}" STF_PYTHON_DLLS_DIR)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "${_stf_python_bootstrap}import pathlib, sys; dll = pathlib.Path(sys.base_prefix) / ('python%d%d.dll' % (sys.version_info[0], sys.version_info[1])); print(dll.as_posix())"
      OUTPUT_VARIABLE STF_PYTHON_RUNTIME_DLL
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    file(TO_CMAKE_PATH "${STF_PYTHON_RUNTIME_DLL}" STF_PYTHON_RUNTIME_DLL)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "${_stf_python_bootstrap}import numpy; print(numpy.get_include())"
      OUTPUT_VARIABLE STF_NUMPY_INCLUDE
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )

    if(NOT STF_BUILD_MODULE)
      execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "${_stf_python_bootstrap}import wx, pathlib; print(pathlib.Path(wx.__file__).resolve().parent.as_posix())"
        OUTPUT_VARIABLE STF_WXPYTHON_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE _stf_wxpy_err
        RESULT_VARIABLE _stf_wxpy_status
      )
      file(TO_CMAKE_PATH "${STF_WXPYTHON_DIR}" STF_WXPYTHON_DIR)

      execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "${_stf_python_bootstrap}import os, sys, wx; sys.stdout.write(os.path.join(os.path.dirname(wx.__spec__.origin), 'include'))"
        OUTPUT_VARIABLE STF_WXPYTHON_INCLUDE_FROM_WX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      file(TO_CMAKE_PATH "${STF_WXPYTHON_INCLUDE_FROM_WX}" STF_WXPYTHON_INCLUDE_FROM_WX)

      set(_stf_wxpython_include_dir "${STF_WXPYTHON_INCLUDE_DIR}")
      if(_stf_wxpython_include_dir)
        if(NOT EXISTS "${_stf_wxpython_include_dir}/wxPython/wxpy_api.h")
          message(FATAL_ERROR "STF_WXPYTHON_INCLUDE_DIR='${_stf_wxpython_include_dir}' does not contain wxPython/wxpy_api.h")
        endif()
      else()
        set(_stf_wxpython_include_hints "")
        if(STF_WXPYTHON_INCLUDE_FROM_WX)
          list(APPEND _stf_wxpython_include_hints "${STF_WXPYTHON_INCLUDE_FROM_WX}")
        endif()
        if(STF_WINDOWS_PYTHON_EXTRA_PATHS)
          list(APPEND _stf_wxpython_include_hints ${STF_WINDOWS_PYTHON_EXTRA_PATHS})
        endif()
        if(STF_WXPYTHON_DIR)
          list(APPEND _stf_wxpython_include_hints "${STF_WXPYTHON_DIR}")
        endif()
        if(STF_PYTHON_PLATLIB)
          list(APPEND _stf_wxpython_include_hints "${STF_PYTHON_PLATLIB}")
        endif()
        if(STF_PYTHON_PURELIB)
          list(APPEND _stf_wxpython_include_hints "${STF_PYTHON_PURELIB}")
        endif()

        foreach(_stf_wx_hint IN LISTS _stf_wxpython_include_hints)
          if(EXISTS "${_stf_wx_hint}/wxPython/wxpy_api.h")
            set(_stf_wxpython_include_dir "${_stf_wx_hint}")
            break()
          endif()
        endforeach()
        unset(_stf_wx_hint)

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

      if(STF_PY_SHELL_BACKEND STREQUAL "LEGACY")
        execute_process(
          COMMAND ${Python3_EXECUTABLE} -c "${_stf_python_bootstrap}from wx.py import shell"
          RESULT_VARIABLE _stf_legacy_shell_status
          ERROR_QUIET
        )
        if(NOT _stf_legacy_shell_status EQUAL 0)
          message(FATAL_ERROR "STF_PY_SHELL_BACKEND=LEGACY requires 'wx.py.shell', which is not available in this Python environment.")
        endif()
      else()
        if(NOT _stf_wxpy_status EQUAL 0)
          string(STRIP "${_stf_wxpy_err}" _stf_wxpy_err)
          message(FATAL_ERROR "STF_PY_SHELL_BACKEND=MODERN requires wxPython to be importable from Python3_EXECUTABLE. ${_stf_wxpy_err}")
        endif()
      endif()
    endif()

    if(STF_NUMPY_INCLUDE STREQUAL "")
      message(FATAL_ERROR "NumPy headers were not detected. Install numpy into the selected Python environment.")
    endif()

    message(STATUS "Stimfit Python shell backend: ${STF_PY_SHELL_BACKEND}")
    message(STATUS "Stimfit Python interpreter: ${Python3_EXECUTABLE}")
    message(STATUS "Stimfit Python platlib: ${STF_PYTHON_PLATLIB}")
    if(WIN32 AND NOT "${STF_WINDOWS_PYTHON_EXTRA_PATHS}" STREQUAL "")
      message(STATUS "Stimfit Python extra paths: ${STF_WINDOWS_PYTHON_EXTRA_PATHS}")
    endif()
  endif()

  find_package(SWIG QUIET)
  unset(_stf_python_bootstrap)
  unset(_stf_wxpython_include_dir)
  unset(_stf_wxpython_include_hints)
  unset(STF_WXPYTHON_INCLUDE_FROM_WX)
  unset(_stf_python_extra_paths)
  unset(_stf_path_escaped)
  unset(_stf_path)
endif()
