include_guard(GLOBAL)

include(CheckSymbolExists)
include(StimfitBiosigExternal)

function(stf_get_python_bootstrap out_var)
  set(_stf_python_bootstrap "")

  if(WIN32)
    set(_stf_python_extra_paths ${STF_WINDOWS_PYTHON_EXTRA_PATHS})
    list(FILTER _stf_python_extra_paths EXCLUDE REGEX "^$")
    foreach(_stf_path IN LISTS _stf_python_extra_paths)
      string(REPLACE "\\" "\\\\" _stf_path_escaped "${_stf_path}")
      string(APPEND _stf_python_bootstrap "import sys; sys.path.insert(0, r'${_stf_path_escaped}'); ")
    endforeach()
  endif()

  set(${out_var} "${_stf_python_bootstrap}" PARENT_SCOPE)
endfunction()

function(stf_find_macos_wx_config out_var)
  set(_stf_wx_config_candidate "")

  if(APPLE)
    find_program(_stf_wx_config_candidate
      NAMES wx-config
      HINTS "/opt/local/bin"
      NO_DEFAULT_PATH
    )

    if(NOT _stf_wx_config_candidate)
      file(GLOB _stf_macports_python_bins LIST_DIRECTORIES FALSE "/opt/local/Library/Frameworks/Python.framework/Versions/*/bin")
      foreach(_stf_python_bin IN LISTS _stf_macports_python_bins)
        find_program(_stf_wx_config_from_python
          NAMES wx-config
          HINTS "${_stf_python_bin}"
          NO_DEFAULT_PATH
        )
        if(_stf_wx_config_from_python)
          set(_stf_wx_config_candidate "${_stf_wx_config_from_python}")
          break()
        endif()
      endforeach()
      unset(_stf_wx_config_from_python)
      unset(_stf_python_bin)
      unset(_stf_macports_python_bins)
    endif()
  endif()

  set(${out_var} "${_stf_wx_config_candidate}" PARENT_SCOPE)
endfunction()

function(stf_import_biosig_target)
  if(TARGET stimfit::biosig)
    return()
  endif()

  set(_stf_biosig_include_dir "${BIOSIG_INCLUDE_DIR}")
  set(_stf_biosig_library "${BIOSIG_LIBRARY}")

  if(NOT _stf_biosig_library)
    find_library(_stf_biosig_library
      NAMES biosig biosig2
      HINTS
        /lib
        /lib64
        /usr/lib
        /usr/lib64
        /lib/${CMAKE_LIBRARY_ARCHITECTURE}
        /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
    )
  endif()

  if(NOT _stf_biosig_library)
    set(_stf_biosig_library_candidates
      "/lib/${CMAKE_LIBRARY_ARCHITECTURE}/libbiosig.so"
      "/lib/${CMAKE_LIBRARY_ARCHITECTURE}/libbiosig.so.3"
      "/lib/${CMAKE_LIBRARY_ARCHITECTURE}/libbiosig.a"
      "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/libbiosig.so"
      "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/libbiosig.so.3"
      "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/libbiosig.a"
      "/lib/x86_64-linux-gnu/libbiosig.so"
      "/lib/x86_64-linux-gnu/libbiosig.so.3"
      "/lib/x86_64-linux-gnu/libbiosig.a"
      "/usr/lib/x86_64-linux-gnu/libbiosig.so"
      "/usr/lib/x86_64-linux-gnu/libbiosig.so.3"
      "/usr/lib/x86_64-linux-gnu/libbiosig.a"
      "/lib/aarch64-linux-gnu/libbiosig.so"
      "/lib/aarch64-linux-gnu/libbiosig.so.3"
      "/lib/aarch64-linux-gnu/libbiosig.a"
      "/usr/lib/aarch64-linux-gnu/libbiosig.so"
      "/usr/lib/aarch64-linux-gnu/libbiosig.so.3"
      "/usr/lib/aarch64-linux-gnu/libbiosig.a"
    )
    foreach(_stf_biosig_candidate IN LISTS _stf_biosig_library_candidates)
      if(EXISTS "${_stf_biosig_candidate}")
        set(_stf_biosig_library "${_stf_biosig_candidate}")
        break()
      endif()
    endforeach()
    unset(_stf_biosig_candidate)
    unset(_stf_biosig_library_candidates)
  endif()

  if(_stf_biosig_library AND NOT _stf_biosig_include_dir)
    get_filename_component(_stf_biosig_lib_dir "${_stf_biosig_library}" DIRECTORY)
    set(_stf_biosig_include_candidates
      "${_stf_biosig_lib_dir}"
      "${_stf_biosig_lib_dir}/include"
      "${_stf_biosig_lib_dir}/../include"
    )
    foreach(_stf_biosig_inc IN LISTS _stf_biosig_include_candidates)
      if(EXISTS "${_stf_biosig_inc}/biosig.h")
        set(_stf_biosig_include_dir "${_stf_biosig_inc}")
        break()
      endif()
    endforeach()
    unset(_stf_biosig_inc)
    unset(_stf_biosig_include_candidates)
    unset(_stf_biosig_lib_dir)
  endif()

  if(NOT _stf_biosig_include_dir)
    find_path(_stf_biosig_include_dir
      NAMES biosig.h
      HINTS
        /usr/include
        /usr/local/include
    )
  endif()

  if(NOT _stf_biosig_include_dir)
    foreach(_stf_biosig_inc_candidate IN ITEMS /usr/include /usr/local/include)
      if(EXISTS "${_stf_biosig_inc_candidate}/biosig.h")
        set(_stf_biosig_include_dir "${_stf_biosig_inc_candidate}")
        break()
      endif()
    endforeach()
    unset(_stf_biosig_inc_candidate)
  endif()

  if(_stf_biosig_library)
    add_library(stimfit::biosig UNKNOWN IMPORTED)
    set_target_properties(stimfit::biosig PROPERTIES IMPORTED_LOCATION "${_stf_biosig_library}")
    if(_stf_biosig_include_dir)
      target_include_directories(stimfit::biosig INTERFACE "${_stf_biosig_include_dir}")
    endif()
    set(BIOSIG_LIBRARY "${_stf_biosig_library}" PARENT_SCOPE)
    set(BIOSIG_INCLUDE_DIR "${_stf_biosig_include_dir}" PARENT_SCOPE)
  endif()
endfunction()

find_package(Threads REQUIRED)

find_package(PkgConfig QUIET)

if(STF_HDF5_PREFIX)
  list(PREPEND CMAKE_PREFIX_PATH "${STF_HDF5_PREFIX}")
  list(PREPEND CMAKE_LIBRARY_PATH "${STF_HDF5_PREFIX}/lib")
  list(PREPEND CMAKE_INCLUDE_PATH "${STF_HDF5_PREFIX}/include")
endif()

# Prefer CMake config exports first (works reliably with vcpkg on Windows).
set(_stf_hdf5_resolved FALSE)
find_package(hdf5 CONFIG QUIET)
find_package(HDF5 CONFIG QUIET)
if(TARGET hdf5::hdf5-shared OR TARGET hdf5::hdf5 OR TARGET hdf5::hdf5-static OR TARGET HDF5::HDF5)
  add_library(stimfit::hdf5 INTERFACE IMPORTED)
  if(TARGET hdf5::hdf5-shared)
    target_link_libraries(stimfit::hdf5 INTERFACE hdf5::hdf5-shared)
  elseif(TARGET hdf5::hdf5)
    target_link_libraries(stimfit::hdf5 INTERFACE hdf5::hdf5)
  elseif(TARGET hdf5::hdf5-static)
    target_link_libraries(stimfit::hdf5 INTERFACE hdf5::hdf5-static)
  else()
    target_link_libraries(stimfit::hdf5 INTERFACE HDF5::HDF5)
  endif()

  if(TARGET hdf5::hdf5_hl-shared)
    target_link_libraries(stimfit::hdf5 INTERFACE hdf5::hdf5_hl-shared)
  elseif(TARGET hdf5::hdf5_hl)
    target_link_libraries(stimfit::hdf5 INTERFACE hdf5::hdf5_hl)
  elseif(TARGET hdf5::hdf5_hl-static)
    target_link_libraries(stimfit::hdf5 INTERFACE hdf5::hdf5_hl-static)
  elseif(TARGET HDF5::hdf5_hl)
    target_link_libraries(stimfit::hdf5 INTERFACE HDF5::hdf5_hl)
  endif()

  set(_stf_hdf5_resolved TRUE)
endif()

if(NOT _stf_hdf5_resolved)
  # Fall back to module mode and then direct library probing.
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
    set(_stf_hdf5_resolved TRUE)
  endif()
endif()

if(NOT _stf_hdf5_resolved)
  find_library(HDF5_LIBRARY NAMES hdf5 hdf5-shared libhdf5)
  find_library(HDF5_HL_LIBRARY NAMES hdf5_hl hdf5_hl-shared libhdf5_hl)
  if(HDF5_LIBRARY AND HDF5_HL_LIBRARY)
    add_library(stimfit::hdf5 INTERFACE IMPORTED)
    target_link_libraries(stimfit::hdf5 INTERFACE ${HDF5_LIBRARY} ${HDF5_HL_LIBRARY})
    if(STF_HDF5_PREFIX)
      target_include_directories(stimfit::hdf5 INTERFACE ${STF_HDF5_PREFIX}/include)
    endif()
    set(_stf_hdf5_resolved TRUE)
  endif()
endif()

if(NOT _stf_hdf5_resolved)
  message(FATAL_ERROR "HDF5 not found. Install HDF5 development files or set STF_HDF5_PREFIX to your HDF5 prefix.")
endif()
unset(_stf_hdf5_resolved)

check_symbol_exists(strptime "time.h" HAVE_STRPTIME_H)

set(STF_BIOSIG_USE_SUBMODULE OFF)
set(STF_BIOSIG_SELECTED_PROVIDER "DISABLED")

if(STF_WITH_BIOSIG)
  set(_stf_biosig_provider_candidates)
  if(STF_BIOSIG_PROVIDER STREQUAL "AUTO")
    if(WIN32)
      list(APPEND _stf_biosig_provider_candidates PATCHED_SUBMODULE SYSTEM SUBMODULE)
    else()
      list(APPEND _stf_biosig_provider_candidates SUBMODULE SYSTEM)
    endif()
  else()
    list(APPEND _stf_biosig_provider_candidates "${STF_BIOSIG_PROVIDER}")
  endif()

  foreach(_stf_biosig_provider IN LISTS _stf_biosig_provider_candidates)
    if(_stf_biosig_provider STREQUAL "PATCHED_SUBMODULE")
      if(WIN32 AND EXISTS "${CMAKE_SOURCE_DIR}/src/biosig/biosig4c++/CMakeLists.txt")
        stf_configure_windows_patched_biosig()
        if(TARGET stimfit::biosig)
          set(STF_BIOSIG_SELECTED_PROVIDER "PATCHED_SUBMODULE")
          break()
        endif()
      endif()
    elseif(_stf_biosig_provider STREQUAL "SUBMODULE")
      if(DEFINED BIOSIG_LIBRARY AND NOT "${BIOSIG_LIBRARY}" STREQUAL "")
        stf_import_biosig_target()
        if(TARGET stimfit::biosig)
          set(STF_BIOSIG_SELECTED_PROVIDER "SUBMODULE")
          break()
        endif()
      elseif(EXISTS "${CMAKE_SOURCE_DIR}/src/biosig/biosig4c++/CMakeLists.txt")
        set(STF_BIOSIG_USE_SUBMODULE ON)
        set(STF_BIOSIG_SELECTED_PROVIDER "SUBMODULE")
        break()
      endif()
    elseif(_stf_biosig_provider STREQUAL "SYSTEM")
      stf_import_biosig_target()
      if(TARGET stimfit::biosig)
        set(STF_BIOSIG_SELECTED_PROVIDER "SYSTEM")
        break()
      endif()
    endif()
  endforeach()

  if(STF_BIOSIG_SELECTED_PROVIDER STREQUAL "DISABLED")
    if(STF_BIOSIG_PROVIDER STREQUAL "AUTO")
      message(WARNING "STF_WITH_BIOSIG is ON but the requested BIOSIG provider was unavailable; turning BIOSIG support OFF")
      set(STF_WITH_BIOSIG OFF CACHE BOOL "Enable BIOSIG support" FORCE)
    else()
      message(FATAL_ERROR "STF_WITH_BIOSIG is ON but STF_BIOSIG_PROVIDER='${STF_BIOSIG_PROVIDER}' could not be resolved. Install system BIOSIG development files or choose STF_BIOSIG_PROVIDER=PATCHED_SUBMODULE or SUBMODULE.")
    endif()
  endif()

  unset(_stf_biosig_provider)
  unset(_stf_biosig_provider_candidates)
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

find_package(LAPACK QUIET)
if(LAPACK_FOUND)
  add_library(stimfit::lapack INTERFACE IMPORTED)
  target_link_libraries(stimfit::lapack INTERFACE ${LAPACK_LIBRARIES})
  target_compile_definitions(stimfit::lapack INTERFACE HAVE_LAPACK)
  if(LAPACK_LIBRARIES MATCHES "openblas")
    target_compile_definitions(stimfit::lapack INTERFACE WITH_OPENBLAS)
  endif()
else()
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    add_library(stimfit::lapack INTERFACE IMPORTED)
    target_link_libraries(stimfit::lapack INTERFACE ${BLAS_LIBRARIES})
    target_compile_definitions(stimfit::lapack INTERFACE HAVE_LAPACK)
    if(BLAS_LIBRARIES MATCHES "openblas")
      target_compile_definitions(stimfit::lapack INTERFACE WITH_OPENBLAS)
    endif()
  else()
    find_library(OPENBLAS_LIBRARY NAMES openblas)
    if(OPENBLAS_LIBRARY)
      add_library(stimfit::lapack UNKNOWN IMPORTED)
      set_target_properties(stimfit::lapack PROPERTIES IMPORTED_LOCATION "${OPENBLAS_LIBRARY}")
      target_compile_definitions(stimfit::lapack INTERFACE WITH_OPENBLAS HAVE_LAPACK)
    else()
      find_library(LAPACK_LIBRARY NAMES lapack lapack3 lapack-3 REQUIRED)
      add_library(stimfit::lapack UNKNOWN IMPORTED)
      set_target_properties(stimfit::lapack PROPERTIES IMPORTED_LOCATION "${LAPACK_LIBRARY}")
      target_compile_definitions(stimfit::lapack INTERFACE HAVE_LAPACK)
    endif()
  endif()
endif()

# When linking against an embedded-Python-capable build (wxPython), the C++
# wxWidgets runtime MUST match the version that wxPython was built against.
# wxPython Phoenix 4.x ships bundled wx 3.2.x DLLs; if stimfit links against
# a different wx (e.g. vcpkg wx 3.3.x) the two runtimes coexist in the same
# process with incompatible ABIs, causing wxPyConvertWrappedPtr to return NULL
# ("Pointer is zero" at startup).  Set STF_WX_PREFER_MODULE_FIND=ON together
# with wxWidgets_ROOT_DIR and wxWidgets_LIB_DIR pointing to the wxPython wx
# tree to compile and link against that exact wx build.
#
# NOTE: vcpkg installs a vcpkg-cmake-wrapper.cmake for wxWidgets that forcibly
# resets wxWidgets_ROOT_DIR / wxWidgets_LIB_DIR with CACHE INTERNAL before
# FindwxWidgets runs, defeating preset values.  When STF_WX_PREFER_MODULE_FIND
# is ON we therefore bypass find_package(wxWidgets) entirely and build the
# interface target directly from the caller-supplied paths.
option(STF_WX_PREFER_MODULE_FIND
  "Bypass find_package(wxWidgets) and build the interface directly from wxWidgets_ROOT_DIR/wxWidgets_LIB_DIR. Required when the C++ wx must match the wxPython wheel's bundled wx (e.g. Phoenix 4.x ships wx 3.2)."
  OFF)

if(NOT STF_BUILD_MODULE)
  add_library(stimfit::wx INTERFACE IMPORTED)
  set(_stf_wx_resolved FALSE)

  if(STF_WX_PREFER_MODULE_FIND)
    # Bypass find_package entirely to prevent vcpkg's cmake wrapper from
    # overriding wxWidgets_ROOT_DIR / wxWidgets_LIB_DIR with CACHE INTERNAL.
    if(NOT wxWidgets_ROOT_DIR)
      message(FATAL_ERROR
        "STF_WX_PREFER_MODULE_FIND=ON requires wxWidgets_ROOT_DIR "
        "(e.g. C:/path/to/Phoenix/ext/wxWidgets)")
    endif()
    if(NOT wxWidgets_LIB_DIR)
      message(FATAL_ERROR
        "STF_WX_PREFER_MODULE_FIND=ON requires wxWidgets_LIB_DIR "
        "(e.g. C:/path/to/Phoenix/ext/wxWidgets/lib/vc140_x64_dll)")
    endif()

    # vcpkg installs all package headers into a single flat include/ directory.
    # Other vcpkg packages (HDF5, FFTW3, ...) add that directory to the include
    # path as transitive interface includes from their imported targets.  Because
    # transitive interface includes are appended after the target's own include
    # directories, we use include_directories(BEFORE ...) at directory scope to
    # ensure Phoenix wx 3.2.x headers appear before vcpkg's wx 3.3.x headers in
    # every target's compile command.
    include_directories(BEFORE
      "${wxWidgets_ROOT_DIR}/include"
      "${wxWidgets_LIB_DIR}/mswu"
    )

    # Include dirs: root/include + lib-dir/mswu (MSVC unicode DLL setup.h)
    target_include_directories(stimfit::wx INTERFACE
      "${wxWidgets_ROOT_DIR}/include"
      "${wxWidgets_LIB_DIR}/mswu"
    )

    # Locate import libs for the required components by glob so that the
    # version number embedded in the filename (e.g. 32u vs 33u) is agnostic.
    # Note: 'base' and 'net' live in the wxbase library; the GUI components
    # live in the wxmsw library.
    foreach(_stf_wx_comp IN ITEMS base core adv aui net)
      if(_stf_wx_comp STREQUAL "base")
        file(GLOB _stf_wx_lib LIST_DIRECTORIES false
          "${wxWidgets_LIB_DIR}/wxbase3*u.lib")
      elseif(_stf_wx_comp STREQUAL "net")
        file(GLOB _stf_wx_lib LIST_DIRECTORIES false
          "${wxWidgets_LIB_DIR}/wxbase3*u_net.lib")
      else()
        file(GLOB _stf_wx_lib LIST_DIRECTORIES false
          "${wxWidgets_LIB_DIR}/wxmsw3*u_${_stf_wx_comp}.lib")
      endif()
      if(NOT _stf_wx_lib)
        message(FATAL_ERROR
          "Could not find wx import lib for component '${_stf_wx_comp}' in "
          "${wxWidgets_LIB_DIR}. Verify wxWidgets_LIB_DIR is correct.")
      endif()
      list(APPEND _stf_wx_libs ${_stf_wx_lib})
      unset(_stf_wx_lib)
    endforeach()
    target_link_libraries(stimfit::wx INTERFACE ${_stf_wx_libs})
    unset(_stf_wx_libs)

    # Standard definitions for a unicode wxWidgets DLL build on Windows
    target_compile_definitions(stimfit::wx INTERFACE WXUSINGDLL _UNICODE UNICODE)
    set(_stf_wx_resolved TRUE)
  endif()

  # Prefer config packages (vcpkg) before legacy FindwxWidgets module.
  if(NOT _stf_wx_resolved)
    find_package(wxWidgets CONFIG QUIET)
    if(TARGET wx::base AND TARGET wx::core AND TARGET wx::adv AND TARGET wx::aui AND TARGET wx::net)
      target_link_libraries(stimfit::wx INTERFACE wx::base wx::core wx::adv wx::aui wx::net)
      set(_stf_wx_resolved TRUE)
    endif()
  endif()

  if(NOT _stf_wx_resolved)
    if(APPLE AND (
        NOT DEFINED wxWidgets_CONFIG_EXECUTABLE
        OR "${wxWidgets_CONFIG_EXECUTABLE}" STREQUAL ""
        OR "${wxWidgets_CONFIG_EXECUTABLE}" MATCHES "-NOTFOUND$"
      ))
      stf_find_macos_wx_config(_stf_wx_config_candidate)
      if(_stf_wx_config_candidate)
        set(wxWidgets_CONFIG_EXECUTABLE "${_stf_wx_config_candidate}" CACHE FILEPATH "Path to wx-config executable" FORCE)
      endif()
      unset(_stf_wx_config_candidate)
    endif()

    find_package(wxWidgets MODULE REQUIRED COMPONENTS base core adv aui net)
    target_include_directories(stimfit::wx INTERFACE ${wxWidgets_INCLUDE_DIRS})
    target_link_libraries(stimfit::wx INTERFACE ${wxWidgets_LIBRARIES})
    target_compile_definitions(stimfit::wx INTERFACE ${wxWidgets_DEFINITIONS})
    set(_stf_wx_resolved TRUE)
  endif()

  unset(_stf_wx_resolved)
else()
  add_library(stimfit::wx INTERFACE IMPORTED)
endif()

if(STF_ENABLE_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  add_library(stimfit::python INTERFACE IMPORTED)
  target_include_directories(stimfit::python INTERFACE ${Python3_INCLUDE_DIRS})
  target_link_libraries(stimfit::python INTERFACE ${Python3_LIBRARIES})

  stf_get_python_bootstrap(_stf_python_bootstrap)

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
endif()
