include_guard(GLOBAL)

include(CheckIncludeFile)

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

check_include_file(strptime.h HAVE_STRPTIME_H)

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

find_package(BLAS QUIET)
find_package(LAPACK QUIET)
if(LAPACK_FOUND)
  add_library(stimfit::lapack INTERFACE IMPORTED)
  target_link_libraries(stimfit::lapack INTERFACE ${LAPACK_LIBRARIES})
elseif(BLAS_FOUND)
  add_library(stimfit::lapack INTERFACE IMPORTED)
  target_link_libraries(stimfit::lapack INTERFACE ${BLAS_LIBRARIES})
else()
  find_library(OPENBLAS_LIBRARY NAMES openblas)
  if(OPENBLAS_LIBRARY)
    add_library(stimfit::lapack UNKNOWN IMPORTED)
    set_target_properties(stimfit::lapack PROPERTIES IMPORTED_LOCATION "${OPENBLAS_LIBRARY}")
  else()
    find_library(LAPACK_LIBRARY NAMES lapack lapack3 lapack-3 REQUIRED)
    add_library(stimfit::lapack UNKNOWN IMPORTED)
    set_target_properties(stimfit::lapack PROPERTIES IMPORTED_LOCATION "${LAPACK_LIBRARY}")
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

  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['platlib'])"
      OUTPUT_VARIABLE STF_PYTHON_PLATLIB
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
      OUTPUT_VARIABLE STF_NUMPY_INCLUDE
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
  endif()

  find_package(SWIG QUIET)
endif()

