include_guard(GLOBAL)

include(CheckIncludeFile)

find_package(Threads REQUIRED)

find_package(PkgConfig QUIET)

if(STF_HDF5_PREFIX)
  list(PREPEND CMAKE_PREFIX_PATH "${STF_HDF5_PREFIX}")
  list(PREPEND CMAKE_LIBRARY_PATH "${STF_HDF5_PREFIX}/lib")
  list(PREPEND CMAKE_INCLUDE_PATH "${STF_HDF5_PREFIX}/include")
endif()

find_package(HDF5 COMPONENTS C HL QUIET)
if(HDF5_FOUND)
  add_library(stimfit::hdf5 INTERFACE IMPORTED)
  target_include_directories(stimfit::hdf5 INTERFACE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(stimfit::hdf5 INTERFACE ${HDF5_LIBRARIES})
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
  find_library(BIOSIG_LIBRARY NAMES biosig)
  if(BIOSIG_LIBRARY)
    add_library(stimfit::biosig UNKNOWN IMPORTED)
    set_target_properties(stimfit::biosig PROPERTIES IMPORTED_LOCATION "${BIOSIG_LIBRARY}")
  else()
    message(WARNING "STF_WITH_BIOSIG is ON but external libbiosig was not found; turning it OFF")
    set(STF_WITH_BIOSIG OFF CACHE BOOL "Use external libbiosig if available" FORCE)
  endif()
endif()

find_library(FFTW3_LIBRARY NAMES fftw3 REQUIRED)
add_library(stimfit::fftw3 UNKNOWN IMPORTED)
set_target_properties(stimfit::fftw3 PROPERTIES IMPORTED_LOCATION "${FFTW3_LIBRARY}")

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

