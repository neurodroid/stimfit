cmake_minimum_required(VERSION 3.21)

project(
  stimfit
  VERSION 0.16.11
  DESCRIPTION "Stimfit electrophysiology analysis suite"
  LANGUAGES C CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(CTest)

include(StimfitOptions)
include(StimfitDependencies)
include(StimfitToolchain)
include(StimfitMigration)

if(WIN32)
  set(STF_RUNTIME_DEPENDENCY_DIRS
    "${CMAKE_BINARY_DIR}/src/stimfit/$<CONFIG>"
    "${CMAKE_BINARY_DIR}/src/libstfio/$<CONFIG>"
    "${CMAKE_BINARY_DIR}/src/libstfnum/$<CONFIG>"
    "$ENV{SystemRoot}/System32"
  )
  if(DEFINED BIOSIG_LIBRARY AND EXISTS "${BIOSIG_LIBRARY}")
    get_filename_component(_stf_biosig_lib_dir "${BIOSIG_LIBRARY}" DIRECTORY)
    list(APPEND STF_RUNTIME_DEPENDENCY_DIRS "${_stf_biosig_lib_dir}")
    unset(_stf_biosig_lib_dir)
  endif()
  if(DEFINED wxWidgets_LIB_DIR AND NOT "${wxWidgets_LIB_DIR}" STREQUAL "")
    list(APPEND STF_RUNTIME_DEPENDENCY_DIRS "${wxWidgets_LIB_DIR}")
  endif()
  if(MSVC)
    file(GLOB STF_MSVC_REDIST_DIRS
      "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Redist/MSVC/*/x64/Microsoft.VC143.CRT"
    )
    list(APPEND STF_RUNTIME_DEPENDENCY_DIRS ${STF_MSVC_REDIST_DIRS})
  endif()

  set(STF_INSTALL_RUNTIME_DEPENDENCY_ARGS
    RUNTIME_DEPENDENCIES
      DIRECTORIES
        ${STF_RUNTIME_DEPENDENCY_DIRS}
      # Exclude well-known system runtime DLL families.
      PRE_EXCLUDE_REGEXES
        "api-ms-.*"
        "ext-ms-.*"
        ".*[/\\]Windows[/\\].*"
      POST_EXCLUDE_REGEXES
        ".*[/\\]system32[/\\].*"
        ".*[/\\]SysWOW64[/\\].*"
  )
else()
  set(STF_INSTALL_RUNTIME_DEPENDENCY_ARGS)
endif()

configure_file(cmake/stfconf.h.in stfconf.h @ONLY)

stf_print_configuration_summary()

add_subdirectory(src)

if(NOT STF_BUILD_MODULE)
  if(WIN32)
    add_executable(stimfit WIN32 src/stimfit/gui/main.cpp)
  else()
    add_executable(stimfit src/stimfit/gui/main.cpp)
  endif()
  target_link_libraries(
    stimfit
    PRIVATE
      stimfit_config
      stimfit_core
      stfio
      stfnum
      stimfit::wx
      stimfit::fftw3
      stimfit::lapack
  )
  if(TARGET biosiglite)
    target_link_libraries(stimfit PRIVATE biosiglite)
  endif()
  install(
    TARGETS stimfit
    ${STF_INSTALL_RUNTIME_DEPENDENCY_ARGS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

  if(WIN32 AND DEFINED wxWidgets_LIB_DIR AND EXISTS "${wxWidgets_LIB_DIR}")
    install(
      DIRECTORY "${wxWidgets_LIB_DIR}/"
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      FILES_MATCHING
        PATTERN "wx*.dll"
    )
  endif()

  if(WIN32 AND DEFINED STF_HDF5_PREFIX)
    if(EXISTS "${STF_HDF5_PREFIX}/bin")
      install(
        DIRECTORY "${STF_HDF5_PREFIX}/bin/"
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        FILES_MATCHING
          PATTERN "hdf*.dll"
      )
    endif()
    if(EXISTS "${STF_HDF5_PREFIX}/lib")
      install(
        DIRECTORY "${STF_HDF5_PREFIX}/lib/"
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        FILES_MATCHING
          PATTERN "hdf*.dll"
      )
    endif()
  endif()

  if(WIN32 AND DEFINED FFTW3_LIBRARY AND EXISTS "${FFTW3_LIBRARY}")
    get_filename_component(_fftw3_lib_dir "${FFTW3_LIBRARY}" DIRECTORY)
    install(
      DIRECTORY "${_fftw3_lib_dir}/"
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      FILES_MATCHING
        PATTERN "libfftw3*.dll"
        PATTERN "fftw3*.dll"
    )
    unset(_fftw3_lib_dir)
  endif()

  if(WIN32 AND DEFINED BIOSIG_LIBRARY AND EXISTS "${BIOSIG_LIBRARY}")
    get_filename_component(_biosig_lib_dir "${BIOSIG_LIBRARY}" DIRECTORY)
    install(
      DIRECTORY "${_biosig_lib_dir}/"
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      FILES_MATCHING
        PATTERN "biosig*.dll"
    )
    unset(_biosig_lib_dir)
  endif()

  if(WIN32)
    if(TARGET biosigshared)
      install(FILES $<TARGET_FILE:biosigshared> DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
    if(TARGET biosig2shared)
      install(FILES $<TARGET_FILE:biosig2shared> DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
  endif()

  if(WIN32 AND DEFINED STF_MSVC_REDIST_DIRS)
    foreach(_stf_msvc_redist_dir IN LISTS STF_MSVC_REDIST_DIRS)
      if(EXISTS "${_stf_msvc_redist_dir}")
        install(
          DIRECTORY "${_stf_msvc_redist_dir}/"
          DESTINATION ${CMAKE_INSTALL_BINDIR}
          FILES_MATCHING
            PATTERN "msvcp*.dll"
            PATTERN "vcruntime*.dll"
            PATTERN "concrt*.dll"
        )
      endif()
    endforeach()
    unset(_stf_msvc_redist_dir)
  endif()

  if(WIN32)
    set(_stf_lapack_runtime_dirs)
    list(APPEND _stf_lapack_runtime_dirs
      "${CMAKE_BINARY_DIR}/src/libstfnum/$<CONFIG>"
      "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin"
      "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/bin"
    )
    foreach(_stf_lapack_dir IN LISTS _stf_lapack_runtime_dirs)
      if(EXISTS "${_stf_lapack_dir}")
        install(
          DIRECTORY "${_stf_lapack_dir}/"
          DESTINATION ${CMAKE_INSTALL_BINDIR}
          FILES_MATCHING
            PATTERN "liblapack*.dll"
            PATTERN "openblas*.dll"
            PATTERN "libgcc_s_seh-1.dll"
            PATTERN "libgfortran-5.dll"
            PATTERN "libquadmath-0.dll"
            PATTERN "libwinpthread-1.dll"
        )
      endif()
    endforeach()
    unset(_stf_lapack_runtime_dirs)
    unset(_stf_lapack_dir)
  endif()

  if(STF_BUILD_TESTS)
    add_executable(
      stimfittest
      src/test/section.cpp
      src/test/channel.cpp
      src/test/recording.cpp
      src/test/fit.cpp
      src/test/measure.cpp
      src/test/gtest/src/gtest-all.cc
      src/test/gtest/src/gtest_main.cc
    )
    target_include_directories(
      stimfittest
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/test/gtest
        ${CMAKE_CURRENT_SOURCE_DIR}/src/test/gtest/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(stimfittest PRIVATE STF_TEST)
    target_link_libraries(
      stimfittest
      PRIVATE
        stimfit_config
        stimfit_core
        stfio
        stfnum
        stimfit::wx
        stimfit::fftw3
        stimfit::lapack
        Threads::Threads
    )
    if(TARGET biosiglite)
      target_link_libraries(stimfittest PRIVATE biosiglite)
    endif()
    add_test(NAME stimfittest COMMAND stimfittest)
    install(
      TARGETS stimfittest
      ${STF_INSTALL_RUNTIME_DEPENDENCY_ARGS}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
  endif()
endif()

