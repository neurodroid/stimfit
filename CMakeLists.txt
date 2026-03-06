cmake_minimum_required(VERSION 3.20)

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

configure_file(cmake/stfconf.h.in stfconf.h @ONLY)

stf_print_configuration_summary()

add_subdirectory(src)

if(NOT STF_BUILD_MODULE)
  add_executable(stimfit src/stimfit/gui/main.cpp)
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
  install(TARGETS stimfit RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

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
endif()

