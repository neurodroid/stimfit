cmake_minimum_required(VERSION 3.21)

set(STF_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
if(NOT EXISTS "${STF_VERSION_FILE}")
  message(FATAL_ERROR "Missing VERSION file at ${STF_VERSION_FILE}")
endif()

file(READ "${STF_VERSION_FILE}" STF_VERSION_RAW)
string(STRIP "${STF_VERSION_RAW}" STF_VERSION)
if(STF_VERSION STREQUAL "")
  message(FATAL_ERROR "VERSION file is empty")
endif()

project(
  stimfit
  VERSION ${STF_VERSION}
  DESCRIPTION "Stimfit electrophysiology analysis suite"
  LANGUAGES C CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(CTest)

include(StimfitOptions)
include(StimfitDependencies)
include(StimfitToolchain)
include(StimfitMacOS)
include(StimfitMigration)

set(STF_COMPONENT_CORE "core")
set(STF_COMPONENT_PYTHON "python")
set(STF_COMPONENT_TESTS "tests")

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

  if(STF_ENABLE_PYTHON)
    if(DEFINED Python3_RUNTIME_LIBRARY_DIRS)
      list(APPEND STF_RUNTIME_DEPENDENCY_DIRS ${Python3_RUNTIME_LIBRARY_DIRS})
    endif()
    if(DEFINED Python3_LIBRARY_DIRS)
      list(APPEND STF_RUNTIME_DEPENDENCY_DIRS ${Python3_LIBRARY_DIRS})
    endif()
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

if(UNIX AND NOT APPLE AND NOT STF_BUILD_MODULE)
  if(STF_BUILD_DEBIAN)
    set(STF_LINUX_INSTALL_RPATH "/usr/lib/stimfit")
  else()
    set(STF_LINUX_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/stimfit")
  endif()

  set(CMAKE_INSTALL_RPATH "${STF_LINUX_INSTALL_RPATH}")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
endif()

set(STF_CONFIG_HEADER_TEMPLATE "${CMAKE_SOURCE_DIR}/cmake/stfconf.h.in")
if(NOT EXISTS "${STF_CONFIG_HEADER_TEMPLATE}")
  set(STF_CONFIG_HEADER_TEMPLATE "${CMAKE_SOURCE_DIR}/stfconf.h.in")
endif()
if(NOT EXISTS "${STF_CONFIG_HEADER_TEMPLATE}")
  message(FATAL_ERROR "Could not find stfconf.h.in in either ${CMAKE_SOURCE_DIR}/cmake or ${CMAKE_SOURCE_DIR}")
endif()
configure_file("${STF_CONFIG_HEADER_TEMPLATE}" stfconf.h)
unset(STF_CONFIG_HEADER_TEMPLATE)
if(APPLE AND STF_MACOS_APP_BUNDLE)
  set(PACKAGE_VERSION "${PROJECT_VERSION}")
  configure_file(
    ${CMAKE_SOURCE_DIR}/dist/macosx/stimfit.plist.in
    ${CMAKE_CURRENT_BINARY_DIR}/stimfit.Info.plist
    @ONLY
  )
endif()

stf_print_configuration_summary()

add_subdirectory(src)

if(NOT STF_BUILD_MODULE)
  if(WIN32)
    add_executable(stimfit WIN32 src/stimfit/gui/main.cpp)
  elseif(APPLE AND STF_MACOS_APP_BUNDLE)
    add_executable(stimfit MACOSX_BUNDLE src/stimfit/gui/main.cpp)
  else()
    add_executable(stimfit src/stimfit/gui/main.cpp)
  endif()

  if(APPLE AND STF_MACOS_APP_BUNDLE)
    set_target_properties(stimfit PROPERTIES
      MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_BINARY_DIR}/stimfit.Info.plist"
      MACOSX_BUNDLE_ICON_FILE "stimfit.icns"
    )
    target_sources(stimfit PRIVATE ${CMAKE_SOURCE_DIR}/dist/macosx/stimfit.icns)
    set_source_files_properties(${CMAKE_SOURCE_DIR}/dist/macosx/stimfit.icns PROPERTIES
      MACOSX_PACKAGE_LOCATION "Resources"
    )
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
  stf_apply_macos_runtime_policy(stimfit)
  if(APPLE AND STF_MACOS_APP_BUNDLE)
    install(
      TARGETS stimfit
      BUNDLE DESTINATION .
      COMPONENT ${STF_COMPONENT_CORE}
    )
  else()
    install(
      TARGETS stimfit
      ${STF_INSTALL_RUNTIME_DEPENDENCY_ARGS}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT ${STF_COMPONENT_CORE}
    )
  endif()

  if(UNIX AND NOT APPLE)
    if(EXISTS "${CMAKE_SOURCE_DIR}/debian/stimfit.desktop")
      install(
        FILES "${CMAKE_SOURCE_DIR}/debian/stimfit.desktop"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/applications"
        COMPONENT ${STF_COMPONENT_CORE}
      )
    endif()

    foreach(_stf_icon_size 16 32 48 128 256 512)
      set(_stf_icon_src "${CMAKE_SOURCE_DIR}/src/stimfit/res/stimfit_${_stf_icon_size}.png")
      if(EXISTS "${_stf_icon_src}")
        install(
          FILES "${_stf_icon_src}"
          DESTINATION "${CMAKE_INSTALL_DATADIR}/icons/hicolor/${_stf_icon_size}x${_stf_icon_size}/apps"
          RENAME "stimfit.png"
          COMPONENT ${STF_COMPONENT_CORE}
        )
      endif()
      unset(_stf_icon_src)
    endforeach()
    unset(_stf_icon_size)
  endif()

  if(WIN32 AND DEFINED wxWidgets_LIB_DIR AND EXISTS "${wxWidgets_LIB_DIR}")
    install(
      DIRECTORY "${wxWidgets_LIB_DIR}/"
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT ${STF_COMPONENT_CORE}
      FILES_MATCHING
        PATTERN "wx*.dll"
    )
  endif()

  if(WIN32 AND DEFINED STF_HDF5_PREFIX)
    if(EXISTS "${STF_HDF5_PREFIX}/bin")
      install(
        DIRECTORY "${STF_HDF5_PREFIX}/bin/"
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${STF_COMPONENT_CORE}
        FILES_MATCHING
          PATTERN "hdf*.dll"
      )
    endif()
    if(EXISTS "${STF_HDF5_PREFIX}/lib")
      install(
        DIRECTORY "${STF_HDF5_PREFIX}/lib/"
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${STF_COMPONENT_CORE}
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
      COMPONENT ${STF_COMPONENT_CORE}
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
      COMPONENT ${STF_COMPONENT_CORE}
      FILES_MATCHING
        PATTERN "biosig*.dll"
    )
    unset(_biosig_lib_dir)
  endif()

  if(WIN32)
    if(TARGET biosigshared)
      install(FILES $<TARGET_FILE:biosigshared> DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${STF_COMPONENT_CORE})
    endif()
    if(TARGET biosig2shared)
      install(FILES $<TARGET_FILE:biosig2shared> DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${STF_COMPONENT_CORE})
    endif()
  endif()

  if(WIN32 AND DEFINED STF_MSVC_REDIST_DIRS)
    foreach(_stf_msvc_redist_dir IN LISTS STF_MSVC_REDIST_DIRS)
      if(EXISTS "${_stf_msvc_redist_dir}")
        install(
          DIRECTORY "${_stf_msvc_redist_dir}/"
          DESTINATION ${CMAKE_INSTALL_BINDIR}
          COMPONENT ${STF_COMPONENT_CORE}
          FILES_MATCHING
            PATTERN "msvcp*.dll"
            PATTERN "vcruntime*.dll"
            PATTERN "concrt*.dll"
        )
      endif()
    endforeach()
    unset(_stf_msvc_redist_dir)
  endif()

  if(WIN32 AND STF_ENABLE_PYTHON)
    if(STF_WINDOWS_COPY_PYTHON_RUNTIME)
      if(DEFINED STF_PYTHON_RUNTIME_DLL AND EXISTS "${STF_PYTHON_RUNTIME_DLL}")
        install(FILES "${STF_PYTHON_RUNTIME_DLL}" DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${STF_COMPONENT_PYTHON})
      else()
        message(WARNING "STF_WINDOWS_COPY_PYTHON_RUNTIME is ON but STF_PYTHON_RUNTIME_DLL was not found.")
      endif()
    endif()

    if(STF_WINDOWS_COPY_PYTHON_STDLIB)
      if(DEFINED STF_PYTHON_STDLIB AND EXISTS "${STF_PYTHON_STDLIB}")
        install(
          DIRECTORY "${STF_PYTHON_STDLIB}/"
          DESTINATION "${CMAKE_INSTALL_BINDIR}/Lib"
          COMPONENT ${STF_COMPONENT_PYTHON}
          PATTERN "site-packages" EXCLUDE
          PATTERN "dist-packages" EXCLUDE
          PATTERN "__pycache__" EXCLUDE
        )
      else()
        message(WARNING "STF_WINDOWS_COPY_PYTHON_STDLIB is ON but STF_PYTHON_STDLIB was not found.")
      endif()
    endif()

    if(STF_WINDOWS_COPY_PYTHON_DLLS)
      if(DEFINED STF_PYTHON_DLLS_DIR AND EXISTS "${STF_PYTHON_DLLS_DIR}")
        install(
          DIRECTORY "${STF_PYTHON_DLLS_DIR}/"
          DESTINATION "${CMAKE_INSTALL_BINDIR}/DLLs"
          COMPONENT ${STF_COMPONENT_PYTHON}
          FILES_MATCHING
            PATTERN "*.pyd"
            PATTERN "*.dll"
        )
      endif()
    endif()

    if(STF_WINDOWS_COPY_PYTHON_SITE_PACKAGES)
      set(_stf_site_packages_root "${CMAKE_INSTALL_BINDIR}/stf-site-packages")
      foreach(_stf_pkg IN LISTS STF_WINDOWS_PYTHON_SITE_PACKAGES)
        set(_stf_pkg_copied OFF)

        if(DEFINED STF_PYTHON_PURELIB AND EXISTS "${STF_PYTHON_PURELIB}/${_stf_pkg}")
          install(DIRECTORY "${STF_PYTHON_PURELIB}/${_stf_pkg}" DESTINATION "${_stf_site_packages_root}" COMPONENT ${STF_COMPONENT_PYTHON})
          set(_stf_pkg_copied ON)
        endif()

        if(DEFINED STF_PYTHON_PURELIB AND EXISTS "${STF_PYTHON_PURELIB}/${_stf_pkg}.libs")
          install(DIRECTORY "${STF_PYTHON_PURELIB}/${_stf_pkg}.libs" DESTINATION "${_stf_site_packages_root}" COMPONENT ${STF_COMPONENT_PYTHON})
          set(_stf_pkg_copied ON)
        endif()

        if(DEFINED STF_PYTHON_PURELIB AND EXISTS "${STF_PYTHON_PURELIB}/${_stf_pkg}.py")
          install(FILES "${STF_PYTHON_PURELIB}/${_stf_pkg}.py" DESTINATION "${_stf_site_packages_root}" COMPONENT ${STF_COMPONENT_PYTHON})
          set(_stf_pkg_copied ON)
        endif()

        if(DEFINED STF_PYTHON_PURELIB)
          file(GLOB _stf_pkg_purelib_pyds "${STF_PYTHON_PURELIB}/${_stf_pkg}*.pyd")
          if(_stf_pkg_purelib_pyds)
            install(FILES ${_stf_pkg_purelib_pyds} DESTINATION "${_stf_site_packages_root}" COMPONENT ${STF_COMPONENT_PYTHON})
            set(_stf_pkg_copied ON)
          endif()
          unset(_stf_pkg_purelib_pyds)
        endif()

        if(DEFINED STF_PYTHON_PLATLIB AND EXISTS "${STF_PYTHON_PLATLIB}/${_stf_pkg}")
          install(DIRECTORY "${STF_PYTHON_PLATLIB}/${_stf_pkg}" DESTINATION "${_stf_site_packages_root}" COMPONENT ${STF_COMPONENT_PYTHON})
          set(_stf_pkg_copied ON)
        endif()

        if(DEFINED STF_PYTHON_PLATLIB AND EXISTS "${STF_PYTHON_PLATLIB}/${_stf_pkg}.libs")
          install(DIRECTORY "${STF_PYTHON_PLATLIB}/${_stf_pkg}.libs" DESTINATION "${_stf_site_packages_root}" COMPONENT ${STF_COMPONENT_PYTHON})
          set(_stf_pkg_copied ON)
        endif()

        if(DEFINED STF_PYTHON_PLATLIB AND EXISTS "${STF_PYTHON_PLATLIB}/${_stf_pkg}.pyd")
          install(FILES "${STF_PYTHON_PLATLIB}/${_stf_pkg}.pyd" DESTINATION "${_stf_site_packages_root}" COMPONENT ${STF_COMPONENT_PYTHON})
          set(_stf_pkg_copied ON)
        endif()

        if(DEFINED STF_PYTHON_PLATLIB)
          file(GLOB _stf_pkg_platlib_pyds "${STF_PYTHON_PLATLIB}/${_stf_pkg}*.pyd")
          if(_stf_pkg_platlib_pyds)
            install(FILES ${_stf_pkg_platlib_pyds} DESTINATION "${_stf_site_packages_root}" COMPONENT ${STF_COMPONENT_PYTHON})
            set(_stf_pkg_copied ON)
          endif()
          unset(_stf_pkg_platlib_pyds)
        endif()

        if(NOT _stf_pkg_copied AND DEFINED STF_WINDOWS_PYTHON_EXTRA_PATHS)
          foreach(_stf_extra_py IN LISTS STF_WINDOWS_PYTHON_EXTRA_PATHS)
            if(EXISTS "${_stf_extra_py}/${_stf_pkg}")
              install(DIRECTORY "${_stf_extra_py}/${_stf_pkg}" DESTINATION "${_stf_site_packages_root}" COMPONENT ${STF_COMPONENT_PYTHON})
              set(_stf_pkg_copied ON)
              break()
            endif()
            if(EXISTS "${_stf_extra_py}/${_stf_pkg}.libs")
              install(DIRECTORY "${_stf_extra_py}/${_stf_pkg}.libs" DESTINATION "${_stf_site_packages_root}" COMPONENT ${STF_COMPONENT_PYTHON})
              set(_stf_pkg_copied ON)
            endif()
            if(EXISTS "${_stf_extra_py}/${_stf_pkg}.py")
              install(FILES "${_stf_extra_py}/${_stf_pkg}.py" DESTINATION "${_stf_site_packages_root}" COMPONENT ${STF_COMPONENT_PYTHON})
              set(_stf_pkg_copied ON)
              break()
            endif()
            file(GLOB _stf_pkg_extra_pyds "${_stf_extra_py}/${_stf_pkg}*.pyd")
            if(_stf_pkg_extra_pyds)
              install(FILES ${_stf_pkg_extra_pyds} DESTINATION "${_stf_site_packages_root}" COMPONENT ${STF_COMPONENT_PYTHON})
              set(_stf_pkg_copied ON)
            endif()
            unset(_stf_pkg_extra_pyds)
          endforeach()
        endif()

        if(NOT _stf_pkg_copied)
          message(WARNING "Requested Python package '${_stf_pkg}' was not found in purelib/platlib and was not copied.")
        endif()
      endforeach()

      if(DEFINED STF_PYTHON_PURELIB AND EXISTS "${STF_PYTHON_PURELIB}/site.py")
        install(FILES "${STF_PYTHON_PURELIB}/site.py" DESTINATION "${_stf_site_packages_root}" COMPONENT ${STF_COMPONENT_PYTHON})
      endif()

      unset(_stf_site_packages_root)
      unset(_stf_pkg)
      unset(_stf_pkg_copied)
      unset(_stf_extra_py)
    endif()
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
          COMPONENT ${STF_COMPONENT_CORE}
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
    add_library(
      stimfit_gtest_main
      STATIC
      src/test/gtest/src/gtest-all.cc
      src/test/gtest/src/gtest_main.cc
    )
    target_include_directories(
      stimfit_gtest_main
      PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/test/gtest
        ${CMAKE_CURRENT_SOURCE_DIR}/src/test/gtest/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(
      stimfit_gtest_main
      PUBLIC
        Threads::Threads
        stimfit_config
    )
    stf_apply_macos_runtime_policy(stimfit_gtest_main)

    add_executable(
      stimfit_containers_test
      src/test/section.cpp
      src/test/channel.cpp
      src/test/recording.cpp
    )
    target_link_libraries(
      stimfit_containers_test
      PRIVATE
        stimfit_config
        stimfit_gtest_main
        stfio_containers
    )
    stf_apply_macos_runtime_policy(stimfit_containers_test)
    add_test(NAME stimfit_containers_test COMMAND stimfit_containers_test)
    install(
      TARGETS stimfit_containers_test
      ${STF_INSTALL_RUNTIME_DEPENDENCY_ARGS}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT ${STF_COMPONENT_TESTS}
    )

    if(STF_BUILD_NUMERIC_TESTS)
      add_executable(
        stimfit_numeric_test
        src/test/fit.cpp
        src/test/measure.cpp
      )
      target_link_libraries(
        stimfit_numeric_test
        PRIVATE
          stimfit_config
          stimfit_gtest_main
          stfnum
      )
      stf_apply_macos_runtime_policy(stimfit_numeric_test)
      add_test(NAME stimfit_numeric_test COMMAND stimfit_numeric_test)
      install(
        TARGETS stimfit_numeric_test
        ${STF_INSTALL_RUNTIME_DEPENDENCY_ARGS}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${STF_COMPONENT_TESTS}
      )
    endif()
  endif()
endif()

set(CPACK_PACKAGE_NAME "Stimfit")
set(CPACK_PACKAGE_VENDOR "Stimfit contributors")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Stimfit electrophysiology analysis suite")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_CONTACT "https://github.com/neurodroid/stimfit")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/gpl-2.0.txt")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "Stimfit")
set(CPACK_MONOLITHIC_INSTALL OFF)

if(WIN32)
  set(CPACK_GENERATOR "INNOSETUP;ZIP")
  if(NOT DEFINED CPACK_INNOSETUP_EXECUTABLE)
    set(_stf_iscc_user_raw "$ENV{LOCALAPPDATA}/Programs/Inno Setup 6/ISCC.exe")
    file(TO_CMAKE_PATH "${_stf_iscc_user_raw}" _stf_iscc_user)
    if(EXISTS "${_stf_iscc_user}")
      set(CPACK_INNOSETUP_EXECUTABLE "${_stf_iscc_user}")
    endif()
    unset(_stf_iscc_user_raw)
    unset(_stf_iscc_user)
  endif()
  set(CPACK_INNOSETUP_ARCHITECTURE x64)
  set(CPACK_INNOSETUP_USE_MODERN_WIZARD ON)
  set(CPACK_INNOSETUP_ALLOW_NO_ICONS ON)
  set(CPACK_PACKAGE_EXECUTABLES "stimfit;Stimfit")
  set(CPACK_COMPONENTS_ALL ${STF_COMPONENT_CORE})
  if(STF_ENABLE_PYTHON)
    list(APPEND CPACK_COMPONENTS_ALL ${STF_COMPONENT_PYTHON})
  endif()
  if(STF_BUILD_TESTS)
    list(APPEND CPACK_COMPONENTS_ALL ${STF_COMPONENT_TESTS})
  endif()

  set(CPACK_COMPONENT_CORE_DISPLAY_NAME "Stimfit Application")
  set(CPACK_COMPONENT_CORE_DESCRIPTION "Stimfit executable and required runtime dependencies")
  set(CPACK_COMPONENT_PYTHON_DISPLAY_NAME "Embedded Python Runtime")
  set(CPACK_COMPONENT_PYTHON_DESCRIPTION "Bundled Python runtime, stdlib, DLLs, and selected site-packages")
  set(CPACK_COMPONENT_TESTS_DISPLAY_NAME "Stimfit Tests")
  set(CPACK_COMPONENT_TESTS_DESCRIPTION "Stimfit test executable")
  set(CPACK_COMPONENT_PYTHON_DEPENDS ${STF_COMPONENT_CORE})
  set(CPACK_COMPONENT_TESTS_DEPENDS ${STF_COMPONENT_CORE})
else()
  set(CPACK_GENERATOR "TGZ")
endif()

include(CPack)
