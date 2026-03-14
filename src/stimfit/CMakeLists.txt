include_guard(DIRECTORY)

add_library(
  stimfit_core
  stf.cpp
  gui/app.cpp
  gui/unopt.cpp
  gui/doc.cpp
  gui/copygrid.cpp
  gui/graph.cpp
  gui/printout.cpp
  gui/parentframe.cpp
  gui/childframe.cpp
  gui/view.cpp
  gui/table.cpp
  gui/zoom.cpp
  gui/dlgs/convertdlg.cpp
  gui/dlgs/cursorsdlg.cpp
  gui/dlgs/eventdlg.cpp
  gui/dlgs/fitseldlg.cpp
  gui/dlgs/smalldlgs.cpp
  gui/usrdlg/usrdlg.cpp
)
stf_apply_macos_runtime_policy(stimfit_core)

if(MSVC)
  target_compile_definitions(stimfit_core PRIVATE STFDLL)
endif()

target_compile_definitions(stimfit_core PRIVATE NOPCH)

target_link_libraries(
  stimfit_core
  PUBLIC
    stimfit_config
    stfio
    stfnum
    stimfit::wx
)

if(STF_ENABLE_PYTHON)
  target_link_libraries(stimfit_core PUBLIC stimfit::python)
  if(DEFINED STF_NUMPY_INCLUDE AND NOT "${STF_NUMPY_INCLUDE}" STREQUAL "")
    target_include_directories(stimfit_core PRIVATE ${STF_NUMPY_INCLUDE})
  endif()
endif()

set_target_properties(stimfit_core PROPERTIES OUTPUT_NAME stimfit)

if(APPLE AND STF_MACOS_APP_BUNDLE)
  set(_stf_core_install_libdir "stimfit.app/Contents/lib/stimfit")
else()
  set(_stf_core_install_libdir "${CMAKE_INSTALL_LIBDIR}/stimfit")
endif()

install(TARGETS stimfit_core
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${STF_COMPONENT_CORE}
  LIBRARY DESTINATION ${_stf_core_install_libdir} COMPONENT ${STF_COMPONENT_CORE}
  ARCHIVE DESTINATION ${_stf_core_install_libdir} COMPONENT ${STF_COMPONENT_CORE}
)
