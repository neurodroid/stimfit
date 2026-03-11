include_guard(DIRECTORY)

add_library(
  stimfit_core
  stf.cpp
  Annotation.cpp
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

if(TARGET biosiglite)
  target_link_libraries(stimfit_core PUBLIC biosiglite)
endif()

if(STF_ENABLE_PYTHON)
  target_link_libraries(stimfit_core PUBLIC stimfit::python)
  if(DEFINED STF_NUMPY_INCLUDE AND NOT "${STF_NUMPY_INCLUDE}" STREQUAL "")
    target_include_directories(stimfit_core PRIVATE ${STF_NUMPY_INCLUDE})
  endif()
endif()

set_target_properties(stimfit_core PROPERTIES OUTPUT_NAME stimfit)

install(TARGETS stimfit_core
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${STF_COMPONENT_CORE}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/stimfit COMPONENT ${STF_COMPONENT_CORE}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/stimfit COMPONENT ${STF_COMPONENT_CORE}
)

