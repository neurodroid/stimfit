include_guard(DIRECTORY)

if(NOT STF_ENABLE_PYTHON)
  return()
endif()

if(NOT SWIG_FOUND)
  message(FATAL_ERROR "SWIG is required for pystfio when STF_ENABLE_PYTHON=ON")
endif()

set(STF_PYSTFIO_WRAP ${CMAKE_CURRENT_BINARY_DIR}/pystfio_wrap.cxx)
set(STF_PYSTFIO_PY ${CMAKE_CURRENT_BINARY_DIR}/stfio.py)

add_custom_command(
  OUTPUT ${STF_PYSTFIO_WRAP} ${STF_PYSTFIO_PY}
  COMMAND ${SWIG_EXECUTABLE} -python -c++ -o ${STF_PYSTFIO_WRAP} -outdir ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/pystfio.i
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/pystfio.i
    ${CMAKE_CURRENT_SOURCE_DIR}/numpy.i
)

add_library(_stfio MODULE pystfio.cxx ${STF_PYSTFIO_WRAP})
stf_apply_macos_runtime_policy(_stfio)
set_target_properties(_stfio PROPERTIES PREFIX "" OUTPUT_NAME "_stfio")
if(WIN32)
  set_target_properties(_stfio PROPERTIES SUFFIX ".pyd")
endif()

target_include_directories(
  _stfio
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${Python3_INCLUDE_DIRS}
)

if(DEFINED STF_NUMPY_INCLUDE AND NOT "${STF_NUMPY_INCLUDE}" STREQUAL "")
  target_include_directories(_stfio PRIVATE ${STF_NUMPY_INCLUDE})
endif()

target_link_libraries(_stfio PRIVATE stimfit_config stfio stfnum stimfit::python)

if(TARGET biosiglite)
  target_link_libraries(_stfio PRIVATE biosiglite)
endif()

if(DEFINED STF_PYTHON_PLATLIB AND NOT "${STF_PYTHON_PLATLIB}" STREQUAL "")
  set(_stfio_install_dir ${STF_PYTHON_PLATLIB}/stfio)
else()
  set(_stfio_install_dir ${CMAKE_INSTALL_LIBDIR}/python-stfio/stfio)
endif()

install(TARGETS _stfio
  LIBRARY DESTINATION ${_stfio_install_dir} COMPONENT ${STF_COMPONENT_PYTHON}
  RUNTIME DESTINATION ${_stfio_install_dir} COMPONENT ${STF_COMPONENT_PYTHON}
)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/stfio_plot.py
  ${CMAKE_CURRENT_SOURCE_DIR}/stfio_neo.py
  ${STF_PYSTFIO_PY}
  DESTINATION ${_stfio_install_dir}
  COMPONENT ${STF_COMPONENT_PYTHON}
)

if(BUILD_TESTING)
  add_test(NAME pystfio_unittest COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unittest_stfio.py)
  set_tests_properties(pystfio_unittest PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
  )
endif()
