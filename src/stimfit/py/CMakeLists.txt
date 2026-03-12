include_guard(DIRECTORY)

if(NOT STF_ENABLE_PYTHON)
  return()
endif()

if(NOT SWIG_FOUND)
  message(FATAL_ERROR "SWIG is required for embedded stimfit python module when STF_ENABLE_PYTHON=ON")
endif()

set(STF_PYSTF_WRAP ${CMAKE_CURRENT_BINARY_DIR}/pystf_wrap.cxx)
set(STF_STF_PY ${CMAKE_CURRENT_BINARY_DIR}/stf.py)

add_custom_command(
  OUTPUT ${STF_PYSTF_WRAP} ${STF_STF_PY}
  COMMAND ${SWIG_EXECUTABLE} -python -c++ -o ${STF_PYSTF_WRAP} -outdir ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/pystf.i
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/pystf.i
    ${CMAKE_CURRENT_SOURCE_DIR}/numpy.i
)

add_library(pystf MODULE pystf.cxx ${STF_PYSTF_WRAP})
stf_apply_macos_runtime_policy(pystf)
set_target_properties(pystf PROPERTIES PREFIX "" OUTPUT_NAME "_stf")
if(WIN32)
  set_target_properties(pystf PROPERTIES SUFFIX ".pyd")
endif()

target_include_directories(
  pystf
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Python3_INCLUDE_DIRS}
)

if(DEFINED STF_NUMPY_INCLUDE AND NOT "${STF_NUMPY_INCLUDE}" STREQUAL "")
  target_include_directories(pystf PRIVATE ${STF_NUMPY_INCLUDE})
endif()

target_link_libraries(
  pystf
  PRIVATE
    stimfit_config
    stimfit_core
    stfio
    stimfit::wx
    stimfit::python
)

if(TARGET biosiglite)
  target_link_libraries(pystf PRIVATE biosiglite)
endif()

set(_pystf_install_dir ${CMAKE_INSTALL_LIBDIR}/stimfit)
install(TARGETS pystf
  LIBRARY DESTINATION ${_pystf_install_dir} COMPONENT ${STF_COMPONENT_PYTHON}
  RUNTIME DESTINATION ${_pystf_install_dir} COMPONENT ${STF_COMPONENT_PYTHON}
)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/ivtools.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mintools.py
  ${CMAKE_CURRENT_SOURCE_DIR}/natools.py
  ${CMAKE_CURRENT_SOURCE_DIR}/minidemo.py
  ${CMAKE_CURRENT_SOURCE_DIR}/charlie.py
  ${CMAKE_CURRENT_SOURCE_DIR}/hdf5tools.py
  ${STF_STF_PY}
  ${CMAKE_CURRENT_SOURCE_DIR}/spells.py
  ${CMAKE_CURRENT_SOURCE_DIR}/tdms.py
  ${CMAKE_CURRENT_SOURCE_DIR}/embedded_init.py
  ${CMAKE_CURRENT_SOURCE_DIR}/embedded_stf.py
  ${CMAKE_CURRENT_SOURCE_DIR}/embedded_shell_modern.py
  ${CMAKE_CURRENT_SOURCE_DIR}/embedded_mpl.py
  ${CMAKE_CURRENT_SOURCE_DIR}/heka.py
  ${CMAKE_CURRENT_SOURCE_DIR}/extensions.py
  ${CMAKE_SOURCE_DIR}/src/pystfio/stfio_plot.py
  DESTINATION ${_pystf_install_dir}
  COMPONENT ${STF_COMPONENT_PYTHON}
)
