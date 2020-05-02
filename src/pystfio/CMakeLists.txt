cmake_minimum_required(VERSION 3.12)

if(NOT Numpy_INCLUDE_DIRS)
    message(FATAL_ERROR "Could not find numpy for ${Python_EXECUTABLE}")
endif()

set_property(SOURCE pystfio.i PROPERTY CPLUSPLUS ON)
swig_add_library(stfio LANGUAGE python SOURCES pystfio.i pystfio.cxx)
target_include_directories(stfio PRIVATE 
    ${Python_INCLUDE_DIRS} 
    ${Numpy_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR})

swig_link_libraries(stfio PRIVATE libstfio libstfnum)

# copy source tree to build directory.
add_custom_command(TARGET stfio POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM)

# make wheel shall make the wheel
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_SOURCE_DIR}/setup.py)

set(STFIO_INSTALL_DIR ${CMAKE_BINARY_DIR}/_stfio_install_dir)
add_custom_target(bdist ALL
    DEPENDS stfio
    COMMAND ${Python_EXECUTABLE} setup.py bdist_wheel
        -b ${STFIO_INSTALL_DIR}/${Python_SITELIB} -k 
    COMMENT "Creating python bdist"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM)

install(FILES stfio_plot.py DESTINATION lib/stimfit)
install(DIRECTORY ${STFIO_INSTALL_DIR}/usr/lib/ DESTINATION lib)

enable_testing()
add_test(NAME test_stfio 
    COMMAND ${Python_EXECUTABLE} unittest_stfio.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
