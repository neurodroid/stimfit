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
swig_link_libraries(stfio PRIVATE stfio_static stfnum_static)


# copy source tree to build directory.
add_custom_command(TARGET stfio POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM)

# make wheel shall make the wheel
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_SOURCE_DIR}/setup.py)

add_custom_target(wheel ALL
    DEPENDS stfio
    COMMAND ${Python_EXECUTABLE} setup.py bdist_wheel 
        -d ${CMAKE_BINARY_DIR}
    COMMENT "Creating python wheel"
    VERBATIM)

enable_testing()
add_test(NAME test_stfio 
    COMMAND Python::Interpreter unittest_stfio.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# install targets
install(CODE 
    "execute_process(COMMAND ${Python_EXECUTABLE} -m pip install
        ${CMAKE_CURRENT_BINARY_DIR} --prefix ${CMAKE_INSTALL_PREFIX})"
    )
