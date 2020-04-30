cmake_minimum_required(VERSION 3.12)

set_property(SOURCE pystfio.i PROPERTY CPLUSPLUS ON)
swig_add_library(stfio LANGUAGE python SOURCES pystfio.i pystfio.cxx)
target_include_directories(stfio PRIVATE ${Python_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
swig_link_libraries(stfio PRIVATE stfio_static stfnum_static)

# copy source tree to build directory.
add_custom_command(TARGET stfio POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM)

# make wheel shall make the whell
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
