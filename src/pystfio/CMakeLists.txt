cmake_minimum_required(VERSION 3.12)

set_property(SOURCE pystfio.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE pystfio.i PROPERTY SWIG_MODULE_NAME stfio)
swig_add_library(pystfio LANGUAGE python SOURCES pystfio.i)

target_include_directories(pystfio PRIVATE ${Python_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

swig_link_libraries(pystfio PRIVATE stfio stfnum)

# Copy test file to build directory.
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/unittest_stfio.py
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

enable_testing()
add_test(NAME test_stfio 
    COMMAND Python::Interpreter unittest_stfio.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
