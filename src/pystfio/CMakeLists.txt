cmake_minimum_required(VERSION 3.12)

if(POLICY CMP0078)
    cmake_policy(SET CMP0078 NEW)
endif()
if(POLICY CMP0086)
    cmake_policy(SET CMP0086 NEW)
endif()


find_package(Python COMPONENTS Interpreter Development)

find_package(SWIG REQUIRED)
include(UseSWIG)

set_property(SOURCE pystfio.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE pystfio.i PROPERTY SWIG_MODULE_NAME _stfio)
swig_add_library(pystfio LANGUAGE python SOURCES pystfio.i)
target_include_directories(pystfio PRIVATE ${Python_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

swig_link_libraries(pystfio stfnum)

enable_testing()
add_test(NAME test_stfio 
    COMMAND Python::Interpreter unittest_stfio.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
