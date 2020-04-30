cmake_minimum_required(VERSION 3.12)

set_property(SOURCE pystf.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE pytf.i PROPERTY SWIG_MODULE_NAME pystf)
swig_add_library(pystf LANGUAGE python SOURCES pystf.i)

target_include_directories(pystf PRIVATE 
    ${Python_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${wxWidgets_INCLUDE_DIRS})

target_link_libraries(pystf PUBLIC ${wxWidgets_LIBRARIES})
