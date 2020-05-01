cmake_minimum_required(VERSION 3.12)

set_property(SOURCE pystf.i PROPERTY CPLUSPLUS ON)

swig_add_library(stf LANGUAGE python SOURCES pystf.cxx pystf.i)

target_include_directories(stf PRIVATE 
    ${Python_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${wxWidgets_INCLUDE_DIRS})

target_link_libraries(stf PUBLIC ${wxWidgets_LIBRARIES})
target_include_directories(stf PRIVATE ${WX_INCLUDE_DIRS})
target_include_directories(stf PRIVATE ${Python_INCLUDE_DIRS})
target_include_directories(stf PRIVATE ${Numpy_INCLUDE_DIRS})

# copy the python source tree to build directory.
add_custom_command(TARGET stf 
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM)
