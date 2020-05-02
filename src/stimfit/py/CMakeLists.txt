cmake_minimum_required(VERSION 3.12)

set_property(SOURCE pystf.i PROPERTY CPLUSPLUS ON)

swig_add_library(stf LANGUAGE python SOURCES pystf.cxx pystf.i)

target_include_directories(stf PRIVATE 
    ${Python_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${wxWidgets_INCLUDE_DIRS})

target_link_libraries(stf PUBLIC libstimfit libstfio ${wxWidgets_LIBRARIES})
target_include_directories(stf PRIVATE ${WX_INCLUDE_DIRS})
target_include_directories(stf PRIVATE ${Python_INCLUDE_DIRS})
target_include_directories(stf PRIVATE ${Numpy_INCLUDE_DIRS})

# copy the python source tree to build directory.
add_custom_command(TARGET stf 
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM)

set(STF_INSTALL_DIR ${CMAKE_BINARY_DIR}/_STF)
file(MAKE_DIRECTORY ${STF_INSTALL_DIR}/${Python_SITELIB})
add_custom_command(TARGET stf POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy 
        ${CMAKE_CURRENT_BINARY_DIR}/stf.py
        $<TARGET_FILE:stf>
        ${STF_INSTALL_DIR}/${Python_SITELIB}
    VERBATIM)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION lib/stimfit/
    FILES_MATCHING PATTERN "*.py")


install(DIRECTORY ${STF_INSTALL_DIR}/usr/lib/ DESTINATION lib)
