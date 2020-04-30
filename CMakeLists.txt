cmake_minimum_required(VERSION 3.12)
project(stimfit)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_definitions(-fmax-errors=10)

add_subdirectory(src/libbiosiglite)
add_subdirectory(src/libstfio)
add_subdirectory(src/libstfnum)
add_subdirectory(src/pystfio)
add_subdirectory(src/stimfit)

enable_testing()
add_subdirectory(src/test)
