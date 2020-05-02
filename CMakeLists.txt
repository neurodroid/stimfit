cmake_minimum_required(VERSION 3.12)
project(stimfit)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(STIMFIT_VERSION 0.16.0) 

# Options.
option(WITH_PYTHON "Build python module" ON)

# Global compiler options.
add_definitions(-fmax-errors=10)

# Configuration 

# where to generate libraries.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(WITH_PYTHON)
    if(POLICY CMP0078)
        cmake_policy(SET CMP0078 NEW)
    endif()
    if(POLICY CMP0086)
        cmake_policy(SET CMP0086 NEW)
    endif()
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    set(Python_EXECUTABLE ${Python3_EXECUTABLE})
    set(Python_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
    set(Python_LIBRARIES ${Python3_LIBRARIES})
    set(Python_SITELIB ${Python3_SITELIB})
    # Find nympy
    execute_process(COMMAND 
        ${Python_EXECUTABLE} -c "import numpy;print(numpy.get_include())"
        OUTPUT_VARIABLE Numpy_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    find_package(SWIG REQUIRED)
    include(UseSWIG)
    add_definitions(-DWITH_PYTHON)
endif()

add_subdirectory(src/libbiosiglite)
add_subdirectory(src/libstfio)
add_subdirectory(src/libstfnum)
add_subdirectory(src/pystfio)
add_subdirectory(src/stimfit)

# Tests
enable_testing()
add_subdirectory(src/test)
