cmake_minimum_required(VERSION 3.10)
project(stimfit)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(STIMFIT_VERSION 0.16.0) 

# Options.
option(WITH_PYTHON "Build python module" ON)

# Global compiler options.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configuration 
if(WITH_PYTHON)
    if(POLICY CMP0078)
        cmake_policy(SET CMP0078 OLD)
    endif()
    if(POLICY CMP0086)
        cmake_policy(SET CMP0086 NEW)
    endif()
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.12")
        find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
        set(Python_EXECUTABLE ${Python3_EXECUTABLE})
        set(Python_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
        set(Python_LIBRARIES ${Python3_LIBRARIES})
        set(Python_SITELIB ${Python3_SITELIB})
    else()
        find_package(PythonInterp 3.5 REQUIRED)
        set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
        
        find_package(PythonLibs REQUIRED)
        set(Python_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS})
        set(Python_LIBRARIES ${PYTHON_LIBRARIES})

        set(PYTHON_SITELIB_CMD 
            "from distutils import sysconfig; print(sysconfig.get_python_lib(prefix='', plat_specific=False))")

        execute_process(COMMAND "${Python_EXECUTABLE}" "-c" "${PYTHON_SITELIB_CMD}"
            OUTPUT_VARIABLE PYTHON_SITELIB
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
    endif()
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
