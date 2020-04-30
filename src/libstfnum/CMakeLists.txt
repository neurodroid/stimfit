cmake_minimum_required(VERSION 3.12)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)

find_package(FFTW REQUIRED)

add_library(stfnum SHARED 
    fit.cpp 
    levmar/lm.c 
    levmar/Axb.c
    levmar/misc.c 
    levmar/lmlec.c 
    levmar/lmbc.c 
    funclib.cpp 
    stfnum.cpp
    measure.cpp)

target_link_libraries(stfnum ${FFTW_LIBRARIES})
