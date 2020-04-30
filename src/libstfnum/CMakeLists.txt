cmake_minimum_required(VERSION 3.12)

find_package(FFTW REQUIRED)

set(SRCS
    fit.cpp 
    levmar/lm.c 
    levmar/Axb.c
    levmar/misc.c 
    levmar/lmlec.c 
    levmar/lmbc.c 
    funclib.cpp 
    stfnum.cpp
    measure.cpp)

add_library(stfnum_static STATIC ${SRCS})
set_target_properties(stfnum_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(stfnum_static PUBLIC ${FFTW_LIBRARIES})
