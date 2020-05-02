cmake_minimum_required(VERSION 3.12)

find_package(LAPACK REQUIRED)
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

add_library(libstfnum SHARED ${SRCS})

set_target_properties(libstfnum PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_NAME stfnum)

target_link_libraries(libstfnum PUBLIC ${FFTW_LIBRARIES} ${LAPACK_LIBRARIES})

install(TARGETS libstfnum DESTINATION lib)
