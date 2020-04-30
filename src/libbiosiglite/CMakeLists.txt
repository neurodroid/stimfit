cmake_minimum_required(VERSION 3.12)

find_package(ZLIB REQUIRED)

set(SRCS biosig4c++/biosig2.c
    biosig4c++/biosig.c
    biosig4c++/biosig-HL7aECG-stub.c
    biosig4c++/biosig-network.c
    biosig4c++/gdftime.c
    biosig4c++/mdc_ecg_codes.c
    biosig4c++/physicalunits.c)

add_library(biosiglite STATIC ${SRCS})
target_link_libraries(biosiglite ${ZLIB_LIBRARIES})
