cmake_minimum_required(VERSION 3.12)

find_package(ZLIB REQUIRED)

set(SRCS
  biosig4c++/t210/sopen_abf_read.c 
  biosig4c++/t210/sopen_alpha_read.c 
  biosig4c++/t210/sopen_axg_read.c 
  biosig4c++/t210/sopen_cfs_read.c 
  biosig4c++/t210/sopen_heka_read.c
  biosig4c++/t210/sopen_igor.c
  biosig4c++/t210/sopen_rhd2000_read.c
  biosig4c++/t210/sopen_scp_read.c
  biosig4c++/t210/scp-decode.cpp
  biosig4c++/t220/crc4scp.c 
  biosig4c++/t220/sopen_scp_write.c 
  biosig4c++/test0/sandbox.c 
  biosig4c++/biosig.c 
  biosig4c++/biosig2.c 
  biosig4c++/biosig-HL7aECG-stub.c 
  biosig4c++/gdftime.c 
  biosig4c++/mdc_ecg_codes.c 
  biosig4c++/physicalunits.c
  biosig4c++/biosig-network.c
  )


add_library(biosiglite SHARED ${SRCS})
target_link_libraries(biosiglite ${ZLIB_LIBRARIES} m)
