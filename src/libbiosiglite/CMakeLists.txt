include_guard(DIRECTORY)

find_program(STF_AWK_EXECUTABLE NAMES gawk awk REQUIRED)

set(STF_BIOSIG4CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../biosig/biosig4c++)
set(STF_BIOSIG4MATLAB_DOC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../biosig/biosig4matlab/doc)

set(_eventcodes_i ${CMAKE_CURRENT_BINARY_DIR}/eventcodes.i)
set(_eventcodegroups_i ${CMAKE_CURRENT_BINARY_DIR}/eventcodegroups.i)
set(_units_i ${CMAKE_CURRENT_BINARY_DIR}/units.i)
set(_annexb_i ${CMAKE_CURRENT_BINARY_DIR}/11073-10102-AnnexB.i)

add_custom_command(
  OUTPUT ${_eventcodes_i} ${_eventcodegroups_i}
  COMMAND ${STF_AWK_EXECUTABLE} -f ${STF_BIOSIG4CPP_DIR}/eventcodes.awk ${STF_BIOSIG4MATLAB_DOC_DIR}/eventcodes.txt
  DEPENDS
    ${STF_BIOSIG4CPP_DIR}/eventcodes.awk
    ${STF_BIOSIG4MATLAB_DOC_DIR}/eventcodes.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
  OUTPUT ${_units_i}
  COMMAND ${STF_AWK_EXECUTABLE} -f ${STF_BIOSIG4CPP_DIR}/units.awk ${STF_BIOSIG4MATLAB_DOC_DIR}/units.csv > ${_units_i}
  DEPENDS
    ${STF_BIOSIG4CPP_DIR}/units.awk
    ${STF_BIOSIG4MATLAB_DOC_DIR}/units.csv
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
  OUTPUT ${_annexb_i}
  COMMAND ${STF_AWK_EXECUTABLE} -f ${STF_BIOSIG4CPP_DIR}/annotatedECG.awk ${STF_BIOSIG4MATLAB_DOC_DIR}/11073-10102-AnnexB.txt > ${_annexb_i}
  DEPENDS
    ${STF_BIOSIG4CPP_DIR}/annotatedECG.awk
    ${STF_BIOSIG4MATLAB_DOC_DIR}/11073-10102-AnnexB.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(biosiglite_generated
  DEPENDS
    ${_eventcodes_i}
    ${_eventcodegroups_i}
    ${_units_i}
    ${_annexb_i}
)

add_library(
  biosiglite
  ../biosig/biosig4c++/t210/sopen_abf_read.c
  ../biosig/biosig4c++/t210/sopen_alpha_read.c
  ../biosig/biosig4c++/t210/sopen_axg_read.c
  ../biosig/biosig4c++/t210/sopen_cfs_read.c
  ../biosig/biosig4c++/t210/sopen_heka_read.c
  ../biosig/biosig4c++/t210/sopen_igor.c
  ../biosig/biosig4c++/t210/sopen_rhd2000_read.c
  ../biosig/biosig4c++/t210/sopen_scp_read.c
  ../biosig/biosig4c++/t210/sopen_matio.c
  ../biosig/biosig4c++/t210/sopen_biosigdump_read.c
  ../biosig/biosig4c++/t210/sopen_sqlite.c
  ../biosig/biosig4c++/t210/sopen_hdf5.c
  ../biosig/biosig4c++/t210/sopen_cadwell_read.c
  ../biosig/biosig4c++/t210/scp-decode.cpp
  ../biosig/biosig4c++/t220/crc4scp.c
  ../biosig/biosig4c++/t220/sopen_scp_write.c
  ../biosig/biosig4c++/t230/sopen_hl7aecg.cpp
  ../biosig/biosig4c++/test0/sandbox.c
  ../biosig/biosig4c++/biosig.c
  ../biosig/biosig4c++/biosig2.c
  ../biosig/biosig4c++/gdftime.c
  ../biosig/biosig4c++/mdc_ecg_codes.c
  ../biosig/biosig4c++/physicalunits.c
)

add_dependencies(biosiglite biosiglite_generated)

target_compile_definitions(biosiglite PUBLIC WITH_BIOSIG WITH_ZLIB WITHOUT_NETWORK)
target_include_directories(
  biosiglite
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../biosig/biosig4c++
    ${CMAKE_CURRENT_SOURCE_DIR}/../biosig/biosig4c++/t210
)

find_library(STF_ZLIB_LIBRARY NAMES z zlib REQUIRED)
target_link_libraries(biosiglite PUBLIC stimfit_config ${STF_ZLIB_LIBRARY})

set_target_properties(biosiglite PROPERTIES OUTPUT_NAME biosiglite)

install(TARGETS biosiglite
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/stimfit
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/stimfit
)

