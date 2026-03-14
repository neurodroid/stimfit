include_guard(DIRECTORY)

find_program(STF_AWK_EXECUTABLE NAMES gawk awk REQUIRED)

set(STF_BIOSIG4CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../biosig/biosig4c++)
set(_stf_biosig_binary_dir ${CMAKE_CURRENT_BINARY_DIR})

set(_stf_biosig_generated_eventcodes ${_stf_biosig_binary_dir}/eventcodes.i)
set(_stf_biosig_generated_eventcodegroups ${_stf_biosig_binary_dir}/eventcodegroups.i)
set(_stf_biosig_generated_units ${_stf_biosig_binary_dir}/units.i)
set(_stf_biosig_generated_annexb ${_stf_biosig_binary_dir}/11073-10102-AnnexB.i)

add_custom_command(
  OUTPUT ${_stf_biosig_generated_eventcodes} ${_stf_biosig_generated_eventcodegroups}
  COMMAND ${STF_AWK_EXECUTABLE} -f ${STF_BIOSIG4CPP_DIR}/eventcodes.awk ${STF_BIOSIG4CPP_DIR}/extern/eventcodes.txt
  DEPENDS
    ${STF_BIOSIG4CPP_DIR}/eventcodes.awk
    ${STF_BIOSIG4CPP_DIR}/extern/eventcodes.txt
  WORKING_DIRECTORY ${_stf_biosig_binary_dir}
)

add_custom_command(
  OUTPUT ${_stf_biosig_generated_units}
  COMMAND ${STF_AWK_EXECUTABLE} -f ${STF_BIOSIG4CPP_DIR}/units.awk ${STF_BIOSIG4CPP_DIR}/extern/units.csv > ${_stf_biosig_generated_units}
  DEPENDS
    ${STF_BIOSIG4CPP_DIR}/units.awk
    ${STF_BIOSIG4CPP_DIR}/extern/units.csv
  WORKING_DIRECTORY ${_stf_biosig_binary_dir}
)

add_custom_command(
  OUTPUT ${_stf_biosig_generated_annexb}
  COMMAND ${STF_AWK_EXECUTABLE} -f ${STF_BIOSIG4CPP_DIR}/annotatedECG.awk ${STF_BIOSIG4CPP_DIR}/extern/11073-10102-AnnexB.txt > ${_stf_biosig_generated_annexb}
  DEPENDS
    ${STF_BIOSIG4CPP_DIR}/annotatedECG.awk
    ${STF_BIOSIG4CPP_DIR}/extern/11073-10102-AnnexB.txt
  WORKING_DIRECTORY ${_stf_biosig_binary_dir}
)

add_custom_target(stimfit_biosig_generated
  DEPENDS
    ${_stf_biosig_generated_eventcodes}
    ${_stf_biosig_generated_eventcodegroups}
    ${_stf_biosig_generated_units}
    ${_stf_biosig_generated_annexb}
)

set(_stf_biosig_sources
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_abf_read.c
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_acqbiopac.c
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_alpha_read.c
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_axg_read.c
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_cadwell_read.c
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_cfs_read.c
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_fiff.c
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_heka_read.c
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_igor.c
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_nicolet.c
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_rhd2000_read.c
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_scp_read.c
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_biosigdump_read.c
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_sqlite.c
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_hdf5.c
  ${STF_BIOSIG4CPP_DIR}/t210/sopen_matio.c
  ${STF_BIOSIG4CPP_DIR}/t210/scp-decode.cpp
  ${STF_BIOSIG4CPP_DIR}/t220/crc4scp.c
  ${STF_BIOSIG4CPP_DIR}/t220/sopen_scp_write.c
  ${STF_BIOSIG4CPP_DIR}/t230/sopen_hl7aecg.cpp
  ${STF_BIOSIG4CPP_DIR}/test0/sandbox.c
  ${STF_BIOSIG4CPP_DIR}/XMLParser/tinystr.cpp
  ${STF_BIOSIG4CPP_DIR}/XMLParser/tinyxml.cpp
  ${STF_BIOSIG4CPP_DIR}/XMLParser/tinyxmlerror.cpp
  ${STF_BIOSIG4CPP_DIR}/XMLParser/tinyxmlparser.cpp
  ${STF_BIOSIG4CPP_DIR}/biosig.c
  ${STF_BIOSIG4CPP_DIR}/biosig2.c
  ${STF_BIOSIG4CPP_DIR}/gdftime.c
  ${STF_BIOSIG4CPP_DIR}/mdc_ecg_codes.c
  ${STF_BIOSIG4CPP_DIR}/physicalunits.c
)

if(WIN32)
  list(APPEND _stf_biosig_sources
    ${STF_BIOSIG4CPP_DIR}/win32/getdelim.c
    ${STF_BIOSIG4CPP_DIR}/win32/getline.c
    ${STF_BIOSIG4CPP_DIR}/win32/getlogin.c
  )
elseif(APPLE)
  list(APPEND _stf_biosig_sources
    ${STF_BIOSIG4CPP_DIR}/win32/getdelim.c
    ${STF_BIOSIG4CPP_DIR}/win32/getline.c
  )
endif()

add_library(stimfit_biosig STATIC ${_stf_biosig_sources})
add_dependencies(stimfit_biosig stimfit_biosig_generated)
stf_apply_macos_runtime_policy(stimfit_biosig)

target_compile_definitions(stimfit_biosig PUBLIC WITH_BIOSIG WITH_ZLIB WITHOUT_NETWORK)
target_include_directories(
  stimfit_biosig
  PUBLIC
    ${_stf_biosig_binary_dir}
    ${STF_BIOSIG4CPP_DIR}
    ${STF_BIOSIG4CPP_DIR}/t210
)
target_link_libraries(stimfit_biosig PUBLIC stimfit_config)

find_library(STF_ZLIB_LIBRARY NAMES z zlib REQUIRED)
target_link_libraries(stimfit_biosig PUBLIC ${STF_ZLIB_LIBRARY})

if(APPLE)
  target_link_libraries(stimfit_biosig PUBLIC "-liconv")
endif()

set_target_properties(stimfit_biosig PROPERTIES OUTPUT_NAME biosig2)

if(APPLE AND STF_MACOS_APP_BUNDLE)
  set(_stf_biosig_install_libdir "stimfit.app/Contents/lib/stimfit")
else()
  set(_stf_biosig_install_libdir "${CMAKE_INSTALL_LIBDIR}/stimfit")
endif()

install(TARGETS stimfit_biosig
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${_stf_biosig_install_libdir}
  ARCHIVE DESTINATION ${_stf_biosig_install_libdir}
)
