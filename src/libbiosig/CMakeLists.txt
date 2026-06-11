include_guard(DIRECTORY)

set(_stf_awk_hints)
if(WIN32)
  list(APPEND _stf_awk_hints
    "C:/Program Files/Git/usr/bin"
    "C:/Program Files (x86)/Git/usr/bin"
    "$ENV{LocalAppData}/Programs/Git/usr/bin"
  )
endif()

find_program(STF_AWK_EXECUTABLE
  NAMES gawk awk
  HINTS ${_stf_awk_hints}
  REQUIRED
)
unset(_stf_awk_hints)

set(STF_BIOSIG4CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../biosig/biosig4c++)
set(STF_BIOSIG_EXTERN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../biosig/biosig4matlab/doc)
set(_stf_biosig_binary_dir ${CMAKE_CURRENT_BINARY_DIR})

if(WIN32)
  set(_stf_shell_executable cmd)
  set(_stf_shell_flag /C)
else()
  set(_stf_shell_executable /bin/sh)
  set(_stf_shell_flag -c)
endif()

set(_stf_biosig_generated_eventcodes ${_stf_biosig_binary_dir}/eventcodes.i)
set(_stf_biosig_generated_eventcodegroups ${_stf_biosig_binary_dir}/eventcodegroups.i)
set(_stf_biosig_generated_units ${_stf_biosig_binary_dir}/units.i)
set(_stf_biosig_generated_annexb ${_stf_biosig_binary_dir}/11073-10102-AnnexB.i)

add_custom_command(
  OUTPUT ${_stf_biosig_generated_eventcodes} ${_stf_biosig_generated_eventcodegroups}
  COMMAND ${STF_AWK_EXECUTABLE} -f ${STF_BIOSIG4CPP_DIR}/eventcodes.awk ${STF_BIOSIG_EXTERN_DIR}/eventcodes.txt
  DEPENDS
    ${STF_BIOSIG4CPP_DIR}/eventcodes.awk
    ${STF_BIOSIG_EXTERN_DIR}/eventcodes.txt
  WORKING_DIRECTORY ${_stf_biosig_binary_dir}
)

add_custom_command(
  OUTPUT ${_stf_biosig_generated_units}
  COMMAND
    ${CMAKE_COMMAND} -E rm -f ${_stf_biosig_generated_units}
  COMMAND
    ${_stf_shell_executable} ${_stf_shell_flag}
    "\"${STF_AWK_EXECUTABLE}\" -f \"${STF_BIOSIG4CPP_DIR}/units.awk\" \"${STF_BIOSIG_EXTERN_DIR}/units.csv\" > \"${_stf_biosig_generated_units}\""
  DEPENDS
    ${STF_BIOSIG4CPP_DIR}/units.awk
    ${STF_BIOSIG_EXTERN_DIR}/units.csv
  WORKING_DIRECTORY ${_stf_biosig_binary_dir}
  VERBATIM
)

add_custom_command(
  OUTPUT ${_stf_biosig_generated_annexb}
  COMMAND
    ${CMAKE_COMMAND} -E rm -f ${_stf_biosig_generated_annexb}
  COMMAND
    ${_stf_shell_executable} ${_stf_shell_flag}
    "\"${STF_AWK_EXECUTABLE}\" -f \"${STF_BIOSIG4CPP_DIR}/annotatedECG.awk\" \"${STF_BIOSIG_EXTERN_DIR}/11073-10102-AnnexB.txt\" > \"${_stf_biosig_generated_annexb}\""
  DEPENDS
    ${STF_BIOSIG4CPP_DIR}/annotatedECG.awk
    ${STF_BIOSIG_EXTERN_DIR}/11073-10102-AnnexB.txt
  WORKING_DIRECTORY ${_stf_biosig_binary_dir}
  VERBATIM
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

find_package(ZLIB REQUIRED)
target_link_libraries(stimfit_biosig PUBLIC ZLIB::ZLIB)

if(APPLE)
  find_package(Iconv REQUIRED)
  target_link_libraries(stimfit_biosig PUBLIC Iconv::Iconv)
endif()

set_target_properties(stimfit_biosig PROPERTIES OUTPUT_NAME biosig2)

if(APPLE AND STF_MACOS_APP_BUNDLE)
  set(_stf_biosig_install_libdir "stimfit.app/Contents/lib/stimfit")
else()
  set(_stf_biosig_install_libdir "${CMAKE_INSTALL_LIBDIR}/stimfit")
endif()

if(APPLE AND STF_MACOS_APP_BUNDLE)
  install(TARGETS stimfit_biosig
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${_stf_biosig_install_libdir}
  )
else()
  install(TARGETS stimfit_biosig
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${_stf_biosig_install_libdir}
    ARCHIVE DESTINATION ${_stf_biosig_install_libdir}
  )
endif()
