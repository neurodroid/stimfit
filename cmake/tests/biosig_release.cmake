cmake_minimum_required(VERSION 3.21)
get_filename_component(_repo_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

# Reconfiguration must replace obsolete cached release and revision pins.
set(STF_BIOSIG_EXPECTED_TAG "v4.0.0" CACHE STRING "Legacy pin" FORCE)
set(STF_BIOSIG_REVISION "7e08ed0dc3cce07708e87d138053955de72c97a9" CACHE STRING "Legacy revision" FORCE)
include("${_repo_root}/cmake/StimfitBiosigVersion.cmake")
if(NOT STF_BIOSIG_EXPECTED_TAG STREQUAL STF_BIOSIG_REVISION)
  message(FATAL_ERROR "The cached legacy pin overrode the revision pin")
endif()
stf_validate_biosig_source("${_repo_root}/src/biosig" _source_commit)
if(NOT _source_commit STREQUAL "" AND NOT _source_commit STREQUAL STF_BIOSIG_REVISION)
  message(FATAL_ERROR "The validated commit differs from the shared revision pin")
endif()

# Source archives retain release metadata validation without requiring Git.
if(NOT DEFINED STF_TEST_ROOT)
  set(STF_TEST_ROOT "${_repo_root}/build/biosig-release-test")
endif()
set(_fixture "${STF_TEST_ROOT}")
file(MAKE_DIRECTORY "${_fixture}")
file(WRITE "${_fixture}/configure.ac" "AC_INIT([biosig], [${STF_BIOSIG_VERSION}])\n")
stf_validate_biosig_source("${_fixture}" _archive_commit)
file(WRITE "${_fixture}/configure.ac" "AC_INIT([biosig], [3.0.3])\n")
file(WRITE "${_fixture}/reject.cmake"
  "include(\"${_repo_root}/cmake/StimfitBiosigVersion.cmake\")\n"
  "stf_validate_biosig_source(\"${_fixture}\" _commit)\n")
execute_process(COMMAND "${CMAKE_COMMAND}" -P "${_fixture}/reject.cmake"
  RESULT_VARIABLE _result OUTPUT_VARIABLE _output ERROR_VARIABLE _error)
if(_result EQUAL 0 OR NOT _error MATCHES "Expected Biosig release ${STF_BIOSIG_VERSION}, found 3.0.3")
  message(FATAL_ERROR "An incorrect Biosig release was not rejected: ${_output}${_error}")
endif()

# Matching release metadata must not allow a different Git revision.
if(NOT _source_commit STREQUAL "")
  file(WRITE "${_fixture}/reject-revision.cmake"
    "include(\"${_repo_root}/cmake/StimfitBiosigVersion.cmake\")\n"
    "set(STF_BIOSIG_REVISION \"0000000000000000000000000000000000000000\")\n"
    "stf_validate_biosig_source(\"${_repo_root}/src/biosig\" _commit)\n")
  execute_process(COMMAND "${CMAKE_COMMAND}" -P "${_fixture}/reject-revision.cmake"
    RESULT_VARIABLE _result OUTPUT_VARIABLE _output ERROR_VARIABLE _error)
  if(_result EQUAL 0 OR NOT _error MATCHES "Biosig submodule mismatch: expected revision")
    message(FATAL_ERROR "An incorrect Biosig revision was not rejected: ${_output}${_error}")
  endif()
endif()
message(STATUS "Biosig revision and release metadata checks passed")
