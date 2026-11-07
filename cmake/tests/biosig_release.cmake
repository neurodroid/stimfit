cmake_minimum_required(VERSION 3.21)
get_filename_component(_repo_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

# Reconfiguring a previous build must replace its obsolete cached release pin.
set(STF_BIOSIG_EXPECTED_TAG "v3.9.3" CACHE STRING "Legacy pin" FORCE)
include("${_repo_root}/cmake/StimfitBiosigVersion.cmake")
if(NOT STF_BIOSIG_EXPECTED_TAG STREQUAL "v${STF_BIOSIG_VERSION}")
  message(FATAL_ERROR "The cached legacy pin overrode the release pin")
endif()
stf_validate_biosig_source("${_repo_root}/src/biosig" _release_commit)

# Exercise archive validation without Git metadata, and reject an API version
# mistakenly used as a release version.
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
message(STATUS "Biosig release pin checks passed")
