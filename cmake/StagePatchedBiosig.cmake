cmake_minimum_required(VERSION 3.21)

if(NOT CMAKE_HOST_WIN32)
  message(FATAL_ERROR "StagePatchedBiosig.cmake is only supported on Windows hosts")
endif()

set(STF_BIOSIG_SOURCE_DIR "" CACHE PATH "Path to the pristine biosig checkout")
set(STF_BIOSIG_STAGE_DIR "" CACHE PATH "Path to the staged patched biosig working tree")
set(STF_BIOSIG_PATCH_DIR "" CACHE PATH "Path to biosig patch files")
set(STF_BIOSIG_SIGNATURE_FILE "" CACHE FILEPATH "Path to the generated biosig signature file")
set(STF_BIOSIG_FORCE_REFRESH OFF CACHE BOOL "Force recreation of the staged biosig tree")

if(STF_BIOSIG_SOURCE_DIR STREQUAL "")
  message(FATAL_ERROR "STF_BIOSIG_SOURCE_DIR must be provided")
endif()

if(STF_BIOSIG_STAGE_DIR STREQUAL "")
  message(FATAL_ERROR "STF_BIOSIG_STAGE_DIR must be provided")
endif()

if(STF_BIOSIG_PATCH_DIR STREQUAL "")
  message(FATAL_ERROR "STF_BIOSIG_PATCH_DIR must be provided")
endif()

if(STF_BIOSIG_SIGNATURE_FILE STREQUAL "")
  message(FATAL_ERROR "STF_BIOSIG_SIGNATURE_FILE must be provided")
endif()

if(NOT EXISTS "${STF_BIOSIG_SOURCE_DIR}/biosig4c++/CMakeLists.txt")
  message(FATAL_ERROR "Invalid STF_BIOSIG_SOURCE_DIR: ${STF_BIOSIG_SOURCE_DIR}")
endif()

if(NOT EXISTS "${STF_BIOSIG_PATCH_DIR}")
  message(FATAL_ERROR "Patch directory does not exist: ${STF_BIOSIG_PATCH_DIR}")
endif()

file(GLOB _stf_biosig_patch_files LIST_DIRECTORIES FALSE "${STF_BIOSIG_PATCH_DIR}/*.patch")
list(SORT _stf_biosig_patch_files)

if(NOT _stf_biosig_patch_files)
  message(FATAL_ERROR "No .patch files found under ${STF_BIOSIG_PATCH_DIR}")
endif()

find_program(STF_GIT_EXECUTABLE NAMES git REQUIRED)

function(_stf_run_checked)
  execute_process(
    COMMAND ${ARGV}
    RESULT_VARIABLE _stf_result
  )
  if(NOT _stf_result EQUAL 0)
    message(FATAL_ERROR "Command failed (${_stf_result}): ${ARGV}")
  endif()
endfunction()

set(_stf_refresh_stage ${STF_BIOSIG_FORCE_REFRESH})
set(_stf_existing_signature_file "${STF_BIOSIG_STAGE_DIR}/.stimfit-biosig-signature.txt")

if(EXISTS "${_stf_existing_signature_file}")
  file(READ "${_stf_existing_signature_file}" _stf_existing_signature)
else()
  set(_stf_existing_signature "")
endif()

file(READ "${STF_BIOSIG_SIGNATURE_FILE}" _stf_requested_signature)

if(NOT EXISTS "${STF_BIOSIG_STAGE_DIR}/.git")
  set(_stf_refresh_stage ON)
endif()

if(NOT _stf_existing_signature STREQUAL _stf_requested_signature)
  set(_stf_refresh_stage ON)
endif()

if(_stf_refresh_stage)
  file(REMOVE_RECURSE "${STF_BIOSIG_STAGE_DIR}")
  get_filename_component(_stf_stage_parent "${STF_BIOSIG_STAGE_DIR}" DIRECTORY)
  file(MAKE_DIRECTORY "${_stf_stage_parent}")
  _stf_run_checked(${STF_GIT_EXECUTABLE} clone --local --no-hardlinks "${STF_BIOSIG_SOURCE_DIR}" "${STF_BIOSIG_STAGE_DIR}")
endif()

_stf_run_checked(${STF_GIT_EXECUTABLE} -C "${STF_BIOSIG_STAGE_DIR}" checkout --force HEAD)
_stf_run_checked(${STF_GIT_EXECUTABLE} -C "${STF_BIOSIG_STAGE_DIR}" reset --hard)
_stf_run_checked(${STF_GIT_EXECUTABLE} -C "${STF_BIOSIG_STAGE_DIR}" clean -fdx)

foreach(_stf_patch IN LISTS _stf_biosig_patch_files)
  execute_process(
    COMMAND ${STF_GIT_EXECUTABLE} -C "${STF_BIOSIG_STAGE_DIR}" apply --3way --whitespace=nowarn "${_stf_patch}"
    RESULT_VARIABLE _stf_patch_result
  )
  if(NOT _stf_patch_result EQUAL 0)
    message(FATAL_ERROR "Failed to apply patch: ${_stf_patch}")
  endif()
endforeach()

file(WRITE "${_stf_existing_signature_file}" "${_stf_requested_signature}")

message(STATUS "Staged patched biosig source tree: ${STF_BIOSIG_STAGE_DIR}")
