cmake_minimum_required(VERSION 3.21)

if(NOT CMAKE_HOST_WIN32)
	message(FATAL_ERROR "PrepareBiosigMSVC.cmake is only supported on Windows hosts")
endif()

set(_repo_root "${CMAKE_CURRENT_LIST_DIR}/..")

set(STF_BIOSIG_SOURCE_DIR "${_repo_root}/src/biosig" CACHE PATH "Path to biosig submodule checkout")
set(STF_BIOSIG_WORK_DIR "${_repo_root}/build/biosig-msvc-src" CACHE PATH "Path to working copy for patched biosig sources")
set(STF_BIOSIG_BUILD_DIR "${_repo_root}/build/biosig-msvc-build" CACHE PATH "Path to out-of-tree biosig build directory")
set(STF_BIOSIG_PATCH_DIR "${CMAKE_CURRENT_LIST_DIR}/patches/biosig-msvc" CACHE PATH "Path to patch queue directory (*.patch)")
set(STF_BIOSIG_EXPECTED_TAG "v3.9.3" CACHE STRING "Expected biosig tag used as patch baseline")
set(STF_BIOSIG_GENERATOR "Visual Studio 17 2022" CACHE STRING "CMake generator for biosig build")
set(STF_BIOSIG_ARCH "x64" CACHE STRING "Architecture used with Visual Studio generator")
set(STF_BIOSIG_CONFIG "Release" CACHE STRING "Configuration to build")
set(STF_BIOSIG_TARGETS "biosig2shared" CACHE STRING "Semicolon-separated target list to build")
set(STF_BIOSIG_CLEAN ON CACHE BOOL "Remove existing work/build directories before preparing")

if(NOT EXISTS "${STF_BIOSIG_SOURCE_DIR}/biosig4c++/CMakeLists.txt")
	message(FATAL_ERROR "Invalid STF_BIOSIG_SOURCE_DIR: ${STF_BIOSIG_SOURCE_DIR}")
endif()

if(NOT EXISTS "${STF_BIOSIG_PATCH_DIR}")
	message(FATAL_ERROR "Patch directory does not exist: ${STF_BIOSIG_PATCH_DIR}")
endif()

file(GLOB _biosig_patch_files "${STF_BIOSIG_PATCH_DIR}/*.patch")
list(SORT _biosig_patch_files)
if(NOT _biosig_patch_files)
	message(FATAL_ERROR "No .patch files found under ${STF_BIOSIG_PATCH_DIR}")
endif()

find_program(STF_GIT_EXECUTABLE NAMES git REQUIRED)

execute_process(
	COMMAND ${STF_GIT_EXECUTABLE} -C "${STF_BIOSIG_SOURCE_DIR}" rev-parse --verify --quiet "refs/tags/${STF_BIOSIG_EXPECTED_TAG}^{commit}"
	RESULT_VARIABLE _biosig_expected_tag_result
	OUTPUT_VARIABLE _biosig_expected_commit
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _biosig_expected_tag_result EQUAL 0)
	message(FATAL_ERROR "Expected biosig tag '${STF_BIOSIG_EXPECTED_TAG}' was not found in ${STF_BIOSIG_SOURCE_DIR}")
endif()

execute_process(
	COMMAND ${STF_GIT_EXECUTABLE} -C "${STF_BIOSIG_SOURCE_DIR}" rev-parse HEAD
	RESULT_VARIABLE _biosig_head_result
	OUTPUT_VARIABLE _biosig_source_head
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _biosig_head_result EQUAL 0)
	message(FATAL_ERROR "Failed to determine biosig HEAD from ${STF_BIOSIG_SOURCE_DIR}")
endif()

if(NOT _biosig_source_head STREQUAL _biosig_expected_commit)
	message(FATAL_ERROR
		"biosig submodule mismatch: expected tag '${STF_BIOSIG_EXPECTED_TAG}' -> ${_biosig_expected_commit}, "
		"but src/biosig is at ${_biosig_source_head}. "
		"Update src/biosig to tag '${STF_BIOSIG_EXPECTED_TAG}' before running PrepareBiosigMSVC.cmake."
	)
endif()

function(_run_checked)
	execute_process(
		COMMAND ${ARGV}
		RESULT_VARIABLE _stf_result
	)
	if(NOT _stf_result EQUAL 0)
		message(FATAL_ERROR "Command failed (${_stf_result}): ${ARGV}")
	endif()
endfunction()

if(STF_BIOSIG_CLEAN)
	file(REMOVE_RECURSE "${STF_BIOSIG_WORK_DIR}")
	file(REMOVE_RECURSE "${STF_BIOSIG_BUILD_DIR}")
endif()

if(NOT EXISTS "${STF_BIOSIG_WORK_DIR}/.git")
	file(MAKE_DIRECTORY "${STF_BIOSIG_WORK_DIR}")
	file(REMOVE_RECURSE "${STF_BIOSIG_WORK_DIR}")
	_run_checked(${STF_GIT_EXECUTABLE} clone --local --no-hardlinks "${STF_BIOSIG_SOURCE_DIR}" "${STF_BIOSIG_WORK_DIR}")
endif()

_run_checked(${STF_GIT_EXECUTABLE} -C "${STF_BIOSIG_WORK_DIR}" config core.autocrlf false)
_run_checked(${STF_GIT_EXECUTABLE} -C "${STF_BIOSIG_WORK_DIR}" config core.eol lf)
_run_checked(${STF_GIT_EXECUTABLE} -C "${STF_BIOSIG_WORK_DIR}" checkout --force "${_biosig_expected_commit}")
_run_checked(${STF_GIT_EXECUTABLE} -C "${STF_BIOSIG_WORK_DIR}" reset --hard)
_run_checked(${STF_GIT_EXECUTABLE} -C "${STF_BIOSIG_WORK_DIR}" clean -fdx)

foreach(_patch IN LISTS _biosig_patch_files)
	execute_process(
		COMMAND ${STF_GIT_EXECUTABLE} -C "${STF_BIOSIG_WORK_DIR}" apply --3way --ignore-space-change --ignore-whitespace --whitespace=nowarn "${_patch}"
		RESULT_VARIABLE _patch_result
	)
	if(NOT _patch_result EQUAL 0)
		message(FATAL_ERROR "Failed to apply patch: ${_patch}")
	endif()
endforeach()

file(MAKE_DIRECTORY "${STF_BIOSIG_BUILD_DIR}")

_run_checked(
	"${CMAKE_COMMAND}"
	-S "${STF_BIOSIG_WORK_DIR}/biosig4c++"
	-B "${STF_BIOSIG_BUILD_DIR}"
	-G "${STF_BIOSIG_GENERATOR}"
	-A "${STF_BIOSIG_ARCH}"
)

set(_build_targets)
foreach(_tgt IN LISTS STF_BIOSIG_TARGETS)
	list(APPEND _build_targets --target "${_tgt}")
endforeach()

_run_checked(
	"${CMAKE_COMMAND}"
	--build "${STF_BIOSIG_BUILD_DIR}"
	--config "${STF_BIOSIG_CONFIG}"
	${_build_targets}
)

set(_biosig_include_dir "${STF_BIOSIG_WORK_DIR}/biosig4c++")
set(_biosig_library "${STF_BIOSIG_BUILD_DIR}/${STF_BIOSIG_CONFIG}/biosig2.lib")

if(NOT EXISTS "${_biosig_library}")
	message(FATAL_ERROR "Expected library was not produced: ${_biosig_library}")
endif()

set(_hint_file "${STF_BIOSIG_BUILD_DIR}/biosig-msvc-paths.cmake")
file(WRITE "${_hint_file}"
"# Generated by cmake/PrepareBiosigMSVC.cmake\n"
"set(BIOSIG_INCLUDE_DIR \"${_biosig_include_dir}\")\n"
"set(BIOSIG_LIBRARY \"${_biosig_library}\")\n"
)

message(STATUS "Prepared patched biosig source: ${STF_BIOSIG_WORK_DIR}")
message(STATUS "Built patched biosig in: ${STF_BIOSIG_BUILD_DIR}")
message(STATUS "BIOSIG_INCLUDE_DIR=${_biosig_include_dir}")
message(STATUS "BIOSIG_LIBRARY=${_biosig_library}")
message(STATUS "Path hint file: ${_hint_file}")
