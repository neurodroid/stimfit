include_guard(GLOBAL)

include(ExternalProject)

function(stf_configure_windows_patched_biosig)
  if(TARGET stimfit::biosig)
    return()
  endif()

  if(NOT WIN32)
    message(FATAL_ERROR "stf_configure_windows_patched_biosig() is only valid on Windows")
  endif()

  set(_stf_biosig_source_dir "${CMAKE_SOURCE_DIR}/src/biosig")
  set(_stf_biosig_patch_dir "${CMAKE_SOURCE_DIR}/cmake/patches/biosig-msvc")
  set(STF_BIOSIG_EXPECTED_TAG "v3.9.3" CACHE STRING "Expected biosig tag for Windows patched-submodule provider")

  if(NOT EXISTS "${_stf_biosig_source_dir}/biosig4c++/CMakeLists.txt")
    message(FATAL_ERROR "Patched biosig provider requires the biosig submodule under ${_stf_biosig_source_dir}")
  endif()

  file(GLOB _stf_biosig_patch_files LIST_DIRECTORIES FALSE "${_stf_biosig_patch_dir}/*.patch")
  list(SORT _stf_biosig_patch_files)
  if(NOT _stf_biosig_patch_files)
    message(FATAL_ERROR "No biosig MSVC patch files were found in ${_stf_biosig_patch_dir}")
  endif()

  find_program(STF_GIT_EXECUTABLE NAMES git REQUIRED)

  execute_process(
    COMMAND ${STF_GIT_EXECUTABLE} -C "${_stf_biosig_source_dir}" rev-parse HEAD
    RESULT_VARIABLE _stf_biosig_rev_parse_result
    OUTPUT_VARIABLE _stf_biosig_head
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _stf_biosig_rev_parse_result EQUAL 0)
    message(FATAL_ERROR "Failed to determine biosig submodule HEAD from ${_stf_biosig_source_dir}")
  endif()

  execute_process(
    COMMAND ${STF_GIT_EXECUTABLE} -C "${_stf_biosig_source_dir}" rev-parse --verify --quiet "refs/tags/${STF_BIOSIG_EXPECTED_TAG}^{commit}"
    RESULT_VARIABLE _stf_biosig_expected_tag_result
    OUTPUT_VARIABLE _stf_biosig_expected_commit
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _stf_biosig_expected_tag_result EQUAL 0)
    message(FATAL_ERROR "Expected biosig tag '${STF_BIOSIG_EXPECTED_TAG}' was not found in ${_stf_biosig_source_dir}")
  endif()

  if(NOT _stf_biosig_head STREQUAL _stf_biosig_expected_commit)
    message(FATAL_ERROR
      "biosig submodule mismatch: expected tag '${STF_BIOSIG_EXPECTED_TAG}' -> ${_stf_biosig_expected_commit}, "
      "but src/biosig HEAD is ${_stf_biosig_head}. "
      "Update src/biosig to tag '${STF_BIOSIG_EXPECTED_TAG}' and reconfigure."
    )
  endif()

  execute_process(
    COMMAND ${STF_GIT_EXECUTABLE} -C "${_stf_biosig_source_dir}" status --porcelain --untracked-files=all
    RESULT_VARIABLE _stf_biosig_status_result
    OUTPUT_VARIABLE _stf_biosig_status
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _stf_biosig_status_result EQUAL 0)
    message(FATAL_ERROR "Failed to determine biosig submodule working tree status from ${_stf_biosig_source_dir}")
  endif()

  set(_stf_patch_manifest "")
  foreach(_stf_patch IN LISTS _stf_biosig_patch_files)
    file(SHA256 "${_stf_patch}" _stf_patch_hash)
    string(APPEND _stf_patch_manifest "${_stf_patch}=${_stf_patch_hash}\n")
  endforeach()

  if(DEFINED CMAKE_GENERATOR_PLATFORM AND NOT CMAKE_GENERATOR_PLATFORM STREQUAL "")
    set(_stf_biosig_platform "${CMAKE_GENERATOR_PLATFORM}")
  elseif(CMAKE_VS_PLATFORM_NAME)
    set(_stf_biosig_platform "${CMAKE_VS_PLATFORM_NAME}")
  else()
    set(_stf_biosig_platform "")
  endif()

  set(_stf_biosig_stage_root "${CMAKE_BINARY_DIR}/_deps/biosig")
  set(_stf_biosig_stage_dir "${_stf_biosig_stage_root}/src")
  set(_stf_biosig_build_dir "${_stf_biosig_stage_root}/build")
  set(_stf_biosig_stamp_dir "${_stf_biosig_stage_root}/stamp")
  set(_stf_biosig_tmp_dir "${_stf_biosig_stage_root}/tmp")
  set(_stf_biosig_signature_file "${_stf_biosig_stage_root}/biosig-input-signature.txt")
  set(STF_BIOSIG_PATCHED_TARGETS "biosig2shared" CACHE STRING "Semicolon-separated biosig targets built for the Windows patched provider")

  file(MAKE_DIRECTORY "${_stf_biosig_stage_root}")
  file(MAKE_DIRECTORY "${_stf_biosig_stamp_dir}")
  file(MAKE_DIRECTORY "${_stf_biosig_tmp_dir}")

  set(_stf_biosig_signature
    "biosig-expected-tag=${STF_BIOSIG_EXPECTED_TAG}\n"
    "biosig-expected-commit=${_stf_biosig_expected_commit}\n"
    "biosig-head=${_stf_biosig_head}\n"
    "biosig-status=${_stf_biosig_status}\n"
    "generator=${CMAKE_GENERATOR}\n"
    "platform=${_stf_biosig_platform}\n"
    "targets=${STF_BIOSIG_PATCHED_TARGETS}\n"
    "patches=\n${_stf_patch_manifest}"
  )
  string(JOIN "" _stf_biosig_signature ${_stf_biosig_signature})
  string(SHA256 _stf_biosig_signature_hash "${_stf_biosig_signature}")
  file(WRITE "${_stf_biosig_signature_file}" "${_stf_biosig_signature_hash}\n${_stf_biosig_signature}")

  set(_stf_biosig_stage_stamp "${_stf_biosig_stamp_dir}/stage-${_stf_biosig_signature_hash}.stamp")

  add_custom_command(
    OUTPUT "${_stf_biosig_stage_stamp}"
    COMMAND "${CMAKE_COMMAND}"
      -DSTF_BIOSIG_SOURCE_DIR=${_stf_biosig_source_dir}
      -DSTF_BIOSIG_STAGE_DIR=${_stf_biosig_stage_dir}
      -DSTF_BIOSIG_PATCH_DIR=${_stf_biosig_patch_dir}
      -DSTF_BIOSIG_SIGNATURE_FILE=${_stf_biosig_signature_file}
      -DSTF_BIOSIG_EXPECTED_TAG=${STF_BIOSIG_EXPECTED_TAG}
      -DSTF_BIOSIG_EXPECTED_COMMIT=${_stf_biosig_expected_commit}
      -P "${CMAKE_SOURCE_DIR}/cmake/StagePatchedBiosig.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_stf_biosig_stage_stamp}"
    DEPENDS
      "${_stf_biosig_source_dir}/biosig4c++/CMakeLists.txt"
      ${_stf_biosig_patch_files}
      "${CMAKE_SOURCE_DIR}/cmake/StagePatchedBiosig.cmake"
    COMMENT "Staging patched biosig working tree"
    VERBATIM
  )

  add_custom_target(stimfit_biosig_stage DEPENDS "${_stf_biosig_stage_stamp}")

  set(_stf_biosig_library_dir "${_stf_biosig_build_dir}/$<CONFIG>")
  set(_stf_biosig_import_library "${_stf_biosig_library_dir}/biosig2.lib")
  set(_stf_biosig_runtime_library "${_stf_biosig_library_dir}/biosig2.dll")
  set(_stf_biosig_include_dir "${_stf_biosig_stage_dir}/biosig4c++")
  set(_stf_biosig_release_import_library "${_stf_biosig_build_dir}/Release/biosig2.lib")
  set(_stf_biosig_release_runtime_library "${_stf_biosig_build_dir}/Release/biosig2.dll")
  set(_stf_biosig_debug_import_library "${_stf_biosig_build_dir}/Debug/biosig2.lib")
  set(_stf_biosig_debug_runtime_library "${_stf_biosig_build_dir}/Debug/biosig2.dll")
  set(_stf_biosig_relwithdebinfo_import_library "${_stf_biosig_build_dir}/RelWithDebInfo/biosig2.lib")
  set(_stf_biosig_relwithdebinfo_runtime_library "${_stf_biosig_build_dir}/RelWithDebInfo/biosig2.dll")
  set(_stf_biosig_minsizerel_import_library "${_stf_biosig_build_dir}/MinSizeRel/biosig2.lib")
  set(_stf_biosig_minsizerel_runtime_library "${_stf_biosig_build_dir}/MinSizeRel/biosig2.dll")

  ExternalProject_Add(stimfit_biosig_external
    SOURCE_DIR "${_stf_biosig_stage_dir}/biosig4c++"
    BINARY_DIR "${_stf_biosig_build_dir}"
    STAMP_DIR "${_stf_biosig_stamp_dir}/external"
    TMP_DIR "${_stf_biosig_tmp_dir}"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    INSTALL_COMMAND ""
    CONFIGURE_HANDLED_BY_BUILD ON
    DEPENDS stimfit_biosig_stage
    CMAKE_GENERATOR "${CMAKE_GENERATOR}"
    CMAKE_GENERATOR_PLATFORM "${_stf_biosig_platform}"
    BUILD_COMMAND
      "${CMAKE_COMMAND}" --build "${_stf_biosig_build_dir}" --config $<CONFIG> --target ${STF_BIOSIG_PATCHED_TARGETS}
    BUILD_BYPRODUCTS
      "${_stf_biosig_import_library}"
      "${_stf_biosig_runtime_library}"
  )

  add_library(stimfit::biosig SHARED IMPORTED GLOBAL)
  set_target_properties(stimfit::biosig PROPERTIES
    IMPORTED_CONFIGURATIONS "Debug;Release;RelWithDebInfo;MinSizeRel"
    IMPORTED_IMPLIB_DEBUG "${_stf_biosig_debug_import_library}"
    IMPORTED_LOCATION_DEBUG "${_stf_biosig_debug_runtime_library}"
    IMPORTED_IMPLIB_RELEASE "${_stf_biosig_release_import_library}"
    IMPORTED_LOCATION_RELEASE "${_stf_biosig_release_runtime_library}"
    IMPORTED_IMPLIB_RELWITHDEBINFO "${_stf_biosig_relwithdebinfo_import_library}"
    IMPORTED_LOCATION_RELWITHDEBINFO "${_stf_biosig_relwithdebinfo_runtime_library}"
    IMPORTED_IMPLIB_MINSIZEREL "${_stf_biosig_minsizerel_import_library}"
    IMPORTED_LOCATION_MINSIZEREL "${_stf_biosig_minsizerel_runtime_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_stf_biosig_include_dir}"
  )
  add_dependencies(stimfit::biosig stimfit_biosig_external)

  set(BIOSIG_LIBRARY "${_stf_biosig_release_import_library}" PARENT_SCOPE)
  set(BIOSIG_INCLUDE_DIR "${_stf_biosig_include_dir}" PARENT_SCOPE)
  set(STF_BIOSIG_RUNTIME_DIR "${_stf_biosig_build_dir}/Release" PARENT_SCOPE)
endfunction()
