include_guard(GLOBAL)

# Pin the upstream release, independently of libbiosig's API/ABI version.
# Normal variables deliberately supersede obsolete pins in existing CMake caches.
set(STF_BIOSIG_VERSION "4.0.0")
set(STF_BIOSIG_EXPECTED_TAG "v${STF_BIOSIG_VERSION}")

function(stf_validate_biosig_source source_dir out_commit)
  if(NOT EXISTS "${source_dir}/configure.ac")
    message(FATAL_ERROR "Biosig ${STF_BIOSIG_VERSION} sources are missing. Run git submodule update --init --recursive.")
  endif()
  file(STRINGS "${source_dir}/configure.ac" _biosig_ac_init REGEX "^AC_INIT")
  if(NOT _biosig_ac_init MATCHES "AC_INIT\\(\\[biosig\\], *\\[([^]]+)\\]\\)")
    message(FATAL_ERROR "Cannot determine the Biosig release from ${source_dir}/configure.ac")
  endif()
  if(NOT CMAKE_MATCH_1 STREQUAL STF_BIOSIG_VERSION)
    message(FATAL_ERROR "Expected Biosig release ${STF_BIOSIG_VERSION}, found ${CMAKE_MATCH_1}. Update src/biosig to ${STF_BIOSIG_EXPECTED_TAG}.")
  endif()

  set(_biosig_head "")
  # Source archives have no Git metadata; the release metadata still gets checked.
  if(EXISTS "${source_dir}/.git")
    find_program(STF_GIT_EXECUTABLE NAMES git REQUIRED)
    execute_process(COMMAND "${STF_GIT_EXECUTABLE}" -C "${source_dir}" rev-parse HEAD
      RESULT_VARIABLE _biosig_result OUTPUT_VARIABLE _biosig_head OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _biosig_result EQUAL 0)
      message(FATAL_ERROR "Cannot determine Biosig HEAD in ${source_dir}")
    endif()
    execute_process(COMMAND "${STF_GIT_EXECUTABLE}" -C "${source_dir}" rev-parse --verify --quiet "${STF_BIOSIG_EXPECTED_TAG}^{commit}"
      RESULT_VARIABLE _biosig_tag_result OUTPUT_VARIABLE _biosig_tag_commit OUTPUT_STRIP_TRAILING_WHITESPACE)
    # Shallow submodule checkouts may contain only the recorded commit, without tags.
    if(_biosig_tag_result EQUAL 0 AND NOT _biosig_head STREQUAL _biosig_tag_commit)
      message(FATAL_ERROR "Biosig submodule mismatch: expected ${STF_BIOSIG_EXPECTED_TAG} (${_biosig_tag_commit}), found ${_biosig_head}.")
    endif()
  endif()
  set(${out_commit} "${_biosig_head}" PARENT_SCOPE)
  message(STATUS "Biosig release: ${STF_BIOSIG_VERSION} (${STF_BIOSIG_EXPECTED_TAG})")
endfunction()
