include_guard(GLOBAL)

# Pin a tested upstream revision, independently of release and API/ABI versions.
# Normal variables supersede obsolete pins in existing CMake caches.
set(STF_BIOSIG_VERSION "4.0.1")
set(STF_BIOSIG_REVISION "fb331df54d0a425e7f7fa67263537ff0852d263d")
# Retain the legacy caller/cache name as an alias for the exact revision.
set(STF_BIOSIG_EXPECTED_TAG "${STF_BIOSIG_REVISION}")

function(stf_validate_biosig_source source_dir out_commit)
  if(NOT EXISTS "${source_dir}/configure.ac")
    message(FATAL_ERROR "Biosig ${STF_BIOSIG_VERSION} sources are missing. Run git submodule update --init --recursive.")
  endif()
  file(STRINGS "${source_dir}/configure.ac" _biosig_ac_init REGEX "^AC_INIT")
  if(NOT _biosig_ac_init MATCHES "AC_INIT\\(\\[biosig\\], *\\[([^]]+)\\]\\)")
    message(FATAL_ERROR "Cannot determine the Biosig release from ${source_dir}/configure.ac")
  endif()
  if(NOT CMAKE_MATCH_1 STREQUAL STF_BIOSIG_VERSION)
    message(FATAL_ERROR "Expected Biosig release ${STF_BIOSIG_VERSION}, found ${CMAKE_MATCH_1}. Update src/biosig to ${STF_BIOSIG_REVISION}.")
  endif()

  set(_biosig_head "")
  # Source archives have no Git metadata; release metadata still gets checked.
  if(EXISTS "${source_dir}/.git")
    find_program(STF_GIT_EXECUTABLE NAMES git REQUIRED)
    execute_process(COMMAND "${STF_GIT_EXECUTABLE}" -C "${source_dir}" rev-parse HEAD
      RESULT_VARIABLE _biosig_result OUTPUT_VARIABLE _biosig_head OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _biosig_result EQUAL 0)
      message(FATAL_ERROR "Cannot determine Biosig HEAD in ${source_dir}")
    endif()
    # Compare directly so shallow checkouts without release tags are checked too.
    if(NOT _biosig_head STREQUAL STF_BIOSIG_REVISION)
      message(FATAL_ERROR "Biosig submodule mismatch: expected revision '${STF_BIOSIG_REVISION}', found '${_biosig_head}'. Run git submodule update --init --recursive.")
    endif()
  endif()
  set(${out_commit} "${_biosig_head}" PARENT_SCOPE)
  message(STATUS "Biosig source: ${STF_BIOSIG_VERSION} (${STF_BIOSIG_REVISION})")
endfunction()
