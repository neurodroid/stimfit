cmake_minimum_required(VERSION 3.21)
include("${CMAKE_CURRENT_LIST_DIR}/../StimfitPythonPackages.cmake")

if(NOT DEFINED STF_TEST_ROOT)
  set(STF_TEST_ROOT "${CMAKE_CURRENT_BINARY_DIR}/build/python-packages-test")
endif()
file(TO_CMAKE_PATH "${STF_TEST_ROOT}" STF_TEST_ROOT)

# Exercise the failure path in a child process, because FATAL_ERROR terminates
# the current script. A DLL directory alone is not an importable module.
if(STF_TEST_MISSING)
  stf_collect_python_package_sources(_sources
    PACKAGES missing ROOTS "${STF_TEST_ROOT}/extra")
  message(FATAL_ERROR "Missing Python module was accepted")
endif()

file(MAKE_DIRECTORY
  "${STF_TEST_ROOT}/primary/numpy"
  "${STF_TEST_ROOT}/primary/numpy.libs"
  "${STF_TEST_ROOT}/primary/numpy-1.0.dist-info"
  "${STF_TEST_ROOT}/extra/numpy"
  "${STF_TEST_ROOT}/extra/numpy.libs"
  "${STF_TEST_ROOT}/extra/matplotlib"
  "${STF_TEST_ROOT}/extra/matplotlib.libs"
  "${STF_TEST_ROOT}/extra/matplotlib-1.0.dist-info"
  "${STF_TEST_ROOT}/extra/mpl_toolkits"
  "${STF_TEST_ROOT}/extra/fontTools"
  "${STF_TEST_ROOT}/extra/fonttools-1.0.dist-info"
  "${STF_TEST_ROOT}/extra/PIL"
  "${STF_TEST_ROOT}/extra/pillow.libs"
  "${STF_TEST_ROOT}/extra/pillow-1.0.dist-info"
  "${STF_TEST_ROOT}/extra/dateutil"
  "${STF_TEST_ROOT}/extra/python_dateutil-1.0.dist-info"
  "${STF_TEST_ROOT}/extra/missing.libs")
file(WRITE "${STF_TEST_ROOT}/extra/six.py" "")
file(WRITE "${STF_TEST_ROOT}/extra/pylab.py" "")
file(WRITE "${STF_TEST_ROOT}/extra/kiwisolver.cp314-win_amd64.pyd" "")
file(WRITE "${STF_TEST_ROOT}/extra/missing_unrelated.pyd" "")

stf_collect_python_package_sources(_sources
  PACKAGES numpy matplotlib PIL dateutil six kiwisolver fontTools
  ROOTS "${STF_TEST_ROOT}/primary" "${STF_TEST_ROOT}/primary" "${STF_TEST_ROOT}/extra")
set(_expected
  "${STF_TEST_ROOT}/primary/numpy"
  "${STF_TEST_ROOT}/primary/numpy.libs"
  "${STF_TEST_ROOT}/primary/numpy-1.0.dist-info"
  "${STF_TEST_ROOT}/extra/matplotlib"
  "${STF_TEST_ROOT}/extra/matplotlib.libs"
  "${STF_TEST_ROOT}/extra/matplotlib-1.0.dist-info"
  "${STF_TEST_ROOT}/extra/mpl_toolkits"
  "${STF_TEST_ROOT}/extra/pylab.py"
  "${STF_TEST_ROOT}/extra/fontTools"
  "${STF_TEST_ROOT}/extra/fonttools-1.0.dist-info"
  "${STF_TEST_ROOT}/extra/PIL"
  "${STF_TEST_ROOT}/extra/pillow.libs"
  "${STF_TEST_ROOT}/extra/pillow-1.0.dist-info"
  "${STF_TEST_ROOT}/extra/dateutil"
  "${STF_TEST_ROOT}/extra/python_dateutil-1.0.dist-info"
  "${STF_TEST_ROOT}/extra/six.py"
  "${STF_TEST_ROOT}/extra/kiwisolver.cp314-win_amd64.pyd")
list(SORT _sources)
list(SORT _expected)
if(NOT _sources STREQUAL _expected)
  message(FATAL_ERROR "Incomplete or mixed wheel contents. Expected: ${_expected}; got: ${_sources}")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}"
  "-DSTF_TEST_ROOT=${STF_TEST_ROOT}" -DSTF_TEST_MISSING=ON
  -P "${CMAKE_CURRENT_LIST_FILE}"
  RESULT_VARIABLE _missing_result ERROR_VARIABLE _missing_error)
if(_missing_result EQUAL 0 OR NOT _missing_error MATCHES "Required Windows Python modules were not found: missing")
  message(FATAL_ERROR "Missing module did not produce a clear failure: ${_missing_error}")
endif()
message(STATUS "Windows Python package collection regression passed")
