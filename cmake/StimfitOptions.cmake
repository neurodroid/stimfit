include_guard(GLOBAL)

option(STF_BUILD_MODULE "Build standalone Python module (like --enable-module)" OFF)
option(STF_BUILD_TESTS "Build gtest-based stimfittest target" OFF)
option(STF_ENABLE_PYTHON "Enable Python integration (like --enable-python)" ON)
option(STF_ENABLE_IPYTHON "Enable IPython shell integration (like --enable-ipython)" OFF)
set(STF_PY_SHELL_BACKEND "MODERN" CACHE STRING "Embedded Python shell backend (MODERN or LEGACY)")
set_property(CACHE STF_PY_SHELL_BACKEND PROPERTY STRINGS MODERN LEGACY)

option(STF_WINDOWS_COPY_PYTHON_RUNTIME "Copy Python runtime DLL to install/bin on Windows" ON)
option(STF_WINDOWS_COPY_PYTHON_STDLIB "Copy Python standard library to install/bin/Lib on Windows" ON)
option(STF_WINDOWS_COPY_PYTHON_DLLS "Copy Python DLLs directory to install/bin/DLLs on Windows" ON)
option(STF_WINDOWS_COPY_PYTHON_SITE_PACKAGES "Copy selected Python site-packages into install/stf-site-packages on Windows" ON)
set(STF_WINDOWS_PYTHON_SITE_PACKAGES "numpy;wx" CACHE STRING "Semicolon-separated Python packages copied to install/stf-site-packages on Windows")
set(STF_WINDOWS_PYTHON_EXTRA_PATHS "" CACHE STRING "Semicolon-separated extra Python import paths (e.g. local Phoenix checkout root)")
set(STF_WXPYTHON_INCLUDE_DIR "" CACHE PATH "Path containing wxPython/wxpy_api.h (e.g. <Phoenix>/wx/include)")
option(STF_ENABLE_PSLOPE "Enable slope cursor measurements (like --enable-pslope)" OFF)
option(STF_ENABLE_AUI "Enable experimental AUI doc/view mode (like --enable-aui)" OFF)
option(STF_BUILD_DEBIAN "Enable Debian-oriented build flags/paths (like --enable-debian)" OFF)

option(STF_WITH_BIOSIG "Use external libbiosig if available" ON)
option(STF_WITH_BIOSIGLITE "Use bundled biosiglite implementation" OFF)
option(STF_USE_BIOSIG_SUBMODULE "Use src/biosig submodule as the default BIOSIG provider" ON)

set(STF_HDF5_PREFIX "" CACHE PATH "Optional HDF5 installation prefix (like --with-hdf5-prefix)")

if(STF_BUILD_MODULE)
  set(STF_ENABLE_PYTHON ON CACHE BOOL "Enable Python integration (like --enable-python)" FORCE)
endif()

string(TOUPPER "${STF_PY_SHELL_BACKEND}" STF_PY_SHELL_BACKEND)
if(NOT STF_PY_SHELL_BACKEND STREQUAL "MODERN" AND NOT STF_PY_SHELL_BACKEND STREQUAL "LEGACY")
  message(FATAL_ERROR "STF_PY_SHELL_BACKEND must be either MODERN or LEGACY")
endif()

if(STF_WITH_BIOSIGLITE)
  set(STF_WITH_BIOSIG OFF CACHE BOOL "Use external libbiosig if available" FORCE)
endif()

