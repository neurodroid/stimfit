include_guard(GLOBAL)

option(STF_BUILD_MODULE "Build standalone Python module (like --enable-module)" OFF)
option(STF_BUILD_TESTS "Build gtest-based stimfittest target" OFF)
option(STF_BUILD_NUMERIC_TESTS "Build stfnum-based gtest suites in addition to minimal container tests" OFF)
option(STF_ENABLE_PYTHON "Enable Python integration (like --enable-python)" ON)
option(STF_MACOS_APP_BUNDLE "Build and install stimfit as a macOS .app bundle" OFF)
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

option(STF_WITH_BIOSIG "Enable BIOSIG support" ON)
set(STF_BIOSIG_PROVIDER "AUTO" CACHE STRING "BIOSIG provider (AUTO, SYSTEM, or SUBMODULE)")
set_property(CACHE STF_BIOSIG_PROVIDER PROPERTY STRINGS AUTO SYSTEM SUBMODULE)

set(STF_HDF5_PREFIX "" CACHE PATH "Optional HDF5 installation prefix (like --with-hdf5-prefix)")

if(STF_BUILD_MODULE)
  set(STF_ENABLE_PYTHON ON CACHE BOOL "Enable Python integration (like --enable-python)" FORCE)
endif()

string(TOUPPER "${STF_PY_SHELL_BACKEND}" STF_PY_SHELL_BACKEND)
if(NOT STF_PY_SHELL_BACKEND STREQUAL "MODERN" AND NOT STF_PY_SHELL_BACKEND STREQUAL "LEGACY")
  message(FATAL_ERROR "STF_PY_SHELL_BACKEND must be either MODERN or LEGACY")
endif()

string(TOUPPER "${STF_BIOSIG_PROVIDER}" STF_BIOSIG_PROVIDER)
if(NOT STF_BIOSIG_PROVIDER STREQUAL "AUTO"
   AND NOT STF_BIOSIG_PROVIDER STREQUAL "SYSTEM"
   AND NOT STF_BIOSIG_PROVIDER STREQUAL "SUBMODULE")
  message(FATAL_ERROR "STF_BIOSIG_PROVIDER must be AUTO, SYSTEM, or SUBMODULE")
endif()

if(DEFINED STF_WITH_BIOSIGLITE)
  set(_STF_WITH_BIOSIGLITE_VALUE "${STF_WITH_BIOSIGLITE}")
  unset(STF_WITH_BIOSIGLITE CACHE)
  unset(STF_WITH_BIOSIGLITE)

  if(_STF_WITH_BIOSIGLITE_VALUE)
    message(DEPRECATION "STF_WITH_BIOSIGLITE is deprecated; use STF_WITH_BIOSIG=ON and STF_BIOSIG_PROVIDER=SUBMODULE instead")
    set(STF_WITH_BIOSIG ON CACHE BOOL "Enable BIOSIG support" FORCE)
    set(STF_BIOSIG_PROVIDER "SUBMODULE" CACHE STRING "BIOSIG provider (AUTO, SYSTEM, or SUBMODULE)" FORCE)
  endif()
endif()

if(DEFINED STF_USE_BIOSIG_SUBMODULE)
  set(_STF_USE_BIOSIG_SUBMODULE_VALUE "${STF_USE_BIOSIG_SUBMODULE}")
  unset(STF_USE_BIOSIG_SUBMODULE CACHE)
  unset(STF_USE_BIOSIG_SUBMODULE)

  if(_STF_USE_BIOSIG_SUBMODULE_VALUE)
    message(DEPRECATION "STF_USE_BIOSIG_SUBMODULE is deprecated; use STF_BIOSIG_PROVIDER=SUBMODULE instead")
    set(STF_BIOSIG_PROVIDER "SUBMODULE" CACHE STRING "BIOSIG provider (AUTO, SYSTEM, or SUBMODULE)" FORCE)
  else()
    message(DEPRECATION "STF_USE_BIOSIG_SUBMODULE is deprecated; use STF_BIOSIG_PROVIDER=SYSTEM instead")
    if(STF_BIOSIG_PROVIDER STREQUAL "AUTO")
      set(STF_BIOSIG_PROVIDER "SYSTEM" CACHE STRING "BIOSIG provider (AUTO, SYSTEM, or SUBMODULE)" FORCE)
    endif()
  endif()

  unset(_STF_USE_BIOSIG_SUBMODULE_VALUE)
endif()

unset(_STF_WITH_BIOSIGLITE_VALUE)
