include_guard(GLOBAL)

option(STF_BUILD_MODULE "Build standalone Python module (like --enable-module)" OFF)
option(STF_ENABLE_PYTHON "Enable Python integration (like --enable-python)" ON)
option(STF_ENABLE_IPYTHON "Enable IPython shell integration (like --enable-ipython)" OFF)
option(STF_ENABLE_PSLOPE "Enable slope cursor measurements (like --enable-pslope)" OFF)
option(STF_ENABLE_AUI "Enable experimental AUI doc/view mode (like --enable-aui)" OFF)
option(STF_BUILD_DEBIAN "Enable Debian-oriented build flags/paths (like --enable-debian)" OFF)

option(STF_WITH_BIOSIG "Use external libbiosig if available" ON)
option(STF_WITH_BIOSIGLITE "Use bundled biosiglite implementation" OFF)

set(STF_HDF5_PREFIX "" CACHE PATH "Optional HDF5 installation prefix (like --with-hdf5-prefix)")

if(STF_BUILD_MODULE)
  set(STF_ENABLE_PYTHON ON CACHE BOOL "Enable Python integration (like --enable-python)" FORCE)
endif()

if(STF_WITH_BIOSIGLITE)
  set(STF_WITH_BIOSIG OFF CACHE BOOL "Use external libbiosig if available" FORCE)
endif()

