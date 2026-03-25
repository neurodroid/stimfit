include_guard(DIRECTORY)

if(STF_WITH_BIOSIG AND STF_BIOSIG_USE_SUBMODULE)
  add_subdirectory(libbiosig)
  add_library(stimfit::biosig ALIAS stimfit_biosig)
endif()

add_subdirectory(libstfio)
add_subdirectory(libstfnum)

if(STF_BUILD_MODULE)
  add_subdirectory(pystfio)
else()
  add_subdirectory(stimfit)
  if(STF_ENABLE_PYTHON)
    add_subdirectory(stimfit/py)
  endif()
endif()
