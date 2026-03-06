include_guard(DIRECTORY)

add_subdirectory(libstfio)
add_subdirectory(libstfnum)

if(STF_WITH_BIOSIGLITE)
  add_subdirectory(libbiosiglite)
endif()

if(STF_BUILD_MODULE)
  add_subdirectory(pystfio)
else()
  add_subdirectory(stimfit)
  if(STF_ENABLE_PYTHON)
    add_subdirectory(stimfit/py)
  endif()
endif()

