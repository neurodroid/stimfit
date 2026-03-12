include_guard(GLOBAL)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions($<$<CONFIG:Debug>:_STFDEBUG>)

if(STF_ENABLE_PYTHON)
  add_compile_definitions(WITH_PYTHON)
endif()

if(STF_ENABLE_IPYTHON)
  add_compile_definitions(IPYTHON)
endif()

if(STF_ENABLE_PYTHON)
  if(STF_PY_SHELL_BACKEND STREQUAL "LEGACY")
    add_compile_definitions(STF_PY_SHELL_BACKEND_LEGACY)
  else()
    add_compile_definitions(STF_PY_SHELL_BACKEND_MODERN)
  endif()
endif()

if(STF_ENABLE_PSLOPE)
  add_compile_definitions(WITH_PSLOPE)
endif()

if(STF_ENABLE_AUI)
  add_compile_definitions(WITH_AUIDOCVIEW)
endif()

if(STF_BUILD_MODULE)
  add_compile_definitions(MODULE_ONLY)
endif()

if(STF_WITH_BIOSIG OR STF_WITH_BIOSIGLITE)
  add_compile_definitions(WITH_BIOSIG)
endif()

if(STF_WITH_BIOSIGLITE)
  add_compile_definitions(WITH_BIOSIGLITE)
endif()

add_compile_definitions(H5_USE_16_API)

if(HAVE_STRPTIME_H)
  add_compile_definitions(HAVE_STRPTIME_H=1)
endif()

add_compile_definitions(PACKAGE_VERSION=\"${PROJECT_VERSION}\")

add_library(stimfit_config INTERFACE)
target_include_directories(stimfit_config INTERFACE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_BINARY_DIR}
)

if(NOT MSVC)
  target_compile_options(stimfit_config INTERFACE
    $<$<COMPILE_LANGUAGE:C>:-fPIC>
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
  )
endif()

target_compile_options(stimfit_config INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4 /Zc:__cplusplus>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<CONFIG:Debug>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O0 -g3>
  $<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O2 -g>
)

if(STF_BUILD_DEBIAN)
  if(DEFINED ENV{CPPFLAGS})
    separate_arguments(_debian_cppflags NATIVE_COMMAND "$ENV{CPPFLAGS}")
    target_compile_options(stimfit_config INTERFACE ${_debian_cppflags})
  endif()
  if(DEFINED ENV{CFLAGS})
    separate_arguments(_debian_cflags NATIVE_COMMAND "$ENV{CFLAGS}")
    target_compile_options(stimfit_config INTERFACE ${_debian_cflags})
  endif()
  if(DEFINED ENV{CXXFLAGS})
    separate_arguments(_debian_cxxflags NATIVE_COMMAND "$ENV{CXXFLAGS}")
    target_compile_options(stimfit_config INTERFACE ${_debian_cxxflags})
  endif()
  if(DEFINED ENV{LDFLAGS})
    separate_arguments(_debian_ldflags NATIVE_COMMAND "$ENV{LDFLAGS}")
    target_link_options(stimfit_config INTERFACE ${_debian_ldflags})
  endif()
endif()
