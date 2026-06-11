include_guard(DIRECTORY)

add_library(
  stfnum
  fit.cpp
  levmar/lm.c
  levmar/Axb.c
  levmar/misc.c
  levmar/lmlec.c
  levmar/lmbc.c
  funclib.cpp
  stfnum.cpp
  measure.cpp
)
stf_apply_macos_runtime_policy(stfnum)

if(MSVC)
  target_compile_definitions(stfnum PRIVATE STFNUMDLL)
endif()

target_link_libraries(
  stfnum
  PUBLIC
    stimfit_config
    stfio
    stimfit::fftw3
    stimfit::lapack
)

set_target_properties(stfnum PROPERTIES OUTPUT_NAME stfnum)

if(APPLE AND STF_MACOS_APP_BUNDLE)
  set(_stf_stfnum_install_libdir "stimfit.app/Contents/lib/stimfit")
else()
  set(_stf_stfnum_install_libdir "${CMAKE_INSTALL_LIBDIR}/stimfit")
endif()

if(APPLE AND STF_MACOS_APP_BUNDLE)
  install(TARGETS stfnum
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${STF_COMPONENT_CORE}
    LIBRARY DESTINATION ${_stf_stfnum_install_libdir} COMPONENT ${STF_COMPONENT_CORE}
  )
else()
  install(TARGETS stfnum
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${STF_COMPONENT_CORE}
    LIBRARY DESTINATION ${_stf_stfnum_install_libdir} COMPONENT ${STF_COMPONENT_CORE}
    ARCHIVE DESTINATION ${_stf_stfnum_install_libdir} COMPONENT ${STF_COMPONENT_CORE}
  )
endif()
