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

target_link_libraries(
  stfnum
  PUBLIC
    stimfit_config
    stimfit::fftw3
    stimfit::lapack
)

set_target_properties(stfnum PROPERTIES OUTPUT_NAME stfnum)

install(TARGETS stfnum
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/stimfit
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/stimfit
)

