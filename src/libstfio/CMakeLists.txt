include_guard(DIRECTORY)

set(STF_STFIO_SOURCES
  channel.cpp
  section.cpp
  recording.cpp
  stfio.cpp
  annotation.cpp
  cfs/cfslib.cpp
  cfs/cfs.c
  hdf5/hdf5lib.cpp
  abf/abflib.cpp
  abf/axon/AxAbfFio32/abffiles.cpp
  abf/axon/AxAbfFio32/csynch.cpp
  abf/axon/AxAbfFio32/filedesc.cpp
  abf/axon/Common/FileReadCache.cpp
  abf/axon/Common/FileIO.cpp
  abf/axon/AxAbfFio32/abfheadr.cpp
  abf/axon/AxAbfFio32/abfhwave.cpp
  abf/axon/AxAbfFio32/Oldheadr.cpp
  abf/axon/AxAbfFio32/abfutil.cpp
  abf/axon/AxAbfFio32/msbincvt.cpp
  abf/axon/Common/unix.cpp
  abf/axon/AxAbfFio32/abferror.cpp
  abf/axon/AxAtfFio32/axatffio32.cpp
  abf/axon/AxAtfFio32/fileio2.cpp
  abf/axon2/ProtocolReaderABF2.cpp
  abf/axon2/SimpleStringCache.cpp
  abf/axon2/abf2headr.cpp
  atf/atflib.cpp
  igor/igorlib.cpp
  igor/CrossPlatformFileIO.c
  igor/WriteWave.c
  intan/common.cpp
  intan/intanlib.cpp
  intan/streams.cpp
)

if(NOT STF_WITH_BIOSIG)
  list(APPEND STF_STFIO_SOURCES
    axg/axglib.cpp
    axg/AxoGraph_ReadWrite.cpp
    axg/fileUtils.cpp
    axg/stringUtils.cpp
    axg/byteswap.cpp
    heka/hekalib.cpp
  )
endif()

if(STF_WITH_BIOSIG)
  list(APPEND STF_STFIO_SOURCES biosig/biosiglib.cpp)
endif()

add_library(stfio ${STF_STFIO_SOURCES})
stf_apply_macos_runtime_policy(stfio)

if(MSVC)
  target_compile_definitions(stfio PRIVATE STFIODLL)
  if(BUILD_SHARED_LIBS)
    set_target_properties(stfio PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()
endif()

target_link_libraries(stfio PUBLIC stimfit_config stimfit::hdf5)

if(TARGET stimfit::biosig)
  target_link_libraries(stfio PUBLIC stimfit::biosig)
endif()

set_target_properties(stfio PROPERTIES OUTPUT_NAME stfio)

if(APPLE AND STF_MACOS_APP_BUNDLE)
  set(_stf_stfio_install_libdir "stimfit.app/Contents/lib/stimfit")
else()
  set(_stf_stfio_install_libdir "${CMAKE_INSTALL_LIBDIR}/stimfit")
endif()

install(TARGETS stfio
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${STF_COMPONENT_CORE}
  LIBRARY DESTINATION ${_stf_stfio_install_libdir} COMPONENT ${STF_COMPONENT_CORE}
  ARCHIVE DESTINATION ${_stf_stfio_install_libdir} COMPONENT ${STF_COMPONENT_CORE}
)
