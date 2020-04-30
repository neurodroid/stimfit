cmake_minimum_required(VERSION 3.12)

find_package(HDF5 COMPONENTS CXX HL REQUIRED)
find_package(LAPACK REQUIRED)

add_library(stfio STATIC
    channel.cpp 
    section.cpp 
    recording.cpp 
    stfio.cpp 
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
    axg/axglib.cpp 
    axg/AxoGraph_ReadWrite.cpp 
    axg/fileUtils.cpp 
    axg/stringUtils.cpp 
    axg/byteswap.cpp 
    heka/hekalib.cpp 
    igor/igorlib.cpp 
    igor/CrossPlatformFileIO.c 
    igor/WriteWave.c 
    intan/common.cpp 
    intan/intanlib.cpp 
    intan/streams.cpp)

target_include_directories(stfio PRIVATE ${HDF5_CXX_INCLUDE_DIRS})
target_include_directories(stfio PRIVATE ${HDF5_HL_INCLUDE_DIRS})
target_link_libraries(stfio PUBLIC ${HDF5_CXX_LIBRARIES})
target_link_libraries(stfio PUBLIC ${HDF5_HL_LIBRARIES})
target_link_libraries(stfio PUBLIC ${LAPACK_LIBRARIES})
