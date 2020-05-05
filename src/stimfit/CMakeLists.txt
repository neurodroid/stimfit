cmake_minimum_required(VERSION 3.10)

find_package(FFTW REQUIRED)

if(APPLE)
    find_package(wxWidgets COMPONENTS net core base propgrid aui adv)
else()
    find_package(wxWidgets COMPONENTS net core base aui adv)
endif()

if(NOT wxWidgets_FOUND)
    message(WARNING "*************************************************"
                    "wxWidgets are not found. Disabling GUI."
                    "*************************************************")
    return()
endif()

# Include path for sip.h and wx_api.h if wx module is importable.
execute_process(COMMAND 
    ${Python_EXECUTABLE} -c 
        "import wx;import os;print(os.path.dirname(wx.__file__))"
    OUTPUT_VARIABLE PYWX_INSTALL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT PYWX_INSTALL_DIR)
    message(WARNING 
        "\n*************************************************"
        "\n${Python_EXECUTABLE} Could not import wx module. Disabling GUI."
        "\n*************************************************")
    return()
endif()

# libstimfit target.
set(SRCS
    ./stf.cpp 
    ./gui/app.cpp 
    ./gui/unopt.cpp 
    ./gui/doc.cpp 
    ./gui/copygrid.cpp 
    ./gui/graph.cpp 
    ./gui/printout.cpp 
    ./gui/parentframe.cpp
    ./gui/childframe.cpp 
    ./gui/view.cpp 
    ./gui/table.cpp 
    ./gui/zoom.cpp 
    ./gui/dlgs/convertdlg.cpp
    ./gui/dlgs/cursorsdlg.cpp 
    ./gui/dlgs/eventdlg.cpp 
    ./gui/dlgs/fitseldlg.cpp 
    ./gui/dlgs/smalldlgs.cpp 
    ./gui/usrdlg/usrdlg.cpp)

add_library(libstimfit SHARED ${SRCS})

set(WX_INCLUDE_DIR ${PYWX_INSTALL_DIR}/include)
set(WX_INCLUDE_DIRS ${WX_INCLUDE_DIR} ${WX_INCLUDE_DIR}/wxPython)
message(STATUS "wx include paths: ${WX_INCLUDE_DIRS}")
target_include_directories(libstimfit PRIVATE ${WX_INCLUDE_DIRS})
target_include_directories(libstimfit PRIVATE ${Python_INCLUDE_DIRS})
target_include_directories(libstimfit PRIVATE ${Numpy_INCLUDE_DIRS})


include(${wxWidgets_USE_FILE})

# find path for sip.h and wx_api.h .
execute_process(COMMAND 
    ${Python_EXECUTABLE} -c 
    "import wx;import os;print(os.path.dirname(wx.__file__))"
    OUTPUT_VARIABLE PYWX_INSTALL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT PYWX_INSTALL_DIR)
    message(STATUS "***********************************")
    message(STATUS "warn: `python3 -c 'import wx'` failed. GUI is disabled.")
    message(STATUS "Only python module will be built.")
    message(STATUS "***********************************")
    set(WITH_WX_GUI FALSE)
    return()
else()
    set(PYWX_INCLUDE_DIR ${PYWX_INSTALL_DIR}/include)
    set(PYWX_INCLUDE_DIRS ${PYWX_INCLUDE_DIR} ${PYWX_INCLUDE_DIR}/wxPython)
    message(STATUS "wx include paths: ${PYWX_INCLUDE_DIRS}")
    target_include_directories(libstimfit PRIVATE ${PYWX_INCLUDE_DIRS})
endif()

target_include_directories(libstimfit PRIVATE ${Python_INCLUDE_DIRS})
target_include_directories(libstimfit PRIVATE ${Numpy_INCLUDE_DIRS})

add_subdirectory(py)


target_include_directories(libstimfit PRIVATE ${wxWidgets_INCLUDE_DIRS})
target_include_directories(libstimfit PRIVATE ${FFTW_INCLUDE_DIRS})

set_target_properties(libstimfit PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CAMKE_BINARY_DIR}"
    LIBRARY_OUTPUT_NAME stimfit)

if(APPLE)
    set_target_properties(libstimfit 
        PROPERTIES 
        LINK_FLAGS "-undefined dynamic_lookup")
endif()

target_link_libraries(libstimfit PUBLIC ${wxWidgets_LIBRARIES})
target_link_libraries(libstimfit PUBLIC libstfnum)

# Executable
add_executable(stimfit.exe gui/main.cpp)
target_include_directories(stimfit.exe PRIVATE ${FFTW_INCLUDE_DIRS})
set_target_properties(stimfit.exe PROPERTIES OUTPUT_NAME stimfit)
target_link_libraries(stimfit.exe PUBLIC libstimfit libstfio libstfnum)

if(WITH_PYTHON)
    target_include_directories(stimfit.exe PRIVATE ${WX_INCLUDE_DIRS})
    target_include_directories(stimfit.exe PRIVATE ${Python_INCLUDE_DIRS} ${Numpy_INCLUDE_DIRS})
    target_link_libraries(stimfit.exe PUBLIC ${Python_LIBRARIES})
endif()

# install
install(TARGETS stimfit.exe DESTINATION bin)
install(TARGETS libstimfit DESTINATION lib)
