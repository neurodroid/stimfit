cmake_minimum_required(VERSION 3.12)

if(OSX)
    find_package(wxWidgets REQUIRED COMPONENTS net core base propgrid aui adv)
else()
    find_package(wxWidgets REQUIRED COMPONENTS net core base aui adv)
endif()
include(${wxWidgets_USE_FILE})

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
    ./gui/usrdlg/usrdlg.cpp
    )

add_library(stimfit_static STATIC ${SRCS})

if(WITH_PYTHON)
    # Include path for sip.h and wx_api.h 
    execute_process(COMMAND 
        ${Python_EXECUTABLE} -c 
            "import wx;import os;print(os.path.dirname(wx.__file__))"
        OUTPUT_VARIABLE PYWX_INSTALL_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(WX_INCLUDE_DIR ${PYWX_INSTALL_DIR}/include)
    set(WX_INCLUDE_DIRS ${WX_INCLUDE_DIR} ${WX_INCLUDE_DIR}/wxPython)
    message(STATUS "wx include paths: ${WX_INCLUDE_DIRS}")
    target_include_directories(stimfit_static PRIVATE ${WX_INCLUDE_DIRS})
    target_include_directories(stimfit_static PRIVATE ${Python_INCLUDE_DIRS})
    target_include_directories(stimfit_static PRIVATE ${Numpy_INCLUDE_DIRS})
endif()

if(WITH_PYTHON)
    add_subdirectory(py)
endif()

target_include_directories(stimfit_static PRIVATE ${wxWidgets_INCLUDE_DIRS})

target_link_libraries(stimfit_static PUBLIC ${wxWidgets_LIBRARIES})
target_link_libraries(stimfit_static PRIVATE stfnum_static)

# binary.
add_executable(stimfit.exe gui/main.cpp)
set_target_properties(stimfit.exe PROPERTIES OUTPUT_NAME "stimfit")
target_link_libraries(stimfit.exe PRIVATE stimfit_static stfio_static stfnum_static)
if(WITH_PYTHON)
    target_include_directories(stimfit.exe PRIVATE ${WX_INCLUDE_DIRS})
    target_include_directories(stimfit.exe PRIVATE ${Python_INCLUDE_DIRS} 
        ${Numpy_INCLUDE_DIRS})
    target_link_libraries(stimfit.exe PUBLIC ${Python_LIBRARIES})
endif()

# install stimfit
install(TARGETS stimfit.exe DESTINATION bin)
