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
    ./gui/usrdlg/usrdlg.cpp)

add_library(stimfit STATIC ${SRCS})

if(WITH_PYTHON)
    add_subdirectory(py)
endif()

target_link_libraries(stimfit PRIVATE stfnum)

# binary.
add_executable(stimfit.exe gui/main.cpp)
target_link_libraries(stimfit.exe PRIVATE stimfit stfio stfnum)
target_include_directories(stimfit PRIVATE ${wxWidgets_INCLUDE_DIRS})
target_link_libraries(stimfit PUBLIC ${wxWidgets_LIBRARIES})

