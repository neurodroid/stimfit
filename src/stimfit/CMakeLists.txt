cmake_minimum_required(VERSION 3.12)

if(OSX)
    find_package(wxWidgets REQUIRED COMPONENTS net core base propgrid aui adv)
else()
    find_package(wxWidgets REQUIRED COMPONENTS net core base aui adv)
endif()

add_library(stimfit STATIC
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

target_link_libraries(stimfit PRIVATE stfnum)

# binary.
add_executable(stimfit.exe gui/main.cpp)
target_link_libraries(stimfit.exe PRIVATE stimfit stfio stfnum)

if(wxWidgets_FOUND)
    include(${wxWidgets_USE_FILE})
    target_include_directories(stimfit PRIVATE ${wxWidgets_INCLUDE_DIRS})
    target_link_libraries(stimfit PUBLIC ${wxWidgets_LIBRARIES})
endif()

