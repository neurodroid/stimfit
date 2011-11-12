#include <stdlib.h>
#include <math.h>

#ifdef _WINDOWS
    #ifdef _DEBUG
        #undef _DEBUG
        #define _UNDEBUG
    #endif
#endif

#ifdef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE_WAS_DEF
#undef _POSIX_C_SOURCE
#endif
#ifdef _XOPEN_SOURCE
#define _XOPEN_SOURCE_WAS_DEF
#undef _XOPEN_SOURCE
#endif
#include <Python.h>
#ifdef _POSIX_C_SOURCE_WAS_DEF
  #ifndef _POSIX_C_SOURCE
    #define _POSIX_C_SOURCE
  #endif
#endif
#ifdef _XOPEN_SOURCE_WAS_DEF
  #ifndef _XOPEN_SOURCE
    #define _XOPEN_SOURCE
  #endif
#endif

#include <iostream>
#include <iomanip>

#ifdef _WINDOWS
    #ifdef _UNDEBUG
        #define _DEBUG
    #endif
#endif

#include "pystfio.h"

stfio::filetype gettype(const std::string& ftype) {
    stfio::filetype stftype = stfio::none;
    if (ftype == "cfs") {
        stftype = stfio::cfs;
    } else if (ftype == "hdf5") {
        stftype = stfio::hdf5;
    } else if (ftype == "abf") {
        stftype = stfio::abf;
    } else if (ftype == "atf") {
        stftype = stfio::atf;
    } else if (ftype == "axg") {
        stftype = stfio::axg;
    } else if (ftype == "heka") {
        stftype = stfio::heka;
    } else if (ftype == "igor") {
        stftype = stfio::igor;
    } else {
        stftype = stfio::none;
    }
    return stftype;
}

bool _read(const std::string& filename, const std::string& ftype, bool verbose, Recording& Data) {

    stfio::filetype stftype = gettype(ftype);
    stfio::txtImportSettings tis;
    StdoutProgressInfo progDlg("File import", "Starting file import", 100, verbose);
    
    try {
        if (!stfio::importFile(filename, stftype, Data, tis, progDlg)) {
            std::cerr << "Error importing file\n";
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error importing file:\n"
                  << e.what() << std::endl;
        return false;
    }
        
    return true;
}
