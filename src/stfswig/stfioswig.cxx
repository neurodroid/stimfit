#include <stdlib.h>
#include <math.h>

#ifdef _WINDOWS
    #ifdef _DEBUG
        #undef _DEBUG
        #define _UNDEBUG
    #endif
#endif

#include <Python.h>

#include <iostream>

#ifdef _WINDOWS
    #ifdef _UNDEBUG
        #define _DEBUG
    #endif
#endif

#include "stfioswig.h"

#include "./../core/stimdefs.h"
#include "./../core/core.h"

stf::filetype gettype(const std::string& ftype) {
    stf::filetype stftype = stf::none;
    if (ftype == "cfs") {
        stftype = stf::cfs;
    } else if (ftype == "hdf5") {
        stftype = stf::hdf5;
    } else if (ftype == "abf") {
        stftype = stf::abf;
    } else if (ftype == "atf") {
        stftype = stf::atf;
    } else if (ftype == "axg") {
        stftype = stf::axg;
    } else if (ftype == "heka") {
        stftype = stf::heka;
    } else {
        stftype = stf::none;
    }
    return stftype;
}

bool _read(const std::string& filename, const std::string& ftype, Recording& Data) {

    stf::filetype stftype = gettype(ftype);
    stf::txtImportSettings tis;
    try {
        if (!stf::importFile(filename, stftype, Data, tis, true)) {
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
