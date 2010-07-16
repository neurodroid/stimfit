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

bool _read(const std::string& filename, const std::string& ftype, Recording& Data) {
    wxString fName(filename);
    stf::txtImportSettings tis;
    stf::filetype stftype = stf::none;
    if (ftype == "cfs") {
        stftype = stf::cfs;
    } else if (ftype == "hdf5") {
        stftype = stf::hdf5;
    } else if (ftype == "abf") {
        stftype = stf::abf;
    } else {
        stftype = stf::none;
    }
         
    if (!stf::importFile(fName, stftype, Data, tis, false)) {
        std::cerr << "Error importing file\n";
        return false;
    }
    return true;
}
