#include <stdlib.h>
#include <math.h>

#ifdef _WINDOWS
    #ifdef _DEBUG
        #undef _DEBUG
        #define _UNDEBUG
    #endif
#endif

#include <Python.h>

#ifdef _WINDOWS
    #ifdef _UNDEBUG
        #define _DEBUG
    #endif
#endif

#include "stfioswig.h"

#include "./../core/stimdefs.h"
#include "./../core/recording.h"

bool _open( const char* filename ) {

    bool res = stf::importFile(filename,
        const wxString& fName,
        stf::filetype type,
        Recording& ReturnData,
        const stf::txtImportSettings& txtImport,
        bool progress
) {
    
    return wxGetApp().OpenFilePy( wxFilename );
}
