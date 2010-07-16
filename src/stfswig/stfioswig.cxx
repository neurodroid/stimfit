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
#include "./../core/core.h"
#include "./../core/recording.h"

bool _open( const char* filename ) {
  wxString fName(filename);
  Recording Data;
  stf::txtImportSettings tis;
  return stf::importFile(fName, stf::cfs, Data, tis, false);
}
