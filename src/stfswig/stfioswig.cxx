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

#include "./../core/recording.h"
