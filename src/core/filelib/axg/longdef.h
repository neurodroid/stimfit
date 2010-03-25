#ifndef LONGDEF_H
#define LONGDEF_H

#include <limits.h>

#if ( __WORDSIZE == 64 ) || defined (__APPLE__)
    #define AXGLONG int
#else
    #define AXGLONG long
#endif

#endif
