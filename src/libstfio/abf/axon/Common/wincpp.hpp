#ifndef STF_AXON_COMMON_WINCPP_HPP
#define STF_AXON_COMMON_WINCPP_HPP

//**********************************************************************************************
//
//    Copyright (c) 1993 Axon Instruments.
//    All rights reserved.
//
//**********************************************************************************************
// HEADER:  WINCPP.HPP
// PURPOSE: Contains common includes. Used for generation of precompiled headers.
// AUTHOR:  BHI  Nov 1993
//

#include "../Common/axodefn.h"
#include "../Common/axodebug.h"
/*
#include "../Common/colors.h"
*/
#include "../Common/adcdac.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <limits.h>
#include <memory.h>  
#include <string.h>
#include <math.h>

#if !defined(_MSC_VER)
#ifndef _TRUNCATE
#define _TRUNCATE ((size_t)-1)
#endif


#endif
static inline int strcpy_s(char *dest, size_t destsz, const char *src)
{
    if (!dest || destsz == 0)
        return 1;

    if (!src)
    {
        dest[0] = '\0';
        return 1;
    }

    strncpy(dest, src, destsz - 1);
    dest[destsz - 1] = '\0';
    return 0;
}

static inline int strncpy_s(char *dest, size_t destsz, const char *src, size_t count)
{
    if (!dest || destsz == 0)
        return 1;

    if (!src)
    {
        dest[0] = '\0';
        return 1;
    }

    size_t copy_len = count;
    if (count == _TRUNCATE || copy_len >= destsz)
        copy_len = destsz - 1;

    strncpy(dest, src, copy_len);
    dest[copy_len] = '\0';
    return 0;
}

static inline int strcat_s(char *dest, size_t destsz, const char *src)
{
    if (!dest || destsz == 0)
        return 1;

    size_t dest_len = strlen(dest);
    if (dest_len >= destsz)
    {
        dest[destsz - 1] = '\0';
        return 1;
    }

    return strncpy_s(dest + dest_len, destsz - dest_len, src, _TRUNCATE);
}
#endif

