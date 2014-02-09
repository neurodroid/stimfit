/*****************************************************************************
**
** machine.h
**
** Copyright (c) Cambridge Electronic Design Limited 1991,1992
**
** This file is included at the start of 'C' or 'C++' source file to define
** things needed to make Macintosh DOS and Windows sources more compatible
**
** Revision History
**
** 10/Jun/91  PNC   First Version
**  3/Mar/92  TDB   Added support for non-Windows DOS. Now expects to be 
**                  included after windows.h.
** 23/Jun/92  GPS   Tidied up. SONAPI definitions moved to son.h
** 27/Jul/92  GPS   Made routines that need to be far in MSDOS as F_xxxxxx
**                  and mapped this to actual name. Also added LPSTR and
**                  DWORD definitions.
** 24/Feb/93  PNC   Added new defines _IS_MSDOS_ for actual msdos (not for
**                  windows) and _IS_WINDOWS_ for windows (16 or 32 bit)
**
** 14/Jun/93  KJ    Made few changes, enabling it to be used by CFS for DOS,
**                  Windows and Macintosh.
**
** 01/Oct/93  PNC   Defined _near for NT 32 bit compile as this is an invalid
**                  keyword under this compiler.
**
** 17/Dec/93  PNC   DllExport and DllImport added to enable classes to be 
**                  exported from dlls more easily in NT 32 bit linker.
**                  Also coord changed to long for 16 bit windows operation.
**
** 20/Jan/94  PNC   Added support for Borland C++ not tested by us, but
**                  tested by a customer.
**
** 25/Jan/94  MAE   Defines added for DOS that should have been there 
**                  previously and F_memmove.
**
** 27/Oct/94  MAE   Added FDBL_MAX defines.
**
** 02/Feb/95  GPS   WINNT changed to WIN32 to accomodate Windows 95.
**
** 08/Jun/95  TDB   _fstrrchr defined as strrchr for WIN32 builds.
**
** 03/May/96  KJ    LPCSTR defined as const char* for Mac builds.
**
** 29/Jul/97  TDB   Tweaked for use with Borland C++ Builder
**
** 03/Dec/98  TDB   Added F_malloc and F_free for Windows purposes
**
*****************************************************************************/

/*
** Borland C++ Builder notes:
**
** This compiler defines both __MSDOS__ and __WIN32__, I have mapped
** __WIN32__ to WIN32 so these defines are used, which work fine.
*/

/*****************************************************************************
012345678901234567890123456789012345678901234567890123456789012345678901234567
*****************************************************************************/

#ifndef __MACHINE__
    #define __MACHINE__

    #include <limits.h>
    #if ( __WORDSIZE == 64 ) || defined (__APPLE__)
        #define CFSLONG int
    #else
        #define CFSLONG long
    #endif

    #include <sys/types.h>    /* Needed for various types               */
    #include <sys/stat.h>                            /*    ditto        */
    
    #include <float.h>        /* for LDBL_DIG                           */

    #if defined(WIN32) || defined (_MSC_VER)
                          /* if its windows define our windows symbol   */
       #define _IS_WINDOWS_   /* WIN32 is defined for 32-bit at moment    */
       #undef _IS_MSDOS_      /* and we arent doing msdos after all         */
    #endif

    #ifdef _INC_WINDOWS       /* the alternative windows symbolic defn      */
       #ifndef _IS_WINDOWS_   /* as above but _INC_WINDOWS is for 16 bit    */
          #define _IS_WINDOWS_
       #endif
       #undef _IS_MSDOS_      /* and we arent doing msdos after all         */
    #endif

    #ifndef TRUE
       #define TRUE 1
       #define FALSE 0
    #endif


    #if defined(_IS_WINDOWS_) && !defined(__MINGW32__)  /* Now set up for windows use */
       #include <Windows.h>
       #define F_memcpy memcpy    /* Define model-independent routines      */
       #define F_memmove memmove
       #define F_strlen strlen
       #define F_strcat strcat
       #define F_strcpy strcpy
       #define F_strcmp strcmp
       #define F_strncat strncat
       #define F_strncpy strncpy
       #define F_strncmp strncmp
       #define F_strchr strchr
       #define _fstrrchr strrchr 
       #define _near              /* stop compiler errors for 32 bit compile*/
       #define DllExport __declspec(dllexport)
       #define DllImport __declspec(dllimport)

       typedef CFSLONG Coord;        /* this is LONG in the MacApp definitions */
       typedef double fdouble;
       #define FDBL_DIG DBL_DIG
       #define FDBL_MAX DBL_MAX
       typedef HGLOBAL THandle;

       #define F_malloc         malloc
       #define F_free           free

       #define M_AllocMem(x)     GlobalAlloc(GMEM_MOVEABLE,x)
       #define M_AllocClear(x)   GlobalAlloc(GMEM_MOVEABLE|GMEM_ZEROINIT,x)
       #define M_FreeMem(x)      GlobalFree(x)
       #define M_LockMem(x)      GlobalLock(x)
       #define M_MoveLockMem(x)  GlobalLock(x)
       #define M_UnlockMem(x)    (GlobalUnlock(x)==0)
       #define M_NewMemSize(x,y) (x = GlobalReAlloc(x,y,GMEM_MOVEABLE))
       #define M_GetMemSize(x)   GlobalSize(x)
    #else
       #define F_memcpy memcpy
       #define F_memmove memmove
       #define F_strlen strlen
       #define F_strcat strcat
       #define F_strcpy strcpy
       #define F_strcmp strcmp
       #define F_strncat strncat
       #define F_strncpy strncpy
       #define F_strncmp strncmp
       #define F_strchr strchr
       #define FAR
       #define PASCAL
       #define _far
       #define _near
       #define DllExport
       #define DllImport

       #define FDBL_DIG LDBL_DIG
       #define FDBL_MAX LDBL_MAX
       typedef char * LPSTR;
       typedef const char * LPCSTR;
       typedef unsigned short WORD;
//        typedef unsigned CFSLONG  DWORD;
       typedef unsigned char  BYTE;
       typedef long double fdouble;
       typedef CFSLONG Coord;     /*  Borrowed from MacApp */
       typedef WORD THandle;
       #define F_malloc         malloc
       #define F_free           free
       #define M_AllocMem(x)     NewHandle(x)
       #define M_AllocClear(x)   NewHandleClear(x)
       #define M_FreeMem(x)      DisposHandle(x)
       #define M_LockMem(x)      (HLock(x),*x)
       #define M_MoveLockMem(x)  (HLockHi(x),*x)
       #define M_UnlockMem(x)    (HUnlock(x),TRUE)
       #define M_NewMemSize(x,y) (SetHandleSize(x,y),MemError() == 0)
       #define M_GetMemSize(x)   GetHandleSize(x)
    #endif /*UNIX*/	
#endif /* not defined __MACHINE__ */
