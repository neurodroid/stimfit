/***************************************************************************** 
**
** cfs.h
**                                                               78 cols --->*
** Header file for MSC version of CFS functions.
** Definitions of the structures and routines for the CFS filing             *
** system. This is the include file for standard use, all access             *
** is by means of functions - no access to the internal CFS data. The file   *
** machine.h provides some common definitions across separate platforms,     *
** note that the MS variants are designed to accomodate medium model - far   *
** pointers are explicitly used where necessary.
**
** CFSAPI Don't declare this to give a pascal type on the Mac, there is a MPW
**        compiler bug that corrupts floats passed to pascal functions!!!!!!
**
*/

#ifndef __CFS__
#define __CFS__

#include "machine.h"

#if 0 //def macintosh                /* define CFSCONVERT in here if you want it */
    // #include <types.h>
    // #include <files.h>
    // #include <errors.h>
    #define  USEHANDLES
    #define  CFSAPI(type) type
    #undef   LLIO                   /* LLIO is not used for Mac             */
#endif                              /* End of the Mac stuff, now for DOS    */

#ifdef _IS_MSDOS_
    #define  qDebug 0               /* only used to debug Mac stuff         */
    #undef   USEHANDLES
    #include <malloc.h>
    #include <dos.h>
    #include <io.h>                         /* MSC I/O function definitions */
    #include <fcntl.h>
    #include <errno.h>
    #define  LLIO                   /* We can use LLIO for MSC/DOS          */
    #define  CFSAPI(type) type _pascal
#endif

#ifdef _IS_WINDOWS_
    #include <io.h>                         /* MSC I/O function definitions */
    #include <fcntl.h>
    #define  qDebug 0               /* only used to debug Mac stuff         */
    #define  CFSAPI(type) type WINAPI
    #ifdef WIN32
      #undef   LLIO
      #undef   USEHANDLES
    #else
      #define  LLIO                     /* We can use LLIO for MSC/Windows  */
      #define  USEHANDLES               /* use handles under 16-bit windows */
    #endif
    #ifdef _MSC_VER
    #pragma warning( disable : 4251 )  // Disable warning messages
    #pragma warning( disable : 4996 )  // Disable warning messages
    #endif
#endif

#if defined(__LINUX__) || defined(__WXMAC__)
    #define  qDebug 0               /* only used to debug Mac stuff         */
    #ifdef __LINUX__
        #include <malloc.h>
    #endif
    #include <stdio.h>                         /* MSC I/O function definitions */
    #include <fcntl.h>
    #include <errno.h>
    #define  CFSAPI(type) type
#endif

#define FILEVAR 0              /* Constants to indicate whether variable is */
#define DSVAR   1                         /* file or data section variable. */
#define INT1    0                            /* DATA VARIABLE STORAGE TYPES */
#define WRD1    1
#define INT2    2
#define WRD2    3
#define INT4    4
#define RL4     5
#define RL8     6
#define LSTR    7
#define SUBSIDIARY 2                            /* Chan Data Storage types */
#define MATRIX  1
#define EQUALSPACED 0                            /* Chan Data Storage types */

#define noFlags 0                                 /* Declare a default flag */

/* Definitions of bits for DS flags */

#define FLAG7   1
#define FLAG6   2
#define FLAG5   4
#define FLAG4   8
#define FLAG3   16
#define FLAG2   32
#define FLAG1   64
#define FLAG0   128
#define FLAG15  256
#define FLAG14  512
#define FLAG13  1024
#define FLAG12  2048
#define FLAG11  4096
#define FLAG10  8192
#define FLAG9   16384
#define FLAG8   32768


/* define numbers of characters in various string types */

#define DESCCHARS    20
#define FNAMECHARS   12
#define COMMENTCHARS 72
#define UNITCHARS    8

/*character arrays used in data structure */

typedef char  TDataType;
typedef char  TCFSKind;
typedef char  TDesc[DESCCHARS+2];        /* Names in descriptions, 20 chars */
typedef char  TFileName[FNAMECHARS+2];              /* File names, 12 chars */
typedef char  TComment[COMMENTCHARS+2];            /* Comment, 72 chars max */
typedef char  TUnits[UNITCHARS+2];                    /* For units, 8 chars */

/* other types for users benefit */

typedef WORD TSFlags;

/*  for data and data section variables */

typedef struct
{
   TDesc     varDesc;                      /* users description of variable */
   TDataType vType;                               /* one of 8 types allowed */
   char      zeroByte;                       /* for MS Pascal compatibility */
   TUnits    varUnits;                              /* users name for units */
   short     vSize;  /* for type lstr gives no. of chars +1 for length byte */
} TVarDesc;

#if defined(__linux__) || defined(__WXMAC__)
typedef char            * TpStr;
typedef const char      * TpCStr;
typedef short           * TpShort;
typedef float           * TpFloat;
typedef long            * TpLong;
typedef void            * TpVoid;
typedef TSFlags         * TpFlags;
typedef TDataType       * TpDType;
typedef TCFSKind        * TpDKind;
typedef TVarDesc        * TpVDesc;
typedef const TVarDesc  * TpCVDesc;
typedef THandle         * TpHandle;
typedef signed char     * TpSStr;
typedef WORD            * TpUShort;

#else
typedef char           FAR * TpStr;
typedef const char     FAR * TpCStr;
typedef short          FAR * TpShort;
typedef float          FAR * TpFloat;
typedef long           FAR * TpLong;
typedef void           FAR * TpVoid;
typedef TSFlags         FAR * TpFlags;
typedef TDataType      FAR * TpDType;
typedef TCFSKind       FAR * TpDKind;
typedef TVarDesc       FAR * TpVDesc;
typedef const TVarDesc FAR * TpCVDesc;
typedef THandle        FAR * TpHandle;
typedef signed char    FAR * TpSStr;
typedef WORD           FAR * TpUShort;
#endif

#if 0 //def macintosh
    typedef int     fDef;        /* file handle means something else on Mac */
#else
  #ifdef WIN32
    typedef HANDLE  fDef;                              /* WIN32 file handle */
  #else
    #ifdef LLIO
      typedef short fDef;                                    /* file handle */
    #else
      typedef FILE* fDef;                              /* stream identifier */
    #endif
  #endif
#endif


#ifdef __cplusplus
extern "C" {
#endif

/*
** Now definitions of the functions defined in the code
*/

CFSAPI(short) CreateCFSFile(TpCStr   fname,
                            TpCStr   comment,
                            WORD     blocksize,
                            short    channels,
                            TpCVDesc fileArray,
                            TpCVDesc DSArray,
                            short    fileVars,
                            short    DSVars);

CFSAPI(void)  SetFileChan(short     handle,
                          short     channel,
                          TpCStr    channelName,
                          TpCStr    yUnits,
                          TpCStr    xUnits,
                          TDataType dataType,
                          TCFSKind  dataKind,
                          short     spacing,
                          short     other);

CFSAPI(void)  SetDSChan(short handle,
                        short channel,
                        WORD  dataSection,
                        long  startOffset,
                        long  points,
                        float yScale,
                        float yOffset,
                        float xScale,
                        float xOffset);

CFSAPI(short) WriteData(short  handle,
                        WORD   dataSection,
                        long   startOffset,
                        WORD   bytes,
                        TpVoid dataADS);

CFSAPI(short) ClearDS(short   handle);

CFSAPI(void)  SetWriteData(short  handle,
                           long   startOffset,
                           long   bytes);

CFSAPI(long)  CFSFileSize(short  handle);

CFSAPI(short) InsertDS(short   handle,
                       WORD    dataSection,
                       TSFlags flagSet);

CFSAPI(short) AppendDS(short   handle,
                       long    lSize,
                       TSFlags flagSet);

CFSAPI(void)  RemoveDS(short  handle,
                       WORD   dataSection);

CFSAPI(void)  SetComment(short  handle,
                         TpCStr comment);

CFSAPI(void)  SetVarVal(short  handle,
                        short  varNo,
                        short  varKind,
                        WORD   dataSection,
                        TpVoid varADS);

CFSAPI(short) CloseCFSFile(short  handle);


CFSAPI(short) OpenCFSFile(TpCStr  fname,
                          short   enableWrite,
                          short   memoryTable);

CFSAPI(void) GetGenInfo(short   handle,
                        TpStr   time,
                        TpStr   date,
                        TpStr   comment);

CFSAPI(void) GetFileInfo(short    handle,
                         TpShort  channels,
                         TpShort  fileVars,
                         TpShort  DSVars,
                         TpUShort dataSections);

CFSAPI(void) GetVarDesc(short   handle,
                        short   varNo,
                        short   varKind,
                        TpShort varSize,
                        TpDType varType,
                        TpStr   units,
                        TpStr   description);

CFSAPI(void) GetVarVal(short  handle,
                       short  varNo,
                       short  varKind,
                       WORD   dataSection,
                       TpVoid varADS);

CFSAPI(void) GetFileChan(short   handle, 
                         short   channel,
                         TpStr   channelName,
                         TpStr   yUnits,
                         TpStr   xUnits,
                         TpDType dataType,
                         TpDKind dataKind,
                         TpShort spacing,
                         TpShort other);

CFSAPI(void) GetDSChan(short   handle,
                       short   channel,
                       WORD    dataSection,
                       TpLong  startOffset,
                       TpLong  points,
                       TpFloat yScale,
                       TpFloat yOffset,
                       TpFloat xScale,
                       TpFloat xOffset);

CFSAPI(WORD) GetChanData(short  handle,
                         short  channel,
                         WORD   dataSection,
                         long   firstElement,
                         WORD   numberElements,
                         TpVoid dataADS,
                         long   areaSize);

CFSAPI(long) GetDSSize(short  handle,
                       WORD   dataSection);

CFSAPI(short) ReadData(short  handle,
                       WORD   dataSection,
                       long   startOffest,
                       WORD   bytes,
                       TpVoid dataADS);

CFSAPI(WORD) DSFlagValue(int   nflag);

CFSAPI(void) DSFlags(short   handle,
                     WORD    dataSection,
                     short   setIt,
                     TpFlags pflagSet);

CFSAPI(short) FileError(TpShort handleNo,
                        TpShort procNo,
                        TpShort errNo);

CFSAPI(short) CommitCFSFile(short handle);

#ifdef __cplusplus
}
#endif

#endif /* __CFS__ */
