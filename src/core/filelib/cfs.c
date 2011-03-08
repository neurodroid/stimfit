/*****************************************************************************
**
** MSC version of CFS functions
**
** Copyright (c) Cambridge Electronic Design Limited 1988 .. 1999
**
** Started by J.Grayer 13th Oct 1990
**
** V 2.11  Maximum file size upped to c. 1 GByte   TDB   12-Jun-91
**
** V 2.12  Changes made so data pointers changed to _far so that the
**         user may use any model and still put large data items outside
**         the default data segment.
**
** V 2.13  JG 17th Jan 1992
**         Re-jigged various things so all non_ANSI functions now appear in
**         msio.h/c.
**         Bug corrected in which table pointer did not get reset to NULL
**         when a file was opened with the pointer table not in memory.
**         Change made to SetComment so it can be used with a file opened in
**         edit mode.
**         Re-did the flag usage.
**         When file opened in edit mode and nothing changed file not updated.
**         Bug in recoverTable fixed.
**
** V 2.14  JG 8th Jun 1992
**         Small change to WriteData so it doesnt truncate the file if number
**         of bytes to write is given as zero.
**         (12th June v.minor change: added return statemnt to TransferTable
**         function.)
**
** V 2.15  JG 24th Jun 1992
**         Change to GetChanData so you can read from the current data section
**         provided you have called SetDSChan.
**
** V 2.16  TDB 24th Jul 1992
**         Changes to GetChanData (above) corrected. GetDSChan and GetHeader
**         adjusted to work online. Began work on tidying code
**
** V 2.17  JCN 24-Aug-92
**         Added routine ClearDS to abandon current DS by setting dataSz back
**           to 0 and resetting fileSz to dataSt. On closing the file no
**           InsertDs will then be called and file will not include the empty
**           data set.
**         Commented out the calls to InternalError in GetMemTable.
**
** V 2.18  TDB 16-Sep-92
**         Removed InternalError calls from GetMemTable, fixed bug in
**         GetMemTable, brought up to v 2.18.
**
** V 2.20  KJ  6-Jun-93, Version for DOS, WINDOWS and MACINTOSH.
**
**         MAE - for macintosh definitions of OpenCFSFile & CreateCFSFile
**              would be good idea to use FSSpec instead of 3 separate params
**              for name, vRefNum & dirID.
**              Also note that changing the type TSFlags from bit fields to
**              a WORD has swapped the 2 flag bytes over on the macintosh,
**              but not on the PC (Horace Townsends convertor programme no
**              longer works.)
**              May want to use FindFolder to get the temporary folder, and
**              create the temporary files in that.
**
** V 2.20  KJ 14-Jul-93, 
**            Tidied up errors handling calls in OpenCFSFile, CloseCFSFile
**            and TransferTable routines. 
**            
** V 2.30  KJ 01-Sep-93, 
**            CommitCFSFile function added.
**            This is used to write the head and data sections info
**            to to the disk file and to commit all the data currently
**            in buffers for the given handle.
**            Assert function implemented whereever possible.
**            Error mechanism altered, so that all errors are logged 
**            so they can be retreived by FileError.
**
** V 2.31 TDB 02/Nov/93
**            Removed requirement for oldnames in link (now ANSII names
**            throughout. Fixed GP faults caused by empty file. Tested
**            with Visual Basic in DLL form, created Visual Basic Interface
**            file. Altered Windows file I/O to use C LLIO (non-stream)
**            mechanisms, as Windows _lopen set does not include set
**            file length function! CFS16.MAK now uses large model.
**
** V 2.32 TDB 25/Feb/94
**            Redid CommitCFSFile so that all errors trapped, code for
**            first error is code returned, and all errors logged using
**            InternalError.
**
** V 2.33 TDB 07/Dec/95
**            Fixed trivial problems causing warnings in 32-bit build, now
**            commit code uses correct close function (probably irrelevant)
**            and MSVC make file etc for 32-bit build generated. Ensured
**            that errors in Open/Create calls are passed to InternalError.
**
** V 2.40 TDB 02/May/96
**            Added CFSFileSize function. TDataKind changed to TCFSKind to
**            prevent name clash with SON library. Extended file open checks
**            to detect file too short to be a CFS file - get BADVER error
**            instead of read error.
**
** V 2.41 TDB 16/May/96
**            Changed pointer parameters to const forms wherever appropriate.
**
** V 2.50 TDB 08/Nov/96
**            Changed GetVarVal, SetVarVal and DSFlags so that all of these
**            can be used freely on new files, with 0 specifing the new DS
**            and other section parameters specifying the DS as expected.
**
** V 2.51 TDB 12/May/97
**            Changed SetDSChan and WriteData as above, both can be used on
**            all data sections of a new file, with DS set to 0 or n.
**
** V 2.52 TDB 27/May/98
**            Altered StoreTable so that it will increase the size of the
**            memory table if necessary - for AppendDS. Also tidied up the
**            AppendDS code.
**
** v 2.60 TDB 24/Feb/99
**            Switched to using non handle-based memory allocation (basically
**            malloc) for WIN32 builds, file table is extended as required
**            to allow unlimited numbers of open files. Now uses WIN32 file
**            I/O functions as required, plus AppendDS has been fixed - it
**            was not calling FileUpdate to get the pointer table off the end
**            of the file - leading to corrupted files.
**
** v 2.61 TDB 28/Apr/99
**            Adjusted handling of DS table so that 32-bit build can handle
**            up to 64000 data sections. Changed LoadData & FileData.
**
** v 2.62 SMP 16/Mar/00
**            ClearDS now returns a value when it succeeds!
**            Fixed AppendDS so it now no longer produces a duff DSHeader
**            if called multiple times
**
** v 2.63 SMP 04/Apr/00
**            Temp file name can now be up to 300 chars long.
**
** v 2.64 SMP 01/Jun/00
**            Buffer sizes >= 64k now work without crashing or inefficency in
**            the 32 bit build.
**
** v 2.65 SMP 16/Aug/00
**            AppendDS no longer looses changes to the previously appended DS
**            In particular the scaling factors are maintained.
**
** v 2.66 SMP 13/Mar/01
**            ASSERT definition changed to use _DEBUG and not DEBUG
**            Meaningless ASSERT removed from AppendDS
**
** v 2.67 SMP 27/Feb/02
**            CreateCFSFile was trying to flag errors by setting a handle to
**            -1. This wouldn't work under Windows so errors went unreported.
**
** v 2.68 SMP 17/Jan/03
**            Added SUBSIDIARY channel kind
*****************************************************************************/

#include <stdio.h>                               /* C library I/O functions */
#include <ctype.h>                     /* C library type checking functions */
#include <stdlib.h>           /* C library data conversion + misc functions */
#include <time.h>                      /* C library time and date functions */
#include <string.h>                /* C library string function definitions */

/*************************************************************************
**
** These are the machine specific definitions and includes. Some of them
** are not really machine specific, but I've put them here anyway. first
** of all are the system defines, next machine global, finally specific
**
** Remove the NDEBUG definition to cause the asserts to be used
** The LLIO (low level i/o) should be undefined to use streams library and
** defined for low level. This is to make it a bit easier to implement the
** code on different machines which may not have the low level stuff.
** LLIO is defined in cfs.h and used in cfs.h and here
**
**
** macintosh and MSC are set by the compiler, and define machine type
** _IS_MSDOS_     set in machine.h for an msdos native mode build
** _IS_WINDOWS_   set in machine.h for windows 16 or 32 bit mode
** qDebug         (mac)set by MPW to enable debug fprintf
**
** NDEBUG         gets rid of the asserts.
** USEHANDLES     To use memory handles in place of pointers. This form is
**                supported by code generated for the Mac and Windows forms.
**                Actually, it is ignored.
**
** LLIO           (dos) if defined, low level I/O is used, otherwise streams.
** CFSCONVERT     (mac) define if we should convert format (little-big endian)
**  [convert      (mac) specifies direction of conversion - for CfsConvert]
**
** #undef  NDEBUG No asserts if this is defined                     
**
*/


#if defined(qDebug) || defined(_STFDEBUG)
    #undef NDEBUG
    #include <assert.h>
    #define ASSERT(x)       assert(x)
#else
    #define NDEBUG 1
    #define ASSERT(x)
#endif

#include "cfs.h"        /* Exported type, constant and function definitions */

#ifdef CFSCONVERT
    #include "CfsConv.h"
#endif

#if 0 //def macintosh
    #pragma segment Cfs
#endif


/* define some constants needed in the program */

#define INITCEDFILES   16                  /* Initial file array length */
#define MAXCEDFILES    2048                /* Max no. files for WINDOWS */
#define NDATATYPE      8             /* number of data types defined in CFS */
#define NDATAKIND      3             /* number of data kinds defined in CFS */
#define MAXCHANS       100    /* max number of channels of data in CFS file */
#define MAXFILVARS     100                /* max File varaibles in CFS file */
#define MAXDSVARS      100                 /* max data Sections in CFS file */
#define MAXSTR         256         /* max CFS string length  including NULL */
#define CEDMARKER      "CEDFILE\""                      /* Version 1 marker */
#define PARTMARK       "CEDFILE"            /* Marker for testing old files */
#define MAXLSEEK       2000000000 /* Roughly 2 GByte, the maximum file size */
#define MAXFORWRD      65535            /* maximum value for unsigned short */
#define MAXNODS      64000     /* alloc restrictions are looser for WIN32 */
#define MAXMEMALLOC    65519        /* get problems if try to allocate more */
#define WHOLEFILECHARS 1024/* to hold whole of temp file name including path */
#define MARKERCHARS    8                       /* characters in file marker */
#define PARTMARKCHARS  7         /* characters to test for old version file */
#define MAXFNCHARS     1024             /* characters to test for file name */

/* define error codes */

#define NOHANDLE  -1
#define BADHANDLE -2
#define NOTWRIT   -3
#define NOTWORE   -4
#define NOTWORR   -5
#define NOTOPEN   -6
#define BADVER    -7
#define NOMEMR    -8
#define BADCREAT  -11
#define BADOPEN   -12
#define READERR   -13
#define WRITERR   -14
#define RDDS      -15
#define WRDS      -16
#define DISKPOS   -17
#define BADINS    -18
#define BADFL     -19
#define BADDESC   -20
#define BADPAR    -21
#define BADCHAN   -22
#define XSDS      -23
#define BADDS     -24
#define BADKIND   -25
#define BADVARN   -26
                            /* to convert marker char to old version number */
#define BADDSZ    -27
#define BADOLDVER -39 
                                                 /* define file access mode */
#define  rMode   0                                         /* READ only */
#define  wMode   2                                        /* WRITE only */
#define  rwMode  0                  /* READ|WRITE used to open new file */

                      /* define types which give data structure of CFS file */
typedef char TBigName[WHOLEFILECHARS+2];              /* for temp file name */
typedef char TMarker[MARKERCHARS];                   /* for CED file marker */

#pragma pack(1)

                          /* 1. for file header (fixed) channel information */
typedef struct
{
    TDesc     chanName;                 /* users name for channel, 20 chars */
    TUnits    unitsY;                             /* name of Yunits 8 chars */
    TUnits    unitsX;                             /* name of xunits 8 chars */
    TDataType dType;                         /* storage type 1 of 8 allowed */
    TCFSKind  dKind;                   /* equalspaced, matrix or subsidiary */
    short     dSpacing;                           /* bytes between elements */
    short     otherChan;                            /* used for matrix data */
} TFilChInfo;

typedef TFilChInfo TFilChArr[MAXCHANS];

              /* 2. for data section (may vary with DS) channel information */

typedef struct
{
    CFSLONG  dataOffset;
    CFSLONG  dataPoints;
    float scaleY;
    float offsetY;
    float scaleX;
    float offsetX;
} TDSChInfo;

typedef TDSChInfo TDSChArr[MAXCHANS];
                                             /* 3. for data section headers */
typedef struct
{
    CFSLONG      lastDS;            /* offset in file of header of previous DS */
    CFSLONG      dataSt;                                /* data start position */
    CFSLONG      dataSz;                                 /* data size in bytes */
    TSFlags   flags;                              /* flags for this section */
    short     dSpace[8];                                     /* spare space */
    TDSChArr  DSChArr;                          /* array of DS channel info */
} TDataHead;
                                                  /* 4. for the file header */
typedef struct
{
    TMarker    marker;
    TFileName  name;
    CFSLONG       fileSz;
    char       timeStr[8];
    char       dateStr[8];
    short      dataChans;                     /* number of channels of data */
    short      filVars;                         /* number of file variables */
    short      datVars;                 /* number of data section variables */
    short      fileHeadSz;
    short      dataHeadSz;
    CFSLONG       endPnt;               /* offset in file of header of last DS */
    WORD       dataSecs;                         /* number of data sections */
    WORD       diskBlkSize;                             /* usually 1 or 512 */
    TComment   commentStr;
    CFSLONG       tablePos; 
                  /* offset in file for start of table of DS header offsets */
    short      fSpace[20];                                         /* spare */
    TFilChArr  FilChArr;                     /* array of fixed channel info */
} TFileHead;


#pragma pack()

#if defined(__linux__) || defined(__APPLE__)
typedef TFileHead  * TpFHead;        /* pointer to start of file header */
typedef TDataHead  * TpDHead;        /* pointer to start of data header */
typedef TDSChInfo  * TpDsInfo;
typedef TFilChInfo * TpChInfo;
#else
typedef TFileHead  FAR * TpFHead;        /* pointer to start of file header */
typedef TDataHead  FAR * TpDHead;        /* pointer to start of data header */
typedef TDSChInfo  FAR * TpDsInfo;
typedef TFilChInfo FAR * TpChInfo;
#endif

typedef enum                    /* define types to describe file in program */
{
    reading,
    writing,
    editing,
    nothing
} TAllowed;

typedef struct
{
    fDef    p ;                                              /* file handle */
    fDef    d ;                                              /* file handle */
} TDOSHdl;

typedef struct
{
    TpVDesc  nameP;           /* pointers to start of variable descriptions */
    TpSStr   dataP;               /* pointers to corresponding data storage */
} TPointers;

typedef struct    /* for storing and sending back current error information */
{
    short eFound;
    short eHandleNo;
    short eProcNo;
    short eErrNo;
} TError;

typedef struct            /* For program to keep track of storage locations */
{
    TAllowed    allowed;          /* current state of this set of TFileInfo */
//    THandle     fHeadH;                     /* handle to area for file head */
//    THandle     dHeadH;                  /* handle to area for data section */
//    THandle     eHeadH;           /* handle to area for insert data section */
//    THandle     tableH;         /* handle to area for offsets of ds headers */
    TpFHead     fileHeadP;       /* to storage allocated for general header */
    TpDHead     dataHeadP;  /* to storage allocated for data section header */
    TpDHead     extHeadP;   /* to storage allocated for Insert data section */
    TPointers   FVPoint;    /* to descriptions and values of file variables */
    TPointers   DSPoint; /* to descrip and values of data section variables */
    TpLong      tableP;  /*  array of offsets in the file of the DS headers */
    TDOSHdl     DOSHdl;
#ifndef macintosh
    TBigName    tempFName;
#endif
    WORD        thisSection;
    short       DSAltered;
} TFileInfo;

/***********************  CFS FILE STRUCTURE   *****************************
**
** The file header consist of a struct of type TFileHead less the final 
** array which is replaced by fixed info array for each channel + file and
** DS vraiable descriptions + byte array containing values of the file data
** variables.
** The file also contains the channel data and the data sections. How these
** are arranged in the file is not defined. Each data section (DS) consists
** of a struct of type TDataHead less the final array which is replaced by
** DS info array for each cahnnel + byte array containing values of the DS
** data variables for that DS.
** The position and size of the actual channel data is in the DS header.
** At the end of the file is a table of pointers to the start of each DS.
**
*****************************************************************************/

/****************************************************************************
**
**  Global Variables Declaration and Initialisation
**  NB declare near so they go in the default data segment.
**
**  1. Declare pointer to FileInfo and initialise to null, set global
**     array length info to zero.
**
*****************************************************************************/

int         g_maxCfsFiles = 0;        /* The length of the array of file info */
TFileInfo*  g_fileInfo = NULL;


/****************************************************************************
**
**  2. Declare an error structure and initialise all fields to zero.
**
*****************************************************************************/

#if defined(__linux__) || defined(__APPLE__)
TError errorInfo = {0,0,0,0};
#else
TError  _near errorInfo = {0,0,0,0};
#endif
char    gWorkStr[MAXFNCHARS];    /* Global var on DS to avoid DLL SS != DS problems */

                                              /* Local function definitions */
static void  CleanUpCfs(void);
static short FindUnusedHandle(void);
static short SetSizes(TpCVDesc theArray,TpShort offsetArray,short numVars);
static void  TransferIn(TpCStr olds,TpStr pNew,BYTE max);
static void  TransferOut(TpCStr olds,TpStr pNew,BYTE max);
static void  SetVarDescs(short numOfVars,TPointers varPoint,
                               TpCVDesc useArray,TpShort Offsets,short vSpace);
static CFSLONG  BlockRound(short handle,CFSLONG raw);
static void  InternalError(short handle,short proc,short err);
static short GetHeader(short handle,WORD getSection);
static short FileData(short handle, TpVoid startP, CFSLONG st, CFSLONG sz);
static short LoadData(short handle, TpVoid startP, CFSLONG st, CFSLONG sz);
static CFSLONG  GetTable(short handle, WORD position);
static void  StoreTable(short handle,WORD position,CFSLONG DSPointer);
static short RecoverTable(short handle,TpLong relSize,TpLong tPos,
                                                 TpUShort dSecs,TpLong fSize);
static short TransferTable(WORD sects, fDef rdHdl, fDef wrHdl);
static short GetMemTable(short handle);
static TpStr AllocateSpace(TpUShort sizeP, WORD steps);
static void  ExtractBytes(TpStr destP,WORD dataOffset,
             TpStr srcP,WORD points,WORD spacing,WORD ptSz);
static short FileUpdate(short handle,TpFHead fileHP);

                                 /* Function definitions for IO functions . */

#if 0 //def macintosh
    static short TempName(short handle,ConstStr255Param name,
                                                ConstStr255Param str2);
    static short CCreat(ConstStr255Param name,short vRefNum,CFSLONG dirID,
                             SignedByte perm, OSType creator,OSType fileType);
    static short CUnlink(ConstStr255Param name,short vRefNum,CFSLONG dirID);
    static short COpen(ConstStr255Param name, short vRefNum, CFSLONG dirID,
                                                             SignedByte perm);
    static short CCloseAndUnlink(fDef handle);

#endif

static short TempName(short handle,TpCStr name,TpStr str2, unsigned str2len);
static short CCreat(TpCStr name, short mode, fDef* pFile);
static short CUnlink(TpCStr path);
static short COpen(TpCStr name, short mode, fDef* pFile);
static short CCloseAndUnlink(fDef handle, TpCStr path);

static CFSLONG   CGetFileLen(fDef handle);
static short  CClose(fDef handle);
static CFSLONG   CLSeek(fDef handle, CFSLONG offset, short mode);
static WORD   CReadHandle(fDef handle,TpStr buffer, WORD bytes);
static WORD   CWriteHandle(fDef handle,TpStr buffer, WORD bytes);
static void   CMovel(TpVoid dest,const TpVoid src, int count);
static short  CSetFileLen(fDef handle, CFSLONG size);
static TpVoid CMemAllcn(int size);
static void   CFreeAllcn(TpVoid p);
static void   CStrTime(char *timeStr);
static void   CStrDate(char *dateStr);

/**************************************************************************
** 
** Function definitions for IO functions .
** Set of C functions which call the MS ones in io.h 
** THESE FUNCTIONS ARE FOR BINARY FILES.
** Changes made and memory allocation functions added to force FAR pointers
** 
***************************************************************************/


/*************************      CCreat      ********************************
**
** CCreat. This is very like the Pascal equiv in MSDOSLIB CREAT but only
**         modes allowed are 0 Read/Write, 1 Read only
**
** Either creates a new file or opens and truncates an existing one.
** The mode is applied to newly opened files only. It sets the allowed
** file access after the file has been closed.
**
** On the Mac we have to explicitly create a new file before opening it
** If the file exists, we must truncate it. Then set the file type and
** creator.
**
** Return is handle value or -ve error code.
**
***************************************************************************/

#if 0 //def macintosh
short CCreat(ConstStr255Param name,short vRefNum,CFSLONG dirID, SignedByte perm,
                                               OSType creator,OSType fileType)
{
    short fileRefNum ;              /* Mac file refnum, MUST be a short */
    OSErr crErr, err ;              /* error codes from create and io ops */
    FInfo fndrInfo ;                /* for get/setting filetype and creator */

/*
** Mac version is a little different, as create fails if file exists
** and open will fail if file does not exist
*/
    crErr = HCreate(vRefNum, dirID, name, creator, fileType);
    err   = HOpen (vRefNum, dirID, name, perm, &fileRefNum);
    
    if (err == noErr)                                    /* opened the file */
    {
        if (crErr != noErr)                        /* if already existed... */
            if ((err = SetEOF(fileRefNum, 0)) == noErr)   /* ...truncate it */
            {                                            /* get finder info */
                err = HGetFInfo(vRefNum, dirID, name, &fndrInfo); 
                if (err == noErr)
                {
                    fndrInfo.fdType = fileType;            /* Set file type */
                    fndrInfo.fdCreator  = creator ;          /* and creator */
                    err = HSetFInfo(vRefNum, dirID, name, &fndrInfo); 
                }
        
            }
    }
    else if (crErr != noErr)                    /* check for error creating */
        err = crErr;                           /* err is now our error code */
        
    if (err != noErr)
        return err ;                             /* This is a Mac OSErr code*/
    return fileRefNum;                            /* save the ref num if ok */

}                                                          /* end of CCreat */
#endif

short CCreat(TpCStr name, short mode, fDef* pFile)
{
    short    sErr = 0;
    #ifdef WIN32
    DWORD    dwMode;
    #else
    short    pmode;
    char     fname[MAXFNCHARS];              /* To get near variable holding string */
    #endif
    fDef     file;                      /* The various types of file handle */

//
//    #ifdef _IS_WINDOWS_
//        int      file;                     /* local copy of the file handle */
//    #else
//        #ifdef LLIO
//            int      file;                 /* local copy of the file handle */
//        #else
//            FILE*    file;                          /* file stream pointer */
//        #endif
//    #endif

    #ifdef WIN32
        if (mode)                         /* Sort out the file access value */
            dwMode = GENERIC_READ;
        else
            dwMode = GENERIC_READ | GENERIC_WRITE;
    #else
        if (mode)
            pmode = S_IREAD;
        else
            pmode = S_IREAD|S_IWRITE;
    #endif

    #ifdef _IS_WINDOWS_
        #ifdef WIN32
            file = CreateFileA(name, dwMode, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
            if (file == INVALID_HANDLE_VALUE)
                sErr = BADCREAT;
    #else
            if (strlen(name) < MAXFNCHARS)
                F_strcpy(fname, name);              /* Get filename in near var */
            file = _open(fname, (int)(O_CREAT|O_RDWR|O_TRUNC|O_BINARY), pmode);
            if (file < 0)
                sErr = 0 - _doserrno;
        #endif
    #else
        #if defined(__linux__) || defined(__APPLE__)
            char*     omode;

            if (mode)                         /* Sort out the file access value */
                omode = "r";
            else
                omode = "w+";

            if (strlen(name) < MAXFNCHARS)
                F_strcpy(fname, name);              /* Get filename in near var */
            file = fopen(fname,omode);
            if (file == 0)
                sErr = -1;
        #else
            #ifdef LLIO
                if (strlen(name) < MAXFNCHARS)
                    F_strcpy(fname, name);              /* Get filename in near var */
                file = open(fname,(int)(O_CREAT|O_RDWR|O_TRUNC|O_BINARY),pmode);
                if (file < 0)
                    sErr = 0 - _doserrno;
            #else
                if (strlen(name) < MAXFNCHARS)
                    F_strcpy(fname, name);              /* Get filename in near var */
                file = fopen(fname,"wb");
                if (file==0)
                    sErr = -1;
            #endif                                              /* if LLIO else */
        #endif                                               /* if Windows else */
    #endif
    if (sErr == 0)
        *pFile = file;

    return sErr;
}                                                          /* end of CCreat */

/*******************  2. CUnlink for deleteing files. ***********************
**
**   Delete file specified by path. Return 0 if ok, -ve error code
**
****************************************************************************/
#if 0 //def macintosh
short CUnlink(ConstStr255Param name,short vRefNum,CFSLONG dirID)
{
    return HDelete(vRefNum, dirID, name);
}
#endif

#if defined(_IS_MSDOS_) || defined(_IS_WINDOWS_)
short CUnlink(TpCStr path)
{
    int     a;

    a = _unlink(path);                                /* C function in io.h */
    if (a < 0) 
        return (short)(0 - _doserrno);
    else
        return 0;
}
#endif

#if defined(__linux__) || defined(__APPLE__)
short CUnlink(TpCStr path)
{
    return remove(path);                                /* C function in io.h */
}
#endif

/*************  3. CClose for closing files given DOS handle ****************
**
**  Close the file associated with handle.
**  Return 0 if ok or -ve error code.
**
****************************************************************************/
short CClose(fDef handle)
{
    int     res = 0;

    #if 0 //def macintosh
        short   vRefNum = 0;    /* only used by Mac routines, MUST BE SHORT */

        res = GetVRefNum (handle,&vRefNum);           /* get volume ref Num */

         /* if this fails then we probably have a serious programming error */

        res = FSClose (handle);
        #if qDebug
            if (res)
                fprintf(stderr, "ERROR: %d CFSCloseFile\n", res) ;
        #endif
        if (!res)                                     /* if close was ok,...*/
            res = FlushVol (NULL, vRefNum);             /* flush the volume */
    #endif

    #ifdef _IS_WINDOWS_
        #ifdef WIN32
            if (!CloseHandle(handle))
                res = BADHANDLE;
        #else
            res = _close(handle);                   /* C function from io.h */
            if (res < 0) 
                res = 0 - _doserrno;
        #endif
    #endif
    
    #ifdef _IS_MSDOS_
        #ifdef LLIO
            res = close(handle);                    /* C function from io.h */
            if (res<0) 
                return (short)-_doserrno;
        #else
            res = fclose(handle);                          /* shut the file */
        #endif
    #endif
    #if defined(__linux__) || defined(__APPLE__)
            res = fclose(handle);
    #endif

    return (short)res;
}                                                          /* end of CClose */


/*************  CCloseAndUnlink closes then deletes the file ****************
**
**  Closes the file associated with handle, and then deletes it
**  Return 0 if ok or -ve error code.
**
****************************************************************************/

#if defined(_IS_MSDOS_) || defined(_IS_WINDOWS_)
short CCloseAndUnlink(fDef handle, TpCStr path)
{
    short err = 0;
    
    err = CClose(handle);
    err = CUnlink(path);

    return err;
}
#endif

#if defined(__linux__) || defined(__APPLE__)
short CCloseAndUnlink(fDef handle, TpCStr path)
{
    short err = 0;
    
    err = CClose(handle);
    err += CUnlink(path);

    return err;
}
#endif

#if 0 //def macintosh
short CCloseAndUnlink(fDef handle)
{
    short err = 0;
    Str31 name;
    
    FCBPBRec paramBlock;
    
    paramBlock.ioCompletion = NULL;
    paramBlock.ioNamePtr    = name;
    paramBlock.ioFCBIndx    = 0;
    paramBlock.ioRefNum     = handle;

    err = PBGetFCBInfo(&paramBlock, FALSE);     /* call before closing file */
    if (err == 0)
    {
        err = CClose(handle);
        if (err == 0)
            err = CUnlink(name,paramBlock.ioFCBVRefNum,paramBlock.ioFCBParID);
    }
    return err;
}
#endif


/*************  4. CLSeek for moving file pointer ***************************
**
**  Move the file pointer (ie. next position for read/write) by offset bytes
**  in file  with DOShandle handle from a start position depending on mode.
**  Returns new file position or -ve error code 
**
*****************************************************************************/
CFSLONG CLSeek(fDef handle,                              /* DOS handle of file */
            CFSLONG offset,                         /* amount to move in bytes */
            short mode)                   /* 0 Move from file start
                                             1 Move from current file position
                                             2 Move from file end */
{
#if 0 //def macintosh
    OSErr err;
    CFSLONG eof;

    mode = 0;                                 /* to prevent warning message */

    err = GetEOF(handle, &eof);      /* if need to seek past current end of */
                                      /*  file then first have to extend it */
    if (err == noErr)
    {
        if (offset > eof)
            err = SetEOF(handle, offset);
            
        if (err == noErr)
            err = SetFPos (handle, fsFromStart, offset);            /* seek */
    }
    
    #if qDebug
    if (err)
        fprintf (stderr,"ERROR: %d SetFPos, offset %d\n",err,offset);
    #endif
    
    return (CFSLONG)err;
#endif

#ifdef _IS_WINDOWS_                                         /* Windows seek */
    CFSLONG     res = 0;
    #ifdef WIN32
        DWORD       dwMode;

        switch (mode)
        {
            case 0 : dwMode = FILE_BEGIN;                  /* start of file */
                     break;
            case 1 : dwMode = FILE_CURRENT; /* current posn of file pointer */
                     break;
            case 2 : dwMode = FILE_END;                      /* end of file */
                     break;
        }

        res = SetFilePointer(handle, offset, NULL, dwMode);
        if (res == 0xFFFFFFFF)
            return DISKPOS;
        else
            return res;
    #else
        short    origin = 0;

        switch (mode)
        {
            case 0 : origin = SEEK_SET;                    /* start of file */
                     break;
            case 1 : origin = SEEK_CUR;     /* current posn of file pointer */
                     break;
            case 2 : origin = SEEK_END;                      /* end of file */
                     break;
        }
        res = _lseek(handle, offset, origin);                  /* LLIO read */
        if (res < 0)
            return 0 - _doserrno;                 /* _doserrno set by LSEEK */
        else
            return res;
    #endif
#endif  /* if _IS_WINDOWS_ else */

#ifdef _IS_MSDOS_                                         /* DOS-style seek */
    CFSLONG     res;
    short    origin = 0;

    switch (mode)
    {
            case 0 : origin = SEEK_SET;                    /* start of file */
                     break;
            case 1 : origin = SEEK_CUR; /* current position of file pointer */
                     break;
            case 2 : origin = SEEK_END;                      /* end of file */
                     break;
    }
    #ifdef LLIO
    {
        res = lseek(handle, offset, origin);                   /* LLIO read */
        if (res < 0)
            return - _doserrno;                   /* _doserrno set by LSEEK */
        else
            return res;
    }
    #else
        res = fseek (handle, offset, origin);                 /* stdio read */
        return res;
    #endif  /* if LLIO else */
#endif  /* if _IS_MSDOS_ else */
#if defined(__linux__) || defined(__APPLE__)
    CFSLONG     res;
    short    origin = 0;

    switch (mode)
    {
            case 0 : origin = SEEK_SET;                    /* start of file */
                     break;
            case 1 : origin = SEEK_CUR; /* current position of file pointer */
                     break;
            case 2 : origin = SEEK_END;                      /* end of file */
                     break;
    }
    res = fseek (handle, offset, origin);                 /* stdio read */
    if (res==0) {
		return ftell(handle);
	} else {
		return -1;
	}
#endif
}                                                          /* end of CLSeek */


/********************  5. Transfer from file to buffer **********************
**
**  Transfer from current file position of file, specified by its DOS file
**  handle, to the buffer the specified number of bytes.(counted from file)
**  Return the number of bytes transferred to the buffer or zero if error.
**  NB C Function _dos_read can have been successful and NOT give numread=bytes
**     It can have read numread bytes then found EOF or the file can be text
**     in which case CRLF is counted as 2 bytes in the file but only
**     1 byte (LF) in numread.
**
*****************************************************************************/

WORD CReadHandle(fDef handle, TpStr buffer, WORD bytes)
{
#ifdef  macintosh
    OSErr   err;
    CFSLONG    nBytes = bytes;             /* Mac read routine expects longint */

    err = FSRead (handle, &nBytes, buffer) ;
    if (err)
    {
    #if qDebug
        fprintf (stderr, "ERROR: %d FSRead, nbytes read %d\n", err, nBytes);
    #endif
        return err;
    }
    else
        return bytes;
#endif

#ifdef _IS_WINDOWS_
    #ifdef WIN32
        DWORD   dwRead;

        if (ReadFile(handle, buffer, bytes, &dwRead, NULL))
            return bytes;
        else
            return 0;
    #else
        unsigned numread;

        numread = _read(handle, buffer, bytes);
        if (numread != bytes)
            return 0;
        else 
            return bytes;
    #endif
#endif

#ifdef _IS_MSDOS_
    #ifdef LLIO
        unsigned numread;
        if (_dos_read(handle,buffer,bytes,&numread) != 0)
            return 0;
        else 
            return (WORD)numread;
    #else
        if (fread(buffer, 1, bytes,handle) != bytes)
            return 0;
        else 
            return bytes;
    #endif /* if LLIO else */
#endif /* if MSDOS */

#if defined(__linux__) || defined(__APPLE__)
        if (fread(buffer,1,bytes,handle) != bytes)
            return 0;
        else
            return bytes;
#endif
};

/********************  6. Transfer from buffer to file **********************
**
**  Transfer to current file position of file, specified by its DOS file
**  handle, from the buffer the specified number of bytes.
**  Return the number of bytes transferred from the buffer or zero if error.
**  NB C Function _dos_write can have been successful and NOT give numwrt=bytes
**       It can have written numwrt bytes then run out of disk space.
**       If the file is of type text LF in the buffer is replaced by 2 bytes
**       CRLF but numwrt is unchanged since it refers to the buffer bytes.
**
*****************************************************************************/

WORD CWriteHandle(fDef handle, TpStr buffer, WORD bytes)
{
#ifdef  macintosh
    OSErr err ;
    CFSLONG nBytes = bytes;                /* Mac read routine expects longint */

    if (bytes == 0)                                    /* Protect our backs */
        return 0;
    err = FSWrite (handle, &nBytes, buffer) ;
    if (err)
        return 0;
    else
        return bytes;
#endif

#ifdef _IS_WINDOWS_
    #ifdef WIN32
        DWORD   dwWrit;

        if (WriteFile(handle, buffer, bytes, &dwWrit, NULL))
            return bytes;
        else
            return 0;
    #else
        WORD  numwrt;

        if (bytes == 0)                                    /* Protect our backs */
            return 0;
        numwrt = (WORD)_write(handle, buffer, bytes);
        if (numwrt != bytes)
            return 0;
        else 
            return bytes;
    #endif
#endif

#ifdef _IS_MSDOS_
    #ifdef LLIO
       unsigned numwrt;

       if (_dos_write(handle,buffer,bytes,&numwrt) != 0) 
           return 0;
       else 
           return (WORD)numwrt;
    #else
       if (fwrite(buffer, 1, bytes,handle) != bytes)
           return 0;
       else 
           return bytes;
    #endif /* if LLIO else */
#endif /* else if MSDOS */
#if defined(__linux__) || defined(__APPLE__)
       if (fwrite(buffer, 1, bytes, handle) != bytes)
           return 0;
       else 
           return bytes;
#endif
}                                                    /* end of CWriteHandle */

/********************  7. Memory transfer ***********************************
**
** Transfer count bytes from src to dest addresses.
**   memcpy is used so any overlap of regions is not checked.
**
*****************************************************************************/

void CMovel(TpVoid dest, const TpVoid src, int count)
{
   TpVoid ret;                              /* for return pointer of memcpy */
   ret = F_memcpy(dest,src,(size_t)count);      /* C function from string.h */
   return;
};                                                         /* end of CMovel */

/******  8. Change file size. NB operates on file handle NOT file name  *****/

short CSetFileLen(fDef handle, CFSLONG size)
/*
** For a file specified by its DOS handle, extend or truncate file to length,
** in bytes, specified by size. File must be open for writing.
** Return 0 if successful, -DOS error code if not.
*/
{
    #ifndef WIN32
        short ecode;
    #endif

    #if 0 //def macintosh
        ecode = SetEOF(handle, size);
        return ecode; 
    #endif

    #ifdef _IS_WINDOWS_
        #ifdef WIN32
            if (SetFilePointer(handle, size, NULL, FILE_BEGIN) != 0xFFFFFFFF)
                if (SetEndOfFile(handle))
                    return 0;
                else
                    return BADHANDLE;
            else
                return DISKPOS;
        #else
            ecode = (short)_chsize(handle, size); /* ANSII C function from io.h */
            if (ecode == 0)
                return 0; 
            else 
                return (short)(0 - _doserrno);
        #endif
    #endif
    
    #ifdef _IS_MSDOS_
        #ifdef LLIO
            ecode = (short)_chsize(handle, size); /* ANSII C function from io.h */
            if (ecode == 0)
                return 0; 
            else 
                return (short)-_doserrno;
        #else
            ecode = (short)_chsize(_fileno(handle), size); /* Streams mechanism */
            if (ecode == 0)
                return 0; 
            else 
                return (short)-_doserrno;
        #endif
    #endif
	#if defined(__linux__) || defined(__APPLE__)
		return -1;
	#endif			
}                                                    /* end of CSetFileLen */


/******  8a. Retrieve file size. NB operates on file handle NOT file name  *****/

CFSLONG CGetFileLen(fDef pFile)
/*
** For a file specified by its DOS handle, find the file size in bytes.
** Returns file size if successful, -DOS error code if not.
*/
{
    CFSLONG    lSize;
    
    #if 0 //def macintosh
//        ecode = SetEOF(handle, size);
//        return ecode; 
    #endif

    #ifdef _IS_WINDOWS_
        #ifdef WIN32
            lSize = GetFileSize(pFile, NULL);
            if (lSize != -1)             
                return lSize; 
            else 
                return BADHANDLE;
        #else
            lSize = _filelength(handle); /* ANSII C function from io.h */
            if (lSize != -1)             
                return lSize; 
            else 
                return (CFSLONG)(0 - _doserrno);
        #endif
    #endif
    
    #ifdef _IS_MSDOS_
        #ifdef LLIO
            lSize = _filelength(handle); /* ANSII C function from io.h */
            if (lSize != -1)             
                return lSize; 
            else 
                return (CFSLONG)(0 - _doserrno);
        #else
            lSize = _filelength(_fileno(handle)); /* Streams mechanism */
            if (lSize != -1)             
                return lSize; 
            else 
                return (CFSLONG)(0 - _doserrno);
        #endif
    #endif
    #if defined(__linux__) || defined(__APPLE__)
	fpos_t cur;
	if (fgetpos(pFile,&cur)!=0)
		return -1;
	if (fseek (pFile, 0, SEEK_END)!=0)
		return -1;
   	lSize=ftell (pFile);
	if (fsetpos(pFile,&cur)!=0)
		return -1;
        return lSize;
    #endif
}                                                    /* end of CGetFileLen */


/**************  9.Open an exisiting file of specified name *****************
**
**  Open file specified by its path.
**  For mode = 0 Read only (default)
**           = 1 Write only
**           = 2 Read/Write.
**  Return DOS file handle if ok DOS error code (-ve) if not.
**
**  for the mac, we need vRefNum, which we hope to get from stdFile SFGetFile
**  also we want to make sure we have a pascal string
**
*****************************************************************************/

#if 0 //def macintosh
short COpen(ConstStr255Param name,short vRefNum,CFSLONG dirID,SignedByte perm)
{
    short i ;                                    /* used as the return code */
    short refNum;

    i = HOpen (vRefNum, dirID, name, perm, &refNum);
    return (i == noErr) ? refNum : i;
}                                                         /* CFSOpenOldFile */
#endif

#if defined(_IS_MSDOS_) || defined(_IS_WINDOWS_)
short COpen(TpCStr name, short mode, fDef* pFile)
{
    short   sRes = 0;
    fDef    file;

    #ifdef  WIN32
        DWORD   dwMode;
    #else
        char    fname[MAXFNCHARS];          /* To get near variable holding string */
        int     oflag;
    #endif

//    #ifdef LLIO
//        int     file;                      /* local copy of the file handle */
//        char     fname[70];          /* To get near variable holding string */
//    #else
//        FILE     *file;                              /* file stream pointer */
//        char     fname[70];          /* To get near variable holding string */
//    #endif

    #ifdef WIN32
        switch (mode)                /* use C library constants to set mode */
        {
           case 1 : dwMode = GENERIC_WRITE;
                    break;
           case 2 : dwMode = GENERIC_READ | GENERIC_WRITE;
                    break;
           default: dwMode = GENERIC_READ;
                    break;
        }
    #else
        switch (mode)                /* use C library constants to set mode */
        {
           case 1 : oflag = O_WRONLY|O_BINARY;
                    break;
           case 2 : oflag = O_RDWR|O_BINARY;
                    break;
           default: oflag = O_RDONLY|O_BINARY;
                    break;
        }
    #endif

    #ifdef _IS_WINDOWS_
        #ifdef WIN32
            file = CreateFileA(name, dwMode, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
            if (file == INVALID_HANDLE_VALUE)
                sRes = BADOPEN;
    #else
            if (strlen(name) < MAXFNCHARS)        
                F_strcpy(fname, name);              /* Get filename in near var */
            file = _open(fname, oflag);
            if (file < 0)
                sRes = 0 - _doserrno;
        #endif
    #else
        #ifdef LLIO
            if (strlen(name) < MAXFNCHARS)    
                F_strcpy(fname, name);              /* Get filename in near var */
            file = open(fname, oflag);
            if (file < 0)
                sRes = 0 - _doserrno;
        #else
            if (strlen(name) < MAXFNCHARS)    
                F_strcpy(fname, name);              /* Get filename in near var */
            file = fopen(fname,"r+b");
            if (file < 0)
                sRes = (short)file;
        #endif                                              /* if LLIO else */
    #endif                                               /* if Windows else */

    if (sRes == 0)
        *pFile = file;

    return sRes;
}                                                           /* end of COpen */
#endif

#if defined(__linux__) || defined(__APPLE__)
short COpen(TpCStr name, short mode, fDef* pFile)
{
    short   sRes = 0;
    char    fname[MAXFNCHARS];          /* To get near variable holding string */
    char*     omode;

    switch (mode)                /* use C library constants to set mode */
    {
        case 1 : omode = "w";
                 break;
        case 2 : omode = "r+";
                 break;
        default: omode = "r";
                 break;
    }
    if (strlen(name) < MAXFNCHARS)    
        F_strcpy(fname, name);              /* Get filename in near var */
     *pFile = fopen(fname,omode);
     if (*pFile == 0)
          sRes = -1;

     return sRes;
}
#endif

/**************  10. Allocate memory on the far heap ************************
**
**  Allocate size bytes of memory outside the default data segment and
**  return a far pointer to it.
**
*****************************************************************************/

TpVoid CMemAllcn(int size)
{
    TpVoid      p;

    p = F_malloc(size);       /* Use a simple, non-handle based, allocation */

//    if ((*dummy = (THandle) M_AllocMem(size)) != NULL)
//        p =  M_MoveLockMem(*dummy);
//    else
//        p = NULL;
    return p;
}


/***************  11.Free memory allocated by above function ****************
**
**  Free memory allocated on far heap by _fmalloc.
**  No return.
**
*****************************************************************************/
void CFreeAllcn(TpVoid p)
{
    F_free(p);                              /* Simple memory free mechanism */
//    if (M_UnlockMem(*dummy))
//        M_FreeMem(*dummy);
}


/***************  12. Get the time as string in required format *************
**
**  Gets time formatted as hh:mm:ss. 
**  timeStr must be at least 9 bytes
**
*****************************************************************************/
void CStrTime(char *timeStr)
{
#if defined(__APPLE__) || defined(__linux__)
    time_t    now;
    struct tm *today;
    
    now   = time(NULL);
    today = localtime(&now);
    strftime(timeStr, 9, "%H:%M:%S", today);
#endif
    
#if defined(_IS_MSDOS_) || defined(_IS_WINDOWS_)
    _strtime(timeStr);                  /* timsStr must be at least 9 bytes */
#endif
}                                                        /* end of CStrTime */


/************  13. Get the date as string in the required format  ***********
**
**  gets the date formatted as dd/mm/yy.
**  dateStr must be at least 9 bytes
**
*****************************************************************************/
void CStrDate(char *dateStr)
{
#if 0 //def macintosh
    time_t now;
    struct tm *today;
    
    now   = time(NULL);
    today = localtime(&now);
    strftime(dateStr, 9, "%d/%m/%y", today);
#endif

#if defined(_IS_MSDOS_) || defined(_IS_WINDOWS_)
    char    sdata[9];

    _strdate(dateStr);                  /* dateStr must be at least 9 bytes */
    F_strncpy(sdata,dateStr,8);                  /* store time without NULL */
    sdata[0] = dateStr[3];
    sdata[1] = dateStr[4];
    sdata[3] = dateStr[0];
    sdata[4] = dateStr[1];
    F_strncpy(dateStr,sdata,8);                  /* store time without NULL */

#endif
#if defined(__linux__) || defined(__APPLE__)
    time_t now;
    struct tm *today;
    
    now   = time(NULL);
    today = localtime(&now);
    strftime(dateStr, 9, "%d/%m/%y", today);
#endif
}                                                        /* end of CStrDate */


/***********************    CreateCFSFile    ********************************
**
**  Function to create CFS file open it for writing and allocate storage
**  space for header and data section structures ready to fill and write to
**  the file when ready.
**
**  On the Mac we have to explicitly create a new file before opening it
**  If the file exists, we must truncate it. Then set the file type and 
**  creator.
**  Returns file handle or -ve error code.
**
*****************************************************************************/

CFSAPI(short) CreateCFSFile(TpCStr    fname,                /* name of file */
                            TpCStr    comment,           /* general comment */
                            WORD     blockSize,                 /* 1 or 512 */
                            short    channels,   /* number of data channels */
                            TpCVDesc  fileArray,/* file variable descriptions */
                            TpCVDesc  DSArray,  /* DS variable descriptions */
                            short    fileVars,      /* no. of file varables */
                            short    DSVars)         /* no. of DS variables */
#if 0 //def macintosh
CFSAPI(short) CreateCFSFile(ConstStr255Param fname,         /* name of file */
                            TpCStr   comment,            /* general comment */
                            WORD     blockSize,                 /* 1 or 512 */
                            short    channels,   /* number of data channels */
                            TpCVDesc fileArray,/* file varable descriptions */
                            TpCVDesc DSArray,    /* DS varable descriptions */
                            short    fileVars,      /* no. of file varables */
                            short    DSVars,         /* no. of DS variables */
                            short    vRefNum,        /* ..... */
                            CFSLONG     dirID,          /* ..... */
                            OSType   creator,        /* ..... */
                            OSType   fileType)       /* ..... */
#endif
{
    short      proc = 18;               /* function number for error record */
    short      retval;                                     /* return value  */
    short      sErr;                           /* for checking return codes */
    short      handle;                                   /* CFS file handle */
    WORD       bytSz;     /* used to calcualte space needed when allocating */
    short      search;                                      /* loop varable */
    short      pathend = 0;                    /* end of path part of fname */
    short      filVarSpace;   /* space needed for file and DS variable data */
    short      DSVarSpace;
    TpShort    filOffsets;                            /* temp store offsets */
    TpShort    DSOffsets;                             /* temp store offsets */
#ifdef _IS_WINDOWS__
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
                          /* pointer for current files info in global array */
    TpFHead    pFileH;                          /* used to shorten pointers */
    TpDHead    pDataH;
    TpChInfo   pFilChInfo;
    TpDsInfo   pDSChInfo;

#if 0 //def macintosh
    
#endif

/* 1. Get program file handle */

    handle = FindUnusedHandle();                          /* 0,1,2 or error */
    if (handle < 0) 
    {
        retval = NOHANDLE;               /* error code for no spare handles */
        InternalError(handle,proc,retval);
        return retval;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */

/* 2. Open file for writing */

        sErr = CCreat(fname, rwMode, &pfileInfo->DOSHdl.d);/* Open data file */
        if (sErr != 0)
        {
            pfileInfo->DOSHdl.d = (fDef)-1;
            retval = sErr;                /* error code for unopenable file */
            InternalError(handle,proc,retval);
            return retval;
        }

    #if 0 //def macintosh
        pfileInfo->DOSHdl.d = CCreat(fname, vRefNum, dirID, rwMode, creator,
                                     fileType);           /* Open data file */

        if (pfileInfo->DOSHdl.d < 0)
        {
            retval = BADCREAT;                /* error code for unopenable file */
            InternalError(handle,proc,retval);
            return retval;
        }
    #endif
        TempName(handle, fname, pfileInfo->tempFName, WHOLEFILECHARS+2);  /* get temp file name */
        sErr = CCreat(pfileInfo->tempFName, rwMode, &pfileInfo->DOSHdl.p);
        if (sErr != 0)
        {
            pfileInfo->DOSHdl.p = (fDef)-1;
            retval = sErr;
            goto Close_1;                       /* close CFS file before return */
        }
                                                          /* open temp file */
    #if 0 //def macintosh
    {
        Str31 tempFName;
        TempName(handle,fname,tempFName);             /* get temp file name */
        pfileInfo->DOSHdl.p = CCreat(tempFName,vRefNum,dirID,
                                rwMode,'trsh','trsh');    /* open temp file */
    }

    if (pfileInfo->DOSHdl.p < 0)
    {
        retval = BADCREAT;
        goto Close_1;                       /* close CFS file before return */
    }
    #endif

/* 3. Get space for file,DS and spare headers */

/*
** error code for problem with file or Ds variable sizing.
** allocate some space for the offset arrays used by SetSizes, there is
** allways 2 bytes of space allocated to avoid problem when either of
** these two equals 0
*/
    DSOffsets  = (TpShort)CMemAllcn((2*DSVars+2));
    filOffsets = (TpShort)CMemAllcn((2*fileVars+2));
    if ((DSOffsets == NULL) || (filOffsets == NULL))
    {
        retval = NOMEMR; 
        goto Close0;
    }
/*
** setsizes computes the total space needed to store the variables described
** and sets up an offset array which says whereabouts each variables would
** start if you stored all the variables in a byte list
*/

    DSVarSpace  = SetSizes(DSArray, DSOffsets, DSVars);
    filVarSpace = SetSizes(fileArray, filOffsets, fileVars);
    if ((filVarSpace < 0) || (DSVarSpace < 0))
    {
        retval = BADDESC;
        goto Close0;
    }
                                            /* File header size consists of */
    bytSz = (WORD)(sizeof(TFileHead)
                      - sizeof(TFilChArr) +    /* File header without array */
              channels*sizeof(TFilChInfo) +        /* Info for each channel */
          ((fileVars+DSVars+2) * sizeof(TVarDesc)) +   /* desc for each var */
                  filVarSpace); /* space computed for actual file variables */
#if defined(WIN32) || defined(__linux__) || defined(__APPLE__)
#else
    if (bytSz > (WORD)MAXMEMALLOC)
    {
        retval = NOMEMR;                 /* error code for not enough space */
        goto Close0;                   /* dont attempt to allocate over max */
    }
#endif
    pfileInfo->fileHeadP = (TpFHead)CMemAllcn(bytSz);
                                          /* allocate space for file header */
    pFileH = pfileInfo->fileHeadP;
    if (pFileH == NULL)
    {
        retval = NOMEMR;                 /* error code for not enough space */
        goto Close0;                        /* clear up files before return */
    }
    pFileH->fileHeadSz  = (short)bytSz;/* save space in bytes for file head */
    pFileH->diskBlkSize = blockSize;           /* store blocksize specified */

/*
** Now do the Data Section header size
*/
    bytSz = (WORD)(sizeof(TDataHead) -
                            sizeof(TDSChArr) + /* Data header without array */
                 (channels * sizeof(TDSChInfo))+   /* Info for each channel */
                                  DSVarSpace);     /* DS varable data space */
                                                  /* round to nearest block */
    bytSz = (WORD)(((bytSz+(WORD)(blockSize-1)) / blockSize)*blockSize);
#if defined(WIN32) || defined(__linux__) || defined(__APPLE__)
#else
    if (bytSz>(WORD)MAXMEMALLOC)
    {
        retval = NOMEMR;
        goto Close1;
    }
#endif
    pfileInfo->dataHeadP= (TpDHead)CMemAllcn(bytSz);
                                            /* allocate space for DS header */
    pDataH = pfileInfo->dataHeadP;
    if (pDataH == NULL)
    {
        retval = NOMEMR;
        goto Close1;              /* release file and allocated file header */
    }
    pFileH->dataHeadSz  = (short)bytSz;    /* store DS header size in bytes */
    pfileInfo->extHeadP = (TpDHead)CMemAllcn(bytSz);
                              /* space for insert block (another DS header) */
    if (pfileInfo->extHeadP == NULL)
    {
        retval = NOMEMR;
        goto Close2;                 /* release file,fileheader,data header */
    }

/* 4. Store variables in header checking range */

    retval = BADPAR;                         /* error code for out of range */
    if ( (channels < 0) || (fileVars < 0) || (DSVars < 0) || 
         (channels >= MAXCHANS) || (fileVars >= MAXFILVARS) ||
         (DSVars >= MAXDSVARS) )
    {
        retval = NOMEMR;
        goto Close3;   /* Release file file header data header insert block */
    }
    pFileH->filVars  = fileVars;     /* store specified number of file vars */
    pFileH->datVars  = DSVars;         /* store number of data section vars */
    pFileH->dataChans= channels; /* store specified number of data channels */

/* 5. Set pointers to within spaces allocated to start of various items */

               /* The fixed channel information following the general occupies
                  fileChArr[channels] and the space straight after this is for
                                      the descrition of the file variables. */
    pfileInfo->FVPoint.nameP = (TpVDesc)(pFileH->FilChArr+channels);

              /* The fixed file variable descriptions occupy nameP[filvars+1]
                           and the data section variables start after them. */

    pfileInfo->DSPoint.nameP = pfileInfo->FVPoint.nameP+fileVars+1;

              /* The data section variable descriptions occupy nameP[DSVars+1]
                                        then come the actual file variables */

    pfileInfo->FVPoint.dataP = (TpSStr)(pfileInfo->DSPoint.nameP + DSVars+1);
              /* The actual data section variables are part of the data
                 header (not fileheader) and come straight after the data
                                                       section channel info */
    pfileInfo->DSPoint.dataP = (TpSStr)(pDataH->DSChArr + channels);

/* 6. Store default values in data structure structs */
                                /* Fixed channel information in File Header */
    for (search = 0;search < channels;search++)
    {
        pFilChInfo = pFileH->FilChArr+search;
                                           /* pointer to relevant chan info */
                                 /* Put null strings in Chan name and units */
        TransferIn("", pFilChInfo->chanName, 0);
        TransferIn("", pFilChInfo->unitsY, 0);
        TransferIn("", pFilChInfo->unitsX, 0);
        pFilChInfo->dType    = INT2;             /* default type is integer */
        pFilChInfo->dKind    = EQUALSPACED;
        pFilChInfo->dSpacing = 2;            /* integers stored in sequence */
        pFilChInfo->otherChan= 0;
    }

/***************************************************************************
** Fixed file variables in the File header
** Copy the variable descriptions to the space allocated (which is in the
** form of variable description structs) storing the offsets in the data
** array for each variable value in the vsize field
****************************************************************************/

    SetVarDescs(fileVars,pfileInfo->FVPoint,fileArray,filOffsets,filVarSpace);
                                     /* Data section variables descriptions */
    SetVarDescs(DSVars,pfileInfo->DSPoint,DSArray,DSOffsets,DSVarSpace);
                             /* Now fill in as much as poss of header block */
    pDataH->lastDS = 0;                        /* no data section there yet */
                /* lastDS of zero means the fileheader is the 'previous' DS */
    pDataH->dataSt = BlockRound(handle,(CFSLONG)pFileH->fileHeadSz);/*round up */
    pFileH->fileSz = pDataH->dataSt;        /* reset so they are consistent */
    pDataH->dataSz = 0;                                    /* fill in later */
    pDataH->flags  = noFlags;                        /* start with no flags */
    for (search = 0;search < 8;search++)
         pDataH->dSpace[search] = 0;                    /* zero spare space */
                                          /* Initialise channel information */
    for (search = 0; search < channels; search++)
    {
        pDSChInfo = pDataH->DSChArr + search;
        pDSChInfo->dataOffset = 0;                     /* no data there yet */
        pDSChInfo->dataPoints = 0;
        pDSChInfo->scaleY     = (float)1.0;
        pDSChInfo->offsetY    = (float)0.0;
        pDSChInfo->scaleX     = (float)1.0;
        pDSChInfo->offsetX    = (float)0.0;
    }
    TransferIn(comment,pFileH->commentStr,COMMENTCHARS);
                                           /* users comment in LSTRING form */
    F_strncpy(pFileH->marker,CEDMARKER,MARKERCHARS); /* marker with no NULL */
    CStrTime(gWorkStr);                                      /* time + NULL */
    F_strncpy(pFileH->timeStr,gWorkStr,8);       /* store time without NULL */
    CStrDate(gWorkStr);                                      /* date + NULL */
    F_strncpy(pFileH->dateStr,gWorkStr,8);       /* store date without NULL */
    pFileH->dataSecs   = 0;                                    /* No DS yet */
    pfileInfo->tableP  = NULL;
           /* table giving file offsets for each DS header is not in memory */
    pFileH->tablePos= 0;  /* position in file where DS offsets table starts */
    pFileH->endPnt  = 0;/* this will hold the offset of the final DS header */
    pfileInfo->allowed    = writing;                     /* file now opened */
    pfileInfo->thisSection= 0xFFFF;             /* not used in writing mode */
    for (search = 0;search < 20; search++)
        pFileH->fSpace[search] = 0;                     /* zero spare space */
    pfileInfo->DSAltered = 0;
                           /* initialise to 0 so that GetHeader does not write
                                 the DS header to file.(InsertDS is only valid
                                             way to do this for a new file. */
    errorInfo.eFound = 0;                           /* no filing errors yet */
    
#if defined(_IS_MSDOS_) || defined(_IS_WINDOWS_)
    /* In file header store the file name without a path or leading spaces
                                                          in LSTRING format */
    search = 0;
                                                /* find 1st non space TpStr */
    while (isspace(fname[search]))
        search++;
    pathend = (short)(search-1);
                   /* If there is no path in fname it ends before the start */
                                 /* look for last occurrence of '\\' or ':' */
    while (search<(short)F_strlen(fname))
    {
        if ((fname[search] == '\\') || (fname[search] == ':'))
            pathend = search;
        search = (short)(search + 1);
    }
    TransferIn(&fname[pathend+1],pFileH->name,FNAMECHARS);
#endif

#if 0 //def macintosh
    p2cstr(fname);
    TransferIn(fname,pFileH->name,FNAMECHARS);
    c2pstr(fname);
#endif

                       /* return space allocated for temp arrays of affsets */
    CFreeAllcn(filOffsets);
    CFreeAllcn(DSOffsets);

    return handle;                      /* all ok so return the file handle */

/*
** Here for the general error handling. Tidy up and return error code,
** also calling Internal Error so that any errors are remembered.
*/
    Close3: CFreeAllcn(pfileInfo->extHeadP);    /* free mem for data insert */
            pfileInfo->extHeadP = NULL;

    Close2: CFreeAllcn(pfileInfo->dataHeadP);        /* and for data header */

    Close1: CFreeAllcn(pfileInfo->fileHeadP);        /* and for file header */

    Close0: 
            CCloseAndUnlink(pfileInfo->DOSHdl.p,
                            pfileInfo->tempFName);      /* delete temp file */
#if 0 //def macintosh
            CCloseAndUnlink(pfileInfo->DOSHdl.p);
#endif
            CFreeAllcn(filOffsets);      /* release space for offset arrays */
            CFreeAllcn(DSOffsets);

    Close_1:
            if (strlen(fname) < MAXFNCHARS)
                F_strcpy(gWorkStr,fname);      /* Get file name into global var */
            CCloseAndUnlink(pfileInfo->DOSHdl.d, gWorkStr); 
                                                   /* delete empty CFS file */
#if 0 //def macintosh
            CCloseAndUnlink(pfileInfo->DOSHdl.d);
#endif

    InternalError(handle,proc,retval);
    return retval;                                  /* retval is error code */
}                                                    /* end of CreatCFSFile */

/************************    SetFileChan    ********************************
**
**  Set the channel information that stays same throughout file and goes
**  in the file header.
**
*****************************************************************************/

CFSAPI(void) SetFileChan(short     handle,           /* program file handle */
                         short     channel,  /* channel numb in CFS storage */
                         TpCStr     channelName,
                                            /* users identifier for channel */
                         TpCStr     yUnits,         /* users name for units */
                         TpCStr     xUnits,/* only for equal spaces often s */
                         TDataType dataType,              /* one of 8 types */
                         TCFSKind  dataKind,              /* one of 2 kinds */
                         short     spacing,
                                     /* bytes between equal spaced elements */
                         short     other)   /* next channel for matrix data */
{
    short       proc = 1;               /* function number for error record */
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
                          /* to point to set of file info with given handle */
    TpChInfo    pFilChInfo;     /* to point to this channels info structure */
    short       ecode;                          /* for return of FileUpdate */

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(channel >= 0);
    ASSERT(channel < g_fileInfo[handle].fileHeadP->dataChans);
    ASSERT(dataType >= 0);
    ASSERT(dataType <  NDATATYPE);
    ASSERT(dataKind >= 0);
    ASSERT(spacing  >= 0);
    ASSERT((g_fileInfo[handle].allowed == writing) ||
            (g_fileInfo[handle].allowed == editing));

    /* 1. Check valid handle given and that its file is open for write/edit */

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,proc,BADHANDLE);
        return;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    if ((pfileInfo->allowed == writing) || (pfileInfo->allowed == editing))
    {
    /* 2. Check channel number in range */

        if ((channel < 0) || (channel >= pfileInfo->fileHeadP->dataChans))
        {
            InternalError(handle,proc,BADCHAN);
            return;
        }
    /* Additional checks on parameters dataType and dataKind */

        if ((dataType < 0) || (dataType >= NDATATYPE))
        {
            InternalError(handle,proc,BADPAR);
            return;
        }
        if ((dataKind < 0) || (dataKind >= NDATAKIND))
        {
            InternalError(handle,proc,BADKIND);
            return;
        }
        if ((spacing < 0) || ((dataKind == MATRIX) && (other < 0)))
        {
            InternalError(handle,proc,BADPAR);
            return;
        }

    /* If this is the first edit made to a file get it ready for the change */
        if ((pfileInfo->allowed == editing) &&
                                        (pfileInfo->fileHeadP->tablePos != 0))
        {
            ecode = FileUpdate(handle,pfileInfo->fileHeadP);
            if (ecode != 0)
            {
                InternalError(handle,proc,ecode);
                return;
            }
        }
                /* channels info parameters look reasonable so fill them in */

        pFilChInfo = pfileInfo->fileHeadP->FilChArr + channel;
                                                           /* pick out chan */
        TransferIn(channelName,pFilChInfo->chanName,DESCCHARS);
        TransferIn(yUnits,pFilChInfo->unitsY,UNITCHARS);
        TransferIn(xUnits,pFilChInfo->unitsX,UNITCHARS);
        pFilChInfo->dType     = dataType;
        pFilChInfo->dKind     = dataKind;
        pFilChInfo->dSpacing  = spacing;
        pFilChInfo->otherChan = other;
    }
    else
        InternalError(handle,proc,NOTWORE);          /* failed allowed test */
    return;
}                                                     /* end of SetFileChan */

/**************************    SetDSChan    ********************************
**
**  Sort out the data section channel info ie the bit that can change
**  with data section.
**
*****************************************************************************/

CFSAPI(void) SetDSChan(short handle,                 /* program file handle */
                       short channel,                 /* CFS channel number */
                       WORD  dataSection,         /* reference when editing */
                       CFSLONG  startOffset,     /* byte offset to 1st element */
                       CFSLONG  points,          /* number of channel elements */
                       float yScale,                          /* data scale */
                       float yOffset,
                       float xScale,
                       float xOffset)
{
    short       proc = 2;                       /* number for this function */
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    TpDsInfo    pDSChInfo;
    short       ecode;

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT((g_fileInfo[handle].allowed == writing) ||
            (g_fileInfo[handle].allowed == editing));
    ASSERT(channel >= 0);
    ASSERT(channel < g_fileInfo[handle].fileHeadP->dataChans);

/* 1. Is Handle ok */

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,proc,BADHANDLE);
        return;
    }
/* 1a. Is file open for writing/editing */

    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    if ((pfileInfo->allowed == writing) || (pfileInfo->allowed == editing))
    {

/* 2. Check channel number */
        if ((channel < 0) || (channel >= pfileInfo->fileHeadP->dataChans))
        {
            InternalError(handle,proc,BADCHAN);
            return;
        }

/* 3. Check data section number if editing and read in its header from
      the CFS file */
        if (pfileInfo->allowed == editing)
        {
            if ((dataSection<1)||(dataSection>pfileInfo->fileHeadP->dataSecs))
            {
                InternalError(handle,proc,BADDS);
                return;
            }
            ecode = GetHeader(handle,dataSection);
            if (ecode != 0)
            {
                InternalError(handle,proc,ecode);
                return;
            }
        }
        else
        {
            if (/*(dataSection<0)||*/(dataSection>pfileInfo->fileHeadP->dataSecs))
            {
                InternalError(handle,proc,BADDS);
                return;
            }
            if (dataSection > 0)
            {
                CMovel(pfileInfo->extHeadP, pfileInfo->dataHeadP,   /* Save current DS header */
                            pfileInfo->fileHeadP->dataHeadSz);

                ecode = GetHeader(handle, dataSection); /* Read in wanted */
                if (ecode != 0)
                {
                    InternalError(handle,proc,ecode);
                    goto Restore;
                }
            }
        }
/* If this is the first edit made to a file get it ready for the change */

        if ((pfileInfo->allowed == editing) && 
                                        (pfileInfo->fileHeadP->tablePos != 0))
        {
            ecode = FileUpdate(handle,pfileInfo->fileHeadP);
            if (ecode != 0)
            {
                InternalError(handle,proc,ecode);
                return;
            }
        }

/* Store information from function parameters in program */
        pDSChInfo = pfileInfo->dataHeadP->DSChArr + channel;
        pDSChInfo->dataOffset = startOffset;
        pDSChInfo->dataPoints = points;
        pDSChInfo->scaleY     = yScale;
        pDSChInfo->offsetY    = yOffset;
        pDSChInfo->scaleX     = xScale;
        pDSChInfo->offsetX    = xOffset;
        if (pfileInfo->allowed == editing)
            pfileInfo->DSAltered = 1;                             /* (TRUE) */
        else
        {
            if (dataSection > 0)      /* If we are at previous data section */
            {
                CFSLONG    tableValue;         /* Write the data back to disk */

                tableValue = GetTable(handle, dataSection);
                if (FileData(handle, pfileInfo->dataHeadP, tableValue,
                         (WORD)pfileInfo->fileHeadP->dataHeadSz) == 0)
                {
                    InternalError(handle, proc, WRITERR);
                    goto Restore;
                }
            }
        }

Restore:
        if ((pfileInfo->allowed == writing) &&
            (dataSection > 0))
        {                     /* Then copy the old data back into position */
            CMovel(pfileInfo->dataHeadP,pfileInfo->extHeadP,
                                            pfileInfo->fileHeadP->dataHeadSz);
        }
    }
    else
        InternalError(handle,proc,NOTWORE);       /* failed writing/editing */

    return;
}                                                      /* end of setDSChans */

/**************************    Write Data   ********************************
**
**  Write actual data into current data block.
**
*****************************************************************************/

CFSAPI(short) WriteData(short  handle,               /* program file handle */
                        WORD   dataSection,        /* data section for data */
                        CFSLONG   startOffset,       /* offset in DS for write */
                        WORD   bytes,           /* number of bytes to write */
                        TpVoid dataADS)    /* ptr to start of data to write */
{
    short       proc = 19;                      /* number for this function */
    short       ecode = 0;
    CFSLONG        endOffset;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    TpDHead     dataHP;

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed != nothing);
    ASSERT(g_fileInfo[handle].allowed != reading);

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,proc,BADHANDLE);
        return BADHANDLE;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */

/* action depends on whether file was opened by CREAT (writing) or OPEN (editing) */
    if ((pfileInfo->allowed == writing) && (dataSection == 0))
    {                                            /* adjust pointers for data size */
        dataHP = pfileInfo->dataHeadP;
        endOffset = dataHP->dataSt + startOffset + bytes;
        if (pfileInfo->fileHeadP->fileSz < endOffset)
        {
            pfileInfo->fileHeadP->fileSz = endOffset;
            dataHP->dataSz = endOffset - dataHP->dataSt;
        }
        ecode = FileData(handle,dataADS,dataHP->dataSt + startOffset,bytes);
        if (ecode == 0)
        {
            InternalError(handle,proc,WRITERR);
            ecode = WRITERR;
        }
        else
            ecode = 0;                 /* Set to no error result otherwise */
    }
    else
    {
        if ((pfileInfo->allowed != editing) &&      /* must be OK to write */
            (pfileInfo->allowed != writing))
        {
            InternalError(handle, proc, NOTWORE);
            return NOTWORE;
        }

        if ((dataSection < 1) || (dataSection > pfileInfo->fileHeadP->dataSecs))
        {
            InternalError(handle,proc,BADDS);  /* check data section number */
            return BADDS;
        }

/* If writing, preserve the current data section header */
        if (pfileInfo->allowed == writing)
        {
            CMovel(pfileInfo->extHeadP, pfileInfo->dataHeadP,
                                        pfileInfo->fileHeadP->dataHeadSz);
        }

        ecode = GetHeader(handle, dataSection);
        if (ecode != 0)                        /* can be READERR or WRITERR */
        {
            InternalError(handle,proc,ecode);  /* check data section number */
            goto Restore;
        }
        if ((startOffset + bytes) > pfileInfo->dataHeadP->dataSz)
        {
            InternalError(handle,proc,BADDSZ); /* check data section number */
            ecode = BADDSZ;
            goto Restore;
        }

/* If this is the first edit made to a file get it ready for the change */
        if ((pfileInfo->fileHeadP->tablePos != 0) &&
            (pfileInfo->allowed == editing))
        {
            ecode = FileUpdate(handle, pfileInfo->fileHeadP);
            if (ecode != 0)
            {
                InternalError(handle,proc,ecode);
                goto Restore;
            }
        }
        ecode = FileData(handle, dataADS, pfileInfo->dataHeadP->dataSt + 
                                                           startOffset,bytes);
        if (ecode == 0)
        {
            InternalError(handle,proc,WRITERR);
            ecode = WRITERR;
            goto Restore;
        }
        else
            ecode = 0;              /* No errors, so clear function return */

    Restore:          /* Copy the old data back into position  if required */
        if (pfileInfo->allowed == writing)
        {
            CMovel(pfileInfo->dataHeadP, pfileInfo->extHeadP,
                                         pfileInfo->fileHeadP->dataHeadSz);
        }
    }
    return ecode;
}                                                       /* end of WriteData */

/**************************   Set Write Data  ******************************
**
** For file opened with CREAT only. Set it up ready for fast sequential write.
** Attempt to by pass piecewise disk allocation and speed up write to disc.
** NB BUFFERS = prameter in CONFIG.SYS needs to be at least 20.
** Any error is put into global error struct.
**
*****************************************************************************/
             

CFSAPI(void) SetWriteData(short  handle,             /* program file handle */
                          CFSLONG   startOffset,
                                       /* byte offset within DS for writing */
                          CFSLONG   bytes)         /* number of bytes to write */
{
    short       proc = 3;                                /* function number */
    char        oneByte;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    TpDHead     dataHP;

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed == writing);


/* 1. Check handle */

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,proc,BADHANDLE);
        return;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    if (pfileInfo->allowed != writing)                /* check created file */
    {
        InternalError(handle,proc,NOTWRIT);
        return;
    }
    dataHP = pfileInfo->dataHeadP;
    if ((bytes >= 0) && (startOffset >= 0))         /* If parameters are ok */
/* 2. move to end of data and write 1 byte to allocate file space */
    {
        if (FileData(handle,dataHP,dataHP->dataSt+startOffset + bytes,1) == 0)
        {                           /* attempt write to wouldbe end of file */
            InternalError(handle,proc,WRITERR);
            return;
        }
    }
    else
    {                                   /* either bytes or startOffset duff */
        InternalError(handle,proc,BADPAR);
        return;
    }
/* 3. move to start of data area and read 1 byte to position of head */

    if (LoadData(handle, &oneByte, dataHP->dataSt+startOffset-1, 1) == 0)
    {
        InternalError(handle,proc,READERR);
        return;
    }
    return;
}                                                    /* end of SetWritedata */


/****************************  CFSFileSize  *********************************
**
**  Return the file size, in bytes. The name is to avoid C library clashes.
**
*****************************************************************************/

CFSAPI(CFSLONG) CFSFileSize(short  handle)
{
    short       proc = 24;                               /* function number */
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    TpFHead     fileHP;

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed != nothing);

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle, proc, BADHANDLE);             /* check handle */
        return BADHANDLE;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    if (pfileInfo->allowed == nothing)  /* and also check for file not open */
    {
        InternalError(handle, proc, NOTOPEN);
        return NOTOPEN;
    }
    fileHP = pfileInfo->fileHeadP;            /* Get pointer to file header */

    return fileHP->fileSz;          /* return the file size from the header */
}


/****************************   Insert DS  ******************************
**
**  Close the current data block and insert it at datasection th data 
**  block in the file. For dataSection 0 put it at the end.
**  Return zero or error code.
**
*****************************************************************************/

CFSAPI(short) InsertDS(short   handle,               /* program file handle */
                       WORD    dataSection,               /* section number */
                       TSFlags flagSet)                /* flags for this DS */
{
    short       proc = 17;                               /* function number */
    WORD        index,relevantSize;
    CFSLONG        gtPlace,gtPlace1;
    CFSLONG        tableValue;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
   TpFHead     fileHP;
    TpDHead     dataHP;

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed == writing);
    ASSERT(dataSection < MAXNODS);

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,proc,BADHANDLE);               /* check handle */
        return BADHANDLE;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */

/* 1. check file open for writing and dataSection valid */

    if (pfileInfo->allowed != writing)
    {
        InternalError(handle,proc,NOTWRIT);
        return NOTWRIT;
    }
    fileHP = pfileInfo->fileHeadP;
    if (fileHP->dataSecs >= MAXNODS)
    {
        InternalError(handle,proc,XSDS);
        return XSDS;
    }
    if (dataSection == 0)
        dataSection = (WORD)(fileHP->dataSecs + 1);
    if ((dataSection < 1) || (dataSection > (fileHP->dataSecs + 1)))
    {
        InternalError(handle,proc,BADDS);
        return BADDS;
    }

/* 2. change the pointer table to accommodate new data section */

    for (index = fileHP->dataSecs;index >= dataSection;index--)
    {                                /* move entries past current DS up one */
        tableValue = GetTable(handle,index);
        StoreTable(handle,(WORD)(index+1),tableValue);
    }
    dataHP = pfileInfo->dataHeadP;

/* 3. change pointers for current section */
                   /* for first DS lastDS=0 ie previous thing is the header */
    if (dataSection == 1)
        dataHP->lastDS = 0;
    else
        dataHP->lastDS = GetTable(handle,(WORD)(dataSection-1));
                                                        /* previous section */
    dataHP->dataSz = fileHP->fileSz - dataHP->dataSt;
                         /* set size of data which should be on end of file */
    gtPlace = dataHP->dataSt + BlockRound(handle, dataHP->dataSz);
                                         /* offset corresponding to this DS */
    StoreTable(handle, dataSection, gtPlace);
                                         /* store offset for this DS header */
/* 4. write the data header to the file */

    dataHP->flags = flagSet;                       /* set flags for this DS */
    if (FileData(handle,dataHP,gtPlace,(WORD)fileHP->dataHeadSz) == 0)
    {
        InternalError(handle,proc,WRITERR);
        return WRITERR;
    }

/* 5. alter lastDS pointer in subsequent header */

    relevantSize = sizeof(TDataHead) - sizeof(TDSChArr);
    if (dataSection < (fileHP->dataSecs + 1))
    {                                                   /* not the last one */
        gtPlace1 = GetTable(handle,(WORD)(dataSection + 1));
                                                      /* offset for next DS */
        if (LoadData(handle,pfileInfo->extHeadP,gtPlace1,relevantSize) == 0)
        {
            InternalError(handle,proc,READERR);
            return READERR;                        /* read header to change */
        }
        pfileInfo->extHeadP->lastDS = gtPlace;           /* make the change */
        if (FileData(handle,pfileInfo->extHeadP,gtPlace1,relevantSize) == 0)
        {
            InternalError(handle,proc,WRITERR);
            return WRITERR;
        }
    }
    else
        fileHP->endPnt = gtPlace;                      /* action for end DS */

/* 6. update fileHeader and dataheader variable */

    fileHP->dataSecs = (WORD)(fileHP->dataSecs + 1);
    fileHP->fileSz = gtPlace + fileHP->dataHeadSz;
    dataHP->dataSt = fileHP->fileSz;
    dataHP->dataSz = 0;
    return 0;                                                     /* all ok */
}                                                        /* end of InsertDS */

/****************************   AppendDS  ******************************
**
**  Add an empty data section onto the end of the file, so that we can
**  write to it. Intended for use with offline files.
**
**  Return zero or error code.
**
*****************************************************************************/

CFSAPI(short) AppendDS(short    handle,              /* program file handle */
                       CFSLONG     lSize,           /* size of the new section */
                       TSFlags  flagSet)               /* flags for this DS */
{
    short       proc = 25;                               /* function number */
    WORD        headSize, thisDS;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
   TpFHead     fileHP;
    TpDHead     dataHP;
    CFSLONG        lPrev, lThis;
    CFSLONG        tableValue;

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT((g_fileInfo[handle].allowed == editing) || (g_fileInfo[handle].allowed == writing));

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,proc,BADHANDLE);               /* check handle */
        return BADHANDLE;
    }

    pfileInfo = &g_fileInfo[handle];     /* point to this files information */

    if (pfileInfo->allowed == writing)      /* If writing just use InsertDS */
        return InsertDS(handle, 0, flagSet);             /* as this does it */

/* 1. check file open for writing and work out this DS number */
    if (pfileInfo->allowed != editing)
    {
        InternalError(handle, proc, NOTWORE);
        return NOTWORE;
    }

    if (pfileInfo->DSAltered == 1)  /* do we need to write this DS out first? */
    {
        tableValue = GetTable(handle,pfileInfo->thisSection);
        pfileInfo->DSAltered = 0;
        if (FileData(handle,pfileInfo->dataHeadP,tableValue,(WORD)
                                       pfileInfo->fileHeadP->dataHeadSz) == 0)
        {
            InternalError(handle,proc,WRITERR);
            return WRITERR;
        }
    }
    pfileInfo->thisSection = 0xFFFF;
/* If this is the first edit made to a file get it ready for the change */
/* This code strips the lookup table from the file size, which will help out! */
    if (pfileInfo->fileHeadP->tablePos != 0)
    {
        short   ecode;

        ecode = FileUpdate(handle, pfileInfo->fileHeadP);
        if (ecode != 0)
        {
            InternalError(handle, proc, ecode);
            return ecode;
        }
    }

    fileHP = pfileInfo->fileHeadP;             /* Pointer to the file info */
    dataHP = pfileInfo->dataHeadP;           /* and the data header buffer */
    if (fileHP->dataSecs >= MAXNODS)
    {
        InternalError(handle,proc,XSDS);
        return XSDS;
    }
    headSize = fileHP->dataHeadSz;           /* Size of DS header in total */
    thisDS = fileHP->dataSecs + 1;  /* The number for the new data section */
    lThis = fileHP->fileSz + BlockRound(handle, lSize); /* New DS head pos */

/* 2. Read in the previous DS header so we can meddle */
    lPrev = GetTable(handle, (WORD)(thisDS - 1));/* offset for prev DS */
    if (LoadData(handle, dataHP, lPrev, headSize) == 0)
    {
        InternalError(handle,proc,READERR);
        return READERR;                           /* read header to change */
    }

/* 3. Start creating the new data section. The new section goes at the end */
    dataHP->dataSt = lPrev + headSize;          /* Start posn for the data */
    dataHP->dataSz = lSize;                            /* Size of the data */
    dataHP->lastDS = lPrev;
    dataHP->flags = flagSet;                      /* set flags for this DS */
    pfileInfo->thisSection = thisDS;   /* now talking about a different DS */

    StoreTable(handle, thisDS, lThis);      /* Get the DS start into table */

/* 4. write the data header to the file */
    if (FileData(handle, dataHP, lThis, headSize) == 0)
    {
        InternalError(handle,proc,WRITERR);
        return WRITERR;
    }

/* 5. update fileHeader and dataheader variable */
    fileHP->dataSecs = thisDS;
    fileHP->endPnt = lThis;               /* pointer to the last DS header */
    fileHP->fileSz = lThis + headSize;                /* Updated file size */

    return 0;                                                    /* all ok */
}                                                       /* end of AppendDS */


/****************************   Get DS Size  ********************************
**
**  Returns size (in bytes) of specified data section for old file  or current
**  data section for new file. (file specified by its program file handle ).
**  The size returned is the disk space for the channel data and doesn't
**  include the DS header.
**  Return is the size or -ve error code.
**
*****************************************************************************/

CFSAPI(CFSLONG) GetDSSize(short  handle,                /* program file handle */
                       WORD   dataSection)
{
    short proc = 22;
    short ecode;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif

                                                            /* check handle */
    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed != nothing);

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,proc,BADHANDLE);
        return BADHANDLE;
    }
                                                       /* check file status */
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    if (pfileInfo->allowed != nothing)
    {                                             /* get header if old file */
        if (pfileInfo->allowed != writing)
        {
            if ((dataSection < 1) || 
                               (dataSection > pfileInfo->fileHeadP->dataSecs))
            {
                InternalError(handle,proc,BADDS);
                return BADDS;
            }
            ecode = GetHeader(handle,dataSection);
            if (ecode < 0)
            {
                InternalError(handle,proc,ecode);
                return ecode;
            }
        }
        return pfileInfo->dataHeadP->dataSz;          /* return stored size */
    }
    else
    {
        InternalError(handle,proc, NOTWORR);
        return NOTWORR;
    }
}                                                       /* end of GetDSSize */

/**************************    Clear DS    ********************************
**
** Clear out data already written in current DS. See WriteData.
**
*****************************************************************************/

CFSAPI(short) ClearDS(short handle)

{
    short       proc = 20;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    TpDHead     dataHP = NULL;

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed == writing);

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,proc,BADHANDLE);
        return BADHANDLE;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */

/* action depends on if file was opened by CREAT(writing) or OPEN (editing) */
    if (pfileInfo->allowed == writing)   /* adjust pointers for data size 0 */
    {
        pfileInfo->fileHeadP->fileSz = dataHP->dataSt;
        pfileInfo->dataHeadP->dataSz = 0;
    }
    else
    {
        InternalError(handle,proc,NOTWRIT);
        return NOTWRIT;              /* cannot do it unless CREAT (writing) */
    }

    return 0;
}

/*****************************   Remove DS  *********************************
**
** Remove a data section from the CFS file. Doesn't actually shorten the
** file but rearranges pointers so blocks dont appear any more.
**
*****************************************************************************/

CFSAPI(void) RemoveDS(short handle,                  /* program file handle */
                      WORD  dataSection)                    /* DS to remove */
{
    short       proc = 4;                       /* number for this function */
    CFSLONG        storeLastDS;
    WORD        relSize,index,ecode;
    CFSLONG        tableValue;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(dataSection > 0);
    ASSERT(dataSection <= g_fileInfo[handle].fileHeadP->dataSecs);
    ASSERT((g_fileInfo[handle].allowed == writing) || 
           (g_fileInfo[handle].allowed == editing));

/* 1. Check handle and data section parameters are valid */

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,proc,BADHANDLE);
        return;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    relSize = sizeof(TDataHead) - sizeof(TDSChArr);
                         /* size of data header without channel infoe array */
    if ((pfileInfo->allowed != writing) && (pfileInfo->allowed != editing))
    {
        InternalError(handle,proc,NOTWORE);
        return;
    }
    if ((dataSection < 1) || (dataSection > pfileInfo->fileHeadP->dataSecs))
    {
        InternalError(handle,proc,BADDS);
        return;
    }
/* If this is the first edit made to a file get it ready for the change */

    if ((pfileInfo->allowed == editing) && 
                                        (pfileInfo->fileHeadP->tablePos != 0))
    {
        ecode = FileUpdate(handle,pfileInfo->fileHeadP);
        if (ecode != 0)
        {
            InternalError(handle,proc,ecode);
            return;
        }
    }
    tableValue = GetTable(handle,dataSection);
    if (LoadData(handle,pfileInfo->extHeadP,tableValue,relSize) == 0)
    {
        InternalError(handle,proc,READERR);
        return;
    }
    storeLastDS = pfileInfo->extHeadP->lastDS;
                                         /* save the pointer to previous DS */
/* 3. If current DS needs Filing do so */

    if (pfileInfo->DSAltered == 1)
    {
        tableValue = GetTable(handle,pfileInfo->thisSection);
        pfileInfo->DSAltered = 0;
        if (FileData(handle,pfileInfo->dataHeadP,tableValue,(WORD)
                                       pfileInfo->fileHeadP->dataHeadSz) == 0)
        {
            InternalError(handle,proc,WRITERR);
            return;
        }
    }
    pfileInfo->thisSection = 0xFFFF;

/* 4. Remove section from pointer table */

    for (index = dataSection;index < pfileInfo->fileHeadP->dataSecs;index++)
    {
         tableValue = GetTable(handle,(WORD)(index + 1));
                                             /* shift all offsets one place */
         StoreTable(handle,index,tableValue);              /* back in array */
    }
/* 5. set lastDS for DS which now has removed ones index (or endPnt) if last */

    if (dataSection < pfileInfo->fileHeadP->dataSecs) /* DS is not last one */
    {
        tableValue = GetTable(handle,dataSection);
                                      /* offset for new DS with this number */
        if (LoadData(handle,pfileInfo->extHeadP,tableValue,relSize) == 0)
        {
            InternalError(handle,proc,READERR);
            return;
        }
        pfileInfo->extHeadP->lastDS = storeLastDS;
                                                /* set new previous section */
        if (FileData(handle,pfileInfo->extHeadP,tableValue,relSize) == 0)
        {
            InternalError(handle,proc,WRITERR);
            return;
        }                                    /* write it back to file again */
    }
    else
        pfileInfo->fileHeadP->endPnt = storeLastDS;    /* if DS is last one */

/* 6. There is now 1 DS less in file */

    pfileInfo->fileHeadP->dataSecs--;
    return;
}                                                       /* end of Remove DS */

/*****************************   Set Comment  *******************************
**
**  Can use this function if file opened with CreateCFSFile or OpenCFSFile in
**  edit mode. Overwrite the comment stored in the file header.
**
*****************************************************************************/

CFSAPI(void) SetComment(short handle,                /* program file handle */
                        TpCStr comment)                 /* comment to store */
{
    short proc = 15;                                     /* function number */
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    TpFHead    fileHP;
    short      ecode;                         /* for return from FileUpdate */
 
    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT((g_fileInfo[handle].allowed == writing) || 
           (g_fileInfo[handle].allowed == editing));

/* check handle and file status */

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,proc,BADHANDLE);
        return;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    fileHP = pfileInfo->fileHeadP;
    if ((pfileInfo->allowed != writing) && (pfileInfo->allowed != editing))
    {
        InternalError(handle,proc,NOTWRIT);
        return;
    }
/* If this is the first edit made to a file get it ready for the change */

    if ((pfileInfo->allowed == editing) && 
                                        (pfileInfo->fileHeadP->tablePos != 0))
    {
        ecode = FileUpdate(handle,pfileInfo->fileHeadP);
        if (ecode != 0)
        {
            InternalError(handle,proc,ecode);
            return;
        }
    }
/* if all ok transfer comment to its place in the file header */

    TransferIn(comment,fileHP->commentStr,COMMENTCHARS);
}                                                      /* end of setComment */


/***************************   Set Var Val *******************************
**
**  Transfer the variable value to the byte list ready for transfer to file.
**
**  If file opened by CreateCFSFile (allowed=writing) and the DS variable
**  value applies to the current data section it will keep the value set
**  for subsequent data sections unless changed.
**
**  File opened by OpenCFSFile (allowed=editing) the new DS variable value
**  applies to the data section specified only.
**
*****************************************************************************/

CFSAPI(void) SetVarVal(short   handle,               /* program file handle */
                       short   varNo,       /* variable number (file or DS) */
                       short   varKind,                 /* FILEVAR or DSVAR */
                       WORD    dataSection,               /* 0 or DS number */
                       TpVoid  varADS) /* location of data to write to file */
{
    short  proc = 5;                                     /* function number */
    short  maxVarNo, ecode;
    int    size = 0;
    int    varOff;
    int    stType;                   /* 1 for string transfers 0 for others */
    TpSStr dest = NULL;
    BYTE   maxLen = 0;
    BYTE   charCount;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    CFSLONG*   pLong = (CFSLONG*)varADS;

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed != nothing);
    ASSERT(g_fileInfo[handle].allowed != reading);

/* NB ALL variable descriptions should be entered before the FIRST value
                                                             is transferred */
/* Check handle parameter */

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,proc,BADHANDLE);
        return;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */

/* Check file status */
    if ((pfileInfo->allowed != writing) && (pfileInfo->allowed != editing))
    {
        InternalError(handle, proc, NOTWORE);
        return;
    }
    stType = 0;                                      /* Assume not a string */

/* Check variable kind and number */
    switch (varKind)
    {
        case FILEVAR:
        {
            maxVarNo = pfileInfo->fileHeadP->filVars;
            if ((varNo >= 0) && (varNo < maxVarNo))
            {         /* retrieve place at which to store variable */
                varOff = pfileInfo->FVPoint.nameP[varNo].vSize;
                dest   = pfileInfo->FVPoint.dataP + varOff;
             /* get variable size from its offset and the next one */

                size   = (short)(pfileInfo->FVPoint.
                                        nameP[varNo+1].vSize-varOff);
                                        /* watch out for lstr type */
                if (pfileInfo->FVPoint.nameP[varNo].vType == LSTR)
                {
             /* size allocated is for max number of chars + 2bytes */
                    maxLen = (BYTE)(size-2);
                                 /* maximum characters in transfer */
                    stType = 1;             /* flag string transfer */
                }
                else
                    maxLen = 255;
            }
            break;
        }
        case DSVAR:
        {
            maxVarNo = pfileInfo->fileHeadP->datVars;
                                                  /* same comments as above */
            if ((varNo >= 0) && (varNo < maxVarNo))
            {
                varOff = pfileInfo->DSPoint.nameP[varNo].vSize;
                dest   = pfileInfo->DSPoint.dataP + varOff;
                size   = (short)(pfileInfo->DSPoint.nameP[varNo+1].
                                                       vSize-varOff);
                if (pfileInfo->DSPoint.nameP[varNo].vType == LSTR)
                {
                    maxLen = (BYTE)(size - 2);
                    stType = 1;
                }
                else
                    maxLen = 255;
             }
             break;
        }
        default:
        {
            InternalError(handle,proc,BADKIND);
            return;
        }
    }

    if ((varNo < 0) || (varNo >= maxVarNo))
    {                          /* can now check variable number is in range */
        InternalError(handle,proc,BADVARN);
        return;
    }

/* Now split depending upon the variable type */
    if (varKind == FILEVAR)
    {
/* If this is the first edit made to a file get it ready for the change */
        if ((pfileInfo->allowed == editing) && 
            (pfileInfo->fileHeadP->tablePos != 0))
        {
            ecode = FileUpdate(handle,pfileInfo->fileHeadP);
            if (ecode != 0)
            {
                InternalError(handle,proc,ecode);
                return;
            }
        }
        if (stType == 0)                            /* not a string to transfer */
            CMovel(dest, varADS, (short)size);
        else
        {                             /* varADS is address of C string to transfer
                                         dest is address of LSTRING for storage */
            charCount = (BYTE)F_strlen((TpStr)varADS);
                                               /* how many cahracters in string */
            if (charCount > maxLen)
                charCount = maxLen;
            TransferIn((TpStr)varADS,dest,charCount);
        }
    }
    else
    {                               /* check datasection is in files range */
        if ((pfileInfo->allowed == writing) && (dataSection == 0))
            dataSection = (WORD)(pfileInfo->fileHeadP->dataSecs + 1);

        if ((dataSection < 1) ||               /* Now check for a legal DS */
           ((dataSection > pfileInfo->fileHeadP->dataSecs) &&
            (pfileInfo->allowed != writing)) ||
           ((dataSection > pfileInfo->fileHeadP->dataSecs + 1) && 
            (pfileInfo->allowed == writing)))
        {
            InternalError(handle,proc,BADDS);
            return;
        }

/* If writing and looking back in file, preserve the data header */
        if ((pfileInfo->allowed == writing) &&
            (dataSection <= pfileInfo->fileHeadP->dataSecs))
        {
            CMovel(pfileInfo->extHeadP, pfileInfo->dataHeadP,
                                        pfileInfo->fileHeadP->dataHeadSz);
        }

/* If editing or reading need to load data header of DS specified */
        if (dataSection <= pfileInfo->fileHeadP->dataSecs)
        {
            ecode = GetHeader(handle,dataSection);      /* load data header */
            if (ecode < 0)            /* error code can be read/write error */
            {
                InternalError(handle,proc,ecode);
                goto Restore;
            }
        }

/* If this is the first edit made to a file get it ready for the change */
        if ((pfileInfo->allowed == editing) && 
            (pfileInfo->fileHeadP->tablePos != 0))
        {
            ecode = FileUpdate(handle, pfileInfo->fileHeadP);
            if (ecode != 0)
            {
                InternalError(handle,proc,ecode);
                goto Restore;
            }
        }

        if (stType == 0)                            /* not a string to transfer */
            CMovel(dest, varADS, (short)size);
        else
        {                             /* varADS is address of C string to transfer
                                         dest is address of LSTRING for storage */
            charCount = (BYTE)F_strlen((TpStr)varADS);
                                               /* how many cahracters in string */
            if (charCount > maxLen)
                charCount = maxLen;
            TransferIn((TpStr)varADS,dest,charCount);
        }
           /* If editing update DS header flag so it gets written to file later */

        if (pfileInfo->allowed == editing)
           pfileInfo->DSAltered = 1;

    Restore:             /* Restore header for DS being written if required */
        if ((pfileInfo->allowed == writing) &&
            (dataSection <= pfileInfo->fileHeadP->dataSecs))
        {
            CFSLONG    tableValue;          // First write the data back to disk

            tableValue = GetTable(handle, dataSection);
            if (FileData(handle, pfileInfo->dataHeadP, tableValue,
                         (WORD)pfileInfo->fileHeadP->dataHeadSz) == 0)
                InternalError(handle, proc, WRITERR);

            /* Then copy the old data back into position */
            CMovel(pfileInfo->dataHeadP,pfileInfo->extHeadP,
                                            pfileInfo->fileHeadP->dataHeadSz);
        }
    }

    return;
}                                                       /* end of SetVarVal */


/**************************   Close CFS File *******************************
**
**  Close the CFS file specified by its program handle.
**  It can have been opened by CreateCFSFile or OpenCFSFile.
**  Return 0 if ok or error code if not.
**
*****************************************************************************/

CFSAPI(short) CloseCFSFile(short handle)
{
    short       proc = 21;
    short       flag = 0,
                exchange = 0,
                retval = 0;
    CFSLONG        tabSize, tableValue;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    TpFHead     fileHP;
    TpDHead     dataHP;
    int         index;

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed != nothing);

/* 1. Check handle */

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,proc,BADHANDLE);
        return BADHANDLE;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    if (pfileInfo->allowed == nothing)
    {
        InternalError(handle,proc,BADHANDLE);
        return BADHANDLE;
    }
    fileHP = pfileInfo->fileHeadP;
    dataHP = pfileInfo->dataHeadP;
    retval = 0;             /* return value for ok. Will overwrite if error */

/* 2. File the last DS if necessary */

    if ((pfileInfo->allowed == writing) && (fileHP->fileSz > dataHP->dataSt))
    {
        if (InsertDS(handle,(WORD)(fileHP->dataSecs+1),dataHP->flags) != 0)
            retval = BADINS;
    }
    if (pfileInfo->allowed != reading)
    {
/* 3. Transfer the pointer table first make sure that any altered DS header
      is filed. */
/* Check that the file actually needs updating */

        if ((pfileInfo->allowed == editing) && (fileHP->tablePos != 0))
        {      /* no change has been made. DO NOT write to file. Tidy up.  */
            CFreeAllcn(pfileInfo->extHeadP);
                   /* if file opened for table pointers close and delete it */
            if (pfileInfo->tableP == NULL)
            {
                                                         /* delete CFS file */
                    retval = CCloseAndUnlink(pfileInfo->DOSHdl.p,
                                                    pfileInfo->tempFName);
                #if 0 //def macintosh
                    retval = CCloseAndUnlink(pfileInfo->DOSHdl.p);
                #endif
                
                if (retval < 0)
                    retval = WRDS;
            }
        }
        else
        {
            if (pfileInfo->DSAltered == 1)
            {
                tableValue = GetTable(handle,pfileInfo->thisSection);
                if (FileData(handle,dataHP,tableValue,fileHP->dataHeadSz) == 0)
                    retval = WRITERR;
            }
                                          /* it is to go on end of CFS file */
            if (CLSeek(pfileInfo->DOSHdl.d, fileHP->fileSz, 0) < 0)
                retval = DISKPOS;

            tabSize = fileHP->dataSecs * 4;                /* size of table */
            if (pfileInfo->tableP == NULL)           /* table not in memory */
            {                               /* copy from start of temp file */
                if (CLSeek(pfileInfo->DOSHdl.p,(CFSLONG)0,0) < 0)
                    retval = DISKPOS;
                exchange = TransferTable(fileHP->dataSecs, pfileInfo->DOSHdl.p,
                                                           pfileInfo->DOSHdl.d);
                if (exchange < 0) 
                    retval = exchange;    /* want to return error if not ok */

                    flag = CCloseAndUnlink(pfileInfo->DOSHdl.p,
                            pfileInfo->tempFName);/* delete pointer file */
                #if 0 //def macintosh
                    flag = CCloseAndUnlink(pfileInfo->DOSHdl.p);
                #endif
                if ((flag < 0) && (exchange >= 0))
                    retval = WRDS;             /* delete pointer table file */
            }
            else                  /* table is in memory so write it to file */
                if (CWriteHandle(pfileInfo->DOSHdl.d,(TpStr)pfileInfo->tableP,
                                    (WORD)tabSize) < (WORD)tabSize)
                    retval = WRDS;

/* 4. Store file header */

          if (exchange < 0) 
             fileHP->tablePos = 0;
           /* if transfer failed, enabling to rebuild table on the next run */
          else
             fileHP->tablePos = fileHP->fileSz;
                                   /* table position at end of rest of file */
          fileHP->fileSz = fileHP->fileSz + tabSize;
                                                  /* add table to file size */
          if (FileData(handle, fileHP, (CFSLONG)0, fileHP->fileHeadSz) == 0)
              retval = WRITERR;
          CFreeAllcn(pfileInfo->extHeadP);
/* release space for insert header applies to writing and editing files     */
                                                     /* set the file length */
          if (CSetFileLen(pfileInfo->DOSHdl.d, pfileInfo->fileHeadP->fileSz) != 0)
              retval = BADFL;
        }
    }
    CClose(pfileInfo->DOSHdl.d);                      /* Close the CFS file */
    CFreeAllcn(pfileInfo->fileHeadP);    /* free space allocated for header */
    CFreeAllcn(pfileInfo->dataHeadP);      /* free space allocated for data */
                                                 /* flag handle as not used */
    pfileInfo->allowed = nothing; /* no need to initialise other file prams */
           /* as they will be initialised on opening/creating next CFS file */

/* if the table is in memory free its allocated space */
    if (pfileInfo->tableP != NULL)
        CFreeAllcn(pfileInfo->tableP);

    for (index = 0; index < g_maxCfsFiles; index++)  /* Finally, check tables */
        if (g_fileInfo[index].allowed != nothing)    /* Any files still open? */
            break;                         /* If so, we exit the loop early */

    if (index >= g_maxCfsFiles)     /* If no files open, will clean up memory */
        CleanUpCfs();

    return retval;                                 /* return 0 or errorcode */
}                                                    /* end of CloseCFSFile */

/**************************   Open CFS File *******************************
**
**  Open en exisiting CFS file for reading or editing.
**  Return the program handle if ok (even if cant fulfil table in memory
**  request) or -ve error code.
**  If only problem is inability to allocate memory for the table, this error
**  is returned via the error record and things are set to proceed with the
**  table on disc.
**
*****************************************************************************/

CFSAPI(short) OpenCFSFile(TpCStr  fname,   /* C string containing file name */
                          short  enableWrite,
                                            /* 1 for editing 0 for readonly */
                          short  memoryTable)
                                     /* 1 for table in memory 0 for on disk */

#if 0 //def macintosh
CFSAPI(short) OpenCFSFile(ConstStr255Param   fname,   
                                      short  enableWrite,
                                            /* 1 for editing 0 for readonly */
                                      short  memoryTable,
                                     /* 1 for table in memory 0 for on disk */
                                      short  vRefNum,   
                                      CFSLONG   dirID)
#endif
{
    short       proc;                                    /* function number */
    short       loop,retval;
    WORD        relevantSize;
    CFSLONG        tblSz;
    short       exchange, handle;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    TpFHead     fileHP;
    TpDHead     dataHP,extHP;

    proc  = 13;            /* initialise here to get rid of compiler warning */
    extHP = NULL;                                          /*  do with goto */
    
/* 1. get a program file handle */

    handle = FindUnusedHandle();
    if (handle < 0)
    {
        InternalError(handle,proc,NOHANDLE);
        return NOHANDLE;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */

/* 2. open the file with required status */
                                               /* use loop as temp variable */
    #if defined(_IS_MSDOS_) || defined(_IS_WINDOWS_)
        loop = 0;
        if (COpen(fname,(short)((enableWrite == 0) ? rMode : wMode), &pfileInfo->DOSHdl.d) != 0)
            loop = -1;
    #endif
    #if defined(__linux__) || defined(__APPLE__)
        loop = 0;
        if (COpen(fname,(short)((enableWrite == 0) ? 0 : 2), &pfileInfo->DOSHdl.d) != 0)
            loop = -1;
    #endif
    #if 0 //def macintosh
        loop = COpen(fname,vRefNum,dirID, (short)((enableWrite == 0) ? rMode : wMode));
        if (loop >= 0)
            pfileInfo->DOSHdl.d = loop;
    #endif
    if (loop < 0)
    {
        InternalError(handle,proc,BADOPEN);
        return BADOPEN;
    }

/* file is now open and DOS handle stored */
/* 2a. Check the file size against minimum */
/* Calculate space needed for minimal file file header */
    relevantSize = sizeof(TFileHead) - sizeof(TFilChArr);
    if (CGetFileLen(pfileInfo->DOSHdl.d) < (CFSLONG)relevantSize)
    {
        retval = BADVER;
        goto Close0;                      /* need to close file before exit */
    }

/* file is now open and DOS handle stored */
/* 3. get memory needed initially for file header and data header */
/* Calculate space needed for just a minimalist file header */
    relevantSize = sizeof(TFileHead) - sizeof(TFilChArr);

    pfileInfo->fileHeadP = (TpFHead)CMemAllcn(relevantSize);
    fileHP = pfileInfo->fileHeadP;
    if (fileHP == NULL)                                /* allocation failed */
    {
        retval = NOMEMR;
        goto Close0;                      /* need to close file before exit */
    }

                           /* read the file header into the space allocated */
    if (LoadData(handle,fileHP,(CFSLONG)0,relevantSize) == 0)
    {
        retval = READERR;                                    /* load failed */
        goto Close1;                   /* free space allated and close file */
    }
          /* check the file marker . first whole of current version marker .*/

    if (F_strncmp(fileHP->marker,CEDMARKER,MARKERCHARS) != 0)
    {
        retval = BADVER;
               /* now 2nd check to sort out old version file from foreigner */
        if (F_strncmp(fileHP->marker,PARTMARK,PARTMARKCHARS) == 0)
        {            /* fish out which old version from ASCII of last TpStr */
            retval = fileHP->marker[PARTMARKCHARS];
            retval = (short)(-(retval+BADOLDVER));
                                            /* convert to error code .Yuck! */
        }
        goto Close1;                       /* tidy up and return error code */
    }
    relevantSize = fileHP->fileHeadSz;
                                /* size of full header including chan stuff */
    CFreeAllcn(pfileInfo->fileHeadP);              /* release smaller space */
    fileHP = (TpFHead)CMemAllcn(relevantSize);
                                                      /* get enough for all */
    if (fileHP == NULL)
    {
        retval = NOMEMR;
        goto Close0;                         /* close file only before exit */
    }
    pfileInfo->fileHeadP = fileHP;      /* store pointer to space allocated */

/* 4. read whole of file header including fixed channel info */

    if (LoadData(handle,fileHP,(CFSLONG)0,relevantSize) == 0)
    {
        retval = READERR;
        goto Close1;
    }
                                          /* allocate space for data header */
    pfileInfo->dataHeadP = (TpDHead)CMemAllcn(fileHP->dataHeadSz);
    dataHP = pfileInfo->dataHeadP;
    if (dataHP == NULL)
    {
        retval = NOMEMR;                 /* failure at DS header allocation */
        goto Close1;
    }
    if (enableWrite != 0)
    {          /* if user wants to edit file get extra DS header for insert */
        pfileInfo->extHeadP = (TpDHead)CMemAllcn(fileHP->dataHeadSz);
        extHP = pfileInfo->extHeadP;
        if (extHP == NULL)
        {
            retval = NOMEMR;
            goto Close2;      /* free DS header, file header and close file */
        }
    }
    else
        pfileInfo->extHeadP = NULL;
                                 /* set explicitly to NULL if not allocated */
/* 5. set pointers within allocated space */
                /* file variable descriptions come after fixed channel data */

    pfileInfo->FVPoint.nameP = (TpVDesc)(fileHP->FilChArr+fileHP->dataChans);

/* DS variable descriptions come after the file variables */

    pfileInfo->DSPoint.nameP = pfileInfo->FVPoint.nameP + fileHP->filVars + 1;

/* file variable values follow the DS variable descriptions */

    pfileInfo->FVPoint.dataP = (TpSStr)(pfileInfo->DSPoint.nameP +
                                                         fileHP->datVars + 1);
/* DS variable values follow the DS channel information */

    pfileInfo->DSPoint.dataP = (TpSStr)(dataHP->DSChArr+fileHP->dataChans);
/* 6. get pointer table in memory or in separate file */

    tblSz = 4 * fileHP->dataSecs;
                                          /* first locate table in CFS file */
    if (fileHP->tablePos == 0)                     /* is pointer table lost */
    {
        if (enableWrite == 0)
        {                       /* need to change file access to read/write */
            if (CClose(pfileInfo->DOSHdl.d) != 0)
            {
                retval = WRITERR;                    /* nearset error poss. */
                goto Close2;
            }
            #if defined(_IS_MSDOS_) || defined(_IS_WINDOWS_)
                loop = 0;
                if (COpen(fname, wMode, &pfileInfo->DOSHdl.d) != 0)
                    loop = -1;
            #endif
            #if 0 //def macintosh
                loop = COpen(fname,vRefNum,dirID,wMode);
                if (loop >= 0)
                    pfileInfo->DOSHdl.d = loop;
            #endif
     	    #if defined(__linux__) || defined(__APPLE__)
        	loop = 0;
        	if (COpen(fname, 2, &pfileInfo->DOSHdl.d) != 0)
            	    loop = -1;
    	    #endif           
	    if (loop < 0)
            {
                retval = BADOPEN;
                goto Close2;
            }
        }

        loop = RecoverTable(handle,&tblSz,&fileHP->tablePos,&fileHP->dataSecs,
                                                             &fileHP->fileSz);
       /* if loop is zero the table will be recovered ie.added to the CFS file
                                       and the file header variable updated */
        if (loop < 0)
        {
            retval = loop;             /* return error code of recovertable */
            goto Close2; /* deallocate file and DS header(s) and close file */
        }
        else   /* success in recovering table. write updated header to file */
        {
            if (FileData(handle,fileHP,(CFSLONG)0,fileHP->fileHeadSz) == 0)
            {
                retval = WRITERR;
                goto Close2;
            }
        }
        if (enableWrite == 0)    /* need to change file access back to read */
        {
            if (CClose(pfileInfo->DOSHdl.d) != 0)
            {
                retval = WRITERR;                    /* nearset error poss. */
                goto Close2;
            }
            #if defined(_IS_MSDOS_) || defined(_IS_WINDOWS_)
                loop = 0;
                if (COpen(fname, rMode, &pfileInfo->DOSHdl.d) != 0)
                    loop = -1;
            #endif
            #if 0 //def macintosh
                loop = COpen(fname,vRefNum,dirID,rMode);
                if (loop >= 0)
                    pfileInfo->DOSHdl.d = loop;
            #endif
            #if defined(__linux__) || defined(__APPLE__)
                loop = 0;
                if (COpen(fname, 0, &pfileInfo->DOSHdl.d) != 0)
                    loop = -1;
            #endif
            if (loop < 0)
            {
                retval = BADOPEN;
                goto Close2;
            }
        }
    }
                             /* position CFS file pointer at start of table */
    if (CLSeek(pfileInfo->DOSHdl.d,fileHP->tablePos,0) < 0)
    {
        retval = DISKPOS;                        /* couldnt find table data */
        goto Close2;
    }
                /* if requested and if possible get pointer table in memory */
    if (memoryTable != 0)              /* request for table to be in memory */
        memoryTable  = GetMemTable(handle);             /* try to get space */

    if (memoryTable != 0)             /* space in memory allcated for table */
    {
        if (!LoadData(handle, pfileInfo->tableP, fileHP->tablePos, tblSz))
        {
            retval = READERR;                         /* couldnt read table */
            goto Close2;
        }
    }
    else                                  /* table to be accessed from file */
    {
        pfileInfo->tableP = NULL;       /* flag pointer table not in memory */
        if (enableWrite != 0)
        {                   /* editing required so make temp file for table */
            #if defined(_IS_MSDOS_) || defined(_IS_WINDOWS_)
                                        /* make and save name for temp file */
                TempName(handle,fname,pfileInfo->tempFName, WHOLEFILECHARS+2);
                if (CCreat(pfileInfo->tempFName, rwMode, &pfileInfo->DOSHdl.p) != 0)
                    pfileInfo->DOSHdl.p = (fDef)-1;
            #endif
            #if 0 //def macintosh
            {
                Str31 tempFName;
                TempName(handle,fname,tempFName);
                pfileInfo->DOSHdl.p = CCreat(tempFName,vRefNum,
                                        dirID,rwMode,'trsh','trsh');
            }
            #endif                                           /* create file */
            if (&(pfileInfo->DOSHdl.p) < 0)                   /* create failed */
            {
                retval = BADCREAT;
                goto Close2;
            }
            if (CLSeek(pfileInfo->DOSHdl.p,(CFSLONG)0,0) != 0)
            {
                retval = BADCREAT;
                goto Close3; /* add close and delete of temp file to tiy up */
            }
            exchange = TransferTable(fileHP->dataSecs,pfileInfo->DOSHdl.d,
                                                         pfileInfo->DOSHdl.p);
                                    /* transfer table from CFS to temp file */
            if (exchange < 0)
            {
                retval = exchange;    /* retrun error code of TransferTable */
                goto Close3;
            }
        }
        else
            pfileInfo->DOSHdl.p = pfileInfo->DOSHdl.d;
              /* for reading only access the table from the CFS file itself */
    }
                                        /* all is well now set files status */
    if (enableWrite != 0)
        pfileInfo->allowed = editing;
    else
        pfileInfo->allowed = reading;
    pfileInfo->thisSection = 0xFFFF;
    pfileInfo->DSAltered   = 0;
    return handle;                        /* return the program file handle */

/* 7. Tidy up in case of failure */

    Close3:if (pfileInfo->tableP == NULL)
                                 /* is there likely to be a temp table file */
           {
              if (&(pfileInfo->DOSHdl.p) != &(pfileInfo->DOSHdl.d))
                                          /* make sure it isnt the CFS file */
              {
                      loop = CCloseAndUnlink(pfileInfo->DOSHdl.p,
                                    pfileInfo->tempFName);   /* delete file */
                  #if 0 //def macintosh
                      loop = CCloseAndUnlink(pfileInfo->DOSHdl.p);
                  #endif
              }
           }

    Close2:if (pfileInfo->tableP != NULL)
           {
               CFreeAllcn(pfileInfo->tableP);
               pfileInfo->tableP = NULL;
           }
           if (pfileInfo->extHeadP != NULL)
           {
               CFreeAllcn(pfileInfo->extHeadP);
               pfileInfo->extHeadP = NULL;
           }
           CFreeAllcn(pfileInfo->dataHeadP);

    Close1:CFreeAllcn(pfileInfo->fileHeadP);

    Close0:loop = CClose(pfileInfo->DOSHdl.d);

    InternalError(handle,proc,retval);    /* InternalError gets the details */
    return retval;                                     /* return error code */
}                                                     /* end of OpenCFSFile */

/**************************   Get Gen Info *******************************
**
**  For CFS file specified by it program file handle, return in the time,date
**  and comment arrays the time and date of creation of the CFS file and
**  its general comment.
**  time and date should point to C strings at least 8 chars long. (This will
**  give time/date chars with no NULL.)
**  comment should point to a C string at least COMMENTCHARS+1 (73) chars long.
**  Error return is via error reord.
**
*****************************************************************************/

CFSAPI(void) GetGenInfo(short   handle,              /* progran file handle */
                        TpStr   time,                  /* to read back time */
                        TpStr   date,                  /* to read back date */
                        TpStr   comment)    /* to read back general comment */
{
    short Proc  = 6;
#ifdef _IS_WINDOWS_                                     /* function number */
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo   *pfileInfo;
#endif
    TpFHead     fileHP;

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed != nothing);

/* check handle specified */
    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,Proc,BADHANDLE);
        return;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    if (pfileInfo->allowed == nothing)
        InternalError(handle,Proc,NOTOPEN);
    else
    {
        fileHP = pfileInfo->fileHeadP;
                     /* for the time and date copy the 8 chars only no NULL */
        F_strncpy(time,fileHP->timeStr,8);
        time[8] = '\0';
        F_strncpy(date,fileHP->dateStr,8);
        date[8] = '\0';
        TransferOut(fileHP->commentStr,comment,COMMENTCHARS);
    }
    return;
}                                                      /* end of GetGenInfo */

/***************************   Get File Info *******************************
**
**  For user to get some values from the file header of the file specified
**  by its program file handle.
**  The values are returned via pointers which on entry should point to
**  variables of the correct type.
**  Error return is via error record.
**
*****************************************************************************/

CFSAPI(void) GetFileInfo(short    handle,            /* program file handle */
                         TpShort  channels, /* to return number of channels */
                         TpShort  fileVars,
                                      /* to return number of file variables */
                         TpShort  DSVars,
                                        /* to return number of DS variables */
                         TpUShort dataSections)
                                                   /* to retrun no. of DS's */
{
    short Proc   = 7;
#ifdef _IS_WINDOWS_                                    /* function number */
    TFileInfo    _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    TpFHead      fileHP;

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed != nothing);

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,Proc,BADHANDLE);
        return;
    }
    pfileInfo = &g_fileInfo[handle];
    if (pfileInfo->allowed != nothing)                   /* if file is open */
    {
        fileHP = pfileInfo->fileHeadP;
    /* set the thing that each parameter points to to the value in the file */

        *channels    = fileHP->dataChans;
        *fileVars    = fileHP->filVars;
        *DSVars      = fileHP->datVars;
        *dataSections= fileHP->dataSecs;
    }
    else
        InternalError(handle,Proc,NOTOPEN);     /* file flagged as not open */
    return;
}                                                     /* end of GeTFileInfo */

/**************************   Get Var Desc  *******************************
**
**  To get from the file, specified by its program handle, information about
**  a particular variable description, specified by its varKind (FILEVAR or
**  DSVAR) and its number (alloted when first stored ).
**  The size and type are returned by variable pointers.
**  The units and description are returned by string pointers.
**  The pointer for units should point to a string of at least UNITCHARS+1 (9)
**  chars and that for the description to a string of at least DESCCHARS+1 (21)
**  chars.
**  Any error is returned via the error record.
**
*****************************************************************************/

CFSAPI(void) GetVarDesc(short   handle,              /* program file handle */
                        short   varNo,        /* file or DS variable number */
                        short   varKind,                /* FILEVAR or DSVAR */
                        TpShort varSize,/* to return variable size in bytes */
                        TpDType varType,         /* to return variable type */
                        TpStr   units,       /* to return units as C string */
                        TpStr   description)
                              /* to return variable description as C string */
{
    short      Proc = 8;                                 /* function number */
    short      size;
    short      maxVarNo;
    short      nextOffset;
    TVarDesc   interDesc;
#ifdef _IS_WINDOWS_
    TFileInfo  _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    TpFHead    fileHP;
                                                    /* check program handle */
    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed != nothing);

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,Proc,BADHANDLE);
        return;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    if (pfileInfo->allowed != nothing)
    {
        fileHP = pfileInfo->fileHeadP;
        switch (varKind)
        {
            case FILEVAR:maxVarNo = (short)(fileHP->filVars-1);
                         break;
            case DSVAR  :maxVarNo = (short)(fileHP->datVars-1);
                         break;
            default     :InternalError(handle,Proc,BADKIND);
                         return;
        }
        if ((varNo < 0) || (varNo > maxVarNo))
        {
            InternalError(handle,Proc,BADVARN);
            return;
        }
        if (varKind == FILEVAR)
        {
            interDesc = pfileInfo->FVPoint.nameP[varNo];

/* inside program vsize filed stores variable offset in byte list 
                                                            (not the  size) */
            nextOffset = pfileInfo->FVPoint.nameP[varNo+1].vSize;
        }
        else
        {
            interDesc  = pfileInfo->DSPoint.nameP[varNo];
            nextOffset = pfileInfo->DSPoint.nameP[varNo+1].vSize;
        }
        *varType = interDesc.vType;              /* to return variable type */
                           /* now sort out the size from the stored offsets */
        size = (short)(nextOffset - interDesc.vSize);
      /* if the variable is an lstring deduct 1 so that what is returned
                  is the length of the C string needed to store it, ie. number
                                             of characters + 1 for the NULL */
        if (interDesc.vType == LSTR)
            size = (short)(size-1);
        *varSize = size;                            /* return size in bytes */

      /* Transfer the units and description from their stored LSTRING
                                            format to the C string provided */
        TransferOut(interDesc.varUnits,units,UNITCHARS);
        TransferOut(interDesc.varDesc,description,DESCCHARS);
    }
    else
        InternalError(handle,Proc,NOTOPEN);
    return;
}                                                      /* end of GetVarDesc */

/***************************   Get Var Val  ********************************
**
**  To get from the file, specified by its program file handle, the actual
**  value of a variable specified by its varKind (FILEVAR or DSVAR) and
**  number (allocated when stored).
**  For DSVAR in the case of files opened by CreatsCFSFile the value for the
**           current DS is returned regardless of the parameter dataSection,
**           in the case of files opened by OpenCFSFile the value for the
**           dataSection specified is returned.
**  The value is returned by copying it to the address, varADS, provided.
**  The user must ensure that varADS is pointing to a large enough piece
**  of free memory to which to copy the variable.
**  String variables are returned as NULL terminated C strings (although stored as LSTRING).
**  Any error is returned via the error record.
**
*****************************************************************************/

CFSAPI(void) GetVarVal(short  handle,                /* program file handle */
                       short  varNo,         /* number of variable required */
                       short  varKind,                   /* FILEVAR orDSVAR */
                       WORD   dataSection,        /* may need to specify DS */
                       TpVoid varADS)             /* for return of variable */
{
    short       Proc = 9;                                /* function number */
    short       maxVarNo,size,ecode;
    TpStr       sourceP;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    TpVDesc     pInterDesc,pnext;

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed != nothing);

    if ((handle < 0) || (handle >= g_maxCfsFiles))           /* check handle */
    {
        InternalError(handle,Proc,BADHANDLE);
        return;
    }

    if ((varKind != FILEVAR) && (varKind != DSVAR))      /* check kind too */
    {
        InternalError(handle,Proc,BADKIND);
        return;
    }

    pfileInfo = &g_fileInfo[handle];     /* point to this files information */

/* 1. Check file is open */

    if (pfileInfo->allowed != nothing)
    {
//
// We split into file/DS sections here as the DS var code is getting a bit
// complex
        if (varKind == FILEVAR)
        {
            maxVarNo = (short)(pfileInfo->fileHeadP->filVars-1);
            if ((varNo < 0) || (varNo > maxVarNo))
            {
                InternalError(handle,Proc,BADVARN);
                return;
            }

            pInterDesc = pfileInfo->FVPoint.nameP + varNo;/* vars descript */
            size = pInterDesc->vSize;           /* actually OFFSET for now */
                                    /* set pointer to next var description */
            pnext = pfileInfo->FVPoint.nameP + varNo + 1;

/* point to variable in its data (char) array, using its offset, size */
            sourceP = pfileInfo->FVPoint.dataP + size;
            size = (short)(pnext->vSize - size); /* set size from offsets */

/* 4. move the variable to the location specified */
            if (pInterDesc->vType == LSTR)
            {
                size = (short)(size-2);
                TransferOut(sourceP,(TpStr)varADS,(BYTE)size);
            }
            else
                CMovel(varADS,sourceP,size);
        }
        else        // Here we do mostly the same for DS vars. Sorry for the
        {       // duplication of code, but it keeps things much easier to do.
            maxVarNo = (short)(pfileInfo->fileHeadP->datVars-1);
            if ((varNo < 0) || (varNo > maxVarNo))
            {
                InternalError(handle,Proc,BADVARN);
                return;
            }

// If DS 0 on a new file, set to be next DS.
            if ((pfileInfo->allowed == writing) && (dataSection == 0))
                dataSection = (WORD)(pfileInfo->fileHeadP->dataSecs + 1);
// Now to check the data section number
            if ((dataSection < 1) || 
               ((dataSection > pfileInfo->fileHeadP->dataSecs) &&
                (pfileInfo->allowed != writing)) ||
               ((dataSection > pfileInfo->fileHeadP->dataSecs + 1) &&
                (pfileInfo->allowed == writing)))
            {
                InternalError(handle, Proc, BADDS);
                return;
            }

/* 2. Look to see if a different data header is needed. If it is and we are */
/*    writing, we will need to save the current header. */
            if (dataSection <= pfileInfo->fileHeadP->dataSecs)
            {
                if (pfileInfo->allowed == writing)  // Save the current header ?
                    CMovel(pfileInfo->extHeadP, pfileInfo->dataHeadP,
                                                pfileInfo->fileHeadP->dataHeadSz);
                ecode = GetHeader(handle, dataSection);  /* read new header */
                if (ecode != 0)       /* could have failed on read or write */
                {
                    InternalError(handle,Proc,ecode);
                    goto Restore;
                }
            }

      /* now sort out where the variable is in its array and how many bytes */
            pInterDesc = pfileInfo->DSPoint.nameP + varNo;
            size    = pInterDesc->vSize;          /* This is offset for now */
            pnext   = pfileInfo->DSPoint.nameP + varNo + 1;
            sourceP = pfileInfo->DSPoint.dataP + size;
            size = (short)(pnext->vSize - size);   /* set size from offsets */
                                /* if variable is lstr things are different */
            if (pInterDesc->vType == LSTR)
            {
                size = (short)(size-2);
                TransferOut(sourceP,(TpStr)varADS,(BYTE)size);
            }
            else
                CMovel(varADS,sourceP,size);

    Restore:             /* Restore header for DS being written if required */
            if ((pfileInfo->allowed == writing) &&
                                  (dataSection <= pfileInfo->fileHeadP->dataSecs))
            {
                CMovel(pfileInfo->dataHeadP,pfileInfo->extHeadP,
                                                pfileInfo->fileHeadP->dataHeadSz);
            }
        }
    }
    else
        InternalError(handle,Proc,NOTOPEN);

    return;
}                                                       /* end of GetVarVal */


/**************************   Get File Chan  *******************************
**
**  To get from file, specified by its program file handle, the fixed
**  information on a channel, specified by its number. (ie the order in which
**  it appears in the CFS file.)
**  The character arrays channelName,yUnits and xUnits must be of sufficient
**  size, (DESCCHARS+1 (21), and UNITCHARS+1 (9) characters), for the return
**  of the C string.
**  The other information is returned via pointers which should on entry
**  point to variables of the correct type.
**  Any error is returned via the error record.
**
*****************************************************************************/

CFSAPI(void) GetFileChan(short   handle,             /* program file handle */
                         short   channel,/* chan for which info is required */
                         TpStr   channelName,     /* to return channel name */
                         TpStr   yUnits,       /* to return channel y units */
                         TpStr   xUnits,    /* EqualSpaced ot subsidiary to
                                                             return x units */
                         TpDType dataTypeP,
                                      /* to return 1 of 8 types for channel */
                         TpDKind dataKindP,/* return equalspaced, matrix or
                                                                 subsidiary */
                         TpShort spacing,
                   /* to return bytes between successive values for channel */
                         TpShort other)   /* for matrix data rets next chan */
{
    short       Proc = 10;
#ifndef _IS_WINDOWS_                               /* function number */
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    TpChInfo    pChInfo;
                                                            /* check handle */
    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed != nothing);
    ASSERT(channel >= 0);
    ASSERT(channel <  g_fileInfo[handle].fileHeadP->dataChans);

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,Proc,BADHANDLE);
        return;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    if (pfileInfo->allowed != nothing)
    {                                            /* check channel parameter */
        if ((channel < 0) || (channel >= pfileInfo->fileHeadP->dataChans))
        {
            InternalError(handle,Proc,BADCHAN);
            return;
        }
        pChInfo = pfileInfo->fileHeadP->FilChArr + channel;
                                    /* transfer character arrays for return */
        TransferOut(pChInfo->chanName,channelName,DESCCHARS);
        TransferOut(pChInfo->unitsY,yUnits,UNITCHARS);
        TransferOut(pChInfo->unitsX,xUnits,UNITCHARS);
           /* set values to which return pointers point to values from file */
        *dataTypeP = pChInfo->dType;
        *dataKindP = pChInfo->dKind;
        *spacing   = pChInfo->dSpacing;
        *other     = pChInfo->otherChan;
    }
    else
        InternalError(handle,Proc,NOTOPEN);
    return;
}                                                     /* end of GetFileChan */

/****************************   Get DS Chan  ********************************
**
**  To get from file, specified by its program file handle, the values of the
**  data section channel information. User specifies which channel. Data
**  section can include the one being written if file opened with
**  CreateCFSFile.  The values are returned via pointers.
**  Any error is returned via the error record.
**
*****************************************************************************/

CFSAPI(void) GetDSChan(short   handle,               /* program file handle */
                       short   channel,  /* channel for which info required */
                       WORD    dataSection,        /* dataSection required  */
                       TpLong  startOffset,      /* to return channel start */
                       TpLong  points, /* to return data points for channel */
                       TpFloat yScale,      /* to return yscale for channel */
                       TpFloat yOffset,   /* to return y offset for channel */
                       TpFloat xScale,     /* to return x scale for channel */
                       TpFloat xOffset)   /* to return x offset for channel */
{
    short       Proc = 11;                               /* function number */
    short       ecode;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    TpDsInfo    pChInfo;

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed != nothing);
    ASSERT(channel >= 0);
    ASSERT(channel < g_fileInfo[handle].fileHeadP->dataChans);

/* 1. Check handle */

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,Proc,BADHANDLE);
        return;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
                                                       /* check file status */
    if (pfileInfo->allowed != nothing)
    {
/* 2. Check channel specified against range in file */

        if ((channel < 0) || (channel >= pfileInfo->fileHeadP->dataChans))
        {
            InternalError(handle,Proc,BADCHAN);
            return;
        }
/* 3. Check the data section number. We can access the current section if writing */

        if ((pfileInfo->allowed == writing) && (dataSection == 0))
            dataSection = (WORD)(pfileInfo->fileHeadP->dataSecs + 1);
        if ((dataSection < 1) ||
           ((dataSection > pfileInfo->fileHeadP->dataSecs) &&
            (pfileInfo->allowed != writing)) ||
           ((dataSection > pfileInfo->fileHeadP->dataSecs + 1) && 
            (pfileInfo->allowed == writing)))
        {
            InternalError(handle,Proc,BADDS);
            return;
        }
/* 4. If writing and looking back in file, preserve the data header */

        if ((pfileInfo->allowed == writing) &&
            (dataSection <= pfileInfo->fileHeadP->dataSecs))
        {
            CMovel(pfileInfo->extHeadP,pfileInfo->dataHeadP,
                                            pfileInfo->fileHeadP->dataHeadSz);
        }
/* 5. If editing or reading need to load data header of DS specified */

        if (dataSection <= pfileInfo->fileHeadP->dataSecs)
        {
            ecode = GetHeader(handle,dataSection);      /* load data header */
            if (ecode < 0)            /* error code can be read/write error */
            {
                InternalError(handle,Proc,ecode);
                goto Restore;
            }
        }
/* extract information required from DS header */

        pChInfo      = pfileInfo->dataHeadP->DSChArr + channel;
        *startOffset = pChInfo->dataOffset;
        *points      = pChInfo->dataPoints;
        *yScale      = pChInfo->scaleY;
        *yOffset     = pChInfo->offsetY;
        *xScale      = pChInfo->scaleX;
        *xOffset     = pChInfo->offsetX;

    Restore:             /* Restore header for DS being written if required */
        if ((pfileInfo->allowed == writing) &&
                              (dataSection <= pfileInfo->fileHeadP->dataSecs))
        {
            CMovel(pfileInfo->dataHeadP,pfileInfo->extHeadP,
                                            pfileInfo->fileHeadP->dataHeadSz);
        }
    }
    else
        InternalError(handle,Proc,NOTOPEN);
    return;
}                                                       /* end of GetDSChan */

/**************************   Get Chan Data *******************************
**
** Return data for a single channel from the CFS file.
**
** User specifies file by its program file handle,
**                channel for which data required,
**                dataSection for which data required,
**                data point in the DS and channel from which to start
**                           transfer. (first point is 0)
**                number of data points to transfer. 0 means transfer from
**                        point specified to end of DS, (or end of buffer).
**
** The data are returned by transfer to a buffer starting at dataADS. This
** should point to an area large enough for the transfer.
** areaSize should be set to the size of this destination buffer (in bytes).
** The transfer will not exceed this area even if the points requested would
** take it beyond it.
** Return is 0 if parameters passed specify 0 points or if there is any error,
** (this may have occurred part way through transferring).
** If the whole operation was successful return is the number of data points
** actually transferred, (this may not be the number requested in numPoints).
** Any error is returned via the error record.
**
*****************************************************************************/

CFSAPI(WORD) GetChanData(short  handle,              /* program file handle */
                         short  channel,                /* channel required */
                         WORD   dataSection,                 /* DS required */
                         CFSLONG   firstElement,
                                 /* data point in channel at which to start */
                         WORD   numberElements,            /* points wanted */
                         TpVoid dataADS,          /* address to transfer to */
                         CFSLONG   areaSize)   /* bytes allocated for transfer */
{
    short   SizeOfData[NDATATYPE];          /* sizes in bytes of data types */
    short   ecode;
    short   Proc = 14;                                   /* function number */
    WORD    elementSize,dataOffset,numSecs;
    WORD    retval;
    WORD    bufferSize,spacing,pointsPerBuffer,buffersNeeded,
            bufferLoop,residueElements;
    CFSLONG    filePos,totalPoints,numElements,longSpace;
    TpStr   dBufferP;
//    THandle dummy;
#ifndef _IS_WINDOWS_
    TFileInfo _near *pfileInfo;
#else 
    TFileInfo *pfileInfo;
#endif
    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed != nothing);
    ASSERT(channel >= 0);
    ASSERT(channel < g_fileInfo[handle].fileHeadP->dataChans);

    SizeOfData[INT1] = 1;           /* set sizes in bytes of each data type */
    SizeOfData[WRD1] = 1;
    SizeOfData[INT2] = 2;
    SizeOfData[WRD2] = 2;
    SizeOfData[INT4] = 4;
    SizeOfData[RL4]  = 4;
    SizeOfData[RL8]  = 8;
    SizeOfData[LSTR] = 1;

    retval = 0;           /* initialise return value. no points transferred */
    dBufferP = NULL;
/*
** 1. Check file handle and status
*/
    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,Proc,BADHANDLE);
        return retval;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    if (pfileInfo->allowed != nothing)       /* Are we allowed to do this ? */
    {
/*
** 2. check channel number
*/
        if ((channel < 0) || (channel >= pfileInfo->fileHeadP->dataChans))
        {
            InternalError(handle,Proc,BADCHAN);
            return retval;
        }
/*
** 3. for file just created copy the header for current section to spare place
*/
        if ((pfileInfo->allowed == writing) &&
                                (dataSection<=pfileInfo->fileHeadP->dataSecs))
        {
            CMovel(pfileInfo->extHeadP, pfileInfo->dataHeadP,
                                        pfileInfo->fileHeadP->dataHeadSz);
        }
/*
** 4. check data section and read in its header
*/
        numSecs = pfileInfo->fileHeadP->dataSecs; 
                                              /* shorthand for number of DS */
        if ((pfileInfo->allowed == writing) && (dataSection == 0))
            dataSection = (WORD)(numSecs+1);
                                          /* Convert 0 to currently writing */
        if ((dataSection < 1) || ((pfileInfo->allowed != writing) && 
            (dataSection>numSecs)) || ((pfileInfo->allowed == writing) && 
                                                   (dataSection>(numSecs+1))))
        {
            InternalError(handle,Proc,BADDS);
            return retval;
        }
        if (dataSection <= numSecs)                /* if not the current DS */
        {
            ecode = GetHeader(handle,dataSection);          /* get right DS */
            if (ecode < 0)                               /* header not read */
            {
                InternalError(handle,Proc,ecode);
                return retval;
            }
        }
/*
** 5. Check space and number of points
*/
        elementSize = SizeOfData[(int)pfileInfo->fileHeadP->FilChArr[channel].dType];
        totalPoints = pfileInfo->dataHeadP->DSChArr[channel].dataPoints;
        if (numberElements == 0)
            numElements = totalPoints;
        else
            numElements = numberElements;
/*
** check there are enough points in data section to satisfy request
*/
        if ((numElements + firstElement) >totalPoints)
            numElements = totalPoints - firstElement;
                               /* check that request does not exceed buffer */
        if ((numElements * elementSize) > areaSize)
            numElements = areaSize/elementSize;
                                           /* limit to unsigned short range */
        if (numElements > MAXFORWRD)
            numElements = MAXFORWRD;
/*
** 6. extract channels
*/
        if (numElements == 0)
            goto Restore;
        spacing = pfileInfo->fileHeadP->FilChArr[channel].dSpacing;

//#ifndef WIN32
                                                     /* get as much as poss */
        if ((numElements*spacing) > MAXMEMALLOC)
            bufferSize = (WORD)(MAXMEMALLOC - (MAXMEMALLOC % spacing));
        else
//#endif
            bufferSize = (WORD)numElements * spacing;

                      /* allocated space must have integral set of channels */
        dBufferP = AllocateSpace(&bufferSize, spacing);

        if (bufferSize == 0)
        {
            InternalError(handle,Proc,NOMEMR);
            goto Restore;
        }
        pointsPerBuffer = (WORD)(bufferSize/spacing);
        buffersNeeded   = (WORD)(((numElements-1)/pointsPerBuffer)+1);
                                               /* includes last part buffer */
        longSpace = spacing;             /* keep it like the Pascal version */
        filePos = pfileInfo->dataHeadP->dataSt + pfileInfo->dataHeadP->DSChArr[channel].
                                        dataOffset + (longSpace*firstElement);
        dataOffset = 0;
                      /* deal with complete buffers other than the last one */
        if (buffersNeeded > 1)
        {
            for (bufferLoop = 0;bufferLoop <= (buffersNeeded-2);bufferLoop++)
            {
                if (LoadData(handle,dBufferP,filePos,bufferSize) == 0)
                {
                   InternalError(handle,Proc,READERR);
                   goto Restore;                     /* tidy up before exit */
                }
                ExtractBytes((TpStr)dataADS,dataOffset, dBufferP,
                                         pointsPerBuffer,spacing,elementSize);
                dataOffset = (WORD)(dataOffset + pointsPerBuffer*elementSize);
                filePos    = filePos + bufferSize;
            }
        }
/*
** now whole buffers are done deal with remainder (last Buffer )
*/
        residueElements = (WORD)(((numElements - 1) % pointsPerBuffer) +1);
        if (LoadData(handle,dBufferP,filePos,(WORD)(((
                               residueElements-1)*spacing)+elementSize)) == 0)
        {
            InternalError(handle,Proc,READERR);
            goto Restore;
        }
        ExtractBytes((TpStr)dataADS,dataOffset, dBufferP,
                                         residueElements,spacing,elementSize);
        retval = (WORD)numElements;
                     /* to return number of points successfully transferred */
/*
** 7. tidy up. if writing copy header back
*/
    Restore:
          if ((pfileInfo->allowed == writing) && (dataSection <= numSecs))
          {
              CMovel(pfileInfo->dataHeadP, pfileInfo->extHeadP,
                                           pfileInfo->fileHeadP->dataHeadSz);
          }
          if (dBufferP != NULL)       /* free memory allocated for transfer */
              CFreeAllcn(dBufferP);
    }
    else
        InternalError(handle,Proc,NOTOPEN);
    return retval;
}                                                     /* end of GetChanData */


/*****************************   Read Data  *********************************
**
**  To read data from a data section.
**  CFS File is specified by its program file handle.
**  DS is specified by its number. (even if opened by CreateCFSFile). 1 is
**  first DS.
**  startOffset is byte offset within the DS at which to start.
**  bytes specifies how much to read. (error if this is off end of DS )
**  dataADS points to the start of the region into which to read the data.
**  The user must allocate at least bytes bytes for this.
**  Return is 0 if ok -ve error code if not.
**
*****************************************************************************/

CFSAPI(short) ReadData(short  handle,                /* program file handle */
                       WORD   dataSection,         /* data section required */
                       CFSLONG   startOffset,
                              /* offset into DS from which to start reading */
                       WORD   bytes,             /* number of bytes to read */
                       TpVoid dataADS)    /* start address to which to read */
{
    short       proc = 23;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    TpDHead     phead;
    WORD        numSecs;
    short       ecode,retval;
                                                            /* check handle */

    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed != nothing);

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,proc,BADHANDLE);
        return BADHANDLE;
    }
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */

    if (pfileInfo->allowed == nothing)                      /* check status */
    {
        InternalError(handle,proc,NOTOPEN);
        return NOTOPEN;
    }
                
/* 1. for file opened by CreateCFSFile copy current DS header to spare */

    numSecs = pfileInfo->fileHeadP->dataSecs;
    if ((pfileInfo->allowed == writing) && (dataSection <= numSecs))
    {
        CMovel((TpVoid)pfileInfo->extHeadP,
           (TpVoid)pfileInfo->dataHeadP,pfileInfo->fileHeadP->dataHeadSz);
    }
/* 2. read in header of required DS */

    if ((dataSection < 1) || ((pfileInfo->allowed != writing) && 
        (dataSection > numSecs)) || ((pfileInfo->allowed == writing) &&
                                               (dataSection > (numSecs + 1))))
    {
        InternalError(handle,proc,BADDS);
        return BADDS;
    }
    if (dataSection <= numSecs)            /* read in if not current header */
    {
        ecode = GetHeader(handle,dataSection);
        if (ecode < 0)
        {
            InternalError(handle,proc,ecode);
            return ecode;
        }
    }
                                               /* right header is now there */
    phead = pfileInfo->dataHeadP;                    /* shorten its pointer */
                           /* check the read spec against what is in the DS */
    retval = BADDSZ;                                 /* in case check fails */
    if ((startOffset < 0) || ((startOffset + bytes) > phead->dataSz))
        goto Restore;                               /* clear up before exit */
/* 3. extract data */

    retval = READERR;                                   /* in case it fails */
    if (LoadData(handle,dataADS,phead->dataSt+startOffset,bytes) == 0)
        goto Restore;
    retval = 0;                /* transfer requested successfully completed */

/* 4. if CreateCFSFile used make sure current DS header back in place */

    Restore:
        if ((pfileInfo->allowed == writing) && (dataSection <= numSecs))
            CMovel((TpVoid)pfileInfo->dataHeadP,
                   (TpVoid)pfileInfo->extHeadP,pfileInfo->fileHeadP->dataHeadSz);
        return retval;                                   /* 0 or error code */
}                                                        /* end of ReadData */

/**************************   DS Flag Value *******************************
**
**  Returns initial value of a DS flag, for index supplied from 1 to 16.
**
*****************************************************************************/

CFSAPI(WORD) DSFlagValue(int   nflag)
{
    const   WORD  flagVal[16] = {FLAG0,FLAG1,FLAG2,FLAG3,FLAG4,FLAG5,FLAG6,
                                 FLAG7,FLAG8,FLAG9,FLAG10,FLAG11,FLAG12,
                                 FLAG13,FLAG14,FLAG15};
                                            /* holds DS flags intial values */
    if ((nflag <= 15) && (nflag >=0))
        return flagVal[nflag];
    else 
        return (WORD) 0;
}


/****************************   DS Flags  ***********************************
**
**  Either gets or sets (specified by setIt) the flags for a specified data
**  section within a CFS file, specified by its program file handle, which
**  has been opened with OpenCFSFile.
**
**  The flags may only be set if writing was enabled when the file was opened.
**  On entry pflagSet should point to a variable in the users program of the
**  type TSFlags. If the flags are being set this varaible should have the
**  required settingd.
**
**  Any error is returned via the error record.
**
*****************************************************************************/

CFSAPI(void) DSFlags(short   handle,                 /* program file handle */
                     WORD    dataSection,     /* DS for which flags set/get */
                     short   setIt,        /* 1 means set 0 means get flags */
                     TpFlags pflagSet) /* pointer to users flagset variable */
{
    short       proc = 12;                               /* function number */
    short       ecode;
    TFileInfo   *pfileInfo;


    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed != nothing);

    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,proc,BADHANDLE);
        return;
    }

/* Check file status. ie OpenCFSFile used */

    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    if (pfileInfo->allowed != nothing)
    {                            /* if setting check status is OK */
        if ((setIt == 1) && (pfileInfo->allowed == reading))
        {
            InternalError(handle,proc,NOTWORE);
            return;
        }
                                    /* check datasection is in files range */
        if ((pfileInfo->allowed == writing) && (dataSection == 0))
            dataSection = (WORD)(pfileInfo->fileHeadP->dataSecs + 1);

        if ((dataSection < 1) ||               /* Now check for a legal DS */
           ((dataSection > pfileInfo->fileHeadP->dataSecs) &&
            (pfileInfo->allowed != writing)) ||
           ((dataSection > pfileInfo->fileHeadP->dataSecs + 1) && 
            (pfileInfo->allowed == writing)))
        {
            InternalError(handle,proc,BADDS);
            return;
        }

/* If writing and looking back in file, preserve the data header */
        if ((pfileInfo->allowed == writing) &&
            (dataSection <= pfileInfo->fileHeadP->dataSecs))
        {
            CMovel(pfileInfo->extHeadP, pfileInfo->dataHeadP,
                                        pfileInfo->fileHeadP->dataHeadSz);
        }

/* If editing or reading need to load data header of DS specified */
        if (dataSection <= pfileInfo->fileHeadP->dataSecs)
        {
            ecode = GetHeader(handle,dataSection);      /* load data header */
            if (ecode < 0)            /* error code can be read/write error */
            {
                InternalError(handle,proc,ecode);
                goto Restore;
            }
        }

/* Get or set flags for data section */
        if (setIt == 1)
        {
    /* If this is the first edit made to a file get it ready for the change */
            if ((pfileInfo->allowed == editing) && (pfileInfo->fileHeadP->tablePos != 0))
            {
                ecode = FileUpdate(handle, pfileInfo->fileHeadP);
                if (ecode != 0)
                {
                    InternalError(handle, proc, ecode);
                    goto Restore;
                }
            }
            pfileInfo->dataHeadP->flags = *pflagSet;
            if (pfileInfo->allowed == editing)  /* What to do now ? */
                pfileInfo->DSAltered = 1;  /* flag header has been altered */
        }
        else
            *pflagSet = pfileInfo->dataHeadP->flags;/* return value in file */

    Restore:             /* Restore header for DS being written if required */
        if ((pfileInfo->allowed == writing) &&
            (dataSection <= pfileInfo->fileHeadP->dataSecs))
        {
            if (setIt == 1)                       // Did we modify the data ?
            {                              // If so, we need to write it back
                CFSLONG    tableValue;

                tableValue = GetTable(handle, dataSection);
                if (FileData(handle, pfileInfo->dataHeadP, tableValue,
                                 (WORD)pfileInfo->fileHeadP->dataHeadSz) == 0)
                    InternalError(handle, proc, WRITERR);
            }
            /* Then copy the old data back into position */
            CMovel(pfileInfo->dataHeadP,pfileInfo->extHeadP,
                                            pfileInfo->fileHeadP->dataHeadSz);
        }
    }
    else
        InternalError(handle,proc,NOTWORR);

    return;
}                                                         /* end of DSFlags */

/*****************************   File Error  *******************************
**
**  Error function to collect information on error which is not instantly
**  fatal.
**  Return of error details is via pointers which on entry should point to
**  variables of the right type.
**  The values returned via these pointers refer to the first error encountered
**  since the function was last called. (If no error was encountered since
**  the last time the function was callled the values returned to will refer
**  to the previous error ).
**  Return value is 1 if an error was encountered since the last time the
**  function was called, 0 if not.
**  Side effect.
**  eFound field of global errorInfo is reinitialised to 0.
**
*****************************************************************************/

CFSAPI(short) FileError(TpShort handleNo,        /* to return handle number */
                        TpShort procNo,       /* to return procedure number */
                        TpShort errNo)              /* to return error code */
{
    short retval;

                                /* return current state of error found flag */
    retval    = errorInfo.eFound;
    *handleNo = errorInfo.eHandleNo;
    *procNo   = errorInfo.eProcNo;
    *errNo    = errorInfo.eErrNo;
    errorInfo.eFound = 0;                           /* reset for next error */
    return retval;
}                                                       /* end of FileError */

/**************************  CommitCFSFile  *********************************
**
** CommitCFSFile:- Commits a file to disk, ie ensures that file secure
** on disk. The precise actions will vary with the OS, needless to say,
** but the DOS mechanism is to update the file header on disk, followed
** by duplicating the file handle and closing it, which updates the
** directory information.
**
** fh     The handle for the file
**
** returns zero or -ve error code
**
*****************************************************************************/

CFSAPI(short) CommitCFSFile(short handle)
{                                        
    short       proc      = 16;                          /* function number */
    short       err       = 0;
    short       retCode   = 0;
    short       restore   = 0;
    int         hand      = 0;
    CFSLONG        gtPlace   = 0;
    CFSLONG        endPtr    = 0;
    CFSLONG        oldDataSz = 0;
    CFSLONG        oldFileSz = 0;
    CFSLONG        oldLastDS = 0;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif


    ASSERT(handle >= 0);
    ASSERT(handle < g_maxCfsFiles);
    ASSERT(g_fileInfo[handle].allowed == writing);
    
    if ((handle < 0) || (handle >= g_maxCfsFiles))
    {
        InternalError(handle,proc,BADHANDLE);
        return BADHANDLE;                            /* handle out of range */
    }

    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    if (pfileInfo->allowed != writing)
    {
        InternalError(handle,proc,NOTWRIT);
        return NOTWRIT;                                  /* not for writing */
    }

    if (pfileInfo->fileHeadP->fileSz > pfileInfo->dataHeadP->dataSt)
    {                                            /* Is there a growing DS ? */
        oldDataSz = pfileInfo->dataHeadP->dataSz;
        oldLastDS = pfileInfo->dataHeadP->lastDS;       /* points to header */
        oldFileSz = pfileInfo->fileHeadP->fileSz;
        endPtr    = pfileInfo->fileHeadP->endPnt;

        if (pfileInfo->fileHeadP->dataSecs == 0)
            pfileInfo->dataHeadP->lastDS = 0;
        else
        {
            pfileInfo->dataHeadP->lastDS = 
                               GetTable(handle, pfileInfo->fileHeadP->dataSecs);
            pfileInfo->dataHeadP->dataSz = pfileInfo->fileHeadP->fileSz -
                                                 pfileInfo->dataHeadP->dataSt;
            gtPlace = pfileInfo->dataHeadP->dataSt + 
                              BlockRound(handle, pfileInfo->dataHeadP->dataSz);
                                                      /* pointer to this DS */
            if (!FileData(handle,pfileInfo->dataHeadP,gtPlace,
                                      (WORD)pfileInfo->fileHeadP->dataHeadSz)) 
                retCode = WRDS;                  /* error writing DS header */

            pfileInfo->fileHeadP->endPnt = gtPlace;
            pfileInfo->fileHeadP->fileSz = gtPlace  + 
                                             pfileInfo->fileHeadP->dataHeadSz;
            pfileInfo->fileHeadP->dataSecs++;          /* inc data sections */
            restore  = 1;
        }
    }
    pfileInfo->fileHeadP->tablePos = 0;         /* flag no table on disk... */
    if (!FileData(handle,pfileInfo->fileHeadP,(CFSLONG)0,
                                      (WORD)pfileInfo->fileHeadP->fileHeadSz))
        if (retCode == 0)
            retCode = WRITERR;                 /* error writing file header */

    #if 0 //def macintosh                                        /* MAC specific */
    {
        short   vRefNum = 0;         /* used by Mac routines, MUST BE SHORT */

        err = GetVRefNum(pfileInfo->DOSHdl.d, &vRefNum);
        if (!err)
        {
            err = FlushVol (NULL, vRefNum);
                                   /* flush data and directory info to disk */
            if (err != 0)
            {
               if (retCode == 0)
                   retCode = WRITERR;          /* error writing file header */
            }
        }
        else
        {
            if (retCode == 0)
               retCode = BADHANDLE;
        }
    }
    #endif

    #ifdef _IS_WINDOWS_                        /* shut the file for Windows */
        #ifdef WIN32
            if (!FlushFileBuffers(pfileInfo->DOSHdl.d))
                retCode = BADHANDLE;
        #else
            hand = _dup(pfileInfo->DOSHdl.d);           /* Duplicate the handle */
            if (hand < 0)
            {
                if (retCode == 0)
                   retCode = NOHANDLE;
            }
            else
            {
                err = (short)_close(hand);    /* close dup handle to write thru */
                if (err != 0)
                {
                    if (retCode == 0)
                       retCode = BADHANDLE;
                }
            }
        #endif
    #endif
    
    #ifdef _IS_MSDOS_
        #ifdef LLIO                                     /* DOS LLIO version */
            err = (short) _commit((int)pfileInfo->DOSHdl.d);
            if (err == -1)
            {
                if (retCode == 0)
                    retCode = BADHANDLE;
            }
        #else                                          /* DOS STDIO version */
            hand = _dup(_fileno(pfileInfo->DOSHdl.d));
                                                    /* Duplicate the handle */
            if (hand < 0)
            {
                if (retCode == 0)
                    retCode = NOHANDLE;
            }
            else
            {
                err = (short)fclose(hand);              /   * shut the file */
                if (err != 0)
                {
                    if (retCode == 0)
                       retCode = BADHANDLE;
                }
            }
        #endif
    #endif

    if (restore)
    {
        pfileInfo->fileHeadP->endPnt = endPtr;
        pfileInfo->fileHeadP->fileSz = oldFileSz;
        pfileInfo->dataHeadP->dataSz = oldDataSz;           /* size of data */
        pfileInfo->dataHeadP->lastDS = oldLastDS;
        pfileInfo->fileHeadP->dataSecs--;              /* dec data sections */
    }
    if (retCode != 0)
        InternalError(handle,proc,retCode);

    return retCode;
}

/****************************************************************************/
/*****************                                  *************************/
/*****************   Local  function  definitions   *************************/
/*****************                                  *************************/
/****************************************************************************/

/**************************  Find Unused Handle  *****************************
**
**  Looks through global array of file information to find first member
**  which is not in use. Its index is the local file handle. -1 means none
**  available.
**
*****************************************************************************/

short FindUnusedHandle(void)
{
    short     search;

    if (g_maxCfsFiles <= 0)                   /* Do the initial allocation? */
    {
        ASSERT(g_fileInfo == NULL);

        g_fileInfo = (TFileInfo*)CMemAllcn(INITCEDFILES * sizeof(TFileInfo));
        if (g_fileInfo == NULL)
            return NOMEMR;                       /* Memory allocation error */

        g_maxCfsFiles = INITCEDFILES;  /* Number of files we have space for */
        for (search = 0; search < g_maxCfsFiles; search++)
            g_fileInfo[search].allowed = nothing;   /* Initialise file info */
    }

    search = g_maxCfsFiles - 1;               /* Start index for the search */
    while ((search >= 0) && (g_fileInfo[search].allowed != nothing))
        search--;

    if ((search < 0) && (g_maxCfsFiles < MAXCEDFILES))  /* If table is full */
    {                                              /* but could be enlarged */
        TFileInfo*  pNew;
        int         num;

        num = g_maxCfsFiles * 2;          /* Normally, just double the size */
        if (num > MAXCEDFILES)
            num = MAXCEDFILES;                    /* but limit at the limit */

        pNew = (TFileInfo*)CMemAllcn(num * sizeof(TFileInfo));/* Get memory */
        if (pNew != NULL)                             /* and if it suceeded */
        {
            for (search = 0; search < num; search++)   /* Initialise memory */
                pNew[search].allowed = nothing;        /* and copy the data */
            memcpy(pNew, g_fileInfo, sizeof(TFileInfo) * g_maxCfsFiles);
            CFreeAllcn(g_fileInfo);              /* Discard old memory area */
            g_fileInfo = pNew;            /* and save new pointer and count */
            g_maxCfsFiles = num;
            search = num - 1;         /* and finally, return the last index */
        }
    }

    return search;
}                                                /* end of FindUnusedHandle */


/*****************************  Clean Up Cfs  *****************************
**
**  Tries to clean up the CFS library memory prior to the library exiting
**  from memory or shutting down. Called as part of Windows DLL cleanup.
**
*****************************************************************************/

void CleanUpCfs(void)
{
    int     i;

    for (i = 0; i < g_maxCfsFiles; i++)
        if (g_fileInfo[i].allowed != nothing)/* Attempt a reliable clean up */
            CloseCFSFile((short)i);
    
    if (g_fileInfo != NULL)
        CFreeAllcn(g_fileInfo);     /* and final memory release and tidy up */
    g_fileInfo = NULL;
    g_maxCfsFiles = 0;
}

/**************************  Temp Name  *************************************
**
**  Make a temporary file name encoding the file handle 
**
****************************************************************************/

#if 0 //def macintosh
short TempName(short handle,ConstStr255Param /*name*/,ConstStr255Param str2)
{
    sprintf(str2,"CFS Temporary %d",handle);    /* encode handle into string */
    c2pstr(str2);
}
#endif

#if defined(_IS_MSDOS_) || defined(_IS_WINDOWS_)
short TempName(short handle, TpCStr name, TpStr str2, unsigned str2len)
{
    short   pathstart;
    short   pathend = 0;
    short   search  = 0;
    char    fname[WHOLEFILECHARS];   /* To get near variable holding string */
    if (strlen(name) < WHOLEFILECHARS)
        F_strcpy(fname, name);                      /* Get filename in near var */
    pathstart = 0;
    while (isspace(fname[pathstart]))
        pathstart++;                                  /* first proper TpStr */
    pathend = (short)(pathstart - 1);
    search  = pathstart;              /* start at proper start of file name */
    while (search <= (short)F_strlen(fname))
        {                                     /* scan fname for end of path */
            if ((fname[search] == '\\') || (fname[search] == ':'))
                pathend = search;
            search++;
        }
                            /* use path if any to start temporary file name */
    if (pathend >= pathstart)
    {
        F_strncpy(str2,fname+pathstart,pathend+1-pathstart);   /* copy path */
        str2[pathend+1-pathstart] = '\0';                       /* add null */
    }
    else {
        if (str2len > 0)
            F_strcpy(str2,"");                  /* or initialise to null string */
    }
    F_strcat(str2,"CFS(TMP).");       /* ad standard part of temp file name */
    _itoa(handle,gWorkStr,10);                 /* encode handle into string */
    F_strcat(str2,gWorkStr);            /* add handle to make complete name */
    return 0;
}                                                        /* end of TempName */


#endif

#if defined(__linux__) || defined(__APPLE__)
short TempName(short handle, TpCStr name, TpStr str2, unsigned str2len) {
    if (str2len > 12)
	F_strcpy(str2,"CFSTMPXXXXXX");
    return (short)mkstemp(str2);
}
#endif

/**************************  Set Sizes  *************************************
**
**  Look at array of variable descriptions and using the types add up how
**  much space will be needed for all the actual variables in the array
**  Return computed space (bytes) or -1 for error.
**
*****************************************************************************/

short SetSizes(TpCVDesc theArray, TpShort offsetArray, short numVars)
{
    short       SizeOfData[NDATATYPE];
    short       search,size,runningTotal,errFlag;
    TpShort     pOffsets;
    TpVDesc     pVarDesc;
                                                /* set sizes for data types */
    SizeOfData[INT1] = 1;
    SizeOfData[WRD1] = 1;
    SizeOfData[INT2] = 2;
    SizeOfData[WRD2] = 2;
    SizeOfData[INT4] = 4;
    SizeOfData[RL4]  = 4;
    SizeOfData[RL8]  = 8;
    SizeOfData[LSTR] = 0;

    errFlag = 0;                                                /* No error */
    runningTotal = 0;                                         /* initialise */
    for (search = 0;search < numVars;search++)
    {
        pOffsets = offsetArray + search;
        pVarDesc = (TpVDesc)theArray + search;   /* Ok to remove const here */
                                                  /* check type is in range */
        if ((pVarDesc->vType < 0) || (pVarDesc->vType >= NDATATYPE))
            return -1;
        size = SizeOfData[(int)pVarDesc->vType];
        if (pVarDesc->vType == LSTR)
            size = (short)(pVarDesc->vSize + 1);          /* allow for NULL */
        if ((size < 0) || (size >= MAXSTR))
            errFlag = 1;
        *pOffsets = runningTotal;    /* save start offset for each variable */
        runningTotal = (short)(runningTotal+size);
    }
    if (errFlag) 
        return -1;
    else
        return runningTotal;
}                                                        /* end of setSizes */

/********************  Character handling functions  ************************
**
**  TransferIn  takes C string format (users storage)
**              transfers to CFS storage format (program storage far heap)
**  TransferOUT takes CFS storage format (program storage far heap)
**              transfers to C string format (users storage).
**  NB The length parameter is VITAL for checking strings do not overflow
**     their char arrays.
**  In both cases the length refers to the number of characters transferred
**  neither the length byte nor the NULL are counted in this.
**
****************************************************************************/

/**************************  Transfer In  ***********************************
**
**  Transfer as much as possible, but not more than max characters, of the
**  NULL terminated C string old, to the character array new which imitates
**  an LSTRING (NULL terminated)
**  NB Since the allocated length of new cannot be checked the function relies
**     on the parameter max to prevent overflow of new which should be
**     declared as char[2+max].
**     new should finish up with new[0] containing the chracter count (0 to 255)
**     then new[1] to new[new[0]] containig the characters and new[new[0]+1]
**     containing the NULL 
**
*****************************************************************************/

void TransferIn(TpCStr olds, TpStr pNew, BYTE max)
{
    BYTE     lengths;
    short    loop;

    lengths = (BYTE)F_strlen(olds); 
                                         /* how many characters to transfer */
    if (lengths > max)
        lengths = max;                       /* check against specified max */
    pNew[0] = lengths;
                /* assigning an unsigned char to char preserves bit pattern */
                                        /* now copy characters on at a time */
    for (loop = 0;loop < (short)lengths;loop++)
        pNew[loop+1] = olds[loop];
    pNew[lengths+1] = '\0';                               /* put NULL on end */
}                                                      /* end of TransferIn */

/*************************  Transfer Out  ***********************************
**
**  Transfer as much as possible but not more than max characters of the
**  Pascal like LSTRING + NULL to an ordinary NULL terminated C string.
**  NB new must have been declared at least char[max+1].
**
*****************************************************************************/

void TransferOut(TpCStr olds, TpStr pNew, BYTE max)
{
    BYTE     lengths;
    short    loop;

    lengths = olds[0];/* get the number of chars in the Pascal type LSTRING */
    if (lengths > max)
        lengths = max;                 /* dont transfer more than max chars */
    for (loop = 0;loop < (short)lengths;loop++)
        pNew[loop] = olds[loop+1];             /* transfer 1 char at a time */
    pNew[lengths] = '\0';                            /* terminate with NULL */
}                                                     /* end of transferOut */

/*************************  Set Var Descs  **********************************
**
**  NB the useArray will contain its string data as C strings. When strings
**     get put into the data structure array in this function they are 
**     converted to the LSTRING+NULL format required.
**  This function sets the data structure variable descriptions and the
**  pointers to the corresponding allocated storage positions.
**
*****************************************************************************/

void SetVarDescs(short     numOfVars,    /* number of variable descriptions */
                 TPointers varPoint,  /* pointers to starts of descriptions to
                                            be done and data storage places */
                 TpCVDesc  useArray,            /* array of values to go in */
                 TpShort   offsets,
                   /* array of computed (setSizes) offsets for data storage */
                 short     vSpace)                /* size of data variable area */
{
    short       setloop;
    TpVDesc     p;

/* for storage in program use varaible description structs */

    for (setloop = 0;setloop < numOfVars;setloop++)
    {
        p = varPoint.nameP + setloop;
                                    /* point to variable description to set */
        p->vSize = offsets[setloop];
                         /* vsize to hold offset in data array for variable */
        p->vType = useArray[setloop].vType;
                                     /* copy type from description provided */
        p->zeroByte = 0; 
                       /* zero extra byte added for MS Pascal compatibility */
        TransferIn(useArray[setloop].varUnits,p->varUnits,UNITCHARS);
                                                            /* units string */
        TransferIn(useArray[setloop].varDesc,p->varDesc,DESCCHARS);
                                                             /* name string */
    }
   /* after the ordinary variable descriptions comes the system one which
      stores the size of the data space neede for all the variables */

    varPoint.nameP[numOfVars].vSize = vSpace;
                           /* now initialise all the variable space to zero */
    for (setloop = 0;setloop < vSpace;setloop++)
        varPoint.dataP[setloop] = 0;
}                                                     /* end of SetVarDescs */

/*************************   Block Round   **********************************
**
**  Round up raw (which represents a storage space) to a whole number of blocks
**  The block size is to be found in the file info for the program file handle
**  specified and is usually 1 or 512.
**
*****************************************************************************/

CFSLONG BlockRound(short handle,CFSLONG raw)
{
    CFSLONG retval;
    short dbs;

    dbs = g_fileInfo[handle].fileHeadP->diskBlkSize;
                                                 /* fish out the block size */
    if (dbs == 1)
        retval = raw;
    else
        retval = ((raw + dbs - 1) / dbs) * dbs;
    return retval;
}                                                      /* end of BlockRound */

/*************************  Internal Error  *********************************
**
**  Set the global errorInfo values for user to look at, UNLESS it has already
**  been used.
**
*****************************************************************************/

void InternalError(short handle,short proc,short err)
{
    if (errorInfo.eFound == 0)                         /* no previous error */
    {
        errorInfo.eFound    = 1;
        errorInfo.eHandleNo = handle;
        errorInfo.eProcNo   = proc;
        errorInfo.eErrNo    = err;
    }
}                                                   /* end of Internalerror */

/****************************  File Data  ***********************************
**
**  Write data to the CFS file corresponding to the program handle.
**  Return 1 if ok 0 if not.
**
*****************************************************************************/

short FileData(short  handle,                                /* file handle */
               TpVoid startP,       /* start address from which to transfer */
               CFSLONG   st,        /* file position to which to start writing */
               CFSLONG   sz)                    /* number of bytes to transfer */
{
    WORD    res;
    TpStr   pDat = (TpStr)startP;

    if ((st < 0) || (st >= MAXLSEEK))
        return 0;
        
    if (CLSeek(g_fileInfo[handle].DOSHdl.d, st, 0) < 0) /* set file pointer */
        return 0;

    if (sz == 0)         /* dont try to write 0 bytes it will truncate file */
        return 1;

    while (sz > 0)                                 /* Loop to read the data */
    {
        WORD    wr;

        if (sz > 64000)                 /* How much to read this time round */
            wr = 64000;
        else
            wr = (WORD)sz;

        res = CWriteHandle(g_fileInfo[handle].DOSHdl.d, pDat, wr); /* do write */
        if (res == wr)                              /* keep going if all OK */
        {
            sz -= wr;                                   /* bytes left to do */
            pDat += wr;                            /* pointer to write data */
        }
        else 
            return 0;                    /* Return failed if not written OK */
    }

    return 1;
}                                                        /* end of FileData */

/****************************  Load Data  ***********************************
**
**  Read data from the CFS file corresponding to the program handle.
**  Return 1 if ok 0 if not.
**
*****************************************************************************/

short LoadData(short    handle,                              /* file handle */
               TpVoid   startP,         /* address in memory to transfer to */
               CFSLONG     st,    /* file position from which to start reading */
               CFSLONG     sz)                  /* number of bytes to transfer */
{
    WORD    res;
    TpStr   pDat = (TpStr)startP;

    if ((st < 0) || (st >= MAXLSEEK))
        return 0;

    if (CLSeek(g_fileInfo[handle].DOSHdl.d,st,0) < 0 )  /* set file pointer */
        return 0;

    while (sz > 0)                                 /* Loop to read the data */
    {
        WORD    wr;

        if (sz > 64000)                 /* How much to read this time round */
            wr = 64000;
        else
            wr = (WORD)sz;

        res = CReadHandle(g_fileInfo[handle].DOSHdl.d, pDat, wr);/* do read */
        if (res == wr)                              /* keep going if all OK */
        {
            sz -= wr;                                   /* bytes left to do */
            pDat += wr;                             /* pointer to read data */
        }
        else 
            return 0;                       /* Return failed if not read OK */
    }

    return 1;
}                                                        /* end of LoadData */

/***************************  Get Table  ***********************************
**
**  Return offset in CFS file for the section,position .
**  if the table has been read in then the pointer tableP in g_fileInfo will 
**  point to it and the entry  at (position-1) will correspond to
**  the offset for the data section,postiton.
**  If the table is not there the offset needs to be read from the
**  temporary file.
**
*****************************************************************************/

CFSLONG GetTable(short handle, WORD position)
{
    CFSLONG        DSPointer;                       /* return for offset value */
    CFSLONG        filePosn;    /* position in temporary file for offset value */
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif

    pfileInfo = &g_fileInfo[handle];     /* point to this files information */

/* if table has been read in retrieve offset from it */

    if (pfileInfo->tableP != NULL)
        DSPointer = pfileInfo->tableP[position-1];
    else
    {                                     /* get it from the temporary file */
        filePosn = (position-1)*4; /* offset for each DS is CFSLONG ie 4 bytes */
        if (pfileInfo->allowed == reading)
            filePosn=filePosn + pfileInfo->fileHeadP->fileSz -
                                             4*pfileInfo->fileHeadP->dataSecs;

       CLSeek(pfileInfo->DOSHdl.p, filePosn, 0);  /* move to place for read */
       CReadHandle(pfileInfo->DOSHdl.p, (TpStr)&DSPointer, 4);   /* 4 bytes */
    }
    return DSPointer;
}                                                        /* end of GetTable */

/***************************  Get Header  ***********************************
** 
**  Reads header of DS requested from CFS file. If current header
**  has been altered it is written to the CFS file before being replaced
**  by the new one.
**  Return value is 0 if all ok and errorcode READERR if failed on reading
**  new section or WRITERR if failed when writing current header
**
*****************************************************************************/
short GetHeader(short handle, WORD getSection)
{
    CFSLONG            tableValue;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif

    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    if (pfileInfo->thisSection != getSection)   /* only get it if different */
    {
        if ((pfileInfo->DSAltered != 0) &&  /* write changed header to file */
            (pfileInfo->allowed != writing))          /* But not if writing */
/*
** can only have DSAltered=1 if GetHeader has already been called and set
** thisSection
*/
        {
            tableValue = GetTable(handle, pfileInfo->thisSection);
            if (FileData(handle, pfileInfo->dataHeadP, tableValue,
                                 (WORD)pfileInfo->fileHeadP->dataHeadSz) != 0)
                pfileInfo->DSAltered = 0;/* If saved can clear altered flag */
            else
                return WRITERR;     /* error if couldnt save current header */
        }
        tableValue = GetTable(handle, getSection);
        if (LoadData(handle,pfileInfo->dataHeadP, tableValue,
                                 (WORD)pfileInfo->fileHeadP->dataHeadSz) == 0)
            return READERR;

        if (pfileInfo->allowed != writing)         /* Keep track of actions */
            pfileInfo->thisSection = getSection;
        else
            pfileInfo->thisSection = 0xFFFF;
    }
    return 0;
}                                                       /* end of getHeader */

/***************************  Store Table  ***********************************
** 
**  Put a CFSLONG offset value into the pointer table or file correspopnding to
**  the entry for data section position.
**
*****************************************************************************/

void StoreTable(short handle, WORD position, CFSLONG DSPointer)
{
    CFSLONG                filePosn;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif

    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
/*
** look to see if pointer table has been read into memory
*/
    if (pfileInfo->tableP != NULL)
    {
        if (position > pfileInfo->fileHeadP->dataSecs)
        {
            TpLong      pNew;
            TpVoid      pOld = pfileInfo->tableP;  /* Previous data pointer */

            pNew = (TpLong)CMemAllcn(position*4);
            if (pNew != NULL)
            {
                CMovel(pNew, pfileInfo->tableP, position*4);
                pfileInfo->tableP = pNew;
                CFreeAllcn(pOld);
            }
            else
            {
/*                if (pfileInfo->tableP == NULL)
                    return NOMEMR;

                BUGBUG what the hell do I do here */
            }
        }
        pfileInfo->tableP[position-1] = DSPointer;
    }
    else
    {                                         /* table is in temporary file */
        filePosn = (CFSLONG)(position-1)*4;     /* each entry occupies 4 bytes */
        CLSeek(pfileInfo->DOSHdl.p,filePosn,0);
        CWriteHandle(pfileInfo->DOSHdl.p,(TpStr)&DSPointer,4);
    }
}                                                      /* end of storeTable */

/***************************  Recover  Table  ********************************
** 
**  Re-build table holding file offsets of data section headers in the event
**  of table size and number of data sections not tallying.
**  Uses the information in the file header and individual data section
**  headers.
**
*****************************************************************************/

short RecoverTable(short    handle,                  /* program file handle */
                   TpLong   relSize,     /* location of table size variable */
                   TpLong   tPos,       /* location of table position variable
                                                    this is for return only */
                   TpUShort dSecs,
                                     /* location of number of DS's variable */
                   TpLong   fSize)        /* location of file size variable */
{
    WORD        foundSecs,expSecs;
    CFSLONG        secPos,tablePos,maxSecPos,fileSz,tableSz,maxSecs;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif
    TpFHead     fileHP;
    short       retval;
    THandle     pHandle = 0;
    
    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    fileHP    = pfileInfo->fileHeadP;
/* extract values from function parameter pointers */

    tableSz   = *relSize;                         /* expected size of table */
    fileSz    = *fSize;                               /* expected file size */
    expSecs   = *dSecs;                 /* expected number of data sections */
    secPos    = fileHP->endPnt;       /* start place of highest numbered DS */
    foundSecs = 0;                                            /* initialise */
    maxSecPos = secPos;      /* initialise for finding last section in file */

    #if 0
    if (tableSz > MAXMEMALLOC)
        return NOMEMR;                         /* largest memory allocation */
    #endif
    pfileInfo->tableP = (TpLong)CMemAllcn(tableSz);
                                                /* allocate space for table */
    if (pfileInfo->tableP == NULL)
        return NOMEMR;

    maxSecs = tableSz/4; 
                      /* memory has been allocated for this number of longs */
    if (maxSecs > MAXNODS)
        maxSecs = MAXNODS;                               /* dont exceed max */

    retval = READERR;
    while (secPos > 0)               /* work back through the data sections */
    {   
        foundSecs = (WORD)(foundSecs + 1);
                                           /* count how many sections found */
        if (foundSecs > (WORD)maxSecs)                  /* fail if too many */
        {
            retval = XSDS;
            goto Restore;
        }
        pfileInfo->tableP[expSecs-foundSecs] = secPos;
                            /* store the one found starting at end of table */
                                   /* read the DS header into the file info */
        if (LoadData(handle,pfileInfo->dataHeadP,secPos,
                                               (WORD)fileHP->dataHeadSz) == 0)
            goto Restore;
        secPos = pfileInfo->dataHeadP->lastDS;     /* extract from header just
                     read in the file position of the previous data section */
        if (secPos > maxSecPos)
            maxSecPos = secPos;    /* keep a trace of largest offset found. */
    }      /* secPos = 0 means that 'previous DS' is in fact the fileheader */

    tableSz  = 4*foundSecs;                     /* new value for table size */
    tablePos = maxSecPos + fileHP->dataHeadSz;   /* end of data header area */
    fileSz   = tablePos + tableSz;                         /* new file size */
                               /* move file pointer to table start position */
    if (CLSeek(pfileInfo->DOSHdl.d,tablePos,0) < 0)
        goto Restore;
    retval = WRITERR;
                                                     /* write table to file */
    if (CWriteHandle(pfileInfo->DOSHdl.d,(TpStr)&pfileInfo->tableP[expSecs-
                                     foundSecs],(WORD)tableSz)< (WORD)tableSz)
        goto Restore;
    expSecs = foundSecs;
    retval  = 0;                                                   /* all ok */
            /* prepare for new value returns via pointer function arguments */
    *relSize = tableSz;
    *tPos    = tablePos;
    *dSecs   = expSecs;
    *fSize   = fileSz;

    Restore:CFreeAllcn(pfileInfo->tableP);
    pfileInfo->tableP = NULL;

    return retval;                                       /* error code or 0 */
}                                                    /* end of recovertable */

/***************************  Transfer Table  ********************************
** 
**  Transfer table from current position of file rdHdl to current position
**  of file wrHdl in 512 (or 4) byte blocks.
**  Return zero or error code
**
*****************************************************************************/

short TransferTable(WORD sects, fDef rdHdl, fDef wrHdl)
{
    short     retval;
    TpShort   transArr;                             /* array 512 bytes long */
    WORD      index,ntran,transSize, lastSize;
//    THandle   pHandle = NULL;
    CFSLONG      lTranBuf;                       /* last ditch transfer buffer */

    retval   = 0;                                 /* return value if all ok */
    if (sects == 0)
        return retval;                                   /* Cover our backs */
    transSize = 512;                            /* what we want to transfer */
    transArr = (TpShort)CMemAllcn(transSize);
    if (transArr == NULL)      /* have another go with little 4 byte blocks */
    {
        transSize = 4;
        transArr  = (TpShort)&lTranBuf;        /* Pointer to stack variable */
    }
    ntran = (WORD)(sects/(transSize/4));
                                      /* number of whole blocks to transfer */
    for (index = 1; index <= ntran; index++)
    {
        if (CReadHandle(rdHdl,(TpStr)transArr,transSize) < transSize)
        {   
            retval = READERR;                          /* retrun error code */
            goto Close1;
        }
        if (CWriteHandle(wrHdl,(TpStr)transArr,transSize) < transSize)
        {
            retval = WRITERR;
            goto Close1;
        }
    }
    if (ntran * (transSize/4) < sects)   /* part of block still to transfer */
    {
        lastSize = (WORD)((sects * 4) - (ntran * transSize));
        if (CReadHandle(rdHdl,(TpStr)transArr, lastSize) < lastSize)
        {   
            retval = READERR;
            goto Close1;
        }

        if (CWriteHandle(wrHdl,(TpStr)transArr, lastSize) < lastSize)
            retval = WRITERR;
    }
    Close1:if (transSize > 4)         /* Don't free if using local variable */
               CFreeAllcn(transArr);

    return retval;
}                                                   /* end of TransferTable */

/***************************  GetMem Table  *********************************
** 
**  Try to alloacte space for pointer table.
**  handle is program file handle, proc is function number from which called.
**  Return 1 if alloaction ok 0 if not.
**
*****************************************************************************/

short GetMemTable(short handle)
{
    CFSLONG            tableSz;
#ifdef _IS_WINDOWS_
    TFileInfo   _near *pfileInfo;
#else
    TFileInfo *pfileInfo;
#endif

    pfileInfo = &g_fileInfo[handle];     /* point to this files information */
    tableSz   = 4*(CFSLONG)pfileInfo->fileHeadP->dataSecs;
                                                /* 4 bytes per data section */
#if 0
    if (tableSz > MAXMEMALLOC)
        return 0;                                                 /* failed */
#endif
    pfileInfo->tableP = (TpLong)CMemAllcn(tableSz);
    if (pfileInfo->tableP == NULL)
        return 0;                                                 /* failed */

    return 1;                                                    /* success */
}                                                     /* end of GetMemTable */


/***************************  Allocate Space  ********************************
** 
**  Function for alloacting space which must be an integral number of steps.
**  On entry sizeP points to the variable holding the size of the space
**                 (in bytes) requested.
**           steps is the size in bytes of which the allocated space must be
**                 an integral multiple.
**  On exit the variable pointed to by sizeP is the size of the space
**           actually allocated.
**  The return value is a pointer to the allocated space, NULL if allocation
**  failed.
**
*****************************************************************************/

TpStr AllocateSpace(TpUShort sizeP, WORD steps)
{
    WORD     buffSize;
    TpStr    buffP;

    buffSize = *sizeP;                              /* required buffer size */
    buffP    = NULL;
    buffSize = (WORD)((buffSize/steps)*steps);
                                    /* truncate to integral number of steps */
    while ((buffP == NULL) && (buffSize > 0))
    {
        buffP = (TpStr)CMemAllcn(buffSize);                /* do allocation */
        if (buffP == NULL)             /* want to have another go if failed */
        {
/*
** try half the size or if this isnt integral number of steps subtract steps
*/
            if (((buffSize/2) % steps) == 0)
                buffSize = (WORD)(buffSize/2);
            else
                buffSize = (WORD)(buffSize - steps);
        }
    }
                              /* prepare to return the final size allocated */
    *sizeP = buffSize;

    return buffP;
}                                                   /* end of AllocateSpace */

/***************************  Extract Bytes  ********************************
** 
**  Function to extract single channel array from array of interleaved data.
**  On entry destP points to the start of array to which to write the
**           single channel data.
**           dataOffset is the offset in that array at which to start writing.
**           srcP points to the start of the interleaved data.
**           points is the number of data points to transfer.
**           spacing is the number of bytes between data points of
**                   the channel required in the source array.
**           ptSz is the number of bytes per point.
**
*****************************************************************************/

void ExtractBytes(TpStr destP,WORD dataOffset,TpStr srcP,
                                           WORD points,WORD spacing,WORD ptSz)
{
    WORD index;

    destP = destP + dataOffset;
    for (index = 0;index < points;index++)
    {
        CMovel(destP,srcP,ptSz);                        /* Transfer 1 point */
        srcP  = srcP  + spacing;            /* move to next point in source */
        destP = destP + ptSz;               /* move to place for next point */
    }
}                                                    /* end of ExtractBytes */

/***************************   File Update  ********************************
** 
**  Called when access to file is editing, the file has not yet been changed
**  (tablePos != 0) but a real change is about to be made.
**  It effectively removes the pointer table from the file and updates the
**  file header accordingly. Returns 0 or error code.
**
*****************************************************************************/

short FileUpdate(short    handle,                    /* program file handle */
                 TpFHead  fileHP)       /* address in memory of file header */
{

/*
** 1. Flag that table pointers removed from file
*/
    fileHP->tablePos = 0;
/*
** 2. Effect removal of memory table by not including it in file length
*/
    fileHP->fileSz = fileHP->fileSz - 4 * fileHP->dataSecs;
                                                          /* 4 bytes per DS */
/*
** 3. Write this information to the file as new file header.
*/
    if (FileData(handle, fileHP, 0L, fileHP->fileHeadSz) == 0)
        return WRITERR;
    else
        return 0;
}                                                      /* end of FileUpdate */

/********************************   E N D  **********************************/
