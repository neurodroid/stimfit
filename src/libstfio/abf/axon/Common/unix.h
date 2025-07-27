// Windows macros to compile on Unix machines.
// Taken from MSDOS.H and from various wine headers.
// Only the absolute minimum has been integrated.
// 2007, CSH, University of Freiburg

// Error return from SetFilePointer()
#ifndef INVALID_SEEK_VALUE
#define INVALID_SEEK_VALUE    (0xFFFFFFFF)
#endif
#define FILE_NULL NULL
#define IDS_ENOMESSAGESTR               4

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
    #include <windows.h>
    #include <stdio.h>
    typedef HANDLE FILEHANDLE;
    #include "resource.h"
    #define __LITTLE_ENDIAN__
#else
    #ifdef __APPLE__
         #include <machine/endian.h>
    #elif defined(__MINGW32__)     
         #define __LITTLE_ENDIAN__
    #else
         #include <endian.h>
    #endif
    #ifndef INVALID_HANDLE_VALUE
    #define INVALID_HANDLE_VALUE ((HANDLE)0xFFFFFFFF)
    #endif

#include "../AxAbfFio32/AxAbffio32.h"
    
#include <stdio.h>
    typedef FILE* FILEHANDLE;

//
// Commonly used typedefs & constants from windows.h.
//
typedef unsigned char  BYTE;
typedef unsigned short WORD;
typedef unsigned ABFLONG  DWORD;
typedef unsigned ABFLONG *LPDWORD;
typedef unsigned int   UINT;
typedef int            INT;
typedef int            BOOL;
typedef long          *PLONG;
typedef unsigned char *LPBYTE;
typedef void          *LPVOID;
typedef const void    *LPCVOID;

//
// Basics
//

#ifndef VOID
#define VOID void
typedef char CHAR;
typedef short SHORT;
typedef long LONG;
#endif

//
// UNICODE (Wide Character) types
//

#ifndef _MAC
typedef wchar_t WCHAR;    // wc,   16-bit UNICODE character
#else
// some Macintosh compilers don't define wchar_t in a convenient location, or define it as a char
typedef unsigned short WCHAR;    // wc,   16-bit UNICODE character
#endif

typedef WCHAR *PWCHAR;
typedef WCHAR *LPWCH, *PWCH;
typedef const WCHAR *LPCWCH, *PCWCH;
typedef WCHAR *NWPSTR;
typedef WCHAR *LPWSTR, *PWSTR;

typedef const WCHAR *LPCWSTR, *PCWSTR;

//
// ANSI (Multi-byte Character) types
//
typedef CHAR *PCHAR;
typedef CHAR *LPCH, *PCH;

typedef const CHAR *LPCCH, *PCCH;
typedef CHAR *NPSTR;
typedef CHAR *LPSTR, *PSTR;
typedef const CHAR *LPCSTR, *PCSTR;

//
// Neutral ANSI/UNICODE types and macros
//

#if defined(__MINGW32__)
   #include <stdarg.h>
   #include <windef.h>
   #include <winbase.h>
#else

#if defined(UNICODE)  		   // r_winnt

#ifndef _TCHAR_DEFINED
typedef WCHAR TCHAR, *PTCHAR;
typedef WCHAR TBYTE , *PTBYTE ;
#define _TCHAR_DEFINED
#endif /* !_TCHAR_DEFINED */

typedef LPWSTR LPTCH, PTCH;
typedef LPWSTR PTSTR, LPTSTR;
typedef LPCWSTR LPCTSTR;
typedef LPWSTR LP;
#define __TEXT(quote) L##quote      // r_winnt

#else   /* UNICODE */               // r_winnt

#ifndef _TCHAR_DEFINED
typedef char TCHAR, *PTCHAR;
typedef unsigned char TBYTE , *PTBYTE ;
#define _TCHAR_DEFINED
#endif /* !_TCHAR_DEFINED */

typedef LPSTR LPTCH, PTCH;
typedef LPSTR PTSTR, LPTSTR;
typedef LPCSTR LPCTSTR;
#define __TEXT(quote) quote         // r_winnt

#endif /* UNICODE */                // r_winnt

// Handle declarations.
typedef void          *HANDLE;
typedef HANDLE         HINSTANCE;
typedef HINSTANCE      HMODULE;

typedef void          *LPOVERLAPPED;
typedef void          *LPSECURITY_ATTRIBUTES;

typedef long long      LONGLONG;
#if  defined(_WIN64)
typedef unsigned __int64 UINT_PTR;
#else
typedef unsigned int   UINT_PTR;
#endif
#endif  //

#define DWORD_PTR UINT_PTR

#define FILE_BEGIN                          0
#define FILE_CURRENT                        1
#define FILE_END                            2
#define FILE_ATTRIBUTE_NORMAL      0x00000080

#define CREATE_NEW                          1
#define CREATE_ALWAYS                       2
#define OPEN_EXISTING                       3

#if !defined(__MINGW32__)
#define GENERIC_READ               0x80000000
#define GENERIC_WRITE              0x40000000

#define FILE_SHARE_READ           0x00000001L

#define NO_ERROR                            0
#define ERROR_HANDLE_EOF                   38

#define TRUE  1
#define FALSE 0

#define LOBYTE(w)              ((BYTE)((DWORD_PTR)(w) & 0xFF))
#define HIBYTE(w)              ((BYTE)((DWORD_PTR)(w) >> 8))

#define LOWORD(l)              ((WORD)((DWORD_PTR)(l) & 0xFFFF))
#define HIWORD(l)              ((WORD)((DWORD_PTR)(l) >> 16))

#define ERROR_TOO_MANY_OPEN_FILES       4
#endif //!defined(__MINGW32__)

#ifndef _MAX_PATH
#define _MAX_DRIVE          3
#define _MAX_FNAME          256
#define _MAX_DIR            _MAX_FNAME
#define _MAX_EXT            _MAX_FNAME
#define _MAX_PATH           260
#endif


    // #define __stdcall __attribute__((__stdcall__))
#if !defined(__MINGW32__)
#define __stdcall
#endif
// gcc uses cdecl as a standard:
#define cdecl
#define WINAPI __stdcall

#ifndef GUID_DEFINED
typedef struct _GUID
{
    // TODO: for WIN64 this need to be redefined
    unsigned int   Data1;
    unsigned short Data2;
    unsigned short Data3;
    unsigned char  Data4[ 8 ];
} GUID;
#define GUID_DEFINED
#endif

#if !defined(__MINGW32__)
void _splitpath(const char* inpath, char * drv, char * dir,
                char* fname, char * ext );
int _strnicmp( LPCSTR str1, LPCSTR str2, size_t n );
void _makepath( char * path, const char * drive,
                const char *directory, const char * filename,
                const char * extension );

/* 64 bit number of 100 nanoseconds intervals since January 1, 1601 */
typedef struct _FILETIME
{
#ifdef __BIG_ENDIAN__
    DWORD  dwHighDateTime;
    DWORD  dwLowDateTime;
#else
    DWORD  dwLowDateTime;
    DWORD  dwHighDateTime;
#endif
} FILETIME, *PFILETIME, *LPFILETIME;

typedef struct _SYSTEMTIME{
    WORD wYear;
    WORD wMonth;
    WORD wDayOfWeek;
    WORD wDay;
    WORD wHour;
    WORD wMinute;
    WORD wSecond;
    WORD wMilliseconds;
} SYSTEMTIME, *PSYSTEMTIME, *LPSYSTEMTIME;
#endif // __MINGW32__

#endif // _MSC_VER

extern HINSTANCE g_hInstance;

//
// Function wrappers
//
    
int WINAPI AXODBG_printf(char *lpsz, ... );
BOOL  WINAPI c_WriteFile( FILEHANDLE hFile, LPCVOID buffer, DWORD bytesToWrite,
                          LPDWORD bytesWritten, LPOVERLAPPED overlapped );
FILEHANDLE WINAPI c_CreateFile( LPCSTR filename, DWORD access, DWORD sharing,
                                 LPSECURITY_ATTRIBUTES sa, DWORD creation,
                                 DWORD attributes_, HANDLE templ);
DWORD WINAPI c_SetFilePointer( FILEHANDLE hFile, LONG distance, LONG *highword, DWORD method );
BOOL  WINAPI c_ReadFile( FILEHANDLE hFile, LPVOID buffer, DWORD bytesToRead,
                         LPDWORD bytesRead, LPOVERLAPPED overlapped );
DWORD WINAPI c_GetFileSize( FILEHANDLE hFile, LPDWORD filesizehigh );
BOOL  WINAPI c_CloseHandle( FILEHANDLE handle );
INT   WINAPI c_LoadString( HINSTANCE instance, UINT resource_id,
                           LPSTR buffer, INT buflen );

#ifdef __cplusplus
}
#endif
