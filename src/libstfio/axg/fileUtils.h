#ifndef FILEUTILS_H
#define FILEUTILS_H

#define kAG_Creator  'AxG2'
#define kAG_DocType  'AxGr'

#define kAGX_Creator 'AxGX'
#define kAGX_DocType  'axgx'

#include "longdef.h"

#if 0
    typedef const int filehandle;
#else
    #if !defined(_WINDOWS) || defined(__MINGW32__)
        #include <cstdio>
        typedef FILE* filehandle;
    #else
        #include "Windows.h"
        typedef HANDLE filehandle;
    #endif
#endif

#include "stringUtils.h"

filehandle OpenFile( const char *fileName );
void CloseFile( filehandle dataRefNum );

int SetFilePosition( filehandle dataRefNum, int posn );
int ReadFromFile( filehandle dataRefNum, AXGLONG count, void *dataToRead );

#endif
