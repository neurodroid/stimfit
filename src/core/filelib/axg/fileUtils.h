#ifndef FILEUTILS_H
#define FILEUTILS_H

#define kAG_Creator  'AxG2'
#define kAG_DocType  'AxGr'

#define kAGX_Creator 'AxGX'
#define kAGX_DocType  'axgx'

#ifdef __WXMAC__
    typedef const int filehandle;
#else
    #ifndef _WINDOWS
        #include <cstdio>
        typedef FILE* filehandle;
    #else
        #include "Windows.h"
        typedef HANDLE filehandle;
    #endif
#endif

filehandle OpenFile( const char *fileName );
void CloseFile( filehandle dataRefNum );

int SetFilePosition( filehandle dataRefNum, int posn );
int ReadFromFile( filehandle dataRefNum, long *count, void *dataToRead );

#endif
