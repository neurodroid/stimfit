#ifndef FILEUTILS_H
#define FILEUTILS_H

#define kAG_Creator  'AxG2'
#define kAG_DocType  'AxGr'

#define kAGX_Creator 'AxGX'
#define kAGX_DocType  'axgx'


#ifdef __WXMAC__
int OpenFile( const char *fileName );
void CloseFile( int dataRefNum );

int SetFilePosition( int dataRefNum, int posn );
int ReadFromFile( int dataRefNum, long *count, void *dataToRead );
#else
#include <cstdio>

FILE* OpenFile( const char *fileName );
void CloseFile( FILE* fh );

int SetFilePosition( FILE* fh, int posn );
int ReadFromFile( FILE* fh, long *count, void *dataToRead );
#endif

#endif
