#ifdef __WXMAC__
#include <Carbon/Carbon.h>
#endif

#include "fileUtils.h"

// Mac-specific file access functions
// On other platforms, replace the following with equivalent functions

/* GetApplicationDirectory returns the volume reference number
   and directory ID for the demo application's directory. */

#ifdef __WXMAC__
OSStatus GetApplicationDirectory( short *vRefNum, long *dirID )
{
    ProcessSerialNumber PSN;
    ProcessInfoRec pinfo;
    FSSpec pspec;
    OSStatus err;

    /* valid parameters */
    if ( vRefNum == NULL || dirID == NULL ) return paramErr;

    /* set up process serial number */
    PSN.highLongOfPSN = 0;
    PSN.lowLongOfPSN = kCurrentProcess;

    /* set up info block */
    pinfo.processInfoLength = sizeof( pinfo );
    pinfo.processName = NULL;
    pinfo.processAppSpec = &pspec;

    /* grab the vrefnum and directory */
    err = GetProcessInformation( &PSN, &pinfo );
    if ( err == noErr )
    {
        *vRefNum = pspec.vRefNum;
        *dirID = pspec.parID;
    }

    return err;
}
#endif

#ifdef __WXMAC__
int OpenFile( const char *fileName )
{
    short dataRefNum = 0;
    short vRefNum;
    long dirID;
    OSErr result;
    FSSpec spec;

    // get the application's directory ID
    result = GetApplicationDirectory( &vRefNum, &dirID );

    if ( result != noErr )
    {
        printf( "Error from GetApplicationDirectory - result = %d", result );
        return 0;
    }

    // Make an FSSpec for the AxoGraph file
    Str255 macFileName;
    CopyCStringToPascal( fileName, macFileName);

    result = FSMakeFSSpec( vRefNum, dirID, macFileName, &spec );

    if ( result != noErr ) {
        printf( "Error from FSMakeFSSpec - result = %d", result );
        return 0;
    }

    // open the selected file
    result = FSpOpenDF( &spec, fsRdPerm, &dataRefNum );

    if ( result != noErr ) {
        printf( "Error from FSpOpenDF - result = %d", result );
        return 0;
    }

    return dataRefNum;
}
#else
FILE* OpenFile( const char *fileName )
{
    return fopen( fileName, "r" );
}
#endif

#ifdef __WXMAC__
void CloseFile( int dataRefNum )
{
    FSClose( dataRefNum );
}
#else
void CloseFile( FILE* fh )
{
    fclose( fh );
}
#endif

#ifdef __WXMAC__
int SetFilePosition( int dataRefNum, int posn )
{
    return SetFPos( dataRefNum, fsFromStart, posn );		// Position the mark
}
#else
int SetFilePosition( FILE* fh, int posn )
{
    return fseek( fh, posn, SEEK_SET );
}
#endif


#ifdef __WXMAC__
int ReadFromFile( int dataRefNum, long *count, void *dataToRead )
{
    return FSRead( dataRefNum, count, dataToRead );
}
#else
int ReadFromFile( FILE* fh, long *count, void *dataToRead )
{
    int res = fread( dataToRead, 1, *count, fh );
    if ( res == *count )
        return 0;
    else
        return 1;
}
#endif
