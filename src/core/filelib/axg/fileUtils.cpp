#ifdef __WXMAC__
#include <Carbon/Carbon.h>
#endif

#include "fileUtils.h"
#ifdef _WINDOWS
#include <wx/wx.h>
#endif

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

filehandle OpenFile( const char *fileName )
{
#ifdef __WXMAC__
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
#endif
#ifndef _WINDOWS
    return fopen( fileName, "r" );
#else
	wxString fileNameU( fileName );
	HANDLE file = CreateFile(fileNameU.c_str(), GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    return file;
#endif
}

void CloseFile( filehandle dataRefNum )
{
#ifdef __WXMAC__
    FSClose( dataRefNum );
#endif
#ifndef _WINDOWS
    fclose( dataRefNum );
#else
    CloseHandle(dataRefNum);
#endif
}

int SetFilePosition( filehandle dataRefNum, int posn )
{
#ifdef __WXMAC__
    return SetFPos( dataRefNum, fsFromStart, posn );		// Position the mark
#endif
#ifndef _WINDOWS
    return fseek( dataRefNum, posn, SEEK_SET );
#else
    if (SetFilePointer(dataRefNum, posn, NULL, FILE_BEGIN) == INVALID_SET_FILE_POINTER)
        return 1;
    else
        return 0;
#endif
}

int ReadFromFile( filehandle dataRefNum, long *count, void *dataToRead )
{
#ifdef __WXMAC__
    return FSRead( dataRefNum, count, dataToRead );
#endif
#ifndef _WINDOWS
    if ( fread( dataToRead, 1, *count, dataRefNum ) == *count )
        return 0;
    else
        return 1;
#else
    DWORD   dwRead;
	short res = ReadFile(dataRefNum, dataToRead, *count, &dwRead, NULL);
    if (res)
        return 0;
    else
        return 1;
#endif
}
