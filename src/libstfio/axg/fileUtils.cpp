#include "fileUtils.h"
#if defined(_WINDOWS) && !defined(__MINGW32__)
#include <sstream>

#ifdef MODULE_ONLY
#include <string>
typedef std::wstring wxString;
#else
#include <wx/wx.h>
#endif

#endif

// Mac-specific file access functions
// On other platforms, replace the following with equivalent functions

/* GetApplicationDirectory returns the volume reference number
   and directory ID for the demo application's directory. */

#if 0
OSStatus GetApplicationDirectory( short *vRefNum, AXGLONG *dirID )
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
#if 0
    short dataRefNum = 0;
    short vRefNum;
    AXGLONG dirID;
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
#if defined(__APPLE__) || defined(__linux__) || defined(__MINGW32__)
    return fopen( fileName, "r" );
#endif
#if defined(_WINDOWS) && !defined(__MINGW32__)
	std::wstringstream fileNameS;
	fileNameS << fileName;
    HANDLE file = CreateFile(fileNameS.str().c_str(), GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    return file;
#endif
}

void CloseFile( filehandle dataRefNum )
{
#if 0
    FSClose( dataRefNum );
    return;
#endif
#if defined(__APPLE__) || defined(__linux__) || defined(__MINGW32__)
    fclose( dataRefNum );
    return;
#endif
#if defined(_WINDOWS) && !defined(__MINGW32__)
    CloseHandle(dataRefNum);
    return;
#endif
}

int SetFilePosition( filehandle dataRefNum, int posn )
{
#if 0
    return SetFPos( dataRefNum, fsFromStart, posn );		// Position the mark
#endif
#if defined(__APPLE__) || defined(__linux__) || defined(__MINGW32__)
    return fseek( dataRefNum, posn, SEEK_SET );
#endif
#if defined(_WINDOWS) && !defined(__MINGW32__)
    if (SetFilePointer(dataRefNum, posn, NULL, FILE_BEGIN) == INVALID_SET_FILE_POINTER)
        return 1;
    else
        return 0;
#endif
}

int ReadFromFile( filehandle dataRefNum, AXGLONG count, void *dataToRead )
{
#if 0
    return FSRead( dataRefNum, &count, dataToRead );
#endif
#if defined(__APPLE__) || defined(__linux__) || defined(__MINGW32__)
    if ( (AXGLONG)fread( dataToRead, 1, count, dataRefNum ) == count )
        return 0;
    else
        return 1;
#endif
#if defined(_WINDOWS) && !defined(__MINGW32__)
    DWORD   dwRead;
	short res = ReadFile(dataRefNum, dataToRead, count, &dwRead, NULL);
    if (res)
        return 0;
    else
        return 1;
#endif
}
