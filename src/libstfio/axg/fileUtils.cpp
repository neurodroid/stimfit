#include "fileUtils.h"
#if defined(_WINDOWS) && !defined(__MINGW32__)
    #include <sstream>
    #include <string>
    typedef std::wstring wxString;
#endif

// Mac-specific file access functions
// On other platforms, replace the following with equivalent functions

/* GetApplicationDirectory returns the volume reference number
   and directory ID for the demo application's directory. */

filehandle OpenFile( const char *fileName )
{
#if defined(_WINDOWS) && !defined(__MINGW32__)
	std::wstringstream fileNameS;
	fileNameS << fileName;
    HANDLE file = CreateFile(fileNameS.str().c_str(), GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    return file;
#else
    return fopen( fileName, "r" );
#endif
}

void CloseFile( filehandle dataRefNum )
{
#if defined(_WINDOWS) && !defined(__MINGW32__)
    CloseHandle(dataRefNum);
    return;
#else
    fclose( dataRefNum );
    return;
#endif
}

int SetFilePosition( filehandle dataRefNum, int posn )
{
#if defined(_WINDOWS) && !defined(__MINGW32__)
    if (SetFilePointer(dataRefNum, posn, NULL, FILE_BEGIN) == INVALID_SET_FILE_POINTER)
        return 1;
    else
        return 0;
#else
    return fseek( dataRefNum, posn, SEEK_SET );
#endif
}

int ReadFromFile( filehandle dataRefNum, AXGLONG *count, void *dataToRead )
{
#if defined(_WINDOWS) && !defined(__MINGW32__)
    DWORD dwRead;
    short res = ReadFile(dataRefNum, dataToRead, *count, &dwRead, NULL);
    if (res)
        return 0;
    else
        return 1;
#else
    if ( (AXGLONG)fread( dataToRead, 1, *count, dataRefNum ) == *count )
        return 0;
    else
        return 1;
#endif
}
