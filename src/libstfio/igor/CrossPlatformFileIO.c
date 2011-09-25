// This file contains utilities for cross-platform file I/O.

#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>

#ifdef WIN32
  #include <Windows.h>
#else
  #include "../axon/Common/unix.h"
#endif
// The Windows headers create the WIN32 symbol if we are compiling for Windows.
// Here, we create an analogous MACINTOSH symbol if we are compiling for Macintosh.
#if (defined(GENERATINGPOWERPC) || defined(GENERATING68K))
	#define MACINTOSH 1
#endif

#include "CrossPlatformFileIO.h"

/*	CPCreateFile(fullFilePath, overwrite, macCreator, macFileType)

	Creates a file with the location and name specified by fullFilePath.
	
	fullFilePath must be a native path.

	If overwrite is true and a file by that name already exists, it first
	deletes the conflicting file. If overwrite is false and a file by that
	name exists, it returns an error.
	
	macFileType is ignored on Windows. On Macintosh, it is used to set
	the new file's type. For example, use 'TEXT' for a text file.
	
	macCreator is ignored on Windows. On Macintosh, it is used to set
	the new file's creator code. For example, use 'IGR0' (last character is zero)
	for an file.
	
	Returns 0 if OK or an error code.
*/
int
CPCreateFile(const char* fullFilePath, int overwrite)
{
	int err;
	err = 0;	
#ifdef _WINDOWS
	if (overwrite)							// Delete file if it exists and if overwrite is specified.
            CPDeleteFile(fullFilePath);			// Ignore error.
#endif

	#ifdef MACINTOSH
		if (err = create(fullFilePath, 0, macCreator, macFileType))
			return err;
		return 0;
	#endif
	
#if 1//def WIN32
	{
		HANDLE fileH;
		long accessMode, shareMode;
		
#ifdef _WINDOWS
		accessMode = GENERIC_READ | GENERIC_WRITE;
		shareMode = 0;
		fileH = CreateFileA(fullFilePath, accessMode, shareMode, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
#else
		fileH = fopen(fullFilePath, "w+b");
#endif
		if (fileH == INVALID_HANDLE_VALUE)
#ifdef _WINDOWS
                    err = GetLastError();
#else
                    err = 1;
#endif
		else
#ifdef _WINDOWS
                    CloseHandle(fileH);
#else
                    fclose(fileH);
#endif
		return err;
	}
	#endif
}

/*	CPDeleteFile(fullFilePath)

	Deletes the file specified by fullFilePath.
	
	fullFilePath must be a native path.
	
	Returns 0 if OK or an error code.
	
	Added for Igor Pro 3.13 but works with any version. However, some error
	codes returned require Igor Pro 3.13 or later, so you will get bogus error
	messages if you return these error codes to earlier versions of Igor.
*/
#ifdef _WINDOWS
int
CPDeleteFile(const char* fullFilePath)
{
	#ifdef MACINTOSH
		int err;

		if (err = fsdelete(fullFilePath, 0))
			return err;
		return 0;
	#endif
	
	{
		int err;

		err = 0;
		if (DeleteFileA(fullFilePath) == 0)
                    err = GetLastError();
		return err;
	}
}
#endif

/*	CPOpenFile(fullFilePath, readOrWrite, fileRefPtr)

	If readOrWrite is zero, opens an existing file for reading and returns a file reference
	via fileRefPtr.

	If readOrWrite is non-zero, opens an existing file for writing or creates a new
	file if none exists and returns a file reference via fileRefPtr.

	fullFilePath must be a native path.
	
	Returns 0 if OK or an error code.
*/
int
CPOpenFile(const char* fullFilePath, int readOrWrite, CP_FILE_REF* fileRefPtr)
{
	*fileRefPtr = fopen(fullFilePath, readOrWrite ? "wb" : "rb");
	if (*fileRefPtr == NULL)
		return CP_FILE_OPEN_ERROR;
	return 0;
}

/*	CPCloseFile(fileRef)

	Closes the referenced file.
	
	Returns 0 if OK or an error code.
	
	Added for Igor Pro 3.13 but works with any version. However, some error
	codes returned require Igor Pro 3.13 or later, so you will get bogus error
	messages if you return these error codes to earlier versions of Igor.
*/
int
CPCloseFile(CP_FILE_REF fileRef)
{
	if (fclose(fileRef))
		return CP_FILE_CLOSE_ERROR;
	return 0;
}

/*	CPReadFile(fileRef, count, buffer, numBytesReadPtr)

	Reads count bytes from the referenced file into the buffer.
	
	If numBytesReadPtr is not NULL, stores the number of bytes read in
	*numBytesReadPtr.
	
	Returns 0 if OK or an error code.
	
	If bytes remain to be read in the file and you ask to read more bytes
	than remain, the remaining bytes are returned and the function result is
	zero. If no bytes remain to be read in the file and you ask to read bytes,
	no bytes are returned and the function result is CP_FILE_EOF_ERROR.
	
	CPReadFile is appropriate when you are reading data of variable size, in
	which case you do not want to consider it an error if the end of file is reached
	before reading all of the bytes that you requested. If you are reading a
	record of fixed size, use use CPReadFile2 instead of CPReadFile.
	
	Added for Igor Pro 3.13 but works with any version. However, some error
	codes returned require Igor Pro 3.13 or later, so you will get bogus error
	messages if you return these error codes to earlier versions of Igor.
*/
int
CPReadFile(CP_FILE_REF fileRef, unsigned long count, void* buffer, unsigned long* numBytesReadPtr)
{
	unsigned long numBytesRead;
	
	if (count == 0) {
		if (numBytesReadPtr != NULL)
			*numBytesReadPtr = 0;
		return 0;
	}
	
	clearerr(fileRef);
	numBytesRead = (DWORD)fread(buffer, 1, count, fileRef);
	if (numBytesReadPtr != NULL)
		*numBytesReadPtr = numBytesRead;
	if (ferror(fileRef))
		return CP_FILE_READ_ERROR;
	if (numBytesRead==0 && CPAtEndOfFile(fileRef))
		return CP_FILE_EOF_ERROR;			// We were at the end of file when asked to read some bytes.
	return 0;
}

/*	CPReadFile2(fileRef, count, buffer, numBytesReadPtr)

	Reads count bytes from the referenced file into the buffer.
	
	If numBytesReadPtr is not NULL, stores the number of bytes read in
	*numBytesReadPtr.
	
	Returns 0 if OK or an error code.
	
	If bytes remain to be read in the file and you ask to read more bytes
	than remain, the remaining bytes are returned and the function result is
	CP_FILE_EOF_ERROR.
	
	CPReadFile2 is appropriate when you are reading a record of fixed size, in
	which case you want to consider it an error if the end of file is reached
	before reading all of the bytes in the record. If you are reading a record
	of variable size then you should use CPReadFile instead of CPReadFile2.
	
	Added for Igor Pro 3.13 but works with any version. However, some error
	codes returned require Igor Pro 3.13 or later, so you will get bogus error
	messages if you return these error codes to earlier versions of Igor.
*/
int
CPReadFile2(CP_FILE_REF fileRef, unsigned long count, void* buffer, unsigned long* numBytesReadPtr)
{
	unsigned long numBytesRead;
	
	if (count == 0) {
		if (numBytesReadPtr != NULL)
			*numBytesReadPtr = 0;
		return 0;
	}
	
	clearerr(fileRef);
	numBytesRead = (DWORD)fread(buffer, 1, count, fileRef);
	if (numBytesReadPtr != NULL)
		*numBytesReadPtr = numBytesRead;
	if (ferror(fileRef))
		return CP_FILE_READ_ERROR;
	if (numBytesRead < count) {					// We did not read all of the bytes requested.
		if (CPAtEndOfFile(fileRef))
			return CP_FILE_EOF_ERROR;			// We hit the end of file.
		return CP_FILE_READ_ERROR;				// Some other occurred but ferror did not reflect it.
	}
	return 0;
}

/*	CPWriteFile(fileRef, count, buffer, numBytesWrittenPtr)

	Writes count bytes from the buffer to the referenced file.
	
	If numBytesWrittenPtr is not NULL, stores the number of bytes written in
	*numBytesWrittenPtr.
	
	Returns 0 if OK or an error code.
	
	Added for Igor Pro 3.13 but works with any version. However, some error
	codes returned require Igor Pro 3.13 or later, so you will get bogus error
	messages if you return these error codes to earlier versions of Igor.
*/
int
CPWriteFile(CP_FILE_REF fileRef, unsigned long count, const void* buffer, unsigned long* numBytesWrittenPtr)
{
	unsigned long numBytesWritten;
	
	if (count == 0) {
		if (numBytesWrittenPtr != NULL)
			*numBytesWrittenPtr = 0;
		return 0;
	}
	
	numBytesWritten = (DWORD)fwrite(buffer, 1, count, fileRef);
	if (numBytesWrittenPtr != NULL)
		*numBytesWrittenPtr = numBytesWritten;
	if (numBytesWritten != count)
		return CP_FILE_WRITE_ERROR;
	return 0;
}

/*	CPGetFilePosition(fileRef, filePosPtr)

	Returns via filePosPtr the current file position of the referenced file.
	
	Returns 0 if OK or an error code.
	
	Added for Igor Pro 3.13 but works with any version. However, some error
	codes returned require Igor Pro 3.13 or later, so you will get bogus error
	messages if you return these error codes to earlier versions of Igor.
*/
int
CPGetFilePosition(CP_FILE_REF fileRef, unsigned long* filePosPtr)
{
	long pos;
	
	pos = ftell(fileRef);
	if (pos == -1L)
		return CP_FILE_POS_ERROR;
	*filePosPtr = pos;
	return 0;
}

/*	CPSetFilePosition(fileRef, filePos, mode)

	Sets the current file position in the referenced file.
	
	If mode is -1, then filePos is relative to the start of the file.
	If mode is 0, then filePos is relative to the current file position.
	If mode is 1, then filePos is relative to the end of the file.
	
	Returns 0 if OK or an error code.
	
	Added for Igor Pro 3.13 but works with any version. However, some error
	codes returned require Igor Pro 3.13 or later, so you will get bogus error
	messages if you return these error codes to earlier versions of Igor.
*/
int
CPSetFilePosition(CP_FILE_REF fileRef, long filePos, int mode)
{
	int seekMode;
	
	switch(mode) {
		case -1:
			seekMode = SEEK_SET;
			break;
		case 0:
			seekMode = SEEK_CUR;
			break;
		case 1:
			seekMode = SEEK_END;
			break;
		default:
			return CP_FILE_POS_ERROR;
	}
	
	if (fseek(fileRef, filePos, seekMode) != 0)
		return CP_FILE_POS_ERROR;
	return 0;
}

/*	CPAtEndOfFile(fileRef)

	Returns 1 if the current file position is at the end of file, 0 if not.
	
	Added for Igor Pro 3.13 but works with any version. However, some error
	codes returned require Igor Pro 3.13 or later, so you will get bogus error
	messages if you return these error codes to earlier versions of Igor.
*/
int
CPAtEndOfFile(CP_FILE_REF fileRef)
{
	if (feof(fileRef))				// Hit end of file?
		return 1;
	return 0;
}

/*	CPNumberOfBytesInFile(fileRef, numBytesPtr)

	Returns via numBytesPtr the total number of bytes in the referenced file.
	
	Returns 0 if OK or an error code.
	
	Added for Igor Pro 3.13 but works with any version. However, some error
	codes returned require Igor Pro 3.13 or later, so you will get bogus error
	messages if you return these error codes to earlier versions of Igor.
*/
int
CPNumberOfBytesInFile(CP_FILE_REF fileRef, unsigned long* numBytesPtr)
{
        long originalPos;

	originalPos = ftell(fileRef);
	if (fseek(fileRef, 0, SEEK_END) != 0)
		return CP_FILE_POS_ERROR;
	*numBytesPtr = ftell(fileRef);
	if (*numBytesPtr == -1L)
		return CP_FILE_POS_ERROR;
	if (fseek(fileRef, originalPos, SEEK_SET) != 0)
		return CP_FILE_POS_ERROR;
	return 0;
}
