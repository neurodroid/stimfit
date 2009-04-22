#define CP_FILE_OPEN_ERROR 10000
#define CP_FILE_CLOSE_ERROR 10001
#define CP_FILE_EOF_ERROR 10002
#define CP_FILE_READ_ERROR 10003
#define CP_FILE_WRITE_ERROR 10004
#define CP_FILE_POS_ERROR 10005

#define CP_FILE_REF FILE*

int CPCreateFile(const char* fullFilePath, int overwrite, long macCreator, long macFileType);
int CPDeleteFile(const char* fullFilePath);
int CPOpenFile(const char* fullFilePath, int readOrWrite, CP_FILE_REF* fileRefPtr);
int CPCloseFile(CP_FILE_REF fileRef);
int CPReadFile(CP_FILE_REF fileRef, unsigned long count, void* buffer, unsigned long* numBytesReadPtr);
int CPReadFile2(CP_FILE_REF fileRef, unsigned long count, void* buffer, unsigned long* numBytesReadPtr);
int CPWriteFile(CP_FILE_REF fileRef, unsigned long count, const void* buffer, unsigned long* numBytesWrittenPtr);
int CPGetFilePosition(CP_FILE_REF fileRef, unsigned long* filePosPtr);
int CPSetFilePosition(CP_FILE_REF fileRef, long filePos, int mode);
int CPAtEndOfFile(CP_FILE_REF fileRef);
int CPNumberOfBytesInFile(CP_FILE_REF fileRef, unsigned long* numBytesPtr);
