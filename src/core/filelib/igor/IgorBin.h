// IgorBin.h -- structures and #defines for dealing with Igor binary data.


#ifdef __cplusplus
extern "C" {
#endif

// All structures written to disk are 2-byte-aligned.
#if GENERATINGPOWERPC
	#pragma options align=mac68k
#endif
#ifdef WIN32
	#pragma pack(2)
#endif

#ifdef WIN32
	typedef void** Handle;
#endif


// From IgorMath.h
#define NT_CMPLX 1			// Complex numbers.
#define NT_FP32 2			// 32 bit fp numbers.
#define NT_FP64 4			// 64 bit fp numbers.
#define NT_I8 8				// 8 bit signed integer. Requires Igor Pro 2.0 or later.
#define NT_I16 	0x10		// 16 bit integer numbers. Requires Igor Pro 2.0 or later.
#define NT_I32 	0x20		// 32 bit integer numbers. Requires Igor Pro 2.0 or later.
#define NT_UNSIGNED 0x40	// Makes above signed integers unsigned. Requires Igor Pro 3.0 or later.


// From wave.h
#define MAXDIMS 4


//	From binary.h

typedef struct BinHeader1 {
	short version;						// Version number for backwards compatibility.
	long wfmSize;						// The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.
	short checksum;						// Checksum over this header and the wave header.
} BinHeader1;

typedef struct BinHeader2 {
	short version;						// Version number for backwards compatibility.
	long wfmSize;						// The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.
	long noteSize;						// The size of the note text.
	long pictSize;						// Reserved. Write zero. Ignore on read.
	short checksum;						// Checksum over this header and the wave header.
} BinHeader2;

typedef struct BinHeader3 {
	short version;						// Version number for backwards compatibility.
	long wfmSize;						// The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.
	long noteSize;						// The size of the note text.
	long formulaSize;					// The size of the dependency formula, if any.
	long pictSize;						// Reserved. Write zero. Ignore on read.
	short checksum;						// Checksum over this header and the wave header.
} BinHeader3;

typedef struct BinHeader5 {
	short version;						// Version number for backwards compatibility.
	short checksum;						// Checksum over this header and the wave header.
	long wfmSize;						// The size of the WaveHeader5 data structure plus the wave data.
	long formulaSize;					// The size of the dependency formula, if any.
	long noteSize;						// The size of the note text.
	long dataEUnitsSize;				// The size of optional extended data units.
	long dimEUnitsSize[MAXDIMS];		// The size of optional extended dimension units.
	long dimLabelsSize[MAXDIMS];		// The size of optional dimension labels.
	long sIndicesSize;					// The size of string indicies if this is a text wave.
	long optionsSize1;					// Reserved. Write zero. Ignore on read.
	long optionsSize2;					// Reserved. Write zero. Ignore on read.
} BinHeader5;


//	From wave.h

#define MAX_WAVE_NAME2 18	// Maximum length of wave name in version 1 and 2 files. Does not include the trailing null.
#define MAX_WAVE_NAME5 31	// Maximum length of wave name in version 5 files. Does not include the trailing null.
#define MAX_UNIT_CHARS 3

//	Header to an array of waveform data.

struct WaveHeader2 {
	short type;							// See types (e.g. NT_FP64) above. Zero for text waves.
	struct WaveHeader2 **next;			// Used in memory only. Write zero. Ignore on read.

	char bname[MAX_WAVE_NAME2+2];		// Name of wave plus trailing null.
	short whVersion;					// Write 0. Ignore on read.
	short srcFldr;						// Used in memory only. Write zero. Ignore on read.
	Handle fileName;					// Used in memory only. Write zero. Ignore on read.

	char dataUnits[MAX_UNIT_CHARS+1];	// Natural data units go here - null if none.
	char xUnits[MAX_UNIT_CHARS+1];		// Natural x-axis units go here - null if none.

	long npnts;							// Number of data points in wave.

	short aModified;					// Used in memory only. Write zero. Ignore on read.
	double hsA,hsB;						// X value for point p = hsA*p + hsB

	short wModified;					// Used in memory only. Write zero. Ignore on read.
	short swModified;					// Used in memory only. Write zero. Ignore on read.
	short fsValid;						// True if full scale values have meaning.
	double topFullScale,botFullScale;	// The min full scale value for wave.
		   
	char useBits;						// Used in memory only. Write zero. Ignore on read.
	char kindBits;						// Reserved. Write zero. Ignore on read.
	void **formula;						// Used in memory only. Write zero. Ignore on read.
	long depID;							// Used in memory only. Write zero. Ignore on read.
	unsigned long creationDate;			// DateTime of creation. Not used in version 1 files.
	char wUnused[2];					// Reserved. Write zero. Ignore on read.

	unsigned long modDate;				// DateTime of last modification.
	Handle waveNoteH;					// Used in memory only. Write zero. Ignore on read.

	float wData[4];						// The start of the array of waveform data.
};
typedef struct WaveHeader2 WaveHeader2;
typedef WaveHeader2 *WavePtr2;
typedef WavePtr2 *waveHandle2;


struct WaveHeader5 {
	struct WaveHeader5 **next;			// link to next wave in linked list.

	unsigned long creationDate;			// DateTime of creation.
	unsigned long modDate;				// DateTime of last modification.

	long npnts;							// Total number of points (multiply dimensions up to first zero).
	short type;							// See types (e.g. NT_FP64) above. Zero for text waves.
	short dLock;						// Reserved. Write zero. Ignore on read.

	char whpad1[6];						// Reserved. Write zero. Ignore on read.
	short whVersion;					// Write 1. Ignore on read.
	char bname[MAX_WAVE_NAME5+1];		// Name of wave plus trailing null.
	long whpad2;						// Reserved. Write zero. Ignore on read.
	struct DataFolder **dFolder;		// Used in memory only. Write zero. Ignore on read.

	// Dimensioning info. [0] == rows, [1] == cols etc
	long nDim[MAXDIMS];					// Number of of items in a dimension -- 0 means no data.
	double sfA[MAXDIMS];				// Index value for element e of dimension d = sfA[d]*e + sfB[d].
	double sfB[MAXDIMS];

	// SI units
	char dataUnits[MAX_UNIT_CHARS+1];			// Natural data units go here - null if none.
	char dimUnits[MAXDIMS][MAX_UNIT_CHARS+1];	// Natural dimension units go here - null if none.

	short fsValid;						// TRUE if full scale values have meaning.
	short whpad3;						// Reserved. Write zero. Ignore on read.
	double topFullScale,botFullScale;	// The max and max full scale value for wave.

	Handle dataEUnits;					// Used in memory only. Write zero. Ignore on read.
	Handle dimEUnits[MAXDIMS];			// Used in memory only. Write zero. Ignore on read.
	Handle dimLabels[MAXDIMS];			// Used in memory only. Write zero. Ignore on read.
	
	Handle waveNoteH;					// Used in memory only. Write zero. Ignore on read.
	long whUnused[16];					// Reserved. Write zero. Ignore on read.

	// The following stuff is considered private to Igor.

	short aModified;					// Used in memory only. Write zero. Ignore on read.
	short wModified;					// Used in memory only. Write zero. Ignore on read.
	short swModified;					// Used in memory only. Write zero. Ignore on read.
	
	char useBits;						// Used in memory only. Write zero. Ignore on read.
	char kindBits;						// Reserved. Write zero. Ignore on read.
	void **formula;						// Used in memory only. Write zero. Ignore on read.
	long depID;							// Used in memory only. Write zero. Ignore on read.
	
	short whpad4;						// Reserved. Write zero. Ignore on read.
	short srcFldr;						// Used in memory only. Write zero. Ignore on read.
	Handle fileName;					// Used in memory only. Write zero. Ignore on read.
	
	long **sIndices;					// Used in memory only. Write zero. Ignore on read.

	float wData[1];						// The start of the array of data. Must be 64 bit aligned.
};
typedef struct WaveHeader5 WaveHeader5;
typedef WaveHeader5 *WavePtr5;
typedef WavePtr5 *WaveHandle5;


#if GENERATINGPOWERPC
	#pragma options align=reset
#endif
#ifdef WIN32
	#pragma pack()
#endif
// All structures written to disk are 2-byte-aligned.

#ifdef __cplusplus
}
#endif
