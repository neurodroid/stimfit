/*

    Copyright (C) 2013 Alois Schloegl <alois.schloegl@gmail.com>
    Copyright (C) 1999 Wavematrix, Inc. Lake Oswego OR, USA

    This file is part of the "BioSig for C/C++" repository
    (biosig4c++) at http://biosig.sf.net/


    BioSig is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/



// IgorBin.h -- structures and #defines for dealing with Igor binary data.

#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

// All structures written to disk are 2-byte-aligned.
#if __GNUC__
	#pragma pack(push,2)
	/*
 	igor uses a 32bit memory model, all pointers are 32 bit. 
	in order to keep the structs containing poiners aligned, 
	all pointers need to be remapped to uint32_t. 
	*/
	#define ptr_t uint32_t
	#define Handle uint32_t
#elif GENERATINGPOWERPC
	#pragma options align=mac68k
#endif


// From IgorMath.h
#define NT_CMPLX 1		// Complex numbers.
#define NT_FP32 2		// 32 bit fp numbers.
#define NT_FP64 4		// 64 bit fp numbers.
#define NT_I8 8			// 8 bit signed integer. Requires Igor Pro 2.0 or later.
#define NT_I16 	0x10		// 16 bit integer numbers. Requires Igor Pro 2.0 or later.
#define NT_I32 	0x20		// 32 bit integer numbers. Requires Igor Pro 2.0 or later.
#define NT_UNSIGNED 0x40	// Makes above signed integers unsigned. Requires Igor Pro 3.0 or later.


// From wave.h
#define MAXDIMS 4


//	From binary.h

typedef struct BinHeader1 {
	int16_t version;		// Version number for backwards compatibility.
	int32_t wfmSize;		// The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.
	int16_t checksum;		// Checksum over this header and the wave header.
} BinHeader1;

typedef struct BinHeader2 {
	int16_t version;		// Version number for backwards compatibility.
	int32_t wfmSize;		// The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.
	int32_t noteSize;		// The size of the note text.
	int32_t pictSize;		// Reserved. Write zero. Ignore on read.
	int16_t checksum;		// Checksum over this header and the wave header.
} BinHeader2;

typedef struct BinHeader3 {
	int16_t version;		// Version number for backwards compatibility.
	int32_t wfmSize;		// The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.
	int32_t noteSize;		// The size of the note text.
	int32_t formulaSize;		// The size of the dependency formula, if any.
	int32_t pictSize;		// Reserved. Write zero. Ignore on read.
	int16_t checksum;		// Checksum over this header and the wave header.
} BinHeader3;

typedef struct BinHeader5 {
	int16_t version;		// Version number for backwards compatibility.
	int16_t checksum;		// Checksum over this header and the wave header.
	int32_t wfmSize;		// The size of the WaveHeader5 data structure plus the wave data.
	int32_t formulaSize;		// The size of the dependency formula, if any.
	int32_t noteSize;		// The size of the note text.
	int32_t dataEUnitsSize;		// The size of optional extended data units.
	int32_t dimEUnitsSize[MAXDIMS];		// The size of optional extended dimension units.
	int32_t dimLabelsSize[MAXDIMS];		// The size of optional dimension labels.
	int32_t sIndicesSize;			// The size of string indicies if this is a text wave.
	int32_t optionsSize1;			// Reserved. Write zero. Ignore on read.
	int32_t optionsSize2;			// Reserved. Write zero. Ignore on read.
} BinHeader5;


//	From wave.h

#define MAX_WAVE_NAME2 18	// Maximum length of wave name in version 1 and 2 files. Does not include the trailing null.
#define MAX_WAVE_NAME5 31	// Maximum length of wave name in version 5 files. Does not include the trailing null.
#define MAX_UNIT_CHARS 3

//	Header to an array of waveform data.

struct WaveHeader2 {
	int16_t type;				// See types (e.g. NT_FP64) above. Zero for text waves.
	// struct WaveHeader2 **next;		// Used in memory only. Write zero. Ignore on read.
	ptr_t next; 

	char bname[MAX_WAVE_NAME2+2];		// Name of wave plus trailing null.
	int16_t whVersion;			// Write 0. Ignore on read.
	int16_t srcFldr;			// Used in memory only. Write zero. Ignore on read.
	Handle fileName;			// Used in memory only. Write zero. Ignore on read.

	char dataUnits[MAX_UNIT_CHARS+1];	// Natural data units go here - null if none.
	char xUnits[MAX_UNIT_CHARS+1];		// Natural x-axis units go here - null if none.

	int32_t npnts;				// Number of data points in wave.

	int16_t aModified;			// Used in memory only. Write zero. Ignore on read.
	double hsA,hsB;				// X value for point p = hsA*p + hsB

	int16_t wModified;			// Used in memory only. Write zero. Ignore on read.
	int16_t swModified;			// Used in memory only. Write zero. Ignore on read.
	int16_t fsValid;			// True if full scale values have meaning.
	double topFullScale,botFullScale;	// The min full scale value for wave.
		   
	char useBits;				// Used in memory only. Write zero. Ignore on read.
	char kindBits;				// Reserved. Write zero. Ignore on read.
	//void **formula;			// Used in memory only. Write zero. Ignore on read.
	ptr_t formula;

	int32_t depID;				// Used in memory only. Write zero. Ignore on read.
	uint32_t creationDate;			// DateTime of creation. Not used in version 1 files.
	char wUnused[2];			// Reserved. Write zero. Ignore on read.

	uint32_t  modDate;			// DateTime of last modification.
	Handle waveNoteH;			// Used in memory only. Write zero. Ignore on read.

	float wData[4];				// The start of the array of waveform data.
};
typedef struct WaveHeader2 WaveHeader2;
typedef WaveHeader2 *WavePtr2;
typedef WavePtr2 *waveHandle2;


struct WaveHeader5 {
	//struct WaveHeader5 **next;		// link to next wave in linked list.
	ptr_t next;

	uint32_t creationDate;			// DateTime of creation.
	uint32_t modDate;			// DateTime of last modification.

	int32_t npnts;				// Total number of points (multiply dimensions up to first zero).
	int16_t type;				// See types (e.g. NT_FP64) above. Zero for text waves.
	int16_t dLock;				// Reserved. Write zero. Ignore on read.

	char whpad1[6];				// Reserved. Write zero. Ignore on read.
	int16_t whVersion;			// Write 1. Ignore on read.
	char bname[MAX_WAVE_NAME5+1];		// Name of wave plus trailing null.
	int32_t whpad2;				// Reserved. Write zero. Ignore on read.
	//struct DataFolder **dFolder;		// Used in memory only. Write zero. Ignore on read.
	ptr_t dFolder; 

	// Dimensioning info. [0] == rows, [1] == cols etc
	int32_t nDim[MAXDIMS];			// Number of of items in a dimension -- 0 means no data.
	double sfA[MAXDIMS];			// Index value for element e of dimension d = sfA[d]*e + sfB[d].
	double sfB[MAXDIMS];

	// SI units
	char dataUnits[MAX_UNIT_CHARS+1];		// Natural data units go here - null if none.
	char dimUnits[MAXDIMS][MAX_UNIT_CHARS+1];	// Natural dimension units go here - null if none.

	int16_t fsValid;			// TRUE if full scale values have meaning.
	int16_t whpad3;				// Reserved. Write zero. Ignore on read.
	double topFullScale,botFullScale;	// The max and max full scale value for wave.

	Handle dataEUnits;			// Used in memory only. Write zero. Ignore on read.
	Handle dimEUnits[MAXDIMS];		// Used in memory only. Write zero. Ignore on read.
	Handle dimLabels[MAXDIMS];		// Used in memory only. Write zero. Ignore on read.
	
	Handle waveNoteH;			// Used in memory only. Write zero. Ignore on read.
	int32_t whUnused[16];			// Reserved. Write zero. Ignore on read.

	// The following stuff is considered private to Igor.

	int16_t aModified;			// Used in memory only. Write zero. Ignore on read.
	int16_t wModified;			// Used in memory only. Write zero. Ignore on read.
	int16_t swModified;			// Used in memory only. Write zero. Ignore on read.
	
	char useBits;				// Used in memory only. Write zero. Ignore on read.
	char kindBits;				// Reserved. Write zero. Ignore on read.
	//void **formula;			// Used in memory only. Write zero. Ignore on read.
	ptr_t formula;
	int32_t depID;				// Used in memory only. Write zero. Ignore on read.
	
	int16_t whpad4;				// Reserved. Write zero. Ignore on read.
	int16_t srcFldr;			// Used in memory only. Write zero. Ignore on read.
	Handle fileName;			// Used in memory only. Write zero. Ignore on read.
	
	//int32_t **sIndices;			// Used in memory only. Write zero. Ignore on read.
	ptr_t sIndices;

	float wData[1];				// The start of the array of data. Must be 64 bit aligned.
};
typedef struct WaveHeader5 WaveHeader5;
typedef WaveHeader5 *WavePtr5;
typedef WavePtr5 *WaveHandle5;

#if __GNUC__
	#pragma pack(pop)
	#undef ptr_t
	#undef Handle
#elif GENERATINGPOWERPC
	#pragma options align=reset
#endif

// All structures written to disk are 2-byte-aligned.

#ifdef __cplusplus
}
#endif
