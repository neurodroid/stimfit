// This file contains an example of writing a wave file.

#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include "IgorBin.h"
#include "CrossPlatformFileIO.h"

/*	Checksum(data,oldcksum,numbytes)

 	Returns shortwise simpleminded checksum over the data.
	ASSUMES data starts on an even boundary.
*/
static int
Checksum(short *data, int oldcksum, int numbytes)
{
	numbytes >>= 1;				// 2 bytes to a short -- ignore trailing odd byte.
	while(numbytes-- > 0)
		oldcksum += *data++;
	return oldcksum&0xffff;
}

/*	NumBytesPerPoint(int type)
	
	Given a numeric wave type, returns the number of data bytes per point.
*/
static int
NumBytesPerPoint(int type)
{
	int numBytesPerPoint;
	
	// Consider the number type, not including the complex bit or the unsigned bit.
	switch(type & ~(NT_CMPLX | NT_UNSIGNED)) {
		case NT_I8:
			numBytesPerPoint = 1;		// char
			break;
		case NT_I16:
			numBytesPerPoint = 2;		// short
			break;
		case NT_I32:
			numBytesPerPoint = 4;		// IGORLONG
			break;
		case NT_FP32:
			numBytesPerPoint = 4;		// float
			break;
		case NT_FP64:
			numBytesPerPoint = 8;		// double
			break;
		default:
			return 0;
			break;
	}

	if (type & NT_CMPLX)
		numBytesPerPoint *= 2;			// Complex wave - twice as many points.
	
	return numBytesPerPoint;
}

/*	WriteVersion2NumericWave(fr, whp, data, waveNote, noteSize)

	Writes an Igor version 2 binary wave with the properties specified in
	whp, the data specified by data, and the wave note specified by waveNote
	and noteSize.
	
	Returns 0 or an error code.
*/
static int
WriteVersion2NumericWave(CP_FILE_REF fr, WaveHeader2* whp, const void* data, const char* waveNote, long noteSize)
{
	unsigned long numBytesToWrite;
	unsigned long numBytesWritten;
	unsigned long waveDataSize;
	int numBytesPerPoint;
	short cksum;
	BinHeader2 bh;
	char padding[16];
	int err;
	
	numBytesPerPoint = NumBytesPerPoint(whp->type);
	if (numBytesPerPoint <= 0) {
		printf("Invalid wave type (0x%x).\n", whp->type);
		return -1;
	}
	waveDataSize = whp->npnts * numBytesPerPoint;
	
	// Prepare the BinHeader structure.
	memset(&bh,0,sizeof(struct BinHeader2));
	bh.version = 2;
	bh.wfmSize = offsetof(WaveHeader2, wData) + waveDataSize + 16;	// Includes 16 bytes padding.
	bh.noteSize = noteSize;

	/*	The checksum is over the BinHeader2 structure and the WaveHeader2 structure.
		The wData field of the WaveHeader2 structure is assumed to contain the same
		data as the first 16 bytes of the actual wave data. This is necessary
		to get the correct checksum.
	*/
	cksum = Checksum((short *)&bh, 0, sizeof(struct BinHeader2));
	cksum = Checksum((short *)whp, cksum, sizeof(struct WaveHeader2));
	bh.checksum = -cksum;

	do {
		// Write the BinHeader.
		numBytesToWrite = sizeof(struct BinHeader2);
		if (err = CPWriteFile(fr, numBytesToWrite, &bh, &numBytesWritten))
			break;
		
		// Write the WaveHeader, up to but not including the wData field.
		numBytesToWrite = offsetof(WaveHeader2, wData);
		if (err = CPWriteFile(fr, numBytesToWrite, whp, &numBytesWritten))
			break;
		
		// Write the wave data.
		numBytesToWrite = waveDataSize;
		if (err = CPWriteFile(fr, numBytesToWrite, data, &numBytesWritten))
			break;

		// Write the 16 byte padding.
		memset(padding, 0, 16);								// Write padding at the end of the wave data.
		numBytesToWrite = 16;
		if (err = CPWriteFile(fr, numBytesToWrite, padding, &numBytesWritten))
			break;
			
		// Now write optional data, in the correct order.
		
		// Write the wave note.
		numBytesToWrite = noteSize;
		if (numBytesToWrite > 0) {
			if (err = CPWriteFile(fr, numBytesToWrite, waveNote, &numBytesWritten))
				break;
		}
					
	} while(0);

	return err;
}

/*	WriteVersion5NumericWave(fr, whp, data, waveNote, noteSize)

	Writes an Igor version 5 binary wave with the properties specified in
	whp, the data specified by data, and the wave note specified by waveNote
	and noteSize.
	
	Returns 0 or an error code.
*/
int
WriteVersion5NumericWave(CP_FILE_REF fr, WaveHeader5* whp, const void* data, const char* waveNote, long noteSize)
{
        unsigned long numBytesToWrite;
	unsigned long numBytesWritten;
	unsigned IGORLONG waveDataSize;
	int numBytesPerPoint;
	short cksum;
	BinHeader5 bh;
	int err;

        printf("sizeof(short): %lu\n", sizeof(short));
        printf("sizeof(int): %lu\n", sizeof(int));
        printf("sizeof(long): %lu\n", sizeof(long));
        printf("sizeof(IGORLONG): %lu\n", sizeof(IGORLONG));
        printf("sizeof(unsigned long): %lu\n", sizeof(unsigned long));
        printf("sizeof(unsigned IGORLONG): %lu\n", sizeof(unsigned IGORLONG));
        printf("sizeof(float): %lu\n", sizeof(float));
        printf("sizeof(double): %lu\n", sizeof(double));
	numBytesPerPoint = NumBytesPerPoint(whp->type);
	if (numBytesPerPoint <= 0) {
		printf("Invalid wave type (0x%x).\n", whp->type);
		return -1;
	}
	waveDataSize = whp->npnts * numBytesPerPoint;
	// Prepare the BinHeader structure.
	memset(&bh,0,sizeof(struct BinHeader5));
	bh.version = 5;
	bh.wfmSize = offsetof(WaveHeader5, wData) + waveDataSize;
	bh.noteSize = noteSize;

	/*	The checksum is over the BinHeader5 structure and the WaveHeader5 structure,
		excluding the wData field.
	*/
	cksum = Checksum((short *)&bh, 0, sizeof(BinHeader5));
        printf("%d\n", cksum);
	cksum = Checksum((short *)whp, cksum, offsetof(WaveHeader5, wData));
        printf("%d\n", cksum);
	bh.checksum = -cksum;
        printf("%d\n", bh.checksum);
	do {
		// Write the BinHeader.
		numBytesToWrite = sizeof(struct BinHeader5);
		if (err = CPWriteFile(fr, numBytesToWrite, &bh, &numBytesWritten))
			break;
		
		// Write the WaveHeader, up to but not including the wData field.
		numBytesToWrite = offsetof(WaveHeader5, wData);
		if (err = CPWriteFile(fr, numBytesToWrite, whp, &numBytesWritten))
			break;
		
		// Write the wave data.
		numBytesToWrite = waveDataSize;
		if (err = CPWriteFile(fr, numBytesToWrite, data, &numBytesWritten))
			break;
			
		// Now write optional data, in the correct order.
		
		// Write the wave note.
		numBytesToWrite = noteSize;
		if (numBytesToWrite > 0) {
			if (err = CPWriteFile(fr, numBytesToWrite, waveNote, &numBytesWritten))
				break;
		}
					
	} while(0);

	return err;
}
