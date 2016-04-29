/*

% Copyright (C) 2005-2016 Alois Schloegl <alois.schloegl@ist.ac.at>
% This file is part of the "BioSig for C/C++" repository
% (biosig4c++) at http://biosig.sf.net/


    This program is free software; you can redistribute it and/or
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


/* test whether HDR.CHANNEL[].{bi,bi8} can be replaced, reduction of header size by about 3%
   currently this is not working, because FAMOS seems to need it.
//#define NO_BI
*/


/* External API definitions */

/****************************************************************************/
/**                                                                        **/
/**                 DEFINITIONS, TYPEDEFS AND MACROS                       **/
/**                                                                        **/
/****************************************************************************/

#ifndef __BIOSIG_EXT_H__
#define __BIOSIG_EXT_H__

#define BIOSIG_VERSION_MAJOR 1
#define BIOSIG_VERSION_MINOR 8
#define BIOSIG_PATCHLEVEL 0
// for backward compatibility
#define BIOSIG_VERSION_STEPPING BIOSIG_PATCHLEVEL
#define BIOSIG_VERSION (BIOSIG_VERSION_MAJOR * 10000 + BIOSIG_VERSION_MINOR * 100 + BIOSIG_PATCHLEVEL)
// biosigCHECK_VERSION returns true if BIOSIG_VERSION is at least a.b.c
#define biosigCHECK_VERSION(a,b,c) (BIOSIG_VERSION >= ( 10000*(a) + 100*(b) + (c) ) )

#if defined(_MSC_VER) && (_MSC_VER < 1600)
#if defined(_WIN64)
    typedef __int64		ssize_t;
    typedef unsigned __int64	size_t;
#else
    typedef __int32		ssize_t;
    typedef unsigned __int32	size_t;
#endif
    typedef unsigned __int64	uint64_t;
    typedef __int64		int64_t;
    typedef unsigned __int32	uint32_t;
    typedef __int32		int32_t;
    typedef unsigned __int16	uint16_t;
    typedef __int16		int16_t;
    typedef unsigned __int8	uint8_t;
    typedef __int8		int8_t;
#else
    #include <inttypes.h>
#endif
#include <stdio.h>
#include <time.h>

#include "gdftime.h"

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

/*
 * pack structures to fullfil following requirements:
 * (1) Matlab v7.3+ requires 8 byte alignment
 * (2) in order to use mingw-compiled libbiosig with MS' VisualStudio,
 *     the structurs must be packed in a MS compatible way.
 */
#pragma pack(push, 8)

//* this is probably redundant to the #pragma pack(8) statement, its here to do it the gnu way, too. */
#ifdef __GNUC__
  #define ATT_ALI __attribute__ ((aligned (8)))
  #define ATT_DEPREC __attribute__ ((deprecated))
#else
  #define ATT_ALI
  #define ATT_DEPREC
#endif

#if defined(_MINGW32__) || defined(__CYGWIN__)
  #pragma ms_struct on
  #define ATT_MSSTRUCT __attribute__ ((ms_struct))
#else
  #define ATT_MSSTRUCT
#endif

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	biosig_data_type    data type of  internal data format
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
typedef double biosig_data_type;


/****************************************************************************/
/**                                                                        **/
/**                CONSTANTS and Global variables                          **/
/**                                                                        **/
/****************************************************************************/


/* for error handling */
enum B4C_ERROR {
	B4C_NO_ERROR=0,
	B4C_FORMAT_UNKNOWN,
	B4C_FORMAT_UNSUPPORTED,
	B4C_CANNOT_OPEN_FILE,
	B4C_CANNOT_WRITE_FILE,
	B4C_CANNOT_APPEND_FILE,
	B4C_INSUFFICIENT_MEMORY,
	B4C_ENDIAN_PROBLEM,
	B4C_CRC_ERROR,
	B4C_DATATYPE_UNSUPPORTED,
	B4C_SCLOSE_FAILED,
	B4C_DECOMPRESSION_FAILED,
	B4C_MEMORY_ALLOCATION_FAILED,
	B4C_RAWDATA_COLLAPSING_FAILED,
	B4C_REREF_FAILED,
	B4C_INCOMPLETE_FILE,
	B4C_UNSPECIFIC_ERROR,
	B4C_CHAR_ENCODING_UNSUPPORTED
};

#ifdef BIN
#undef BIN 	// needed for biosig4perl
#endif
#ifdef EVENT
#undef EVENT 	// defined by MS VC++
#endif

	/* list of file formats */
enum FileFormat {
	noFile, unknown,
	ABF, ABF2, ACQ, ACR_NEMA, AIFC, AIFF, AINF, alpha, ARFF,
	ASCII_IBI, ASCII, AU, ASF, ATES, ATF, AVI, AXG, Axona,
	BCI2000, BDF, BESA, BIN, BKR, BLSC, BMP, BNI, BSCS,
	BrainVision, BrainVisionVAmp, BrainVisionMarker, BZ2,
	CDF, CFS, CFWB, CNT, CTF, DICOM, DEMG,
	EBS, EDF, EEG1100, EEProbe, EEProbe2, EEProbeAvr, EGI,
	EGIS, ELF, EMBLA, EMSA, ePrime, ET_MEG, ETG4000, EVENT, EXIF,
	FAMOS, FEF, FIFF, FITS, FLAC, GDF, GDF1,
	GIF, GTF, GZIP, HDF, HL7aECG, HEKA,
	IBW, ISHNE, ITX, JPEG, JSON, Lexicor,
	Matlab, MFER, MIDI, MIT, MM, MSI, MSVCLIB, MS_LNK, MX,
	native, NeuroLoggerHEX, NetCDF, NEURON, NEV, NEX1, NIFTI, NUMPY,
	OGG, OpenXDF,
	PBMA, PBMN, PDF, PDP, Persyst, PGMA, PGMB,
	PLEXON, PNG, PNM, POLY5, PPMA, PPMB, PS,
	RDF, RIFF,
	SASXPT, SCP_ECG, SIGIF, Sigma, SMA, SMR, SND, SQLite,
	SPSS, STATA, SVG, SXI, SYNERGY,
	TDMS, TIFF, TMS32, TMSiLOG, TRC, UNIPRO, VRML, VTK,
	WAV, WCP, WG1, WinEEG, WMF, XML, XPM,
	Z, ZIP, ZIP2
};

#if (BIOSIG_VERSION >= 10900)
#error ABI change: order enum FileFormat alphabethically
#endif

/*
   error handling should use error variables local to each HDR
   otherwise, sopen() etc. is not re-entrant.

   Therefore, use of variables B4C_ERRNUM and B4C_ERRMSG is deprecated;
   Instead, use biosigERROR for setting error status, and
   serror2(hdr), hdr->AS.B4C_ERRNUM, hdr->AS.B4C_ERRMSG for error handling.

 */
ATT_DEPREC extern int B4C_ERRNUM;
ATT_DEPREC extern const char *B4C_ERRMSG;


/*
This part has moved into biosig-dev.h in v1.4.1, because VERBOSE_LEVEL is just
used for debugging and should not be exposed to common applications
#ifdef NDEBUG
#define VERBOSE_LEVEL 0		// turn off debugging information
#else
extern int VERBOSE_LEVEL; 	// used for debugging
#endif
*/

/****************************************************************************/
/**                                                                        **/
/**                 DEFINITIONS, TYPEDEFS AND MACROS                       **/
/**                                                                        **/
/****************************************************************************/



typedef int64_t 		nrec_t;	/* type for number of records */

/****************************************************************************/
/**                                                                        **/
/**                     TYPEDEFS AND STRUCTURES                            **/
/**                                                                        **/
/****************************************************************************/

/*
	This structure defines the header for each channel (variable header)
 */
// TODO: change fixed length strings to dynamically allocated strings
#define MAX_LENGTH_LABEL 	80	// TMS: 40, AXG: 79
#define MAX_LENGTH_TRANSDUCER 	80
#if (BIOSIG_VERSION < 10600)
  #define MAX_LENGTH_PHYSDIM    20	// DEPRECATED - DO NOT USE
#else
  #undef MAX_LENGTH_PHYSDIM
#endif
#define MAX_LENGTH_PID	 	80  	// length of Patient ID: MFER<65, GDF<67, EDF/BDF<81, etc.
#define MAX_LENGTH_RID		80	// length of Recording ID: EDF,GDF,BDF<80, HL7 ?
#define MAX_LENGTH_NAME 	132	// max length of personal name: MFER<=128, EBS<=33*4
#define MAX_LENGTH_MANUF 	128	// max length of manufacturer field: MFER<128
#define MAX_LENGTH_TECHNICIAN 	128	// max length of manufacturer field: SCP<41

typedef struct CHANNEL_STRUCT {
	double 		PhysMin ATT_ALI;	/* physical minimum */
	double 		PhysMax ATT_ALI;	/* physical maximum */
	double 		DigMin 	ATT_ALI;	/* digital minimum */
	double	 	DigMax 	ATT_ALI;	/* digital maximum */
	double		Cal 	ATT_ALI;	/* gain factor */
	double		Off 	ATT_ALI;	/* bias */

	char		Label[MAX_LENGTH_LABEL+1] ATT_ALI; 	/* Label of channel */
	char		OnOff	ATT_ALI;	/* 0: channel is off, not consider for data output; 1: channel is turned on; 2: channel containing time axis */
	uint16_t	LeadIdCode ATT_ALI;	/* Lead identification code */
	char 		Transducer[MAX_LENGTH_TRANSDUCER+1] ATT_ALI;	/* transducer e.g. EEG: Ag-AgCl electrodes */
#ifdef MAX_LENGTH_PHYSDIM
        char            PhysDim[MAX_LENGTH_PHYSDIM+1] ATT_ALI ATT_DEPREC;       /* DONOT USE - use PhysDim3(PhysDimCode) instead */
#endif
	uint16_t	PhysDimCode ATT_ALI;	/* code for physical dimension - PhysDim3(PhysDimCode) returns corresponding string */

	float 		TOffset 	ATT_ALI;	/* time delay of sampling */
	float 		LowPass		ATT_ALI;	/* lowpass filter */
	float 		HighPass	ATT_ALI;	/* high pass */
	float 		Notch		ATT_ALI;	/* notch filter */
	float 		XYZ[3]		ATT_ALI;	/* sensor position */

	union {
        /* context specific channel information */
	float 		Impedance	ATT_ALI;   	/* Electrode Impedance in Ohm, defined only if PhysDim = _Volt */
	float 		fZ        	ATT_ALI;   	/* ICG probe frequency, defined only if PhysDim = _Ohm */
	} ATT_ALI;

	/* this part should not be used by application programs */
	uint8_t*	bufptr		ATT_ALI;	/* pointer to buffer: NRec<=1 and bi,bi8 not used */
	uint32_t 	SPR 		ATT_ALI;	/* samples per record (block) */
	uint32_t	bi 		ATT_ALI;	/* start byte (byte index) of channel within data block */
	uint32_t	bi8 		ATT_ALI;	/* start bit  (bit index) of channel within data block */
	uint16_t 	GDFTYP 		ATT_ALI;	/* data type */
} CHANNEL_TYPE	ATT_ALI ATT_MSSTRUCT;


/*
	This structure defines the general (fixed) header
*/
typedef struct HDR_STRUCT {

	char* 	        FileName ATT_ALI;       /* FileName - dynamically allocated, local copy of file name */

	union {
#ifndef VERSION
	// workaround in case VERSION is already defined as macro, kept for backwards compatibility
	float 		VERSION;		/* GDF version number */
#endif
	float 		Version;		/* GDF version number */
	} ATT_ALI;

	union {
#ifndef TYPE
	// workaround in case TYPE is already defined as macro, kept for backwards compatibility
	enum FileFormat TYPE;		 	/* type of file format */
#endif
	enum FileFormat Type; 			/* type of file format */
	} ATT_ALI;

	struct {
		size_t 			size[2] ATT_ALI; /* size {rows, columns} of data block	 */
		biosig_data_type* 	block ATT_ALI; 	 /* data block */
	} data ATT_ALI;

	uint8_t 	IPaddr[16] ATT_ALI; 	/* IP address of recording device (if applicable) */
	double 		SampleRate ATT_ALI;	/* Sampling rate */
	nrec_t  	NRec 	ATT_ALI;	/* number of records/blocks -1 indicates length is unknown. */
	gdf_time 	T0 	ATT_ALI; 	/* starttime of recording */
	uint32_t 	HeadLen ATT_ALI;	/* length of header in bytes */
	uint32_t 	SPR 	ATT_ALI;	/* samples per block (when different sampling rates are used, this is the LCM(CHANNEL[..].SPR) */
	uint32_t  	LOC[4] 	ATT_ALI;	/* location of recording according to RFC1876 */
	uint16_t 	NS 	ATT_ALI;	/* number of channels */
	int16_t 	tzmin 	ATT_ALI;	/* time zone : minutes east of UTC */

#ifdef CHOLMOD_H
	cholmod_sparse  *Calib ATT_ALI;                  /* re-referencing matrix */
#else
        void        *Calib ATT_ALI;                  /* re-referencing matrix */
#endif
	CHANNEL_TYPE 	*rerefCHANNEL ATT_ALI;

	/* Patient specific information */
	struct {
		gdf_time 	Birthday; 	/* Birthday of Patient */
		// 		Age;		// the age is HDR.T0 - HDR.Patient.Birthday, even if T0 and Birthday are not known
		uint16_t	Headsize[3]; 	/* circumference, nasion-inion, left-right mastoid in millimeter;  */
		char		Name[MAX_LENGTH_NAME+1]; /* because for privacy protection it is by default not supported, support is turned on with FLAG.ANONYMOUS */
//		char*		Name;	// because for privacy protection it is by default not supported, support is turned on with FLAG.ANONYMOUS
		char		Id[MAX_LENGTH_PID+1];	/* patient identification, identification code as used in hospital  */
		uint8_t		Weight;		/* weight in kilograms [kg] 0:unkown, 255: overflow  */
		uint8_t		Height;		/* height in centimeter [cm] 0:unkown, 255: overflow  */
		//		BMI;		// the body-mass index = weight[kg]/height[m]^2
		/* Patient classification */
		int8_t	 	Sex;		/* 0:Unknown, 1: Male, 2: Female */
		int8_t		Handedness;	/* 0:Unknown, 1: Right, 2: Left, 3: Equal */
		int8_t		Smoking;	/* 0:Unknown, 1: NO, 2: YES */
		int8_t		AlcoholAbuse;	/* 0:Unknown, 1: NO, 2: YES */
		int8_t		DrugAbuse;	/* 0:Unknown, 1: NO, 2: YES */
		int8_t		Medication;	/* 0:Unknown, 1: NO, 2: YES */
		struct {
			int8_t 	Visual;		/* 0:Unknown, 1: NO, 2: YES, 3: Corrected */
			int8_t 	Heart;		/* 0:Unknown, 1: NO, 2: YES, 3: Pacemaker */
		} Impairment;
	} Patient ATT_ALI;

	struct {
		char		Recording[MAX_LENGTH_RID+1]; 	/* HL7, EDF, GDF, BDF replaces HDR.AS.RID */
		char* 		Technician;
		char* 		Hospital;	/* recording institution */
		uint64_t 	Equipment; 	/* identifies this software */
		struct {
			/* see
				SCP: section1, tag14,
				MFER: tag23:  "Manufacturer^model^version number^serial number"
			*/
			const char*	Name;
			const char*	Model;
			const char*	Version;
			const char*	SerialNumber;
			char	_field[MAX_LENGTH_MANUF+1];	/* buffer */
		} Manufacturer;
	} ID ATT_ALI;

	/* position of electrodes; see also HDR.CHANNEL[k].XYZ */
	struct {
		float		REF[3];	/* XYZ position of reference electrode */
		float		GND[3];	/* XYZ position of ground electrode */
	} ELEC ATT_ALI;

	/* EVENTTABLE */
	struct {
		double  	SampleRate ATT_ALI;	/* for converting POS and DUR into seconds  */
		uint16_t 	*TYP ATT_ALI;	/* defined at http://biosig.svn.sourceforge.net/viewvc/biosig/trunk/biosig/doc/eventcodes.txt */
		uint32_t 	*POS ATT_ALI;	/* starting position [in samples] using a 0-based indexing */
		uint32_t 	*DUR ATT_ALI;	/* duration [in samples] */
		uint16_t 	*CHN ATT_ALI;	/* channel number; 0: all channels  */
#if (BIOSIG_VERSION >= 10500)
		gdf_time        *TimeStamp ATT_ALI;  /* store time stamps */
#endif
		const char*	*CodeDesc ATT_ALI;	/* describtion of "free text"/"user specific" events (encoded with TYP=0..255 */
		uint32_t  	N ATT_ALI;	/* number of events */
		uint16_t	LenCodeDesc ATT_ALI;	/* length of CodeDesc Table */
	} EVENT ATT_ALI;

	struct {	/* flags */
		char		OVERFLOWDETECTION; 	/* overflow & saturation detection 0: OFF, !=0 ON */
		char		UCAL; 		/* UnCalibration  0: scaling  !=0: NO scaling - raw data return  */
		char		ANONYMOUS; 	/* 1: anonymous mode, no personal names are processed */
		char		ROW_BASED_CHANNELS;     /* 0: column-based data [default]; 1: row-based data */
		char		TARGETSEGMENT; /* in multi-segment files (like Nihon-Khoden, EEG1100), it is used to select a segment */
	} FLAG ATT_ALI;

	CHANNEL_TYPE 	*CHANNEL ATT_ALI;
		// moving CHANNEL after the next struct (HDR.FILE) gives problems at AMD64 MEX-file.
		// perhaps some alignment problem.

	struct {	/* File specific data  */
#ifdef ZLIB_H
		gzFile		gzFID;
#else
		void*		gzFID;
#endif
#ifdef _BZLIB_H
//		BZFILE*		bzFID;
#endif
		FILE* 		FID;		/* file handle  */
		size_t 		size;		/* size of file - experimental: only partly supported */
		size_t 		POS;		/* current reading/writing position [in blocks] */
		//size_t 	POS2;		// current reading/writing position [in samples] */
		int		Des;		/* file descriptor */
		int		DES;		/* descriptor for streams */
		uint8_t		OPEN; 		/* 0: closed, 1:read, 2: write */
		uint8_t		LittleEndian;   /* 1 if file is LittleEndian data format and 0 for big endian data format*/
		uint8_t		COMPRESSION;    /* 0: no compression 9: best compression */
	} FILE ATT_ALI;

	/*	internal variables (not public)  */
	struct {
		const char*	B4C_ERRMSG;	/* error message */
//		char 		PID[MAX_LENGTH_PID+1];	// use HDR.Patient.Id instead
//		char* 		RID;		// recording identification
		uint32_t 	bpb;  		/* total bytes per block */
		uint32_t 	bpb8;  		/* total bits per block */

		uint8_t*	Header;
		uint8_t*	rawEventData;
		uint8_t*	rawdata; 	/* raw data block */
		size_t		first;		/* first block loaded in buffer - this is equivalent to hdr->FILE.POS */
		size_t		length;		/* number of block(s) loaded in buffer */
		uint8_t*	auxBUF;  	/* auxillary buffer - used for storing EVENT.CodeDesc, MIT FMT infor, alpha:rawdata header */
		union {
		    char*	bci2000;	/* application specific free text field */
		    char*	fpulse;
		    char*	stimfit;
		};
		uint32_t	SegSel[5];	/* segment selection in a hirachical data formats, e.g. sweeps in HEKA/PatchMaster format */
		enum B4C_ERROR	B4C_ERRNUM;	/* error code */
		char		flag_collapsed_rawdata; /* 0 if rawdata contain obsolete channels, too. 	*/
	} AS ATT_ALI;

	void *aECG;				/* used as an pointer to (non-standard) auxilary information - mostly used for hacks */
	uint64_t viewtime; 			/* used by edfbrowser */

#if (BIOSIG_VERSION >= 10500)
	struct {
		/*
			This part contains Section 7-11 of the SCP-ECG format
			without its 16 byte "Section ID header".
			These sections are also stored in GDF Header 3 (tag 9-13)
			It is mostly used for SCP<->GDF conversion.

			The pointers points into hdr->AS.Header,
			so do not dynamically re-allocate the pointers.
		*/
		const uint8_t* Section7;
		const uint8_t* Section8;
		const uint8_t* Section9;
		const uint8_t* Section10;
		const uint8_t* Section11;
		uint32_t Section7Length;
		uint32_t Section8Length;
		uint32_t Section9Length;
		uint32_t Section10Length;
		uint32_t Section11Length;
	} SCP;
#endif

} HDRTYPE ATT_MSSTRUCT;

/*
	This structure defines codes and groups of the event table
 */

// Desription of event codes
struct etd_t {
        uint16_t typ;		// used in HDR.EVENT.TYP
        uint16_t groupid;	// defines the group id as used in EventCodeGroups below
        const char* desc;	// name/description of event code // const decrease signifitiantly number of warning
} ATT_MSSTRUCT;
// Groups of event codes
struct event_groups_t {
        uint16_t groupid;
        const char* GroupDescription; // const decrease signifitiantly number of warning
} ATT_MSSTRUCT;
struct FileFormatStringTable_t {
	enum FileFormat	fmt;
	const char*	FileTypeString;
} ATT_MSSTRUCT;
struct NomenclatureAnnotatedECG_t {
	uint16_t part;
	uint16_t code10;
	uint32_t cf_code10;
	const char *refid;
} ATT_MSSTRUCT;

extern const struct etd_t ETD [];
extern const struct event_groups_t EventCodeGroups [];
extern const struct FileFormatStringTable_t FileFormatStringTable [];


/* reset structure packing to default settings */
#pragma pack(pop)
#if defined(_MINGW32__) || defined(__CYGWIN__)
#pragma ms_struct reset
#endif

/****************************************************************************/
/**                                                                        **/
/**                     EXPORTED FUNCTIONS                                 **/
/**                                                                        **/
/****************************************************************************/

#ifdef __cplusplus
EXTERN_C {
#endif

uint32_t get_biosig_version ();
/* 	returns the version number in hex-decimal representation
	get_biosig_version() & 0x00ff0000 :  major version number
	get_biosig_version() & 0x0000ff00 :  minor version number
	get_biosig_version() & 0x000000ff :  patch level
 --------------------------------------------------------------- */

HDRTYPE* constructHDR(const unsigned NS, const unsigned N_EVENT);
/* 	allocates memory initializes header HDR of type HDRTYPE
	with NS channels an N_EVENT event elements
 --------------------------------------------------------------- */


void 	 destructHDR(HDRTYPE* hdr);
/* 	destroys the header *hdr and frees allocated memory
 --------------------------------------------------------------- */

HDRTYPE* getfiletype(HDRTYPE* hdr);
/* 	identify file format from header information
	input:
		hdr->AS.Header contains header of hdr->HeadLen bytes
		hdr->TYPE must be unknown, otherwise no FileFormat evaluation is performed
	output:
		hdr->TYPE	file format
		hdr->VERSION	is defined for some selected formats e.g. ACQ, EDF, BDF, GDF
 --------------------------------------------------------------- */

HDRTYPE* sopen(const char* FileName, const char* MODE, HDRTYPE* hdr);
/*	FileName: name of file
	Mode: "r" is reading mode, requires FileName
	Mode: "w" is writing mode, hdr contains the header information
		If the number of records is not known, set hdr->NRec=-1 and
		sclose will fill in the correct number.
	Mode: "a" is append mode,
		if file exists, header and eventtable is read,
		position pointer is set to end of data in order to add
		more data. If file is successfully opened, the header structure
		of the existing file is used, and any different specification in
		hdr is discarded.
		If file is not compressed, it can be used for read and write,
  		for compressed files, only appending at the end of file is possible.
		Currently, append mode is supported only for the GDF format.

	hdr should be generated with constructHDR, and the necessary fields
	must be defined. In read-mode, hdr can be NULL; however,
	hdr->FLAG... can be used to turn off spurious warnings. In write-mode,
	the whole header information must be defined.
	In append mode, it is recommended to provide whole header information,
	which must be equivalent to the header info of an existing file.
	After calling sopen, the file header is read or written, and
	the position pointer points to the beginning of the data section
	in append mode, the position pointer points to the end of the data section.
 --------------------------------------------------------------- */

int 	sclose(HDRTYPE* hdr);
/* 	closes the file corresponding to hdr
	file handles are closed, the position pointer becomes meaningless
	Note: hdr is not destroyed; use destructHDR() to free the memory of hdr
	if hdr was opened in writing mode, the event table is added to the file
	and if hdr->NRec=-1, the number of records is obtained from the
	    position pointer and written into the header,
 --------------------------------------------------------------- */

size_t	sread(biosig_data_type* DATA, size_t START, size_t LEN, HDRTYPE* hdr);
/*	LEN data segments are read from file associated with hdr, starting from
	segment START. The data is copied into DATA; if DATA == NULL, a
	sufficient amount of memory is allocated, and the pointer to the data
	is available in hdr->data.block.

	In total, LEN*hdr->SPR*NS samples are read and stored in
	data type of biosig_data_type (currently double).
	NS is the number of channels with non-zero hdr->CHANNEL[].OnOff.
	The number of successfully read data blocks is returned.

 	A pointer to the data block is also available from hdr->data.block,
	the number of columns and rows is available from
	hdr->data.size[0] and hdr->data.size[1] respectively.

	Channels k with (hdr->CHANNEL[k].SPR==0) are interpreted as sparsely
	sampled channels [for details see specification ofGDF v2 or larger].
	The sample values are also returned in DATA the corresponding
  	sampling time, the values in between the sparse sampling times are
	set to DigMin. (Applying the flags UCAL and OVERFLOWDETECTION will
	convert this into PhysMin and NaN, resp. see below).

	The following flags will influence the result.
 	hdr->FLAG.UCAL = 0 	scales the data to its physical values
 	hdr->FLAG.UCAL = 1 	does not apply the scaling factors

 	hdr->FLAG.OVERFLOWDETECTION = 0 does not apply overflow detection
 	hdr->FLAG.OVERFLOWDETECTION = 1: replaces all values that exceed
		the dynamic range (defined by Phys/Dig/Min/Max)

	hdr->FLAG.ROW_BASED_CHANNELS = 0 each channel is in one column
	hdr->FLAG.ROW_BASED_CHANNELS = 1 each channel is in one row
 --------------------------------------------------------------- */

#ifdef __GSL_MATRIX_DOUBLE_H__
size_t	gsl_sread(gsl_matrix* DATA, size_t START, size_t LEN, HDRTYPE* hdr);
/*	same as sread but return data is of type gsl_matrix
 --------------------------------------------------------------- */
#endif

size_t  swrite(const biosig_data_type *DATA, size_t NELEM, HDRTYPE* hdr);
/*	DATA contains the next NELEM data segment(s) for writing.
 *	hdr contains the definition of the header information and was generated by sopen
 *	the number of successfully written segments is returned;
 --------------------------------------------------------------- */


int	seof(HDRTYPE* hdr);
/*	returns 1 if end of file is reached.
 --------------------------------------------------------------- */


void	srewind(HDRTYPE* hdr);
/*	postions file pointer to the beginning
 *
 *	Currently, this function is meaning less because sread requires always the start value
 --------------------------------------------------------------- */


int 	sseek(HDRTYPE* hdr, long int offset, int whence);
/*	positions file pointer
 *
 *	Currently, this function is meaning less because sread requires always the start value
 --------------------------------------------------------------- */


long int stell(HDRTYPE* hdr);
/*	returns position of file point in segments
 --------------------------------------------------------------- */

#ifndef  ONLYGDF
ATT_DEPREC int serror();
/*	handles errors; it reports whether an error has occured.
 *	if yes, an error message is displayed, and the error status is reset.
 * 	the return value is 0 if no error has occured, otherwise the error code
 *	is returned.
 *  IMPORTANT NOTE:
 *	serror() uses the global error variables B4C_ERRNUM and B4C_ERRMSG,
 *	which is not re-entrant, because two opened files share the same
 *	error variables.
 --------------------------------------------------------------- */
#endif //ONLYGDF

int 	serror2(HDRTYPE* hdr);
/*	handles errors; it reports whether an error has occured.
 *	if yes, an error message is displayed, and the error status is reset.
 * 	the return value is 0 if no error has occured, otherwise the error code
 *	is returned.
 --------------------------------------------------------------- */

int 	biosig_check_error(HDRTYPE *hdr);
/* 	returns error status but does not handle/reset it.
 * 	it can be used for checking whether some error status has been set
 --------------------------------------------------------------- */
char*   biosig_get_errormsg(HDRTYPE *hdr);
/* 	returns error message but does not reset it.
 * 	memory for the error message is allocated and need to be freed
 *      by the calling application
 --------------------------------------------------------------- */


int 	sflush_gdf_event_table(HDRTYPE* hdr);
/*	writes the event table of file hdr. hdr must define a file in GDF format
 *  	and can be opened as read or write.
 *	In case of success, the return value is 0.
 --------------------------------------------------------------- */

int 	cachingWholeFile(HDRTYPE* hdr);
/*	caching: load data of whole file into buffer
 *		 this will speed up data access, especially in interactive mode
 --------------------------------------------------------------- */


int RerefCHANNEL(HDRTYPE *hdr, void *ReRef, char rrtype);
/* rerefCHAN
        defines rereferencing of channels,
        hdr->Calib defines the rereferencing matrix
        hdr->rerefCHANNEL is defined.
        hdr->rerefCHANNEL[.].Label is  by some heuristics from hdr->CHANNEL
                either the maximum scaling factor
        if ReRef is NULL, rereferencing is turned off (hdr->Calib and
        hdr->rerefCHANNEL are reset to NULL).
        if rrtype==1, Reref is a filename pointing to a MatrixMarket file
        if rrtype==2, Reref must be a pointer to a cholmod sparse matrix (cholmod_sparse*)
        In case of an error (mismatch of dimensions), a non-zero is returned,
        and serror() is set.

        rr is a pointer to a rereferencing matrix
        rrtype determines the type of pointer
        rrtype=0: no rereferencing, RR is ignored (NULL)
               1: pointer to MarketMatrix file (char*)
               2: pointer to a sparse cholmod matrix  (cholmod_sparse*)
 ------------------------------------------------------------------------*/

const char* GetFileTypeString(enum FileFormat FMT);
/*	returns a string with file format
 --------------------------------------------------------------- */

enum FileFormat GetFileTypeFromString(const char *);
/*	returns file format from string
 --------------------------------------------------------------- */


/* =============================================================
	utility functions for handling of event table
   ============================================================= */

void sort_eventtable(HDRTYPE *hdr);
/* sort event table with respect to hdr->EVENT.POS
  --------------------------------------------------------------*/

size_t reallocEventTable(HDRTYPE *hdr, size_t EventN);
/*------------------------------------------------------------------------
	re-allocates memory for Eventtable. 
	hdr->EVENT.N contains actual number of events
	EventN determines the size of the allocated memory

  return value:
	in case of success, EVENT_N is returned
	in case of failure SIZE_MAX is returned;
  ------------------------------------------------------------------------*/

void convert2to4_eventtable(HDRTYPE *hdr);
/* converts event table from {TYP,POS} to [TYP,POS,CHN,DUR} format
  -------------------------------------------------------------- */

void convert4to2_eventtable(HDRTYPE *hdr);
/* converts event table from [TYP,POS,CHN,DUR} to {TYP,POS} format
	all CHN[k] must be 0
  -------------------------------------------------------------- */

const char* GetEventDescription(HDRTYPE *hdr, size_t n);
/* returns clear text description of n-th event,
   considers also user-defined events.
  -------------------------------------------------------------- */

void FreeTextEvent(HDRTYPE* hdr, size_t N, const char* annotation);
/*  adds free text annotation to event table for the N-th event.
	the EVENT.TYP[N] is identified from the table EVENT.CodeDesc
	if annotations is not listed in CodeDesc, it is added to CodeDesc
	The table is limited to 256 entries, because the table EventCodes
	allows only codes 0-255 as user specific entry. If the description
	table contains more than 255 entries, an error is set.
  ------------------------------------------------------------------------*/

/* =============================================================
	utility functions for handling of physical dimensons
   ============================================================= */

uint16_t PhysDimCode(const char* PhysDim);
/* Encodes  Physical Dimension as 16bit integer according to
   ISO/IEEE 11073-10101:2004 Vital Signs Units of Measurement
 --------------------------------------------------------------- */

char* PhysDim(uint16_t PhysDimCode, char *PhysDimText);
/* DEPRECATED: USE INSTEAD PhysDim3(uint16_t PhysDimCode)
   It's included just for backwards compatibility
   converts HDR.CHANNEL[k].PhysDimCode into a readable Physical Dimension
   the memory for PhysDim must be preallocated, its maximum length is
   defined by (MAX_LENGTH_PHYSDIM+1)
 --------------------------------------------------------------- */

const char* PhysDim3(uint16_t PhysDimCode);
/* converts PhysDimCode into a readable Physical Dimension
 --------------------------------------------------------------- */

double PhysDimScale(uint16_t PhysDimCode);
/* returns scaling factor of physical dimension
	e.g. 0.001 for milli, 1000 for kilo etc.
 --------------------------------------------------------------- */

int biosig_set_hdr_ipaddr(HDRTYPE *hdr, const char *hostname);
/* set the field HDR.IPaddr based on the IP address of hostname

   Return value:
	0: hdr->IPaddr is set
	otherwise hdr->IPaddr is not set
  ---------------------------------------------------------------*/


/* =============================================================
	printing of header information
   ============================================================= */

int	hdr2ascii(HDRTYPE* hdr, FILE *fid, int VERBOSITY);
/*	writes the header information is ascii format the stream defined by fid
 *	Typically fid is stdout. VERBOSITY defines how detailed the information is.
 *	VERBOSITY=0 or 1 report just some basic information,
 *	VERBOSITY=2 reports als the channel information
 *	VERBOSITY=3 provides in addition the event table.
 *	VERBOSITY=8 for debugging
 *	VERBOSITY=9 for debugging
 *	VERBOSITY=-1 header and event table is shown in JSON format
 --------------------------------------------------------------- */

ATT_DEPREC int hdr2json (HDRTYPE *hdr, FILE *fid);
int fprintf_hdr2json(FILE *stream, HDRTYPE* hdr);
/* prints header in json format into stream;
   hdr2json is the old form and deprecated,
   use fprintf_hdr2json instead
 --------------------------------------------------------------- */

int asprintf_hdr2json(char **str, HDRTYPE* hdr);
/* prints header in json format into *str;
   memory for str is automatically allocated and must be freed
   after usage.
 --------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif


/****************************************************************************/
/**                                                                        **/
/**                               EOF                                      **/
/**                                                                        **/
/****************************************************************************/

#endif	/* __BIOSIG_EXT_H__ */

