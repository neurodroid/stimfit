/*

% Copyright (C) 2005-2018 Alois Schloegl <alois.schloegl@gmail.com>
% This file is part of the "BioSig for C/C++" repository 
% (biosig4c++) at http://biosig.sf.net/ 


    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.


 */

/* 
	Internal definitions (external API is defined in biosig.h) 
*/

/****************************************************************************/
/**                                                                        **/
/**                CONSTANTS and Global variables                          **/
/**                                                                        **/
/****************************************************************************/

#ifndef __BIOSIG_INTERNAL_H__
#define __BIOSIG_INTERNAL_H__

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#if defined(__MINGW32__)
#include <sys/param.h>
#endif
#include <time.h>
#include "physicalunits.h"


#ifdef __cplusplus
extern "C" {
#endif

#ifdef NDEBUG
#define VERBOSE_LEVEL 0 	// turn off debugging information, but its only used without NDEBUG
#else
extern int VERBOSE_LEVEL; 	// used for debugging, variable is always defined
#endif


/*
	Including ZLIB enables reading gzipped files (they are decompressed on-the-fly)
	The output files can be zipped, too.
 */

#ifdef HAVE_ZLIB
#include <zlib.h>
#ifndef ZLIB_H
    #if defined(__MINGW64__)
	#include "win64/zlib/zlib.h"
    #elif defined(__MINGW32__)
	#include "win32/zlib/include/zlib.h"
    #endif
#endif
#endif

#ifdef HAVE_CHOLMOD
    #ifdef __APPLE__
        #include <cholmod.h>
    #else
        #include <suitesparse/cholmod.h>
    #endif
#endif

#ifdef HAVE_HDF5
    #include <hdf5.h>
#endif
#ifdef WITH_NIFTI
    #include <nifti1.h>
#endif


#ifdef WITH_GSL
    #include <gsl/gsl_matrix_double.h>
#endif

#ifdef	__WIN32__
#define FILESEP '\\'
char *getlogin (void);
#else
#define FILESEP '/'
#endif


/* test whether HDR.CHANNEL[].{bi,bi8} can be replaced, reduction of header size by about 3%
   currently this is not working, because FAMOS seems to need it.
//#define NO_BI
*/

/* External API definitions - this was part of old biosig.h */
// #include "biosig.h"

/****************************************************************************/
/**                                                                        **/
/**                 DEFINITIONS, TYPEDEFS AND MACROS                       **/
/**                                                                        **/
/****************************************************************************/

#define BIOSIG_VERSION_MAJOR 1
#define BIOSIG_VERSION_MINOR 9
#define BIOSIG_PATCHLEVEL    2
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
    typedef __int16		int16_t;
    typedef unsigned __int8	uint8_t;
    typedef __int8		int8_t;
#else
    #include <inttypes.h>
#endif

#include "gdftime.h"

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
	Z, ZIP, ZIP2, RHD2000,
	invalid=0xffff
};


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
		// workaround for transition to cleaner fieldnames
		float VERSION;		/* GDF version number */
		float Version;		/* GDF version number */
	} ATT_ALI;

	union {
		// workaround for transition to cleaner fieldnames
		enum FileFormat TYPE;		 	/* type of file format */
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

		/* Patient Name:
		 * can consist of up to three components, separated by the unit separator ascii(31), 0x1f, containing in that order
			Last name, first name, second last name (see also SCP-ECG specification EN1064, Section 1, tag 0, 1, and 3)
		 * for privacy protection this field is by default not supported, support can be turned on with FLAG.ANONYMOUS
                 */
		char		Name[MAX_LENGTH_NAME+1];

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




#define GCC_VERSION (__GNUC__ * 10000  + __GNUC_MINOR__ * 100  + __GNUC_PATCHLEVEL__)

#if 0

#elif defined(__linux__) 
#  include <endian.h>
#  include <byteswap.h>

#elif defined(__GLIBC__)	// for Hurd
#  include <endian.h>
#  include <byteswap.h>

#elif defined(__CYGWIN__)
#  include <endian.h>
#  include <byteswap.h>

#elif defined(__WIN32__) || defined(_WIN32)
#  include <stdlib.h>
#  define __BIG_ENDIAN		4321
#  define __LITTLE_ENDIAN	1234
#  define __BYTE_ORDER		__LITTLE_ENDIAN
#  define bswap_16(x) __builtin_bswap16(x)
#  define bswap_32(x) __builtin_bswap32(x)
#  define bswap_64(x) __builtin_bswap64(x)

#	include <winsock2.h>
#	if defined(__MINGW32__)
#	    include <sys/param.h>
#	endif
#	if BYTE_ORDER == LITTLE_ENDIAN
#		define htobe16(x) htons(x)
#		define htole16(x) (x)
#		define be16toh(x) ntohs(x)
#		define le16toh(x) (x)

#		define htobe32(x) htonl(x)
#		define htole32(x) (x)
#		define be32toh(x) ntohl(x)
#		define le32toh(x) (x)

#		define htole64(x) (x)
#		if defined(__MINGW32__)
#       	    define htobe64(x) __builtin_bswap64(x)
#       	    define be64toh(x) __builtin_bswap64(x)
#       	else
#       	    define ntohll(x) (((_int64)(ntohl((int)((x << 32) >> 32))) << 32) | (unsigned int)ntohl(((int)(x >> 32))))
#       	    define htonll(x) ntohll(x)
#       	    define htobe64(x) htonll(x)
#       	    define be64toh(x) ntohll(x)
#       	endif
#		define le64toh(x) (x)

#	elif BYTE_ORDER == BIG_ENDIAN
		/* that would be xbox 360 */
#		define htobe16(x) (x)
#		define htole16(x) __builtin_bswap16(x)
#		define be16toh(x) (x)
#		define le16toh(x) __builtin_bswap16(x)

#		define htobe32(x) (x)
#		define htole32(x) __builtin_bswap32(x)
#		define be32toh(x) (x)
#		define le32toh(x) __builtin_bswap32(x)

#		define htobe64(x) (x)
#		define htole64(x) __builtin_bswap64(x)
#		define be64toh(x) (x)
#		define le64toh(x) __builtin_bswap64(x)

#	else
#		error byte order not supported
#	endif

#elif defined(__NetBSD__)
#  include <sys/bswap.h>
#  define __BIG_ENDIAN _BIG_ENDIAN
#  define __LITTLE_ENDIAN _LITTLE_ENDIAN
#  define __BYTE_ORDER _BYTE_ORDER
#  define bswap_16(x) bswap16(x)
#  define bswap_32(x) bswap32(x)
#  define bswap_64(x) bswap64(x)

#elif defined(__APPLE__)
#	define __BIG_ENDIAN      4321
#	define __LITTLE_ENDIAN  1234
#if (defined(__LITTLE_ENDIAN__) && (__LITTLE_ENDIAN__ == 1))
	#define __BYTE_ORDER __LITTLE_ENDIAN
#else
	#define __BYTE_ORDER __BIG_ENDIAN
#endif

#	include <libkern/OSByteOrder.h>
#	define bswap_16 OSSwapInt16
#	define bswap_32 OSSwapInt32
#	define bswap_64 OSSwapInt64

#	define htobe16(x) OSSwapHostToBigInt16(x)
#	define htole16(x) OSSwapHostToLittleInt16(x)
#	define be16toh(x) OSSwapBigToHostInt16(x)
#	define le16toh(x) OSSwapLittleToHostInt16(x)

#	define htobe32(x) OSSwapHostToBigInt32(x)
#	define htole32(x) OSSwapHostToLittleInt32(x)
#	define be32toh(x) OSSwapBigToHostInt32(x)
#	define le32toh(x) OSSwapLittleToHostInt32(x)

#	define htobe64(x) OSSwapHostToBigInt64(x)
#	define htole64(x) OSSwapHostToLittleInt64(x)
#	define be64toh(x) OSSwapBigToHostInt64(x)
#	define le64toh(x) OSSwapLittleToHostInt64(x)

#elif defined(__OpenBSD__)
#	include <sys/endian.h>
#	define bswap_16 __swap16
#	define bswap_32 __swap32
#	define bswap_64 __swap64

#elif defined(__NetBSD__) || defined(__FreeBSD__) || defined(__DragonFly__)
#	include <sys/endian.h>
#	define be16toh(x) betoh16(x)
#	define le16toh(x) letoh16(x)
#	define be32toh(x) betoh32(x)
#	define le32toh(x) letoh32(x)
#	define be64toh(x) betoh64(x)
#	define le64toh(x) letoh64(x)

#elif (defined(BSD) && (BSD >= 199103)) && !defined(__GLIBC__)
#  include <machine/endian.h>
#  define __BIG_ENDIAN _BIG_ENDIAN
#  define __LITTLE_ENDIAN _LITTLE_ENDIAN
#  define __BYTE_ORDER _BYTE_ORDER
#  define bswap_16(x) __bswap16(x)
#  define bswap_32(x) __bswap32(x)
#  define bswap_64(x) __bswap64(x)

#elif defined(__GNUC__) 
   /* use byteswap macros from the host system, hopefully optimized ones ;-) */
#  include <endian.h>
#  include <byteswap.h>
#  define bswap_16(x) __bswap_16 (x)
#  define bswap_32(x) __bswap_32 (x)
#  define bswap_64(x) __bswap_64 (x)

#elif defined(__sparc__) 
#  define __BIG_ENDIAN  	4321
#  define __LITTLE_ENDIAN  	1234
#  define __BYTE_ORDER 	__BIG_ENDIAN

#else
#  error Unknown platform
#endif 

#if defined(__sparc__)

# ifndef bswap_16
#  define bswap_16(x)   \
	((((x) & 0xff00) >> 8) | (((x) & 0x00ff) << 8))
# endif

# ifndef bswap_32
#  define bswap_32(x)   \
	 ((((x) & 0xff000000) >> 24) \
        | (((x) & 0x00ff0000) >> 8)  \
	| (((x) & 0x0000ff00) << 8)  \
	| (((x) & 0x000000ff) << 24))

# endif

# ifndef bswap_64
#  define bswap_64(x) \
      	 ((((x) & 0xff00000000000000ull) >> 56)	\
      	| (((x) & 0x00ff000000000000ull) >> 40)	\
      	| (((x) & 0x0000ff0000000000ull) >> 24)	\
      	| (((x) & 0x000000ff00000000ull) >> 8)	\
      	| (((x) & 0x00000000ff000000ull) << 8)	\
      	| (((x) & 0x0000000000ff0000ull) << 24)	\
      	| (((x) & 0x000000000000ff00ull) << 40)	\
      	| (((x) & 0x00000000000000ffull) << 56))
# endif

#endif


#if !defined(__BIG_ENDIAN) && !defined(__LITTLE_ENDIAN) 
#error  ENDIANITY is not known 
#endif 

static inline uint16_t leu16p(const void* i) {
	uint16_t a;
	memcpy(&a, i, sizeof(a));
	return (le16toh(a));
}
static inline int16_t lei16p(const void* i) {
	uint16_t a;
	memcpy(&a, i, sizeof(a));
	return ((int16_t)le16toh(a));
}
static inline uint32_t leu32p(const void* i) {
	uint32_t a;
	memcpy(&a, i, sizeof(a));
	return (le32toh(a));
}
static inline int32_t lei32p(const void* i) {
	uint32_t a;
	memcpy(&a, i, sizeof(a));
	return ((int32_t)le32toh(a));
}
static inline uint64_t leu64p(const void* i) {
	uint64_t a;
	memcpy(&a, i, sizeof(a));
	return (le64toh(a));
}
static inline int64_t lei64p(const void* i) {
	uint64_t a;
	memcpy(&a, i, sizeof(a));
	return ((int64_t)le64toh(a));
}

static inline uint16_t beu16p(const void* i) {
	uint16_t a;
	memcpy(&a, i, sizeof(a));
	return ((int16_t)be16toh(a));
}
static inline int16_t bei16p(const void* i) {
	uint16_t a;
	memcpy(&a, i, sizeof(a));
	return ((int16_t)be16toh(a));
}
static inline uint32_t beu32p(const void* i) {
	uint32_t a;
	memcpy(&a, i, sizeof(a));
	return (be32toh(a));
}
static inline int32_t bei32p(const void* i) {
	uint32_t a;
	memcpy(&a, i, sizeof(a));
	return ((int32_t)be32toh(a));
}
static inline uint64_t beu64p(const void* i) {
	uint64_t a;
	memcpy(&a, i, sizeof(a));
	return ((int64_t)be64toh(a));
}
static inline int64_t bei64p(const void* i) {
	uint64_t a;
	memcpy(&a, i, sizeof(a));
	return ((int64_t)be64toh(a));
}

static inline void leu16a(uint16_t i, void* r) {
	i = htole16(i);
	memcpy(r, &i, sizeof(i));
}
static inline void lei16a( int16_t i, void* r) {
	i = htole16(i);
	memcpy(r, &i, sizeof(i));
}
static inline void leu32a(uint32_t i, void* r) {
	i = htole32(i);
	memcpy(r, &i, sizeof(i));
}
static inline void lei32a( int32_t i, void* r) {
	i = htole32(i);
	memcpy(r, &i, sizeof(i));
}
static inline void leu64a(uint64_t i, void* r) {
	i = htole64(i);
	memcpy(r, &i, sizeof(i));
}
static inline void lei64a( int64_t i, void* r) {
	i = htole64(i);
	memcpy(r, &i, sizeof(i));
}

static inline void beu16a(uint16_t i, void* r) {
	i = htobe16(i);
	memcpy(r, &i, sizeof(i));
};
static inline void bei16a( int16_t i, void* r) {
	i = htobe16(i);
	memcpy(r, &i, sizeof(i));
}
static inline void beu32a(uint32_t i, void* r) {
	i = htobe32(i);
	memcpy(r, &i, sizeof(i));
}
static inline void bei32a( int32_t i, void* r) {
	i = htobe32(i);
	memcpy(r, &i, sizeof(i));
}
static inline void beu64a(uint64_t i, void* r) {
	i = htobe64(i);
	memcpy(r, &i, sizeof(i));
}
static inline void bei64a( int64_t i, void* r) {
	i = htobe64(i);
	memcpy(r, &i, sizeof(i));
}

static inline float lef32p(const void* i) {
	// decode little endian float pointer
	uint32_t o;
	union {
		uint32_t i;
		float   r;
	} c;
	memcpy(&o,i,4);
	c.i = le32toh(o);
	return(c.r);
}
static inline double lef64p(const void* i) {
	// decode little endian double pointer
	uint64_t o=0;
	union {
		uint64_t i;
		double   r;
	} c;
	memcpy(&o,i,8);
	c.i = le64toh(o);
	return(c.r);
}
static inline float bef32p(const void* i) {
	// decode little endian float pointer
	uint32_t o;
	union {
		uint32_t i;
		float   r;
	} c;
	memcpy(&o,i,4);
	c.i = be32toh(o);
	return(c.r);
}
static inline double bef64p(const void* i) {
	// decode little endian double pointer
	uint64_t o=0;
	union {
		uint64_t i;
		double   r;
	} c;
	memcpy(&o,i,8);
	c.i = be64toh(o);
	return(c.r);
}

static inline void lef32a( float i, void* r) {
	uint32_t i32;
	memcpy(&i32, &i, sizeof(i));
	i32 = le32toh(i32);
	memcpy(r, &i32, sizeof(i32));
}
static inline void lef64a(  double i, void* r) {
	uint64_t i64;
	memcpy(&i64, &i, sizeof(i));
	i64 = le64toh(i64);
	memcpy(r, &i64, sizeof(i64));
}
static inline void bef32a(   float i, void* r) {
	uint32_t i32;
	memcpy(&i32, &i, sizeof(i));
	i32 = be32toh(i32);
	memcpy(r, &i32, sizeof(i32));
}
static inline void bef64a(  double i, void* r) {
	uint64_t i64;
	memcpy(&i64, &i, sizeof(i));
	i64 = be64toh(i64);
	memcpy(r, &i64, sizeof(i64));
}

#ifndef NAN
# define NAN (0.0/0.0)        /* used for encoding of missing values */
#endif
#ifndef INFINITY
# define INFINITY (1.0/0.0)   /* positive infinity */
#endif
#ifndef isfinite
# define isfinite(a) (-INFINITY < (a) && (a) < INFINITY)
#endif

/*
    The macro IS_SET() can be used to test for defines in 
	if (IS_SET(...)) {
	}
    as well as in 
        #if (IS_SET(...)) 
	#endif
    http://www.thepowerbase.com/2012/04/latest-release-of-linux-contains-code-developed-via-google-plus/
*/
#define macrotest_1 ,
#define IS_SET(macro) is_set_(macro)
#define is_set_(value) is_set__(macrotest_##value)
#define is_set__(comma) is_set___(comma 1, 0)
#define is_set___(_, v, ...) v


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	global constants and variables
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef VERBOSE_LEVEL
extern int   VERBOSE_LEVEL; 	// used for debugging
#endif



/****************************************************************************/
/**                                                                        **/
/**                     TYPEDEFS AND STRUCTURES                            **/
/**                                                                        **/
/****************************************************************************/


/*
	This structure defines the fields used for "VitalFEF"
 */
typedef struct asn1 {
	void *pduType;
	void *SAS;
} ASN1_t;

/*
	This structure defines the fields used for "Annotated ECG"
 */
typedef struct aecg {
	char*		test;		/* test field for annotated ECG */
	
	float		diastolicBloodPressure;		/* in mmHg */
	float		systolicBloodPressure;		/* in mmHg */
	char*		MedicationDrugs;
	char*		ReferringPhysician;
	char*		LatestConfirmingPhysician;
	char*		Diagnosis;
	uint8_t		EmergencyLevel; /* 0: routine 1-10: increased emergency level */

	float		HeartRate;	
	float		P_wave[2]; 	/* start and end  */
	float		QRS_wave[2]; 	/* start and end  */
	float		T_wave[2]; 	/* start and end  */
	float		P_QRS_T_axes[3];

	/***** SCP only fields *****/
	struct {	
		uint8_t	HUFFMAN;
		uint8_t	REF_BEAT;
		uint8_t	DIFF;
		uint8_t	BIMODAL;
	} FLAG;
        struct {
		//uint8_t tag14[41],tag15[41];
	        struct {
			uint16_t INST_NUMBER;		/* tag 14, byte 1-2  */
			uint16_t DEPT_NUMBER;		/* tag 14, byte 3-4  */
			uint16_t DEVICE_ID;		/* tag 14, byte 5-6  */
			uint8_t  DeviceType;		/* tag 14, byte 7: 0: Cart, 1: System (or Host)  */
			uint8_t MANUF_CODE;		/* tag 14, byte 8 (MANUF_CODE has to be 255) */
			char*   MOD_DESC;		/* tag 14, byte 9 (MOD_DESC has to be "Cart1") */
			uint8_t VERSION;		/* tag 14, byte 15 (VERSION has to be 20) */
			uint8_t PROT_COMP_LEVEL;	/* tag 14, byte 16 (PROT_COMP_LEVEL has to be 0xA0 => level II) */
			uint8_t LANG_SUPP_CODE;		/* tag 14, byte 17 (LANG_SUPP_CODE has to be 0x00 => Ascii only, latin and 1-byte code) */
			uint8_t ECG_CAP_DEV;		/* tag 14, byte 18 (ECG_CAP_DEV has to be 0xD0 => Acquire, (No Analysis), Print and Store) */
			uint8_t MAINS_FREQ;		/* tag 14, byte 19 (MAINS_FREQ has to be 0: unspecified, 1: 50 Hz, 2: 60Hz) */
			char 	reserved[22]; 		/* char[35-19] reserved; */			
			char* 	ANAL_PROG_REV_NUM;
			char* 	SERIAL_NUMBER_ACQ_DEV;
			char* 	ACQ_DEV_SYS_SW_ID;
			char* 	ACQ_DEV_SCP_SW; 	/* tag 14, byte 38 (SCP_IMPL_SW has to be "OpenECG XML-SCP 1.00") */
			char* 	ACQ_DEV_MANUF;		/* tag 14, byte 38 (ACQ_DEV_MANUF has to be "Manufacturer") */
        	} Tag14, Tag15; 
        } Section1;
        struct {
        	size_t   StartPtr;
        	size_t	 Length;
        } Section5;
        struct {
        	size_t   StartPtr;
        	size_t	 Length;
        } Section6;
        struct {
        	char	 Confirmed; // 0: original report (not overread); 1:Confirmed report; 2: Overread report (not confirmed)
		struct tm t; 
		uint8_t	 NumberOfStatements;
		char 	 **Statements;
        } Section8;
        struct {
        	char*    StartPtr;
        	size_t	 Length;
        } Section9;
        struct {
        	size_t   StartPtr;
        	size_t	 Length;
        } Section10;
        struct {
        	char	 Confirmed; // 0: original report (not overread); 1:Confirmed report; 2: Overread report (not confirmed)
		struct tm t; 
		uint8_t	 NumberOfStatements;
		char 	 **Statements;
        } Section11;
        struct {
		size_t   StartPtr;
		size_t	 Length;
        } Section12;

} aECG_TYPE;

/****************************************************************************/
/**                                                                        **/
/**                     INTERNAL FUNCTIONS                                 **/
/**                                                                        **/
/****************************************************************************/

/*
        file access wrapper: use ZLIB (if available) or STDIO
 */
HDRTYPE* 	ifopen(HDRTYPE* hdr, const char* mode );
int 		ifclose(HDRTYPE* hdr);
int             ifeof(HDRTYPE* hdr);
int 		ifflush(HDRTYPE* hdr);
size_t 		ifread(void* buf, size_t size, size_t nmemb, HDRTYPE* hdr);
size_t 		ifwrite(void* buf, size_t size, size_t nmemb, HDRTYPE* hdr);
int             ifprintf(HDRTYPE* hdr, const char *format, va_list arg);
int             ifputc(int c, HDRTYPE* hdr);
int 		ifgetc(HDRTYPE* hdr);
char*           ifgets(char *str, int n, HDRTYPE* hdr);
int             ifseek(HDRTYPE* hdr, long offset, int whence );
long            iftell(HDRTYPE* hdr);
int 		ifgetpos(HDRTYPE* hdr, size_t *pos);
int             iferror(HDRTYPE* hdr);


/*
	various utility functions 
*/

uint32_t gcd(uint32_t A, uint32_t B);
uint32_t lcm(uint32_t A, uint32_t B);

extern const uint16_t GDFTYP_BITS[];
extern const char *LEAD_ID_TABLE[];

uint16_t CRCEvaluate(uint8_t* datablock, uint32_t datalength);
int16_t CRCCheck(uint8_t* datablock, uint32_t datalength);

#if (BIOSIG_VERSION < 10700)
// this deprecated since Aug 2013, v1.5.7
#ifndef _WIN32
ATT_DEPREC int strcmpi(const char* str1, const char* str2); // use strcasecmp() instead
#endif
ATT_DEPREC int strncmpi(const char* str1, const char* str2, size_t n); // use strncasecmp() instead
#endif


int month_string2int(const char *s);


int u32cmp(const void *a, const void *b); 

void biosigERROR(HDRTYPE *hdr, enum B4C_ERROR errnum, const char *errmsg);
/*
	sets the local and the (deprecated) global error variables B4C_ERRNUM and B4C_ERRMSG
	the global error variables are kept for backwards compatibility.
*/


/*
	some important functions used internally, 
	the interface for these functios is a bit clumsy and are
	therefore not exported to standard user applications. 
*/

void struct2gdfbin(HDRTYPE *hdr);
int gdfbin2struct(HDRTYPE *hdr);
/* struct2gdfbin and gdfbin2struct
	convert between the streamed header information (as in a GDF file or 
	on a network connection) and the header structure HDRTYPE 
	Specifically, the fixed header, the variable hadder and the optional 
	header information (header 1,2 and 3). This incluedes the 
	description of the user-specified events (TYP=1..255), but not the 
	event table itself. 
 ------------------------------------------------------------------------*/

size_t hdrEVT2rawEVT(HDRTYPE *hdr);
void rawEVT2hdrEVT(HDRTYPE *hdr, size_t length_rawEventTable);
/* rawEVT2hdrEVT and hdrEVT2rawEVT
	convert between streamed event table and the structure
	HDRTYPE.EVENT.
 ------------------------------------------------------------------------*/

int NumberOfChannels(HDRTYPE *hdr); 
/*
        returns the number of channels returned by SREAD. 
        This might be different than the number of data channels in the file
        because of status,event and annotation channels, and because some 
        rereferencing is applied
 ------------------------------------------------------------------------*/


size_t reallocEventTable(HDRTYPE *hdr, size_t EventN);
/*
	allocate, and resize memory of event table
 ------------------------------------------------------------------------*/

void FreeGlobalEventCodeTable();
/*
	free memory allocated for global event code
 ------------------------------------------------------------------------*/

size_t	sread_raw(size_t START, size_t LEN, HDRTYPE* hdr, char flag, void *buf, size_t bufsize);
/* sread_raw: 
	LEN data segments are read from file associated with hdr, starting from 
	segment START.

	If buf==NULL,  a sufficient amount of memory is (re-)allocated in
	hdr->AS.rawdata and the data is copied into  hdr->AS.rawdata, and LEN*hdr->AS.bpb bytes
	are read and stored.

	If buf points to some memory location of size bufsize, the data is stored
	in buf, no reallocation of memory is possible, and only the
	minimum(bufsize, LEN*hdr->AS.bpb) is stored.

	No Overflowdetection or calibration is applied.

	The number of successfully read data blocks is returned, this can be smaller 
	than LEN at the end of the file, of when bufsize is not large enough.

	The data can be "cached", this means
	that more than the requested number of blocks is available in hdr->AS.rawdata. 
	hdr->AS.first and hdr->AS.length contain the number of the first 
	block and the number of blocks, respectively.  
 --------------------------------------------------------------- */

size_t bpb8_collapsed_rawdata(HDRTYPE *hdr);
/* bpb8_collapsed_rawdata
	computes the bits per block when rawdata is collapsed
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

const char* GetFileTypeString(enum FileFormat FMT);
/*	returns a string with file format
 --------------------------------------------------------------- */

enum FileFormat GetFileTypeFromString(const char *);
/*	returns file format from string
 --------------------------------------------------------------- */


#ifdef __cplusplus
}
#endif 

/****************************************************************************/
/**                                                                        **/
/**                               EOF                                      **/
/**                                                                        **/
/****************************************************************************/

#endif	/* BIOSIG_INTERNAL_H */
