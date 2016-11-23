/*

% Copyright (C) 2005-2016 Alois Schloegl <alois.schloegl@gmail.com>
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



/****************************************************************************/
/**                                                                        **/
/**                 DEFINITIONS, TYPEDEFS AND MACROS                       **/
/**                                                                        **/
/****************************************************************************/
#ifndef __BIOSIG_INTERNAL_H__
#define __BIOSIG_INTERNAL_H__

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/param.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif
extern int VERBOSE_LEVEL; 	// used for debugging, variable is always defined


#ifdef NDEBUG
#define VERBOSE_LEVEL 0 	// turn off debugging information, but its only used without NDEBUG
#endif


/*
	Including ZLIB enables reading gzipped files (they are decompressed on-the-fly)
	The output files can be zipped, too.
 */

#ifdef WITH_ZLIB
#include <zlib.h>
#ifndef ZLIB_H
    #if defined(__MINGW64__)
	#include "win64/zlib/zlib.h"
    #elif defined(__MINGW32__)
	#include "win32/zlib/include/zlib.h"
    #endif
#endif
#endif

#ifdef WITH_CHOLMOD
    #ifdef __APPLE__
        #include <cholmod.h>
    #else
        #include <suitesparse/cholmod.h>
    #endif
#endif

#ifdef WITH_HDF5
    #include <hdf5.h>
#endif
#ifdef WITH_NIFTI
    #include <nifti1.h>
#endif


#ifdef WITH_GSL
    #include <gsl/gsl_matrix_double.h>
#endif

#include "biosig.h"

#ifdef	__WIN32__
#define FILESEP '\\'
char *getlogin (void);
#else
#define FILESEP '/'
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

#elif defined(__WIN32__)
#  include <stdlib.h>
#  define __BIG_ENDIAN		4321
#  define __LITTLE_ENDIAN	1234
#  define __BYTE_ORDER		__LITTLE_ENDIAN
#  define bswap_16(x) __builtin_bswap16(x)
#  define bswap_32(x) __builtin_bswap32(x)
#  define bswap_64(x) __builtin_bswap64(x)

#	include <winsock2.h>
#	include <sys/param.h>

#	if BYTE_ORDER == LITTLE_ENDIAN
#		define htobe16(x) htons(x)
#		define htole16(x) (x)
#		define be16toh(x) ntohs(x)
#		define le16toh(x) (x)

#		define htobe32(x) htonl(x)
#		define htole32(x) (x)
#		define be32toh(x) ntohl(x)
#		define le32toh(x) (x)

#		define htobe64(x) __builtin_bswap64(x)
#		define htole64(x) (x)
#		define be64toh(x) __builtin_bswap64(x)
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


#ifdef __cplusplus
}
#endif


#ifndef NAN
# define NAN (0.0/0.0)        /* used for encoding of missing values */
#endif
#ifndef INFINITY
# define INFINITY (1.0/0.0)   /* positive infinity */
#endif
#ifndef isfinite
# define isfinite(a) (-INFINITY < (a) && (a) < INFINITY)
#endif
#ifndef isnan
# define isnan(a) ((a)!=(a))
#endif


#define min(a,b)	(((a) < (b)) ? (a) : (b))
#define max(a,b)	(((a) > (b)) ? (a) : (b))


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

} aECG_TYPE;

/****************************************************************************/
/**                                                                        **/
/**                     INTERNAL FUNCTIONS                                 **/
/**                                                                        **/
/****************************************************************************/

#ifdef __cplusplus
EXTERN_C {
#endif 

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

#ifdef __cplusplus
}
#endif 

/****************************************************************************/
/**                                                                        **/
/**                               EOF                                      **/
/**                                                                        **/
/****************************************************************************/

#endif	/* BIOSIG_INTERNAL_H */
