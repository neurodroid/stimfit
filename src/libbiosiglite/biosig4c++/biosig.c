/*

    Copyright (C) 2005-2021 Alois Schloegl <alois.schloegl@gmail.com>
    Copyright (C) 2011 Stoyan Mihaylov
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

/*

	Library function for reading and writing of varios biosignal data formats.
	It provides also one reference implementation for reading and writing of the
	GDF data format [1].

	Features:
	- reading and writing of EDF, BDF, GDF1, GDF2, CWFB, HL7aECG, SCP files
	- reading of ACQ, AINF, BKR, BrainVision, CNT, DEMG, EGI, ETG4000, MFER files
	The full list of supported formats is shown at
	http://pub.ist.ac.at/~schloegl/biosig/TESTED

	implemented functions:
	- SOPEN, SREAD, SWRITE, SCLOSE, SEOF, SSEEK, STELL, SREWIND


	References:
	[1] GDF - A general data format for biomedical signals.
		available online http://arxiv.org/abs/cs.DB/0608052

*/


/* TODO: ensure that hdr->CHANNEL[.].TOffset gets initialized after every alloc() */

#define _GNU_SOURCE

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <locale.h>
#include <math.h>      // define macro isnan()
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
// Can't include sys/stat.h or sopen is declared twice.
#include <sys/types.h>
struct stat {
  _dev_t st_dev;
  _ino_t st_ino;
  unsigned short st_mode;
  short st_nlink;
  short st_uid;
  short st_gid;
  _dev_t st_rdev;
  _off_t st_size;
  time_t st_atime;
  time_t st_mtime;
  time_t st_ctime;
};
int __cdecl stat(const char *_Filename,struct stat *_Stat);
#else
  #include <sys/stat.h>
#endif


#ifdef WITH_CURL
#  include <curl/curl.h>
#endif

int VERBOSE_LEVEL __attribute__ ((visibility ("default") )) = 0;	// this variable is always available, but only used without NDEBUG

#include "biosig.h"
#include "biosig-network.h"


#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #define FILESEP '\\'
#else
  #include <netdb.h>
  #include <netinet/in.h>   /* sockaddr_in and sockaddr_in6 definitions.      */
  #include <pwd.h>
  #include <unistd.h>
  #define FILESEP '/'
#endif

#define min(a,b)        (((a) < (b)) ? (a) : (b))
#define max(a,b)        (((a) > (b)) ? (a) : (b))

char* getlogin (void);
char* xgethostname (void);

/*-----------------------------------------------------------------------
   error handling should use error variables local to each HDR
   otherwise, sopen() etc. is not re-entrant.

   Therefore, use of variables B4C_ERRNUM and B4C_ERRMSG is deprecated;
   Use instead serror2(hdr), hdr->AS.B4C_ERRNUM, hdr->AS.B4C_ERRMSG.
  ----------------------------------------------------------------------- */
// do not expose deprecated interface in libgdf
#ifndef  ONLYGDF
ATT_DEPREC int B4C_ERRNUM = 0;
ATT_DEPREC const char *B4C_ERRMSG;
#endif


#ifdef HAVE_CHOLMOD
    cholmod_common CHOLMOD_COMMON_VAR;
void CSstop() {
	cholmod_finish(&CHOLMOD_COMMON_VAR);
}
void CSstart () {
	cholmod_start (&CHOLMOD_COMMON_VAR) ; /* start CHOLMOD */
	atexit (&CSstop) ;
}
#endif


#ifndef  ONLYGDF

#ifdef __cplusplus
extern "C" {
#endif

int sopen_SCP_read     (HDRTYPE* hdr);
int sopen_SCP_write    (HDRTYPE* hdr);
int sopen_HL7aECG_read (HDRTYPE* hdr);
void sopen_HL7aECG_write(HDRTYPE* hdr);
void sopen_abf_read    (HDRTYPE* hdr);
void sopen_abf2_read   (HDRTYPE* hdr);
void sopen_axg_read    (HDRTYPE* hdr);
void sopen_alpha_read  (HDRTYPE* hdr);
void sopen_cadwell_read(HDRTYPE* hdr);
void sopen_biosigdump_read (HDRTYPE* hdr);
void sopen_cfs_read    (HDRTYPE* hdr);
void sopen_FAMOS_read  (HDRTYPE* hdr);
void sopen_fiff_read   (HDRTYPE* hdr);
int sclose_HL7aECG_write(HDRTYPE* hdr);
void sopen_ibw_read    (HDRTYPE* hdr);
void sopen_itx_read    (HDRTYPE* hdr);
void sopen_smr_read    (HDRTYPE* hdr);
void sopen_rhd2000_read (HDRTYPE* hdr);
void sopen_rhs2000_read (HDRTYPE* hdr);
void sopen_intan_clp_read (HDRTYPE* hdr);
#ifdef WITH_TDMS
void sopen_tdms_read   (HDRTYPE* hdr);
#endif
int sopen_trc_read   (HDRTYPE* hdr);
int sopen_unipro_read   (HDRTYPE* hdr);
#ifdef WITH_FEF
int sopen_fef_read(HDRTYPE* hdr);
int sclose_fef_read(HDRTYPE* hdr);
#endif
void sopen_heka(HDRTYPE* hdr,FILE *fid);
int sopen_hdf5        (HDRTYPE *hdr);
int sopen_matlab      (HDRTYPE *hdr);
int sopen_sqlite      (HDRTYPE* hdr);
#if defined(WITH_DICOM) || defined(WITH_DCMTK)
int sopen_dicom_read(HDRTYPE* hdr);
#endif

void sopen_atf_read(HDRTYPE* hdr);
void sread_atf(HDRTYPE* hdr);


#ifdef __cplusplus
}
#endif

#endif //ONLYGDF

const uint16_t GDFTYP_BITS[] __attribute__ ((visibility ("default") )) = {
	8, 8, 8,16,16,32,32,64,64,32,64, 0, 0, 0, 0, 0,   /* 0  */
	32,64,128,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   /* 16 */
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   /* 32 */
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   /* 48 */
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   /* 64 */
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	16,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   /* 128: EEG1100 coder,  */
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 8, 0,10, 0,12, 0, 0, 0,16,    /* 256 - 271*/
	0, 0, 0, 0, 0, 0, 0,24, 0, 0, 0, 0, 0, 0, 0,32,    /* 255+24 = bit24, 3 byte */
	0, 0, 0, 0, 0, 0, 0,40, 0, 0, 0, 0, 0, 0, 0,48,
	0, 0, 0, 0, 0, 0, 0,56, 0, 0, 0, 0, 0, 0, 0,64,
	0, 0, 0, 0, 0, 0, 0,72, 0, 0, 0, 0, 0, 0, 0,80,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    /* 384 - 399*/
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 8, 0,10, 0,12, 0, 0, 0,16,    /* 512 - 527*/
	0, 0, 0, 0, 0, 0, 0,24, 0, 0, 0, 0, 0, 0, 0,32,
	0, 0, 0, 0, 0, 0, 0,40, 0, 0, 0, 0, 0, 0, 0,48,
	0, 0, 0, 0, 0, 0, 0,56, 0, 0, 0, 0, 0, 0, 0,64,
	0, 0, 0, 0, 0, 0, 0,72, 0, 0, 0, 0, 0, 0, 0,80 };

const char *gdftyp_string[] = {
	"char","int8","uint8","int16","uint16","int32","uint32","int64","uint64",
	"","","","","","","","float32","float64","float128"
	};

const char *LEAD_ID_TABLE[] = { "unspecified",
	"I","II","V1","V2","V3","V4","V5","V6",
	"V7","V2R","V3R","V4R","V5R","V6R","V7R","X",
	"Y","Z","CC5","CM5","LA","RA","LL","fI",
	"fE","fC","fA","fM","fF","fH","dI",
	"dII","dV1","dV2","dV3","dV4","dV5",
	"dV6","dV7","dV2R","dV3R","dV4R","dV5R",
	"dV6R","dV7R","dX","dY","dZ","dCC5","dCM5",
	"dLA","dRA","dLL","dfI","dfE","dfC","dfA",
	"dfM","dfF","dfH","III","aVR","aVL","aVF",
	"aVRneg","V8","V9","V8R","V9R","D","A","J",
	"Defib","Extern","A1","A2","A3","A4","dV8",
	"dV9","dV8R","dV9R","dD","dA","dJ","Chest",
	"V","VR","VL","VF","MCL","MCL1","MCL2","MCL3",
	"MCL4","MCL5","MCL6","CC","CC1","CC2","CC3",
	"CC4","CC6","CC7","CM","CM1","CM2","CM3","CM4",
	"CM6","dIII","daVR","daVL","daVF","daVRneg","dChest",
	"dV","dVR","dVL","dVF","CM7","CH5","CS5","CB5","CR5",
	"ML","AB1","AB2","AB3","AB4","ES","AS","AI","S",
	"dDefib","dExtern","dA1","dA2","dA3","dA4","dMCL1",
	"dMCL2","dMCL3","dMCL4","dMCL5","dMCL6","RL","CV5RL",
	"CV6LL","CV6LU","V10","dMCL","dCC","dCC1","dCC2",
	"dCC3","dCC4","dCC6","dCC7","dCM","dCM1","dCM2",
	"dCM3","dCM4","dCM6","dCM7","dCH5","dCS5","dCB5",
	"dCR5","dML","dAB1","dAB2","dAB3","dAB4","dES",
	"dAS","dAI","dS","dRL","dCV5RL","dCV6LL","dCV6LU","dV10"
/*   EEG Leads - non consecutive index
	,"NZ","FPZ","AFZ","FZ","FCZ","CZ","CPZ","PZ",
	"POZ","OZ","IZ","FP1","FP2","F1","F2","F3","F4",
	"F5","F6","F7","F8","F9","F10","FC1","FC2","FC3",
	"FC4","FC5","FC6","FT7","FT8","FT9","FT10","C1",
	"C2","C3","C4","C5","C6","CP1","CP2","CP3","CP4",
	"CP5","CP6","P1","P2","P3","P4","P5","P6","P9",
	"P10","O1","O2","AF3","AF4","AF7","AF8","PO3",
	"PO4","PO7","PO8","T3","T7","T4","T8","T5","P7",
	"T6","P8","T9","T10","TP7","TP8","TP9","TP10",
	"A1","A2","T1","T2","PG1","PG2","SP1","SP2",
	"E0","EL1","EL2","EL3","EL4","EL5","EL6","EL7",
	"ER1","ER2","ER3","ER4","ER5","ER6","ER7","ELL",
	"ERL","ELA","ELB","ERA","ERB"
*/
	, "\0\0" };  // stop marker


#ifndef  ONLYGDF
/*
        This information was obtained from here:
        http://www.physionet.org/physiotools/wfdb/lib/ecgcodes.h
*/
const char *MIT_EVENT_DESC[] = {
        "normal beat",
        "left bundle branch block beat",
        "right bundle branch block beat",
        "aberrated atrial premature beat",
        "premature ventricular contraction",
        "fusion of ventricular and normal beat",
        "nodal (junctional) premature beat",
        "atrial premature contraction",
        "premature or ectopic supraventricular beat",
        "ventricular escape beat",
        "nodal (junctional) escape beat",
        "paced beat",
        "unclassifiable beat",
        "signal quality change",
        "condition 15",
        "isolated QRS-like artifact",
        "condition 17",
        "ST change",
        "T-wave change",
        "systole",
        "diastole",
        "comment annotation",
        "measurement annotation",
        "P-wave peak",
        "left or right bundle branch block",
        "non-conducted pacer spike",
        "T-wave peak",
        "rhythm change",
        "U-wave peak",
        "learning",
        "ventricular flutter wave",
        "start of ventricular flutter/fibrillation",
        "end of ventricular flutter/fibrillation",
        "atrial escape beat",
        "supraventricular escape beat",
        "link to external data (aux contains URL)",
        "non-conducted P-wave (blocked APB)",
        "fusion of paced and normal beat",
        "PQ junction (beginning of QRS)",
        "J point (end of QRS)",
        "R-on-T premature ventricular contraction",
        "condition 42",
        "condition 43",
        "condition 44",
        "condition 45",
        "condition 46",
        "condition 47",
        "condition 48",
        "not-QRS (not a getann/putann code)",        // code = 0 is mapped to 49(ACMAX)
        ""};
#endif //ONLYGDF


/* --------------------------------------------------- *
 *	Predefined Event Code Table                    *
 * --------------------------------------------------- */
#if (BIOSIG_VERSION < 10500)
ATT_DEPREC static uint8_t GLOBAL_EVENTCODES_ISLOADED = 0;
ATT_DEPREC struct global_t {
	uint16_t LenCodeDesc;
	uint16_t *CodeIndex;
	const char **CodeDesc;
	char  	 *EventCodesTextBuffer;
} Global;  // deprecated since Oct 2012, v1.4.0
#endif

// event table desription
const struct etd_t ETD [] = {
#include "eventcodes.i"
	{0, 0, ""}
};

// event groups
const struct event_groups_t EventCodeGroups [] = {
#include "eventcodegroups.i"
	{0xffff,  "end-of-table" },
};


/****************************************************************************/
/**                                                                        **/
/**                      INTERNAL FUNCTIONS                                **/
/**                                                                        **/
/****************************************************************************/

// greatest common divisor
uint32_t gcd(uint32_t A, uint32_t B) {
	uint32_t t;
	if (A<B) {t=B; B=A; A=t;};
	while (B>0) {
		t = B;
		B = A%B;
		A = t;
	}
	return(A);
};

// least common multiple - used for obtaining the common HDR.SPR
uint32_t lcm(uint32_t A, uint32_t B) {
	if (A==0 || B==0) {
		fprintf(stderr,"%s (line %d) %s(%d,%d)\n",__FILE__,__LINE__,__func__,A,B);
		return 0;
	}
	// return(A*(B/gcd(A,B)) with overflow detection
	uint64_t A64 = A;
	A64 *= B/gcd(A,B);
	if (A64 > 0x00000000ffffffffllu) {
		fprintf(stderr,"Error: HDR.SPR=LCM(%u,%u) overflows and does not fit into uint32.\n",(unsigned)A,(unsigned)B);
	}
	return((uint32_t)A64);
};



#ifndef  ONLYGDF

void* mfer_swap8b(uint8_t *buf, int8_t len, char FLAG_SWAP)
{
	if (VERBOSE_LEVEL==9)
        	fprintf(stdout,"swap=%i %i %i \nlen=%i %2x%2x%2x%2x%2x%2x%2x%2x\n",
			(int)FLAG_SWAP, __BYTE_ORDER, __LITTLE_ENDIAN, (int)len,
			(unsigned)buf[0],(unsigned)buf[1],(unsigned)buf[2],(unsigned)buf[3],
			(unsigned)buf[4],(unsigned)buf[5],(unsigned)buf[6],(unsigned)buf[7] );

#ifndef S_SPLINT_S
#if __BYTE_ORDER == __BIG_ENDIAN
        if (FLAG_SWAP) {
        	unsigned k;
                for (k=len; k < sizeof(uint64_t); buf[k++]=0);
                *(uint64_t*)buf = bswap_64(*(uint64_t*)buf);
        } else {
                *(uint64_t*)buf >>= (sizeof(uint64_t)-len)*8;
	}
#elif __BYTE_ORDER == __LITTLE_ENDIAN
        if (FLAG_SWAP) {
                *(uint64_t*)buf = bswap_64(*(uint64_t*)buf) >> (sizeof(uint64_t)-len)*8;
        } else {
        	unsigned k;
		for (k=len; k < sizeof(uint64_t); buf[k++]=0) {};
	}

#endif
#endif
	if (VERBOSE_LEVEL==9)
		fprintf(stdout,"%2x%2x%2x%2x%2x%2x%2x%2x %i %f\n",
			buf[0],buf[1],buf[2],buf[3],buf[4],buf[5],
			buf[6],buf[7],(int)*(uint64_t*)buf,*(double*)buf );

	return(buf);
}

/* --------------------------------
 * float to ascii[8] conversion
 * -------------------------------- */
int ftoa8(char* buf, double num)
{
	// used for converting scaling factors Dig/Phys/Min/Max into EDF header
	// Important note: buf may need more than len+1 bytes. make sure there is enough memory allocated.
	double f1,f2;

	if (num==ceil(num))
		sprintf(buf,"%d",(int)num);
	else
		sprintf(buf,"%f",num);

	f1 = atof(buf);
	buf[8] = 0; 	// truncate
	f2 = atof(buf);

	return (fabs(f1-f2) > (fabs(f1)+fabs(f2)) * 1e-6);
}

int is_nihonkohden_signature(char *str)
{
  return (!(
	strncmp(str, "EEG-1200A V01.00", 16) &&
	strncmp(str, "EEG-1100A V01.00", 16) &&
	strncmp(str, "EEG-1100B V01.00", 16) &&
	strncmp(str, "EEG-1100C V01.00", 16) &&
	strncmp(str, "QI-403A   V01.00", 16) &&
  	strncmp(str, "QI-403A   V02.00", 16) &&
  	strncmp(str, "EEG-2100  V01.00", 16) &&
  	strncmp(str, "EEG-2100  V02.00", 16) &&
  	strncmp(str, "DAE-2100D V01.30", 16) &&
  	strncmp(str, "DAE-2100D V02.00", 16)
  ));
}

#endif //ONLYGDF


#if (BIOSIG_VERSION < 10700)
int strncmpi(const char* str1, const char* str2, size_t n)
{
	fprintf(stderr,"Warning from libbiosig: use of function strncmpi() is deprecated - use instead strncasecmp()\n");
	return strncasecmp(str1,str2,n);
}
int strcmpi(const char* str1, const char* str2)
{
	fprintf(stderr,"Warning from libbiosig: use of function strcmpi() is deprecated - use instead strcasecmp()\n");
	return strcasecmp(str1,str2);
}
#endif

/*
	Converts name of month int numeric value.
*/
int month_string2int(const char *s) {
	const char ListOfMonth[12][4] = {"JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"};
	int k;
	for (k = 0; k < 12; k++)
		if (!strncasecmp(s, ListOfMonth[k], 3)) return k;

	return -1;
}


/*
	compare uint32_t
*/
int u32cmp(const void *a,const void *b)
{
	return((int)(*(uint32_t*)a - *(uint32_t*)b));
}


/*
	Interface for mixed use of ZLIB and STDIO
	If ZLIB is not available, STDIO is used.
	If ZLIB is availabe, HDR.FILE.COMPRESSION tells
	whether STDIO or ZLIB is used.
 */

HDRTYPE* ifopen(HDRTYPE* hdr, const char* mode) {
#ifdef ZLIB_H
	if (hdr->FILE.COMPRESSION)
	{
	hdr->FILE.gzFID = gzopen(hdr->FileName, mode);
	hdr->FILE.OPEN = (hdr->FILE.gzFID != NULL);
	} else
#endif
	{
	hdr->FILE.FID = fopen(hdr->FileName, mode);
	hdr->FILE.OPEN = (hdr->FILE.FID != NULL);
	}
	return(hdr);
}

int ifclose(HDRTYPE* hdr) {
	hdr->FILE.OPEN = 0;
#ifdef ZLIB_H
	if (hdr->FILE.COMPRESSION)
		return(gzclose(hdr->FILE.gzFID));
	else
#endif
	return(fclose(hdr->FILE.FID));
}

int ifflush(HDRTYPE* hdr) {
#ifdef ZLIB_H
	if (hdr->FILE.COMPRESSION)
		return(gzflush(hdr->FILE.gzFID,Z_FINISH));
	else
#endif
	return(fflush(hdr->FILE.FID));
}

size_t ifread(void* ptr, size_t size, size_t nmemb, HDRTYPE* hdr) {
#ifdef ZLIB_H
	if (hdr->FILE.COMPRESSION>0)
		return(gzread(hdr->FILE.gzFID, ptr, size * nmemb)/size);
	else
#endif
	return(fread(ptr, size, nmemb, hdr->FILE.FID));
}

size_t ifwrite(void* ptr, size_t size, size_t nmemb, HDRTYPE* hdr) {
#ifdef ZLIB_H
	if (hdr->FILE.COMPRESSION)
	return(gzwrite(hdr->FILE.gzFID, ptr, size*nmemb)/size);
	else
#endif
	return(fwrite(ptr, size, nmemb, hdr->FILE.FID));
}

int ifprintf(HDRTYPE* hdr, const char *format, va_list arg) {
#ifdef ZLIB_H
	if (hdr->FILE.COMPRESSION)
	return(gzprintf(hdr->FILE.gzFID, format, arg));
	else
#endif
	return(fprintf(hdr->FILE.FID, format, arg));
}

int ifputc(int c, HDRTYPE* hdr) {
#ifdef ZLIB_H
	if (hdr->FILE.COMPRESSION)
	return(gzputc(hdr->FILE.gzFID, c));
	else
#endif
	return(fputc(c,hdr->FILE.FID));
}

int ifgetc(HDRTYPE* hdr) {
#ifdef ZLIB_H
	if (hdr->FILE.COMPRESSION)
	return(gzgetc(hdr->FILE.gzFID));
	else
#endif
	return(fgetc(hdr->FILE.FID));
}

char* ifgets(char *str, int n, HDRTYPE* hdr) {
#ifdef ZLIB_H
	if (hdr->FILE.COMPRESSION)
	return(gzgets(hdr->FILE.gzFID, str, n));
	else
#endif
	return(fgets(str,n,hdr->FILE.FID));
}

int ifseek(HDRTYPE* hdr, ssize_t offset, int whence) {
#ifdef ZLIB_H
	if (hdr->FILE.COMPRESSION) {
	if (whence==SEEK_END)
		fprintf(stdout,"Warning SEEK_END is not supported but used in gzseek/ifseek.\nThis can cause undefined behaviour.\n");
	return(gzseek(hdr->FILE.gzFID,offset,whence));
	} else
#endif
#if defined(__MINGW64__)
	return(_fseeki64(hdr->FILE.FID,offset,whence));
#else
	return(fseek(hdr->FILE.FID,offset,whence));
#endif
}

ssize_t iftell(HDRTYPE* hdr) {
#ifdef ZLIB_H
	if (hdr->FILE.COMPRESSION)
	return(gztell(hdr->FILE.gzFID));
	else
#endif
	return(ftell(hdr->FILE.FID));
}

int ifsetpos(HDRTYPE* hdr, size_t *pos) {
#if __gnu_linux__
	// gnu linux on sparc needs this
	fpos_t p;
	p.__pos = *pos;
#elif __sparc__ || __APPLE__ || __MINGW32__ || ANDROID || __NetBSD__ || __CYGWIN__ || __FreeBSD__
	fpos_t p = *pos;
#else
	fpos_t p;
	p.__pos = *pos;
#endif

#ifdef ZLIB_H
	if (hdr->FILE.COMPRESSION) {
		gzseek(hdr->FILE.gzFID,*pos,SEEK_SET);
		size_t pos1 = *pos;
		*pos = gztell(hdr->FILE.gzFID);
		return(*pos - pos1);
	}
	else
#endif
	{
	int c= fsetpos(hdr->FILE.FID,&p);
#if __gnu_linux__
	// gnu linux on sparc needs this
	*pos = p.__pos;
#elif __sparc__ || __APPLE__ || __MINGW32__ || ANDROID || __NetBSD__ || __CYGWIN__ || __FreeBSD__
	*pos = p;
#else
	*pos = p.__pos;
#endif
	return(c);
	}
}

int ifgetpos(HDRTYPE* hdr, size_t *pos) {
#ifdef ZLIB_H
	if (hdr->FILE.COMPRESSION) {
		z_off_t p = gztell(hdr->FILE.gzFID);
		if (p<0) return(-1);
		else {
			*pos = p;
			return(0);
		}
	} else
#endif
	{
		fpos_t p;
		int c = fgetpos(hdr->FILE.FID, &p);
#if __gnu_linux__
		// gnu linux on sparc needs this
		*pos = p.__pos;	// ugly hack but working
#elif __sparc__ || __APPLE__ || __MINGW32__ || ANDROID || __NetBSD__ || __CYGWIN__ || __FreeBSD__
		*pos = p;
#else
		*pos = p.__pos;	// ugly hack but working
#endif
		return(c);
	}
}

int ifeof(HDRTYPE* hdr) {
#ifdef ZLIB_H
	if (hdr->FILE.COMPRESSION)
	return(gzeof(hdr->FILE.gzFID));
	else
#endif
	return(feof(hdr->FILE.FID));
}

int iferror(HDRTYPE* hdr) {
#ifdef ZLIB_H
	if (hdr->FILE.COMPRESSION) {
		int errnum;
		const char *tmp = gzerror(hdr->FILE.gzFID,&errnum);
		fprintf(stderr,"GZERROR: %i %s \n",errnum, tmp);
		return(errnum);
	}
	else
#endif
	return(ferror(hdr->FILE.FID));
}


/*------------------------------------------------------------------------
	sort event table according to EVENT.POS
  ------------------------------------------------------------------------*/
struct event {
	uint32_t POS;
	uint32_t DUR;
	uint16_t TYP;
	uint16_t CHN;
#if (BIOSIG_VERSION >= 10500)
	gdf_time TimeStamp;
#endif
};

int compare_eventpos(const void *e1, const void *e2) {
	return(((struct event*)(e1))->POS - ((struct event*)(e2))->POS);
}

void sort_eventtable(HDRTYPE *hdr) {
	size_t k;
	struct event *entry = (struct event*) calloc(hdr->EVENT.N, sizeof(struct event));
	if ((hdr->EVENT.DUR != NULL) && (hdr->EVENT.CHN != NULL))
	for (k=0; k < hdr->EVENT.N; k++) {
		entry[k].TYP = hdr->EVENT.TYP[k];
		entry[k].POS = hdr->EVENT.POS[k];
		entry[k].CHN = hdr->EVENT.CHN[k];
		entry[k].DUR = hdr->EVENT.DUR[k];
	}
	else
	for (k=0; k < hdr->EVENT.N; k++) {
		entry[k].TYP = hdr->EVENT.TYP[k];
		entry[k].POS = hdr->EVENT.POS[k];
	}
#if (BIOSIG_VERSION >= 10500)
	if (hdr->EVENT.TimeStamp != NULL)
	for (k=0; k < hdr->EVENT.N; k++) {
		entry[k].TimeStamp = hdr->EVENT.TimeStamp[k];
	}
#endif

	qsort(entry, hdr->EVENT.N, sizeof(struct event), &compare_eventpos);

	if ((hdr->EVENT.DUR != NULL) && (hdr->EVENT.CHN != NULL))
	for (k=0; k < hdr->EVENT.N; k++) {
		hdr->EVENT.TYP[k] = entry[k].TYP;
		hdr->EVENT.POS[k] = entry[k].POS;
		hdr->EVENT.CHN[k] = entry[k].CHN;
		hdr->EVENT.DUR[k] = entry[k].DUR;
	}
	else
	for (k=0; k < hdr->EVENT.N; k++) {
		hdr->EVENT.TYP[k] = entry[k].TYP;
		hdr->EVENT.POS[k] = entry[k].POS;
	}

#if (BIOSIG_VERSION >= 10500)
	if (hdr->EVENT.TimeStamp != NULL)
	for (k=0; k < hdr->EVENT.N; k++) {
		hdr->EVENT.TimeStamp[k] = entry[k].TimeStamp;
	}
#endif

	free(entry);
}

/*------------------------------------------------------------------------
	re-allocates memory for Eventtable.
	hdr->EVENT.N contains actual number of events
	EventN determines the size of the allocated memory

  return value:
	in case of success, EVENT_N is returned
	in case of failure SIZE_MAX is returned;
  ------------------------------------------------------------------------*/
size_t reallocEventTable(HDRTYPE *hdr, size_t EventN)
{
	size_t n;
	hdr->EVENT.POS = (uint32_t*)realloc(hdr->EVENT.POS, EventN * sizeof(*hdr->EVENT.POS));
	hdr->EVENT.DUR = (uint32_t*)realloc(hdr->EVENT.DUR, EventN * sizeof(*hdr->EVENT.DUR));
	hdr->EVENT.TYP = (uint16_t*)realloc(hdr->EVENT.TYP, EventN * sizeof(*hdr->EVENT.TYP));
	hdr->EVENT.CHN = (uint16_t*)realloc(hdr->EVENT.CHN, EventN * sizeof(*hdr->EVENT.CHN));
#if (BIOSIG_VERSION >= 10500)
	hdr->EVENT.TimeStamp = (gdf_time*)realloc(hdr->EVENT.TimeStamp, EventN * sizeof(gdf_time));
#endif

	if (hdr->EVENT.POS==NULL) return SIZE_MAX;
	if (hdr->EVENT.TYP==NULL) return SIZE_MAX;
	if (hdr->EVENT.CHN==NULL) return SIZE_MAX;
	if (hdr->EVENT.DUR==NULL) return SIZE_MAX;
	if (hdr->EVENT.TimeStamp==NULL) return SIZE_MAX;

	for (n = hdr->EVENT.N; n< EventN; n++) {
		hdr->EVENT.TYP[n] = 0;
		hdr->EVENT.CHN[n] = 0;
		hdr->EVENT.DUR[n] = 0;
#if (BIOSIG_VERSION >= 10500)
		hdr->EVENT.TimeStamp[n] = 0;
#endif
	}
	return EventN;
}



/*------------------------------------------------------------------------
	converts event table from {TYP,POS} to [TYP,POS,CHN,DUR} format
  ------------------------------------------------------------------------*/
void convert2to4_eventtable(HDRTYPE *hdr) {
	size_t k1,k2,N=hdr->EVENT.N;

	sort_eventtable(hdr);

	if (hdr->EVENT.DUR == NULL)
		hdr->EVENT.DUR = (typeof(hdr->EVENT.DUR)) calloc(N,sizeof(*hdr->EVENT.DUR));
	if (hdr->EVENT.CHN == NULL)
		hdr->EVENT.CHN = (typeof(hdr->EVENT.CHN)) calloc(N,sizeof(*hdr->EVENT.CHN));

	for (k1=0; k1<N; k1++) {
		typeof(*hdr->EVENT.TYP) typ =  hdr->EVENT.TYP[k1];
		if ((typ < 0x8000) && (typ>0)  && !hdr->EVENT.DUR[k1])
		for (k2 = k1+1; k2<N; k2++) {
			if ((typ|0x8000) == hdr->EVENT.TYP[k2]) {
				hdr->EVENT.DUR[k1] = hdr->EVENT.POS[k2] - hdr->EVENT.POS[k1];
				hdr->EVENT.TYP[k2] = 0;
				break;
			}
		}
	}
	for (k1=0,k2=0; k1<N; k1++) {
		if (k2!=k1) {
			hdr->EVENT.TYP[k2]=hdr->EVENT.TYP[k1];
			hdr->EVENT.POS[k2]=hdr->EVENT.POS[k1];
			hdr->EVENT.DUR[k2]=hdr->EVENT.DUR[k1];
			hdr->EVENT.CHN[k2]=hdr->EVENT.CHN[k1];
#if (BIOSIG_VERSION >= 10500)
			if (hdr->EVENT.TimeStamp != NULL)
				hdr->EVENT.TimeStamp[k2] = hdr->EVENT.TimeStamp[k1];
#endif
		}
		if (hdr->EVENT.TYP[k1]) k2++;
	}
	hdr->EVENT.N = k2;
}
/*------------------------------------------------------------------------
	converts event table from [TYP,POS,CHN,DUR} to {TYP,POS} format
  ------------------------------------------------------------------------*/
void convert4to2_eventtable(HDRTYPE *hdr) {
	size_t k1,k2,N = hdr->EVENT.N;
	if ((hdr->EVENT.DUR == NULL) || (hdr->EVENT.CHN == NULL)) return;

	for (k1=0; k1<N; k1++)
		if (hdr->EVENT.CHN[k1]) return;

	hdr->EVENT.TYP = (typeof(hdr->EVENT.TYP)) realloc(hdr->EVENT.TYP,2*N*sizeof(*hdr->EVENT.TYP));
	hdr->EVENT.POS = (typeof(hdr->EVENT.POS)) realloc(hdr->EVENT.POS,2*N*sizeof(*hdr->EVENT.POS));
#if (BIOSIG_VERSION >= 10500)
	hdr->EVENT.TimeStamp = (gdf_time*) realloc(hdr->EVENT.TimeStamp,2*N*sizeof(gdf_time));
#endif

	for (k1=0,k2=N; k1<N; k1++)
		if (hdr->EVENT.DUR[k1]) {
			hdr->EVENT.TYP[k2] = hdr->EVENT.TYP[k1] | 0x8000;
			hdr->EVENT.POS[k2] = hdr->EVENT.POS[k1] + hdr->EVENT.DUR[k1];
#if (BIOSIG_VERSION >= 10500)
			hdr->EVENT.TimeStamp[k2] = hdr->EVENT.TimeStamp[k1] + lround(ldexp(hdr->EVENT.DUR[k1]/(hdr->EVENT.SampleRate*24*3600),32));
#endif
			k2++;
		}
	hdr->EVENT.N = k2;

	free(hdr->EVENT.CHN); hdr->EVENT.CHN=NULL;
	free(hdr->EVENT.DUR); hdr->EVENT.DUR=NULL;
	sort_eventtable(hdr);
}


/*------------------------------------------------------------------------
	write GDF event table
	utility function for SCLOSE and SFLUSH_GDF_EVENT_TABLE

	TODO: writing of TimeStamps
  ------------------------------------------------------------------------*/
void write_gdf_eventtable(HDRTYPE *hdr)
{
	uint32_t	k32u;
	uint8_t 	buf[88];
	char flag;


fprintf(stdout,"write_gdf_eventtable is obsolete - use hdrEVT2rawEVT instead;\n");

	ifseek(hdr, hdr->HeadLen + hdr->AS.bpb*hdr->NRec, SEEK_SET);
	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"WriteEventTable: %p %p %p %p\t",hdr->EVENT.TYP,hdr->EVENT.POS,hdr->EVENT.DUR,hdr->EVENT.CHN);
	flag = (hdr->EVENT.DUR != NULL) && (hdr->EVENT.CHN != NULL);
	if (flag)   // any DUR or CHN is larger than 0
		for (k32u=0, flag=0; (k32u<hdr->EVENT.N) && !flag; k32u++)
			flag |= hdr->EVENT.CHN[k32u] || hdr->EVENT.DUR[k32u];

	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"flag=%d.\n",flag);

	buf[0] = (flag ? 3 : 1);
	if (hdr->VERSION < 1.94) {
		k32u   = lround(hdr->EVENT.SampleRate);
		buf[1] =  k32u      & 0x000000FF;
		buf[2] = (k32u>>8 ) & 0x000000FF;
		buf[3] = (k32u>>16) & 0x000000FF;
		leu32a(hdr->EVENT.N, buf+4);
	}
	else {
		k32u   = hdr->EVENT.N;
		buf[1] =  k32u      & 0x000000FF;
		buf[2] = (k32u>>8 ) & 0x000000FF;
		buf[3] = (k32u>>16) & 0x000000FF;
		lef32a(hdr->EVENT.SampleRate, buf+4);
	};
	for (k32u=0; k32u<hdr->EVENT.N; k32u++) {
		hdr->EVENT.POS[k32u] = htole32(hdr->EVENT.POS[k32u]);
		hdr->EVENT.TYP[k32u] = htole16(hdr->EVENT.TYP[k32u]);
	}
	ifwrite(buf, 8, 1, hdr);
	ifwrite(hdr->EVENT.POS, sizeof(*hdr->EVENT.POS), hdr->EVENT.N, hdr);
	ifwrite(hdr->EVENT.TYP, sizeof(*hdr->EVENT.TYP), hdr->EVENT.N, hdr);
	if (flag) {
		for (k32u=0; k32u<hdr->EVENT.N; k32u++) {
			hdr->EVENT.DUR[k32u] = le32toh(hdr->EVENT.DUR[k32u]);
			hdr->EVENT.CHN[k32u] = le16toh(hdr->EVENT.CHN[k32u]);
		}
		ifwrite(hdr->EVENT.CHN, sizeof(*hdr->EVENT.CHN), hdr->EVENT.N,hdr);
		ifwrite(hdr->EVENT.DUR, sizeof(*hdr->EVENT.DUR), hdr->EVENT.N,hdr);
	}
}


#if (BIOSIG_VERSION < 10500)
/* Stubs for deprecated functions */
ATT_DEPREC void FreeGlobalEventCodeTable() {} // deprecated since Oct 2012, v1.4.0
ATT_DEPREC void LoadGlobalEventCodeTable() {} // deprecated since Oct 2012, v1.4.0
#endif

/*------------------------------------------------------------------------
	adds free text annotation to event table
	the EVENT.TYP is identified from the table EVENT.CodeDesc
	if annotations is not listed in CodeDesc, it is added to CodeDesc
	The table is limited to 256 entries, because the table EventCodes
	allows only codes 0-255 as user specific entry.
  ------------------------------------------------------------------------*/
void FreeTextEvent(HDRTYPE* hdr,size_t N_EVENT, const char* annotation) {
	/* free text annotations encoded as user specific events (codes 1-255) */

/* !!!
	annotation is not copied, but it is assumed that annotation string is also available after return
	usually, the string is available in hdr->AS.Header; still this can disappear (free, or rellocated)
	before the Event table is destroyed.
   !!! */

	size_t k;
	int flag;
//	static int LengthCodeDesc = 0;
	if (hdr->EVENT.CodeDesc == NULL) {
		hdr->EVENT.CodeDesc = (typeof(hdr->EVENT.CodeDesc)) realloc(hdr->EVENT.CodeDesc,257*sizeof(*hdr->EVENT.CodeDesc));
		hdr->EVENT.CodeDesc[0] = "";	// typ==0, is always empty
		hdr->EVENT.LenCodeDesc = 1;
	}

	if (annotation == NULL) {
		hdr->EVENT.TYP[N_EVENT] = 0;
		return;
	}

	// First, compare text with any predefined event description
	for (k=0; ETD[k].typ != 0; k++) {
		if (!strcmp(ETD[k].desc, annotation)) {
			// annotation is already a predefined event
			hdr->EVENT.TYP[N_EVENT] = ETD[k].typ;
			return;
		}
	}

	// Second, compare text with user-defined event description
	flag=1;
	for (k=0; (k < hdr->EVENT.LenCodeDesc) && flag; k++) {
		if (!strncmp(hdr->EVENT.CodeDesc[k], annotation, strlen(annotation))) {
			hdr->EVENT.TYP[N_EVENT] = k;
			flag = 0;
		}
	}

	// Third, add event description if needed
	if (flag && (hdr->EVENT.LenCodeDesc < 256)) {
		hdr->EVENT.TYP[N_EVENT] = hdr->EVENT.LenCodeDesc;
		hdr->EVENT.CodeDesc[hdr->EVENT.LenCodeDesc] = annotation;
		hdr->EVENT.LenCodeDesc++;
	}

	if (hdr->EVENT.LenCodeDesc > 255) {
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Maximum number of user-defined events (256) exceeded");
	}
}

/*------------------------------------------------------------------------
      returns clear text description of n-th event
  -------------------------------------------------------------- */
const char* GetEventDescription(HDRTYPE *hdr, size_t N) {
        if (hdr==NULL || N >= hdr->EVENT.N) return NULL;
        uint16_t TYP = hdr->EVENT.TYP[N];

        if (TYP < hdr->EVENT.LenCodeDesc) // user-specified events, TYP < 256
                return hdr->EVENT.CodeDesc[TYP];

	if (TYP < 256) // not defined by user
		return NULL;

        // end of event: encoded by 0x8000+TYP
	if ((hdr->EVENT.TYP[N] & 0x8000) && (hdr->TYPE==GDF))
			return (NULL);

	if ((hdr->EVENT.TYP[N] == 0x7fff) && (hdr->TYPE==GDF))
			return "[neds]";

        // event definition according to GDF's eventcodes.txt table
        uint16_t k;
        for (k=0; ETD[k].typ != 0; k++)
                if (ETD[k].typ==TYP)
                        return ETD[k].desc;

        fprintf(stderr,"Warning: invalid event type 0x%04x\n",TYP);
        return (NULL);
}

/*------------------------------------------------------------------------
	DUR2VAL converts sparse sample values in the event table
	from the DUR format (uint32, machine endian) to the sample value.
	Endianity of the platform is considered.
  ------------------------------------------------------------------------*/
double dur2val(uint32_t DUR, uint16_t gdftyp) {

	if (gdftyp==5)
		return (double)(int32_t)DUR;
	if (gdftyp==6)
		return (double)(uint32_t)DUR;
	if (gdftyp==16) {
		float fDur;
		memcpy(&fDur,&DUR,4);
		return (double)fDur;
	}
	union {
		uint32_t t32;
		uint16_t t16[2];
		uint8_t  t8[4];
	} u;
	/*
		make sure u32 is always little endian like in the GDF file
		and only then extract the sample value
	*/
	u.t32 = htole32(DUR);

	if (gdftyp==1)
		return (double)(int8_t)(u.t8[0]);
	if (gdftyp==2)
		return (double)(uint8_t)(u.t8[0]);
	if (gdftyp==3)
		return (double)(int16_t)le16toh(u.t16[0]);
	if (gdftyp==4)
		return (double)(uint16_t)le16toh(u.t16[0]);

	return NAN;
}


/*------------------------------------------------------------------------
	getTimeChannelNumber
	searches all channels, whether one channel contains the time axis

	Return value:
	the number of the first channel containing the time axis is returned.
	if no time channel is found, 0 is returned;

	Note: a 1-based indexing is used, the corresponding time channel is used
	the header of the time channel is in hdr->CHANNEL[getTimeChannelNumber(hdr)-1]
  ------------------------------------------------------------------------*/
int getTimeChannelNumber(HDRTYPE* hdr) {
	typeof(hdr->NS) k;

	for (k=0; k<hdr->NS; k++)
		if (hdr->CHANNEL[k].OnOff==2)
			return (k+1);

	return 0;
}


/*------------------------------------------------------------------------
	biosig_set_hdr_ipaddr
	set the field HDR.IPaddr based on the IP address of hostname

	Return value:
	 0: hdr->IPaddr is set
	otherwise hdr->IPaddr is not set
  ------------------------------------------------------------------------*/
int biosig_set_hdr_ipaddr(HDRTYPE *hdr, const char *hostname) {

	struct addrinfo hints;
	struct addrinfo *result, *rp;

	memset(&hints, 0, sizeof(struct addrinfo));
	hints.ai_family   = AF_UNSPEC;    /* Allow IPv4 or IPv6 */
	hints.ai_socktype = 0;
	hints.ai_flags    = 0;
	hints.ai_protocol = 0;          /* Any protocol */

	int s = getaddrinfo(hostname, NULL, &hints, &result);

	if (s != 0) return -1;	// IPaddr can not be set

	for (rp = result; rp != NULL; rp = rp->ai_next) {
		if ( rp->ai_family == AF_INET6)
			memcpy(hdr->IPaddr, &(((struct sockaddr_in6 *)rp->ai_addr)->sin6_addr), 16);

		else if ( rp->ai_family == AF_INET) {
			memcpy(hdr->IPaddr, &(((struct sockaddr_in *)rp->ai_addr)->sin_addr.s_addr), 4);
			memset(hdr->IPaddr+4, 0, 12);
		}
		break; // set first found address
	}

	freeaddrinfo(result);
	return 0;
}


/****************************************************************************/
/**                                                                        **/
/**                     EXPORTED FUNCTIONS                                 **/
/**                                                                        **/
/****************************************************************************/

uint32_t get_biosig_version (void) {
	return ((BIOSIG_VERSION_MAJOR<<16) + (BIOSIG_VERSION_MINOR<<8) + BIOSIG_PATCHLEVEL);
}


/****************************************************************************/
/**                     INIT HDR                                           **/
/****************************************************************************/
#define Header1 ((char*)hdr->AS.Header)

HDRTYPE* constructHDR(const unsigned NS, const unsigned N_EVENT)
{
/*
	HDR is initialized, memory is allocated for
	NS channels and N_EVENT number of events.

	The purpose is to define all parameters at an initial step.
	No parameters must remain undefined.
 */
	HDRTYPE* hdr = (HDRTYPE*)malloc(sizeof(HDRTYPE));

	union {
		uint32_t testword;
		uint8_t  testbyte[sizeof(uint32_t)];
	} EndianTest;
    	int k,k1;
	uint8_t	LittleEndian;
	size_t BitsPerBlock;

	EndianTest.testword = 0x4a3b2c1d;
	LittleEndian = (EndianTest.testbyte[0]==0x1d && EndianTest.testbyte[1]==0x2c  && EndianTest.testbyte[2]==0x3b  && EndianTest.testbyte[3]==0x4a );

	assert (  ( LittleEndian && __BYTE_ORDER == __LITTLE_ENDIAN)
	       || (!LittleEndian && __BYTE_ORDER == __BIG_ENDIAN   ) );

	hdr->FileName = NULL;
	hdr->FILE.OPEN = 0;
	hdr->FILE.FID = 0;
	hdr->FILE.POS = 0;
	hdr->FILE.Des = 0;
	hdr->FILE.COMPRESSION = 0;
	hdr->FILE.size = 0;
#ifdef ZLIB_H
	hdr->FILE.gzFID = 0;
#endif

	hdr->AS.B4C_ERRNUM = B4C_NO_ERROR;
	hdr->AS.B4C_ERRMSG = NULL;
	hdr->AS.Header = NULL;
	hdr->AS.rawEventData = NULL;
	hdr->AS.auxBUF = NULL;
	hdr->AS.bpb    = 0;

	hdr->TYPE = noFile;
	hdr->VERSION = 2.0;
	hdr->AS.rawdata = NULL; 		//(uint8_t*) malloc(0);
	hdr->AS.flag_collapsed_rawdata = 0;	// is rawdata not collapsed
	hdr->AS.first = 0;
	hdr->AS.length  = 0;  			// no data loaded
	memset(hdr->AS.SegSel,0,sizeof(hdr->AS.SegSel));
	hdr->Calib = NULL;
	hdr->rerefCHANNEL = NULL;

	hdr->NRec = 0;
	hdr->SPR  = 0;
	hdr->NS = NS;
	hdr->SampleRate = 4321.5;
	hdr->Patient.Id[0]=0;
	strcpy(hdr->ID.Recording,"00000000");
	hdr->data.size[0] = 0; 	// rows
	hdr->data.size[1] = 0;  // columns
	hdr->data.block = NULL;
#if __FreeBSD__ || __APPLE__ || __NetBSD__
	time_t t=time(NULL);
	struct tm *tt = localtime(&t);
	hdr->tzmin    = tt->tm_gmtoff/60;
	hdr->T0       = t_time2gdf_time(time(NULL)-tt->tm_gmtoff); // localtime
#else
	hdr->T0    = t_time2gdf_time(time(NULL)-timezone); // localtime
	hdr->tzmin = -timezone/60;      // convert from seconds west of UTC to minutes east;
#endif
	{
	uint8_t Equipment[8] = "b4c_1.5 ";
	Equipment[4] = BIOSIG_VERSION_MAJOR+'0';
	Equipment[6] = BIOSIG_VERSION_MINOR+'0';
	memcpy(&(hdr->ID.Equipment), &Equipment, 8);
	}

	hdr->ID.Manufacturer._field[0]    = 0;
	hdr->ID.Manufacturer.Name         = NULL;
	hdr->ID.Manufacturer.Model        = NULL;
	hdr->ID.Manufacturer.Version      = NULL;
	hdr->ID.Manufacturer.SerialNumber = NULL;
	hdr->ID.Technician 	= NULL;
	hdr->ID.Hospital 	= NULL;

	memset(hdr->IPaddr, 0, 16);
	{	// some local variables are used only in this block
#ifdef _WIN32
   #if 1
	// getlogin() a flawfinder level [4] issue, recommended to use getpwuid(geteuid()) but not available on Windows
	hdr->ID.Technician = strdup(getlogin());
   #else	// this compiles but stops with "Program error" on wine
	char str[1001];
	GetUserName(str,1000);
	if (VERBOSE_LEVEL>7)  fprintf(stdout,"Name:%s\n",str);
	hdr->ID.Technician = strdup(str);
   #endif
#else
	char *username = NULL;
	struct passwd *p = getpwuid(geteuid());
	if (p != NULL)
		username = p->pw_name;
	if (username)
		hdr->ID.Technician = strdup(username);

#endif
	}


#ifndef WITHOUT_NETWORK
#ifdef _WIN32
	WSADATA wsadata;
	WSAStartup(MAKEWORD(1,1), &wsadata);
#endif
	{
      	// set default IP address to local IP address
	char *localhostname;
	localhostname = xgethostname();
	if (localhostname) {
		biosig_set_hdr_ipaddr(hdr, localhostname);
		free (localhostname);
	}
	}
#ifdef _WIN32
	WSACleanup();
#endif
#endif // not WITHOUT_NETWORK

	hdr->Patient.Name[0] 	= 0;
	//hdr->Patient.Id[0] 	= 0;
	hdr->Patient.Birthday 	= (gdf_time)0;        // Unknown;
      	hdr->Patient.Medication = 0;	// 0:Unknown, 1: NO, 2: YES
      	hdr->Patient.DrugAbuse 	= 0;	// 0:Unknown, 1: NO, 2: YES
      	hdr->Patient.AlcoholAbuse=0;	// 0:Unknown, 1: NO, 2: YES
      	hdr->Patient.Smoking 	= 0;	// 0:Unknown, 1: NO, 2: YES
      	hdr->Patient.Sex 	= 0;	// 0:Unknown, 1: Male, 2: Female
      	hdr->Patient.Handedness = 0;	// 0:Unknown, 1: Right, 2: Left, 3: Equal
      	hdr->Patient.Impairment.Visual = 0;	// 0:Unknown, 1: NO, 2: YES, 3: Corrected
      	hdr->Patient.Impairment.Heart  = 0;	// 0:Unknown, 1: NO, 2: YES, 3: Pacemaker
      	hdr->Patient.Weight 	= 0;	// 0:Unknown
      	hdr->Patient.Height 	= 0;	// 0:Unknown

      	for (k1=0; k1<3; k1++) {
      		hdr->Patient.Headsize[k1] = 0;        // Unknown;
      		hdr->ELEC.REF[k1] = 0.0;
      		hdr->ELEC.GND[k1] = 0.0;
      	}
	hdr->LOC[0] = 0x00292929;
	hdr->LOC[1] = 48*3600000+(1<<31); 	// latitude
	hdr->LOC[2] = 15*3600000+(1<<31); 	// longitude
	hdr->LOC[3] = 35000; 	 	//altitude in centimeter above sea level

	hdr->FLAG.UCAL = 0; 		// un-calibration OFF (auto-scaling ON)
	hdr->FLAG.OVERFLOWDETECTION = 1; 	// overflow detection ON
	hdr->FLAG.ANONYMOUS = 1; 	// <>0: no personal names are processed
	hdr->FLAG.TARGETSEGMENT = 1;	// read 1st segment
	hdr->FLAG.ROW_BASED_CHANNELS=0;

       	// define variable header
	hdr->CHANNEL = (CHANNEL_TYPE*)calloc(hdr->NS, sizeof(CHANNEL_TYPE));
	BitsPerBlock = 0;
	for (k=0;k<hdr->NS;k++)	{
		size_t nbits;
		CHANNEL_TYPE *hc = hdr->CHANNEL+k;
	      	hc->Label[0]  = 0;
	      	hc->LeadIdCode= 0;
	      	strcpy(hc->Transducer, "EEG: Ag-AgCl electrodes");
	      	hc->PhysDimCode = 19+4256; // uV
	      	hc->PhysMax   = +100;
	      	hc->PhysMin   = -100;
	      	hc->DigMax    = +2047;
	      	hc->DigMin    = -2048;
	      	hc->Cal	      = NAN;
	      	hc->Off	      = 0.0;
	      	hc->TOffset   = 0.0;
	      	hc->GDFTYP    = 3;	// int16
	      	hc->SPR       = 1;	// one sample per block
	      	hc->bi 	      = 2*k;
	      	hc->bi8	      = BitsPerBlock;
	 	nbits = GDFTYP_BITS[hc->GDFTYP]*hc->SPR;
		BitsPerBlock   += nbits;
	      	hc->OnOff     = 1;
	      	hc->HighPass  = 0.16;
	      	hc->LowPass   = 70.0;
	      	hc->Notch     = 50;
	      	hc->Impedance = INFINITY;
	      	hc->fZ        = NAN;
	      	hc->XYZ[0] 	= 0.0;
	      	hc->XYZ[1] 	= 0.0;
	      	hc->XYZ[2] 	= 0.0;
	}

	// define EVENT structure
	hdr->EVENT.N = N_EVENT;
	hdr->EVENT.SampleRate = 0;
	hdr->EVENT.CodeDesc = NULL;
	hdr->EVENT.LenCodeDesc = 0;
	if (hdr->EVENT.N) {
		hdr->EVENT.POS = (uint32_t*) calloc(hdr->EVENT.N, sizeof(*hdr->EVENT.POS));
		hdr->EVENT.TYP = (uint16_t*) calloc(hdr->EVENT.N, sizeof(*hdr->EVENT.TYP));
		hdr->EVENT.DUR = (uint32_t*) calloc(hdr->EVENT.N, sizeof(*hdr->EVENT.DUR));
		hdr->EVENT.CHN = (uint16_t*) calloc(hdr->EVENT.N, sizeof(*hdr->EVENT.CHN));
#if (BIOSIG_VERSION >= 10500)
		hdr->EVENT.TimeStamp = (gdf_time*) calloc(hdr->EVENT.N, sizeof(gdf_time));
#endif
	} else {
		hdr->EVENT.POS = NULL;
		hdr->EVENT.TYP = NULL;
		hdr->EVENT.DUR = NULL;
		hdr->EVENT.CHN = NULL;
#if (BIOSIG_VERSION >= 10500)
		hdr->EVENT.TimeStamp = NULL;
#endif
	}

	// initialize specialized fields
	hdr->aECG = NULL;
	hdr->AS.bci2000 = NULL;

#if (BIOSIG_VERSION >= 10500)
	hdr->SCP.Section7  = NULL;
	hdr->SCP.Section8  = NULL;
	hdr->SCP.Section9  = NULL;
	hdr->SCP.Section10 = NULL;
	hdr->SCP.Section11 = NULL;
	hdr->SCP.Section7Length  = 0;
	hdr->SCP.Section8Length  = 0;
	hdr->SCP.Section9Length  = 0;
	hdr->SCP.Section10Length = 0;
	hdr->SCP.Section11Length = 0;
#endif

	return(hdr);
}

/* just for debugging
void debug_showptr(HDRTYPE* hdr) {
	fprintf(stdout,"=========================\n");
	fprintf(stdout,"&AS.Header=%p\n",hdr->AS.Header);
	fprintf(stdout,"&AS.auxBUF=%p\n",hdr->AS.auxBUF);
	fprintf(stdout,"&aECG=%p\n",hdr->aECG);
	fprintf(stdout,"&AS.bci2000=%p\n",hdr->AS.bci2000);
	fprintf(stdout,"&AS.rawEventData=%p\n",hdr->AS.rawEventData);
	fprintf(stdout,"&AS.rawData=%p\n",hdr->AS.rawdata);
	fprintf(stdout,"&data.block=%p\n",hdr->data.block);
	fprintf(stdout,"&CHANNEL=%p\n",hdr->CHANNEL);
	fprintf(stdout,"&EVENT.POS=%p\n",hdr->EVENT.POS);
	fprintf(stdout,"&EVENT.TYP=%p\n",hdr->EVENT.TYP);
	fprintf(stdout,"&EVENT.DUR=%p\n",hdr->EVENT.DUR);
	fprintf(stdout,"&EVENT.CHN=%p\n",hdr->EVENT.CHN);
	fprintf(stdout,"&EVENT.CodeDesc=%p\n",hdr->EVENT.CodeDesc);
	fprintf(stdout,"&FileName=%p %s\n",&hdr->FileName,hdr->FileName);
	fprintf(stdout,"&Hospital=%p\n",hdr->ID.Hospital);
}
*/

void destructHDR(HDRTYPE* hdr) {

	if (hdr==NULL) return;

	sclose(hdr);

	if (VERBOSE_LEVEL>7) fprintf(stdout,"destructHDR(%s): free HDR.aECG\n",hdr->FileName);
#if (BIOSIG_VERSION < 10500)
    	if (hdr->aECG != NULL) {
		if (((struct aecg*)hdr->aECG)->Section8.NumberOfStatements>0)
			free(((struct aecg*)hdr->aECG)->Section8.Statements);
		if (((struct aecg*)hdr->aECG)->Section11.NumberOfStatements>0)
			free(((struct aecg*)hdr->aECG)->Section11.Statements);
    		free(hdr->aECG);
    	}
#endif

	if (hdr->ID.Technician != NULL) free(hdr->ID.Technician);
	if (hdr->ID.Hospital   != NULL) free(hdr->ID.Hospital);

    	if (hdr->AS.bci2000 != NULL) free(hdr->AS.bci2000);

	if (VERBOSE_LEVEL>7)  fprintf(stdout,"destructHDR: free HDR.AS.rawdata @%p\n",hdr->AS.rawdata);

	// in case of SCPv3, rawdata can be loaded into Header
	if ( (hdr->AS.rawdata < hdr->AS.Header) || (hdr->AS.rawdata > (hdr->AS.Header+hdr->HeadLen)) )
		if (hdr->AS.rawdata != NULL) free(hdr->AS.rawdata);

	if (VERBOSE_LEVEL>7)  fprintf(stdout,"destructHDR: free HDR.data.block @%p\n",hdr->data.block);

    	if (hdr->data.block != NULL) free(hdr->data.block);

	if (VERBOSE_LEVEL>7)  fprintf(stdout,"destructHDR: free HDR.CHANNEL[] @%p %p\n",hdr->CHANNEL,hdr->rerefCHANNEL);

    	if (hdr->CHANNEL != NULL) free(hdr->CHANNEL);

	if (VERBOSE_LEVEL>7)  fprintf(stdout,"destructHDR: free HDR.AS.Header\n");

    	if (hdr->AS.rawEventData != NULL) free(hdr->AS.rawEventData);
    	if (hdr->AS.Header != NULL) free(hdr->AS.Header);

	if (VERBOSE_LEVEL>7)  fprintf(stdout,"destructHDR: free Event Table %p %p %p %p \n",hdr->EVENT.TYP,hdr->EVENT.POS,hdr->EVENT.DUR,hdr->EVENT.CHN);

    	if (hdr->EVENT.POS != NULL)  free(hdr->EVENT.POS);
    	if (hdr->EVENT.TYP != NULL)  free(hdr->EVENT.TYP);
    	if (hdr->EVENT.DUR != NULL)  free(hdr->EVENT.DUR);
    	if (hdr->EVENT.CHN != NULL)  free(hdr->EVENT.CHN);
#if (BIOSIG_VERSION >= 10500)
	if (hdr->EVENT.TimeStamp)    free(hdr->EVENT.TimeStamp);
#endif
    	if (hdr->EVENT.CodeDesc != NULL) free(hdr->EVENT.CodeDesc);

	if (VERBOSE_LEVEL>7)  fprintf(stdout,"destructHDR: free HDR.AS.auxBUF\n");

    	if (hdr->AS.auxBUF != NULL) free(hdr->AS.auxBUF);

	if (VERBOSE_LEVEL>7)  fprintf(stdout,"destructHDR: free HDR.rerefCHANNEL\n");

#ifdef CHOLMOD_H
        //if (hdr->Calib) cholmod_print_sparse(hdr->Calib,"destructHDR hdr->Calib",&CHOLMOD_COMMON_VAR);
	if (VERBOSE_LEVEL>7)  fprintf(stdout,"destructHDR: free hdr->Calib\n");
	if (hdr->Calib) cholmod_free_sparse(&hdr->Calib, &CHOLMOD_COMMON_VAR);

	if (VERBOSE_LEVEL>7)  fprintf(stdout,"destructHDR: free hdr->rerefCHANNEL %p\n",hdr->rerefCHANNEL);
	if (hdr->rerefCHANNEL) free(hdr->rerefCHANNEL);
#endif

	if (VERBOSE_LEVEL>7)  fprintf(stdout,"destructHDR: free HDR\n");

	if (hdr->FileName != NULL) free(hdr->FileName);

	if (hdr != NULL) free(hdr);
	return;
}


/****************************************************************************/
/**           INITIALIZE FIELDS OF A SINGLE CHANNEL TO DEFAULT VALUES      **/
/****************************************************************************/
void init_channel(struct CHANNEL_STRUCT *hc) {
	hc->PhysMin	= -1e9;
	hc->PhysMax	= +1e9;
	hc->DigMin	= ldexp(-1,15);
	hc->DigMax	= ldexp(1,15)-1;
	hc->Cal		= 1.0;
	hc->Off		= 0.0;

	hc->Label[0]	= '\0';
	hc->OnOff	= 1;
	hc->LeadIdCode	= 0;
	hc->Transducer[0] = '\0';
	hc->PhysDimCode	= 0;
#ifdef MAX_LENGTH_PHYSDIM
	hc->PhysDim[0]	= '?';
#endif
	hc->TOffset	= 0.0;
	hc->HighPass	= NAN;
	hc->LowPass	= NAN;
	hc->Notch	= NAN;
	hc->XYZ[0]	= 0;
	hc->XYZ[1]	= 0;
	hc->XYZ[2]	= 0;
	hc->Impedance	= NAN;

	hc->bufptr 	= NULL;
	hc->SPR 	= 1;
	hc->bi	 	= 0;
	hc->bi8	 	= 0;
	hc->GDFTYP 	= 3; // int16

}


/*  http://www.ietf.org/rfc/rfc1952.txt   */
const char *MAGIC_NUMBER_GZIP = "\x1f\x8B\x08";

/****************************************************************************/
/**                     GETFILETYPE                                        **/
/****************************************************************************/
HDRTYPE* getfiletype(HDRTYPE* hdr)
/*
	input:
		hdr->AS.Header1 contains first block of hdr->HeadLen bytes
		hdr->TYPE must be unknown, otherwise no FileFormat evaluation is performed
	output:
		hdr->TYPE	file format
		hdr->VERSION	is defined for some selected formats e.g. ACQ, EDF, BDF, GDF
 */
{
	// ToDo: use LEN to detect buffer overflow

    	hdr->TYPE = unknown;

	if (VERBOSE_LEVEL>7) fprintf(stdout,"[%s line %i]! %i\n", __func__,__LINE__, hdr->HeadLen);

#ifndef  ONLYGDF
   	const uint8_t MAGIC_NUMBER_FEF1[] = {67,69,78,13,10,0x1a,4,0x84};
	const uint8_t MAGIC_NUMBER_FEF2[] = {67,69,78,0x13,0x10,0x1a,4,0x84};
	const uint8_t MAGIC_NUMBER_Z[]    = {31,157,144};
	// const uint8_t MAGIC_NUMBER_ZIP[]  = {80,75,3,4};
	const uint8_t MAGIC_NUMBER_TIFF_l32[] = {73,73,42,0};
	const uint8_t MAGIC_NUMBER_TIFF_b32[] = {77,77,0,42};
	const uint8_t MAGIC_NUMBER_TIFF_l64[] = {73,73,43,0,8,0,0,0};
	const uint8_t MAGIC_NUMBER_TIFF_b64[] = {77,77,0,43,0,8,0,0};
	const uint8_t MAGIC_NUMBER_DICOM[]    = {8,0,5,0,10,0,0,0,73,83,79,95,73,82,32,49,48,48};
	const uint8_t MAGIC_NUMBER_UNIPRO[]   = {40,0,4,1,44,1,102,2,146,3,44,0,190,3};
	const uint8_t MAGIC_NUMBER_SYNERGY[]  =  {83,121,110,101,114,103,121,0,48,49,50,46,48,48,51,46,48,48,48,46,48,48,48,0,28,0,0,0,2,0,0,0};
	const char* MAGIC_NUMBER_BRAINVISION       = "Brain Vision Data Exchange Header File";
	const char* MAGIC_NUMBER_BRAINVISION1      = "Brain Vision V-Amp Data Header File Version";
	const char* MAGIC_NUMBER_BRAINVISIONMARKER = "Brain Vision Data Exchange Marker File, Version";
	const uint8_t MAGIC_NUMBER_NICOLET_WFT[] = {0x33,0,0x32,0,0x31,0,0x30,0};

    	/******** read 1st (fixed)  header  *******/
  	uint32_t U32 = leu32p(hdr->AS.Header+2);
	uint32_t MAGIC_EN1064_Section0Length  = leu32p(hdr->AS.Header+10);

	if ((U32>=30) & (U32<=45)) {
    		hdr->VERSION = (float)U32;
    		U32 = leu32p(hdr->AS.Header+6);
		if      ((hdr->VERSION <34.0) && (U32 == 150)) hdr->TYPE = ACQ;
		else if ((hdr->VERSION <35.0) && (U32 == 164)) hdr->TYPE = ACQ;
		else if ((hdr->VERSION <36.0) && (U32 == 326)) hdr->TYPE = ACQ;
		else if ((hdr->VERSION <37.0) && (U32 == 886)) hdr->TYPE = ACQ;
		else if ((hdr->VERSION <38.0) && (U32 ==1894)) hdr->TYPE = ACQ;
		else if ((hdr->VERSION <41.0) && (U32 ==1896)) hdr->TYPE = ACQ;
		else if ((hdr->VERSION <43.0) && (U32 ==1944)) hdr->TYPE = ACQ;
		//else if ((hdr->VERSION <45.0) && (U32 ==2976)) hdr->TYPE = ACQ;
		else if ((hdr->VERSION <45.0) && (U32 >=2220)) hdr->TYPE = ACQ;
		else if ((hdr->VERSION>=45.0) && (U32 ==(12944+160))) hdr->TYPE = ACQ;

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s ACQ %f %i\n", __FILE__, __LINE__, __func__, hdr->VERSION, U32);

	    	if (hdr->TYPE == ACQ) {
    			hdr->HeadLen = U32; // length of fixed header
			hdr->FILE.LittleEndian = 1;
    			return(hdr);
    		}
    	}

	U32 = beu32p(hdr->AS.Header+2);
	if ((U32==83)) {
		hdr->VERSION = (float)U32;
		U32 = beu32p(hdr->AS.Header+6);

		if      ((hdr->VERSION == 83) & (U32 == 1564)) hdr->TYPE = ACQ;

		if (hdr->TYPE == ACQ) {
			hdr->HeadLen = U32; // length of fixed header
			hdr->FILE.LittleEndian = 0;
			return(hdr);
		}
	}

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s (..): %u %u %u\n", __FILE__,__LINE__,__func__,beu32p(hdr->AS.Header+2),beu32p(hdr->AS.Header+6),beu32p(hdr->AS.Header+10));

#endif //ONLYGDF

	if (VERBOSE_LEVEL>7) fprintf(stdout,"(%s line %i: %x %x!  <%8s> TYPE=<%s>\n",__func__,__LINE__, leu16p(hdr->AS.Header),leu16p(hdr->AS.Header+154),hdr->AS.Header,GetFileTypeString(hdr->TYPE));

	if (hdr->TYPE != unknown)
		return(hdr);

#ifndef  ONLYGDF
	else if (!memcmp(hdr->AS.Header, "ABF ", 4)) {
    	// else if (!memcmp(Header1,"ABF \x66\x66\xE6\x3F",4)) { // ABF v1.8
		hdr->TYPE    = ABF;
    		hdr->VERSION = lef32p(hdr->AS.Header+4);
    	}
	else if (!memcmp(hdr->AS.Header, "ABF2\x00\x00", 6) && ( hdr->AS.Header[6] < 10 )  && ( hdr->AS.Header[7] < 10 ) ) {
		hdr->TYPE    = ABF2;
		hdr->VERSION = hdr->AS.Header[7] + ( hdr->AS.Header[6] / 10.0 );
    	}
    	else if (!memcmp(Header1+20,"ACR-NEMA",8))
	    	hdr->TYPE = ACR_NEMA;
    	else if (strstr(Header1,"ALPHA-TRACE-MEDICAL"))
	    	hdr->TYPE = alpha;
    	else if (!memcmp(Header1,"ATES MEDICA SOFT. EEG for Windows",33))
	    	hdr->TYPE = ATES;
    	else if (!memcmp(Header1,"ATF\x09",4))
    	        hdr->TYPE = ATF;
	else if (!memcmp(Header1,"AxGr",4)) {
		hdr->TYPE = AXG;
		hdr->VERSION = bei16p(hdr->AS.Header+4);
	}
	else if (!memcmp(Header1,"axgx",4)) {
		hdr->TYPE = AXG;
		hdr->VERSION = bei32p(hdr->AS.Header+4);
	}
    	else if (!memcmp(Header1,"ADU1",4) || !memcmp(Header1,"ADU2",4)  )
    	        hdr->TYPE = Axona;

    	else if (!memcmp(Header1,"HeaderLen=",10)) {
	    	hdr->TYPE = BCI2000;
	    	hdr->VERSION = 1.0;
	}
    	else if (!memcmp(Header1,"BCI2000V",8)) {
	    	hdr->TYPE = BCI2000;
	    	hdr->VERSION = 1.1;
	}
    	else if (!memcmp(Header1+1,"BIOSEMI",7) && (hdr->AS.Header[0]==0xff) && (hdr->HeadLen > 255)) {
    		hdr->TYPE = BDF;
    		hdr->VERSION = -1;
    	}
    	else if (!memcmp(Header1,"#BIOSIG ASCII",13))
	    	hdr->TYPE = ASCII;
    	else if (!memcmp(Header1,"#BIOSIG BINARY",14))
	    	hdr->TYPE = BIN;
    	else if ((leu16p(hdr->AS.Header)==207) && (leu16p(hdr->AS.Header+154)==0))
	    	hdr->TYPE = BKR;
    	else if (!memcmp(Header1+34,"BLSC",4))
	    	hdr->TYPE = BLSC;
    	else if (!memcmp(Header1,"bscs://",7))
	    	hdr->TYPE = BSCS;
    	else if (((beu16p(hdr->AS.Header)==0x0311) && (beu32p(hdr->AS.Header+4)==0x0809B002)
    		 && (leu16p(hdr->AS.Header+2) > 240) && (leu16p(hdr->AS.Header+2) < 250))  		// v2.40 - v2.50
    		 || !memcmp(hdr->AS.Header+307, "E\x00\x00\x00\x00\x00\x00\x00DAT", 11)
    		)
	    	hdr->TYPE = BLSC;
    	else if (!memcmp(Header1,"#BIOSIGDUMP v1.0",16)) {
	    	hdr->TYPE = BiosigDump;
	    	hdr->VERSION = 1.0;
	}
    	else if (!memcmp(Header1,"FileFormat = BNI-1-BALTIMORE",28))
	    	hdr->TYPE = BNI;
	else if (!memcmp(Header1, MAGIC_NUMBER_NICOLET_WFT, 8))	{ // WFT/Nicolet format
		hdr->TYPE = WFT;
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s \n", __FILE__, __LINE__, __func__);
	}
        else if (!memcmp(Header1,MAGIC_NUMBER_BRAINVISION,strlen(MAGIC_NUMBER_BRAINVISION)) || ((leu32p(hdr->AS.Header)==0x42bfbbef) && !memcmp(Header1+3, MAGIC_NUMBER_BRAINVISION,38)))
                hdr->TYPE = BrainVision;
        else if (!memcmp(Header1,MAGIC_NUMBER_BRAINVISION1,strlen(MAGIC_NUMBER_BRAINVISION1)))
                hdr->TYPE = BrainVisionVAmp;
        else if (!memcmp(Header1,MAGIC_NUMBER_BRAINVISIONMARKER,strlen(MAGIC_NUMBER_BRAINVISIONMARKER)))
                hdr->TYPE = BrainVisionMarker;
    	else if (!memcmp(Header1,"BZh91",5))
	    	hdr->TYPE = BZ2;
    	else if (!memcmp(Header1,"CDF",3))
	    	hdr->TYPE = CDF;
	else if (!memcmp(Header1,"CEDFILE",7))
		hdr->TYPE = CFS;
	else if (!memcmp(Header1+2,"(C) CED 87",10))
		hdr->TYPE = SMR;        // CED's SMR/SON format
	else if (!memcmp(Header1,"CFWB\1\0\0\0",8))
		hdr->TYPE = CFWB;
    	else if (!memcmp(Header1,"Version 3.0",11))
	    	hdr->TYPE = CNT;

    	else if (!memcmp(Header1,"MEG4",4))
	    	hdr->TYPE = CTF;
    	else if (!memcmp(Header1,"CTF_MRI_FORMAT VER 2.2",22))
	    	hdr->TYPE = CTF;
    	else if (!memcmp(Header1,"PATH OF DATASET:",16))
	    	hdr->TYPE = CTF;

    	else if (!memcmp(Header1,"DEMG",4))
	    	hdr->TYPE = DEMG;
    	else if (!memcmp(Header1+128,"DICM\x02\x00\x00\x00",8))
	    	hdr->TYPE = DICOM;
    	else if (!memcmp(Header1, MAGIC_NUMBER_DICOM,sizeof(MAGIC_NUMBER_DICOM)))
	    	hdr->TYPE = DICOM;
    	else if (!memcmp(Header1+12, MAGIC_NUMBER_DICOM,sizeof(MAGIC_NUMBER_DICOM)))
	    	hdr->TYPE = DICOM;
    	else if (!memcmp(Header1+12, MAGIC_NUMBER_DICOM,8))
	    	hdr->TYPE = DICOM;

	else if (!memcmp(Header1, "SctHdr\0\0Directory\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\xff\xff\xff\xff\xff\xff\xff\xff\0\0\0\0\0\0\0\0", 48)
	      && (leu32p((hdr->AS.Header+0x3C))==0x68) ) {
		hdr->TYPE = EAS;
	}
	else if (!memcmp(Header1,"Easy3File",10)) {
		hdr->TYPE = EZ3;
	}

    	else if (!memcmp(Header1,"EBS\x94\x0a\x13\x1a\x0d",8))
	    	hdr->TYPE = EBS;

    	else if (!memcmp(Header1,"0       ",8) && (hdr->HeadLen > 255)) {
	    	hdr->TYPE = EDF;
	    	hdr->VERSION = 0;
	}

	/* Nihon Kohden */
	else if (is_nihonkohden_signature((char*)Header1) && is_nihonkohden_signature((char*)(Header1+0x81))) {
	    	hdr->TYPE = EEG1100;
		hdr->VERSION = strtod((char*)Header1+11,NULL);
	}
    	else if (!memcmp(Header1, "RIFF",4) && !memcmp(Header1+8, "CNT ",4))
	    	hdr->TYPE = EEProbe;
    	else if (!memcmp(Header1, "EEP V2.0",8))
	    	hdr->TYPE = EEProbe;
    	else if (!memcmp(Header1, "\x26\x00\x10\x00",4))	// AVR
	    	hdr->TYPE = EEProbe;

    	else if (( bei32p(hdr->AS.Header) == 0x01020304) &&
    		 ((beu16p(hdr->AS.Header+4) == 0xffff) || (beu16p(hdr->AS.Header+4) == 3)) )
    	{
	    	hdr->TYPE = EGIS;
	    	hdr->FILE.LittleEndian = 0;
    	}
    	else if (( lei32p(hdr->AS.Header) == 0x01020304) &&
    		 ((leu16p(hdr->AS.Header+4) == 0xffff) || (leu16p(hdr->AS.Header+4) == 3)) )
	{
	    	hdr->TYPE = EGIS;
	    	hdr->FILE.LittleEndian = 1;
    	}

    	else if ((beu32p(hdr->AS.Header) > 1) && (beu32p(hdr->AS.Header) < 8) && !hdr->AS.Header[6]  && !hdr->AS.Header[8]  && !hdr->AS.Header[10]  && !hdr->AS.Header[12]  && !hdr->AS.Header[14]  && !hdr->AS.Header[26] ) {
		/* sanity check: the high byte of month, day, hour, min, sec and bits must be zero */
	    	hdr->TYPE = EGI;
	    	hdr->VERSION = hdr->AS.Header[3];
    	}
	else if (*(uint32_t*)(Header1) == htobe32(0x7f454c46))
	    	hdr->TYPE = ELF;
	else if ( (hdr->HeadLen > 64) && !memcmp(Header1+0x30,"GALNT EEG DATA",14))
		hdr->TYPE = EBNEURO;
    	else if ( (hdr->HeadLen > 14) && !memcmp(Header1,"Embla data file",15))
	    	hdr->TYPE = EMBLA;
    	else if ( (hdr->HeadLen > 4) && ( !memcmp(Header1,"PBJ",3) || !memcmp(Header1,"BPC",3) ) )
	    	hdr->TYPE = EMSA;
    	else if (strstr(Header1,"Subject") && strstr(Header1,"Target.OnsetTime") && strstr(Header1,"Target.RTTime") && strstr(Header1,"Target.RESP"))
	    	hdr->TYPE = ePrime;
    	else if (!memcmp(Header1,"[Header]",8))
	    	hdr->TYPE = ET_MEG;
    	else if ( (hdr->HeadLen > 19) && !memcmp(Header1,"Header\r\nFile Version'",20))
	    	hdr->TYPE = ETG4000;
    	else if (!memcmp(Header1,"|CF,",4))
	    	hdr->TYPE = FAMOS;

    	else if (!memcmp(Header1,MAGIC_NUMBER_FEF1,sizeof(MAGIC_NUMBER_FEF1)) || !memcmp(Header1,MAGIC_NUMBER_FEF2,sizeof(MAGIC_NUMBER_FEF1))) {
	    	hdr->TYPE = FEF;
		char tmp[9];tmp[8] = 0;
		memcpy(tmp,hdr->AS.Header+8,8);
		hdr->VERSION = (float)atol(tmp);
    	}
	else if (!memcmp(hdr->AS.Header,   "\0\0\0\x64\0\0\0\x1f\0\0\0\x14\0\0\0\0\0\1",4) &&
		 !memcmp(hdr->AS.Header+36,"\0\0\0\x65\0\0\0\3\0\0\0\4\0\0",14) &&
		 !memcmp(hdr->AS.Header+56,"\0\0\0\x6a\0\0\0\3\0\0\0\4\0\0\0\0\xff\xff\xff\xff\0\0",22)
		)
		hdr->TYPE = FIFF;
    	else if (!memcmp(Header1,"fLaC",4))
	    	hdr->TYPE = FLAC;
#endif //ONLYGDF

    	else if (!memcmp(Header1,"GDF",3) && (hdr->HeadLen > 255)) {
	    	hdr->TYPE = GDF;
		char tmp[6]; tmp[5] = 0;
		memcpy(tmp,hdr->AS.Header+3, 5);
	    	hdr->VERSION 	= strtod(tmp,NULL);
	}

#ifndef  ONLYGDF
    	else if (!memcmp(Header1,"GIF87a",6))
	    	hdr->TYPE = GIF;
    	else if (!memcmp(Header1,"GIF89a",6))
	    	hdr->TYPE = GIF;
    	else if ( (hdr->HeadLen > 21) && !memcmp(Header1,"GALILEO EEG TRACE FILE",22))
	    	hdr->TYPE = GTF;
	else if (!memcmp(Header1,MAGIC_NUMBER_GZIP,strlen(MAGIC_NUMBER_GZIP)))  {
		hdr->TYPE = GZIP;
//		hdr->FILE.COMPRESSION = 1;
	}
    	else if (!memcmp(Header1,"\x89HDF\x0d\x0a\x1a\x0a",8))
	    	hdr->TYPE = HDF;
	else if (!memcmp(Header1,"DATA\0\0\0\0",8)) {
		hdr->TYPE = HEKA;
		hdr->VERSION = 0;
	}
	else if (!memcmp(Header1,"DAT1\0\0\0\0",8)) {
		hdr->TYPE = HEKA;
		hdr->VERSION = 1;
	}
	else if (!memcmp(Header1,"DAT2\0\0\0\0",8)) {
		hdr->TYPE = HEKA;
		hdr->VERSION = 2;
	}
    	else if (!memcmp(Header1,"IGOR",4))
	    	hdr->TYPE = ITX;
	else if (*(int16_t*)Header1==0x0001 ||
		 *(int16_t*)Header1==0x0002 ||
		 *(int16_t*)Header1==0x0003 ||
		 *(int16_t*)Header1==0x0005 ) {
		 /* no swapping */
		hdr->TYPE = IBW;
		hdr->FILE.LittleEndian = (__BYTE_ORDER == __LITTLE_ENDIAN);
		hdr->VERSION = *(int16_t*)Header1;
	}
	else if (*(int16_t*)Header1==0x0100 ||
		 *(int16_t*)Header1==0x0200 ||
		 *(int16_t*)Header1==0x0300 ||
		 *(int16_t*)Header1==0x0500 ) {
		 /* data need to be swapped */
		hdr->TYPE = IBW;
		hdr->FILE.LittleEndian = (__BYTE_ORDER == __BIG_ENDIAN);
		hdr->VERSION = bswap_16(*(int16_t*)Header1);
	}
	else if (!memcmp(Header1,"ANN  1.0",8))
		hdr->TYPE = ISHNE;
    	else if (!memcmp(Header1,"ISHNE1.0",8))
	    	hdr->TYPE = ISHNE;
    	else if (!memcmp(Header1,"@  MFER ",8))
	    	hdr->TYPE = MFER;
    	else if (!memcmp(Header1,"@ MFR ",6))
	    	hdr->TYPE = MFER;
    	else if (!memcmp(Header1,"MATLAB 5.0 MAT-file, ",7) && !memcmp(Header1+10," MAT-file, ",11) ) {
	    	hdr->TYPE = Matlab;
		hdr->VERSION = (Header1[7]-'0') + (Header1[9]-'0')/10.0;
	}
    	else if (!memcmp(Header1,"%%MatrixMarket",14))
	    	hdr->TYPE = MM;
/*    	else if (!memcmp(Header1,"MThd\000\000\000\001\000",9))
	    	hdr->TYPE = MIDI;
*/

    	else if (!memcmp(Header1,"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x3E\x00\x03\x00\xFE\xFF\x09\x00\x06",0x21))
	    	hdr->TYPE = MSI;

	else if (!strcmp(Header1,"(*This is a Mathematica binary dump file. It can be loaded with Get.*)"))
		hdr->TYPE = MX;

	else if ( (hdr->HeadLen>346) &&
		  (Header1[344]=='n') && (Header1[347]=='\0') && \
		  ((Header1[345]=='i') || (Header1[345]=='+') ) && \
		   (Header1[346]>'0') && (Header1[346]<='9') ) {
			hdr->TYPE = NIFTI;
			hdr->VERSION = Header1[346]-'0';
		}
    	else if ( (hdr->HeadLen > 344) && (!memcmp(Header1+344,"ni1",4) || !memcmp(Header1+344,"n+1",4) ) )
	    	hdr->TYPE = NIFTI;
    	else if (!memcmp(Header1,"NEURALEV",8) || !memcmp(Header1,"N.EV.",6) )
	    	hdr->TYPE = NEV;
	else if (!memcmp(Header1,"NEX1",3))
	    	hdr->TYPE = NEX1;
    	else if ( (hdr->HeadLen > 31) && !memcmp(Header1,"Logging Start\x0aLogger SW Version: ",31))
	    	hdr->TYPE = NeuroLoggerHEX;
    	else if (!memcmp(Header1,"Neuron",6))
	    	hdr->TYPE = NEURON;
	else if (!memcmp(Header1,"\x93NUMPY",6)) {
		hdr->TYPE = NUMPY;
	}
    	else if (!memcmp(Header1,"[FileInfo]",10))
	    	hdr->TYPE = Persyst;
    	else if (!memcmp(Header1,"SXDF",4))
	    	hdr->TYPE = OpenXDF;
	else if (!memcmp(Header1,"PLEX",4)) {
		hdr->TYPE = PLEXON;
		hdr->VERSION=1.0;
	}
	else if (!memcmp(Header1+10,"PLEXON",6)) {
		hdr->TYPE = PLEXON;
		hdr->VERSION=2.0;
	}
	else if (!memcmp(Header1,"\x02\x27\x91\xC6",4)) {
		hdr->TYPE = RHD2000;	// Intan RHD2000 format
		hdr->FILE.LittleEndian = 1;
	}
	else if (!memcmp(Header1,"\xAC\x27\x91\xD6",4)) {
		hdr->TYPE = RHS2000;	// Intan RHS2000 format
		hdr->FILE.LittleEndian = 1;
	}
	else if (!memcmp(Header1,"\x81\xa4\xb1\xf3",4) & (leu16p(Header1+8) < 2)) {
		hdr->TYPE = IntanCLP;	// Intan CLP format, we'll use same read for now
		hdr->FILE.LittleEndian = 1;
	}
    	else if (!memcmp(Header1,"\x55\xAA\x00\xb0",2)) {
	    	hdr->TYPE = RDF;	// UCSD ERPSS aquisition system
	    	hdr->FILE.LittleEndian = 1;
	}
    	else if (!memcmp(Header1,"\xAA\x55\xb0\x00",2)) {
	    	hdr->TYPE = RDF;	// UCSD ERPSS aquisition system
	    	hdr->FILE.LittleEndian = 0;
	}
    	else if (!memcmp(Header1,"RIFF",4)) {
	    	hdr->TYPE = RIFF;
	    	if (!memcmp(Header1+8,"WAVE",4))
	    		hdr->TYPE = WAV;
	    	if (!memcmp(Header1+8,"AIF",3))
	    		hdr->TYPE = AIFF;
	    	if (!memcmp(Header1+8,"AVI ",4))
	    		hdr->TYPE = AVI;
	}
	// general SCP
	else if (  (hdr->HeadLen>32) &&
                   ( MAGIC_EN1064_Section0Length    >  120)
		&& ( MAGIC_EN1064_Section0Length    <  16+10*1024)
		&& ((MAGIC_EN1064_Section0Length%10)== 6)
		&& (*(uint16_t*)(hdr->AS.Header+ 8) == 0x0000)
		&& (leu32p(hdr->AS.Header+10) == leu32p(hdr->AS.Header+24))
		&& (  (!memcmp(hdr->AS.Header+16,"SCPECG\0\0",8))
		   || (*(uint64_t*)(hdr->AS.Header+16) == 0)
		   )
		&& (leu32p(hdr->AS.Header+28) == (uint32_t)0x00000007)
		&& (leu16p(hdr->AS.Header+32) == (uint16_t)0x0001)
		) {
	    	hdr->TYPE = SCP_ECG;
	}
/*
	// special SCP files - header is strange, files can be decoded
	else if (  (leu32p(hdr->AS.Header+10) == 136)
		&& (*(uint16_t*)(hdr->AS.Header+ 8) == 0x0000)
		&& (  (!memcmp(hdr->AS.Header+14,"\x0A\x01\x25\x01\x99\x01\xE7\x49\0\0",10))
		   || (!memcmp(hdr->AS.Header+14,"\x0A\x00\x90\x80\0\0\x78\x80\0\0",10))
		   || (!memcmp(hdr->AS.Header+14,"\x0A\xCD\xCD\xCD\xCD\xCD\xCD\xCD\0\0",10))
		   )
		&& (leu32p(hdr->AS.Header+24) == 136)
		&& (leu32p(hdr->AS.Header+28) == 0x0007)
		&& (leu16p(hdr->AS.Header+32) == 0x0001)
		)  {
	    	hdr->TYPE = SCP_ECG;
	    	hdr->VERSION = -2;
	}
	else if (  (leu32p(hdr->AS.Header+10)       == 136)
		&& (*(uint16_t*)(hdr->AS.Header+ 8) == 0x0000)
		&& (*(uint8_t*) (hdr->AS.Header+14) == 0x0A)
		&& (*(uint8_t*) (hdr->AS.Header+15) == 0x0B)
		&& (*(uint32_t*)(hdr->AS.Header+16) == 0)
		&& (*(uint32_t*)(hdr->AS.Header+20) == 0)
		&& (*(uint32_t*)(hdr->AS.Header+24) == 0)
		&& (*(uint32_t*)(hdr->AS.Header+28) == 0)
		&& (leu16p(hdr->AS.Header+32)       == 0x0001)
		) {
	    	hdr->TYPE = SCP_ECG;
	    	hdr->VERSION = -3;
	}
*/
/*
	// special SCP files - header is strange, files cannot be decoded
	else if (  (leu32p(hdr->AS.Header+10) == 136)
		&& (*(uint16_t*)(hdr->AS.Header+ 8) == 0x0000)
		&& (leu16p(hdr->AS.Header+14) == 0x0b0b)
		&& (!memcmp(hdr->AS.Header+16,"x06SCPECG",7))
		)  {
	    	hdr->TYPE = SCP_ECG;
	    	hdr->VERSION = -1;
	}
	else if (  (leu32p(hdr->AS.Header+10) == 136)
		&& (*(uint16_t*)(hdr->AS.Header+ 8) == 0x0000)
		&& (leu16p(hdr->AS.Header+14) == 0x0d0d)
		&& (!memcmp(hdr->AS.Header+16,"SCPEGC\0\0",8))
		&& (leu32p(hdr->AS.Header+24) == 136)
		&& (leu32p(hdr->AS.Header+28) == 0x0007)
		&& (leu16p(hdr->AS.Header+32) == 0x0001)
		)  {
	    	hdr->TYPE = SCP_ECG;
	    	hdr->VERSION = -4;
	}
*/

	else if ((hdr->HeadLen > 78) && !memcmp(Header1,"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!000000000000000000000000000000",78))
		hdr->TYPE = SASXPT;	// SAS Transport file format (XPORT)
	else if (!memcmp(Header1,"$FL2@(#) SPSS DATA FILE",8)) {
		hdr->TYPE = SPSS;	// SPSS file format
		switch (*(uint32_t*)(Header1+64)) {
		case 0x00000002:
		case 0x00000003:
		    	hdr->FILE.LittleEndian = 1;
		    	break;
		case 0x02000000:
		case 0x03000000:
		    	hdr->FILE.LittleEndian = 0;
		    	break;
		}
	}
	else if ((Header1[0]==0x71 || Header1[0]==0x72) && (Header1[1]==1 || Header1[1]==2) && Header1[2]==1  && Header1[3]==0 )
		hdr->TYPE = STATA;
	else if (!memcmp(Header1,"IAvSFo",6))
		hdr->TYPE = SIGIF;
	else if (!memcmp(Header1,"position,duration,channel,type,name\n",35))
		hdr->TYPE = SigViewerEventsCSV;
	else if ((hdr->HeadLen>23) && !memcmp(Header1,"SQLite format 3\000",16) && Header1[21]==64 && Header1[22]==32 && Header1[23]==32 )
		hdr->TYPE = SQLite;
	else if ((hdr->HeadLen>23) && !memcmp(Header1,"\"Snap-Master Data File\"",24))
	    	hdr->TYPE = SMA;
	else if (!memcmp(Header1,".snd",5))
		hdr->TYPE = SND;
	else if (!memcmp(Header1,".snd",5))
		hdr->TYPE = SND;

	else if (!memcmp(Header1,"TDSm",4))
		hdr->TYPE = TDMS; 	// http://www.ni.com/white-paper/5696/en

	else if ((hdr->HeadLen>30) && !memcmp(Header1,"POLY SAMPLE FILEversion ",24) && !memcmp(Header1+28, "\x0d\x0a\x1a",3))
		hdr->TYPE = TMS32;
	else if ((hdr->HeadLen>35) && !memcmp(Header1,"FileId=TMSi PortiLab sample log file\x0a\x0dVersion=",35))
		hdr->TYPE = TMSiLOG;
	else if (!memcmp(Header1,MAGIC_NUMBER_TIFF_l32,4))
		hdr->TYPE = TIFF;
	else if (!memcmp(Header1,MAGIC_NUMBER_TIFF_b32,4))
		hdr->TYPE = TIFF;
	else if (!memcmp(Header1,MAGIC_NUMBER_TIFF_l64,8))
		hdr->TYPE = TIFF;
	else if (!memcmp(Header1,MAGIC_NUMBER_TIFF_b64,8))
		hdr->TYPE = TIFF;
	else if (!memcmp(Header1,"#VRML",5))
		hdr->TYPE = VRML;
	else if ((hdr->HeadLen > 17) && !memcmp(hdr->AS.Header+4,MAGIC_NUMBER_UNIPRO,14))
		hdr->TYPE = UNIPRO;

	else if (!memcmp(Header1,MAGIC_NUMBER_SYNERGY,sizeof(MAGIC_NUMBER_SYNERGY))
		&& !strncmp(Header1+63,"CRawDataElement",15)
		&& !strncmp(Header1+85,"CRawDataBuffer",14) )  {
		hdr->TYPE = SYNERGY;
    	}
	else if ((hdr->HeadLen > 23) && !memcmp(Header1,"# vtk DataFile Version ",23)) {
		hdr->TYPE = VTK;
		char tmp[4]; tmp[3]=0;
		memcpy(tmp,(char*)Header1+23,3);
		hdr->VERSION = strtod(tmp,NULL);
	}
	else if (!strncmp(Header1,"Serial number",13))
		hdr->TYPE = ASCII_IBI;
	else if (!memcmp(Header1,"VER=9\r\nCTIME=",13))
		hdr->TYPE = WCP;
	else if (!memcmp(Header1,"\xAF\xFE\xDA\xDA",4) || !memcmp(Header1,"\xDA\xDA\xFE\xAF",4) || !memcmp(Header1,"\x55\x55\xFE\xAF",4) )
		hdr->TYPE = WG1;	// Walter Graphtek
	else if (!memcmp(Header1,MAGIC_NUMBER_Z,3))
		hdr->TYPE = Z;
	else if (!strncmp(Header1,"PK\003\004",4))
		hdr->TYPE = ZIP;
	else if (!strncmp(Header1,"PK\005\006",4))
		hdr->TYPE = ZIP;
	else if (!strncmp(Header1,"!<arch>\n",8))
		hdr->TYPE = MSVCLIB;
/*
	else if (!strncmp(Header1,"XDF",3))
		hdr->TYPE = XDF;
 */
	else if (!strncmp(Header1,"ZIP2",4))
		hdr->TYPE = ZIP2;
	else if ((hdr->HeadLen>13) && !memcmp(Header1,"<?xml version",13))
		hdr->TYPE = HL7aECG;
	else if ( (leu32p(hdr->AS.Header) & 0x00FFFFFFL) == 0x00BFBBEFL
		&& !memcmp(Header1+3,"<?xml version",13))
		hdr->TYPE = HL7aECG;	// UTF8
	else if (leu16p(hdr->AS.Header)==0xFFFE)
	{	hdr->TYPE = XML; // UTF16 BigEndian
		hdr->FILE.LittleEndian = 0;
    	}
	else if (leu16p(hdr->AS.Header)==0xFEFF)
	{	hdr->TYPE = XML; // UTF16 LittleEndian
		hdr->FILE.LittleEndian = 1;
    	}
	else if ((hdr->HeadLen>40) && !memcmp(hdr->AS.Header,"V3.0            ",16) && !memcmp(hdr->AS.Header+32,"[PatInfo]",9)) {
		hdr->TYPE = Sigma;
		hdr->VERSION = 3.0;
	}
	else if ((hdr->HeadLen > 175) && (hdr->AS.Header[175] < 5))
	{	hdr->TYPE = TRC; // Micromed *.TRC format
		hdr->FILE.LittleEndian = 1;
    	}
	else if (!memcmp(hdr->AS.Header,"\x4c\x00\x00\x00\x01\x14\x02\x00\x00\x00\x00\x00\xC0\x00\x00\x00\x00\x00\x46",20))
	{	hdr->TYPE = MS_LNK; // Microsoft *.LNK format
		hdr->FILE.LittleEndian = 1;
	}
	else if ((hdr->HeadLen > 175) && (hdr->AS.Header[175] < 5))
	{	hdr->TYPE = TRC; // Micromed *.TRC format
		hdr->FILE.LittleEndian = 1;
	}
	else {
		// if the first 4 bytes represent the file length
		struct stat FileBuf;
		if (stat(hdr->FileName,&FileBuf)==0
		&& (leu32p(hdr->AS.Header+2)==FileBuf.st_size)
		&& (leu16p(hdr->AS.Header)==0) )
		{
			// Cardioview 3000 generates such SCP files
			hdr->TYPE = SCP_ECG;
			hdr->FILE.LittleEndian = 1;
			fprintf(stderr,"Warning SOPEN (SCP): this kind of an SCP file predates the official SCP (EN1064) standard and is not fully implemented.\n" );
			if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i): %i %s %s \n",__FILE__,__LINE__,hdr->TYPE,GetFileTypeString(hdr->TYPE),hdr->FileName);
		}
		if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i): 0x%x 0x%x \n",__FILE__,__LINE__,leu32p(hdr->AS.Header),(int)FileBuf.st_size);
	}

#endif //ONLYGDF

	if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i): %i %s %s \n",__FILE__,__LINE__,hdr->TYPE,GetFileTypeString(hdr->TYPE),hdr->FileName);

	return(hdr);
}


const struct FileFormatStringTable_t FileFormatStringTable[] = {
	{ unknown,    	"unknown" },
	{ alpha,    	"alpha" },
	{ ABF,    	"ABF" },
	{ ABF2,    	"ABF2" },
	{ ACQ,    	"ACQ" },
	{ ACR_NEMA,    	"ACR_NEMA" },
	{ AINF,    	"AINF" },
	{ AIFC,    	"AIFC" },
	{ AIFF,    	"AIFF" },
	{ ARC,    	"ARC(Cadwell)" },
	{ ARFF,    	"ARFF" },
	{ ASCII,    	"ASCII" },
	{ ATES,    	"ATES" },
	{ ATF,    	"ATF" },
	{ AU,    	"AU" },
	{ AXG,    	"AXG" },
	{ Axona,    	"Axona" },
	{ BCI2000,    	"BCI2000" },
	{ BDF,    	"BDF" },
	{ BESA,    	"BESA" },
	{ BIN,    	"BINARY" },
	{ BiosigDump,   "BIOSIGDUMP" },
	{ BKR,    	"BKR" },
	{ BLSC,    	"BLSC" },
	{ BMP,    	"BMP" },
	{ BNI,    	"BNI-1-Baltimore/Nicolet" },
	{ BrainVision,  "BrainVision" },
	{ BrainVisionVAmp, "BrainVision" },
	{ BrainVisionMarker, "BrainVision" },
	{ BZ2,    	"BZ2" },
	{ CDF,    	"CDF" },
	{ CFS,    	"CFS" },
	{ CFWB,    	"CFWB" },
	{ CNT,    	"CNT" },
	{ CTF,    	"CTF" },
	{ DEMG,    	"DEMG" },
	{ DICOM,    	"DICOM" },
	{ EAS,    	"EAS(Cadwell)" },
	{ EBNEURO,	"EBNEURO"},
	{ EBS,    	"EBS" },
	{ EDF,    	"EDF" },
	{ EEG1100,    	"EEG1100" },
	{ EEProbe,    	"EEProbe" },
	{ EGI,    	"EGI" },
	{ EGIS,    	"EGIS" },
	{ ELF,    	"ELF" },
	{ EMBLA,    	"EMBLA" },
	{ EMSA,    	"EMSA" },
	{ ePrime,    	"ePrime" },
	{ ET_MEG,    	"ET-MEG" },
	{ ETG4000,    	"ETG4000" },
	{ EVENT,    	"EVENT" },
	{ EXIF,    	"EXIF" },
	{ EZ3,    	"EZ3(Cadwell)" },
	{ FAMOS,    	"FAMOS" },
	{ FEF,    	"FEF" },
	{ FIFF,    	"FIFF" },
	{ FITS,    	"FITS" },
	{ FLAC,    	"FLAC" },
	{ GDF,    	"GDF" },
	{ GIF,    	"GIF" },
	{ GTF,    	"GTF" },
	{ GZIP,    	"GZIP" },
	{ HDF,    	"HDF" },
	{ HEKA,    	"HEKA" },
	{ HL7aECG,    	"HL7aECG" },
	{ IBW,    	"IBW" },
	{ ITX,    	"ITX" },
	{ ISHNE,    	"ISHNE" },
	{ JPEG,    	"JPEG" },
	{ JSON,    	"JSON" },
	{ Matlab,    	"MAT" },
	{ MFER,    	"MFER" },
	{ MIDI,    	"MIDI" },
	{ MIT,    	"MIT" },
	{ MM,    	"MatrixMarket" },
	{ MSI,    	"MSI" },
	{ MS_LNK,    	".LNK" },
	{ MSVCLIB,    	"MS VC++ Library" },
	{ MX,    	"Mathematica serialized package format" },
	{ native,    	"native" },
	{ NeuroLoggerHEX, "NeuroLoggerHEX"},
	{ NetCDF,    	"NetCDF" },
	{ NEV,    	"NEV" },
	{ NEX1,    	"NEX" },
	{ NIFTI,    	"NIFTI" },
	{ NEURON,    	"NEURON" },
	{ NUMPY,    	"NUMPY" },
	{ Persyst,    	"Persyst" },
	{ OGG,    	"OGG" },
	{ PDP,    	"PDP" },
	{ PLEXON,    	"PLEXON" },
	{ RDF,    	"RDF" },
	{ IntanCLP,    	"IntanCLP" },
	{ RHD2000,    	"RHD2000" },
	{ RHS2000,    	"RHS2000" },
	{ RIFF,    	"RIFF" },
	{ SASXPT,    	"SAS_XPORT" },
	{ SCP_ECG,    	"SCP" },
	{ SIGIF,    	"SIGIF" },
	{ Sigma,    	"Sigma" },
	{ SigViewerEventsCSV, "SigViewer's CSV event table"},
	{ SMA,    	"SMA" },
	{ SMR,    	"SON/SMR" },
	{ SND,    	"SND" },
	{ SPSS,    	"SPSS" },
	{ SQLite,    	"SQLite" },
	{ STATA,    	"STATA" },
	{ SVG,    	"SVG" },
	{ SYNERGY,      "SYNERGY"},
	{ TDMS,    	"TDMS (NI)" },
	{ TIFF,    	"TIFF" },
	{ TMS32,    	"TMS32" },
	{ TMSiLOG,    	"TMSiLOG" },
	{ TRC,    	"TRC" },
	{ UNIPRO,    	"UNIPRO" },
	{ VRML,    	"VRML" },
	{ VTK,    	"VTK" },
	{ WAV,    	"WAV" },
	{ WCP,    	"WCP" },
	{ WFT,    	"WFT/Nicolet" },
	{ WG1,    	"Walter Graphtek" },
	{ WMF,    	"WMF" },
	{ XDF,    	"XDF" },
	{ XML,    	"XML" },
	{ ZIP,    	"ZIP" },
	{ ZIP2,    	"ZIP2" },
	{ Z,    	"Z" },
	{ noFile,    	NULL }
} ;


/* ------------------------------------------
 *   	returns string of file type
 * ------------------------------------------- */
const char* GetFileTypeString(enum FileFormat FMT) {
	uint16_t k;
	for (k=0; ; k++) {
		if (FMT==FileFormatStringTable[k].fmt)
			return (FileFormatStringTable[k].FileTypeString);
		if (noFile==FileFormatStringTable[k].fmt)  	// stopping criteria: last element in FileFormatStringTable
			return (NULL);
	}
}

/* ------------------------------------------
 *   	returns file type from type string
 * ------------------------------------------- */
enum FileFormat GetFileTypeFromString(const char *FileTypeString) {
	uint16_t k;
	for (k=0; ; k++) {
		if (FileFormatStringTable[k].FileTypeString == NULL) 	// stopping criteria: last element in FileFormatStringTable
			return (noFile);
		if (!strcmp(FileFormatStringTable[k].FileTypeString, FileTypeString))
			return (FileFormatStringTable[k].fmt);
	}
}


/****************************************************************************/
/**                     struct2gdfbin                                      **/
/****************************************************************************/
void struct2gdfbin(HDRTYPE *hdr)
{
	size_t k;
	char tmp[81];
	uint32_t Dur[2];

	// NS number of channels selected for writing
     	typeof(hdr->NS)  NS = 0;
	for (k=0; k<hdr->NS; k++) {
		CHANNEL_TYPE *hc = hdr->CHANNEL+k;
		if (hc->OnOff) NS++;
		hc->Cal = (hc->PhysMax-hc->PhysMin)/(hc->DigMax-hc->DigMin);
		hc->Off =  hc->PhysMin-hc->Cal*hc->DigMin;
	}

 	    	hdr->HeadLen = (NS+1)*256;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %p\n", __func__, __LINE__, hdr->HeadLen, hdr->EVENT.LenCodeDesc, hdr->EVENT.CodeDesc);

		/******
		 *	The size of Header 3 is computed by going through all TLV triples,
		 *	and compute HeadLen to allocate sufficient amount of memory
		 *	Header 3 is filled later in a 2nd scan below
		 ******/

		/* writing header 3, in Tag-Length-Value from
		 */
		uint32_t TagNLen[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
		uint8_t tag=1;
		if (hdr->EVENT.LenCodeDesc > 1) {	// first entry is always empty - no need to save tag1
			for (k=0; k<hdr->EVENT.LenCodeDesc; k++)
		     		TagNLen[tag] += strlen(hdr->EVENT.CodeDesc[k])+1;
	     		TagNLen[tag] += 1; 			// acounts for terminating \0
	     		hdr->HeadLen += 4+TagNLen[tag];
	     	}
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
	     	tag = 2;
	     	if (hdr->AS.bci2000 != NULL) {
	     		TagNLen[tag] = strlen(hdr->AS.bci2000)+1;
	     		hdr->HeadLen += 4+TagNLen[tag];
	     	}
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
	     	tag = 3;
	     	if ((hdr->ID.Manufacturer.Name != NULL) || (hdr->ID.Manufacturer.Model != NULL) || (hdr->ID.Manufacturer.Version != NULL) || (hdr->ID.Manufacturer.SerialNumber != NULL)) {
	     		if (hdr->ID.Manufacturer.Name == NULL) hdr->ID.Manufacturer.Name="";
	     		if (hdr->ID.Manufacturer.Model == NULL) hdr->ID.Manufacturer.Model="";
	     		if (hdr->ID.Manufacturer.Version == NULL) hdr->ID.Manufacturer.Version="";
	     		if (hdr->ID.Manufacturer.SerialNumber == NULL) hdr->ID.Manufacturer.SerialNumber="";

	     		TagNLen[tag] = strlen(hdr->ID.Manufacturer.Name)+strlen(hdr->ID.Manufacturer.Model)+strlen(hdr->ID.Manufacturer.Version)+strlen(hdr->ID.Manufacturer.SerialNumber)+4;
	     		hdr->HeadLen += 4+TagNLen[tag];
	     	}
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
	     	tag = 4;
/* OBSOLETE
	     	char FLAG_SENSOR_ORIENTATION = 0;
	     	for (k=0; k<hdr->NS; k++) {
	     		FLAG_SENSOR_ORIENTATION |= hdr->CHANNEL[k].Orientation[0] != (float)0.0;
	     		FLAG_SENSOR_ORIENTATION |= hdr->CHANNEL[k].Orientation[1] != (float)0.0;
	     		FLAG_SENSOR_ORIENTATION |= hdr->CHANNEL[k].Orientation[2] != (float)0.0;
	     		FLAG_SENSOR_ORIENTATION |= hdr->CHANNEL[k].Area != (float)0.0;
	     	}
	     	if (FLAG_SENSOR_ORIENTATION)
	     		TagNLen[tag] = hdr->NS*sizeof(float)*4;
     		hdr->HeadLen += 4+TagNLen[tag];
*/
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
	     	tag = 5;
	     	for (k=0; k<16; k++) {
	     		if (hdr->IPaddr[k]) {
		     		if (k<4) TagNLen[tag] = 4;
		     		else 	 TagNLen[tag] = 16;
		     	}
	     		hdr->HeadLen += 4+TagNLen[tag];
	     	}
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
	     	tag = 6;
     		TagNLen[tag] = hdr->ID.Technician==NULL ? 0 :  strlen(hdr->ID.Technician);
	     	if (TagNLen[tag]) {
	     		TagNLen[tag]++;
	     		hdr->HeadLen += 4+TagNLen[tag];
	     	}
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
	     	tag = 7;
		if (hdr->ID.Hospital!=NULL) {
	     		TagNLen[tag] = strlen(hdr->ID.Hospital);
		     	if (TagNLen[tag]) {
	     			TagNLen[tag]++;
	     			hdr->HeadLen += 4+TagNLen[tag];
	     		}
	     	}
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);

#if (BIOSIG_VERSION >= 10500)
		tag = 9;
		if (hdr->SCP.Section7 != NULL) {
			TagNLen[tag] = hdr->SCP.Section7Length;  // leu32p(hdr->SCP.Section7+4);
			if (TagNLen[tag]) {
				hdr->HeadLen += 4+TagNLen[tag];
			}
		}
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
		tag = 10;
		if (hdr->SCP.Section8 != NULL) {
			TagNLen[tag] = hdr->SCP.Section8Length;  // leu32p(hdr->SCP.Section8+4);
			if (TagNLen[tag]) {
				hdr->HeadLen += 4+TagNLen[tag];
			}
		}
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
		tag = 11;
		if (hdr->SCP.Section9 != NULL) {
			TagNLen[tag] = hdr->SCP.Section9Length;  // leu32p(hdr->SCP.Section9+4);
			if (TagNLen[tag]) {
				hdr->HeadLen += 4+TagNLen[tag];
			}
		}
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
		tag = 12;
		if (hdr->SCP.Section10 != NULL) {
			TagNLen[tag] = hdr->SCP.Section10Length;  // leu32p(hdr->SCP.Section10+4);
			if (TagNLen[tag]) {
				hdr->HeadLen += 4+TagNLen[tag];
			}
		}
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
		tag = 13;
		if (hdr->SCP.Section11 != NULL) {
			TagNLen[tag] = hdr->SCP.Section11Length;  // leu32p(hdr->SCP.Section11+4);
			if (TagNLen[tag]) {
				hdr->HeadLen += 4+TagNLen[tag];
			}
		}
#endif

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
	     	/* end */

		if (hdr->TYPE==GDF) {
			if (0.0 < hdr->VERSION && hdr->VERSION < 1.9) {
				hdr->VERSION = 1.25;
			}
			else if (hdr->VERSION < 3.0) {
				// this is currently still the default version
#if (BIOSIG_VERSION >= 10500)
				hdr->VERSION = 2.51;
#else
				hdr->VERSION = 2.22;
#endif
			}
			else {
				hdr->VERSION = 3.0;
			}

			// in case of GDF v2, make HeadLen a multiple of 256.
			if ((hdr->VERSION > 2.0) && (hdr->HeadLen & 0x00ff))
				hdr->HeadLen = (hdr->HeadLen & 0xff00) + 256;
		}
		else if (hdr->TYPE==GDF1) {
			fprintf(stderr,"libbiosig@sopen(hdr,\"w\") with hdr->TYPE=GDF1 is deprecated. Use hdr->TYPE=GDF and hdr->VERSION=1.25 instead\n");
			hdr->VERSION = 1.25;
			hdr->TYPE = GDF;
		}


		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, 1, hdr->HeadLen, TagNLen[1]);

		if (hdr->SCP.Section7 || hdr->SCP.Section8 || hdr->SCP.Section9 || hdr->SCP.Section10 || hdr->SCP.Section11) {
			// use auxillary pointer in order to keep SCP sections in memory
			if (hdr->aECG) free(hdr->aECG);
			hdr->aECG = hdr->AS.Header;
			hdr->AS.Header = (uint8_t*) realloc(NULL, hdr->HeadLen);
		}
		else {
			hdr->AS.Header = (uint8_t*) realloc(hdr->AS.Header, hdr->HeadLen);
		}
	    	if (hdr->AS.Header == NULL) {
	    		biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Memory allocation failed");
			return;
	    	}
		memset(Header1, 0, 256*(1+hdr->NS));
	     	sprintf((char*)hdr->AS.Header,"GDF %4.2f",hdr->VERSION);
	    	uint8_t* Header2 = hdr->AS.Header+256;

		uint16_t maxlen = 66;
		if (hdr->VERSION < 1.90) maxlen = 80;
		size_t l1 = (hdr->Patient.Id==NULL) ? 0 : strlen(hdr->Patient.Id);
		size_t l2 = (hdr->Patient.Name==NULL) ? 0 : strlen(hdr->Patient.Name);
		if (0 < l1 && l1 < maxlen) {
			for (k=0; hdr->Patient.Id[k]; k++)
				if (isspace(hdr->Patient.Id[k]))
					hdr->Patient.Id[k] = '_';

	     		strncpy(Header1+8, hdr->Patient.Id, l1+1);
		}
		else {
		     	strncpy(Header1+8, "X X",4);
			l1 = 1;
		}

		if (!hdr->FLAG.ANONYMOUS && (0 < l2) && (l1+l2+1 < maxlen) ) {
		     	Header1[8+l1] = ' ';
		     	strcpy(Header1+8+1+l1, hdr->Patient.Name);	/* Flawfinder: ignore *** length is already checked with l1+l2+1 */
		}
		else if (l1+3 < maxlen)
		     	strcpy(Header1+8+l1, " X");

		if (hdr->VERSION>1.90) {
	     		Header1[84] = (hdr->Patient.Smoking%4) + ((hdr->Patient.AlcoholAbuse%4)<<2) + ((hdr->Patient.DrugAbuse%4)<<4) + ((hdr->Patient.Medication%4)<<6);
	     		Header1[85] =  hdr->Patient.Weight;
	     		Header1[86] =  hdr->Patient.Height;
	     		Header1[87] = (hdr->Patient.Sex%4) + ((hdr->Patient.Handedness%4)<<2) + ((hdr->Patient.Impairment.Visual%4)<<4) + ((hdr->Patient.Impairment.Heart%4)<<6);
		}

	     	size_t len = strlen(hdr->ID.Recording);
	     	memcpy(Header1+88, hdr->ID.Recording, min(len,80));
                Header1[88 + min(len,80)] = 0;
		if (hdr->VERSION>1.90) {
			memcpy(Header1+152, &hdr->LOC, 16);
#if __BYTE_ORDER == __BIG_ENDIAN
			*(uint32_t*) (Header1+152) = htole32( *(uint32_t*) (Header1+152) );
			*(uint32_t*) (Header1+156) = htole32( *(uint32_t*) (Header1+156) );
			*(uint32_t*) (Header1+160) = htole32( *(uint32_t*) (Header1+160) );
			*(uint32_t*) (Header1+164) = htole32( *(uint32_t*) (Header1+164) );
#endif
		}

		if (hdr->VERSION<1.90) {
    			struct tm *t = gdf_time2tm_time(hdr->T0);

			sprintf(tmp,"%04i%02i%02i%02i%02i%02i00",t->tm_year+1900,t->tm_mon+1,t->tm_mday,t->tm_hour,t->tm_min,t->tm_sec);
			memcpy(hdr->AS.Header+168,tmp,max(strlen(tmp),16));
			leu32a(hdr->HeadLen, hdr->AS.Header+184);

			memcpy(Header1+192, &hdr->ID.Equipment, 8);
			// FIXME: 200: LabId, 208 TechId, 216, Serial No //
		}
		else {
			//memcpy(Header1+168, &hdr->T0, 8);
			leu64a(hdr->T0, hdr->AS.Header+168);
			//memcpy(Header1+176, &hdr->Patient.Birthday, 8);
			leu64a(hdr->Patient.Birthday, hdr->AS.Header+176);
			// *(uint16_t*)(Header1+184) = (hdr->HeadLen>>8)+(hdr->HeadLen%256>0);
			leu32a(hdr->HeadLen>>8, hdr->AS.Header+184);

			memcpy(hdr->AS.Header+192, &hdr->ID.Equipment, 8);
			memcpy(hdr->AS.Header+200, &hdr->IPaddr, 6);
			memcpy(hdr->AS.Header+206, &hdr->Patient.Headsize, 6);
			lef32a(hdr->ELEC.REF[0], hdr->AS.Header+212);
			lef32a(hdr->ELEC.REF[1], hdr->AS.Header+216);
			lef32a(hdr->ELEC.REF[2], hdr->AS.Header+220);
			lef32a(hdr->ELEC.GND[0], hdr->AS.Header+224);
			lef32a(hdr->ELEC.GND[1], hdr->AS.Header+228);
			lef32a(hdr->ELEC.GND[2], hdr->AS.Header+232);
		}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %x\n", __func__, __LINE__, hdr->HeadLen,leu32p(hdr->AS.Header+184));


		leu64a(hdr->NRec, hdr->AS.Header+236);

	/* FIXME: this part should make the records as small as possible
		size_t DIV = 1, div;
		for (k=0; k<hdr->NS; k++) {
			div = hdr->SPR/hdr->CHANNEL[k].SPR;
			if (div>DIV) DIV=div;
		}
		for (k=0; k<hdr->NS; k++) {
			hdr->CHANNEL[k].SPR = (hdr->CHANNEL[k].SPR*DIV)/hdr->SPR;
		}
		hdr->NRec *= hdr->SPR/DIV;
		hdr->SPR  = DIV;
	*/

		double fDur = hdr->SPR/hdr->SampleRate;
		if (hdr->NS==0 && 0.0 < hdr->EVENT.SampleRate && hdr->EVENT.SampleRate < INFINITY)
			fDur = 1.0 / hdr->EVENT.SampleRate;

		if (hdr->VERSION < 2.21) {
			/* Duration is expressed as an fraction of integers */
			double dtmp1, dtmp2;
			dtmp2 = modf(fDur, &dtmp1);
			// approximate real with rational number
			if (fabs(dtmp2) < DBL_EPSILON) {
				Dur[0] = lround(fDur);
				Dur[1] = 1;
			}
			else {
				Dur[1] = lround(1.0 / dtmp2 );
				Dur[0] = lround(1.0 + dtmp1 * Dur[1]);
			}

			leu32a(Dur[0], hdr->AS.Header+244);
			leu32a(Dur[1], hdr->AS.Header+248);
		}
		else
			lef64a(fDur, hdr->AS.Header+244);

		leu16a(NS, hdr->AS.Header + 252);
		if (hdr->VERSION > 2.4) {
			lei16a(hdr->tzmin, hdr->AS.Header+254);
		}

	     	/* define HDR.Header2
	     	this requires checking the arguments in the fields of the struct HDR.CHANNEL
	     	and filling in the bytes in HDR.Header2.
	     	*/
		typeof(k) k2=0;
		for (k=0; k<hdr->NS; k++)
		if (hdr->CHANNEL[k].OnOff)
		{
			const char *tmpstr;
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;

			if ( (0 < hc->LeadIdCode) && (hc->LeadIdCode * sizeof(&LEAD_ID_TABLE[0]) < sizeof(LEAD_ID_TABLE) ) )
				tmpstr = LEAD_ID_TABLE[hc->LeadIdCode];
			else
				tmpstr = hc->Label;

			len = strlen(tmpstr)+1;
		     	memcpy(Header2+16*k2,tmpstr,min(len,16));
                        Header2[16*k2+min(len,16)] = 0;

		     	len = strlen(hdr->CHANNEL[k].Transducer);
		     	memcpy(Header2+80*k2 + 16*NS, hdr->CHANNEL[k].Transducer, min(len,80));
			Header2[80*k2 + min(len,80) + 16*NS] = 0;

			tmpstr = PhysDim3(hdr->CHANNEL[k].PhysDimCode);
			len = strlen(tmpstr)+1;
		     	if (hdr->VERSION < 1.9)
		     		memcpy(Header2+ 8*k2 + 96*NS, tmpstr, min(8,len));
		     	else {
		     		memcpy(Header2+ 6*k2 + 96*NS, tmpstr, min(6,len));
				leu16a(hdr->CHANNEL[k].PhysDimCode, Header2+ 2*k2 + 102*NS);
			};

			lef64a(hdr->CHANNEL[k].PhysMin, Header2 + 8*k2 + 104*NS);
		     	lef64a(hdr->CHANNEL[k].PhysMax, Header2 + 8*k2 + 112*NS);
		     	if (hdr->VERSION < 1.9) {
				lei64a((int64_t)hdr->CHANNEL[k].DigMin, Header2 + 8*k2 + 120*NS);
				lei64a((int64_t)hdr->CHANNEL[k].DigMax, Header2 + 8*k2 + 128*NS);
			     	// FIXME // memcpy(Header2 + 80*k + 136*hdr->NS,hdr->CHANNEL[k].PreFilt,max(80,strlen(hdr->CHANNEL[k].PreFilt)));
			}
			else {
				lef64a(hdr->CHANNEL[k].DigMin, Header2 + 8*k2 + 120*NS);
				lef64a(hdr->CHANNEL[k].DigMax, Header2 + 8*k2 + 128*NS);
			     	if (hdr->VERSION >= 2.22) lef32a(hdr->CHANNEL[k].TOffset, Header2 + 4*k2 + 200*NS);	// GDF222
				lef32a(hdr->CHANNEL[k].LowPass, Header2 + 4*k2 + 204*NS);
				lef32a(hdr->CHANNEL[k].HighPass, Header2 + 4*k2 + 208*NS);
				lef32a(hdr->CHANNEL[k].Notch, Header2 + 4*k2 + 212*NS);

				lef32a(hdr->CHANNEL[k].XYZ[0], Header2 + 4*k2 + 224*NS);
				lef32a(hdr->CHANNEL[k].XYZ[1], Header2 + 4*k2 + 228*NS);
				lef32a(hdr->CHANNEL[k].XYZ[2], Header2 + 4*k2 + 232*NS);

        		     	if (hdr->VERSION < (float)2.19)
       	     				Header2[k2+236*NS] = (uint8_t)ceil(log10(min(39e8,hdr->CHANNEL[k].Impedance))/log10(2.0)*8.0-0.5);

        		     	else switch (hdr->CHANNEL[k].PhysDimCode & 0xFFE0) {
        		     	        // context-specific header 2 area
        		     	        case 4256:
						lef32a((float)hdr->CHANNEL[k].Impedance, Header2+236*NS+20*k2);
        	     				break;
        		     	        case 4288:
						lef32a((float)hdr->CHANNEL[k].fZ, Header2+236*NS+20*k2);
        	     				break;
        	     			// default:        // reserved area
                                }
		     	}
			leu32a(hdr->CHANNEL[k].SPR, Header2 + 4*k2 + 216*NS);
			leu32a(hdr->CHANNEL[k].GDFTYP, Header2 + 4*k2 + 220*NS);
		     	k2++;
		}

		if (errno==34) errno = 0; // reset numerical overflow error

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d)  %d %s\n", __func__, __LINE__, errno, strerror(errno));

		/*****
		 *	This is the 2nd scan of Header3 - memory is allocated, now H3 is filled in with content
		 *****/
	    	Header2 = hdr->AS.Header+(NS+1)*256;
	    	tag = 1;
	     	if (TagNLen[tag]>0) {
			leu32a(tag + (TagNLen[tag]<<8), Header2); // Tag=1 & Length of Tag 1
	     		size_t pos = 4;
	     		for (k=0; k<hdr->EVENT.LenCodeDesc; k++) {
	     			strcpy((char*)(Header2+pos),hdr->EVENT.CodeDesc[k]);     /* Flawfinder: ignore *** memory is allocated after 1st H3 scan above */
		     		pos += strlen(hdr->EVENT.CodeDesc[k])+1;
		     	}
		     	Header2[pos]=0; 	// terminating NULL
	     		Header2 += pos+1;
	     	}
		tag = 2;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);

	     	if (TagNLen[tag]>0) {
			leu32a(tag + (TagNLen[tag]<<8), Header2); // Tag=2 & Length of Tag 2
     			strcpy((char*)(Header2+4),hdr->AS.bci2000);			/* Flawfinder: ignore *** memory is allocated after 1st H3 scan above */
			Header2 += 4+TagNLen[tag];
	     	}
	     	tag = 3;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);

	     	if (TagNLen[tag]>0) {
			leu32a(tag + (TagNLen[tag]<<8), Header2); // Tag=3 & Length of Tag 3
			if (VERBOSE_LEVEL>8) fprintf(stdout,"SOPEN(GDF)w: tag=%i,len=%i\n",tag,TagNLen[tag]);
	     		memset(Header2+4,0,TagNLen[tag]);
	     		size_t len = 0;

     			strcpy((char*)(Header2+4), hdr->ID.Manufacturer.Name);		/* Flawfinder: ignore *** memory is allocated after 1st H3 scan above */
		     	if (hdr->ID.Manufacturer.Name != NULL)
		     		len += strlen(hdr->ID.Manufacturer.Name);

     			strcpy((char*)(Header2+5+len), hdr->ID.Manufacturer.Model);	/* Flawfinder: ignore *** memory is allocated after 1st H3 scan above */
		     	if (hdr->ID.Manufacturer.Model != NULL)
		     		len += strlen(hdr->ID.Manufacturer.Model);

     			strcpy((char*)(Header2+6+len), hdr->ID.Manufacturer.Version);	/* Flawfinder: ignore *** memory is allocated after 1st H3 scan above */
		     	if (hdr->ID.Manufacturer.Version != NULL)
		     		len += strlen(hdr->ID.Manufacturer.Version);

     			strcpy((char*)(Header2+7+len), hdr->ID.Manufacturer.SerialNumber);	/* Flawfinder: ignore *** memory is allocated after 1st H3 scan above */
		     	if (hdr->ID.Manufacturer.SerialNumber != NULL)
		     		len += strlen(hdr->ID.Manufacturer.SerialNumber);
			Header2 += 4+TagNLen[tag];

	     	}
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
/*
	     	tag = 4;
	     	if (TagNLen[tag]>0) {

			*(uint32_t*)(Header2) = htole32(tag + (TagNLen[tag]<<8)); // Tag=4 & Length of Tag 4
	     		Header2 += 4;
	     		for (k=0; k<hdr->NS; k++) {
				*(uint32_t*)(Header2 + 4*k)             = le32toh(*(uint32_t*)(hdr->CHANNEL[k].Orientation+0));
				*(uint32_t*)(Header2 + 4*k + 4*hdr->NS) = le32toh(*(uint32_t*)(hdr->CHANNEL[k].Orientation+1));
				*(uint32_t*)(Header2 + 4*k + 8*hdr->NS) = le32toh(*(uint32_t*)(hdr->CHANNEL[k].Orientation+2));
				*(uint32_t*)(Header2 + 4*k +12*hdr->NS) = le32toh(*(uint32_t*)(&(hdr->CHANNEL[k].Area));
	     		}
     			Header2 += 4*sizeof(float)*hdr->NS;
	     	}
*/
	     	tag = 5;
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
	     	if (TagNLen[tag]>0) {
			leu32a(tag + (TagNLen[tag]<<8), Header2); // Tag=5 & Length of Tag 5
     			memcpy(Header2+4,hdr->IPaddr,TagNLen[tag]);
			Header2 += 4+TagNLen[tag];
	     	}

	     	tag = 6;
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
	     	if (TagNLen[tag]>0) {
			leu32a(tag + (TagNLen[tag]<<8), Header2); // Tag=6 & Length of Tag 6
     			strcpy((char*)(Header2+4),hdr->ID.Technician);		/* Flawfinder: ignore *** memory is allocated after 1st H3 scan above */
			Header2 += 4+TagNLen[tag];
	     	}

	     	tag = 7;
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
	     	if (TagNLen[tag]>0) {
			leu32a(tag + (TagNLen[tag]<<8), Header2); // Tag=7 & Length of Tag 7
     			strcpy((char*)(Header2+4),hdr->ID.Hospital);		/* Flawfinder: ignore *** memory is allocated after 1st H3 scan above */
			Header2 += 4+TagNLen[tag];
	     	}

#if (BIOSIG_VERSION >= 10500)
		tag = 9;
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
		if (TagNLen[tag]>0) {
			leu32a(tag + (TagNLen[tag]<<8), Header2); 	// Tag=9 & Length of Tag 9
			memcpy((char*)(Header2+4),hdr->SCP.Section7, TagNLen[tag]);		/* Flawfinder: ignore *** memory is allocated after 1st H3 scan above */
			hdr->SCP.Section7 = Header2+4;
			Header2 += 4+TagNLen[tag];
		}
		tag = 10;
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
		if (TagNLen[tag]>0) {
			leu32a(tag + (TagNLen[tag]<<8), Header2); 	// Tag=10 & Length of Tag 10
			memcpy((char*)(Header2+4),hdr->SCP.Section8, TagNLen[tag]);		/* Flawfinder: ignore *** memory is allocated after 1st H3 scan above */
			hdr->SCP.Section8 = Header2+4;
			Header2 += 4+TagNLen[tag];
		}
		tag = 11;
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
		if (TagNLen[tag]>0) {
			leu32a(tag + (TagNLen[tag]<<8), Header2); 	// Tag=11 & Length of Tag 11
			memcpy((char*)(Header2+4),hdr->SCP.Section9, TagNLen[tag]);		/* Flawfinder: ignore *** memory is allocated after 1st H3 scan above */
			hdr->SCP.Section9 = Header2+4;
			Header2 += 4+TagNLen[tag];
		}
		tag = 12;
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
		if (TagNLen[tag]>0) {
			leu32a(tag + (TagNLen[tag]<<8), Header2); 	// Tag=12 & Length of Tag 12
			memcpy((char*)(Header2+4),hdr->SCP.Section10, TagNLen[tag]);		/* Flawfinder: ignore *** memory is allocated after 1st H3 scan above */
			hdr->SCP.Section10 = Header2+4;
			Header2 += 4+TagNLen[tag];
		}
		tag = 13;
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i %i %i\n", __func__, __LINE__, tag, hdr->HeadLen, TagNLen[tag]);
		if (TagNLen[tag]>0) {
			leu32a(tag + (TagNLen[tag]<<8), Header2); 	// Tag=13 & Length of Tag 13
			memcpy((char*)(Header2+4),hdr->SCP.Section11, TagNLen[tag]);		/* Flawfinder: ignore *** memory is allocated after 1st H3 scan above */
			hdr->SCP.Section11 = Header2+4;
			Header2 += 4+TagNLen[tag];
		}
#endif

		while (Header2 < (hdr->AS.Header + hdr->HeadLen) ) {
			*Header2 = 0;
			 Header2++;
		}
		if (hdr->aECG) {
			free(hdr->aECG);
			hdr->aECG=NULL;
		}

		if (VERBOSE_LEVEL>8) fprintf(stdout,"GDFw [339] %p %p\n", Header1,Header2);
}

/****************************************************************************
       gdfbin2struct
	converts flat file into hdr structure
 ****************************************************************************/
int gdfbin2struct(HDRTYPE *hdr)
{
    	unsigned int 	k;
    	char 		tmp[81];
    	double 		Dur;
//	char*		ptr_str;
	struct tm 	tm_time;
//	time_t		tt;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %p\n",__func__,__LINE__,hdr->AS.Header);

		if (!memcmp("GDF",(char*)(hdr->AS.Header+3),3)) {
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Only GDF is supported");
			return (hdr->AS.B4C_ERRNUM);
		}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) #%i \n",__func__,__LINE__, (int)hdr->NS);

      	    	strncpy(tmp,(char*)(hdr->AS.Header+3),5); tmp[5]=0;
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) #%i  Ver=<%s>\n",__func__,__LINE__, (int)hdr->NS,tmp);
	    	hdr->VERSION 	= atof(tmp);
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) #%i  Ver=<%s>\n",__func__,__LINE__, (int)hdr->NS,tmp);

	    	hdr->NRec 	= lei64p(hdr->AS.Header+236);
	    	hdr->NS   	= leu16p(hdr->AS.Header+252);

		hdr->tzmin = (hdr->VERSION > 2.4) ? lei16p(hdr->AS.Header+254) : 0;

		if (hdr->VERSION < 2.21)
			Dur = (double)leu32p(hdr->AS.Header+244)/(double)leu32p(hdr->AS.Header+248);
		else
			Dur = lef64p(hdr->AS.Header+244);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) #%i  Ver=%g\n",__func__,__LINE__, hdr->NS, hdr->VERSION);

	    	if (hdr->VERSION > 1.90) {

		    	hdr->HeadLen 	= leu16p(hdr->AS.Header+184)<<8;
		    	int len = min(66,MAX_LENGTH_PID);
	    		strncpy(hdr->Patient.Id,(const char*)hdr->AS.Header+8,len);
	    		hdr->Patient.Id[len]=0;
	    		len = min(64,MAX_LENGTH_RID);
	    		strncpy(hdr->ID.Recording,(const char*)hdr->AS.Header+88,len);
	    		hdr->ID.Recording[len]=0;
	    		strtok(hdr->Patient.Id," ");
	    		char *tmpptr = strtok(NULL," ");
	    		if ((!hdr->FLAG.ANONYMOUS) && (tmpptr != NULL)) {
				// strncpy(hdr->Patient.Name,tmpptr,Header1+8-tmpptr);
		    		strncpy(hdr->Patient.Name,tmpptr,MAX_LENGTH_NAME);
		    	}

			if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) FMT=%s Ver=%4.2f\n",__func__,__LINE__, GetFileTypeString(hdr->TYPE), hdr->VERSION);

	    		hdr->Patient.Smoking      =  Header1[84]%4;
	    		hdr->Patient.AlcoholAbuse = (Header1[84]>>2)%4;
	    		hdr->Patient.DrugAbuse    = (Header1[84]>>4)%4;
	    		hdr->Patient.Medication   = (Header1[84]>>6)%4;
	    		hdr->Patient.Weight       =  Header1[85];
	    		hdr->Patient.Height       =  Header1[86];
	    		hdr->Patient.Sex       	  =  Header1[87]%4;
	    		hdr->Patient.Handedness   = (Header1[87]>>2)%4;
	    		hdr->Patient.Impairment.Visual = (Header1[87]>>4)%4;
	    		hdr->Patient.Impairment.Heart  = (Header1[87]>>6)%4;

#if __BYTE_ORDER == __BIG_ENDIAN
			*(uint32_t*)(hdr->AS.Header+156) = bswap_32(*(uint32_t*)(hdr->AS.Header+156));
			*(uint32_t*)(hdr->AS.Header+160) = bswap_32(*(uint32_t*)(hdr->AS.Header+160));
			*(uint32_t*)(hdr->AS.Header+164) = bswap_32(*(uint32_t*)(hdr->AS.Header+164));
#endif
			if (hdr->AS.Header[156]) {
				hdr->LOC[0] = 0x00292929;
				memcpy(&hdr->LOC[1], hdr->AS.Header+156, 12);
			}
			else {
#if __BYTE_ORDER == __BIG_ENDIAN
				*(uint32_t*) (hdr->AS.Header+152) = bswap_32(*(uint32_t*)(hdr->AS.Header+152));
#endif
				memcpy(&hdr->LOC, hdr->AS.Header+152, 16);
			}

			if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) FMT=%s Ver=%4.2f\n",__func__,__LINE__, GetFileTypeString(hdr->TYPE), hdr->VERSION);

			hdr->T0 		= lei64p(hdr->AS.Header+168);
			hdr->Patient.Birthday 	= lei64p(hdr->AS.Header+176);
			// memcpy(&hdr->T0, Header1+168,8);
			// memcpy(&hdr->Patient.Birthday, Header1+176, 8);

			hdr->ID.Equipment 	= lei64p(hdr->AS.Header+192);
			if (hdr->VERSION < (float)2.10) memcpy(hdr->IPaddr, Header1+200,4);
			hdr->Patient.Headsize[0]= leu16p(hdr->AS.Header+206);
			hdr->Patient.Headsize[1]= leu16p(hdr->AS.Header+208);
			hdr->Patient.Headsize[2]= leu16p(hdr->AS.Header+210);

			//memcpy(&hdr->ELEC.REF, Header1+212,12);
			//memcpy(&hdr->ELEC.GND, Header1+224,12);
			hdr->ELEC.REF[0]   = lef32p(hdr->AS.Header+212);
			hdr->ELEC.REF[1]   = lef32p(hdr->AS.Header+216);
			hdr->ELEC.REF[2]   = lef32p(hdr->AS.Header+220);
			hdr->ELEC.GND[0]   = lef32p(hdr->AS.Header+224);
			hdr->ELEC.GND[1]   = lef32p(hdr->AS.Header+228);
			hdr->ELEC.GND[2]   = lef32p(hdr->AS.Header+232);
		    	if (hdr->VERSION > 100000.0) {
		    		fprintf(stdout,"%e \nb4c %c %i %c. %c%c%c%c%c%c%c\n",hdr->VERSION,169,2007,65,83,99,104,108,246,103,108);
		    		FILE *fid = fopen("/tmp/b4c_tmp","wb");
		    		if (fid != NULL) {
			    		fprintf(fid,"\nb4c %f \n%c %i %c.%c%c%c%c%c%c%c\n",hdr->VERSION,169,2007,65,83,99,104,108,246,103,108);
			    		fclose(fid);
			    	}
		    	}

			if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) FMT=%s Ver=%4.2f\n",__func__,__LINE__, GetFileTypeString(hdr->TYPE), hdr->VERSION);

	    	}
	    	else if (hdr->VERSION > 0.0) {
	    		strncpy(hdr->Patient.Id,Header1+8,min(80,MAX_LENGTH_PID));
	    		hdr->Patient.Id[min(80,MAX_LENGTH_PID)] = 0;
			strncpy(hdr->ID.Recording,(const char*)Header1+88,min(80,MAX_LENGTH_RID));
	    		hdr->ID.Recording[min(80,MAX_LENGTH_RID)] = 0;
	    		strtok(hdr->Patient.Id," ");
	    		char *tmpptr = strtok(NULL," ");
	    		if ((!hdr->FLAG.ANONYMOUS) && (tmpptr != NULL)) {
				// strncpy(hdr->Patient.Name,tmpptr,Header1+8-tmpptr);
		    		strncpy(hdr->Patient.Name,tmpptr,MAX_LENGTH_NAME);
		    	}

			memset(tmp,0,5);
			strncpy(tmp,Header1+168+12,2);
	    		tm_time.tm_sec  = atoi(tmp);
			strncpy(tmp,Header1+168+10,2);
	    		tm_time.tm_min  = atoi(tmp);
			strncpy(tmp,Header1+168+ 8,2);
	    		tm_time.tm_hour = atoi(tmp);
			strncpy(tmp,Header1+168+ 6,2);
	    		tm_time.tm_mday = atoi(tmp);
			strncpy(tmp,Header1+168+ 4,2);
	    		tm_time.tm_mon  = atoi(tmp)-1;
			strncpy(tmp,Header1+168   ,4);
	    		tm_time.tm_year = atoi(tmp)-1900;
	    		tm_time.tm_isdst= -1;

			hdr->T0 = tm_time2gdf_time(&tm_time);
		    	hdr->HeadLen 	= leu64p(hdr->AS.Header+184);
	    	}
	    	else {
    			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Error SOPEN(GDF); invalid version number.");
	    		return (hdr->AS.B4C_ERRNUM);
	    	}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) #%i Ver=%4.2f\n",__func__,__LINE__, hdr->NS, hdr->VERSION);

		if (hdr->HeadLen < (256u * (hdr->NS + 1u))) {
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "(GDF) Length of Header is too small");
			return (hdr->AS.B4C_ERRNUM);
		}

		hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		uint8_t *Header2 = hdr->AS.Header+256;

		hdr->AS.bpb=0;
		size_t bpb8 = 0;
		for (k=0; k<hdr->NS; k++)	{
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;

			if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) #%i/%i\n",__func__,__LINE__, k, hdr->NS);

			hc->LeadIdCode = 0;
			size_t len = min(16, MAX_LENGTH_LABEL);
			strncpy(hc->Label,(char*)Header2 + 16*k, len);
			hc->Label[len] = 0;

if (VERBOSE_LEVEL>7) fprintf(stdout,"#%2i: <%s> %i %i\n",k,hc->Label,(int)len,(int)strlen(hc->Label));

			len = min(MAX_LENGTH_TRANSDUCER, 80);
			memcpy(hc->Transducer, (char*)Header2 + 16*hdr->NS + 80*k, len);
			hc->Transducer[len] = 0;

			if (VERBOSE_LEVEL>7) fprintf(stdout,"[GDF 212] #=%i/%i %s\n",k,hdr->NS,hc->Label);

			hc->PhysMin = lef64p(Header2+ 8*k + 104*hdr->NS);
			hc->PhysMax = lef64p(Header2+ 8*k + 112*hdr->NS);

			hc->SPR     = leu32p(Header2+ 4*k + 216*hdr->NS);
			hc->GDFTYP  = leu16p(Header2+ 4*k + 220*hdr->NS);
			hc->OnOff   = 1;
			hc->bi      = bpb8>>3;
			hc->bi8     = bpb8;
			size_t nbits = (GDFTYP_BITS[hc->GDFTYP]*(size_t)hc->SPR);
			bpb8 += nbits;

			if (hdr->VERSION < 1.90) {
				char p[9];
				strncpy(p, (char*)Header2 + 8*k + 96*hdr->NS,8);
				p[8] = 0; // remove trailing blanks
				int k1;
				for (k1=7; (k1>0) && isspace(p[k1]); p[k1--] = 0) {};

				hc->PhysDimCode = PhysDimCode(p);

				hc->DigMin   = (double) lei64p(Header2 + 8*k + 120*hdr->NS);
				hc->DigMax   = (double) lei64p(Header2 + 8*k + 128*hdr->NS);

				char *PreFilt  = (char*)(Header2+ 68*k + 136*hdr->NS);
				hc->LowPass  = NAN;
				hc->HighPass = NAN;
				hc->Notch    = NAN;
				hc->TOffset  = NAN;
				float lf,hf;
				if (sscanf(PreFilt,"%f - %f Hz",&lf,&hf)==2) {
					hc->LowPass  = hf;
					hc->HighPass = lf;
				}
			}
			else {
				hc->PhysDimCode = leu16p(Header2+ 2*k + 102*hdr->NS);

				hc->DigMin   = lef64p(Header2+ 8*k + 120*hdr->NS);
				hc->DigMax   = lef64p(Header2+ 8*k + 128*hdr->NS);

				hc->LowPass  = lef32p(Header2+ 4*k + 204*hdr->NS);
				hc->HighPass = lef32p(Header2+ 4*k + 208*hdr->NS);
				hc->Notch    = lef32p(Header2+ 4*k + 212*hdr->NS);
				hc->XYZ[0]   = lef32p(Header2+ 4*k + 224*hdr->NS);
				hc->XYZ[1]   = lef32p(Header2+ 4*k + 228*hdr->NS);
				hc->XYZ[2]   = lef32p(Header2+ 4*k + 232*hdr->NS);
				// memcpy(&hc->XYZ,Header2 + 4*k + 224*hdr->NS,12);
				hc->Impedance= ldexp(1.0, (uint8_t)Header2[k + 236*hdr->NS]/8);

			     	if (hdr->VERSION < 2.22)
					hc->TOffset  = NAN;
			     	else
			     		hc->TOffset  = lef32p(Header2 + 4 * k + 200 * hdr->NS);

        		     	if (hdr->VERSION < (float)2.19)
        				hc->Impedance = ldexp(1.0, (uint8_t)Header2[k + 236*hdr->NS]/8);
        		     	else switch(hdr->CHANNEL[k].PhysDimCode & 0xFFE0) {
        		     	        // context-specific header 2 area
        		     	        case 4256:
                				hc->Impedance = *(float*)(Header2+236*hdr->NS+20*k);
        	     				break;
        		     	        case 4288:
        		     	                hc->fZ = *(float*)(Header2+236*hdr->NS+20*k);
        	     				break;
        	     			// default:        // reserved area
                                }
			}
			hc->Cal = (hc->PhysMax-hc->PhysMin)/(hc->DigMax-hc->DigMin);
			hc->Off =  hc->PhysMin-hc->Cal*hc->DigMin;
		}
		hdr->AS.bpb = bpb8>>3;
		if (bpb8 & 0x07) {		// each block must use whole number of bytes
			hdr->AS.bpb++;
			hdr->AS.bpb8 = hdr->AS.bpb<<3;
		}

		if (VERBOSE_LEVEL>8) fprintf(stdout,"[213] FMT=%s Ver=%4.2f\n",GetFileTypeString(hdr->TYPE),hdr->VERSION);

		for (k=0, hdr->SPR=1; k<hdr->NS;k++) {

			if (VERBOSE_LEVEL>8) fprintf(stdout,"[GDF 214] #=%i\n",k);

			if (hdr->CHANNEL[k].SPR)
				hdr->SPR = lcm(hdr->SPR,hdr->CHANNEL[k].SPR);

			if (GDFTYP_BITS[hdr->CHANNEL[k].GDFTYP]==0) {
				biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "GDF: Invalid or unsupported GDFTYP");
				return(hdr->AS.B4C_ERRNUM);
			}
		}
		hdr->SampleRate = ((double)(hdr->SPR))/Dur;

		if (VERBOSE_LEVEL>8) fprintf(stdout,"[219] FMT=%s Ver=%4.2f\n",GetFileTypeString(hdr->TYPE),hdr->VERSION);

		/* read GDF Header 3 - experimental */
		if ((hdr->HeadLen > 256u*(hdr->NS+1u)) && (hdr->VERSION>=(float)2.10)) {
		    	uint8_t *Header2 = hdr->AS.Header + 256*(hdr->NS+1);
		    	uint8_t tag = 0xff;
		    	size_t pos=0,len=0;
	    		tag = (uint8_t)Header2[0];

			if (VERBOSE_LEVEL>8) fprintf(stdout,"[220] GDFr3: %i %i Tag=%i\n",hdr->HeadLen,hdr->NS,tag);

		    	while ((pos < (hdr->HeadLen-256*(hdr->NS+1)-4)) && (tag>0)) {
		    		len = leu32p(Header2+pos)>>8;

    				if (VERBOSE_LEVEL>8) fprintf(stdout,"GDFr3: Tag=%i Len=%i pos=%i\n",tag,(int)len,(int)pos);

		    		if (0) {}
		    		else if (tag==1) {
		    			// user-specific events i.e. free text annotations
if (VERBOSE_LEVEL>6) fprintf(stdout,"user-specific events defined\n");
					hdr->AS.auxBUF = (uint8_t*) realloc(hdr->AS.auxBUF,len);
					memcpy(hdr->AS.auxBUF, Header2+pos+4, len);
					hdr->EVENT.CodeDesc = (typeof(hdr->EVENT.CodeDesc)) realloc(hdr->EVENT.CodeDesc,257*sizeof(*hdr->EVENT.CodeDesc));
					hdr->EVENT.CodeDesc[0] = "";	// typ==0, is always empty
					hdr->EVENT.LenCodeDesc = 1;
					k = 1;
					while (hdr->AS.auxBUF[k]) {
						hdr->EVENT.CodeDesc[hdr->EVENT.LenCodeDesc++] = (char*)(hdr->AS.auxBUF+k);
						k += strlen((char*)(hdr->AS.auxBUF+k))+1;
					}
		    		}
		    		else if (tag==2) {
		    			/* BCI 2000 information */
		    			hdr->AS.bci2000 = (char*) realloc(hdr->AS.bci2000,len+1);
		    			memcpy(hdr->AS.bci2000,Header2+pos+4,len);
		    			hdr->AS.bci2000[len]=0;
		    		}
		    		else if (tag==3) {
		    			/* manufacture information */
		    			if (len > MAX_LENGTH_MANUF) {
		    				fprintf(stderr,"Warning: length of Manufacturer information (%i) exceeds length of %i bytes\n", (int)len, MAX_LENGTH_MANUF);
		    				len = MAX_LENGTH_MANUF;
		    			}
		    			memcpy(hdr->ID.Manufacturer._field,Header2+pos+4,len);
		    			hdr->ID.Manufacturer._field[MAX_LENGTH_MANUF]=0;
		    			hdr->ID.Manufacturer.Name = hdr->ID.Manufacturer._field;
		    			hdr->ID.Manufacturer.Model= hdr->ID.Manufacturer.Name+strlen(hdr->ID.Manufacturer.Name)+1;
		    			hdr->ID.Manufacturer.Version = hdr->ID.Manufacturer.Model+strlen(hdr->ID.Manufacturer.Model)+1;
		    			hdr->ID.Manufacturer.SerialNumber = hdr->ID.Manufacturer.Version+strlen(hdr->ID.Manufacturer.Version)+1;
		    		}
		    		else if (0) {
		    			// (tag==4) {
		    			/* sensor orientation */
/*		    			// OBSOLETE
		    			for (k=0; k<hdr->NS; k++) {
		    				hdr->CHANNEL[k].Orientation[0] = lef32p(Header2+pos+4+4*k);
		    				hdr->CHANNEL[k].Orientation[1] = lef32p(Header2+pos+4+4*k+hdr->NS*4);
		    				hdr->CHANNEL[k].Orientation[2] = lef32p(Header2+pos+4+4*k+hdr->NS*8);
		    				// if (len >= 12*hdr->NS)
			    				hdr->CHANNEL[k].Area   = lef32p(Header2+pos+4+4*k+hdr->NS*12);
						if (VERBOSE_LEVEL>8)
							fprintf(stdout,"GDF tag=4 #%i pos=%i/%i: %f\n",k,pos,len,hdr->CHANNEL[k].Area);
		    			}
*/		    		}
		    		else if (tag==5) {
		    			/* IP address  */
		    			memcpy(hdr->IPaddr,Header2+pos+4,len);
		    		}
		    		else if (tag==6) {
		    			/* Technician  */
					hdr->ID.Technician = (char*)realloc(hdr->ID.Technician,len+1);
					memcpy(hdr->ID.Technician,Header2+pos+4, len);
					hdr->ID.Technician[len]=0;
		    		}
		    		else if (tag==7) {
		    			// recording institution
					// hdr->ID.Hospital = strndup((char*)(Header2+pos+4),len);
					hdr->ID.Hospital = malloc(len+1);
					if (hdr->ID.Hospital) {
						hdr->ID.Hospital[len] = 0;
						strncpy(hdr->ID.Hospital,(char*)Header2+pos+4,len);
					}
		    		}

#if (BIOSIG_VERSION >= 10500)
				else if (tag==9) {
					hdr->SCP.Section7 = Header2+pos+4;
					hdr->SCP.Section7Length = len;
				}
				else if (tag==10) {
					hdr->SCP.Section8 = Header2+pos+4;
					hdr->SCP.Section8Length = len;
				}
				else if (tag==11) {
					hdr->SCP.Section9 = Header2+pos+4;
					hdr->SCP.Section9Length = len;
				}
				else if (tag==12) {
					hdr->SCP.Section10 = Header2+pos+4;
					hdr->SCP.Section10Length = len;
				}
				else if (tag==13) {
					hdr->SCP.Section11 = Header2+pos+4;
					hdr->SCP.Section11Length = len;
				}
#endif

		    		/* further tags may include
		    		- Manufacturer: SCP, MFER, GDF1
		    		- Orientation of MEG channels
		    		- Study ID
		    		- BCI: session, run
		    		*/

		    		pos+= 4+len;
		    		tag = (uint8_t)Header2[pos];

				if (VERBOSE_LEVEL>8) fprintf(stdout,"GDFr3: next Tag=%i pos=%i\n",tag,(int)pos);

		    	}
		}

		// if (VERBOSE_LEVEL>8) fprintf(stdout,"[GDF 217] #=%li\n",iftell(hdr));
		return(hdr->AS.B4C_ERRNUM);
}

/*********************************************************************************
	hdrEVT2rawEVT(HDRTYPE *hdr)
	converts structure HDR.EVENT into raw event data (hdr->AS.rawEventData)

	TODO: support of EVENT.TimeStamp
 *********************************************************************************/
size_t hdrEVT2rawEVT(HDRTYPE *hdr) {

	size_t k32u;
	char flag = (hdr->EVENT.DUR != NULL) && (hdr->EVENT.CHN != NULL) ? 3 : 1;
	if (flag==3)   // any DUR or CHN is larger than 0
		for (k32u=0, flag=1; k32u < hdr->EVENT.N; k32u++)
			if (hdr->EVENT.CHN[k32u] || hdr->EVENT.DUR[k32u]) {
				flag = 3;
				break;
			}

#if (BIOSIG_VERSION >= 10500)
	if (hdr->EVENT.TimeStamp != NULL) {
		flag = flag | 0x04;
	}
#endif

	int sze;
	sze  = (flag & 2) ? 12 : 6;
	sze += (flag & 4) ?  8 : 0;
	size_t len = 8+hdr->EVENT.N*sze;
	hdr->AS.rawEventData = (uint8_t*) realloc(hdr->AS.rawEventData,len);
	uint8_t *buf = hdr->AS.rawEventData;

	buf[0] = flag;
	if (hdr->VERSION < 1.94) {
		k32u   = lround(hdr->EVENT.SampleRate);
		buf[1] =  k32u      & 0x000000FF;
		buf[2] = (k32u>>8 ) & 0x000000FF;
		buf[3] = (k32u>>16) & 0x000000FF;
		leu32a(hdr->EVENT.N, buf+4);
	}
	else {
		k32u   = hdr->EVENT.N;
		buf[1] =  k32u      & 0x000000FF;
		buf[2] = (k32u>>8 ) & 0x000000FF;
		buf[3] = (k32u>>16) & 0x000000FF;
		lef32a(hdr->EVENT.SampleRate, buf+4);
	};
	uint8_t *buf1=hdr->AS.rawEventData+8;
	uint8_t *buf2=hdr->AS.rawEventData+8+hdr->EVENT.N*4;
	for (k32u=0; k32u<hdr->EVENT.N; k32u++) {
		*(uint32_t*)(buf1+k32u*4) = htole32(hdr->EVENT.POS[k32u]+1); // convert from 0-based (biosig4c++) to 1-based (GDF) indexing
		*(uint16_t*)(buf2+k32u*2) = htole16(hdr->EVENT.TYP[k32u]);
	}
	if (flag & 2) {
		buf1 = hdr->AS.rawEventData+8+hdr->EVENT.N*6;
		buf2 = hdr->AS.rawEventData+8+hdr->EVENT.N*8;
		for (k32u=0; k32u<hdr->EVENT.N; k32u++) {
			*(uint16_t*)(buf1+k32u*2) = htole16(hdr->EVENT.CHN[k32u]);
			*(uint32_t*)(buf2+k32u*4) = htole32(hdr->EVENT.DUR[k32u]);
		}
	}
#if (BIOSIG_VERSION >= 10500)
	if (flag & 4) {
		buf1 = hdr->AS.rawEventData+8+hdr->EVENT.N*(sze-8);
		for (k32u=0; k32u<hdr->EVENT.N; k32u++) {
			*(uint64_t*)(buf1+k32u*8) = htole64(hdr->EVENT.TimeStamp[k32u]);
		}
	}
#endif
	return(len);
}

/*********************************************************************************
	rawEVT2hdrEVT(HDRTYPE *hdr)
	converts raw event data (hdr->AS.rawEventData) into structure HDR.EVENT

	TODO: support of EVENT.TimeStamp
 *********************************************************************************/
void rawEVT2hdrEVT(HDRTYPE *hdr, size_t length_rawEventData) {
	// TODO: avoid additional copying
	size_t k;
			uint8_t *buf = hdr->AS.rawEventData;
			if ((buf==NULL) || (length_rawEventData < 8)) {
				hdr->EVENT.N = 0;
				return;
			}

			if (hdr->VERSION < 1.94) {
				if (buf[1] | buf[2] | buf[3])
					hdr->EVENT.SampleRate = buf[1] + (buf[2] + buf[3]*256.0)*256.0;
				else {
					fprintf(stdout,"Warning GDF v1: SampleRate in Eventtable is not set in %s !!!\n",hdr->FileName);
					hdr->EVENT.SampleRate = hdr->SampleRate;
				}
				hdr->EVENT.N = leu32p(buf + 4);
			}
			else	{
				hdr->EVENT.N = buf[1] + (buf[2] + buf[3]*256)*256;
				hdr->EVENT.SampleRate = lef32p(buf + 4);
			}

			char flag = buf[0];
			int sze = (flag & 2) ? 12 : 6;
			if (flag & 4) sze+=8;

			if (sze*hdr->EVENT.N+8 < length_rawEventData) {
				hdr->EVENT.N = 0;
		                biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Error GDF: event table is corrupted");
		                return;
			}

			if (hdr->NS==0 && !isfinite(hdr->SampleRate)) hdr->SampleRate = hdr->EVENT.SampleRate;

	 		hdr->EVENT.POS = (uint32_t*) realloc(hdr->EVENT.POS, hdr->EVENT.N*sizeof(*hdr->EVENT.POS) );
			hdr->EVENT.TYP = (uint16_t*) realloc(hdr->EVENT.TYP, hdr->EVENT.N*sizeof(*hdr->EVENT.TYP) );

			uint8_t *buf1 = hdr->AS.rawEventData+8;
			uint8_t *buf2 = hdr->AS.rawEventData+8+4*hdr->EVENT.N;
			for (k=0; k < hdr->EVENT.N; k++) {
				// POS & TYP
				hdr->EVENT.POS[k] = leu32p(buf1 + k*4)-1;  // convert from 1-based (GDF) to 0-based (biosig4c++) indexing
				hdr->EVENT.TYP[k] = leu16p(buf2 + k*2);
			}
			if (flag & 2) {
				// DUR & CHN
				hdr->EVENT.DUR = (uint32_t*) realloc(hdr->EVENT.DUR,hdr->EVENT.N*sizeof(*hdr->EVENT.DUR));
				hdr->EVENT.CHN = (uint16_t*) realloc(hdr->EVENT.CHN,hdr->EVENT.N*sizeof(*hdr->EVENT.CHN));

				buf1 = hdr->AS.rawEventData+8+6*hdr->EVENT.N;
				buf2 = hdr->AS.rawEventData+8+8*hdr->EVENT.N;
				for (k=0; k < hdr->EVENT.N; k++) {
					hdr->EVENT.CHN[k] = leu16p(buf1 + k*2);
					hdr->EVENT.DUR[k] = leu32p(buf2 + k*4);
				}
			}
			else {
				hdr->EVENT.DUR = NULL;
				hdr->EVENT.CHN = NULL;
			}
#if (BIOSIG_VERSION >= 10500)
			if (flag & 4) {
				// TimeStamp
				hdr->EVENT.TimeStamp = (gdf_time*) realloc(hdr->EVENT.TimeStamp, hdr->EVENT.N*sizeof(*hdr->EVENT.TimeStamp));
				buf1 = hdr->AS.rawEventData+8+hdr->EVENT.N*(sze-8);
				for (k=0; k < hdr->EVENT.N; k++) {
					hdr->EVENT.TimeStamp[k] = leu64p(buf1 + k*8);
				}
			} else {
				hdr->EVENT.TimeStamp = NULL;
			}
#endif
}

int NumberOfChannels(HDRTYPE *hdr)
{
        unsigned int k,NS;
        for (k=0, NS=0; k<hdr->NS; k++)
                if (hdr->CHANNEL[k].OnOff==1) NS++;

#ifdef CHOLMOD_H
        if (hdr->Calib == NULL)
                return (NS);

        if (NS == hdr->Calib->nrow)
                return (hdr->Calib->ncol);
#endif
        return(hdr->NS);
}

int RerefCHANNEL(HDRTYPE *hdr, void *arg2, char Mode)
{
#ifndef CHOLMOD_H
                if (!arg2 || !Mode) return(0); // do nothing

                biosigERROR(hdr, B4C_REREF_FAILED, "Error RerefCHANNEL: cholmod library is missing");
                return(1);
#else
                if (arg2==NULL) Mode = 0; // do nothing

                cholmod_sparse *ReRef=NULL;
		uint16_t flag,NS;
                size_t i,j,k;
                long r;
		char flagLabelIsSet = 0;

                switch (Mode) {
                case 1: {
                        HDRTYPE *RR = sopen((const char*)arg2,"r",NULL);
                        ReRef       = RR->Calib;
			if (RR->rerefCHANNEL != NULL) {
                                flagLabelIsSet = 1;
				if (hdr->rerefCHANNEL) free(hdr->rerefCHANNEL);
				hdr->rerefCHANNEL = RR->rerefCHANNEL;
				RR->rerefCHANNEL  = NULL;
                        }
                        RR->Calib   = NULL; // do not destroy ReRef
                        destructHDR(RR);
                        RR = NULL;
                        break;
                        }
                case 2: ReRef = (cholmod_sparse*) arg2;
                        CSstart();
                        break;
                }

                if ((ReRef==NULL) || !Mode) {
                        // reset rereferencing

        		if (hdr->Calib != NULL)
				 cholmod_free_sparse(&hdr->Calib, &CHOLMOD_COMMON_VAR);
                        hdr->Calib = ReRef;
        		if (hdr->rerefCHANNEL) free(hdr->rerefCHANNEL);
        		hdr->rerefCHANNEL = NULL;

        	        return(0);
                }
                cholmod_sparse *A = ReRef;

                // check dimensions
                for (k=0, NS=0; k<hdr->NS; k++)
                        if (hdr->CHANNEL[k].OnOff) NS++;
                if (NS - A->nrow) {
                        biosigERROR(hdr, B4C_REREF_FAILED, "Error REREF_CHAN: size of data does not fit ReRef-matrix");
                        return(1);
                }

                // allocate memory
		if (hdr->Calib != NULL)
                        cholmod_free_sparse(&hdr->Calib, &CHOLMOD_COMMON_VAR);

		if (VERBOSE_LEVEL>8) {
			CHOLMOD_COMMON_VAR.print = 5;
			cholmod_print_sparse(ReRef,"HDR.Calib", &CHOLMOD_COMMON_VAR);
		}

                hdr->Calib = ReRef;
                if (hdr->rerefCHANNEL==NULL)
			hdr->rerefCHANNEL = (CHANNEL_TYPE*) realloc(hdr->rerefCHANNEL, A->ncol*sizeof(CHANNEL_TYPE));

		CHANNEL_TYPE *NEWCHANNEL = hdr->rerefCHANNEL;
                hdr->FLAG.ROW_BASED_CHANNELS = 1;

                // check each component
       		for (i=0; i<A->ncol; i++)         // i .. column index
       		{
			flag = 0;
			int mix = -1, oix = -1, pix = -1;
			double m  = 0.0;
			double v;
			for (j = *((unsigned*)(A->p)+i); j < *((unsigned*)(A->p)+i+1); j++) {

				v = *(((double*)A->x)+j);
				r = *(((int*)A->i)+j);        // r .. row index

				if (v>m) {
					m = v;
					mix = r;
				}
				if (v==1.0) {
					if (oix<0)
						oix = r;
					else
						fprintf(stderr,"Warning: ambiguous channel information (in new #%i,%i more than one scaling factor of 1.0 is used.) \n",(int)i,(int)j);
				}
				if (v) {
					if (pix == -1) {
                				//memcpy(NEWCHANNEL+i, hdr->CHANNEL+r, sizeof(CHANNEL_TYPE));
					        NEWCHANNEL[i].PhysDimCode = hdr->CHANNEL[r].PhysDimCode;
					        NEWCHANNEL[i].LowPass 	  = hdr->CHANNEL[r].LowPass;
					        NEWCHANNEL[i].HighPass    = hdr->CHANNEL[r].HighPass;
					        NEWCHANNEL[i].Notch 	  = hdr->CHANNEL[r].Notch;
					        NEWCHANNEL[i].SPR 	  = hdr->CHANNEL[r].SPR;
					        NEWCHANNEL[i].GDFTYP      = hdr->CHANNEL[r].GDFTYP;
				                NEWCHANNEL[i].Impedance   = fabs(v)*hdr->CHANNEL[r].Impedance;
						NEWCHANNEL[i].OnOff       = 1;
						NEWCHANNEL[i].LeadIdCode  = 0;
						if (!flagLabelIsSet) memcpy(NEWCHANNEL[i].Label, hdr->CHANNEL[r].Label, MAX_LENGTH_LABEL);
				                pix = 0;
					}
					else {
					        if (NEWCHANNEL[i].PhysDimCode != hdr->CHANNEL[r].PhysDimCode)
					                NEWCHANNEL[i].PhysDimCode = 0;
					        if (NEWCHANNEL[i].LowPass != hdr->CHANNEL[r].LowPass)
					                NEWCHANNEL[i].LowPass = NAN;
					        if (NEWCHANNEL[i].HighPass != hdr->CHANNEL[r].HighPass)
					                NEWCHANNEL[i].HighPass = NAN;
					        if (NEWCHANNEL[i].Notch != hdr->CHANNEL[r].Notch)
					                NEWCHANNEL[i].Notch = NAN;

					        if (NEWCHANNEL[i].SPR != hdr->CHANNEL[r].SPR)
					                NEWCHANNEL[i].SPR = lcm(NEWCHANNEL[i].SPR, hdr->CHANNEL[r].SPR);
					        if (NEWCHANNEL[i].GDFTYP != hdr->CHANNEL[r].GDFTYP)
					                NEWCHANNEL[i].GDFTYP = max(NEWCHANNEL[i].GDFTYP, hdr->CHANNEL[r].GDFTYP);

				                NEWCHANNEL[i].Impedance += fabs(v)*NEWCHANNEL[r].Impedance;
				                NEWCHANNEL[i].GDFTYP = 16;
					}
				}
				if (r >= hdr->NS) {
					flag = 1;
					fprintf(stderr,"Error: index (%i) in channel (%i) exceeds number of channels (%i)\n",(int)r,(int)i,hdr->NS);
				}
			}

			// heuristic to determine hdr->CHANNEL[k].Label;
			if (oix>-1) r=oix;        // use the info from channel with a scaling of 1.0 ;
			else if (mix>-1) r=mix;   // use the info from channel with the largest scale;
			else r = -1;

			if (flagLabelIsSet)
                               ;
			else if (!flag && (r<hdr->NS) && (r>=0)) {
			        // if successful
			        memcpy(NEWCHANNEL[i].Label, hdr->CHANNEL[r].Label, MAX_LENGTH_LABEL);
                        }
			else {
			        sprintf(NEWCHANNEL[i].Label,"component #%i",(int)i);
			}
                }
		return(0);
#endif
}



/****************************************************************************
 *                     READ_HEADER_1
 *
 *
 ****************************************************************************/
int read_header(HDRTYPE *hdr) {
/*
	input:
		hdr must be an open file able to read from
		hdr->TYPE must be unknown, otherwise no FileFormat evaluation is performed
		hdr->FILE.size
	output:
		defines whole header structure and event table
	return value:
	0	no error
	-1	error reading header 1
	-2	error reading header 2
	-3	error reading event table
 */

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): %i %i %f\n",__func__,__LINE__, (int)hdr->FILE.size, (int)hdr->HeadLen, hdr->VERSION);

	size_t count = hdr->HeadLen;
	if (hdr->HeadLen<=512) {
		ifseek(hdr, count, SEEK_SET);
		hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, 513);
		count += ifread(hdr->AS.Header+hdr->HeadLen, 1, 512-count, hdr);
		getfiletype(hdr);
	}
    	char tmp[6];
    	strncpy(tmp,(char*)hdr->AS.Header+3,5); tmp[5]=0;
    	hdr->VERSION 	= atof(tmp);

	// currently, only GDF is supported
	if ( (hdr->TYPE != GDF) || (hdr->VERSION < 0.01) )
		return ( -1 );

    	if (hdr->VERSION > 1.90)
	    	hdr->HeadLen = leu16p(hdr->AS.Header+184)<<8;
	else
	    	hdr->HeadLen = leu64p(hdr->AS.Header+184);

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): %i %i %i %f\n", __func__, __LINE__,(int)hdr->FILE.size, (int)hdr->HeadLen, (int)count, hdr->VERSION);

	hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,hdr->HeadLen);
        if (count < hdr->HeadLen) {
		ifseek(hdr, count, SEEK_SET);
	    	count += ifread(hdr->AS.Header+count, 1, hdr->HeadLen-count, hdr);
	}

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): %i %i %i %f\n",__func__, __LINE__, (int)hdr->FILE.size, (int)hdr->HeadLen, (int)count, hdr->VERSION);

        if (count < hdr->HeadLen) {
		if (VERBOSE_LEVEL>7) fprintf(stdout,"ambiguous GDF header size: %i %i\n",(int)count,hdr->HeadLen);
                biosigERROR(hdr, B4C_INCOMPLETE_FILE, "reading GDF header failed");
                return(-2);
	}

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): %i %i %i %f\n",__func__, __LINE__, (int)hdr->FILE.size, (int)hdr->HeadLen, (int)count, hdr->VERSION);

	if ( gdfbin2struct(hdr) ) {
	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): %i %i %i %f\n",__func__, __LINE__, (int)hdr->FILE.size, (int)hdr->HeadLen, (int)count, hdr->VERSION);
		return(-2);
	}

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): %i %i %i %f\n",__func__, __LINE__, (int)hdr->FILE.size, (int)hdr->HeadLen, (int)count, hdr->VERSION);

 	hdr->EVENT.N   = 0;
	hdr->EVENT.POS = NULL;
	hdr->EVENT.TYP = NULL;
	hdr->EVENT.DUR = NULL;
	hdr->EVENT.CHN = NULL;
#if (BIOSIG_VERSION >= 10500)
	hdr->EVENT.TimeStamp = NULL;
#endif

	if (hdr->NRec < 0) {
		hdr->NRec = (hdr->FILE.size - hdr->HeadLen)/hdr->AS.bpb;
		if (hdr->AS.rawEventData!=NULL) {
			free(hdr->AS.rawEventData);
			hdr->AS.rawEventData=NULL;
		}
	}
	else if (hdr->FILE.size > hdr->HeadLen + hdr->AS.bpb*(size_t)hdr->NRec + 8)
	{
			if (VERBOSE_LEVEL > 7)
				fprintf(stdout,"GDF EVENT: %i,%i %i,%i,%i\n",(int)hdr->FILE.size, (int)(hdr->HeadLen + hdr->AS.bpb*hdr->NRec + 8), hdr->HeadLen, hdr->AS.bpb, (int)hdr->NRec);

			ifseek(hdr, hdr->HeadLen + hdr->AS.bpb*hdr->NRec, SEEK_SET);
			// READ EVENTTABLE
			hdr->AS.rawEventData = (uint8_t*)realloc(hdr->AS.rawEventData,8);
			size_t c = ifread(hdr->AS.rawEventData, sizeof(uint8_t), 8, hdr);
    			uint8_t *buf = hdr->AS.rawEventData;

			if (c<8) {
				hdr->EVENT.N = 0;
			}
			else if (hdr->VERSION < 1.94) {
				hdr->EVENT.N = leu32p(buf + 4);
			}
			else {
				hdr->EVENT.N = buf[1] + (buf[2] + buf[3]*256)*256;
			}

			if (VERBOSE_LEVEL > 7)
				fprintf(stdout,"EVENT.N = %i,%i\n",hdr->EVENT.N,(int)c);

			char flag = buf[0];
			int sze = (flag & 2) ? 12 : 6;
			if (flag & 4) sze+=8;

			hdr->AS.rawEventData = (uint8_t*)realloc(hdr->AS.rawEventData,8+hdr->EVENT.N*sze);
			c = ifread(hdr->AS.rawEventData+8, sze, hdr->EVENT.N, hdr);
			ifseek(hdr, hdr->HeadLen, SEEK_SET);
			if (c < hdr->EVENT.N) {
                                biosigERROR(hdr, B4C_INCOMPLETE_FILE, "reading GDF eventtable failed");
                                return(-3);
			}
			rawEVT2hdrEVT(hdr, 8+hdr->EVENT.N*sze);
		}
		else
			hdr->EVENT.N = 0;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"[228] FMT=%s Ver=%4.2f\n",GetFileTypeString(hdr->TYPE),hdr->VERSION);

	return (0);
}


/****************************************************************************/
/**                     SOPEN                                              **/
/****************************************************************************/
HDRTYPE* sopen(const char* FileName, const char* MODE, HDRTYPE* hdr) {

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d): sopen(%s,%s)\n",__func__,__LINE__, FileName, MODE);

	return sopen_extended(FileName, MODE, hdr, NULL);
}

HDRTYPE* sopen_extended(const char* FileName, const char* MODE, HDRTYPE* hdr, biosig_options_type *biosig_options) {
/*
	MODE="r"
		reads file and returns HDR
	MODE="w"
		writes HDR into file
 */

//    	unsigned int 	k2;
//    	uint32_t	k32u;
    	size_t	 	count;
#ifndef  ONLYGDF
//    	double 		Dur;
	char*		ptr_str;
	struct tm 	tm_time;
//	time_t		tt;

	const char	GENDER[] = "XMFX";
	const uint16_t	CFWB_GDFTYP[] = {17,16,3};
	const float	CNT_SETTINGS_NOTCH[] = {0.0, 50.0, 60.0};
	const float	CNT_SETTINGS_LOWPASS[] = {30, 40, 50, 70, 100, 200, 500, 1000, 1500, 2000, 2500, 3000};
	const float	CNT_SETTINGS_HIGHPASS[] = {NAN, 0, .05, .1, .15, .3, 1, 5, 10, 30, 100, 150, 300};
	uint16_t	BCI2000_StatusVectorLength=0;	// specific for BCI2000 format
#endif //ONLYGDF

	biosig_options_type default_options;
	default_options.free_text_event_limiter="\0";

	if (biosig_options==NULL) biosig_options = &default_options;

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s(%s,%s) (line %d): --delimiter=<%s> %p\n",__func__, FileName, MODE, __LINE__, biosig_options->free_text_event_limiter, biosig_options);

	if (FileName == NULL) {
		biosigERROR(hdr, B4C_CANNOT_OPEN_FILE, "no filename specified");
		return (hdr);
	}
	if (hdr==NULL)
		hdr = constructHDR(0,0);	// initializes fields that may stay undefined during SOPEN

	if (FileName != NULL) {
		if (hdr->FileName) free(hdr->FileName);
		hdr->FileName = strdup(FileName);
	}

	if (VERBOSE_LEVEL>6)
		fprintf(stdout,"SOPEN( %s, %s) open=%i\n",FileName, MODE, hdr->FILE.OPEN);

	setlocale(LC_NUMERIC,"C");

	// hdr->FLAG.SWAP = (__BYTE_ORDER == __BIG_ENDIAN); 	// default: most data formats are little endian
	hdr->FILE.LittleEndian = 1;

if (!strncmp(MODE,"a",1)) {

	/***** 	SOPEN APPEND *****/
	HDRTYPE *hdr2 = NULL;
	struct stat FileBuf;
	if (stat(FileName, &FileBuf)==0)
		hdr->FILE.size = FileBuf.st_size;
	else
		hdr->FILE.size = 0;

	if (hdr->FILE.size==0) {
		if (hdr->FILE.OPEN) ifclose(hdr);
		return( sopen(FileName, "w", hdr) );
	}
	else if (hdr->FILE.size < 256) {
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Error SOPEN(APPEND);  file format not supported.");
		return (hdr);
	}
	else {
		// read header of existing file
		hdr2 = sopen(FileName, "r", hdr2);
		sclose(hdr2);
	};

	if (hdr2->TYPE != GDF) {
		// currently only GDF is tested and supported
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Error SOPEN(APPEND);  file format not supported.");
		destructHDR(hdr2);
		return (hdr);
	}

	// test for additional restrictions
	if ( hdr2->EVENT.N > 0 && hdr2->FILE.COMPRESSION ) {
		// gzopen does not support "rb+" (simultaneous read/write) but can only append at the end of file
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Error SOPEN(GDF APPEND);  cannot append to compressed GDF file containing event table.");
		destructHDR(hdr2);
		return (hdr);
	}

	// use header of existing file, sopen does hdr=hdr2, and open files for writing.
	destructHDR(hdr);
	if (hdr2->FILE.COMPRESSION)
		hdr = ifopen(hdr2, "ab");
	else {
		hdr = ifopen(hdr2, "rb+");
		ifseek(hdr, hdr->HeadLen + hdr->NRec*hdr->AS.bpb, SEEK_SET);
	}
	if (!hdr->FILE.OPEN) {
		biosigERROR(hdr, B4C_CANNOT_OPEN_FILE, "Error SOPEN(APPEND); Cannot open file.");
		return(hdr);
	}
	hdr->FILE.OPEN = 2;
}

else if (!strncmp(MODE,"r",1)) {
	/***** 	SOPEN READ *****/

#ifndef WITHOUT_NETWORK
	if (!memcmp(hdr->FileName,"bscs://",7)) {
		uint64_t ID;
    		char *hostname = (char*)hdr->FileName+7;
    		char *t = strrchr(hostname,'/');
    		if (t==NULL) {
			biosigERROR(hdr, B4C_CANNOT_OPEN_FILE, "SOPEN-NETWORK: file identifier not specifed");
    			return(hdr);
    		}
    		*t=0;
		cat64(t+1, &ID);
		int sd,s;
		sd = bscs_connect(hostname);
		if (sd<0) {
			biosigERROR(hdr, B4C_CANNOT_OPEN_FILE, "could not connect to server");
			return(hdr);
		}
  		hdr->FILE.Des = sd;
		s  = bscs_open(sd, &ID);
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%i = bscs_open\n",s);
  		s  = bscs_requ_hdr(sd,hdr);
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%i = bscs_requ_hdr\n",s);
  		s  = bscs_requ_evt(sd,hdr);
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%i = bscs_requ_evt\n",s);
  		hdr->FILE.OPEN = 1;
  		return(hdr);
    	}
#endif

	// modern cpu's have cache lines of 4096 bytes, so for performance reasons we use this size as well.
	const size_t PAGESIZE=4096;
	/* reading some formats may imply that at least 512 bytes are read,
           if you want to use a smaller page size, double check whether your format(s) are correctly handled. */
	assert(PAGESIZE >= 512);
	hdr->AS.Header = (uint8_t*)malloc(PAGESIZE+1);

	size_t k;
#ifndef  ONLYGDF
	size_t name=0,ext=0;
	for (k=0; hdr->FileName[k]; k++) {
		if (hdr->FileName[k]==FILESEP) name = k+1;
		if (hdr->FileName[k]=='.')     ext  = k+1;
	}

	const char *FileExt  = hdr->FileName+ext;
	const char *FileName = hdr->FileName+name;
#endif //ONLYGDF

#ifdef __CURL_CURL_H
	if (! strncmp(hdr->FileName,"file://", 7)
         || ! strncmp(hdr->FileName,"ftp://", 6)
         || ! strncmp(hdr->FileName,"http://", 7)
         || ! strncmp(hdr->FileName,"https://", 8) )
	{
	        CURL *curl;
	        char errbuffer[CURL_ERROR_SIZE];

	        if ((curl = curl_easy_init()) != NULL) {
			FILE *tmpfid = tmpfile();

	                curl_easy_setopt(curl, CURLOPT_URL, hdr->FileName);
	                curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
	                curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errbuffer);
			if (VERBOSE_LEVEL > 6)
		                curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
			curl_easy_setopt(curl, CURLOPT_WRITEDATA, tmpfid);
	                if (curl_easy_perform(curl) != CURLE_OK) {
				fprintf(stderr,"CURL ERROR: %s\n",errbuffer);
				biosigERROR(hdr, B4C_CANNOT_OPEN_FILE, "Error SOPEN(READ); file download failed.");
				fclose(tmpfid);
				return(hdr);
			}

			/*
				associate temporary file with input stream
				channeling everything through zlib ensures that *.gz files
				are automatically decompressed
				According to http://www.acm.uiuc.edu/webmonkeys/book/c_guide/2.12.html#tmpfile,
				the tmpfile will be removed when stream is closed
			*/
			fseek(tmpfid,0,SEEK_SET);
			hdr->FILE.gzFID = gzdopen(fileno(tmpfid), "r");
		        hdr->FILE.COMPRESSION = 1;
	                curl_easy_cleanup(curl);

			/* */
			count = ifread(hdr->AS.Header, 1, PAGESIZE, hdr);
			hdr->AS.Header[count]=0;
	        }
	} else
#endif
	{
		if (VERBOSE_LEVEL>7) fprintf(stdout,"SOPEN 101: <%s>\n",hdr->FileName);

#ifndef  ONLYGDF
		/* AINF */
		if (!strcmp(FileExt, "ainf")) {
			if (VERBOSE_LEVEL>8) fprintf(stdout,"getfiletype ainf1 %s %i\n",hdr->FileName,(int)ext);
			char* AINF_RAW_FILENAME = (char*)calloc(strlen(hdr->FileName)+5,sizeof(char));
			strncpy(AINF_RAW_FILENAME, hdr->FileName,ext);
			strcpy(AINF_RAW_FILENAME+ext, "raw");
			FILE* fid1=fopen(AINF_RAW_FILENAME,"rb");
			if (fid1) {
				fclose(fid1);
				hdr->TYPE = AINF;
			}
			free(AINF_RAW_FILENAME);
		}
		else if (!strcmp(FileExt, "raw")) {
			char* AINF_RAW_FILENAME = (char*)calloc(strlen(hdr->FileName)+5,sizeof(char));
			strncpy(AINF_RAW_FILENAME, hdr->FileName,ext);
			strcpy(AINF_RAW_FILENAME+ext, "ainf");
			FILE* fid1=fopen(AINF_RAW_FILENAME,"r");
			if (fid1) {
				fclose(fid1);
				hdr->TYPE = AINF;
			}
			free(AINF_RAW_FILENAME);
		}
#endif //ONLYGDF

	        hdr->FILE.COMPRESSION = 0;
		hdr   = ifopen(hdr,"rb");
		if (!hdr->FILE.OPEN) {
			biosigERROR(hdr, B4C_CANNOT_OPEN_FILE, "Error SOPEN(READ); Cannot open file.");
	    		return(hdr);
		}
		if (VERBOSE_LEVEL>7) fprintf(stdout,"SOPEN 101:\n");
		count = ifread(hdr->AS.Header, 1, PAGESIZE, hdr);
		if (count<25) {
			if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d): %s(...) count = %d\n",__FILE__,__LINE__,__func__,count);
			biosigERROR(hdr, B4C_CANNOT_OPEN_FILE, "Error SOPEN(READ); file is empty (or too short)");
			ifclose(hdr);
			return(hdr);
		}
		hdr->AS.Header[count]=0;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) count=%i\n", __func__, __LINE__,(int)count);

		if (!memcmp(Header1,MAGIC_NUMBER_GZIP,strlen(MAGIC_NUMBER_GZIP))) {
#ifdef ZLIB_H
			if (VERBOSE_LEVEL>7) fprintf(stdout,"[221] %i\n",(int)count);

			ifseek(hdr, 0, SEEK_SET);
			hdr->FILE.gzFID = gzdopen(fileno(hdr->FILE.FID),"r");
		        hdr->FILE.COMPRESSION = (uint8_t)1;
			hdr->FILE.FID = NULL;
			count = ifread(hdr->AS.Header, 1, PAGESIZE, hdr);
			hdr->AS.Header[count]=0;
#else
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Error SOPEN(READ); *.gz file not supported because not linked with zlib.");
#endif
	    	}

	}

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...): count%i\n",__FILE__,__LINE__,__func__,(int)count);
	hdr->HeadLen = count;
	getfiletype(hdr);
	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) FMT=%s Ver=%4.2f\n",__FILE__,__LINE__,GetFileTypeString(hdr->TYPE),hdr->VERSION);

#ifndef  ONLYGDF
	if (hdr->TYPE != unknown)
		;
    	else if (!memcmp(Header1,FileName,strspn(FileName,".")) && (!strcmp(FileExt,"HEA") || !strcmp(FileExt,"hea") ))
	    	hdr->TYPE = MIT;
	else if (count < 512) {
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Error SOPEN(READ): file is too short\n");
		return(hdr);
	}
#endif //ONLYGDF

    	if (hdr->TYPE == unknown) {
    		biosigERROR(hdr, B4C_FORMAT_UNKNOWN, "ERROR BIOSIG4C++ SOPEN(read): Dataformat not known.\n");
    		ifclose(hdr);
		return(hdr);
	}

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) FMT=%s Ver=%4.2f\n",__FILE__,__LINE__,GetFileTypeString(hdr->TYPE),hdr->VERSION);

    	count = iftell(hdr);
    	hdr->AS.first  =  0;
    	hdr->AS.length =  0;
	hdr->AS.bpb    = -1; 	// errorneous value: ensures that hdr->AS.bpb will be defined

#ifndef WITHOUT_NETWORK
	if (!memcmp(hdr->AS.Header,"bscs://",7)) {
		hdr->AS.Header[count]=0;

		uint64_t ID;
    		char *hostname = Header1+7;
		Header1[6]=0;
    		char *t = strrchr(hostname,'/');
    		if (t==NULL) {
			biosigERROR(hdr, B4C_CANNOT_OPEN_FILE, "SOPEN-NETWORK: file identifier not specifed");
    			return(hdr);
    		}
    		t[0]=0;
		cat64(t+1, &ID);
		int sd,s;
		sd = bscs_connect(hostname);
		if (sd<0) {
    			fprintf(stderr,"could not connect to %s\n",hostname);
			biosigERROR(hdr, B4C_CANNOT_OPEN_FILE, "could not connect to server");
			return(hdr);
		}
  		hdr->FILE.Des = sd;
		s  = bscs_open(sd, &ID);
  		s  = bscs_requ_hdr(sd,hdr);
  		s  = bscs_requ_evt(sd,hdr);
  		hdr->FILE.OPEN = 1;
  		return(hdr);
    	}
    	else
#endif

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) FMT=%s Ver=%4.2f\n",__FILE__,__LINE__,GetFileTypeString(hdr->TYPE),hdr->VERSION);

	if (hdr->TYPE == GDF) {

		struct stat FileBuf;
		if (stat(hdr->FileName,&FileBuf)==0) hdr->FILE.size = FileBuf.st_size;

	    	if ( read_header(hdr) ) {
			return (hdr);
		}

    	}
#ifndef  ONLYGDF
    	else if ((hdr->TYPE == EDF) || (hdr->TYPE == BDF))	{
                if (count < 256) {
                        biosigERROR(hdr,  B4C_INCOMPLETE_FILE, "reading BDF/EDF fixed header failed");
                        return(hdr);
                }

		typeof(hdr->NS)	StatusChannel = 0;
		int annotStartBi=-1;
		int annotEndBi=-1;
		int annotNumBytesPerBlock=0;

		int last = min(MAX_LENGTH_PID, 80);
		strncpy(hdr->Patient.Id, Header1+8, last);
		while ((0 <= last) && (isspace(hdr->Patient.Id[--last])));
		hdr->Patient.Id[last+1]=0;

		last = min(MAX_LENGTH_RID, 80);
		memcpy(hdr->ID.Recording, Header1+88, last);
		while ((0 <= last) && (isspace(hdr->ID.Recording[--last])));
		hdr->ID.Recording[last+1]=0;

		if (VERBOSE_LEVEL>8) fprintf(stdout,"[EDF 211] #=%li\nT0=<%16s>",iftell(hdr),Header1+168);

		// TODO: sanity check of T0
		char tmp[81];
		memset(tmp,0,9);
		strncpy(tmp,Header1+168+14,2);
    		tm_time.tm_sec  = atoi(tmp);
    		strncpy(tmp,Header1+168+11,2);
    		tm_time.tm_min  = atoi(tmp);
    		strncpy(tmp,Header1+168+8,2);
    		tm_time.tm_hour = atoi(tmp);
    		strncpy(tmp,Header1+168,2);
    		tm_time.tm_mday = atoi(tmp);
    		strncpy(tmp,Header1+168+3,2);
    		tm_time.tm_mon  = atoi(tmp)-1;
    		strncpy(tmp,Header1+168+6,2);
    		tm_time.tm_year = atoi(tmp);
    		tm_time.tm_year+= (tm_time.tm_year < 70 ? 100 : 0);

		hdr->EVENT.N 	= 0;
		memset(tmp,0,9);
	    	hdr->NS		= atoi(memcpy(tmp,Header1+252,4));
	    	hdr->HeadLen	= atoi(memcpy(tmp,Header1+184,8));
		if (hdr->HeadLen < ((hdr->NS+1u)*256)) {
			biosigERROR(hdr, B4C_UNSPECIFIC_ERROR, "EDF/BDF corrupted: HDR.NS and HDR.HeadLen do not fit");
			if (VERBOSE_LEVEL > 7)
				fprintf(stdout,"HeadLen=%i,%i\n",hdr->HeadLen ,(hdr->NS+1)<<8);
		};

	    	hdr->NRec	= atoi(strncpy(tmp,Header1+236,8));
	    	//Dur		= atof(strncpy(tmp,Header1+244,8));

		if (VERBOSE_LEVEL>7) fprintf(stdout,"[EDF 211b] #=%li\nT0=%s\n",iftell(hdr),asctime(&tm_time));

		if (!strncmp(Header1+192,"EDF+",4)) {

		if (VERBOSE_LEVEL>7) fprintf(stdout,"[EDF 211c+] <%s>\n",hdr->Patient.Id);

	    		strtok(hdr->Patient.Id," ");
	    		ptr_str = strtok(NULL," ");
			if (ptr_str!=NULL) {
				// define Id, Sex, Birthday, Name
		if (VERBOSE_LEVEL>7) fprintf(stdout,"[EDF 211c+] <%p>\n",ptr_str);

	    		hdr->Patient.Sex = (ptr_str[0]=='f')*2 + (ptr_str[0]=='F')*2 + (ptr_str[0]=='M') + (ptr_str[0]=='m');
	    		ptr_str = strtok(NULL," ");	// startdate
	    		char *tmpptr = strtok(NULL," ");
	    		if ((!hdr->FLAG.ANONYMOUS) && (tmpptr != NULL)) {
				strncpy(hdr->Patient.Name,tmpptr,MAX_LENGTH_NAME);
				hdr->Patient.Name[MAX_LENGTH_NAME]=0;
		    	}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"[EDF 211c] #=%li\n",iftell(hdr));

			if (strlen(ptr_str)==11) {
				struct tm t1;
				char *strMDay=strtok(ptr_str,"-");
				char *strMonth=strtok(NULL,"-");
				char *strYear=strtok(NULL,"-");
				for (k=0; strMonth[k]>0; ++k) strMonth[k]= toupper(strMonth[k]);	// convert to uppper case

				t1.tm_mday = atoi(strMDay);
				t1.tm_mon  = month_string2int(strMonth);
				t1.tm_year = atoi(strYear) - 1900;
		    		t1.tm_sec  = 0;
		    		t1.tm_min  = 0;
		    		t1.tm_hour = 12;
		    		t1.tm_isdst= -1;
		    		hdr->Patient.Birthday = tm_time2gdf_time(&t1);
		    	}}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"[EDF 211d] <%s>\n",hdr->ID.Recording);

			if (!strncmp(Header1+88,"Startdate ",10)) {
				size_t pos = strcspn(Header1+88+10," ")+10;
				strncpy(hdr->ID.Recording, Header1+88+pos+1, 80-pos);
				hdr->ID.Recording[80-pos-1] = 0;
				if (strtok(hdr->ID.Recording," ")!=NULL) {
					char *tech = strtok(NULL," ");
					if (hdr->ID.Technician) free(hdr->ID.Technician);
					hdr->ID.Technician = (tech != NULL) ?  strdup(tech) : NULL;
					hdr->ID.Manufacturer.Name  = strtok(NULL," ");
				}

				Header1[167]=0;
		    		strtok(Header1+88," ");
	    			ptr_str = strtok(NULL," ");
				// check EDF+ Startdate against T0
		if (VERBOSE_LEVEL>7) fprintf(stdout,"[EDF 211e-] <%s>\n",ptr_str);
				/* TODO:
					fix "Startdate X ..."

				*/
				if (strcmp(ptr_str,"X")) {
					int d,m,y;
					d = atoi(strtok(ptr_str,"-"));
		if (VERBOSE_LEVEL>7) fprintf(stdout,"[EDF 211e] <%s>\n",ptr_str);
					ptr_str = strtok(NULL,"-");
		if (VERBOSE_LEVEL>7) fprintf(stdout,"[EDF 211f] <%s>\n",ptr_str);
					strcpy(tmp,ptr_str);
		    			for (k=0; k<strlen(tmp); ++k) tmp[k]=toupper(tmp[k]);	// convert to uppper case
					m = month_string2int(tmp);
		if (VERBOSE_LEVEL>7) fprintf(stdout,"[EDF 211g] <%s>\n",tmp);
	    				y = atoi(strtok(NULL,"-")) - 1900;
		if (VERBOSE_LEVEL>7) fprintf(stdout,"[EDF 211h] <%i>\n",tm_time.tm_year);

		    			if ((tm_time.tm_mday == d) && (tm_time.tm_mon == m)) {
		    				tm_time.tm_year = y;
				    		tm_time.tm_isdst= -1;
				    	}
					else {
	    					fprintf(stderr,"Error SOPEN(EDF+): recording dates do not match %i/%i <> %i/%i\n",d,m,tm_time.tm_mday,tm_time.tm_mon);
	    				}
		    		}
			}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"[EDF 211z] #=%li\n",iftell(hdr));

		}
		hdr->T0 = tm_time2gdf_time(&tm_time);  // note: sub-second information will be extracted from first annotation
		if (VERBOSE_LEVEL>7) fprintf(stdout,"[EDF 212] #=%li\n",iftell(hdr));

		if (hdr->NS==0) return(hdr);

	    	hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
	    	hdr->AS.Header = (uint8_t*) realloc(Header1,hdr->HeadLen);
	    	char *Header2 = (char*)hdr->AS.Header+256;
		if (hdr->HeadLen > count)
			count  += ifread(hdr->AS.Header+count, 1, hdr->HeadLen-count, hdr);

                if (count < hdr->HeadLen) {
                        biosigERROR(hdr, B4C_INCOMPLETE_FILE, "reading BDF/EDF variable header failed");
                        return(hdr);
                }

                /* identify buggy NeuroLoggerEDF  export with bytes 236-257
                	EDF requires that the fields are left-justified, thus the first byte should be different than a space
			Therefore, the probability of a false positive detection is highly unlikely.
                */
                char FLAG_BUGGY_NEUROLOGGER_EDF = !strncmp(Header1+236," 1       0       8   ",21) && Header1[0x180]==' '
						&& Header1[0x7c0]==' ' && Header1[0x400]==' ' && Header1[0x440]==' '
						&& Header1[0x480]==' ' && Header1[0x4c0]==' ' && Header1[0x500]==' ';

                if (FLAG_BUGGY_NEUROLOGGER_EDF) for (k=236; k<9*256; k++) Header1[k-1]=Header1[k];

		char p[9];
		hdr->AS.bpb = 0;
		size_t BitsPerBlock = 0;
		for (k=0, hdr->SPR = 1; k<hdr->NS; k++)	{
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"[EDF 213] #%i/%i\n",(int)k,hdr->NS);

			hc->LeadIdCode = 0;
			last = min(MAX_LENGTH_LABEL, 16);
			strncpy(hc->Label, Header2 + 16*k, last);
			while ((0 <= last) && (isspace(hc->Label[--last]))) ;
			hc->Label[last+1]=0;

			last = min(80,MAX_LENGTH_TRANSDUCER);
			strncpy(hc->Transducer, Header2+80*k+16*hdr->NS, last);
			while ((0 <= last) && (isspace(hc->Transducer[--last])));
			hc->Transducer[last+1]=0;

			// PhysDim -> PhysDimCode
			last = 8;
			memcpy(p,Header2 + 8*k + 96*hdr->NS, last);
			while ((0 <= last) && (isspace(p[--last])));
			p[last+1]=0;

			hc->PhysDimCode = PhysDimCode(p);
			tmp[8] = 0;

			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"[EDF 215a] #%i/%i\n",(int)k,hdr->NS);

			hc->PhysMin = atof(strncpy(tmp,Header2 + 8*k + 104*hdr->NS,8));
			hc->PhysMax = atof(strncpy(tmp,Header2 + 8*k + 112*hdr->NS,8));
			hc->DigMin  = atof(strncpy(tmp,Header2 + 8*k + 120*hdr->NS,8));
			hc->DigMax  = atof(strncpy(tmp,Header2 + 8*k + 128*hdr->NS,8));

			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"[EDF 215b] #%i/%i\n",(int)k,hdr->NS);

			hc->Cal     = (hc->PhysMax - hc->PhysMin) / (hc->DigMax-hc->DigMin);
			hc->Off     =  hc->PhysMin - hc->Cal*hc->DigMin;

			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"[EDF 215c] #%i: NS=%i NRec=%i \n",(int)k,hdr->NS,(int)hdr->NRec);

			hc->LeadIdCode  = 0;
			hc->SPR     	= atol(strncpy(tmp, Header2 + 8*k + 216*hdr->NS, 8));
			hc->GDFTYP  	= ((hdr->TYPE != BDF) ? 3 : 255+24);
			hc->OnOff   	= 1;

			hc->bi 		= hdr->AS.bpb;
			hc->bi8     	= BitsPerBlock;
			size_t nbits 	= GDFTYP_BITS[hc->GDFTYP]*(size_t)hc->SPR;
			BitsPerBlock   += nbits;
			uint32_t nbytes = nbits>>3;
			hdr->AS.bpb    += nbytes;

			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"[EDF 216] #%i/%i/%i/%i/%i/%i\n",(int)k,hdr->NS,nbytes,hdr->AS.bpb,hc->SPR,hdr->SPR);

			hc->LowPass  = NAN;
			hc->HighPass = NAN;
			hc->Notch    = NAN;
			hc->TOffset  = NAN;
			hc->Impedance = NAN;

			// decode filter information into hdr->Filter.{Lowpass, Highpass, Notch}
			uint8_t kk;
			char PreFilt[81];
			strncpy(PreFilt, Header2+ 80*k + 136*hdr->NS, 80);
			for (kk=0; kk<80; kk++) PreFilt[kk] = toupper(PreFilt[kk]);
			PreFilt[80] = 0;

			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"#%i# <%s>\n",(int)k,PreFilt);

			char *s1;
			s1 = strstr(PreFilt,"HP:");
			if (s1) hc->HighPass = strtod(s1+3, &s1);
			s1 = strstr(PreFilt,"LP:");
			if (s1) hc->LowPass  = strtod(s1+3, &s1);
			s1 = strstr(PreFilt,"NOTCH:");
			if (s1) hc->Notch    = strtod(s1+6, &s1);

			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"#%i# HP: %fHz  LP:%fHz NOTCH=%f\n",(int)k,hc->HighPass,hc->LowPass,hc->Notch);

			if ((hdr->TYPE==EDF) && !strcmp(hc->Label,"EDF Annotations")) {
				hc->OnOff = 0;
				if (annotStartBi < 0) annotStartBi = hc->bi;
				annotEndBi = hdr->AS.bpb;
				annotNumBytesPerBlock += nbytes;
			}
			else if ((hdr->TYPE==BDF) && !strcmp(hc->Label,"BDF Annotations")) {
				hc->OnOff = 0;
				if (annotStartBi < 0) annotStartBi = hc->bi;
				annotEndBi = hdr->AS.bpb;
				annotNumBytesPerBlock += nbytes;
			}
			if ((hdr->TYPE==BDF) && !strcmp(hc->Label,"Status")) {
				hc->OnOff = 0;
				StatusChannel = k+1;
			}
			if (hc->OnOff) {
				// common sampling rate is based only on date channels but not annotation channels
				hdr->SPR = lcm(hdr->SPR, hc->SPR);
			}

			if (VERBOSE_LEVEL>7) fprintf(stdout,"[EDF 219] #%i/%i/%i\n",(int)k,hdr->NS,hdr->SPR);

		}
		hdr->FLAG.OVERFLOWDETECTION = 0; 	// EDF does not support automated overflow and saturation detection
	    	double Dur	= atof(strncpy(tmp,Header1+244,8));
	    	if (Dur==0.0 && FLAG_BUGGY_NEUROLOGGER_EDF) Dur = hdr->SPR/496.0;
		hdr->SampleRate = hdr->SPR/Dur;

		if (VERBOSE_LEVEL>8) fprintf(stdout,"[EDF 220] #=%i SPR=%i Dur=%g\n",(int)iftell(hdr),(int)hdr->SPR, Dur);

		if (hdr->NRec <= 0) {
        		struct stat FileBuf;
        		stat(hdr->FileName,&FileBuf);
			hdr->NRec = (FileBuf.st_size - hdr->HeadLen)/hdr->AS.bpb;
		}

		if (annotStartBi + annotNumBytesPerBlock - annotEndBi)
			fprintf(stdout, "WARNING: this file has multiple non-contigous blocks of EDF+/BDF+ annotations channels - annotation channels are not decoded");
		else {
			/* read Annotation and Status channel and extract event information */
			size_t bpb	= annotNumBytesPerBlock;
			size_t len 	= bpb * hdr->NRec;
			uint8_t *Marker = (uint8_t*)malloc(len + 1);
			size_t skip 	= hdr->AS.bpb - bpb;
			ifseek(hdr, hdr->HeadLen + annotStartBi, SEEK_SET);
			nrec_t k3;
			for (k3=0; k3<hdr->NRec; k3++) {
			    	ifread(Marker+k3*bpb, 1, bpb, hdr);
				ifseek(hdr, skip, SEEK_CUR);
			}
			Marker[hdr->NRec*bpb] = 20; // terminating marker
			size_t N_EVENT = 0;
			hdr->EVENT.SampleRate = hdr->SampleRate;

			/* convert EDF+/BDF+ annotation channel into event table */
			char flag_subsec_isset = 0;
			for (k3 = 0; k3 < hdr->NRec; k3++) {
				double timeKeeping = 0;
				char *line = (char*)(Marker + k3 * bpb);

				char flag = !strncmp(Header1+193,"DF+D",4); // no time keeping for EDF+C
				while (line < (char*)(Marker + (k3+1) * bpb)) {
					// loop through all annotations within a segment

if (VERBOSE_LEVEL>7) fprintf(stdout,"EDF+ line<%s>\n",line);

					char *next = strchr(line,0); // next points to end of annotation

					char *s1 = strtok(line,"\x14");
					char *s2 = strtok(NULL,"\x14");
					char *s3 = strtok(NULL,"\x14");
					char *tstr   = strtok(s1,"\x14\x15");
					char *durstr = strtok(NULL,"\x14\x15");

					if (tstr==NULL) {
						// TODO: check whether this is needed based on the EDF+ specs or whether it is an incorrect
						fprintf(stderr,"Warning EDF+ events: tstr not defined\n");
if (VERBOSE_LEVEL>7) fprintf(stdout,"%s(line %i): EDF+ line<%s>\n",__FILE__,__LINE__,line);
						break;
					}

					double t = atof(tstr);

					/* set sub-second start time: see also
						https://github.com/mne-tools/mne-python/pull/7875
						https://www.edfplus.info/specs/edfplus.html#tal
					*/
					if ( !flag_subsec_isset && (s2==NULL) && (tstr[0]=='+' || tstr[0]=='-' ) ) {
						// first t contains subsecond information of starttime
						if (VERBOSE_LEVEL>7) fprintf(stdout,"T0=%20f + %f s\n",ldexp(hdr->T0,-32)*24*3600,t);
						hdr->T0 += ldexp(t / (24 * 3600.0), +32);
						if (VERBOSE_LEVEL>7) fprintf(stdout,"T0=%20f\n", ldexp(hdr->T0,-32) * 24 * 3600);
					}
					flag_subsec_isset = 1;

					if (flag>0 || s2!=NULL) {
						if (N_EVENT <= hdr->EVENT.N+1) {
							N_EVENT = reallocEventTable(hdr, max(6,hdr->EVENT.N*2));
							if (N_EVENT == SIZE_MAX) {
								biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
								return (hdr);
							};
						}

						/*
							This is a workaround to read BDF data with large number of free text events containing meta information
							Free text is limited to the first occurence of limiter character, default is "\0" which does not remove anything
						 */
						s2 = strtok(s2, biosig_options->free_text_event_limiter);

						switch (flag) {
						case 0:
							// EDF+C
							FreeTextEvent(hdr, hdr->EVENT.N, s2);   // set hdr->EVENT.TYP
							hdr->EVENT.POS[hdr->EVENT.N] = round(t * hdr->EVENT.SampleRate);
							break;
						case 1:
							// EDF+D: marker for beginning of segment
							hdr->EVENT.TYP[hdr->EVENT.N] = 0x7ffe;
							hdr->EVENT.POS[hdr->EVENT.N] = k3 * hdr->SPR;
							timeKeeping = t;
							flag = 2;
							break;
						default:
							// EDF+D: real annotation
							FreeTextEvent(hdr, hdr->EVENT.N, s2);   // set hdr->EVENT.TYP
							hdr->EVENT.POS[hdr->EVENT.N] = k3 * hdr->SPR + round((t-timeKeeping) * hdr->EVENT.SampleRate);
							break;
						}

#if (BIOSIG_VERSION >= 10500)
						hdr->EVENT.TimeStamp[hdr->EVENT.N] = hdr->T0 + ldexp(t/(24*3600),32);
#endif
						hdr->EVENT.DUR[hdr->EVENT.N] = durstr ? (atof(durstr)*hdr->EVENT.SampleRate) : 0;
						hdr->EVENT.CHN[hdr->EVENT.N] = 0;
						hdr->EVENT.N++;
					}

if (VERBOSE_LEVEL>7) fprintf(stdout,"EDF+ event\n\ts1:\t<%s>\n\ts2:\t<%s>\n\ts3:\t<%s>\n\tsdelay:\t<%s>\n\tdur:\t<%s>\n\t\n",s1,s2,s3,tstr,durstr);

					for (line=next; *line==0; line++) {};  // skip \0's and set line to start of next annotation
				}
			}

			hdr->AS.auxBUF = Marker;	// contains EVENT.CodeDesc strings
		}	/* End reading if Annotation channel */

		if (StatusChannel) {
				/* read Status channel and extract event information */
				CHANNEL_TYPE *hc = hdr->CHANNEL+StatusChannel-1;

				size_t sz   	= GDFTYP_BITS[hc->GDFTYP]>>3;
				size_t len 	= hc->SPR * hdr->NRec * sz;
				uint8_t *Marker = (uint8_t*)malloc(len + 1);
				size_t skip 	= hdr->AS.bpb - hc->SPR * sz;
				ifseek(hdr, hdr->HeadLen + hc->bi, SEEK_SET);
				nrec_t k3;
				for (k3=0; k3<hdr->NRec; k3++) {
				    	ifread(Marker+k3*hc->SPR * sz, 1, hc->SPR * sz, hdr);
					ifseek(hdr, skip, SEEK_CUR);
				}
				size_t N_EVENT  = 0;
				hdr->EVENT.SampleRate = hdr->SampleRate;

				/* convert BDF status channel into event table*/
				uint32_t d1, d0;
				for (d0=0, k=0; k < len/3; d0 = d1, k++) {
					d1 = ((uint32_t)Marker[3*k+2]<<16) + ((uint32_t)Marker[3*k+1]<<8) + (uint32_t)Marker[3*k];

				/*	count raising edges */
					if (d1 & 0x010000)
						d1 = 0;
					else
						d1 &=  0x00ffff;

					if (d0 < d1) ++N_EVENT;

				/* 	raising and falling edges
					if ((d1 & 0x010000) != (d0 & 0x010000)) ++N_EVENT;
					if ((d1 & 0x00ffff) != (d0 & 0x00ffff)) ++N_EVENT;
				*/
				}


				hdr->EVENT.POS = (uint32_t*)realloc(hdr->EVENT.POS, (hdr->EVENT.N + N_EVENT) * sizeof(*hdr->EVENT.POS));
				hdr->EVENT.TYP = (uint16_t*)realloc(hdr->EVENT.TYP, (hdr->EVENT.N + N_EVENT) * sizeof(*hdr->EVENT.TYP));
				if (hdr->EVENT.POS==NULL || hdr->EVENT.TYP==NULL) {
					biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
					return (hdr);
				}

				for (d0=0, k=0; k < len/3; d0=d1, k++) {
					d1 = ((uint32_t)Marker[3*k+2]<<16) + ((uint32_t)Marker[3*k+1]<<8) + (uint32_t)Marker[3*k];

				/*	raising edges */
					if (d1 & 0x010000)
						d1  = 0;
					else
						d1 &= 0x00ffff;

					if (d0 < d1) {
						hdr->EVENT.POS[hdr->EVENT.N] = k;        // 0-based indexing
						hdr->EVENT.TYP[hdr->EVENT.N] = d1;
						++hdr->EVENT.N;
					}

				/* 	raising and falling edges
					if ((d1 & 0x010000) != (d0 & 0x010000)) {
						hdr->EVENT.POS[hdr->EVENT.N] = k;        // 0-based indexing
						hdr->EVENT.TYP[hdr->EVENT.N] = 0x7ffe;
						++hdr->EVENT.N;
					}

					if ((d1 & 0x00ffff) != (d0 & 0x00ffff)) {
						hdr->EVENT.POS[hdr->EVENT.N] = k;        // 0-based indexing
						uint16_t d2 = d1 & 0x00ffff;
						if (!d2) d2 = (uint16_t)(d0 & 0x00ffff) | 0x8000;
						hdr->EVENT.TYP[hdr->EVENT.N] = d2;
						++hdr->EVENT.N;
						if (d2==0x7ffe)
							fprintf(stdout,"Warning: BDF file %s uses ambiguous code 0x7ffe; For details see file eventcodes.txt. \n",hdr->FileName);
					}
				*/
				}
				free(Marker);

		}	/* End reading BDF Status channel */

		ifseek(hdr, hdr->HeadLen, SEEK_SET);
	}

	else if (hdr->TYPE==ABF) {
		hdr->HeadLen = count;
		sopen_abf_read(hdr);
	}

	else if (hdr->TYPE==ABF2) {
		hdr->HeadLen = count;
		sopen_abf2_read(hdr);
	}

	else if (hdr->TYPE==ATF) {
		// READ ATF
		hdr->HeadLen = count;
		hdr->VERSION = atof((char*)(hdr->AS.Header+4));

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) FMT=%s Ver=%4.2f\n",__FILE__,__LINE__,GetFileTypeString(hdr->TYPE),hdr->VERSION);

		if (hdr->FILE.COMPRESSION) {
			biosigERROR(hdr, B4C_DATATYPE_UNSUPPORTED, "compressed ATF file format not supported");
			return hdr;
		}
		sopen_atf_read(hdr);

	}

	else if (hdr->TYPE==ACQ) {

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s: try loading ACQ header \n", __FILE__, __LINE__, __func__);

		if ( !hdr->FILE.LittleEndian ) {
			hdr->NS = bei16p(hdr->AS.Header + 10);
			hdr->HeadLen += hdr->NS*1714;

			if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s (..): v%g noChan=%u HeadLen=%u\n", \
				__FILE__, __LINE__, __func__, hdr->VERSION, hdr->NS, hdr->HeadLen);

			biosigERROR(hdr, B4C_DATATYPE_UNSUPPORTED, "BigEndian ACQ file format is currently not supported");
			return(hdr);
		}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s \n", __FILE__, __LINE__, __func__);

		/* defined in http://biopac.com/AppNotes/app156FileFormat/FileFormat.htm */
		hdr->NS   = lei16p(hdr->AS.Header+10);
		hdr->SampleRate = 1000.0/lef64p(hdr->AS.Header+16);
		hdr->NRec = 1;
		hdr->SPR  = 1;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s \n", __FILE__, __LINE__, __func__);

		// add "per channel data section"
		if (hdr->VERSION<38.0)		// Version 3.0+
			hdr->HeadLen += hdr->NS*122;
		else if (hdr->VERSION<39.0)	// Version 3.7.0+
			hdr->HeadLen += hdr->NS*252;
		else if (hdr->VERSION<42.0)	// Version 3.7.3+
			hdr->HeadLen += hdr->NS*254;
		else if (hdr->VERSION<43.0)	// Version 3.7.3+
			hdr->HeadLen += hdr->NS*256;
		else 				// Version 3.8.2+
			hdr->HeadLen += hdr->NS*262;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s \n", __FILE__, __LINE__, __func__);

		hdr->HeadLen  += 4;
		// read header up to nLenght and nID of foreign data section
		hdr->AS.Header = (uint8_t*) realloc(hdr->AS.Header, hdr->HeadLen);
		if (hdr->HeadLen > count)
			count += ifread(Header1+count, 1, hdr->HeadLen-count, hdr);
		uint32_t POS   = hdr->HeadLen;
		// read "foreign data section" and "per channel data types section"
		hdr->HeadLen  += leu16p(hdr->AS.Header + hdr->HeadLen-4) - 4;

		// read "foreign data section" and "per channel data types section"
		hdr->HeadLen  += 4*hdr->NS;
		hdr->AS.Header = (uint8_t*)realloc(Header1, hdr->HeadLen+8);
		if (hdr->HeadLen > POS)
			count += ifread(Header1+POS, 1, hdr->HeadLen-POS, hdr);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s %i/%i %i/%i %i/%i %i/%i %i/%i \n", \
			__FILE__, __LINE__, __func__, \
			leu16p(hdr->AS.Header+hdr->HeadLen-20), leu16p(hdr->AS.Header+hdr->HeadLen-18), \
			leu16p(hdr->AS.Header+hdr->HeadLen-16), leu16p(hdr->AS.Header+hdr->HeadLen-14), \
			leu16p(hdr->AS.Header+hdr->HeadLen-12), leu16p(hdr->AS.Header+hdr->HeadLen-10), \
			leu16p(hdr->AS.Header+hdr->HeadLen-8),  leu16p(hdr->AS.Header+hdr->HeadLen-6), \
			leu16p(hdr->AS.Header+hdr->HeadLen-4),  leu16p(hdr->AS.Header+hdr->HeadLen-2) \
			);

		// define channel specific header information
		hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		uint32_t* ACQ_NoSamples = (uint32_t*) calloc(hdr->NS, sizeof(uint32_t));
		//uint16_t CHAN;
    		POS = leu32p(hdr->AS.Header+6);
    		size_t minBufLenXVarDiv = -1;	// maximum integer value
		for (k = 0; k < hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;

	    		uint8_t* Header2 = hdr->AS.Header+POS;
			hc->LeadIdCode = 0;
			hc->Transducer[0] = '\0';

			//CHAN = leu16p(Header2+4);
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s: #%i %i %i %i %i\n", \
			__FILE__, __LINE__, __func__, (int)k, (int)leu32p(Header2), (int)leu16p(Header2+4), (int)leu32p(Header2+88), (int)leu16p(Header2+250));

			int len=min(MAX_LENGTH_LABEL,40);
			strncpy(hc->Label,(char*)Header2+6,len);
			hc->Label[len]=0;

			char tmp[21];
			strncpy(tmp,(char*)Header2+68,20); tmp[20]=0;
			/* ACQ uses none-standard way of encoding physical units
			   Convert to ISO/IEEE 11073-10101 */
			if (!strcmp(tmp,"Volts"))
				hc->PhysDimCode = 4256;
			else if (!strcmp(tmp,"Seconds"))
				hc->PhysDimCode = 2176;
			else if (!strcmp(tmp,"deg C"))
				hc->PhysDimCode = 6048;
			else if (!strcmp(tmp,"microsiemen"))
				hc->PhysDimCode = 8307;
			else
				hc->PhysDimCode = PhysDimCode(tmp);

			hc->Off     = lef64p(Header2+52);
			hc->Cal     = lef64p(Header2+60);

			hc->OnOff   = 1;
			hc->SPR     = 1;
			if (hdr->VERSION >= 38.0) {
				hc->SPR  = leu16p(Header2+250);  // used here as Divider

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s: %i:%i\n", __FILE__, __LINE__, __func__, (int)hdr->SPR, (int)hc->SPR);

				if (hc->SPR > 1)
					hdr->SPR = lcm(hdr->SPR, hc->SPR);
				else
					hc->SPR  = 1;
			}

			ACQ_NoSamples[k] = leu32p(Header2+88);
			size_t tmp64 = leu32p(Header2+88) * hc->SPR;
			if (minBufLenXVarDiv > tmp64) minBufLenXVarDiv = tmp64;

			POS += leu32p((uint8_t*)Header2);
		}
		hdr->NRec = minBufLenXVarDiv;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s: POS=%i\n", __FILE__, __LINE__, __func__, POS);
		/// foreign data section - skip
		POS += leu16p(hdr->AS.Header+POS);
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s: POS=%i,%i\n", __FILE__, __LINE__, __func__, (int)POS, (int)count);
		if (POS+2 > count) {
			if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s: POS=%i,%i\n", __FILE__, __LINE__, __func__, (int)POS, (int)count);

			hdr->AS.Header = (uint8_t*) realloc(hdr->AS.Header, POS+2);
			count  += ifread(Header1+count, 1, POS+2-count, hdr);
		}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s: POS=%i\n", __FILE__, __LINE__, __func__, POS);

		size_t DataLen=0;
		for (k=0, hdr->AS.bpb=0; k<hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			if ((hdr->VERSION>=38.0) && (hc->SPR > 1))
				hc->SPR = hdr->SPR/hc->SPR;  // convert DIVIDER into SPR

			uint16_t u16 = leu16p(hdr->AS.Header+POS+2);
			switch (u16)	{
			case 1:
				hc->GDFTYP = 17;  // double
				DataLen   += ACQ_NoSamples[k]<<3;
				hc->DigMax =  1e9;
				hc->DigMin = -1e9;
				break;
			case 2:
				hc->GDFTYP = 3;   // int
				DataLen   += ACQ_NoSamples[k]<<1;
				hc->DigMax =  32767;
				hc->DigMin = -32678;
				break;
			default:

				if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s: #%i type=%i \n", __FILE__, __LINE__, __func__, (int)k, (int)u16);

				biosigERROR(hdr, B4C_UNSPECIFIC_ERROR, "SOPEN(ACQ-READ): invalid channel type.");
			};
			hc->PhysMax = hc->DigMax * hc->Cal + hc->Off;
			hc->PhysMin = hc->DigMin * hc->Cal + hc->Off;
			hc->bi 	  = hdr->AS.bpb;
		      	hdr->AS.bpb += (GDFTYP_BITS[hc->GDFTYP]*hc->SPR)>>3;
			POS +=4;
		}
		free(ACQ_NoSamples);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s: POS=%i,%i\n", __FILE__, __LINE__, __func__, (int)POS, (int)count);
		/// Markers header section - skip
		POS += leu16p(hdr->AS.Header+POS);
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s: POS=%i,%i\n", __FILE__, __LINE__, __func__, (int)POS, (int)count);
		if (POS+2 > count) {
			if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s: POS=%i,%i\n", __FILE__, __LINE__, __func__, (int)POS, (int)count);
			hdr->AS.Header = (uint8_t*) realloc(hdr->AS.Header, POS+2);
			count  += ifread(Header1+count, 1, POS+2-count, hdr);
		}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s: POS=%i,%i\n", __FILE__, __LINE__, __func__, (int)POS, (int)count);
		/// Markers header section - skip
		POS += leu16p(hdr->AS.Header+POS);
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s: POS=%i,%i\n", __FILE__, __LINE__, __func__, (int)POS, (int)count);
		if (POS+2 > count) {
			if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s: POS=%i,%i\n", __FILE__, __LINE__, __func__, (int)POS, (int)count);
			hdr->AS.Header = (uint8_t*) realloc(hdr->AS.Header, POS+2);
			count  += ifread(Header1+count, 1, POS+2-count, hdr);
		}
//		hdr->HeadLen = count + 1;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s: POS=%i HeadLen=%i\n", __FILE__, __LINE__, __func__, POS, (int)hdr->HeadLen);

/*  ### FIXME ###
	reading Marker section

	    	POS     = hdr->HeadLen;
#ifdef ZLIB_H
		gzseek(hdr->FILE.FID, hdr->HeadLen+DataLen, SEEK_SET); // start of markers header section
	    	count   = gzread(hdr->FILE.FID, Header1+POS, 8);
#else
		fseek(hdr->FILE.FID, hdr->HeadLen+DataLen, SEEK_SET); // start of markers header section
	    	count   = fread(Header1+POS, 1, 8, hdr->FILE.FID);
#endif
	    	size_t LengthMarkerItemSection = (leu32p(Header1+POS));

	    	hdr->EVENT.N = (leu32p(Header1+POS+4));
	    	Header1 = (char*)realloc(Header1,hdr->HeadLen+8+LengthMarkerItemSection);
	    	POS    += 8;
#ifdef ZLIB_H
	    	count   = gzread(hdr->FILE.FID, Header1+POS, LengthMarkerItemSection);
#else
	    	count   = fread(Header1+POS, 1, LengthMarkerItemSection, hdr->FILE.FID);
#endif
		hdr->EVENT.TYP = (uint16_t*)calloc(hdr->EVENT.N,2);
		hdr->EVENT.POS = (uint32_t*)calloc(hdr->EVENT.N,4);
		for (k=0; k<hdr->EVENT.N; k++)
		{
fprintf(stdout,"ACQ EVENT: %i POS: %i\n",k,POS);
			hdr->EVENT.POS[k] = leu32p(Header1+POS);
			POS += 12 + leu16p(Header1+POS+10);
		}
*/
		ifseek(hdr, hdr->HeadLen, SEEK_SET);
	}

	else if (hdr->TYPE==AINF) {
		ifclose(hdr);
		char *filename = hdr->FileName; // keep input file name
		char* tmpfile = (char*)calloc(strlen(hdr->FileName)+5,1);
		strcpy(tmpfile, hdr->FileName);			// Flawfinder: ignore
		char* ext = strrchr(tmpfile,'.')+1;

		/* open and read header file */
		strcpy(ext,"ainf");
		hdr->FileName = tmpfile;
		hdr = ifopen(hdr,"rb");
		count = 0;
		while (!ifeof(hdr)) {
			size_t bufsiz = max(2*count, PAGESIZE);
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, bufsiz+1);
			count  += ifread(hdr->AS.Header+count, 1, bufsiz-count, hdr);
		}
		hdr->AS.Header[count]=0;
		hdr->HeadLen = count;
		ifclose(hdr);

		char *t1= NULL;
		char *t = strtok((char*)hdr->AS.Header,"\xA\xD");
		while ((t) && !strncmp(t,"#",1)) {
			char* p;
			if ((p = strstr(t,"sfreq ="))) t1 = p;
			t = strtok(NULL,"\xA\xD");
		}

		hdr->SampleRate = atof(strtok(t1+7," "));
		hdr->SPR = 1;
		hdr->NS = 0;
		hdr->AS.bpb = 4;
		while (t) {
			int chno1=-1, chno2=-1;
			double f1,f2;
			char *label = NULL;

#if !defined __STDC_VERSION__ || __STDC_VERSION__ < 199901L
			sscanf(t,"%d %as %d %lf %lf",&chno1,&label,&chno2,&f1,&f2);
#else
			sscanf(t,"%d %ms %d %lf %lf",&chno1,&label,&chno2,&f1,&f2);
#endif

			k = hdr->NS++;
			hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL,hdr->NS*sizeof(CHANNEL_TYPE));
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			snprintf(hc->Label, MAX_LENGTH_LABEL+1, "%s %03i",label, chno2);

			hc->Transducer[0] = 0;
			hc->LeadIdCode = 0;
			hc->SPR    = 1;
			hc->Cal    = f1*f2;
			hc->Off    = 0.0;
			hc->OnOff  = 1;
			hc->GDFTYP = 3;
			hc->DigMax =  32767;
			hc->DigMin = -32678;
			hc->PhysMax= hc->DigMax * hc->Cal + hc->Off;
			hc->PhysMin= hc->DigMin * hc->Cal + hc->Off;
			hc->bi  = hdr->AS.bpb;
			hdr->AS.bpb += 2;

			if (strcmp(label,"MEG")==0)
				hc->PhysDimCode = 1446; // "T/m"
			else
				hc->PhysDimCode = 4256; // "V"

			if (label) free(label);
		 	t = strtok(NULL,"\x0a\x0d");
		}

		/* open data file */
		strcpy(ext,"raw");
		struct stat FileBuf;
		stat(hdr->FileName,&FileBuf);

		hdr = ifopen(hdr,"rb");
		hdr->NRec = FileBuf.st_size/hdr->AS.bpb;
		hdr->HeadLen   = 0;
		// hdr->FLAG.SWAP = (__BYTE_ORDER == __LITTLE_ENDIAN);  	// AINF is big endian
		hdr->FILE.LittleEndian = 0;
		/* restore input file name, and free temporary file name  */
		hdr->FileName = filename;
		free(tmpfile);
	}

    	else if (hdr->TYPE==alpha) {
		ifclose(hdr); 	// close already opened file (typically its .../alpha.alp)
		sopen_alpha_read(hdr);
	}

	else if (hdr->TYPE==AXG) {
		sopen_axg_read(hdr);
	}

    	else if (hdr->TYPE==Axona) {
    		fprintf(stdout, "Axona: alpha version. \n");

    		hdr->AS.bpb 	= 12 + 20 + 2 * 192 + 16;
    		hdr->NS 	=  4 + 64;
    		hdr->SPR 	=  3;
    		hdr->SampleRate = 24e3;
		struct stat FileBuf;
		if (stat(hdr->FileName, &FileBuf)==0) hdr->FILE.size = FileBuf.st_size;
    		hdr->NRec 	= hdr->FILE.size / hdr->AS.bpb;

		hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		CHANNEL_TYPE *hc;
		for (k = 0; k < hdr->NS; k++) {
			hc = hdr->CHANNEL + k;
			// hc->PhysDimCode = 4256; // "V"
			hc->PhysDimCode   = 0;
			hc->Transducer[0] = 0;

			hc->LeadIdCode = 0;
			hc->SPR        = hdr->SPR;
			hc->Cal        = 1.0;
			hc->Off        = 0.0;
			hc->OnOff      = 1;

			hc->LeadIdCode = 0;
			hc->Notch      = NAN;
			hc->LowPass    = NAN;
			hc->HighPass   = NAN;
		}

		for (k = 0; k+4 < hdr->NS; k++) {
			hc = hdr->CHANNEL + k + 4;
			sprintf(hc->Label, "#%02i", (int)k+1);
			hc->PhysDimCode = 4256; // "V"
			hc->PhysDimCode = 4274; // "mV"
			hc->PhysDimCode = 4275; // "uV"
			hc->GDFTYP  = 3;
			hc->DigMax  =  32767;
			hc->DigMin  = -32678;
			hc->bi      = 32 + 2*k;
		}

			hc = hdr->CHANNEL;
			strcpy(hc->Label, "PacketNumber");
			hc->SPR     = 1;
			hc->GDFTYP  = 6;	// uint32
			hc->DigMin  = 0.0;
			hc->DigMax  = ldexp(1.0, 32) - 1.0;
			hc->bi      = 4;

			hc = hdr->CHANNEL + 1;
			strcpy(hc->Label, "Digital I/O");
			hc->SPR     = 1;
			hc->GDFTYP  = 6;	// uint32
			hc->DigMin  = 0.0;
			hc->DigMax  = ldexp(1.0, 32) - 1.0;
			hc->bi      = 8;

			hc = hdr->CHANNEL + 2;
			strcpy(hc->Label, "FunKey");
			hc->SPR     = 1;
			hc->GDFTYP  = 2;	// uint8
			hc->DigMin  = 0.0;
			hc->DigMax  = 255.0;
			hc->bi      = 416;

			hc = hdr->CHANNEL + 3;
			strcpy(hc->Label, "Key Code");
			hc->SPR     = 1;
			hc->GDFTYP  = 2;	// uint8
			hc->DigMin  = 0.0;
			hc->DigMax  = 255.0;
			hc->bi 	    = 417;

		for (k = 0; k < hdr->NS; k++) {
			hc = hdr->CHANNEL + k;
			hc->PhysMax = hc->DigMax * hc->Cal + hc->Off;
			hc->PhysMin = hc->DigMin * hc->Cal + hc->Off;
		}

		hdr->HeadLen = 0;
		ifseek(hdr, 0, SEEK_SET);

		HDRTYPE H1;
		H1.FILE.COMPRESSION = 1; 	// try always with zlib, libz reverts to  no compression anyway.
		H1.FileName = malloc(strlen(hdr->FileName)+5);
		strcpy(H1.FileName, hdr->FileName); 		// Flawfinder: ignore
		char *e = strrchr(H1.FileName,'.');
		if (e==NULL) e = H1.FileName+strlen(H1.FileName);
		strcpy(e,".set");
		ifopen(&H1, "r");
		unsigned MaxLineLen = 1000;
		char *line = malloc(MaxLineLen);
		double PhysMax = NAN;
		//char* ifgets(char *str, int n, HDRTYPE* hdr) {
		while (!ifeof(&H1)) {
			ifgets(line, MaxLineLen, &H1);
			if (MaxLineLen <= strlen(line)) {
				fprintf(stderr,"Warning (Axona): line ## in file <%s> exceeds maximum length\n",H1.FileName);
			}
			char *tag = strtok(line," \t\n\r");
			char *val = strtok(NULL,"\n\r");

			if (tag==NULL || val==NULL)
				;
			else if (!strcmp(tag,"trial date"))
				;
			else if (!strcmp(tag,"trial time"))
				;
			else if (!strcmp(tag,"experimenter"))
				hdr->ID.Technician = strdup(val);
			else if (!strcmp(tag,"sw_version"))
				;
			else if (!strcmp(tag,"ADC_fullscale_mv")) {
				char *e;
				PhysMax = strtod(val, &e);
				if (e==NULL) continue; // ignore value because its invalid

			}
			else if (!strncmp(tag,"gain_ch_",4)) {
				char *e;
				size_t ch = strtol(tag+8, &e, 10);
				if (e==NULL || ch >= hdr->NS) continue; // ignore value because its invalid
				double Cal = strtod(val, &e);
				if (e==NULL) continue; // ignore value because its invalid

				hdr->CHANNEL[ch].Cal = 1.0/Cal;
				hdr->CHANNEL[ch].Off = 0.0;
				hdr->CHANNEL[ch].PhysMax = +PhysMax;
				hdr->CHANNEL[ch].PhysMin = -PhysMax;
				hdr->CHANNEL[ch].DigMax  = +PhysMax/Cal;
				hdr->CHANNEL[ch].DigMin  = -PhysMax/Cal;
			}
			else if (!strcmp(tag,"rawRate")) {
				char *e;
				double fs = strtod(val, &e);
				if (e==NULL) continue; // ignore value because its invalid
				hdr->SampleRate = fs;
			}
			else if (!strcmp(tag,"")) {
				;
			}
		}
		free(line);
		ifclose(&H1);
		free(H1.FileName);
	}

    	else if ((hdr->TYPE==ASCII) || (hdr->TYPE==BIN)) {
		while (!ifeof(hdr)) {
			size_t bufsiz = 65536;
		    	hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,count+bufsiz+1);
		    	count  += ifread(hdr->AS.Header+count,1,bufsiz,hdr);
		}
		hdr->AS.Header[count]=0;
		hdr->HeadLen = count;
		ifclose(hdr);

		hdr->NS   = 0;
		hdr->NRec = 1;
		hdr->SPR  = 1;
		hdr->AS.bpb = 0;
		double Fs = 1.0;
		size_t N  = 0;
		char status = 0;
		char *val   = NULL;
		const char sep[] = " =\x09";
		double duration = 0;
		size_t lengthRawData = 0;
		uint8_t FLAG_NUMBER_OF_FIELDS_READ;	// used to trigger consolidation of channel info
		CHANNEL_TYPE *cp = NULL;
		char *datfile = NULL;
		uint16_t gdftyp = 0;

		char *line  = strtok((char*)hdr->AS.Header,"\x0a\x0d");
		while (line!=NULL) {

			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"ASCII read line [%i]: <%s>\n",status,line);

			if (!strncmp(line,"[Header 1]",10))
				status = 1;
			else if (!strncmp(line,"[Header 2]",10)) {
				status = 2;
				hdr->NS = 0;
				FLAG_NUMBER_OF_FIELDS_READ=0;
			}
			else if (!strncmp(line,"[EVENT TABLE]",13)) {
				status = 3;
				hdr->EVENT.SampleRate = hdr->SampleRate;
				N = 0;
			}

			val = strchr(line,'=');
			if ((val != NULL) && (status<3)) {
				val += strspn(val,sep);
				size_t c;
				c = strspn(val,"#");
				if (c) val[c] = 0; // remove comments
				c = strcspn(line,sep);
				if (c) line[c] = 0; // deblank
				FLAG_NUMBER_OF_FIELDS_READ++;
			}
			if (VERBOSE_LEVEL>8) fprintf(stdout,"BIN <%s>=<%s> \n",line,val);

			if (status==1) {
				if (!strcmp(line,"Duration"))
					duration = atof(val);
				//else if (!strncmp(line,"NumberOfChannels"))
				else if (!strcmp(line,"Patient.Id"))
					strncpy(hdr->Patient.Id,val,MAX_LENGTH_PID);
				else if (!strcmp(line,"Patient.Birthday")) {
					struct tm t;
					sscanf(val,"%04i-%02i-%02i %02i:%02i:%02i",&t.tm_year,&t.tm_mon,&t.tm_mday,&t.tm_hour,&t.tm_min,&t.tm_sec);
					t.tm_year -=1900;
					t.tm_mon--;
					t.tm_isdst = -1;
					hdr->Patient.Birthday = tm_time2gdf_time(&t);
				}
				else if (!strcmp(line,"Patient.Weight"))
					hdr->Patient.Weight = atoi(val);
				else if (!strcmp(line,"Patient.Height"))
					hdr->Patient.Height = atoi(val);
				else if (!strcmp(line,"Patient.Gender"))
					hdr->Patient.Sex = atoi(val);
				else if (!strcmp(line,"Patient.Handedness"))
					hdr->Patient.Handedness = atoi(val);
				else if (!strcmp(line,"Patient.Smoking"))
					hdr->Patient.Smoking = atoi(val);
				else if (!strcmp(line,"Patient.AlcoholAbuse"))
					hdr->Patient.AlcoholAbuse = atoi(val);
				else if (!strcmp(line,"Patient.DrugAbuse"))
					hdr->Patient.DrugAbuse = atoi(val);
				else if (!strcmp(line,"Patient.Medication"))
					hdr->Patient.Medication = atoi(val);
				else if (!strcmp(line,"Recording.ID"))
					strncpy(hdr->ID.Recording,val,MAX_LENGTH_RID);
				else if (!strcmp(line,"Recording.Time")) {
					struct tm t;
					sscanf(val,"%04i-%02i-%02i %02i:%02i:%02i",&t.tm_year,&t.tm_mon,&t.tm_mday,&t.tm_hour,&t.tm_min,&t.tm_sec);
					t.tm_year -= 1900;
					t.tm_mon--;
					t.tm_isdst = -1;
					hdr->T0 = tm_time2gdf_time(&t);
				}
				else if (!strcmp(line,"Timezone")) {
					int m;
					if (sscanf(val,"%i min", &m) > 0)
						hdr->tzmin = m;
				}
				else if (!strcmp(line,"Recording.IPaddress")) {
#ifndef WITHOUT_NETWORK
#ifdef _WIN32
					WSADATA wsadata;
					WSAStartup(MAKEWORD(1,1), &wsadata);
#endif

					biosig_set_hdr_ipaddr(hdr, val);

#ifdef _WIN32
					WSACleanup();
#endif
#endif // not WITHOUT_NETWORK
				}
				else if (!strcmp(line,"Recording.Technician")) {
					if (hdr->ID.Technician) free(hdr->ID.Technician);
					hdr->ID.Technician = strdup(val);
				}
				else if (!strcmp(line,"Manufacturer.Name"))
					hdr->ID.Manufacturer.Name = val;
				else if (!strcmp(line,"Manufacturer.Model"))
					hdr->ID.Manufacturer.Model = val;
				else if (!strcmp(line,"Manufacturer.Version"))
					hdr->ID.Manufacturer.Version = val;
				else if (!strcmp(line,"Manufacturer.SerialNumber"))
					hdr->ID.Manufacturer.SerialNumber = val;
			}

			else if (status==2) {
				if (!strcmp(line,"Filename")) {

					// add next channel
					++hdr->NS;
					hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS*sizeof(CHANNEL_TYPE));
					cp = hdr->CHANNEL+hdr->NS-1;
					cp->Transducer[0] = 0;
					cp->bi = hdr->AS.bpb;
					cp->PhysDimCode = 0;
					cp->HighPass = NAN;
					cp->LowPass  = NAN;
					cp->Notch    = NAN;
					cp->Impedance= NAN;
					cp->fZ       = NAN;
					cp->LeadIdCode = 0;
					datfile      = val;

					FLAG_NUMBER_OF_FIELDS_READ = 1;

				}

				else if (!strcmp(line,"Label"))
					strncpy(cp->Label,val,MAX_LENGTH_LABEL);

				else if (!strcmp(line,"GDFTYP")) {
					if      (!strcmp(val,"int8"))	gdftyp = 1;
					else if (!strcmp(val,"uint8"))	gdftyp = 2;
					else if (!strcmp(val,"int16"))	gdftyp = 3;
					else if (!strcmp(val,"uint16"))	gdftyp = 4;
					else if (!strcmp(val,"int32"))	gdftyp = 5;
					else if (!strcmp(val,"uint32"))	gdftyp = 6;
					else if (!strcmp(val,"int64"))	gdftyp = 7;
					else if (!strcmp(val,"uint64"))	gdftyp = 8;
					else if (!strcmp(val,"float32"))	gdftyp = 16;
					else if (!strcmp(val,"float64"))	gdftyp = 17;
					else if (!strcmp(val,"float128"))	gdftyp = 18;
					else if (!strcmp(val,"ascii"))	gdftyp = 0xfffe;
					else 				gdftyp = atoi(val);

				}
				else if (!strcmp(line,"PhysicalUnits"))
					cp->PhysDimCode = PhysDimCode(val);
				else if (!strcmp(line,"PhysDimCode")) {
					// If PhysicalUnits and PhysDimCode conflict, PhysicalUnits gets the preference
					if (!cp->PhysDimCode)
						cp->PhysDimCode = atoi(val);
				}
				else if (!strcmp(line,"Transducer"))
					strncpy(cp->Transducer,val,MAX_LENGTH_TRANSDUCER);

				else if (!strcmp(line,"SamplingRate"))
					Fs = atof(val);

				else if (!strcmp(line,"NumberOfSamples")) {
					cp->SPR = atol(val);
					if (cp->SPR>0) hdr->SPR = lcm(hdr->SPR,cp->SPR);

					if ((gdftyp>0) && (gdftyp<256)) {
						cp->GDFTYP = gdftyp;

						FILE *fid = fopen(datfile,"rb");
						if (fid != NULL) {
							size_t bufsiz = (size_t)cp->SPR*GDFTYP_BITS[cp->GDFTYP]>>3;
							hdr->AS.rawdata = (uint8_t*) realloc(hdr->AS.rawdata,lengthRawData+bufsiz+1);
							count = fread(hdr->AS.rawdata+lengthRawData,1,bufsiz+1,fid);
							if (count != bufsiz)
								fprintf(stderr,"Warning SOPEN(BIN) #%i: mismatch between sample number and file size (%i,%i)\n",hdr->NS-1,(int)count,(int)bufsiz);
							lengthRawData += bufsiz;
							fclose(fid);
						}
						else if (cp->SPR > 0) {
							cp->SPR = 0;
							fprintf(stderr,"Warning SOPEN(BIN) #%i: data file (%s) not found\n",hdr->NS,datfile);
						}
					}
					else if (gdftyp==0xfffe) {
						cp->GDFTYP = 17;	// double

						struct stat FileBuf;
						stat(datfile, &FileBuf);

						FILE *fid = fopen(datfile,"rb");
						if (fid != NULL) {
							char *buf = (char*)malloc(FileBuf.st_size+1);
							count = fread(buf, 1, FileBuf.st_size, fid);
							fclose(fid);
							buf[count] = 0;

							size_t sz = GDFTYP_BITS[cp->GDFTYP]>>3;
							const size_t bufsiz = cp->SPR * sz;
							hdr->AS.rawdata = (uint8_t*) realloc(hdr->AS.rawdata, lengthRawData+bufsiz);

							char *bufbak  = buf; 	// backup copy
							char **endptr = &bufbak;
							for (k = 0; k < cp->SPR; k++) {
								double d = strtod(*endptr,endptr);
								*(double*)(hdr->AS.rawdata+lengthRawData+sz*k) = d;
							}
							lengthRawData += bufsiz;
							free(buf);
						}
						else if (cp->SPR > 0) {
							cp->SPR = 0;
							fprintf(stderr,"Warning SOPEN(BIN) #%i: data file (%s) not found\n",hdr->NS,datfile);
						}
					}
					else {
						biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "ASCII/BIN: data type unsupported");
					}
					hdr->AS.bpb  = lengthRawData;
				}
				else if (!strcmp(line,"HighPassFilter"))
					cp->HighPass = atof(val);
				else if (!strcmp(line,"LowPassFilter"))
					cp->LowPass = atof(val);
				else if (!strcmp(line,"NotchFilter"))
					cp->Notch = atof(val);
				else if (!strcmp(line,"DigMax"))
					cp->DigMax = atof(val);
				else if (!strcmp(line,"DigMin"))
					cp->DigMin = atof(val);
				else if (!strcmp(line,"PhysMax"))
					cp->PhysMax = atof(val);
				else if (!strcmp(line,"PhysMin"))
					cp->PhysMin = atof(val);
				else if (!strcmp(line,"Impedance"))
					cp->Impedance = atof(val);
				else if (!strcmp(line,"freqZ"))
					cp->fZ = atof(val);
				else if (!strncmp(line,"Position",8)) {
					sscanf(val,"%f \t%f \t%f",cp->XYZ,cp->XYZ+1,cp->XYZ+2);

					// consolidate previous channel
					if ((((size_t)cp->SPR*GDFTYP_BITS[cp->GDFTYP] >> 3) != (hdr->AS.bpb-cp->bi)) && (hdr->TYPE==BIN)) {
						fprintf(stdout,"Warning SOPEN(BIN): problems with channel %i - filesize %i does not fit header info %"PRIiPTR"\n",(int)k+1, hdr->AS.bpb-hdr->CHANNEL[k].bi, (GDFTYP_BITS[hdr->CHANNEL[k].GDFTYP]*(size_t)hdr->CHANNEL[k].SPR) >> 3);
					}

					hdr->SampleRate = hdr->SPR/duration;
					cp->LeadIdCode = 0;
					cp->OnOff = 1;
					cp->Cal = (cp->PhysMax - cp->PhysMin) / (cp->DigMax - cp->DigMin);
					cp->Off =  cp->PhysMin - cp->Cal*cp->DigMin;
				}
			}

			else if (status==3) {
				if (!strncmp(line,"0x",2)) {

					if (hdr->EVENT.N+1 >= N) {
						N = max(PAGESIZE, 2*N);
						if (N != reallocEventTable(hdr, N)) {
							biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
							return (hdr);
						};
					}

					val = line+2;
					int i;
					sscanf(val,"%04x",&i);
					if (i>0xffff)
						fprintf(stdout,"Warning: Type %i of event %i does not fit in 16bit\n",i,hdr->EVENT.N);
					else
						hdr->EVENT.TYP[hdr->EVENT.N] = (typeof(hdr->EVENT.TYP[0]))i;

					double d;
					val = strchr(val,'\t')+1;
					sscanf(val,"%lf",&d);

					hdr->EVENT.POS[hdr->EVENT.N] = (typeof(*hdr->EVENT.POS))round(d*hdr->EVENT.SampleRate);  // 0-based indexing
#if (BIOSIG_VERSION >= 10500)
					hdr->EVENT.TimeStamp[hdr->EVENT.N] = 0;
#endif

					val = strchr(val,'\t')+1;
					if (val[0]!='\t') {
						sscanf(val,"%lf",&d);
						hdr->EVENT.DUR[hdr->EVENT.N] = (typeof(*hdr->EVENT.POS))round(d*hdr->EVENT.SampleRate);
					}
					else
						hdr->EVENT.DUR[hdr->EVENT.N] = 0;

					val = strchr(val,'\t')+1;
					if (val[0]!='\t') {
						sscanf(val,"%d",&i);
						if (i>0xffff)
							fprintf(stdout,"Warning: channel number %i of event %i does not fit in 16bit\n",i,hdr->EVENT.N);
						else
							hdr->EVENT.CHN[hdr->EVENT.N] = i;
					}
					else
						hdr->EVENT.CHN[hdr->EVENT.N] = 0;

					val = strchr(val,'\t')+1;
					if ((hdr->EVENT.TYP[hdr->EVENT.N]==0x7fff) && (hdr->EVENT.CHN[hdr->EVENT.N]>0) && (!hdr->CHANNEL[hdr->EVENT.CHN[hdr->EVENT.N]-1].SPR)) {
						sscanf(val,"%d",&hdr->EVENT.DUR[hdr->EVENT.N]);
					}
					++hdr->EVENT.N;
				}
			}
			line = strtok(NULL,"\x0a\x0d");
		}
		hdr->AS.length = hdr->NRec;
    	}

	else if (hdr->TYPE==BCI2000) {
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) FMT=%s Ver=%4.2f\n",__FILE__,__LINE__,GetFileTypeString(hdr->TYPE),hdr->VERSION);

		char *ptr, *t1;

		/* decode header length */
		hdr->HeadLen = 0;
		ptr = strstr((char*)hdr->AS.Header,"HeaderLen=");
		if (ptr==NULL)
			biosigERROR(hdr, B4C_FORMAT_UNKNOWN, "not a BCI2000 format");
		else {
			/* read whole header */
			hdr->HeadLen = (typeof(hdr->HeadLen)) strtod(ptr+10,&ptr);
			if (count <= hdr->HeadLen) {
				hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, hdr->HeadLen+1);
				count   += ifread(hdr->AS.Header+count,1,hdr->HeadLen-count,hdr);
			}
			else
				ifseek(hdr,hdr->HeadLen,SEEK_SET);
		}
		hdr->AS.Header[hdr->HeadLen]=0;
		hdr->AS.bci2000 = (char*)realloc(hdr->AS.bci2000, hdr->HeadLen+1);
		memcpy(hdr->AS.bci2000, hdr->AS.Header, hdr->HeadLen+1);

		/* decode number of channels */
		t1  = strtok((char*)hdr->AS.Header,"\x0a\x0d");
		ptr = strstr(t1,"SourceCh=");
		if (ptr==NULL)
			biosigERROR(hdr, B4C_FORMAT_UNKNOWN, "not a BCI2000 format");
		else
			hdr->NS = (typeof(hdr->NS)) strtod(ptr+9,&ptr);

		/* decode length of state vector */
		ptr = strstr(t1,"StatevectorLen=");
		if (ptr==NULL)
			biosigERROR(hdr, B4C_FORMAT_UNKNOWN, "not a BCI2000 format");
		else
		    	BCI2000_StatusVectorLength = (size_t) strtod(ptr+15,&ptr);

		/* decode data format */
		ptr = strstr(ptr,"DataFormat=");
		uint16_t gdftyp=3;
		if (ptr == NULL) gdftyp = 3;
		else if (!strncmp(ptr+12,"int16",3))	gdftyp = 3;
		else if (!strncmp(ptr+12,"int32",5))	gdftyp = 5;
		else if (!strncmp(ptr+12,"float32",5))	gdftyp = 16;
		else if (!strncmp(ptr+12,"int24",5))	gdftyp = 255+24;
		else if (!strncmp(ptr+12,"uint16",3))	gdftyp = 4;
		else if (!strncmp(ptr+12,"uint32",5))	gdftyp = 6;
		else if (!strncmp(ptr+12,"uint24",5))	gdftyp = 511+24;
		else if (!strncmp(ptr+12,"float64",6))	gdftyp = 17;
		else biosigERROR(hdr, B4C_FORMAT_UNKNOWN, "SOPEN(BCI2000): invalid file format");

		if (hdr->AS.B4C_ERRNUM) {
			return(hdr);
		}

		if (hdr->FLAG.OVERFLOWDETECTION) {
			fprintf(stderr,"WARNING: Automated overflowdetection not supported in BCI2000 file %s\n",hdr->FileName);
			hdr->FLAG.OVERFLOWDETECTION = 0;
		}

		hdr->SPR = 1;
		double gain=0.0, offset=0.0, digmin=0.0, digmax=0.0;
		size_t tc_len=0,tc_pos=0, rs_len=0,rs_pos=0, fb_len=0,fb_pos=0;
		char TargetOrientation=0;

		hdr->AS.bpb = 0;
		hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL,hdr->NS*sizeof(CHANNEL_TYPE));
		for (k=0; k<hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			hc->Transducer[0] = 0;
			sprintf(hc->Label,"#%03i",(int)k+1);
			hc->Cal    = gain;
			hc->Off    = offset;
			hc->PhysDimCode = 4275; // uV
			hc->LeadIdCode = 0;
			hc->OnOff  = 1;
			hc->SPR    = 1;
			hc->GDFTYP = gdftyp;
			hc->bi     = hdr->AS.bpb;
			hdr->AS.bpb    += (GDFTYP_BITS[hc->GDFTYP] * (size_t)hc->SPR)>>3;
		}
		if (hdr->TYPE==BCI2000)
			hdr->AS.bpb += BCI2000_StatusVectorLength;

		int status = 0;
		ptr = strtok(NULL,"\x0a\x0d");
		while (ptr != NULL) {

			if (VERBOSE_LEVEL>8)
				fprintf(stdout,"[203] %i:  %s !\n",status,ptr);

			if (!strncmp(ptr,"[ State Vector Definition ]",26))
				status = 1;
			else if (!strncmp(ptr,"[ Parameter Definition ]",24))
				status = 2;
			else if (!strncmp(ptr,"[ ",2))
				status = 3;

			else if (status==1) {
				int  i[4];
				char *item = NULL;
#if !defined __STDC_VERSION__ || __STDC_VERSION__ < 199901L
				sscanf(ptr,"%as %i %i %i %i",&item,i,i+1,i+2,i+3);
#else
				sscanf(ptr,"%ms %i %i %i %i",&item,i,i+1,i+2,i+3);
#endif
				if (!strcmp(item,"TargetCode")) {
					tc_pos = i[2]*8 + i[3];
					tc_len = i[0];
				}
				else if (!strcmp(item,"ResultCode")) {
					rs_pos = i[2]*8 + i[3];
					rs_len = i[0];
				}
				else if (!strcmp(item,"Feedback")) {
					fb_pos = i[2]*8 + i[3];
					fb_len = i[0];
				}
				if (item) free(item);
			}

			else if (status==2) {
				t1 = strstr(ptr,"ChannelNames=");
				if (t1 != NULL) {
		    			unsigned NS = (unsigned)strtod(t1+13,&ptr);
		    			for (k=0; k<NS; k++) {
		    				while (isspace(ptr[0])) ++ptr;
		    				int k1=0;
		    				while (!isspace(ptr[k1])) ++k1;
		    				ptr += k1;
		    				if (k1>MAX_LENGTH_LABEL) k1=MAX_LENGTH_LABEL;
		    				strncpy(hdr->CHANNEL[k].Label,ptr-k1,k1);
						hdr->CHANNEL[k].Label[k1]=0; 	// terminating 0
		    			}
		    		}

				t1 = strstr(ptr,"SamplingRate=");
				if (t1 != NULL)	hdr->SampleRate = strtod(t1+14,&ptr);

				t1 = strstr(ptr,"SourceChGain=");
				if (t1 != NULL) {
		    			unsigned NS = (unsigned) strtod(t1+13,&ptr);
		    			for (k=0; k<NS; k++) hdr->CHANNEL[k].Cal = strtod(ptr,&ptr);
		    			for (; k<hdr->NS; k++) hdr->CHANNEL[k].Cal = hdr->CHANNEL[k-1].Cal;
		    		}
				t1 = strstr(ptr,"SourceChOffset=");
				if (t1 != NULL) {
		    			unsigned NS = (unsigned) strtod(t1+15,&ptr);
		    			for (k=0; k<NS; k++) hdr->CHANNEL[k].Off = strtod(ptr,&ptr);
		    			for (; k<hdr->NS; k++) hdr->CHANNEL[k].Off = hdr->CHANNEL[k-1].Off;
		    		}
				t1 = strstr(ptr,"SourceMin=");
				if (t1 != NULL)	digmin = strtod(t1+10,&ptr);

				t1 = strstr(ptr,"SourceMax=");
				if (t1 != NULL) digmax = strtod(t1+10,&ptr);

				t1 = strstr(ptr,"StorageTime=");
				if (t1 != NULL) {
					char *t2 = strstr(t1,"%20");
					while (t2!=NULL) {
						memset(t2,' ',3);
						t2 = strstr(t1,"%20");
					}

					char tmp[20];
					int c=sscanf(t1+12,"%03s %03s %2u %2u:%2u:%2u %4u",tmp+10,tmp,&tm_time.tm_mday,&tm_time.tm_hour,&tm_time.tm_min,&tm_time.tm_sec,&tm_time.tm_year);
					if (c==7) {
						tm_time.tm_isdst = -1;
						tm_time.tm_year -= 1900;
						tm_time.tm_mon   = month_string2int(tmp);
						hdr->T0 = tm_time2gdf_time(&tm_time);
					}
				}
				t1 = strstr(ptr,"TargetOrientation=");
				if (t1 != NULL)	TargetOrientation = (char) strtod(t1+18, &ptr);

			// else if (status==3);

			}
			ptr = strtok(NULL,"\x0a\x0d");
		}

		for (k=0; k<hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			hc->DigMax = digmax;
			hc->DigMin = digmin;
			hc->PhysMax= hc->DigMax * hc->Cal + hc->Off;
			hc->PhysMin= hc->DigMin * hc->Cal + hc->Off;
		}
		hdr->AS.bpb = (hdr->NS * (GDFTYP_BITS[gdftyp]>>3) + BCI2000_StatusVectorLength);

		/* decode state vector into event table */
		hdr->EVENT.SampleRate = hdr->SampleRate;
	        size_t skip = hdr->NS * (GDFTYP_BITS[gdftyp]>>3);
	        size_t N = 0;
	        count 	 = 0;
	        uint8_t *StatusVector = (uint8_t*) malloc(BCI2000_StatusVectorLength*2);
		uint32_t b0=0,b1=0,b2=0,b3,b4=0,b5;
	        while (!ifeof(hdr)) {
		        ifseek(hdr, skip, SEEK_CUR);
			ifread(StatusVector + BCI2000_StatusVectorLength*(count & 1), 1, BCI2000_StatusVectorLength, hdr);
			if (memcmp(StatusVector, StatusVector+BCI2000_StatusVectorLength, BCI2000_StatusVectorLength)) {
				if (N+4 >= hdr->EVENT.N) {
					hdr->EVENT.N  += 1024;
					if (SIZE_MAX == reallocEventTable(hdr, hdr->EVENT.N)) {
						biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
						return (hdr);
					};
				}

#if (BIOSIG_VERSION >= 10500)
				hdr->EVENT.TimeStamp[N] = 0;
#endif
				/*
					event codes according to
					http://www.bci2000.org/wiki/index.php/User_Reference:GDFFileWriter
					http://biosig.cvs.sourceforge.net/biosig/biosig/doc/eventcodes.txt?view=markup
				*/

				/* decode ResultCode */
				b3 = *(uint32_t*)(StatusVector + BCI2000_StatusVectorLength*(count & 1) + (rs_pos>>3));
				b3 = (b3 >> (rs_pos & 7)) & ((1<<rs_len)-1);
				if (b3 != b2) {
					if (b3>b2) hdr->EVENT.TYP[N] = ( b3==b1 ? 0x0381 : 0x0382);
					else 	   hdr->EVENT.TYP[N] = ( b2==b0 ? 0x8381 : 0x8382);
					hdr->EVENT.POS[N] = count;        // 0-based indexing
					N++;
					b2 = b3;
				}

				/* decode TargetCode */
				b1 = *(uint32_t*)(StatusVector + BCI2000_StatusVectorLength*(count & 1) + (tc_pos>>3));
				b1 = (b1 >> (tc_pos & 7)) & ((1<<tc_len)-1);
				if (b1 != b0) {
					if (TargetOrientation==1) {	// vertical
						switch ((int)b1-(int)b0) {
						case  1: hdr->EVENT.TYP[N] = 0x030c; break;
						case  2: hdr->EVENT.TYP[N] = 0x0306; break;
						case -1: hdr->EVENT.TYP[N] = 0x830c; break;
						case -2: hdr->EVENT.TYP[N] = 0x8306; break;
						default:
							if (b1>b0) hdr->EVENT.TYP[N] = 0x0300 + b1 - b0;
							else       hdr->EVENT.TYP[N] = 0x8300 + b0 - b1;
						}
					}
					else {
						if (b1>b0) hdr->EVENT.TYP[N] = 0x0300 + b1 - b0;
						else       hdr->EVENT.TYP[N] = 0x8300 + b0 - b1;
					}

					hdr->EVENT.POS[N] = count;        // 0-based indexing
					N++;
					b0 = b1;
				}

				/* decode Feedback */
				b5 = *(uint32_t*)(StatusVector + BCI2000_StatusVectorLength*(count & 1) + (fb_pos>>3));
				b5 = (b5 >> (fb_pos & 7)) & ((1<<fb_len)-1);
				if (b5 > b4)
					hdr->EVENT.TYP[N] = 0x030d;
				else if (b5 < b4)
					hdr->EVENT.TYP[N] = 0x830d;
				if (b5 != b4) {
					hdr->EVENT.POS[N] = count;        // 0-based indexing
					N++;
					b4 = b5;
				}
			}
			count++;
		}
		hdr->EVENT.N = N;
		free(StatusVector);
		hdr->NRec = (iftell(hdr) - hdr->HeadLen) / hdr->AS.bpb;
	        ifseek(hdr, hdr->HeadLen, SEEK_SET);

		if (VERBOSE_LEVEL>8) fprintf(stdout,"[209] header finished!\n");
	}

	else if (hdr->TYPE==BKR) {

		if (VERBOSE_LEVEL>8) fprintf(stdout,"libbiosig/sopen (BKR)\n");

	    	hdr->HeadLen 	 = 1024;
	    	hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, hdr->HeadLen);
		if (hdr->HeadLen > count)
			count   += ifread(hdr->AS.Header+count, 1, hdr->HeadLen-count, hdr);
		hdr->NS  	 = leu16p(hdr->AS.Header+2);
		hdr->NRec   	 = leu32p(hdr->AS.Header+6);
		hdr->SPR  	 = leu32p(hdr->AS.Header+10);
		hdr->NRec 	*= hdr->SPR;
		hdr->SPR  	 = 1;
		hdr->T0 	 = 0;        // Unknown;
		hdr->SampleRate	 = leu16p(hdr->AS.Header+4);

	    	/* extract more header information */
	    	hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL,hdr->NS*sizeof(CHANNEL_TYPE));
		for (k=0; k<hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			sprintf(hc->Label,"# %02i",(int)k);
			hc->Transducer[0] = '\0';
		    	hc->GDFTYP 	= 3;
		    	hc->SPR 	= 1; // *(int32_t*)(Header1+56);
		    	hc->LowPass	= lef32p(hdr->AS.Header+22);
		    	hc->HighPass = lef32p(hdr->AS.Header+26);
		    	hc->Notch	= -1.0;  // unknown
		    	hc->PhysMax	= (double)leu16p(hdr->AS.Header+14);
		    	hc->DigMax	= (double)leu16p(hdr->AS.Header+16);
		    	hc->PhysMin	= -hc->PhysMax;
		    	hc->DigMin	= -hc->DigMax;
		    	hc->Cal	 	= hc->PhysMax/hc->DigMax;
		    	hc->Off	 	= 0.0;
			hc->OnOff    	= 1;
		    	hc->PhysDimCode = 4275; // uV
		    	hc->LeadIdCode  = 0;
		    	hc->bi      	= k*2;
		}
		hdr->AS.bpb = hdr->NS*2;
		hdr->FLAG.OVERFLOWDETECTION = 0; 	// BKR does not support automated overflow and saturation detection
	}

	else if (hdr->TYPE==BLSC) {
		hdr->HeadLen = hdr->AS.Header[1]<<7;
		if (count<hdr->HeadLen) {
		    	hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, hdr->HeadLen);
		    	count   += ifread(hdr->AS.Header+count,1,hdr->HeadLen-count,hdr);
		}

		hdr->VERSION  = leu16p(hdr->AS.Header+2)/100.0;
		hdr->SampleRate = 128;
		hdr->SPR = 1;
		hdr->NS  = hdr->AS.Header[346];

		const uint32_t GAIN[] = {
			0,50000,75000,100000,150000,200000,250000,300000,  //0-7
			0,5000,7500,10000,15000,20000,25000,30000,  //8-15
			0,500,750,1000,1500,2000,2500,3000,  //16-23
			10,50,75,100,150,200,250,300  //24-31
			};

	    	hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL,hdr->NS*sizeof(CHANNEL_TYPE));
		for (k=0; k<hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			hc->Label[0] 	= 0;
			hc->Transducer[0] = '\0';
		    	hc->GDFTYP 	= 2;
		    	hc->SPR 	= hdr->SPR; // *(int32_t*)(Header1+56);
		    	hc->LowPass	= -1.0;
		    	hc->HighPass 	= -1.0;
    			hc->Notch	= -1.0;  // unknown
		    	hc->DigMax	= 255;
		    	hc->DigMin	= 0;

#define SENS  	leu16p(hdr->AS.Header+467)
#define CALUV 	leu16p(hdr->AS.Header+469)
#define CV 	hdr->AS.Header[425+k]
#define DC 	hdr->AS.Header[446+k]
#define gain 	GAIN[hdr->AS.Header[602+k]]

if (VERBOSE_LEVEL>8)
	fprintf(stdout,"#%i sens=%i caluv=%i cv=%i dc=%i Gain=%i\n",(int)k,SENS,CALUV,CV,DC,gain);

			double cal, off;
		    	if (hdr->AS.Header[5]==0) {
		    		// external amplifier
				cal = 0.2*CALUV*SENS/CV;
				off = -DC*cal;
		    	}
		    	else {
		    		// internal amplifier
			    	cal = 4e6/(CV*gain);
				off = -(128+(DC-128)*gain/3e5)*cal;
			}

#undef SENS
#undef CALUV
#undef CV
#undef DC
#undef gain

		    	hc->Cal	 = cal;
		    	hc->Off	 = off;
		    	hc->PhysMax	 = hc->DigMax * cal + off;
		    	hc->PhysMin	 = hc->DigMin * cal + off;
			hc->OnOff    = 1;
		    	hc->PhysDimCode = 4275; // uV
    			hc->LeadIdCode  = 0;
			hc->bi 	= k*hc->SPR*(GDFTYP_BITS[2]>>3);
		}
		hdr->AS.bpb     = hdr->NS*hdr->SPR*(GDFTYP_BITS[2]>>3);

		struct stat FileBuf;
		stat(hdr->FileName,&FileBuf);
		hdr->NRec = FileBuf.st_size/hdr->NS;
	        ifseek(hdr, hdr->HeadLen, SEEK_SET);

	}

	else if (hdr->TYPE==WFT) {
		// WFT/Nicolet
		hdr->HeadLen = atol((char*)hdr->AS.Header+8);
		if (count<hdr->HeadLen) {
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, hdr->HeadLen);
			count   += ifread(hdr->AS.Header+count,1,hdr->HeadLen-count,hdr);
		}
		uint16_t gdftyp=3;  // int16_t

		// File_size
		char *next = strchr(hdr->AS.Header+8,0)+1;
		while (*next==32) next++;
		hdr->FILE.size = atol(next);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// File format version
		next = strchr(next,0)+1;
		while (*next==32) next++;
		hdr->VERSION = atol(next);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		hdr->NS   = 1;
		hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL,hdr->NS*sizeof(CHANNEL_TYPE));
		CHANNEL_TYPE *hc = hdr->CHANNEL;
		hc->OnOff = 1;
		hc->LeadIdCode = 0;
		hc->Transducer[0]=0;
		hc->TOffset = 0.0;
		hc->LowPass = 0.0;
		hc->HighPass = 0.0;
		hc->Notch   = 0.0;
		hc->XYZ[0]  = 0;
		hc->XYZ[1]  = 0;
		hc->XYZ[2]  = 0;
		hc->Impedance = NAN;
		hc->bi = 0;
		hc->bi8 = 0;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// Waveform title
		next = strchr(next,0)+1;
		while (*next==32) next++;
		memcpy(hc->Label, next, MAX_LENGTH_LABEL+1);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		struct tm T0;
		T0.tm_sec=0;
		T0.tm_min=0;
		T0.tm_hour=0;
		// date year
		next = strchr(next,0)+1;
		while (*next==32) next++;
		T0.tm_year = atol(next);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// date month
		next = strchr(next,0)+1;
		while (*next==32) next++;
		T0.tm_mon = atol(next)-1;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// date day
		next = strchr(next,0)+1;
		while (*next==32) next++;
		T0.tm_mday = atol(next);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// milliseconds since midnight
		next = strchr(next,0)+1;
		while (*next==32) next++;
		long msec = atol(next);
		T0.tm_sec = msec / 1000;
		msec      = msec % 1000;
		hdr->T0 = tm_time2gdf_time(&T0) + ldexp(msec/(3600*24*1000.0),32) ;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// date day
		next = strchr(next,0)+1;
		while (*next==32) next++;
		hdr->SPR  = 1;
		hdr->NRec = atol(next);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		hc->Transducer[0] = '\0';
		hc->GDFTYP 	= 3;	// int16
		hc->SPR 	= hdr->SPR; //
		hc->LowPass	= -1.0;
		hc->HighPass 	= -1.0;
		hc->Notch	= -1.0;  // unknown
		hc->DigMax	= 32767;
		hc->DigMin	= -32768;

		hc->OnOff    	= 1;
		hc->PhysDimCode = 4275; // uV
		hc->LeadIdCode  = 0;

		// vertical zero
		next = strchr(next,0)+1;
		while (*next==32) next++;
		int vz = atol(next);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// vertical norm
		next = strchr(next,0)+1;
		while (*next==32) next++;
		double vn = atof(next);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// user vertical zero
		next = strchr(next,0)+1;
		while (*next==32) next++;
		double uvz = atof(next);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// user vertical norm
		next = strchr(next,0)+1;
		while (*next==32) next++;
		double uvn = atof(next);
		hc->Cal = vn*uvn;
		hc->Off = uvz - vz*vn*uvn;
		hc->PhysMax	= hc->DigMax * hc->Cal + hc->Off;
		hc->PhysMin	= hc->DigMin * hc->Cal + hc->Off;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// user vertical label
		next = strchr(next,0)+1;
		while (*next==32) next++;
		const char *physdim = next;
		if (!memcmp(next,"V ",2))
			hc->PhysDimCode = 4256;
		else
			hc->PhysDimCode = PhysDimCode(next);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// user horizontal zero
		next = strchr(next,0)+1;
		while (*next==32) next++;
		hdr->SampleRate = 1.0/atof(next);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// user horizontal norm
		next = strchr(next,0)+1;
		while (*next==32) next++;
		double uhn = atof(next);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// user horizontal label
		next = strchr(next,0)+1;
		while (*next==32) next++;
		const char *my_physdim = next;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// user notes
		next = strchr(next,0)+1;
		while (*next==32) next++;
		const char *user_notes = next;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// audit
		next = strchr(next,0)+1;
		while (*next==32) next++;
		const char *audit = next;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// nicolet_digitizer_type
		next = strchr(next,0)+1;
		while (*next==32) next++;
		const char nicolet_digitizer_type = *next;
		strcpy(hdr->ID.Manufacturer._field, "Nicolet");
		strcpy(hdr->ID.Manufacturer._field+8, next);
		hdr->ID.Manufacturer.Name  = hdr->ID.Manufacturer._field;
		hdr->ID.Manufacturer.Model = hdr->ID.Manufacturer._field+8;
		hdr->ID.Manufacturer.Version = NULL;
		hdr->ID.Manufacturer.SerialNumber = NULL;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// bytes per data point
		next = strchr(next,0)+1;
		while (*next==32) next++;
		hdr->AS.bpb = atol(next);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		// resolution
		next = strchr(next,0)+1;
		while (*next==32) next++;
		int resolution = atol(next);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s line %d: %s(...) <%s>:\n", __FILE__,__LINE__,__func__,next);

		int k=24;
		hdr->SampleRate = 200;	// unknown
		while (next < (char*)(hdr->AS.Header + hdr->HeadLen)) {

			if (k==189) hdr->SampleRate = 1.0/atof(next);

			if (VERBOSE_LEVEL>8) fprintf(stdout,"%s line %d: %s(...) : %d <%s>\n", __FILE__,__LINE__,__func__, k, next);
			next = strchr(next,0)+1;
			while (*next==32) next++;
			k++;
		}
	}

	else if (hdr->TYPE==BiosigDump) {
	        hdr->HeadLen = count;
		sopen_biosigdump_read(hdr);
	}

	else if (hdr->TYPE==BNI) {
		// BNI-1-Baltimore/Nicolet
		char *line = strtok((char*)hdr->AS.Header,"\x0a\x0d");
		fprintf(stderr,"Warning SOPEN: BNI not implemented - experimental code!\n");
		double cal=0,age;
		char *Label=NULL;
		struct tm t;
		while (line != NULL) {
			size_t c1 = strcspn(line," =");
			size_t c2 = strspn(line+c1," =");
			char *val = line+c1+c2;
			if (!strncmp(line,"PatientId",9))
				strncpy(hdr->Patient.Id,val,MAX_LENGTH_PID);
			else if (!strncasecmp(line,"Sex",3))
				hdr->Patient.Sex = 1*(toupper(val[0])=='M')+2*(toupper(val[0])=='F');
			else if (!strncasecmp(line,"medication",11))
				hdr->Patient.Medication = val==NULL ? 1 : 2;
			else if (!strncasecmp(line,"diagnosis",10)) {
			}
			else if (!strncasecmp(line,"MontageRaw",9))
				Label = val;
			else if (!strncasecmp(line,"Age",3))
				age = atol(val);
			else if (!strncasecmp(line,"Date",c1))
				sscanf(val,"%02i/%02i/%02i",&t.tm_mon,&t.tm_mday,&t.tm_year);
			else if (!strncasecmp(line,"Time",c1))
				sscanf(val,"%02i:%02i:%02i",&t.tm_hour,&t.tm_min,&t.tm_sec);
			else if (!strncasecmp(line,"Rate",c1))
				hdr->SampleRate = atol(val);
			else if (!strncasecmp(line,"NchanFile",9))
				hdr->NS = atol(val);
			else if (!strncasecmp(line,"UvPerBit",c1))
				cal = atof(val);
			else if (!strncasecmp(line,"[Events]",c1)) {
				// not implemented yet
			}
			else
				fprintf(stdout,"SOPEN(BNI): unknown field %s=%s\n",line,val);

			line = strtok(NULL,"\x0a\x0d");
		}
		hdr->T0 = tm_time2gdf_time(&t);
	    	hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL,hdr->NS*sizeof(CHANNEL_TYPE));
		for (k=0; k<hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			if (!k) strncpy(hc->Label, strtok(Label,","),MAX_LENGTH_LABEL);
			else 	strncpy(hc->Label, strtok(NULL,","),MAX_LENGTH_LABEL);

			hc->Transducer[0] = '\0';
		    	hc->GDFTYP 	= 0xffff;	// unknown - triggers error status
		    	hc->SPR 	= 1; //
		    	hc->LowPass	= -1.0;
		    	hc->HighPass 	= -1.0;
    			hc->Notch	= -1.0;  // unknown
		    	hc->DigMax	= 32767;
		    	hc->DigMin	= -32768;

		    	hc->Cal	 	= cal;
		    	hc->Off	 	= 0.0;
		    	hc->PhysMax	= hc->DigMax * cal;
		    	hc->PhysMin	= hc->DigMin * cal;
			hc->OnOff    	= 1;
		    	hc->PhysDimCode = 4275; // uV
    			hc->LeadIdCode  = 0;
    			//hc->bi 	= k*GDFTYP_BITS[hc->GDFTYP]>>3;
		}
	}

	else if (hdr->TYPE==BrainVisionMarker) {

		while (!ifeof(hdr)) {
			size_t bufsiz  = max(count*2, PAGESIZE);
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,bufsiz+1);
			count += ifread(hdr->AS.Header+count,1,bufsiz-count,hdr);
		}
		hdr->AS.Header[count]=0;
		hdr->HeadLen = count;
		ifclose(hdr);

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"SOPEN(BV): marker file read.\n");

		int seq = 0;
		/* decode marker file */


		char *t,*t1="    ";
		t  = Header1;
		t += strcspn(Header1,"\x0A\x0D");
		t += strspn(t,"\x0A\x0D");
		//char *t1 = strtok(Header1,"\x0A\x0D");
		// skip first line
		size_t N_EVENT=0;
		hdr->EVENT.N=0;
		do {
			t1 = t;
			t += strcspn(t,"\x0A\x0D");
			t += strspn(t,"\x0A\x0D");
			t[-1]=0;

			if (VERBOSE_LEVEL>8) fprintf(stdout,"%i <%s>\n",seq,t1);

			if (!strncmp(t1,";",1))
				;
			else if (!strncmp(t1,"[Common Infos]",14))
				seq = 1;
			else if (!strncmp(t1,"[Marker Infos]",14))
				seq = 2;

			else if (seq==1)
				;
			else if ((seq==2) && !strncmp(t1,"Mk",2)) {
				int p1 = strcspn(t1,"=");
				int p2 = p1 + 1 + strcspn(t1+p1+1,",");
				int p3 = p2 + 1 + strcspn(t1+p2+1,",");
				int p4 = p3 + 1 + strcspn(t1+p3+1,",");
				int p5 = p4 + 1 + strcspn(t1+p4+1,",");
				int p6 = p5 + 1 + strcspn(t1+p5+1,",");

			if (VERBOSE_LEVEL>8) fprintf(stdout,"  %i %i %i %i %i %i \n",p1,p2,p3,p4,p5,p6);

				t1[p1]=0;
				t1[p2]=0;
				t1[p3]=0;
				t1[p4]=0;
				t1[p5]=0;

				if (hdr->EVENT.N <= N_EVENT) {
					hdr->EVENT.N  += 256;
					if (reallocEventTable(hdr, hdr->EVENT.N) == SIZE_MAX) {
						biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
						return (hdr);
					};
				}
				hdr->EVENT.TYP[N_EVENT] = atol(t1+p2+2);
				hdr->EVENT.POS[N_EVENT] = atol(t1+p3+1)-1;        // 0-based indexing
				hdr->EVENT.DUR[N_EVENT] = atol(t1+p4+1);
				hdr->EVENT.CHN[N_EVENT] = atol(t1+p5+1);
#if (BIOSIG_VERSION >= 10500)
				hdr->EVENT.TimeStamp[N_EVENT] = 0;
#endif
				if (!strncmp(t1+p1+1,"New Segment",11)) {
					hdr->EVENT.TYP[N_EVENT] = 0x7ffe;

					char* t2 = t1+p6+1;
					t2[14]=0;	tm_time.tm_sec  = atoi(t2+12);
					t2[12]=0;	tm_time.tm_min  = atoi(t2+10);
					t2[10]=0;	tm_time.tm_hour = atoi(t2+8);
					t2[8] =0;	tm_time.tm_mday = atoi(t2+6);
					t2[6] =0;	tm_time.tm_mon  = atoi(t2+4)-1;
					t2[4] =0;	tm_time.tm_year = atoi(t2)-1900;
					hdr->T0 = tm_time2gdf_time(&tm_time);
				}
				else {
					if (VERBOSE_LEVEL>8) fprintf(stdout,"#%02i <%s>\n",(int)N_EVENT,t1+p2+1);
					FreeTextEvent(hdr,N_EVENT,t1+p2+1);
				}

				++N_EVENT;
			}
		}
		while (strlen(t1)>0);

		// free(vmrk);
		hdr->AS.auxBUF = hdr->AS.Header;
		hdr->AS.Header = NULL;
		hdr->EVENT.N   = N_EVENT;
		hdr->TYPE      = EVENT;

	}

	else if ((hdr->TYPE==BrainVision) || (hdr->TYPE==BrainVisionVAmp)) {
		/* open and read header file */
		// ifclose(hdr);
		char *filename = hdr->FileName; // keep input file name
		char* tmpfile = (char*)calloc(strlen(hdr->FileName)+5,1);
		strcpy(tmpfile, hdr->FileName);		// Flawfinder: ignore
		hdr->FileName = tmpfile;
		char* ext = strrchr((char*)hdr->FileName,'.')+1;

		while (!ifeof(hdr)) {
			size_t bufsiz = max(2*count, PAGESIZE);
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, bufsiz+1);
			count  += ifread(hdr->AS.Header+count, 1, bufsiz-count, hdr);
		}
		hdr->AS.Header[count]=0;
		hdr->HeadLen = count;
		ifclose(hdr);

		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"SOPEN(BV): header file read.\n");

		int seq = 0;

		/* decode header information */
		hdr->FLAG.OVERFLOWDETECTION = 0;
		seq = 0;
		uint16_t gdftyp=3;
		char FLAG_ASCII = 0;
		hdr->FILE.LittleEndian = 1;	// default little endian
		double physmax=1e6,physmin=-1e6,digmax=1e6,digmin=-1e6,cal=1.0,off=0.0;
		enum o_t{VEC,MUL} orientation = MUL;
		char DECIMALSYMBOL='.';
		int  SKIPLINES=0, SKIPCOLUMNS=0;
		size_t npts=0;

		char *t;
		size_t pos;
		// skip first line with <CR><LF>
		const char EOL[] = "\r\n";
		pos  = strcspn(Header1,EOL);
		pos += strspn(Header1+pos,EOL);
		while (pos < hdr->HeadLen) {
			t    = Header1+pos;	// start of line
			pos += strcspn(t,EOL);
			Header1[pos] = 0;	// line terminator
			pos += strspn(Header1+pos+1,EOL)+1; // skip <CR><LF>

			if (VERBOSE_LEVEL>7) fprintf(stdout,"[212]: %i pos=%i <%s>, ERR=%i\n",seq,(int)pos,t,hdr->AS.B4C_ERRNUM);

			if (!strncmp(t,";",1)) 	// comments
				;
			else if (!strncmp(t,"[Common Infos]",14))
				seq = 1;
			else if (!strncmp(t,"[Binary Infos]",14))
				seq = 2;
			else if (!strncmp(t,"[ASCII Infos]",13)) {
				seq = 2;
				FLAG_ASCII = 1;
				gdftyp = 17;

//				biosigERROR(hdr, B4C_DATATYPE_UNSUPPORTED, "Error SOPEN(BrainVision): ASCII-format not supported (yet).");
			}
			else if (!strncmp(t,"[Channel Infos]",14)) {
				seq = 3;

				/* open data file */
				if (FLAG_ASCII) hdr = ifopen(hdr,"rt");
				else 	        hdr = ifopen(hdr,"rb");

				hdr->AS.bpb = (hdr->NS*GDFTYP_BITS[gdftyp])>>3;
				if (hdr->TYPE==BrainVisionVAmp) hdr->AS.bpb += 4;
				if (!npts) {
					struct stat FileBuf;
					stat(hdr->FileName,&FileBuf);
					npts = FileBuf.st_size/hdr->AS.bpb;
		        	}

				/* restore input file name, and free temporary file name  */
				hdr->FileName = filename;
				free(tmpfile);

				if (orientation == VEC) {
					hdr->SPR = npts;
					hdr->NRec= 1;
					hdr->AS.bpb*= hdr->SPR;
				} else {
					hdr->SPR = 1;
					hdr->NRec= npts;
				}

			    	hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL,hdr->NS*sizeof(CHANNEL_TYPE));
				for (k=0; k<hdr->NS; k++) {
					CHANNEL_TYPE *hc = hdr->CHANNEL+k;
					hc->Label[0] = 0;
					hc->Transducer[0] = '\0';
				    	hc->GDFTYP 	= gdftyp;
				    	hc->SPR 	= hdr->SPR; // *(int32_t*)(Header1+56);
				    	hc->LowPass	= -1.0;
				    	hc->HighPass	= -1.0;
		    			hc->Notch	= -1.0;  // unknown
				    	hc->PhysMax	= physmax;
				    	hc->DigMax	= digmax;
				    	hc->PhysMin	= physmin;
				    	hc->DigMin	= digmin;
				    	hc->Impedance	= NAN;
				    	hc->Cal		= cal;
				    	hc->Off		= off;
					hc->OnOff   	= 1;
				    	hc->PhysDimCode = 4275; // uV
		    			hc->LeadIdCode  = 0;
					size_t bi8      = k*(size_t)hdr->SPR*GDFTYP_BITS[gdftyp];
					hc->bi8         = bi8;
					hc->bi          = bi8>>3;
				}

				if (VERBOSE_LEVEL>7) fprintf(stdout,"BVA210 seq=%i,pos=%i,%i <%s> bpb=%i\n",seq,(int)pos,hdr->HeadLen,t,hdr->AS.bpb);
			}
			//else if (!strncmp(t,"[Common Infos]",14))
			//	seq = 4;
			else if (!strncmp(t,"[Coordinates]",13))
				seq = 5;
			else if (!strncmp(t,"[Comment]",9))
				seq = 6;
			else if (!strncmp(t,"[",1))
				seq = 9;


			else if (seq==1) {
				if      (!strncmp(t,"DataFile=",9))
					strcpy(ext, strrchr(t,'.') + 1);

				else if (!strncmp(t,"MarkerFile=",11)) {

					char* mrkfile = (char*)calloc(strlen(hdr->FileName)+strlen(t),1);

					if (strrchr(hdr->FileName,FILESEP)) {
						strcpy(mrkfile, hdr->FileName);			// Flawfinder: ignore
						strcpy(strrchr(mrkfile,FILESEP)+1, t+11);	// Flawfinder: ignore
					} else
						strcpy(mrkfile,t+11);		// Flawfinder: ignore

					if (VERBOSE_LEVEL>7)
						fprintf(stdout,"SOPEN marker file <%s>.\n",mrkfile);

					HDRTYPE *hdr2 = sopen(mrkfile,"r",NULL);

					hdr->T0 = hdr2->T0;
					memcpy(&hdr->EVENT,&hdr2->EVENT,sizeof(hdr2->EVENT));
					hdr->AS.auxBUF = hdr2->AS.auxBUF;  // contains the free text annotation
					// do not de-allocate event table when hdr2 is deconstructed
					memset(&hdr2->EVENT,0,sizeof(hdr2->EVENT));
					hdr2->AS.auxBUF = NULL;
					sclose(hdr2);
					destructHDR(hdr2);
					free(mrkfile);
					biosigERROR(hdr, B4C_NO_ERROR, NULL); // reset error status - missing or incorrect marker file is not critical
				}
				else if (!strncmp(t,"DataFormat=BINARY",11))
					;
				else if (!strncmp(t,"DataFormat=ASCII",16)) {
					FLAG_ASCII = 1;
					gdftyp     = 17;
//					biosigERROR(hdr, B4C_DATATYPE_UNSUPPORTED, "Error SOPEN(BrainVision): ASCII-format not supported (yet).");
				}
				else if (!strncmp(t,"DataOrientation=VECTORIZED",25))
					orientation = VEC;
				else if (!strncmp(t,"DataOrientation=MULTIPLEXED",26))
					orientation = MUL;
				else if (!strncmp(t,"DataType=TIMEDOMAIN",19))
					;
				else if (!strncmp(t,"DataType=",9)) {
					biosigERROR(hdr, B4C_DATATYPE_UNSUPPORTED, "Error SOPEN(BrainVision): DataType is not TIMEDOMAIN");
				}
				else if (!strncmp(t,"NumberOfChannels=",17)) {
					hdr->NS = atoi(t+17);
				}
				else if (!strncmp(t,"DataPoints=",11)) {
					npts = atol(t+11);
				}
				else if (!strncmp(t,"SamplingInterval=",17)) {
					hdr->SampleRate = 1e6/atof(t+17);
					hdr->EVENT.SampleRate = hdr->SampleRate;
				}
			}
			else if (seq==2) {
				if      (!strncmp(t,"BinaryFormat=IEEE_FLOAT_32",26)) {
					gdftyp = 16;
					digmax =  physmax/cal;
					digmin =  physmin/cal;
				}
				else if (!strncmp(t,"BinaryFormat=INT_16",19)) {
					gdftyp =  3;
					digmax =  32767;
					digmin = -32768;
					hdr->FLAG.OVERFLOWDETECTION = 1;
				}
				else if (!strncmp(t,"BinaryFormat=UINT_16",20)) {
					gdftyp = 4;
					digmax = 65535;
					digmin = 0;
					hdr->FLAG.OVERFLOWDETECTION = 1;
				}
				else if (!strncmp(t,"BinaryFormat",12)) {
					biosigERROR(hdr, B4C_DATATYPE_UNSUPPORTED, "Error SOPEN(BrainVision): BinaryFormat=<unknown>");
				}
				else if (!strncmp(t,"UseBigEndianOrder=NO",20)) {
					// hdr->FLAG.SWAP = (__BYTE_ORDER == __BIG_ENDIAN);
					hdr->FILE.LittleEndian = 1;
				}
				else if (!strncmp(t,"UseBigEndianOrder=YES",21)) {
					// hdr->FLAG.SWAP = (__BYTE_ORDER == __LITTLE_ENDIAN);
					hdr->FILE.LittleEndian = 0;
				}
				else if (!strncmp(t,"DecimalSymbol=",14)) {
					DECIMALSYMBOL = t[14];
				}
				else if (!strncmp(t,"SkipLines=",10)) {
					SKIPLINES = atoi(t+10);
				}
				else if (!strncmp(t,"SkipColumns=",12)) {
					SKIPCOLUMNS = atoi(t+12);
				}
				else if (0) {
					biosigERROR(hdr, B4C_DATATYPE_UNSUPPORTED, "Error SOPEN(BrainVision): BinaryFormat=<unknown>");
					return(hdr);
				}
			}
			else if (seq==3) {
				if (VERBOSE_LEVEL==9)
					fprintf(stdout,"BVA: seq=%i,line=<%s>,ERR=%i\n",seq,t,hdr->AS.B4C_ERRNUM );

				if (!strncmp(t,"Ch",2)) {
					char* ptr;

					if (VERBOSE_LEVEL==9) fprintf(stdout,"%s\n",t);

					int n = strtoul(t+2, &ptr, 10)-1;
					if ((n < 0) || (n >= hdr->NS)) {
						biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Error SOPEN(BrainVision): invalid channel number");
						ifclose(hdr);
						return(hdr);
					}
					size_t len = min(strcspn(ptr+1,","),MAX_LENGTH_LABEL);
					strncpy(hdr->CHANNEL[n].Label,ptr+1,len);
					hdr->CHANNEL[n].Label[len]=0;
					ptr += len+2;
					ptr += strcspn(ptr,",")+1;
					if (strlen(ptr)>0) {
						double tmp = atof(ptr);
						if (tmp) hdr->CHANNEL[n].Cal = tmp;
						hdr->CHANNEL[n].PhysMax = hdr->CHANNEL[n].DigMax * hdr->CHANNEL[n].Cal ;
						hdr->CHANNEL[n].PhysMin = hdr->CHANNEL[n].DigMin * hdr->CHANNEL[n].Cal ;
					}

					if (VERBOSE_LEVEL==9)
						fprintf(stdout,"Ch%02i=%s,,%s(%f)\n",n,hdr->CHANNEL[n].Label,ptr,hdr->CHANNEL[n].Cal );
				}
			}
			else if (seq==4) {
			}
			else if (seq==5) {
			}
			else if (seq==6) {
			}

			// t = strtok(NULL,"\x0a\x0d");	// extract next line
		}
		hdr->HeadLen  = 0;
	    	if (FLAG_ASCII) {
	    		count = 0;
			size_t bufsiz  = hdr->NS*hdr->SPR*hdr->NRec*16;
			while (!ifeof(hdr)) {
			    	hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,count+bufsiz+1);
		    		count += ifread(hdr->AS.Header+count,1,bufsiz,hdr);
			}
			ifclose(hdr);
			hdr->AS.Header[count]=0;	// terminating null character

			size_t pos=0;
			if (DECIMALSYMBOL != '.')
				do {
					if (hdr->AS.Header[pos]==DECIMALSYMBOL)
						hdr->AS.Header[pos] = '.';
				} while (hdr->AS.Header[++pos]);

			pos = 0;
	    		while (SKIPLINES>0) {
				while (!iscntrl(hdr->AS.Header[pos])) pos++; 	// skip line
				while ( iscntrl(hdr->AS.Header[pos])) pos++;	// skip line feed and carriage return
				SKIPLINES--;
	    		}

			hdr->AS.rawdata = (uint8_t*)malloc(hdr->NS*npts*sizeof(double));
			char* POS=(char*)(hdr->AS.Header+pos);
			for (k=0; k < hdr->NS*npts; k++) {
		    		if (((orientation==MUL) && !(k%hdr->NS)) ||
		    		    ((orientation==VEC) && !(k%npts))) {
		    		    	double d;
			    		int sc = SKIPCOLUMNS;
					while (sc--) d=strtod(POS,&POS);	// skip value, return value is ignored
	    			}
		    		*(double*)(hdr->AS.rawdata+k*sizeof(double)) = strtod(POS,&POS);
	    		}
	    		hdr->TYPE = native;
			hdr->AS.length  = hdr->NRec;
	    	}
	}

	else if (hdr->TYPE==CFS) {
	        hdr->HeadLen = count;
		sopen_cfs_read(hdr);
	}

	else if (hdr->TYPE==SMR) {
	        hdr->HeadLen = count;
		sopen_smr_read(hdr);
	}

	else if (hdr->TYPE==CFWB) {
	    	hdr->SampleRate = 1.0/lef64p(hdr->AS.Header+8);
		hdr->SPR    	= 1;
	    	tm_time.tm_year = lei32p(hdr->AS.Header+16) - 1900;
	    	tm_time.tm_mon  = lei32p(hdr->AS.Header+20) - 1;
	    	tm_time.tm_mday = lei32p(hdr->AS.Header+24);
	    	tm_time.tm_hour = lei32p(hdr->AS.Header+28);
	    	tm_time.tm_min  = lei32p(hdr->AS.Header+32);
	    	tm_time.tm_sec  = (int)lef64p(hdr->AS.Header+36);
		tm_time.tm_isdst = -1;

    		hdr->T0 	= tm_time2gdf_time(&tm_time);
	    	// = *(double*)(Header1+44);	// pre-trigger time
	    	hdr->NS   	= leu32p(hdr->AS.Header+52);
	    	hdr->NRec	= leu32p(hdr->AS.Header+56);
#define CFWB_FLAG_TIME_CHANNEL  (*(int32_t*)(Header1+60))	// TimeChannel
	    	//  	= *(int32_t*)(Header1+64);	// DataFormat

	    	hdr->HeadLen = 68 + hdr->NS*96;
	    	hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,hdr->HeadLen);
	    	if (count<=hdr->HeadLen)
			count += ifread(hdr->AS.Header+count, 1, hdr->HeadLen-count, hdr);
		else
	    		ifseek(hdr, hdr->HeadLen, SEEK_SET);

		uint16_t gdftyp = leu32p(hdr->AS.Header+64);
		hdr->AS.bpb = (CFWB_FLAG_TIME_CHANNEL ? GDFTYP_BITS[CFWB_GDFTYP[gdftyp-1]]>>3 : 0);

	    	hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL,hdr->NS*sizeof(CHANNEL_TYPE));
		for (k=0; k<hdr->NS; k++)	{
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
		    	uint8_t* Header2 = hdr->AS.Header+68+k*96;
			hc->Transducer[0] = '\0';
		    	hc->GDFTYP 	= CFWB_GDFTYP[gdftyp-1];
		    	hc->SPR 	= 1; // *(int32_t*)(Header1+56);
		    	strncpy(hc->Label, (char*)Header2, min(32,MAX_LENGTH_LABEL));
		    	char p[17];
		    	memcpy(p, (char*)Header2+32, 16);
		    	p[16] = 0;
		    	hc->PhysDimCode = PhysDimCode(p);
		    	hc->LeadIdCode  = 0;
		    	hc->Cal	= lef64p(Header2+64);
		    	hc->Off	= lef64p(Header2+72);
		    	hc->PhysMax	= lef64p(Header2+80);
		    	hc->PhysMin	= lef64p(Header2+88);
		    	hc->DigMax	= (hc->PhysMax - hc->Off) / hc->Cal;
		    	hc->DigMin	= (hc->PhysMin - hc->Off) / hc->Cal;
			hc->OnOff    	= 1;
			hc->bi    	= hdr->AS.bpb;
			hdr->AS.bpb += GDFTYP_BITS[hc->GDFTYP]>>3;
		}
		hdr->FLAG.OVERFLOWDETECTION = 0; 	// CFWB does not support automated overflow and saturation detection
	}

	else if (hdr->TYPE==CNT) {

		if (VERBOSE_LEVEL>7) fprintf(stdout, "%s: Neuroscan format (count=%d)\n",__func__, (int)count);

		// TODO: fix handling of AVG and EEG files
		if (count < 900) {
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, 900);
			count  += ifread(hdr->AS.Header+count, 1, 900-count, hdr);
		}
		hdr->VERSION = atof((char*)hdr->AS.Header + 8);

		int8_t FLAG_CNT32 = 0;
		uint16_t gdftyp = 0;
	    	uint8_t minor_revision = hdr->AS.Header[804];
	    	size_t eventtablepos = leu32p(hdr->AS.Header+886);
	    	uint32_t nextfilepos = leu32p(hdr->AS.Header+12);

		if (VERBOSE_LEVEL > 7)
			fprintf(stdout,"%s: Neuroscan format: minor revision %i eventtablepos: %i nextfilepos: %i\n", __func__, minor_revision, (unsigned)eventtablepos, nextfilepos);

		/* make base of filename */
		size_t i=0, j=0;
		while (hdr->FileName[i] != '\0') {
			if ((hdr->FileName[i]=='/') || (hdr->FileName[i]=='\\')) { j=i+1; }
			i++;
		}
		/* skip the extension '.cnt' of filename base and copy to Patient.Id */
		strncpy(hdr->Patient.Id, hdr->FileName+j, min(MAX_LENGTH_PID,strlen(hdr->FileName)-j-4));
		hdr->Patient.Id[MAX_LENGTH_PID] = 0;

	    	ptr_str = (char*)hdr->AS.Header+136;
    		hdr->Patient.Sex = (ptr_str[0]=='f')*2 + (ptr_str[0]=='F')*2 + (ptr_str[0]=='M') + (ptr_str[0]=='m');
	    	ptr_str = (char*)hdr->AS.Header+137;
	    	hdr->Patient.Handedness = (ptr_str[0]=='r')*2 + (ptr_str[0]=='R')*2 + (ptr_str[0]=='L') + (ptr_str[0]=='l');
	    	ptr_str = (char*)hdr->AS.Header+225;

		char tmp[6];
		tmp[2] = '\0'; 		// make sure tmp is 0-terminated
	    	tm_time.tm_sec  = atoi(memcpy(tmp,ptr_str+16,2));
    		tm_time.tm_min  = atoi(memcpy(tmp,ptr_str+13,2));
    		tm_time.tm_hour = atoi(memcpy(tmp,ptr_str+10,2));
    		tm_time.tm_mday = atoi(memcpy(tmp,ptr_str,2));
    		tm_time.tm_mon  = atoi(memcpy(tmp,ptr_str+3,2))-1;
    		tm_time.tm_year = atoi(memcpy(tmp,ptr_str+6,2));

	    	if (tm_time.tm_year<=80)    	tm_time.tm_year += 100;
		hdr->T0 = tm_time2gdf_time(&tm_time);

		hdr->NS  = leu16p(hdr->AS.Header+370);
	    	hdr->HeadLen = 900+hdr->NS*75;
		hdr->SampleRate = leu16p(hdr->AS.Header+376);
		hdr->AS.bpb = hdr->NS*2;

		if (hdr->AS.Header[20]==1) {
			// Neuroscan EEG
			hdr->NRec = leu16p(hdr->AS.Header+362);
			hdr->SPR  = leu16p(hdr->AS.Header+368);
	    		hdr->AS.bpb = 2*hdr->NS*hdr->SPR+1+2+2+4+2+2;
	    		size_t bpb4 = 4*hdr->NS*hdr->SPR+1+2+2+4+2+2;
			struct stat FileBuf;

			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"%s (line %d): Neuroscan format: minor rev=%i bpb2:%i bpb4:%i\n", __func__,__LINE__, minor_revision, (unsigned)hdr->AS.bpb, (unsigned)bpb4);

		    	switch (minor_revision) {
		    	case 9:
		    		// TODO: FIXME
				fprintf(stderr,"Warning biosig/%s (line %d) (CNT/EEG): minor revision %i is experimental\n", __func__,__LINE__, minor_revision);
		    		gdftyp = 3;
		    		hdr->FILE.LittleEndian = 0;
				stat(hdr->FileName,&FileBuf);
				if (hdr->NRec <= 0) {
					hdr->NRec = (min(FileBuf.st_size, nextfilepos) - hdr->HeadLen)/hdr->AS.bpb;
				}
		    		break;

		    	case 12:
		    		gdftyp = 3;
		    		eventtablepos = hdr->HeadLen + hdr->NRec*hdr->AS.bpb;
		    		break;

		    	default:
				if (minor_revision != 16)
					fprintf(stderr,"Warning biosig/%s (line %d) sopen (CNT/EEG): minor revision %i not tested\n", __func__,__LINE__, minor_revision);

				if (VERBOSE_LEVEL>7)
					fprintf(stdout,"biosig/%s (line %d) (CNT/EEG):  %i %i %i %i %i %i \n", __func__,__LINE__, (int)hdr->NRec, hdr->SPR, hdr->NS, (int)eventtablepos, (int)(hdr->AS.bpb * hdr->NRec + hdr->HeadLen), (int)(bpb4 * hdr->NRec + hdr->HeadLen));

	    			if ((size_t)(hdr->AS.bpb * hdr->NRec + hdr->HeadLen) == eventtablepos)
	    				gdftyp = 3;
	    			else if ((bpb4 * hdr->NRec + hdr->HeadLen) == eventtablepos) {
	    				hdr->AS.bpb = bpb4;
	    				gdftyp = 5;
	    			}
	    			else {
	    				biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "CNT/EEG: type of format not supported");
			    		return(hdr);
	    			}
		    	}
		}
		else {
			// Neuroscan CNT
			hdr->SPR    = 1;
			eventtablepos = leu32p(hdr->AS.Header+886);
			if (nextfilepos > 0) {
				ifseek (hdr,nextfilepos+52,SEEK_SET);
				FLAG_CNT32 = (ifgetc(hdr)==1);
				ifseek (hdr,count,SEEK_SET);
			}

	    		gdftyp      = FLAG_CNT32 ? 5 : 3;
		    	hdr->AS.bpb = hdr->NS*GDFTYP_BITS[gdftyp]/8;
			hdr->NRec   = (eventtablepos - hdr->HeadLen) / hdr->AS.bpb;

			if (VERBOSE_LEVEL > 7)
				fprintf(stdout,"biosig/%s (line %d) (CNT):  %i %i %i %i %i \n", __func__,__LINE__, (int)hdr->NRec, hdr->SPR, hdr->NS, (int)eventtablepos, (int)(hdr->AS.bpb * hdr->NRec + hdr->HeadLen) );
		}

		if (count < hdr->HeadLen) {
			hdr->AS.Header = (uint8_t*) realloc(Header1, hdr->HeadLen);
			count  += ifread(Header1+count, 1, hdr->HeadLen-count, hdr);
		}

	    	hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
	    	size_t bi = 0;
		for (k=0; k<hdr->NS; k++)	{
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
		    	uint8_t* Header2 = hdr->AS.Header+900+k*75;
			hc->Transducer[0] = '\0';
		    	hc->GDFTYP 	= gdftyp;
		    	hc->SPR 	= hdr->SPR; // *(int32_t*)(Header1+56);


		    	const size_t len = min(10, MAX_LENGTH_LABEL);

if (VERBOSE_LEVEL > 7) fprintf(stdout,"biosig/%s (line %d): #%d label <%s>\n", __func__,__LINE__,(int)k, (char*) Header2 );

		    	strncpy(hc->Label, (char*)Header2, len);
		    	hc->Label[len]  = 0;
		    	hc->LeadIdCode  = 0;
			hc->PhysDimCode = 4256+19;  // uV
		    	hc->Cal		= lef32p(Header2+59);
		    	hc->Cal    	*= lef32p(Header2+71)/204.8;
		    	hc->Off		= lef32p(Header2+47) * hc->Cal;
		    	hc->HighPass	= CNT_SETTINGS_HIGHPASS[(uint8_t)Header2[64]];
		    	hc->LowPass	= CNT_SETTINGS_LOWPASS[(uint8_t)Header2[65]];
		    	hc->Notch	= CNT_SETTINGS_NOTCH[(uint8_t)Header1[682]];
			hc->OnOff       = 1;

			if (FLAG_CNT32) {
			  	hc->DigMax	=  (double)(0x007fffff);
			    	hc->DigMin	= -(double)(int32_t)(0xff800000);
			}
			else {
			    	hc->DigMax	=  (double)32767;
			    	hc->DigMin	= -(double)32768;
			}
		    	hc->PhysMax	= hc->DigMax * hc->Cal + hc->Off;
		    	hc->PhysMin	= hc->DigMin * hc->Cal + hc->Off;
			hc->bi    	= bi;
			bi 		+= (size_t)hdr->SPR * (GDFTYP_BITS[hc->GDFTYP]>>3);
		}

	    	if ((eventtablepos < nextfilepos) && !ifseek(hdr, eventtablepos, SEEK_SET)) {
		    	/* read event table */
			hdr->EVENT.SampleRate = hdr->SampleRate;
			ifread(tmp, 9, 1, hdr);
			int8_t   TeegType   = tmp[0];
			uint32_t TeegSize   = leu32p(tmp+1);
			// uint32_t TeegOffset = leu32p(tmp+5); // not used

			int fieldsize;
			switch (TeegType) {
			case 2:
			case 3:  fieldsize = 19; break;
			default: fieldsize = 8;
			}

			uint8_t* buf = (uint8_t*)malloc(TeegSize);
			count = ifread(buf, 1, TeegSize, hdr);
			hdr->EVENT.N   = count/fieldsize;
			if (reallocEventTable(hdr, hdr->EVENT.N) == SIZE_MAX) {
				biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
				return (hdr);
			};

			hdr->EVENT.DUR=NULL;
			hdr->EVENT.CHN=NULL;
			for  (k = 0; k < hdr->EVENT.N; k++) {
				hdr->EVENT.TYP[k] = leu16p(buf+k*fieldsize);	// stimulus type
				uint8_t tmp8 = buf[k*fieldsize+3];
				if (tmp8>0) {
					if (hdr->EVENT.TYP[k]>0)
						fprintf(stdout,"Warning %s (line %d) event %i: both, stimulus and response, codes (%i/%i) are non-zero. response code is ignored.\n",__func__,__LINE__, (int)k+1,hdr->EVENT.TYP[k],tmp8);
					else
						hdr->EVENT.TYP[k] |= tmp8 | 0x80;	// response type
				}
				hdr->EVENT.POS[k] = leu32p(buf+4+k*fieldsize);        // 0-based indexing
				if (TeegType != 3)
					hdr->EVENT.POS[k] = (hdr->EVENT.POS[k] - hdr->HeadLen) / hdr->AS.bpb;
#if (BIOSIG_VERSION >= 10500)
				hdr->EVENT.TimeStamp[k] = 0;
#endif
			}
			free(buf);
	    	}
	    	ifseek(hdr, hdr->HeadLen, SEEK_SET);
		hdr->FLAG.OVERFLOWDETECTION = 0; 	// automated overflow and saturation detection not supported
	}

	else if (hdr->TYPE==CTF) {

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s: CTF[101]: %s\n", __func__, hdr->FileName);

		char *f0        = hdr->FileName;
		char *f1 	= (char*)malloc(strlen(f0)+6);
		strcpy(f1, f0);				// Flawfinder: ignore
		strcpy(strrchr(f1,'.')+1,"res4");	// Flawfinder: ignore

		if (VERBOSE_LEVEL>8) fprintf(stdout,"CTF[102]: %s\n\t%s\n",f0,f1);

		if (strcmp(strrchr(hdr->FileName,'.'),".res4")) {
			if (VERBOSE_LEVEL>8) fprintf(stdout,"CTF[103]:\n");
			ifclose(hdr);
			hdr->FileName = f1;
			hdr = ifopen(hdr,"rb");
			count = 0;
		}

		hdr->HeadLen = 1844;
		if (count < hdr->HeadLen) {
			hdr->AS.Header  = (uint8_t*)realloc(hdr->AS.Header,hdr->HeadLen);
			count += ifread(hdr->AS.Header+count,1,hdr->HeadLen-count,hdr);
		}

		if (VERBOSE_LEVEL>8) fprintf(stdout,"CTF[104]: %i %s\n\t%s\n",(int)count,f0,f1);

		struct tm t;
		sscanf((char*)(hdr->AS.Header+778),"%d:%d:%d",&t.tm_hour,&t.tm_min,&t.tm_sec);
		sscanf((char*)(hdr->AS.Header+778+255),"%d/%d/%d",&t.tm_mday,&t.tm_mon,&t.tm_year);
		--t.tm_mon;
		hdr->T0 = tm_time2gdf_time(&t);

		hdr->SPR 	= bei32p(hdr->AS.Header+1288);
		hdr->NS  	= bei16p(hdr->AS.Header+1292);
		hdr->SampleRate = bef64p(hdr->AS.Header+1296);
		// double Dur	= bef64p(hdr->AS.Header+1304);
		hdr->NRec	= bei16p(hdr->AS.Header+1312);
		strncpy(hdr->Patient.Id,(char*)(hdr->AS.Header+1712),min(MAX_LENGTH_PID,32));
		int32_t CTF_RunSize  = bei32p(hdr->AS.Header+1836);
		//int32_t CTF_RunSize2 = bei32p(hdr->AS.Header+1844);

		hdr->HeadLen=1844+CTF_RunSize+2;
		if (count < hdr->HeadLen) {
			hdr->AS.Header  = (uint8_t*)realloc(hdr->AS.Header,hdr->HeadLen);
			count += ifread(hdr->AS.Header+count,1,hdr->HeadLen-count,hdr);
		}
		int16_t CTF_NumberOfFilters = bei16p(hdr->AS.Header+1844+CTF_RunSize);
		hdr->HeadLen = 1844+CTF_RunSize+2+CTF_NumberOfFilters*26+hdr->NS*(32+48+1280);
		if (count < hdr->HeadLen) {
			hdr->AS.Header  = (uint8_t*)realloc(hdr->AS.Header,hdr->HeadLen);
			count += ifread(hdr->AS.Header+count,1,hdr->HeadLen-count,hdr);
		}
		ifclose(hdr);

		size_t pos = 1846+CTF_RunSize+CTF_NumberOfFilters*26;

	    	hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
	    	hdr->AS.bpb = 0;
		for (k=0; k<hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;

			strncpy(hc->Label,(const char*)(hdr->AS.Header+pos+k*32),min(32,MAX_LENGTH_LABEL));
			hc->Label[min(MAX_LENGTH_LABEL,32)]=0;

			if (VERBOSE_LEVEL>8)
				fprintf(stdout,"CTF[107]: #%i\t%x\t%s\n",(int)k,(int)(pos+k*32),hc->Label);

			int16_t index = bei16p(hdr->AS.Header+pos+hdr->NS*32+k*(48+1280)); // index
			hc->Cal = 1.0/bef64p(hdr->AS.Header+pos+hdr->NS*32+k*(48+1280)+16);
			switch (index) {
			case 0:
			case 1:
			case 9:
				hc->Cal /= bef64p(hdr->AS.Header+pos+hdr->NS*32+k*(48+1280)+8);
			}

		    	hc->GDFTYP 	= 5;
		    	hc->SPR 	= hdr->SPR;
		    	hc->LeadIdCode  = 0;
		    	hc->Off	= 0.0;
			hc->OnOff   = 1;

			hc->PhysDimCode = 0;
			hc->Transducer[0] = 0;
		    	hc->DigMax	= ldexp( 1.0,31);
		    	hc->DigMin	= ldexp(-1.0,31);
		    	hc->PhysMax	= hc->DigMax * hc->Cal + hc->Off;
		    	hc->PhysMin	= hc->DigMin * hc->Cal + hc->Off;

			hc->bi    = hdr->AS.bpb;
			hdr->AS.bpb += hdr->SPR*(GDFTYP_BITS[hc->GDFTYP]>>3);
		}

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"CTF[109] %s: \n",hdr->FileName);

		/********** read marker file **********/
		char *f2 = (char*)malloc(strlen(f0)+16);
		strcpy(f2, f0);					// Flawfinder: ignore
		strcpy(strrchr(f2,FILESEP)+1,"MarkerFile.mrk");	// Flawfinder: ignore
		hdr->EVENT.SampleRate = hdr->SampleRate;
		hdr->EVENT.N = 0;

		hdr->FileName = f2;
       		hdr = ifopen(hdr,"rb");
	    	if (hdr->FILE.OPEN) {
			count = 0;
	    		char *vmrk=NULL;
			while (!ifeof(hdr)) {
				size_t bufsiz = max(2*count, PAGESIZE);
				vmrk   = (char*)realloc(vmrk, bufsiz+1);
				count += ifread(vmrk+count, 1, bufsiz-count, hdr);
			}
		    	vmrk[count] = 0;	// add terminating \0 character
			ifclose(hdr);

			char *t1, *t2;
			float u1,u2;
			t1 = strstr(vmrk,"TRIAL NUMBER");
			t2 = strtok(t1,"\x0a\x0d");
			size_t N = 0;
			t2 = strtok(NULL,"\x0a\x0d");
			while (t2 != NULL) {
				sscanf(t2,"%f %f",&u1,&u2);

				if (N+1 >= hdr->EVENT.N) {
					hdr->EVENT.N  += 256;
					if (reallocEventTable(hdr, hdr->EVENT.N) == SIZE_MAX) {
						biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
						return (hdr);
					};
				}
				hdr->EVENT.TYP[N] = 1;
				hdr->EVENT.POS[N] = (uint32_t)(u1*hdr->SPR+u2*hdr->SampleRate);
				hdr->EVENT.DUR[N] = 0;
				hdr->EVENT.CHN[N] = 0;
#if (BIOSIG_VERSION >= 10500)
				hdr->EVENT.TimeStamp[N] = 0;
#endif
				N++;

				t2 = strtok(NULL,"\x0a\x0d");
			}
			hdr->EVENT.N = N;
			free(vmrk);
		}
		free(f2);
		/********** end reading event/marker file **********/


		strcpy(strrchr(f1,'.')+1,"meg4");
		hdr->FileName = f1;
		hdr = ifopen(hdr,"rb");
	    	hdr->HeadLen  = 8;
		hdr->HeadLen  = ifread(hdr->AS.Header,1,8,hdr);
		// hdr->FLAG.SWAP= (__BYTE_ORDER == __LITTLE_ENDIAN);
		hdr->FILE.LittleEndian = 0;

		hdr->FileName = f0;
		free(f1);

	}

	else if (hdr->TYPE==DEMG) {
	    	hdr->VERSION 	= leu16p(hdr->AS.Header+4);
	    	hdr->NS		= leu16p(hdr->AS.Header+6);
	    	hdr->SPR	= 1;
	    	hdr->SampleRate = leu32p(hdr->AS.Header+8);
	    	hdr->NRec	= leu32p(hdr->AS.Header+12);

		uint16_t gdftyp = 16;
		uint8_t  bits    = hdr->AS.Header[16];
		double   PhysMin = (double)(int8_t)hdr->AS.Header[17];
		double   PhysMax = (double)(int8_t)hdr->AS.Header[18];
		double	 Cal = 1.0;
		double	 Off = 0.0;

		if (hdr->VERSION==1) {
			gdftyp = 16;	// float32
			Cal = 1.0;
			Off = 0.0;
		}
		else if (hdr->VERSION==2) {
			gdftyp = 4;	// uint16
			Cal = (PhysMax-PhysMin)/((1<<bits) - 1.0);
			Off = (double)PhysMin;
		}
		double DigMax = (PhysMax-Off)/Cal;
		double DigMin = (PhysMin-Off)/Cal;
	    	hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
	    	hdr->AS.bpb = 0;
		for (k=0; k < hdr->NS; k++) {
			CHANNEL_TYPE* hc = hdr->CHANNEL+k;
			hc->GDFTYP   = gdftyp;
			hc->SPR      = 1;
			hc->Cal      = Cal;
			hc->Off      = Off;
			hc->OnOff    = 1;
			hc->Transducer[0] = '\0';
			hc->LowPass  = 450;
			hc->HighPass = 20;
			hc->PhysMax  = PhysMax;
			hc->PhysMin  = PhysMin;
			hc->DigMax   = DigMax;
			hc->DigMin   = DigMin;
		    	hc->LeadIdCode  = 0;
			hc->bi    = hdr->AS.bpb;
			hdr->AS.bpb += GDFTYP_BITS[gdftyp]>>3;
		}
		hdr->FLAG.OVERFLOWDETECTION = 0; 	// automated overflow and saturation detection not supported
	    	hdr->HeadLen = 19;
	    	ifseek(hdr, 19, SEEK_SET);
	}

	else if ((hdr->TYPE==EAS) || (hdr->TYPE==EZ3) || (hdr->TYPE==ARC)) {
		while (!ifeof(hdr)) {
			size_t bufsiz = max(2*count, PAGESIZE);
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, bufsiz+1);
			count  += ifread(hdr->AS.Header+count, 1, bufsiz-count, hdr);
		}
		hdr->AS.Header[count]=0;
		hdr->HeadLen = count;
		ifclose(hdr);

		sopen_cadwell_read(hdr);
	}
	else if (hdr->TYPE==EBS) {

		fprintf(stderr,"Warning SOPEN(EBS): support for EBS format is experimental\n");

		/**  Fixed Header (32 bytes)  **/
		uint32_t EncodingID = beu32p(hdr->AS.Header+8);
		hdr->NS  = beu32p(hdr->AS.Header+12);
		hdr->SPR = beu64p(hdr->AS.Header+16);
		uint64_t datalen = beu64p(hdr->AS.Header+24);

		enum encoding {
			TIB_16 = 0x00000000,
			CIB_16 = 0x00000001,
			TIL_16 = 0x00000002,
			CIL_16 = 0x00000003,
			TI_16D = 0x00000010,
			CI_16D = 0x00000011
		};

		hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL,hdr->NS*sizeof(CHANNEL_TYPE));
		size_t pos = 32;
		uint32_t tag, len;
                /**  Variable Header  **/
                tag = beu32p(hdr->AS.Header+pos);
                while (tag) {
	                len = beu32p(hdr->AS.Header+pos+4)<<2;
	                pos += 8;
	                if (count < pos+len+8) {
	                	hdr->AS.Header = (uint8_t*) realloc(hdr->AS.Header,count*2);
	                	count += ifread(hdr->AS.Header+count, 1, count, hdr);
	                }
			if (VERBOSE_LEVEL>8)
		                fprintf(stdout,"%6i %6i tag=%08x len=%5i: |%c%c%c%c| %s\n", (int)pos, (int)count,tag, len, Header1[0x015f], Header1[0x0160], Header1[0x0161], Header1[0x0162], hdr->AS.Header+pos);

        		/* Appendix A */
        		switch (tag) {
        		case 0x00000002: break;
        		case 0x00000004:
        			strncpy(hdr->Patient.Name,Header1+pos,MAX_LENGTH_NAME);
        			break;
        		case 0x00000006:
        			strncpy(hdr->Patient.Id,Header1+pos,MAX_LENGTH_PID);
        			break;
        		case 0x00000008: {
        			struct tm t;
        			t.tm_mday = (Header1[pos+6]-'0')*10 + (Header1[pos+7]-'0');
        			Header1[pos+6] = 0;
        			t.tm_mon  = atoi(Header1+pos+4) + 1;
        			Header1[pos+4] = 0;
        			t.tm_year = atoi(Header1+pos) - 1900;
				t.tm_hour = 0;
				t.tm_min = 0;
				t.tm_sec = 0;
				hdr->Patient.Birthday = tm_time2gdf_time(&t);
        			break;
        			}
        		case 0x0000000a:
        			hdr->Patient.Sex = bei32p(hdr->AS.Header+pos);
        			break;
        		case 0x00000010:
        			hdr->SampleRate = atof(Header1+pos);
        			break;
        		case 0x00000012:
				// strndup(hdr->ID.Hospital,Header1+pos,len);
				hdr->ID.Hospital = malloc(len+1);
				if (hdr->ID.Hospital) {
					hdr->ID.Hospital[len] = 0;
					strncpy(hdr->ID.Hospital,Header1+pos,len);
				}
        			break;

        		case 0x00000003: // units
        			{
        			int k;
				char* ptr = Header1+pos;
        			for (k=0; k < hdr->NS; k++) {
					CHANNEL_TYPE *hc = hdr->CHANNEL + k;
       					hc->Cal = strtod(ptr, &ptr);
					hc->PhysDimCode = PhysDimCode(ptr);
        			}
        			}
        			break;

        		case 0x00000005:
        			{
        			int k;
				char* ptr = Header1+pos;
        			for (k=0; k < hdr->NS; k++) {
					CHANNEL_TYPE *hc = hdr->CHANNEL + k;
					int c = 0;
        				while (beu32p(ptr)) {
						if (VERBOSE_LEVEL>8)
							fprintf(stdout,"0x05: [%i %i] |%c%c%c%c%c%c%c%c|\n",k,c,ptr[0],ptr[1],ptr[2],ptr[3],ptr[4],ptr[5],ptr[6],ptr[7]);

						if ((*ptr) && (c<=MAX_LENGTH_LABEL)) {
        						hc->Label[c++] = *ptr;
        					}
        					ptr++;
        				}
					if (VERBOSE_LEVEL>7)
						fprintf(stdout,"0x05: %08x\n",beu32p(ptr));
					hc->Label[c] = 0;
        				ptr += 4;
        				while (bei32p(ptr)) ptr++;
        				ptr += 4;
				}
				}
				break;

        		case 0x0000000b: // recording time
        			if (Header1[pos+8]=='T') {
	        			struct tm t;
					t.tm_sec = atoi(Header1+pos+13);
        				Header1[pos+13] = 0;
					t.tm_min = atoi(Header1+pos+11);
        				Header1[pos+11] = 0;
					t.tm_hour = atoi(Header1+pos+9);
        				Header1[pos+8] = 0;
        				t.tm_mday = atoi(Header1+pos+6);
        				Header1[pos+6] = 0;
        				t.tm_mon  = atoi(Header1+pos+4) + 1;
	        			Header1[pos+4] = 0;
        				t.tm_year = atoi(Header1+pos) - 1900;
					hdr->T0 = tm_time2gdf_time(&t);
					if (VERBOSE_LEVEL>8)
		       				fprintf(stdout,"<%s>, T0 = %s\n",Header1+pos,asctime(&t));
        			}
				if (VERBOSE_LEVEL>8)
	       				fprintf(stdout,"<%s>\n",Header1+pos);
        			break;
        		case 0x0000000f: // filter
        			{
        			int k;
				char* ptr = Header1+pos;
        			for (k=0; k < hdr->NS; k++) {
					CHANNEL_TYPE *hc = hdr->CHANNEL + k;
        				switch (beu32p(ptr)) {
        				case 1: // lowpass
        					hc->LowPass  = strtod(ptr+4, &ptr);
        					break;
        				case 2: // high pass
        					hc->HighPass = strtod(ptr+4, &ptr);
        					break;
        				default:
						fprintf(stderr,"Warning SOPEN (EBS): unknown filter\n");
        				}
        				while (bei32p(ptr) != -1) ptr++;
        				ptr += 4;
        			}
        			}
        			break;
        		}

			pos += len;
	                tag = beu32p(hdr->AS.Header+pos);
                }
                hdr->HeadLen = pos;
                ifseek(hdr,pos,SEEK_SET);
		hdr->AS.first = 0;
		hdr->AS.length = 0;

		if ((bei64p(hdr->AS.Header+24)==-1) && (bei64p(hdr->AS.Header+24)==-1)) {
			/* if data length is not present */
			struct stat FileBuf;
			stat(hdr->FileName,&FileBuf);
			hdr->FILE.size = FileBuf.st_size;
			datalen = (hdr->FILE.size - hdr->HeadLen);
		}
		else 	datalen <<= 2;

                /**  Encoded Signal Data (4*d bytes)  **/
                size_t spr = datalen/(2*hdr->NS);
                switch (EncodingID) {
                case TIB_16:
                	hdr->SPR = 1;
                	hdr->NRec = spr;
                	hdr->FILE.LittleEndian = 0;
                	break;
                case CIB_16:
                	hdr->SPR = spr;
                	hdr->NRec = 1;
                	hdr->FILE.LittleEndian = 0;
                	break;
                case TIL_16:
                	hdr->SPR = 1;
                	hdr->NRec = spr;
                	hdr->FILE.LittleEndian = 1;
                	break;
                case CIL_16:
                	hdr->SPR = spr;
                	hdr->NRec = 1;
                	hdr->FILE.LittleEndian = 1;
                	break;
                case TI_16D:
                case CI_16D:
                default:
                	biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "EBS: unsupported Encoding");
                	return(hdr);
                }

		typeof(hdr->NS) k;
		for (k = 0; k < hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL + k;
			hc->GDFTYP = 3; 	// int16
			hc->SPR    = hdr->SPR; 	// int16
			hc->bi     = k*2;
			hc->Off    = 0.0;
			hc->OnOff  = 1;
			hc->DigMax = (double)32767;
			hc->DigMin = (double)-32768;
			hc->PhysMax = hc->DigMax*hc->Cal;
			hc->PhysMin = hc->DigMin*hc->Cal;
			hc->Transducer[0] = 0;
			hc->LeadIdCode = 0;
			hc->Notch     = NAN;
			hc->Impedance = INFINITY;
		      	hc->fZ        = NAN;
			hc->XYZ[0] = 0.0;
			hc->XYZ[1] = 0.0;
			hc->XYZ[2] = 0.0;
		}
		hdr->AS.bpb = hdr->SPR*hdr->NS*2;

                /**  Optional Second Variable Header  **/

	}

	else if (hdr->TYPE==EEG1100) {
		// the information of this format is derived from nk2edf-0.43beta-src of Teunis van Beelen

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d) %s(...)\n",__FILE__,__LINE__,__func__);

		char *fn = (char*)malloc((strlen(hdr->FileName)+5)*sizeof(char));
		strcpy(fn, hdr->FileName);	// Flawfinder: ignore
		char *LOG=NULL;

		/* read .pnt */
			if (strrchr(fn,FILESEP))
				strncpy(strrchr(fn,FILESEP)+1, (char*)(hdr->AS.Header + 32), 4);
			else
				strncpy(fn, (char*)(hdr->AS.Header + 32), 4);

			FILE *fid = fopen(fn,"rb");
			if (fid != NULL) {
				count = 0;

			 	while (!feof(fid)) {
					size_t r = max(count*2, PAGESIZE);
					LOG = (char*) realloc(LOG,r+1);
					count += fread(LOG+count,1,r-count,fid);
			 	}
				fclose(fid);

				LOG[count] = 0;
				// Name: @0x062e

				if (!hdr->FLAG.ANONYMOUS) {
					strncpy(hdr->Patient.Name, LOG+0x62e, MAX_LENGTH_PID);
					hdr->Patient.Name[MAX_LENGTH_NAME] = 0;
				}

				// Id: @0x0604
				strncpy(hdr->Patient.Id, LOG+0x604, MAX_LENGTH_PID);
				hdr->Patient.Id[MAX_LENGTH_PID] = 0;

				// Gender: @0x064a
				hdr->Patient.Sex = (toupper(LOG[0x064a])=='M') + 2*(toupper(LOG[0x064a])=='F') + 2*(toupper(LOG[0x064a])=='W');

				// Birthday: @0x0660
				sscanf((char*)(LOG+0x0660),"%04u/%02u/%02u",&tm_time.tm_year,&tm_time.tm_mon,&tm_time.tm_mday);
				tm_time.tm_hour  = 12;
				tm_time.tm_min   = 0;
				tm_time.tm_sec   = 0;
				tm_time.tm_year -= 1900;
				tm_time.tm_mon--;
				tm_time.tm_isdst = -1;
				hdr->Patient.Birthday = tm_time2gdf_time(&tm_time);

			}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d) %s(...)\n",__FILE__,__LINE__,__func__);

		size_t n1,n2,k2,pos1,pos2;
		n1 = hdr->AS.Header[145];
		if ((n1*20+0x92) > count) {
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,n1*20+0x92+1);
			count += ifread(hdr->AS.Header+count, 1, n1*20+0x92-count,hdr);
		}
		// Start date: @0x0040
		sscanf((char*)(hdr->AS.Header+0x40),"%04u%02u%02u%02u%02u%02u",&tm_time.tm_year,&tm_time.tm_mon,&tm_time.tm_mday,&tm_time.tm_hour,&tm_time.tm_min,&tm_time.tm_sec);
		tm_time.tm_year -= 1900;
		tm_time.tm_mon--;	// Jan=0, Feb=1, ...
		tm_time.tm_isdst = -1;
		//hdr->T0 = tm_time2gdf_time(&tm_time);

		int TARGET_SEGMENT = hdr->FLAG.TARGETSEGMENT;
		int numSegments = 0;

		size_t Total_NRec = 0;
		uint8_t *h2 = (uint8_t*)malloc(22);
		uint8_t *h3 = (uint8_t*)malloc(40);
		for (k=0; k<n1; k++) {

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d) %s(...) k=%d\n",__FILE__,__LINE__,__func__,k);

			pos1 = leu32p(hdr->AS.Header+146+k*20);
			ifseek(hdr, pos1, SEEK_SET);
			ifread(h2, 1, 22, hdr);
			n2 = h2[17];
			if (n2>1) {
				h2 = (uint8_t*)realloc(h2,2+n2*20);
				ifread(h2+22, 1, 2+n2*20-22, hdr);
			}
			if (reallocEventTable(hdr, hdr->EVENT.N+n2) == SIZE_MAX) {
				biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
				return (hdr);
			};

			for (k2=0; k2<n2; k2++) {
				pos2 = leu32p(h2 + 18 + k2*20);

				ifseek(hdr, pos2, SEEK_SET);
				size_t pos3 = ifread(h3, 1, 40, hdr);

				// fprintf(stdout,"@%i: <%s>\n",pos3,(char*)h3+1);
				if (!strncmp((char*)h3+1,"TIME",4)) {
					sscanf((char*)(h3+5),"%02u%02u%02u",&tm_time.tm_hour,&tm_time.tm_min,&tm_time.tm_sec);
				}

				typeof(hdr->NS) NS = h3[38];
				typeof(hdr->SampleRate) SampleRate = leu16p(h3+26) & 0x3fff;
				nrec_t NRec = (nrec_t)(leu32p(h3+28) * SampleRate * 0.1);
				size_t HeadLen = pos2 + 39 + 10*NS;

				hdr->EVENT.TYP[hdr->EVENT.N] = 0x7ffe;
				hdr->EVENT.POS[hdr->EVENT.N] = Total_NRec;
				hdr->EVENT.DUR[hdr->EVENT.N] = NRec;
				hdr->EVENT.CHN[hdr->EVENT.N] = 0;
#if (BIOSIG_VERSION >= 10500)
				hdr->EVENT.TimeStamp[hdr->EVENT.N] = 0;
#endif
				Total_NRec += NRec;
				hdr->EVENT.N++;
				numSegments++;

				--TARGET_SEGMENT;	// decrease target segment counter
				if (TARGET_SEGMENT != 0) {
					continue;
				}

				hdr->T0 = tm_time2gdf_time(&tm_time);
				hdr->NS = NS;
				hdr->SampleRate = SampleRate;
				hdr->EVENT.SampleRate = SampleRate;
				hdr->NRec = NRec;
				hdr->HeadLen = HeadLen;
				hdr->SPR = 1;
				int16_t gdftyp = 128; // Nihon-Kohden int16 format
				hdr->AS.bpb = ((hdr->NS*GDFTYP_BITS[gdftyp])>>3)+2;

				// fprintf(stdout,"NK k=%i <%s> k2=%i <%s>\n",k,h2+1,k2,h3+1);
				// fprintf(stdout,"[%i %i]:pos=%u (%x) length=%Li(%Lx).\n",k,k2,pos2,pos2,hdr->NRec*(hdr->NS+1)*2,hdr->NRec*(hdr->NS+1)*2);

				h3 = (uint8_t*)realloc(h3,32 + hdr->NS*10);
				pos3 += ifread(h3+pos3, 1, 32+hdr->NS*10 - pos3, hdr);

				hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
				int k3;
				for (k3=0; k3<hdr->NS; k3++) {
					CHANNEL_TYPE *hc = hdr->CHANNEL+k3;
					uint8_t u8 = h3[39+k3*10];
					switch (u8) {
					case 0: strcpy(hc->Label,"Fp1"); break;
					case 1: strcpy(hc->Label,"Fp2"); break;
					case 2: strcpy(hc->Label,"F3"); break;
					case 3: strcpy(hc->Label,"F4"); break;
					case 4: strcpy(hc->Label,"C3"); break;
					case 5: strcpy(hc->Label,"C4"); break;
					case 6: strcpy(hc->Label,"P3"); break;
					case 7: strcpy(hc->Label,"P4"); break;
					case 8: strcpy(hc->Label,"O1"); break;
					case 9: strcpy(hc->Label,"O2"); break;
					case 10: strcpy(hc->Label,"F7"); break;
					case 11: strcpy(hc->Label,"F8"); break;
					case 12: strcpy(hc->Label,"T3"); break;
					case 13: strcpy(hc->Label,"T4"); break;
					case 14: strcpy(hc->Label,"T5"); break;
					case 15: strcpy(hc->Label,"T6"); break;
					case 16: strcpy(hc->Label,"Fz"); break;
					case 17: strcpy(hc->Label,"Cz"); break;
					case 18: strcpy(hc->Label,"Pz"); break;
					case 19: strcpy(hc->Label,"E"); break;
					case 20: strcpy(hc->Label,"PG1"); break;
					case 21: strcpy(hc->Label,"PG2"); break;
					case 22: strcpy(hc->Label,"A1"); break;
					case 23: strcpy(hc->Label,"A2"); break;
					case 24: strcpy(hc->Label,"T1"); break;
					case 25: strcpy(hc->Label,"T2"); break;

					case 74: strcpy(hc->Label,"BN1"); break;
					case 75: strcpy(hc->Label,"BN2"); break;
					case 76: strcpy(hc->Label,"Mark1"); break;
					case 77: strcpy(hc->Label,"Mark2"); break;

					case 100: strcpy(hc->Label,"X12/BP1"); break;
					case 101: strcpy(hc->Label,"X13/BP2"); break;
					case 102: strcpy(hc->Label,"X14/BP3"); break;
					case 103: strcpy(hc->Label,"X15/BP4"); break;

					case 254: strcpy(hc->Label,"-"); break;
					case 255: strcpy(hc->Label,"Z"); break;
					default:
						if 	((25<u8)&&(u8<=36)) sprintf(hc->Label,"X%u",u8-25);
						else if ((36<u8)&&(u8<=41)) strcpy(hc->Label,"-");
						else if ((41<u8)&&(u8<=73)) sprintf(hc->Label,"DC%02u",u8-41);
						else if ((77<u8)&&(u8<=99)) strcpy(hc->Label,"-");
						else if	((103<u8)&&(u8<=254)) sprintf(hc->Label,"X%u",u8-88);
					}

					if ((41<u8) && (u8<=73)) {
						hc->PhysDimCode = 4274;	// mV
						hc->PhysMin = -12002.9;
						hc->PhysMax =  12002.56;
					} else {
						hc->PhysDimCode = 4275;    // uV
						hc->PhysMin = -3200.0;
						hc->PhysMax =  3200.0*((1<<15)-1)/(1<<15);
					}
					hc->GDFTYP =  128;	// Nihon-Kohden int16 format
					hc->DigMax =  32767.0;
					hc->DigMin = -32768.0;

					hc->Cal   	= (hc->PhysMax - hc->PhysMin) / (hc->DigMax - hc->DigMin);
					hc->Off   	=  hc->PhysMin - hc->Cal * hc->DigMin;
					hc->SPR    = 1;
				    	hc->LeadIdCode = 0;
					hc->OnOff  = 1;
					hc->Transducer[0] = 0;
					hc->bi = (k3*GDFTYP_BITS[gdftyp])>>3;

					// hc->LowPass  = 0.1;
					// hc->HighPass = 100;
					// hdr->CHANNEL[k3].Notch    = 0;
				}
			}
		}
		free(h2);
		free(h3);
		ifseek(hdr, hdr->HeadLen, SEEK_SET);
		if ((numSegments>1) && (hdr->FLAG.TARGETSEGMENT==1))
			fprintf(stdout,"File %s has more than one (%i) segment; use TARGET_SEGMENT argument to select other segments.\n",hdr->FileName,numSegments);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d) %s(...) <%s>\n",__FILE__,__LINE__,__func__,fn);

		/* read .log */
		char *c = strrchr(fn,'.');
		if (c != NULL) {
			strcpy(c+1,"log");
			FILE *fid = fopen(fn,"rb");
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d) %s(...) <%s>\n",__FILE__,__LINE__,__func__,fn);
			if (fid == NULL) {
				strcpy(c+1,"LOG");
				fid = fopen(fn,"rb");
			}
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d) %s(...) <%s>\n",__FILE__,__LINE__,__func__,fn);
			if (fid != NULL) {

				count = 0;
			 	while (!feof(fid)) {
					size_t c = max(2*count, 11520);
					LOG = (char*) realloc(LOG, c+1);
					count += fread(LOG+count, 1, c-count, fid);
			 	}
				fclose(fid);
				LOG[count]=0;

				for (k=0; k<(unsigned)LOG[145]; k++) {

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d) %s(...) %i,%i<%s>\n",__FILE__,__LINE__,__func__,(int)k,(int)count,fn);

					//uint32_t lba = leu32p(LOG+146+k*20);
					uint32_t lba = atoi(LOG+146+k*20);

				if (VERBOSE_LEVEL>7) fprintf(stdout,"EEG1100 [253]: <%d> %d 0x%x\n",(int)k,lba,lba);
//break;		// FIXME: there is at least one EEG1100C file that breaks this
					uint32_t N = LOG[lba+18];

				if (VERBOSE_LEVEL>7) fprintf(stdout,"EEG1100 [254]: <%d> %d %d\n",(int)k,(int)lba,N);

					if (reallocEventTable(hdr, hdr->EVENT.N+N) == SIZE_MAX) {
						biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
						return (hdr);
					};

					size_t k1;
					for (k1=0; k1<N; k1++) {

						if (VERBOSE_LEVEL>7) fprintf(stdout,"EEG1100 [257]: [%d,%d] N=%d,%d\n",(int)k,(int)k1,N,hdr->EVENT.N);
						if (VERBOSE_LEVEL>8) fprintf(stdout,"EEG1100 [258]: [%d,%d] N=%d, <%s>\n",(int)k,(int)k1,hdr->EVENT.N,(char*)(LOG+lba+20+k1*45));

//						FreeTextEvent(hdr,hdr->EVENT.N,(char*)(LOG+lba+20+k1*45));
					if (VERBOSE_LEVEL>7) {
						fprintf(stdout,"   <%s>\n   <%s>\n",(char*)(LOG+lba+9+k1*45),(char*)(LOG+lba+29+k1*45));
						int kk; for (kk=0; kk<45; kk++) putchar(LOG[lba+9+k1*45+kk]);
						putchar('\n');
					}

						char *desc = (char*)(LOG+lba+9+k1*45);

						if (desc[0] == 0) continue;
/*
						char secstr[7];
						memcpy(secstr, LOG+lba+29+k1*45, 6);
						secstr[6] = 0;
*/
						if (VERBOSE_LEVEL>7) fprintf(stdout,"EEG1100 [259]: <%s> <%s>",desc,(char*)LOG+lba+29+k1*45);

/*
						int c = sscanf((char*)(LOG+lba+46+k1*45),"(%02u%02u%02u%02u%02u%02u)",&tm_time.tm_year,&tm_time.tm_mon,&tm_time.tm_mday,&tm_time.tm_hour,&tm_time.tm_min,&tm_time.tm_sec);

						if (c<6) continue;

						tm_time.tm_year += tm_time.tm_year<20 ? 100:0;
						tm_time.tm_mon--;	// Jan=0, Feb=1, ...
						gdf_time t0 = tm_time2gdf_time(&tm_time);

						char tmpstr[80];
						strftime(tmpstr,80,"%Y-%m-%d %H:%M:%S", &tm_time);
						if (VERBOSE_LEVEL>7) fprintf(stdout,"EEG1100 [261]: %s\n",tmpstr);
*/
						if (1) //(t0 >= hdr->T0)
						{
							hdr->EVENT.TYP[hdr->EVENT.N] = 1;
							//hdr->EVENT.POS[hdr->EVENT.N] = (uint32_t)(ldexp(t0 - hdr->T0,-32)*86400*hdr->SampleRate);        // 0-based indexing
							//hdr->EVENT.POS[hdr->EVENT.N] = (uint32_t)(atoi(strtok((char*)(LOG+lba+29+k1*45),"("))*hdr->SampleRate);
							hdr->EVENT.POS[hdr->EVENT.N] = (uint32_t)(atoi(strtok((char*)(LOG+lba+29+k1*45),"("))*hdr->SampleRate);
							hdr->EVENT.DUR[hdr->EVENT.N] = 0;
							hdr->EVENT.CHN[hdr->EVENT.N] = 0;
#if (BIOSIG_VERSION >= 10500)
							hdr->EVENT.TimeStamp[hdr->EVENT.N] = 0;
#endif
							FreeTextEvent(hdr,hdr->EVENT.N,desc);
							hdr->EVENT.N++;
						}
					}
				}
			}
		}
		hdr->AS.auxBUF = (uint8_t*)LOG;
		free(fn);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d) %s(...)\n",__FILE__,__LINE__,__func__);
	}

	else if (hdr->TYPE==EEProbe) {
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "EEProbe currently not supported");
	}

	else if (hdr->TYPE==EGI) {

		fprintf(stdout,"Reading EGI is under construction\n");

		uint16_t NEC = 0;	// specific for EGI format
		uint16_t gdftyp = 3;

		// BigEndian
		hdr->FILE.LittleEndian = 0;
		hdr->VERSION	= beu32p(hdr->AS.Header);
		if      (hdr->VERSION==2 || hdr->VERSION==3)	gdftyp = 3;	// int32
		else if (hdr->VERSION==4 || hdr->VERSION==5)	gdftyp = 16;	// float
		else if (hdr->VERSION==6 || hdr->VERSION==7)	gdftyp = 17;	// double

		tm_time.tm_year = beu16p(hdr->AS.Header+4) - 1900;
		tm_time.tm_mon  = beu16p(hdr->AS.Header+6) - 1;
		tm_time.tm_mday = beu16p(hdr->AS.Header+8);
		tm_time.tm_hour = beu16p(hdr->AS.Header+10);
		tm_time.tm_min  = beu16p(hdr->AS.Header+12);
		tm_time.tm_sec  = beu16p(hdr->AS.Header+14);
		// tm_time.tm_sec  = beu32p(Header1+16)/1000; // not supported by tm_time

		hdr->T0 = tm_time2gdf_time(&tm_time);
		hdr->SampleRate = beu16p(hdr->AS.Header+20);
		hdr->NS         = beu16p(hdr->AS.Header+22);
		// uint16_t  Gain  = beu16p(Header1+24);	// not used
		uint16_t  Bits  = beu16p(hdr->AS.Header+26);
		uint16_t PhysMax= beu16p(hdr->AS.Header+28);
		size_t POS;
		if (hdr->AS.Header[3] & 0x01)
		{ 	// Version 3,5,7
			POS = 32;
			for (k=0; k < beu16p(hdr->AS.Header+30); k++) {
				char tmp[256];
				int  len = hdr->AS.Header[POS];
				strncpy(tmp,Header1+POS,len);
				tmp[len]=0;
				if (VERBOSE_LEVEL>7)
					fprintf(stdout,"EGI categorie %i: <%s>\n",(int)k,tmp);

				POS += *(hdr->AS.Header+POS);	// skip EGI categories
				if (POS > count-8) {
					hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,2*count);
					count += ifread(hdr->AS.Header,1,count,hdr);
				}
			}

			hdr->NRec= beu16p(hdr->AS.Header+POS);
			hdr->SPR = beu32p(hdr->AS.Header+POS+2);
			NEC = beu16p(hdr->AS.Header+POS+6);	// EGI.N
			POS += 8;
		}
		else
		{ 	// Version 2,4,6
			hdr->NRec = beu32p(hdr->AS.Header+30);
			NEC = beu16p(hdr->AS.Header+34);	// EGI.N
			hdr->SPR  = 1;
			/* see also end-of-sopen
			hdr->AS.spb = hdr->SPR+NEC;
			hdr->AS.bpb = (hdr->NS + NEC)*GDFTYP_BITS[hdr->CHANNEL[0].GDFTYP]>>3;
			*/
			POS = 36;
		}

		/* read event code description */
		hdr->AS.auxBUF = (uint8_t*) realloc(hdr->AS.auxBUF,5*NEC);
		hdr->EVENT.CodeDesc = (typeof(hdr->EVENT.CodeDesc)) realloc(hdr->EVENT.CodeDesc,257*sizeof(*hdr->EVENT.CodeDesc));
		hdr->EVENT.CodeDesc[0] = "";	// typ==0, is always empty
		hdr->EVENT.LenCodeDesc = NEC+1;
		for (k=0; k < NEC; k++) {
			memcpy(hdr->AS.auxBUF+5*k,Header1+POS,4);
			hdr->AS.auxBUF[5*k+4]=0;
			hdr->EVENT.CodeDesc[k+1] = (char*)hdr->AS.auxBUF+5*k;
			POS += 4;
		}
		hdr->HeadLen = POS;
		ifseek(hdr,hdr->HeadLen,SEEK_SET);

		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		for (k=0; k<hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			hc->GDFTYP = gdftyp;
			hc->PhysDimCode = 4275;  // "uV"
			hc->LeadIdCode  = 0;
			hc->Transducer[0] = 0;
			sprintf(hc->Label,"# %03i",(int)k);
			hc->Cal	= PhysMax/ldexp(1,Bits);
			hc->Off	= 0;
			hc->SPR	= hdr->SPR;
			hc->bi	= k*hdr->SPR*(GDFTYP_BITS[gdftyp]>>3);

			if (VERBOSE_LEVEL>8)
				fprintf(stdout,"SOPEN(EGI): #%i %i %i\n",(int)k,Bits, PhysMax);

			if (Bits && PhysMax) {
				hc->PhysMax = PhysMax;
				hc->PhysMin = -PhysMax;
				hc->DigMax  = ldexp(1,Bits);
				hc->DigMin  = ldexp(-1,Bits);
			}
			else {
/*			hc->PhysMax = PhysMax;
			hc->PhysMin = -PhysMax;
			hc->DigMax  = ldexp(1,Bits);
			hc->DigMin  = ldexp(-1,Bits);
*/			hc->Cal     = 1.0;
			hc->OnOff   = 1;
			}
		}
		hdr->AS.bpb = (hdr->NS*hdr->SPR + NEC) * (GDFTYP_BITS[gdftyp]>>3);
		if (hdr->AS.Header[3] & 0x01)	// triggered
			hdr->AS.bpb += 6;

		size_t N = 0;
		if (NEC > 0) {
			/* read event information */

			size_t sz	= GDFTYP_BITS[gdftyp]>>3;
			uint8_t *buf 	= (uint8_t*)calloc(NEC, sz);
			uint8_t *buf8 	= (uint8_t*)calloc(NEC*2, 1);
			size_t *ix 	= (size_t*)calloc(NEC, sizeof(size_t));

			size_t skip	= hdr->AS.bpb - NEC * sz;
			ifseek(hdr, hdr->HeadLen + skip, SEEK_SET);
			typeof(NEC) k1;
			nrec_t k;
			for (k=0; (k < hdr->NRec*hdr->SPR) && !ifeof(hdr); k++) {
				ifread(buf, sz,   NEC, hdr);
				ifseek(hdr, skip, SEEK_CUR);

				int off0, off1;
				if (k & 0x01)
				{ 	off0 = 0; off1=NEC; }
				else
				{ 	off0 = NEC; off1=0; }

				memset(buf8+off1,0,NEC);	// reset
				for (k1=0; k1 < NEC * sz; k1++)
					if (buf[k1]) buf8[ off1 + k1/sz ] = 1;

				for (k1=0; k1 < NEC ; k1++) {
					if (buf8[off1 + k1] && !buf8[off0 + k1]) {
						/* rising edge */
						ix[k1] = k;
					}
					else if (!buf8[off1 + k1] && buf8[off0 + k1]) {
						/* falling edge */
						if (N <= (hdr->EVENT.N + NEC*2)) {
							N += (hdr->EVENT.N+NEC)*2;	// allocate memory for this and the terminating line.
							if (reallocEventTable(hdr, N) == SIZE_MAX) {
								biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
								return (hdr);
							};
						}
						hdr->EVENT.TYP[hdr->EVENT.N] = k1+1;
						hdr->EVENT.POS[hdr->EVENT.N] = ix[k1];        // 0-based indexing
						hdr->EVENT.CHN[hdr->EVENT.N] = 0;
						hdr->EVENT.DUR[hdr->EVENT.N] = k-ix[k1];
#if (BIOSIG_VERSION >= 10500)
						hdr->EVENT.TimeStamp[hdr->EVENT.N] = 0;
#endif
						hdr->EVENT.N++;
						ix[k1] = 0;
					}
				}
			}

			for (k1 = 0; k1 < NEC; k1++)
			if (ix[k1]) {
				/* end of data */
				hdr->EVENT.TYP[hdr->EVENT.N] = k1+1;
				hdr->EVENT.POS[hdr->EVENT.N] = ix[k1];        // 0-based indexing
				hdr->EVENT.CHN[hdr->EVENT.N] = 0;
				hdr->EVENT.DUR[hdr->EVENT.N] = k-ix[k1];
#if (BIOSIG_VERSION >= 10500)
				hdr->EVENT.TimeStamp[hdr->EVENT.N] = 0;
#endif
				hdr->EVENT.N++;
				ix[k1] = 0;
			}

			hdr->EVENT.SampleRate = hdr->SampleRate;
			free(buf);
			free(buf8);
			free(ix);
		}
		ifseek(hdr,hdr->HeadLen,SEEK_SET);

		/* TODO: check EGI format */
	}


#ifdef WITH_EGIS
	else if (hdr->TYPE==EGIS) {
		fprintf(stdout,"Reading EGIS is under construction\n");

#if __BYTE_ORDER == __BIG_ENDIAN
		char FLAG_SWAP = hdr->FILE.LittleEndian;
#elif __BYTE_ORDER == __LITTLE_ENDIAN
		char FLAG_SWAP = hdr->FILE.LittleEndian;
#endif
		hdr->VERSION = *(int16_t*) mfer_swap8b(hdr->AS.Header+4, sizeof(int16_t), char FLAG_SWAP);
		hdr->HeadLen = *(uint16_t*) mfer_swap8b(hdr->AS.Header+6, sizeof(uint16_t), char FLAG_SWAP);
		//hdr->HeadLen = *(uint32_t*) mfer_swap8b(hdr->AS.Header+8, sizeof(uint32_t), char FLAG_SWAP);

		/* read file */
		hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,hdr->HeadLen+1);
		if (hdr->HeadLen > count)
			count += ifread(hdr->AS.Header+count, 1, hdr->HeadLen-count, hdr);
		hdr->AS.Header[count]=0;

	}
#endif

#ifdef WITH_EMBLA
	else if (hdr->TYPE==EMBLA) {

		while (!ifeof(hdr)) {
			size_t bufsiz = max(2*count, PAGESIZE);
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, bufsiz+1);
			count  += ifread(hdr->AS.Header+count, 1, bufsiz-count, hdr);
		}
		hdr->AS.Header[count]=0;
		hdr->HeadLen = count;
		ifclose(hdr);

		count = 48;

		int chan;
		uint32_t cal;
		float chan32;
		uint16_t pdc=0;
		while (count+8 < hdr->HeadLen) {
			uint32_t tag = leu32p(hdr->AS.Header+count);
			uint32_t len = leu32p(hdr->AS.Header+count+4);
			count+=8;
/*
			uint32_t taglen[2];
			uint32_t *tag = &taglen[0];
			uint32_t *len = &taglen[1];
			size_t c = ifread(taglen, 4, 2, hdr);
			if (ifeof(hdr)) break;
*/
			if (VERBOSE_LEVEL > 7) {
				int ssz = min(80,len);
				char S[81];
				strncpy(S, hdr->AS.Header+count, ssz); S[ssz]=0;
				fprintf(stdout,"tag %8d [%d]: <%s>\n",tag,len, S);
			}

			switch (tag) {
			case 32:
				hdr->SPR = len/2;
//				hdr->AS.rawdata = realloc(hdr->AS.rawdata,len);
				break;
			case 133:
				chan = leu16p(hdr->AS.Header+count);
				fprintf(stdout,"\tchan=%d\n",chan);
				break;
			case 134:	// Sampling Rate
				hdr->SampleRate=leu32p(hdr->AS.Header+count)/1000.0;
				fprintf(stdout,"\tFs=%g #134\n",hdr->SampleRate);
				break;
			case 135:	//
				cal=leu32p(hdr->AS.Header+count)/1000.0;
				hc->Cal = (cal==1 ? 1.0 : cal*1e-9);
				break;
			case 136:	// session count
				fprintf(stdout,"\t%d (session count)\n",leu32p(hdr->AS.Header+count));
				break;
			case 137:	// Sampling Rate
				hdr->SampleRate=lef64p(hdr->AS.Header+count);
				fprintf(stdout,"\tFs=%g #137\n",hdr->SampleRate);
				break;
			case 141:
				chan32 = lef32p(hdr->AS.Header+count);
				fprintf(stdout,"\tchan32=%g\n",chan32);
				break;
			case 144:	// Label
				strncpy(hc->Label, hdr->AS.Header+count, MAX_LENGTH_LABEL);
				hc->Label[min(MAX_LENGTH_LABEL,len)]=0;
				break;
			case 153:	// Label
				pdc=PhysDimCode(hdr->AS.Header+count);
				fprintf(stdout,"\tpdc=0x%x\t<%s>\n",pdc,PhysDim3(pdc));
				break;
			case 208:	// Patient Name
				if (!hdr->FLAG.ANONYMOUS)
					strncpy(hdr->Patient.Name, hdr->AS.Header+count, MAX_LENGTH_NAME);
					hdr->Patient.Name[min(MAX_LENGTH_NAME,len)]=0;
				break;
			case 209:	// Patient Name
				strncpy(hdr->Patient.Id, hdr->AS.Header+count, MAX_LENGTH_PID);
				hdr->Patient.Id[min(MAX_LENGTH_PID,len)]=0;
				break;
			default:
				;
			}
			count+=len;
		}


		hdr->NS = 1;
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		for (k=0; k < hdr->NS; k++) {
			CHANNEL_TYPE* hc = hdr->CHANNEL+k;
			hc->OnOff    = 1;
			hc->GDFTYP   = 3;
			hc->SPR      = hdr->SPR;
			hc->Cal      = 0.1;
			hc->Off      = 0.0;
			hc->Transducer[0] = '\0';
			hc->LowPass  = NAN;
			hc->HighPass = NAN;
			hc->PhysMax  =  3276.7;
			hc->PhysMin  = -3276.8;
			hc->DigMax   =  32767;
			hc->DigMin   = -32768;
			hc->LeadIdCode  = 0;
			hc->PhysDimCode = 4275;	//uV
			hc->bi   = k*hdr->SPR*2;

			char *label = (char*)(hdr->AS.Header+1034+k*512);
			const size_t len    = min(16,MAX_LENGTH_LABEL);
			if ( (hdr->AS.Header[1025+k*512]=='E') && strlen(label)<13) {
				strcpy(hc->Label, "EEG ");
				strcat(hc->Label, label);		// Flawfinder: ignore
			}
			else {
				strncpy(hc->Label, label, len);
				hc->Label[len]=0;
			}
		}

	}

#endif // EMBLA
	else if (hdr->TYPE==EMSA) {

		hdr->NS = (uint8_t)hdr->AS.Header[3];
		hdr->HeadLen = 1024 + hdr->NS*512;
		if (count < hdr->HeadLen) {
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, hdr->HeadLen);
			count  += ifread(hdr->AS.Header+count, 1, hdr->HeadLen-count, hdr);
		}
		if (count < hdr->HeadLen) {
			biosigERROR(hdr, B4C_INCOMPLETE_FILE, "EMSA file corrupted");
		}
		hdr->HeadLen = count;

		sprintf(hdr->Patient.Id,"%05i",leu32p(hdr->AS.Header+4));
		unsigned seq_nr = hdr->AS.Header[8];
		uint16_t fs = leu16p(hdr->AS.Header+9);
		if (fs % 10)
			hdr->SPR = fs;
		else
			hdr->SPR = fs/10;

		hdr->AS.bpb = 2*hdr->NS*hdr->SPR;
		hdr->NRec = (hdr->FILE.size - hdr->HeadLen) / hdr->AS.bpb;
		hdr->SampleRate = fs;

		{
			struct tm t;
			char tmp[9];
			// Birthday
			strncpy(tmp, (char*)(hdr->AS.Header+169), 8);
			for (k=0; k<8; k++)
				if (tmp[k]<'0' || tmp[k]>'9')
					biosigERROR(hdr, B4C_FORMAT_UNKNOWN, NULL); // error ;
			tmp[8] = 0; t.tm_mday = atoi(tmp+6);
			tmp[6] = 0; t.tm_mon = atoi(tmp+4)-1;
			tmp[4] = 0; t.tm_year  = atoi(tmp+4)-1900;
			t.tm_hour = 12;
			t.tm_min = 0;
			t.tm_sec = 0;
			hdr->Patient.Birthday = tm_time2gdf_time(&t);

			// startdate
			strncpy(tmp, (char*)hdr->AS.Header+205, 8);
			for (k=0; k<8; k++)
				if (tmp[k]<'0' || tmp[k]>'9')
					biosigERROR(hdr, B4C_FORMAT_UNKNOWN, NULL); // error ;
			tmp[8] = 0; t.tm_mday  = atoi(tmp+6);
			tmp[6] = 0; t.tm_mon = atoi(tmp+4)-1;
			tmp[4] = 0; t.tm_year  = atoi(tmp+4)-1900;

			// starttime
			strncpy(tmp, (char*)hdr->AS.Header+214, 8);
			for (k=0; k<8; k++) {
				if ((k==2 || k==5) && tmp[k] != ':')
					biosigERROR(hdr, B4C_FORMAT_UNKNOWN, NULL); // error ;
				else if (tmp[k]<'0' || tmp[k]>'9')
					biosigERROR(hdr, B4C_FORMAT_UNKNOWN, NULL); // error ;
			}
			tmp[8] = 0; t.tm_sec = atoi(tmp+6);
			tmp[5] = 0; t.tm_min = atoi(tmp+3);
			tmp[2] = 0; t.tm_hour= atoi(tmp);
			hdr->T0 = tm_time2gdf_time(&t);

			if (hdr->AS.B4C_ERRNUM)
				biosigERROR(hdr, B4C_FORMAT_UNKNOWN, "Reading EMSA file failed - invalid data / time format");

		}

		size_t len = min(MAX_LENGTH_NAME,30);
		strncpy(hdr->Patient.Name, (char*)hdr->AS.Header+11, len);
		hdr->Patient.Name[len]=0;

		// equipment
		len = min(MAX_LENGTH_MANUF,40);
		strncpy(hdr->ID.Manufacturer._field, (char*)hdr->AS.Header+309, len);
		hdr->ID.Manufacturer._field[len]=0;

		char c = toupper(hdr->AS.Header[203]);
		hdr->Patient.Sex = (c=='M') + (c=='F')*2;
		c = hdr->AS.Header[204];
		hdr->Patient.Handedness = (c=='D') + (c=='E')*2; //D->1: right-handed, E->2: left-handed, 0 unknown

		hdr->Patient.Weight = atoi((char*)(hdr->AS.Header+351));

		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		for (k=0; k < hdr->NS; k++) {
			CHANNEL_TYPE* hc = hdr->CHANNEL+k;
			hc->OnOff    = 1;
			hc->GDFTYP   = 3;
			hc->SPR      = hdr->SPR;
			hc->Cal      = 0.1;
			hc->Off      = 0.0;
			hc->Transducer[0] = '\0';
			hc->LowPass  = NAN;
			hc->HighPass = NAN;
			hc->PhysMax  =  3276.7;
			hc->PhysMin  = -3276.8;
			hc->DigMax   =  32767;
			hc->DigMin   = -32768;
		    	hc->LeadIdCode  = 0;
		    	hc->PhysDimCode = 4275;	//uV
	    		hc->bi   = k*hdr->SPR*2;

		    	char *label = (char*)(hdr->AS.Header+1034+k*512);
		    	len    = min(16,MAX_LENGTH_LABEL);
			if ( (hdr->AS.Header[1025+k*512]=='E') && strlen(label)<13) {
				strcpy(hc->Label, "EEG ");
				strcat(hc->Label, label);	// Flawfinder: ignore
			}
			else {
			    	strncpy(hc->Label, label, len);
				hc->Label[len]=0;
			}
		}

		/* read event file */
		char* tmpfile = (char*)calloc(strlen(hdr->FileName)+4, 1);
		strcpy(tmpfile, hdr->FileName);
		char* ext = strrchr(tmpfile,'.');
		if (ext != NULL)
			strcpy(ext+1,"LBK");	// Flawfinder: ignore
		else
			strcat(tmpfile,".LBK");	// Flawfinder: ignore

		FILE *fid = fopen(tmpfile,"rb");
		if (fid==NULL) {
			if (ext != NULL)
				strcpy(ext+1,"lbk");
			else
				strcat(tmpfile,".lbk");
		}
		if (fid != NULL) {
			size_t N_EVENTS = 0;
			const int sz = 69;
			char buf[sz+1];
			hdr->EVENT.SampleRate = hdr->SampleRate;
			while (!feof(fid)) {
				if (fread(buf,sz,1,fid) <= 0) break;

				// starttime
				char *tmp = buf;
				for (k=0; k<8; k++) {
					if ((k==2 || k==5) && tmp[k] != ':')
						biosigERROR(hdr, B4C_FORMAT_UNKNOWN, NULL); // error ;
					else if (tmp[k]<'0' || tmp[k]>'9')
						biosigERROR(hdr, B4C_FORMAT_UNKNOWN, NULL); // error ;
				}
				tmp[2] = 0;
				tmp[5] = 0;
				tmp[8] = 0;
				size_t tstart = atoi(tmp)*3600 + atoi(tmp+3)*60 + atoi(tmp+6);

				fread(buf,sz,1,fid);

				// endtime
				tmp = buf+9;
				for (k=0; k<8; k++) {
					if ((k==2 || k==5) && tmp[k] != ':')
						biosigERROR(hdr, B4C_FORMAT_UNKNOWN, NULL); // error ;
					else if (tmp[k]<'0' || tmp[k]>'9')
						biosigERROR(hdr, B4C_FORMAT_UNKNOWN, NULL); // error ;
				}
				tmp[2] = 0;
				tmp[5] = 0;
				tmp[8] = 0;
				size_t tend = atoi(tmp)*3600 + atoi(tmp+3)*60 + atoi(tmp+6);
				if (tend<tstart) tend+=24*3600;

				tmp = buf+18;
				k=68;
				while (k>18 && isspace(buf[k])) k--;
				buf[k+1]=0;

				if (hdr->EVENT.N+2 >= N_EVENTS) {
					// memory allocation if needed
					N_EVENTS = max(128, N_EVENTS*2);
					if (reallocEventTable(hdr, N_EVENTS) == SIZE_MAX) {
						biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
						return (hdr);
					};
				}
				FreeTextEvent(hdr,hdr->EVENT.N,(char*)buf);
				hdr->EVENT.POS[hdr->EVENT.N] = tstart*hdr->EVENT.SampleRate;
				hdr->EVENT.DUR[hdr->EVENT.N] = (tend-tstart)*hdr->EVENT.SampleRate;
				hdr->EVENT.CHN[hdr->EVENT.N] = 0;
#if (BIOSIG_VERSION >= 10500)
				hdr->EVENT.TimeStamp[hdr->EVENT.N] = 0;
#endif
				hdr->EVENT.N++;
			}
		}
		free(tmpfile);
	}

    	else if (hdr->TYPE==ePrime) {
		/* read file */
		while (!ifeof(hdr)) {
			size_t bufsiz = max(2*count,1<<16);
		    	hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, bufsiz+1);
		    	count  += ifread(hdr->AS.Header+count, 1, bufsiz-count, hdr);
		}
		hdr->AS.Header[count]=0;
		hdr->HeadLen = count;
		ifclose(hdr);

		struct Target {
			size_t OnsetTime;
			size_t RTTime;
			size_t RT;
			size_t TrigTarget;
			uint8_t RESP;
			char *Stimulus;
		} Target;
		Target.RESP = 0xff;
		Target.Stimulus = NULL;
		Target.OnsetTime = 0;
		Target.RTTime = 0;
		Target.RT = 0;
		Target.TrigTarget = 0;

		int colSubject      = -1, colSampleRate = -1, colDate = -1, colTime = -1, colOnsetTime = -1;
		int colResponseTime = -1, colRTTime = -1, colStimulus = -1, colTrigTarget = -1, colRESP = -1;
		size_t N_EVENTS = 0;
		struct tm t;
		char nextRow = 0;

		int col=0, row=0, len;
		char *f = (char*)hdr->AS.Header;
		while (*f != 0) {
			len = strcspn(f,"\t\n\r");
			col++;
			if (f[len]==9) {
				nextRow = 0;
			}
			else if ( f[len]==10 || f[len]==13 || f[len]==0 ) {
				nextRow = 1;
				if (row>0) {
					if (hdr->EVENT.N+2 >= N_EVENTS) {
						// memory allocation if needed
						N_EVENTS = max(128, N_EVENTS*2);
						if (reallocEventTable(hdr, N_EVENTS) == SIZE_MAX) {
							biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
							return (hdr);
						};
					}

					// add trigger event
					if (Target.Stimulus != NULL || Target.TrigTarget > 0) {
						if (Target.Stimulus)
							FreeTextEvent(hdr, hdr->EVENT.N, Target.Stimulus);
						else
							hdr->EVENT.TYP[hdr->EVENT.N] = Target.TrigTarget;
						hdr->EVENT.POS[hdr->EVENT.N] = Target.OnsetTime;
						hdr->EVENT.DUR[hdr->EVENT.N] = Target.RT;
						hdr->EVENT.CHN[hdr->EVENT.N] = 0;
#if (BIOSIG_VERSION >= 10500)
						hdr->EVENT.TimeStamp[hdr->EVENT.N] = 0;
#endif
						hdr->EVENT.N++;
					}

					if (Target.RESP < 0x0f) {
						// add response event
						hdr->EVENT.TYP[hdr->EVENT.N] = Target.RESP + 0x0140;	// eventcodes.txt: response codes are in the range between 0x0140 to 0x014f
						hdr->EVENT.POS[hdr->EVENT.N] = Target.OnsetTime + Target.RT;
						hdr->EVENT.CHN[hdr->EVENT.N] = 0;
						hdr->EVENT.DUR[hdr->EVENT.N] = 0;
#if (BIOSIG_VERSION >= 10500)
						hdr->EVENT.TimeStamp[hdr->EVENT.N] = 0;
#endif
						hdr->EVENT.N++;
					};
				}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"=======%i: %i\t%s\t%i\t%i\t%i\t%i\n", hdr->EVENT.N-1, (int)Target.TrigTarget, Target.Stimulus, (int)Target.OnsetTime, (int)Target.RT, (int)Target.RTTime, Target.RESP);

				Target.RESP = 0xff;
				Target.Stimulus = NULL;
				Target.OnsetTime = 0;
				Target.RTTime = 0;
				Target.RT = 0;
				Target.TrigTarget = 0;
			}
			f[len] = 0;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"r%i\tc%i\t%s\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\n",(int)row,(int)col,f,colSubject, colSampleRate, colDate, colTime, colOnsetTime, colResponseTime, colStimulus, colRTTime);

			if (row==0) {
				// decode header line
				if (!strcmp(f,"Subject")) {
					colSubject = col;
				}
				else if (!strcmp(f,"Display.RefreshRate")) {
					colSampleRate = col;
				}
				else if (!strcmp(f,"SessionDate")) {
					colDate = col;
				}
				else if (!strcmp(f,"SessionTime")) {
					colTime = col;
				}
				else if (strstr(f,"Target.OnsetTime")) {
					colOnsetTime = col;
				}
				else if (strstr(f,"Target.RTTime")) {
					colRTTime = col;
				}
				else if (strstr(f,"Target.RT")) {
					colResponseTime = col;
				}
				else if (!strcmp(f,"Stimulus")) {
					colStimulus = col;
				}
				else if (!strcmp(f,"TrigTarget")) {
					colTrigTarget = col;
				}
				else if (strstr(f,"Target.RESP")) {
					colRESP = col;
				}
			}
			else {

				// decode line of body
				if (row==1) {
					t.tm_isdst = 0;
					char *eptr;
					if (col==colTime) {
						t.tm_hour = strtol(f, &eptr, 10);
						t.tm_min  = strtol(eptr+1, &eptr, 10);
						t.tm_sec  = strtol(eptr+1, &eptr, 10);
					}
					else if (col==colDate) {
						t.tm_mon  = strtol(f, &eptr, 10)-1;
						t.tm_mday = strtol(eptr+1, &eptr, 10);
						t.tm_year = strtol(eptr+1, &eptr, 10)-1900;
					}
					else if (col==colSubject) {
						strncpy(hdr->Patient.Id, f, MAX_LENGTH_PID);
					}
					else if (col==colSampleRate) {
						hdr->EVENT.SampleRate = atof(f);
					}
				}

				if (col==colOnsetTime) {
					Target.OnsetTime = atol(f);
				}
				else if (col==colResponseTime) {
					Target.RT = atoi(f);
				}
				else if (col==colRTTime) {
					Target.RTTime = atol(f);
				}
				else if (col==colStimulus) {
					Target.Stimulus = f;
				}
				else if (col==colTrigTarget) {
					Target.TrigTarget = atoi(f);
				}
				else if ((col==colRESP) && strlen(f)) {
					Target.RESP = atoi(f);
				}

			}

			f += len+1;
			if (nextRow) {
				f   += strspn(f,"\n\r");
				row += nextRow;
				col = 0;
			}
		};
		hdr->T0 = tm_time2gdf_time(&t);
	}

	else if (hdr->TYPE==SigViewerEventsCSV) {
		while (!ifeof(hdr)) {
			size_t bufsiz = max(2*count,1<<16);
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, bufsiz+1);
			count  += ifread(hdr->AS.Header+count, 1, bufsiz-count, hdr);
		}
		hdr->AS.Header[count]=0;
		hdr->HeadLen = count;
		ifclose(hdr);

		// position,duration,channel,type,name
		size_t N_EVENT=0,N=0;
		char *nextLine=NULL;
		char *line = strtok_r(hdr->AS.Header, "\n\r", &nextLine);	// skip first line
		while (line != NULL) {
			line = strtok_r(NULL, "\n\r" ,&nextLine);
			if (line==NULL) break;

			char *nextToken=NULL;
			char *tok1 = strtok_r(line, ",", &nextToken);
			char *tok2 = strtok_r(NULL, ",", &nextToken);
			char *tok3 = strtok_r(NULL, ",", &nextToken);
			char *tok4 = strtok_r(NULL, ",", &nextToken);
			char *tok5 = strtok_r(NULL, ",", &nextToken);

			if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d): <%s> <%s> <%s> <%s> <%s>\n",__FILE__,__LINE__, tok1,tok2,tok3,tok4,tok5);

			if (!tok1 || !tok2 || !tok3 || !tok4) continue;

			if (N_EVENT <= N) {
				N_EVENT = reallocEventTable(hdr, max(256,N_EVENT*2));
				if (N_EVENT == SIZE_MAX) {
					biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
					return (hdr);
				};
			}

			hdr->EVENT.POS[N] = (uint32_t)atol(tok1);
			hdr->EVENT.DUR[N] = (uint32_t)atol(tok2);

			int CHN = atoi(tok3);
			hdr->EVENT.CHN[N] = (CHN < 0) ? 0 : CHN+1;
			if (hdr->NS < CHN) hdr->NS = CHN+1;

			uint16_t TYP = (uint16_t)atoi(tok4);
			hdr->EVENT.TYP[N] = TYP;

			// read free text event description
			if ((0 < TYP) && (TYP < 255)) {
				if (hdr->EVENT.LenCodeDesc==0) {
					// allocate memory
					hdr->EVENT.LenCodeDesc = 257;
					hdr->EVENT.CodeDesc = (typeof(hdr->EVENT.CodeDesc)) realloc(hdr->EVENT.CodeDesc,257*sizeof(*hdr->EVENT.CodeDesc));
					hdr->EVENT.CodeDesc[0] = "";	// typ==0, is always empty
					for (k=0; k<=256; k++)
						hdr->EVENT.CodeDesc[k] = NULL;
				}
				if (hdr->EVENT.CodeDesc[TYP]==NULL)
					hdr->EVENT.CodeDesc[TYP] = tok5;
			}
			if (TYP>0) N++;		// skip events with TYP==0
		}
		hdr->AS.auxBUF=hdr->AS.Header;
		hdr->AS.Header=NULL;
		hdr->EVENT.SampleRate = NAN;
		hdr->EVENT.N = N;
		hdr->TYPE    = EVENT;
		hdr->NS      = 0;
	}

    	else if (hdr->TYPE==ET_MEG) {
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "FLT/ET-MEG format not supported");
	}

	else if (hdr->TYPE==ETG4000) {
		/* read file */
		while (!ifeof(hdr)) {
			size_t bufsiz = max(2*count,1<<16);
		    	hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, bufsiz+1);
		    	count  += ifread(hdr->AS.Header+count, 1, bufsiz-count, hdr);
		}
		hdr->AS.Header[count]=0;
		hdr->HeadLen = count;
		ifclose(hdr);

		if (VERBOSE_LEVEL==9)
			fprintf(stdout,"Size of File %s is %i\n",hdr->FileName,(int)count);

		/* decode header section */
		char dlm[2];
		dlm[0] = (char)hdr->AS.Header[20];
		dlm[1] = 0;
		hdr->VERSION = -1;
		char FLAG_StimType_STIM = 0;
		char *t = strtok((char*)hdr->AS.Header,"\xA\xD");
		char *ag=NULL, *dg=NULL, *label;
		double lpf=-1.0,hpf=-1.0,age=0.0;
		while (strncmp(t,"Probe",5)) {
			if (VERBOSE_LEVEL==9)
				fprintf(stderr,"-> %s\n",t);

			if (!strncmp(t,"File Version",12))
				hdr->VERSION = atof(strpbrk(t,dlm)+1);
			else if (!strncmp(t,"Name",4))
				strncpy(hdr->Patient.Id,strpbrk(t,dlm)+1,MAX_LENGTH_PID);
			else if (!strncmp(t,"Sex",3))
				hdr->Patient.Sex = ((toupper(*strpbrk(t,dlm)+1)=='F')<<1) + (toupper(*strpbrk(t,dlm)+1)=='M');
			else if (!strncmp(t,"Age",3)) {
				char *tmp1 = strpbrk(t,dlm)+1;
				size_t c = strcspn(tmp1,"0123456789");
				char buf[20];
				age = atof(strncpy(buf,tmp1,19));
				if (tmp1[c]=='y')
					age *= 365.25;
				else if (tmp1[c]=='m')
					age *= 30;
			}
			else if (!strncmp(t,"Date",4)) {
				sscanf(strpbrk(t,dlm)+1,"%d/%d/%d %d:%d",&(tm_time.tm_year),&(tm_time.tm_mon),&(tm_time.tm_mday),&(tm_time.tm_hour),&(tm_time.tm_min));
				tm_time.tm_sec = 0;
				tm_time.tm_year -= 1900;
				tm_time.tm_mon -= 1;
				hdr->T0 = tm_time2gdf_time(&tm_time);
			}
			else if (!strncmp(t,"HPF[Hz]",7))
				hpf = atof(strpbrk(t,dlm)+1);
			else if (!strncmp(t,"LPF[Hz]",7))
				lpf = atof(strpbrk(t,dlm)+1);
			else if (!strncmp(t,"Analog Gain",11))
				ag = strpbrk(t,dlm);
			else if (!strncmp(t,"Digital Gain",12))
				dg = strpbrk(t,dlm)+1;
			else if (!strncmp(t,"Sampling Period[s]",18))
				hdr->SampleRate = 1.0/atof(strpbrk(t,dlm)+1);
			else if (!strncmp(t,"StimType",8))
				FLAG_StimType_STIM = !strncmp(t+9,"STIM",4);

			t = strtok(NULL,"\xA\xD");
		}
		if (VERBOSE_LEVEL==9)
			fprintf(stderr,"\nNS=%i\n-> %s\n",hdr->NS,t);

		hdr->Patient.Birthday = hdr->T0 - (uint64_t)ldexp(age,32);
		hdr->NS = 0;
		while (ag != NULL) {
			++hdr->NS;
			ag = strpbrk(ag+1,dlm);
		}
		hdr->NS >>= 1;

		if (VERBOSE_LEVEL==9)
			fprintf(stderr,"\n-V=%i NS=%i\n-> %s\n",VERBOSE_LEVEL,hdr->NS,t);

	    	label = strpbrk(t,dlm) + 1;
	    	//uint16_t gdftyp = 16;			// use float32 as internal buffer
	    	uint16_t gdftyp = 17;			// use float64 as internal buffer
		double DigMax = 1.0, DigMin = -1.0;
		hdr->FLAG.OVERFLOWDETECTION = 0; 	// automated overflow and saturation detection not supported
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		for (k=0; k < hdr->NS; k++) {
			CHANNEL_TYPE* hc = hdr->CHANNEL+k;
			hc->OnOff    = 1;
			hc->GDFTYP   = gdftyp;
			hc->SPR      = 1;
			hc->Cal      = 1.0;
			hc->Off      = 0.0;
			hc->Transducer[0] = '\0';
			hc->LowPass  = lpf;
			hc->HighPass = hpf;
			hc->PhysMax  = DigMax;
			hc->PhysMin  = DigMin;
			hc->DigMax   = DigMax;
			hc->DigMin   = DigMin;
		    	hc->LeadIdCode  = 0;
		    	hc->PhysDimCode = 65362;	//mmol l-1 mm
	    		hc->bi   = k*GDFTYP_BITS[gdftyp]>>3;

		    	size_t c     = strcspn(label,dlm);
		    	size_t c1    = min(c,MAX_LENGTH_LABEL);
		    	strncpy(hc->Label, label, c1);
		    	hc->Label[c1]= 0;
		    	label += c+1;

			if (VERBOSE_LEVEL>8)
				fprintf(stderr,"-> Label #%02i: len(%i) %s\n",(int)k,(int)c1,hc->Label);
		}
		hdr->SPR    = 1;
		hdr->NRec   = 0;
    		hdr->AS.bpb = hdr->NS*GDFTYP_BITS[gdftyp]>>3;

		/* decode data section */
		// hdr->FLAG.SWAP = 0;
		hdr->FILE.LittleEndian = (__BYTE_ORDER == __LITTLE_ENDIAN);

		uint32_t pos;
		int Mark=0,hh,mm,ss,ds,BodyMovement,RemovalMark,PreScan;
		size_t NEV = 16;
		hdr->EVENT.N = 0;
		hdr->EVENT.SampleRate = hdr->SampleRate;
		hdr->EVENT.DUR = NULL;
		hdr->EVENT.CHN = NULL;

		pos = atol(strtok(NULL,dlm));
		while (pos) {
			hdr->AS.rawdata = (uint8_t*) realloc(hdr->AS.rawdata, (((size_t)hdr->NRec+1) * hdr->NS * GDFTYP_BITS[gdftyp])>>3);
			for (k=0; k < hdr->NS; k++) {
			if (gdftyp==16)
				*(float*)(hdr->AS.rawdata  + (((size_t)hdr->NRec*hdr->NS+k)*(GDFTYP_BITS[gdftyp]>>3))) = (float)atof(strtok(NULL,dlm));
			else if (gdftyp==17)
				*(double*)(hdr->AS.rawdata + (((size_t)hdr->NRec*hdr->NS+k)*(GDFTYP_BITS[gdftyp]>>3))) = atof(strtok(NULL,dlm));
			}
			++hdr->NRec;

			Mark = atoi(strtok(NULL,dlm));
			if (Mark) {
                                if (hdr->EVENT.N+1 >= NEV) {
                                        NEV<<=1;        // double allocated memory
        		 		hdr->EVENT.POS = (uint32_t*) realloc(hdr->EVENT.POS, NEV*sizeof(*hdr->EVENT.POS) );
        				hdr->EVENT.TYP = (uint16_t*) realloc(hdr->EVENT.TYP, NEV*sizeof(*hdr->EVENT.TYP) );
#if (BIOSIG_VERSION >= 10500)
					hdr->EVENT.TimeStamp = (gdf_time*)realloc(hdr->EVENT.TimeStamp, NEV*sizeof(gdf_time));
#endif
        			}
				hdr->EVENT.POS[hdr->EVENT.N] = pos;         // 0-based indexing
				hdr->EVENT.TYP[hdr->EVENT.N] = Mark;
#if (BIOSIG_VERSION >= 10500)
				hdr->EVENT.TimeStamp[hdr->EVENT.N] = 0;
#endif
				if (FLAG_StimType_STIM && !(hdr->EVENT.N & 0x01))
					hdr->EVENT.TYP[hdr->EVENT.N] = Mark | 0x8000;
				++hdr->EVENT.N;
			}
			sscanf(strtok(NULL,dlm),"%d:%d:%d.%d",&hh,&mm,&ss,&ds);
			BodyMovement 	= atoi(strtok(NULL,dlm));
			RemovalMark 	= atoi(strtok(NULL,dlm));
			PreScan 	= atoi(strtok(NULL,"\xA\xD"));

			if (VERBOSE_LEVEL>8)
				fprintf(stdout,"%d: %d %02d:%02d:%02d.%02d %d %d %d\n",pos,Mark,hh,mm,ss,ds,BodyMovement,RemovalMark,PreScan);

			pos = atol(strtok(NULL,dlm));
		};

		if (FLAG_StimType_STIM && (hdr->EVENT.N & 0x01)) {
			/* if needed, add End-Of-Event marker */
			++hdr->EVENT.N;
	 		hdr->EVENT.POS = (uint32_t*) realloc(hdr->EVENT.POS, hdr->EVENT.N*sizeof(*hdr->EVENT.POS) );
			hdr->EVENT.TYP = (uint16_t*) realloc(hdr->EVENT.TYP, hdr->EVENT.N*sizeof(*hdr->EVENT.TYP) );
#if (BIOSIG_VERSION >= 10500)
			hdr->EVENT.TimeStamp = (gdf_time*)realloc(hdr->EVENT.TimeStamp, hdr->EVENT.N*sizeof(gdf_time));
			hdr->EVENT.TimeStamp[hdr->EVENT.N-1] = 0;
#endif
			hdr->EVENT.POS[hdr->EVENT.N-1] = pos;         // 0-based indexing
			hdr->EVENT.TYP[hdr->EVENT.N-1] = Mark | 0x8000;
		}
		hdr->AS.length  = hdr->NRec;
	}

#ifdef WITH_FAMOS
    	else if (hdr->TYPE==FAMOS) {
    		hdr->HeadLen=count;
		sopen_FAMOS_read(hdr);
	}
#endif

    	else if (hdr->TYPE==FEF) {
#ifdef WITH_FEF
		size_t bufsiz = 1l<<24;
		while (!ifeof(hdr)) {
		    	hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,count+bufsiz+1);
		    	count  += ifread(hdr->AS.Header+count,1,bufsiz,hdr);
		}
		hdr->AS.Header[count]=0;
		hdr->HeadLen = count;

		char tmp[9];
    		tmp[8] = 0;
    		memcpy(tmp, hdr->AS.Header+8, 8);
    		hdr->VERSION = atol(tmp)/100.0;
    		memcpy(tmp, hdr->AS.Header+24, 8);
    		hdr->FILE.LittleEndian = !atol(tmp);
    		ifseek(hdr,32,SEEK_SET);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"ASN1 [401] %i\n",(int)count);
		sopen_fef_read(hdr);
		if (VERBOSE_LEVEL>7) fprintf(stdout,"ASN1 [491]\n");
#else
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "VITAL/FEF Format not supported");
		return(hdr);

#endif
	}

	else if (hdr->TYPE==FIFF) {
		hdr->HeadLen = count;
		sopen_fiff_read(hdr);
	}

	else if (hdr->TYPE==GTF) {
		/* read file */
		while (!ifeof(hdr)) {
			size_t bufsiz = max(2*count,1<<16);
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, bufsiz);
			count  += ifread(hdr->AS.Header+count, 1, bufsiz-count, hdr);
		}
		ifclose(hdr);
		hdr->HeadLen    = 512+15306+8146;
		count          -= hdr->HeadLen;

		char tmp[8];
		strncpy(tmp,hdr->AS.Header+34,2); tmp[2]=0;
		hdr->NS  = atoi(tmp);
		strncpy(tmp,hdr->AS.Header+36,3); tmp[3]=0;
		hdr->SampleRate = atoi(tmp);
		hdr->SPR = 10 * hdr->SampleRate;
		if (hdr->NS <= 0 || hdr->SampleRate<=0.0)
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Invalid GTF header");

	        hdr->AS.bpb = hdr->SampleRate*240+2048;
		hdr->NRec   = floor(count / hdr->AS.bpb);
		//hdr->AS.bpb = hdr->NS*GDFTYP_BITS[gdftyp]>>3;

		const float tau[]     = {0.01, 0.03, 0.1, 0.3, 1};	// ignored for now
		const float Lowpass[] = {30, 70};	// ignored for now
		const float Sens[]    = {.5, .7, 1, 1.4, 2, 5, 7, 10, 14, 20, 50, 70, 100, 140, 200};
		// FIXME
		// x    = reshape(s4(13:6:1932,:),32,HDR.NRec*HDR.Dur);
		// Cal  = Sens(x(1:HDR.NS,:)+1)'/4;

		uint16_t gdftyp = 1;			// int8
		double DigMax = 127, DigMin = -127;
		hdr->FLAG.OVERFLOWDETECTION = 1; 	// automated overflow and saturation detection supported
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		for (int k=0; k < hdr->NS; k++) {
			CHANNEL_TYPE* hc = hdr->CHANNEL+k;
			hc->OnOff    = 1;
			hc->GDFTYP   = gdftyp;
			hc->SPR      = hdr->SPR;
			hc->Cal      = 1.0;
			hc->Off      = 0.0;
			hc->Transducer[0] = '\0';
			hc->LowPass  = NAN;	// TODO: get switch info about Lowpass filter
			hc->HighPass = NAN;	// TODO: get switch info about Lowpass filter
			hc->PhysMax  = DigMax;
			hc->PhysMin  = DigMin;
			hc->DigMax   = DigMax;
			hc->DigMin   = DigMin;
			hc->LeadIdCode  = 0;
			hc->PhysDimCode = 4275;	// uV
			hc->bi       = k*GDFTYP_BITS[gdftyp]>>3;

			assert(MAX_LENGTH_LABEL > 32);
			char* tmp = hdr->AS.Header+512+15306+1070+k*32;
			// Trim trailing space
			int c;
			for (c=31; isspace(tmp[c]) && c>0; c--);
			memcpy(hc->Label, tmp, c);
			hc->Label[c] = 0;

			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"%s (line %d) : Label #%02d: len(%d) <%s>\n",__FILE__,__LINE__, k, c, hc->Label);
		}
		hdr->AS.rawdata = hdr->AS.Header+512+15306+8146;
		// hdr->AS.rawdata = hdr->AS.Header+count+9248;

		for (size_t m=0; m < hdr->NRec; m++) {
			size_t t2pos = 9248 + m * hdr->AS.bpb;
			for (size_t l=0; l < hdr->SampleRate*240; l++) {
				;
			}
		}
	}

    	else if (hdr->TYPE==HDF) {
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d): %s(...)\n", __FILE__,__LINE__,__func__);
		if (sopen_hdf5(hdr) != 0) return(hdr);
	}

    	else if (hdr->TYPE==HEKA) {
    		// HEKA PatchMaster file format
		hdr->HeadLen = count;

		FILE *itx = fopen((char*)hdr->aECG, "w");
		hdr->aECG = NULL; 	// reset auxillary pointer

    		sopen_heka(hdr, itx);

		if (itx) fclose(itx);

	}

	else if (hdr->TYPE==IBW) {
		struct stat FileBuf;
		stat(hdr->FileName, &FileBuf);
		hdr->FILE.size = FileBuf.st_size;

		sopen_ibw_read(hdr);
	}

    	else if (hdr->TYPE==ITX) {
		sopen_itx_read(hdr);
	}

    	else if (hdr->TYPE==ISHNE) {

		char flagANN = !strncmp((char*)hdr->AS.Header,"ANN",3);

		fprintf(stderr,"Warning SOPEN(ISHNE): support for ISHNE format is experimental\n");

                   // unknown, generic, X,Y,Z, I-VF, V1-V6, ES, AS, AI
    	        uint16_t Table1[] = {0,0,16,17,18,1,2,87,88,89,90,3,4,5,6,7,8,131,132,133};
    	        size_t len;
    	        struct tm  t;

		hdr->HeadLen = lei32p(hdr->AS.Header+22);
                if (count < hdr->HeadLen) {
		    	hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,hdr->HeadLen);
		    	count  += ifread(hdr->AS.Header+count,1,hdr->HeadLen-count,hdr);
		}
		hdr->HeadLen = count;

                if (VERBOSE_LEVEL>6) {
                	fprintf(stdout,"SOPEN(ISNHE): @%p %i\n",hdr->AS.Header,hdr->HeadLen);
                	fprintf(stdout,"SOPEN(ISNHE): @%p %x %x %x %x %x %x\n",hdr->AS.Header,hdr->AS.Header[132],hdr->AS.Header[133],hdr->AS.Header[134],hdr->AS.Header[135],hdr->AS.Header[136],hdr->AS.Header[137]);
                	for (k=0;k<522;k++) {
				fprintf(stdout,"%02x ",hdr->AS.Header[k]);
				if (k%32==0) fputc('\n',stdout);
			}
                }
		//int offsetVarHdr = lei32p(hdr->AS.Header+18);
		hdr->VERSION = (float)lei16p(hdr->AS.Header+26);

		if (!hdr->FLAG.ANONYMOUS) {
			len = min(40, MAX_LENGTH_NAME);
			char *s;
			s = (char*)(hdr->AS.Header+68);	// lastname
			size_t slen = strlen(s);
			int len1 = min(40, slen);
			memcpy(hdr->Patient.Name, s, len1);
			hdr->Patient.Name[len1] = 0x1f;	// unit separator ascii(31)

			s = (char*)(hdr->AS.Header+28);	// firstname
			int len2 = min(strlen(s), MAX_LENGTH_NAME-len-1);
			strncpy(hdr->Patient.Name+len1+1, s, len2);
			hdr->Patient.Name[len1+len2+1] = 0;
		}
                len = min(20, MAX_LENGTH_PID);
		strncpy(hdr->Patient.Id, (char*)(hdr->AS.Header+108), len);
		hdr->Patient.Id[len] = 0;

                hdr->Patient.Sex = lei16p(hdr->AS.Header+128);
                // Race = lei16p(hdr->AS.Header+128);

                t.tm_mday  = lei16p(hdr->AS.Header + 132);
                t.tm_mon   = lei16p(hdr->AS.Header + 134) - 1;
                t.tm_year  = lei16p(hdr->AS.Header + 136) - 1900;
                t.tm_hour  = 12;
                t.tm_min   = 0;
                t.tm_sec   = 0;
                t.tm_isdst = 0;
                if (VERBOSE_LEVEL>6) {
                	fprintf(stdout,"SOPEN(ISNHE): Birthday: %04i-%02i-%02i %02i:%02i:%02i\n",t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec);
                	fprintf(stdout,"SOPEN(ISNHE): @%p %x %x %x %x %x %x\n",hdr->AS.Header,hdr->AS.Header[132],hdr->AS.Header[133],hdr->AS.Header[134],hdr->AS.Header[135],hdr->AS.Header[136],hdr->AS.Header[137]);
                }
		if (t.tm_mday>0 && t.tm_mon>=0 && t.tm_year>=0)
	                hdr->Patient.Birthday = tm_time2gdf_time(&t);

		t.tm_mday  = leu16p(hdr->AS.Header + 138);
		t.tm_mon   = leu16p(hdr->AS.Header + 140)-1;
		t.tm_year  = leu16p(hdr->AS.Header + 142)-1900;

		t.tm_hour  = leu16p(hdr->AS.Header + 150);
		t.tm_min   = leu16p(hdr->AS.Header + 152);
		t.tm_sec   = leu16p(hdr->AS.Header + 154);
                hdr->T0    = tm_time2gdf_time(&t);

                hdr->NS    = lei16p(hdr->AS.Header + 156);
		hdr->AS.bpb= hdr->NS * 2;
		hdr->SPR   = 1;
                hdr->SampleRate = lei16p(hdr->AS.Header + 272);
                hdr->Patient.Impairment.Heart = lei16p(hdr->AS.Header+230) ? 3 : 0;    // Pacemaker
		{
			struct stat FileBuf;
			stat(hdr->FileName,&FileBuf);
			hdr->FILE.size = FileBuf.st_size;
		}
		if (flagANN) {
			hdr->NRec=0;
			hdr->EVENT.N = (hdr->FILE.size - hdr->HeadLen)/4;
			hdr->EVENT.TYP = (typeof(hdr->EVENT.TYP)) realloc(hdr->EVENT.TYP, hdr->EVENT.N * sizeof(*hdr->EVENT.TYP));
			hdr->EVENT.POS = (typeof(hdr->EVENT.POS)) realloc(hdr->EVENT.POS, hdr->EVENT.N * sizeof(*hdr->EVENT.POS));

                        /* define user specified events according to ECG Annotation format of http://thew-project.org/THEWFileFormat.htm */
			hdr->EVENT.CodeDesc = (typeof(hdr->EVENT.CodeDesc)) realloc(hdr->EVENT.CodeDesc,10*sizeof(*hdr->EVENT.CodeDesc));
			hdr->EVENT.CodeDesc[0]="";
			hdr->EVENT.CodeDesc[1]="Normal beat";
			hdr->EVENT.CodeDesc[2]="Premature ventricular contraction";
			hdr->EVENT.CodeDesc[3]="Supraventricular premature or ectopic beat";
			hdr->EVENT.CodeDesc[4]="Calibration Pulse";
			hdr->EVENT.CodeDesc[5]="Bundle branch block beat";
			hdr->EVENT.CodeDesc[6]="Pace";
			hdr->EVENT.CodeDesc[7]="Artfact";
			hdr->EVENT.CodeDesc[8]="Unknown";
			hdr->EVENT.CodeDesc[9]="NULL";
			hdr->EVENT.LenCodeDesc = 9;

			uint8_t evt[4];
			ifseek(hdr, lei32p(hdr->AS.Header+22), SEEK_SET);
			size_t N = 0, pos=0;
			while (!ifeof(hdr)) {
				if (!ifread(evt, 1, 4, hdr)) break;

				uint16_t typ = 8;
				switch ((char)(evt[0])) {
				case 'N': typ = 1; break;
				case 'V': typ = 2; break;
				case 'S': typ = 3; break;
				case 'C': typ = 4; break;
				case 'B': typ = 5; break;
				case 'P': typ = 6; break;
				case 'X': typ = 7; break;
				case '!': typ = 0x7ffe; break;
				case 'U': typ = 8; break;
				default: continue;
				}
				pos += leu16p(evt+2);
				hdr->EVENT.POS[N] = pos;
				hdr->EVENT.TYP[N] = typ;
				N++;
			}
			hdr->EVENT.N = N;
		}
		else {
			hdr->EVENT.N = 0;
			hdr->NRec = min(leu32p(hdr->AS.Header+14),
				(hdr->FILE.size - hdr->HeadLen)/hdr->AS.bpb );
		}

		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		for (k=0; k < hdr->NS; k++) {
			CHANNEL_TYPE* hc = hdr->CHANNEL+k;
			hc->OnOff    = 1;
			if (hdr->VERSION == 1) {
        			hc->GDFTYP   = 3;        //int16 - 2th complement
        			hc->DigMax   = (double)(int16_t)(0x7fff);
        			hc->DigMin   = (double)(int16_t)(0x8000);
        		}
        		else {
        			hc->GDFTYP   = 4;        //uint16
        			hc->DigMax   = (double)(uint16_t)(0xffff);
        			hc->DigMin   = (double)(uint16_t)(0x0000);
        		}
			hc->SPR      = 1;
			hc->Cal      = lei16p(hdr->AS.Header + 206 + 2*k);
			hc->Off      = 0.0;
			hc->Transducer[0] = '\0';
			hc->LowPass  = NAN;
			hc->HighPass = NAN;
			hc->PhysMax  = hc->Cal * hc->DigMax;
			hc->PhysMin  = hc->Cal * hc->DigMin;
		    	hc->LeadIdCode  = Table1[lei16p(hdr->AS.Header + 158 + 2*k)];
		    	hc->PhysDimCode = 4276;	        // nV

	    		hc->bi       = k*2;
	    		strcpy(hc->Label, LEAD_ID_TABLE[hc->LeadIdCode]);
		}
		ifseek(hdr, lei32p(hdr->AS.Header+22), SEEK_SET);
    		hdr->FILE.POS = 0;
	}

    	else if (hdr->TYPE==Matlab) {
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d): %s(...)\n", __FILE__,__LINE__,__func__);
		if (sopen_matlab(hdr) != 0) return(hdr);
	}

    	else if (hdr->TYPE==MFER) {
		/*
		// ISO/TS 11073-92001:2007(E), Table 5, p.9
		physicalunits({'V','mmHg','Pa','cm H2O', 'mmHg s-1','dyn','N','%','°C','min-1','s-1','Ohm','A','rpm','W','dB','kg','J','dyne s m-2 cm-5','l','l s-1','l min-1','cd'})
		*/
    		const uint16_t MFER_PhysDimCodeTable[30] = {
    			4256, 3872, 3840, 3904,65330,	// Volt, mmHg, Pa, mmH2O, mmHg/s
			3808, 3776,  544, 6048, 2528,	// dyne, N, %, °C, 1/min
    			4264, 4288, 4160,65376, 4032,	// 1/s, Ohm, A, rpm, W
    			6448, 1731, 3968, 6016, 1600,	// dB, kg, J, dyne s m-2 cm-5, l
    			3040, 3072, 4480,    0,    0,	// l/s, l/min, cd
    			   0,    0,    0,    0,    0,	//
		};

		hdr->FLAG.OVERFLOWDETECTION = 0; 	// MFER does not support automated overflow and saturation detection

	    	uint8_t buf[128];
		void* ptrbuf = buf;
		uint8_t gdftyp = 3; 	// default: int16
		uint8_t UnitCode=0;
		double Cal = 1.0, Off = 0.0;
		char SWAP = ( __BYTE_ORDER == __LITTLE_ENDIAN);   // default of MFER is BigEndian
		hdr->FILE.LittleEndian = 0;
		hdr->SampleRate = 1000; 	// default sampling rate is 1000 Hz
		hdr->NS = 1;	 		// default number of channels is 1
		/* TAG */
		uint8_t tag = hdr->AS.Header[0];
    		ifseek(hdr,1,SEEK_SET);
    		int curPos = 1;
		size_t N_EVENT=0;	// number of events, memory is allocated for in the event table.
		while (!ifeof(hdr)) {
			uint32_t len, val32=0;
			int32_t  chan=-1;
			uint8_t tmplen;

			if (tag==255)
				break;
			else if (tag==63) {
				/* CONTEXT */
				curPos += ifread(buf,1,1,hdr);
				chan = buf[0] & 0x7f;
				while (buf[0] & 0x80) {
					curPos += ifread(buf,1,1,hdr);
					chan    = (chan<<7) + (buf[0] & 0x7f);
				}
			}

			/* LENGTH */
			curPos += ifread(&tmplen,1,1,hdr);
			char FlagInfiniteLength = 0;
			if ((tag==63) && (tmplen==0x80)) {
				FlagInfiniteLength = -1; //Infinite Length
				len = 0;
			}
			else if (tmplen & 0x80) {
				tmplen &= 0x7f;
				curPos += ifread(&buf,1,tmplen,hdr);
				len = 0;
				k = 0;
				while (k<tmplen)
					len = (len<<8) + buf[k++];
			}
			else
				len = tmplen;

			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"MFER: tag=%3i chan=%2i len=%i %3i curPos=%i %li\n",tag,chan,tmplen,len,curPos,iftell(hdr));

			/* VALUE */
			if (tag==0) {
				if (len!=1) fprintf(stderr,"Warning MFER tag0 incorrect length %i!=1\n",len);
				curPos += ifread(buf,1,len,hdr);
			}
			else if (tag==1) {
				// Endianity
				if (len!=1) fprintf(stderr,"Warning MFER tag1 incorrect length %i!=1\n",len);
					ifseek(hdr,len-1,SEEK_CUR);
				curPos += ifread(buf,1,1,hdr);
				hdr->FILE.LittleEndian = buf[0];
#if (__BYTE_ORDER == __BIG_ENDIAN)
				SWAP = hdr->FILE.LittleEndian;
#elif (__BYTE_ORDER == __LITTLE_ENDIAN)
				SWAP = !hdr->FILE.LittleEndian;
#endif
			}
			else if (tag==2) {
				// Version
				uint8_t v[3];
				if (len!=3) fprintf(stderr,"Warning MFER tag2 incorrect length %i!=3\n",len);
				curPos += ifread(&v,1,3,hdr);
				hdr->VERSION = v[0] + (v[1]<10 ? v[1]/10.0 : (v[1]<100 ? v[1]/100.0 : v[1]/1000.0));
				}
			else if (tag==3) {
				// character code
				char v[17];
				if (len>16) fprintf(stderr,"Warning MFER tag2 incorrect length %i>16\n",len);
				curPos += ifread(&v,1,len,hdr);
				v[len]  = 0;
if (VERBOSE_LEVEL>7) fprintf(stdout,"MFER: character code <%s>\n",v);
			}
			else if (tag==4) {
				// SPR
				if (len>4) fprintf(stderr,"Warning MFER tag4 incorrect length %i>4\n",len);
				curPos += ifread(buf,1,len,hdr);
				hdr->SPR = *(int64_t*) mfer_swap8b(buf, len, SWAP);
if (VERBOSE_LEVEL>7) fprintf(stdout,"MFER: TLV %i %i %i \n",tag,len,(int)hdr->SPR);
			}
			else if (tag==5)     //0x05: number of channels
			{
				uint16_t oldNS=hdr->NS;
				if (len>4) fprintf(stderr,"Warning MFER tag5 incorrect length %i>4\n",len);
				curPos += ifread(buf,1,len,hdr);
				hdr->NS = *(int64_t*) mfer_swap8b(buf, len, SWAP);
if (VERBOSE_LEVEL>7) fprintf(stdout,"MFER: TLV %i %i %i \n",tag,len,(int)hdr->NS);
				hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS*sizeof(CHANNEL_TYPE));
				for (k=oldNS; k<hdr->NS; k++) {
					CHANNEL_TYPE *hc = hdr->CHANNEL+k;
					hc->SPR = 0;
					hc->PhysDimCode = 4275;	// uV : default value in Table 5, ISO/FDIS 22077-1(E)ISO/WD 22077-1
					hc->Cal = 1.0;
					hc->Off = 0.0;
					hc->OnOff = 1;
					hc->LeadIdCode = 0;
					hc->GDFTYP = 3;
					hc->Transducer[0] = 0;
				}
			}
			else if (tag==6) 	// 0x06 "number of sequences"
			{
				// NRec
				if (len>4) fprintf(stderr,"Warning MFER tag6 incorrect length %i>4\n",len);
				curPos += ifread(buf,1,len,hdr);
				hdr->NRec = *(int64_t*) mfer_swap8b(buf, len, SWAP);
if (VERBOSE_LEVEL>7) fprintf(stdout,"MFER: TLV %i %i %i \n",tag,len,(int)hdr->NRec);
			}
			else if (tag==8) {
				if (len>2) fprintf(stderr,"Warning MFER tag8 incorrect length %i>2\n",len);
				curPos += ifread(buf,1,len,hdr);
			/*	// NOT USED
				// Type of Waveform
				union {
					uint8_t TypeOfWaveForm8[2];
					uint16_t TypeOfWaveForm;
				} t;
				if (len==1)
					t.TypeOfWaveForm = buf[0];
				else {
					t.TypeOfWaveForm8[0] = buf[0];
					t.TypeOfWaveForm8[1] = buf[1];
					if (SWAP)
						t.TypeOfWaveForm = bswap_16(t.TypeOfWaveForm);
				}
			*/
			}
			else if (tag==10) {
				// GDFTYP
				if (len!=1) fprintf(stderr,"warning MFER tag10 incorrect length %i!=1\n",len);
				curPos += ifread(&gdftyp,1,1,hdr);
				if 	(gdftyp==0)	gdftyp=3; // int16
				else if (gdftyp==1)	gdftyp=4; // uint16
				else if (gdftyp==2)	gdftyp=5; // int32
				else if (gdftyp==3)	gdftyp=2; // uint8
				else if (gdftyp==4)	gdftyp=4; // bit16
				else if (gdftyp==5)	gdftyp=1; // int8
				else if (gdftyp==6)	gdftyp=6; // uint32
				else if (gdftyp==7)	gdftyp=16; // float32
				else if (gdftyp==8)	gdftyp=17; // float64
				else if (gdftyp==9)	//gdftyp=2; // 8 bit AHA compression
					fprintf(stdout,"Warning: MFER compressed format not supported\n");
				else			gdftyp=3;
			}
			else if (tag==11)    //0x0B
			{
				// Fs
				if (len>6) fprintf(stderr,"Warning MFER tag11 incorrect length %i>6\n",len);
				double  fval;
				curPos += ifread(buf,1,len,hdr);
				fval = *(int64_t*) mfer_swap8b(buf+2, len-2, SWAP);

				hdr->SampleRate = fval*pow(10.0, (int8_t)buf[1]);
				if (buf[0]==1)  // s
					hdr->SampleRate = 1.0/hdr->SampleRate;

if (VERBOSE_LEVEL>7) fprintf(stdout,"MFER: TLV %i %i %i %i %g \n",tag,len,buf[0], buf[1], hdr->SampleRate);

			}
			else if (tag==12)    //0x0C
			{
				// sampling resolution
				if (len>6) fprintf(stderr,"Warning MFER tag12 incorrect length %i>6\n",len);
				val32   = 0;
				int8_t  v8;
				curPos += ifread(&UnitCode,1,1,hdr);
				curPos += ifread(&v8,1,1,hdr);
				curPos += ifread(buf,1,len-2,hdr);
				Cal = *(int64_t*) mfer_swap8b(buf, len-2, SWAP);
				Cal *= pow(10.0,v8);
				if (!MFER_PhysDimCodeTable[UnitCode])
					fprintf(stderr,"Warning MFER: unsupported physical unit (code=%i)\n", UnitCode);
			}
			else if (tag==13) {
				if (len>8) fprintf(stderr,"Warning MFER tag13 incorrect length %i>8\n",len);
				curPos += ifread(&buf,1,len,hdr);
				if      (gdftyp == 1) Off = ( int8_t)buf[0];
				else if (gdftyp == 2) Off = (uint8_t)buf[0];
				else if (SWAP) {
					if      (gdftyp == 3) Off = ( int16_t)bswap_16(*( int16_t*)ptrbuf);
					else if (gdftyp == 4) Off = (uint16_t)bswap_16(*(uint16_t*)ptrbuf);
					else if (gdftyp == 5) Off = ( int32_t)bswap_32(*( int32_t*)ptrbuf);
					else if (gdftyp == 6) Off = (uint32_t)bswap_32(*(uint32_t*)ptrbuf);
					else if (gdftyp == 7) Off = ( int64_t)bswap_64(*( int64_t*)ptrbuf);
					else if (gdftyp == 8) Off = (uint64_t)bswap_64(*(uint64_t*)ptrbuf);
					else if (gdftyp ==16) {
						*(uint32_t*)ptrbuf = bswap_32(*(uint32_t*)ptrbuf);
						Off = *(float*)ptrbuf;
					}
					else if (gdftyp ==17) {
						*(uint64_t*)ptrbuf = bswap_64(*(uint64_t*)ptrbuf);
						Off = *(double*)ptrbuf;
					}
				}
				else {
					if      (gdftyp == 3) Off = *( int16_t*)ptrbuf;
					else if (gdftyp == 4) Off = *(uint16_t*)ptrbuf;
					else if (gdftyp == 5) Off = *( int32_t*)ptrbuf;
					else if (gdftyp == 6) Off = *(uint32_t*)ptrbuf;
					else if (gdftyp == 7) Off = *( int64_t*)ptrbuf;
					else if (gdftyp == 8) Off = *(uint64_t*)ptrbuf;
					else if (gdftyp ==16) Off = *(float*)ptrbuf;
					else if (gdftyp ==17) Off = *(double*)ptrbuf;
				}
			}
			else if (tag==22) {
				// MWF_NTE (16h): Comment
				char buf[257];
				ifread(buf,1,min(256,len),hdr);
				buf[min(256,len)]=0;
				if (VERBOSE_LEVEL > 7) fprintf(stdout,"MFER comment (tag=22): %s\n",buf);
				size_t POS=0, CHN=0;
				const char *Desc = NULL;

				char *ptrP1 = strstr(buf,"<P=");
				if (ptrP1) {
					ptrP1+=3;
					char *ptrP2 = strstr(ptrP1, ">");
					if (ptrP2) {
						*ptrP2=0;
						ptrP2++;
						POS = atol(ptrP1);
						Desc = ptrP2;
					}
				}
				char *ptrP3 = strstr(buf,"<C=");
				if (ptrP3==NULL) {
					ptrP3 = strstr(buf,"<L=");
				}
				if (ptrP3) {
					ptrP3+=3;
					char *ptrP2 = strstr(ptrP3, ">");
					if (ptrP2) {
						*ptrP2=0;
						ptrP2++;
						CHN = strtol(ptrP3, &ptrP3, 10);
					}
				}
				if (POS>0 && Desc) {
					size_t N = hdr->EVENT.N;
					if (N_EVENT <= N) {
						N_EVENT = reallocEventTable(hdr, N_EVENT);
						if (N_EVENT == SIZE_MAX) {
							biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
							return (hdr);
						};
					}
					hdr->EVENT.POS[N] = POS;
					hdr->EVENT.CHN[N] = CHN;
					FreeTextEvent(hdr, N, Desc);   // sets hdr->EVENT.TYP[n]
					hdr->EVENT.N = N+1;
				}
				curPos += len;
			}
			else if (tag==23) {
				// manufacturer information: "Manufacturer^model^version number^serial number"
				ifread(hdr->ID.Manufacturer._field,1,min(MAX_LENGTH_MANUF,len),hdr);
				if (len>MAX_LENGTH_MANUF) {
					fprintf(stderr,"Warning MFER tag23 incorrect length %i>128\n",len);
					ifseek(hdr,len-MAX_LENGTH_MANUF,SEEK_CUR);
				}
				curPos += len;
				for (k=0; isprint(hdr->ID.Manufacturer._field[k]) && (k<MAX_LENGTH_MANUF); k++) ;
				hdr->ID.Manufacturer._field[k]    = 0;
				hdr->ID.Manufacturer.Name         = strtok(hdr->ID.Manufacturer._field,"^");
				hdr->ID.Manufacturer.Model        = strtok(NULL,"^");
				hdr->ID.Manufacturer.Version      = strtok(NULL,"^");
				hdr->ID.Manufacturer.SerialNumber = strtok(NULL,"^");
		     		if (hdr->ID.Manufacturer.Name == NULL) hdr->ID.Manufacturer.Name="\0";
		     		if (hdr->ID.Manufacturer.Model == NULL) hdr->ID.Manufacturer.Model="\0";
	     			if (hdr->ID.Manufacturer.Version == NULL) hdr->ID.Manufacturer.Version="\0";
	     			if (hdr->ID.Manufacturer.SerialNumber == NULL) hdr->ID.Manufacturer.SerialNumber="\0";
			}
			else if (tag==30)     //0x1e: waveform data
			{
				// data block
				hdr->AS.rawdata = (uint8_t*)realloc(hdr->AS.rawdata,len);
				hdr->HeadLen    = curPos;
				curPos += ifread(hdr->AS.rawdata,1,len,hdr);
				hdr->AS.first = 0;
				hdr->AS.length= hdr->NRec;
			}
			else if (tag==63) {
				uint8_t tag2=255, len2=255;

				count = 0;
				while ((count<len) && !(FlagInfiniteLength && len2==0 && tag2==0)){
					curPos += ifread(&tag2,1,1,hdr);
					curPos += ifread(&len2,1,1,hdr);
					if (VERBOSE_LEVEL==9)
						fprintf(stdout,"MFER: tag=%3i chan=%2i len=%-4i tag2=%3i len2=%3i curPos=%i %li count=%4i\n",tag,chan,len,tag2,len2,curPos,iftell(hdr),(int)count);

					if (FlagInfiniteLength && len2==0 && tag2==0) break;

					count  += (2+len2);
					curPos += ifread(&buf,1,len2,hdr);
					if (tag2==4) {
						// SPR
						if (len2>4) fprintf(stderr,"Warning MFER tag63-4 incorrect length %i>4\n",len2);
						int64_t SPR = *(int64_t*) mfer_swap8b(buf, len2, SWAP);
						hdr->SPR = (chan==0) ? SPR : lcm(SPR, hdr->SPR);
						hdr->CHANNEL[chan].SPR = SPR;

if (VERBOSE_LEVEL>7) fprintf(stdout,"MFER: TLV %i %i %i %i %i %i %i %i %i\n",tag,len, chan, tag2,len2, buf[0], buf[1], (int)hdr->SPR, (int)hdr->CHANNEL[chan].SPR);

					}
					else if (tag2==9) {	//leadname
						if (len2==2)
							hdr->CHANNEL[chan].LeadIdCode = 0;
						else if (len2==1)
							hdr->CHANNEL[chan].LeadIdCode = buf[0];
						else if (len2<=32)
							strncpy(hdr->CHANNEL[chan].Label,(char*)buf,len2);
						else
							fprintf(stderr,"Warning MFER tag63-9 incorrect length %i>32\n",len2);
					}
					else if (tag2==10) {
						// GDFTYP
						if (len2!=1) fprintf(stderr,"warning MFER tag63-10 incorrect length %i!=1\n",len2);
						if 	(buf[0]==0)	gdftyp=3; // int16
						else if (buf[0]==1)	gdftyp=4; // uint16
						else if (buf[0]==2)	gdftyp=5; // int32
						else if (buf[0]==3)	gdftyp=2; // uint8
						else if (buf[0]==4)	gdftyp=4; // bit16
						else if (buf[0]==5)	gdftyp=1; // int8
						else if (buf[0]==6)	gdftyp=6; // uint32
						else if (buf[0]==7)	gdftyp=16; // float32
						else if (buf[0]==8)	gdftyp=17; // float64
						else if (buf[0]==9)	//gdftyp=2; // 8 bit AHA compression
							fprintf(stdout,"Warning: MFER compressed format not supported\n");
						else			gdftyp=3;

						hdr->CHANNEL[chan].GDFTYP = gdftyp;
if (VERBOSE_LEVEL>7) fprintf(stdout,"MFER: TLV %i %i %i %i %i %i %i\n",tag, len, chan, tag2, len2, buf[0], gdftyp);

					}
					else if (tag2==11) {	// sampling rate
						if (len2>6) fprintf(stderr,"Warning MFER tag63-11 incorrect length %i>6\n",len2);
						double  fval;
						fval = *(int64_t*) mfer_swap8b(buf+2, len2-2, SWAP);

						fval *= pow(10.0, (int8_t)(buf[1]));
						if (buf[0]==1)  // s
							fval = 1.0/fval;

						hdr->CHANNEL[chan].SPR = lround(hdr->SPR * fval / hdr->SampleRate);

if (VERBOSE_LEVEL>7) fprintf(stdout,"MFER: TLV %i %i %i %i %i %i %i %g %i %g\n",tag,len, chan, tag2,len2, buf[0], buf[1], fval, (int)hdr->SPR, hdr->SampleRate);

					}
					else if (tag2==12) {	// MWF_SEN (0Ch): Sampling resolution
						CHANNEL_TYPE *hc = hdr->CHANNEL+chan;
						hc->PhysDimCode = 4275;	// uV : default value in Table 5, ISO/FDIS 22077-1(E)ISO/WD 22077-1
						hc->Cal  = *(int64_t*) mfer_swap8b(buf+2, len2-2, SWAP);
						hc->Cal *= pow(10.0, (int8_t)(buf[1]));
						hc->PhysDimCode = MFER_PhysDimCodeTable[buf[0]];
					}

					else if (tag2==13) {	// Offset
						gdftyp = hdr->CHANNEL[chan].GDFTYP;
						if      (gdftyp == 1) Off = ( int8_t)buf[0];
						else if (gdftyp == 2) Off = (uint8_t)buf[0];
						else if (SWAP) {
							if      (gdftyp == 3) Off = ( int16_t)bswap_16(*( int16_t*)buf);
							else if (gdftyp == 4) Off = (uint16_t)bswap_16(*(uint16_t*)buf);
							else if (gdftyp == 5) Off = ( int32_t)bswap_32(*( int32_t*)buf);
							else if (gdftyp == 6) Off = (uint32_t)bswap_32(*(uint32_t*)buf);
							else if (gdftyp == 7) Off = ( int64_t)bswap_64(*( int64_t*)buf);
							else if (gdftyp == 8) Off = (uint64_t)bswap_64(*(uint64_t*)buf);
							else if (gdftyp ==16) {
								*(uint32_t*)ptrbuf = bswap_32(*(uint32_t*)buf);
								Off = *(float*)ptrbuf;
							}
							else if (gdftyp ==17) {
								uint64_t u64 = bswap_64(*(uint64_t*)ptrbuf);
								Off = *(double*)&u64;
							}
						}
						else {
							if      (gdftyp == 3) Off = *( int16_t*)buf;
							else if (gdftyp == 4) Off = *(uint16_t*)buf;
							else if (gdftyp == 5) Off = *( int32_t*)buf;
							else if (gdftyp == 6) Off = *(uint32_t*)buf;
							else if (gdftyp == 7) Off = *( int64_t*)buf;
							else if (gdftyp == 8) Off = *(uint64_t*)buf;
							else if (gdftyp ==16) Off = *(float*)buf;
							else if (gdftyp ==17) Off = *(double*)buf;
						}
						hdr->CHANNEL[chan].Off = Off;
						/* TODO
							convert to Phys/Dig/Min/Max
						*/
					}

					else if (tag2==18) {	// null value
						// FIXME: needed for overflow detection
						if (len2>6)
							fprintf(stderr,"Warning MFER tag63-12 incorrect length %i>6\n", len2);
						if (!MFER_PhysDimCodeTable[UnitCode])
							fprintf(stderr,"Warning MFER: unsupported physical unit (code=%i)\n", UnitCode);

						hdr->CHANNEL[chan].PhysDimCode = MFER_PhysDimCodeTable[UnitCode];
						double cal = *(int64_t*) mfer_swap8b(buf+2, len2-2, SWAP);
						hdr->CHANNEL[chan].Cal = cal * pow(10.0,(int8_t)buf[1]);
					}
					else {
						if (VERBOSE_LEVEL==9)
							fprintf(stdout,"tag=63-%i (len=%i) not supported\n",tag2,len2);
					}
				}
			}
			else if (tag==64)     //0x40
			{
				// preamble
				char tmp[256];
				curPos += ifread(tmp,1,len,hdr);
				if (VERBOSE_LEVEL>7) {
					fprintf(stdout,"Preamble: pos=%i|",curPos);
					for (k=0; k<len; k++) fprintf(stdout,"%c",tmp[k]);
					fprintf(stdout,"|\n");
				}
			}
			else if (tag==65)     //0x41: patient event
			{
				// event table

				curPos += ifread(buf,1,len,hdr);
				if (len>2) {
					size_t N = hdr->EVENT.N;
#ifdef CURRENTLY_NOT_AVAILABLE
				// FIXME: biosig_set_number_of_events is currently part of biosig2 interface
					if (N_EVENT <= N) {
						N_EVENT = biosig_set_number_of_events(hdr, max(16, N*2));
					}

					if (VERBOSE_LEVEL > 7)
						fprintf(stdout,"MFER-event: N=%i\n",hdr->EVENT.N);

#if (BIOSIG_VERSION >= 10500)
					hdr->EVENT.TimeStamp[N] = 0;
#endif

					hdr->EVENT.CHN[N] = 0;
					hdr->EVENT.DUR[N] = 0;
					if (SWAP) {
						hdr->EVENT.TYP[N] = bswap_16(*(uint16_t*)ptrbuf);
						hdr->EVENT.POS[N] = bswap_32(*(uint32_t*)(buf+2));   // 0-based indexing
						if (len>6)
							hdr->EVENT.DUR[N] = bswap_32(*(uint32_t*)(buf+6));
					}
					else {
						hdr->EVENT.TYP[N] = *(uint16_t*)ptrbuf;
						hdr->EVENT.POS[N] = *(uint32_t*)(buf+2);   // 0-based indexing
						if (len>6)
							hdr->EVENT.DUR[N] = *(uint32_t*)(buf+6);
					}
					hdr->EVENT.N = N+1;
#endif //CURRENTLY_NOT_AVAILABLE
				}
			}
			else if (tag==66)     //0x42: NIPB, SpO2(value)
			{
			}
			else if (tag==67)     //0x43: Sample skew
			{
				int skew=0;
				curPos += ifread(&skew, 1, len,hdr);
if (VERBOSE_LEVEL>2)
	fprintf(stdout,"MFER: sample skew %i ns\n",skew);
			}
			else if (tag==70)     //0x46: digital signature
			{
if (VERBOSE_LEVEL>2)
	fprintf(stdout,"MFER: digital signature \n");
			}

			else if (tag==103)     //0x67 Group definition
			{
if (VERBOSE_LEVEL>2)
	fprintf(stdout,"MFER: Group definition\n");
			}

			else if (tag==129)   //0x81
			{
				if (!hdr->FLAG.ANONYMOUS)
					curPos += ifread(hdr->Patient.Name,1,len,hdr);
				else 	{
					ifseek(hdr,len,SEEK_CUR);
					curPos += len;
				}
			}

			else if (tag==130)    //0x82
			{
				// Patient Id
				if (len>64) fprintf(stderr,"Warning MFER tag131 incorrect length %i>64\n",len);
				if (len>MAX_LENGTH_PID) {
					ifread(hdr->Patient.Id,1,MAX_LENGTH_PID,hdr);
					ifseek(hdr,MAX_LENGTH_PID-len,SEEK_CUR);
					curPos += len;
				}
				else
					curPos += ifread(hdr->Patient.Id,1,len,hdr);
			}

			else if (tag==131)    //0x83
			{
				// Patient Age
				if (len!=7) fprintf(stderr,"Warning MFER tag131 incorrect length %i!=7\n",len);
				curPos += ifread(buf,1,len,hdr);
				uint16_t t16;
				memcpy(&t16, buf+3, 2);
				if (SWAP) t16 = bswap_16(t16);
		    		tm_time.tm_year = t16 - 1900;
		    		tm_time.tm_mon  = buf[5]-1;
		    		tm_time.tm_mday = buf[6];
		    		tm_time.tm_hour = 12;
		    		tm_time.tm_min  = 0;
		    		tm_time.tm_sec  = 0;
				hdr->Patient.Birthday  = tm_time2gdf_time(&tm_time);
				//hdr->Patient.Age = buf[0] + cswap_u16(*(uint16_t*)(buf+1))/365.25;
			}
			else if (tag==132)    //0x84
			{
				// Patient Sex
				if (len!=1) fprintf(stderr,"Warning MFER tag132 incorrect length %i!=1\n",len);
				curPos += ifread(&hdr->Patient.Sex,1,len,hdr);
			}
			else if (tag==133)    //0x85
			{
				curPos += ifread(buf,1,len,hdr);
				uint16_t t16, u16;
				memcpy(&t16, buf+3, 2);
				if (SWAP) t16 = bswap_16(t16);
		    		tm_time.tm_year = t16 - 1900;
		    		tm_time.tm_mon  = buf[2] - 1;
		    		tm_time.tm_mday = buf[3];
		    		tm_time.tm_hour = buf[4];
		    		tm_time.tm_min  = buf[5];
		    		tm_time.tm_sec  = buf[6];

				hdr->T0  = tm_time2gdf_time(&tm_time);
				// add milli- and micro-seconds
				memcpy(&t16, buf+7, 2);
				memcpy(&u16, buf+9, 2);
				if (SWAP)
					hdr->T0 += (uint64_t) ( bswap_16(t16) * 1e+3 + bswap_16(u16) * ldexp(1.0,32) / (24*3600e6) );
				else
					hdr->T0 += (uint64_t) (          t16  * 1e+3 +          u16  * ldexp(1.0,32) / (24*3600e6) );
			}
			else if (tag==135)     //0x67 Object identifier
			{
if (VERBOSE_LEVEL>2)
	fprintf(stdout,"MFER: object identifier\n");
			}
			else {
		    		curPos += len;
		    		ifseek(hdr,len,SEEK_CUR);
				if (VERBOSE_LEVEL>7)
					fprintf(stdout,"tag=%i (len=%i) not supported\n",tag,len);
		    	}

		    	if (curPos != iftell(hdr))
				fprintf(stdout,"positions differ %i %li \n",curPos,iftell(hdr));

			/* TAG */
			int sz=ifread(&tag,1,1,hdr);
			curPos += sz;
	 	}
		hdr->FLAG.OVERFLOWDETECTION = 0; 	// overflow detection OFF - not supported
	 	hdr->AS.bpb = 0;
	 	for (k=0; k<hdr->NS; k++) {

	 		if (VERBOSE_LEVEL>8)
	 			fprintf(stdout,"sopen(MFER): #%i\n",(int)k);

			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			hc->Transducer[0] = 0;
	 		if (!hc->PhysDimCode) hc->PhysDimCode = MFER_PhysDimCodeTable[UnitCode];
	 		if (hc->Cal==1.0) hc->Cal = Cal;
	 		hc->Off = Off * hc->Cal;
			if (!hc->SPR) hc->SPR = hdr->SPR;
			if (hc->GDFTYP<16)
				if (hc->GDFTYP & 0x01) {
		 			hc->DigMax = ldexp( 1.0,GDFTYP_BITS[gdftyp]-1) - 1.0;
		 			hc->DigMin = ldexp(-1.0,GDFTYP_BITS[gdftyp]-1);
	 			}
	 			else {
	 				hc->DigMax = ldexp( 1.0,GDFTYP_BITS[gdftyp]);
		 			hc->DigMin = 0.0;
	 			}
	 		else {
	 			hc->DigMax =  INFINITY;
		 		hc->DigMin = -INFINITY;
	 		}
	 		hc->PhysMax = hc->DigMax * hc->Cal + hc->Off;
	 		hc->PhysMin = hc->DigMin * hc->Cal + hc->Off;
	    		hc->OnOff   = 1;
	    		hc->bi      = hdr->AS.bpb;
			hdr->AS.bpb += hdr->SPR*(GDFTYP_BITS[gdftyp]>>3);
	 	}

		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"[MFER] -V=%i NE=%i\n",VERBOSE_LEVEL,hdr->EVENT.N);
	}

	else if (hdr->TYPE==MIT) {
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s(...): %i \n",__FILE__,__LINE__,__func__,VERBOSE_LEVEL);

    		size_t bufsiz = 1024;
	    	while (!ifeof(hdr)) {
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,count+bufsiz);
		    	count += ifread(hdr->AS.Header+count, 1, bufsiz, hdr);
	    	}
	    	ifclose(hdr);

		/* MIT: decode header information */
		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"%s (line %i) %s(...): %s \n",__FILE__,__LINE__,__func__, (char*)hdr->AS.Header);

	    	hdr->SampleRate = 250.0;
	    	hdr->NRec = 0;
	    	hdr->SPR  = 1;
	    	size_t NumberOfSegments = 1;
		char *ptr = (char*)hdr->AS.Header;
	    	char *line;
	    	do line = strtok((char*)hdr->AS.Header,"\x0d\x0a"); while ((line != NULL) && (line[0]=='#'));

		ptr = strpbrk(line,"\x09\x0a\x0d\x20"); 	// skip 1st field
		ptr[0] = 0;
		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"%s (line %i) %s(...): %s \n",__FILE__,__LINE__,__func__, ptr);

		if (strchr(line,'/') != NULL) {
			NumberOfSegments = atol(strchr(line,'/')+1);
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "MIT/HEA/PhysioBank: multi-segment records are not supported");
		}
		hdr->NS = (typeof(hdr->NS))strtod(ptr+1,&ptr);		// number of channels

		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"%s (line %i) %s(...): NS=%i %p\n",__FILE__,__LINE__,__func__, hdr->NS, ptr);

	    	if ((ptr != NULL) && strlen(ptr)) {
			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"%s (line %i) %s(...) MIT : 123: <%s>\n",__FILE__,__LINE__,__func__, ptr);

			hdr->SampleRate = strtod(ptr,&ptr);
			if (ptr[0]=='/') {
				double CounterFrequency = strtod(ptr+1,&ptr);
				if (fabs(CounterFrequency-hdr->SampleRate) > 1e-5) {
					biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "MIT format: Sampling rate and counter frequency differ - this is currently not supported!");
				}
			}
			if (ptr[0]=='(') {
				double BaseCounterValue = strtod(ptr+1,&ptr);
				if (BaseCounterValue) {
					biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "MIT format: BaseCounterValue is not zero - this is currently not supported !");
				}
				ptr++; // skip ")"
			}
	    	}
	    	if ((ptr != NULL) && strlen(ptr)) {
			hdr->NRec = (nrec_t)strtod(ptr,&ptr);
		}
	    	if ((ptr != NULL) && strlen(ptr)) {
			struct tm t;
			sscanf(ptr," %u:%u:%u %u/%u/%u",&t.tm_hour,&t.tm_min,&t.tm_sec,&t.tm_mday,&t.tm_mon,&t.tm_year);
			t.tm_mon--;
			t.tm_year -= 1900;
			t.tm_isdst = -1;
			hdr->T0 = tm_time2gdf_time(&t);
		}

                if (VERBOSE_LEVEL>8) hdr2ascii(hdr,stdout,2);	// channel header not parsed yet

		int fmt=0,FMT=0;
		size_t MUL=1;
		char **DatFiles = (char**)calloc(hdr->NS, sizeof(char*));
		size_t *ByteOffset = (size_t*)calloc(hdr->NS, sizeof(size_t));
		size_t nDatFiles = 0;
		uint16_t gdftyp,NUM=1,DEN=1;
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		hdr->AS.bpb8 = 0;
		hdr->AS.bpb  = 0;
		for (k=0; k < hdr->NS; k++) {

			double skew=0;
			//double byteoffset=0;
			double ADCgain=200;
			double baseline=0;
			double ADCresolution=12;
			double ADCzero=0;
			double InitialValue=0;
			double BlockSize=0;

			CHANNEL_TYPE* hc = hdr->CHANNEL+k;
			if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d): %s(...)\n",__FILE__,__LINE__,__func__);

		    	do line = strtok(NULL,"\x0d\x0a"); while (line[0]=='#'); // read next line

			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"%s (line %i) %s(...): %i/%i <%s>\n",__FILE__,__LINE__,__func__, (int)k, hdr->NS, line);

			for (ptr=line; !isspace(ptr[0]); ptr++) {}; 	// skip 1st field
			ptr[0]=0;
			if (k==0)
				DatFiles[nDatFiles++]=line;
			else if (strcmp(DatFiles[nDatFiles-1],line))
				DatFiles[nDatFiles++]=line;

			fmt = (typeof(fmt))strtod(ptr+1,&ptr);
			if (k==0) FMT = fmt;
			else if (FMT != fmt) {
				biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "MIT/HEA/PhysioBank: different formats within a single data set is not supported");
			}
			if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d): %s(...)\n",__FILE__,__LINE__,__func__);

			size_t DIV=1;
			if (ptr[0]=='x') {
				DIV = (size_t)strtod(ptr+1, &ptr);
				hdr->CHANNEL[k].SPR *= DIV;
				MUL = lcm(MUL, DIV);
			}
			hdr->CHANNEL[k].SPR = DIV;
			if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d): %s(...)\n",__FILE__,__LINE__,__func__);

			if (ptr[0]==':') skew = strtod(ptr+1,&ptr);
			if (ptr[0]=='+') ByteOffset[k] = (size_t)strtod(ptr+1,&ptr);

			if (ptr != NULL) ADCgain = strtod(ptr+1,&ptr);
			if (ADCgain==0) ADCgain=200;	// DEFGAIN: https://www.physionet.org/physiotools/wag/header-5.htm

			if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d): %s(...)\n",__FILE__,__LINE__,__func__);
			if (ptr[0] == '(') {
				baseline = strtod(ptr+1,&ptr);
				ptr++;
			}
			hc->PhysDimCode = 4274; // mV
			if (ptr[0] == '/') {
				char  *PhysUnits = ++ptr;
				while (!isspace(ptr[0])) ++ptr;
				ptr[0] = 0;
				hc->PhysDimCode = PhysDimCode(PhysUnits);
			}
			if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d): %s(...)\n",__FILE__,__LINE__,__func__);

			if (ptr != NULL) ADCresolution = strtod(ptr+1,&ptr);
			if (ptr != NULL) ADCzero       = strtod(ptr+1,&ptr);
			if (ptr != NULL) InitialValue  = strtod(ptr+1,&ptr);
			else InitialValue = ADCzero;

			double checksum;
			if (ptr != NULL) checksum = strtod(ptr+1,&ptr);
			if (ptr != NULL) BlockSize = strtod(ptr+1,&ptr);
			while (isspace(ptr[0])) ++ptr;

			strncpy(hdr->CHANNEL[k].Label,ptr,MAX_LENGTH_LABEL);
			if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d): %s(...)\n",__FILE__,__LINE__,__func__);

			hc->Cal      = 1/ADCgain;
			hc->Off      = -ADCzero*hc->Cal;
			hc->OnOff    = 1;
			hc->Transducer[0] = '\0';
			hc->LowPass  = -1;
			hc->HighPass = -1;

			if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d): %s(...)\n",__FILE__,__LINE__,__func__);
			// hdr->FLAG.SWAP = (__BYTE_ORDER == __BIG_ENDIAN);
			hdr->FILE.LittleEndian = 1;
			switch (fmt) {
			case 8:
				gdftyp = 1;
				hc->DigMax =  127.0;
				hc->DigMin = -128.0;
				biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "MIT/HEA/PhysioBank format 8(diff) not supported");
				break;
			case 80:
				gdftyp = 2; 	// uint8;
				hc->Off= -128*hc->Cal;
				hc->DigMax = 255.0;
				hc->DigMin = 0.0;
				break;
			case 16:
				if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d): %s(...)\n",__FILE__,__LINE__,__func__);
			 	gdftyp = 3;
				NUM = 2; DEN = 1;
				hc->DigMax = ldexp( 1.0,15)-1.0;
				hc->DigMin = ldexp(-1.0,15);
				break;
			case 24:
			 	gdftyp = 255+24;
				NUM = 3; DEN = 1;
				hc->DigMax = ldexp( 1.0,23)-1.0;
				hc->DigMin = ldexp(-1.0,23);
				break;
			case 32:
			 	gdftyp = 5;
				NUM = 4; DEN = 1;
				hc->DigMax = ldexp( 1.0,31)-1.0;
				hc->DigMin = ldexp(-1.0,31);
				break;
			case 61:
				gdftyp = 3;
				// hdr->FLAG.SWAP = !(__BYTE_ORDER == __BIG_ENDIAN);
				hdr->FILE.LittleEndian = 0;
				NUM = 2; DEN = 1;
				hc->DigMax =  ldexp( 1.0,15)-1.0;
				hc->DigMin =  ldexp(-1.0,15);
				break;
			case 160:
			 	gdftyp = 4; 	// uint16;
				hc->Off= ldexp(-1.0,15)*hc->Cal;
				NUM = 2; DEN = 1;
				hc->DigMax = ldexp(1.0,16)-1.0;
				hc->DigMin = 0.0;
				break;
			case 212:
			 	gdftyp = 255+12;
				NUM = 3; DEN = 2;
				hc->DigMax =  ldexp( 1.0,11)-1.0;
				hc->DigMin =  ldexp(-1.0,11);
				break;
			case 310:
			case 311:
				gdftyp = 255+10;
				NUM = 4; DEN = 3;
				hc->DigMax = ldexp( 1.0,9)-1.0;
				hc->DigMin = ldexp(-1.0,9);
				biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "MIT/HEA/PhysioBank format 310/311 not supported");
				break;
			default:
				gdftyp = 0xffff;
				biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "MIT/HEA/PhysioBank: unknown format");
			}

			hc->GDFTYP   = gdftyp;
		    	hc->LeadIdCode  = 0;
	 		hc->PhysMax = hc->DigMax * hc->Cal + hc->Off;
	 		hc->PhysMin = hc->DigMin * hc->Cal + hc->Off;
	 		hc->bi8 = hdr->AS.bpb8;
			hdr->AS.bpb8 += (hdr->SPR*NUM<<3)/DEN;
	 		hc->bi = hdr->AS.bpb;
			hdr->AS.bpb += hdr->AS.bpb8>>3;

			if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d): %s(...)\n",__FILE__,__LINE__,__func__);
		}
		hdr->SampleRate *= MUL;
		hdr->SPR 	*= MUL;

		if (VERBOSE_LEVEL > 7) hdr2ascii(hdr,stdout,4);
		if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d): %s(...)\n",__FILE__,__LINE__,__func__);

		/* read age, sex etc. */
		line = strtok(NULL,"\x0d\x0a");
		if (line != NULL) {
			char *t1;
			double age=0.0;
			for (k=0; k<strlen(line); k++) line[k]=toupper(line[k]);
			t1 = strstr(line,"AGE:");
			if (t1 != NULL) age = strtod(t1+4,&ptr);
			t1 = strstr(line,"AGE>:");
			if (t1 != NULL) age = strtod(t1+5,&ptr);
			if (age>0.0)
				hdr->Patient.Birthday = hdr->T0 - (uint64_t)ldexp(age*365.25,32);

			t1 = strstr(line,"SEX:");
			if (t1 != NULL) t1 += 4;
			else {
				t1 = strstr(line,"SEX>:");
				if (t1 != NULL) t1 += 5;
			}
			if (t1 != NULL) {
			        while (isspace(t1[0])) t1++;
			        hdr->Patient.Sex = (t1[0]=='M') + 2* (t1[0]=='F');
			}
		}

		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"%s (line %i) %s(...): %i (%i) %s FMT=%i + %i\n",__FILE__,__LINE__,__func__, (int)k+1,(int)nDatFiles,DatFiles[0],fmt,(int)ByteOffset[0]);

		/* MIT: read ATR annotation file */
		char *f0 = hdr->FileName;
		char *f1 = (char*) malloc(strlen(hdr->FileName)+5);
		strcpy(f1,hdr->FileName);		// Flawfinder: ignore
		strcpy(strrchr(f1,'.')+1,"atr");	// Flawfinder: ignore
		hdr->FileName = f1;

		hdr   = ifopen(hdr,"r");
		if (!hdr->FILE.OPEN) {
			// if no *.atr file, try *.qrs
			strcpy(strrchr(f1,'.')+1,"qrs");	// Flawfinder: ignore
			hdr   = ifopen(hdr,"r");
		}
		if (!hdr->FILE.OPEN) {
			// *.ecg
			strcpy(strrchr(f1,'.')+1,"ecg");	// Flawfinder: ignore
			hdr   = ifopen(hdr,"r");
		}

		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"%s (line %i) %s(...): <%s> %i %i\n",__FILE__,__LINE__,__func__, hdr->FileName,hdr->FILE.OPEN,(int)bufsiz);

		if (hdr->FILE.OPEN) {
        		uint16_t *Marker=NULL;
        		count = 0;

		    	while (!ifeof(hdr)) {
                                if (bufsiz<1024) bufsiz = 1024;
                                bufsiz *= 2;
				void *tmp = realloc(Marker, 2 * bufsiz );
                                Marker = (uint16_t*) tmp;
			    	count += ifread (Marker+count, 2, bufsiz-count, hdr);
		    	}
		    	ifclose(hdr);
                        Marker[count]=0;

                        /* define user specified events according to http://www.physionet.org/physiotools/wfdb/lib/ecgcodes.h */
                        hdr->EVENT.CodeDesc = (typeof(hdr->EVENT.CodeDesc)) realloc(hdr->EVENT.CodeDesc,257*sizeof(*hdr->EVENT.CodeDesc));
                        hdr->EVENT.CodeDesc[0] = "";
                        for (k=0; strlen(MIT_EVENT_DESC[k])>0; k++) {
				if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s(...) [MIT 182]:  %i\n",__FILE__,__LINE__,__func__, (int)k);

                                hdr->EVENT.CodeDesc[k+1] = (char*)MIT_EVENT_DESC[k];   // hack to satisfy MinGW (gcc version 4.2.1-sjlj)
                                if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s(...) [MIT 182]: %i %s %s\n",__FILE__,__LINE__,__func__, (int)k,MIT_EVENT_DESC[k],hdr->EVENT.CodeDesc[k]);
                        }
        		hdr->EVENT.LenCodeDesc = k+1;

			if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s(...)[MIT 183] %s %i\n",__FILE__,__LINE__,__func__, f1,(int)count);

			/* decode ATR annotation information */
			size_t N = count;
			hdr->EVENT.TYP = (typeof(hdr->EVENT.TYP)) realloc(hdr->EVENT.TYP,N*sizeof(*hdr->EVENT.TYP));
			hdr->EVENT.POS = (typeof(hdr->EVENT.POS)) realloc(hdr->EVENT.POS,N*sizeof(*hdr->EVENT.POS));
			hdr->EVENT.CHN = (typeof(hdr->EVENT.CHN)) realloc(hdr->EVENT.CHN,N*sizeof(*hdr->EVENT.CHN));

			hdr->EVENT.N   = 0;
			hdr->EVENT.SampleRate = hdr->SampleRate;
			uint16_t chn   = 0;
			size_t pos     = 0;
			char flag_chn  = 0;
			for (k=0; (k<N) && Marker[k]; k++) {
				uint16_t a16 = leu16p(Marker+k);
				uint16_t A   = a16 >> 10;
				uint16_t len = a16 & 0x03ff;

				if (VERBOSE_LEVEL>8)
					fprintf(stdout,"%s (line %i) %s(...)[MIT 183] k=%i/%i N=%i A=%i l=%i\n", __FILE__,__LINE__,__func__, (int)k, (int)N, (int)hdr->EVENT.N, a16>>10, len);

				switch (A) {
				case 59:	// SKIP
					pos += (((uint32_t)leu16p(Marker+k+1))<<16) + leu16p(Marker+k+2);
					k   += 2;
					break;
				case 60:	// NUM
				case 61:	// SUB
					break;
				case 62: 	// CHN
					chn = len;
					flag_chn = flag_chn || chn;
					break;
				case 63: 	// AUX
					k += (len+1)/2;
					break;
				default:
					pos += len;
					// code = 0 is mapped to 49(ACMAX), see MIT_EVENT_DESC and http://www.physionet.org/physiotools/wfdb/lib/ecgcodes.h
					hdr->EVENT.TYP[hdr->EVENT.N] = (A==0 ? 49 : A);
					hdr->EVENT.POS[hdr->EVENT.N] = pos-1;   // convert to 0-based indexing
					hdr->EVENT.CHN[hdr->EVENT.N] = chn;
					++hdr->EVENT.N;
				}
			}
			if (flag_chn)
				hdr->EVENT.DUR = (typeof(hdr->EVENT.DUR)) realloc(hdr->EVENT.DUR,N*sizeof(*hdr->EVENT.DUR));
			else {
				free(hdr->EVENT.CHN);
				hdr->EVENT.CHN = NULL;
			}
			free(Marker);
		}
		free(f1);

		hdr->FileName = f0;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s(...)[MIT 185] \n",__FILE__,__LINE__,__func__);

		/* MIT: open data file */
		if (nDatFiles == 1) {
        		//uint8_t *Marker=NULL;
        		count = 0;

			char *f0 = hdr->FileName;
			char *f1 = (char*) malloc(strlen(hdr->FileName)+strlen(DatFiles[0])+2);
			strcpy(f1,hdr->FileName);
			hdr->FileName = f1;
			char *ptr = strrchr(f1,FILESEP);
			if (ptr != NULL)
				strcpy(ptr+1,DatFiles[0]);
			else
				strcpy(f1,DatFiles[0]);

			hdr->HeadLen = ByteOffset[0];
			hdr = ifopen(hdr,"rb");
			ifseek(hdr, hdr->HeadLen, SEEK_SET);

			count  = 0;
                        bufsiz = 1024;
		    	while (!ifeof(hdr)) {
                                bufsiz *= 2;
				void *tmpptr = realloc(hdr->AS.rawdata, bufsiz + 1 );
				hdr->AS.rawdata = (uint8_t*) tmpptr;
			    	count += ifread (hdr->AS.rawdata+count, 1, bufsiz-count, hdr);
		    	}
		    	ifclose(hdr);

			free(f1);
			hdr->FileName = f0;

			if (!hdr->NRec) {
				hdr->NRec = count/(hdr->AS.bpb);
			}
		}

		if (VERBOSE_LEVEL > 7)
			fprintf(stdout,"%s (line %i) %s(...)[MIT 198] #%i: (%i) %s FMT=%i\n",__FILE__,__LINE__,__func__,(int)k+1,(int)nDatFiles,DatFiles[0],fmt);

		free(DatFiles);
		free(ByteOffset);

		if (VERBOSE_LEVEL > 7)
			fprintf(stdout,"%s (line %i) %s(...)[MIT 199] #%i: (%i) %s FMT=%i\n",__FILE__,__LINE__,__func__,(int)k+1,(int)nDatFiles,DatFiles[0],fmt);

		if (nDatFiles != 1) {
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "MIT/HEA/PhysioBank: multiply data files within a single data set is not supported");
			return(hdr);
		}
		hdr->AS.length  = hdr->NRec;

	} /* END OF MIT FORMAT */

#ifdef CHOLMOD_H
	else if ((hdr->TYPE==MM) && (!hdr->FILE.COMPRESSION)) {

		if (VERBOSE_LEVEL>7) fprintf(stdout,"[MM 001] %i,%i\n",hdr->HeadLen,hdr->FILE.COMPRESSION);

		while (!ifeof(hdr)) {
			count = max(5000, hdr->HeadLen*2);
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, count);
			hdr->HeadLen  += ifread(hdr->AS.Header + hdr->HeadLen, 1, count - hdr->HeadLen - 1, hdr);
		}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"[MM 003] %i,%i\n",hdr->HeadLen,hdr->FILE.COMPRESSION);

		char *line = strtok((char*)hdr->AS.Header, "\x0a\x0d");
		char status = 0;
		unsigned long ns = 0;
		while (line != NULL) {

		if (VERBOSE_LEVEL>7) fprintf(stdout,"[MM 013] <%s>\n",line);

			if ((line[0]=='%') && (line[1]=='%') && isspace(line[2])) {
				if (!strncmp(line+3,"LABELS",6))
					status = 1;
				else if (!strncmp(line+3,"ENDLABEL",8)) {
					status = 0;
					break;
				}
				if (status) {
					int k = 3;
					while (isspace(line[k])) k++;
					unsigned long ch = strtoul(line+k, &line, 10);
					while (isspace(line[0])) line++;
					if (ch >= ns) {
						hdr->rerefCHANNEL = (CHANNEL_TYPE*)realloc(hdr->rerefCHANNEL, ch*sizeof(CHANNEL_TYPE));

						while (ns < ch) {
							hdr->rerefCHANNEL[ns].Label[0] = 0;
							hdr->rerefCHANNEL[ns].Transducer[0] = 0;
							ns++;
						}
					}
					strncpy(hdr->rerefCHANNEL[ch-1].Label, line, MAX_LENGTH_LABEL);
					hdr->rerefCHANNEL[ch-1].OnOff = 1;
					hdr->rerefCHANNEL[ch-1].Label[MAX_LENGTH_LABEL] = 0;
					hdr->rerefCHANNEL[ch-1].LeadIdCode = 0;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"[MM 027] %i <%s>\n",(int)ch,line);
				}
			}
			line = strtok(NULL,"\x0a\x0d");
		}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"[MM 033]\n");

		ifseek(hdr,0,SEEK_SET);

		CSstart();	// init cholmod library
                CHOLMOD_COMMON_VAR.print = 5;
                hdr->Calib = cholmod_read_sparse (hdr->FILE.FID, &CHOLMOD_COMMON_VAR); /* read in a matrix */

                if (VERBOSE_LEVEL>7)
                        cholmod_print_sparse (hdr->Calib, "Calib", &CHOLMOD_COMMON_VAR); /* print the matrix */

		ifclose(hdr);
		if (VERBOSE_LEVEL>7) fprintf(stdout,"[MM 999]\n");
		return(hdr);
	} /* END OF MatrixMarket */
#endif

	else if (hdr->TYPE==NEURON) {
		hdr->HeadLen = count;

	if (VERBOSE_LEVEL>7) fprintf(stdout,"NEURON: start\n");

		size_t count;
		while (!ifeof(hdr)) {
			count = max(1<<20, hdr->HeadLen*2);
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, count);
			hdr->HeadLen  += ifread(hdr->AS.Header + hdr->HeadLen, 1, count - hdr->HeadLen - 1, hdr);
		}
		hdr->AS.Header[hdr->HeadLen] = 0;
		hdr->NS  = 1;
		hdr->SPR = 1;
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		hdr->CHANNEL[0].GDFTYP   = 17;
		hdr->CHANNEL[0].Cal      = 1.0;
		hdr->CHANNEL[0].Off      = 0.0;
		hdr->CHANNEL[0].PhysMin  = -1e9;
		hdr->CHANNEL[0].PhysMax  = +1e9;
		hdr->CHANNEL[0].DigMin   = -1e9;
		hdr->CHANNEL[0].DigMax   = +1e9;
		hdr->AS.bpb = sizeof(double);
		hdr->CHANNEL[0].bi       = 0;
		hdr->CHANNEL[0].bi8      = 0;
		hdr->CHANNEL[0].LeadIdCode = 0;
		hdr->CHANNEL[0].SPR      = hdr->SPR;
		hdr->CHANNEL[0].LowPass  = NAN;
		hdr->CHANNEL[0].HighPass = NAN;
		hdr->CHANNEL[0].Notch    = NAN;
		hdr->CHANNEL[0].Transducer[0] = 0;

	if (VERBOSE_LEVEL>7) fprintf(stdout,"NEURON 202: \n");

		char *t = strtok( (char*)hdr->AS.Header, "\x0A\x0D");
		char status = 0;
		size_t spr  = 0;

		while (t != NULL) {

	if (VERBOSE_LEVEL>8) fprintf(stdout,"NEURON 301: <%s>\n", t);

			if (status==0) {
				if (!strncmp(t,"Header:", 7))
					status = 1;
			}
			else if (status==1) {
	if (VERBOSE_LEVEL>7) fprintf(stdout,"NEURON 311: <%s>\n",t);
				char *val = t+strlen(t);
				while (isspace(*(--val))) {}; val[1]=0;	// remove trailing blanks
				val = strchr(t,':');			// find right value
				val[0] = 0;
				while (isspace(*(++val))) {};

				if (!strncmp(t,"Data", 7)) {
					status=2;
					spr = 0;
				}
				else if (!strcmp(t,"SampleInt"))
					hdr->SampleRate = 1.0 / atof(val);

				else if (!strcmp(t,"Points")) {
					hdr->NRec = atoi(val);
					hdr->AS.rawdata = (uint8_t*)realloc(hdr->AS.rawdata, sizeof(double) * hdr->NRec);
				}
				else if (!strcmp(t,"XUnit")) {
					uint16_t xunits = PhysDimCode(val);
					double   scale = PhysDimScale(xunits);
					if ((xunits & 0xffe0)==2176) hdr->SampleRate /= scale;
					else fprintf(stdout, "Error NEURON: invalid XUnits <%s>\n", val);
				}

				else if (!strcmp(t,"YUnit")) {
	if (VERBOSE_LEVEL>7) fprintf(stdout,"NEURON 321: Yunits:<%s>\n",val);
					hdr->CHANNEL[0].PhysDimCode = PhysDimCode(val);
				}
				else if (!strcmp(t,"Method")) {
					strncpy(hdr->CHANNEL[0].Label, val, MAX_LENGTH_LABEL);
				}
			}
			else if (status==2) {
				if (strpbrk(t,"0123456789")) {
					// ignore non-numeric (e.g. emtpy) lines
					*(double*)(hdr->AS.rawdata + spr*sizeof(double)) = atof(t);
					spr++;
				}
				if (hdr->NRec >= 0)
				if (spr >= (size_t)hdr->NRec) {
					void *ptr = realloc(hdr->AS.rawdata, 2 * min(spr, (size_t)hdr->NRec) * sizeof(double));
					if (ptr==NULL) break;
					hdr->AS.rawdata = (uint8_t*)ptr;
				}
			}
			t = strtok(NULL, "\x0A\x0D");
		}
		free(hdr->AS.Header);
		hdr->AS.Header = NULL;
		hdr->AS.first  = 0;
		hdr->AS.length = spr;
	}

        else if (hdr->TYPE == NeuroLoggerHEX) {
        	hdr->NS = 8;
        	// uint16_t gdftyp = 2;

		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "NeuroLogger HEX format not supported, yet");
		return(hdr);

	}

#if defined(WITH_NEV)
	else if (hdr->TYPE==NEV) {

		fprintf(stdout,"Support for NEV format is under construction - most likely its not useful yet.\n");

		if (VERBOSE_LEVEL>7) fprintf(stdout,"SOPEN (NEV)\n");

		hdr->VERSION = beu16p(hdr->AS.Header+8)>>8;
		switch (beu16p(hdr->AS.Header+8)) {
		case 0x0100:	// readnev1
		case 0x0101:	// readnev1_1
		case 0x0200:	// readnev2
			//biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, NULL);
			break;
		default:
			biosigERROR(hdr, B4C_FORMAT_UNKNOWN, NULL);
		}
		const int H1Len = 28+16+32+256+4;

		/* read Basic Header */
		// uint16_t fileFormat = beu16p(hdr->AS.Header+10);
		uint32_t HeadLen = leu32p(hdr->AS.Header+12);
		if (HeadLen < H1Len) {
			biosigERROR(hdr, B4C_INCOMPLETE_FILE, NULL);
			return(hdr);
		}
		hdr->AS.bpb = leu32p(hdr->AS.Header+16);
		uint32_t TimeStepFrequency = leu32p(hdr->AS.Header+20);
		// samples = TimeStepFrequency / 10; // Freq in 0.1 seconds
		hdr->SampleRate = leu32p(hdr->AS.Header+24);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"SOPEN (NEV) [210] %d %d \n", (int)count, (int)HeadLen);

		if (count<HeadLen) {
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,HeadLen);
			count += ifread(hdr->AS.Header+count, 1, HeadLen-count, hdr);
		}
		uint32_t extHdrN = leu32p(hdr->AS.Header+H1Len-4);

		if (count < H1Len + 32*extHdrN) {
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,HeadLen);
			count += ifread(hdr->AS.Header+count, 1, HeadLen-count, hdr);
		}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"SOPEN (NEV) [220] %d %d %d \n", extHdrN, (int)count, (int)HeadLen);

		struct tm t;
		t.tm_year = leu32p(hdr->AS.Header+28);
		t.tm_mon  = leu32p(hdr->AS.Header+30);
		//t.tm_wday = beu32p(hdr->AS.Header+32);
		t.tm_mday = leu32p(hdr->AS.Header+34);
		t.tm_hour = leu32p(hdr->AS.Header+36);
		t.tm_min  = leu32p(hdr->AS.Header+38);
		t.tm_sec  = leu32p(hdr->AS.Header+40);
		//milliseconds = beu32p(hdr->AS.Header+42);
		hdr->T0   = tm_time2gdf_time(&t);

		double time_interval = 1e3 * (hdr->AS.bpb-8) / TimeStepFrequency;

		 if (VERBOSE_LEVEL>7) hdr2ascii(hdr,stdout,2);

		//******** read Extended Header *********
		const char *nameOfElectrode, *extraComment, *continuedComment, *mapfile;

		const char *H2 = (char*)hdr->AS.Header + H1Len;
		hdr->NS = 0;
    		for (k = 0; k < extHdrN; k++) {
			const char *identifier =  H2 + k*32;

			if (VERBOSE_LEVEL>8) {
				char tmp[9];tmp[8]=0;
				char tmp24[25]; tmp24[24]=0;
				memcpy(tmp,identifier,8);
				memcpy(tmp24,identifier+8,24);

				fprintf(stdout,"SOPEN (NEV) [225] %d %d <%s> <%s>\n",(int)k,hdr->NS,tmp,tmp24);
			}

			if (!memcmp (identifier, "NEUEVWAV",8)) hdr->NS++;
		}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"SOPEN (NEV) [230]\n");

		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));

		uint16_t NS = 0;
    		for (k = 0; k < extHdrN; k++) {
			const char *identifier =  H2 + k*32;
			if (!memcmp (identifier, "ARRAYNME",8)) {
				nameOfElectrode = identifier + 8;
			}
			else if (!memcmp (identifier, "ECOMMENT",8)) {
				extraComment = identifier + 8;
			}
			else if (!memcmp (identifier, "CCOMMENT",8)) {
				continuedComment = identifier + 8;
			}
			else if (!memcmp (identifier, "MAPFILE",8)) {
				mapfile = identifier + 8;
			}
			else if (!memcmp (identifier, "NEUEVWAV",8)) {
				//neuralEventWaveform = identifier + 8;
				CHANNEL_TYPE *hc = hdr->CHANNEL+(NS++);
				sprintf(hc->Label,"#%d",leu16p(identifier + 8));	// electrodeId
				hc->Transducer[0] = 0;
				// (uint8_t)(identifier + 8 + 2);	// module
				// (uint8_t)(identifier + 8 + 3);	// channel
				hc->OnOff = 1;
				hc->Cal = leu16p(identifier+ 8 + 4);	// scaling factor
				// beu16p(identifier + 8 + 6);	// energyTreshold
				hc->Off = 0;
				hc->DigMax = lei16p(identifier + 8 + 8);	// high threshold
				hc->DigMin = lei16p(identifier + 8 + 10);	// low threshold
				// (uint8_t)(identifier + 8 + 11);	// sortedUnitsInChannel
				hc->GDFTYP = 2 * identifier[8 + 12];	// bytesPerWaveformSample
				hc->PhysMax = hc->DigMax*hc->Cal;
				hc->PhysMin = hc->DigMin*hc->Cal;
				hc->LeadIdCode = 0;
				hc->Transducer[0] = 0;
				hc->PhysDimCode = 0;
				hc->TOffset = 0;
				hc->LowPass = NAN;
				hc->HighPass = NAN;
				hc->Notch = NAN;
				hc->XYZ[0] = 0;
				hc->XYZ[1] = 0;
				hc->XYZ[2] = 0;
				hc->Impedance = NAN;

				hc->SPR = 0;
				hc->bi = hdr->AS.bpb;
				hc->bi8 = hdr->AS.bpb*8;
				hc->bufptr = NULL;

			}
			else if (!memcmp (identifier, "NSASEXEV",8)) {
				char *nsas = identifier + 8;
			}
			else  {
/*
				// IGNORE
				biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "NEV: unknown extended header");
*/
			}
		}
		return(hdr);
	}
#endif

	else if (hdr->TYPE==NEX1) {

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i)\n",__func__,__LINE__);

		if (count < 284) {
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,284);
			count  += ifread(hdr->AS.Header + count, 1, 284 - count, hdr);
		}

		uint8_t v = hdr->AS.Header[3]-'0';
		const int H1LEN = (v==1) ? (4 + 4 + 256 +   8 + 4*4 + 256)
					 : (4 + 4 + 256 +   8 +   8 +  4 +  8 + 64);
		const int H2LEN = (v==1) ? (4 + 4 +  64 + 6*4 + 4*8 + 12 + 16 + 52)
					 : (4 + 4 +  64 + 2*8 + 2*4 +  8 + 32 + 4*8 + 4*4 + 60);

		uint32_t k      = leu32p(hdr->AS.Header + 280);

		if (count < H1LEN + k * H2LEN) {
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, H1LEN + k * H2LEN);
			count  += ifread(hdr->AS.Header + count, 1, H1LEN + k * H2LEN - count, hdr);
		}

		hdr->HeadLen          = count;
		hdr->VERSION          = leu32p(hdr->AS.Header + 4) / 100.0;
		hdr->SampleRate       = lef64p(hdr->AS.Header + 264);
		hdr->SPR              = 1;
		hdr->EVENT.SampleRate = lef64p(hdr->AS.Header + 264);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i)\n",__func__,__LINE__);

		if (k > 0xffff) {
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "NEX format has more than 65535 channels");
			return (hdr);
		}

		while (!ifeof(hdr)) {
			void *tmpptr = realloc(hdr->AS.Header, count*2);
			if (tmpptr)
				hdr->AS.Header = (uint8_t*)tmpptr;
			else {
				biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Not enough memory to read NEX file");
				return(hdr);
			}
			count  += ifread(hdr->AS.Header + count, 1, count, hdr);
		}


		hdr->NS = k;
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS*sizeof(CHANNEL_TYPE));

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i)\n",__func__,__LINE__);

		for (k=0; k < hdr->NS; k++) {

			if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): VarHdr # %i\n",__func__,__LINE__, k);

			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			uint32_t type = leu32p(hdr->AS.Header + H1LEN + k*H2LEN);

			hc->OnOff = (type==5);

			strncpy(hc->Label, hdr->AS.Header + H1LEN + k*H2LEN + 8, min(64,MAX_LENGTH_LABEL));
			hc->Label[min(64, MAX_LENGTH_LABEL)] = 0;
			hc->Transducer[0] = 0;

			size_t n;
			if (v==5) {
				hc->GDFTYP = (leu32p(hdr->AS.Header + H1LEN + k*H2LEN + 92)==1) ? 16 : 3;
				hc->PhysDimCode = PhysDimCode(hdr->AS.Header + H1LEN + k*H2LEN + 5*8 + 64);
				n       = leu64p(hdr->AS.Header + 80 + H1LEN + k*H2LEN);
				hc->Cal = lef64p(hdr->AS.Header + 64+8*5+32 + H1LEN + k*H2LEN);
				hc->Off = lef64p(hdr->AS.Header + 64+8*5+40 + H1LEN + k*H2LEN);
				hc->SPR = leu64p(hdr->AS.Header + 64+8*5+48 + H1LEN + k*H2LEN);
				hc->bufptr = hdr->AS.Header + leu64p(hdr->AS.Header + 64+8 + H1LEN + k*H2LEN);
			}
			else {
				hc->GDFTYP = 3;
				hc->PhysDimCode = PhysDimCode("mV");
				n       = leu32p(hdr->AS.Header + 76 + H1LEN + k*H2LEN);
				hc->Cal = lef64p(hdr->AS.Header + 64+8*4+3*8    + H1LEN + k*H2LEN);
				hc->Off = lef64p(hdr->AS.Header + 64+8*4+3*8+20 + H1LEN + k*H2LEN);
				hc->SPR = leu32p(hdr->AS.Header + 64+8*4+4*8    + H1LEN + k*H2LEN);
				hc->bufptr = hdr->AS.Header+leu32p(hdr->AS.Header + 64+8 + H1LEN + k*H2LEN);
			}

			if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): VarHdr # %i %i %i %i \n",__func__,__LINE__, k,v,type,(int)n);

			switch (type) {
			case 2:
			//case 6:
			case 0:
			case 1:
				hdr->EVENT.N += n;
			}
			//if (hc->OnOff) hdr->SPR = lcm(hdr->SPR, hc->SPR);
		}

		if (hdr->EVENT.N > 0) {
			size_t N=hdr->EVENT.N;
			hdr->EVENT.N=0;
			reallocEventTable(hdr,N);

			N = 0;
			for (k=0; k < hdr->NS; k++) {
				if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): VarHdr # %i\n",__func__,__LINE__, k);
				CHANNEL_TYPE *hc = hdr->CHANNEL+k;
				uint32_t type = leu32p(hdr->AS.Header + H1LEN + k*H2LEN);

				size_t n,l;
				uint16_t gdftyp = 5;
				if (v==5) {
					n = leu64p(hdr->AS.Header + 80 + H1LEN + k*H2LEN);
					if (leu32p(hdr->AS.Header + 88 + H1LEN + k*H2LEN))
						gdftyp=7;
				}
				else
					n = leu32p(hdr->AS.Header + 76 + H1LEN + k*H2LEN);


				switch (type) {
				case 2:
					if (gdftyp==5) {
						for (l=0; l<n; l++)
							hdr->EVENT.DUR[N+l] = leu32p(hc->bufptr+4*(l+n));
					}
					else {
						for (l=0; l<n; l++)
							hdr->EVENT.DUR[N+l] = leu64p(hc->bufptr+8*(l+n));
					}
				case 0:
				case 1:
				//case 6:
					if (gdftyp==5) {
						for (l=0; l<n; l++)
							hdr->EVENT.POS[N+l] = leu32p(hc->bufptr+4*l);
					}
					else {
						for (l=0; l<n; l++)
							hdr->EVENT.POS[N+l] = leu64p(hc->bufptr+8*l);
					}

					for (l=0; l<n; l++) {
						hdr->EVENT.TYP[N+l] = type;
						hdr->EVENT.CHN[N+l] = k;
						//hdr->EVENT.TimeStamp[N+l] = 0;
					}
				}
				N+=n;
			}
			hdr->EVENT.N=N;
		}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i)\n",__func__,__LINE__);

		hdr2ascii(hdr,stdout,4);

		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Support for NEX format is not ready yet");
		return(hdr);
	}

	else if (hdr->TYPE==NIFTI) {
		if (count<352)
		    	count += ifread(hdr->AS.Header+count, 1, 352-count, hdr);

	    	// nifti_1_header *NIFTI_HDR = (nifti_1_header*)hdr-AS.Header;
	    	char SWAP = *(int16_t*)(Header1+40) > 7;
#if   (__BYTE_ORDER == __BIG_ENDIAN)
		hdr->FILE.LittleEndian = SWAP;
#elif (__BYTE_ORDER == __LITTLE_ENDIAN)
		hdr->FILE.LittleEndian = !SWAP;
#endif
	    	if (!SWAP) {
		    	hdr->HeadLen = (size_t)*(float*)(Header1+80);
		}
		else {
			union {uint32_t u32; float f32;} u;
			u.u32 = bswap_32(*(uint32_t*)(Header1+108));
		    	hdr->HeadLen = (size_t)u.f32;
		}

		if (Header1[345]=='i') {
			ifclose(hdr);
			char *f0 = hdr->FileName;
			char *f1 = (char*)malloc(strlen(hdr->FileName)+4);
			strcpy(f1,hdr->FileName);
			strcpy(strrchr(f1,'.') + 1, "img");	// Flawfinder: ignore
			hdr->FileName = f1;
			hdr = ifopen(hdr,"r");
			hdr->FileName = f0;
		}
		else
			ifseek(hdr,hdr->HeadLen,SEEK_SET);

#ifdef _NIFTI_HEADER_
		nifti_1_header *NIFTI_HDR = (nifti_1_header*)hdr->AS.Header;
#endif

		ifclose(hdr);
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format NIFTI not supported");
		return(hdr);
	}

	else if (hdr->TYPE==NUMPY) {
		/*
			There is no way to extract sampling rate and scaling factors from numpy files.
			For this reason, numpy is not going to be a supported data format.
		*/
		fprintf(stderr,"Warning SOPEN (NUMPY): sampling rate, scaling, physical units etc. are not supported, and are most likely incorrect.");
		hdr->VERSION = hdr->AS.Header[6]+hdr->AS.Header[7]/100;
		hdr->HeadLen = leu16p(hdr->AS.Header+8);
		if (count < hdr->HeadLen) {
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, hdr->HeadLen+1);
			count += ifread(hdr->AS.Header + count, 1, hdr->HeadLen - count, hdr);
		}
		hdr->AS.Header[count]=0;

		hdr->NS=1;
		hdr->SPR=0;
		hdr->NRec=1;
		uint16_t gdftyp = 0;

		const char *h=(char*)hdr->AS.Header+10;
		int flag_order = (strstr(h,"'fortran_order': False")==NULL) + (strstr(h,"'fortran_order': True")==NULL)*2;
		switch (flag_order) {
		case 1:
			break;
		case 2:
			break;
		default:
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "format NUMPY: fortran_order not specified or invalid");
		}
		char *descr = strstr(h,"'descr':");
		if (descr==NULL)
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format NUMPY: descr not defined");
		else {
			descr += 8;
			descr += strspn(descr," \t'");
			descr[strcspn(descr," \t'")]=0;
			if (descr[0]=='<')
				hdr->FILE.LittleEndian = 1;
			else if (descr[0]=='>')
				hdr->FILE.LittleEndian = 0;
			else
				biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format NUMPY: field 'descr': endianity undefined");

			if      (!strcmp(descr+1,"f8")) gdftyp = 17;
			else if (!strcmp(descr+1,"f4")) gdftyp = 16;
			else if (!strcmp(descr+1,"u8")) gdftyp = 8;
			else if (!strcmp(descr+1,"i8")) gdftyp = 7;
			else if (!strcmp(descr+1,"u4")) gdftyp = 6;
			else if (!strcmp(descr+1,"i4")) gdftyp = 5;
			else if (!strcmp(descr+1,"u2")) gdftyp = 4;
			else if (!strcmp(descr+1,"i2")) gdftyp = 3;
			else if (!strcmp(descr+1,"u1")) gdftyp = 2;
			else if (!strcmp(descr+1,"i1")) gdftyp = 1;
			else
				biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format NUMPY: field 'descr': not supported");
		}

		char *shapestr = strstr(h,"'shape':");
		if (shapestr==NULL)
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format NUMPY: shape not defined");
		else {
			int n = 0;
			char *tmpstr = strchr(shapestr,'(') + 1;
			while (tmpstr && *tmpstr) {
				*strchr(tmpstr,')') = 0;	// terminating \0
				char *next = strchr(tmpstr,',');
				if (next) {
					*next=0;
					long dim = atol(tmpstr);
					switch (n) {
					case 0: hdr->SPR =dim; break;
					case 1: hdr->NS  =dim; break;
					//case 2: hdr->NRec=dim; break;
					default:
						biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "format NUMPY: shape not supported");
					}
					n++;
					tmpstr = next+1;
				}
			}
		}
		hdr->AS.bpb = hdr->NS*GDFTYP_BITS[gdftyp]>>3;
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL,hdr->NS*sizeof(CHANNEL_TYPE));
		typeof (hdr->NS) k;
		for (k=0; k<hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			hc->Transducer[0] = 0;
			hc->Label[0] = 0;
			hc->GDFTYP = gdftyp;
			hc->SPR    = hdr->SPR;
			hc->Cal    = 1.0;
			hc->Off    = 0.0;
		}

		if (VERBOSE_LEVEL > 6)
			fprintf(stdout,"NUMPY:\n%s",(char*)hdr->AS.Header+10);

		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format NUMPY not supported");
		return(hdr);
	}

    	else if (hdr->TYPE==Persyst) {

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d) (Persyst) [225]\n", __func__,__LINE__);

		size_t c=1;
		while (~ifeof(hdr) && c) {

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d) (Persyst) [25] %d\n",__func__,__LINE__,(int)count);

			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,count*2+1);
		    	c = ifread(hdr->AS.Header + count, 1, count, hdr);
			count += c;
		}
		hdr->AS.Header[count] = 0;
		ifclose(hdr);
		hdr->SPR = 1;

		int32_t gdftyp = 3;
		double Cal = 1.0;
		int status = 0;
		char *remHDR=(char*)hdr->AS.Header;
		const char *FirstName=NULL, *MiddleName=NULL, *SurName=NULL;
		char *datfile = NULL;
		struct tm RecTime;
		size_t NEvent = 0;
		hdr->FLAG.OVERFLOWDETECTION = 0; // overflow detection is not supported for this format
		double DigMax =  ldexp(1,-15);
		double DigMin = -ldexp(1,-15)-1;
		char *line;
		char flag_interleaved = 1;
		while (1) {
			if (*remHDR == '\0') break;
			// line = strsep(&remHDR,"\n\r");
			line = remHDR;
			remHDR = strpbrk(remHDR,"\n\r\0");
			*remHDR++ = 0;

			remHDR += strspn(remHDR,"\n\r");

			if (!strncmp(line,"[FileInfo]",10))
				status = 1;
			else if (!strncmp(line,"[ChannelMap]",12)) {
				status = 2;
				hdr->AS.bpb = hdr->NS*GDFTYP_BITS[gdftyp]>>3;
				hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL,hdr->NS*sizeof(CHANNEL_TYPE));
				uint16_t ch;
				for (ch=0; ch < hdr->NS; ch++) {
					CHANNEL_TYPE *hc = hdr->CHANNEL+ch;

					hc->PhysMax = DigMax*Cal;
					hc->PhysMin = DigMin*Cal;
					hc->DigMax  = DigMax;
					hc->DigMin  = DigMin;
					hc->Cal     = Cal;
					hc->Off     = 0.0;
					hc->OnOff   = 1;

					hc->Label[0]      = 0;
					hc->LeadIdCode    = 0;
					hc->Transducer[0] = 0;
					hc->PhysDimCode   = 0; 	//TODO
					hc->GDFTYP        = gdftyp;

					hc->TOffset   = NAN;
					hc->LowPass   = NAN;
					hc->HighPass  = NAN;
					hc->Notch     = NAN;
					hc->XYZ[0]    = 0.0;
					hc->XYZ[1]    = 0.0;
					hc->XYZ[2]    = 0.0;
					hc->Impedance = NAN;

					hc->SPR = hdr->SPR;
					hc->bi8 = ch*GDFTYP_BITS[gdftyp];
					hc->bi  = hc->bi8>>3;
					hc->bufptr = NULL;
				}
			}
			else if (!strncmp(line,"[Sheets]",8))
				status = 3;
			else if (!strncmp(line,"[Comments]",10))
				status = 4;
			else if (!strncmp(line,"[Patient]",9))
				status = 5;
			else if (!strncmp(line,"[SampleTimes]",13))
				status = 6;
			else {

				switch (status) {
				case 1: {
					char *tag = line;
					char *val = strchr(line,'=');
					*val= 0;		// replace "=" with terminating \0
					val++;			// next character is the start of the value parameters

					if (!strcmp(tag,"File")) {
						datfile = strrchr(val,'/');
						if (!datfile) datfile = strrchr(val,'\\')+1;
						if (!datfile) datfile = val;
					}
					else if (!strcmp(line,"FileType"))
						flag_interleaved = !strcmp(val,"Interleaved");
					else if (!strcmp(line,"SamplingRate"))
						hdr->SampleRate = atof(val);
					else if (!strcmp(line,"Calibration"))
						Cal = atof(val);
					else if (!strcmp(line,"WaveformCount"))
						hdr->NS = atol(val);
					else if (!strcmp(line,"DataType")) {
						switch (atol(val)) {
						case 0: gdftyp = 3; 	// int 16
							hdr->FILE.LittleEndian = 1;
							hdr->AS.bpb *= 2;
							DigMin = -ldexp(1.0,-15)-1;
							DigMax =  ldexp(1.0,-15);
							break;
						case 4: gdftyp = 3; 	// int16
							hdr->FILE.LittleEndian = 0;
							hdr->AS.bpb *= 2;
							DigMin = -ldexp(1.0,-15)-1;
							DigMax =  ldexp(1.0,-15);
							break;
						case 6: gdftyp = 1; 	// int8
							DigMin = -ldexp(1.0,-7)-1;
							DigMax =  ldexp(1.0,-7);
							break;
						default:
							biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format Persyst: unsupported data type");
						}
					}
					break;
				}
				case 2: {
					char *tag = line;
					char *val = strchr(line,'=');
					*val = 0;		// replace "=" with terminating \0
					val++;			// next character is the start of the value parameters

					int channo = atol(val)-1;
					if (0 <= channo && channo < hdr->NS) {
						strncpy(hdr->CHANNEL[channo].Label, tag, MAX_LENGTH_LABEL);
					}
					break;
				}
				case 3: {
					break;
				}
				case 4: {
					if (NEvent < hdr->EVENT.N+2) {
						NEvent  += max(128,NEvent);
						if (reallocEventTable(hdr, NEvent) == SIZE_MAX) {
							biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
							return (hdr);
						};
					}
					char *tmp2,*tmp1 = line;
					tmp2 = strchr(tmp1,',');   *tmp2 = 0;
					hdr->EVENT.POS[hdr->EVENT.N] = atof(tmp1)*hdr->SampleRate;
					tmp1 = strchr(tmp2+1,','); *tmp1 = 0;
					hdr->EVENT.DUR[hdr->EVENT.N]  = atof(tmp1)*hdr->SampleRate;
					tmp2 = strchr(tmp1+1,','); *tmp2 = 0; 	// ignore next field
					tmp1 = strchr(tmp2+1,','); *tmp1 = 0;  	// ignore next field
					char *Desc = tmp1+1;
					FreeTextEvent(hdr,hdr->EVENT.N,Desc);
					hdr->EVENT.N++;
					break;
				}
				case 5: {
					char *val = strchr(line,'=');
					*val= 0;		// replace "=" with terminating \0
					val++;			// next character is the start of the value parameters

					if (!strcmp(line,"First"))
						FirstName=val;
					else if (!strcmp(line,"MI"))
						MiddleName=val;
					else if (!strcmp(line,"Last"))
						SurName=val;
					else if (!strcmp(line,"Hand"))
						hdr->Patient.Handedness = (toupper(val[0])=='R') + 2*(toupper(val[0])=='L') ;
					else if (!strcmp(line,"Sex"))
						hdr->Patient.Sex = (toupper(val[0])=='M') + 2*(toupper(val[0])=='F') ;
					else if (!strcmp(line,"BirthDate")) {
						struct tm t;
						t.tm_year = atol(val+6);
						if (t.tm_year < 80) t.tm_year+=100;
						val[5]=0;
						t.tm_mday = atol(val+3);
						val[2]=0;
						t.tm_mon = atol(val);

						t.tm_hour = 12;
						t.tm_min = 0;
						t.tm_sec = 0;
						hdr->Patient.Birthday = tm_time2gdf_time(&t);
					}
					else if (!strcmp(line,"TestDate")) {
						RecTime.tm_year = atol(val+6);
						if (RecTime.tm_year < 80) RecTime.tm_year+=100;
						val[5]=0;
						RecTime.tm_mday  =  atol(val+3);
						val[2]=0;
						RecTime.tm_mon = atol(val);
					}
					else if (!strcmp(line,"TestTime")) {
						RecTime.tm_sec = atol(val+6);
						val[5]=0;
						RecTime.tm_min  =  atol(val+3);
						val[2]=0;
						RecTime.tm_hour = atol(val);
					}
					else if (!strcmp(line,"ID")) {
						strncpy(hdr->Patient.Id,val,MAX_LENGTH_PID);
						hdr->Patient.Id[MAX_LENGTH_PID] = 0;
					}
					/* Omitted, because it is not important, identification through ID, and T0; quality is determined by Technician
					else if (!strcmp(line,"Physician"))
						Physician=val;
					*/
					else if (!strcmp(line,"Technician"))
						hdr->ID.Technician = strdup(val);
					else if (!strcmp(line,"Medications"))
						hdr->Patient.Medication = (val!=NULL) && strlen(val)>0;

					break;
					}
				case 6: {
					break;
					}
				case 7: {
					break;
					}
				}
			}
		}


		if (VERBOSE_LEVEL>7) fprintf(stdout,"SOPEN (Persyst) [260] %d<%s>\n",status,line);

		hdr->T0 = tm_time2gdf_time(&RecTime);


		if (VERBOSE_LEVEL>7) fprintf(stdout,"SOPEN (Persyst) [270] %d<%s>\n",status,line);

		if (!hdr->FLAG.ANONYMOUS) {
			size_t len = 0, len0=0;
			if (SurName!=NULL) len += strlen(SurName);
			if (len < MAX_LENGTH_NAME) {
				strcpy(hdr->Patient.Name, SurName);
				hdr->Patient.Name[len]=0x1f;
				len0 = ++len;
			}
			if (FirstName!=NULL) len += strlen(FirstName);
			if (len < MAX_LENGTH_NAME) {
				strcpy(hdr->Patient.Name+len0, FirstName);
				hdr->Patient.Name[len]=' ';
				len0 = ++len;
			}
			if (MiddleName!=NULL) len += strlen(MiddleName);
			if (len < MAX_LENGTH_NAME) {
				strcpy(hdr->Patient.Name+len0, MiddleName);
				hdr->Patient.Name[len0]=' ';
			}
			hdr->Patient.Name[len]=0;
		}


		if (VERBOSE_LEVEL>7) fprintf(stdout,"SOPEN (Persyst) [280] %d<%s>\n",status,datfile);

		size_t len = strlen(hdr->FileName);
		char *FileName = hdr->FileName;
		hdr->FileName = (char*) malloc(len+strlen(datfile)+2);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"SOPEN (Persyst) [283] %d<%s>  %d/%d\n",(int)len,datfile,(int)hdr->SPR,(int)hdr->NRec);

		if (strspn(FileName,"/\\")) {
			strcpy(hdr->FileName, FileName);
			char *tmpstr = strrchr(hdr->FileName,'/')+1;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"SOPEN (Persyst) [285] %d<%s>\n",(int)len,tmpstr);

			if (tmpstr==NULL)
				tmpstr = strrchr(hdr->FileName,'\\')+1;

			if (tmpstr!=NULL)
				strcpy(tmpstr,datfile);
			else {
		    		biosigERROR(hdr, B4C_CANNOT_OPEN_FILE, "Format Persyst: cannot open dat file.");
			}
		}
		else {
			strcpy(hdr->FileName, datfile);
		}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"SOPEN (Persyst) [290] %d<%s>\n",status, hdr->FileName);

		struct stat FileBuf;
		if (stat(hdr->FileName, &FileBuf)==0) {
			hdr->FILE.size = FileBuf.st_size;
			hdr->NRec = FileBuf.st_size*8/(hdr->NS*GDFTYP_BITS[gdftyp]);

			if (!flag_interleaved) {
				hdr->SPR = hdr->NRec;
				hdr->NRec = 1;

				uint16_t ch;
				for (ch=0; ch < hdr->NS; ch++) {
					CHANNEL_TYPE *hc = hdr->CHANNEL+ch;

					hc->SPR = hdr->SPR;
					size_t bi8 = ch * (size_t)hdr->SPR * GDFTYP_BITS[gdftyp];
					hc->bi8 = bi8;
					hc->bi  = bi8>>3;
				}
				hdr->AS.bpb = FileBuf.st_size;
			}
		}
		else {
	    		biosigERROR(hdr, B4C_CANNOT_OPEN_FILE, "Format Persyst: cannot open dat file.");
		}

		ifopen(hdr,"r");
		hdr->HeadLen = 0;	// datfile has no header

		free(hdr->FileName);
		hdr->FileName = FileName;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"SOPEN (Persyst) [298] %d %d %d\n",(int)FileBuf.st_size,(int)hdr->AS.bpb,(int)(FileBuf.st_size/hdr->AS.bpb));

	}

	else if (hdr->TYPE==PLEXON) {
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format PLEXON not supported");
		return(hdr);

	}
	else if (hdr->TYPE==RDF) {

		// UCSD ERPSS aquisition system

		#define RH_F_CONV   0x0001   /* converted from other format */
		#define RH_F_RCOMP  0x0002   /* raw file is compressed */
		#define RH_F_DCMAP  0x4000   /* channel mapping used during dig. */

		#define RH_CHANS       256   /* maximum # channels */
		#define RH_DESC         64   /* max # chars in description */
		#define RH_CHDESC        8   /* max # chars in channel descriptor */
		#define RH_SFILL         4   /* short filler */
		#define RH_LFILL         6   /* long filler */
		#define RH_ALOG        828   /* max # chars in ASCII log */

		struct rawhdr {
			uint16_t rh_byteswab;          /* ERPSS byte swab indicator */
			uint16_t rh_magic;             /* file magic number */
			uint16_t rh_flags;             /* flags */
			uint16_t rh_nchans;            /* # channels */
			uint16_t rh_calsize;           /* (if norm, |cal|) */
			uint16_t rh_res;               /* (if norm, in pts/unit) */
			uint16_t rh_pmod;              /* # times processed */
			uint16_t rh_dvers;             /* dig. program version */
			uint16_t rh_l2recsize;         /* log 2 record size */
			uint16_t rh_recsize;           /* record size in pts */
			uint16_t rh_errdetect;         /* error detection in effect */
			uint16_t rh_chksum;            /* error detection chksum */
			uint16_t rh_tcomp;             /* log 2  time comp/exp */
			uint16_t rh_narbins;           /* (# art. rej. count slots) */
			uint16_t rh_sfill[RH_SFILL];   /* short filler (to 16 slots) */
			uint16_t rh_nrmcs[RH_CHANS];   /* (cal sizes used for norm.) */
			uint32_t rh_time;              /* creation time, secs since 1970 */
			uint32_t rh_speriod;           /* digitization sampling period */
			uint32_t rh_lfill[RH_LFILL];   /* long filler (to 8 slots) */
			char     rh_chdesc[RH_CHANS][RH_CHDESC]; /* chan descriptions */
			char     rh_chmap[RH_CHANS];   /* input chan mapping array */
			char     rh_subdesc[RH_DESC];  /* subject description */
			char     rh_expdesc[RH_DESC];  /* experimenter description */
			char     rh_ename[RH_DESC];    /* experiment name */
			char     rh_hname[RH_DESC];    /* host machine name */
			char     rh_filedesc[RH_DESC]; /* file description */
			char     rh_arbdescs[RH_DESC]; /* (art. rej. descriptions) */
			char     rh_alog[RH_ALOG];     /* ASCII log */
		};

		if (count < sizeof(struct rawhdr)) {
			hdr->HeadLen = sizeof(struct rawhdr);
		    	hdr->AS.Header = (uint8_t*) realloc(hdr->AS.Header,hdr->HeadLen+1);
    			count += ifread(Header1+count, 1, hdr->HeadLen-count, hdr);
			hdr->AS.Header[hdr->HeadLen]=0;
		}

		hdr->NS   = *(uint16_t*)(hdr->AS.Header+2);
		time_t T0 = (time_t)*(uint32_t*)(hdr->AS.Header+32);	// seconds since 1970
		hdr->T0   = t_time2gdf_time(T0);
		hdr->SampleRate = 1e6 / (*(uint32_t*)(hdr->AS.Header+36));
		strncpy(hdr->Patient.Id, (const char*)hdr->AS.Header+32+24+256*9, min(64,MAX_LENGTH_PID));
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		for (k=0; k<hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL + k;
			hc->OnOff = 1;
			strncpy(hc->Label,(char*)(hdr->AS.Header+32+24+8*k),8);
			hc->Transducer[0] = 0;
			hc->LeadIdCode = 0;
		}
    		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format RDF (UCSD ERPSS) not supported");
	}

	else if (hdr->TYPE==IntanCLP) {
		sopen_intan_clp_read(hdr);
	}
	else if (hdr->TYPE==RHD2000) {
		sopen_rhd2000_read(hdr);
	}
	else if (hdr->TYPE==RHS2000) {
		sopen_rhs2000_read(hdr);
	}

	else if (hdr->TYPE==SCP_ECG) {
		hdr->HeadLen   = leu32p(hdr->AS.Header+2);
		if (count < hdr->HeadLen) {
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,hdr->HeadLen);
			count += ifread(hdr->AS.Header+count, 1, hdr->HeadLen-count, hdr);
		}
		uint16_t crc   = CRCEvaluate(hdr->AS.Header+2,hdr->HeadLen-2);
		if ( leu16p(hdr->AS.Header) != crc) {
			biosigERROR(hdr, B4C_CRC_ERROR, "Warning SOPEN(SCP-READ): Bad CRC!");
		}
		hdr->Version = hdr->AS.Header[14]/10.0;
		sopen_SCP_read(hdr);

		serror2(hdr);	// report and reset error, but continue
		// hdr->FLAG.SWAP = 0; 	// no swapping
		hdr->FILE.LittleEndian = (__BYTE_ORDER == __LITTLE_ENDIAN); 	// no swapping
		hdr->AS.length = hdr->NRec;
	}

	else if (hdr->TYPE==Sigma) {  /********* Sigma PLpro ************/
		hdr->HeadLen = leu32p(hdr->AS.Header+16);
		if (count < hdr->HeadLen) {
			hdr->AS.Header = (uint8_t*) realloc(hdr->AS.Header,hdr->HeadLen+1);
			count += ifread(Header1+count, 1, hdr->HeadLen-count, hdr);
		}
		hdr->AS.Header[count]=0;

		struct tm t;
		char *tag, *val;
		size_t pos = leu32p(hdr->AS.Header+28);
		uint8_t len;

		typeof(hdr->NS) k;
		for (k=0; k<5; k++) {
#define line ((char*)(hdr->AS.Header+pos))
			len = strcspn(line,"\x0a\x0d");
			line[len] = 0;
			tag = line;
			val = strchr(line,'=');
			if (val!=NULL) {
				val[0] = 0;
				val++;
			}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%i: %s=%s\n",k,tag,val);

			if (0) {}
			//else if (!strcmp(tag,"Name")) {}
			//else if (!strcmp(tag,"Vorname")) {}
			else if (!strcmp(tag,"GebDat")) {
				sscanf(val,"%02u.%02u.%04u",&t.tm_mday,&t.tm_mon,&t.tm_year);
				t.tm_year -=1900;
				t.tm_mon--;
				t.tm_hour = 12;
				t.tm_min = 0;
				t.tm_sec = 0;
				t.tm_isdst = -1;
				hdr->T0 = tm_time2gdf_time(&t);
			}
			else if (!strcmp(tag,"ID"))
				strncpy(hdr->Patient.Id,val,MAX_LENGTH_PID);

			pos += len+1;
			while ((line[0]==10) || (line[0]==13)) pos++;
		}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"333 SIGMA  pos=%i, 0x%x\n", (int)pos, (int)pos);

		hdr->NS = leu16p(hdr->AS.Header+pos);
		hdr->SampleRate = 128;
		hdr->SPR  = 1;
		hdr->NRec = -1;		// unknown
		struct stat stbuf;
		if(!stat(hdr->FileName, &stbuf)) {
		if (!hdr->FILE.COMPRESSION)
			hdr->NRec = (stbuf.st_size-hdr->HeadLen)/(2*hdr->NS);
		else
			fprintf(stdout,"Compressed Sigma file (%s) is currently not supported. Uncompress file and try again.", hdr->FileName);
		}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"333 SIGMA NS=%i/0x%x, Fs=%f, SPR=%i, NRec=%i\n",hdr->NS,hdr->NS, hdr->SampleRate,hdr->SPR,(int)hdr->NRec);

	       	// define variable header
		pos     = 148;
		hdr->FLAG.UCAL = 1;
		hdr->FLAG.OVERFLOWDETECTION = 0;
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));

		uint16_t p[] = {4,19,19,19,19+2,19,19,19,19+8,9,11};	// difference of positions of string elements within variable header
		for (k=1; k < sizeof(p)/sizeof(p[0]); k++) p[k] += p[k-1];	// relative position

		double *fs = (double*) malloc(hdr->NS * sizeof(double));
		double minFs = INFINITY;
		for (k=0; k<hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			pos = 148 + k*203;
			// ch = lei16p(hdr->AS.Header+pos);
			double val;
			hc->GDFTYP = 3;
			hc->OnOff  = 1;
			hc->SPR    = 1;
		      	hc->DigMax    = (int16_t)0x7fff;
		      	hc->DigMin    = (int16_t)0x8000;
	      		hc->PhysMax   = hc->DigMax;
		      	hc->PhysMin   = hc->DigMin;
//		      	hc->Cal       = 1.0;
	      		hc->Off       = 0.0;
		      	hc->HighPass  = NAN;
		      	hc->LowPass   = NAN;
		      	hc->Notch     = *(int16_t*)(hdr->AS.Header+pos+2) ? 1.0 : 0.0;
		      	hc->Impedance = INFINITY;
		      	hc->fZ        = NAN;
	      		hc->XYZ[0]    = 0.0;
		      	hc->XYZ[1]    = 0.0;
		      	hc->XYZ[2]    = 0.0;
			hc->LeadIdCode = 0;
			hc->Transducer[0] = 0;
			hc->Label[0] = 0;

			unsigned k1;
			for (k1 = sizeof(p)/sizeof(p[0]); k1>0; ) {
				k1--;
				len = hdr->AS.Header[pos+p[k1]];
				hdr->AS.Header[pos+p[k1]+len+1] = 0;
				val = atof((char*)(hdr->AS.Header+pos+p[k1]+1));
				switch (k1) {
				case 0: 	// Abtastrate
					fs[k] = val;
					if (hdr->SampleRate < fs[k]) hdr->SampleRate=fs[k];
					if (minFs > fs[k]) minFs=fs[k];
					break;
				case 1: 	// obere Grenzfrequenz
				      	hc->LowPass = val;
				      	break;
				case 2: 	// untere Grenzfrequenz
				      	hc->HighPass = val;
				      	break;
				case 6: 	// Electrodenimpedanz
				      	hc->Impedance = val;
				      	break;
				case 7: 	// Sensitivitaet Verstaerker
				      	hc->Cal = val;
				      	break;
				case 8: 	// Elektrodenbezeichnung
					strcpy(hc->Label, (char*)(hdr->AS.Header+pos+p[k1]+1));
					break;
				case 10: 	// Masseinheit
					hc->PhysDimCode = PhysDimCode((char*)(hdr->AS.Header+pos+p[k1]+1));
					break;
				case 11: 	//
					strcpy(hc->Transducer, (char*)(hdr->AS.Header+pos+p[k1]+1));
					break;
				case 3: 	// gerfac ?
				case 4: 	// Kalibriergroesse
				case 5: 	// Kanaldimension
				case 9: 	// Bezeichnung der Referenzelektrode
					{};
				}

			}
			hc->Off    = lei16p(hdr->AS.Header+pos+ 80) * hc->Cal;
			hc->XYZ[0] = lei32p(hdr->AS.Header+pos+158);
			hc->XYZ[1] = lei32p(hdr->AS.Header+pos+162);
			hc->XYZ[2] = lei32p(hdr->AS.Header+pos+166);
	  	}
#undef line
		hdr->SPR = (typeof(hdr->SPR))round(hdr->SampleRate/minFs);
		for (k=0,hdr->AS.bpb=0; k<hdr->NS; k++) {
			hdr->CHANNEL[k].SPR = (typeof(hdr->SPR))round(fs[k]/minFs);
		      	hdr->CHANNEL[k].bi = hdr->AS.bpb;
		      	hdr->AS.bpb += hdr->CHANNEL[k].SPR*2;
		}
		free(fs);
	}	/******* end of Sigma PLpro ********/

	else if (hdr->TYPE==SQLite) {
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d): %s(...)\n", __FILE__,__LINE__,__func__);
		if (sopen_sqlite(hdr)) return(hdr);
	}

/*
	else if (hdr->TYPE==SMA) {
		char *delim = "=\x0a\x0d\x22";
	}
*/
	else if (hdr->TYPE==SND) {
		/* read file */
		// hdr->FLAG.SWAP  = (__BYTE_ORDER == __LITTLE_ENDIAN);
		hdr->FILE.LittleEndian = 0;
		hdr->HeadLen  	= beu32p(hdr->AS.Header+4);
		size_t datlen 	= beu32p(hdr->AS.Header+8);
		uint32_t filetype = beu32p(hdr->AS.Header+12);
		hdr->SampleRate	= (double)beu32p(hdr->AS.Header+16);
		hdr->NS	      	= beu32p(hdr->AS.Header+20);
		if (count<hdr->HeadLen) {
		    	hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,hdr->HeadLen);
		    	count  += ifread(hdr->AS.Header+count,1,hdr->HeadLen-count,hdr);
		}
		else {
		    	hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,hdr->HeadLen);
			ifseek(hdr,hdr->HeadLen,SEEK_SET);
		}
	    	const uint16_t	SND_GDFTYP[] = {0,2,1,3,255+24,5,16,17,0,0,0,2,4,511+25,6};
		uint16_t gdftyp = SND_GDFTYP[filetype];
		hdr->AS.bpb = hdr->NS * GDFTYP_BITS[gdftyp]>>3;
		double Cal = 1;
		if ((filetype>1) && (filetype<6))
			Cal = ldexp(1,-GDFTYP_BITS[gdftyp]);

		hdr->NRec = datlen/hdr->AS.bpb;
		hdr->SPR  = 1;

		hdr->FLAG.OVERFLOWDETECTION = 0; 	// automated overflow and saturation detection not supported
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		double digmax = ldexp(1,GDFTYP_BITS[gdftyp]);
		for (k=0,hdr->AS.bpb=0; k < hdr->NS; k++) {
			CHANNEL_TYPE* hc = hdr->CHANNEL+k;
			hc->OnOff    = 1;
			hc->GDFTYP   = gdftyp;
			hc->SPR      = 1;
			hc->Cal      = Cal;
			hc->Off      = 0.0;
			hc->Transducer[0] = '\0';
			hc->LowPass  = -1;
			hc->HighPass = -1;
			hc->PhysMax  =  1.0;
			hc->PhysMin  = -1.0;
			hc->DigMax   =  digmax;
			hc->DigMin   = -digmax;
		    	hc->LeadIdCode  = 0;
		    	hc->PhysDimCode = 0;
		    	hc->Label[0] = 0;
		    	hc->bi    = hdr->AS.bpb;
			hdr->AS.bpb  += GDFTYP_BITS[gdftyp]>>3;
		}
	}

#if defined(WITH_TDMS)
	else if (hdr->TYPE==TDMS) {
		sopen_tdms_read(hdr);
	}
#endif

	else if (hdr->TYPE==TMS32) {
		hdr->VERSION 	= leu16p(hdr->AS.Header+31);
		hdr->SampleRate = leu16p(hdr->AS.Header+114);
		size_t NS 	= leu16p(hdr->AS.Header+119);
		hdr->HeadLen 	= 217 + NS*136;
		if (hdr->HeadLen > count) {
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,hdr->HeadLen);
		    	count += ifread(hdr->AS.Header+count, 1, hdr->HeadLen-count, hdr);
		} else
			ifseek(hdr,hdr->HeadLen,SEEK_SET);

		// size_t filesize	= lei32p(hdr->AS.Header+121);
		tm_time.tm_year = lei16p(hdr->AS.Header+129)-1900;
		tm_time.tm_mon 	= lei16p(hdr->AS.Header+131)-1;
		tm_time.tm_mday = lei16p(hdr->AS.Header+133);
		tm_time.tm_hour = lei16p(hdr->AS.Header+137);
		tm_time.tm_min 	= lei16p(hdr->AS.Header+139);
		tm_time.tm_sec 	= lei16p(hdr->AS.Header+141);
		tm_time.tm_isdst= -1;
		hdr->T0 	= tm_time2gdf_time(&tm_time);
		hdr->NRec 	= lei32p(hdr->AS.Header+143);
		hdr->SPR 	= leu16p(hdr->AS.Header+147);
		//hdr->AS.bpb 	= leu16p(hdr->AS.Header+149)+86;
		hdr->AS.bpb = 86;
		hdr->FLAG.OVERFLOWDETECTION = 0;

		hdr->CHANNEL = (CHANNEL_TYPE*)calloc(NS, sizeof(CHANNEL_TYPE));
		size_t aux = 0;
		for (k=0; k < NS; k += 1) {
			CHANNEL_TYPE* hc = hdr->CHANNEL+aux;
			uint8_t StringLength = hdr->AS.Header[217+k*136];
			char *SignalName = (char*)(hdr->AS.Header+218+k*136);
			if (!strncmp(SignalName, "(Lo) ", 5)) {
				strncpy(hc->Label,SignalName+5,StringLength-5);
				hc->GDFTYP = 16;
				aux += 1;
				hc->Label[StringLength-5] = 0;
				hc->bi    = hdr->AS.bpb;
			}
			else if (!strncmp(SignalName, "(Hi) ", 5)) {
			}
			else {
				strncpy(hc->Label, SignalName, StringLength);
				hc->GDFTYP = 3;
				aux += 1;
				hc->Label[StringLength] = 0;
				hc->bi    = hdr->AS.bpb;
			}

			StringLength = hdr->AS.Header[45+217+k*136];
			char tmp[256];
			strncpy(tmp, (char*)(hdr->AS.Header+46+217+k*136), StringLength);
			tmp[StringLength] = 0;
		    	hc->PhysDimCode = PhysDimCode(tmp);
			hc->PhysMin  = lef32p(hdr->AS.Header+56+217+k*136);
			hc->PhysMax  = lef32p(hdr->AS.Header+60+217+k*136);
			hc->DigMin   = lef32p(hdr->AS.Header+64+217+k*136);
			hc->DigMax   = lef32p(hdr->AS.Header+68+217+k*136);

			hc->Cal      = (hc->PhysMax-hc->PhysMin)/(hc->DigMax-hc->DigMin);
                	hc->Off      = hc->PhysMin - hc->Cal * hc->DigMin;

			hc->OnOff    = 1;
			hc->SPR      = hdr->SPR;
			hc->Transducer[0] = '\0';
			hc->LowPass  = -1;
			hc->HighPass = -1;
		    	hc->LeadIdCode  = 0;
			//hdr->AS.bpb    += 2 * hc->SPR;
			hdr->AS.bpb    += hc->SPR * (GDFTYP_BITS[hc->GDFTYP]>>3);

			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"k=%i\tLabel=%s [%s]\tNS=%i\tpos=%i\n",(int)k,SignalName,tmp,(int)NS,(int)iftell(hdr));

		}
		hdr->NS = aux;
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL,hdr->NS*sizeof(CHANNEL_TYPE));
	}

	else if (hdr->TYPE==TMSiLOG) {
		/* read header
		      docu says HeadLen = 141+275*NS, but our example has 135+277*NS;
		 */
		while (!ifeof(hdr)) {
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,count*2+1);
			count += ifread(hdr->AS.Header + count, 1, count, hdr);
		}
	    	ifclose(hdr);
	    	hdr->AS.Header[count] = 0;

	    	hdr->NS    = 0;
	    	hdr->SPR   = 1;
	    	hdr->NRec  = 1;
	    	double duration = 0.0;
	    	uint16_t gdftyp = 0;

	    	char *line = strstr(Header1,"Signals=");
		if (line) {
			char tmp[5];
			strncpy(tmp,line+8,5);
	 	    	hdr->NS    = atoi(tmp);
		}
		if (!hdr->NS) {
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "TMSiLOG: multiple data files not supported");
		}
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));

		char *filename = NULL;
		line = strtok(Header1,"\x0d\x0a");
		while (line) {
			char *val = strchr(line,'=');
			val[0] = 0;
			val++;

			if (!strcmp(line,"FileId")) {}
			else if (!strcmp(line,"Version"))
				hdr->VERSION = atoi(val);
			else if (!strcmp(line,"DateTime")) {
				struct tm t;
				sscanf(val,"%04d/%02d/%02d-%02d:%02d:%02d",&t.tm_year,&t.tm_mon,&t.tm_mday,&t.tm_hour,&t.tm_min,&t.tm_sec);
				t.tm_year -= 1900;
				t.tm_mon--;
				t.tm_isdst =-1;
			}
			else if (!strcmp(line,"Format")) {
				if      (!strcmp(val,"Float32")) gdftyp = 16;
				else if (!strcmp(val,"Int32  ")) gdftyp = 5;
				else if (!strcmp(val,"Int16  ")) gdftyp = 3;
				else if (!strcmp(val,"Ascii  ")) gdftyp = 0xfffe;
				else                             gdftyp = 0xffff;
			}
			else if (!strcmp(line,"Length")) {
				duration = atof(val);
			}
			else if (!strcmp(line,"Signals")) {
				hdr->NS = atoi(val);
			}
			else if (!strncmp(line,"Signal",6)) {
				char tmp[5];
				strncpy(tmp,line+6,4);
				size_t ch = atoi(tmp)-1;
				char *field = line+11;

				if (!strcmp(field,"Name"))
					strncpy(hdr->CHANNEL[ch].Label,val,MAX_LENGTH_LABEL);
				else if (!strcmp(field,"UnitName"))
					hdr->CHANNEL[ch].PhysDimCode=PhysDimCode(val);
				else if (!strcmp(field,"Resolution"))
					hdr->CHANNEL[ch].Cal=atof(val);
				else if (!strcmp(field,"StoreRate")) {
					hdr->NRec = (nrec_t)atof(val)*duration;
					hdr->CHANNEL[ch].SPR = 1;
					// hdr->CHANNEL[ch].SPR=atof(val)*duration;
					//hdr->SPR = lcm(hdr->SPR,hdr->CHANNEL[ch].SPR);
				}
				else if (!strcmp(field,"File")) {
					if (!filename)
						filename = val;
					else if (strcmp(val, filename)) {
						fprintf(stdout,"<%s><%s>",val,filename);
						biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "TMSiLOG: multiple data files not supported");
					}
				}
				else if (!strcmp(field,"Index")) {}
				else
					fprintf(stdout,"TMSi Signal%04i.%s = <%s>\n",(int)ch,field,val);
			}
			else
				fprintf(stdout,"TMSi %s = <%s>\n",line,val);

			line = strtok(NULL,"\x0d\x0a");
		}

		hdr->SampleRate = hdr->SPR*hdr->NRec/duration;
		hdr->NRec *= hdr->SPR;
		hdr->SPR   = 1;
		char *fullfilename = (char*)malloc(strlen(hdr->FileName)+strlen(filename)+1);

		if (!filename) {
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "TMSiLOG: data file not specified");
		}
		else if (strrchr(hdr->FileName,FILESEP)) {
			strcpy(fullfilename,hdr->FileName);
			strcpy(strrchr(fullfilename,FILESEP)+1,filename);
		}
		else {
			strcpy(fullfilename,filename);
		}
		filename = NULL; // filename had a pointer to hdr->AS.Header; could be released here

		if (gdftyp < 1000) {
			FILE *fid = fopen(fullfilename,"rb");

			if (!fid) {
				biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "TMSiLOG: data file not found");
			}
			else {
				int16_t h[3];
				size_t c = fread(h, 2, 3, fid);
				if      (h[2]==16)  h[2] = 3;
				else if (h[2]==32)  h[2] = 5;
				else if (h[2]==288) h[2] = 16;
				else                h[2] = 0xffff;  	// this triggers the error trap

				if ( (c<2) || (h[0] != hdr->NS) || (((double)h[1]) != hdr->SampleRate) || (h[2] != gdftyp) ) {
					biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "TMSiLOG: Binary file corrupted");
				}
				else {
					size_t sz = (size_t)hdr->NS*hdr->SPR*hdr->NRec*(GDFTYP_BITS[gdftyp]>>3);
					hdr->AS.rawdata = (uint8_t*)realloc(hdr->AS.rawdata,sz);
					c = fread(hdr->AS.rawdata, hdr->NRec, hdr->SPR*hdr->NS*GDFTYP_BITS[gdftyp]>>3, fid);
					if (c < sz) hdr->NRec = c;
				}
				fclose(fid);
			}
		}
		else if (gdftyp==0xfffe) {
			// double Fs;
			gdftyp = 17; 	// ascii is converted to double
			FILE *fid = fopen(fullfilename,"rt");

			if (!fid) {
				biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "TMSiLOG: data file not found");
			}
			else {
				size_t sz  = (hdr->NS+1)*24;
				char *line = (char*) malloc(sz);
				// read and ignore (i.e. skip) 3 lines
				int c = 3;
				while (c>0) {
					if (fgetc(fid)=='\n') {
						c--;
						int ch = fgetc(fid);
						// skip '\r'
						if (ch=='\r') ungetc(ch,fid);
					}
				}

				// TODO: sanity checks
				if (0) {
					biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "TMSiLOG: Binary file corrupted");
				}
				else {
					// hdr->FLAG.SWAP = 0; 	// no swaping required
					typeof(hdr->NS) k2;
					size_t k;
					hdr->FILE.LittleEndian = (__BYTE_ORDER == __LITTLE_ENDIAN);
					size_t sz = (size_t)hdr->NS * hdr->SPR * hdr->NRec * GDFTYP_BITS[gdftyp]>>3;
					hdr->AS.rawdata = (uint8_t*)realloc(hdr->AS.rawdata,sz);
					for (k=0; k < (size_t)hdr->SPR*hdr->NRec; k++)
					if (fgets(line,sz,fid)) {
						char *next;
						strtoul(line, &next, 10);	// skip index entry
						for (k2=0;k2<hdr->NS;k2++)
							*(double*)(hdr->AS.rawdata+(k*hdr->NS+k2)*sizeof(double)) = strtod(next,&next);
					}
					else {
						for (k2=0;k2<hdr->NS;k2++)
							*(double*)(hdr->AS.rawdata+(k*hdr->NS+k2)*sizeof(double)) = NAN;
					}
				}
				free(line);
				fclose(fid);
			}
		}
		free(fullfilename);
		hdr->AS.length  = hdr->NRec;

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"TMSi [149] \n");


		hdr->AS.bpb = 0;
		for (k=0; k<hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;

			if (VERBOSE_LEVEL>8)
				fprintf(stdout,"TMSi [151] %i\n",(int)k);

			hc->GDFTYP = gdftyp;
			if (gdftyp==3) {
				hc->DigMax = (double)(int16_t)0x7fff;
				hc->DigMin = (double)(int16_t)0x8000;
			}
			else if (gdftyp==5) {
				hc->DigMax = (double)(int32_t)0x7fffffff;
				hc->DigMin = (double)(int32_t)0x80000000;
			}
			else {
				hc->DigMax = 1.0;
				hc->DigMin = 0.0;
				hdr->FLAG.OVERFLOWDETECTION = 0;	// no overflow detection
			}
			hc->PhysMax = hc->DigMax * hc->Cal;
			hc->PhysMin = hc->DigMin * hc->Cal;
		      	hc->LeadIdCode = 0;
	      		hc->Transducer[0] = 0;
		      	hc->SPR       = 1;	// one sample per block
		      	hc->OnOff     = 1;
	      		hc->HighPass  = NAN;
		      	hc->LowPass   = NAN;
		      	hc->Notch     = NAN;
	      		hc->Impedance = INFINITY;
		      	hc->fZ        = NAN;
		      	hc->XYZ[0] = 0.0;
		      	hc->XYZ[1] = 0.0;
		      	hc->XYZ[2] = 0.0;
		      	hc->bi  = hdr->AS.bpb;
			hdr->AS.bpb += GDFTYP_BITS[gdftyp]>>3;
		}
	}

	else if (hdr->TYPE==AIFF) {
		// hdr->FLAG.SWAP  = (__BYTE_ORDER == __LITTLE_ENDIAN);
		hdr->FILE.LittleEndian = 0;
		uint8_t *tag;
		uint32_t tagsize;
		//uint16_t gdftyp;
		size_t pos;
		tagsize  = beu32p(hdr->AS.Header+4);
		tagsize += tagsize & 0x0001;
		pos 	 = 12;
		tag	 = hdr->AS.Header+pos;
		while (1) {
			tagsize  = beu32p(hdr->AS.Header+pos+4);
			tagsize += tagsize & 0x0001;
			if (!strncmp((char*)tag,"COMM",4)) {
			}
			else if (!strncmp((char*)tag,"DATA",4)) {
			}

			pos += tagsize;
			tag  = hdr->AS.Header+pos;
		}
		/// TODO, FIXME
	}

#if defined(WITH_WAV) || defined(WITH_AVI) || defined(WITH_RIFF)
	else if ((hdr->TYPE==WAV)||(hdr->TYPE==AVI)||(hdr->TYPE==RIFF)) {
		// hdr->FLAG.SWAP  = (__BYTE_ORDER == __BIG_ENDIAN);
		hdr->FILE.LittleEndian = 1;
		uint8_t *tag;
		uint32_t tagsize;
		uint16_t gdftyp;
		uint16_t format=0, bits = 0;
		double Cal=1.0, Off=0.0;
		size_t pos;
		tagsize  = leu32p(hdr->AS.Header+4);
		tagsize += tagsize & 0x0001;
		pos 	 = 12;
		tag	 = hdr->AS.Header+pos;
		while (1) {
			tagsize  = leu32p(hdr->AS.Header+pos+4);
			tagsize += tagsize & 0x0001;
			if (!strncmp((char*)tag,"fmt ",4)) {
				format	 	= leu16p(hdr->AS.Header+pos+4);
				hdr->NS 	= leu16p(hdr->AS.Header+pos+4+2);
				hdr->SampleRate = (double)leu32p(hdr->AS.Header+pos+4+4);
				if (format==1) {
					bits 	= leu16p(hdr->AS.Header+pos+4+14);
					Cal 	= ldexp(1,-8*(bits/8 + ((bits%8) > 0)));
					if 	(bits <= 8) {
						gdftyp = 2;
						Off = 0.5;
					}
					else if (bits <= 16)
						gdftyp = 3;
					else if (bits <= 24)
						gdftyp = 255+24;
					else if (bits <= 32)
						gdftyp = 5;
				}
				else
					fprintf(stdout,"Warning (WAV): format not supported.\n");
			}
			else if (!strncmp((char*)tag,"data",4)) {
				if (format==1) {
					hdr->AS.bpb = hdr->NS * ((bits/8) + ((bits%8)>0));
					hdr->SPR    = tagsize/hdr->AS.bpb;
				}
			}

			pos += tagsize;
			tag  = hdr->AS.Header+pos;
		/// TODO, FIXME
		}
	}
#endif

	else if (hdr->TYPE==ASCII_IBI) {

		hdr->NS   = 0;
		hdr->NRec = 0;
		hdr->SPR  = 1;
		hdr->AS.bpb = 0;
		ifseek(hdr,0,SEEK_SET);
		char line[81];
		ifgets(line,80,hdr);
		size_t N = 0;
		hdr->EVENT.N = 0;
		while (!ifeof(hdr) && strlen(line)) {

			if (isdigit(line[0])) {
				struct tm t;
				int ms,rri;
				char *desc = NULL;

#if !defined __STDC_VERSION__ || __STDC_VERSION__ < 199901L
				sscanf(line,"%02u-%02u-%02u %02u:%02u:%02u %03u %as %u", &t.tm_mday, &t.tm_mon, &t.tm_year, &t.tm_hour, &t.tm_min, &t.tm_sec, &ms, &desc, &rri);
#else
				sscanf(line,"%02u-%02u-%02u %02u:%02u:%02u %03u %ms %u", &t.tm_mday, &t.tm_mon, &t.tm_year, &t.tm_hour, &t.tm_min, &t.tm_sec, &ms, &desc, &rri);
#endif
				if (t.tm_year < 1970) t.tm_year += 100;
				t.tm_mon--;
				t.tm_isdst = -1;

				if (N+1 >= hdr->EVENT.N) {
					hdr->EVENT.N   += max(4096,hdr->EVENT.N);
			 		hdr->EVENT.POS = (typeof(hdr->EVENT.POS)) realloc(hdr->EVENT.POS, hdr->EVENT.N*sizeof(*hdr->EVENT.POS) );
					hdr->EVENT.TYP = (typeof(hdr->EVENT.TYP)) realloc(hdr->EVENT.TYP, hdr->EVENT.N*sizeof(*hdr->EVENT.TYP) );
				}
				if (N==0) {
					hdr->T0 = (gdf_time)(tm_time2gdf_time(&t) + ldexp((ms-rri)/(24*3600*1e3),32));
					hdr->EVENT.POS[0] = 0;
					hdr->EVENT.TYP[0] = 0x0501;
					hdr->EVENT.POS[1] = rri;
					hdr->EVENT.TYP[1] = 0x0501;
					N = 1;
				}
				else {
					hdr->EVENT.POS[N] = hdr->EVENT.POS[N-1] + rri;
				}

				if (!strcmp(desc,"IBI"))
					hdr->EVENT.TYP[N] = 0x0501;
				else
					FreeTextEvent(hdr,N,desc);

				++N;
				if (desc) free(desc);
			}
			else {
				strtok(line,":");
				char *v = strtok(NULL,":");
				if (!strncmp(line,"File version",12))
					hdr->VERSION = atof(v);
				else if (!hdr->FLAG.ANONYMOUS && !strncmp(line,"File version",12))
					strncpy(hdr->Patient.Id,v,MAX_LENGTH_PID);
			}
			ifgets(line,80,hdr);
		}
	    	ifclose(hdr);
	    	hdr->EVENT.N = N;
	    	hdr->SampleRate = 1000.0;
	    	hdr->EVENT.SampleRate = 1000.0;
	    	hdr->TYPE = EVENT;
		hdr->data.block = NULL;
		hdr->data.size[0] = 0;
		hdr->data.size[1] = 0;
	}

#if defined(WITH_DICOM) || defined(WITH_GDCM) || defined(WITH_DCMTK)
	else if (hdr->TYPE==DICOM) {
		fprintf(stderr,"DICOM support is very (!!!) experimental!\n");

		hdr->HeadLen = count;
		sopen_dicom_read(hdr);

    		return(hdr);
	}
#endif

	else if (hdr->TYPE==HL7aECG || hdr->TYPE==XML) {
		sopen_HL7aECG_read(hdr);
		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"[181] #%i\n",hdr->NS);

    		if (hdr->AS.B4C_ERRNUM) return(hdr);
    		// hdr->FLAG.SWAP = 0;
		hdr->FILE.LittleEndian = (__BYTE_ORDER == __LITTLE_ENDIAN); // no swapping
		hdr->AS.length  = hdr->NRec;
	}

#ifdef WITH_MICROMED
    	else if (hdr->TYPE==TRC) {
    		sopen_TRC_read(hdr);
	}
#endif

	else if (hdr->TYPE==UNIPRO) {
		sopen_unipro_read(hdr);
		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"[181] #%i\n",hdr->NS);

    		if (hdr->AS.B4C_ERRNUM) return(hdr);
    		// hdr->FLAG.SWAP = 0;
	}

	else if (hdr->TYPE==WCP) {

		fprintf(stderr,"%s (line %i) %s: WARNING WCP support is experimental!\n", __FILE__,__LINE__,__func__);

		const int WCP_HEADER_LENGTH=2048;
		if (hdr->HeadLen < WCP_HEADER_LENGTH) {
			hdr->AS.Header = realloc(hdr->AS.Header, WCP_HEADER_LENGTH);
			hdr->HeadLen  += ifread(hdr->AS.Header+hdr->HeadLen, 1, WCP_HEADER_LENGTH - hdr->HeadLen, hdr);
		}
		int ADCMAX=0;

		char  *tok = strtok(hdr->AS.Header,"\r\n");
		while (tok != NULL) {
			char *sep = strchr(tok,'=');
			*sep = 0;
			char *val = sep+1;
			char *key = tok;
			if (!strcmp(key,"VER"))
				hdr->VERSION=atof(val);
			else if (!strcmp(key,"RTIME")) {
				struct tm T;
#if !defined(_WIN32)
				strptime(val,"%d/%m/%Y %T",&T);
#else
				char *p=val+strlen(val);
				do p--; while (isdigit(*p) && (p>val));
				T.tm_sec = atoi(p+1);
				*p=0;
				do p--; while (isdigit(*p) && (p>val));
				T.tm_min = atoi(p+1);
				*p=0;
				do p--; while (isdigit(*p) && (p>val));
				T.tm_hour = atoi(p+1);
				*p=0;
				do p--; while (isdigit(*p) && (p>val));
				T.tm_year = atoi(p+1);
				*p=0;
				do p--; while (isdigit(*p) && (p>val));
				T.tm_mon = atoi(p+1);
				*p=0;
				do p--; while (isdigit(*p) && (p>val));
				T.tm_mday = atoi(p+1);
#endif
				hdr->T0 = tm_time2gdf_time(&T);
			}
			else if (!strcmp(key,"NC")) {
				hdr->NS=atoi(val);
				hdr->CHANNEL = realloc(hdr->CHANNEL, sizeof(CHANNEL_TYPE)*hdr->NS);
			}
			else if (!strcmp(key,"DT"))
				hdr->SampleRate=1.0/atof(val);
			else if (!strcmp(key,"ADCMAX"))
				ADCMAX=atoi(val);
			else if (!strcmp(key,"NR"))
				hdr->NRec=atoi(val);
			else if (key[0]=='Y') {
				int chan = atoi(key+2);
				CHANNEL_TYPE *hc = hdr->CHANNEL+chan;
				switch (key[1]) {
				case 'U':	// YU
					hc->PhysDimCode = PhysDimCode(val);
					break;
				case 'N':	// YN
					strncpy(hc->Label, val, MAX_LENGTH_LABEL+1);
					break;
				case 'G':	// YG
					hc->Cal = atof(val);
					break;
				}
			}
			else {
				fprintf(stdout,"%s-WCP: %s=%s ignored\n",__func__,key,val);
			}
			tok = strtok(NULL,"\r\n");
		}

		uint16_t gdftyp=3;
		if (ADCMAX==32767) gdftyp = 3;

		struct stat FileBuf;
		stat(hdr->FileName,&FileBuf);

		hdr->SPR  = 1;
		typeof (hdr->NS) k;
		size_t bpb8=0;
		for (k = 0; k < hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			hc->Transducer[0] = 0;
			hc->GDFTYP  =  gdftyp;
			hc->OnOff   =  1;
			hc->bi      =  bpb8>>3;
			hc->Off     =  0.0;
			hc->DigMax  =  ADCMAX;
			hc->DigMin  = -ADCMAX;
			hc->PhysMax =  ADCMAX*hc->Cal;
			hc->PhysMin = -ADCMAX*hc->Cal;
			hc->SPR     = 1;
			hc->TOffset = 0.0;
			hc->LowPass = 0.0;
			hc->HighPass= 0.0;
			hc->Notch   = 0.0;
			hc->Impedance     = NAN;
			hc->LeadIdCode    = 0;
			hc->Transducer[0] = 0;
			hc->XYZ[0]  = 0.0;
			hc->XYZ[1]  = 0.0;
			hc->XYZ[2]  = 0.0;
			hc->bufptr  = NULL;
			bpb8 += GDFTYP_BITS[gdftyp];
		}
		hdr->AS.bpb = bpb8>>3;
		hdr->NRec   = (FileBuf.st_size - WCP_HEADER_LENGTH)/hdr->AS.bpb;
	}

	else if (hdr->TYPE==WG1) {
		uint32_t VER = leu32p(hdr->AS.Header);
		if (VER==0xAFFE5555) {
			// FIXME: this version is currently not implemented
			if (count < 5120) {
				hdr->AS.Header = realloc(hdr->AS.Header, 5120);
				count += ifread(hdr->AS.Header,5120-count,1,hdr);
			}
			hdr->HeadLen = count;
			hdr->NS   = leu16p(hdr->AS.Header+0x40);
			hdr->NRec = leu16p(hdr->AS.Header+0x110);	// total number of blocks
			/*
			uint16_t gdftyp = 1;
			uint16_t startblock = leu16p(hdr->AS.Header+0x112);
			// FIXME:
			*/
	    		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "ERROR BIOSIG SOPEN(READ): WG1 0x5555FEAF format is not supported yet");
		}
		else {
			hdr->SampleRate = 1e6 / leu32p(Header1+16);
			hdr->NS = leu16p(Header1+22);
	    		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "ERROR BIOSIG SOPEN(READ): WG1 data format is not supported yet");
		}
    		return(hdr);
	}
#endif //ONLYGDF
	else {
    		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "ERROR BIOSIG SOPEN(READ): data format is not supported");
    		ifclose(hdr);
    		return(hdr);
	}

	hdr->FILE.POS = 0;

	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"[189] #%i\n",hdr->NS);

	for (k=0; k<hdr->NS; k++) {
		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"[190] #%i: LeadIdCode=%i\n",(int)k,hdr->CHANNEL[k].LeadIdCode);

		// set HDR.PhysDim - this part will become obsolete
/*
		k1 = hdr->CHANNEL[k].PhysDimCode;
		if (k1>0) hdr->CHANNEL[k].PhysDim = PhysDim3(k1);
*/
		// set HDR.PhysDimCode
		if (hdr->CHANNEL[k].LeadIdCode == 0) {
			int k1;
			if (!strncmp(hdr->CHANNEL[k].Label, "MDC_ECG_LEAD_", 13)) {
				// MDC_ECG_LEAD_*  - ignore case  //
				for (k1=0; strcasecmp(hdr->CHANNEL[k].Label+13,LEAD_ID_TABLE[k1]) && LEAD_ID_TABLE[k1][0]; k1++) {};
				if (LEAD_ID_TABLE[k1][0])
					hdr->CHANNEL[k].LeadIdCode = k1;
			}
			else {
				for (k1=0; strcmp(hdr->CHANNEL[k].Label, LEAD_ID_TABLE[k1]) && LEAD_ID_TABLE[k1][0]; k1++) {};
				if (LEAD_ID_TABLE[k1][0])
					hdr->CHANNEL[k].LeadIdCode = k1;
			}
		}

		// based on ISO/DIS 11073-91064, EN 1064:2005+A1:2007 (E)
		if (200 <= hdr->CHANNEL[k].LeadIdCode)
			strcpy(hdr->CHANNEL[k].Label,"(Manufacturere specific)");
		else if (185 <= hdr->CHANNEL[k].LeadIdCode)
			strcpy(hdr->CHANNEL[k].Label,"(reserved for future expansion)");
		else if (hdr->CHANNEL[k].LeadIdCode)
			strcpy(hdr->CHANNEL[k].Label,LEAD_ID_TABLE[hdr->CHANNEL[k].LeadIdCode]);	// Flawfinder: ignore

	}

	if (!hdr->EVENT.SampleRate) hdr->EVENT.SampleRate = hdr->SampleRate;
	/*
	convert2to4_eventtable(hdr);
	convert into canonical form if needed
	*/

}
else if (!strncmp(MODE,"w",1))	 /* --- WRITE --- */
{

	hdr->FILE.COMPRESSION = hdr->FILE.COMPRESSION || strchr(MODE,'z');
	if ( (hdr->Patient.Id==NULL) || !strlen(hdr->Patient.Id))
		strcpy(hdr->Patient.Id,"00000000");

#ifndef WITHOUT_NETWORK
    	if (!memcmp(hdr->FileName,"bscs://",7)) {
    		// network: write to server
                const char *hostname = hdr->FileName+7;
                char *tmp= (char*)strchr(hostname,'/');
                if (tmp != NULL) tmp[0]=0;   // ignore terminating slash

                uint64_t ID=0;
                int sd, s;
		sd = bscs_connect(hostname);
		if (sd<0) {
			fprintf(stdout,"could not connect to <%s> (err %i)\n",hostname, sd);
			biosigERROR(hdr, B4C_CANNOT_OPEN_FILE, "could not connect to server");
			return(hdr);
		}
  		hdr->FILE.Des = sd;
		s  = bscs_open(sd, &ID);
  		s  = bscs_send_hdr(sd,hdr);
  		hdr->FILE.OPEN = 2;
		fprintf(stdout,"write file to bscs://%s/%016"PRIx64"\n",hostname,ID);
  		return(hdr);
	}
#endif

	// NS number of channels selected for writing
     	typeof(hdr->NS) NS = 0;
     	{
     		typeof(hdr->NS) k;
		for (k=0; k<hdr->NS; k++)
			if (hdr->CHANNEL[k].OnOff) NS++;
	}

	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"sopen-W ns=%i (%s)\n",NS,GetFileTypeString(hdr->TYPE));

#ifndef  ONLYGDF
    	if ((hdr->TYPE==ASCII) || (hdr->TYPE==BIN)) {

		size_t k;
    		FILE *fid = fopen(hdr->FileName,"w");
		hdr->FILE.LittleEndian = 1;

    		fprintf(fid,"#BIOSIG %s\n", (hdr->TYPE==ASCII ? "ASCII" : "BINARY"));
    		fprintf(fid,"#   comments start with #\n\n");
    		fprintf(fid,"Filename\t= %s\t # (this file)\n",hdr->FileName);
    		fprintf(fid,"\n[Header 1]\n");
    		// fprintf(fid,"\n[Header 1]\nNumberOfChannels\t= %i\n",hdr->NS);
    		//fprintf(fid,"NRec\t= %i\n",hdr->NRec);
    		fprintf(fid,"Duration         \t= %f\t# in seconds\n",hdr->SPR*hdr->NRec/hdr->SampleRate);
		char tmp[40];
		snprintf_gdfdatetime(tmp, sizeof(tmp), hdr->T0);
		fprintf(fid,"Recording.Time    \t= %s\t# YYYY-MM-DD hh:mm:ss.uuuuuu\n",tmp);
		fprintf(fid,"Timezone          \t= +%i min\n",hdr->tzmin);

    		fprintf(fid,"Patient.Id        \t= %s\n",hdr->Patient.Id);
		snprintf_gdfdate(tmp, sizeof(tmp), hdr->Patient.Birthday);
		fprintf(fid,"Patient.Birthday  \t= %s\t# YYYY-MM-DD\n",tmp);
    		fprintf(fid,"Patient.Weight    \t= %i\t# in [kg]\n",hdr->Patient.Weight);
    		fprintf(fid,"Patient.Height    \t= %i\t# in [cm]\n",hdr->Patient.Height);
    		fprintf(fid,"Patient.Gender    \t= %i\t# 0:Unknown, 1: Male, 2: Female, 9: Unspecified\n",hdr->Patient.Sex);
    		fprintf(fid,"Patient.Handedness\t= %i\t# 0:Unknown, 1: Right, 2: Left, 3: Equal\n",hdr->Patient.Handedness);
    		fprintf(fid,"Patient.Smoking   \t= %i\t# 0:Unknown, 1: NO, 2: YES\n",hdr->Patient.Sex);
    		fprintf(fid,"Patient.AlcoholAbuse\t= %i\t# 0:Unknown, 1: NO, 2: YES\n",hdr->Patient.AlcoholAbuse);
    		fprintf(fid,"Patient.DrugAbuse \t= %i\t# 0:Unknown, 1: NO, 2: YES \n",hdr->Patient.DrugAbuse);
    		fprintf(fid,"Patient.Medication\t= %i\t# 0:Unknown, 1: NO, 2: YES \n",hdr->Patient.Medication);
		fprintf(fid,"Recording.ID      \t= %s\n",hdr->ID.Recording);
		uint8_t IPv6=0;
		for (k=4; k<16; k++) IPv6 |= hdr->IPaddr[k];
		if (IPv6)
			fprintf(fid,"Recording.IPaddress \t= %02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x\n",hdr->IPaddr[0],hdr->IPaddr[1],hdr->IPaddr[2],hdr->IPaddr[3],hdr->IPaddr[4],hdr->IPaddr[5],hdr->IPaddr[6],hdr->IPaddr[7],hdr->IPaddr[8],hdr->IPaddr[9],hdr->IPaddr[10],hdr->IPaddr[11],hdr->IPaddr[12],hdr->IPaddr[13],hdr->IPaddr[14],hdr->IPaddr[15]);
		else
			fprintf(fid,"Recording.IPaddress \t= %u.%u.%u.%u\n",hdr->IPaddr[0],hdr->IPaddr[1],hdr->IPaddr[2],hdr->IPaddr[3]);
		fprintf(fid,"Recording.Technician\t= %s\n",hdr->ID.Technician);
		fprintf(fid,"Manufacturer.Name \t= %s\n",hdr->ID.Manufacturer.Name);
		fprintf(fid,"Manufacturer.Model\t= %s\n",hdr->ID.Manufacturer.Model);
		fprintf(fid,"Manufacturer.Version\t= %s\n",hdr->ID.Manufacturer.Version);
		fprintf(fid,"Manufacturer.SerialNumber\t= %s\n",hdr->ID.Manufacturer.SerialNumber);


   		fprintf(fid,"\n[Header 2]\n");
		k = strlen(hdr->FileName);
		char* fn = (char*)calloc(k + 10,1);
		strcpy(fn, hdr->FileName);
		char *e = strrchr(fn,'.');
		if (e==NULL) {
			fn[k] = '.';
			e = fn+k;
		}

		e[1] = (hdr->TYPE == ASCII ? 'a' : 's');
		e+=2;

    		for (k=0; k<hdr->NS; k++)
    		if (hdr->CHANNEL[k].OnOff) {
    			if (hdr->FILE.COMPRESSION) sprintf(e,"%02i_gz",(int)k+1);
    			else sprintf(e,"%02i",(int)k+1);
	    		fprintf(fid,"Filename  \t= %s\n",fn);
	    		fprintf(fid,"Label     \t= %s\n",hdr->CHANNEL[k].Label);
	    		if (hdr->TYPE==ASCII)
		    		fprintf(fid,"GDFTYP    \t= ascii\n");
	    		else if (hdr->TYPE==BIN) {
	    			const char *gdftyp;
	    			switch (hdr->CHANNEL[k].GDFTYP) {
	    			case 1:	gdftyp="int8"; break;
	    			case 2:	gdftyp="uint8"; break;
	    			case 3:	gdftyp="int16"; break;
	    			case 4:	gdftyp="uint16"; break;
	    			case 5:	gdftyp="int32"; break;
	    			case 6:	gdftyp="uint32"; break;
	    			case 7:	gdftyp="int64"; break;
	    			case 8:	gdftyp="uint64"; break;
	    			case 16: gdftyp="float32"; break;
	    			case 17: gdftyp="float64"; break;
	    			case 18: gdftyp="float128"; break;
	    			case 255+24: gdftyp="bit24"; break;
	    			case 511+24: gdftyp="ubit24"; break;
	    			case 255+12: gdftyp="bit12"; break;
	    			case 511+12: gdftyp="ubit12"; break;
	    			default: gdftyp = "unknown";
	    			}
		    		fprintf(fid,"GDFTYP    \t= %s\n",gdftyp);
	    		}

	    		fprintf(fid,"Transducer\t= %s\n",hdr->CHANNEL[k].Transducer);
	    		fprintf(fid,"PhysicalUnits\t= %s\n",PhysDim3(hdr->CHANNEL[k].PhysDimCode));
	    		fprintf(fid,"PhysDimCode\t= %i\n",hdr->CHANNEL[k].PhysDimCode);
	    		fprintf(fid,"DigMax   \t= %f\n",hdr->CHANNEL[k].DigMax);
	    		fprintf(fid,"DigMin   \t= %f\n",hdr->CHANNEL[k].DigMin);
	    		fprintf(fid,"PhysMax  \t= %g\n",hdr->CHANNEL[k].PhysMax);
	    		fprintf(fid,"PhysMin  \t= %g\n",hdr->CHANNEL[k].PhysMin);
	    		fprintf(fid,"SamplingRate\t= %f\n",hdr->CHANNEL[k].SPR*hdr->SampleRate/hdr->SPR);
			fprintf(fid,"NumberOfSamples\t= %i\t# 0 indicates a channel with sparse samples\n",(int)(hdr->CHANNEL[k].SPR*hdr->NRec));
	    		fprintf(fid,"HighPassFilter\t= %f\n",hdr->CHANNEL[k].HighPass);
	    		fprintf(fid,"LowPassFilter\t= %f\n",hdr->CHANNEL[k].LowPass);
	    		fprintf(fid,"NotchFilter\t= %f\n",hdr->CHANNEL[k].Notch);
	    		switch (hdr->CHANNEL[k].PhysDimCode & 0xffe0) {
	       		case 4256:        // Voltage data
	    		        fprintf(fid,"Impedance\t= %f\n",hdr->CHANNEL[k].Impedance);
	    		        break;
	    		case 4288:         // Impedance data
	    		        fprintf(fid,"freqZ\t= %f\n",hdr->CHANNEL[k].fZ);
	    		        break;
	    		}
	    		fprintf(fid,"PositionXYZ\t= %f\t%f\t%f\n",hdr->CHANNEL[k].XYZ[0],hdr->CHANNEL[k].XYZ[1],hdr->CHANNEL[k].XYZ[2]);
//	    		fprintf(fid,"OrientationXYZ\t= %f\t%f\t%f\n",hdr->CHANNEL[k].Orientation[0],hdr->CHANNEL[k].Orientation[1],hdr->CHANNEL[k].Orientation[2]);
//	    		fprintf(fid,"Area     \t= %f\n",hdr->CHANNEL[k].Area);

	    		fprintf(fid,"\n");
			hdr->CHANNEL[k].SPR *= hdr->NRec;
    		}
		hdr->SPR *= hdr->NRec;
		hdr->NRec = 1;

		fprintf(fid,"[EVENT TABLE]\n");
		fprintf(fid,"TYP\tPOS [s]\tDUR [s]\tCHN\tVAL/Desc");

		for (k=0; k<hdr->EVENT.N; k++) {

			fprintf(fid,"\n0x%04x\t%f\t",hdr->EVENT.TYP[k],hdr->EVENT.POS[k]/hdr->EVENT.SampleRate);   // EVENT.POS uses 0-based indexing
			if (hdr->EVENT.DUR != NULL) {
				typeof(*hdr->EVENT.DUR) DUR;
				DUR = (hdr->EVENT.TYP[k]==0x7fff) ? 0 : hdr->EVENT.DUR[k];
				fprintf(fid,"%f\t%d\t", DUR/hdr->EVENT.SampleRate, hdr->EVENT.CHN[k]);
			}
			else
				fprintf(fid,"\t\t");

			if (hdr->EVENT.TYP[k] == 0x7fff) {
				typeof(hdr->NS) chan = hdr->EVENT.CHN[k] - 1;
				double val = dur2val(hdr->EVENT.DUR[k], hdr->CHANNEL[chan].GDFTYP);

				val *= hdr->CHANNEL[chan].Cal;
				val += hdr->CHANNEL[chan].Off;

				fprintf(fid,"%g\t# sparse sample ", val);	// value of sparse samples
			}
			else {
				const char *str = GetEventDescription(hdr,k);
 				if (str) fprintf(fid,"%s",str);
			}
		}
		fclose(fid);
		free(fn);
		hdr->FILE.POS  = 0;
    	}
	else if (hdr->TYPE==ATF) {
		// Write ATF

		hdr->HeadLen  = 0;
		hdr->FILE.POS = 0;
		hdr->FILE.FID = fopen(hdr->FileName, "w");

		typeof(hdr->NS) k, NS = 1;
		for (k = 0; k < hdr->NS; k++) {
			NS += (hdr->CHANNEL[k].OnOff > 0);
		}

		hdr->HeadLen += fprintf(hdr->FILE.FID, "ATF\t1.0\n0\t%d", NS);

		char sep = '\n';
		if (getTimeChannelNumber(hdr) == 0) {
			hdr->HeadLen += fprintf(hdr->FILE.FID, "%c\"Time (ms)\"", sep);
			sep = '\t';
		}

		for (k = 0; k < hdr->NS; k++) {
			if (hdr->CHANNEL[k].OnOff) {
				hdr->HeadLen += fprintf(hdr->FILE.FID, "%c\"%s (%s)\"", sep, hdr->CHANNEL[k].Label, PhysDim3(hdr->CHANNEL[k].PhysDimCode));
				sep = '\t';
			}
		}
	}
	else if (hdr->TYPE==BrainVision) {

		if (VERBOSE_LEVEL>8) fprintf(stdout,"BVA-write: [210]\n");

		char* tmpfile = (char*)calloc(strlen(hdr->FileName)+6,1);
		strcpy(tmpfile,hdr->FileName);			// Flawfinder: ignore
		char* ext = strrchr(tmpfile,'.');
		if (ext != NULL) strcpy(ext+1,"vhdr");		// Flawfinder: ignore
		else 		strcat(tmpfile,".vhdr");	// Flawfinder: ignore

		if (VERBOSE_LEVEL>8) fprintf(stdout,"BVA-write: [211]\n");

    		hdr->HeadLen = 0;
    		FILE *fid = fopen(tmpfile,"wb");
    		fprintf(fid,"Brain Vision Data Exchange Header File Version 1.0\r\n");
    		fprintf(fid,"; Data created by BioSig4C++\r\n\r\n");
    		fprintf(fid,"[Common Infos]\r\n");
    		fprintf(fid,"DataFile=%s\r\n",hdr->FileName);
    		fprintf(fid,"MarkerFile=%s\r\n",strcpy(strrchr(tmpfile,'.')+1,"vhdr"));
    		fprintf(fid,"DataFormat=BINARY\r\n");
    		fprintf(fid,"; Data orientation: MULTIPLEXED=ch1,pt1, ch2,pt1 ...\r\n");
    		fprintf(fid,"DataOrientation=MULTIPLEXED\r\n");
    		hdr->NRec *= hdr->SPR;
		hdr->SPR = 1;
    		fprintf(fid,"NumberOfChannels=%i\r\n",hdr->NS);
    		fprintf(fid,"; Sampling interval in microseconds\r\n");
    		fprintf(fid,"SamplingInterval=%f\r\n\r\n",1e6/hdr->SampleRate);

		if (VERBOSE_LEVEL>8) fprintf(stdout,"BVA-write: [212]\n");

    		fprintf(fid,"[Binary Infos]\r\nBinaryFormat=");
		uint16_t gdftyp = 0;
		typeof(hdr->NS) k;
    		for (k=0; k<hdr->NS; k++)
    			if (gdftyp < hdr->CHANNEL[k].GDFTYP)
    				gdftyp = hdr->CHANNEL[k].GDFTYP;
    		if (gdftyp<4) {
    			gdftyp = 3;
    			fprintf(fid,"INT_16");
    		}
    		else {
    			gdftyp = 16;
 	   		fprintf(fid,"IEEE_FLOAT_32");
		}

		if (VERBOSE_LEVEL>8) fprintf(stdout,"BVA-write: [214] gdftyp=%i NS=%i\n",gdftyp,hdr->NS);

		hdr->AS.bpb = (size_t)hdr->NS * hdr->SPR * GDFTYP_BITS[gdftyp] >> 3;

    		fprintf(fid,"\r\n\r\n[Channel Infos]\r\n");
    		fprintf(fid,"; Each entry: Ch<Channel number>=<Name>,<Reference channel name>,\r\n");
    		fprintf(fid,"; <Resolution in \"Unit\">,<Unit>,,<Future extensions..\r\n");
    		fprintf(fid,"; Fields are delimited by commas, some fields might be omitted (empty).\r\n");
    		fprintf(fid,"; Commas in channel names are coded as \"\\1\".\r\n");
    		for (k=0; k<hdr->NS; k++) {

			if (VERBOSE_LEVEL>8) fprintf(stdout,"BVA-write: [220] %i\n",k);

			hdr->CHANNEL[k].SPR = hdr->SPR;
			hdr->CHANNEL[k].GDFTYP = gdftyp;
    			char Label[MAX_LENGTH_LABEL+1];
			strcpy(Label,hdr->CHANNEL[k].Label);			// Flawfinder: ignore
    			size_t k1;
    			for (k1=0; Label[k1]; k1++) if (Label[k1]==',') Label[k1]=1;
	    		fprintf(fid,"Ch%d=%s,,1,%s\r\n",k+1,Label,PhysDim3(hdr->CHANNEL[k].PhysDimCode));
    		}
    		fprintf(fid,"\r\n\r\n[Coordinates]\r\n");
    		// fprintf(fid,"; Each entry: Ch<Channel number>=<Radius>,<Theta>,<Phi>\n\r");
    		fprintf(fid,"; Each entry: Ch<Channel number>=<X>,<Y>,<Z>\r\n");
    		for (k=0; k<hdr->NS; k++)
	    		fprintf(fid,"Ch%i=%f,%f,%f\r\n",k+1,hdr->CHANNEL[k].XYZ[0],hdr->CHANNEL[k].XYZ[1],hdr->CHANNEL[k].XYZ[2]);

		if (VERBOSE_LEVEL>8) fprintf(stdout,"BVA-write: [222]\n");

    		fprintf(fid,"\r\n\r\n[Comment]\r\n\r\n");

		fprintf(fid,"A m p l i f i e r  S e t u p\r\n");
		fprintf(fid,"============================\r\n");
		fprintf(fid,"Number of channels: %i\r\n",hdr->NS);
		fprintf(fid,"Sampling Rate [Hz]: %f\r\n",hdr->SampleRate);
		fprintf(fid,"Sampling Interval [µS]: %f\r\n",1e6/hdr->SampleRate);
		fprintf(fid,"Channels\r\n--------\r\n");
		fprintf(fid,"#     Name      Phys. Chn.    Resolution [µV]  Low Cutoff [s]   High Cutoff [Hz]   Notch [Hz]\n\r");
    		for (k=0; k<hdr->NS; k++) {
			fprintf(fid,"\r\n%6i %13s %17i %18f",k+1,hdr->CHANNEL[k].Label,k+1,hdr->CHANNEL[k].Cal);

			if (hdr->CHANNEL[k].HighPass>0)
				fprintf(fid," %15f",1/(2*3.141592653589793238462643383279502884197169399375*hdr->CHANNEL[k].HighPass));
			else
				fprintf(fid,"\t-");

			if (hdr->CHANNEL[k].LowPass>0)
				fprintf(fid," %15f",hdr->CHANNEL[k].LowPass);
			else
				fprintf(fid,"\t-");

			if (hdr->CHANNEL[k].Notch>0)
				fprintf(fid," %f",hdr->CHANNEL[k].Notch);
			else
				fprintf(fid,"\t-");
		}

    		fprintf(fid,"\r\n\r\nImpedance [kOhm] :\r\n\r\n");
    		for (k=0; k<hdr->NS; k++)
    		if (isnan(hdr->CHANNEL[k].Impedance))
			fprintf(fid,"%s:\t\t-\r\n",hdr->CHANNEL[k].Label);
		else
			fprintf(fid,"%s:\t\t%f\r\n",hdr->CHANNEL[k].Label,hdr->CHANNEL[k].Impedance);


		fclose(fid);

		strcpy(strrchr(tmpfile,'.')+1,"vmrk");
		fid = fopen(tmpfile,"wb");
    		fprintf(fid,"Brain Vision Data Exchange Marker File, Version 1.0\r\n");
    		fprintf(fid,"; Data created by BioSig4C++\r\n\r\n");
    		fprintf(fid,"[Common Infos]\r\n");
    		fprintf(fid,"DataFile=%s\r\n\r\n",hdr->FileName);
    		fprintf(fid,"[Marker Infos]\r\n\r\n");
    		fprintf(fid,"; Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>,\r\n");
    		fprintf(fid,"; <Size in data points>, <Channel number (0 = marker is related to all channels)>\r\n");
    		fprintf(fid,"; Fields are delimited by commas, some fields might be omitted (empty).\r\n");
    		fprintf(fid,"; Commas in type or description text are coded as \"\\1\".\r\n");
    		struct tm *T0 = gdf_time2tm_time(hdr->T0);
		uint32_t us = (hdr->T0*24*3600 - floor(hdr->T0*24*3600))*1e6;
    		fprintf(fid,"Mk1=New Segment,,1,1,0,%04u%02u%02u%02u%02u%02u%06u",T0->tm_year+1900,T0->tm_mon+1,T0->tm_mday,T0->tm_hour,T0->tm_min,T0->tm_sec,us); // 20081002150147124211

		if ((hdr->EVENT.DUR==NULL) && (hdr->EVENT.CHN==NULL))
	    		for (k=0; k<hdr->EVENT.N; k++) {
				fprintf(fid,"\r\nMk%i=,0x%04x,%u,1,0",k+2,hdr->EVENT.TYP[k],hdr->EVENT.POS[k]+1);  // convert to 1-based indexing
   			}
    		else
    			for (k=0; k<hdr->EVENT.N; k++) {
				fprintf(fid,"\r\nMk%i=,0x%04x,%u,%u,%u",k+2,hdr->EVENT.TYP[k],hdr->EVENT.POS[k]+1,hdr->EVENT.DUR[k],hdr->EVENT.CHN[k]); // convert EVENT.POS to 1-based indexing
	   		}
		fclose(fid);

		free(tmpfile);

		if (VERBOSE_LEVEL>8) fprintf(stdout,"BVA-write: [290] %s %s\n",tmpfile,hdr->FileName);
    	}

    	else if (hdr->TYPE==CFWB) {
	     	hdr->HeadLen = 68 + NS*96;
	    	hdr->AS.Header = (uint8_t*)malloc(hdr->HeadLen);
	    	uint8_t* Header2 = hdr->AS.Header+68;
		memset(hdr->AS.Header,0,hdr->HeadLen);
	    	memcpy(hdr->AS.Header,"CFWB\1\0\0\0",8);
		lef64a(1/hdr->SampleRate, hdr->AS.Header+8);

		lef64a(0.0,  hdr->AS.Header+44);	// pretrigger time
	    	leu32a(NS, hdr->AS.Header+52);
	    	hdr->NRec *= hdr->SPR; hdr->SPR = 1;
	    	leu32a(hdr->NRec, hdr->AS.Header+56); // number of samples
	    	lei32a(0, hdr->AS.Header+60);	// 1: time channel

	    	int32_t gdftyp = 3; // 1:double, 2:float, 3: int16; see CFWB_GDFTYP too.
		typeof(hdr->NS) k,k2;
		for (k=0; k<hdr->NS; k++)
		if (hdr->CHANNEL[k].OnOff)
		{
			/* if int16 is not sufficient, use float or double */
			if (hdr->CHANNEL[k].GDFTYP>16)
				gdftyp = min(gdftyp,1);	// double
			else if (hdr->CHANNEL[k].GDFTYP>3)
				gdftyp = min(gdftyp,2);	// float
		}
		lei32a(gdftyp, hdr->AS.Header+64);	// 1: double, 2: float, 3:short

		for (k=0,k2=0; k<hdr->NS; k++)
		if (hdr->CHANNEL[k].OnOff)
		{
	    		hdr->CHANNEL[k].SPR = 1;
			hdr->CHANNEL[k].GDFTYP = CFWB_GDFTYP[gdftyp-1];
			const char *tmpstr;
			if (hdr->CHANNEL[k].LeadIdCode)
				tmpstr = LEAD_ID_TABLE[hdr->CHANNEL[k].LeadIdCode];
			else
				tmpstr = hdr->CHANNEL[k].Label;
		     	size_t len = strlen(tmpstr);
		     	memcpy(Header2+96*k2, tmpstr, min(len,32));

		     	tmpstr = PhysDim3(hdr->CHANNEL[k].PhysDimCode);
			if (tmpstr != NULL) {
			     	len = strlen(tmpstr)+1;
			     	memcpy(Header2+96*k2+32, tmpstr, min(len,32));
			}

			lef64a(hdr->CHANNEL[k].Cal, Header2+96*k2+64);
			lef64a(hdr->CHANNEL[k].Off, Header2+96*k2+72);
			lef64a(hdr->CHANNEL[k].PhysMax, Header2+96*k2+80);
			lef64a(hdr->CHANNEL[k].PhysMin, Header2+96*k2+88);
			k2++;
		}
	}

	else

#endif //ONLYGDF

	      if ((hdr->TYPE==GDF) || (hdr->TYPE==GDF1)) {
		/* use of GDF1 is deprecated, instead hdr->TYPE=GDF and hdr->VERSION=1.25 should be used.
                   a test and a warning is about this is implemented within struct2gdfbin
		*/
		struct2gdfbin(hdr);

		size_t bpb8 = 0;
		typeof(hdr->NS) k;
		for (k=0, hdr->AS.bpb=0; k<hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			hc->bi8 = bpb8;
			hc->bi  = bpb8>>3;
			if (hc->OnOff)
				bpb8 += (GDFTYP_BITS[hc->GDFTYP] * hc->SPR);
		}
		hdr->AS.bpb8 = bpb8;
		hdr->AS.bpb  = bpb8>>3;
		if (bpb8 & 0x07) {		// each block must use whole number of bytes
			hdr->AS.bpb++;
			hdr->AS.bpb8 = hdr->AS.bpb<<3;
		}

		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"GDFw h3\n");

	}

#ifndef  ONLYGDF

    	else if ((hdr->TYPE==EDF) || (hdr->TYPE==BDF)) {
	     	hdr->HeadLen   = (NS+1)*256;
	    	hdr->AS.Header = (uint8_t*)malloc(hdr->HeadLen);
	    	char* Header2  = (char*)hdr->AS.Header+256;
		memset(Header1,' ',hdr->HeadLen);
		if (hdr->TYPE==BDF) {
			Header1[0] = 255;
	     		memcpy(Header1+1,"BIOSEMI",7);
		}
		else {
			Header1[0] = '0';
	     	}

		char tmp[81];
		if (hdr->Patient.Birthday>1)
			strfgdftime(tmp,81,"%02d-%b-%04Y",hdr->Patient.Birthday);
		else strcpy(tmp,"X");

		if (strlen(hdr->Patient.Id) > 0) {
			size_t k;
			for (k=0; hdr->Patient.Id[k]; k++)
				if (isspace(hdr->Patient.Id[k]))
					hdr->Patient.Id[k] = '_';
		}

	    	char cmd[256];
		if (!hdr->FLAG.ANONYMOUS)
			snprintf(cmd,MAX_LENGTH_PID+1,"%s %c %s %s",hdr->Patient.Id,GENDER[hdr->Patient.Sex],tmp,hdr->Patient.Name);
		else
			snprintf(cmd,MAX_LENGTH_PID+1,"%s %c %s X",hdr->Patient.Id,GENDER[hdr->Patient.Sex],tmp);

	     	memcpy(Header1+8, cmd, strlen(cmd));

		if (hdr->T0 > 1)
			strfgdftime(tmp,81,"%d-%b-%Y", hdr->T0);
		else strcpy(tmp,"X");

		char *tmpstr = hdr->ID.Technician;
		if (!tmpstr || !strlen(tmp)) tmpstr = "X";
		size_t len = snprintf(cmd,sizeof(cmd),"Startdate %s X %s ", tmp, tmpstr);
	     	memcpy(Header1+88, cmd, len);
	     	memcpy(Header1+88+len, &hdr->ID.Equipment, 8);

		strfgdftime(tmp,81,"%d.%m.%y%H.%M.%S",hdr->T0);
	     	memcpy(Header1+168, tmp, 16);

		len = sprintf(tmp,"%i",hdr->HeadLen);
		if (len>8) fprintf(stderr,"Warning: HeaderLength is (%s) too long (%i>8).\n",tmp,(int)len);
	     	memcpy(Header1+184, tmp, len);
	     	memcpy(Header1+192, "EDF+C  ", 5);

		len = sprintf(tmp,"%u",(int)hdr->NRec);
		if (len>8) fprintf(stderr,"Warning: NRec is (%s) too long (%i>8).\n",tmp,(int)len);
	     	memcpy(Header1+236, tmp, len);

		len = sprintf(tmp,"%f",hdr->SPR/hdr->SampleRate);
		if (len>8) fprintf(stderr,"Warning: Duration is (%s) too long (%i>8).\n",tmp,(int)len);
	     	memcpy(Header1+244, tmp, len);

		len = sprintf(tmp,"%i",NS);
		if (len>4) fprintf(stderr,"Warning: NS is (%s) too long (%i>4).\n",tmp,(int)len);
	     	memcpy(Header1+252, tmp, len);

		typeof(hdr->NS) k,k2;
		for (k=0,k2=0; k<hdr->NS; k++)
		if (hdr->CHANNEL[k].OnOff)
		{
			const char *tmpstr;
			if (hdr->CHANNEL[k].LeadIdCode)
				tmpstr = LEAD_ID_TABLE[hdr->CHANNEL[k].LeadIdCode];
			else
				tmpstr = hdr->CHANNEL[k].Label;
		     	len = strlen(tmpstr);
			if (len>16)
			//fprintf(stderr,"Warning: Label (%s) of channel %i is to long.\n",hdr->CHANNEL[k].Label,k);
		     	fprintf(stderr,"Warning: Label of channel %i,%i is too long (%i>16).\n",k,k2, (int)len);
		     	memcpy(Header2+16*k2,tmpstr,min(len,16));

		     	len = strlen(hdr->CHANNEL[k].Transducer);
			if (len>80) fprintf(stderr,"Warning: Transducer of channel %i,%i is too long (%i>80).\n",k,k2, (int)len);
		     	memcpy(Header2+80*k2 + 16*NS,hdr->CHANNEL[k].Transducer,min(len,80));

		     	tmpstr = PhysDim3(hdr->CHANNEL[k].PhysDimCode);
		     	if (tmpstr) {
				len = strlen(tmpstr);
			     	if (len>8) fprintf(stderr,"Warning: Physical Dimension (%s) of channel %i is too long (%i>8).\n",tmpstr,k,(int)len);
			     	memcpy(Header2 + 8*k2 + 96*NS, tmpstr, min(len,8));
			}

			if (ftoa8(tmp,hdr->CHANNEL[k].PhysMin))
				fprintf(stderr,"Warning: PhysMin (%f)(%s) of channel %i does not fit into 8 bytes of EDF header.\n",hdr->CHANNEL[k].PhysMin,tmp,k);
		     	memcpy(Header2 + 8*k2 + 104*NS, tmp, strlen(tmp));
			if (ftoa8(tmp,hdr->CHANNEL[k].PhysMax))
				fprintf(stderr,"Warning: PhysMax (%f)(%s) of channel %i does not fit into 8 bytes of EDF header.\n",hdr->CHANNEL[k].PhysMax,tmp,k);
		     	memcpy(Header2 + 8*k2 + 112*NS, tmp, strlen(tmp));
			if (ftoa8(tmp,hdr->CHANNEL[k].DigMin))
				fprintf(stderr,"Warning: DigMin (%f)(%s) of channel %i does not fit into 8 bytes of EDF header.\n",hdr->CHANNEL[k].DigMin,tmp,k);
		     	memcpy(Header2 + 8*k2 + 120*NS, tmp, strlen(tmp));
			if (ftoa8(tmp,hdr->CHANNEL[k].DigMax))
				fprintf(stderr,"Warning: DigMax (%f)(%s) of channel %i does not fit into 8 bytes of EDF header.\n",hdr->CHANNEL[k].DigMax,tmp,k);
		     	memcpy(Header2 + 8*k2 + 128*NS, tmp, strlen(tmp));

			if (hdr->CHANNEL[k].Notch>0)
				len = sprintf(tmp,"HP:%fHz LP:%fHz Notch:%fHz",hdr->CHANNEL[k].HighPass,hdr->CHANNEL[k].LowPass,hdr->CHANNEL[k].Notch);
			else
				len = sprintf(tmp,"HP:%fHz LP:%fHz",hdr->CHANNEL[k].HighPass,hdr->CHANNEL[k].LowPass);
		     	memcpy(Header2+ 80*k2 + 136*NS,tmp,min(80,len));

			len = sprintf(tmp,"%i",hdr->CHANNEL[k].SPR);
			if (len>8) fprintf(stderr,"Warning: SPR (%s) of channel %i is to long (%i)>8.\n",tmp,k,(int)len);
		     	memcpy(Header2+ 8*k2 + 216*NS,tmp,min(8,len));
		     	hdr->CHANNEL[k].GDFTYP = ( (hdr->TYPE != BDF) ? 3 : 255+24);
		     	k2++;
		}
	}

    	else if (hdr->TYPE==HL7aECG) {
		sopen_HL7aECG_write(hdr);

		// hdr->FLAG.SWAP = 0;
		hdr->FILE.LittleEndian = (__BYTE_ORDER == __LITTLE_ENDIAN); // no byte-swapping
	}

    	else if (hdr->TYPE==MFER) {
    		uint8_t tag;
    		size_t  len, curPos=0;
	     	hdr->HeadLen   = 32+128+3*6+3 +80000;
	    	hdr->AS.Header = (uint8_t*)malloc(hdr->HeadLen);
		memset(Header1, ' ', hdr->HeadLen);

		hdr->FILE.LittleEndian = 0;

		fprintf(stderr,"Warning SOPEN(MFER): write support for MFER format under construction\n");
		/* FIXME & TODO:
		   known issues:
			Label
			Sampling Rate
			HeadLen
			Encoding of data block
		*/

		// tag 64: preamble
		// Header1[curPos] = 64;
		// len =32;
		curPos = 34;
		strncpy(Header1,"@  MFER                                ",curPos);
		// Header1[curPos+1] = len;
		// curPos = len+2;

		if (VERBOSE_LEVEL>8) fprintf(stdout,"[MFER 711]:\n");

		// tag 23: Manufacturer
		tag = 23;
		Header1[curPos] = tag;
		{
			char *str = Header1+curPos+2;
			strncpy(str, hdr->ID.Manufacturer.Name, MAX_LENGTH_MANUF);
			size_t l2,l1 = strlen(str);

			l2 = (hdr->ID.Manufacturer.Model==NULL) ? MAX_LENGTH_MANUF*2 : strlen(hdr->ID.Manufacturer.Model);
			str[l1++]='^';
			if (l1+l2 <= MAX_LENGTH_MANUF) {
				memcpy(str+l1, hdr->ID.Manufacturer.Model, l2);
				l1 += l2;
			}

			l2 = (hdr->ID.Manufacturer.Version==NULL) ? MAX_LENGTH_MANUF*2 : strlen(hdr->ID.Manufacturer.Version);
			str[l1++]='^';
			if (l1+l2 <= MAX_LENGTH_MANUF) {
				memcpy(str+l1, hdr->ID.Manufacturer.Version, l2);
				l1 += l2;
			}

			l2 = (hdr->ID.Manufacturer.SerialNumber==NULL) ? MAX_LENGTH_MANUF*2 : strlen(hdr->ID.Manufacturer.SerialNumber);
			str[l1++]='^';
			if (l1+l2 <= MAX_LENGTH_MANUF) {
				memcpy(str+l1, hdr->ID.Manufacturer.SerialNumber, l2);
				l1 += l2;
			}
			len = min(l1, MAX_LENGTH_MANUF);
			str[len]=0;
		}
		Header1[curPos] = tag;
		if (len<128) {
			hdr->AS.Header[curPos+1] = len;
			curPos += len+2;
		} else
			fprintf(stderr,"Warning MFER(W) Tag23 (manufacturer) too long len=%i>128\n",(int)len);

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"Write MFER: tag=%i,len%i,curPos=%i\n",tag,(int)len,(int)curPos);

		// tag 1: Endianity
		// use default BigEndianity

		if (VERBOSE_LEVEL>8) fprintf(stdout,"[MFER 720-4]:\n");

		// tag 4: SPR
		tag = 4;
		len = sizeof(uint32_t);
		Header1[curPos++] = tag;
		Header1[curPos++] = len;
		beu32a(hdr->SPR, hdr->AS.Header+curPos);
		curPos += len;

		if (VERBOSE_LEVEL>8) fprintf(stdout,"[MFER 720-5]:\n");

		// tag 5: NS
		tag = 5;
		len = sizeof(uint16_t);
		Header1[curPos++] = tag;
		Header1[curPos++] = len;
		beu16a(hdr->NS, hdr->AS.Header+curPos);
		curPos += len;

		if (VERBOSE_LEVEL>8) fprintf(stdout,"[MFER 720-6]:\n");

		// tag 6: NRec
		tag = 6;
		len = sizeof(uint32_t);
		Header1[curPos++] = tag;
		Header1[curPos++] = len;
		beu32a(hdr->NRec, hdr->AS.Header+curPos);
		curPos += len;

		if (VERBOSE_LEVEL>8) fprintf(stdout,"[MFER 720-8]:\n");

		// tag 8: Waveform: unidentified
		tag = 8;
		len = sizeof(uint8_t);
		Header1[curPos++] = tag;
		Header1[curPos++] = len;
		*(Header1+curPos) = 0; // unidentified
		curPos += len;

		if (VERBOSE_LEVEL>8) fprintf(stdout,"[MFER 720-129]:\n");

		// tag 129: Patient Name
		if (!hdr->FLAG.ANONYMOUS) {
			tag = 129;
			len = strlen(hdr->Patient.Name);
			Header1[curPos++] = tag;
			Header1[curPos++] = len;
			strcpy(Header1+curPos,hdr->Patient.Name);
			curPos += len;
		}

		// tag 130: Patient Id
		tag = 130;
		len = strlen(hdr->Patient.Id);
		Header1[curPos++] = tag;
		Header1[curPos++] = len;
		strcpy(Header1+curPos,hdr->Patient.Id);
		curPos += len;

		// tag 131: Patient Age
		if (hdr->Patient.Birthday>0) {
			tag = 131;
			len = 7;
			struct tm *t = gdf_time2tm_time(hdr->Patient.Birthday);
			hdr->AS.Header[curPos++] = tag;
			hdr->AS.Header[curPos++] = len;
			hdr->AS.Header[curPos] = (uint8_t)((hdr->T0 - hdr->Patient.Birthday)/365.25);
			double tmpf64 = (hdr->T0 - hdr->Patient.Birthday);
			tmpf64 -= 365.25*floor(tmpf64/365.25);
			beu16a((uint16_t)tmpf64, Header1+curPos+1);
			beu16a(t->tm_year+1900, Header1+curPos+3);
			hdr->AS.Header[curPos+5] = (t->tm_mon+1);
			hdr->AS.Header[curPos+6] = (t->tm_mday);
			curPos += len;
		}

		if (VERBOSE_LEVEL>8) fprintf(stdout,"[MFER 720-132]:\n");

		// tag 132: Patient Sex
		tag = 132;
		Header1[curPos]   = tag;
		Header1[curPos+1] = 1;
		Header1[curPos+2] = hdr->Patient.Sex;
		curPos += 3;

		// tag 133: Recording time
		tag = 133;
		len = 11;
		{
			struct tm *t = gdf_time2tm_time(hdr->T0);
			hdr->AS.Header[curPos++] = tag;
			hdr->AS.Header[curPos++] = len;
			beu16a(t->tm_year+1900, hdr->AS.Header+curPos);
			hdr->AS.Header[curPos+2] = (uint8_t)(t->tm_mon+1);
			hdr->AS.Header[curPos+3] = (uint8_t)(t->tm_mday);
			hdr->AS.Header[curPos+4] = (uint8_t)(t->tm_hour);
			hdr->AS.Header[curPos+5] = (uint8_t)(t->tm_min);
			hdr->AS.Header[curPos+6] = (uint8_t)(t->tm_sec);
			memset(hdr->AS.Header+curPos+7, 0, 4);
			curPos += len;
		}


		// tag  9: LeadId
		// tag 10: gdftyp
		// tag 11: SampleRate
		// tag 12: Cal
		// tag 13: Off
		hdr->HeadLen = curPos;
		// tag 63: channel-specific settings

		if (VERBOSE_LEVEL>8) fprintf(stdout,"[MFER 720-63]:\n");

		tag = 63;
		size_t ch;
		for (ch=0; ch<hdr->NS; ch++) {

		if (VERBOSE_LEVEL>8) fprintf(stdout,"[MFER 720-63 #%i/%i %i]:\n",(int)ch,hdr->NS,hdr->CHANNEL[ch].LeadIdCode);

		 	// FIXME: this is broken
			len = 0;
			Header1[curPos++] = tag;
			if (ch<128)
				Header1[curPos++] = ch;
			else {
				Header1[curPos++] = (ch >> 7) | 0x80;
				Header1[curPos++] = (ch && 0x7f);
			}
			// tag1  9: LeadId
			size_t ix = curPos;
			size_t len1 = 0;
			Header1[ix++] = 9;
			if (hdr->CHANNEL[ch].LeadIdCode>0) {
				hdr->AS.Header[ix++] = 2;
				leu16a(hdr->CHANNEL[ch].LeadIdCode, hdr->AS.Header+ix);
				len1 = 2;

			} else {
				len1 = strlen(hdr->CHANNEL[ch].Label);
				Header1[ix++] = len1;
				strcpy(Header1+ix, hdr->CHANNEL[ch].Label);
			}
			// tag1 10: gdftyp
			// tag1 11: SampleRate
			// tag1 12: Cal
			// tag1 13: Off

			len += len1+ix-curPos;
			hdr->AS.Header[curPos] = len;
			curPos += len+curPos;
		}
		// tag 30: data
	}

    	else if (hdr->TYPE==SCP_ECG) {
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s(line %i) %s -> SOPEN_SCP_WRITE v%f\n",__FILE__,__LINE__,__func__,hdr->VERSION);
    		sopen_SCP_write(hdr);
    		if (hdr->AS.B4C_ERRNUM) return(hdr);
	}

#ifdef WITH_TMSiLOG
    	else if (hdr->TYPE==TMSiLOG) {
    		// ###FIXME: writing of TMSi-LOG file is experimental and not completed
	    	FILE *fid = fopen(hdr->FileName,"wb");
		fprintf(fid,"FileId=TMSi PortiLab sample log file\n\rVersion=0001\n\r",NULL);
		struct tm *t = gdf_time2tm_time(hdr->T0);
		fprintf(fid,"DateTime=%04d/02d/02d-02d:02d:02d\n\r",t->tm_year+1900,t->tm_mon+1,t->tm_mday,t->tm_hour,t->tm_min,t->tm_sec);
		fprintf(fid,"Format=Float32\n\rLength=%f\n\rSignals=%04i\n\r",hdr->NRec*hdr->SPR/hdr->SampleRate,hdr->NS);
		const char* fn = strrchr(hdr->FileName,FILESEP);
		if (!fn) fn=hdr->FileName;
		size_t len = strcspn(fn,".");
		char* fn2 = (char*)malloc(len+1);
		strncpy(fn2,fn,len);
		fn2[len]=0;
		for (k=0; k<hdr->NS; k++) {
			fprintf(fid,"Signal%04d.Name=%s\n\r",k+1,hdr->CHANNEL[k].Label);
			fprintf(fid,"Signal%04d.UnitName=%s\n\r",k+1,PhysDim3(hdr->CHANNEL[k].PhysDimCode));
			fprintf(fid,"Signal%04d.Resolution=%f\n\r",k+1,hdr->CHANNEL[k].Cal);
			fprintf(fid,"Signal%04d.StoreRate=%f\n\r",k+1,hdr->SampleRate);
			fprintf(fid,"Signal%04d.File=%s.asc\n\r",k+1,fn2);
			fprintf(fid,"Signal%04d.Index=%04d\n\r",k+1,k+1);
		}
		fprintf(fid,"\n\r\n\r");
		fclose(fid);

		// ###FIXME: this belongs into SWRITE
		// write data file
		fn2 = (char*) realloc(fn2, strlen(hdr->FileName)+5);
		strcpy(fn2,hdr->FileName);		// Flawfinder: ignore
		strcpy(strrchr(fn2,'.'),".asc");	// Flawfinder: ignore
    		// hdr->FileName = fn2;
	    	fid = fopen(fn2,"wb");
	    	fprintf(fid,"%d\tHz\n\r\n\rN",hdr->SampleRate);
		for (k=0; k<hdr->NS; k++) {
			fprintf(fid,"\t%s(%s)", hdr->CHANNEL[k].Label, PhysDim3(hdr->CHANNEL[k].PhysDimCode));
		}
		for (k1=0; k1<hdr->SPR*hdr->NRec; k1++) {
			fprintf(fid,"\n%i",k1);
			for (k=0; k<hdr->NS; k++) {
				// TODO: Row/Column ordering
				fprintf(fid,"\t%f",hdr->data.block[]);
			}
		}

		fclose(fid);
		free(fn2);
	}
#endif  // WITH_TMSiLOG

#endif //ONLYGDF


	else {
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Writing of format not supported");
		return(NULL);
	}

	if ((hdr->TYPE != ASCII) && (hdr->TYPE != ATF) && (hdr->TYPE != BIN) && (hdr->TYPE != HL7aECG) && (hdr->TYPE != TMSiLOG)){
	    	hdr = ifopen(hdr,"wb");

		if (!hdr->FILE.COMPRESSION && (hdr->FILE.FID == NULL) ) {
			biosigERROR(hdr, B4C_CANNOT_WRITE_FILE, "Unable to open file for writing");
			return(NULL);
		}
#ifdef ZLIB_H
		else if (hdr->FILE.COMPRESSION && (hdr->FILE.gzFID == NULL) ){
			biosigERROR(hdr, B4C_CANNOT_WRITE_FILE, "Unable to open file for writing");
			return(NULL);
		}
#endif
		if(hdr->TYPE != SCP_ECG){
			ifwrite(Header1, sizeof(char), hdr->HeadLen, hdr);
		}

		hdr->FILE.OPEN = 2;
		hdr->FILE.POS  = 0;
	}

	size_t bpb8 = 0;

#ifndef  ONLYGDF
	if (hdr->TYPE==AINF) {
		hdr->AS.bpb = 4;
		bpb8 = 32;
	}
	else
#endif //ONLYGDF
		hdr->AS.bpb = 0;

	typeof(hdr->NS) k;
	for (k=0, hdr->SPR = 1; k < hdr->NS; k++) {
		hdr->CHANNEL[k].bi  = bpb8>>3;
		hdr->CHANNEL[k].bi8 = bpb8;
		if (hdr->CHANNEL[k].OnOff) {
			bpb8 += (GDFTYP_BITS[hdr->CHANNEL[k].GDFTYP] * hdr->CHANNEL[k].SPR);
			if (hdr->CHANNEL[k].SPR > 0)  // ignore sparse channels
				hdr->SPR = lcm(hdr->SPR, hdr->CHANNEL[k].SPR);
		}
	}
	hdr->AS.bpb8 = bpb8;
	hdr->AS.bpb  = bpb8>>3;
	if ((hdr->TYPE==GDF) && (bpb8 & 0x07)) {
		// each block must use whole number of bytes
		hdr->AS.bpb++;
		hdr->AS.bpb8 = hdr->AS.bpb<<3;
	}

}	// end of branch "write"


#ifndef ANDROID
    if (VERBOSE_LEVEL > 7) {
	 //There is a way to send messages in Android to log, but I dont know it yet. Stoyan
 	//There is problem with some files printing deubg info.
 	//And debug in NDK is bad idea in Android
	if (hdr->FILE.POS != 0)
		fprintf(stdout,"Debugging Information: (Format=%d) %s FILE.POS=%d is not zero.\n",hdr->TYPE,hdr->FileName,(int)hdr->FILE.POS);

	typeof(hdr->NS) k;
	for (k=0; k<hdr->NS; k++)
	if  (GDFTYP_BITS[hdr->CHANNEL[k].GDFTYP] % 8) {

	if (hdr->TYPE==alpha)
		; // 12bit alpha is well tested
	else if  ((__BYTE_ORDER == __LITTLE_ENDIAN) && !hdr->FILE.LittleEndian)
			fprintf(stdout,"GDFTYP=%i [12bit LE/BE] not well tested\n",hdr->CHANNEL[k].GDFTYP);
	else if  ((__BYTE_ORDER == __LITTLE_ENDIAN) && hdr->FILE.LittleEndian)
			fprintf(stdout,"GDFTYP=%i [12bit LE/LE] not well tested\n",hdr->CHANNEL[k].GDFTYP);
	else if  ((__BYTE_ORDER == __BIG_ENDIAN) && hdr->FILE.LittleEndian)
			fprintf(stdout,"GDFTYP=%i [12bit BE/LE] not well tested\n",hdr->CHANNEL[k].GDFTYP);
	else if  ((__BYTE_ORDER == __BIG_ENDIAN) && !hdr->FILE.LittleEndian)
			fprintf(stdout,"GDFTYP=%i [12bit BE/BE] not well tested\n",hdr->CHANNEL[k].GDFTYP);
	}
    }
	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"sopen{return} %i %s\n", hdr->AS.B4C_ERRNUM,GetFileTypeString(hdr->TYPE) );
#endif
	return(hdr);
}  // end of SOPEN



/****************************************************************************
	bpb8_collapsed_rawdata
	 computes the bytes per block when rawdata is collapsed
 ****************************************************************************/
size_t bpb8_collapsed_rawdata(HDRTYPE *hdr)
{
	size_t bpb8=0;
	CHANNEL_TYPE *CHptr;
	typeof(hdr->NS) k;
	for (k=0; k<hdr->NS; k++) {
		CHptr 	= hdr->CHANNEL+k;
		if (CHptr->OnOff) bpb8 += (size_t)CHptr->SPR*GDFTYP_BITS[CHptr->GDFTYP];
	}
	return(bpb8);
}

/* ***************************************************************************
   collapse raw data
	this function is used to remove obsolete channels (e.g.
	status and annotation channels because the information
	as been already converted into the event table)
	that are not needed in GDF.

	if buf==NULL, hdr->AS.rawdata will be collapsed

 ****************************************************************************/
void collapse_rawdata(HDRTYPE *hdr, void *buf, size_t count) {

	CHANNEL_TYPE *CHptr;
	size_t bpb, k4;
	size_t bi1=0, bi2=0, SZ;
	int num3Segments=0,k1;

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): started\n",__func__,__LINE__);

	bpb = bpb8_collapsed_rawdata(hdr);
	if (bpb == hdr->AS.bpb<<3) return; 	// no collapsing needed

	if ((bpb & 7) || (hdr->AS.bpb8 & 7)) {
		biosigERROR(hdr, B4C_RAWDATA_COLLAPSING_FAILED, "collapse_rawdata: does not support bitfields");
	}
	bpb >>= 3;

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): bpb=%i/%i\n",__func__,__LINE__,(int)bpb,hdr->AS.bpb);

	if (buf == NULL) {
		buf   = hdr->AS.rawdata;
		count = hdr->AS.length;
	}

	// prepare idxlist for copying segments within a single block (i.e. record)
	size_t *idxList= alloca(3*hdr->NS*sizeof(size_t));
	CHptr = hdr->CHANNEL;
	while (1) {
		SZ = 0;
		while (!CHptr->OnOff && (CHptr < hdr->CHANNEL+hdr->NS) ) {
			SZ += (size_t)CHptr->SPR * GDFTYP_BITS[CHptr->GDFTYP];
			if (SZ & 7) biosigERROR(hdr, B4C_RAWDATA_COLLAPSING_FAILED, "collapse_rawdata: does not support bitfields");
			CHptr++;
		}
		bi1 += SZ;

		SZ = 0;
		while (CHptr->OnOff && (CHptr < hdr->CHANNEL+hdr->NS)) {
			SZ += (size_t)CHptr->SPR * GDFTYP_BITS[CHptr->GDFTYP];
			if (SZ & 7) biosigERROR(hdr, B4C_RAWDATA_COLLAPSING_FAILED, "collapse_rawdata: does not support bitfields");
			CHptr++;
		}

		if (SZ > 0) {
			SZ >>= 3;
			idxList[num3Segments]   = bi2;	// offset of destination
			idxList[num3Segments+1] = bi1;	// offset of source
			idxList[num3Segments+2] = SZ;	// size
			num3Segments           += 3;
	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): #%i  src:%i dest:%i size:%i\n",__func__,__LINE__,num3Segments/3,(int)bi1,(int)bi2,(int)SZ);
		}

	if (CHptr >= hdr->CHANNEL+hdr->NS) break;
		bi1 += SZ;
		bi2 += SZ;
	}

	for (k4 = 0; k4 < count; k4++) {
		void *src  = buf + k4*hdr->AS.bpb;
		void *dest = buf + k4*bpb;
		for (k1=0; k1 < num3Segments; k1+=3)
			if ((dest + idxList[k1]) != (src + idxList[k1+1]))
				memcpy(dest + idxList[k1], src + idxList[k1+1], idxList[k1+2]);
	}

	if (buf == hdr->AS.rawdata) {
		hdr->AS.flag_collapsed_rawdata = 1;	// rawdata is now "collapsed"
	}
}

/****************************************************************************/
/**	SREAD_RAW : segment-based                                          **/
/****************************************************************************/
size_t sread_raw(size_t start, size_t length, HDRTYPE* hdr, char flag, void *buf, size_t bufsize) {
/*
 *	Reads LENGTH blocks with HDR.AS.bpb BYTES each
 * 	(and HDR.SPR samples).
 *	Rawdata is available in hdr->AS.rawdata.
 *
 *        start <0: read from current position
 *             >=0: start reading from position start
 *        length  : try to read length blocks
 *
 *	  flag!=0 : unused channels (those channels k where HDR.CHANNEL[k].OnOff==0)
 *		are collapsed
 *
 * 	  for reading whole data section, bufsize must be length*hdr->AS.bpb (also if flag is set)
 */

	if (buf != NULL) {
		if (length > (bufsize / hdr->AS.bpb)) {
			fprintf(stderr, "Warning %s (line %i): bufsize is not large enough for converting %i blocks.\n", \
				__func__, __LINE__, (int)length);
			length = bufsize / hdr->AS.bpb;
		}

		if ( (hdr->AS.first <= start) && ((start+length) <= (hdr->AS.first+hdr->AS.length)) ) {
			/****  copy from rawData if available:
				- better performance
				- required for some data formats (e.g. CFS, when hdr->AS.rawdata is populated in SOPEN)
			 ****/

			if (!hdr->AS.flag_collapsed_rawdata) {
				memcpy(buf, hdr->AS.rawdata + (start - hdr->AS.first) * hdr->AS.bpb, bufsize);
				if (flag) collapse_rawdata(hdr, buf, length);
				return (length);
			}
			else if (flag) {
				size_t bpb = bpb8_collapsed_rawdata(hdr)>>3;
				memcpy(buf, hdr->AS.rawdata + (start - hdr->AS.first) * bpb, bufsize);
				return (bufsize / bpb);
			}
			// else if (hdr->AS.flag_collapsed_rawdata && !flag) is handled below
		}
	}

	if (hdr->AS.flag_collapsed_rawdata && !flag)
		hdr->AS.length = 0; // 	force reloading of data

	size_t	count, nelem;

	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"%s (line %i): start=%d length=%d nrec=%d POS=%d bpb=%i\n",__func__,__LINE__, \
			(int)start,(int)length,(int)hdr->NRec, (int)hdr->FILE.POS, hdr->AS.bpb);

	if ((nrec_t)start > hdr->NRec)
		return(0);
	else if ((ssize_t)start < 0)
		start = hdr->FILE.POS;

	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"%s (line %i): %d %d %d %d\n",__func__,__LINE__, (int)start, (int)length, (int)hdr->NRec, (int)hdr->FILE.POS);

	// limit reading to end of data block
	if (hdr->NRec<0)
		nelem = length;
	else if (start >= (size_t)hdr->NRec)
		nelem = 0;
	else
		nelem = min(length, hdr->NRec - start);

	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"%s (line %i): %i %i %i %i %i %p\n",__func__,__LINE__, \
			(int)start, (int)length, (int)nelem, (int)hdr->NRec, (int)hdr->FILE.POS, hdr->AS.rawdata);

	if ( (buf == NULL) && (start >= hdr->AS.first) && ( (start + nelem) <= (hdr->AS.first + hdr->AS.length) ) ) {
		// Caching, no file-IO, data is already loaded into hdr->AS.rawdata
		hdr->FILE.POS = start;
		count = nelem;

		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"%s (line %i): \n",__func__,__LINE__);

	}
#ifndef WITHOUT_NETWORK
	else if (hdr->FILE.Des > 0) {
		// network connection
		int s = bscs_requ_dat(hdr->FILE.Des, start, length,hdr);
		count = hdr->AS.length;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"sread-raw from network: 222 count=%i\n",(int)count);
	}
#endif
	else {

		assert(hdr->TYPE != CFS);	// CFS data has been already cached in SOPEN
		assert(hdr->TYPE != SMR);	// CFS data has been already cached in SOPEN

		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"%s (line %i): \n",__func__,__LINE__);

		// read required data block(s)
		if (ifseek(hdr, start*hdr->AS.bpb + hdr->HeadLen, SEEK_SET)<0) {
			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"--%i %i %i %i \n",(int)(start*hdr->AS.bpb + hdr->HeadLen), (int)start, (int)hdr->AS.bpb, (int)hdr->HeadLen);
			return(0);
		}
		else
			hdr->FILE.POS = start;

		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"%s (line %i): bpb=%i\n",__func__,__LINE__,(int)hdr->AS.bpb);

		// allocate AS.rawdata
		void* tmpptr = buf;
		if (buf == NULL) {
			tmpptr = realloc(hdr->AS.rawdata, hdr->AS.bpb*nelem);
			if ((tmpptr!=NULL) || (hdr->AS.bpb*nelem==0)) {
				if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i)  %i %i \n",__func__,__LINE__,(int)hdr->AS.bpb,(int)nelem);
				hdr->AS.rawdata = (uint8_t*) tmpptr;
			}
			else {
				biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "memory allocation failed");
				return(0);
			}
		}

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"#sread(%i %li)\n",(int)(hdr->HeadLen + hdr->FILE.POS*hdr->AS.bpb), iftell(hdr));

		// read data
		count = ifread(tmpptr, hdr->AS.bpb, nelem, hdr);
		if (buf != NULL) {
			hdr->AS.flag_collapsed_rawdata = 0;	// is rawdata not collapsed
			hdr->AS.first = start;
			hdr->AS.length= count;
		}

		if (count < nelem) {
			fprintf(stderr,"warning: less than the number of requested blocks read (%i/%i) from file %s - something went wrong\n",(int)count,(int)nelem,hdr->FileName);
			if (VERBOSE_LEVEL>7)
				fprintf(stderr,"warning: only %i instead of %i blocks read - something went wrong (bpb=%i,pos=%li)\n",(int)count,(int)nelem,hdr->AS.bpb,iftell(hdr));
		}
	}
	// (uncollapsed) data is now in buffer hdr->AS.rawdata

	if (flag) {
		collapse_rawdata(hdr, NULL, 0);
	}
	return(count);
}

/****************************************************************************
 	caching: load data of whole file into buffer
		 this will speed up data access, especially in interactive mode
 ****************************************************************************/
int cachingWholeFile(HDRTYPE* hdr) {

	sread_raw(0,hdr->NRec,hdr, 0, NULL, 0);

	return((hdr->AS.first != 0) || (hdr->AS.length != (size_t)hdr->NRec));
}



/****************************************************************************/
/**	SREAD : segment-based                                              **/
/****************************************************************************/
size_t sread(biosig_data_type* data, size_t start, size_t length, HDRTYPE* hdr) {
/*
 *	Reads LENGTH blocks with HDR.AS.bpb BYTES each
 * 	(and HDR.SPR samples).
 *	Rawdata is available in hdr->AS.rawdata.
 *      Output data is available in hdr->data.block.
 *      If the request can be completed, hdr->data.block contains
 *	LENGTH*HDR.SPR samples and HDR.NS channels.
 *	The size of the output data is availabe in hdr->data.size.
 *
 *      hdr->FLAG.LittleEndian controls swapping
 *
 *      hdr->CHANNEL[k].OnOff 	controls whether channel k is loaded or not
 *
 *	data is a pointer to a memory array to write the data.
 *	if data is NULL, memory is allocated and the pointer is returned
 *	in hdr->data.block.
 *
 *	channel selection is controlled by hdr->CHANNEL[k].OnOff
 *
 *        start <0: read from current position
 *             >=0: start reading from position start
 *        length  : try to read length blocks
 *
 *
 * ToDo:
 *	- sample-based loading
 *
 */

	size_t			count,k1,k2,k4,k5=0,NS;//bi,bi8;
	size_t			toffset;	// time offset for rawdata
	biosig_data_type	*data1=NULL;


	if (VERBOSE_LEVEL>6)
		fprintf(stdout,"%s( %p, %i, %i, %s ) MODE=%i bpb=%i\n",__func__,data, (int)start, (int)length, hdr->FileName, hdr->FILE.OPEN, (int)hdr->AS.bpb);

	if ((ssize_t)start < 0) start=hdr->FILE.POS;

	if (start >= (size_t)hdr->NRec) return(0);

	switch (hdr->TYPE) {
	case AXG:
	case ABF2:
	case ATF:
	case SMR: // data is already cached
		count = hdr->NRec;
		break;
	default:
		count = sread_raw(start, length, hdr, 0, NULL, 0);
	}

	if (hdr->AS.B4C_ERRNUM) return(0);

	toffset = start - hdr->AS.first;

	// set position of file handle
	size_t POS = hdr->FILE.POS;
	hdr->FILE.POS += count;

	// count number of selected channels
	for (k1=0,NS=0; k1<hdr->NS; ++k1)
		if (hdr->CHANNEL[k1].OnOff) ++NS;

	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"%s (line %i): count=%i pos=[%i,%i,%i,%i], size of data = %ix%ix%ix%i = %i\n", __func__, __LINE__, \
			(int)count,(int)start,(int)length,(int)POS,(int)hdr->FILE.POS,(int)hdr->SPR, (int)count, \
			(int)NS, (int)sizeof(biosig_data_type), (int)(hdr->SPR * count * NS * sizeof(biosig_data_type)));

#ifndef ANDROID
//Stoyan: Arm has some problem with log2 - or I dont know how to fix it - it exists but do not work.
        if (log2(hdr->SPR) + log2(count) + log2(NS) + log2(sizeof(biosig_data_type)) + 1 >= sizeof(size_t)*8) {
                // used to check the 2GByte limit on 32bit systems
                biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Size of required data buffer too large (exceeds size_t addressable space)");
                return(0);
        }
#endif
	// transfer RAW into BIOSIG data format
	if ((data==NULL) || hdr->Calib) {
		// local data memory required
		size_t sz = hdr->SPR * count * NS * sizeof(biosig_data_type);
		void *tmpptr = realloc(hdr->data.block, sz);
		if (tmpptr!=NULL || !sz)
			data1 = (biosig_data_type*) tmpptr;
		else {
                        biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "memory allocation failed - not enough memory");
                        return(0);
		}
		hdr->data.block = data1;
	}
	else
		data1 = data;

	char ALPHA12BIT = (hdr->TYPE==alpha) && (hdr->NS>0) && (hdr->CHANNEL[0].GDFTYP==(255+12));
	char MIT12BIT   = (hdr->TYPE==MIT  ) && (hdr->NS>0) && (hdr->CHANNEL[0].GDFTYP==(255+12));
#if (__BYTE_ORDER == __BIG_ENDIAN)
	char SWAP = hdr->FILE.LittleEndian;
#elif (__BYTE_ORDER == __LITTLE_ENDIAN)
	char SWAP = !hdr->FILE.LittleEndian;
#endif

	int stride = 1;

#ifndef  ONLYGDF
	if (hdr->TYPE==Axona)
		stride = 64;
	else if (hdr->TYPE==TMS32)
		stride = hdr->NS;
#endif //ONLYGDF

	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"%s (line %i): alpha12bit=%i SWAP=%i spr=%i   %p\n",__func__,__LINE__, ALPHA12BIT, SWAP, hdr->SPR, hdr->AS.rawdata);

	for (k1=0,k2=0; k1<hdr->NS; k1++) {
		CHANNEL_TYPE *CHptr = hdr->CHANNEL+k1;

	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"%s (line %i) #%i#%i: alpha12bit=%i SWAP=%i spr=%i   %p | bi=%i bpb=%i \n",__func__,__LINE__, (int)k1, (int)k2, ALPHA12BIT, SWAP, hdr->SPR, hdr->AS.rawdata,(int)CHptr->bi,(int)hdr->AS.bpb );

	if (CHptr->OnOff) {	/* read selected channels only */
	if (CHptr->SPR > 0) {
		size_t DIV 	= hdr->SPR/CHptr->SPR;
		uint16_t GDFTYP = CHptr->GDFTYP;
		size_t SZ  	= GDFTYP_BITS[GDFTYP];
		int32_t int32_value = 0;
		uint8_t bitoff = 0;

		union {int16_t i16; uint16_t u16; uint32_t i32; float f32; uint64_t i64; double f64;} u;

		// TODO:  MIT data types
		for (k4 = 0; k4 < count; k4++)
		{  	uint8_t *ptr1;

#ifndef  ONLYGDF
			if (hdr->TYPE == FEF) {
				ptr1 = CHptr->bufptr;
			}
			else
#endif //ONLYGDF
				ptr1 = hdr->AS.rawdata + (k4+toffset)*hdr->AS.bpb + CHptr->bi;


		for (k5 = 0; k5 < CHptr->SPR; k5++)
		{

		biosig_data_type sample_value;
		uint8_t *ptr = ptr1 + (stride * k5 * SZ >> 3);

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"%s (line %i): k_i = [%d %d %d %d ] 0x%08x[%g] @%p => ",__func__,__LINE__,(int)k1,(int)k2,(int)k4,(int)k5,(int)leu32p(ptr),lef64p(ptr),ptr);

		switch (GDFTYP) {
		case 1:
			sample_value = (biosig_data_type)(*(int8_t*)ptr);
			break;
		case 2:
			sample_value = (biosig_data_type)(*(uint8_t*)ptr);
			break;
		case 3:
			if (SWAP) {
				sample_value = (biosig_data_type)(int16_t)bswap_16(*(int16_t*)ptr);
			}
			else {
				sample_value = (biosig_data_type)(*(int16_t*)ptr);
			}
			break;
		case 4:
			if (SWAP) {
				sample_value = (biosig_data_type)(uint16_t)bswap_16(*(uint16_t*)ptr);
			}
			else {
				sample_value = (biosig_data_type)(*(uint16_t*)ptr);
			}
			break;
		case 5:
			if (SWAP) {
				sample_value = (biosig_data_type)(int32_t)bswap_32(*(int32_t*)ptr);
			}
			else {
				sample_value = (biosig_data_type)(*(int32_t*)ptr);
			}
			break;
		case 6:
			if (SWAP) {
				sample_value = (biosig_data_type)(uint32_t)bswap_32(*(uint32_t*)ptr);
			}
			else {
				sample_value = (biosig_data_type)(*(uint32_t*)ptr);
			}
			break;
		case 7:
			if (SWAP) {
				sample_value = (biosig_data_type)(int64_t)bswap_64(*(int64_t*)ptr);
			}
			else {
				sample_value = (biosig_data_type)(*(int64_t*)ptr);
			}
			break;
		case 8:
			if (SWAP) {
				sample_value = (biosig_data_type)(uint64_t)bswap_64(*(uint64_t*)ptr);
			}
			else {
				sample_value = (biosig_data_type)(*(uint64_t*)ptr);
			}
			break;
		case 16:
			if (SWAP) {
				u.i32 = bswap_32(*(uint32_t*)(ptr));
				sample_value = (biosig_data_type)(u.f32);
			}
			else {
				sample_value = (biosig_data_type)(*(float*)(ptr));
			}
			break;

		case 17:
			if (SWAP) {
				u.i64 = bswap_64(*(uint64_t*)(ptr));
				sample_value = (biosig_data_type)(u.f64);
			}
			else {
				sample_value = (biosig_data_type)(*(double*)(ptr));
			}
			break;

		case 128:	// Nihon-Kohden little-endian int16 format
			u.u16 = leu16p(ptr) + 0x8000;
			sample_value = (biosig_data_type) (u.i16);
			break;

		case 255+12:
			if (ALPHA12BIT) {
				// get source address
				size_t off = (k4+toffset)*hdr->NS*SZ + hdr->CHANNEL[k1].bi8 + k5*SZ;
				ptr = hdr->AS.rawdata + (off>>3);

				if (off & 0x07)
					u.i16 = ptr[1] + ((ptr[0] & 0x0f)<<8);
				else
					u.i16 = (ptr[0]<<4) + (ptr[1] >> 4);

				if (u.i16 & 0x0800) u.i16 -= 0x1000;
				sample_value = (biosig_data_type)u.i16;
			}
			else if (MIT12BIT) {
				size_t off = (k4+toffset)*hdr->NS*SZ + hdr->CHANNEL[k1].bi8 + k5*SZ;
				ptr = hdr->AS.rawdata + (off>>3);
				//bitoff = k5*SZ & 0x07;
				if (off & 0x07)
					u.i16 = (((uint16_t)ptr[0] & 0xf0) << 4) + ptr[1];
				else
					//u.i16 = ((uint16_t)ptr[0]<<4) + (ptr[1] & 0x0f);
					u.i16 = leu16p(ptr) & 0x0fff;

				if (u.i16 & 0x0800) u.i16 -= 0x1000;
				sample_value = (biosig_data_type)u.i16;
			}
			else if (hdr->FILE.LittleEndian) {
				bitoff = k5*SZ & 0x07;
#if __BYTE_ORDER == __BIG_ENDIAN
				u.i16 = (leu16p(ptr) >> (4-bitoff)) & 0x0FFF;
#elif __BYTE_ORDER == __LITTLE_ENDIAN
				u.i16 = (leu16p(ptr) >> bitoff) & 0x0FFF;
#endif
				if (u.i16 & 0x0800) u.i16 -= 0x1000;
				sample_value = (biosig_data_type)u.i16;
			}
			else {
				bitoff = k5*SZ & 0x07;
#if __BYTE_ORDER == __BIG_ENDIAN
				u.i16 = (beu16p(ptr) >> (4-bitoff)) & 0x0FFF;
#elif __BYTE_ORDER == __LITTLE_ENDIAN
				u.i16 = (beu16p(ptr) >> bitoff) & 0x0FFF;
#endif
				if (u.i16 & 0x0800) u.i16 -= 0x1000;
				sample_value = (biosig_data_type)u.i16;
			}
			break;

		case 511+12:
			bitoff = k5*SZ & 0x07;
			if (hdr->FILE.LittleEndian) {
#if __BYTE_ORDER == __BIG_ENDIAN
				sample_value = (biosig_data_type)((leu16p(ptr) >> (4-bitoff)) & 0x0FFF);
#elif __BYTE_ORDER == __LITTLE_ENDIAN
				sample_value = (biosig_data_type)((leu16p(ptr) >> bitoff) & 0x0FFF);
#endif
			} else {
#if __BYTE_ORDER == __BIG_ENDIAN
				sample_value = (biosig_data_type)((beu16p(ptr) >> (4-bitoff)) & 0x0FFF);
#elif __BYTE_ORDER == __LITTLE_ENDIAN
				sample_value = (biosig_data_type)((beu16p(ptr) >> (4-bitoff)) & 0x0FFF);
#endif
			}

		case 255+24:
			if (hdr->FILE.LittleEndian) {
				int32_value = (*(uint8_t*)(ptr)) + (*(uint8_t*)(ptr+1)<<8) + (*(int8_t*)(ptr+2)*(1<<16));
				sample_value = (biosig_data_type)int32_value;
			}
			else {
				int32_value = (*(uint8_t*)(ptr+2)) + (*(uint8_t*)(ptr+1)<<8) + (*(int8_t*)(ptr)*(1<<16));
				sample_value = (biosig_data_type)int32_value;
			}
			break;

		case 511+24:
			if (hdr->FILE.LittleEndian) {
				int32_value = (*(uint8_t*)(ptr)) + (*(uint8_t*)(ptr+1)<<8) + (*(uint8_t*)(ptr+2)<<16);
				sample_value = (biosig_data_type)int32_value;
			}
			else {
				int32_value = (*(uint8_t*)(ptr+2)) + (*(uint8_t*)(ptr+1)<<8) + (*(uint8_t*)(ptr)<<16);
				sample_value = (biosig_data_type)int32_value;
			}
			break;

		default:
			if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i) GDFTYP=%i %i %i \n", __FILE__, __LINE__, GDFTYP, (int)k1, (int)k2);
			biosigERROR(hdr, B4C_DATATYPE_UNSUPPORTED, "Error SREAD: datatype not supported");
			return(-1);

		}	// end switch

		// overflow and saturation detection
		if ((hdr->FLAG.OVERFLOWDETECTION) && ((sample_value <= CHptr->DigMin) || (sample_value >= CHptr->DigMax)))
			sample_value = NAN; 	// missing value

		if (!hdr->FLAG.UCAL)	// scaling
			sample_value = sample_value * CHptr->Cal + CHptr->Off;

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"%g\n",sample_value);

		// resampling 1->DIV samples
		if (hdr->FLAG.ROW_BASED_CHANNELS) {
			size_t k3;
			for (k3=0; k3 < DIV; k3++)
				data1[k2 + (k4*hdr->SPR + k5*DIV + k3)*NS] = sample_value; // row-based channels
		} else {
			size_t k3;
			for (k3=0; k3 < DIV; k3++)
				data1[k2*count*hdr->SPR + k4*hdr->SPR + k5*DIV + k3] = sample_value; // column-based channels
		}

		}	// end for (k5 ....
		}	// end for (k4 ....

	}
	k2++;
	}}

	if (hdr->FLAG.ROW_BASED_CHANNELS) {
		hdr->data.size[0] = k2;			// rows
		hdr->data.size[1] = hdr->SPR*count;	// columns
	} else {
		hdr->data.size[0] = hdr->SPR*count;	// rows
		hdr->data.size[1] = k2;			// columns
	}

	/* read sparse samples */
	if (((hdr->TYPE==GDF) && (hdr->VERSION > 1.9)) || (hdr->TYPE==PDP)) {

		for (k1=0,k2=0; k1<hdr->NS; k1++) {
			CHANNEL_TYPE *CHptr = hdr->CHANNEL+k1;
			// Initialize sparse channels with NANs
			if (CHptr->OnOff) {	/* read selected channels only */
				if (CHptr->SPR==0) {
					// sparsely sampled channels are stored in event table
					if (hdr->FLAG.ROW_BASED_CHANNELS) {
						for (k5 = 0; k5 < hdr->SPR*count; k5++)
							data1[k2 + k5*NS] = CHptr->DigMin;		// row-based channels
					} else {
						for (k5 = 0; k5 < hdr->SPR*count; k5++)
							data1[k2*count*hdr->SPR + k5] = CHptr->DigMin; 	// column-based channels
					}
				}
				k2++;
			}
		}

		double c = hdr->SPR / hdr->SampleRate * hdr->EVENT.SampleRate;
		size_t *ChanList = (size_t*)calloc(hdr->NS+1,sizeof(size_t));

		// Note: ChanList and EVENT.CHN start with index=1 (not 0)
		size_t ch = 0;
		for (k1=0; k1<hdr->NS; k1++) // list of selected channels
			ChanList[k1+1]= (hdr->CHANNEL[k1].OnOff ? ++ch : 0);

		for (k1=0; k1<hdr->EVENT.N; k1++)
		if (hdr->EVENT.TYP[k1] == 0x7fff) 	// select non-equidistant sampled value
		if (ChanList[hdr->EVENT.CHN[k1]] > 0)	// if channel is selected
		if ((hdr->EVENT.POS[k1] >= POS*c) && (hdr->EVENT.POS[k1] < hdr->FILE.POS*c)) {
			biosig_data_type sample_value;
			uint8_t *ptr = (uint8_t*)(hdr->EVENT.DUR + k1);

			k2 = ChanList[hdr->EVENT.CHN[k1]]-1;
			CHANNEL_TYPE *CHptr = hdr->CHANNEL+k2;
			size_t DIV 	= (uint32_t)ceil(hdr->SampleRate/hdr->EVENT.SampleRate);
			uint16_t GDFTYP = CHptr->GDFTYP;
//			size_t SZ  	= GDFTYP_BITS[GDFTYP]>>3;	// obsolete
			int32_t int32_value = 0;

			if (0);
			else if (GDFTYP==3)
				sample_value = (biosig_data_type)lei16p(ptr);
			else if (GDFTYP==4)
				sample_value = (biosig_data_type)leu16p(ptr);
			else if (GDFTYP==16)
				sample_value = (biosig_data_type)lef32p(ptr);
/*			else if (GDFTYP==17)
				sample_value = (biosig_data_type)lef64p(ptr);
*/			else if (GDFTYP==0)
				sample_value = (biosig_data_type)(*(char*)ptr);
			else if (GDFTYP==1)
				sample_value = (biosig_data_type)(*(int8_t*)ptr);
			else if (GDFTYP==2)
				sample_value = (biosig_data_type)(*(uint8_t*)ptr);
			else if (GDFTYP==5)
				sample_value = (biosig_data_type)lei32p(ptr);
			else if (GDFTYP==6)
				sample_value = (biosig_data_type)leu32p(ptr);
/*			else if (GDFTYP==7)
				sample_value = (biosig_data_type)(*(int64_t*)ptr);
			else if (GDFTYP==8)
				sample_value = (biosig_data_type)(*(uint64_t*)ptr);
*/			else if (GDFTYP==255+24) {
				// assume LITTLE_ENDIAN format
				int32_value = (*(uint8_t*)(ptr)) + (*(uint8_t*)(ptr+1)<<8) + (*(int8_t*)(ptr+2)*(1<<16));
				sample_value = (biosig_data_type)int32_value;
			}
			else if (GDFTYP==511+24) {
				// assume LITTLE_ENDIAN format
				int32_value = (*(uint8_t*)(ptr)) + (*(uint8_t*)(ptr+1)<<8) + (*(uint8_t*)(ptr+2)<<16);
				sample_value = (biosig_data_type)int32_value;
			}
			else {
				if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i) GDFTYP=%i %i %i \n", __FILE__, __LINE__, GDFTYP,(int)k1,(int)k2);
				biosigERROR(hdr, B4C_DATATYPE_UNSUPPORTED, "Error SREAD: datatype not supported");
				return(0);
			}

			// overflow and saturation detection
			if ((hdr->FLAG.OVERFLOWDETECTION) && ((sample_value<=CHptr->DigMin) || (sample_value>=CHptr->DigMax)))
				sample_value = NAN; 	// missing value

			if (!hdr->FLAG.UCAL)	// scaling
				sample_value = sample_value * CHptr->Cal + CHptr->Off;

			// resampling 1->DIV samples
			k5  = (hdr->EVENT.POS[k1]/c - POS)*hdr->SPR;
			if (hdr->FLAG.ROW_BASED_CHANNELS) {
				size_t k3;
				for (k3=0; k3 < DIV; k3++)
					data1[k2 + (k5 + k3)*NS] = sample_value;
			} else {
				size_t k3;
				for (k3=0; k3 < DIV; k3++)
					data1[k2 * count * hdr->SPR + k5 + k3] = sample_value;
			}

		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"E%02i: s(%d,%d)= %d %e %e %e\n",(int)k1,(int)k2,hdr->EVENT.CHN[k1],leu32p(ptr),sample_value,(*(double*)(ptr)),(*(float*)(ptr)));

		}
		free(ChanList);
	}
	else if (hdr->TYPE==TMS32) {
		// post-processing TMS32 files: last block can contain undefined samples
		size_t spr = lei32p(hdr->AS.Header+121);
		if (hdr->FILE.POS*hdr->SPR > spr)
		for (k2=0; k2<NS; k2++)
		for (k5 = spr - POS*hdr->SPR; k5 < hdr->SPR*count; k5++) {
			if (hdr->FLAG.ROW_BASED_CHANNELS)
				data1[k2 + k5*NS] = NAN;		// row-based channels
			else
				data1[k2*count*hdr->SPR + k5] = NAN; 	// column-based channels
		}
	}

#ifdef CHOLMOD_H
	if (hdr->Calib) {
        if (!hdr->FLAG.ROW_BASED_CHANNELS)
                fprintf(stderr,"Error SREAD: Re-Referencing on column-based data not supported.");
        else {
			cholmod_dense X,Y;
			X.nrow = hdr->data.size[0];
			X.ncol = hdr->data.size[1];
			X.d    = hdr->data.size[0];
			X.nzmax= hdr->data.size[1]*hdr->data.size[0];
			X.x    = data1;
                        X.xtype = CHOLMOD_REAL;
                        X.dtype = CHOLMOD_DOUBLE;

			Y.nrow = hdr->Calib->ncol;
			Y.ncol = hdr->data.size[1];
			Y.d    = Y.nrow;
			Y.nzmax= Y.nrow * Y.ncol;
			if (data)
				Y.x    = data;
			else
				Y.x    = malloc(Y.nzmax*sizeof(double));

                        Y.xtype = CHOLMOD_REAL;
                        Y.dtype = CHOLMOD_DOUBLE;

			double alpha[]={1,0},beta[]={0,0};

			cholmod_sdmult(hdr->Calib,1,alpha,beta,&X,&Y,&CHOLMOD_COMMON_VAR);

			if (VERBOSE_LEVEL>8) fprintf(stdout,"%f -> %f\n",*(double*)X.x,*(double*)Y.x);
			free(X.x);
			if (data==NULL)
				hdr->data.block = (biosig_data_type*)Y.x;
			else
				hdr->data.block = NULL;

        		hdr->data.size[0] = Y.nrow;

	}
	}
#endif

	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"sread - end \n");

//VERBOSE_LEVEL = V;

	return(count);

}  // end of SREAD


#ifdef __GSL_MATRIX_DOUBLE_H__
/****************************************************************************/
/**	GSL_SREAD : GSL-version of sread                                   **/
/****************************************************************************/
size_t gsl_sread(gsl_matrix* m, size_t start, size_t length, HDRTYPE* hdr) {
/* 	same as sread but return data is of type gsl_matrix
*/
	// TODO: testing

        size_t count = sread(NULL, start, length, hdr);
	size_t n = hdr->data.size[0]*hdr->data.size[1];

	if (m->owner && m->block) gsl_block_free(m->block);
	m->block = gsl_block_alloc(n);
	m->block->data = hdr->data.block;

	m->size1 = hdr->data.size[1];
	m->tda   = hdr->data.size[0];
	m->size2 = hdr->data.size[0];
	m->data  = m->block->data;
	m->owner = 1;
	hdr->data.block = NULL;

	return(count);
}
#endif


/****************************************************************************/
/**                     SWRITE                                             **/
/****************************************************************************/
size_t swrite(const biosig_data_type *data, size_t nelem, HDRTYPE* hdr) {
/*
 *	writes NELEM blocks with HDR.AS.bpb BYTES each,
 */
	uint8_t		*ptr;
	size_t		count=0,k1,k2,k4,k5,DIV,SZ=0;
	int 		GDFTYP;
	CHANNEL_TYPE*	CHptr;
	biosig_data_type 	sample_value, iCal, iOff;
	union {
		int8_t i8;
		uint8_t u8;
		int16_t i16;
		uint16_t u16;
		int32_t i32;
		uint32_t u32;
		int64_t i64;
		uint64_t u64;
	} val;

	if (VERBOSE_LEVEL>6)
		fprintf(stdout,"%s( %p, %i, %s ) MODE=%i\n",__func__, data, (int)nelem, hdr->FileName, hdr->FILE.OPEN);

		// write data

#define MAX_INT8   ((int8_t)0x7f)
#define MIN_INT8   ((int8_t)0x80)
#define MAX_UINT8  ((uint8_t)0xff)
#define MIN_UINT8  ((uint8_t)0)
#define MAX_INT16  ((int16_t)0x7fff)
#define MIN_INT16  ((int16_t)0x8000)
#define MAX_UINT16 ((uint16_t)0xffff)
#define MIN_UINT16 ((uint16_t)0)
#define MAX_INT24  ((int32_t)0x007fffff)
#define MIN_INT24  ((int32_t)0xff800000)
#define MAX_UINT24 ((uint32_t)0x00ffffff)
#define MIN_UINT24 ((uint32_t)0)
#define MAX_INT32  ((int32_t)0x7fffffff)
#define MIN_INT32  ((int32_t)0x80000000)
#define MAX_UINT32 ((uint32_t)0xffffffff)
#define MIN_UINT32 ((uint32_t)0)
#define MAX_INT64  ((((uint64_t)1)<<63)-1)
#define MIN_INT64  ((int64_t)((uint64_t)1)<<63)
#define MAX_UINT64 ((uint64_t)0xffffffffffffffffl)
#define MIN_UINT64 ((uint64_t)0)


	size_t bpb8 = bpb8_collapsed_rawdata(hdr);

	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"%s (line %i): <%s> sz=%i\n",__func__,__LINE__,hdr->FileName,(int)(hdr->NRec*bpb8>>3));

	if (hdr->TYPE==ATF) {
		if (VERBOSE_LEVEL>7) fprintf(stdout,"ATF swrite\n");

		nrec_t nr = hdr->data.size[(int)hdr->FLAG.ROW_BASED_CHANNELS];        // if collapsed data, use k2, otherwise use k1
		assert(nr == hdr->NRec * hdr->SPR);

		typeof(hdr->NS) k,k2;
		nrec_t c = 0;

		unsigned timeChan = getTimeChannelNumber(hdr);

		if (hdr->data.size[1-hdr->FLAG.ROW_BASED_CHANNELS] < hdr->NS) {
			// if collapsed data, use k2, otherwise use k1
			for (c = 0; c < nr; c++) {
				char *sep = "\n";
				if (timeChan == 0) {
					fprintf(hdr->FILE.FID,"%s%.16g",sep,(++hdr->FILE.POS)*1000.0/hdr->SampleRate);
					sep = "\t";
				}
				for (k = 0, k2=0; k < hdr->NS; k++) {
					if (hdr->CHANNEL[k].OnOff) {
						size_t idx;
						if (hdr->FLAG.ROW_BASED_CHANNELS)
							idx = k2 + c * hdr->data.size[0];
						else
							idx = k2 * hdr->data.size[0] + c;

						fprintf(hdr->FILE.FID,"%s%.16g",sep,data[idx]);
						sep = "\t";
						k2++;
					}
				}
			}
		}
		else {	// if not collapsed data, use k1
			for (c = 0; c < nr; c++) {
				char *sep = "\n";
				if (timeChan == 0) {
					fprintf(hdr->FILE.FID,"%s%.16g",sep,(++hdr->FILE.POS)*1000.0/hdr->SampleRate);
					sep = "\t";
				}
				for (k = 0; k < hdr->NS; k++) {
					if (hdr->CHANNEL[k].OnOff) {
						size_t idx;
						if (hdr->FLAG.ROW_BASED_CHANNELS)
							idx = k + c * hdr->data.size[0];
						else
							idx = k * hdr->data.size[0] + c;

						fprintf(hdr->FILE.FID,"%s%.16g", sep, data[idx]);
						sep = "\t";
					}
				}
			}
		}
		return nr;
		// end write ATF
	}


	if ((hdr->NRec*bpb8>0) && (hdr->TYPE != SCP_ECG)) {
	// memory allocation for SCP is done in SOPEN_SCP_WRITE Section 6
		ptr = (typeof(ptr))realloc(hdr->AS.rawdata, (hdr->NRec*bpb8>>3)+1);
		if (ptr==NULL) {
			biosigERROR(hdr, B4C_INSUFFICIENT_MEMORY, "SWRITE: memory allocation failed.");
			return(0);
		}
		hdr->AS.rawdata = (uint8_t*)ptr;
	}


	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"%s (line %i): %dx%d %d\n",__func__,__LINE__,(int)hdr->NRec,hdr->SPR,hdr->NS);

	size_t bi8 = 0;
	for (k1=0,k2=0; k1<hdr->NS; k1++) {
	CHptr 	= hdr->CHANNEL+k1;

	if (CHptr->OnOff != 0) {
	if (CHptr->SPR) {

		DIV 	= hdr->SPR/CHptr->SPR;
		GDFTYP 	= CHptr->GDFTYP;
		SZ  	= GDFTYP_BITS[GDFTYP];
		iCal	= 1/CHptr->Cal;
		//iOff	= CHptr->DigMin - CHptr->PhysMin*iCal;
		iOff	= -CHptr->Off*iCal;

		size_t col = (hdr->data.size[1-hdr->FLAG.ROW_BASED_CHANNELS]<hdr->NS) ? k2 : k1;        // if collapsed data, use k2, otherwise use k1

	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"%s (line %i): #%i gdftyp=%i %i %i %i %f %f %f %f %i\n",
			__func__,__LINE__,(int)k1,GDFTYP,(int)bi8,(int)SZ,(int)CHptr->SPR,CHptr->Cal,CHptr->Off,iCal,iOff,(int)bpb8);

		for (k4 = 0; k4 < (size_t)hdr->NRec; k4++) {
			if (VERBOSE_LEVEL>8)
				fprintf(stdout,"%s (line %i): #%i: [%i %i] %i %i %i %i %i\n",
					__func__,__LINE__,(int)k1,(int)hdr->data.size[0],(int)hdr->data.size[1],(int)k4,(int)0,(int)hdr->SPR,(int)DIV,(int)nelem);

		for (k5 = 0; k5 < CHptr->SPR; k5++) {

			if (VERBOSE_LEVEL>8)
				fprintf(stdout,"%s (line %i): #%i: [%i %i] %i %i %i %i %i\n",
					__func__,__LINE__,(int)k1,(int)hdr->data.size[0],(int)hdr->data.size[1],(int)k4,(int)k5,(int)hdr->SPR,(int)DIV,(int)nelem);

			size_t k3=0;
			if (VERBOSE_LEVEL>8)
				fprintf(stdout,"%s (line %i): [%i %i %i %i %i] %i %i %i %i %i %i\n",
					__func__,__LINE__,(int)k1,(int)k2,(int)k3,(int)k4,(int)k5,(int)col,
					(int)hdr->data.size[0],(int)hdr->data.size[1],(int)hdr->SPR,(int)nelem,(int)hdr->NRec);

			sample_value = 0.0;
        		if (hdr->FLAG.ROW_BASED_CHANNELS) {
				for (k3=0; k3 < DIV; k3++)
        				sample_value += data[col + (k4*hdr->SPR + k5*DIV + k3)*hdr->data.size[0]];
                        }
            		else {
				for (k3=0; k3 < DIV; k3++)
        				sample_value += data[col*nelem*hdr->SPR + k4*hdr->SPR + k5*DIV + k3];
                        }
			sample_value /= DIV;

			if (VERBOSE_LEVEL>8)
				fprintf(stdout,"%s (line %i): %f/%i\n",__func__,__LINE__,sample_value,(int)DIV);

			if (!hdr->FLAG.UCAL)	// scaling
				sample_value = sample_value * iCal + iOff;

			// get target address
			//ptr = hdr->AS.rawdata + k4*hdr->AS.bpb + hdr->CHANNEL[k1].bi + k5*SZ;
			//ptr = hdr->AS.rawdata + (k4*bpb8 + bi8 + k5*SZ)>>3;

			//size_t off = k4*hdr->AS.bpb8 + hdr->CHANNEL[k1].bi8 + (k5*SZ);
			size_t off = k4*bpb8 + bi8 + (k5*SZ);
			ptr = hdr->AS.rawdata + (off>>3);

			if (VERBOSE_LEVEL>8)
				fprintf(stdout,"%s (line %i): %i %i %i %f %p %p\n",
					__func__,__LINE__,(int)k4,(int)k5,(int)(off>>3),sample_value, hdr->AS.Header, ptr);

			// mapping of raw data type to (biosig_data_type)
			switch (GDFTYP) {
			case 3:
				if      (sample_value > MAX_INT16) val.i16 = MAX_INT16;
				else if (sample_value > MIN_INT16) val.i16 = (int16_t) sample_value;
				else     val.i16 = MIN_INT16;
				lei16a(val.i16, ptr);
				break;

			case 4:
				if      (sample_value > MAX_UINT16) val.u16 = MAX_UINT16;
				else if (sample_value > MIN_UINT16) val.u16 = (uint16_t) sample_value;
				else     val.u16 = MIN_UINT16;
				leu16a(val.u16, ptr);
				break;

			case 16:
				lef32a((float)sample_value, ptr);
				break;

			case 17:
				lef64a((double)sample_value, ptr);
				break;

			case 0:
				if      (sample_value > MAX_INT8) val.i8 = MAX_INT8;
				else if (sample_value > MIN_INT8) val.i8 = (int8_t) sample_value;
				else     val.i8 = MIN_INT8;
				*(int8_t*)ptr = val.i8;
				break;

			case 1:
				if      (sample_value > MAX_INT8) val.i8 = MAX_INT8;
				else if (sample_value > MIN_INT8) val.i8 = (int8_t) sample_value;
				else     val.i8 = MIN_INT8;
				*(int8_t*)ptr = val.i8;
				break;

			case 2:
				if      (sample_value > MAX_UINT8) val.u8 = MAX_UINT8;
				else if (sample_value > MIN_UINT8) val.u8 = (uint8_t) sample_value;
				else      val.u8 = MIN_UINT8;
				*(uint8_t*)ptr = val.u8;
				break;

			case 5:
				if      (sample_value > ldexp(1.0,31)-1) val.i32 = MAX_INT32;
				else if (sample_value > ldexp(-1.0,31)) val.i32 = (int32_t) sample_value;
				else     val.i32 = MIN_INT32;
				lei32a(val.i32, ptr);
				break;

			case 6:
				if      (sample_value > ldexp(1.0,32)-1.0) val.u32 = MAX_UINT32;
				else if (sample_value > 0.0) val.u32 = (uint32_t) sample_value;
				else     val.u32 = MIN_UINT32;
				leu32a(val.u32, ptr);
				break;
			case 7:
				if      (sample_value > ldexp(1.0,63)-1.0) val.i64 = MAX_INT64;
				else if (sample_value > -ldexp(1.0,63)) val.i64 = (int64_t) sample_value;
				else     val.i64 = MIN_INT64;
				lei64a(val.i64, ptr);
				break;
			case 8:
				if      (sample_value > ldexp(1.0,64)-1.0) val.u64 = (uint64_t)(-1);
				else if (sample_value > 0.0) val.u64 = (uint64_t) sample_value;
				else     val.u64 = 0;
				leu64a(val.u64, ptr);
				break;

			case 255+24:
				if      (sample_value > MAX_INT24) val.i32 = MAX_INT24;
				else if (sample_value > MIN_INT24) val.i32 = (int32_t) sample_value;
				else     val.i32 = MIN_INT24;
				*(uint8_t*)ptr = (uint8_t)(val.i32 & 0x000000ff);
				*((uint8_t*)ptr+1) = (uint8_t)((val.i32>>8) & 0x000000ff);
				*((uint8_t*)ptr+2) = (uint8_t)((val.i32>>16) & 0x000000ff);
				break;

			case 511+24:
				if      (sample_value > MAX_UINT24) val.i32 = MAX_UINT24;
				else if (sample_value > MIN_UINT24) val.i32 = (int32_t) sample_value;
				else     val.i32 = MIN_UINT24;
				*(uint8_t*)ptr     =  val.i32 & 0x000000ff;
				*((uint8_t*)ptr+1) = (uint8_t)((val.i32>>8) & 0x000000ff);
				*((uint8_t*)ptr+2) = (uint8_t)((val.i32>>16) & 0x000000ff);
				break;

			case 255+12:
			case 511+12: {
				if (GDFTYP == 255+12) {
					if      (sample_value > ((1<<11)-1)) val.i16 =  (1<<11)-1;
					else if (sample_value > -(1<<11))  val.i16 = (int16_t) sample_value;
					else     val.i16 = -(1<<11);
				}
				else if (GDFTYP == 511+12) {
					if      (sample_value > ((1<<12)-1)) val.u16 =  (1<<12)-1;
					else if (sample_value > 0)  val.u16 = (int16_t) sample_value;
					else     val.u16 = 0;
				}

				if (hdr->FILE.LittleEndian) {
					uint16_t acc = leu16p(ptr);
					if (off)
						leu16a( (acc & 0x000F) | (val.u16<<4), ptr);
					else
						leu16a( (acc & 0xF000) | (val.u16 & 0x0FFF), ptr);
				}
				else {
					uint16_t acc = beu16p(ptr);
					if (!off)
						beu16a( (acc & 0x000F) | (val.u16<<4), ptr);
					else
						beu16a( (acc & 0xF000) | (val.u16 & 0x0FFF), ptr);
				}
				break;
			}
			default:
				biosigERROR(hdr, B4C_DATATYPE_UNSUPPORTED, "SWRITE: datatype not supported");
				return(0);
			}
		}        // end for k5
		}        // end for k4
		}        // end if SPR
		k2++;
		bi8 += SZ*CHptr->SPR;
	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"swrite 314 %i\n",(int)k2);

	}       // end if OnOff
	}	// end for k1

	if (VERBOSE_LEVEL>7)
		fprintf(stdout,"swrite 315 <%s>\n", hdr->FileName );

#ifndef WITHOUT_NETWORK
	if (hdr->FILE.Des>0) {

	if (VERBOSE_LEVEL>7) fprintf(stdout,"bscs_send_dat sz=%i\n",(int)(hdr->NRec*bpb8>>3));

		int s = bscs_send_dat(hdr->FILE.Des,hdr->AS.rawdata,hdr->NRec*bpb8>>3);

	if (VERBOSE_LEVEL>7) fprintf(stdout,"bscs_send_dat succeeded %i\n",s);

	}
	else
#endif

#ifndef  ONLYGDF
	if ((hdr->TYPE == ASCII) || (hdr->TYPE == BIN)) {
		HDRTYPE H1;
		H1.CHANNEL = NULL;
		H1.FILE.COMPRESSION = hdr->FILE.COMPRESSION;

		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"swrite ASCII/BIN\n");

		k1 = strlen(hdr->FileName);
		char* fn = (char*)calloc(k1 + 10,1);
		strcpy(fn, hdr->FileName);
		char *e = strrchr(fn,'.');
		if (e==NULL) {
			fn[k1] = '.';
			e = fn+k1;
		}
		e[1] = (hdr->TYPE == ASCII ? 'a' : 's');
		e+=2;

		for (k1=0; k1<hdr->NS; k1++)
		if (hdr->CHANNEL[k1].OnOff) {
		//if (hdr->CHANNEL[k1].OnOff && hdr->CHANNEL[k1].SPR) {
			/* Off channels and sparse channels (SPR) are not exported;
				sparse samples are available throught the header file
				containing the event table.
			*/
			CHptr 	= hdr->CHANNEL+k1;
    			if (hdr->FILE.COMPRESSION) sprintf(e,"%02i_gz",(int)k1+1);
    			else sprintf(e,"%02i",(int)k1+1);

    			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"#%i: %s\n",(int)k1,fn);

			H1.FileName = fn;
			ifopen(&H1,"wb");

			if (hdr->TYPE == ASCII) {
				typeof(hdr->SPR) SPR;
				if (CHptr->SPR>0) {
					DIV = hdr->SPR/CHptr->SPR;
					SPR = CHptr->SPR;
				}
				else {
					DIV = 1;
					SPR = hdr->SPR;
				}
				size_t k2;
				for (k2=0; k2 < SPR*(size_t)hdr->NRec; k2++) {
					biosig_data_type i = 0.0;
					size_t k3;
						// TODO: row channels
					if (hdr->FLAG.ROW_BASED_CHANNELS)
						for (k3=0; k3<DIV; k3++)
							i += hdr->data.block[k1+(k2*DIV+k3)*hdr->data.size[0]];
					else
						// assumes column channels
						for (k3=0; k3<DIV; k3++)
							i += hdr->data.block[hdr->SPR*hdr->NRec*k1+k2*DIV+k3];

/*
        		if (hdr->FLAG.ROW_BASED_CHANNELS) {
            			for (k3=0, sample_value=0.0; k3 < DIV; k3++)
        				sample_value += data[col + (k4*hdr->SPR + k5*DIV + k3)*hdr->data.size[0]];
                        }
            		else {
            			for (k3=0, sample_value=0.0; k3 < DIV; k3++)
        				sample_value += data[col*nelem*hdr->SPR + k4*hdr->SPR + k5*DIV + k3];
                        }
*/

#ifdef ZLIB_H
					if (H1.FILE.COMPRESSION)
						gzprintf(H1.FILE.gzFID,"%g\n",i/DIV);
					else
#endif
						fprintf(H1.FILE.FID,"%g\n",i/DIV);
				}
			}
			else if (hdr->TYPE == BIN) {
				size_t nbytes	= ((size_t)hdr->CHANNEL[k1].SPR * GDFTYP_BITS[hdr->CHANNEL[k1].GDFTYP])>>3;
				ifwrite(hdr->AS.rawdata+hdr->CHANNEL[k1].bi, nbytes, hdr->NRec, &H1);
			}
			ifclose(&H1);
		}
		count = hdr->NRec;
		free(fn);
	}
	else
#endif //ONLYGDF
	     if ((hdr->TYPE != SCP_ECG) && (hdr->TYPE != HL7aECG)) {
		// for SCP: writing to file is done in SCLOSE

		if (VERBOSE_LEVEL>7) fprintf(stdout,"swrite 317 <%s>\n", hdr->FileName );

		count = ifwrite((uint8_t*)(hdr->AS.rawdata), hdr->AS.bpb, hdr->NRec, hdr);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"swrite 319 <%i>\n", (int)count);

	}
	else { 	// SCP_ECG, HL7aECG
#ifdef  ONLYGDF
		assert(0);
#endif //ONLYGDF
		count = 1;
	}

	// set position of file handle
	hdr->FILE.POS += count;

	return(count);

}  // end of SWRITE


/****************************************************************************/
/**                     SEOF                                               **/
/****************************************************************************/
int seof(HDRTYPE* hdr)
{
	return (hdr->FILE.POS >= (size_t)hdr->NRec);
}


/****************************************************************************/
/**                     SREWIND                                            **/
/****************************************************************************/
void srewind(HDRTYPE* hdr)
{
	sseek(hdr,0,SEEK_SET);
	return;
}


/****************************************************************************/
/**                     SSEEK                                              **/
/****************************************************************************/
int sseek(HDRTYPE* hdr, ssize_t offset, int whence)
{
	int64_t pos=0;

	if    	(whence < 0)
		pos = offset * hdr->AS.bpb;
	else if (whence == 0)
		pos = (hdr->FILE.POS + offset) * hdr->AS.bpb;
	else if (whence > 0)
		pos = (hdr->NRec + offset) * hdr->AS.bpb;

	if ((pos < 0) | (pos > hdr->NRec * hdr->AS.bpb))
		return(-1);
	else if (ifseek(hdr, pos + hdr->HeadLen, SEEK_SET))
		return(-1);

	hdr->FILE.POS = pos / (hdr->AS.bpb);
	return(0);

}  // end of SSEEK


/****************************************************************************/
/**                     STELL                                              **/
/****************************************************************************/
ssize_t stell(HDRTYPE* hdr)
{
	ssize_t pos = iftell(hdr);

	if (pos<0)
		return(-1);
	else if ((size_t)pos != (hdr->FILE.POS * hdr->AS.bpb + hdr->HeadLen))
		return(-1);
	else
		return(hdr->FILE.POS);

}  // end of STELL


/****************************************************************************/
/**                     SCLOSE                                             **/
/****************************************************************************/
int sclose(HDRTYPE* hdr) {
	ssize_t pos, len;

	if (VERBOSE_LEVEL>6)
		fprintf(stdout,"SCLOSE( %s ) MODE=%i\n",hdr->FileName, hdr->FILE.OPEN);

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): sclose\n",__FILE__,__LINE__);

        if (hdr==NULL) return(0);

	size_t k;
	for (k=0; k<hdr->NS; k++) {
		// replace Nihon-Kohden code with standard code
		if (hdr->CHANNEL[k].GDFTYP==128)
			hdr->CHANNEL[k].GDFTYP=3;
	}

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): sclose OPEN=%i %s\n",__FILE__,__LINE__,hdr->FILE.OPEN,GetFileTypeString(hdr->TYPE));

#if defined(WITH_FEF) && !defined(ONLYGDF)
	if (hdr->TYPE == FEF) sclose_fef_read(hdr);
#endif

#ifndef WITHOUT_NETWORK
	if (hdr->FILE.Des>0) {
		// network connection
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): sclose\n",__FILE__,__LINE__);

		if (hdr->FILE.OPEN > 1) bscs_send_evt(hdr->FILE.Des,hdr);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): sclose\n",__FILE__,__LINE__);

  		int s = bscs_close(hdr->FILE.Des);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): sclose\n",__FILE__,__LINE__);

  		if (s & ERR_MASK) {
			if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): bscs_close failed (err %i %08x)\n",__FILE__,__LINE__,s,s);
			biosigERROR(hdr, B4C_SCLOSE_FAILED, "bscs_close failed");
  		}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): sclose\n",__FILE__,__LINE__);

  		hdr->FILE.Des = 0;
  		hdr->FILE.OPEN = 0;
		bscs_disconnect(hdr->FILE.Des);
	}
	else
#endif
	if ((hdr->FILE.OPEN>1) && ((hdr->TYPE==GDF) || (hdr->TYPE==EDF) || (hdr->TYPE==BDF)))
	{

		if (VERBOSE_LEVEL>7) fprintf(stdout,"sclose(121) nrec= %i\n",(int)hdr->NRec);

		// WRITE HDR.NRec
		pos = (iftell(hdr)-hdr->HeadLen);
		if (hdr->NRec<0)
		{
			union {
				char tmp[88];
				int64_t i64;
			} t;

			if (pos>0) 	hdr->NRec = pos/hdr->AS.bpb;
			else		hdr->NRec = 0;
			if (hdr->TYPE==GDF) {
				t.i64 = htole64(hdr->NRec);
				len = sizeof(hdr->NRec);
			}
			else {
				len = sprintf(t.tmp,"%d",(int)hdr->NRec);
				if (len>8) fprintf(stderr,"Warning: NRec is (%s) to long.\n",t.tmp);
			}
			/* ### FIXME : gzseek supports only forward seek */
			if (hdr->FILE.COMPRESSION>0)
				fprintf(stderr,"Warning: writing NRec in gz-file requires gzseek which may not be supported.\n");
			ifseek(hdr,236,SEEK_SET);
			ifwrite(t.tmp,len,1,hdr);
		}

		if (VERBOSE_LEVEL>7)
			fprintf(stdout, "888: File Type=%s ,N#of Events %i,bpb=%i\n",GetFileTypeString(hdr->TYPE),hdr->EVENT.N,hdr->AS.bpb);

		if ((hdr->TYPE==GDF) && (hdr->EVENT.N>0)) {

			size_t len = hdrEVT2rawEVT(hdr);
			ifseek(hdr, hdr->HeadLen + hdr->AS.bpb*hdr->NRec, SEEK_SET);
			ifwrite(hdr->AS.rawEventData, len, 1, hdr);

//			write_gdf_eventtable(hdr);
		}

	}

#ifndef  ONLYGDF
	else if ((hdr->FILE.OPEN>1) && (hdr->TYPE==ATF)) {
		fprintf(hdr->FILE.FID, "\n");
	}
	else if ((hdr->FILE.OPEN>1) && (hdr->TYPE==SCP_ECG)) {
		uint16_t 	crc;
		uint8_t*	ptr; 	// pointer to memory mapping of the file layout

		hdr->AS.rawdata = NULL;
		struct aecg* aECG = (struct aecg*)hdr->aECG;
		if (aECG->Section5.Length>0) {
			// compute CRC for Section 5
			uint16_t crc = CRCEvaluate(hdr->AS.Header + aECG->Section5.StartPtr+2,aECG->Section5.Length-2); // compute CRC
			leu16a(crc, hdr->AS.Header + aECG->Section5.StartPtr);
		}
		if (aECG->Section6.Length>0) {
			// compute CRC for Section 6
			uint16_t crc = CRCEvaluate(hdr->AS.Header + aECG->Section6.StartPtr+2,aECG->Section6.Length-2); // compute CRC
			leu16a(crc, hdr->AS.Header + aECG->Section6.StartPtr);
		}
		if (aECG->Section7.Length>0) {
			// compute CRC for Section 7
			uint16_t crc = CRCEvaluate(hdr->AS.Header + aECG->Section7.StartPtr+2,aECG->Section7.Length-2); // compute CRC
			leu16a(crc, hdr->AS.Header + aECG->Section7.StartPtr);
		}
		if ((aECG->Section12.Length>0) && (hdr->VERSION > 2.5)) {
			// compute CRC for Section 12
			uint16_t crc = CRCEvaluate(hdr->AS.Header + aECG->Section12.StartPtr+2,aECG->Section12.Length-2); // compute CRC
			leu16a(crc, hdr->AS.Header + aECG->Section12.StartPtr);
		}
		// compute crc and len and write to preamble
		ptr = hdr->AS.Header;
		leu32a(hdr->HeadLen, ptr+2);
		crc = CRCEvaluate(ptr+2,hdr->HeadLen-2);
		leu16a(crc, ptr);
		ifwrite(hdr->AS.Header, sizeof(char), hdr->HeadLen, hdr);
	}
	else if ((hdr->FILE.OPEN>1) && (hdr->TYPE==HL7aECG))
	{
		sclose_HL7aECG_write(hdr);
		hdr->FILE.OPEN = 0;
	}
#endif //ONLYGDF

	if (hdr->FILE.OPEN > 0) {
		int status = ifclose(hdr);
		if (status) iferror(hdr);
		hdr->FILE.OPEN = 0;
    	}

    	return(0);
}


/****************************************************************************/
/**                     Error Handling                                     **/
/****************************************************************************/
void biosigERROR(HDRTYPE *hdr, enum B4C_ERROR errnum, const char *errmsg) {
/*
	sets the local and the (deprecated) global error variables B4C_ERRNUM and B4C_ERRMSG
	the global error variables are kept for backwards compatibility.
*/
#if (BIOSIG_VERSION < 10500)
#ifndef  ONLYGDF
	B4C_ERRNUM = errnum;
	B4C_ERRMSG = errmsg;
#endif //ONLYGDF
#endif
	hdr->AS.B4C_ERRNUM = errnum;
	hdr->AS.B4C_ERRMSG = errmsg;
}

#if (BIOSIG_VERSION < 10500)
#ifndef  ONLYGDF
// do not expose deprecated interface in libgdf
int serror() {
	int status = B4C_ERRNUM;
	fprintf(stderr,"Warning: use of function SERROR() is deprecated - use SERROR2() instead");
	if (status) {
		fprintf(stderr,"ERROR %i: %s\n",B4C_ERRNUM,B4C_ERRMSG);
		B4C_ERRNUM = B4C_NO_ERROR;
	}
	return(status);
}
#endif //ONLYGDF
#endif

int serror2(HDRTYPE *hdr) {
	int status = hdr->AS.B4C_ERRNUM;
	if (status) {
		fprintf(stderr,"ERROR %i: %s\n",hdr->AS.B4C_ERRNUM,hdr->AS.B4C_ERRMSG);
		hdr->AS.B4C_ERRNUM = B4C_NO_ERROR;
		hdr->AS.B4C_ERRMSG = NULL;
	}
	return(status);
}

char *biosig_get_errormsg(HDRTYPE *hdr) {
	if (hdr==NULL) return NULL;
	if (hdr->AS.B4C_ERRNUM==0) return NULL;
	return strdup(hdr->AS.B4C_ERRMSG);
};

int biosig_check_error(HDRTYPE *hdr) {
	if (hdr==NULL) return B4C_NO_ERROR;
	return hdr->AS.B4C_ERRNUM;
};


/****************************************************************************/
/*        Write / Update Event Table in GDF file                            */
/*                                                                          */
/* returns 0 in case of success                                             */
/* returns -1 in case of failure                                            */
/****************************************************************************/
int sflush_gdf_event_table(HDRTYPE* hdr)
{
	if ((hdr->TYPE!=GDF) || hdr->FILE.COMPRESSION)
		return(-1);

	ssize_t filepos = iftell(hdr);
	ifclose(hdr);
	hdr = ifopen(hdr,"rb+");
	if (!hdr->FILE.OPEN) {
		/* file cannot be opened in write mode */
		hdr = ifopen(hdr,"rb");
		return(-1);
	}

	size_t len = hdrEVT2rawEVT(hdr);
	ifseek(hdr, hdr->HeadLen + hdr->AS.bpb*hdr->NRec, SEEK_SET);
	ifwrite(hdr->AS.rawEventData, len, 1, hdr);
//	write_gdf_eventtable(hdr);

	ifseek(hdr,filepos,SEEK_SET);

	return(0);
}


void fprintf_json_double(FILE *fid, const char* Name, double val) {
	fprintf(fid,"\t\t\"%s\"\t: %g", Name, val);
}

/****************************************************************************/
/**                     HDR2ASCII                                          **/
/**	displaying header information                                      **/
/****************************************************************************/
int asprintf_hdr2json(char **str, HDRTYPE *hdr)
{
        size_t k;
	char tmp[41];
	char flag_comma = 0;

	size_t sz = 25*50 + hdr->NS * 16 * 50 + hdr->EVENT.N * 6 * 50;	// rough estimate of memory needed
	size_t c  = 0;
	*str = (char*) realloc(*str, sz);
#define STR ((*str)+c)

if (VERBOSE_LEVEL>7) fprintf(stdout, "asprintf_hdr2json: sz=%i\n", (int)sz);

	size_t NumberOfSweeps = (hdr->SPR*hdr->NRec > 0);
        size_t NumberOfUserSpecifiedEvents = 0;
        for (k = 0; k < hdr->EVENT.N; k++) {
                if (hdr->EVENT.TYP[k] < 255)
                        NumberOfUserSpecifiedEvents++;
                else if (hdr->EVENT.TYP[k]==0x7ffe)
                        NumberOfSweeps++;
        }

        c += sprintf(STR, "\n{\n");
        c += sprintf(STR, "\t\"TYPE\"\t: \"%s\",\n",GetFileTypeString(hdr->TYPE));
        c += sprintf(STR, "\t\"VERSION\"\t: %4.2f,\n",hdr->VERSION);

	c += sprintf(STR, "\t\"Filename\"\t: \"%s\",\n",hdr->FileName);
	c += sprintf(STR, "\t\"NumberOfChannels\"\t: %i,\n",(int)hdr->NS);
	c += sprintf(STR, "\t\"NumberOfRecords\"\t: %i,\n",(int)hdr->NRec);
	c += sprintf(STR, "\t\"SamplesPerRecords\"\t: %i,\n",(int)hdr->SPR);
	c += sprintf(STR, "\t\"NumberOfSamples\"\t: %i,\n",(int)(hdr->NRec*hdr->SPR));
	if ((0.0 <= hdr->SampleRate) && (hdr->SampleRate < INFINITY))
		c += sprintf(STR, "\t\"Samplingrate\"\t: %f,\n",hdr->SampleRate);
	snprintf_gdfdatetime(tmp, 40, hdr->T0);
	c += sprintf(STR, "\t\"StartOfRecording\"\t: \"%s\",\n",tmp);
	c += sprintf(STR, "\t\"TimezoneMinutesEastOfUTC\"\t: %i,\n", hdr->tzmin);
	c += sprintf(STR, "\t\"NumberOfSweeps\"\t: %d,\n",(unsigned)NumberOfSweeps);
	c += sprintf(STR, "\t\"NumberOfGroupsOrUserSpecifiedEvents\"\t: %d,\n", (unsigned)NumberOfUserSpecifiedEvents);

	c += sprintf(STR, "\t\"Patient\"\t: {\n");
	if (strlen(hdr->Patient.Name)) {
		c += sprintf(STR, "\t\t\"Name\"\t: \"%s\",\n", hdr->Patient.Name);
		char Name[MAX_LENGTH_NAME+1];
		strcpy(Name, hdr->Patient.Name);
		char *lastname = strtok(Name,"\x1f");
		char *firstname = strtok(NULL,"\x1f");
		char *secondlastname = strtok(NULL,"\x1f");
		c += sprintf(STR, "\t\t\"Lastname\"\t: \"%s\",\n", lastname);
		c += sprintf(STR, "\t\t\"Firstname\"\t: \"%s\",\n", firstname);
		c += sprintf(STR, "\t\t\"Second_Lastname\"\t: \"%s\",\n", secondlastname);
	}
	if (hdr->Patient.Id)   c += sprintf(STR, "\t\t\"Id\"\t: \"%s\",\n", hdr->Patient.Id);
	if (hdr->Patient.Weight) c += sprintf(STR, "\t\t\"Weight\"\t: \"%i kg\",\n", hdr->Patient.Weight);
	if (hdr->Patient.Height) c += sprintf(STR, "\t\t\"Height\"\t: \"%i cm\",\n", hdr->Patient.Height);
	if (hdr->Patient.Birthday>0) c += sprintf(STR, "\t\t\"Age\"\t: %i,\n", (int)((hdr->T0 - hdr->Patient.Birthday)/ldexp(365.25,32)) );
	c += sprintf(STR, "\t\t\"Gender\"\t: \"%s\"\n", hdr->Patient.Sex==1 ? "Male" : "Female");        // no comma at the end because its the last element
	c += sprintf(STR, "\t},\n");   // end-of-Patient

	if (hdr->ID.Manufacturer.Name || hdr->ID.Manufacturer.Model || hdr->ID.Manufacturer.Version || hdr->ID.Manufacturer.SerialNumber) {
		c += sprintf(STR,"\t\"Manufacturer\"\t: {\n");
		flag_comma = 0;
		if (hdr->ID.Manufacturer.Name) {
			c += sprintf(STR,"\t\t\"Name\"\t: \"%s\"", hdr->ID.Manufacturer.Name);
			flag_comma = 1;
		}
		if (hdr->ID.Manufacturer.Model) {
			if (flag_comma) c += sprintf(STR,",\n");
			c += sprintf(STR,"\t\t\"Model\"\t: \"%s\"", hdr->ID.Manufacturer.Model);
			flag_comma = 1;
		}
		if (hdr->ID.Manufacturer.Version) {
			if (flag_comma) c += sprintf(STR,",\n");
			c += sprintf(STR,"\t\t\"Version\"\t: \"%s\"", hdr->ID.Manufacturer.Version);
			flag_comma = 1;
		}
		if (hdr->ID.Manufacturer.SerialNumber) {
			if (flag_comma) c += sprintf(STR,",\n");
			c += sprintf(STR,"\t\t\"SerialNumber\"\t: \"%s\"", hdr->ID.Manufacturer.SerialNumber);
		}
		c += sprintf(STR,"\n\t},\n");   // end-of-Manufacturer
	}

        c += sprintf(STR,"\t\"CHANNEL\"\t: [");

if (VERBOSE_LEVEL>7) fprintf(stdout, "asprintf_hdr2json: count=%i\n", (int)c);


	for (k = 0; k < hdr->NS; k++) {

		if (sz < c + 1000) {
			// double allocated memory
			sz *= 2;
			*str = (char*) realloc(*str, sz);
		}

		CHANNEL_TYPE *hc = hdr->CHANNEL+k;
                if (k>0) c += sprintf(STR,",");
                c += sprintf(STR,"\n\t\t{\n");
		c += sprintf(STR,"\t\t\"ChannelNumber\"\t: %i,\n", (int)k+1);
		c += sprintf(STR,"\t\t\"Label\"\t: \"%s\",\n", hc->Label);
		double fs = hdr->SampleRate * hc->SPR/hdr->SPR;
		if ((0.0 <= fs) && (fs < INFINITY)) c += sprintf(STR, "\t\t\"Samplingrate\"\t: %f,\n", fs);
		if ( hc->Transducer && strlen(hc->Transducer) ) c += sprintf(STR,"\t\t\"Transducer\"\t: \"%s\",\n", hc->Transducer);
		if (!isnan(hc->PhysMax)) c += sprintf(STR,"\t\t\"PhysicalMaximum\"\t: %g,\n", hc->PhysMax);
		if (!isnan(hc->PhysMin)) c += sprintf(STR,"\t\t\"PhysicalMinimum\"\t: %g,\n", hc->PhysMin);
		if (!isnan(hc->DigMax))  c += sprintf(STR,"\t\t\"DigitalMaximum\"\t: %f,\n", hc->DigMax);
		if (!isnan(hc->DigMin))  c += sprintf(STR,"\t\t\"DigitalMinimum\"\t: %f,\n", hc->DigMin);
		if (!isnan(hc->Cal))     c += sprintf(STR,"\t\t\"scaling\"\t: %g,\n", hc->Cal);
		if (!isnan(hc->Off))     c += sprintf(STR,"\t\t\"offset\"\t: %g,\n", hc->Off);
		if (!isnan(hc->TOffset)) c += sprintf(STR,"\t\t\"TimeDelay\"\t: %g", hc->TOffset);
		uint8_t flag = (0 < hc->LowPass && hc->LowPass<INFINITY) | ((0 < hc->HighPass && hc->HighPass<INFINITY)<<1) | ((0 < hc->Notch && hc->Notch<INFINITY)<<2);
		if (flag) {
			c += sprintf(STR, "\t\t\"Filter\" : {\n");
			if (flag & 0x01) c += sprintf(STR, "\t\t\t\"Lowpass\"\t: %g%c\n", hc->LowPass, flag & 0x06 ? ',' : ' ');
			if (flag & 0x02) c += sprintf(STR, "\t\t\t\"Highpass\"\t: %g%c\n", hc->HighPass, flag & 0x04 ? ',' : ' ' );
			if (flag & 0x04) c += sprintf(STR, "\t\t\t\"Notch\"\t: %g\n", hc->Notch);
			c += sprintf(STR, "\n\t\t},\n");
		}
		switch (hc->PhysDimCode & 0xffe0) {
		case 4256: // Volt
			if (!isnan(hc->Impedance)) c += sprintf(STR, "\t\t\"Impedance\"\t: %g,\n", hc->Impedance);
			break;
		case 4288: // Ohm
			if (!isnan(hc->fZ)) c += sprintf(STR, "\t\t\"fZ\"\t: %g,\n", hc->fZ);
			break;
		}
		c += sprintf(STR,"\t\t\"PhysicalUnit\"\t: \"%s\"", PhysDim3(hc->PhysDimCode));
		c += sprintf(STR, "\n\t\t}");   // end-of-CHANNEL
	}
        c += sprintf(STR, "\n\t]");   // end-of-CHANNELS

if (VERBOSE_LEVEL>7) fprintf(stdout, "asprintf_hdr2json: count=%i\n", (int)c);

        if (hdr->EVENT.N>0) {
	    c += sprintf(STR, ",\n\t\"EVENT\"\t: [");
            flag_comma = 0;
            for (k = 0; k < hdr->EVENT.N; k++) {
	    	if ( hdr->EVENT.TYP[k] == 0 ) continue;

		if (sz < c + 1000) {
			// double allocated memory
			sz *= 2;
			*str = (char*) realloc(*str, sz);
		}

                if ( flag_comma ) c += sprintf(STR,",");
                c += sprintf(STR, "\n\t\t{\n");
                c += sprintf(STR, "\t\t\"TYP\"\t: \"0x%04x\",\n", hdr->EVENT.TYP[k]);
                c += sprintf(STR, "\t\t\"POS\"\t: %f", hdr->EVENT.POS[k]/hdr->EVENT.SampleRate);
                if (hdr->EVENT.CHN && hdr->EVENT.DUR) {
			if (hdr->EVENT.CHN[k])
	                        c += sprintf(STR, ",\n\t\t\"CHN\"\t: %d", hdr->EVENT.CHN[k]);
			if (hdr->EVENT.TYP[k] != 0x7fff)
	                        c += sprintf(STR, ",\n\t\t\"DUR\"\t: %f", hdr->EVENT.DUR[k]/hdr->EVENT.SampleRate);
                }

		if (hdr->EVENT.TimeStamp != NULL && hdr->EVENT.TimeStamp[k] != 0) {
			char buf[255];
			snprintf_gdfdatetime(buf, sizeof(buf), hdr->EVENT.TimeStamp[k]);
                        c += sprintf(STR,",\n\t\t\"TimeStamp\"\t: \"%s\"", buf);
		}

		if (hdr->EVENT.TYP[k] == 0x7fff) {
			// c += sprintf(STR, ",\n\t\t\"Description\"\t: \"[sparse sample]\"");
			typeof(hdr->NS) chan = hdr->EVENT.CHN[k] - 1;

			double val = dur2val(hdr->EVENT.DUR[k], hdr->CHANNEL[chan].GDFTYP);

			val *= hdr->CHANNEL[chan].Cal;
			val += hdr->CHANNEL[chan].Off;

			if (isfinite(val))
				c += sprintf(STR, ",\n\t\t\"Value\"\t: %g", val);        // no comma at the end because its the last element
		}
		else {
			const char *tmpstr = GetEventDescription(hdr,k);
			if (tmpstr != NULL)
				c += sprintf(STR, ",\n\t\t\"Description\"\t: \"%s\"", tmpstr);        // no comma at the end because its the last element
		}

                c += sprintf(STR, "\n\t\t}");
		flag_comma = 1;
            }
            c += sprintf(STR, "\n\t]");   // end-of-EVENT
        }
        c += sprintf(STR, "\n}\n");

if (VERBOSE_LEVEL>7) fprintf(stdout, "asprintf_hdr2json: count=%i\n", (int)c);

        return (0);
#undef STR
}


/****************************************************************************/
/**                     HDR2ASCII                                          **/
/**	displaying header information                                      **/
/****************************************************************************/
#if (BIOSIG_VERSION < 10500)
// for backwards compatibility
ATT_DEPREC int hdr2json( HDRTYPE *hdr, FILE *fid)  {
	return fprintf_hdr2json(fid, hdr);
} // deprecated since Oct 2012, v1.4.0
#endif

int fprintf_hdr2json(FILE *fid, HDRTYPE* hdr)
{
        size_t k;
	char tmp[41];
	char flag_comma = 0;

	size_t NumberOfSweeps = (hdr->SPR*hdr->NRec > 0);
        size_t NumberOfUserSpecifiedEvents = 0;
        for (k = 0; k < hdr->EVENT.N; k++) {
                if (hdr->EVENT.TYP[k] < 255)
                        NumberOfUserSpecifiedEvents++;
                else if (hdr->EVENT.TYP[k]==0x7ffe)
                        NumberOfSweeps++;
        }

        fprintf(fid,"{\n");
        fprintf(fid,"\t\"TYPE\"\t: \"%s\",\n",GetFileTypeString(hdr->TYPE));
        fprintf(fid,"\t\"VERSION\"\t: %4.2f,\n",hdr->VERSION);

	fprintf(fid,"\t\"Filename\"\t: \"%s\",\n",hdr->FileName);
	fprintf(fid,"\t\"NumberOfChannels\"\t: %i,\n",(int)hdr->NS);
	fprintf(fid,"\t\"NumberOfRecords\"\t: %i,\n",(int)hdr->NRec);
	fprintf(fid,"\t\"SamplesPerRecords\"\t: %i,\n",(int)hdr->SPR);
	fprintf(fid,"\t\"NumberOfSamples\"\t: %i,\n",(int)(hdr->NRec*hdr->SPR));
	if ((0.0 <= hdr->SampleRate) && (hdr->SampleRate < INFINITY))
		fprintf(fid,"\t\"Samplingrate\"\t: %f,\n", hdr->SampleRate);

	snprintf_gdfdatetime(tmp, 40, hdr->T0);
	fprintf(fid,"\t\"StartOfRecording\"\t: \"%s\",\n",tmp);
	fprintf(fid,"\t\"TimezoneMinutesEastOfUTC\"\t: %i,\n", hdr->tzmin);
	fprintf(fid,"\t\"NumberOfSweeps\"\t: %d,\n",(unsigned)NumberOfSweeps);
	fprintf(fid,"\t\"NumberOfGroupsOrUserSpecifiedEvents\"\t: %d,\n",(unsigned)NumberOfUserSpecifiedEvents);

	fprintf(fid,"\t\"Patient\"\t: {\n");
	if (strlen(hdr->Patient.Name)) {
		fprintf(fid, "\t\t\"Name\"\t: \"%s\",\n", hdr->Patient.Name);
		char Name[MAX_LENGTH_NAME+1];
		strcpy(Name, hdr->Patient.Name);
		char *lastname = strtok(Name,"\x1f");
		char *firstname = strtok(NULL,"\x1f");
		char *secondlastname = strtok(NULL,"\x1f");
		fprintf(fid, "\t\t\"Lastname\"\t: \"%s\",\n", lastname);
		fprintf(fid, "\t\t\"Firstname\"\t: \"%s\",\n", firstname);
		fprintf(fid, "\t\t\"Second_Lastname\"\t: \"%s\",\n", secondlastname);
	}
	if (hdr->Patient.Id)     fprintf(fid,"\t\t\"Id\"\t: \"%s\",\n", hdr->Patient.Id);
	if (hdr->Patient.Weight) fprintf(fid,"\t\t\"Weight\"\t: \"%i kg\",\n", hdr->Patient.Weight);
	if (hdr->Patient.Height) fprintf(fid,"\t\t\"Height\"\t: \"%i cm\",\n", hdr->Patient.Height);
	if (hdr->Patient.Birthday>0) fprintf(fid,"\t\t\"Age\"\t: %i,\n", (int)((hdr->T0 - hdr->Patient.Birthday)/ldexp(365.25,32)) );
	fprintf(fid,"\t\t\"Gender\"\t: \"%s\"\n", hdr->Patient.Sex==1 ? "Male" : "Female");        // no comma at the end because its the last element
	fprintf(fid,"\t},\n");   // end-of-Patient

	if (hdr->ID.Manufacturer.Name || hdr->ID.Manufacturer.Model || hdr->ID.Manufacturer.Version || hdr->ID.Manufacturer.SerialNumber) {
		fprintf(fid,"\t\"Manufacturer\"\t: {\n");
		flag_comma = 0;
		if (hdr->ID.Manufacturer.Name) {
			fprintf(fid,"\t\t\"Name\"\t: \"%s\"", hdr->ID.Manufacturer.Name);
			flag_comma = 1;
		}
		if (hdr->ID.Manufacturer.Model) {
			if (flag_comma) fprintf(fid,",\n");
			fprintf(fid,"\t\t\"Model\"\t: \"%s\"", hdr->ID.Manufacturer.Model);
			flag_comma = 1;
		}
		if (hdr->ID.Manufacturer.Version) {
			if (flag_comma) fprintf(fid,",\n");
			fprintf(fid,"\t\t\"Version\"\t: \"%s\"", hdr->ID.Manufacturer.Version);
			flag_comma = 1;
		}
		if (hdr->ID.Manufacturer.SerialNumber) {
			if (flag_comma) fprintf(fid,",\n");
			fprintf(fid,"\t\t\"SerialNumber\"\t: \"%s\"", hdr->ID.Manufacturer.SerialNumber);        // no comma at the end because its the last element
		}
		fprintf(fid,"\n\t},\n");   // end-of-Manufacturer
	}

        fprintf(fid,"\t\"CHANNEL\"\t: [");
	for (k = 0; k < hdr->NS; k++) {
		CHANNEL_TYPE *hc = hdr->CHANNEL+k;
                if (k>0) fprintf(fid,",");
                fprintf(fid,"\n\t\t{\n");
		fprintf(fid,"\t\t\"ChannelNumber\"\t: %i,\n", (int)k+1);
		fprintf(fid,"\t\t\"Label\"\t: \"%s\",\n", hc->Label);
		double fs = hdr->SampleRate * hc->SPR/hdr->SPR;
		if ((0.0 <= fs) && (fs < INFINITY)) fprintf(fid,"\t\t\"Samplingrate\"\t: %f,\n", fs);
		if ( hc->Transducer && strlen(hc->Transducer) ) fprintf(fid,"\t\t\"Transducer\"\t: \"%s\",\n", hc->Transducer);
		if (!isnan(hc->PhysMax)) fprintf(fid,"\t\t\"PhysicalMaximum\"\t: %g,\n", hc->PhysMax);
		if (!isnan(hc->PhysMin)) fprintf(fid,"\t\t\"PhysicalMinimum\"\t: %g,\n", hc->PhysMin);
		if (!isnan(hc->DigMax))  fprintf(fid,"\t\t\"DigitalMaximum\"\t: %f,\n", hc->DigMax);
		if (!isnan(hc->DigMin))  fprintf(fid,"\t\t\"DigitalMinimum\"\t: %f,\n", hc->DigMin);
		if (!isnan(hc->Cal))     fprintf(fid,"\t\t\"scaling\"\t: %g,\n", hc->Cal);
		if (!isnan(hc->Off))     fprintf(fid,"\t\t\"offset\"\t: %g,\n", hc->Off);
		if (!isnan(hc->TOffset)) fprintf(fid,"\t\t\"TimeDelay\"\t: %g,\n", hc->TOffset);
		uint8_t flag = (0 < hc->LowPass && hc->LowPass<INFINITY) | ((0 < hc->HighPass && hc->HighPass<INFINITY)<<1) | ((0 < hc->Notch && hc->Notch<INFINITY)<<2);
		if (flag) {
			fprintf(fid,"\t\t\"Filter\" : {\n");
			if (flag & 0x01) fprintf(fid,"\t\t\t\"Lowpass\"\t: %g%c\n",hc->LowPass, flag & 0x06 ? ',' : ' ');
			if (flag & 0x02) fprintf(fid,"\t\t\t\"Highpass\"\t: %g%c\n",hc->HighPass, flag & 0x04 ? ',' : ' ' );
			if (flag & 0x04) fprintf(fid,"\t\t\t\"Notch\"\t: %g\n",hc->Notch);
			fprintf(fid,"\n\t\t},\n");
		}
		switch (hc->PhysDimCode & 0xffe0) {
		case 4256: // Volt
			if (!isnan(hc->Impedance)) fprintf(fid,"\t\t\"Impedance\"\t: %g,\n", hc->Impedance);
			break;
		case 4288: // Ohm
			if (!isnan(hc->fZ)) fprintf(fid,"\t\t\"fZ\"\t: %g,\n", hc->fZ);
			break;
		}
		fprintf(fid,"\t\t\"PhysicalUnit\"\t: \"%s\"", PhysDim3(hc->PhysDimCode));	// no comma at the end because its the last element
		fprintf(fid,"\n\t\t}");   // end-of-CHANNEL
	}
        fprintf(fid,"\n\t]");   // end-of-CHANNELS

        if (hdr->EVENT.N>0) {
            flag_comma = 0;
            fprintf(fid,",\n\t\"EVENT\"\t: [");
            for (k = 0; k < hdr->EVENT.N; k++) {
                if ( hdr->EVENT.TYP[k] == 0 ) continue;
                if ( flag_comma ) fprintf(fid,",");
                fprintf(fid,"\n\t\t{\n");
                fprintf(fid,"\t\t\"TYP\"\t: \"0x%04x\",\n", hdr->EVENT.TYP[k]);
                fprintf(fid,"\t\t\"POS\"\t: %f", hdr->EVENT.POS[k]/hdr->EVENT.SampleRate);
                if (hdr->EVENT.CHN && hdr->EVENT.DUR) {
			if (hdr->EVENT.CHN[k])
	                        fprintf(fid,",\n\t\t\"CHN\"\t: %d", hdr->EVENT.CHN[k]);
			if (hdr->EVENT.TYP[k] != 0x7fff)
	                        fprintf(fid,",\n\t\t\"DUR\"\t: %f", hdr->EVENT.DUR[k]/hdr->EVENT.SampleRate);
                }

		if (hdr->EVENT.TimeStamp != NULL && hdr->EVENT.TimeStamp[k] != 0) {
			char buf[255];
			snprintf_gdfdatetime(buf,sizeof(buf), hdr->EVENT.TimeStamp[k]);
                        fprintf(fid,",\n\t\t\"TimeStamp\"\t: \"%s\"", buf);
		}

		if ((hdr->EVENT.TYP[k] == 0x7fff) && (hdr->TYPE==GDF)) {
			//fprintf(fid,"\t\t\"Description\"\t: [neds]\n");        // no comma at the end because its the last element

			//fprintf(fid,",\n\t\t\"Description\"\t: \"[sparse sample]\"");
			typeof(hdr->NS) chan = hdr->EVENT.CHN[k] - 1;

			double val = dur2val(hdr->EVENT.DUR[k], hdr->CHANNEL[chan].GDFTYP);

			val *= hdr->CHANNEL[chan].Cal;
			val += hdr->CHANNEL[chan].Off;

			if (isfinite(val))
				fprintf(fid,",\n\t\t\"Value\"\t: %g", val);        // no comma at the end because its the last element
		}
		else {
			const char *str = GetEventDescription(hdr,k);
			if (str != NULL)
				fprintf(fid,",\n\t\t\"Description\"\t: \"%s\"",str);        // no comma at the end because its the last element
		}

		fprintf(fid,"\n\t\t}");
		flag_comma = 1;
            }
            fprintf(fid,"\n\t]");   // end-of-EVENT
        }
        fprintf(fid,"\n}\n");
        return (0);
}



/****************************************************************************/
/**                     HDR2ASCII                                          **/
/**	displaying header information                                      **/
/****************************************************************************/
int hdr2ascii(HDRTYPE* hdr, FILE *fid, int VERBOSE)
{
	CHANNEL_TYPE* 	cp;
	struct tm  	*T0;
	float		age;

	if (VERBOSE==7) {
		char tmp[60];
		snprintf_gdfdatetime(tmp, sizeof(tmp), hdr->T0);
		fprintf(fid,"\tStartOfRecording: %s\nbci2000: %p\n",tmp,hdr->AS.bci2000);
		return(0);
	}

	if (VERBOSE==-1) {
		return(fprintf_hdr2json(fid, hdr));
	}

	if (VERBOSE>0) {
		/* demographic information */
		fprintf(fid,"\n===========================================\n[FIXED HEADER]\n");
//		fprintf(fid,"\nPID:\t|%s|\nPatient:\n",hdr->AS.PID);
		fprintf(fid,   "Recording:\n\tID              : %s\n",hdr->ID.Recording);
		fprintf(fid,               "\tInstitution     : %s\n",hdr->ID.Hospital);
		fprintf(fid,               "\tTechnician      : %s\t# default: localuser\n",hdr->ID.Technician);
		char tmp[60];
		strncpy(tmp,(char*)&hdr->ID.Equipment,8);
		tmp[8] = 0;
		fprintf(fid,               "\tEquipment       : %s\n",tmp);
		if (VERBOSE_LEVEL>8)
			fprintf(fid,       "\t                  %#.16"PRIx64"\n",(uint64_t)hdr->ID.Equipment);
		uint8_t k,IPv6=0;
		for (k=4; k<16; k++) IPv6 |= hdr->IPaddr[k];
		if (IPv6) fprintf(fid,     "\tIPv6 address    : %02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x",hdr->IPaddr[0],hdr->IPaddr[1],hdr->IPaddr[2],hdr->IPaddr[3],hdr->IPaddr[4],hdr->IPaddr[5],hdr->IPaddr[6],hdr->IPaddr[7],hdr->IPaddr[8],hdr->IPaddr[9],hdr->IPaddr[10],hdr->IPaddr[11],hdr->IPaddr[12],hdr->IPaddr[13],hdr->IPaddr[14],hdr->IPaddr[15]);
		else fprintf(fid,          "\tIPv4 address    : %u.%u.%u.%u",hdr->IPaddr[0],hdr->IPaddr[1],hdr->IPaddr[2],hdr->IPaddr[3]);

		fprintf(fid,"\t # default:local host\nManufacturer:\n\tName            : %s\n",hdr->ID.Manufacturer.Name);
		fprintf(fid,               "\tModel           : %s\n",hdr->ID.Manufacturer.Model);
		fprintf(fid,               "\tVersion         : %s\n",hdr->ID.Manufacturer.Version);
		fprintf(fid,               "\tSerialNumber    : %s\n",hdr->ID.Manufacturer.SerialNumber);
		fprintf(fid,     "Patient:\n\tID              : %s\n",hdr->Patient.Id);
		if (strlen(hdr->Patient.Name)) {
			fprintf(fid,               "\tName            : %s\n",hdr->Patient.Name);
			char Name[MAX_LENGTH_NAME+1];
			strcpy(Name, hdr->Patient.Name);
			char *lastname = strtok(Name,"\x1f");
			char *firstname = strtok(NULL,"\x1f");
			char *secondlastname = strtok(NULL,"\x1f");
			fprintf(fid,"\t\tLastname        : %s\n",lastname);
			fprintf(fid,"\t\tFirstname       : %s\n",firstname);
			fprintf(fid,"\t\tSecondLastName  : %s\n",secondlastname);
		}

		if (hdr->Patient.Birthday>0)
			age = (hdr->T0 - hdr->Patient.Birthday)/ldexp(365.25,32);
		else
			age = NAN;

		if (hdr->Patient.Height)
			fprintf(fid,"\tHeight          : %i cm\n",hdr->Patient.Height);
		if (hdr->Patient.Height)
			fprintf(stdout,"\tWeight          : %i kg\n",hdr->Patient.Weight);

		const char *Gender[] = {"unknown","male","female","unknown"};
		const char *EyeImpairment[] = {"unknown","no","yes","corrected"};
		const char *HeartImpairment[] = {"unknown","no","yes","pacemaker"};
		fprintf(fid,"\tGender          : %s\n",Gender[hdr->Patient.Sex]);
		fprintf(fid,"\tEye Impairment  : %s\n",EyeImpairment[hdr->Patient.Impairment.Visual]);
		fprintf(fid,"\tHeart Impairment: %s\n",HeartImpairment[hdr->Patient.Impairment.Heart]);
		if (hdr->Patient.Birthday) {
			snprintf_gdfdatetime(tmp, sizeof(tmp), hdr->T0);
			fprintf(fid,"\tAge             : %4.1f years\n\tBirthday        : (%.6f) %s ",age,ldexp(hdr->Patient.Birthday,-32),tmp);
		}
		else
			fprintf(fid,"\tAge             : ----\n\tBirthday        : unknown\n");

		snprintf_gdfdatetime(tmp, sizeof(tmp), hdr->T0);
		fprintf(fid,"\tStartOfRecording: (%.6f) %s",ldexp(hdr->T0,-32),tmp);
		fprintf(fid,"\tTimezone        : %+i min\n\n", hdr->tzmin);
		if (hdr->AS.bci2000 != NULL) {
		   if (VERBOSE < 4) {
			size_t c = min(39,strcspn(hdr->AS.bci2000,"\xa\xd"));
			strncpy(tmp, hdr->AS.bci2000, c); tmp[c]=0;
			fprintf(fid,"BCI2000 [%i]\t\t: <%s...>\n",(int)strlen(hdr->AS.bci2000),tmp);
		   }
		   else {
			fprintf(fid,"BCI2000 [%i]:\n%s\n",(int)strlen(hdr->AS.bci2000),hdr->AS.bci2000);
		   }
		}
		fprintf(fid,"bpb=%i\n",hdr->AS.bpb);
		fprintf(fid,"row-based=%i\n",hdr->FLAG.ROW_BASED_CHANNELS);
		fprintf(fid,"uncalib  =%i\n",hdr->FLAG.UCAL);
		fprintf(fid,"OFdetect =%i\n",hdr->FLAG.OVERFLOWDETECTION);

	}

	if (VERBOSE>1) {
		/* display header information */
		fprintf(fid,"FileName:\t%s\nType    :\t%s\nVersion :\t%4.2f\nHeadLen :\t%i\n",hdr->FileName,GetFileTypeString(hdr->TYPE),hdr->VERSION,hdr->HeadLen);
//		fprintf(fid,"NoChannels:\t%i\nSPR:\t\t%i\nNRec:\t\t%Li\nDuration[s]:\t%u/%u\nFs:\t\t%f\n",hdr->NS,hdr->SPR,hdr->NRec,hdr->Dur[0],hdr->Dur[1],hdr->SampleRate);
		fprintf(fid,"NoChannels:\t%i\nSPR:\t\t%i\nNRec:\t\t%li\nFs:\t\t%f\n",hdr->NS,hdr->SPR,(long)hdr->NRec,hdr->SampleRate);
		fprintf(fid,"Events/Annotations:\t%i\nEvents/SampleRate:\t%f\n",hdr->EVENT.N,hdr->EVENT.SampleRate);
	}

	if (VERBOSE>2) {
		/* channel settings */
		fprintf(fid,"\n[CHANNEL HEADER] %p",hdr->CHANNEL);
		fprintf(fid,"\nNo  LeadId Label\tFs[Hz]\tSPR\tGDFTYP\tCal\tOff\tPhysDim\tPhysMax \tPhysMin \tDigMax  \tDigMin  \tHighPass\tLowPass \tNotch   \tdelay [s]\tX\tY\tZ");
		size_t k;
#ifdef CHOLMOD_H
                typeof(hdr->NS) NS = hdr->NS;
                if (hdr->Calib) NS += hdr->Calib->ncol;
		for (k=0; k<NS; k++) {
		        if (k<hdr->NS)
        			cp = hdr->CHANNEL+k;
        		else
        			cp = hdr->rerefCHANNEL + k - hdr->NS;
#else
		for (k=0; k<hdr->NS; k++) {
       			cp = hdr->CHANNEL+k;
#endif

			const char *tmpstr = cp->Label;
			if (tmpstr==NULL || strlen(tmpstr)==0) tmpstr = LEAD_ID_TABLE[cp->LeadIdCode];

			fprintf(fid,"\n#%02i: %3i %i %-17s\t%5f %5i", (int)k+1, cp->LeadIdCode, cp->bi8, tmpstr, cp->SPR*hdr->SampleRate/hdr->SPR, cp->SPR);

			if      (cp->GDFTYP<20)  fprintf(fid," %s  ",gdftyp_string[cp->GDFTYP]);
			else if (cp->GDFTYP>511) fprintf(fid, " bit%i  ", cp->GDFTYP-511);
			else if (cp->GDFTYP>255) fprintf(fid, " bit%i  ", cp->GDFTYP-255);

			tmpstr = PhysDim3(cp->PhysDimCode);
			if (tmpstr==NULL) tmpstr="\0";
			fprintf(fid,"%e %e %s\t%g\t%g\t%5f\t%5f\t%5f\t%5f\t%5f\t%5g\t%5f\t%5f\t%5f",
				cp->Cal, cp->Off, tmpstr,
				cp->PhysMax, cp->PhysMin, cp->DigMax, cp->DigMin,cp->HighPass,cp->LowPass,cp->Notch,cp->TOffset,
				cp->XYZ[0],cp->XYZ[1],cp->XYZ[2]);
			//fprintf(fid,"\t %3i", cp->SPR);
		}
	}

	if (VERBOSE>3) {
		/* channel settings */
		fprintf(fid,"\n\n[EVENT TABLE %i] N=%i Fs=%f", (hdr->EVENT.TimeStamp!=NULL) + (hdr->EVENT.TYP!=NULL) + (hdr->EVENT.POS!=NULL) + (hdr->EVENT.CHN!=NULL) + (hdr->EVENT.DUR!=NULL), hdr->EVENT.N,hdr->EVENT.SampleRate);
		fprintf(fid,"\nNo\tTYP\tPOS\tCHN\tDUR/VAL\tDesc");

		size_t k;
		for (k=0; k<hdr->EVENT.N; k++) {
			fprintf(fid,"\n%5i\t0x%04x\t%d",(int)(k+1),hdr->EVENT.TYP[k],hdr->EVENT.POS[k]);

			if (hdr->EVENT.TimeStamp != NULL && hdr->EVENT.TimeStamp[k] != 0) {
				char buf[255];
				snprintf_gdfdatetime(buf,sizeof(buf), hdr->EVENT.TimeStamp[k]);
				fprintf(fid,"\t%s",buf);
			}

			if (hdr->EVENT.TYP[k] == 0x7fff)
				fprintf(fid,"\t%d",hdr->EVENT.CHN[k]);
			else if (hdr->EVENT.DUR != NULL)
				fprintf(fid,"\t%d\t%5d",hdr->EVENT.CHN[k],hdr->EVENT.DUR[k]);

			if ((hdr->EVENT.TYP[k] == 0x7fff) && (hdr->TYPE==GDF)) {
				typeof(hdr->NS) chan = hdr->EVENT.CHN[k]-1;

				double val = dur2val(hdr->EVENT.DUR[k], hdr->CHANNEL[chan].GDFTYP);

				val *= hdr->CHANNEL[chan].Cal;
				val += hdr->CHANNEL[chan].Off;

				fprintf(fid, "\t%g %s\t## sparse sample", val, PhysDim3(hdr->CHANNEL[chan].PhysDimCode));        // no comma at the end because its the last element
			}
			else {
				const char *str = GetEventDescription(hdr,k);
				if (str) fprintf(fid,"\t\t%s",str);
			}
		}
	}

	if (VERBOSE>4) {
		const char* StatusString[] = {"Original (not overread)", "Confirmed", "Overread (not confirmed)", "unknown"};
#if (BIOSIG_VERSION >= 10500)

		if (hdr->SCP.Section7) {
			fprintf(stdout,"\n\n=== SCP Section 7: Global measurements ===\n");
			fprintf(stdout,"\n\n  (report of this section is not implemented yet \n");
		}
		if (hdr->SCP.Section8) {
			struct tm t;
			t.tm_year = leu16p(hdr->SCP.Section8+1)-1900;
			t.tm_mon  = hdr->SCP.Section8[3]-1;
			t.tm_mday = hdr->SCP.Section8[4];
			t.tm_hour = hdr->SCP.Section8[5];
			t.tm_min  = hdr->SCP.Section8[6];
			t.tm_sec  = hdr->SCP.Section8[7];
			uint8_t NumberOfStatements = hdr->SCP.Section8[8];

			fprintf(stdout,"\n\n=== SCP Section 8: Storage of full text interpretive statements ===\n");
			fprintf(stdout,"Report %04i-%02i-%02i %02ih%02im%02is (Status=%s) %i statements\n",t.tm_year+1900,t.tm_mon+1,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec,StatusString[min(hdr->SCP.Section8[0],3)], NumberOfStatements);

			uint32_t curSectPos = 9;
			uint8_t k;
			for (k=0; k < NumberOfStatements;k++) {
				if (curSectPos+3 > hdr->SCP.Section8Length) break;
				fprintf(stdout, "%s\n", (char*)(hdr->SCP.Section8+curSectPos+3));
				curSectPos += 3+leu16p(hdr->SCP.Section8+curSectPos+1);
			}
		}
		if (hdr->SCP.Section9) {
			struct tm t;
			t.tm_year = leu16p(hdr->SCP.Section9+1)-1900;
			t.tm_mon  = hdr->SCP.Section9[3]-1;
			t.tm_mday = hdr->SCP.Section9[4];
			t.tm_hour = hdr->SCP.Section9[5];
			t.tm_min  = hdr->SCP.Section9[6];
			t.tm_sec  = hdr->SCP.Section9[7];
			uint8_t NumberOfStatements = hdr->SCP.Section9[8];

			fprintf(stdout,"\n\n=== SCP Section 9: Storing manufacturer specific interpretive statements and data related to the overreading trail ===\n");
			fprintf(stdout,"Report %04i-%02i-%02i %02ih%02im%02is (Status=%s) %i statements\n",t.tm_year+1900,t.tm_mon+1,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec,StatusString[min(hdr->SCP.Section8[0],3)], NumberOfStatements);

			uint32_t curSectPos = 9;
			uint8_t k;
			for (k=0; k < NumberOfStatements;k++) {
				if (curSectPos+3 > hdr->SCP.Section9Length) break;
				fprintf(stdout, "%s\n", (char*)(hdr->SCP.Section9+curSectPos+3));
				curSectPos += 3+leu16p(hdr->SCP.Section9+curSectPos+1);
			}
		}
		if (hdr->SCP.Section10) {
			fprintf(stdout,"\n\n=== SCP Section 10: Lead measurement block ===\n");
			fprintf(stdout,"\n\n  (report of this section is not implemented yet \n");
		}
		if (hdr->SCP.Section11) {
			struct tm t;
			t.tm_year = leu16p(hdr->SCP.Section11+1)-1900;
			t.tm_mon  = hdr->SCP.Section11[3]-1;
			t.tm_mday = hdr->SCP.Section11[4];
			t.tm_hour = hdr->SCP.Section11[5];
			t.tm_min  = hdr->SCP.Section11[6];
			t.tm_sec  = hdr->SCP.Section11[7];
			uint8_t NumberOfStatements = hdr->SCP.Section11[8];

			fprintf(stdout,"\n\n=== SCP Section 11: Storage of the universal ECG interpretive statement codes ===\n");
			fprintf(stdout,"Report %04i-%02i-%02i %02ih%02im%02is (Status=%s) %i statements\n",t.tm_year+1900,t.tm_mon+1,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec,StatusString[min(hdr->SCP.Section8[0],3)], NumberOfStatements);

			uint32_t curSectPos = 9;
			uint8_t k;
			for (k=0; k < NumberOfStatements;k++) {
				if (curSectPos+3 > hdr->SCP.Section11Length) break;
				fprintf(stdout, "%s\n", (char*)(hdr->SCP.Section11+curSectPos+3));
				curSectPos += 3+leu16p(hdr->SCP.Section11+curSectPos+1);
			}
		}
#else
		if (hdr->aECG && (hdr->TYPE==SCP_ECG)) {
			struct aecg* aECG = (struct aecg*)hdr->aECG;
			fprintf(stdout,"\nInstitution Number: %i\n",aECG->Section1.Tag14.INST_NUMBER);
			fprintf(stdout,"DepartmentNumber : %i\n",aECG->Section1.Tag14.DEPT_NUMBER);
			fprintf(stdout,"Device Id        : %i\n",aECG->Section1.Tag14.DEVICE_ID);
			fprintf(stdout,"Device Type      : %i\n",aECG->Section1.Tag14.DeviceType);
			fprintf(stdout,"Manufacture code : %i\n",aECG->Section1.Tag14.MANUF_CODE);
			fprintf(stdout,"MOD_DESC         : %s\n",aECG->Section1.Tag14.MOD_DESC);
			fprintf(stdout,"Version          : %i\n",aECG->Section1.Tag14.VERSION);
			fprintf(stdout,"ProtCompLevel    : %i\n",aECG->Section1.Tag14.PROT_COMP_LEVEL);
			fprintf(stdout,"LangSuppCode     : %i\n",aECG->Section1.Tag14.LANG_SUPP_CODE);
			fprintf(stdout,"ECG_CAP_DEV      : %i\n",aECG->Section1.Tag14.ECG_CAP_DEV);
			fprintf(stdout,"Mains Frequency  : %i\n",aECG->Section1.Tag14.MAINS_FREQ);
/*
			fprintf(stdout,"ANAL_PROG_REV_NUM    : %s\n",aECG->Section1.Tag14.ANAL_PROG_REV_NUM);
			fprintf(stdout,"SERIAL_NUMBER_ACQ_DEV: %s\n",aECG->Section1.Tag14.SERIAL_NUMBER_ACQ_DEV);
			fprintf(stdout,"ACQ_DEV_SYS_SW_ID    : %i\n",aECG->Section1.Tag14.ACQ_DEV_SYS_SW_ID);
			fprintf(stdout,"ACQ_DEV_SCP_SW       : %i\n",aECG->Section1.Tag14.ACQ_DEV_SCP_SW);
*/
			fprintf(stdout,"ACQ_DEV_MANUF        : %s\n",aECG->Section1.Tag14.ACQ_DEV_MANUF);
			fprintf(stdout,"Compression  HUFFMAN : %i\n",aECG->FLAG.HUFFMAN);
			fprintf(stdout,"Compression  REF-BEAT: %i\n",aECG->FLAG.REF_BEAT);
			fprintf(stdout,"Compression  BIMODAL : %i\n",aECG->FLAG.BIMODAL);
			fprintf(stdout,"Compression  DIFF    : %i\n",aECG->FLAG.DIFF);
			if ((aECG->systolicBloodPressure > 0.0) || (aECG->diastolicBloodPressure > 0.0))
				fprintf(stdout,"Blood pressure (systolic/diastolic) : %3.0f/%3.0f mmHg\n",aECG->systolicBloodPressure,aECG->diastolicBloodPressure);


			uint8_t k;
			if (aECG->Section8.NumberOfStatements>0) {
				fprintf(stdout,"\n\nReport %04i-%02i-%02i %02ih%02im%02is (Status=%s)\n",aECG->Section8.t.tm_year+1900,aECG->Section8.t.tm_mon+1,aECG->Section8.t.tm_mday,aECG->Section8.t.tm_hour,aECG->Section8.t.tm_min,aECG->Section8.t.tm_sec,StatusString[min(aECG->Section8.Confirmed,3)]);
				for (k=0; k<aECG->Section8.NumberOfStatements;k++) {
					fprintf(stdout,"%s\n",aECG->Section8.Statements[k]);
				}
			}

			if ( aECG->Section11.NumberOfStatements > 0 ) {
				fprintf(stdout,"\n\nReport %04i-%02i-%02i %02ih%02im%02is (Status=%s)\n",aECG->Section11.t.tm_year+1900,aECG->Section11.t.tm_mon+1,aECG->Section11.t.tm_mday,aECG->Section11.t.tm_hour,aECG->Section11.t.tm_min,aECG->Section11.t.tm_sec,StatusString[min(aECG->Section11.Confirmed,3)]);
				for (k=0; k<aECG->Section11.NumberOfStatements;k++) {
					fprintf(stdout,"%s\n",aECG->Section11.Statements[k]);
				}
			}

			fprintf(stdout,"\n\nSection9:\n%s\n\n",aECG->Section9.StartPtr);
		}
#endif  // BIOSIGVERSION < 10500
	}
	fprintf(fid,"\n\n");

	return(0);
} 	/* end of HDR2ASCII */


/****************************************************************************/
/**                                                                        **/
/**                               EOF                                      **/
/**                                                                        **/
/****************************************************************************/

