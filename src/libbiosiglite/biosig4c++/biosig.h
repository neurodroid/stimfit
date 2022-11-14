/*

    Copyright (C) 2012-2018 Alois Schloegl <alois.schloegl@gmail.com>
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


#ifndef __LIBBIOSIG2_H__
#define __LIBBIOSIG2_H__

#include "biosig-dev.h"

#define BIOSIG_FLAG_COMPRESSION        0x0001
#define BIOSIG_FLAG_UCAL               0x0002
#define BIOSIG_FLAG_OVERFLOWDETECTION  0x0004
#define BIOSIG_FLAG_ROW_BASED_CHANNELS 0x0008


#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif

/****************************************************************************/
/**                                                                        **/
/**                     EXPORTED FUNCTIONS : Level 1                       **/
/**                                                                        **/
/****************************************************************************/

uint32_t get_biosig_version (void);
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


int 	sseek(HDRTYPE* hdr, ssize_t offset, int whence);
/*	positions file pointer
 *
 *	Currently, this function is meaning less because sread requires always the start value
 --------------------------------------------------------------- */


ssize_t stell(HDRTYPE* hdr);
/*	returns position of file point in segments
 --------------------------------------------------------------- */

#ifndef  ONLYGDF
ATT_DEPREC int serror(void);
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

#ifndef __PHYSICALUNITS_H__
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
#endif

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
HDRTYPE* constructHDR(const unsigned NS, const unsigned N_EVENT);
/* 	allocates memory initializes header HDR of type HDRTYPE
	with NS channels an N_EVENT event elements
 --------------------------------------------------------------- */
void 	 destructHDR(HDRTYPE* hdr);
/* 	destroys the header *hdr and frees allocated memory
 --------------------------------------------------------------- */



/****************************************************************************/
/**                                                                        **/
/**                     EXPORTED FUNCTIONS : Level 2                       **/
/**                                                                        **/
/****************************************************************************/


/* =============================================================
	setter and getter functions for accessing fields of HDRTYPE
	these functions are currently experimential and are likely to change
   ============================================================= */

/* get, set and check function of filetype */
enum FileFormat biosig_get_filetype(HDRTYPE *hdr);
int biosig_set_filetype(HDRTYPE *hdr, enum FileFormat format);
#define biosig_check_filetype(a,b) (biosig_get_filetype(a)==b)

ATT_DEPREC int biosig_set_flags(HDRTYPE *hdr, char compression, char ucal, char overflowdetection);
int biosig_get_flag(HDRTYPE *hdr, unsigned flags);
int biosig_set_flag(HDRTYPE *hdr, unsigned flags);
int biosig_reset_flag(HDRTYPE *hdr, unsigned flags);

int biosig_set_targetsegment(HDRTYPE *hdr, unsigned targetsegment);
int biosig_get_targetsegment(HDRTYPE *hdr);

const char* biosig_get_filename(HDRTYPE *hdr);
float biosig_get_version(HDRTYPE *hdr);

int biosig_set_segment_selection(HDRTYPE *hdr, int k, uint32_t argSweepSel);
uint32_t* biosig_get_segment_selection(HDRTYPE *hdr);


// returns error message in memory allocated with strdup
int biosig_check_error(HDRTYPE *hdr);
char *biosig_get_errormsg(HDRTYPE *hdr);

long biosig_get_number_of_channels(HDRTYPE *hdr);
size_t biosig_get_number_of_samples(HDRTYPE *hdr);
ATT_DEPREC size_t biosig_get_number_of_samples_per_record(HDRTYPE *hdr);
size_t biosig_get_number_of_records(HDRTYPE *hdr);
size_t biosig_get_number_of_segments(HDRTYPE *hdr);

int biosig_set_number_of_channels(HDRTYPE *hdr, int ns);
int biosig_set_number_of_samples(HDRTYPE *hdr, ssize_t nrec, ssize_t spr);
#define biosig_set_number_of_samples_per_record(h,n)  biosig_set_number_of_samples(h,-1,n)
#define biosig_set_number_of_records(h,n)             biosig_set_number_of_samples(h,n,-1)
// ATT_DEPREC int biosig_set_number_of_segments(HDRTYPE *hdr, );

int biosig_get_datablock(HDRTYPE *hdr, biosig_data_type **data, size_t *rows, size_t *columns);
biosig_data_type* biosig_get_data(HDRTYPE *hdr, char flag);

double biosig_get_samplerate(HDRTYPE *hdr);
int biosig_set_samplerate(HDRTYPE *hdr, double fs);

size_t biosig_get_number_of_events(HDRTYPE *hdr);
size_t biosig_set_number_of_events(HDRTYPE *hdr, size_t N);

// get n-th event, variables pointing to NULL are ignored
int biosig_get_nth_event(HDRTYPE *hdr, size_t n, uint16_t *typ, uint32_t *pos, uint16_t *chn, uint32_t *dur, gdf_time *timestamp, const char **desc);
/* set n-th event, variables pointing to NULL are ignored
   typ or  Desc can be used to determine the type of the event.
   if both, typ and Desc, are not NULL, the result is undefined */
int biosig_set_nth_event(HDRTYPE *hdr, size_t n, uint16_t* typ, uint32_t *pos, uint16_t *chn, uint32_t *dur, gdf_time *timestamp, char *Desc);

double biosig_get_eventtable_samplerate(HDRTYPE *hdr);
int    biosig_set_eventtable_samplerate(HDRTYPE *hdr, double fs);
int    biosig_change_eventtable_samplerate(HDRTYPE *hdr, double fs);


int biosig_get_startdatetime(HDRTYPE *hdr, struct tm *T);
int biosig_set_startdatetime(HDRTYPE *hdr, struct tm T);
gdf_time biosig_get_startdatetime_gdf(HDRTYPE *hdr);
int biosig_set_startdatetime_gdf(HDRTYPE *hdr, gdf_time T);

int biosig_get_birthdate(HDRTYPE *hdr, struct tm *T);
int biosig_set_birthdate(HDRTYPE *hdr, struct tm T);

const char* biosig_get_patient_name(HDRTYPE *hdr);
const char* biosig_get_patient_id(HDRTYPE *hdr);

const char* biosig_get_recording_id(HDRTYPE *hdr);
const char* biosig_get_technician(HDRTYPE *hdr);
const char* biosig_get_manufacturer_name(HDRTYPE *hdr);
const char* biosig_get_manufacturer_model(HDRTYPE *hdr);
const char* biosig_get_manufacturer_version(HDRTYPE *hdr);
const char* biosig_get_manufacturer_serial_number(HDRTYPE *hdr);
const char* biosig_get_application_specific_information(HDRTYPE *hdr);

int biosig_set_patient_name(HDRTYPE *hdr, const char* rid);
int biosig_set_patient_name_structured(HDRTYPE *hdr, const char* LastName, const char* FirstName, const char* SecondLastName);
int biosig_set_patient_id(HDRTYPE *hdr, const char* rid);
int biosig_set_recording_id(HDRTYPE *hdr, const char* rid);
int biosig_set_technician(HDRTYPE *hdr, const char* rid);
int biosig_set_manufacturer_name(HDRTYPE *hdr, const char* rid);
int biosig_set_manufacturer_model(HDRTYPE *hdr, const char* rid);
int biosig_set_manufacturer_version(HDRTYPE *hdr, const char* rid);
int biosig_set_manufacturer_serial_number(HDRTYPE *hdr, const char* rid);
int biosig_set_application_specific_information(HDRTYPE *hdr, const char* appinfo);

double biosig_get_channel_samplerate(HDRTYPE *hdr, int chan);
int biosig_set_channel_samplerate_and_samples_per_record(HDRTYPE *hdr, int chan, ssize_t spr, double fs);


/* =============================================================
	setter and getter functions for accessing fields of CHANNEL_TYPE
	these functions are currently experimential and are likely to change
   ============================================================= */

// returns M-th channel, M is zero-based
CHANNEL_TYPE* biosig_get_channel(HDRTYPE *hdr, int M);

const char* biosig_channel_get_label(CHANNEL_TYPE *chan);
int         biosig_channel_set_label(CHANNEL_TYPE *chan, const char* label);

uint16_t    biosig_channel_get_physdimcode(CHANNEL_TYPE *chan);
const char* biosig_channel_get_physdim(CHANNEL_TYPE *chan);
#define     biosig_channel_get_unit(h) biosig_channel_get_physdim(h)

int         biosig_channel_set_physdimcode(CHANNEL_TYPE *chan, uint16_t physdimcode);
#define     biosig_channel_set_physdim(a,b) biosig_channel_set_physdimcode(a, PhysDimCode(b))
#define     biosig_channel_set_unit(a,b) biosig_channel_set_physdimcode(a, PhysDimCode(b))

// this will affect result of next SREAD when flag.ucal==0
int     biosig_channel_change_scale_to_physdimcode(CHANNEL_TYPE *chan, uint16_t physdimcode);
#define biosig_channel_change_scale_to_unitcode(a,b) biosig_channel_set_scale_to_physdimcode(a, b)
#define biosig_channel_change_scale_to_physdim(a,b) biosig_channel_set_scale_to_physdimcode(a, PhysDimCode(b))
#define biosig_channel_change_scale_to_unit(a,b) biosig_channel_set_scale_to_physdimcode(a, PhysDimCode(b))

int biosig_channel_get_scaling(CHANNEL_TYPE *chan, double *PhysMax, double *PhysMin, double *DigMax, double *DigMin);
int biosig_channel_set_scaling(CHANNEL_TYPE *chan, double PhysMax, double PhysMin, double DigMax, double DigMin);
double biosig_channel_get_cal(CHANNEL_TYPE *chan);
double biosig_channel_get_off(CHANNEL_TYPE *chan);
ATT_DEPREC int biosig_channel_set_cal(CHANNEL_TYPE *chan, double cal);
ATT_DEPREC int biosig_channel_set_off(CHANNEL_TYPE *chan, double off);

int    biosig_channel_get_filter(CHANNEL_TYPE *chan, double *LowPass, double *HighPass, double *Notch);
int    biosig_channel_set_filter(CHANNEL_TYPE *chan, double LowPass, double HighPass, double Notch);

double biosig_channel_get_timing_offset(CHANNEL_TYPE *hc);
int    biosig_channel_set_timing_offset(CHANNEL_TYPE *hc, double off);

double biosig_channel_get_impedance(CHANNEL_TYPE *hc);
int    biosig_channel_set_impedance(CHANNEL_TYPE *hc, double val);

/*
double biosig_channel_get_samplerate(CHANNEL_TYPE *hc);
int    biosig_channel_set_samplerate_and_samples_per_record(CHANNEL_TYPE *hc, size_t spr, double val);
*/

size_t biosig_channel_get_samples_per_record(CHANNEL_TYPE *hc);
int    biosig_channel_set_samples_per_record(CHANNEL_TYPE *hc, size_t spr);

uint16_t biosig_channel_get_datatype(CHANNEL_TYPE *hc);
int  biosig_channel_set_datatype(CHANNEL_TYPE *hc, uint16_t gdftyp);
#define biosig_channel_set_datatype_to_int8(h)		biosig_channel_set_datatype(h,1)
#define biosig_channel_set_datatype_to_uint8(h)		biosig_channel_set_datatype(h,2)
#define biosig_channel_set_datatype_to_int16(h)		biosig_channel_set_datatype(h,3)
#define biosig_channel_set_datatype_to_uint16(h)	biosig_channel_set_datatype(h,4)
#define biosig_channel_set_datatype_to_int32(h)		biosig_channel_set_datatype(h,5)
#define biosig_channel_set_datatype_to_uint32(h)	biosig_channel_set_datatype(h,6)
#define biosig_channel_set_datatype_to_int64(h)		biosig_channel_set_datatype(h,7)
#define biosig_channel_set_datatype_to_uint64(h)	biosig_channel_set_datatype(h,8)
#define biosig_channel_set_datatype_to_float(h)		biosig_channel_set_datatype(h,16)
#define biosig_channel_set_datatype_to_single(h)	biosig_channel_set_datatype(h,16)
#define biosig_channel_set_datatype_to_double(h)	biosig_channel_set_datatype(h,17)

const char *biosig_channel_get_transducer(CHANNEL_TYPE *hc);
int biosig_channel_set_transducer(CHANNEL_TYPE *hc, const char *transducer);

#pragma GCC visibility pop

typedef HDRTYPE *biosig_handle_t ;

#if defined(MAKE_EDFLIB)

// definitions according to edflib v1.09
#define edflib_version()			(109)
#define EDFLIB_MAX_ANNOTATION_LEN 	512

#define EDFLIB_FILETYPE_EDF                  0
#define EDFLIB_FILETYPE_EDFPLUS              1
#define EDFLIB_FILETYPE_BDF                  2
#define EDFLIB_FILETYPE_BDFPLUS              3
#define EDFLIB_MALLOC_ERROR                 -1
#define EDFLIB_NO_SUCH_FILE_OR_DIRECTORY    -2
#define EDFLIB_FILE_CONTAINS_FORMAT_ERRORS  -3
#define EDFLIB_MAXFILES_REACHED             -4
#define EDFLIB_FILE_READ_ERROR              -5
#define EDFLIB_FILE_ALREADY_OPENED          -6
#define EDFLIB_FILETYPE_ERROR               -7
#define EDFLIB_FILE_WRITE_ERROR             -8
#define EDFLIB_NUMBER_OF_SIGNALS_INVALID    -9
#define EDFLIB_FILE_IS_DISCONTINUOUS       -10
#define EDFLIB_INVALID_READ_ANNOTS_VALUE   -11

/* values for annotations */
#define EDFLIB_DO_NOT_READ_ANNOTATIONS 0
#define EDFLIB_READ_ANNOTATIONS        1
#define EDFLIB_READ_ALL_ANNOTATIONS    2

/* the following defines are possible errors returned by edfopen_file_writeonly() */
#define EDFLIB_NO_SIGNALS                  -20
#define EDFLIB_TOO_MANY_SIGNALS            -21
#define EDFLIB_NO_SAMPLES_IN_RECORD        -22
#define EDFLIB_DIGMIN_IS_DIGMAX            -23
#define EDFLIB_DIGMAX_LOWER_THAN_DIGMIN    -24
#define EDFLIB_PHYSMIN_IS_PHYSMAX          -25

#define EDFLIB_TIME_DIMENSION (10000000LL)
#define EDFLIB_MAXSIGNALS 256
#define EDFLIB_MAX_ANNOTATION_LEN 512

#define EDFSEEK_SET 0
#define EDFSEEK_CUR 1
#define EDFSEEK_END 2


struct edf_annotation_struct {                       /* this structure is used for annotations */
        size_t onset;                                /* onset time of the event, expressed in units of 100 nanoSeconds and relative to the starttime in the header */
        size_t duration;                              /* duration time, this is a null-terminated ASCII text-string */
        char annotation[EDFLIB_MAX_ANNOTATION_LEN + 1]; /* description of the event in UTF-8, this is a null terminated string */
       };

int edfopen_file_writeonly(const char *path, int filetype, int number_of_signals);
#define edfopen_file_readonly(a,c) 		biosig_open_file_readonly(a,c)
#define edfclose_file(handle) 			biosig_close_file(handle)
int edfread_physical_samples(int handle, int edfsignal, int n, double *buf);
int edfread_digital_samples(int handle, int edfsignal, int n, int *buf);
long long edfseek(int handle, int biosig_signal, long long offset, int whence);
long long edftell(int handle, int biosig_signal);
int edfrewind(int handle, int edfsignal);
//#define edf_get_annotation(a,b,c)               biosig_get_annotation(a,b,c) 
int edf_get_annotation(int handle, int n, struct edf_annotation_struct *annot);
//#define edfopen_file_writeonly(a,b,c)		biosig_open_file_writeonly(a,b,c)		
int biosig_open_file_writeonly(const char *path, enum FileFormat filetype, int number_of_signals);
#define edf_set_samplefrequency(a,b,c)		biosig_set_samplefrequency(a,b,c)
#define edf_set_physical_maximum(a,b,c) 	biosig_set_physical_maximum(a,b,c)
#define edf_set_physical_minimum(a,b,c) 	biosig_set_physical_minimum(a,b,c)
#define edf_set_digital_maximum(a,b,c)		biosig_set_digital_maximum(a,b,(double)(c))
#define edf_set_digital_minimum(a,b,c)		biosig_set_digital_minimum(a,b,(double)(c))
#define edf_set_label(a,b,c) 			biosig_set_label(a,b,c)
#define edf_set_prefilter(a,b,c) 		biosig_set_prefilter(a,b,c)
#define edf_set_transducer(a,b,c) 		biosig_set_transducer(a,b,c)
#define edf_set_physical_dimension(a,b,c) 	biosig_set_physical_dimension(a,b,c)
int edf_set_startdatetime(int handle, int startdate_year, int startdate_month, int startdate_day, int starttime_hour, int starttime_minute, int starttime_second);
#define edf_set_patientname(a,b)		biosig_set_patientname(a,b)
#define edf_set_patientcode(a,b)		biosig_set_patientcode(a,b)
//#define edf_set_gender(a,b)			biosig_set_gender(a,b)
int edf_set_gender(int handle, int gender);
int edf_set_birthdate(int handle, int birthdate_year, int birthdate_month, int birthdate_day);
#define edf_set_patient_additional(a,b)		biosig_set_patient_additional(a,b)
#define edf_set_admincode(a,b)			biosig_set_admincode(a,b)
#define edf_set_technician(a,b)			biosig_set_technician(a,b)
#define edf_set_equipment(a,b)			biosig_set_equipment(a,b)

int edf_set_recording_additional(int handle, const char *recording_additional);
int edfwrite_physical_samples(int handle, double *buf);
int edf_blockwrite_physical_samples(int handle, double *buf);
int edfwrite_digital_samples(int handle, int *buf);
int edf_blockwrite_digital_samples(int handle, int *buf);

#define edfwrite_annotation_utf8(a,b,c,d) 	biosig_write_annotation_utf8(a,b,c,d) 
#define edfwrite_annotation_latin1(a,b,c,d) 	biosig_write_annotation_latin1(a,b,c,d) 
#define edf_set_datarecord_duration(a,b)	biosig_set_datarecord_duration(a,b)

#endif 


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif
