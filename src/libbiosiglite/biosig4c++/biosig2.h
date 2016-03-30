/*

    Copyright (C) 2012,2013 Alois Schloegl <alois.schloegl@gmail.com>
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

// TODO: eventually biosig.h should not be removed
#include "biosig.h"

#define BIOSIG_FLAG_COMPRESSION        0x0001
#define BIOSIG_FLAG_UCAL               0x0002
#define BIOSIG_FLAG_OVERFLOWDETECTION  0x0004
#define BIOSIG_FLAG_ROW_BASED_CHANNELS 0x0008

#ifdef __cplusplus
extern "C" {
#endif

HDRTYPE* constructHDR(const unsigned NS, const unsigned N_EVENT);
/* 	allocates memory initializes header HDR of type HDRTYPE
	with NS channels an N_EVENT event elements
 --------------------------------------------------------------- */
void 	 destructHDR(HDRTYPE* hdr);
/* 	destroys the header *hdr and frees allocated memory
 --------------------------------------------------------------- */

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


/*
        DO NOT USE         DO NOT USE         DO NOT USE         DO NOT USE

        the functions below are experimental and have not been used so far
        in any productions system
        They will be removed or significantly changed .

        DO NOT USE         DO NOT USE         DO NOT USE         DO NOT USE
*/

struct ATT_DEPREC biosig_annotation_struct {       /* this structure is used for annotations */
        size_t onset;                   /* onset time of the event, expressed in units of 100 nanoSeconds and relative to the starttime in the header */
        size_t duration;                /* duration time, this is a null-terminated ASCII text-string */
        const char *annotation; 	/* description of the event in UTF-8, this is a null terminated string */
       };

typedef HDRTYPE *biosig_handle_t ;

ATT_DEPREC int biosig_lib_version(void);

ATT_DEPREC int biosig_open_file_readonly(const char *path, int read_annotations);

ATT_DEPREC int biosig_close_file(int handle);
ATT_DEPREC int biosig_read_samples(int handle, size_t channel, size_t n, double *buf, unsigned char UCAL);
ATT_DEPREC int biosig_read_physical_samples(int handle, size_t channel, size_t n, double *buf);
ATT_DEPREC int biosig_read_digital_samples(int handle, size_t channel, size_t n, double *buf);
//#define biosig_read_physical_samples(a,b,c,d) biosig_read_samples(a,b,c,d,0) 
//#define biosig_read_digital_samples(a,b,c,d)  biosig_read_samples(a,b,c,d,1) 
ATT_DEPREC size_t biosig_seek(int handle, long long offset, int whence);
ATT_DEPREC size_t biosig_tell(int handle);
ATT_DEPREC void biosig_rewind(int handle, int biosig_signal);
ATT_DEPREC int biosig_get_annotation(int handle, size_t n, struct biosig_annotation_struct *annot);
ATT_DEPREC int biosig_open_file_writeonly(const char *path, enum FileFormat filetype, int number_of_signals);

ATT_DEPREC double biosig_get_global_samplefrequency(int handle);
ATT_DEPREC int biosig_set_global_samplefrequency(int handle, double samplefrequency);

ATT_DEPREC double biosig_get_samplefrequency(int handle, int biosig_signal);
ATT_DEPREC int biosig_set_samplefrequency(int handle, int biosig_signal,  double samplefrequency);

ATT_DEPREC double biosig_get_physical_maximum(int handle, int biosig_signal);
ATT_DEPREC int biosig_set_physical_maximum(int handle, int biosig_signal, double phys_max);

ATT_DEPREC double biosig_get_physical_minimum(int handle, int biosig_signal);
ATT_DEPREC int biosig_set_physical_minimum(int handle, int biosig_signal, double phys_min);

ATT_DEPREC double biosig_get_digital_maximum(int handle, int biosig_signal);
ATT_DEPREC int biosig_set_digital_maximum(int handle, int biosig_signal, double dig_max);

ATT_DEPREC double biosig_get_digital_minimum(int handle, int biosig_signal);
ATT_DEPREC int biosig_set_digital_minimum(int handle, int biosig_signal, double dig_min);

ATT_DEPREC const char *biosig_get_label(int handle, int biosig_signal);
ATT_DEPREC int biosig_set_label(int handle, int biosig_signal, const char *label);

//const char *biosig_get_prefilter(int handle, int biosig_signal);
ATT_DEPREC int biosig_set_prefilter(int handle, int biosig_signal, const char *prefilter);
ATT_DEPREC double biosig_get_highpassfilter(int handle, int biosig_signal);
ATT_DEPREC int biosig_set_highpassfilter(int handle, int biosig_signal, double frequency);
ATT_DEPREC double biosig_get_lowpassfilter(int handle, int biosig_signal);
ATT_DEPREC int biosig_set_lowpassfilter(int handle, int biosig_signal, double frequency);
ATT_DEPREC double biosig_get_notchfilter(int handle, int biosig_signal);
ATT_DEPREC int biosig_set_notchfilter(int handle, int biosig_signal, double frequency);

ATT_DEPREC const char *biosig_get_transducer(int handle, int biosig_signal);
ATT_DEPREC int biosig_set_transducer(int handle, int biosig_signal, const char *transducer);

ATT_DEPREC const char *biosig_get_physical_dimension(int handle, int biosig_signal);
ATT_DEPREC int biosig_set_physical_dimension(int handle, int biosig_signal, const char *phys_dim);

/*
int biosig_get_startdatetime(int handle, struct tm *T);
int biosig_set_startdatetime(int handle, const struct tm *T);
*/

ATT_DEPREC const char *biosig_get_patientname(int handle);
ATT_DEPREC int biosig_set_patientname(int handle, const char *patientname);
ATT_DEPREC const char *biosig_get_patientcode(int handle);
ATT_DEPREC int biosig_set_patientcode(int handle, const char *patientcode);
ATT_DEPREC int biosig_get_gender(int handle);
ATT_DEPREC int biosig_set_gender(int handle, int gender);

/*
int biosig_get_birthdate(int handle, struct tm *T);
int biosig_set_birthdate(int handle, const struct tm *T);
*/

ATT_DEPREC int biosig_set_patient_additional(int handle, const char *patient_additional);
ATT_DEPREC int biosig_set_admincode(int handle, const char *admincode);
/*
const char *biosig_get_technician(int handle);
int biosig_set_technician(int handle, const char *technician);
*/
ATT_DEPREC int biosig_set_equipment(int handle, const char *equipment);
ATT_DEPREC int biosig_set_recording_additional(int handle, const char *recording_additional);

ATT_DEPREC int biosig_write_physical_samples(int handle, double *buf);
ATT_DEPREC int biosig_blockwrite_physical_samples(int handle, double *buf);
ATT_DEPREC int biosig_write_digital_samples(int handle, int *buf);
ATT_DEPREC int biosig_blockwrite_digital_samples(int handle, int *buf);
ATT_DEPREC int biosig_write_annotation_utf8(int handle, size_t onset, size_t duration, const char *description);
ATT_DEPREC int biosig_write_annotation_latin1(int handle, size_t onset, size_t duration, const char *description);
ATT_DEPREC int biosig_set_datarecord_duration(int handle, double duration);


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
