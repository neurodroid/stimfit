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


#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "biosig.h"


/* =============================================================
	setter and getter functions for accessing fields of HDRTYPE
   ============================================================= */

enum FileFormat biosig_get_filetype(HDRTYPE *hdr) {
	if (hdr==NULL) return noFile;
	return hdr->TYPE;
}
int biosig_set_filetype(HDRTYPE *hdr, enum FileFormat format) {
	if (hdr==NULL) return -1;
	hdr->TYPE=format;
	if (format==GDF)
	    hdr->VERSION = 1.0/0.0; // use latest version
	return 0;
}

#if (BIOSIG_VERSION < 10700)
ATT_DEPREC int biosig_set_flags(HDRTYPE *hdr, char compression, char ucal, char overflowdetection) {
	fprintf(stderr,"Warning libbiosig2: function biosig_set_flags() is deprecated, use biosig_(re)set_flag() instead\n");
	if (hdr==NULL) return -1;
	hdr->FLAG.UCAL = ucal;
	hdr->FLAG.OVERFLOWDETECTION = overflowdetection;
	hdr->FILE.COMPRESSION = compression;
	return 0;
}
#endif

int biosig_get_flag(HDRTYPE *hdr, unsigned flags) {
	if (hdr==NULL) return -1;
	return flags & ( \
		(!!hdr->FLAG.OVERFLOWDETECTION) * (unsigned)BIOSIG_FLAG_OVERFLOWDETECTION \
		+ (!!hdr->FLAG.UCAL) * (unsigned)BIOSIG_FLAG_UCAL \
		+ (!!hdr->FILE.COMPRESSION) * (unsigned)BIOSIG_FLAG_COMPRESSION \
		+ (!!hdr->FLAG.UCAL) * (unsigned)BIOSIG_FLAG_UCAL \
		+ (!!hdr->FLAG.ROW_BASED_CHANNELS)* (unsigned)BIOSIG_FLAG_ROW_BASED_CHANNELS \
		) ;
}

int biosig_set_flag(HDRTYPE *hdr, unsigned flags) {
	if (hdr==NULL) return -1;
	hdr->FLAG.UCAL               |= !!(flags & BIOSIG_FLAG_UCAL);
	hdr->FLAG.OVERFLOWDETECTION  |= !!(flags & BIOSIG_FLAG_OVERFLOWDETECTION);
	hdr->FILE.COMPRESSION        |= !!(flags & BIOSIG_FLAG_COMPRESSION);
	hdr->FLAG.ROW_BASED_CHANNELS |= !!(flags & BIOSIG_FLAG_ROW_BASED_CHANNELS);
	return 0;
};

int biosig_reset_flag(HDRTYPE *hdr, unsigned flags) {
	if (hdr==NULL) return -1;
	hdr->FLAG.UCAL               &= !(flags & BIOSIG_FLAG_UCAL);
	hdr->FLAG.OVERFLOWDETECTION  &= !(flags & BIOSIG_FLAG_OVERFLOWDETECTION);
	hdr->FILE.COMPRESSION        &= !(flags & BIOSIG_FLAG_COMPRESSION);
	hdr->FLAG.ROW_BASED_CHANNELS &= !(flags & BIOSIG_FLAG_ROW_BASED_CHANNELS);
	return 0;
};

int biosig_get_targetsegment(HDRTYPE *hdr) {
	if (hdr==NULL) return -1;
	return hdr->FLAG.TARGETSEGMENT;
};

const char* biosig_get_filename(HDRTYPE *hdr) {
	if (hdr==NULL) return NULL;
	return hdr->FileName;
};
float biosig_get_version(HDRTYPE *hdr) {
	if (hdr==NULL) return NAN;
	return hdr->VERSION;
};


int biosig_set_targetsegment(HDRTYPE *hdr, unsigned targetsegment) {
	return biosig_set_segment_selection(hdr, 0, targetsegment);
};
int biosig_set_segment_selection(HDRTYPE *hdr, int k, uint32_t argSweepSel) {;
	if (hdr==NULL) return -1;
	if (k>5 || k<0) return -3;
	if (k==0) {
		if (argSweepSel > 127) {
			fprintf(stderr,"Warning libbiosig2: biosig_set_targetsegment is larger than 127 (%i)\n", argSweepSel);
			return -2;
		}
		hdr->FLAG.TARGETSEGMENT = argSweepSel;
	}
	else
		hdr->AS.SegSel[k-1] = argSweepSel;
	return 0;
}
uint32_t* biosig_get_segment_selection(HDRTYPE *hdr) {
	if (hdr==NULL) return NULL;
	return (uint32_t*)&(hdr->AS.SegSel);
};

long biosig_get_number_of_channels(HDRTYPE *hdr) {
	if (hdr==NULL) return -1;
	long k,m;
	for (k=0,m=0; k<hdr->NS; k++)
		if (hdr->CHANNEL[k].OnOff==1) {
			m++;
		}
	return m;
}
size_t biosig_get_number_of_records(HDRTYPE *hdr) {
	if (hdr==NULL) return -1;
	return hdr->NRec;
}
size_t biosig_get_number_of_samples(HDRTYPE *hdr) {
	if (hdr==NULL) return -1;
	return hdr->NRec*hdr->SPR;
}
size_t biosig_get_number_of_samples_per_record(HDRTYPE *hdr) {
	if (hdr==NULL) return -1;
	return hdr->SPR;
}
size_t biosig_get_number_of_segments(HDRTYPE *hdr) {
	if (hdr==NULL) return 0;
	if (hdr->SPR==0) return 0;
	size_t k, n;
	for (k=0, n=1; k<hdr->EVENT.N; k++)
		if (hdr->EVENT.TYP[k]==0x7ffe) n++;
	return n;
}

int biosig_set_number_of_channels(HDRTYPE *hdr, int ns) {
	if (hdr==NULL) return -1;
	// define variable header
	void *ptr = realloc(hdr->CHANNEL, ns*sizeof(CHANNEL_TYPE));
	if (ptr==NULL) return -1;
	hdr->CHANNEL = (CHANNEL_TYPE*)ptr;
	int k;
	for (k=hdr->NS; k < ns; k++) {
		// initialize new channels
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
		hc->bi8	      = 16*k;
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
	hdr->NS = ns;
	return 0;
}
int biosig_set_number_of_samples(HDRTYPE *hdr, ssize_t nrec, ssize_t spr) {
	if (hdr==NULL) return -1;
	if (nrec >= 0) hdr->NRec = nrec;
	if (spr  >= 0) hdr->SPR  = spr;
	return 0;
}
//ATT_DEPREC int biosig_set_number_of_segments(HDRTYPE *hdr, )

int biosig_get_datablock(HDRTYPE *hdr, double **data, size_t *rows, size_t *columns ) {
	if (hdr==NULL) return -1;
	*data = hdr->data.block;
	*rows = hdr->data.size[0];
	*columns = hdr->data.size[1];
	return 0;
}
biosig_data_type* biosig_get_data(HDRTYPE *hdr, char flag ) {
	if (hdr==NULL) return NULL;
        hdr->FLAG.ROW_BASED_CHANNELS = flag;
        sread(NULL, 0, hdr->NRec, hdr);
	return hdr->data.block;
}
double biosig_get_samplerate(HDRTYPE *hdr) {
	if (hdr==NULL) return NAN;
	return hdr->SampleRate;
}
int biosig_set_samplerate(HDRTYPE *hdr, double fs) {
	if (hdr==NULL) return -1;
	hdr->SampleRate=fs;
	return 0;
}


size_t biosig_get_number_of_events(HDRTYPE *hdr) {
	if (hdr==NULL) return 0;
	return hdr->EVENT.N;
}
size_t biosig_set_number_of_events(HDRTYPE *hdr, size_t N) {
	if (hdr==NULL) return 0;
	size_t k;
	hdr->EVENT.POS = (uint32_t*) realloc(hdr->EVENT.POS, N * 4 );
	hdr->EVENT.TYP = (uint16_t*) realloc(hdr->EVENT.TYP, N * 2 );
	for (k = hdr->EVENT.N; k<N; k++) {
		hdr->EVENT.POS[k] = 0;
		hdr->EVENT.TYP[k] = 0;
	}
	k = ( (hdr->EVENT.DUR==NULL) || (hdr->EVENT.CHN==NULL) ) ? 0 : hdr->EVENT.N;
	hdr->EVENT.DUR = (uint32_t*) realloc(hdr->EVENT.DUR, N * 4 );
	hdr->EVENT.CHN = (uint16_t*) realloc(hdr->EVENT.CHN, N * 2 );
	for (; k<N; k++) {
		hdr->EVENT.CHN[k] = 0;
		hdr->EVENT.DUR[k] = 0;
	}
	k = (hdr->EVENT.TimeStamp==NULL) ? 0 : hdr->EVENT.N;
	hdr->EVENT.TimeStamp = (gdf_time*) realloc(hdr->EVENT.TimeStamp, N * 8 );
	for (; k<N; k++) {
		hdr->EVENT.TimeStamp[k] = 0;
	}
	hdr->EVENT.N = N;
	return hdr->EVENT.N;
}

int biosig_get_nth_event(HDRTYPE *hdr, size_t n, uint16_t *typ, uint32_t *pos, uint16_t *chn, uint32_t *dur, gdf_time *timestamp, const char **desc) {
	if (hdr==NULL) return -1;
	if (hdr->EVENT.N <= n) return -1;
	uint16_t TYP=hdr->EVENT.TYP[n];
	if (typ != NULL)
		*typ = TYP;
	if (pos != NULL)
		*pos = hdr->EVENT.POS[n];
	if (chn != NULL)
		*chn = (hdr->EVENT.CHN==NULL) ? 0 : hdr->EVENT.CHN[n];
	if (dur != NULL)
		*dur = (hdr->EVENT.DUR==NULL) ? 0 : hdr->EVENT.DUR[n];
	if (timestamp != NULL)
		*timestamp = (hdr->EVENT.TimeStamp==NULL) ? 0 : hdr->EVENT.TimeStamp[n];
	if ( (desc != NULL) )
		*desc = (TYP < hdr->EVENT.LenCodeDesc) ? hdr->EVENT.CodeDesc[TYP] : NULL;
	return 0;
}
int biosig_set_nth_event(HDRTYPE *hdr, size_t n, uint16_t* typ, uint32_t *pos, uint16_t *chn, uint32_t *dur, gdf_time *timestamp, char *Desc) {
	if (hdr==NULL) return -1;
	if (hdr->EVENT.N <= n)
		biosig_set_number_of_events(hdr, n+1);

	if (typ != NULL)
		hdr->EVENT.TYP[n] = *typ;
	else if (typ == NULL)
		FreeTextEvent(hdr, n, Desc);   // sets hdr->EVENT.TYP[n]

	if (pos != NULL)
		hdr->EVENT.POS[n] = *pos;
	if (chn != NULL)
		hdr->EVENT.CHN[n] = *chn;
	if (dur != NULL)
		hdr->EVENT.DUR[n] = *dur;
	if (timestamp != NULL)
		hdr->EVENT.TimeStamp[n] = *timestamp;

	return 0;
}

double biosig_get_eventtable_samplerate(HDRTYPE *hdr) {
	if (hdr==NULL) return NAN;
	return hdr->EVENT.SampleRate;
}
int biosig_set_eventtable_samplerate(HDRTYPE *hdr, double fs) {
	if (hdr==NULL) return -1;
	hdr->EVENT.SampleRate=fs;
	return 0;
}
int biosig_change_eventtable_samplerate(HDRTYPE *hdr, double fs) {
	if (hdr==NULL) return -1;
	if (hdr->EVENT.SampleRate==fs) return 0;
	size_t k;
	double ratio = fs/hdr->EVENT.SampleRate;
	for (k = 0; k < hdr->EVENT.N; k++) {
		uint32_t POS = hdr->EVENT.POS[k];
		hdr->EVENT.POS[k] = ratio*POS;
		if (hdr->EVENT.DUR != NULL)
			hdr->EVENT.DUR[k] = (POS + hdr->EVENT.DUR[k]) * ratio - hdr->EVENT.POS[k];
	}
	hdr->EVENT.SampleRate=fs;
	return 0;
}

// deprecated because time resolution is lost, use gdftime and its tools instead.
__attribute__ ((deprecated)) int biosig_get_startdatetime(HDRTYPE *hdr, struct tm *T) {
	if (hdr==NULL) return -1;
	gdf_time2tm_time_r(hdr->T0, T);
	return (ldexp(hdr->T0,-32)<100.0);
}
int biosig_set_startdatetime(HDRTYPE *hdr, struct tm T) {
	if (hdr==NULL) return -1;
	hdr->T0 = tm_time2gdf_time(&T);
	return (ldexp(hdr->T0,-32)<100.0);
}

gdf_time biosig_get_startdatetime_gdf(HDRTYPE *hdr) {
	if (hdr==NULL) return 0;
	return(hdr->T0);
}
int biosig_set_startdatetime_gdf(HDRTYPE *hdr, gdf_time T) {
	if (hdr==NULL) return -1;
	hdr->T0 = T;
	return (ldexp(hdr->T0,-32)<100.0);
}

// deprecated because time resolution is lost, use gdftime and its tools instead.
__attribute__ ((deprecated)) int biosig_get_birthdate(HDRTYPE *hdr, struct tm *T) {
	if (hdr==NULL) return -1;
	gdf_time2tm_time_r(hdr->Patient.Birthday, T);
	return (ldexp(hdr->Patient.Birthday,-32)<100.0);
}
int biosig_set_birthdate(HDRTYPE *hdr, struct tm T) {
	if (hdr==NULL) return -1;
	hdr->Patient.Birthday = tm_time2gdf_time(&T);
	return (ldexp(hdr->Patient.Birthday,-32)<100.0);
}

const char* biosig_get_patient_name(HDRTYPE *hdr) {
	if (hdr==NULL) return NULL;
	return hdr->Patient.Name;
}

const char* biosig_get_patient_lastname(HDRTYPE *hdr, size_t *LengthLastName) {
	if (hdr==NULL) return NULL;
	*LengthLastName	=  strcspn(hdr->Patient.Name, "\x1f");
	return hdr->Patient.Name;
}
const char* biosig_get_patient_firstname(HDRTYPE *hdr, size_t *LengthFirstName) {
	if (hdr==NULL) return NULL;
	char *tmpstr = strchr(hdr->Patient.Name, 0x1f);
	if (tmpstr==NULL) {
		*LengthFirstName = 0;
		return NULL;
	}
	*LengthFirstName = strcspn(tmpstr, "\x1f");
	return tmpstr;
}
const char* biosig_get_patient_secondlastname(HDRTYPE *hdr, size_t *LengthSecondLastName) {
	if (hdr==NULL) return NULL;
	char *tmpstr = strchr(hdr->Patient.Name, 0x1f);
	if (tmpstr != NULL)
		tmpstr = strchr(tmpstr, 0x1f);
	if (tmpstr==NULL) {
		*LengthSecondLastName = 0;
		return NULL;
	}
	*LengthSecondLastName = strcspn(tmpstr, "\x1f");
	return tmpstr;
}


const char* biosig_get_patient_id(HDRTYPE *hdr) {
	if (hdr==NULL) return NULL;
	return hdr->Patient.Id;
}
const char* biosig_get_recording_id(HDRTYPE *hdr) {
	if (hdr==NULL) return NULL;
	return hdr->ID.Recording;
}
const char* biosig_get_technician(HDRTYPE *hdr) {
	if (hdr==NULL) return NULL;
	return hdr->ID.Technician;
}
const char* biosig_get_manufacturer_name(HDRTYPE *hdr) {
	if (hdr==NULL) return NULL;
	return hdr->ID.Manufacturer.Name;
}
const char* biosig_get_manufacturer_model(HDRTYPE *hdr) {
	if (hdr==NULL) return NULL;
	return hdr->ID.Manufacturer.Model;
}
const char* biosig_get_manufacturer_version(HDRTYPE *hdr) {
	if (hdr==NULL) return NULL;
	return hdr->ID.Manufacturer.Version;
}
const char* biosig_get_manufacturer_serial_number(HDRTYPE *hdr) {
	if (hdr==NULL) return NULL;
	return hdr->ID.Manufacturer.SerialNumber;
}
const char* biosig_get_application_specific_information(HDRTYPE *hdr) {
	if (hdr==NULL) return NULL;
	return hdr->AS.bci2000;
}

int biosig_set_patient_name(HDRTYPE *hdr, const char* name) {
	if (hdr==NULL) return -1;
	strncpy(hdr->Patient.Name, name, MAX_LENGTH_NAME);
	hdr->Patient.Name[MAX_LENGTH_NAME]=0;
}

int biosig_set_patient_name_structured(HDRTYPE *hdr, const char* LastName, const char* FirstName, const char* SecondLastName) {
	if (hdr==NULL) return -1;
	size_t len1 = (LastName ? strlen(LastName) : 0 );
	size_t len2 = (FirstName ? strlen(FirstName) : 0 );
	size_t len3 = (SecondLastName ? strlen(SecondLastName) : 0 );
	if (len1+len2+len3+2 > MAX_LENGTH_NAME) {
		fprintf(stderr,"Error in function %s(...): total length of name too large (%i > %i)\n",__func__, (int)(len1+len2+len3+2), MAX_LENGTH_NAME);
		return -1;
	}
	strcpy(hdr->Patient.Name, LastName);					// Flawfinder: ignore
	if (FirstName != NULL) {
		hdr->Patient.Name[len1]=0x1f;
		strcpy(hdr->Patient.Name+len1+1, FirstName);			// Flawfinder: ignore
	}
	if (SecondLastName != NULL) {
		hdr->Patient.Name[len1+len2+1]=0x1f;
		strcpy(hdr->Patient.Name+len1+len2+2, SecondLastName);		// Flawfinder: ignore
	}
	return 0;
}
int biosig_set_patient_id(HDRTYPE *hdr, const char* id) {
	if (hdr==NULL) return -1;
	strncpy(hdr->Patient.Id, id, MAX_LENGTH_PID);
	hdr->Patient.Id[MAX_LENGTH_PID]=0;
	return 0;
}
int biosig_set_recording_id(HDRTYPE *hdr, const char* rid) {
	if (hdr==NULL) return -1;
	strncpy(hdr->ID.Recording, rid, MAX_LENGTH_RID);
	hdr->ID.Recording[MAX_LENGTH_RID]=0;
	return 0;
}
int biosig_set_technician(HDRTYPE *hdr, const char* technician) {
	if (hdr==NULL) return -1;
	hdr->ID.Technician = (char*)technician;
	return 0;
}
int biosig_set_manufacturer_name(HDRTYPE *hdr, const char* rid) {
	if (hdr==NULL) return -1;
	hdr->ID.Manufacturer.Name = (char*)rid;
	return 0;
}
int biosig_set_manufacturer_model(HDRTYPE *hdr, const char* rid) {
	if (hdr==NULL) return -1;
	hdr->ID.Manufacturer.Model = rid;
	return 0;
}
int biosig_set_manufacturer_version(HDRTYPE *hdr, const char* rid) {
	if (hdr==NULL) return -1;
	hdr->ID.Manufacturer.Version = rid;
	return 0;
}
int biosig_set_manufacturer_serial_number(HDRTYPE *hdr, const char* rid) {
	if (hdr==NULL) return -1;
	hdr->ID.Manufacturer.SerialNumber = rid;
	return 0;
}
int biosig_set_application_specific_information(HDRTYPE *hdr, const char* appinfo) {
	if (hdr==NULL) return -1;
	hdr->AS.bci2000 = strdup(appinfo);
	return 0;
}

// returns M-th channel, M is 0-based
CHANNEL_TYPE* biosig_get_channel(HDRTYPE *hdr, int M) {
	if (hdr==NULL) return NULL;
	typeof(hdr->NS) k,m;
	for (k=0,m=0; k<hdr->NS; k++)
		if (hdr->CHANNEL[k].OnOff==1) {
			if (M==m) return hdr->CHANNEL+k;
			m++;
		}
	return NULL;
}

int biosig_channel_change_scale_to_physdimcode(CHANNEL_TYPE *hc, uint16_t physdimcode) {
	if (hc==NULL) return -1;
	if (hc->PhysDimCode == physdimcode) return 0; 	// nothing to do
	if ( (hc->PhysDimCode & 0xffe0) != (physdimcode & 0xffe0) ) return -2; 	// units do not match
        double scale = PhysDimScale(hc->PhysDimCode);
        scale /= PhysDimScale(physdimcode);
	hc->PhysDimCode = physdimcode;
        hc->PhysMax *= scale;
        hc->PhysMin *= scale;
        hc->Cal *= scale;
        hc->Off *= scale;
	return(0);
}
const char* biosig_channel_get_label(CHANNEL_TYPE *hc) {
	if (hc==NULL) return NULL;
	return hc->Label;
}
uint16_t biosig_channel_get_physdimcode(CHANNEL_TYPE *hc) {
	if (hc==NULL) return 0;
	return hc->PhysDimCode;
}
const char* biosig_channel_get_physdim(CHANNEL_TYPE *hc) {
	if (hc==NULL) return NULL;
	return PhysDim3(hc->PhysDimCode);
}
int biosig_channel_set_label(CHANNEL_TYPE *hc, const char* label) {
	if (hc==NULL) return -1;
	strncpy(hc->Label, label, MAX_LENGTH_LABEL);
	hc->Label[MAX_LENGTH_LABEL]=0;
	return 0;
}
int biosig_channel_set_physdimcode(CHANNEL_TYPE *hc, uint16_t physdimcode) {
	if (hc==NULL) return -1;
	hc->PhysDimCode = physdimcode;
	return 0;
}


int biosig_channel_get_scaling(CHANNEL_TYPE *hc, double *PhysMax, double *PhysMin, double *DigMax, double *DigMin) {
	if (hc==NULL) return -1;
	if (PhysMax != NULL)
		*PhysMax = hc->PhysMax;
	if (PhysMin != NULL)
		*PhysMax = hc->PhysMin;
	if (DigMax != NULL)
		*DigMax = hc->DigMax;
	if (DigMin != NULL)
		*DigMin = hc->DigMin;
	return 0;
}
int biosig_channel_set_scaling(CHANNEL_TYPE *hc, double PhysMax, double PhysMin, double DigMax, double DigMin) {
	if (hc==NULL) return -1;
	hc->PhysMax = PhysMax;
	hc->PhysMin = PhysMin;
	hc->DigMax = DigMax;
	hc->DigMin = DigMin;
	hc->Cal    = ( PhysMax - PhysMin) / ( DigMax - DigMin );
	hc->Off    = PhysMin - DigMin * hc->Cal;
	return 0;
}

double biosig_channel_get_cal(CHANNEL_TYPE *hc) {
	if (hc==NULL) return -1;
	double cal = ( hc->PhysMax - hc->PhysMin) / ( hc->DigMax - hc->DigMin );
	assert(cal==hc->Cal);
	return (cal);
}
double biosig_channel_get_off(CHANNEL_TYPE *hc) {
	if (hc==NULL) return -1;
	double off = hc->PhysMin - hc->DigMin * hc->Cal;
	assert(off==hc->Off);
	return off;
}

int biosig_channel_set_cal(CHANNEL_TYPE *hc, double cal) {
	if (hc==NULL) return -1;
	hc->Cal = cal;
	return 0;
}
int biosig_channel_set_off(CHANNEL_TYPE *hc, double off) {
	if (hc==NULL) return -1;
	hc->Off = off;
	return 0;
}

int biosig_channel_get_filter(CHANNEL_TYPE *hc, double *LowPass, double *HighPass, double *Notch) {
	if (hc==NULL) return -1;
	if (LowPass != NULL)
		*LowPass = hc->LowPass;
	if (HighPass != NULL)
		*HighPass = hc->HighPass;
	if (Notch != NULL)
		*Notch = hc->Notch;
	return 0;
}
int biosig_channel_set_filter(CHANNEL_TYPE *hc, double LowPass, double HighPass, double Notch) {
	if (hc==NULL) return -1;
	hc->LowPass = LowPass;
	hc->HighPass = HighPass;
	hc->Notch = Notch;
	return 0;
}

double biosig_channel_get_timing_offset(CHANNEL_TYPE *hc) {
	if (hc==NULL) return -1;
	return hc->TOffset;
}
int biosig_channel_set_timing_offset(CHANNEL_TYPE *hc, double off) {
	if (hc==NULL) return -1;
	hc->TOffset = off;
	return 0;
}

double biosig_channel_get_impedance(CHANNEL_TYPE *hc) {
	if (hc==NULL) return -1;
	return ( (hc->PhysDimCode & 0x7ffe) == 4256 ) ? hc->Impedance : NAN;
}
int biosig_channel_set_impedance(CHANNEL_TYPE *hc, double val) {
	if (hc==NULL) return -1;
	if ( (hc->PhysDimCode & 0x7ffe) != 4256 ) return -1;
	hc->Impedance = val;
	return 0;
}

uint16_t biosig_channel_get_datatype(CHANNEL_TYPE *hc) {
	if (hc==NULL) return -1;
	return hc->GDFTYP;
}
int biosig_channel_set_datatype(CHANNEL_TYPE *hc, uint16_t gdftyp) {
	if (hc==NULL) return -1;
	hc->GDFTYP = gdftyp;
	return 0;
}

double biosig_get_channel_samplerate(HDRTYPE *hdr, int chan) {
	CHANNEL_TYPE *hc = biosig_get_channel(hdr, chan);
	if (hc==NULL) return -1;
	if (hdr==NULL) return -1;
	return (hdr->SampleRate * hc->SPR / hdr->SPR);
}
size_t biosig_channel_get_samples_per_record(CHANNEL_TYPE *hc) {
	if (hc==NULL) return -1;
	return hc->SPR;
}
int	biosig_channel_set_samples_per_record(CHANNEL_TYPE *hc, size_t spr)  {
	if (hc==NULL) return -1;
	hc->SPR = spr;
	return 0;
}

int  biosig_set_channel_samplerate_and_samples_per_record(HDRTYPE *hdr, int chan, ssize_t spr, double fs)  {
	CHANNEL_TYPE *hc = biosig_get_channel(hdr,chan);
	if (hc==NULL) return -1;
	if ((spr <= 0) && (fs >= 0.0)) {
		hc->SPR = hdr->SPR * fs / hdr->SampleRate;
		return 0;
	}
	if ((spr >= 0) && (fs != fs)) {
		hc->SPR = spr;
		return 0;
	}
	assert (hdr->SampleRate * hc->SPR == fs * hdr->SPR);
	return (hdr->SampleRate * hc->SPR != fs * hdr->SPR);
}

const char *biosig_channel_get_transducer(CHANNEL_TYPE *hc) {

	if (hc==NULL) return(NULL);
	return (hc->Transducer);
}

int biosig_channel_set_transducer(CHANNEL_TYPE *hc, const char *transducer) {

	if (hc==NULL) return(-1);
	strncpy(hc->Transducer, transducer, MAX_LENGTH_TRANSDUCER+1);

	return (0);
}




/*
        DO NOT USE         DO NOT USE         DO NOT USE         DO NOT USE

        the functions below are experimental and have not been used so far
        in any productions system
        They will be removed or significantly changed .

        DO NOT USE         DO NOT USE         DO NOT USE         DO NOT USE
*/

#define hdrlistlen 64
struct hdrlist_t {
	HDRTYPE *hdr;		// header information as defined in level 1 interface
	//const char *filename; // name of file, is always hdr->FileName
	uint16_t NS; 	        // number of effective channels, only CHANNEL[].OnOff==1 are considered
	size_t *chanpos; 	// position of file handle for each channel
} ; 

struct hdrlist_t hdrlist[hdrlistlen];

CHANNEL_TYPE *getChannelHeader(HDRTYPE *hdr, uint16_t channel) {
	// returns channel header - skip Off-channels
	CHANNEL_TYPE *hc = hdr->CHANNEL;
	typeof(hdr->NS) chan = 0; 
	while (1) {
		if (hc->OnOff==1) {
			if (chan==channel) return hc;
			chan++;
		}
		hc++;
	}
	return NULL;
}


int biosig_lib_version(void) {
	return (BIOSIG_VERSION);
}

biosig_handle_t biosig2_open_file_readonly(const char *path, int read_annotations) {
/* 

	on success returns handle. 
*/
	HDRTYPE *hdr = sopen(path,"r",NULL);
	if (serror2(hdr)) {
		destructHDR(hdr);
		return(NULL);
	}
	if (read_annotations)
	        sort_eventtable(hdr);
	return(hdr);
}

int biosig_open_file_readonly(const char *path, int read_annotations) {
/* 

	on success returns handle. 
*/
	int k = 0;
	while (k < hdrlistlen && hdrlist[k].hdr != NULL) k++;
	if (k >= hdrlistlen) return(-1);
	HDRTYPE *hdr = sopen(path,"r",NULL);
	hdrlist[k].hdr = hdr;
	//hdrlist[k].filename = hdr->FileName;
	hdrlist[k].NS  = 0; 
	hdrlist[k].chanpos  = calloc(hdrlist[k].NS,sizeof(size_t)); 

        if (read_annotations)
                sort_eventtable(hdrlist[k].hdr);

	return(k);
}

int biosig2_close_file(biosig_handle_t hdr) {
	destructHDR(hdr);
	return(0);
}

int biosig_close_file(int handle) {
	destructHDR(hdrlist[handle].hdr);
	hdrlist[handle].hdr = NULL;
	if (hdrlist[handle].chanpos) free(hdrlist[handle].chanpos);
	hdrlist[handle].NS  = 0; 
	//hdrlist[handle].filename = NULL;
	
#if 0
	int k;
	for (k=0; k<hdrlistlen; k++)
		if (hdrlist[k].hdr!=NULL) return(0); 
	free(hdrlist);
#endif 
	return(0);
}

int biosig_read_samples(int handle, size_t channel, size_t n, double *buf, unsigned char UCAL) {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL || hdrlist[handle].NS<=channel ) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;

	CHANNEL_TYPE *hc = getChannelHeader(hdr,channel);

	size_t stride = 1; // stride between consecutive samples of same channel, depends on data orientation hdr->FLAG.ROW_BASED_CHANNELS
	size_t div = hdr->SPR/hc->SPR; 	// stride if sample rate of channel is smaller than the overall sampling rate

	size_t POS = hdrlist[handle].chanpos[channel]*div;	// 
	size_t LEN = n*div;
	size_t startpos = POS/hdr->SPR;  // round towards 0
	size_t endpos = (POS+LEN)/hdr->SPR + ((POS+LEN)%hdr->SPR != 0);  // round towards infinity

	if (hdr->AS.first > startpos || (endpos-startpos) > hdr->AS.length || hdr->FLAG.UCAL!=UCAL) {
		// read data when not in data buffer hdr->data.block
		hdr->FLAG.UCAL = UCAL; 
		sread(NULL, startpos, endpos - startpos, hdr);
	}	

	// when starting position is not aligned with start of data
	size_t offset = hdr->AS.first * hdr->SPR - POS; 

	// find starting position and stride of data 
	double *data = hdr->data.block;
	if (hdr->FLAG.ROW_BASED_CHANNELS) {
		stride = hdr->data.size[0];
		data = hdr->data.block + channel + offset * stride;
	} 
	else {
		data = hdr->data.block + offset + channel * hdr->data.size[0];
	}
	size_t k;
	for (k = 0; k < n; k++) {
		buf[k] = data[k*div*stride];	// copy data into output buffer
	}
	hdrlist[handle].chanpos[channel] += n; // update position pointer of channel chan
	return (0);
}

/*
int biosig_read_physical_samples(int handle, size_t biosig_signal, size_t n, double *buf) {
	return biosig_read_samples(handle, biosig_signal, n, buf, (unsigned char)(0));
}

int biosig_read_digital_samples(int handle, size_t biosig_signal, size_t n, double *buf) {
	return biosig_read_samples(handle, biosig_signal, n, buf, (unsigned char)(1));
}
*/

size_t biosig_seek(int handle, long long offset, int whence) {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	sseek(hdr, offset, whence);
	return (hdr->FILE.POS);
}

size_t biosig_tell(int handle) {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	return(stell(hdrlist[handle].hdr));
}

void biosig_rewind(int handle, int biosig_signal) {
/* It is equivalent to: (void) biosig_seek(int handle, int biosig_signal, 0LL, biosig_SEEK_SET) */
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return;
	srewind(hdrlist[handle].hdr);
}

biosig_handle_t biosig2_open_file_writeonly(const char *path, enum FileFormat filetype, int number_of_signals) {

        /* TODO: does not open file and write to file */
	HDRTYPE *hdr = constructHDR(number_of_signals,0);
        hdr->FLAG.UCAL = 0;
        hdr->FLAG.OVERFLOWDETECTION = 0;
        hdr->FILE.COMPRESSION = 0;

	return(hdr); 
}

int biosig_open_file_writeonly(const char *path, enum FileFormat filetype, int number_of_signals) {

        /* TODO: does not open file and write to file */
#if 1
	int k = 0;
	while (k < hdrlistlen && hdrlist[k].hdr != NULL) k++;
	if (k>=hdrlistlen) return -1;

	HDRTYPE *hdr = constructHDR(number_of_signals,0);
#else
	HDRTYPE *hdr = constructHDR(number_of_signals,0);
	if (hdr==NULL) return (-1); 

	hdr->FileName = strdup(path);
	hdr->TYPE = filetype;
	int k = 0;
	while (k < hdrlistlen && hdrlist[k].hdr != NULL) k++;
	if (k>=hdrlistlen) {
		void *ptr = realloc(hdrlist, (k+1)*sizeof(*hdrlist));
		if (ptr==NULL) return (-1); 
		hdrlist = (struct hdrlist_t*) ptr;
		hdrlistlen = k+1;
	}
#endif
        hdr->FLAG.UCAL = 0;
        hdr->FLAG.OVERFLOWDETECTION = 0;
        hdr->FILE.COMPRESSION = 0;

	hdrlist[k].hdr  = hdr;
	return(0); 
}

double biosig_get_global_samplefrequency(int handle) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(NAN);
	return (hdrlist[handle].hdr->SampleRate);
}

int biosig_set_global_samplefrequency(int handle, double samplefrequency) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	hdrlist[handle].hdr->SampleRate = samplefrequency;

	return 0;
}

double biosig_get_samplefrequency(int handle, int biosig_signal) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(NAN);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(NAN);

	return (hdr->SampleRate*hdr->CHANNEL[biosig_signal].SPR/hdr->SPR);
}

int biosig_set_samplefrequency(int handle, int biosig_signal, double samplefrequency) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = 0;
	int ch;
        for (ch = 0; ch < hdr->NS; ch++) {
                if (hdr->CHANNEL[ch].OnOff==1) {
                        if (ns==biosig_signal) break;
                        ns++;
                }
        }
	if (ch >= hdr->NS) return(-1);

	// FIXME: resulting sampling rate might depend on calling order; what's the overall sampling rate ? 
	if (samplefrequency != hdr->SampleRate) {
		double spr = samplefrequency * hdr->SPR / hdr->SampleRate;
		hdr->CHANNEL[biosig_signal].SPR = spr;
		if (spr != ceil(spr)) return (-2);
		return 0;
	}
	hdr->CHANNEL[ch].SPR = hdr->SPR;
	return 0;
}

double biosig_get_physical_maximum(int handle, int biosig_signal) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(NAN);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(NAN);

	return (hdr->CHANNEL[biosig_signal].PhysMax);
}

int biosig_set_physical_maximum(int handle, int biosig_signal, double phys_max) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(-1);

	hdr->CHANNEL[biosig_signal].PhysMax = phys_max;
	return (0);
}

double biosig_get_physical_minimum(int handle, int biosig_signal) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(NAN);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(NAN);

	return (hdr->CHANNEL[biosig_signal].PhysMin);
}

int biosig_set_physical_minimum(int handle, int biosig_signal, double phys_min) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(-1);

	hdr->CHANNEL[biosig_signal].PhysMin = phys_min;
	return (0);
}

double biosig_get_digital_maximum(int handle, int biosig_signal) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(NAN);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(NAN);

	return (hdr->CHANNEL[biosig_signal].DigMax);
}

int biosig_set_digital_maximum(int handle, int biosig_signal, double dig_max) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(-1);

	hdr->CHANNEL[biosig_signal].DigMax = dig_max;
	return (0);
}

double biosig_get_digital_minimum(int handle, int biosig_signal) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(NAN);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(NAN);

	return (hdr->CHANNEL[biosig_signal].DigMin);
}

int biosig_set_digital_minimum(int handle, int biosig_signal, double dig_min) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(-1);

	hdr->CHANNEL[biosig_signal].DigMin = dig_min;
	return (0);
}

const char *biosig_get_label(int handle, int biosig_signal) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(NULL);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(NULL);

	return (hdr->CHANNEL[biosig_signal].Label);
}

int biosig_set_label(int handle, int biosig_signal, const char *label) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(-1);

	strncpy(hdr->CHANNEL[biosig_signal].Label, label, MAX_LENGTH_LABEL);
	return (0);
}


int biosig_set_prefilter(int handle, int biosig_signal, const char *prefilter) {
        // TODO: parse prefilter and call biosig_set_{highpass,lowpass,notch}filter
        return fprintf(stderr,"Warning: biosig_set_prefilter(...) is not implemented, use instead biosig_set_highpassfilter(),biosig_set_lowpassfilter(),biosig_set_notchfilter().\n");
}
        
int biosig_set_highpassfilter(int handle, int biosig_signal, double frequency) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(-1);

	hdr->CHANNEL[biosig_signal].HighPass = frequency;

	return 0; 
}

int biosig_set_lowpassfilter(int handle, int biosig_signal, double frequency) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(-1);

	hdr->CHANNEL[biosig_signal].LowPass = frequency;

	return 0; 
}

int biosig_set_notchfilter(int handle, int biosig_signal, double frequency) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(-1);

	hdr->CHANNEL[biosig_signal].Notch = frequency;

	return (0);
}


const char *biosig_get_transducer(int handle, int biosig_signal) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(NULL);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(NULL);

	return (hdr->CHANNEL[biosig_signal].Transducer);
}

int biosig_set_transducer(int handle, int biosig_signal, const char *transducer) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(-1);

	strncpy(hdr->CHANNEL[biosig_signal].Transducer, transducer, MAX_LENGTH_TRANSDUCER+1);

	return (0);
}


const char *biosig_physical_dimension(int handle, int biosig_signal) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(NULL);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(NULL);

	return (PhysDim3(hdr->CHANNEL[biosig_signal].PhysDimCode));
}

int biosig_set_physical_dimension(int handle, int biosig_signal, const char *phys_dim) {

	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	typeof(hdr->NS) ns = hdr->NS;
	if (biosig_signal >= ns) return(-1);

	hdr->CHANNEL[biosig_signal].PhysDimCode = PhysDimCode(phys_dim);

	return (0);
}

int edf_set_startdatetime(int handle, int startdate_year, int startdate_month, int startdate_day, int starttime_hour, int starttime_minute, int starttime_second) {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	struct tm T;
	T.tm_year = startdate_year;
	T.tm_mon  = startdate_month;
	T.tm_mday = startdate_day;
	T.tm_hour = starttime_hour;
	T.tm_min  = starttime_minute;
	T.tm_sec  = starttime_second;
	hdr->T0   = tm_time2gdf_time(&T);
	return (0);
}

const char *biosig_get_patientname(int handle)  {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(NULL);
	return(hdrlist[handle].hdr->Patient.Name);
}
int biosig_set_patientname(int handle, const char *patientname) {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	strncpy(hdrlist[handle].hdr->Patient.Name, patientname, MAX_LENGTH_NAME+1);
	return (0);
}

const char *biosig_get_patientcode(int handle)  {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(NULL);
	return(hdrlist[handle].hdr->Patient.Id);
}
int biosig_set_patientcode(int handle, const char *patientcode) {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	strncpy(hdrlist[handle].hdr->Patient.Id, patientcode, MAX_LENGTH_PID+1);
	return(0);
}

int biosig_get_gender(int handle) {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(0);
	return(hdrlist[handle].hdr->Patient.Sex);
}

int biosig_set_gender(int handle, int gender) {
	if (gender<0 || gender>9) return (-1); 
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	switch (gender) {
	case  1 :
	case 'm':
	case 'M':
		hdrlist[handle].hdr->Patient.Sex = 1;
		return(0);
	case  2 :
	case 'f':
	case 'F':
		hdrlist[handle].hdr->Patient.Sex = 2;
		return(0);
	default:
		return(0); 
	}
}

int edf_set_birthdate(int handle, int birthdate_year, int birthdate_month, int birthdate_day) {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	struct tm T;
	T.tm_year = birthdate_year;
	T.tm_mon  = birthdate_month;
	T.tm_mday = birthdate_day;
	T.tm_hour = 12;
	T.tm_min  = 0;
	T.tm_sec  = 0;
	hdr->Patient.Birthday = tm_time2gdf_time(&T);
	return (0);
}

int biosig_set_patient_additional(int handle, const char *patient_additional) {
	fprintf(stderr,"Warning: biosig_set_patient_additional() not supported.\n");
	return (-1);
}

int biosig_set_admincode(int handle, const char *admincode) {
	fprintf(stderr,"Warning: biosig_set_admincode() not supported.\n");
	return (-1);
}

/*
const char *biosig_get_technician(int handle)  {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(NULL);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	return(hdr->ID.Technician);
}
int biosig_set_technician(int handle, const char *technician) {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	hdr->ID.Technician = realloc(hdr->ID.Technician, strlen(technician)+1);
	strcpy(hdr->ID.Technician, technician);
	return(0);
}
*/

// TODO: implement the following functions
int biosig_set_equipment(int handle, const char *equipment) {
	return (-1);
}
int biosig_set_recording_additional(int handle, const char *recording_additional) {
	return (-1);
}
int biosig_write_physical_samples(int handle, double *buf) {
	return (-1);
}
int biosig_blockwrite_physical_samples(int handle, double *buf) {
	return (-1);
}
int biosig_write_digital_samples(int handle, int *buf) {
	return (-1);
}
int biosig_blockwrite_digital_samples(int handle, int *buf) {
	return (-1);
}

int biosig_write_annotation(int handle, size_t onset, size_t duration, const char *description) {
	/* onset and duration are in samples */
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;

	size_t N = hdr->EVENT.N++;
	hdr->EVENT.POS = (uint32_t*) realloc(hdr->EVENT.POS, hdr->EVENT.N*sizeof(*(hdr->EVENT.POS)) );
	hdr->EVENT.TYP = (uint16_t*) realloc(hdr->EVENT.TYP, hdr->EVENT.N*sizeof(*(hdr->EVENT.TYP)) );
	hdr->EVENT.DUR = (uint32_t*) realloc(hdr->EVENT.DUR, hdr->EVENT.N*sizeof(*(hdr->EVENT.DUR)) );
	hdr->EVENT.CHN = (uint16_t*) realloc(hdr->EVENT.CHN, hdr->EVENT.N*sizeof(*(hdr->EVENT.CHN)) );

	hdr->EVENT.POS[N] = onset;	
	hdr->EVENT.DUR[N] = duration;
	hdr->EVENT.CHN[N] = 0;
	FreeTextEvent(hdr, N, description);
	return (hdr->AS.B4C_ERRNUM);
}

int biosig_write_annotation_utf8(int handle, size_t onset, size_t duration, const char *description) {
	fprintf(stdout,"biosig_write_annotation_latin1(): It's recommended to use biosig_write_annotation() instead.\n");
	return ( biosig_write_annotation(handle, onset, duration, description) );
}
int biosig_write_annotation_latin1(int handle, size_t onset, size_t duration, const char *description) {
	fprintf(stdout,"biosig_write_annotation_latin1(): It's recommended to use biosig_write_annotation() instead.\n");
	return ( biosig_write_annotation(handle, onset, duration, description) );
}

int biosig_set_datarecord_duration(int handle, double duration) {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	double spr   = hdr->SampleRate * duration;
	size_t rspr  = round(spr);
	if (fabs(spr - rspr) > 1e-8*spr) {
		fprintf(stderr,"Warning biosig_set_datarecord_duration(): number of samples is not integer (%g) - rounded to integers (%i)\n",spr,(int)rspr);
	}
	hdr->SPR = (size_t)rspr;
	return 0;
}

/******************************************************************************************
	biosig_unserialize_header: converts memory buffer into header structure HDR

	biosig_unserialize: converts memory buffer into header structure HDR, and
	if data != NULL, data samples will be read into a matrix,
		the starting address of this data matrix will be stored in *data
		point to *data
	input:
		mem : buffer
		len : length of buffer mem
		start: starting position to extract data
		length: number of samples for extracting data,
		flags: BIOSIG_FLAG_UCAL | BIOSIG_FLAG_OVERFLOWDETECTION | BIOSIG_FLAG_ROW_BASED_CHANNELS
	output:
		*data will contain start address to matrix data samples, of size
		hdr->NS * (hdr->SPR * hdr->NRec) or its transpose form depending on flags
	return value:
		header structure HDRTYPE* hdr.
 ******************************************************************************************/
HDRTYPE* biosig_unserialize(void *mem, size_t len, size_t start, size_t length, biosig_data_type **data, int flags) {

	fprintf(stdout,"%s (line %i) %s:\n",__FILE__,__LINE__,__func__);

	HDRTYPE *hdr = constructHDR(0,0);

	// decode header
	fprintf(stdout,"%s (line %i) %s:\n",__FILE__,__LINE__,__func__);

	hdr->AS.Header = mem;
	// in case of error, memory is deallocated through destructHDR(hdr);
	if (gdfbin2struct(hdr)) return(hdr);
	// in case of success, memory is managed by its own pointer *mem
	hdr->AS.Header = NULL;

	fprintf(stdout,"%s (line %i) %s:\n",__FILE__,__LINE__,__func__);

	// get data block
	biosig_set_flag(hdr, flags);
	if (data != NULL) {
		hdr->AS.rawdata = mem+hdr->HeadLen;
		size_t L = sread(*data, start, length, hdr);
		*data    = hdr->data.block;
		hdr->data.block = NULL;
	}
	hdr->AS.rawdata = NULL;

	fprintf(stdout,"%s (line %i) %s:\n",__FILE__,__LINE__,__func__);

	// read eventtable
	hdr->AS.rawEventData = (hdr->NRec != -1) ? mem + hdr->HeadLen + hdr->NRec*hdr->AS.bpb : NULL;
	rawEVT2hdrEVT(hdr, len - hdr->HeadLen - hdr->NRec*hdr->AS.bpb);

	// in case of success, memory is managed by its own pointer *mem
	hdr->AS.rawEventData = NULL;

	fprintf(stdout,"%s (line %i) %s:\n",__FILE__,__LINE__,__func__);

	return(hdr);
}

/******************************************************************************************
	biosig_serialize: converts header structure into memory buffer
	input:
		hdr: header structure, including event table.

	output:
		*mem will contain start address of buffer
		*len will contain length of buffer mem.
 ******************************************************************************************/

void* biosig_serialize(HDRTYPE *hdr, void **mem, size_t *len) {
	// encode header

	fprintf(stdout,"%s (line %i) %s:\n",__FILE__,__LINE__,__func__);

	hdr->TYPE=GDF;
	hdr->VERSION=3.0;

	struct2gdfbin(hdr);

	fprintf(stdout,"%s (line %i) %s:\n",__FILE__,__LINE__,__func__);

	// write event table
	size_t len3 = hdrEVT2rawEVT(hdr);

	fprintf(stdout,"%s (line %i) %s:\n",__FILE__,__LINE__,__func__);

	size_t len0 = hdr->HeadLen + hdr->NRec*hdr->AS.bpb + len3;
	char* M = (char*)realloc(*mem,len0);
	if (M == NULL) return(NULL);

	*mem = M;
	*len = len0;
	// write header into buffer
	memcpy(M, hdr->AS.Header, hdr->HeadLen);

	fprintf(stdout,"%s (line %i) %s:\n",__FILE__,__LINE__,__func__);

	// write data into buffer, and collapse unused channels
	size_t count = sread_raw(0, hdr->NRec, hdr, 1, M + hdr->HeadLen, hdr->NRec*hdr->AS.bpb);

	fprintf(stdout,"%s (line %i) %s:\n",__FILE__,__LINE__,__func__);

	// write event table into buffer
	memcpy(M + hdr->HeadLen + hdr->NRec*hdr->AS.bpb, hdr->AS.rawEventData, len3);

	fprintf(stdout,"%s (line %i) %s:\n",__FILE__,__LINE__,__func__);

	return(M);
}




#if defined(MAKE_EDFLIB)

int edfopen_file_writeonly(const char *path, int filetype, int number_of_signals) {
	enum FileFormat fmt=unknown; 
 	switch (filetype) {
	case EDFLIB_FILETYPE_EDF:
	case EDFLIB_FILETYPE_EDFPLUS:
		fmt = EDF;
		break;
	case EDFLIB_FILETYPE_BDF:
	case EDFLIB_FILETYPE_BDFPLUS:
		fmt = EDF;
		break;
	default:
		return(-1); 
	}
	return(biosig_open_file_writeonly(path, fmt, number_of_signals));
}

int edf_set_gender(int handle, int gender) {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	hdr->Patient.Sex = (gender==1) + (gender==0)*2 ;
}

int edfread_physical_samples(int handle, int edfsignal, int n, double *buf) {
	fprintf(stderr,"error: edfread_physical_samples - use biosig_read_physical_samples instead.\n");
	return(-1);
}

int edfread_digital_samples(int handle, int edfsignal, int n, int *buf) {
	fprintf(stderr,"error: edfread_digital_samples - use biosig_read_digital_samples instead.\n");
	return(-1);
}

long long edfseek(int handle, int channel, long long offset, int whence) {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL || hdrlist[handle].NS<=channel ) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;

	switch (whence) {
	case SEEK_SET:
		hdrlist[handle].chanpos[channel] = offset; // update position pointer of channel chan
		break;
	case SEEK_CUR:
		hdrlist[handle].chanpos[channel] += offset; // update position pointer of channel chan
		break;
	case SEEK_END: {
		CHANNEL_TYPE *hc = getChannelHeader(hdr,channel);
		hdrlist[handle].chanpos[channel] = hdr->NRec*hc->SPR + offset; // update position pointer of channel chan
		break;
		}
	}	
	return (hdrlist[handle].chanpos[channel]);
}

long long edftell(int handle, int channel) {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL || hdrlist[handle].NS<=channel ) return(-1);
	return ( hdrlist[handle].chanpos[channel] );
}

int edfrewind(int handle, int channel) {
/* It is equivalent to: (void) edf_seek(int handle, int biosig_signal, 0LL, SEEK_SET) */
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL || hdrlist[handle].NS<=channel ) return(-1);
	hdrlist[handle].chanpos[channel] = 0;
	return(0);
}

int edf_get_annotation(int handle, int n, struct edf_annotation_struct *annot) {
	if (handle<0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;

	annot->onset = hdr->EVENT.POS[n]*1e4/hdr->EVENT.SampleRate;
	annot->duration = hdr->EVENT.DUR[n]*1e4/hdr->EVENT.SampleRate;
	strncpy(annot->annotation,GetEventDescription(hdr, n),sizeof(annot->annotation));

	return(0);
}

int edfwrite_annotation(int handle, size_t onset, size_t duration, const char *description) {
	/* onset and duration are multiples of 100 microseconds */
	if (handle < 0 || handle >= hdrlistlen || hdrlist[handle].hdr==NULL) return(-1);
	HDRTYPE *hdr = hdrlist[handle].hdr;
	return (biosig_write_annotation(handle, onset*1e-4*hdr->EVENT.SampleRate, duration*1e-4*hdr->EVENT.SampleRate, description));
}

/*
   TODO: the following functions neeed to be implemented 	
*/
int edf_set_recording_additional(int handle, const char *recording_additional) {
	return fprintf(stderr,"this function is not implemented, yet.\n");
}

int edfwrite_physical_samples(int handle, double *buf) {
	return fprintf(stderr,"this function is not implemented, yet.\n");
}

int edf_blockwrite_physical_samples(int handle, double *buf) {
	return fprintf(stderr,"this function is not implemented, yet.\n");
}

/*
int edfwrite_digital_samples(int handle, int *buf) {
	return fprintf(stderr,"this function is not implemented, yet.\n");
}
*/

int edf_blockwrite_digital_samples(int handle, int *buf) {
	return fprintf(stderr,"this function is not implemented, yet.\n");
}


#endif 


