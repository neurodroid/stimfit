/*

    Copyright (C) 2010,2011,2012,2015,2016 Alois Schloegl <alois.schloegl@ist.ac.at>

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
#include <ctype.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "../biosig-dev.h"

#define min(a,b)        (((a) < (b)) ? (a) : (b))

uint16_t cfs_data_type(uint8_t dataType) {
	switch (dataType) {
	case 0:	//int8
		return 1;
	case 1: //uint8
		return 3;
	case 2: //int16
		return 3;
	case 3:	// uint16
		return 4;
	case 4: // int32
		return 5;
	case 5: //float
		return 16;
	case 6: // double
		return 17;
	case 7: // char
		return 0;
	default:
		return 0xffff;
	}
}


/*
    trim_trailing_space:
	trims trailing whitespaces from pascal string (pstr), and the string
	becomes always 0-terminated. This may mean that the last string character
	might be deleted, if all bytes are filled with non-zero values.

	no memory allocation or reallication is performed, maxLength+1 is
	the size of the memory allocation for pstr (pstr[0] to pstr[maxLength]).
	maxLength specify the string length w/o count or terminating zero byte.

	pascal string: first byte contains lengths, followed by characters, not null-terminated
*/
char *trim_trailing_space(uint8_t *pstr, unsigned maxLength) {
	unsigned len = min(pstr[0], maxLength);
	while (isspace(pstr[len]) && (len>0)) {
		len--;
	}

	if (len==maxLength) fprintf(stdout,"Warning %s: last and %i-th  character of string <%c%c%c%c...> has been deleted\n",__func__,len,pstr[1],pstr[2],pstr[3],pstr[4]);

	len = min(len+1,maxLength);
	pstr[len]=0;
	pstr[0]=len;

	return (pstr+1);
}


const char *Signal6_StateTable[]={
                        "","State 1","State 2","State 3","State 4","State 5","State 6","State 7","State 8","State 9",
		"State 10","State 11","State 12","State 13","State 14","State 15","State 16","State 17","State 18","State 19",
		"State 20","State 21","State 22","State 23","State 24","State 25","State 26","State 27","State 28","State 29",
		"State 30","State 31","State 32","State 33","State 34","State 35","State 36","State 37","State 38","State 39",
		"State 40","State 41","State 42","State 43","State 44","State 45","State 46","State 47","State 48","State 49",
		"State 50","State 51","State 52","State 53","State 54","State 55","State 56","State 57","State 58","State 59",
		"State 60","State 61","State 62","State 63","State 64","State 65","State 66","State 67","State 68","State 69",
		"State 70" };

EXTERN_C void sopen_cfs_read(HDRTYPE* hdr) {
/*
	this function will be called by the function SOPEN in "biosig.c"

	Input:
		char* Header	// contains the file content

	Output:
		HDRTYPE *hdr	// defines the HDR structure accoring to "biosig.h"
*/

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s:%i sopen_cfs_read started - %i bytes already loaded\n",__FILE__,__LINE__,hdr->HeadLen);

#define H1LEN (8+14+4+8+8+2+2+2+2+2+4+2+2+74+4+40)

		size_t count = hdr->HeadLen;
		char flag_FPulse = 0;    // indicates whether data was recorded using FPulse

#define CFS_NEW		// this flag allows to switch back to old version

		while (!ifeof(hdr)) {
			hdr->AS.Header = (uint8_t*) realloc(hdr->AS.Header,count*2+1);
			count += ifread(hdr->AS.Header+count,1,count,hdr);
		}
		hdr->AS.Header[count] = 0;
		hdr->FLAG.OVERFLOWDETECTION = 0;

		/*
			Implementation is based on the following reference:
		        CFS - The CED Filing System October 2006
		*/

		/* General Header */
		// uint32_t filesize = leu32p(hdr->AS.Header+22);	// unused
		hdr->FILE.size = leu32p(hdr->AS.Header+0x16);	//    file size
		hdr->NS    = leu16p(hdr->AS.Header+0x2a);	// 6  number of channels
		uint16_t n = leu16p(hdr->AS.Header+0x2c);	// 7  number of file variables
		uint16_t d = leu16p(hdr->AS.Header+0x2e);	// 8  number of data section variables
		uint16_t FileHeaderSize = leu16p(hdr->AS.Header+0x30);	// 9  byte size of file header
		uint16_t DataHeaderSize = leu16p(hdr->AS.Header+0x32);	// 10 byte size of data section header
		uint32_t LastDataSectionHeaderOffset = leu32p(hdr->AS.Header+0x34);	// 11 last data section header offset
		uint16_t NumberOfDataSections = leu16p(hdr->AS.Header+0x38);	// 12 last data section header offset

		// decode start time and date
		{
			struct tm t;
			char strtmp[9];
			memcpy(strtmp, hdr->AS.Header+0x1a, 8);
			strtmp[8] = 0; // terminating null character
if (VERBOSE_LEVEL>7) fprintf(stdout,"%s:%i sopen_cfs_read started [%s]\n",__FILE__,__LINE__,strtmp);

			t.tm_hour = atoi(strtok(strtmp,":/"));
			t.tm_min  = atoi(strtok(NULL,":/"));
			t.tm_sec  = atoi(strtok(NULL,":/"));
			memcpy(strtmp, hdr->AS.Header+0x22, 8);
			t.tm_mday = atoi(strtok(strtmp,"-:/"));
			t.tm_mon  = atoi(strtok(NULL,"-:/"))-1;
			t.tm_year = atoi(strtok(NULL,"-:/"));
			if (t.tm_year<39) t.tm_year+=100;
			hdr->T0 = tm_time2gdf_time(&t);
		}

		if (NumberOfDataSections) {
			hdr->EVENT.TYP = (typeof(hdr->EVENT.TYP)) realloc(hdr->EVENT.TYP, (hdr->EVENT.N + 2*NumberOfDataSections - 1) * sizeof(*hdr->EVENT.TYP));
			hdr->EVENT.POS = (typeof(hdr->EVENT.POS)) realloc(hdr->EVENT.POS, (hdr->EVENT.N + 2*NumberOfDataSections - 1) * sizeof(*hdr->EVENT.POS));
			hdr->EVENT.TimeStamp = (typeof(hdr->EVENT.TimeStamp)) realloc(hdr->EVENT.TimeStamp, (hdr->EVENT.N + 2*NumberOfDataSections - 1) * sizeof(*hdr->EVENT.TimeStamp));
		}

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) CFS - %d,%d,%d,0x%x,0x%x,0x%x,%d,0x%x\n",__FILE__,__LINE__,hdr->NS,n,d,FileHeaderSize,DataHeaderSize,LastDataSectionHeaderOffset,NumberOfDataSections,leu32p(hdr->AS.Header+0x86));

		/* channel information */
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
#define H2LEN (22+10+10+1+1+2+2)
 		char* H2 = (char*)(hdr->AS.Header + H1LEN);
		double xPhysDimScale[100];		// CFS is limited to 99 channels
		typeof(hdr->NS) NS = hdr->NS;
		uint8_t k;
		uint32_t bpb=0;
		for (k = 0; k < hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL + k;
			/*
				1 offset because CFS uses pascal type strings (first byte contains string length)
				in addition, the strings are \0 terminated.
			*/
			hc->OnOff = 1;
			hc->LeadIdCode = 0;
			hc->bi8 = 0;

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s(line %i) Channel #%i/%i: %i<%s>/%i<%s>\n",__FILE__,__LINE__,k+1, hdr->NS, H2[22 + k*H2LEN], H2 + 23 + k*H2LEN, H2[32 + k*H2LEN], H2 + 33 + k*H2LEN);

			strncpy(hc->Label,              trim_trailing_space(H2 + k*H2LEN, 20), 21);	// Channel name
			hc->PhysDimCode  = PhysDimCode (trim_trailing_space(H2 + 22 + k*H2LEN,9));		// Y axis units
			xPhysDimScale[k] = PhysDimScale(PhysDimCode(trim_trailing_space(H2 + 32 + k*H2LEN,9)));	// X axis units

			uint8_t  dataType  = H2[42 + k*H2LEN];
			uint8_t  dataKind  = H2[43 + k*H2LEN];
			//uint16_t byteSpace = leu16p(H2+44 + k*H2LEN);		// stride
			//uint16_t next      = leu16p(H2+46 + k*H2LEN);
			hc->GDFTYP = cfs_data_type(dataType);
			if (hc->GDFTYP > 580)	// sizeof GDFTYP_BITS[]
				biosigERROR(hdr, B4C_DATATYPE_UNSUPPORTED, "CFS: unknown/unsupported data type !");

			if (H2[43 + k * H2LEN] == 1) {
				// Matrix data does not return an extra channel, but contains Markers and goes into the event table.
				hc->OnOff = 0;
				NS--;
			}
			hc->bi = bpb;
			bpb += GDFTYP_BITS[hc->GDFTYP]>>3;	// per single sample
			hc->Impedance = NAN;
			hc->TOffset  = NAN;
			hc->LowPass  = NAN;
			hc->HighPass = NAN;
			hc->Notch    = NAN;
			hc->XYZ[0]   = NAN;
			hc->XYZ[1]   = NAN;
			hc->XYZ[2]   = NAN;

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s(line %i) Channel #%i: [%s](%i/%i) <%s>/<%s> ByteSpace%i,Next#%i\n",
			__FILE__, __LINE__, k+1, H2 + 1 + k*H2LEN, dataType, dataKind,
			H2 + 23 + k*H2LEN, H2 + 33 + k*H2LEN, leu16p(H2+44+k*H2LEN), leu16p(H2+46+k*H2LEN));

		}
		hdr->AS.bpb = bpb;

		size_t datapos = H1LEN + H2LEN*hdr->NS;

		/* file variable information :
		   will be extracted as application specific information
		   // n*36 bytes
                 */
if (VERBOSE_LEVEL>7) fprintf(stdout,"\n******* file variable information (n=%i) *********\n", n);
		hdr->AS.bci2000= realloc(hdr->AS.bci2000, 3);
		hdr->AS.bci2000[0]='\0';
		for (k = 0; k < n; k++) {
			int i=-1; double f=NAN;
			size_t pos   = datapos + k*36;
			char   *desc = (char*)(hdr->AS.Header+pos+1);
			uint16_t typ = leu16p(hdr->AS.Header+pos+22);
			char   *unit = (char*)(hdr->AS.Header+pos+25);
			uint16_t off = leu16p(hdr->AS.Header+pos+34);

//			if (flag_FPulse && !strcmp(desc, "Spare")) continue;

			/*
			   H1LEN 	General Header
			   H2LEN*NS 	Channel Information a 48 byte
			   n*36		File Variable information a 36 byte
			   d*36		DS Variable information
			*/
			size_t p3 = H1LEN + H2LEN*hdr->NS + (n+d+2)*36 + off;
			/***** file variable k *****/
			switch (typ) {
			case 0:
			case 1: i = hdr->AS.Header[p3]; break;
			case 2: i = lei16p(hdr->AS.Header+p3); break;
			case 3: i = leu16p(hdr->AS.Header+p3); break;
			case 4: i = lei32p(hdr->AS.Header+p3); break;
			case 5: f = lef32p(hdr->AS.Header+p3); break;
			case 6: f = lef64p(hdr->AS.Header+p3); break;
			}
if (VERBOSE_LEVEL>8) 	{
			fprintf(stdout,"#%2i [%i:%i] <%s>",k,typ,off,desc);
			if (typ<1) ;
			else if (typ<5) fprintf(stdout,"[%d]",i);
			else if (typ<7) fprintf(stdout,"[%g]",f);
			else if (typ==7) fprintf(stdout,"[%s]",hdr->AS.Header+p3+1);
			fprintf(stdout,"[%s]\n",unit);
			}
else if (VERBOSE_LEVEL>7)
			{
			if (typ==7) fprintf(stdout,"#%2i [%i:] <%s> <%s> <%s>\n",k,typ,desc, hdr->AS.Header+p3+1,unit);
			}

			if ((typ==7) && !strncmp(desc,"Script",6)) {
				char *scriptline=(char*)hdr->AS.Header+p3+1;
				if (!strncmp(desc,"ScriptBlock",11)) {
					// only the script block is extracted
					assert(hdr->AS.Header[p3]==strlen(scriptline));
					assert(hdr->AS.Header[p3+1+hdr->AS.Header[p3]]==0);

					// replace '\r' <CR> with '\n' <NEWLINE> in scriptline
					while (*scriptline) {
						switch (*scriptline) {
						case '\r': *scriptline='\n';
						}
						scriptline++;
					}

					scriptline=(char*)hdr->AS.Header+p3+1;
					hdr->AS.bci2000=realloc(hdr->AS.bci2000, strlen(hdr->AS.bci2000) + strlen(scriptline) + 1);
					strcat(hdr->AS.bci2000, scriptline);		// Flawfinder: ignore
				}
				else if (!strncmp(desc,"Scriptline",10)) {
					hdr->AS.bci2000=realloc(hdr->AS.bci2000, strlen(hdr->AS.bci2000) + strlen(scriptline) + 3);
					strcat(hdr->AS.bci2000, scriptline);	// Flawfinder: ignore
					strcat(hdr->AS.bci2000, "\n");		// Flawfinder: ignore
				}
			}
			if (k==0) {
				// File variable zero
				flag_FPulse = !strcmp(unit,"FPulse");

				if (typ==2)
					hdr->VERSION = i/100.0;
				else
					fprintf(stderr,"Warning (%s line %i): File variable zeros is not of type INT2\n",__func__,__LINE__);

				char *e = (char*)&(hdr->ID.Equipment);
				memcpy(e,unit,8);
				lei16a(i,e+6);
			}
		}
		hdr->ID.Manufacturer.Name = "CED - Cambridge Electronic Devices";
		hdr->ID.Manufacturer.Model = NULL;
		hdr->ID.Manufacturer.Version = NULL;
		hdr->ID.Manufacturer.SerialNumber = NULL;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"\n******* Data Section variable information (n=%i,%i)*********\n", d,NumberOfDataSections);
		datapos = LastDataSectionHeaderOffset; //H1LEN + H2LEN*hdr->NS + n*36;
		// reverse order of data sections
		uint32_t *DATAPOS = (uint32_t*)malloc(sizeof(uint32_t)*NumberOfDataSections);

		uint16_t m;
		for (m = NumberOfDataSections; 0 < m; ) {
			DATAPOS[--m] = datapos;
			datapos      = leu32p(hdr->AS.Header + datapos);
			if (VERBOSE_LEVEL>7) fprintf(stdout, "%s:%i sopen_cfs_read started: section %"PRIi16" pos %"PRIuPTR" 0x%"PRIxPTR"\n",__FILE__,__LINE__,m,datapos,datapos);
		}

		if (hdr->AS.SegSel[0] > NumberOfDataSections) {
			fprintf(stderr,"Warning loading CFS file: selected sweep number is larger than number of sweeps [%d,%d] - no data is loaded\n",hdr->AS.SegSel[0], NumberOfDataSections);
			NumberOfDataSections = 0;
		}
		else if (0 < hdr->AS.SegSel[0]) {
			// hack: if sweep is selected, use same method than for data with a single sweep
			DATAPOS[0] = DATAPOS[hdr->AS.SegSel[0]-1];
			NumberOfDataSections = 1;
		}

//		void *VarChanInfoPos = hdr->AS.Header + datapos + 30;  // unused
		char flag_ChanInfoChanged = 0;
		hdr->NRec = NumberOfDataSections;
		int Signal6_CodeDescLen = 0;
		size_t SPR = 0, SZ = 0;
		for (m = 0; m < NumberOfDataSections; m++) {
			int32_t tmp_event_typ=0;
			double  tmp_event_pos=0;

			datapos = DATAPOS[m];
			for (k = 0; k < d; k++) {
				/***** DS variable k information *****/
				int i=-1; double f=NAN;
				size_t pos = H1LEN + H2LEN*hdr->NS + (n+k+1)*36;
				char   *desc = (char*)(hdr->AS.Header+pos+1);
				uint16_t typ = leu16p(hdr->AS.Header+pos+22);
				char   *unit = (char*)(hdr->AS.Header+pos+25);
				uint16_t off = leu16p(hdr->AS.Header+pos+34);

				if (flag_FPulse && !strcmp(desc, "Spare")) continue;

				/***** data section variable k *****/
				size_t p3   = DATAPOS[m] + 30 + 24 * hdr->NS + off;

				switch (typ) {
				case 0:
				case 1: i = hdr->AS.Header[p3]; break;
				case 2: i = lei16p(hdr->AS.Header+p3); break;
				case 3: i = leu16p(hdr->AS.Header+p3); break;
				case 4: i = lei32p(hdr->AS.Header+p3); break;
				case 5: f = lef32p(hdr->AS.Header+p3); break;
				case 6: f = lef64p(hdr->AS.Header+p3); break;
				}

				if ((k==1) && (typ==4) && !strcmp(desc,"State")) {
					tmp_event_typ = i;
					if ((Signal6_CodeDescLen < i) && (i < 256))
						Signal6_CodeDescLen = i;
				}
				else if ((k==3) && (typ==6) && !strcmp(desc,"Start") && !strcmp(unit,"s"))
					tmp_event_pos = f;

if (VERBOSE_LEVEL>7) 		{
				fprintf(stdout,"#%2i [%i:%i] <%s>",k,typ,off,desc);
				if (typ<1) ;
				else if (typ<5) fprintf(stdout,"[%d]",i);
				else if (typ<7) fprintf(stdout,"[%g]",f);
				else if (typ==7) fprintf(stdout,"[%s]",hdr->AS.Header+p3+1);
				fprintf(stdout,"[%s]\n",unit);
				}
			}

			typeof (hdr->T0) TS = tmp_event_pos ? (hdr->T0 + ldexp(tmp_event_pos/(24.0*3600.0),32)) : 0;
			if (m>0) {
				hdr->EVENT.TYP[hdr->EVENT.N] = 0x7ffe;
				hdr->EVENT.POS[hdr->EVENT.N] = SPR;
				hdr->EVENT.TimeStamp[hdr->EVENT.N] = TS;
				hdr->EVENT.N++;
			}
			if (tmp_event_typ > 0) {
				hdr->EVENT.TYP[hdr->EVENT.N] = tmp_event_typ;
				hdr->EVENT.POS[hdr->EVENT.N] = SPR;
				hdr->EVENT.TimeStamp[hdr->EVENT.N] = TS;
				hdr->EVENT.N++;
			}

			if (!leu32p(hdr->AS.Header+datapos+8)) continue; 	// empty segment

			if (VERBOSE_LEVEL>7) fprintf(stdout,"\n******* DATA SECTION --%03i-- %i *********\n",m,flag_ChanInfoChanged);
			if (VERBOSE_LEVEL>7) fprintf(stdout,"\n[DS#%3i] 0x%x 0x%x [0x%x 0x%x szChanData=%i] 0x02%x\n", m, FileHeaderSize, \
						(int)datapos, leu32p(hdr->AS.Header+datapos), leu32p(hdr->AS.Header+datapos+4), \
						leu32p(hdr->AS.Header+datapos+8), leu16p(hdr->AS.Header+datapos+12));

			uint32_t sz    = 0;
			uint32_t bpb   = 0, spr = 0;
			hdr->AS.first  = 0;
			hdr->AS.length = 0;
			char flag_firstchan = 1;
			uint32_t xspr0 = 0;

			for (k = 0; k < hdr->NS; k++) {
				uint8_t *pos = hdr->AS.Header + datapos + 30 + 24 * k;

				CHANNEL_TYPE *hc = hdr->CHANNEL + k;

				// uint32_t bi  = leu32p(pos);	// unused
				uint32_t xspr = leu32p(pos+4);
				float Cal    = lef32p(pos+8);
				float Off    = lef32p(pos+12);
				double XCal  = lef32p(pos+16);
				double XOff  = lef32p(pos+20);// unused
				if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): %i/%i %i/%i %f+%f change SPR:%i->%i, Cal:%f->%f, Off: %f->%f\n", \
							__FILE__,__LINE__, (int)m, (int)NumberOfDataSections, (int)k,(int)hdr->NS, XCal, XOff, \
							(int)hc->SPR,(int)xspr,hc->Cal,Cal,hc->Off,Off);

				if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) [%i %i] %i %i %p %i\n",
							__FILE__, __LINE__, m, k, xspr0, xspr, pos, (int)(datapos + 30 + 24 * k));

				if (xspr0 == 0)
					xspr0 = xspr;
				else if (xspr>0)
					xspr0 = lcm(xspr0,xspr);
				else if (xspr==0)
					if (VERBOSE_LEVEL>7) fprintf(stdout,"Warning : #%i of section %i contains %i samples \n", (int)k, (int)m, (int)xspr);

				if (m > 0) {
					if ( (hc->Cal != Cal)
					  || (hc->Off != Off)
					   )
					switch (hc->GDFTYP) {
					case 0:
					case 1:
					case 2:
					case 3:
					case 4:
					case 5:
					case 6:
					case 16:
						hc->GDFTYP = 16; 	// float, single precision
						break;
					case 7:
					case 8:
					case 17:
						hc->GDFTYP = 17; 	// double
						break;
					case 18:  	// quadruple
					default: ;
					}
				}
				else {
					hc->Cal = Cal;
					hc->Off = Off;
				}

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): %i #%i: SPR=%i=%i=%i  x%f+-%f %i x%g %g %g\n",
					__func__,__LINE__,m,k,spr,(int)SPR,hc->SPR,hc->Cal,hc->Off,hc->bi,xPhysDimScale[k], XCal, XOff);

				double Fs = 1.0 / (xPhysDimScale[k] * XCal);
				if ( (hc->OnOff == 0) || (XCal == 0.0) ) {
					; // do nothing:
				}
				else if (flag_firstchan) {
					hdr->SampleRate = Fs;
					flag_firstchan = 0;
				}
				else if (fabs(hdr->SampleRate - Fs) > 1e-3) {
					biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "CED/CFS: different sampling rates are not supported\n");
				}

				sz  += hc->SPR * GDFTYP_BITS[hc->GDFTYP] >> 3;
				hc->bi = bpb;
				bpb += GDFTYP_BITS[hc->GDFTYP]>>3;	// per single sample
				hdr->AS.length += hc->SPR;

			}

			SPR += xspr0;
			SZ  += sz;
			hdr->AS.bpb = bpb;
		}
		if (Signal6_CodeDescLen > 0) {

			if (Signal6_CodeDescLen > 70) {
				// 19 is sizeof Signal6_StateTable
				fprintf(stderr, "Warning %s (line %i): Event code description exceed table (Signal6_CodeDescLen=%i)\n",__func__,__LINE__,Signal6_CodeDescLen);
				Signal6_CodeDescLen = 70;
			}

			hdr->EVENT.LenCodeDesc = Signal6_CodeDescLen+1; //70 is sizeof(Signal6_StateTable)
			hdr->EVENT.CodeDesc    = realloc(hdr->EVENT.CodeDesc, sizeof(char*) * (Signal6_CodeDescLen+1));
			for (int k=0; k <= Signal6_CodeDescLen; k++) {
				hdr->EVENT.CodeDesc[k] = Signal6_StateTable[k];
			}
		}

		/*
		The previous and the next loop have been split, in order to avoid
		multiple reallocations. Multiple reallocations seem to be a major
		issue for performance problems of Windows (a 50 MB file with 480 sweeps took 50 s to load).
		However, the two loops might contain some redundant operations.
		*/
		hdr->AS.flag_collapsed_rawdata = 0;
		void *ptr = realloc(hdr->AS.rawdata, hdr->AS.bpb * SPR);
		if (!ptr) biosigERROR(hdr,B4C_MEMORY_ALLOCATION_FAILED, "CFS: memory allocation failed in line __LINE__");
		hdr->AS.rawdata = (uint8_t*)ptr;

		SPR = 0, SZ = 0;
		for (m = 0; m < NumberOfDataSections; m++) {

			datapos = DATAPOS[m];
			if (!leu32p(hdr->AS.Header+datapos+8)) continue; 	// empty segment

			uint32_t sz    = 0;
			uint32_t bpb   = 0, spr = 0;
			hdr->AS.first  = 0;
			hdr->AS.length = 0;
			char flag_firstchan = 1;
			uint32_t xspr0 = 0;
			float Cal, Off;
			for (k = 0; k < hdr->NS; k++) {
				uint8_t *pos = hdr->AS.Header + datapos + 30 + 24 * k;

				CHANNEL_TYPE *hc = hdr->CHANNEL + k;

				//uint32_t bi = leu32p(pos);	// unused
				uint32_t xspr = leu32p(pos+4);
				hc->SPR     = leu32p(pos+4);
				hc->Cal     = lef32p(pos+8);
				hc->Off     = lef32p(pos+12);
				Cal         = lef32p(pos+8);
				Off         = lef32p(pos+12);
				double XCal = lef32p(pos+16);
				double XOff = lef32p(pos+20);// unused

				if (xspr0 == 0)
					xspr0 = xspr;
				else if (xspr>0)
					xspr0 = lcm(xspr0, xspr);
				else if (xspr==0)
					if (VERBOSE_LEVEL>7) fprintf(stdout,"Warning : #%i of section %i contains %i samples \n", (int)k, (int)m, (int)xspr);

				assert( (xspr==0) || ((xspr0 % xspr) == 0) );

				double Fs = 1.0 / (xPhysDimScale[k] * XCal);
				if ( (hc->OnOff == 0) || (XCal == 0.0) ) {
					; // do nothing:
				}
				else if (flag_firstchan) {
					hdr->SampleRate = Fs;
					flag_firstchan = 0;
				}
				else if (fabs(hdr->SampleRate - Fs) > 1e-3) {
					biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "CED/CFS: different sampling rates are not supported\n");
				}

				sz  += xspr * GDFTYP_BITS[hc->GDFTYP] >> 3;
				hc->bi = bpb;
				bpb += GDFTYP_BITS[hc->GDFTYP]>>3;	// per single sample
				hdr->AS.length += xspr;

			}

			if (xspr0 > 0) {
				/* hack: copy data into a single block (rawdata)
				   this is used as a data cache, no I/O is needed in sread, at the cost that sopen is slower
				   sread_raw does not attempt to reload the data
				 */

				hdr->AS.first = 0;
				uint8_t ns = 0;
				for (k = 0; k < hdr->NS; k++) {
					uint8_t *pos = hdr->AS.Header + datapos + 30 + 24 * k;
					CHANNEL_TYPE *hc = hdr->CHANNEL + k;

					uint32_t memoffset = leu32p(pos);
					uint8_t *srcaddr = hdr->AS.Header + leu32p(hdr->AS.Header+datapos + 4) ;
					//uint16_t byteSpace = leu16p(H2+44 + k*H2LEN);

					uint8_t  dataType  = H2[42 + k*H2LEN];
					uint8_t  dataKind  = H2[43 + k*H2LEN];			// equidistant, Subsidiary or Matrix data
					uint16_t stride    = leu16p(H2+44 + k*H2LEN);		// byteSpace
					uint16_t next      = leu16p(H2+46 + k*H2LEN);
					uint16_t gdftyp    = cfs_data_type(dataType);
					if (gdftyp > 580)	// sizeof GDFTYP_BITS[]
						biosigERROR(hdr, B4C_DATATYPE_UNSUPPORTED, "CFS: unknown/unsupported data type !");

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) #%i %i,%i %i %i %i %i: %i @%p %i\n",
					__func__, __LINE__, k, dataType, dataKind, hc->SPR, gdftyp,hc->GDFTYP, stride, memoffset,
					srcaddr, leu32p(hdr->AS.Header+datapos + 4) + leu32p(hdr->AS.Header + datapos + 30 + 24 * k));

					if (hc->OnOff==0) continue;	// Time and Marker channels are not supported, yet

					if ((hc->SPR > 0) && (xspr0 != hc->SPR)) {
						biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "CED/CFS: samples-per-record (SPR) changes between channels - this is not supported yet.");
					}

					size_t k2;
					if (gdftyp == hc->GDFTYP) {
					    for (k2 = 0; k2 < xspr0; k2++) {

						uint8_t *ptr = srcaddr + memoffset + k2*stride;
						uint8_t *dest = hdr->AS.rawdata + hc->bi + (SPR + k2) * hdr->AS.bpb;
						double val=NAN;

						switch (gdftyp) {
						// reorder for performance reasons - more frequent gdftyp's come first
						case 3:  //val = lei16p(ptr);
						case 4:  //val = leu16p(ptr);
							memcpy(dest, ptr, 2);
							break;
						case 16: //val = lef32p(ptr);
							memcpy(dest, ptr, 4);
							break;
						case 7:  //val = lei64p(ptr);
						case 8:  //val = leu64p(ptr);
						case 17: //val = lef64p(ptr);
							memcpy(dest, ptr, 8);
							break;
						case 0:  //val = *(   char*) ptr;
						case 1:  //val = *( int8_t*) ptr;
						case 2:  //val = *(uint8_t*) ptr;
							*dest = *ptr;
							break;
						case 5:  val = lei32p(ptr);
						case 6:  //val = leu32p(ptr);
							memcpy(dest, ptr, 4);
							break;
						default:
							val = NAN;
							biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "CED/CFS: invalid data type");
						}

						if (hc->OnOff) continue;

						if (!strncmp(hc->Label,"Marker",6) && hc->PhysDimCode==2176 && hc->GDFTYP==5 && next != 0) {
							// matrix data might contain time markers.

							// memory allocation for additional events - more efficient implementation would be nice
							hdr->EVENT.TYP = (typeof(hdr->EVENT.TYP)) realloc(hdr->EVENT.TYP, (hdr->EVENT.N + NumberOfDataSections) * sizeof(*hdr->EVENT.TYP));
							hdr->EVENT.POS = (typeof(hdr->EVENT.POS)) realloc(hdr->EVENT.POS, (hdr->EVENT.N + NumberOfDataSections) * sizeof(*hdr->EVENT.POS));

							/*
							char Desc[2]; Desc[0] = srcaddr[hdr->CHANNEL[next].bi + k2*stride]; Desc[1] = 0;
								this does currently not work because FreeTextEvent expects that
								the string constant is available as long as hdr, which is not the case here.
							*/

						 	// typically a single character within a 4 byte integer, this should be sufficient to ensure \0 termination
							char *Desc = (char*)srcaddr + hdr->CHANNEL[next].bi + k2 * stride;
							Desc[1] = 0;
							FreeTextEvent(hdr, hdr->EVENT.N, Desc);
							hdr->EVENT.POS[hdr->EVENT.N] = lround( (val * hc->Cal + hc->Off) * hdr->SampleRate) + SPR;
							hdr->EVENT.N++;
						}
					    }
					}
					else {
					    for (k2 = 0; k2 < xspr0; k2++) {

						uint8_t *ptr = srcaddr + memoffset + k2*stride;
						uint8_t *dest = hdr->AS.rawdata + hc->bi + (SPR + k2) * hdr->AS.bpb;
						double val;

						if (hc->GDFTYP==16) {	// float
							switch (gdftyp) {
							// reorder for performance reasons - more frequent gdftyp's come first
							case 3:  val = lei16p(ptr) * Cal + Off;
								break;
							case 4:  val = leu16p(ptr) * Cal + Off;
								break;
							case 16: val = lef32p(ptr) * Cal + Off;
								break;
							case 7:  val = lei64p(ptr) * Cal + Off;
								break;
							case 8:  val = leu64p(ptr) * Cal + Off;
								break;
							case 17: val = lef64p(ptr) * Cal + Off;
								break;
							case 0:  val = (*(   char*) ptr) * Cal + Off;
								break;
							case 1:  val = (*( int8_t*) ptr) * Cal + Off;
								break;
							case 2:  val = (*(uint8_t*) ptr) * Cal + Off;
								break;
							case 5:  val = lei32p(ptr) * Cal + Off;
								break;
							case 6:  val = leu32p(ptr) * Cal + Off;
								break;
							default:
								val = NAN;
								biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "CED/CFS: invalid data type");
							}

							lef32a(val, dest);	// hdr->AS.rawdata uses the native endian format of the platform
						}
						else {
							// any conversion to anything but float is not supported yet.
							biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "CED/CFS: data type changes between segments - this is not supported yet.");
						}

						// scaling has been already consider when converting to float, do not scale in sread anymore.
						hc->Cal = 1.0;
						hc->Off = 0.0;
						if (hc->OnOff) continue;

						if (!strncmp(hc->Label,"Marker",6) && hc->PhysDimCode==2176 && hc->GDFTYP==5 && next != 0) {
							// matrix data might contain time markers.

							// memory allocation for additional events - more efficient implementation would be nice
							hdr->EVENT.TYP = (typeof(hdr->EVENT.TYP)) realloc(hdr->EVENT.TYP, (hdr->EVENT.N + NumberOfDataSections) * sizeof(*hdr->EVENT.TYP));
							hdr->EVENT.POS = (typeof(hdr->EVENT.POS)) realloc(hdr->EVENT.POS, (hdr->EVENT.N + NumberOfDataSections) * sizeof(*hdr->EVENT.POS));

							/*
							char Desc[2]; Desc[0] = srcaddr[hdr->CHANNEL[next].bi + k2*stride]; Desc[1] = 0;
								this does currently not work because FreeTextEvent expects that
								the string constant is available as long as hdr, which is not the case here.
							*/

							// typically a single character within a 4 byte integer, this should be sufficient to ensure \0 termination
							char *Desc = (char*)srcaddr + hdr->CHANNEL[next].bi + k2 * stride;
							Desc[1] = 0;
							FreeTextEvent(hdr, hdr->EVENT.N, Desc);
							hdr->EVENT.POS[hdr->EVENT.N] = lround( (val * hc->Cal + hc->Off) * hdr->SampleRate) + SPR;
							hdr->EVENT.N++;
						}
					    }
					}
					ns += hc->OnOff;
				}
			}

			SPR += xspr0;
			SZ  += sz;

			datapos = leu32p(hdr->AS.Header + datapos);
		}
		free(DATAPOS);

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): SPR=%i=%i NRec=%i  @%p\n",__FILE__,__LINE__,(int)SPR,hdr->SPR,(int)hdr->NRec, hdr->AS.rawdata);

		// set variables such that sread_raw does not attempt to reload the data
		hdr->AS.first = 0;
		hdr->EVENT.SampleRate = hdr->SampleRate;
		if (NumberOfDataSections < 1) {
			hdr->SPR = 0;
		}
		else  {
			hdr->SPR       = 1;
			hdr->NRec      = SPR;
			hdr->AS.length = SPR;

			size_t bpb = 0;
			for (k = 0; k < hdr->NS; k++) {
				CHANNEL_TYPE *hc = hdr->CHANNEL + k;
				assert(hc->bi==bpb);
				bpb        += GDFTYP_BITS[hc->GDFTYP] >> 3;
				hc->SPR     = hdr->SPR;
			}
			assert(hdr->AS.bpb==bpb);
		}

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): SPR=%i=%i NRec=%i\n",__func__,__LINE__,(int)SPR,hdr->SPR,(int)hdr->NRec);

		datapos   = FileHeaderSize;  //+DataHeaderSize;

		if (flag_ChanInfoChanged) {
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "CED/CFS: varying channel information not supported");
		}

		for (k = 0; k < hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL + k;
			switch (hc->GDFTYP) {
			case 0:
			case 1:
				hc->DigMax  =  127;
				hc->DigMin  = -128;
				break;
			case 2:
				hc->DigMax  =  255;
				hc->DigMin  =  0;
				break;
			case 3:
				hc->DigMax  = (int16_t)0x7fff;
				hc->DigMin  = (int16_t)0x8000;
				break;
			case 4:
				hc->DigMax  = 0xffff;
				hc->DigMin  = 0;
				break;
			case 16:
			case 17:
				hc->DigMax  =  1e9;
				hc->DigMin  = -1e9;
				break;
			}
if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (%i): %f %f %f %f %f %f\n",__FILE__,__LINE__,hc->PhysMax, hc->DigMax, hc->Cal, hc->Off,hc->PhysMin, hc->DigMin );
			hc->PhysMax = hc->DigMax * hc->Cal + hc->Off;
			hc->PhysMin = hc->DigMin * hc->Cal + hc->Off;
if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (%i): %f %f %f %f %f %f\n",__FILE__,__LINE__,hc->PhysMax, hc->DigMax, hc->Cal, hc->Off,hc->PhysMin, hc->DigMin );
		}

#undef H1LEN
}



/*****************************************************************************

	SOPEN_SMR_READ for reading SON/SMR files


	son.h, sonintl.h are used just for understanding the file format.
	and are not needed for the core functionality.


        TODO:
	Currently waveform data is supported,
        events and marker information is ignored


 *****************************************************************************/

#if defined(WITH_SON)
  #include "sonintl.h"
  // defines __SONINTL__
#else
#define ADCdataBlkSize  32000

#endif


EXTERN_C void sopen_smr_read(HDRTYPE* hdr) {
        /*TODO: implemnt SON/SMR format */
	fprintf(stdout,"SOPEN: Support for CED's SMR/SON format is under construction \n");

	size_t count = hdr->HeadLen;
	if (count < 512) {
			hdr->HeadLen = 512;
			// make sure fixed header (first 512 bytes) are read
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,hdr->HeadLen+1);
			count += ifread(hdr->AS.Header+count, 1, hdr->HeadLen-count, hdr);
			hdr->AS.Header[count]=0;
	}

	// get Endianity, Version and Header size
	hdr->FILE.LittleEndian = (*(uint16_t*)(hdr->AS.Header+38) == 0);   // 0x0000: little endian, 0x0101: big endian
	if (hdr->FILE.LittleEndian) {
		hdr->VERSION = leu16p(hdr->AS.Header);
		hdr->HeadLen = leu32p(hdr->AS.Header + 26 ); // first data
	} else {
		hdr->VERSION = beu16p(hdr->AS.Header);
		hdr->HeadLen = beu32p(hdr->AS.Header + 26); // first data
		// TODO: relax this restriction
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "SON/SMR: big-endian file not supported,yet");
	}

	size_t HeadLen = hdr->HeadLen;
	size_t size_factor = 1;
	if (hdr->VERSION >= 9) {
		size_factor = 512;
		HeadLen *= size_factor;
		if (sizeof(size_t) <= 4) {
			// TODO: relax that restriction
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "SON/SMR: version 9 (bigfile) not supported,yet");
			return;
		}
	}

	if (count < HeadLen) {
		// read channel header and extra data (i.e. everything up to firstData)
		hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,HeadLen+1);
		count += ifread(hdr->AS.Header+count, 1, HeadLen-count, hdr);
		hdr->AS.Header[count]=0;
	}


#ifdef __SONINTL__
	if (VERBOSE_LEVEL > 6) {

		TFileHead *tfh = (void*)(hdr->AS.Header);
		TChannel  *tc  = (void*)(hdr->AS.Header+512);

		// debugging information
		fprintf(stdout,"\tsizeof TFileHead %i\n\tsizeof TChannel %i\n\tsizeof TDataBlock %i\n",
			(int)sizeof(TFileHead),(int)sizeof(TChannel),(int)sizeof(TDataBlock));

		fprintf(stdout,"%i NS=%i %i %i %i\n",
			(int)sizeof(*tfh),hdr->NS,(int)offsetof(TFileHead,channels),(int)sizeof(TChannel),(int)sizeof(TDataBlock));

		fprintf(stdout,"systemID\t%i %i %i\n",(int)offsetof(TFileHead,systemID),
			(int)sizeof(tfh->systemID),lei16p(hdr->AS.Header+offsetof(TFileHead,systemID)));
		fprintf(stdout,"copyright\t%i %i\n",(int)offsetof(TFileHead,copyright),(int)sizeof(tfh->copyright));
		fprintf(stdout,"creator  \t%i %i <%8s>\n",(int)offsetof(TFileHead,creator),
			(int)sizeof(tfh->creator),hdr->AS.Header+offsetof(TFileHead,creator));
		fprintf(stdout,"usPerTime\t%i %i %i\n",(int)offsetof(TFileHead,usPerTime),
			(int)sizeof(tfh->usPerTime),lei16p(hdr->AS.Header+offsetof(TFileHead,usPerTime)));
		fprintf(stdout,"timePerADC\t%i %i %i\n",(int)offsetof(TFileHead,timePerADC),
			(int)sizeof(tfh->timePerADC),lei16p(hdr->AS.Header+offsetof(TFileHead,timePerADC)));
		fprintf(stdout,"fileState\t%i %i %i\n",(int)offsetof(TFileHead,fileState),
			(int)sizeof(tfh->fileState),lei16p(hdr->AS.Header+offsetof(TFileHead,fileState)));
		fprintf(stdout,"firstData\t%i %i %i\n",(int)offsetof(TFileHead,firstData),
			(int)sizeof(tfh->firstData),lei32p(hdr->AS.Header+offsetof(TFileHead,firstData)));
		fprintf(stdout,"channels\t%i %i 0x%04x\n",(int)offsetof(TFileHead,channels),
			(int)sizeof(tfh->channels),lei16p(hdr->AS.Header+offsetof(TFileHead,channels)));
		fprintf(stdout,"chanSize\t%i %i 0x%04x\n",(int)offsetof(TFileHead,chanSize),
			(int)sizeof(tfh->chanSize),lei16p(hdr->AS.Header+offsetof(TFileHead,chanSize)));
		fprintf(stdout,"extraData\t%i %i %i\n",(int)offsetof(TFileHead,extraData),
			(int)sizeof(tfh->extraData),lei16p(hdr->AS.Header+offsetof(TFileHead,extraData)));
		fprintf(stdout,"bufferSz\t%i %i %i\n",(int)offsetof(TFileHead,bufferSz),
			(int)sizeof(tfh->bufferSz),lei16p(hdr->AS.Header+offsetof(TFileHead,bufferSz)));
		fprintf(stdout,"osFormat\t%i %i %i\n",(int)offsetof(TFileHead,osFormat),
			(int)sizeof(tfh->osFormat),lei16p(hdr->AS.Header+offsetof(TFileHead,osFormat)));
		fprintf(stdout,"maxFTime\t%i %i %i\n",(int)offsetof(TFileHead,maxFTime),
			(int)sizeof(tfh->maxFTime),lei32p(hdr->AS.Header+offsetof(TFileHead,maxFTime)));
		fprintf(stdout,"dTimeBase\t%i %i %g\n",(int)offsetof(TFileHead,dTimeBase),
			(int)sizeof(tfh->dTimeBase),lef64p(hdr->AS.Header+offsetof(TFileHead,dTimeBase)));
		fprintf(stdout,"timeDate\t%i %i\n",(int)offsetof(TFileHead,timeDate),(int)sizeof(tfh->timeDate));
		fprintf(stdout,"cAlignFlag\t%i %i\n",(int)offsetof(TFileHead,cAlignFlag),(int)sizeof(tfh->cAlignFlag));
		fprintf(stdout,"pad0     \t%i %i\n",(int)offsetof(TFileHead,pad0),(int)sizeof(tfh->pad0));
		fprintf(stdout,"LUTable  \t%i %i\n",(int)offsetof(TFileHead,LUTable),(int)sizeof(tfh->LUTable));
		fprintf(stdout,"pad      \t%i %i\n",(int)offsetof(TFileHead,pad),(int)sizeof(tfh->pad));
		fprintf(stdout,"fileComment\t%i %i\n",(int)offsetof(TFileHead,fileComment),(int)sizeof(tfh->fileComment));

		fprintf(stdout,"==CHANNEL==\n");
		fprintf(stdout,"delSize  \t%i %i\n",(int)offsetof(TChannel,delSize),(int)sizeof(tc->delSize));
		fprintf(stdout,"nextDelBlock\t%i %i\n",(int)offsetof(TChannel,nextDelBlock),(int)sizeof(tc->nextDelBlock));
		fprintf(stdout,"firstBlock\t%i %i\n",(int)offsetof(TChannel,firstBlock),(int)sizeof(tc->firstBlock));
		fprintf(stdout,"lastBlock\t%i %i\n",(int)offsetof(TChannel,lastBlock),(int)sizeof(tc->lastBlock));
		fprintf(stdout,"blocks   \t%i %i\n",(int)offsetof(TChannel,blocks),(int)sizeof(tc->blocks));
		fprintf(stdout,"nExtra   \t%i %i\n",(int)offsetof(TChannel,nExtra),(int)sizeof(tc->nExtra));
		fprintf(stdout,"preTrig  \t%i %i\n",(int)offsetof(TChannel,preTrig),(int)sizeof(tc->preTrig));
		fprintf(stdout,"blocksMSW\t%i %i\n",(int)offsetof(TChannel,blocksMSW),(int)sizeof(tc->blocksMSW));
		fprintf(stdout,"phySz    \t%i %i\n",(int)offsetof(TChannel,phySz),(int)sizeof(tc->phySz));
		fprintf(stdout,"maxData  \t%i %i\n",(int)offsetof(TChannel,maxData),(int)sizeof(tc->maxData));
		fprintf(stdout,"comment  \t%i %i\n",(int)offsetof(TChannel,comment),(int)sizeof(tc->comment));

		fprintf(stdout,"maxChanTime  \t%i %i\n",(int)offsetof(TChannel,maxChanTime),(int)sizeof(tc->maxChanTime));

		fprintf(stdout,"lChanDvd\t%i %i\n",(int)offsetof(TChannel,lChanDvd),(int)sizeof(tc->lChanDvd));
		fprintf(stdout,"phyChan  \t%i %i\n",(int)offsetof(TChannel,phyChan),(int)sizeof(tc->phyChan));
		fprintf(stdout,"title    \t%i %i %s\n",(int)offsetof(TChannel,title),(int)sizeof(tc->title),((char*)&(tc->title))+1);
		fprintf(stdout,"idealRate\t%i %i\n",(int)offsetof(TChannel,idealRate),(int)sizeof(tc->idealRate));
		fprintf(stdout,"kind     \t%i %i\n",(int)offsetof(TChannel,kind),(int)sizeof(tc->kind));
		fprintf(stdout,"delSizeMSB\t%i %i\n",(int)offsetof(TChannel,delSizeMSB),(int)sizeof(tc->delSizeMSB));

		fprintf(stdout,"v.adc.scale\t%i %i\n",(int)offsetof(TChannel,v.adc.scale),(int)sizeof(tc->v.adc.scale));
		fprintf(stdout,"v.adc.offset\t%i %i\n",(int)offsetof(TChannel,v.adc.offset),(int)sizeof(tc->v.adc.offset));
		fprintf(stdout,"v.adc.units\t%i %i\n",(int)offsetof(TChannel,v.adc.units),(int)sizeof(tc->v.adc.units));
		fprintf(stdout,"v.adc.divide\t%i %i\n",(int)offsetof(TChannel,v.adc.divide),(int)sizeof(tc->v.adc.divide));

		fprintf(stdout,"v.event.initLow\t%i %i\n",(int)offsetof(TChannel,v.event.initLow),(int)sizeof(tc->v.event.initLow));
		fprintf(stdout,"v.event.nextLow\t%i %i\n",(int)offsetof(TChannel,v.event.nextLow),(int)sizeof(tc->v.event.nextLow));

		fprintf(stdout,"v.real.min\t%i %i\n",(int)offsetof(TChannel,v.real.min),(int)sizeof(tc->v.real.min));
		fprintf(stdout,"v.real.max\t%i %i\n",(int)offsetof(TChannel,v.real.max),(int)sizeof(tc->v.real.max));
		fprintf(stdout,"v.real.units\t%i %i\n",(int)offsetof(TChannel,v.real.units),(int)sizeof(tc->v.real.units));

		assert(sizeof(TSONTimeDate)==8);
		assert(sizeof(TFileHead)==512);
		assert(sizeof(TChannel)==140);
		assert(sizeof(TDataBlock)==64020);
	}
#endif

	uint16_t timePerADC;
	uint32_t maxFTime;
	double timebase,dTimeBase;
	memcpy(&hdr->ID.Equipment, hdr->AS.Header+12, 8);
	if (hdr->FILE.LittleEndian) {
		timebase = hdr->VERSION < 6 ? 1e-6 : lef64p(hdr->AS.Header + 44);
		hdr->SampleRate = 1.0 / (timebase * leu16p(hdr->AS.Header + 20) );
		hdr->NS         = leu16p(hdr->AS.Header + 30 );
		timePerADC      = lei16p(hdr->AS.Header + 22);
		maxFTime 	= leu32p(hdr->AS.Header + 40);
		dTimeBase	= lef64p(hdr->AS.Header + 44);
	} else {
		timebase = hdr->VERSION < 6 ? 1e-6 : bef64p(hdr->AS.Header + 44);
		hdr->SampleRate = 1.0 / (timebase * beu16p(hdr->AS.Header + 20));
		hdr->NS         = beu16p(hdr->AS.Header + 30);
		timePerADC      = bei16p(hdr->AS.Header + 22);
		maxFTime 	= beu32p(hdr->AS.Header + 40);
		dTimeBase	= bef64p(hdr->AS.Header + 44);
	}
	hdr->SPR = 1;

	while (!ifeof(hdr)) {
		// read channel header and extra data
		hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,hdr->HeadLen*2);
		hdr->HeadLen  *= 2;
		count += ifread(hdr->AS.Header+count, 1, hdr->HeadLen-count, hdr);
	}
	hdr->HeadLen = count;

	/*********************************************
	  read channel header
 	 *********************************************/
	hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));

	typeof(hdr->NS) k;
	size_t bpb = 0;
	for (k = 0; k < hdr->NS; k++) {
		uint32_t off = 512 + k*140;
		CHANNEL_TYPE *hc    = hdr->CHANNEL+k;

		hc->Cal   = lef32p(hdr->AS.Header + off + 124);	// v.adc.scale
		hc->Off   = lef32p(hdr->AS.Header + off + 128);	// v.adc.off
		hc->OnOff = 0;   //*(int16_t*)(hdr->AS.Header + off + 106) != 0;
		hc->SPR   = 0;
		hc->GDFTYP = 3;
		hc->LeadIdCode = 0;

		int stringLength = hdr->AS.Header[off+108];
		assert(stringLength < MAX_LENGTH_LABEL);
		memcpy(hc->Label, hdr->AS.Header+off+108+1, stringLength);
		hc->Label[stringLength] = '\0';

		{
		// extract Physical units (pascal string - first byte indicates the length, no 0-terminator)
		stringLength = hdr->AS.Header[off+132];
		char PhysicalUnit[6];
		assert( stringLength < sizeof(PhysicalUnit) );
		memcpy(PhysicalUnit, hdr->AS.Header+off+132+1, stringLength);
		PhysicalUnit[stringLength] = '\0';
		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): #%i <%s> <%s>\n",__FILE__,__LINE__, k+1, hc->Label, PhysicalUnit);

		if (!strcmp(PhysicalUnit,"Volt") || !strcmp(PhysicalUnit," Volt"))
			hc->PhysDimCode = 4256;
		else if (!strcmp(PhysicalUnit,"mVolt"))
			hc->PhysDimCode = 4274;
		else if (!strcmp(PhysicalUnit,"uVolt"))
			hc->PhysDimCode = 4275;
		else
			hc->PhysDimCode = PhysDimCode(PhysicalUnit);
		}

		uint32_t firstBlock = leu32p(hdr->AS.Header+off+6);
		uint32_t lastBlock  = leu32p(hdr->AS.Header+off+10);

		hc->bi = bpb;
		hc->bi8= 0;
		uint8_t kind = hdr->AS.Header[off+122];
		switch (kind) {
		case 0: //
			//    ChanOff=0,          /* the channel is OFF - */
			hc->OnOff = 0;
			break;
		case 1: {
			//    Adc,                /* a 16-bit waveform channel */
			hc->OnOff   = 1;
			hc->GDFTYP  = 3;
			hc->Cal     = lef32p(hdr->AS.Header + off + 124) / 6553.6;
			hc->Off     = lef32p(hdr->AS.Header + off + 128);
			hc->DigMax  = (double)(int16_t)0x7fff;
			hc->DigMin  = (double)(int16_t)0x8000;
			hc->PhysMax = hc->DigMax * hc->Cal + hc->Off;
			hc->PhysMin = hc->DigMin * hc->Cal + hc->Off;

			uint32_t blocks = leu16p(hdr->AS.Header+off+14);
			size_t bpbnew   = bpb + blocks * ADCdataBlkSize * 2;
			hdr->AS.rawdata = realloc(hdr->AS.rawdata,bpbnew);

			uint32_t pos = firstBlock;
			while (1) {
				uint32_t items = leu16p(hdr->AS.Header+pos+18);
				memcpy(hdr->AS.rawdata+bpb, hdr->AS.Header+pos+20, items*2);
				bpb     += items*2;
				hc->SPR += items;
			if (pos == lastBlock) break;
				pos  = leu32p(hdr->AS.Header+pos+4);
			};

			hdr->SPR    = lcm(hdr->SPR, hc->SPR);
			break;
		}
		case 2:
			//    EventFall,          /* Event times (falling edges) */
			hc->OnOff = 0;
			break;
		case 3:
			//    EventRise,          /* Event times (rising edges) */
			hc->OnOff = 0;
			break;
		case 4:
			//    EventBoth,          /* Event times (both edges) */
			hc->OnOff = 0;
			break;
		case 5:
			//    Marker,             /* Event time plus 4 8-bit codes */
			hc->OnOff = 0;
			break;
		case 6:
			//    AdcMark,            /* Marker plus Adc waveform data */
			hc->OnOff = 0;
			break;
		case 7:
			//    RealMark,           /* Marker plus float numbers */
			hc->OnOff = 0;
			break;
		case 8:
			//    TextMark,           /* Marker plus text string */
			hc->OnOff = 0;
			break;
		case 9: {
			//    RealWave            /* waveform of float numbers */
			hc->OnOff   = 1;
			hc->GDFTYP  = 16;
			hc->LeadIdCode = 0;
			hc->Cal     = lef32p(hdr->AS.Header + off + 124) / 6553.6;
			hc->Off     = lef32p(hdr->AS.Header + off + 128);
			hc->DigMax  = +1e9;
			hc->DigMin  = -1e9;
			hc->PhysMax = hc->DigMax * hc->Cal + hc->Off;
			hc->PhysMin = hc->DigMin * hc->Cal + hc->Off;

			uint32_t blocks = leu16p(hdr->AS.Header+off+14);
			size_t bpbnew   = bpb + blocks * ADCdataBlkSize * 4;
			hdr->AS.rawdata = realloc(hdr->AS.rawdata, bpbnew);

			uint32_t pos = firstBlock;
			while (1) {
				uint32_t items = leu16p(hdr->AS.Header+pos+18);
				memcpy(hdr->AS.rawdata+bpb, hdr->AS.Header+pos+20, items*4);
				bpb     += items*4;
				hc->SPR += items;
			if (pos == lastBlock) break;
				pos  = leu32p(hdr->AS.Header+pos+4);
			}

			hdr->SPR    = lcm(hdr->SPR, hc->SPR);
			break;
		}
		default:
			//    unknown
			hc->OnOff = 0;
			fprintf(stderr,"SMR/SON: channel %i ignored - unknown type %i\n",k,kind);
		}

		if (VERBOSE_LEVEL > 6) {
			char tmp[98-26+1];
			fprintf(stdout,"[%i].delSize\t%i\n",k,lei16p(hdr->AS.Header+off));

			fprintf(stdout,"[%i].nextDelBlock\t%i\n",k,leu32p(hdr->AS.Header+off+2));
			fprintf(stdout,"[%i].firstBlock\t%i\n",k,leu32p(hdr->AS.Header+off+6));
			fprintf(stdout,"[%i].lastBlock\t%i\n",k,leu32p(hdr->AS.Header+off+10));

			fprintf(stdout,"[%i].blocks\t%i\n",k,leu16p(hdr->AS.Header+off+14));
			fprintf(stdout,"[%i].nExtra\t%i\n",k,leu16p(hdr->AS.Header+off+16));
			fprintf(stdout,"[%i].preTrig\t%i\n",k,lei16p(hdr->AS.Header+off+18));
			fprintf(stdout,"[%i].blocksMSW\t%i\n",k,lei16p(hdr->AS.Header+off+20));
			fprintf(stdout,"[%i].phySz\t%i\n",k,lei16p(hdr->AS.Header+off+22));
			fprintf(stdout,"[%i].maxData\t%i\n",k,lei16p(hdr->AS.Header+off+24));

			stringLength = hdr->AS.Header[off+26];
			assert(stringLength < sizeof(tmp));
			memcpy(tmp, hdr->AS.Header+off+26+1, 9);
			tmp[stringLength] = 0;
			fprintf(stdout,"[%i].comment\t<%s>\n",(int)k,tmp);

			fprintf(stdout,"[%i].maxChanTime\t%i\t%i\n",(int)k,lei32p(hdr->AS.Header+off+98),*(int32_t*)(hdr->AS.Header+off+98));
			fprintf(stdout,"[%i].lChanDvd\t%i\n",(int)k,lei32p(hdr->AS.Header+off+102));
			fprintf(stdout,"[%i].phyChan\t%i\n",(int)k,lei16p(hdr->AS.Header+off+106));

			stringLength = hdr->AS.Header[off+108];
			assert(stringLength < sizeof(tmp));
			memcpy(tmp, hdr->AS.Header+off+108+1,9);
			tmp[stringLength] = 0;
			fprintf(stdout,"[%i].title\t<%s>\n",(int)k,tmp);

			fprintf(stdout,"[%i].idealRate\t%f\n",(int)k,lef32p(hdr->AS.Header+off+118));
			fprintf(stdout,"[%i].kind\t%i\n",(int)k,*(hdr->AS.Header+off+122));
			fprintf(stdout,"[%i].delSizeMSB\t%i\n",(int)k,*(hdr->AS.Header+off+123));

			fprintf(stdout,"[%i].v.adc.scale\t%f\n",(int)k,lef32p(hdr->AS.Header+off+124));
			fprintf(stdout,"[%i].v.adc.offset\t%f\n",(int)k,lef32p(hdr->AS.Header+off+128));

			stringLength = hdr->AS.Header[off+132];
			assert(stringLength < sizeof(tmp));
			memcpy(tmp, hdr->AS.Header+off+132+1, 5);
			tmp[stringLength] = 0;
			fprintf(stdout,"[%i].v.adc.units\t%s\n", (int)k, tmp);

			fprintf(stdout,"[%i].v.adc.divide\t%i\n", (int)k, lei16p(hdr->AS.Header+off+138));

			fprintf(stdout,"[%i].v.real.max\t%f\n", (int)k, lef32p(hdr->AS.Header+off+124));
			fprintf(stdout,"[%i].v.real.min\t%f\n", (int)k, lef32p(hdr->AS.Header+off+128));
			fprintf(stdout,"[%i].v.real.units\t%s\n", (int)k, hdr->AS.Header+off+132+1);

			fprintf(stdout,"[%i].v.event\t%0x\t%g\n", (int)k, leu32p(hdr->AS.Header+off+124), lef32p(hdr->AS.Header+off+124));
		}
        }

	hdr->NRec       = 1;
	hdr->AS.bpb	= bpb;
}


