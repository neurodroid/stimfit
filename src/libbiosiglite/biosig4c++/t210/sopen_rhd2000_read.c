/*

    Copyright (C) 2019 Alois Schloegl <alois.schloegl@gmail.com>

    This file is part of the "BioSig for C/C++" repository
    (biosig4c++) at http://biosig.sf.net/

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.

 */

/*

Referemces:
[1]  RHD2000 Application Note: Data File Formats, Intan TECHNOLOGIES, LLC
     Downloaded 2019-08-30 from
     https://www.intantech.com/files/Intan_RHD2000_data_file_formats.pdf

*/

#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <iconv.h>
#include <errno.h>

#if !defined(__APPLE__) && defined (_LIBICONV_H)
 #define iconv		libiconv
 #define iconv_open	libiconv_open
 #define iconv_close	libiconv_close
#endif

#include "../gdftime.h"
#include "../biosig-dev.h"

#define min(a,b)        (((a) < (b)) ? (a) : (b))
#define max(a,b)        (((a) > (b)) ? (a) : (b))

typedef struct {
	int32_t length;	// this is little endian as defined in spec, use le32toh() when using it
	void *string;
} QSTRING_T;

#ifdef __cplusplus
extern "C" {
#endif

/* check whether sufficient data of the header information has been read,
   reads more data if needed, advance position pointer pos to the end of the string;
*/
QSTRING_T *read_qstring(HDRTYPE* hdr, size_t *pos) {
	int32_t len0 = lei32p(hdr->AS.Header + (*pos));
	*pos += 4;
	int32_t len = max(0,len0);

	// after each qstring at most 28 bytes are loaded before the next check for a qstring
	// This check is also needed when qstring is empty
	size_t SIZE = *pos+len+100;
	if (SIZE > hdr->HeadLen) {
		SIZE = max(SIZE, 2*hdr->HeadLen);	// always double size of header
		void *ptr = realloc(hdr->AS.Header, SIZE);
		if (ptr==NULL) {
			biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Format Intan RHD2000 - memory allocation failed");
			return NULL;
		}
		hdr->AS.Header = (uint8_t*)ptr;
		hdr->HeadLen += ifread(hdr->AS.Header+hdr->HeadLen, 1, SIZE-hdr->HeadLen, hdr);
	}

	if (len0 < 0) return NULL;

	if ((*pos + len) > hdr->HeadLen)
		biosigERROR(hdr, B4C_INCOMPLETE_FILE, "Format Intan RHD2000 - incomplete file");

	QSTRING_T *qString=(QSTRING_T *)(hdr->AS.Header+(*pos));
	*pos += len;
	return qString;
}


int sopen_intan_clp_read(HDRTYPE* hdr) {

	uint16_t NumADCs=0, NumChips=0, NumChan=0;

	float minor = leu16p(hdr->AS.Header+6);
	minor      *= (minor < 10) ? 0.1 : 0.01;
	hdr->VERSION = leu16p(hdr->AS.Header+4) + minor;

	uint16_t datatype=leu16p(hdr->AS.Header+8);
	switch (datatype) {
	case 1: NumADCs=leu16p(hdr->AS.Header+10);
		hdr->SampleRate = lef32p(hdr->AS.Header+24);
	case 0: break;
	default:
		// this should never ever occurs, because getfiletype checks this
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format Intan CLP - datatype unknown");
		return -1;
	}

	size_t HeadLen=leu16p(hdr->AS.Header+10+(datatype*2));
	// read header
	if (HeadLen > hdr->HeadLen) {
		hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, HeadLen+1);
		hdr->HeadLen  += ifread(hdr->AS.Header+HeadLen, 1, HeadLen - hdr->HeadLen, hdr);
	}
	hdr->AS.Header[hdr->HeadLen]=0;
	if (HeadLen > hdr->HeadLen) {
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format Intan/CLP - file is too short");
		return -1;
	}
	ifseek(hdr, HeadLen, SEEK_SET);

	// read recording date and time
	size_t pos=12+(datatype*2);
	{
		struct tm t0;
		t0.tm_year = leu16p(hdr->AS.Header+pos);
		t0.tm_mon = leu16p(hdr->AS.Header+pos+2);
		t0.tm_mday = leu16p(hdr->AS.Header+pos+4);
		t0.tm_hour = leu16p(hdr->AS.Header+pos+6);
		t0.tm_min = leu16p(hdr->AS.Header+pos+8);
		t0.tm_sec = leu16p(hdr->AS.Header+pos+10);
		hdr->T0 = tm_time2gdf_time(&t0);
	}

	switch (datatype) {
	case 0:
		// If this is the standard data file (including clamp and measured data),
		HeadLen=leu16p(hdr->AS.Header+10);
		/* TODO:
			read chips
			read settings
		*/
		break;
	case 1:
		// If this is the aux data file (including Digital Ins/Outs and ADC data)
		NumADCs=leu16p(hdr->AS.Header+10);
		HeadLen=leu16p(hdr->AS.Header+12);

		hdr->SampleRate = lef32p(hdr->AS.Header+24);
	default:
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format Intan CLP - datatype unknown");
		return -1;
	}

	if (datatype==0) {
		// read chips
		pos = 24;
		NumChips=leu16p(hdr->AS.Header+pos);
		NumChan=leu16p(hdr->AS.Header+pos+2);
		for (uint16_t k1 = 0; k1 < NumChips; k1++) {
		for (uint16_t k2 = 0; k2 < NumChan; k2++) {
			// read one header per chip
			//14*2+4+4+16+20+4
		}
			// read
		}
		pos += ((14*2+4+4+16+20+4) * NumChan + 8) * NumChips;
		// read settings

		hdr->NS    = 4;
		hdr->SPR   = 1;
		hdr->NRec  = -1;
		hdr->AS.bpb= 16;

		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		strcpy(hdr->CHANNEL[0].Label,"Time");
		strcpy(hdr->CHANNEL[1].Label,"Clamp");
		strcpy(hdr->CHANNEL[2].Label,"TotalClamp");
		strcpy(hdr->CHANNEL[3].Label,"Measured");

		for (int k=0; k<4; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			hc->Transducer[0]='\0';
			hc->OnOff=1;
			hc->GDFTYP=16;
			hc->DigMax=+1e9;
			hc->DigMin=-1e9;
			hc->Cal=1;
			hc->Off=0;
		}
		hdr->CHANNEL[0].GDFTYP=6;	//uint32
		hdr->CHANNEL[0].DigMax=ldexp(1l,32)-1;	//uint32
		hdr->CHANNEL[0].DigMin=0.0;	//uint32
		hdr->CHANNEL[0].Cal=1.0/hdr->SampleRate;	//uint32
		for (int k=0; k<4; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			hc->PhysMax = hc->DigMax * hc->Cal + hc->Off;
			hc->PhysMin = hc->DigMin * hc->Cal + hc->Off;
		}
	}

	biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format Intan/CLP not supported");
	return -1;
}

int sopen_rhs2000_read(HDRTYPE* hdr) {
	biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format Intan/RHS2000 not supported");
	return -1;
}

int sopen_rhd2000_read(HDRTYPE* hdr) {
/*
	this function is a stub or placeholder and need to be defined in order to be useful.
	It will be called by the function SOPEN in "biosig.c"

	Input:
		char* Header	// contains the file content

	Output:
		HDRTYPE *hdr	// defines the HDR structure accoring to "biosig.h"
*/

		float minor = leu16p(hdr->AS.Header+6);
		minor      *= (minor < 10) ? 0.1 : 0.01;
		hdr->VERSION = leu16p(hdr->AS.Header+4) + minor;

		hdr->NS = 1;
		hdr->SampleRate = lef32p(hdr->AS.Header+8);

		float HighPass = ( leu16p(hdr->AS.Header+12) ? 0.0 : lef32p(hdr->AS.Header+14) );
		      HighPass = max( HighPass, lef32p(hdr->AS.Header+18) );
		float LowPass = lef32p(hdr->AS.Header+22);
		const int ListNotch[] = {0,50,60};
		uint16_t tmp = leu16p(hdr->AS.Header+34);
		if (tmp>2) tmp=0;
		float Notch = ListNotch[tmp];
		float fZ_desired   = lef32p(hdr->AS.Header+40); // desired impedance test frequency
		float fZ_actual    = lef32p(hdr->AS.Header+44); // actual impedance test frequency

		size_t pos = 48;
		QSTRING_T *note1 = read_qstring(hdr, &pos);
		QSTRING_T *note2 = read_qstring(hdr, &pos);
		QSTRING_T *note3 = read_qstring(hdr, &pos);

		uint16_t numberTemperatureSensors = leu16p(hdr->AS.Header+pos);
		pos += 2;

		int boardMode = leu16p(hdr->AS.Header+pos);
		pos += 2;
		float PhysMin=0.;
		float PhysMax=1.;
		float Cal=1, DigOff=0;
		switch (boardMode) {
		case 0: PhysMax=3.3; Cal=0.000050354; break;
		case 1: PhysMin=-5.0; PhysMax=5.0; Cal = 0.00015259; break;
		case 13: PhysMin=-10.24; PhysMax=10.24; Cal = 0.0003125; break;
		default:
			fprintf(stderr,"%s (line %d): Intan/RHD2000 - unknown Boardmode %d\n", __func__,__LINE__,boardMode);
			// boardMode unknown
		};

		QSTRING_T *referenceChannelName = read_qstring(hdr, &pos);

		uint16_t numberOfSignalGroups = leu16p(hdr->AS.Header+pos);
		pos += 2;

		hdr->NS = 1;
		uint32_t chan = 1 ;
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		CHANNEL_TYPE *hc = hdr->CHANNEL;
		strcpy(hc->Label,"Time");
		hc->Transducer[0] = 0;
		hc->OnOff  = 2;
		hc->GDFTYP = 5; //int32_t
		hc->DigMin = -ldexp(1,31);
		hc->DigMax = ldexp(1,31)-1;
		hdr->SPR   = (hdr->Version < 2.0) ? 60 : 128;

		for (uint16_t k = 0; k<numberOfSignalGroups; k++) {
			QSTRING_T *groupName = read_qstring(hdr, &pos);
			QSTRING_T *groupNamePrefix = read_qstring(hdr, &pos);

			uint16_t enabled = leu16p(hdr->AS.Header+pos);
			pos += 2;
			uint16_t numberChannelsInGroup = leu16p(hdr->AS.Header+pos);
			pos += 2;
			uint16_t numberAmplifierChannels = leu16p(hdr->AS.Header+pos);
			pos += 2;

			if (enabled && (numberChannelsInGroup > 0)) {
				hdr->NS += numberChannelsInGroup;
				hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));

				for (; chan < hdr->NS; chan++); {
					CHANNEL_TYPE *hc = hdr->CHANNEL + chan;

					QSTRING_T *nativeChannelName = read_qstring(hdr, &pos);
					if (nativeChannelName != NULL) {
						// translate into biosig HDR channel structure
						iconv_t CD = iconv_open ("UTF-16LE","UTF-8");
						size_t inlen = le32toh(nativeChannelName->length);
						size_t outlen = MAX_LENGTH_LABEL+1;
						char *outbuf = hc->Label;
						size_t iconv_res = iconv (CD, nativeChannelName->string, &inlen, &outbuf, &outlen);
						iconv_close (CD);
					}

					QSTRING_T *customChannelName = read_qstring(hdr, &pos);
					if (customChannelName != NULL) {
						// translate into biosig HDR channel structure
						iconv_t CD = iconv_open ("UTF-16LE","UTF-8");
						size_t inlen = le32toh(customChannelName->length);
						size_t outlen = MAX_LENGTH_LABEL+1;
						char *outbuf = hc->Label;
						size_t iconv_res = iconv (CD, customChannelName->string, &inlen, &outbuf, &outlen);
						iconv_close (CD);
					}

					uint16_t customOrder = leu16p(hdr->AS.Header+pos);
					pos += 2;
					uint16_t nativeOrder = leu16p(hdr->AS.Header+pos);
					pos += 2;
					uint16_t signalType = leu16p(hdr->AS.Header+pos);
					pos += 2;
					uint16_t channelEnabled = leu16p(hdr->AS.Header+pos);
					hc->OnOff = channelEnabled;
					pos += 2;
					uint16_t chipChannel = leu16p(hdr->AS.Header+pos);
					pos += 2;
					uint16_t boardStream = leu16p(hdr->AS.Header+pos);
					pos += 2;

					uint16_t triggerMode = leu16p(hdr->AS.Header+pos);
					pos += 2;
					int16_t voltageThreshold = leu16p(hdr->AS.Header+pos); // uV
					pos += 2;
					uint16_t triggerChannel = leu16p(hdr->AS.Header+pos);
					pos += 2;
					float ImpedanceMagnitude = lef32p(hdr->AS.Header+pos); // Ohm
					pos += 4;
					float ImpedancePhase = lef32p(hdr->AS.Header+pos); 	// degree
					pos += 4;

					// translate into biosig HDR channel structure
					hc->GDFTYP = 4; 	// uint16_t
					hc->DigMin = 0;
					hc->DigMax = 0xffff;
					hc->SPR    = (signalType<3) ? 60 : 128;
					switch (signalType) {
					case 0: hc->SPR = hdr->SPR;
						hc->Cal = 0.195;
						hc->PhysMin = (hc->DigMin-32768) * hc->Cal;
						hc->PhysMax = (hc->DigMin-32768) * hc->Cal;
						break;
					case 1: hc->SPR = hdr->SPR/4;
						hc->Cal = 0.0000374;
						hc->PhysMin = hc->DigMin * hc->Cal;
						hc->PhysMax = hc->DigMin * hc->Cal;
						break;
					case 2: hc->SPR = 1;
						hc->Cal = 0.0000748;
						hc->PhysMin = hc->DigMin * hc->Cal;
						hc->PhysMax = hc->DigMin * hc->Cal;
						break;
					case 3: // depends on board mode, range is computed above
						hc->SPR = hdr->SPR;
						hc->PhysMin = PhysMin;
						hc->PhysMax = PhysMax;
						hc->Cal = 0.0000748;
						break;
					case 4: hc->SPR = hdr->SPR;
						hc->PhysMin = hc->DigMin;
						hc->PhysMax = hc->DigMax;
						break;
					case 5: hc->SPR = hdr->SPR;
						hc->PhysMin = hc->DigMin;
						hc->PhysMax = hc->DigMax;
						break;
					default:
						;
					}
					hc->Off = hc->PhysMin - hc->DigMin * hc->Cal;
					hc->Transducer[0]=0;

				if (VERBOSE_LEVEL >7)
					fprintf(stdout, "%s (line %d): Intan/RHD2000:  #%d %d %s\n",__FILE__,__LINE__, chan, hc->OnOff, hc->Label);

					if (!(chipChannel<32) || !(signalType<6)) {
						biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format Intan RHD2000 - not conformant to specification");
						return -1;
					}
				}
			}
		}
		hdr->HeadLen = pos;

	biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format Intan RHD2000 not supported");
	return -1;
}


#ifdef __cplusplus
}
#endif

