/*
    Copyright (C) 2019,2020 Alois Schloegl <alois.schloegl@gmail.com>

    This file is part of the "BioSig for C/C++" repository
    (biosig4c++) at http://biosig.sf.net/

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.
 */

/*
References:
[1]  RHD2000 Application Note: Data File Formats, Intan TECHNOLOGIES, LLC
     Downloaded 2020-01-14 from
     http://www.intantech.com/files/Intan_RHD2000_data_file_formats.pdf
[2]  RHS2000 Data File Formats - Intan Tech
     http://www.intantech.com/files/Intan_RHS2000_data_file_formats.pdf
*/

#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <iconv.h>
#include <errno.h>
#include <sys/stat.h>

#if !defined(__APPLE__) && defined (_LIBICONV_H)
 #define iconv		libiconv
 #define iconv_open	libiconv_open
 #define iconv_close	libiconv_close
#endif

#include "../gdftime.h"
#include "../biosig-dev.h"

#define min(a,b)        (((a) < (b)) ? (a) : (b))
#define max(a,b)        (((a) > (b)) ? (a) : (b))

#ifdef __cplusplus
extern "C" {
#endif

/*
   get_qstring has the following functions and side effects:
   *) check whether sufficient data of the header information has been read,
      reads more data if needed,
   *) advance position pointer pos to the end of the string;
   *) convert QString into UTF-8 String in outbuf
*/

void read_qstring(HDRTYPE* hdr, size_t *pos, char *outbuf, size_t outlen) {
	uint32_t len = leu32p(hdr->AS.Header + (*pos));
	*pos += 4;
	if (len==(uint32_t)(-1)) return;

	// after each qstring at most 28 bytes are loaded before the next check for a qstring
	// This check is also needed when qstring is empty
	size_t SIZE = *pos+len+100;
	if (SIZE > hdr->HeadLen) {
		SIZE = max(SIZE, 2*hdr->HeadLen);	// always double size of header
		void *ptr = realloc(hdr->AS.Header, SIZE);
		if (ptr==NULL) {
			biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Format Intan RH[DS]2000 - memory allocation failed");
			return;
		}
		hdr->AS.Header = (uint8_t*)ptr;
		hdr->HeadLen += ifread(hdr->AS.Header+hdr->HeadLen, 1, SIZE-hdr->HeadLen, hdr);
	}

	if ((*pos + len ) > hdr->HeadLen)
		biosigERROR(hdr, B4C_INCOMPLETE_FILE, "Format Intan RHD2000 - incomplete file");

	// convert qString into UTF-8 string
	if (outbuf != NULL) {
		iconv_t CD   = iconv_open ("UTF-8", "UTF-16LE");
		size_t inlen = len;
		char *inbuf  = hdr->AS.Header+(*pos);
		size_t iconv_res = iconv (CD, &inbuf, &inlen, &outbuf, &outlen);
		*outbuf = '\0';
		iconv_close (CD);
	}
	*pos += len;
	return;
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
		/* TODO:
			read chips
			read settings
		*/
		break;
	case 1:
		// If this is the aux data file (including Digital Ins/Outs and ADC data)
	default:
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format Intan CLP - datatype unknown");
		return -1;
	}

	switch (datatype) {
	case 0: {
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
		// see below // strcpy(hdr->CHANNEL[0].Label,"Time");
		strcpy(hdr->CHANNEL[1].Label,"Clamp");
		strcpy(hdr->CHANNEL[2].Label,"TotalClamp");
		strcpy(hdr->CHANNEL[3].Label,"Measured");

		for (int k=0; k < hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			hc->Transducer[0]='\0';
			hc->OnOff=1;
			hc->GDFTYP=16;
			hc->DigMax=+1e9;
			hc->DigMin=-1e9;
			hc->Cal=1;
			hc->Off=0;
		}
		break;
	    }
	case 1: {
		hdr->NS    = NumADCs;
		hdr->SPR   = 1;
		hdr->NRec  = -1;
		hdr->AS.bpb= 4 + 2 + 2 + 2 * NumADCs;

		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		for (int k=0; k < hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			sprintf(hdr->CHANNEL[3].Label, "ADC%d", k-2);
			hc->Transducer[0]='\0';
			hc->OnOff=1;
			hc->GDFTYP=4;
			hc->DigMax=65535.0;
			hc->DigMin=0;
			hc->Cal = (k < 3) ? 1.0 : 0.0003125;
			hc->Off = (k < 3) ? 0.0 : -10.24;
		}
		// see below // strcpy(hdr->CHANNEL[0].Label, "Time");
		strcpy(hdr->CHANNEL[1].Label,"DigitalIn");
		strcpy(hdr->CHANNEL[2].Label,"DigitalOut");
		break;
	    }
	default:
		;
	}

	hdr->CHANNEL[0].OnOff=2;	//uint32
	hdr->CHANNEL[0].GDFTYP=6;	//uint32
	hdr->CHANNEL[0].DigMax=ldexp(1l,32)-1;	//uint32
	hdr->CHANNEL[0].DigMin=0.0;	//uint32
	hdr->CHANNEL[0].Cal=1.0/hdr->SampleRate;	//uint32
	hdr->CHANNEL[0].PhysDimCode = 2176;	//uint32
	strcpy(hdr->CHANNEL[0].Label, "Time");

	hdr->AS.bpb = 0;
	for (int k = 0; k < hdr->NS; k++) {
		CHANNEL_TYPE *hc = hdr->CHANNEL+k;
		hc->PhysMax   = hc->DigMax * hc->Cal + hc->Off;
		hc->PhysMin   = hc->DigMin * hc->Cal + hc->Off;
		hc->LeadIdCode = 0;
		hc->PhysDimCode = 0;
		hc->TOffset   = 0;
		hc->LowPass   = NAN;
		hc->HighPass  = NAN;
		hc->Notch     = NAN;
		hc->Impedance = NAN;
		hc->fZ        = NAN;
		hc->SPR       = 1;
		memset(hc->XYZ, 0, 12);
		hc->bi        = hdr->AS.bpb;
		hdr->AS.bpb  += (GDFTYP_BITS[hc->GDFTYP]*(size_t)hc->SPR) >> 3;
	}

	biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format Intan/CLP not supported");
	return -1;
}

/*
     RHS2000 Data File Formats - Intan Tech
     http://www.intantech.com/files/Intan_RHS2000_data_file_formats.pdf
*/
int sopen_rhs2000_read(HDRTYPE* hdr) {

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %u); %s(...) [%u]\n",__FILE__,__LINE__,__func__,hdr->HeadLen);

		// 8 bytes
		float minor = leu16p(hdr->AS.Header+6);
		minor      *= (minor < 10) ? 0.1 : 0.01;
		hdr->VERSION = leu16p(hdr->AS.Header+4) + minor;

		// +38 = 46 bytes
		hdr->NS = 1;
		hdr->SampleRate = lef32p(hdr->AS.Header+8);

		float HighPass = ( leu16p(hdr->AS.Header+12) ? lef32p(hdr->AS.Header+14) : 0.0 );
		      HighPass = max( HighPass, lef32p(hdr->AS.Header+18) );
		float LowPass = lef32p(hdr->AS.Header+26);

		// +10 = 56 bytes
		const int ListNotch[] = {0,50,60};
		uint16_t tmp = leu16p(hdr->AS.Header+46);
		if (tmp>2) tmp=0;
		float Notch = ListNotch[tmp];
		float fZ_desired   = lef32p(hdr->AS.Header+46); // desired impedance test frequency
		float fZ_actual    = lef32p(hdr->AS.Header+50); // actual impedance test frequency

		// +4 = 60 bytes
		// +12 = 72 bytes
		float StimStepSize = lef32p(hdr->AS.Header+58); // Stim Step Size  [A]
		float ChargeRecoveryCurrentLimit = lef32p(hdr->AS.Header+62);	// [A]
		float ChargeRecoveryTargetVoltage = lef32p(hdr->AS.Header+66); // [V]

		size_t pos = 72;
		read_qstring(hdr, &pos, NULL, 0);	// note1
		read_qstring(hdr, &pos, NULL, 0);	// note2
		read_qstring(hdr, &pos, NULL, 0);	// note3

		uint16_t flag_DC_amplifier_data_saved = leu16p(hdr->AS.Header+pos) > 0;
		pos += 2;
		uint16_t BoardMode = leu16p(hdr->AS.Header+pos);
		pos += 2;

		char *ReferenceChannelName = NULL;
		read_qstring(hdr, &pos, ReferenceChannelName, 0);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %u); %s(...)  %u %u\n", __FILE__, __LINE__, __func__, (unsigned)pos, hdr->HeadLen);

		uint16_t numberOfSignalGroups = leu16p(hdr->AS.Header+pos);
		pos += 2;
		uint16_t NS = 0;

		hdr->SPR = 128;
		size_t bi = (0+4)*hdr->SPR;

		// read all signal groups
		for (int nsg=0; nsg < numberOfSignalGroups; nsg++) {
			char SignalGroupName[101], SignalGroupPrefix[101];
			read_qstring(hdr, &pos, SignalGroupName, 100);
			read_qstring(hdr, &pos, SignalGroupPrefix, 100);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %u); %s(...) group=%u %u SGP<%s> SGP<%s>\n",__FILE__,__LINE__,__func__, nsg, (unsigned)pos, SignalGroupName,SignalGroupPrefix );

			uint16_t flag_SignalGroupEnabled = leu16p(hdr->AS.Header+pos);
			pos += 2;
			uint16_t NumberOfChannelsInSignalGroup = leu16p(hdr->AS.Header+pos);
			pos += 2;
			uint16_t NumberOfAmplifierChannelsInSignalGroup = leu16p(hdr->AS.Header+pos);
			pos += 2;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %u); %s(...) group=%u %u+%u %d\n", __FILE__, __LINE__, __func__, nsg, NS, NumberOfChannelsInSignalGroup, (int)pos);

			if (flag_SignalGroupEnabled) {
				hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, (1+NS+NumberOfChannelsInSignalGroup*(2+flag_DC_amplifier_data_saved)) * sizeof(CHANNEL_TYPE));
				for (unsigned k = 0; k < NumberOfChannelsInSignalGroup; k++) {
					char NativeChannelName[MAX_LENGTH_LABEL+1];
					char CustomChannelName[MAX_LENGTH_LABEL+1];
					read_qstring(hdr, &pos, NativeChannelName, MAX_LENGTH_LABEL);
					read_qstring(hdr, &pos, CustomChannelName, MAX_LENGTH_LABEL);
					pos += 4;
					int16_t SignalType = lei16p(hdr->AS.Header+pos);
					pos += 2;
					int16_t ChannelEnabled = lei16p(hdr->AS.Header+pos);
					pos += 2;
					int16_t ChipChannel = lei16p(hdr->AS.Header+pos);
					pos += 2;
					int16_t CommandStream = lei16p(hdr->AS.Header+pos);
					pos += 2;
					int16_t BoardStream = lei16p(hdr->AS.Header+pos);
					pos += 2;
					int16_t SpikeScopeVoltageTriggerMode = lei16p(hdr->AS.Header+pos);
					pos += 2;
					int16_t SpikeScopeVoltageThreshold = lei16p(hdr->AS.Header+pos);
					pos += 2;
					int16_t SpikeScopeDigitalTriggerChannel = lei16p(hdr->AS.Header+pos);
					pos += 2;
					int16_t SpikeScopeDigitalTriggerEdgePolarity = lei16p(hdr->AS.Header+pos);
					pos += 2;
					float ElectrodeImpedanceMagnitude = lef32p(hdr->AS.Header+pos);
					pos += 4;
					float ElectrodeImpedancePhase = lef32p(hdr->AS.Header+pos);
					pos += 4;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %u); %s(...) group=%u %u %d SignalType=%d ChannelEnabled=%d flagDC=%d pos=%d NS=%u bi=%u\n",
			__FILE__,__LINE__,__func__,
			nsg,k,ChannelEnabled,SignalType,ChannelEnabled,flag_DC_amplifier_data_saved,(int)pos,NS,(unsigned)bi);

					int nn = 1 + (SignalType==0) + flag_DC_amplifier_data_saved*(SignalType==0);
					char *ChannelName = NULL;
					for (int k2=0; k2 < ChannelEnabled * nn; k2++) {
						NS++;	// first channel is Time channel
						CHANNEL_TYPE *hc = hdr->CHANNEL + NS;
						strcpy(hc->Label, NativeChannelName);
#ifdef MAX_LENGTH_PHYSDIM
						strcpy(hc->PhysDim,"?");
#endif
						hc->OnOff  = 1;
						hc->GDFTYP = 4; // uint16
						hc->bi     = bi;
						hc->bi8    = bi << 3;
						hc->LeadIdCode = 0;
						hc->DigMin = 0.0;
						hc->DigMax = ldexp(1,16)-1;

						hc->PhysDimCode = 0; // [?] default
						hc->Cal    = 1;	//default
						hc->Off    = 0; // default
						hc->SPR    = hdr->SPR; //default
						switch (SignalType) {
						case 0: 	// RHS2000 amplifier channel
							if (k2==0) {
								ChannelName = hc->Label;
								hc->Cal    = 0.195;
								hc->Off    = -32768 * hc->Cal;
							}
							else if ((k2+1)==nn) {
								// Stimulation Data
								sprintf(hc->Label,"Stim %s",ChannelName);
								hc->PhysDimCode = 4160; // [A]
								hc->Cal    = StimStepSize;
								hc->Off    = 0;
							}
							else {
								sprintf(hc->Label,"DC_AmpData %s",ChannelName);
								hc->Cal    = 19.23;
								hc->Off    = -512 * hc->Cal;
							}
							hc->PhysDimCode = 4256; // [V]
							break;
						/*
						case 2: 	// supply voltage channel
							hc->SPR    = 1;
							hc->Cal    = 0.0000748;
							hc->Off    = 0;
							hc->PhysDimCode = 4256; // [V]
							break
						*/
						case 3: 	// Analog input channel
						case 4: 	// Analog output channel
							hc->PhysDimCode = 4256; // [V]
							hc->Cal    = 0.0003125;
							hc->Off    = -32768 * hc->Cal;
							break;
						case 5: 	// Digital Input channel
						case 6: 	// Digital Output channel
							hc->PhysDimCode = 0; // [1]
							break;
						default:
							biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format Intan RHS2000 - not conformant to specification");
						}
						bi += hc->SPR*2;
						hc->PhysMin = hc->DigMin * hc->Cal + hc->Off;
						hc->PhysMax = hc->DigMax * hc->Cal + hc->Off;

						hc->bufptr = NULL;
						hc->TOffset = 0;
						hc->LowPass = LowPass;
						hc->HighPass = HighPass;
						hc->Notch = Notch;
						hc->XYZ[0] = NAN;
						hc->XYZ[1] = NAN;
						hc->XYZ[2] = NAN;
						hc->Impedance = ElectrodeImpedanceMagnitude;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %u); %s(...) %u %2u %3u Label<%s>\n",__FILE__,__LINE__,__func__,nsg,k,NS,hc->Label);

					}
				}
			}
		}
		{	// channel 0 - time channel
						CHANNEL_TYPE *hc = hdr->CHANNEL+0;
						hc->OnOff = 2;
						strcpy(hc->Label, "time");
						strcpy(hc->Transducer, "");
#ifdef MAX_LENGTH_PHYSDIM
						strcpy(hc->PhysDim,"s");
#endif
						hc->bi      = 0;
						hc->bufptr  = NULL;

						hc->OnOff  = 2;		// time channel
						hc->SPR    = hdr->SPR;
						hc->GDFTYP = 6; // uint32
						hc->bi     = 0;
						hc->bi8    = hc->bi << 3;
						hc->LeadIdCode = 0;
						hc->DigMin = 0.0;
						hc->DigMax = ldexp(1,32)-1;
						hc->Off    = 0.0;
						hc->Cal    = 1.0/hdr->SampleRate;
						hc->PhysDimCode = 2176; // [s]
						hc->PhysMin = 0;
						hc->PhysMax = hc->DigMax*hc->Cal;

						hc->bufptr = NULL;
						hc->TOffset = 0;
						hc->LowPass = 0;
						hc->HighPass = INFINITY;
						hc->Notch = 0;
						hc->XYZ[0] = NAN;
						hc->XYZ[1] = NAN;
						hc->XYZ[2] = NAN;
						hc->Impedance = NAN;
		}

		hdr->HeadLen = pos;
		hdr->NRec = -1;
		hdr->NS = NS+1;
		hdr->AS.bpb = bi;
		ifseek(hdr, hdr->HeadLen, SEEK_SET);

		struct stat FileBuf;
		if (stat(hdr->FileName,&FileBuf)==0) hdr->FILE.size = FileBuf.st_size;
		hdr->NRec = (hdr->FILE.size - hdr->HeadLen) / hdr->AS.bpb;

		return 0;
}

/*
     RHD2000 Data File Formats - Intan Tech
     http://www.intantech.com/files/Intan_RHD2000_data_file_formats.pdf
*/
int sopen_rhd2000_read(HDRTYPE* hdr) {

		float minor = leu16p(hdr->AS.Header+6);
		minor      *= (minor < 10) ? 0.1 : 0.01;
		hdr->VERSION = leu16p(hdr->AS.Header+4) + minor;

		hdr->NS = 1;
		hdr->SampleRate = lef32p(hdr->AS.Header+8);

		float HighPass = ( leu16p(hdr->AS.Header+12) ? lef32p(hdr->AS.Header+14) : 0.0 );
		      HighPass = max( HighPass, lef32p(hdr->AS.Header+18) );
		float LowPass = lef32p(hdr->AS.Header+22);

		const int ListNotch[] = {0,50,60};
		uint16_t tmp = leu16p(hdr->AS.Header+34);
		if (tmp>2) tmp=0;
		float Notch = ListNotch[tmp];
		float fZ_desired   = lef32p(hdr->AS.Header+40); // desired impedance test frequency
		float fZ_actual    = lef32p(hdr->AS.Header+44); // actual impedance test frequency

		size_t pos = 48;
		read_qstring(hdr, &pos, NULL, 0);	// note1
		read_qstring(hdr, &pos, NULL, 0);	// note2
		read_qstring(hdr, &pos, NULL, 0);	// note3

		uint16_t numberTemperatureSensors = leu16p(hdr->AS.Header+pos);
		pos += 2;

		int BoardMode = leu16p(hdr->AS.Header+pos);
		pos += 2;

/*
		float PhysMin=0.;
		float PhysMax=1.;
		float Cal=1, DigOff=0;
		switch (BoardMode) {
		case 0: PhysMax=3.3; Cal=0.000050354; break;
		case 1: PhysMin=-5.0; PhysMax=5.0; Cal = 0.00015259; break;
		case 13: PhysMin=-10.24; PhysMax=10.24; Cal = 0.0003125; break;
		default:
			fprintf(stderr,"%s (line %d): Intan/RHD2000 - unknown Boardmode %d\n", __func__,__LINE__,BoardMode);
			// boardMode unknown
		};
*/

		char *referenceChannelName = NULL;
		read_qstring(hdr, &pos, referenceChannelName, 0);

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
		unsigned bi = 0;
		for (uint16_t k = 0; k<numberOfSignalGroups; k++) {
			char *groupName = NULL;
			read_qstring(hdr, &pos, groupName, 0);
			char *groupNamePrefix = NULL;
			read_qstring(hdr, &pos, groupNamePrefix, 0);

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

					char *NativeChannelName = NULL;
					read_qstring(hdr, &pos, NativeChannelName, MAX_LENGTH_LABEL);
					char *CustomChannelName = NULL;
					read_qstring(hdr, &pos, CustomChannelName, 0);

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
						hc->PhysDimCode = 0; // [?] default
						hc->Cal    = 1;	//default
						hc->Off    = 0; // default
						hc->SPR    = hdr->SPR; //default
						switch (signalType) {
						case 0: 	// amplifier channel
							hc->Cal    = 0.195;
							hc->Off    = -32768*hc->Cal;
							hc->PhysDimCode = 4256; // [V]
							break;
						case 1: 	// auxilary input channel
							hc->SPR    = hdr->SPR/4;
							hc->Cal    = 0.195;
							hc->Off    = -32768*hc->Cal;
							break;
						case 2: 	// supply voltage channel
							hc->SPR    = 1;
							hc->Cal    = 0.0000748;
							hc->Off    = 0;
							hc->PhysDimCode = 4256; // [V]
							break;
						case -1: 	// Temperature Sensor channel
							hc->GDFTYP = 3; // int16
							hc->SPR    = 1;
							hc->Cal    = 0.01;
							hc->Off    = 0;
							hc->PhysDimCode = 6048; // [°C]
							break;
						case 3: 	// USB board ADC input channel
							hc->SPR    = hc->SPR;
							hc->PhysDimCode = 4256; // [V]
							switch(BoardMode) {
							case 0:
								hc->Cal    = 0.000050354;
								hc->Off    = 0;
								break;
							case 1:
								hc->Cal    = 0.00015259;
								hc->Off    = 3-2768*hc->Cal;
								break;
							case 13:
								hc->Cal    = 0.0003125;
								hc->Off    = -32768*hc->Cal;
								break;
							}
							break;
						case 4: 	// USB board digital input channel
							;
						case 5: 	// USB board digital output channel
							;
						}
						bi += hc->SPR*2;
						hc->PhysMin = hc->DigMin * hc->Cal + hc->Off;
						hc->PhysMax = hc->DigMax * hc->Cal + hc->Off;

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
