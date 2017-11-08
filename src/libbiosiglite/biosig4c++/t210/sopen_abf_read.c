/*

    $Id: sopen_abf_read.c,v 1.2 2009-02-12 16:15:17 schloegl Exp $
    Copyright (C) 2012,2013,2014,2015 Alois Schloegl <alois.schloegl@gmail.com>

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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../biosig-dev.h"

#define ABFLONG int32_t
#include "axon_structs.h"	// ABF2
#include "abfheadr.h"		// ABF1

#define min(a,b)        (((a) < (b)) ? (a) : (b))
#define max(a,b)        (((a) > (b)) ? (a) : (b))

/*
	read data block of ATF file into the cache
	also HDR.NRec is defined here
*/

#if defined(WITH_ATF)
EXTERN_C void sopen_atf_read(HDRTYPE* hdr) {

		ifseek(hdr,0,SEEK_SET);
		size_t ll;
		char *line = NULL;
		ssize_t nc;
		typeof (hdr->NS) k;

		// first line - skip: contains "ATF\t1.0" or alike
		nc = getline(&line, &ll, hdr->FILE.FID);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"ATF line <%s>\n",line);

		// 2nd line: number of rows and colums
		if (line!=NULL) { free(line); line=NULL; }  // allocate line buffer as needed
		nc = getline(&line, &ll, hdr->FILE.FID);
		char *str = line;
		hdr->NRec = strtoul(str,&str,10);
		hdr->SPR  = 1;
		hdr->NS   = strtoul(str,&str,10);
		hdr->SampleRate = 1.0;
		hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));

		if (VERBOSE_LEVEL>7) fprintf(stdout,"ATF line <%s>\n",line);

		// 3rd line: channel label
		if (line!=NULL) { free(line); line=NULL; }  // allocate line buffer as needed
		nc  = getline(&line, &ll, hdr->FILE.FID);

		if (VERBOSE_LEVEL>7) fprintf(stdout,"ATF line <%s>\n",line);

		typeof(hdr->NS) TIMECHANNEL = 0;

		/* TODO:
			distinguish between ATF files with single channel and multiple sections and
			and ATF files with single section and multiple channels
		 */
		int flag_sections_only=1;

		str = strtok(line,"\t\n\r");
		for (k = 0; k < hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			//init_channel(hc);
			hc->GDFTYP 	= 17; // float64
			hc->PhysMin	= -1e9;
			hc->PhysMax	= +1e9;
			hc->DigMin	= -1e9;
			hc->DigMax	= +1e9;
			hc->Cal		= 1.0;
			hc->Off		= 0.0;

			//hc->Label[0]  = '\0';
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
			hc->bi8	 	= k * GDFTYP_BITS[hc->GDFTYP];
			hc->bi	 	= hc->bi8 / 8;
			hdr->AS.bpb    += GDFTYP_BITS[hc->GDFTYP]/8;

			if (str != NULL) {
				size_t len = strlen(str);
				strncpy(hc->Label, str+1, len-2); // do not copy quotes
				if (strncmp(hc->Label,"Section",7)) flag_sections_only=0;

				// extract physical units enclosed in parenthesis "Label (units)"
				str = strchr(str+1,'(');
				if (str != NULL) {
					char *str2 = strchr(str,')');
					if (str2 != NULL) {
						*str2 = 0;
						hc->PhysDimCode = PhysDimCode(str+1);
					}
				}
				if (!strncasecmp(hc->Label,"time",4)) {
					TIMECHANNEL = k+1;
					hc->OnOff   = 0;   // mark channel as containing the time axis
					if (k==0) hdr->SampleRate /= PhysDimScale(hc->PhysDimCode);
				}
			}
			// extract next label
			str = strtok(NULL,"\t\n\r");

			if (VERBOSE_LEVEL>7) fprintf(stdout,"ATF label #%i:<%s>\n",(int)k,hc->Label);

		}
		hdr->HeadLen = iftell(hdr);

		// 4th line: 1st sample of time channel
		if (line!=NULL) { free(line); line=NULL; }  // allocate line buffer as needed
		nc  = getline(&line, &ll, hdr->FILE.FID);
		double t1 = strtod(line, &str);

		// 5th line: 2nd sample of time channel
		if (line!=NULL) { free(line); line=NULL; }  // allocate line buffer as needed
		nc  = getline(&line, &ll, hdr->FILE.FID);
		double t2 = strtod(line, &str);

		hdr->SampleRate /= (t2-t1);
		if (line!=NULL) { free(line); line=NULL; }  // allocate line buffer as needed

		/*
		if ((size_t)(hdr->NRec * hdr->SPR) <= (ln+1) ) {
			hdr->NRec = max(1024, ln*2);
			hdr->AS.rawdata = realloc(hdr->AS.rawdata, hdr->NRec * hdr->SPR * hdr->AS.bpb);
		}

			TODO:
			 this marks that no data has been read, and
			 hdr->SampleRate, are not defined
		 */
		//biosigERROR(hdr, B4C_DATATYPE_UNSUPPORTED, "support for ATF files not complete");

		hdr->AS.rawdata = NULL;

}

EXTERN_C void sread_atf(HDRTYPE* hdr) {

	if (VERBOSE_LEVEL>6) fprintf(stdout,"SREAD ATF [%i,%i]\n",(unsigned)hdr->NRec, (unsigned)hdr->SPR);

	if (hdr->AS.rawdata != NULL) return;
	//if (hdr->NRec * hdr->SPR > 0)
		hdr->AS.rawdata = malloc(hdr->NRec * hdr->SPR * hdr->AS.bpb);

	ifseek(hdr, hdr->HeadLen, SEEK_SET);

	size_t ll;
	char *line = NULL;

	if (VERBOSE_LEVEL>6) fprintf(stdout,"SREAD ATF\n");

	size_t ln = 0;
	while (~ifeof(hdr)) {
		if (line!=NULL) { free(line); line=NULL; }  // allocate line buffer as needed
		ssize_t nc = getline(&line, &ll, hdr->FILE.FID);
		if (nc < 0) break;

		if (VERBOSE_LEVEL>8) fprintf(stdout,"SREAD ATF 2 %i\t<%s>\n",(unsigned)ln,line );

		if ((hdr->NRec * hdr->SPR) <= (long)(ln+1) ) {
			hdr->NRec = max(1024, ln*2);
			hdr->AS.rawdata = realloc(hdr->AS.rawdata, hdr->NRec * hdr->SPR * hdr->AS.bpb);
		}

		if (VERBOSE_LEVEL>8) fprintf(stdout,"SREAD ATF 4 %i\t<%s>\n",(unsigned)ln,line );

		char *str = strtok(line,"\t");
		typeof(hdr->NS) k;
		for (k = 0; k < hdr->NS; k++) {
			*(double*)(hdr->AS.rawdata + ln*hdr->AS.bpb + hdr->CHANNEL[k].bi) = strtod(str, &str);
			// extract next value
			//str = strtok(NULL,"\t");
		}
		if (VERBOSE_LEVEL>8) fprintf(stdout,"SREAD ATF 6 %i\t<%s>\n",(unsigned)ln,line );

		ln++;
	}
	free(line);
	ifclose(hdr);

	hdr->NRec = ln;
	hdr->AS.first  = 0;
	hdr->AS.length = hdr->NRec;
}
#endif

EXTERN_C void sopen_abf_read(HDRTYPE* hdr) {	
/*
	this function will be called by the function SOPEN in "biosig.c"
	Input: 
		char* Header	// contains the file content
	Output: 
		HDRTYPE *hdr	// defines the HDR structure accoring to "biosig.h"
*/	

	if (VERBOSE_LEVEL>7) fprintf(stdout,"sopen_abf_read 101\n");

	if (VERBOSE_LEVEL>7) fprintf(stderr,"%s (line %i) %"PRIiPTR" %i %i %"PRIiPTR"\n",__FILE__,__LINE__,
                sizeof(struct ABFFileHeader),ABF_OLDHEADERSIZE, ABF_HEADERSIZE,
                offsetof(struct ABFFileHeader, lHeaderSize));

	if (VERBOSE_LEVEL>7) fprintf(stderr,"%s (line %i) %i %"PRIiPTR" %i\n",__FILE__,__LINE__,
		ABF_BLOCKSIZE, offsetof(struct ABFFileHeader, lDataSectionPtr),
		ABF_BLOCKSIZE*lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lDataSectionPtr)) );

		size_t count = hdr->HeadLen; 	
		hdr->VERSION = lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fFileVersionNumber));

		hdr->HeadLen = ABF_BLOCKSIZE * lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lDataSectionPtr));
		if (count < hdr->HeadLen) {
			hdr->AS.Header = (uint8_t*) realloc(hdr->AS.Header, hdr->HeadLen);
			count         += ifread(hdr->AS.Header+count, 1, hdr->HeadLen-count, hdr);
		}
		if ( count < (hdr->VERSION < 1.6 ? ABF_OLDHEADERSIZE : ABF_HEADERSIZE)
		  || count < hdr->HeadLen )
		{
			biosigERROR(hdr, B4C_INCOMPLETE_FILE, "Reading ABF Header block failed"); return;
		}
		hdr->HeadLen = count;

		{
			struct tm t;
			uint32_t u = leu32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lFileStartDate));
			t.tm_year = u / 10000 - 1900;
			t.tm_mon  = (u % 10000)/100 - 1;
			t.tm_mday = (u % 100);
			u = leu32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lFileStartTime));
			t.tm_hour = u / 3600;
			t.tm_min = (u % 3600)/60;
			t.tm_sec = (u % 60);
			//u = leu16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nFileStartMillisecs));
			hdr->T0 = tm_time2gdf_time(&t);
		}

		uint16_t gdftyp = 3;
		switch (lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nDataFormat))) {
		case 0: gdftyp = 3; break;
		case 1: gdftyp = 16; break;
		}

	if (VERBOSE_LEVEL>7) fprintf(stderr,"%s (line %i)\n",__FILE__,__LINE__);

		size_t slen;
		slen = min(MAX_LENGTH_MANUF, sizeof(((struct ABFFileHeader*)(hdr->AS.Header))->sCreatorInfo));
		slen = min(MAX_LENGTH_MANUF, ABF_CREATORINFOLEN);
		strncpy(hdr->ID.Manufacturer._field, (char*)hdr->AS.Header + offsetof(struct ABFFileHeader, sCreatorInfo), slen);
		hdr->ID.Manufacturer._field[slen] = 0;
		hdr->ID.Manufacturer.Name = hdr->ID.Manufacturer._field;

		if (hdr->Version > 1.5) {
			slen = min(MAX_LENGTH_RID, sizeof(((struct ABFFileHeader*)(hdr->AS.Header))->sFileComment));
			strncpy(hdr->ID.Recording,(char*)hdr->AS.Header + offsetof(struct ABFFileHeader, sFileComment), slen);
			hdr->ID.Recording[slen] = 0;
		}

		if (VERBOSE_LEVEL>7) {
			fprintf(stdout,"sCreatorInfo:\t%s\n",hdr->AS.Header + offsetof(struct ABFFileHeader, sCreatorInfo));
			fprintf(stdout,"_sFileComment:\t%s\n",hdr->AS.Header + offsetof(struct ABFFileHeader, _sFileComment));
			if (hdr->Version > 1.5)
				fprintf(stdout,"sFileComment:\t%s\n",hdr->AS.Header + offsetof(struct ABFFileHeader, sFileComment));


			fprintf(stdout,"\nlHeaderSize:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lHeaderSize)));
			fprintf(stdout,"lTagSectionPtr:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lTagSectionPtr)));
			fprintf(stdout,"lNumTagEntries:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lNumTagEntries)));
			fprintf(stdout,"lVoiceTagPtr:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lVoiceTagPtr)));
			fprintf(stdout,"lVoiceTagEntries:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lVoiceTagEntries)));
			fprintf(stdout,"lSynchArrayPtr:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lSynchArrayPtr)));
			fprintf(stdout,"lSynchArraySize:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lSynchArraySize)));

			fprintf(stdout,"\nlDataSectionPtr:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lDataSectionPtr)));
			fprintf(stdout,"lScopeConfigPtr:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lScopeConfigPtr)));
			fprintf(stdout,"lNumScopes:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lNumScopes)));
			fprintf(stdout,"_lDACFilePtr:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, _lDACFilePtr)));
			fprintf(stdout,"_lDACFileNumEpisodes:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, _lDACFileNumEpisodes)));
			fprintf(stdout,"lDeltaArrayPtr:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lDeltaArrayPtr)));
			fprintf(stdout,"lNumDeltas:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lNumDeltas)));
			fprintf(stdout,"nDataFormat:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nDataFormat)));
			fprintf(stdout,"nSimultaneousScan:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nSimultaneousScan)));
			fprintf(stdout,"lStatisticsConfigPtr:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lStatisticsConfigPtr)));
			fprintf(stdout,"lAnnotationSectionPtr:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lAnnotationSectionPtr)));
			fprintf(stdout,"lNumAnnotations:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lNumAnnotations)));

			fprintf(stdout,"\nlNumSamplesPerEpisode:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lNumSamplesPerEpisode)));
			fprintf(stdout,"lPreTriggerSamples:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lPreTriggerSamples)));
			fprintf(stdout,"lEpisodesPerRun:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lEpisodesPerRun)));
			fprintf(stdout,"lActualAcqLength:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lActualAcqLength)));
			fprintf(stdout,"lActualEpisodes:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lActualEpisodes)));
			fprintf(stdout,"lRunsPerTrial:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lRunsPerTrial)));
			fprintf(stdout,"lNumberOfTrials:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lNumberOfTrials)));
			fprintf(stdout,"fADCRange:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCRange)));
			fprintf(stdout,"fDACRange:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fDACRange)));
			fprintf(stdout,"lADCResolution:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lADCResolution)));
			fprintf(stdout,"lDACResolution:\t%i\n",lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lDACResolution)));

			fprintf(stdout,"\nchannel_count_acquired:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, channel_count_acquired)));
			fprintf(stdout,"nADCNumChannels:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nADCNumChannels)));
			fprintf(stdout,"fADCSampleInterval:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCSampleInterval)));
			fprintf(stdout,"nDigitalDACChannel:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nDigitalDACChannel)));
			fprintf(stdout,"nOperationMode:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nOperationMode)));
			fprintf(stdout,"nDigitalEnable:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nDigitalEnable)));

			fprintf(stdout,"\nfFileVersionNumber:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fFileVersionNumber)));
			fprintf(stdout,"fHeaderVersionNumber:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fHeaderVersionNumber)));
			fprintf(stdout,"fADCSampleInterval:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCSampleInterval)));
			fprintf(stdout,"fADCSecondSampleInterval:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCSecondSampleInterval)));
			fprintf(stdout,"fSynchTimeUnit:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fSynchTimeUnit)));
			fprintf(stdout,"fSecondsPerRun:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fSecondsPerRun)));
			fprintf(stdout,"fTriggerThreshold:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fTriggerThreshold)));

			fprintf(stdout,"\nfScopeOutputInterval:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fScopeOutputInterval)));
			fprintf(stdout,"fEpisodeStartToStart:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fEpisodeStartToStart)));
			fprintf(stdout,"fRunStartToStart:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fRunStartToStart)));
			fprintf(stdout,"fTrialStartToStart:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fTrialStartToStart)));
			fprintf(stdout,"fStatisticsPeriod:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fStatisticsPeriod)));
			fprintf(stdout,"fADCRange:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCRange)));
			fprintf(stdout,"fDACRange:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fTriggerThreshold)));
			fprintf(stdout,"fTriggerThreshold:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fDACRange)));
			fprintf(stdout,"dFileDuration:\t%g\n",lef64p(hdr->AS.Header + offsetof(struct ABFFileHeader, dFileDuration)));
			fprintf(stdout,"fTriggerThreshold:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fTriggerThreshold)));

		}

	if (VERBOSE_LEVEL>7) fprintf(stdout,"sopen_abf_read 201\n");
	if (VERBOSE_LEVEL>7) fprintf(stderr,"%s (line %i)\n",__FILE__,__LINE__);


		hdr->NS = lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nADCNumChannels));
		if (lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nDigitalEnable)))
			hdr->NS += lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nDigitalDACChannel));

		hdr->SampleRate = 1e6 / (hdr->NS * lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCSampleInterval)));

		hdr->NRec = lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lActualAcqLength)) / lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nADCNumChannels));
		hdr->SPR  = 1;
		hdr->AS.bpb = hdr->NS*GDFTYP_BITS[gdftyp]/8;

		hdr->CHANNEL = realloc(hdr->CHANNEL, hdr->NS*sizeof(CHANNEL_TYPE));
		uint32_t k1=0,k;	// ABF is 32 bits only, no need for more
		for (k=0; k < ABF_ADCCOUNT + ABF_DACCOUNT; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k1;
			if ((k < ABF_ADCCOUNT) && (lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nADCSamplingSeq) + 2 * k) >= 0)) {
				hc->OnOff = 1;
				hc->bufptr = NULL;
				hc->LeadIdCode = 0;
				int ll = min(ABF_ADCNAMELEN, MAX_LENGTH_LABEL);
				strncpy(hc->Label, (char*)hdr->AS.Header + offsetof(struct ABFFileHeader, sADCChannelName) + k*ABF_ADCNAMELEN, ll);
				while (ll > 0 && isspace(hc->Label[--ll]));
				hc->Label[ll+1] = 0;
				char units[ABF_ADCUNITLEN+1];
				{
					memcpy(units, (char*)hdr->AS.Header + offsetof(struct ABFFileHeader, sADCUnits) + k*ABF_ADCUNITLEN, ABF_ADCUNITLEN);
					units[ABF_ADCUNITLEN] = 0;
					int p=ABF_ADCUNITLEN; 	while ( (0<p) && isspace(units[--p])) units[p]=0;  // remove trailing white space
					hc->PhysDimCode = PhysDimCode(units);
				}
				hc->Transducer[0] = '\0';
				hc->LowPass  = lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fSignalLowpassFilter) + 4 * k);
				hc->HighPass = lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fSignalHighpassFilter) + 4 * k);
				hc->Notch    = NAN;
				hc->TOffset  = NAN;
				hc->XYZ[0]   = 0;
				hc->XYZ[1]   = 0;
				hc->XYZ[2]   = 0;
				hc->GDFTYP   = gdftyp;
				hc->SPR      = hdr->SPR;
				hc->bi       = k1*GDFTYP_BITS[gdftyp]/8;
				hc->bi8      = k1*GDFTYP_BITS[gdftyp];
				double PhysMax = lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCRange));
				double DigMax  = (double)lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lADCResolution));
				double fTotalScaleFactor = lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fInstrumentScaleFactor) + 4 * k)
							 * lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCProgrammableGain) + 4 * k);
				hc->Off      = -lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fInstrumentOffset) + 4 * k);
				if (lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nSignalType) + 2 * k) != 0) {
					hc->Off           -= lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fInstrumentOffset) + 4 * k);
					fTotalScaleFactor *= lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fSignalGain) + 4 * k);
				}
				if (hdr->Version > 1.5 &&
				    lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nTelegraphEnable) + 2 * k) != 0) 
				{
					fTotalScaleFactor *= lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fTelegraphAdditGain) + 4 * k);
				}
				hc->Cal      = PhysMax / (fTotalScaleFactor * DigMax);

				hc->DigMax   =  DigMax-1.0;
				hc->DigMin   = -hc->DigMax;
				hc->PhysMax  = hc->DigMax * hc->Cal;
				hc->PhysMin  = hc->DigMin * hc->Cal;
				if (VERBOSE_LEVEL>7) {
					fprintf(stdout,"==== CHANNEL %i [%s] ====\n",k,units);
					fprintf(stdout,"nSignalType:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nSignalType) + 2 * k));
					fprintf(stdout,"nADCPtoLChannelMap:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nADCPtoLChannelMap) + 2 * k));
					fprintf(stdout,"nADCSamplingSeq:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nADCSamplingSeq) + 2 * k));
					fprintf(stdout,"fADCProgrammableGain:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCProgrammableGain) + 4 * k));
					fprintf(stdout,"fADCRange:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCRange) + 4 * k));

					fprintf(stdout,"fADCDisplayAmplification:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCDisplayAmplification) + 4 * k));
					fprintf(stdout,"fADCDisplayOffset:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCDisplayOffset) + 4 * k));
					fprintf(stdout,"fInstrumentScaleFactor:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fInstrumentScaleFactor) + 4 * k));
					fprintf(stdout,"fInstrumentOffset:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fInstrumentOffset) + 4 * k));
					fprintf(stdout,"fSignalGain:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fSignalGain) + 4 * k));
					fprintf(stdout,"fSignalOffset:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fSignalOffset) + 4 * k));
					fprintf(stdout,"fSignalLowpassFilter:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fSignalLowpassFilter) + 4 * k));
					fprintf(stdout,"fSignalHighpassFilter:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fSignalHighpassFilter) + 4 * k));
					if (hdr->Version > 1.5)
						fprintf(stdout,"fTelegraphAdditGain:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fTelegraphAdditGain) + 4 * k));
				}
				k1++;
			}
			else if (k < ABF_ADCCOUNT) {
				// do not increase k1;
				if (VERBOSE_LEVEL>7) {
					fprintf(stdout,"==== CHANNEL %i ====\n",k);

					fprintf(stdout,"nADCPtoLChannelMap:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nADCPtoLChannelMap) + 2 * k));
					fprintf(stdout,"nADCSamplingSeq:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nADCSamplingSeq) + 2 * k));

					fprintf(stdout,"fADCProgrammableGain:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCProgrammableGain) + 4 * k));

					fprintf(stdout,"fADCDisplayAmplification:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCDisplayAmplification) + 4 * k));
					fprintf(stdout,"fADCDisplayOffset:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCDisplayOffset) + 4 * k));
					fprintf(stdout,"fInstrumentScaleFactor:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fInstrumentScaleFactor) + 4 * k));
					fprintf(stdout,"fInstrumentOffset:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fInstrumentOffset) + 4 * k));
					fprintf(stdout,"fSignalGain:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fSignalGain) + 4 * k));
					fprintf(stdout,"fSignalOffset:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fSignalOffset) + 4 * k));
					fprintf(stdout,"fSignalLowpassFilter:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fSignalLowpassFilter) + 4 * k));
					fprintf(stdout,"fSignalHighpassFilter:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fSignalHighpassFilter) + 4 * k));
				}
			}
			else if ( (k1 < ABF_ADCCOUNT+ABF_DACCOUNT) && (k < hdr->NS) ) {
				hc->OnOff = 1;
				hc->bufptr = NULL;
				hc->LeadIdCode = 0;
				strncpy(hc->Label, (char*)hdr->AS.Header + offsetof(struct ABFFileHeader, sDACChannelName) + (k-ABF_ADCCOUNT)*ABF_DACNAMELEN,
					min(ABF_DACNAMELEN,MAX_LENGTH_LABEL));
				hc->Label[ABF_DACNAMELEN] = 0;

				char units[ABF_DACUNITLEN+1]; {
					memcpy(units, hdr->AS.Header + offsetof(struct ABFFileHeader, sDACChannelUnits) + (k-ABF_ADCCOUNT)*ABF_DACUNITLEN, ABF_DACUNITLEN);
					units[ABF_DACUNITLEN] = 0;
					int p=ABF_ADCUNITLEN; 	while ( (0<p) && isspace(units[--p])) units[p]=0;  // remove trailing white space
					hc->PhysDimCode = PhysDimCode(units);
				}
				hc->Transducer[0] = '\0';
				hc->HighPass = NAN;
				hc->LowPass  = NAN;
				hc->Notch    = NAN;
				hc->TOffset  = NAN;
				hc->XYZ[0]   = 0;
				hc->XYZ[1]   = 0;
				hc->XYZ[2]   = 0;
				hc->GDFTYP   = gdftyp;
				hc->SPR = hdr->SPR;
				hc->bi       = k1*GDFTYP_BITS[gdftyp]/8;
				hc->bi8      = k1*GDFTYP_BITS[gdftyp];

				double PhysMax = lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fDACRange));
				double DigMax  = (double)lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lDACResolution));
				hc->Cal = PhysMax/DigMax;
				hc->Off = 0.0;
				hc->PhysMax  = PhysMax * (DigMax-1.0) / DigMax;
				hc->PhysMin  = -hc->PhysMax;
				hc->DigMax  = DigMax-1;
				hc->DigMin  = -hc->DigMax;

				if (VERBOSE_LEVEL>7) {
					fprintf(stdout,"==== CHANNEL %i [%s] ====\nfDACScaleFactor:\t%f\n",k,units,
						lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fDACScaleFactor) + 4 * (k-ABF_ADCCOUNT)));
					fprintf(stdout,"fDACHoldingLevel:\t%f\n",lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fDACHoldingLevel) + 4 * (k-ABF_ADCCOUNT)));
				}
				k1++;
			}
		}

		/* ===== EVENT TABLE ===== */
		uint32_t n1,n2;
		n1 = lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lActualEpisodes)) - 1;
		n2 = lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lNumTagEntries));
		hdr->EVENT.N = n1+n2;

		/* add breaks between sweeps */
		size_t spr = lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lNumSamplesPerEpisode))/hdr->NS;
		hdr->EVENT.SampleRate = hdr->SampleRate;
		hdr->EVENT.POS = (uint32_t*) realloc(hdr->EVENT.POS, hdr->EVENT.N * sizeof(*hdr->EVENT.POS));
		hdr->EVENT.TYP = (uint16_t*) realloc(hdr->EVENT.TYP, hdr->EVENT.N * sizeof(*hdr->EVENT.TYP));
		for (k=0; k < n1; k++) {
			hdr->EVENT.TYP[k] = 0x7ffe;
			hdr->EVENT.POS[k] = (k+1)*spr;
		}
		/* add tags */
		hdr->AS.auxBUF = realloc(hdr->AS.auxBUF, n2 * sizeof(struct ABFTag));
		ifseek(hdr, lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lTagSectionPtr))*ABF_BLOCKSIZE, SEEK_SET);
		count = ifread(hdr->AS.auxBUF, sizeof(struct ABFTag), n2, hdr);
		if (count>255) {
			count = 255;
			fprintf(stderr,"Warning ABF: more than 255 tags cannot be read");
		}
		hdr->EVENT.N = n1+count;
		for (k=0; k < count; k++) {
			uint8_t *abftag = hdr->AS.auxBUF + k * sizeof(struct ABFTag);
			hdr->EVENT.POS[k+n1] = leu32p(abftag)/hdr->NS;
			abftag[ABF_TAGCOMMENTLEN+4-1]=0;
			FreeTextEvent(hdr, k+n1, (char*)(abftag+4));
		}

		// set HeadLen to begin of data block
		hdr->HeadLen = ABF_BLOCKSIZE * lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lDataSectionPtr));
		ifseek(hdr, hdr->HeadLen, SEEK_SET);

}  // end of sopen_abf_read


size_t readABF2block(uint8_t *block, HDRTYPE *hdr, struct ABF_Section *S) {
	S->uBlockIndex  = leu32p(block + offsetof(struct ABF_Section, uBlockIndex));
	if (S->uBlockIndex==0) return 0;

	S->uBytes = leu32p(block + offsetof(struct ABF_Section, uBytes));
	if (S->uBytes==0) return 0;

	S->llNumEntries = leu64p(block + offsetof(struct ABF_Section, llNumEntries));
	hdr->AS.auxBUF = realloc(hdr->AS.auxBUF, S->llNumEntries*S->uBytes);
	ifseek(hdr, S->uBlockIndex*(size_t)512, SEEK_SET);
	return ifread(hdr->AS.auxBUF, 1, S->llNumEntries*S->uBytes, hdr);
}


EXTERN_C void sopen_abf2_read(HDRTYPE* hdr) {
/*
	this function will be called by the function SOPEN in "biosig.c"
	Input:
		char* Header	// contains the file content
	Output:
		HDRTYPE *hdr	// defines the HDR structure accoring to "biosig.h"
*/

	if (VERBOSE_LEVEL>7) fprintf(stdout,"sopen_abf2_read 101\n");

//	biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "ABF2 format currently not supported"); return;

		fprintf(stdout,"Warning ABF2 v%4.2f: implementation is not complete!\n", hdr->VERSION);

		if (hdr->HeadLen < 512) {
		    	hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, 512);
			hdr->HeadLen  += ifread(hdr->AS.Header+hdr->HeadLen, 1, 512-hdr->HeadLen, hdr);
		}

		{
			struct tm t;
			uint32_t u = leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, uFileStartDate));
				t.tm_year = u / 10000 - 1900;
			t.tm_mon  = (u % 10000)/100 - 1;
			t.tm_mday = (u % 100);
			u = leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, uFileStartTimeMS))/1000;
			t.tm_hour = u / 3600;
			t.tm_min = (u % 3600)/60;
			t.tm_sec = (u % 60);
			//u = leu16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nFileStartMillisecs));
			hdr->T0 = tm_time2gdf_time(&t);
		}

		hdr->SPR = 1;
		uint16_t gdftyp = 3;
		switch (lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nDataFormat))) {
		case 0: gdftyp = 3; break;
		case 1: gdftyp = 16; break;
		}

		if (VERBOSE_LEVEL>7) {
			fprintf(stdout,"\nuFileInfoSize:\t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, uFileInfoSize)));
			fprintf(stdout,"uActualEpisodes:\t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, uActualEpisodes)));
			fprintf(stdout,"uFileStartDate:\t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, uFileStartDate)));
			fprintf(stdout,"uFileStartTimeMS:\t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, uFileStartTimeMS)));
			fprintf(stdout,"uStopwatchTime:\t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, uStopwatchTime)));

			fprintf(stdout,"nFileType:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABF_FileInfo, nFileType)));
			fprintf(stdout,"nDataFormat:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABF_FileInfo, nDataFormat)));
			fprintf(stdout,"nSimultaneousScan:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABF_FileInfo, nSimultaneousScan)));
			fprintf(stdout,"nCRCEnable:\t%i\n",lei16p(hdr->AS.Header + offsetof(struct ABF_FileInfo, nCRCEnable)));

			fprintf(stdout,"uFileCRC:          \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, uFileCRC)));
			fprintf(stdout,"uCreatorVersion:   \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, uCreatorVersion)));
			fprintf(stdout,"uCreatorNameIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, uCreatorNameIndex)));
			fprintf(stdout,"uModifierVersion:  \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, uModifierVersion)));
			fprintf(stdout,"uModifierNameIndex:\t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, uModifierNameIndex)));
			fprintf(stdout,"uProtocolPathIndex:\t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, uProtocolPathIndex)));

			fprintf(stdout,"ProtocolSection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, ProtocolSection.uBlockIndex)));
			fprintf(stdout,"ProtocolSection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, ProtocolSection.uBytes)));
			fprintf(stdout,"ProtocolSection.llNumEntries:\t%i\n",(int)lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, ProtocolSection.llNumEntries)));

			fprintf(stdout,"ADCSection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, ADCSection.uBlockIndex)));
			fprintf(stdout,"ADCSection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, ADCSection.uBytes)));
			fprintf(stdout,"ADCSection.llNumEntries:\t%i\n",(int)lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, ADCSection.llNumEntries)));

			fprintf(stdout,"DACSection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, DACSection.uBlockIndex)));
			fprintf(stdout,"DACSection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, DACSection.uBytes)));
			fprintf(stdout,"DACSection.llNumEntries:\t%i\n",(int)lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, DACSection.llNumEntries)));

			fprintf(stdout,"EpochSection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, EpochSection.uBlockIndex)));
			fprintf(stdout,"EpochSection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, EpochSection.uBytes)));
			fprintf(stdout,"EpochSection.llNumEntries:\t%i\n",(int)lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, EpochSection.llNumEntries)));

			fprintf(stdout,"ADCPerDACSection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, ADCPerDACSection.uBlockIndex)));
			fprintf(stdout,"ADCPerDACSection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, ADCPerDACSection.uBytes)));
			fprintf(stdout,"ADCPerDACSection.llNumEntries:\t%i\n",(int)lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, ADCPerDACSection.llNumEntries)));

			fprintf(stdout,"EpochPerDACSection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, EpochPerDACSection.uBlockIndex)));
			fprintf(stdout,"EpochPerDACSection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, EpochPerDACSection.uBytes)));
			fprintf(stdout,"EpochPerDACSection.llNumEntries:\t%i\n",(int)lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, EpochPerDACSection.llNumEntries)));

			fprintf(stdout,"UserListSection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, UserListSection.uBlockIndex)));
			fprintf(stdout,"UserListSection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, UserListSection.uBytes)));
			fprintf(stdout,"UserListSection.llNumEntries:\t%i\n",(int)lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, UserListSection.llNumEntries)));

			fprintf(stdout,"StatsRegionSection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, StatsRegionSection.uBlockIndex)));
			fprintf(stdout,"StatsRegionSection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, StatsRegionSection.uBytes)));
			fprintf(stdout,"StatsRegionSection.llNumEntries:\t%i\n",(int)lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, StatsRegionSection.llNumEntries)));

			fprintf(stdout,"MathSection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, MathSection.uBlockIndex)));
			fprintf(stdout,"MathSection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, MathSection.uBytes)));
			fprintf(stdout,"MathSection.llNumEntries:\t%i\n",(int)lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, MathSection.llNumEntries)));

			fprintf(stdout,"StringsSection.uBlockIndex:\t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, StringsSection.uBlockIndex)));
			fprintf(stdout,"StringsSection.uBytes:     \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, StringsSection.uBytes)));
			fprintf(stdout,"StringsSection.llNumEntries:\t%i\n",(int)lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, StringsSection.llNumEntries)));

			fprintf(stdout,"DataSection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, DataSection.uBlockIndex)));
			fprintf(stdout,"DataSection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, DataSection.uBytes)));
			fprintf(stdout,"DataSection.llNumEntries:\t%"PRIi64"\n",lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, DataSection.llNumEntries)));

			fprintf(stdout,"TagSection.uBlockIndex:  \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, TagSection.uBlockIndex)));
			fprintf(stdout,"TagSection.uBytes:       \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, TagSection.uBytes)));
			fprintf(stdout,"TagSection.llNumEntries: \t%"PRIi64"\n",lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, TagSection.llNumEntries)));

			fprintf(stdout,"ScopeSection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, ScopeSection.uBlockIndex)));
			fprintf(stdout,"ScopeSection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, ScopeSection.uBytes)));
			fprintf(stdout,"ScopeSection.llNumEntries:\t%"PRIi64"\n",lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, ScopeSection.llNumEntries)));

			fprintf(stdout,"DeltaSection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, DeltaSection.uBlockIndex)));
			fprintf(stdout,"DeltaSection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, DeltaSection.uBytes)));
			fprintf(stdout,"DeltaSection.llNumEntries:\t%"PRIi64"\n",lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, DeltaSection.llNumEntries)));

			fprintf(stdout,"VoiceTagSection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, VoiceTagSection.uBlockIndex)));
			fprintf(stdout,"VoiceTagSection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, VoiceTagSection.uBytes)));
			fprintf(stdout,"VoiceTagSection.llNumEntries:\t%"PRIi64"\n",lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, VoiceTagSection.llNumEntries)));

			fprintf(stdout,"SynchArraySection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, SynchArraySection.uBlockIndex)));
			fprintf(stdout,"SynchArraySection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, SynchArraySection.uBytes)));
			fprintf(stdout,"SynchArraySection.llNumEntries:\t%"PRIi64"\n",lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, SynchArraySection.llNumEntries)));

			fprintf(stdout,"AnnotationSection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, AnnotationSection.uBlockIndex)));
			fprintf(stdout,"AnnotationSection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, AnnotationSection.uBytes)));
			fprintf(stdout,"AnnotationSection.llNumEntries:\t%"PRIi64"\n",lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, AnnotationSection.llNumEntries)));

			fprintf(stdout,"StatsSection.uBlockIndex: \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, StatsSection.uBlockIndex)));
			fprintf(stdout,"StatsSection.uBytes:      \t%i\n",leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, StatsSection.uBytes)));
			fprintf(stdout,"StatsSection.llNumEntries:\t%"PRIi64"\n",lei64p(hdr->AS.Header + offsetof(struct ABF_FileInfo, StatsSection.llNumEntries)));

		}

		struct ABF_Section S;
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, ProtocolSection), hdr, &S);
		float fADCRange = lef32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, fADCRange));
		float fDACRange = lef32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, fDACRange));
		int32_t lADCResolution = lei32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, lADCResolution));
		int32_t lDACResolution = lei32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, lDACResolution));
		hdr->SampleRate =  1e6 / lef32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, fADCSequenceInterval));

		if (VERBOSE_LEVEL>7) {
			fprintf(stdout,"nOperationMode:\t%i\n", lei16p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, nOperationMode)));
			fprintf(stdout,"fADCSequenceInterval:\t%g\n", lef32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, fADCSequenceInterval)));
			fprintf(stdout,"fSecondsPerRun:\t%g\n", lef32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, fSecondsPerRun)));
			fprintf(stdout,"fSecondsPerRun:\t%g\n", lef32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, fSecondsPerRun)));
			fprintf(stdout,"lNumSamplesPerEpisode:\t%i\n", lei32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, lNumSamplesPerEpisode)));
			fprintf(stdout,"lPreTriggerSamples:\t%i\n", lei32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, lPreTriggerSamples)));
			fprintf(stdout,"lEpisodesPerRun:\t%i\n", lei32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, lEpisodesPerRun)));
			fprintf(stdout,"lRunsPerTrial:\t%i\n",  lei32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, lRunsPerTrial)));
			fprintf(stdout,"lNumberOfTrials:\t%i\n", lei32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, lNumberOfTrials)));

			fprintf(stdout,"nDigitalEnable:\t%i\n", lei16p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, nDigitalEnable)));
			fprintf(stdout,"nActiveDACChannel:\t%i\n", lei16p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, nActiveDACChannel)));
			fprintf(stdout,"nDigitalHolding:\t%i\n", lei16p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, nDigitalHolding)));
			fprintf(stdout,"nDigitalInterEpisode:\t%i\n", lei16p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, nDigitalInterEpisode)));
			fprintf(stdout,"nDigitalDACChannel:\t%i\n", lei16p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, nDigitalDACChannel)));

			fprintf(stdout,"nDigitizerADCs:\t%i\n", lei16p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, nDigitizerADCs)));
			fprintf(stdout,"nDigitizerDACs:\t%i\n", lei16p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, nDigitizerDACs)));
			fprintf(stdout,"nDigitizerTotalDigitalOuts:\t%i\n", lei16p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, nDigitizerTotalDigitalOuts)));
			fprintf(stdout,"nDigitizerSynchDigitalOuts:\t%i\n", lei16p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, nDigitizerSynchDigitalOuts)));
			fprintf(stdout,"nDigitizerType:\t%i\n", lei16p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, nDigitizerType)));

			fprintf(stdout,"fADCRange:\t%f\n", lef32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, fADCRange)));
			fprintf(stdout,"fDACRange:\t%f\n", lef32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, fDACRange)));
			fprintf(stdout,"lADCResolution:\t%i\n", lei32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, lADCResolution)));
			fprintf(stdout,"lDACResolution:\t%i\n", lei32p(hdr->AS.auxBUF + offsetof(struct ABF_ProtocolInfo, lDACResolution)));
		}

		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, ADCSection), hdr, &S);
		hdr->NS = S.llNumEntries;
		hdr->CHANNEL = realloc(hdr->CHANNEL, hdr->NS*sizeof(CHANNEL_TYPE));
		int k;
		for (k = 0; k < hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			hc->bufptr = NULL;
			hc->LeadIdCode = 0;
			hc->OnOff = 1;

			hc->LowPass  = lef32p(hdr->AS.auxBUF + S.uBytes*k + offsetof(struct ABF_ADCInfo, fSignalLowpassFilter));
			hc->HighPass = lef32p(hdr->AS.auxBUF + S.uBytes*k + offsetof(struct ABF_ADCInfo, fSignalHighpassFilter));
			hc->GDFTYP   = gdftyp;
			hc->SPR      = hdr->SPR;
			hc->bi       = k*GDFTYP_BITS[gdftyp] / 8;

/*
			strncpy(hc->Label, (char*)hdr->AS.Header + offsetof(struct ABFFileHeader, sADCChannelName) + k*ABF_ADCNAMELEN, min(ABF_ADCNAMELEN,MAX_LENGTH_LABEL));
			hc->Label[ABF_ADCNAMELEN] = 0;
			char units[ABF_ADCUNITLEN+1]; {
				memcpy(units, (char*)hdr->AS.Header + offsetof(struct ABFFileHeader, sADCUnits) + k*ABF_ADCUNITLEN, ABF_ADCUNITLEN);
				units[ABF_ADCUNITLEN] = 0;
				int p=ABF_ADCUNITLEN; 	while ( (0<p) && isspace(units[--p])) units[p]=0;  // remove trailing white space
				hc->PhysDimCode = PhysDimCode(units);
			}
*/

			double PhysMax = fADCRange;
			double DigMax  = lADCResolution;
			hc->Cal      = lef32p(hdr->AS.auxBUF + S.uBytes*k + offsetof(struct ABF_ADCInfo, fInstrumentScaleFactor)) / lADCResolution;
			hc->Off      = lef32p(hdr->AS.auxBUF + S.uBytes*k + offsetof(struct ABF_ADCInfo, fInstrumentOffset));
			hc->DigMax   = lADCResolution-1.0;
			hc->DigMin   = -hc->DigMax;
			hc->PhysMax  = hc->DigMax * hc->Cal;
			hc->PhysMin  = hc->DigMin * hc->Cal;

if (VERBOSE_LEVEL>7) {

				fprintf(stdout,"nADCNum:\t%i\n", lei16p(hdr->AS.auxBUF + S.uBytes*k + offsetof(struct ABF_ADCInfo, nADCNum)));
				fprintf(stdout,"nADCSamplingSeq:\t%i\n", lei16p(hdr->AS.auxBUF + S.uBytes*k + offsetof(struct ABF_ADCInfo, nADCSamplingSeq)));
				fprintf(stdout,"nADCPtoLChannelMap:\t%i\n", lei16p(hdr->AS.auxBUF + S.uBytes*k + offsetof(struct ABF_ADCInfo, nADCPtoLChannelMap)));
				fprintf(stdout,"fADCProgrammableGain:\t%f\n", lef32p(hdr->AS.auxBUF + S.uBytes*k + offsetof(struct ABF_ADCInfo, fADCProgrammableGain)));
				fprintf(stdout,"fInstrumentScaleFactor:\t%f\n", lef32p(hdr->AS.auxBUF + S.uBytes*k + offsetof(struct ABF_ADCInfo, fInstrumentScaleFactor)));
				fprintf(stdout,"fInstrumentOffset:\t%f\n", lef32p(hdr->AS.auxBUF + S.uBytes*k + offsetof(struct ABF_ADCInfo, fInstrumentOffset)));
				fprintf(stdout,"fSignalGain:\t%f\n", lef32p(hdr->AS.auxBUF + S.uBytes*k + offsetof(struct ABF_ADCInfo, fSignalGain)));
				fprintf(stdout,"fSignalOffset:\t%f\n", lef32p(hdr->AS.auxBUF + S.uBytes*k + offsetof(struct ABF_ADCInfo, fSignalOffset)));
				fprintf(stdout,"fSignalLowpassFilter:\t%f\n", lef32p(hdr->AS.auxBUF + S.uBytes*k + offsetof(struct ABF_ADCInfo, fSignalLowpassFilter)));
				fprintf(stdout,"fSignalHighpassFilter:\t%f\n", lef32p(hdr->AS.auxBUF + S.uBytes*k + offsetof(struct ABF_ADCInfo, fSignalHighpassFilter)));
			}
		}
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, DACSection), hdr, &S);
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, EpochSection), hdr, &S);
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, ADCPerDACSection), hdr, &S);
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, EpochPerDACSection), hdr, &S);
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, UserListSection), hdr, &S);
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, StatsRegionSection), hdr, &S);
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, MathSection), hdr, &S);
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, StringsSection), hdr, &S);
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, DataSection), hdr, &S);
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, TagSection), hdr, &S);
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, ScopeSection), hdr, &S);
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, DeltaSection), hdr, &S);
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, VoiceTagSection), hdr, &S);
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, SynchArraySection), hdr, &S);
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, AnnotationSection), hdr, &S);
		readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, StatsSection), hdr, &S);


/*
		hdr->NS = lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nADCNumChannels));
		if (lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nDigitalEnable)))
			hdr->NS += lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nDigitalDACChannel));

		hdr->SampleRate = 1e6 / (hdr->NS * lef32p(hdr->AS.Header + offsetof(struct ABFFileHeader, fADCSampleInterval)));

		hdr->NRec = lei32p(hdr->AS.Header + offsetof(struct ABFFileHeader, lActualAcqLength)) / lei16p(hdr->AS.Header + offsetof(struct ABFFileHeader, nADCNumChannels));
		hdr->SPR  = 1;
		hdr->AS.bpb = hdr->NS*GDFTYP_BITS[gdftyp]/8;

		hdr->CHANNEL = realloc(hdr->CHANNEL, hdr->NS*sizeof(CHANNEL_TYPE));
*/

		// beginning of data block
		hdr->HeadLen = leu32p(hdr->AS.Header + offsetof(struct ABF_FileInfo, DataSection.uBlockIndex)) * 512;

}

