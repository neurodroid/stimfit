/*

    Copyright (C) 2013,2014 Alois Schloegl <alois.schloegl@gmail.com>

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

#define _GNU_SOURCE

/*
#include <assert.h>
#include <ctype.h>
#include <stddef.h>
*/
#include <ctype.h>
#include <math.h>      // define macro isnan()
#include <stdlib.h>
#include <string.h>
#include "../biosig-dev.h"
#include "../igor/IgorBin.h"

#define IGOROLD 1	// will be used for testing and migrating to new version

#ifdef __cplusplus
extern "C" {
#endif


#define ITX_MAXLINELENGTH 400

char *IgorChanLabel(char *inLabel, HDRTYPE *hdr, size_t *ngroup, size_t *nseries, size_t *nsweep, size_t *ns) {
	/*
		extract Channel Label of IGOR ITX data format
	*/

	*ns = 0;
	// static char Label[ITX_MAXLINELENGTH+1];
	int k, s = 0, pos4=0, pos1=0;
	for (k = strlen(inLabel); inLabel[k] < ' '; k--);
	inLabel[k+1] = 0;

	while (inLabel[k] >= ' ') {
		while ( inLabel[k] >= '0' && inLabel[k] <= '9' )
			k--;
		if (inLabel[k]=='_') {
			s++;
			if (s==1) pos4 = k;
			if (s==4) pos1 = k;
			k--;
		}
		if ( inLabel[k] < '0' || inLabel[k] > '9' )
			break;
	}

	if (3 < s) {
		char nvar = 0;
		for (k = strlen(inLabel); 0 < k && nvar < 4; k--) {
			if (inLabel[k] == '_') {
				inLabel[k] = 0;
				char  *v = inLabel+k+1;
				size_t n = atol(v);

				switch (nvar) {
				case 0: *ns = n;
					nvar++;
					break;
				case 1: *nsweep = n;
					nvar++;
					break;
				case 2: *nseries = n;
					nvar++;
					break;
				case 3: *ngroup = n;
					nvar++;
					break;
				}
				inLabel[k] = 0;
			}
		}
		for (k=1; inLabel[pos4+k-1]; k++) {
			inLabel[pos1+k] = inLabel[pos4+k];
		}
	}

	if ((*ns)+1 > hdr->NS) {	// another channel
		hdr->NS = (*ns)+1;
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
	}

	return(inLabel);
}


/* compute igor checksum */
int ibwChecksum(int16_t *data, int flag_swap, int oldcksum, int numbytes) {
	numbytes >>= 1;				// 2 bytes to a short -- ignore trailing odd byte.
	if (flag_swap) {
		while(numbytes-- > 0)
			oldcksum += bswap_16(*(data++));
	} else {
		while(numbytes-- > 0)
			oldcksum += *(data++);
	}
	return oldcksum&0xffff;
}


/*
void ReorderBytes(void *p, int bytesPerPoint, long numValues)	// Reverses byte order.
{
	unsigned char ch, *p1, *p2, *pEnd;

	pEnd = (unsigned char *)p + numValues*bytesPerPoint;
	while (p < (void *)pEnd) {
		p1 = p;
		p2 = (unsigned char *)p + bytesPerPoint-1;
		while (p1 < p2) {
			ch = *p1;
			*p1++ = *p2;
			*p2-- = ch;
		}
		p = (unsigned char *)p + bytesPerPoint;
	}
}
*/

void ReorderShort(void* sp) {
	*(uint16_t*)sp = bswap_16(*(uint16_t*)sp);
}

void ReorderLong(void* lp) {
	*(uint32_t*)lp = bswap_32(*(uint32_t*)lp);
}

void ReorderDouble(void* dp) {
	*(uint64_t*)dp = bswap_64(*(uint64_t*)dp);
}

void ReorderBinHeader1(BinHeader1* p) {
	ReorderShort(&p->version);
	ReorderLong(&p->wfmSize);
	ReorderShort(&p->checksum);
}

void ReorderBinHeader2(BinHeader2* p) {
	ReorderShort(&p->version);
	ReorderLong(&p->wfmSize);
	ReorderLong(&p->noteSize);
//	ReorderLong(&p->pictSize);
	ReorderShort(&p->checksum);
}

void ReorderBinHeader3(BinHeader3* p) {
	ReorderShort(&p->version);
	ReorderLong(&p->wfmSize);
	ReorderLong(&p->noteSize);
	ReorderLong(&p->formulaSize);
//	ReorderLong(&p->pictSize);
	ReorderShort(&p->checksum);
}

void ReorderBinHeader5(BinHeader5* p) {
	ReorderShort(&p->version);
	ReorderShort(&p->checksum);
	ReorderLong(&p->wfmSize);
	ReorderLong(&p->formulaSize);
	ReorderLong(&p->noteSize);
	ReorderLong(&p->dataEUnitsSize);
//	ReorderBytes(&p->dimEUnitsSize, 4, 4);
	ReorderLong(&p->dimEUnitsSize[0]);
	ReorderLong(&p->dimEUnitsSize[1]);
	ReorderLong(&p->dimEUnitsSize[2]);
	ReorderLong(&p->dimEUnitsSize[3]);
//	ReorderBytes(&p->dimLabelsSize, 4, 4);
	ReorderLong(&p->dimLabelsSize[0]);
	ReorderLong(&p->dimLabelsSize[1]);
	ReorderLong(&p->dimLabelsSize[2]);
	ReorderLong(&p->dimLabelsSize[3]);
	ReorderLong(&p->sIndicesSize);
//	ReorderLong(&p->optionsSize1);
//	ReorderLong(&p->optionsSize2);
}

void ReorderWaveHeader2(WaveHeader2* p) {
	ReorderShort(&p->type);
//	ReorderLong(&p->next);
	// char bname does not need to be reordered.
//	ReorderShort(&p->whVersion);
//	ReorderShort(&p->srcFldr);
//	ReorderLong(&p->fileName);
	// char dataUnits does not need to be reordered.
	// char xUnits does not need to be reordered.
	ReorderLong(&p->npnts);
//	ReorderShort(&p->aModified);
	ReorderDouble(&p->hsA);
	ReorderDouble(&p->hsB);
//	ReorderShort(&p->wModified);
//	ReorderShort(&p->swModified);
	ReorderShort(&p->fsValid);
	ReorderDouble(&p->topFullScale);
	ReorderDouble(&p->botFullScale);
	// char useBits does not need to be reordered.
	// char kindBits does not need to be reordered.
//	ReorderLong(&p->formula);
//	ReorderLong(&p->depID);
	ReorderLong(&p->creationDate);
	// char wUnused does not need to be reordered.
	ReorderLong(&p->modDate);
//	ReorderLong(&p->waveNoteH);
	// The wData field marks the start of the wave data which will be reordered separately.
}

void ReorderWaveHeader5(WaveHeader5* p) {
//	ReorderLong(&p->next);
	ReorderLong(&p->creationDate);
	ReorderLong(&p->modDate);
	ReorderLong(&p->npnts);
	ReorderShort(&p->type);
//	ReorderShort(&p->dLock);
	// char whpad1 does not need to be reordered.
//	ReorderShort(&p->whVersion);
	// char bname does not need to be reordered.
//	ReorderLong(&p->whpad2);
//	ReorderLong(&p->dFolder);
//	ReorderBytes(&p->nDim, 4, 4);
	ReorderLong(&p->nDim[0]);
	ReorderLong(&p->nDim[1]);
	ReorderLong(&p->nDim[2]);
	ReorderLong(&p->nDim[3]);
//	ReorderBytes(&p->sfA, 8, 4);
	ReorderDouble(&p->sfA[0]);
	ReorderDouble(&p->sfA[1]);
	ReorderDouble(&p->sfA[2]);
	ReorderDouble(&p->sfA[3]);
//	ReorderBytes(&p->sfB, 8, 4);
	ReorderDouble(&p->sfB[0]);
	ReorderDouble(&p->sfB[1]);
	ReorderDouble(&p->sfB[2]);
	ReorderDouble(&p->sfB[3]);
	// char dataUnits does not need to be reordered.
	// char dimUnits does not need to be reordered.
	ReorderShort(&p->fsValid);
//	ReorderShort(&p->whpad3);
	ReorderDouble(&p->topFullScale);
	ReorderDouble(&p->botFullScale);
/*
	// according to IgorBin.h, the following stuff can be ignored for reading IBW files //

	ReorderLong(&p->dataEUnits);
//	ReorderBytes(&p->dimEUnits, 4, 4);
	ReorderLong(&p->dimEUnits[0]);
	ReorderLong(&p->dimEUnits[1]);
	ReorderLong(&p->dimEUnits[2]);
	ReorderLong(&p->dimEUnits[3]);
//	ReorderBytes(&p->dimLabels, 4, 4);
	ReorderLong(&p->dimLabels[0]);
	ReorderLong(&p->dimLabels[1]);
	ReorderLong(&p->dimLabels[2]);
	ReorderLong(&p->dimLabels[3]);
	ReorderLong(&p->waveNoteH);
//	ReorderBytes(&p->whUnused, 4, 16);
	ReorderShort(&p->aModified);
	ReorderShort(&p->wModified);
	ReorderShort(&p->swModified);
	// char useBits does not need to be reordered.
	// char kindBits does not need to be reordered.
	ReorderLong(&p->formula);
	ReorderLong(&p->depID);
	ReorderShort(&p->whpad4);
	ReorderShort(&p->srcFldr);
	ReorderLong(&p->fileName);
	ReorderLong(&p->sIndices);
	// The wData field marks the start of the wave data which will be reordered separately.
*/
}


void sopen_ibw_read (HDRTYPE* hdr) {
/*
	this function will be called by the function SOPEN in "biosig.c"

	Input:
		char* Header	// contains the file content

	Output:
		HDRTYPE *hdr	// defines the HDR structure accoring to "biosig.h"
*/

	fprintf(stdout,"Warning: support for IBW is very experimental\n");

	uint16_t version = *(uint16_t*)hdr->AS.Header;
	char flag_swap = (version & 0xFF) == 0;
	if (flag_swap)
		version = bswap_16(version);

	unsigned count=0;
	int binHeaderSize;
	int waveHeaderSize;

	switch (version) {
	case 1: binHeaderSize=sizeof(BinHeader1); waveHeaderSize=sizeof(WaveHeader2); break;
	case 2: binHeaderSize=sizeof(BinHeader2); waveHeaderSize=sizeof(WaveHeader2); break;
	case 3: binHeaderSize=sizeof(BinHeader3); waveHeaderSize=sizeof(WaveHeader2); break;
	case 5: binHeaderSize=sizeof(BinHeader5); waveHeaderSize=sizeof(WaveHeader5); break;
	default:
		if (VERBOSE_LEVEL>7) fprintf(stderr,"ver=%x \n",version);
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Igor/IBW: unsupported version number");
		return;
	}
	count = binHeaderSize+waveHeaderSize;

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s(line %i): IBW v%i %i %i %i\n",__FILE__,__LINE__,version, binHeaderSize,waveHeaderSize, (int)count);

	if (hdr->HeadLen < count) {
		hdr->AS.Header = realloc(hdr->AS.Header, count+1);
		hdr->HeadLen  += ifread(hdr->AS.Header + hdr->HeadLen, 1, count-hdr->HeadLen, hdr);
	}

	// compute check sum
	if ((int)hdr->VERSION == 5) count -= 4;

	void *buffer = hdr->AS.Header;
	// Check the checksum.
	int crc;
	if ((ibwChecksum((int16_t*)buffer, flag_swap, 0, count))) {
		if (VERBOSE_LEVEL>7) fprintf(stderr,"ver=%x crc = %x  %i %i \n",version, crc, binHeaderSize, waveHeaderSize);
		biosigERROR(hdr, B4C_CRC_ERROR, "Igor/IBW: checksum error");
		return;
	}

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s(line %i): sizeof BinHeaders %i %i %i %i\n",__FILE__,__LINE__,
			(int)sizeof(BinHeader1),(int)sizeof(BinHeader2),(int)sizeof(BinHeader3),(int)sizeof(BinHeader5));

	// Do byte reordering if the file is from another platform.
	if (flag_swap) {
		version = bswap_16(version);
		switch(version) {
			case 1:
				ReorderBinHeader1((BinHeader1*)hdr->AS.Header);
				break;
			case 2:
				ReorderBinHeader2((BinHeader2*)hdr->AS.Header);
				break;
			case 3:
				ReorderBinHeader3((BinHeader3*)hdr->AS.Header);
				break;
			case 5:
				ReorderBinHeader5((BinHeader5*)hdr->AS.Header);
				break;
		}
		switch(version) {
			case 1:				// Version 1 and 2 files use WaveHeader2.
			case 2:
			case 3:
				ReorderWaveHeader2((WaveHeader2*)(hdr->AS.Header+binHeaderSize));
				break;
			case 5:
				ReorderWaveHeader5((WaveHeader5*)(hdr->AS.Header+binHeaderSize));
				break;
		}
	}

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s(line %i): sizeof WaveHeaders %i %i %i v%i\n",__FILE__,__LINE__,
				(int)sizeof(WaveHeader2), (int)sizeof(WaveHeader5), (int)iftell(hdr),version);

	// Read some of the BinHeader fields.
	int16_t type = 0;			// See types (e.g. NT_FP64) above. Zero for text waves.

	hdr->NS = 1;
	hdr->SampleRate = 1.0;
	hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));

	// Read some of the WaveHeader fields.
	switch(version) {
		case 1:
		case 2:
		case 3:
			{
				WaveHeader2* w2;
				w2 = (WaveHeader2*)(buffer+binHeaderSize);
				type = w2->type;
				strncpy(hdr->CHANNEL[0].Label, w2->bname, MAX_LENGTH_LABEL);
				hdr->CHANNEL[0].PhysDimCode = PhysDimCode(w2->dataUnits);
				hdr->CHANNEL[0].SPR = hdr->SPR = 1;
				hdr->NRec = w2->npnts;
#ifdef IGOROLD
				hdr->CHANNEL[0].Cal = w2->hsA;
				hdr->CHANNEL[0].Off = w2->hsB;
/*
				hdr->CHANNEL[0].DigMax = (w2->topFullScale-w2->hsB) / w2->hsA;
				hdr->CHANNEL[0].DigMin = (w2->botFullScale-w2->hsB) / w2->hsA;
*/
#else
				uint16_t pdc = PhysDimCode(w5->dimUnits[0]);
				hdr->SampleRate /= w2->hsA * (pdc==0 ? 0.001 : PhysDimScale(pdc));	// if physical units unspecified, assume millisecond
				hdr->CHANNEL[0].PhysMax = w2->topFullScale;
				hdr->CHANNEL[0].PhysMin = w2->botFullScale;
#endif
				hdr->FLAG.OVERFLOWDETECTION = !w2->fsValid;

				if (VERBOSE_LEVEL>7) fprintf(stdout,"%s(line %i): %f %f <%s> <%s>\n",__FILE__,__LINE__,w2->hsA,w2->hsB,w2->xUnits,w2->dataUnits);
				if (VERBOSE_LEVEL>7) fprintf(stdout,"%s(line %i): %f %f\n",__FILE__,__LINE__,w2->topFullScale,w2->botFullScale);

				hdr->HeadLen = binHeaderSize+waveHeaderSize-16;    // 16 = size of wData field in WaveHeader2 structure.
			}
			break;

		case 5:
			{
				WaveHeader5* w5;
				w5 = (WaveHeader5*)(buffer+binHeaderSize);
				type = w5->type;

				int k;
				size_t nTraces=1;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s(line %i): %i %i %i %i\n", __FILE__, __LINE__, w5->nDim[0], w5->nDim[1], w5->nDim[2], w5->nDim[3]);

				for (k=1; (w5->nDim[k]!=0) && (k<4); k++) {
					// count number of traces
					nTraces *= w5->nDim[k];
				}
				if (nTraces > 1) {
					// set break marker between traces
					hdr->EVENT.N = nTraces-1;
					if (reallocEventTable(hdr, hdr->EVENT.N) == SIZE_MAX) {
						biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
						return;
					};
					size_t n;
					for (n = 0; n < hdr->EVENT.N; n++) {
						hdr->EVENT.TYP[n] = 0x7ffe;
						hdr->EVENT.POS[n] = (n+1)*w5->nDim[0];
						hdr->EVENT.DUR[n] = 0;
						hdr->EVENT.CHN[n] = 0;
					}
				}

				if (VERBOSE_LEVEL>7) {
					for (k=0; k<4; k++)
						fprintf(stdout,"%i\t%f\t%f\n",w5->nDim[k],w5->sfA[k],w5->sfB[k]);
				}

				strncpy(hdr->CHANNEL[0].Label, w5->bname, MAX_LENGTH_LABEL);
				hdr->CHANNEL[0].PhysDimCode = PhysDimCode(w5->dataUnits);
				hdr->CHANNEL[0].SPR = hdr->SPR = 1;
				hdr->NRec        = w5->npnts;
				uint16_t pdc = PhysDimCode(w5->dimUnits[0]);
				hdr->SampleRate /= w5->sfA[0] * (pdc==0 ? 0.001 : PhysDimScale(pdc));	// if physical units unspecified, assume millisecond

				if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): %g.x+%g \n",__FILE__,__LINE__,w5->sfA[0],w5->sfB[0]);
				if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): |%s|%s|%s|%s|\n",__FILE__,__LINE__,w5->dimUnits[0],w5->dimUnits[1],w5->dimUnits[2],w5->dimUnits[3]);

#ifdef IGOROLD
				hdr->CHANNEL[0].Cal = 1.0;
				hdr->CHANNEL[0].Off = 0.0;
#else
				hdr->CHANNEL[0].PhysMax = w5->topFullScale;
				hdr->CHANNEL[0].PhysMin = w5->botFullScale;
#endif
				if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): %f %f\n",__FILE__,__LINE__,w5->topFullScale,w5->botFullScale);

				hdr->FLAG.OVERFLOWDETECTION = !w5->fsValid;

				hdr->HeadLen = binHeaderSize+waveHeaderSize-4;    // 4 = size of wData field in WaveHeader5 structure.
			}
			break;
	}

	if (VERBOSE_LEVEL > 7) fprintf(stdout, "%s (line %i) Wave name=%s, npnts=%d, type=0x%x.\n", __FILE__, __LINE__, hdr->CHANNEL[0].Label, (int)hdr->NRec, type);

	uint16_t gdftyp;
	double digmin=NAN, digmax=NAN;
	// Consider the number type, not including the complex bit or the unsigned bit.
	switch(type & ~(NT_CMPLX | NT_UNSIGNED)) {
		case NT_I8:
			gdftyp = 1;
			hdr->AS.bpb = 1;		// char
			break;
			if (type & NT_UNSIGNED) {
				gdftyp++;
				digmin = ldexp(-1,7);
				digmax = ldexp( 1,7)-1;
			} else {
				digmin = 0;
				digmax = ldexp( 1,8)-1;
			}
			break;
		case NT_I16:
			gdftyp = 3;
			hdr->AS.bpb = 2;		// short
			if (type & NT_UNSIGNED) {
				gdftyp++;
				digmin = ldexp(-1,15);
				digmax = ldexp( 1,15)-1;
			} else {
				digmin = 0;
				digmax = ldexp( 1,16)-1;
			}
			break;
		case NT_I32:
			gdftyp = 5;
			hdr->AS.bpb = 4;		// long
			if (type & NT_UNSIGNED) {
				gdftyp++;
				digmin = ldexp(-1,31);
				digmax = ldexp( 1,31)-1;
			} else {
				digmin = 0;
				digmax = ldexp( 1,32)-1;
			}
			break;
		case NT_FP32:
			gdftyp = 16;
			hdr->AS.bpb = 4;		// float
			digmin = -__FLT_MAX__;
			digmax = __FLT_MAX__;
			break;
		case NT_FP64:
			gdftyp = 17;
			hdr->AS.bpb = 8;		// double
			digmin = -__DBL_MAX__;
			digmax = __DBL_MAX__;
			break;
		case 0: 				// text waves
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Igor/IBW: text waves not supported");
			return;
		default:
			if (VERBOSE_LEVEL>7) fprintf(stderr,"type=%x \n",version);
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Igor/IBW: unsupported or unknown data type");
			return;
			break;
	}

	if (type & NT_CMPLX) {
		hdr->AS.bpb *= 2;			// Complex wave - twice as many points.
		hdr->NS     *= 2;
		hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		strcpy(hdr->CHANNEL[2].Label,"imag part");
		hdr->CHANNEL[1].Cal = 1.0;
		hdr->CHANNEL[1].Off = 0.0;
		hdr->CHANNEL[1].SPR = hdr->SPR;
	}

	typeof (hdr->NS) k;
	size_t bpb=0;
	for (k = 0; k < hdr->NS; k++) {
		CHANNEL_TYPE *hc = hdr->CHANNEL+k;
		hc->GDFTYP = gdftyp;
		hc->OnOff  = 1;
		hc->DigMin = digmin;
		hc->DigMax = digmax;
#ifdef IGOROLD
		hc->PhysMax = digmax * hc->Cal + hc->Off;
		hc->PhysMin = digmin * hc->Cal + hc->Off;
#else
		hc->Cal = (hc->PhysMax - hc->PhysMin) / (digmax - digmin);
		hc->Off =  hc->PhysMin - hc->DigMin * hc->Cal;

		if (VERBOSE_LEVEL > 7) fprintf(stdout, "%s (line %i) %f %f %f %f.\n", __FILE__, __LINE__, hc->PhysMax, hc->PhysMin, digmax, digmin);
#endif
		hc->LeadIdCode = 0;
		hc->bi = bpb;
		hc->Transducer[0] = 0;
		hc->TOffset = 0;
		hc->LowPass = NAN;
		hc->HighPass = NAN;
		hc->Notch = NAN;
		hc->Impedance = NAN;
		hc->XYZ[0] = 0;
		hc->XYZ[1] = 0;
		hc->XYZ[2] = 0;
		bpb += GDFTYP_BITS[gdftyp]/8;
	}

	hdr->FILE.POS = 0;
	hdr->AS.first = 0;
	hdr->AS.length= 0;
	hdr->data.block = NULL;
	hdr->AS.rawdata = NULL;

	if (VERBOSE_LEVEL > 7)
		fprintf(stdout, "%s (line %i) %i %i %i 0x%x\n", __FILE__, __LINE__,
			(int)hdr->HeadLen, (int)hdr->FILE.size, (int)(hdr->AS.bpb*hdr->NRec), (int)(hdr->HeadLen+hdr->AS.bpb*hdr->NRec));

#ifndef IGOROLD
	size_t endpos = hdr->HeadLen + hdr->AS.bpb * hdr->NRec;
	if (endpos < hdr->FILE.size) {
		/*
		 *  If data were recorded with NeuroMatic/NClamp: http://www.neuromatic.thinkrandom.com/
		 *  some additional information like SamplingRate, Scaling, etc. can be obtained from
		 *  a text block at the end of a file.
		 */

		size_t sz = hdr->FILE.size-endpos;
		char *tmpstr = malloc(sz+1);
		ifseek(hdr, endpos, SEEK_SET);
		ifread(tmpstr, 1, sz, hdr);
		tmpstr[sz]=0;
		char *ptr = tmpstr;
		// skip 0-bytes at the beginning of the block
		while ((ptr != (tmpstr+sz)) && (*ptr==0)) {
			ptr++;
		}
		if (ptr != tmpstr + sz) {
			// if block is not empty
			if (VERBOSE_LEVEL > 7) fprintf(stdout, "%s (line %i) [%i]<%s>\n", __FILE__, __LINE__, sz, ptr);

			// parse lines
			char *line = strtok(ptr,"\n\r\0");
			CHANNEL_TYPE *hc = hdr->CHANNEL+0;
			struct tm t;
			while (line != NULL) {

				if (VERBOSE_LEVEL > 7) fprintf(stdout, "%s (line %i) <%s>\n", __FILE__, __LINE__, line);

				size_t p = strcspn(line,":");	// search for field delimiter

				if (!strncmp(line,"ADCname",p)) {
					strncpy(hc->Label,line+p+1,MAX_LENGTH_LABEL+1);
					hc->Label[MAX_LENGTH_LABEL]=0;
				}
				else if (!strncmp(line,"ADCunits",p)) {
					hc->PhysDimCode = PhysDimCode(line+p+1);
					if (VERBOSE_LEVEL > 7) fprintf(stdout, "%s (line %i) %s<%s>\n", __FILE__, __LINE__, line+p+1, PhysDim3(hc->PhysDimCode));
				}
				else if (!strncmp(line,"ADCunitsX",p)) {
					if (VERBOSE_LEVEL > 7) fprintf(stdout, "%s (line %i) Fs=%f %s\n", __FILE__, __LINE__, hdr->SampleRate, line+p+1);
					if (!strcmp(line+p+1,"msec")) line[p+3]=0;
					hdr->SampleRate /= PhysDimScale(PhysDimCode(line+p+1));
					if (VERBOSE_LEVEL > 7) fprintf(stdout, "%s (line %i) %f Hz\n", __FILE__, __LINE__, hdr->SampleRate);
				}
				else if (!strncmp(line,"ADCscale",p)) {
					hc->Cal     = strtod(line+p+1,NULL);
					if (VERBOSE_LEVEL > 7) fprintf(stdout, "%s (line %i) Cal %f %s\n", __FILE__, __LINE__, hc->Cal, line+p+1);
					hc->PhysMax = hc->DigMax * hc->Cal;
					hc->PhysMin = hc->DigMin * hc->Cal;
					hc->Label[MAX_LENGTH_LABEL]=0;
				}
				else if (!strncmp(line,"Time",p)) {
					ptr = line;   // replace separator with :
					while (*ptr) {
						if (*ptr==',') *ptr=':';
						ptr++;
					}
					strptime(line+p+1,"%H:%M:%S",&t);
					if (VERBOSE_LEVEL > 7) fprintf(stdout, "%s (line %i) %s\n", __FILE__, __LINE__, line);
					if (VERBOSE_LEVEL > 7) {
						char tmp[30];
						strftime(tmp,30,"%F %T",&t);
						fprintf(stdout, "%s (line %i) %s\n", __FILE__, __LINE__, tmp);
					}
				}
				else if (!strncmp(line,"Date",p)) {
					strptime(line+p+1,"%d %b %Y",&t);
					t.tm_hour = 0;
					t.tm_min  = 0;
					t.tm_sec  = 0;
					if (VERBOSE_LEVEL > 7) {
						char tmp[30];
						strftime(tmp,30,"%F %T",&t);
						fprintf(stdout, "%s (line %i) %s\n", __FILE__, __LINE__, tmp);
					}
				}
				else if (!strncmp(line,"Time Stamp",p)) {
					//hdr->SampleRate *= hdr->SPR*hdr->NRec/strtod(line+p+1,NULL);
				}

				line = strtok(NULL, "\n\r\0");
			}
			hdr->T0 = tm_time2gdf_time(&t);
		}

		if (tmpstr) free(tmpstr);
	}
#endif   // not defined IGOROLD
}

/*
 * utility functions for managing List of Sweep names
 */

struct sweepnames_t {
	size_t idx;
	char *name;
	struct sweepnames_t *next;
};

size_t search_sweepnames(struct sweepnames_t* list, const char* name) {
	/* first element starts with 1
	   zero is returned in case of empty list
	*/
	struct sweepnames_t* next=list;
	for (next=list; next!=NULL; next=next->next) {
		if (!strcmp(next->name, name))
			return next->idx;
	}
	return 0;
}

struct sweepnames_t* add_sweepnames(struct sweepnames_t* list, const char* name) {
	struct sweepnames_t* next = (struct sweepnames_t*)malloc(sizeof(struct sweepnames_t));
	next->name = strdup(name);
	next->idx  = list==NULL ? 1 : list->idx+1;
	next->next = list;
	return next;
}

size_t count_sweepnames(struct sweepnames_t* list) {
	size_t count = 0;
	for (; list!=NULL; list=list->next)
		count++;
	return count;
}

void clear_sweepnames(struct sweepnames_t* list) {
	if (list==NULL) return;
	if (list->name) free(list->name);
	clear_sweepnames(list->next);
	free(list->next);
}

void sopen_itx_read (HDRTYPE* hdr) {
#define IGOR_MAXLENLINE 400

		char line[IGOR_MAXLENLINE+1];
		char flagData = 0;
		char flagSupported = 1;

	        if (VERBOSE_LEVEL>7)
                        fprintf(stdout,"%s (line %i) start reading %s,v%4.2f format (%i)\n",__FILE__,__LINE__,GetFileTypeString(hdr->TYPE),hdr->VERSION,ifeof(hdr));

		typeof(hdr->SPR) SPR = 0, spr = 0;
		typeof(hdr->NS)  ns  = 0;
		int chanNo=0, sweepNo=0;
		hdr->SPR = 0;
		hdr->NRec= 0;

		struct sweepnames_t* sweepname_list=NULL;

		/*
			checks the structure of the file and extracts formating information
		*/

		ifseek(hdr,0,SEEK_SET);
		int c = ifgetc(hdr);  //  read first character
		while (!ifeof(hdr)) {

	        if (VERBOSE_LEVEL>8)
                        fprintf(stdout,"%s (line %i) start reading %s,v%4.2f format (%i)\n",__FILE__,__LINE__,GetFileTypeString(hdr->TYPE),hdr->VERSION,(int)iftell(hdr));

			int i = 0;
			while ( (c != -1) &&  (c != 10) &&  (c != 13) && (i < IGOR_MAXLENLINE) ) {
				// terminate when any line break occurs
				line[i++] = c;
				c = ifgetc(hdr);
			};
			line[i] = 0;
			while (isspace(c)) c = ifgetc(hdr); 	// keep first non-space character of next line in buffer

	        if (VERBOSE_LEVEL>8)
                        fprintf(stdout,"\t%s (line %i) <%s> (%i)\n",__FILE__,__LINE__,line,(int)iftell(hdr));

			if (!strncmp(line,"BEGIN",5)) {
				flagData = 1;
				spr = 0;
				hdr->CHANNEL[ns].bi = SPR*sizeof(double);
			}
			else if (!strncmp(line,"END",3)) {
				flagData = 0;
                                if ((SPR!=0) && (SPR != spr)) {
					flagSupported = 0;
					if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) ITX (not supported): %i, %i \n",__FILE__,__LINE__, SPR, spr);
				}
				else
					SPR = spr;

				if (ns==0) hdr->SPR += SPR;
			}

			else if (!strncmp(line,"X SetScale/P x",14)) {
				/*
				This line seems to have inconsistant formating, some times an comma is following the first 14 bytes, some times no comma is found.
				here are some examples
				X SetScale/P x, 0.000000000E+00,  5.000000000E-05,"s", AP101222bp_1_63_1_2
				X SetScale/P x 0,5e-05,"s", f0ch1w1_1_0to0_Co_TCh; SetScale y 0,1e-09,"A", f0ch1w1_1_0to0_Co_TCh
				*/

				double TOffset = atof(strtok(line+15,","));
				if (isnan(hdr->CHANNEL[ns].TOffset))
					hdr->CHANNEL[ns].TOffset = TOffset;
				else if (fabs(hdr->CHANNEL[ns].TOffset - TOffset) > 1e-12)
					fprintf(stderr,"Warning TOffsets in channel #%i do not match (%f,%f)", ns, hdr->CHANNEL[ns].TOffset, TOffset);

				double dur = atof(strtok(NULL,","));
				char *p = strchr(line,'"');
				if (p != NULL) {
					p++;
					char *p2 = strchr(p,'"');
					if (p2 != NULL) *p2=0;
					dur *= PhysDimScale(PhysDimCode(p));
				}

				double div = spr / (hdr->SampleRate * dur);
				if (ns==0) {
					hdr->SampleRate = 1.0 / dur;
				}
				else if (hdr->SampleRate == 1.0 / dur)
					;
				else if (div == floor(div)) {
					hdr->SampleRate *= div;
				}
			}

			else if (!strncmp(line,"X SetScale y,",13)) {
				char *p = strchr(line,'"');
				if (p!=NULL) {
					p++;
					char *p2 = strchr(p,'"');
					if (p2!=NULL) *p2=0;
					if (hdr->CHANNEL[ns].PhysDimCode == 0)
						hdr->CHANNEL[ns].PhysDimCode = PhysDimCode(p);
					else if (hdr->CHANNEL[ns].PhysDimCode != PhysDimCode(p)) {
						flagSupported = 0;	// physical units do not match
	if  (VERBOSE_LEVEL>7) fprintf(stdout,"[%s:%i] ITX (not supported): %i, %i,<%s> \n",__FILE__,__LINE__, hdr->CHANNEL[ns].PhysDimCode,PhysDimCode(p),p);
					}
				}
			}
			else if (!strncmp(line,"WAVES",5)) {

               if (VERBOSE_LEVEL>7)
                        fprintf(stdout,"%s (line %i): <%s>#%i: %i/%i\n",__FILE__,__LINE__,line,(int)ns,(int)spr,(int)hdr->SPR);

				char *p;
				p = strrchr(line,'_');
				ns = 0;
				sweepNo = 0;
				if (p != NULL) {
					chanNo  = strtol(p+1, NULL, 10);
					if (chanNo > 0)
						ns = chanNo - 1; // if decoding fails, assume there is only a single channel

					p[0] = 0;

					if (search_sweepnames(sweepname_list, line) == 0) {
						// when sweep not found, add to list
						sweepname_list = add_sweepnames(sweepname_list, line);
					}
				}

		if  (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) ITX (supported %i): %i, %i, %i, %i, %i\n",__FILE__,__LINE__, flagSupported, (int)ns, (int)spr, (int)hdr->SPR, chanNo, sweepNo);

				if (ns >= hdr->NS) {
					hdr->NS = ns+1;
					hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
				}

		if (VERBOSE_LEVEL>7)
                        fprintf(stdout,"%s (line %i): <%s>#%i: %i/%i\n",__FILE__,__LINE__,line,(int)ns,(int)spr,(int)hdr->SPR);

				CHANNEL_TYPE* hc = hdr->CHANNEL + ns;
				strncpy(hc->Label, line+6, MAX_LENGTH_LABEL);

		if (VERBOSE_LEVEL>7)
                        fprintf(stdout,"%s (line %i): <%s>#%i: %i/%i\n",__FILE__,__LINE__,line,(int)ns,(int)spr,(int)hdr->SPR);

				hc->OnOff    = 1;
				hc->GDFTYP   = 17;
				hc->DigMax   = (double)(int16_t)(0x7fff);
				hc->DigMin   = (double)(int16_t)(0x8000);
				hc->LeadIdCode = 0;

				hc->Cal      = 1.0;
				hc->Off      = 0.0;
				hc->Transducer[0] = '\0';
				hc->LowPass  = NAN;
				hc->HighPass = NAN;
				hc->TOffset  = NAN;
				hc->PhysMax  = hc->Cal * hc->DigMax;
				hc->PhysMin  = hc->Cal * hc->DigMin;
				hc->PhysDimCode = 0;

				// decode channel number and sweep number
                if (VERBOSE_LEVEL>7)
                        fprintf(stdout,"%s (line %i): <%s>#%i: %i/%i\n",__FILE__,__LINE__,line,(int)ns,(int)spr,(int)hdr->SPR);

			}
			else if (flagData)
				spr++;
		}

                if (VERBOSE_LEVEL>7)
                        fprintf(stdout,"%s (line %i): scanning %s,v%4.2f format (supported: %i)\n",__FILE__,__LINE__,GetFileTypeString(hdr->TYPE),hdr->VERSION,flagSupported);

		if (!flagSupported) {
			clear_sweepnames(sweepname_list);
			biosigERROR(hdr, hdr->AS.B4C_ERRNUM,
 "This ITX format is not supported. Possible reasons: not generated by Heka-Patchmaster, corrupted, physical units do not match between sweeps, or do not fulfil some other requirements");
			return;
		}

		hdr->NRec = count_sweepnames(sweepname_list);
                if (VERBOSE_LEVEL>7)
                        fprintf(stdout,"%s (line %i): [%i,%i,%i] = %i, %i\n",__FILE__,__LINE__,(int)hdr->NS,(int)hdr->SPR,(int)hdr->NRec,(int)hdr->NRec*hdr->SPR*hdr->NS, (int)hdr->AS.bpb);

		hdr->EVENT.N = hdr->NRec - 1;
		hdr->EVENT.SampleRate = hdr->SampleRate;
		hdr->EVENT.POS = (uint32_t*) realloc(hdr->EVENT.POS, hdr->EVENT.N * sizeof(*hdr->EVENT.POS));
		hdr->EVENT.TYP = (uint16_t*) realloc(hdr->EVENT.TYP, hdr->EVENT.N * sizeof(*hdr->EVENT.TYP));
		hdr->EVENT.CHN = (uint16_t*) realloc(hdr->EVENT.CHN, hdr->EVENT.N * sizeof(*hdr->EVENT.CHN));
		hdr->EVENT.DUR = (uint32_t*) realloc(hdr->EVENT.DUR, hdr->EVENT.N * sizeof(*hdr->EVENT.DUR));
#if (BIOSIG_VERSION >= 10500)
		hdr->EVENT.TimeStamp = (gdf_time*)realloc(hdr->EVENT.TimeStamp, hdr->EVENT.N*sizeof(gdf_time));
#endif

		hdr->NRec = hdr->SPR;
		hdr->SPR  = 1;
		hdr->AS.first  = 0;
		hdr->AS.length = hdr->NRec;
		hdr->AS.bpb = sizeof(double)*hdr->NS;
		for (ns=0; ns < hdr->NS; ns++) {
			hdr->CHANNEL[ns].SPR = hdr->SPR;
			hdr->CHANNEL[ns].bi  = sizeof(double)*ns;
		}

		double *data = (double*)realloc(hdr->AS.rawdata,hdr->NRec*hdr->SPR*hdr->NS*sizeof(double));
		hdr->FILE.LittleEndian = (__BYTE_ORDER == __LITTLE_ENDIAN);   // no swapping
		hdr->AS.rawdata = (uint8_t*) data;

		/*
			reads and converts data into biosig structure
		*/
		spr = 0;SPR = 0;
		ifseek(hdr, 0, SEEK_SET);
		c = ifgetc(hdr);	// read first character
		while (!ifeof(hdr)) {
			int i = 0;
			while ( (c != -1) &&  (c != 10) &&  (c != 13) && (i < IGOR_MAXLENLINE) ) {
				// terminate when any line break occurs
				line[i++] = c;
				c = ifgetc(hdr);
			};
			line[i] = 0;
			while (isspace(c)) c = ifgetc(hdr); 	// keep first non-space character of next line in buffer

			if (!strncmp(line,"BEGIN",5)) {
				flagData = 1;
				spr = 0;
			}
			else if (!strncmp(line,"END",3)) {
				flagData = 0;
				if (chanNo+1 == hdr->NS) SPR += spr;
			}
			else if (!strncmp(line,"X SetScale y,",13)) {
				//ns++;
			}
			else if (!strncmp(line,"WAVES",5)) {
				// decode channel number and sweep number

				chanNo = 0;
				sweepNo= 0;
				char *p;
				p = strrchr(line,'_');
				if (p != NULL) {
					chanNo  = strtol(p+1,NULL,10);
					if (chanNo > 0) chanNo--; 	// if decoding fails, assume there is only a single channel.
					p[0] = 0;
					sweepNo = search_sweepnames(sweepname_list, line);
					if (sweepNo > 0) sweepNo--; 	// if decoding fails, assume there is only a single sweep
				}
				spr = 0;
				if (sweepNo > 0 && chanNo==0) {
					hdr->EVENT.POS[sweepNo-1] = SPR;
					hdr->EVENT.TYP[sweepNo-1] = 0x7ffe;
					hdr->EVENT.DUR[sweepNo-1] = 0;
					hdr->EVENT.CHN[sweepNo-1] = 0;
#if (BIOSIG_VERSION >= 10500)
					hdr->EVENT.TimeStamp[sweepNo-1] = 0;
#endif
				}
			}
			else if (flagData) {
				double val = atof(line);
				data[hdr->NS*(SPR + spr) + chanNo] = val;
				spr++;
			}
		}
		clear_sweepnames(sweepname_list);
		hdr->EVENT.N = sweepNo;

                if (VERBOSE_LEVEL>7)
			fprintf(stdout,"%s (line %i): reading %s,v%4.2f format finished \n",__FILE__,__LINE__,GetFileTypeString(hdr->TYPE),hdr->VERSION);

		hdr->SPR   = 1;
		hdr->NRec *= hdr->SPR;
		hdr->AS.first  = 0;
		hdr->AS.length = hdr->NRec;
		hdr->AS.bpb = sizeof(double)*hdr->NS;
#undef IGOR_MAXLENLINE

}


#ifdef __cplusplus
}
#endif
