/*
    sopen_axg_read is a helper function to sopen, reading AXG data. 

    Copyright (C) 2008-2014 Alois Schloegl <alois.schloegl@gmail.com>
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
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <iconv.h>

#if !defined(__APPLE__) && defined (_LIBICONV_H)
 #define iconv		libiconv
 #define iconv_open	libiconv_open
 #define iconv_close	libiconv_close
#endif


#include "../biosig-dev.h"

#define min(a,b)        (((a) < (b)) ? (a) : (b))
#define max(a,b)        (((a) > (b)) ? (a) : (b))

void sopen_axg_read(HDRTYPE* hdr) {

		hdr->FILE.LittleEndian = 0;

		// read whole file into RAM
		size_t count = hdr->HeadLen;
		while (!ifeof(hdr)) {
			const int minsize = 1024;
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, 2*count+minsize);
			count += ifread(hdr->AS.Header+count, 1, count+minsize, hdr);
		}
		ifclose(hdr);

if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i) %p %i\n", __FILE__, __LINE__, hdr->CHANNEL, hdr->NS );

		int32_t nCol;
		switch ((int) hdr->VERSION) {
		case 1:
		case 2:
			nCol         = bei16p(hdr->AS.Header+6);
			hdr->HeadLen = 8;
			break;
		case 3:
		case 4:
		case 5:
		case 6:
			nCol      = bei32p(hdr->AS.Header+8);
			hdr->HeadLen = 12;
			break;
		default:
			biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"AXG - unsupported version number");
			return;
		}

		/* hack: for now each trace (i.e. column) becomes a separate channel -
			later the traces of the channels will be reorganized
		 */
		CHANNEL_TYPE *TEMPCHANNEL = (CHANNEL_TYPE*)malloc(nCol*sizeof(CHANNEL_TYPE));
		char **ValLabel = (char**)malloc(nCol*sizeof(char*));
		uint32_t *keyLabel = (uint32_t*)malloc(nCol*sizeof(uint32_t));

if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i) %p %i\n", __FILE__, __LINE__, hdr->CHANNEL, hdr->NS );

		/******* read all column/traces ******/
		int32_t k;
		uint8_t *pos = hdr->AS.Header+hdr->HeadLen;
		hdr->SPR    = beu32p(pos);
		switch ((int) hdr->VERSION) {
		case 1:
			for (k = 0; k < nCol; k++) {
				CHANNEL_TYPE *hc = TEMPCHANNEL+k;
				hc->GDFTYP = 16; 	//float
				hc->PhysDimCode = 0;
				hc->SPR    = beu32p(pos);

				int strlen = pos[4];   // string in Pascal format
				if (strlen > 79) {
					biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"AXG - invalid title length ");
					return;
				}

				/*  Organize channels  */
				uint32_t i;
				for (i = 0; i < hdr->NS; i++) {
					// check if channel with same title already exists
					if (!memcmp(ValLabel[hdr->NS], pos+4, strlen)) {
						keyLabel[k] = hdr->NS;
						break;
					}
				}
				if (i==hdr->NS) {
					// in case of new title, add another channel
					ValLabel[hdr->NS] = (char*)pos+4;
					keyLabel[k] = hdr->NS;
					hdr->NS++;
				}

				// start of data section
				hc->bufptr = pos+84;
				pos += 84 + hc->SPR * sizeof(float);

			}
			break;
		case 2:
			for (k = 0; k < nCol; k++) {
				CHANNEL_TYPE *hc = TEMPCHANNEL+k;
				hc->GDFTYP = 3;		// int16
				hc->PhysDimCode = 0;
				hc->SPR    = beu32p(pos);
				if (k==0) {
					hc->Off    = bef32p(pos+84);
					hc->Cal    = bef32p(pos+88);
					hc->bufptr = NULL;
					hdr->SampleRate = 1.0 / hc->Cal;
				}
				else {
					hc->Cal    = bef32p(pos+84);
					hc->Off    = 0.0;
					hc->bufptr = pos + 88;
				}
				int strlen = pos[4];   // string in Pascal format
				if (strlen > 79) {
					biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"AXG - invalid title length ");
					return;
				}

				biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"AXG - version 2 not supported yet ");
				return;

				// start of data sectioB
				pos += (k==0 ? 92 : 88 + hc->SPR * sizeof(int16_t) );

			}
			break;
		case 6:
			for (k=0; k < nCol; k++) {

if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i) %p %i\n", __FILE__, __LINE__, hdr->CHANNEL, k );

				CHANNEL_TYPE *hc  = TEMPCHANNEL+k;
				hc->SPR           = beu32p(pos);
				uint32_t datatype = beu32p(pos+4);
				size_t titleLen   = beu32p(pos+8);
				char *inbuf       = (char*)pos + 12;
				hc->bufptr        = pos + 12 + titleLen;

if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i) %i %i %i\n", __FILE__, __LINE__, (int)datatype, (int)titleLen, (int)hc->SPR);
				/*
				// The only types used for data file columns are...
				//   ShortArrayType = 4     IntArrayType = 5
				//   FloatArrayType = 6     DoubleArrayType = 7
				//   SeriesArrayType = 9    ScaledShortArrayType = 10
				*/
				hc->Cal = 1.0;
				hc->Off = 0.0;
				hc->GDFTYP = datatype;	//TEMPCHANNEL.GDFTYP uses a different encoding than standard GDFTYP
				hc->OnOff = 1;
				switch (datatype) {
				case 4: // int16
				case 5: // int32
				case 6: // float32
				case 7: // double
					break;
				case 9: hc->GDFTYP = 17;  // series
					// double firstval  = bef64p(hc->bufptr);
					double increment = bef64p(hc->bufptr+8);
					hc->bufptr = NULL;
					hc->OnOff = 0;
					if (!memcmp(inbuf,"\0T\0i\0m\0e\0 \0(\0s\0)\0",8)) {
						hdr->SampleRate = 1.0/increment;
						//hc->OnOff = 2;	// time axis
					}
					else {
						biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "AXG: series data not being a Time axis is not supported. ");
						return;
					}
					break;
				case 10: // scaled short
					hc->Cal = bef64p(hc->bufptr);
					hc->Off = bef64p(hc->bufptr+8);
					break;
				default:
					hc->OnOff = 0;
					biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "error reading AXG: unsupported data type");
					return;
				}

if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i) %p %i\n", __FILE__, __LINE__, hdr->CHANNEL, k );
				if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i): %i %i %i %i\n", __FILE__, __LINE__, (int)hc->SPR, (int)datatype, (int)titleLen, (int)(pos-hdr->AS.Header) );

				/*  Organize channels
					find number of channels and
					setup data structure that assignes each column to a channel
					ValLabel contains the different Labels - one for each channel
					keyLabel contains the channel number for the corresponding column
				*/
				uint32_t i;
				for (i = 0; i < hdr->NS; i++) {
					// check if channel with same title already exists
					uint32_t prevTitleLen = beu32p((uint8_t*)(ValLabel[i])-4);
					if ((titleLen == prevTitleLen) && !memcmp(ValLabel[i], pos+12, titleLen)) {
						keyLabel[k] = i;
						break;
					}
				}
if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i) %p %i\n", __FILE__, __LINE__, hdr->CHANNEL, k );
				if (i==hdr->NS) {
					// in case of new title, add another channel
					ValLabel[hdr->NS] = (char*)pos+12; 	// pointer to title of channel 'nLabel', length of title is stored in beu32p(pos+8)
					keyLabel[k] = hdr->NS;
					hdr->NS++;
				}

if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i) %p %i\n", __FILE__, __LINE__, hdr->CHANNEL, k );
				// pointer to data sections
				hc->bufptr = pos + 12 + titleLen;


				// move pos to the starting position of the next column
				pos += 12 + titleLen;
				// position of next column
				switch (datatype) {
				case 4:
					pos += hc->SPR * sizeof(int16_t);
					break;
				case 5: //int32
				case 6: //float
					pos += hc->SPR * 4;
					break;
				case 7:
					pos += hc->SPR * sizeof(double);
					break;
				case 9:
					pos += 2 * sizeof(double);
					break;
				case 10:
					pos += 2 * sizeof(double) + hc->SPR * sizeof(int16_t);
					break;
				default:
					biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"error reading AXG: unsupported data type");
				}
				if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i): %i %i %i %i\n", __FILE__, __LINE__, (int)hc->SPR, (int)datatype, (int)titleLen, (int)(pos-hdr->AS.Header) );

			}
			break;
		default:
			biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"AXG version is not supported");
		}

if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i) %p %p %i %i\n", __FILE__, __LINE__, TEMPCHANNEL, hdr->CHANNEL, (int)hdr->NS , (int)sizeof(CHANNEL_TYPE));

		/* convert columns/traces into channels */
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		uint32_t ns;
		for (ns=0; ns < hdr->NS; ns++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL + ns;
			hc->SPR = 0;
			hc->GDFTYP = 0;
			hc->OnOff = 1;
		}
		size_t EventN = 0;
		hdr->SPR = 0;

if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i) NS=%i nCol=%i\n", __FILE__, __LINE__, hdr->NS, nCol );

		int flag_traces_of_first_sweep_done=0;
		for (k=0; k < nCol; k++) {
			/*
				copy essential parameters ´(GDFTYP, OnOff, Cal, Off) from TEMPCHANNEL
				and keep track of the number of samples SPR for each channel
			*/

			// define GDFTYP, Cal, Off
			ns = keyLabel[k];	// channel number for current column
			CHANNEL_TYPE *hc = hdr->CHANNEL + ns;

			hc->SPR   += TEMPCHANNEL[k].SPR;

			switch (TEMPCHANNEL[k].GDFTYP) {
			case 4: // int16
				if (hc->GDFTYP < 3) hc->GDFTYP = 3;
				break;
			case 5: // int32
				if (hc->GDFTYP < 5) hc->GDFTYP = 5;
				break;
			case 6: // float32
				if (hc->GDFTYP < 16) hc->GDFTYP = 16;
				break;
			case 7: // double
				if (hc->GDFTYP < 17) hc->GDFTYP = 17;
				break;
			case 10: hc->GDFTYP = 3;  // scaled short
				if (hc->GDFTYP < 16) hc->GDFTYP = 16;
			}

			if (!flag_traces_of_first_sweep_done) {
				hc->Cal    = TEMPCHANNEL[k].Cal;
				hc->Off    = TEMPCHANNEL[k].Off;
			}
			else {
				if (hc->Cal != TEMPCHANNEL[k].Cal || hc->Off != TEMPCHANNEL[k].Off) {
					// in case input is scaled short, output shoud be float
					hc->GDFTYP = max(16, hc->GDFTYP);
				}
			}

			if (hdr->SPR < hc->SPR) hdr->SPR = hc->SPR;

			if (ns+1 == hdr->NS) {
				flag_traces_of_first_sweep_done = 1;
				// if current column corresponds to last channel, ...
				// check if all traces of the same sweep have the same length, and ...
				for (ns=0; ns < hdr->NS; ns++) {
					CHANNEL_TYPE *hc = hdr->CHANNEL + ns;
					if (hc->OnOff != 1) continue;
					else if (hdr->SPR != hc->SPR) {
						biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"AXG - SPR differs between channel");
						return;
					}
				}

				// ... add segment break in event table.
				if ( hdr->EVENT.N + 1 >= EventN ) {
					EventN += max(EventN, 16);
					hdr->EVENT.POS = (uint32_t*)realloc(hdr->EVENT.POS, EventN * sizeof(*hdr->EVENT.POS));
					hdr->EVENT.TYP = (uint16_t*)realloc(hdr->EVENT.TYP, EventN * sizeof(*hdr->EVENT.TYP));
				}
				hdr->EVENT.TYP[hdr->EVENT.N] = 0x7ffe;
				hdr->EVENT.POS[hdr->EVENT.N] = hdr->SPR;
				hdr->EVENT.N++;
			}
		}
		hdr->EVENT.N--;		// ignore last separator event 

		hdr->NRec = hdr->SPR;
		hdr->SPR = 1;
		uint32_t bi8 = 0;
		for (ns=0; ns < hdr->NS; ns++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+ns;
			hc->SPR = hdr->SPR;
			hc->bi8 = bi8;
			hc->bi  = bi8/8;
			if (hc->OnOff != 1)
				hc->SPR = 0;
			else
				bi8 += GDFTYP_BITS[hc->GDFTYP];
		}
		hdr->AS.bpb = bi8/8;

		for (ns=0; ns < hdr->NS; ns++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+ns;

			// define hdr->channel[.].Label, hdr->channel[.].PhysDim
			if (hdr->Version <= 2) {
				// PascalToCString(ValLabel[ns]); 	// shift by 1 byte and terminate 0 char
				int strlen = min(ValLabel[ns][0],MAX_LENGTH_LABEL);
				strncpy(hc->Label, (ValLabel[ns])+1, strlen);

				char *u1 = strrchr(ValLabel[ns],'(');
				char *u2 = strrchr(ValLabel[ns],')');
				if (u1 != NULL && u2 != NULL && u1 < u2) {
					*u1 = 0;
					*u2 = 0;
					hc->PhysDimCode = PhysDimCode(u1+1);
				}
			}
			else if (hdr->Version <= 6) {
				char *inbuf       = ValLabel[ns];
				size_t inlen      = beu32p((uint8_t*)(ValLabel[ns])-4);
				char *outbuf      = hc->Label;
				size_t outlen     = MAX_LENGTH_LABEL+1;
#if  defined(_ICONV_H) || defined (_LIBICONV_H)
				iconv_t ICONV = iconv_open("UTF-8","UCS-2BE");
				size_t reticonv = iconv(ICONV, &inbuf, &inlen, &outbuf, &outlen);
				iconv_close(ICONV);

				if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i): %i %i %i %"PRIiPTR"\n", __FILE__, __LINE__, (int)hc->SPR, (int)inlen, (int)(pos-hdr->AS.Header), reticonv );

				if (reticonv == (size_t)(-1) ) {
					perror("AXG - conversion of title failed!!!");
					biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"AXG - conversion of title failed");
					return;
				}
				*outbuf=0;
#else
				++inbuf;
				int i = min(MAX_LENGTH_LABEL, titleLen/2);
				for (; i>0 ; i-- ) {
					*outbuf= *inbuf;
					inbuf += 2;
					outbuf++;
				}
				outbuf = 0;
#endif
				char *u1 = strrchr(hc->Label,'(');
				char *u2 = strrchr(hc->Label,')');
				if (u1 != NULL && u2 != NULL && u1 < u2) {
					*u1 = 0;
					*u2 = 0;
					hc->PhysDimCode = PhysDimCode(u1+1);
				}
			}

			// these might be reorganized below
			hc->DigMax  =  1e9;
			hc->DigMin  = -1e9;
			hc->PhysMax = hc->DigMax;
			hc->PhysMin = hc->DigMin;

			hc->LeadIdCode = 0;
			hc->Transducer[0] = 0;

			hc->Cal     =  1.0;
			hc->Off     =  0.0;

			hc->TOffset   = 0;
			hc->HighPass  = NAN;
			hc->LowPass   = NAN;
			hc->Notch     = NAN;
			hc->Impedance = INFINITY;
			hc->fZ        = NAN;
			hc->XYZ[0] = 0.0;
			hc->XYZ[1] = 0.0;
			hc->XYZ[2] = 0.0;
		}

		hdr->AS.rawdata = (uint8_t*)realloc( hdr->AS.rawdata, hdr->AS.bpb*hdr->SPR*hdr->NRec);

		for (ns=0; ns < hdr->NS; ns++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL + ns;
			hc->SPR = 0;
		}
		for (k=0; k < nCol; k++) {

			ns = keyLabel[k];
			CHANNEL_TYPE *hc = hdr->CHANNEL + ns;

			if (hc->OnOff != 1) continue;

			uint32_t i;
			switch (hc->GDFTYP) {
			case 3:
				assert(TEMPCHANNEL[k].GDFTYP==4);
				for (i=0; i < TEMPCHANNEL[k].SPR; i++) {
					*(int16_t*)(hdr->AS.rawdata + hc->bi + (hc->SPR + i) * hdr->AS.bpb) = bei16p(TEMPCHANNEL[k].bufptr + i*2);
				}
				break;
			case 5:
				switch (TEMPCHANNEL[k].GDFTYP) {
				case 4:
					for (i=0; i < TEMPCHANNEL[k].SPR; i++) {
						*(int32_t*)(hdr->AS.rawdata + hc->bi + (hc->SPR + i) * hdr->AS.bpb) = bei16p(TEMPCHANNEL[k].bufptr + i*2);
					}
					break;
				case 5:
					for (i=0; i < TEMPCHANNEL[k].SPR; i++) {
						*(int32_t*)(hdr->AS.rawdata + hc->bi + (hc->SPR + i) * hdr->AS.bpb) = bei32p(TEMPCHANNEL[k].bufptr + i*4);
					}
					break;
				default: 
					biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"AXG - data conversion not supported ");
					return;
				}
				break;
			case 16:
				switch (TEMPCHANNEL[k].GDFTYP) {
				case 4:
					for (i=0; i < TEMPCHANNEL[k].SPR; i++) {
						*(float*)(hdr->AS.rawdata + hc->bi + (hc->SPR + i) * hdr->AS.bpb) = (float)bei16p(TEMPCHANNEL[k].bufptr + i*2);
					}
					break;
				case 5:
					for (i=0; i < TEMPCHANNEL[k].SPR; i++) {
						*(float*)(hdr->AS.rawdata + hc->bi + (hc->SPR + i) * hdr->AS.bpb) = (float)bei32p(TEMPCHANNEL[k].bufptr + i*4);
					}
					break;
				case 6:
					for (i=0; i < TEMPCHANNEL[k].SPR; i++) {
						*(float*)(hdr->AS.rawdata + hc->bi + (hc->SPR + i) * hdr->AS.bpb) = bef32p(TEMPCHANNEL[k].bufptr + i*4);
					}
					break;
				case 10:
					for (i=0; i < TEMPCHANNEL[k].SPR; i++) {
						*(float*)(hdr->AS.rawdata + hc->bi + (hc->SPR + i) * hdr->AS.bpb) = bei16p(TEMPCHANNEL[k].bufptr + i*2) * TEMPCHANNEL[k].Cal + TEMPCHANNEL[k].Off;
					}
					break;
				default: 
					biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"AXG - data conversion not supported ");
					return;
				}
				break;
			case 17:
				switch (TEMPCHANNEL[k].GDFTYP) {
				case 4:
					for (i=0; i < TEMPCHANNEL[k].SPR; i++) {
						*(double*)(hdr->AS.rawdata + hc->bi + (hc->SPR + i) * hdr->AS.bpb) = (double)bei16p(TEMPCHANNEL[k].bufptr + i*2);
					}
					break;
				case 5:
					for (i=0; i < TEMPCHANNEL[k].SPR; i++) {
						*(double*)(hdr->AS.rawdata + hc->bi + (hc->SPR + i) * hdr->AS.bpb) = (double)bei32p(TEMPCHANNEL[k].bufptr + i*4);
					}
					break;
				case 6:
					for (i=0; i < TEMPCHANNEL[k].SPR; i++) {
						*(double*)(hdr->AS.rawdata + hc->bi + (hc->SPR + i) * hdr->AS.bpb) = (double)bef32p(TEMPCHANNEL[k].bufptr + i*4);
					}
					break;
				case 7:
					for (i=0; i < TEMPCHANNEL[k].SPR; i++) {
						*(double*)(hdr->AS.rawdata + hc->bi + (hc->SPR + i) * hdr->AS.bpb) = bef64p(TEMPCHANNEL[k].bufptr + i*8);
					}
					break;
				case 10:
					for (i=0; i < TEMPCHANNEL[k].SPR; i++) {
						*(double*)(hdr->AS.rawdata + hc->bi + (hc->SPR + i) * hdr->AS.bpb) = bei16p(TEMPCHANNEL[k].bufptr + i*2) * TEMPCHANNEL[k].Cal + TEMPCHANNEL[k].Off;
					}
					break;
				default: 
					biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"AXG - data conversion not supported ");
					return;
				}
				break;
			default: 
				biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"AXG - unsupported target data type");
				return;
			}
			hc->SPR += TEMPCHANNEL[k].SPR;
		}

		for (ns=0; ns < hdr->NS; ns++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL + ns;
			hc->SPR = hdr->SPR;
		}
		// free intermediate data structure to reorganized column/trace to channels
		if(TEMPCHANNEL) free(TEMPCHANNEL);
		if(keyLabel) free(keyLabel);
		if(ValLabel) free(ValLabel);

		// data is stored on hdr->AS.rawdata in such a way that swapping must not be applied
		hdr->FILE.LittleEndian = (__BYTE_ORDER == __LITTLE_ENDIAN);
		hdr->AS.first  = 0;
		hdr->AS.length = (size_t)hdr->NRec;

		// read Comments
		size_t szComments = beu32p(pos);
		char  *inbuf       = (char*)pos+4;
		char  *Comments    = malloc(szComments+1);
		char  *outbuf      = Comments;
		size_t outlen     = szComments+1;
		size_t inlen      = szComments;

		iconv_t ICONV = iconv_open("UTF-8","UCS-2BE");
		size_t reticonv = iconv(ICONV, &inbuf, &inlen, &outbuf, &outlen);
		iconv_close(ICONV);
		if (reticonv == (size_t)(-1) ) {
			perror("AXG - conversion of comments failed!!!");
			biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"AXG - conversion of comments failed");
			return;
		}
		Comments[outlen]=0;

		if (VERBOSE_LEVEL >7)
			fprintf(stdout,"\n=== COMMENT === \n %s\n",Comments);
		pos += 4+szComments;


		// read Notes
		size_t szNotes  = beu32p(pos);
		inbuf           = (char*)pos+4;
		char *Notes     = malloc(szNotes+1);
		outbuf = Notes;
		outlen = szNotes+1;
		inlen  = szNotes;

		ICONV    = iconv_open("UTF-8","UCS-2BE");
		reticonv = iconv(ICONV, &inbuf, &inlen, &outbuf, &outlen);
		iconv_close(ICONV);
		if ( reticonv == (size_t)(-1) ) {
			biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"AXG - conversion of Notes failed");
			return;
		}
		Notes[outlen]=0;

		if (VERBOSE_LEVEL >7)
			fprintf(stdout,"=== NOTES === \n %s\n",Notes);
		pos += 4+szNotes;

		/******  parse Date and Time ********/
		struct tm T; 
#ifdef __GLIBC__
		strptime(strstr(Notes,"Created on ")+11, "%a %b %d %Y", &T);
		strptime(strstr(Notes,"acquisition at ")+15, "%T", &T);
		hdr->T0 = tm_time2gdf_time(&T);
#else
		char DATE[30];
		strncpy(DATE, strstr(Notes,"Created on ")+11, 30);
		DATE[29] = 0;
		strtok(DATE, "\n\r");	// cut at newline
		char *tmp = strtok(DATE, " ");	// day of week - skip 

		tmp = strtok(NULL, " ");	// abreviated month name
		T.tm_mon = month_string2int(tmp);
		
		tmp = strtok(NULL, " ");	// day of month
		T.tm_mday = atoi(tmp); 

		tmp = strtok(NULL

, " ");	// year 
		T.tm_year = atoi(tmp) - 1900; 

		strncpy(DATE, strstr(Notes,"acquisition at ")+15, 9);
		DATE[9] = 0;
		tmp = strtok(DATE, " :");
		T.tm_hour = atoi(tmp);
		tmp = strtok(NULL, " :");
		T.tm_min  = atoi(tmp);
		tmp = strtok(NULL, " :");
		T.tm_sec  = atoi(tmp);

		hdr->T0 = tm_time2gdf_time(&T);

#endif

		hdr->AS.fpulse = Notes; 
		free(Comments);

if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %i)\n", __FILE__, __LINE__ );

}



