/*

    Copyright (C) 2008-2013,2018 Alois Schloegl <alois.schloegl@gmail.com>
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

#include "../biosig.h"

/* TODO: 
	- need to separate sopen_heka() and sread_heka()
	- data swapping 
*/

#define min(a,b)        (((a) < (b)) ? (a) : (b))
#define max(a,b)        (((a) > (b)) ? (a) : (b))

/****************************************************************************
   rational :
     computes the rational approximation of a floating point number
     such that n/d is an approximation for r with an relative
     error smaller than tol

     see Octave's rat.m
 ****************************************************************************/
void rational (double x, double tol, long *n, long *d) {

        if (x != x) {		// i.e. isnan(x)
                *n = 0;
                *d = 0;
                return;
        }

	if (!finite(x)) {
	        *n = x>0; 	// i.e. sign(x)
                *d = 0;
                return;
        }

	tol *= fabs(x);
	*n   = lround(x);
	*d   = 1;
	double frac = x - *n;
	long lastn  = 1, lastd = 0;

	while (fabs((*d) * x - (*n) ) >= fabs((*d) * tol)) {
	        double flip = 1.0/frac;
	        long step   = lround(flip);
	        frac = flip - step;

	        long nextn = *n, nextd = *d;
	        *n = *n * step + lastn;
	        *d = *d * step + lastd;
	        lastn = nextn;
	        lastd = nextd;
	}

	if (*d < 0) {
	        *n = - *n;
	        *d = - *d;
	}
}


/****************************************************************************
   heka2gdftime 
     converts heka time format into gdftime 
 ****************************************************************************/
gdf_time heka2gdftime(double t) {		
	t -= 1580970496; if (t<0) t += 4294967296; t += 9561652096;
	return (uint64_t)ldexp(t/(24.0*60*60) + 584755, 32); // +datenum(1601,1,1));
}		

/****************************************************************************
   sopen_heka
     reads heka format 

     if itx is not null, the file is converted into an ITX formated file 
     and streamed to itx, too. 	
 ****************************************************************************/
void sopen_heka(HDRTYPE* hdr, FILE *itx) {
	size_t count = hdr->HeadLen;

	if (hdr->TYPE==HEKA && hdr->VERSION > 1) {

		int32_t Levels=0;
		uint16_t k;
		//int32_t *Sizes=NULL;
		int32_t Counts[5], counts[5]; //, Sizes[5];
		memset(Counts,0,20);
		memset(counts,0,20);
		//memset(Sizes,0,20);
		uint32_t StartOfData=0,StartOfPulse=0;

		union {
			struct {
				int32_t Root;
				int32_t Group;
				int32_t Series;
				int32_t Sweep;
				int32_t Trace;
			} Rec;
			int32_t all[5];
		} Sizes;


    		// HEKA PatchMaster file format

		count = hdr->HeadLen;
		struct stat FileBuf;
		stat(hdr->FileName,&FileBuf);
		hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, FileBuf.st_size);
		count += ifread(hdr->AS.Header+count, 1, 1024-count, hdr);
		hdr->HeadLen = count;

		hdr->FILE.LittleEndian = *(uint8_t*)(hdr->AS.Header+52) > 0;
		char SWAP = ( hdr->FILE.LittleEndian && (__BYTE_ORDER == __BIG_ENDIAN))  \
			 || (!hdr->FILE.LittleEndian && (__BYTE_ORDER == __LITTLE_ENDIAN));

		if (SWAP) {
			biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Heka/Patchmaster format requires data swapping - this is not supported yet.");
			return;
		}
		SWAP = 0;  // might be useful for compile time optimization

		/* get file size and read whole file */
		count += ifread(hdr->AS.Header+count, 1, FileBuf.st_size - count, hdr);

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s(...): %i bytes read\n",__FILE__,__LINE__,__func__, count);

		// double oTime;
		uint32_t nItems;
		if (hdr->FILE.LittleEndian) {
			// oTime  = lef64p(hdr->AS.Header+40);	// not used
			nItems = leu32p(hdr->AS.Header+48);
		}
		else {
			// oTime  = bef64p(hdr->AS.Header+40);	// not used
			nItems = beu32p(hdr->AS.Header+48);
		}

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s(...): nItems=%i\n",__FILE__,__LINE__,__func__, nItems);

		if (hdr->VERSION == 1) {
			Sizes.Rec.Root   = 544;
			Sizes.Rec.Group  = 128;
			Sizes.Rec.Series = 1120;
			Sizes.Rec.Sweep  = 160;
			Sizes.Rec.Trace  = 296;
		}
		else if (hdr->VERSION == 2)
		for (k=0; k < min(12,nItems); k++) {

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): HEKA nItems=%i\n",__func__,__LINE__, k);

			uint32_t start  = *(uint32_t*)(hdr->AS.Header+k*16+64);
			uint32_t length = *(uint32_t*)(hdr->AS.Header+k*16+64+4);
			if (SWAP) {
				start  = bswap_32(start);
				length = bswap_32(length);
			}
			uint8_t *ext = hdr->AS.Header + k*16 + 64 + 8;

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): HEKA #%i: <%s> [%i:+%i]\n",__func__,__LINE__,k,ext,start,length);

			if (!start) break;

			if ((start+8) > count) {
				biosigERROR(hdr,  B4C_INCOMPLETE_FILE, "Heka/Patchmaster: file is corrupted - segment with pulse data is not available!");
				return;
			}

			if (!memcmp(ext,".pul\0\0\0\0",8)) {
				// find pulse data
				ifseek(hdr, start, SEEK_SET);

				//magic  = *(int32_t*)(hdr->AS.Header+start);
				Levels = *(int32_t*)(hdr->AS.Header+start+4);
				if (SWAP) Levels = bswap_32(Levels);
				if (Levels>5) {
					biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Heka/Patchmaster format with more than 5 levels not supported");
					return;
				}

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): HEKA #%i    Levels=%i\n",__func__,__LINE__,k,Levels);

				memcpy(Sizes.all,hdr->AS.Header+start+8,sizeof(int32_t)*Levels);
				if (SWAP) {
					int l;
					for (l=0; l < Levels; l++) Sizes.all[l] = bswap_32(Sizes.all[l]);
				}

if (VERBOSE_LEVEL>7) {int l; for (l=0; l < Levels; l++) fprintf(stdout,"%s (line %i): HEKA #%i       %i\n",__func__,__LINE__,l, Sizes.all[l]); }

				StartOfPulse = start + 8 + 4 * Levels;
			}

			else if (!memcmp(ext,".dat\0\0\0\0",8)) {
				StartOfData = start;
			}
		}

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s(...): \n",__FILE__,__LINE__,__func__);

		// if (!Sizes) free(Sizes); Sizes=NULL;

/* DONE: HEKA, check channel number and label 
	pass 1:
		+ get number of sweeps
		+ get number of channels
		+ check whether all traces of a single sweep have the same SPR, and Fs
		+ check whether channelnumber (TrAdcChannel), scaling (DataScaler) and Label fit among all sweeps
		+ extract the total number of samples
		+ physical units
		+ level 4 may have no children
		+ count event descriptions Level2/SeLabel
	pass 2:
		+ initialize data to NAN
		+ skip sweeps if selected channel is not in it
		+ Y scale, physical scale
		+ Event.CodeDescription, Events,
		resampling
*/

		uint32_t k1=0, k2=0, k3=0, k4=0;
		uint32_t K1=0, K2=0, K3=0, K4=0, K5=0;
		double t;
		size_t pos;

		// read K1
		if (SWAP) {
			K1 		= bswap_32(*(uint32_t*)(hdr->AS.Header + StartOfPulse + Sizes.Rec.Root));
			hdr->VERSION 	= bswap_32(*(uint32_t*)(hdr->AS.Header + StartOfPulse));
			union {
				double f64;
				uint64_t u64;
			} c;	
			c.u64 = bswap_64(*(uint64_t*)(hdr->AS.Header + StartOfPulse + 520));
			t = c.f64;
		} else {
			K1 		= (*(uint32_t*)(hdr->AS.Header + StartOfPulse + Sizes.Rec.Root));
			hdr->VERSION 	= (*(uint32_t*)(hdr->AS.Header + StartOfPulse));
			t  		= (*(double*)(hdr->AS.Header + StartOfPulse + 520));
		}

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s(...): \n",__FILE__,__LINE__,__func__);
		
		hdr->T0 = heka2gdftime(t);	// this is when when heka was started, data is recorded later.
		hdr->SampleRate = 0.0;
		double *DT = NULL; 	// list of sampling intervals per channel
		hdr->SPR = 0;

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %s(...): %p\n",__FILE__,__LINE__,__func__,hdr->EVENT.CodeDesc);

		/*******************************************************************************************************
			HEKA: read structural information
 		 *******************************************************************************************************/

		pos = StartOfPulse + Sizes.Rec.Root + 4;
		size_t EventN=0;

		for (k1=0; k1<K1; k1++)	{
		// read group

if (VERBOSE_LEVEL>7) fprintf(stdout,"HEKA L1 @%i=\t%i/%i \n",(int)(pos+StartOfData),k1,K1);

			pos += Sizes.Rec.Group+4;
			// read number of children
			K2 = (*(uint32_t*)(hdr->AS.Header+pos-4));

			hdr->AS.auxBUF = (uint8_t*)realloc(hdr->AS.auxBUF,K2*33);	// used to store name of series
			for (k2=0; k2<K2; k2++)	{
				// read series
				union {
					double   f64;
					uint64_t u64;
				} Delay;
				char *SeLabel =  (char*)(hdr->AS.Header+pos+4);		// max 32 bytes
				strncpy((char*)hdr->AS.auxBUF + 33*k2, (char*)hdr->AS.Header+pos+4, 32); hdr->AS.auxBUF[33*k2+32] = 0;
				SeLabel = (char*)hdr->AS.auxBUF + 33*k2;
				double tt  = *(double*)(hdr->AS.Header+pos+136);		// time of series. TODO: this time should be taken into account 
				Delay.u64 = bswap_64(*(uint64_t*)(hdr->AS.Header+pos+472+176));

				gdf_time t = heka2gdftime(tt);
		
				struct tm tm;
				gdf_time2tm_time_r(t,&tm); 
if (VERBOSE_LEVEL>7) fprintf(stdout,"HEKA L2 @%i=%s %f\t%i/%i %i/%i     t=%.17g %s\n",(int)(pos+StartOfData),SeLabel,Delay.f64,k1,K1,k2,K2,ldexp(t,-32),asctime(&tm));

				pos += Sizes.Rec.Series + 4;
				// read number of children
				K3 = (*(uint32_t*)(hdr->AS.Header+pos-4));
				if (EventN <= hdr->EVENT.N + K3 + 2) {
					EventN = max(max(16,EventN),hdr->EVENT.N+K3+2) * 2;
					if (reallocEventTable(hdr, EventN) == SIZE_MAX) {
						biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "Allocating memory for event table failed.");
						return;
					};
				}

				if (!hdr->AS.SegSel[0] && !hdr->AS.SegSel[1] && !hdr->AS.SegSel[2]) {
					// in case of reading the whole file (no sweep selection), include marker for start of series
					FreeTextEvent(hdr, hdr->EVENT.N, SeLabel);
					hdr->EVENT.POS[hdr->EVENT.N] = hdr->SPR;	// within reading the structure, hdr->SPR is used as a intermediate variable counting the number of samples
#if (BIOSIG_VERSION >= 10500)
					hdr->EVENT.TimeStamp[hdr->EVENT.N] = t;
#endif
					hdr->EVENT.N++;
				}

				for (k3=0; k3<K3; k3++)	{
					// read sweep
					hdr->NRec++; 	// increase number of sweeps
					size_t SPR = 0, spr = 0;
					gdf_time t   = heka2gdftime(*(double*)(hdr->AS.Header+pos+48));		// time of sweep. TODO: this should be taken into account 

					gdf_time2tm_time_r(t,&tm); 

if (VERBOSE_LEVEL>7) fprintf(stdout,"HEKA L3 @%i= %fHz\t%i/%i %i/%i %i/%i %s\n",(int)(pos+StartOfData),hdr->SampleRate,k1,K1,k2,K2,k3,K3,asctime(&tm));

					char flagSweepSelected = (hdr->AS.SegSel[0]==0 || k1+1==hdr->AS.SegSel[0])
						              && (hdr->AS.SegSel[1]==0 || k2+1==hdr->AS.SegSel[1])
							      && (hdr->AS.SegSel[2]==0 || k3+1==hdr->AS.SegSel[2]);

					// hdr->SPR
					if (hdr->SPR==0)
						hdr->T0 = t; 		// start time of first recording determines the start time of the recording
					else if (flagSweepSelected && hdr->SPR > 0) {
						// marker for start of sweep
						hdr->EVENT.POS[hdr->EVENT.N] = hdr->SPR;	// within reading the structure, hdr->SPR is used as a intermediate variable counting the number of samples
						hdr->EVENT.TYP[hdr->EVENT.N] = 0x7ffe;
#if (BIOSIG_VERSION >= 10500)
						hdr->EVENT.TimeStamp[hdr->EVENT.N] = t;
#endif
						hdr->EVENT.N++;
					}

					pos += Sizes.Rec.Sweep + 4;
					// read number of children
					K4 = (*(uint32_t*)(hdr->AS.Header+pos-4));
					for (k4=0; k4<K4; k4++)	{
						// read trace
						double DigMin, DigMax;
						uint16_t gdftyp  = 0;
						uint32_t ns      = (*(uint32_t*)(hdr->AS.Header+pos+36));
						uint32_t DataPos = (*(uint32_t*)(hdr->AS.Header+pos+40));
						spr              = (*(uint32_t*)(hdr->AS.Header+pos+44));
						double DataScaler= (*(double*)(hdr->AS.Header+pos+72));
						double Toffset   = (*(double*)(hdr->AS.Header+pos+80));		// time offset of 
						uint16_t pdc     = PhysDimCode((char*)(hdr->AS.Header + pos + 96));
						double dT        = (*(double*)(hdr->AS.Header+pos+104));
						//double XStart    = (*(double*)(hdr->AS.Header+pos+112));
						uint16_t XUnits  = PhysDimCode((char*)(hdr->AS.Header+pos+120));
						double YRange    = (*(double*)(hdr->AS.Header+pos+128));
						double YOffset   = (*(double*)(hdr->AS.Header+pos+136));
						double Bandwidth = (*(double*)(hdr->AS.Header+pos+144));
						//double PipetteResistance  = (*(double*)(hdr->AS.Header+pos+152));
						double RsValue   = (*(double*)(hdr->AS.Header+pos+192));

						uint8_t ValidYRange = hdr->AS.Header[pos+220];
						uint16_t AdcChan = (*(uint16_t*)(hdr->AS.Header+pos+222));
						/* obsolete: range is defined by DigMin/DigMax * DataScaler + YOffset
						double PhysMin   = (*(double*)(hdr->AS.Header+pos+224));
						double PhysMax   = (*(double*)(hdr->AS.Header+pos+232));
						*/

if (VERBOSE_LEVEL>7) fprintf(stdout, "%s (line %i): %i %i %i %i %i %g %g 0x%x xUnits=%i %g %g %g %g %i %i\n", __FILE__,__LINE__, k1, k2, k3, k4, ns, DataScaler, Toffset, pdc, XUnits, YRange, YOffset, Bandwidth, RsValue, ValidYRange, AdcChan);

						switch (hdr->AS.Header[pos+70]) {
						case 0: gdftyp = 3; 		//int16
							/*
								It seems that the range is 1.024*(2^15-1)/2^15 nA or V
								and symetric around zero. i.e. YOffset is zero
							*/
							DigMax =  ldexp(1.0,15) - 1.0;
							DigMin = -DigMax;
							break;
						case 1: gdftyp = 5; 		//int32
							DigMax =  ldexp(1.0, 31) - 1.0;
							DigMin = -DigMax;
							break;
						case 2: gdftyp = 16; 		//float32
							DigMax =  1e9;
							DigMin = -1e9;
							break;
						case 3: gdftyp = 17; 		//float64
							DigMax =  1e9;
							DigMin = -1e9;
							break;
						default:
							DigMax =  NAN;
							DigMin =  NAN;
							biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Heka/Patchmaster: data type not supported");
						};

						if (SWAP) {
 							AdcChan = bswap_16(AdcChan);
 							ns  = bswap_32(ns);
 							DataPos = bswap_32(DataPos);
							spr = bswap_32(spr);
							// avoid breaking strict-aliasing rules
							union {
								double f64;
								uint64_t u64;
							} c;	
							c.f64 = dT;      c.u64 = bswap_64(c.u64); dT      = c.f64;
							c.f64 = YRange;  c.u64 = bswap_64(c.u64); YRange  = c.f64;
							c.f64 = YOffset; c.u64 = bswap_64(c.u64); YOffset = c.f64;
							//c.f64 = PhysMax; c.u64 = bswap_64(c.u64); PhysMax = c.f64;
							//c.f64 = PhysMin; c.u64 = bswap_64(c.u64); PhysMin = c.f64;
							c.f64 = Toffset; c.u64 = bswap_64(c.u64); Toffset = c.f64;
 						}

						if (YOffset != 0.0) 
							fprintf(stderr,"!!! WARNING !!!  HEKA: the offset is not zero - "
							"this case is not tested and might result in incorrect scaling of "
							"the data,\n!!! YOU ARE WARNED !!!\n"); 

						// scale to standard units - no prefix 	
						double Cal = DataScaler * PhysDimScale(pdc);
						double Off = YOffset * PhysDimScale(pdc);
						pdc &= 0xffe0; 
						float Fs = 1.0 / ( dT  * PhysDimScale(XUnits) ) ;  // float is used to avoid spurios accuracy, round to single precision accuracy

						if (flagSweepSelected) {

							if (hdr->SampleRate <= 0.0) hdr->SampleRate = Fs;
                                                        if (fabs(hdr->SampleRate - Fs) > 1e-9*Fs) {
								long DIV1 = 1, DIV2 = 1;
								rational(hdr->SampleRate*dT*PhysDimScale(XUnits), 1e-6, &DIV2, &DIV1);

								if (DIV1 > 1) {
									if ( ((size_t)DIV1 * hdr->SPR) > 0xffffffffffffffff) {
										fprintf(stderr,"!!! WARNING sopen_heka(%s) !!! due to resampling, the data will have more then 2^31 samples !!!\n", hdr->FileName);
										biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"HEKA file has more than 2^32 samples - this is not supported yet");
									}
									hdr->SPR *= DIV1;
									hdr->SampleRate *= DIV1;
							                hdr->EVENT.SampleRate = hdr->SampleRate;
									size_t n = 0;
									while (n < hdr->EVENT.N)
										hdr->EVENT.POS[n++] *= DIV1;
								}	
								if (DIV2 > 1) spr *= DIV2;
							}

							// samples per sweep
							if (k4==0) SPR = spr;
							else if (SPR != spr) {
								biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Heka/Patchmaster: number of samples among channels within a single sweep do not match.");
								return;
							}
						}

						char *Label = (char*)hdr->AS.Header+pos+4;
						for (ns=0; ns < hdr->NS; ns++) {
							if (!strcmp(hdr->CHANNEL[ns].Label,Label)) break;
						}

if (VERBOSE_LEVEL>7) fprintf(stdout,"HEKA L4 @%i= #%i,%i, %s %f %fHz\t%i/%i %i/%i %i/%i %i/%i \n",(int)(pos+StartOfData),ns,AdcChan,Label,hdr->SampleRate,Fs,k1,K1,k2,K2,k3,K3,k4,K4);
		
						CHANNEL_TYPE *hc;
						if (ns >= hdr->NS) {
							hdr->NS = ns + 1;
#ifdef WITH_TIMESTAMPCHANNEL
							// allocate memory for an extra time stamp channel, which is define only after the end of the channel loop - see below
							hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, (hdr->NS + 1) * sizeof(CHANNEL_TYPE));  
#else
							hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));  
#endif
							hc = hdr->CHANNEL + ns;
							strncpy(hc->Label, Label, max(32, MAX_LENGTH_LABEL));
							hc->Label[max(32,MAX_LENGTH_LABEL)] = 0;
							hc->Transducer[0] = 0;
                                                        hc->SPR     = 1;
							hc->PhysDimCode = pdc;
                                                        hc->OnOff   = 1;
							hc->GDFTYP  = gdftyp;
							hc->LeadIdCode = 0;
							hc->DigMin  = DigMin;
							hc->DigMax  = DigMax;

							// TODO: case of non-zero YOffset is not tested //
							hc->PhysMax = DigMax * Cal + Off;
							hc->PhysMin = DigMin * Cal + Off;

							hc->Cal = Cal;
							hc->Off = Off;
							hc->TOffset = Toffset;

#ifndef NDEBUG
							double Cal2 = (hc->PhysMax - hc->PhysMin) / (hc->DigMax - hc->DigMin);
							double Off2 = hc->PhysMin - Cal2 * hc->DigMin;
							double Off3 = hc->PhysMax - Cal2 * hc->DigMax;

if (VERBOSE_LEVEL>6) fprintf(stdout,"HEKA L5 @%i= #%i,%i, %s %g/%g %g/%g \n",(int)(pos+StartOfData),ns,AdcChan,Label,Cal,Cal2,Off,Off2);

							assert(Cal==Cal2);
							assert(Off==Off2);
							assert(Off==Off3);
#endif

							/* TODO: fix remaining channel header  */
							/* LowPass, HighPass, Notch, Impedance, */
							hc->HighPass = NAN;
							hc->LowPass = (Bandwidth > 0) ? Bandwidth : NAN;
							hc->Notch = NAN;
							hc->Impedance = (RsValue > 0) ? RsValue : NAN;	

							DT = (double*) realloc(DT, hdr->NS*sizeof(double));
							DT[ns] = dT;
						}
						else {
							/*
							   channel has been already defined in earlier sweep.
							   check compatibility and adapt internal format when needed
							*/
							hc = hdr->CHANNEL + ns;
							double PhysMax = DigMax * Cal + Off;
							double PhysMin = DigMin * Cal + Off;
							// get max value to avoid false positive saturation detection when scaling changes
							if (hc->PhysMax < PhysMax) hc->PhysMax = PhysMax;
							if (hc->PhysMin > PhysMin) hc->PhysMin = PhysMin;

							if (hc->GDFTYP < gdftyp) {
								/* when data type changes, use the largest data type */
								if (4 < hc->GDFTYP && hc->GDFTYP < 9 && gdftyp==16)
									/* (U)INT32, (U)INT64 + FLOAT32 -> DOUBLE */
									hc->GDFTYP = 17; 	
								else 
									hc->GDFTYP = gdftyp; 
							}
							else if (hc->GDFTYP > gdftyp) {
								/* when data type changes, use the largest data type */
								if (4 < gdftyp && gdftyp < 9 && hc->GDFTYP==16)
									/* (U)INT32, (U)INT64 + FLOAT32 -> DOUBLE */
									hc->GDFTYP = 17; 	
							}

							if (fabs(hc->Cal - Cal) > 1e-9*Cal) {
								/* when scaling changes from sweep to sweep, use floating point numbers internally. */
								if (hc->GDFTYP < 5) // int16 or smaller 
									hc->GDFTYP = 16;
								else if (hc->GDFTYP < 9) // int32, int64 -> double
									hc->GDFTYP = 17;
							} 

							if ((pdc & 0xFFE0) != (hc->PhysDimCode & 0xFFE0)) {
	                                                        fprintf(stdout, "ERROR: [%i,%i,%i,%i] Yunits in %s do not match %04x(%s) ! %04x(%s)\n",k1,k2,k3,k4, Label, pdc, PhysDim3(pdc), hc->PhysDimCode, PhysDim3(hc->PhysDimCode));
								biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Heka/Patchmaster: Yunits do not match");
							}
	                                                if ( ( VERBOSE_LEVEL > 7 ) && ( fabs( DT[ns] - dT) > 1e-9 * dT) ) {
								fprintf(stdout, "%s (line %i) different sampling rates [%i,%i,%i,%i]#%i,%f/%f \n",__FILE__,__LINE__,(int)k1,(int)k2,(int)k3,(int)k4,(int)ns, 1.0/DT[ns],1.0/dT);
	                                                }
						}

						if (YOffset) {
							biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Heka/Patchmaster: YOffset is not zero");
						}
						if (hdr->AS.Header[pos+220] != 1) {
							fprintf(stderr,"WARNING Heka/Patchmaster: ValidYRange not set to 1 but %i in sweep [%i,%i,%i,%i]\n", hdr->AS.Header[pos+220],k1+1,k2+1,k3+1,k4+1);
						}

if (VERBOSE_LEVEL>7) fprintf(stdout,"HEKA L6 @%i= #%i,%i, %s %f-%fHz\t%i/%i %i/%i %i/%i %i/%i \n",(int)(pos+StartOfData),ns,AdcChan,Label,hdr->SampleRate,Fs,k1,K1,k2,K2,k3,K3,k4,K4);

						pos += Sizes.Rec.Trace+4;
						// read number of children -- this should be 0 - ALWAYS;
						K5 = (*(uint32_t*)(hdr->AS.Header+pos-4));
						if (K5) {
							biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Heka/Patchmaster: Level 4 has some children");
						}
					}	// end loop k4

					// if sweep is selected, add number of samples to counter 
					if (flagSweepSelected) {
						if ( hdr->SPR > 0xffffffffffffffffu-SPR) {
							biosigERROR(hdr,B4C_FORMAT_UNSUPPORTED,"HEKA file has more than 2^32 samples - this is not supported yet");
						}
						hdr->SPR += SPR;
					}
				}		// end loop k3
			}			// end loop k2
		}				// end loop k1

#ifndef NO_BI
		if (DT) free(DT);
#else
		size_t *BI = (size_t*) DT;      // DT is not used anymore, use space for BI
#endif
                DT = NULL;

#ifdef WITH_TIMESTAMPCHANNEL
		{
			/*
				define time stamp channel, memory is already allocated above
			*/		
			CHANNEL_TYPE *hc = hdr->CHANNEL + hdr->NS; 
			hc->GDFTYP  = 7; 	// corresponds to int64_t, gdf_time
			strcpy(hc->Label,"Timestamp");
			hc->Transducer[0]=0;
			hc->PhysDimCode = 2272; // units: days [d]
                        hc->LeadIdCode = 0;
                        hc->SPR     = 1;
			hc->Cal     = ldexp(1.0, -32); 
			hc->Off     = 0.0; 
			hc->OnOff   = 1; 
			hc->DigMax  = ldexp( 1.0, 61); 	
			hc->DigMin  = 0; 	
			hc->PhysMax = hc->DigMax * hc->Cal;  	
			hc->PhysMin = hc->DigMin * hc->Cal;
			hc->TOffset   = 0.0; 
			hc->Impedance = NAN; 
			hc->HighPass = NAN; 
			hc->LowPass  = NAN; 
			hc->Notch    = NAN; 
			hc->XYZ[0] = 0.0; 
			hc->XYZ[1] = 0.0; 
			hc->XYZ[2] = 0.0; 

			hdr->NS++;
		}
#endif

		hdr->NRec = 1;
		hdr->AS.bpb = 0;
		for (k = 0; k < hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL + k;

			hc->Cal = (hc->PhysMax - hc->PhysMin) / (hc->DigMax - hc->DigMin); 
			hc->Off = hc->PhysMin - hc->DigMin * hc->Cal;
#ifndef NO_BI
			hc->bi = hdr->AS.bpb;
#else
			BI[k] = hdr->AS.bpb;
#endif
			hc->SPR = hdr->SPR;
			hdr->AS.bpb += hc->SPR * (GDFTYP_BITS[hc->GDFTYP]>>3);	// multiplation must not exceed 32 bit limit
		}

		if (hdr->AS.B4C_ERRNUM) {
#ifdef NO_BI
			if (BI) free(BI);
#endif
 			return;
		}
		hdr->ID.Manufacturer.Name = "HEKA/Patchmaster"; 



/******************************************************************************
      SREAD_HEKA 

      void sread_heka(HDRTYPE* hdr, FILE *itx, ... ) {

 ******************************************************************************/

if (VERBOSE_LEVEL > 7) fprintf(stdout,"HEKA: 400: %"PRIi64"  %"PRIi32" %"PRIi64"\n",hdr->NRec, hdr->AS.bpb, hdr->NRec * (size_t)hdr->AS.bpb);

		size_t sz = hdr->NRec * (size_t)hdr->AS.bpb;
		if (sz/hdr->NRec < hdr->AS.bpb) {
                        biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "memory allocation failed - more than 2GB required but platform supports only 32 bit!");
                        return;
		}

		void* tmpptr = realloc(hdr->AS.rawdata, sz);
		if (tmpptr!=NULL) 
			hdr->AS.rawdata = (uint8_t*) tmpptr;
		else {
                        biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "memory allocation failed - not enough memory!");
                        return;
		}	
		assert(hdr->NRec >= 0);
		memset(hdr->AS.rawdata, 0xff, hdr->NRec * (size_t)hdr->AS.bpb); 	// initialize with NAN's


#ifdef NO_BI
#define _BI (BI[k])
#else
#define _BI (hc->bi)
#endif
		/* initialize with NAN's */
		for (k=0; k<hdr->NS; k++) {
			size_t k1;
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			switch (hc->GDFTYP) {
			case 3:
				for (k1=0; k1<hc->SPR; k1++) {
                                        *(uint16_t*)(hdr->AS.rawdata + _BI + k1 * 2) = 0x8000;
                                }
				break;
			case 5:
				for (k1=0; k1<hc->SPR; k1++) *(uint32_t*)(hdr->AS.rawdata + _BI + k1 * 4) = 0x80000000;
				break;
			case 7:
				for (k1=0; k1<hc->SPR; k1++) *(int64_t*)(hdr->AS.rawdata + _BI + k1 * 4) = 0x8000000000000000LL;
				break;
			case 16:
				for (k1=0; k1<hc->SPR; k1++) *(float*)(hdr->AS.rawdata + _BI + k1 * 4) = NAN;
				break;
			case 17:
				for (k1=0; k1<hc->SPR; k1++) *(double*)(hdr->AS.rawdata + _BI + k1 * 8) = NAN;
				break;
			}
		}
#undef _BI

		char *WAVENAME = NULL; 
		if (itx) {
			fprintf(itx, "IGOR\r\nX Silent 1\r\n");
			const char *fn = strrchr(hdr->FileName,'\\');
			if (fn) fn++;
			else fn = strrchr(hdr->FileName,'/');
			if (fn) fn++;
			else fn = hdr->FileName;

			size_t len = strspn(fn,"."); 
			WAVENAME = (char*)malloc(strlen(hdr->FileName)+7);
			if (len) 
				strncpy(WAVENAME, fn, len); 
			else 
				strcpy(WAVENAME, fn); 		// Flawfinder: ignore
		}

if (VERBOSE_LEVEL>7) hdr2ascii(hdr,stdout,4);

		/*******************************************************************************************************
			HEKA: read data blocks
 		 *******************************************************************************************************/
		uint32_t SPR = 0;
		pos = StartOfPulse + Sizes.Rec.Root + 4;
		for (k1=0; k1<K1; k1++)	{
		// read group

if (VERBOSE_LEVEL>7) fprintf(stdout,"HEKA+L1 @%i=\t%i/%i \n",(int)(pos+StartOfData),k1,K1);

			pos += Sizes.Rec.Group+4;
			// read number of children
			K2 = (*(uint32_t*)(hdr->AS.Header+pos-4));

			for (k2=0; k2<K2; k2++)	{
				// read series
				union {
					double   f64;
					uint64_t u64;
				} Delay;
				uint32_t spr = 0;
				char *SeLabel = (char*)(hdr->AS.Header+pos+4);		// max 32 bytes
				Delay.u64 = bswap_64(*(uint64_t*)(hdr->AS.Header+pos+472+176));

if (VERBOSE_LEVEL>7) fprintf(stdout,"HEKA+L2 @%i=%s %f\t%i/%i %i/%i \n",(int)(pos+StartOfData),SeLabel,Delay.f64,k1,K1,k2,K2);

				/* move to reading of data */
				pos += Sizes.Rec.Series+4;
				// read number of children
				K3 = (*(uint32_t*)(hdr->AS.Header+pos-4));
				for (k3=0; k3<K3; k3++)	{
#if defined(WITH_TIMESTAMPCHANNEL)

#ifdef NO_BI
#define _BI (BI[hdr->NS-1])
#else
#define _BI (hdr->CHANNEL[hdr->NS-1].bi)
#endif
					gdf_time t = heka2gdftime(*(double*)(hdr->AS.Header+pos+48));		// time of sweep. TODO: this should be taken into account 
					*(int64_t*)(hdr->AS.rawdata + _BI + SPR * 8) = t;
#undef _BI

#endif // WITH_TIMESTAMPCHANNEL
					// read sweep
					char flagSweepSelected = (hdr->AS.SegSel[0]==0 || k1+1==hdr->AS.SegSel[0])
						              && (hdr->AS.SegSel[1]==0 || k2+1==hdr->AS.SegSel[1])
							      && (hdr->AS.SegSel[2]==0 || k3+1==hdr->AS.SegSel[2]);

if (VERBOSE_LEVEL>7) fprintf(stdout,"HEKA+L3 @%i=\t%i/%i %i/%i %i/%i sel=%i\n",(int)(pos+StartOfData),k1,K1,k2,K2,k3,K3,flagSweepSelected);

					pos += Sizes.Rec.Sweep + 4;
					// read number of children
					K4 = (*(uint32_t*)(hdr->AS.Header+pos-4));
					size_t DIV=1;
					for (k4=0; k4<K4; k4++)	{
						if (!flagSweepSelected) {
							pos += Sizes.Rec.Trace+4;
							continue;
						}

						// read trace
						uint16_t gdftyp  = 0;	
						uint32_t ns      = (*(uint32_t*)(hdr->AS.Header+pos+36));
						uint32_t DataPos = (*(uint32_t*)(hdr->AS.Header+pos+40));
						spr              = (*(uint32_t*)(hdr->AS.Header+pos+44));
						double DataScaler= (*(double*)(hdr->AS.Header+pos+72));
						double Toffset   = (*(double*)(hdr->AS.Header+pos+80));		// time offset of 
						uint16_t pdc     = PhysDimCode((char*)(hdr->AS.Header + pos + 96));
						char *physdim    = (char*)(hdr->AS.Header + pos + 96);
						double dT        = (*(double*)(hdr->AS.Header+pos+104));
//						double XStart    = (*(double*)(hdr->AS.Header+pos+112));
//						uint16_t XUnits  = PhysDimCode((char*)(hdr->AS.Header+pos+120));
						double YRange    = (*(double*)(hdr->AS.Header+pos+128));
						double YOffset   = (*(double*)(hdr->AS.Header+pos+136));
//						double Bandwidth = (*(double*)(hdr->AS.Header+pos+144));
						uint16_t AdcChan = (*(uint16_t*)(hdr->AS.Header+pos+222));
/*
						double PhysMin   = (*(double*)(hdr->AS.Header+pos+224));
						double PhysMax   = (*(double*)(hdr->AS.Header+pos+232));
*/
						switch (hdr->AS.Header[pos+70]) {
						case 0: gdftyp = 3;  break;	// int16
						case 1: gdftyp = 5;  break;	// int32
						case 2: gdftyp = 16; break;	// float32
						case 3: gdftyp = 17; break;	// float64
						default: 
							biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Heka/Patchmaster unknown data type is used");
						};

						if (SWAP) {
 							AdcChan  = bswap_16(AdcChan);
 							ns       = bswap_32(ns);
 							DataPos  = bswap_32(DataPos);
							spr      = bswap_32(spr);
							// avoid breaking strict-aliasing rules
							union {
								double f64;
								uint64_t u64;
							} c;	
							c.f64 = dT;      c.u64 = bswap_64(c.u64); dT      = c.f64;
							c.f64 = YRange;  c.u64 = bswap_64(c.u64); YRange  = c.f64;
							c.f64 = YOffset; c.u64 = bswap_64(c.u64); YOffset = c.f64;
/*
							c.f64 = PhysMax; c.u64 = bswap_64(c.u64); PhysMax = c.f64;
							c.f64 = PhysMin; c.u64 = bswap_64(c.u64); PhysMin = c.f64;
*/
							c.f64 = Toffset; c.u64 = bswap_64(c.u64); Toffset = c.f64;
 						}
                                                double Fs  = round(1.0 / dT);
						DIV = round(hdr->SampleRate / Fs);

						char *Label = (char*)(hdr->AS.Header+pos+4);
						for (ns=0; ns < hdr->NS; ns++) {
							if (!strcmp(hdr->CHANNEL[ns].Label, Label)) break;
						}
						CHANNEL_TYPE *hc = hdr->CHANNEL+ns;

if (VERBOSE_LEVEL>7) fprintf(stdout,"HEKA+L4 @%i= #%i,%i,%i/%i %s\t%i/%i %i/%i %i/%i %i/%i DIV=%i,%i,%i\n",(int)(pos+StartOfData),ns,AdcChan,spr,SPR,Label,k1,K1,k2,K2,k3,K3,k4,K4,(int)DIV,gdftyp,hc->GDFTYP);

						if (itx) {
							uint32_t k5;
							double Cal = DataScaler;

							assert(hdr->CHANNEL[ns].Off==0.0);
							double Off = 0.0; 

							fprintf(itx, "\r\nWAVES %s_%i_%i_%i_%i\r\nBEGIN\r\n", WAVENAME,k1+1,k2+1,k3+1,k4+1);
							switch (hc->GDFTYP) {
							case 3:  
								for (k5 = 0; k5 < spr; ++k5)
									fprintf(itx,"% e\n", (double)*(int16_t*)(hdr->AS.Header + DataPos + k5 * 2) * Cal + Off);
								break;
							case 5:
								for (k5 = 0; k5 < spr; ++k5)
									fprintf(itx,"% e\n", (double)*(int32_t*)(hdr->AS.Header + DataPos + k5 * 4) * Cal + Off);
								break;
							case 16: 
								for (k5 = 0; k5 < spr; ++k5)
									fprintf(itx,"% e\n", (double)*(float*)(hdr->AS.Header + DataPos + k5 * 4) * Cal + Off);
								break;
							case 17: 
								for (k5 = 0; k5 < spr; ++k5)
									fprintf(itx,"% e\n", *(double*)(hdr->AS.Header + DataPos + k5 * 8) * Cal + Off);
								break;
							}
							fprintf(itx, "END\r\nX SetScale/P x, %g, %g, \"s\", %s_%i_%i_%i_%i\r\n", Toffset, dT, WAVENAME, k1+1,k2+1,k3+1,k4+1);
							fprintf(itx, "X SetScale y,0,0,\"%s\", %s_%i_%i_%i_%i\n", physdim, WAVENAME, k1+1,k2+1,k3+1,k4+1);
						}	

#ifdef NO_BI
#define _BI (BI[ns])
#else
#define _BI (hc->bi)
#endif
						// no need to check byte order because File.Endian is set and endian conversion is done in sread
						if ((DIV==1) && (gdftyp == hc->GDFTYP))	{
							uint16_t sz = GDFTYP_BITS[hc->GDFTYP]>>3;	
							memcpy(hdr->AS.rawdata + _BI + SPR * sz, hdr->AS.Header + DataPos, spr * sz);
						}
						else if (1) {
							double Cal = DataScaler * PhysDimScale(pdc) / hdr->CHANNEL[ns].Cal;
							assert(Cal==1.0 || hc->GDFTYP > 15); // when scaling changes, target data type is always float/double -> see above
							uint32_t k5,k6;
							switch (gdftyp) {
							case 3: 
								switch (hc->GDFTYP) {
								case 3: 
									for (k5 = 0; k5 < spr; ++k5) {
										int16_t ival = *(int16_t*)(hdr->AS.Header + DataPos + k5 * 2);
										for (k6 = 0; k6 < DIV; ++k6) 
											*(int16_t*)(hdr->AS.rawdata + _BI + (SPR + k5*DIV + k6) * 2) = ival;
									}
									break;
								case 5: 
									for (k5 = 0; k5 < spr; ++k5) {
										int16_t ival = *(int16_t*)(hdr->AS.Header + DataPos + k5 * 2);
										for (k6 = 0; k6 < DIV; ++k6) 
											*(int32_t*)(hdr->AS.rawdata + _BI + (SPR + k5*DIV + k6) * 4) = (int32_t)ival;
									}
									break;
								case 16: 
									for (k5 = 0; k5 < spr; ++k5) {
										int16_t ival = *(int16_t*)(hdr->AS.Header + DataPos + k5 * 2);
										for (k6 = 0; k6 < DIV; ++k6) 
											*(float*)(hdr->AS.rawdata + _BI + (SPR + k5*DIV + k6) * 4) = (float)ival * Cal;
									}
									break;
								case 17: 
									for (k5 = 0; k5 < spr; ++k5) {
										int16_t ival = *(int16_t*)(hdr->AS.Header + DataPos + k5 * 2);
										for (k6 = 0; k6 < DIV; ++k6) 
											*(double*)(hdr->AS.rawdata + _BI + (SPR + k5*DIV + k6) * 8) = (double)ival * Cal;
									}
									break;
								}
								break;
							case 5:
								switch (hc->GDFTYP) {
								case 5: 
									for (k5 = 0; k5 < spr; ++k5) {
										int32_t ival = *(int32_t*)(hdr->AS.Header + DataPos + k5 * 4);
										for (k6 = 0; k6 < DIV; ++k6) 
											*(int32_t*)(hdr->AS.rawdata + _BI + (SPR + k5*DIV + k6) * 4) = ival;
									}
									break;
								case 16: 
									for (k5 = 0; k5 < spr; ++k5) {
										int32_t ival = *(int32_t*)(hdr->AS.Header + DataPos + k5 * 4);
										for (k6 = 0; k6 < DIV; ++k6) 
											*(float*)(hdr->AS.rawdata + _BI + (SPR + k5*DIV + k6) * 4) = (float)ival * Cal;
									}
									break;
								case 17: 
									for (k5 = 0; k5 < spr; ++k5) {
										int32_t ival = *(int32_t*)(hdr->AS.Header + DataPos + k5 * 4);
										for (k6 = 0; k6 < DIV; ++k6) 
											*(double*)(hdr->AS.rawdata + _BI + (SPR + k5*DIV + k6) * 8) = (double)ival * Cal;
									}
									break;
								}
								break;
							case 16:
								switch (hc->GDFTYP) {
								case 16: 
									for (k5 = 0; k5 < spr; ++k5) {
										float ival = *(float*)(hdr->AS.Header + DataPos + k5 * 4);
										for (k6 = 0; k6 < DIV; ++k6) 
											*(float*)(hdr->AS.rawdata + _BI + (SPR + k5*DIV + k6) * 4) = ival * Cal;
									}
									break;
								case 17: 
									for (k5 = 0; k5 < spr; ++k5) {
										float ival = *(float*)(hdr->AS.Header + DataPos + k5 * 4);
										for (k6 = 0; k6 < DIV; ++k6) 
											*(double*)(hdr->AS.rawdata + _BI + (SPR + k5*DIV + k6) * 8) = (double)ival * Cal;
									}
									break;
								}
								break;
							case 17:
								switch (hc->GDFTYP) {
								case 17: 
									for (k5 = 0; k5 < spr; ++k5) {
										double ival = *(double*)(hdr->AS.Header + DataPos + k5 * 8);
										for (k6 = 0; k6 < DIV; ++k6) 
											*(double*)(hdr->AS.rawdata + _BI + (SPR + k5*DIV + k6) * 8) = ival * Cal;
									}
									break;
								}
								break;
							}
						}
#undef _BI
						pos += Sizes.Rec.Trace+4;

					}
					if (flagSweepSelected) SPR += spr * DIV;
				}
			}
		}
#ifdef NO_BI
		if (BI) free(BI);
#endif
		hdr->AS.first  = 0;
		hdr->AS.length = hdr->NRec;
		free(hdr->AS.Header);
		hdr->AS.Header = NULL;

		if (VERBOSE_LEVEL>7) fprintf(stdout,"End of SOPEN_HEKA\n");
	}

	else {
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Heka/Patchmaster format has unsupported version number");
        }
}


#ifdef __cplusplus
}
#endif

