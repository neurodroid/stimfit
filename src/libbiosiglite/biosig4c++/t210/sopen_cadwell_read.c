/*

    Copyright (C) 2021 Alois Schloegl <alois.schloegl@gmail.com>

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

#include "../biosig.h"

/******************************************************************************
	read CADWELL file formats (EAS, EZ3, ARC)
 ******************************************************************************/
void sopen_cadwell_read(HDRTYPE* hdr) {

	if (VERBOSE_LEVEL > 8) fprintf(stdout,"%s (line %d) %s(hdr)\n", __FILE__, __LINE__, __func__);

	if (hdr->TYPE==EAS) {
		/* 12 bit ADC ?
		   200 Hz
		   77*800 samples
		   EEGData section has a periodicity of 202*2 (404 bytes)
			800samples*16channels*2byte=25600  = 0x6400)
		*/
		hdr->NS  = 16;  // so far all example files had 16 channels
		hdr->SPR  = 1;
		hdr->NRec = 0;
		hdr->SampleRate = 250;
		unsigned lengthHeader0 = leu32p(hdr->AS.Header + 0x30);
		unsigned lengthHeader1 = leu32p(hdr->AS.Header + 0x34);	// this is varying amoung data sets - meaning unknown
		assert(lengthHeader0==0x0400);

		for (int k=0; k*0x20 < lengthHeader1; k++) {
			char *sectName =       hdr->AS.Header + 0x6c + k*0x20;
			size_t sectPos= leu32p(hdr->AS.Header + 0x7c + k*0x20);
			size_t sectN1 = leu32p(hdr->AS.Header + 0x80 + k*0x20);
			size_t sectN2 = leu32p(hdr->AS.Header + 0x84 + k*0x20);
			size_t sectN3 = leu32p(hdr->AS.Header + 0x88 + k*0x20);

			if (!sectPos
			 || memcmp("SctHdr\0\0", hdr->AS.Header+sectPos, 8)
			 || memcmp(hdr->AS.Header+sectPos+8, sectName, 16))
			{
				if (VERBOSE_LEVEL > 8) fprintf(stdout,"%s (line %d): break loop (0x%x %s)\n",__FILE__,__LINE__, sectPos, sectName);
				break;
			}

			uint64_t curSec, nextSec;
			int flag=1;
			do {
				curSec  = leu64p(hdr->AS.Header + sectPos + 24);
				nextSec = leu64p(hdr->AS.Header + sectPos + 32);

		if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d): 0x%08x %s  0x%08lx 0x%08lx \n",__FILE__,__LINE__, sectPos, sectName, curSec, nextSec);
				if (flag && !strcmp(sectName,"EEGData")) {
		if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d): 0x%08x %s  0x%08lx 0x%08lx \n",__FILE__,__LINE__, sectPos, sectName, curSec, nextSec);
					FILE *fid2=fopen("tmp.bin","w");
					fwrite(hdr->AS.Header + curSec+120, 1, nextSec-curSec-120,fid2);
					fclose(fid2);
					FILE *fid=fopen("tmp.asc","w");
					for (size_t k0=curSec+8*15; k0 < nextSec; k0+=2) {
						fprintf(fid,"%d\n",bei16p(hdr->AS.Header + curSec + k0+1));
					}
					fclose(fid);
					flag = 0;
				}
				sectPos = nextSec;
			} while (nextSec != (size_t)-1L);
		}
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format EAS(Cadwell): unsupported ");
	}

	else if (hdr->TYPE==EZ3) {
		hdr->VERSION = strtod((char*)hdr->AS.Header+21, NULL);
		// 16 bit ADC ?
		// 250 Hz ?

		if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d)0x%8x\n",__FILE__,__LINE__,hdr->HeadLen);

		uint32_t posH1  = leu32p(hdr->AS.Header + 0x10);
		if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d)0x%8x\n",__FILE__,__LINE__,posH1);

		uint32_t posH1b = leu32p(hdr->AS.Header + 0x20);
		if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d)0x%8x\n",__FILE__,__LINE__,posH1b);

		uint32_t posH2  = leu32p(hdr->AS.Header + 0x38);
		if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d)0x%8x\n",__FILE__,__LINE__,posH2);

		uint32_t posH2b = leu32p(hdr->AS.Header + posH1 + 0x38);

		if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d) 0x%08x 0x%08x 0x%08x 0x%08x \n",__FILE__,__LINE__,posH1,posH1b,posH2,posH2b);

		assert(posH1==posH1b);
		assert(posH2==posH2b);

		if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d)\n",__FILE__,__LINE__);

		// start date/time
		{
			char *tmp = hdr->AS.Header + 0x5c;
			struct tm t;
			t.tm_year = strtol(tmp,&tmp,10)-1900;
			t.tm_mon = strtol(tmp+1,&tmp,10)-1;
			t.tm_mday = strtol(tmp+1,&tmp,10);
			t.tm_hour = strtol(tmp+1,&tmp,10);
			t.tm_min = strtol(tmp+1,&tmp,10);
			t.tm_sec = strtol(tmp+1,&tmp,10);
			hdr->T0 = tm_time2gdf_time(&t);
		}

		uint32_t pos0 = posH1+0x40;
		for (size_t k = posH1+0x40; k < posH2; k += 0x30) {
			char *tmp    = hdr->AS.Header + k + 1;
			uint32_t pos = leu32p(hdr->AS.Header + k + 0x28);
			if (tmp[0]=='\0') break;

			if (VERBOSE_LEVEL > 7) fprintf(stdout,"%s (line %d): Label=<%s> %10d(0x%08x) [sz=%d]\n",__FILE__,__LINE__, tmp, pos, pos, pos-pos0);

			if (pos<pos0) break;

			uint32_t V16=leu32p(hdr->AS.Header+pos+16);	// related to size current block, seems to match next pos-pos0-64, V16+pos+64 is next pos.
			uint32_t V20=leu32p(hdr->AS.Header+pos+20);	// always(?) 0
			uint32_t V24=leu32p(hdr->AS.Header+pos+24);	// related to block type, or position, seems to be a decimal number, 0,1,27000,151000,329000
			uint32_t V32=leu32p(hdr->AS.Header+pos+32);	// import for EEG001 blocks ?
			char*    next    =  hdr->AS.Header+pos+40;	// string, refers to name of next block ??
			uint32_t nextpos =  leu32p(hdr->AS.Header + k + 0x28 + 0x30);
			uint32_t V64=leu32p(hdr->AS.Header+pos+64);	// maybe some size information of a subblock ?
			uint32_t V68=leu32p(hdr->AS.Header+pos+68);	// ?
			uint32_t V72=leu32p(hdr->AS.Header+pos+72);	// ?
			uint32_t V76=leu32p(hdr->AS.Header+pos+76);	// ? seems to refer to next block
			uint32_t V80=leu32p(hdr->AS.Header+pos+80);	// ? seems to refer to next block
			uint32_t V324=leu32p(hdr->AS.Header+pos+324);	// seems to be the same as V24, at least when V64 is large enough

			uint32_t V236=leu32p(hdr->AS.Header+pos+236);	// seems to be important for EEG001 blocks ?
			uint32_t V237=leu32p(hdr->AS.Header+pos+237);	// seems to be important for EEG001 blocks ?
			uint32_t V244=leu32p(hdr->AS.Header+pos+244);	// seems to be important for EEG001 blocks ?
			uint32_t V245=leu32p(hdr->AS.Header+pos+245);	// seems to be important for EEG001 blocks ?
			uint32_t V260=leu32p(hdr->AS.Header+pos+260);	// seems to be important for EEG001 blocks ?
			uint32_t V261=leu32p(hdr->AS.Header+pos+261);	// seems to be important for EEG001 blocks ?
			uint32_t V332=leu32p(hdr->AS.Header+pos+332);	// seems to be important for EEG001 blocks ?

			char TMP[1024];
			strcpy(TMP, "cadwell.debug.");
			strcat(TMP, next);
			FILE* fid = fopen(TMP,"a"); fprintf(fid,"%d\n",pos); fclose(fid);
			pos0=pos;

			if (VERBOSE_LEVEL > 7) {
				if (memcmp(hdr->AS.Header+pos,"EasyDCWYAAA\0@\0\0\0",16) || leu32p(hdr->AS.Header+pos+20) )
					fprintf(stdout,"%s (line %d): unexpected header info <%s><%s>0x08x\n",__FILE__,__LINE__, hdr->AS.Header+pos, hdr->AS.Header+pos+12, V20);

				if (nextpos != V16+pos+64)
					fprintf(stdout,"%s (line %d): unexpected header info %d!=%d\n",__FILE__,__LINE__, nextpos, V16+pos+64);

				fprintf(stdout,"%s (line %d): unknown header info (block type, position ?) %8d(0x%08x) <%s><%s>\n",__FILE__,__LINE__, V24, V24, tmp, next);

				if (( V64 > 328) && (V24 != V324))
					fprintf(stdout,"%s (line %d): *(@+64) %ld(0x%16lx) *(@+324) %d(0x%08x)\n",__FILE__,__LINE__, V64, V64, V324, V324);
			}

#if 0
			if (!strncmp(tmp,"EEG001",7)) {
				// it seems all EEG001 data blocks do contain these "magic" values
				if (V64 != 0x02f3504dL)
					fprintf(stdout,"%s (line %d): [%s] *(@+64) %20d(0x%16x)\n",__FILE__,__LINE__, tmp, V64, V64);
				if (V68 != 0x4da8564a)
					fprintf(stdout,"%s (line %d): [%s] *(@+68) %20d(0x%16x)\n",__FILE__,__LINE__, tmp, V68, V68);
				if (V72 != 0x03b)
					fprintf(stdout,"%s (line %d): [%s] *(@+72) %20d(0x%16x)\n",__FILE__,__LINE__, tmp, V72, V72);
			}
			else if (!strncmp(next,"EVENT001",9)) {
				// it seems all EEG001 data blocks do contain these "magic" values
				if (V64 != 0x100)
					fprintf(stdout,"%s (line %d): [%s] *(@+64) %20d(0x%16x)\n",__FILE__,__LINE__, tmp, V64, V64);
				if (V68 != 0xffffff00)
					fprintf(stdout,"%s (line %d): [%s] *(@+68) %20d(0x%16x)\n",__FILE__,__LINE__, tmp, V68, V68);
				if (V72 != 0x1ff)
					fprintf(stdout,"%s (line %d): [%s] *(@+72) %20d(0x%16x)\n",__FILE__,__LINE__, tmp, V72, V72);
			}
			else {
				// it seems all EEG001 data blocks do contain these "magic" values
				if (V64 != 0x100)
					fprintf(stdout,"%s (line %d): [%s] *(@+64) %20d(0x%16x)\n",__FILE__,__LINE__, tmp, V64, V64);
				if (V68 != 0xffffff00)
					fprintf(stdout,"%s (line %d): [%s] *(@+68) %20d(0x%16x)\n",__FILE__,__LINE__, tmp, V68, V68);
				if (V72 != 0x01ff)
					fprintf(stdout,"%s (line %d): [%s] *(@+72) %20d(0x%16x)\n",__FILE__,__LINE__, tmp, V72, V72);
			}
#endif

			if (!strncmp(next, "EEG001",7)) {
				/* positions with differences in the range of [0..371]
					    16    17    24    25    32   237   238   245   261   262   324   325   332
				*/
				if (V76 != 0x2addcb81L)
					fprintf(stdout,"%s (line %d): [%s] *(@+76) %20d(0x%16x)\n",__FILE__,__LINE__, next, V76, V76);
				if (V80 != 0x02)
					fprintf(stdout,"%s (line %d): [%s] *(@+80) %20d(0x%16x)\n",__FILE__,__LINE__, next, V80, V80);

				if (V32 != 1000)
					fprintf(stdout,"%s (line %d): [%s] *(@+32) %20d(0x%16x)\n",__FILE__,__LINE__, next, V32, V32);

				if (V237 != V24)
					fprintf(stdout,"%s (line %d): [%s] *(@+237) %20d(0x%16x)\n",__FILE__,__LINE__, next, V237, V237);

				if (V245 != 1000)
					fprintf(stdout,"%s (line %d): [%s] *(@+245) %20d(0x%16x)\n",__FILE__,__LINE__, next, V245, V245);

				if (V261 != V24)
					fprintf(stdout,"%s (line %d): [%s] *(@+261) %20d(0x%16x)\n",__FILE__,__LINE__, next, V261, V261);

				if (V324 != V24)
					fprintf(stdout,"%s (line %d): [%s(?)] *(@+324) %20d(0x%16x)\n",__FILE__,__LINE__, next, V324, V324);

				if (V332 != 250)	// sampling rate, SPR ?
					fprintf(stdout,"%s (line %d): [%s/SPR(?)] *(@+332) %20d(0x%16x)\n",__FILE__,__LINE__, next, V332, V332);

				if (V332*4 != V32 || V32 != V245)
					fprintf(stdout,"%s (line %d): [%s(?)]  V32,V245,4xV332 do not match %d != %d != %d\n",__FILE__,__LINE__, next, V32, V245, V332);

/*
				start at 372-872: block1
				block2 615-1114:
				block2 1433-
				next interval seem seem to have variable lengths
				(signed) int16
				blocklengths 308 samples, 250 real samples + ?
*/
				strcpy(TMP, "cadwell.debug.eeg.txt");
				FILE* fid = fopen(TMP,"a");
				for (size_t kk=0; kk<V332; kk++) {
					int16_t v1=lei16p(hdr->AS.Header+pos+371+kk*2);
					int16_t v2=lei16p(hdr->AS.Header+pos+371+(V332*2+63*2)+kk*2);
					int16_t v3=lei16p(hdr->AS.Header+pos+371+(V332*2+63*2)*2+kk*2);
					fprintf(fid, "%d\t%d\t%d\n", v1,v2,v3);
				}
				fclose(fid);
			}
			else if (!strncmp(next,"EVENT001",9)) {
				/* positions with differences in the range of [0..2051]
				   18    537    538    539    540    541    542    543    544    545    546    547    548    549    550    551    552   1217   1402
				   after 2052, almost all bytes a diffent
				*/
				if (V76 != 0)
					fprintf(stdout,"%s (line %d): EVENT *(@+76) %20d(0x%16x)\n",__FILE__,__LINE__, V76, V76);
				if (V80 != 0x20c00)
					fprintf(stdout,"%s (line %d): EVENT *(@+80) %20d(0x%16x)\n",__FILE__,__LINE__, V80, V80);
				for (int kk=536; kk<556; kk+=4) {
					uint32_t v = leu32p(hdr->AS.Header+pos+kk);
					fprintf(stdout,"%s (line %d): [%s] *(@+%10d)\t%d(0x%16x)\n",__FILE__,__LINE__, next, kk, v, v);
				}

				if (hdr->AS.Header[pos+165]!='E' || hdr->AS.Header[pos+202]!='t' ) {
					if (VERBOSE_LEVEL > 7)
						fprintf(stdout,"%s (line %d): unexpected info in EVENT header <%c><%c>\n",__FILE__,__LINE__, hdr->AS.Header[pos+165],hdr->AS.Header[pos+202]);
				}
			}
			else {
				if (V76 != 0)
					fprintf(stdout,"%s (line %d): [%s]->[%s] *(@+76) %20d(0x%16x)\n",__FILE__,__LINE__, tmp,next, V76, V76);
				if (V80 != 0x20c00)
					fprintf(stdout,"%s (line %d): [%s]->[%s] *(@+80) %20d(0x%16x)\n",__FILE__,__LINE__, tmp,next, V80, V80);
			}
		}
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format EZ3(Cadwell): unsupported ");

	}
	else if (hdr->TYPE==ARC) {
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format ARC(Cadwell): unsupported ");
	}
}

