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
#ifdef WITH_MATIO
#include <matio.h>
#endif
#include "../biosig.h"


int sopen_matlab(HDRTYPE* hdr) {
#ifdef WITH_MATIO
        /*
                file hdr->FileName is already opened and hdr->HeadLen bytes are read
                These are available from hdr->AS.Header.

                ToDo: populate hdr
			sanity checks
			memory leaks
        */
	ifclose(hdr);
	//size_t count = hdr->HeadLen;
	// TODO: identify which type/origin of MATLAB

        fprintf(stdout, "Trying to read Matlab data using MATIO v%i.%i.%i\n", MATIO_MAJOR_VERSION, MATIO_MINOR_VERSION, MATIO_RELEASE_LEVEL);

	mat_t *matfile = Mat_Open(hdr->FileName, MAT_ACC_RDONLY);
	matvar_t *EEG=NULL, *pnts=NULL, *nbchan=NULL, *trials=NULL, *srate=NULL, *data=NULL, *chanlocs=NULL, *event=NULL;
	if (matfile != NULL) {
		EEG    = Mat_VarRead(matfile, "EEG" );
		if (EEG != NULL) {
			Mat_VarReadDataAll(matfile, EEG );
			pnts   = Mat_VarGetStructField(EEG, "pnts", MAT_BY_NAME, 0);
			nbchan = Mat_VarGetStructField(EEG, "nbchan", MAT_BY_NAME, 0);
			trials = Mat_VarGetStructField(EEG, "trials", MAT_BY_NAME, 0);
			srate  = Mat_VarGetStructField(EEG, "srate", MAT_BY_NAME, 0);
			data   = Mat_VarGetStructField(EEG, "data", MAT_BY_NAME, 0);
			chanlocs = Mat_VarGetStructField(EEG, "chanlocs", MAT_BY_NAME, 0);
			event    = Mat_VarGetStructField(EEG, "event", MAT_BY_NAME, 0);

			hdr->NS  = *(double*)(nbchan->data);
			hdr->SPR = *(double*)(pnts->data);
			hdr->NRec= *(double*)(trials->data);
			hdr->SampleRate = *(double*)(srate->data);

/* TODO CB
			hdr->NRec 	 = ;
			hdr->SPR  	 = ;
			hdr->T0 	 = 0;        // Unknown;
*/
			uint16_t gdftyp  = 16;	// 16: float; 17: double
			hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
			size_t k;
			for (k=0; k<hdr->NS; k++) {
				CHANNEL_TYPE *hc = hdr->CHANNEL+k;
				sprintf(hc->Label,"#%2d",k+1);
				hc->SPR = hdr->SPR;
/* TODO CB
				hc->GDFTYP = gdftyp;
				hc->Transducer[0] = '\0';
				hc->LowPass	= ;
				hc->HighPass = ;
				hc->Notch	= ;  // unknown
				hc->PhysMax	= ;
				hc->DigMax	= ;
				hc->PhysMin	= ;
				hc->DigMin	= ;
				hc->Cal	 	= 1.0;
				hc->Off	 	= 0.0;
				hc->OnOff    	= 1;
				hc->PhysDimCode = 4275; // uV
				hc->LeadIdCode  = 0;
				hc->bi      	= k*GDFTYP_BITS[gdftyp]>>3;	// TODO AS
*/
			}

			size_t sz = hdr->NS*hdr->SPR*hdr->NRec*GDFTYP_BITS[gdftyp]>>3;
			hdr->AS.rawdata = realloc(hdr->AS.rawdata, sz);
/* TODO CB
			memcpy(hdr->AS.rawdata,...,sz);
*/
			hdr->EVENT.N = 0; 	// TODO CB
			hdr->EVENT.POS = (uint32_t*) realloc(hdr->EVENT.POS, hdr->EVENT.N*sizeof(*hdr->EVENT.POS));
			hdr->EVENT.TYP = (uint16_t*) realloc(hdr->EVENT.TYP, hdr->EVENT.N*sizeof(*hdr->EVENT.TYP));
			hdr->EVENT.DUR = (uint32_t*) realloc(hdr->EVENT.DUR, hdr->EVENT.N*sizeof(*hdr->EVENT.DUR));
			hdr->EVENT.CHN = (uint16_t*) realloc(hdr->EVENT.CHN, hdr->EVENT.N*sizeof(*hdr->EVENT.CHN));
			for (k=0; k<hdr->EVENT.N; k++) {
/* TODO CB
				hdr->EVENT.TYP[k] =
				FreeTextEvent(hdr, k, annotation)
				hdr->EVENT.POS[k] =
				hdr->EVENT.CHN[k] = 0;
				hdr->EVENT.DUR[k] = 0;
*/
			}

		hdr->AS.bpb = hdr->NS*2;
		hdr->FLAG.OVERFLOWDETECTION = 0; 	// BKR does not support automated overflow and saturation detection


			Mat_VarPrint(pnts,   1);
			Mat_VarPrint(nbchan, 1);
			Mat_VarPrint(trials, 1);
			Mat_VarPrint(srate,  1);
			//Mat_VarPrint(data,   1);
			//Mat_VarPrint(chanlocs, 1);
			//Mat_VarPrint(event,  1);


			Mat_VarFree(EEG);
		}

		Mat_Close(matfile);
	}

	biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Error reading MATLAB file");
	return(-1);

#else	// WITH_MATIO
	biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "SOPEN(MATIO): - matlab format not supported - libbiosig need to be recompiled with libmatio support.");
	ifclose(hdr);
	return(-1);
#endif	// WITH_MATIO
}

