/*
    sandbox is used for development and under constraction work
    The functions here are either under construction or experimental.
    The functions will be either fixed, then they are moved to another place;
    or the functions are discarded. Do not rely on the interface in this function

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

#include "../biosig-dev.h"
#include "../biosig.h"

// these functions are stubs

#ifdef WITH_DCMTK
#undef WITH_DICOM
#undef WITH_GDCM

EXTERN_C int sopen_dicom_read(HDRTYPE* hdr) {
	fprintf(stdout,"DCMTK is used to read dicom files.\n");
}

#endif

#ifdef HAVE_HDF
#include <hdf5.h>
#endif
#ifdef HAVE_MATIO
#include <matio.h>
#endif


#ifdef WITH_GDCM
#undef WITH_DICOM

#include "gdcmReader.h"
//#include "gdcmImageReader.h"
//#include "gdcmWriter.h"
#include "gdcmDataSet.h"
#include "gdcmAttribute.h"

//#include "gdcmCommon.h"
//#include "gdcmPreamble.h"
#include "gdcmFile.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmWaveform.h"

/*
 This is the list from gdcmconv.cxx
#include "gdcmReader.h"
#include "gdcmFileDerivation.h"
#include "gdcmAnonymizer.h"
#include "gdcmVersion.h"
#include "gdcmPixmapReader.h"
#include "gdcmPixmapWriter.h"
#include "gdcmWriter.h"
#include "gdcmSystem.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmDataSet.h"
#include "gdcmIconImageGenerator.h"
#include "gdcmAttribute.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmUIDGenerator.h"
#include "gdcmImage.h"
#include "gdcmImageChangeTransferSyntax.h"
#include "gdcmImageApplyLookupTable.h"
#include "gdcmImageFragmentSplitter.h"
#include "gdcmImageChangePlanarConfiguration.h"
#include "gdcmImageChangePhotometricInterpretation.h"
#include "gdcmFileExplicitFilter.h"
#include "gdcmJPEG2000Codec.h"
#include "gdcmJPEGCodec.h"
#include "gdcmJPEGLSCodec.h"
#include "gdcmSequenceOfFragments.h"
*/


EXTERN_C int sopen_dicom_read(HDRTYPE* hdr) {

	fprintf(stdout,"GDCM is used to read dicom files.\n");

	gdcm::Reader reader;
        const gdcm::DataElement *de;
	reader.SetFileName( hdr->FileName );
	if ( !reader.Read() ) {
		fprintf(stdout,"%s (line %i) \n",__FILE__,__LINE__);
		return 1;
	}

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i)\n",__FILE__,__LINE__);
	gdcm::File &file = reader.GetFile();
	gdcm::FileMetaInformation &header = file.GetHeader();

	if ( header.FindDataElement( gdcm::Tag(0x0002, 0x0013 ) ) )
		const gdcm::DataElement &de = header.GetDataElement( gdcm::Tag(0x0002, 0x0013) );

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i)\n",__FILE__,__LINE__);
	gdcm::DataSet &ds = file.GetDataSet();

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i)\n",__FILE__,__LINE__);
	if ( header.FindDataElement( gdcm::Tag(0x0002, 0x0010 ) ) )
		de = &header.GetDataElement( gdcm::Tag(0x0002, 0x0010) );

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): attr <0x0002,0x0010> len=%i\n",__FILE__,__LINE__,de->GetByteValue() );

	if (0) {
		gdcm::Attribute<0x28,0x100> at;
		at.SetFromDataElement( ds.GetDataElement( at.GetTag() ) );
		if( at.GetValue() != 8 ) {

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): attr <0x0002,0x0010> GetValue\n",__FILE__,__LINE__ );

			return 1;
		}
		at.SetValue( 32 );
		ds.Replace( at.GetAsDataElement() );
	}
	{


if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): attr <0x0008,0x002a>\n",__FILE__,__LINE__);
		gdcm::Attribute<0x0008,0x002a> at;
 		ds.GetDataElement( at.GetTag() );
		at.SetFromDataElement( ds.GetDataElement( at.GetTag() ) );
if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): attr <0x0008,0x002a>\n",__FILE__,__LINE__);

//		fprintf(stdout,"DCM: [0008,002a]: %i %p\n", at.GetNumberOfValues(), at.GetValue());
	}

	{

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): attr <0x0008,0x0023>\n",__FILE__,__LINE__);
		gdcm::Attribute<0x0008,0x0023> at;
 		ds.GetDataElement( at.GetTag() );
		at.SetFromDataElement( ds.GetDataElement( at.GetTag() ) );
if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): attr <0x0008,0x0023>\n",__FILE__,__LINE__);
	}
	if (1) {
if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): attr <0x003a,0x0005>\n",__FILE__,__LINE__);
		gdcm::Attribute<0x003a,0x0005> NumberOfWaveformChannels;
if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): attr <0x003a,0x0005> %i\n",__FILE__,__LINE__, NumberOfWaveformChannels.GetValue());
// 		ds.GetDataElement( NumberOfWaveformChannels.GetTag() );
//		NumberOfWaveformChannels.SetFromDataElement( ds.GetDataElement( NumberOfWaveformChannels.GetTag() ) );
if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): attr <0x003a,0x0005> %i\n",__FILE__,__LINE__, NumberOfWaveformChannels.GetValue());
	}
	{
if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): attr <0x003a,0x0010>\n",__FILE__,__LINE__);
		gdcm::Attribute<0x003a,0x0010> NumberOfWaveformSamples;
if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): attr <0x003a,0x0010> %i\n",__FILE__,__LINE__, NumberOfWaveformSamples.GetValue());

//		fprintf(stdout,"DCM: [0008,0023]: %i %p\n",at.GetNumberOfValues(), at.GetValue());
	}

		gdcm::Attribute<0x003a,0x001a> SamplingFrequency;
if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i): attr <0x003a,0x001a> %f\n",__FILE__,__LINE__, SamplingFrequency.GetValue());

	return 0;
}
#endif


#ifdef __cplusplus
extern "C" {
#endif

#ifdef WITH_HDF
int sopen_hdf5(HDRTYPE* hdr) {
        /*
                file hdr->FileName is already opened and hdr->HeadLen bytes are read
                These are available from hdr->AS.Header.

                ToDo: populate hdr
        */
	size_t count = hdr->HeadLen;
        fprintf(stdout,"Trying to read HDF data using \"%s\"\n",H5_VERS_INFO);

	ifclose(hdr);

        return(-1);
}
#endif

#ifdef HAVE_MATIO
int sopen_matlab(HDRTYPE* hdr) {
        /*
                file hdr->FileName is already opened and hdr->HeadLen bytes are read
                These are available from hdr->AS.Header.

                ToDo: populate hdr
			sanity checks
			memory leaks
        */
	ifclose(hdr);
	//size_t count = hdr->HeadLen;

        fprintf(stdout, "Trying to read Matlab data using MATIO v%i.%i.%i\n", MATIO_MAJOR_VERSION, MATIO_MINOR_VERSION, MATIO_RELEASE_LEVEL);

	mat_t *matfile = Mat_Open(hdr->FileName, MAT_ACC_RDONLY);
	matvar_t *EEG=NULL, *pnts=NULL, *nbchan=NULL, *trials=NULL, *srate=NULL, *data=NULL, *chanlocs=NULL, *event=NULL;
	if (matfile != NULL) {
		EEG    = Mat_VarRead(matfile, "EEG" );
		if (EEG != NULL) {
			Mat_VarReadDataAll(matfile, EEG );
			pnts   = Mat_VarGetStructField(EEG, "pnts", BY_NAME, 0);
			nbchan = Mat_VarGetStructField(EEG, "nbchan", BY_NAME, 0);
			trials = Mat_VarGetStructField(EEG, "trials", BY_NAME, 0);
			srate  = Mat_VarGetStructField(EEG, "srate", BY_NAME, 0);
			data   = Mat_VarGetStructField(EEG, "data", BY_NAME, 0);
			chanlocs = Mat_VarGetStructField(EEG, "chanlocs", BY_NAME, 0);
			event    = Mat_VarGetStructField(EEG, "event", BY_NAME, 0);

			hdr->NS  = *(double*)(nbchan->data);
			hdr->SPR = *(double*)(pnts->data);
			hdr->NRec= *(double*)(trials->data);
			hdr->SampleRate = *(double*)(srate->data);

/* TODO CB
			hdr->NRec 	 = ;
			hdr->SPR  	 = ;
			hdr->T0 	 = 0;        // Unknown;
			uint16_t gdftyp  = ;	16: float; 17: double
*/
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

        return (0);
}
#endif



int sopen_fiff_read(HDRTYPE* hdr) {
	/* TODO: implement FIFF support
	        define all fields in hdr->....
		currently only the first hdr->HeadLen bytes are stored in
		hdr->AS.Header, all other fields still need to be defined.
	*/
	size_t k;

	/* define basic header */
	hdr->NS = 0;
	// hdr->.... fill in all

	/* define channel headers */
	hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
	for (k = 0; k < hdr->NS; k++) {
		CHANNEL_TYPE *hc = hdr->CHANNEL + k;
	}

	/* define event table */
	hdr->EVENT.N = 0;
	//reallocEventTable(hdr, 0);

	/* report status header and return */
	hdr2ascii(hdr,stdout,4);
	biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "FIFF support not completed");
	return 0;
}


int sopen_unipro_read(HDRTYPE* hdr) {
		hdr->FILE.LittleEndian = (__BYTE_ORDER == __LITTLE_ENDIAN);
		char *Header1 = (char*)hdr->AS.Header;
		struct tm t0;
		char tmp[5];
		memset(tmp,0,5);
		strncpy(tmp,Header1+0x9c,2);
		t0.tm_mon = atoi(tmp)-1;
		strncpy(tmp,Header1+0x9e,2);
		t0.tm_mday = atoi(tmp);
		strncpy(tmp,Header1+0xa1,2);
		t0.tm_hour = atoi(tmp);
		strncpy(tmp,Header1+0xa3,2);
		t0.tm_min = atoi(tmp);
		strncpy(tmp,Header1+0xa5,2);
		t0.tm_sec = atoi(tmp);
		strncpy(tmp,Header1+0x98,4);
		t0.tm_year = atoi(tmp)-1900;
		hdr->T0 = tm_time2gdf_time(&t0);

		memset(tmp,0,5);
		strncpy(tmp,Header1+0x85,2);
		t0.tm_mday = atoi(tmp);
		strncpy(tmp,Header1+0x83,2);
		t0.tm_mon = atoi(tmp)-1;
		strncpy(tmp,Header1+0x7f,4);
		t0.tm_year = atoi(tmp)-1900;
		hdr->Patient.Birthday = tm_time2gdf_time(&t0);

		// filesize = leu32p(hdr->AS.Header + 0x24);
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "UNIPRO not supported");
		return(0);
}


#ifdef WITH_DICOM
int sopen_dicom_read(HDRTYPE* hdr) {

		fprintf(stdout,"home-made parser is used to read dicom files.\n");

		char FLAG_implicite_VR = 0;
		int EndOfGroup2=-1;

		if (hdr->HeadLen<132) {
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, 132);
		    	hdr->HeadLen += ifread(hdr->AS.Header+hdr->HeadLen, 1, 132-hdr->HeadLen, hdr);
		}
		size_t count = hdr->HeadLen;
		size_t pos = 128;
		while (!hdr->AS.Header[pos] && (pos<128)) pos++;
		if ((pos==128) && !memcmp(hdr->AS.Header+128,"DICM",4)) {
//			FLAG_implicite_VR = 0;
			pos = 132;
		}
		else
			pos = 0;

		size_t bufsiz = 16384;
		while (!ifeof(hdr)) {
			hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header, count+bufsiz+1);
		    	count += ifread(hdr->AS.Header+count, 1, bufsiz, hdr);
		    	bufsiz *= 2;
		}
	    	ifclose(hdr);
	    	hdr->AS.Header[count] = 0;

		uint16_t nextTag[2];

		struct tm T0;
		char flag_t0=0;
		char flag_ignored;
		uint32_t Tag;
		uint32_t Len;
		nextTag[0] = *(uint16_t*)(hdr->AS.Header+pos);
		nextTag[1] = *(uint16_t*)(hdr->AS.Header+pos+2);
		while (pos < count) {

			if ((__BYTE_ORDER == __BIG_ENDIAN) ^ !hdr->FILE.LittleEndian) {
				// swapping required
				Tag  = (((uint32_t)bswap_16(nextTag[0])) << 16) + bswap_16(nextTag[1]);
				pos += 4;
				if (FLAG_implicite_VR) {
					Len = bswap_32(*(uint32_t*)(hdr->AS.Header+pos));
					pos += 4;
				}
				else {
					// explicite_VR
					if (pos+4 > count) break;

					if (memcmp(hdr->AS.Header+pos,"OB",2)
					 && memcmp(hdr->AS.Header+pos,"OW",2)
					 && memcmp(hdr->AS.Header+pos,"OF",2)
					 && memcmp(hdr->AS.Header+pos,"SQ",2)
					 && memcmp(hdr->AS.Header+pos,"UT",2)
					 && memcmp(hdr->AS.Header+pos,"UN",2) ) {
						Len = bswap_16(*(uint16_t*)(hdr->AS.Header+pos+2));
						pos += 4;
					}
					else {
						Len = bswap_32(*(uint32_t*)(hdr->AS.Header+pos+4));
						pos += 8;
					}
				}
			}
			else {
				// no swapping
				Tag  = (((uint32_t)nextTag[0]) << 16) + nextTag[1];
				pos += 4;
				if (FLAG_implicite_VR) {
					Len = *(uint32_t*)(hdr->AS.Header+pos);
					pos += 4;
				}
				else {
					// explicite_VR
					if (pos+4 > count) break;

					if (memcmp(hdr->AS.Header+pos,"OB",2)
					 && memcmp(hdr->AS.Header+pos,"OW",2)
					 && memcmp(hdr->AS.Header+pos,"OF",2)
					 && memcmp(hdr->AS.Header+pos,"SQ",2)
					 && memcmp(hdr->AS.Header+pos,"UT",2)
					 && memcmp(hdr->AS.Header+pos,"UN",2) ) {
						Len = *(uint16_t*)(hdr->AS.Header+pos+2);
						pos += 4;
					}
					else {
						Len = *(uint32_t*)(hdr->AS.Header+pos+4);
						pos += 8;
					}
				}
			}

			/*
			    backup next tag, this allows setting of terminating 0
			*/
			if (pos+Len < count) {
				nextTag[0] = *(uint16_t*)(hdr->AS.Header+pos+Len);
				nextTag[1] = *(uint16_t*)(hdr->AS.Header+pos+Len+2);
				hdr->AS.Header[pos+Len] = 0;
	    		}


			flag_ignored = 0;
			if (VERBOSE_LEVEL>8)
				fprintf(stdout,"        %6x:   (%04x,%04x) %8d\t%s\n",pos,Tag>>16,Tag&0x0ffff,Len,(char*)hdr->AS.Header+pos);

			switch (Tag) {


			/* elements of group 0x0002 use always
				Explicite VR little Endian encoding
			*/
			case 0x00020000: {
				int c = 0;
				if (!memcmp(hdr->AS.Header+pos-4,"UL",2))
					c = leu32p(hdr->AS.Header+pos);
				else if (!memcmp(hdr->AS.Header+pos-4,"SL",2))
					c = lei32p(hdr->AS.Header+pos);
				else if (!memcmp(hdr->AS.Header+pos-4,"US",2))
					c = leu16p(hdr->AS.Header+pos);
				else if (!memcmp(hdr->AS.Header+pos-4,"SS",2))
					c = lei16p(hdr->AS.Header+pos);
				else  {

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i) %i<%i %i>\n",__FILE__,__LINE__,pos,hdr->AS.Header[pos-8],hdr->AS.Header[pos-7]);
					biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "DICOM (0002,0000): unsupported");
				}
				EndOfGroup2 = c + pos;
				break;
				}
			case 0x00020001:
				break;

			case 0x00020002: {
				hdr->NS = 1;
				char *t = (char*)hdr->AS.Header+pos;
				while (isspace(*t)) t++;	// deblank
				char *ct[] = {	"1.2.840.10008.5.1.4.1.1.9.1.1",
						"1.2.840.10008.5.1.4.1.1.9.1.2",
						"1.2.840.10008.5.1.4.1.1.9.1.3",
						"1.2.840.10008.5.1.4.1.1.9.2.1",
						"1.2.840.10008.5.1.4.1.1.9.3.1",
						"1.2.840.10008.5.1.4.1.1.9.4.1"
						};
				if (!strcmp(t,*ct)) hdr->NS = 12;
				break;
				}

			case 0x00020003:
				break;

			case 0x00020010: {
				char *t = (char*)hdr->AS.Header+pos;
				while (isspace(*t)) t++;	// deblank
				char *ct[] = {	"1.2.840.10008.1.2",
						"1.2.840.10008.1.2.1",
						"1.2.840.10008.1.2.1.99",
						"1.2.840.10008.1.2.2"
						};

				if      (!strcmp(t,*ct))   FLAG_implicite_VR = 1;
				else if (!strcmp(t,*ct+1)) FLAG_implicite_VR = 0;
				else if (!strcmp(t,*ct+3)) FLAG_implicite_VR = 1;
				break;
				}

			case 0x00080020:  // StudyDate
			case 0x00080023:  // ContentDate
				{
				hdr->AS.Header[pos+8]=0;
				T0.tm_mday = atoi((char*)hdr->AS.Header+pos+6);
				hdr->AS.Header[pos+6]=0;
				T0.tm_mon = atoi((char*)hdr->AS.Header+pos+4)-1;
				hdr->AS.Header[pos+4]=0;
				T0.tm_year = atoi((char*)hdr->AS.Header+pos)-1900;
				flag_t0 |= 1;
				break;
				}
			case 0x0008002a:  // AcquisitionDateTime
				{
				struct tm t0;
				hdr->AS.Header[pos+14]=0;
				t0.tm_sec = atoi((char*)hdr->AS.Header+pos+12);
				hdr->AS.Header[pos+12]=0;
				t0.tm_min = atoi((char*)hdr->AS.Header+pos+10);
				hdr->AS.Header[pos+10]=0;
				t0.tm_hour = atoi((char*)hdr->AS.Header+pos+8);
				hdr->AS.Header[pos+8]=0;
				t0.tm_mday = atoi((char*)hdr->AS.Header+pos+6);
				hdr->AS.Header[pos+6]=0;
				t0.tm_mon = atoi((char*)hdr->AS.Header+pos+4)-1;
				hdr->AS.Header[pos+4]=0;
				t0.tm_year = atoi((char*)hdr->AS.Header+pos)-1900;

				hdr->T0 = tm_time2gdf_time(&t0);
				break;
				}
			case 0x00080030:  // StudyTime
			case 0x00080033:  // ContentTime
				{
				hdr->AS.Header[pos+6]=0;
				T0.tm_sec = atoi((char*)hdr->AS.Header+pos+4);
				hdr->AS.Header[pos+4]=0;
				T0.tm_min = atoi((char*)hdr->AS.Header+pos+2);
				hdr->AS.Header[pos+2]=0;
				T0.tm_hour = atoi((char*)hdr->AS.Header+pos);
				flag_t0 |= 2;
				break;
				}
			case 0x00080070:  // Manufacturer
				{
				strncpy(hdr->ID.Manufacturer._field,(char*)hdr->AS.Header+pos,MAX_LENGTH_MANUF);
				hdr->ID.Manufacturer.Name = hdr->ID.Manufacturer._field;
				break;
				}
			case 0x00081050:  // Performing Physician
				{
				strncpy(hdr->ID.Technician,(char*)hdr->AS.Header+pos,MAX_LENGTH_TECHNICIAN);
				break;
				}
			case 0x00081090: // Manufacturer Model
				{
				const size_t nn = strlen(hdr->ID.Manufacturer.Name)+1;
				strncpy(hdr->ID.Manufacturer._field+nn,(char*)hdr->AS.Header+pos,MAX_LENGTH_MANUF-nn-1);
				hdr->ID.Manufacturer.Model = hdr->ID.Manufacturer._field+nn;
				break;
				}

			case 0x00100010: // Name
				if (!hdr->FLAG.ANONYMOUS) {
					strncpy(hdr->Patient.Name,(char*)hdr->AS.Header+pos,MAX_LENGTH_NAME);
					hdr->Patient.Name[MAX_LENGTH_NAME]=0;
				}
				break;
			case 0x00100020: // ID
				strncpy(hdr->Patient.Id,(char*)hdr->AS.Header+pos,MAX_LENGTH_PID);
				hdr->Patient.Id[MAX_LENGTH_PID]=0;
				break;

			case 0x00100030: // Birthday
				{
				struct tm t0;
				t0.tm_sec = 0;
				t0.tm_min = 0;
				t0.tm_hour = 12;

				hdr->AS.Header[pos+8]=0;
				t0.tm_mday = atoi((char*)hdr->AS.Header+pos+6);
				hdr->AS.Header[pos+6]=0;
				t0.tm_mon = atoi((char*)hdr->AS.Header+pos+4)-1;
				hdr->AS.Header[pos+4]=0;
				t0.tm_year = atoi((char*)hdr->AS.Header+pos)-1900;

				hdr->Patient.Birthday = tm_time2gdf_time(&t0);
				break;
				}
			case 0x00100040:
				hdr->Patient.Sex = (toupper(hdr->AS.Header[pos])=='M') + 2*(toupper(hdr->AS.Header[pos])=='F');
				break;

			case 0x00101010: //Age
				break;
			case 0x00101020:
				hdr->Patient.Height = (uint8_t)(atof((char*)hdr->AS.Header+pos)*100.0);
				break;
			case 0x00101030:
				hdr->Patient.Weight = (uint8_t)atoi((char*)hdr->AS.Header+pos);
				break;

			default:
				flag_ignored = 1;
				if (VERBOSE_LEVEL<7)
					fprintf(stdout,"ignored %6x:   (%04x,%04x) %8d\t%s\n",pos,Tag>>16,Tag&0x0ffff,Len,(char*)hdr->AS.Header+pos);

			}

			if (VERBOSE_LEVEL>6) {
			if (!FLAG_implicite_VR || (Tag < 0x00030000))
				fprintf(stdout,"%s %6x:   (%04x,%04x) %8d %c%c \t%s\n",(flag_ignored?"ignored":"       "),pos,Tag>>16,Tag&0x0ffff,Len,hdr->AS.Header[pos-8],hdr->AS.Header[pos-7],(char*)hdr->AS.Header+pos);
			else
				fprintf(stdout,"%s %6x:   (%04x,%04x) %8d\t%s\n",(flag_ignored?"ignored":"       "),pos,Tag>>16,Tag&0x0ffff,Len,(char*)hdr->AS.Header+pos);
			}
			pos += Len + (Len & 0x01 ? 1 : 0); // even number of bytes

		}
		if (flag_t0 == 3) hdr->T0 = tm_time2gdf_time(&T0);

if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %i)\n",__FILE__,__LINE__);

	return(0);
}
#endif


#ifdef WITH_PDP
#include "../NONFREE/sopen_pdp_read.c"
#endif

#ifdef WITH_TRC
#include "../NONFREE/sopen_trc_read.c"
#endif

#ifdef __cplusplus
}
#endif

