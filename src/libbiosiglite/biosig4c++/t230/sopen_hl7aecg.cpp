/*

    $Id$
    Copyright (C) 2006,2007,2009,2011,2012 Alois Schloegl <alois.schloegl@gmail.com>
    Copyright (C) 2007 Elias Apostolopoulos
    Copyright (C) 2011 Stoyan Mihaylov

    This file is part of the "BioSig for C/C++" repository 
    (biosig4c++) at http://biosig.sf.net/ 


    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.


 */


#include <stdio.h>	// system includes
#include <stdlib.h>	// for strtod(3)
#include <strings.h>

#include "../biosig-dev.h"
#ifdef WITH_LIBXML2
#  include <libxml/parser.h>
#  include <libxml/tree.h>
#else 
#  include "../XMLParser/tinyxml.h"
#endif


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
                convert time in string format into gdf-time format
                Currently, the following formats are supported 
                        YYYYMMDDhhmmss.uuuuuu        
                        YYYYMMDDhhmmss        
                        YYYYMMDD
                in case of error, zero is returned        
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
EXTERN_C gdf_time str_time2gdf_time(const char *t1) {

        struct tm t0; 
        gdf_time T;
        double fracsec = 0.0; 
        double f;
        int len;
#define MAXLEN  22        
        char t[MAXLEN+1];

        if (t1==NULL) return(0);
        len = strlen(t1);
        if (len>MAXLEN) return(0);
        if (len<8) return(0);
        strncpy(t,t1,MAXLEN);
        t[len] = 0;	

        if (VERBOSE_LEVEL>8) 
                fprintf(stdout,"str_time2gdf_time: [%i]<%s>\n",len,t1);
 
        char *p = strrchr(t,'.');
        if (p==NULL) {
                // no comma
                p = t+len; 
        }
        else { 
                for (p++, f=0.1; p[0]; p++, f=f/10) {
                        if (p[0]<'0' || p[0]>'9') return(0);
                        fracsec += (p[0]-'0')*f;
                }        
                p = strrchr(t,'.');
        }       

        if (VERBOSE_LEVEL>8) 
                fprintf(stdout,"str_time2gdf_time: [%i]<%s>\n",len,t1);
 
        if (len>=14) {
                // decode hhmmss
        	p[0] = '\0'; p-=2;
	        t0.tm_sec  = atoi(p);  	
	        p[0] = '\0'; p-=2;
	        t0.tm_min  = atoi(p);
	        p[0] = '\0'; p-=2;
	        t0.tm_hour = atoi(p);
        	p[0] = '\0'; 
	}
	else {
	        t0.tm_sec  = 0;  	
	        t0.tm_min  = 0;
	        t0.tm_hour = 0;
	}
	p -= 2;
	t0.tm_mday = atoi(p);

	p[0] = '\0'; p-=2;
	t0.tm_mon  = atoi(p)-1;

	p[0] = '\0'; p-=4;
	t0.tm_year = atoi(t)-1900;
	t0.tm_isdst  = -1;
	T = tm_time2gdf_time(&t0);

	if (fracsec>0)
	        T += ldexp(fracsec/86400,32);
	        
        if (VERBOSE_LEVEL>8) 
                fprintf(stdout,"str_time2gdf_time: [%i]<%s>\n",len,t1);
 
        return(T);
}


EXTERN_C int sopen_HL7aECG_read(HDRTYPE* hdr) {
/*
	this function is a stub or placeholder and need to be defined in order to be useful.
	It will be called by the function SOPEN in "biosig.c"

	Input: 
		char* Header	// contains the file content
		
	Output: 
		HDRTYPE *hdr	// defines the HDR structure accoring to "biosig.h"
*/

	char tmp[80]; 

	if (VERBOSE_LEVEL > 7) fprintf(stdout,"hl7r: [410]\n"); 

#ifdef WITH_LIBXML2
	fprintf(stderr,"Warning: LIBXML2 is used instead of TinyXML - support for HL7aECG is very experimental and must not be used for production use! You are warned\n");
#else
	TiXmlDocument doc(hdr->FileName);

	if (VERBOSE_LEVEL > 7) fprintf(stdout,"hl7r: [411]\n"); 

	if ( doc.LoadFile() ) {

	if (VERBOSE_LEVEL > 7) fprintf(stdout,"hl7r: [412]\n"); 

	    TiXmlHandle hDoc(&doc);
	    TiXmlHandle geECG = hDoc.FirstChild("CardiologyXML");
	    TiXmlHandle IHE = hDoc.FirstChild("IHEDocumentList");
	    TiXmlHandle aECG = hDoc.FirstChild("AnnotatedECG");
	    TiXmlHandle SierraECG = hDoc.FirstChild("restingECG");
	    TiXmlHandle SierraECG2 = hDoc.FirstChild("restingecgdata");

	    if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [412]\n"); 

	    if (SierraECG.Element()) {
		fprintf(stdout,"Great! Philips Sierra ECG is recognized\n");
	    }	    

	    else if (SierraECG2.Element()) {
		const char *t; 
		const char *e;
		float notch = 0, lowpass=0, highpass=0; 
		//uint16_t gdftyp = 16; 
		struct tm t0;

		fprintf(stdout,"Great! Philips Sierra ECG 2 is recognized\n");
		TiXmlHandle H = SierraECG2.FirstChild("dataacquisition");

		if (H.Element()) {
			e = SierraECG2.FirstChild("dataacquisition").Element()->Attribute("date");
			t0.tm_year = (int)strtod(e,(char**)&e) - 1900;
			t0.tm_mon = (int)strtod(e+1,(char**)&e) - 1 ;
			t0.tm_mday = (int)strtod(e+1,(char**)&e);
			e = SierraECG2.FirstChild("dataacquisition").Element()->Attribute("time");
			t0.tm_hour = (int)strtod(e+1,(char**)&e);
			t0.tm_min = (int)strtod(e+1,(char**)&e);
			t0.tm_sec = (int)strtod(e+1,(char**)&e);
			hdr->T0 = tm_time2gdf_time(&t0);
		}

		H = SierraECG2.FirstChild("dataacquisition").FirstChild("signalcharacteristics").FirstChild("acsetting");
		if (H.Element()) notch = atof(H.Element()->GetText());
		H = SierraECG2.FirstChild("reportinfo").FirstChild("reportbandwidth").FirstChild("lowpassfiltersetting");
		if (H.Element()) lowpass = atof(H.Element()->GetText());
		H = SierraECG2.FirstChild("reportinfo").FirstChild("reportbandwidth").FirstChild("highpassfiltersetting");
		if (H.Element()) highpass = atof(H.Element()->GetText());

		H = SierraECG2.FirstChild("dataacquisition").FirstChild("acquirer").FirstChild("institutionname");
		if (H.Element()) hdr->ID.Hospital = strdup(H.Element()->GetText());

		H = SierraECG2.FirstChild("dataacquisition").FirstChild("signalcharacteristics").FirstChild("samplingrate");
		if (H.Element()) hdr->SampleRate = atof(H.Element()->GetText());

		H = SierraECG2.FirstChild("dataacquisition").FirstChild("signalcharacteristics").FirstChild("signalresolution");
		double Cal = 1.0; if (H.Element()) Cal = atof(H.Element()->GetText());

		H = SierraECG2.FirstChild("dataacquisition").FirstChild("signalcharacteristics").FirstChild("numberchannelsvalid");
		if (H.Element()) hdr->NS = atoi(H.Element()->GetText());

		H = SierraECG2.FirstChild("dataacquisition").FirstChild("signalcharacteristics").FirstChild("numberchannelsallocated");
		if (H.Element()) {
			if (hdr->NS != atoi(H.Element()->GetText()) ) 
				fprintf(stdout,"SierraECG: number of channels is ambigous\n");
		}

		H = SierraECG2.FirstChild("patient").FirstChild("generalpatientdata").FirstChild("patientid");
		if (H.Element()) memcpy(hdr->Patient.Id, H.Element()->GetText(), MAX_LENGTH_PID);
		hdr->Patient.Name[0] = 0;
		size_t NameLength = 0;
		H = SierraECG2.FirstChild("patient").FirstChild("generalpatientdata").FirstChild("name").FirstChild("firstname");
		if (H.Element()) {
			strncpy(hdr->Patient.Name, H.Element()->GetText(), MAX_LENGTH_NAME);
			hdr->Patient.Name[MAX_LENGTH_NAME]=0;
			NameLength = strlen(hdr->Patient.Name);
		}
		H = SierraECG2.FirstChild("patient").FirstChild("generalpatientdata").FirstChild("name").FirstChild("middlename");
		if (H.Element()) {
			const char *str = H.Element()->GetText();
			size_t l2 = strlen(str); 
			if (NameLength+l2+1 < MAX_LENGTH_NAME) {
				hdr->Patient.Name[NameLength]= ' ';
				strncpy(hdr->Patient.Name+NameLength+1, str, l2+1);
				NameLength += l2+1;
			}
		}
		H = SierraECG2.FirstChild("patient").FirstChild("generalpatientdata").FirstChild("name").FirstChild("lastname");
		if (H.Element()) {
			const char *str = H.Element()->GetText();
			size_t l2 = strlen(str); 
			if (NameLength+l2+1 < MAX_LENGTH_NAME) {
				hdr->Patient.Name[NameLength]= ' ';
				strncpy(hdr->Patient.Name+NameLength+1, str, l2+1);
				NameLength += l2+1;
			}
		}
		H = SierraECG2.FirstChild("patient").FirstChild("generalpatientdata").FirstChild("age").FirstChild("years");
//		if (H.Element()) hdr->Patient.Age != atoi(H.Element()->GetText()) 

		H = SierraECG2.FirstChild("patient").FirstChild("generalpatientdata").FirstChild("sex");
		if (H.Element()) {
			t = H.Element()->GetText();
			hdr->Patient.Sex = (t[0]=='M') + (t[0]=='m') + 2 * ( (t[0]=='F') + (t[0]=='f') ); 
		}

		H = SierraECG2.FirstChild("waveforms").FirstChild("parsedwaveforms");
		if (H.Element()) {
			hdr->SPR = atof(H.Element()->Attribute("durationperchannel"))*hdr->SampleRate/1000;
		}
		hdr->NRec = 1; 	
		hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));
		for (uint16_t k=0; k<hdr->NS; k++) {
			CHANNEL_TYPE *hc = hdr->CHANNEL+k;
			hc->GDFTYP   = 16;
			sprintf(hc->Label,"#%i",k);
			hc->Cal      = Cal;
			hc->Off      = 0.0;
			hc->OnOff    = 1;
			hc->DigMin   = -ldexp(1,15);
			hc->DigMax   = +ldexp(1,15)-1;
			hc->PhysMin  = hc->DigMin * Cal;
			hc->PhysMax  = hc->DigMax * Cal;
			hc->PhysDimCode = PhysDimCode("nV");
			//strcpy(hc->PhysDim,"nV");
			hc->bi       = k*hdr->SPR*4;
			hc->bi8      = 0;
			hc->LeadIdCode = 0;
			hc->SPR      = hdr->SPR;
			hc->LowPass  = lowpass;
			hc->HighPass = highpass;
			hc->Notch    = notch;
			hc->TOffset  = 0; 
			hc->XYZ[0]   = 0; 
			hc->XYZ[1]   = 0; 
			hc->XYZ[2]   = 0; 
		}
		hdr->AS.bpb = sizeof(float)*hdr->NS;
		hdr->AS.rawdata = (uint8_t*)realloc(hdr->AS.rawdata, hdr->AS.bpb*hdr->NRec*hdr->SPR);
		hdr->AS.first = 0;
		hdr->AS.length= hdr->NRec;

		if (H.Element()) {
			const char *e = H.Element()->GetText();
			size_t k = 0;
			while (e != NULL && k < hdr->SPR * (size_t)hdr->NRec * hdr->NS) {
				((float*)hdr->AS.rawdata)[k++] = (float)strtod(e,(char**)&e);
			}
		}

	    }	    
	    else if (geECG.Element()) {
			hdr->ID.Manufacturer.Name = "GE";

			TiXmlHandle H = geECG.FirstChild("ClinicalInfo").FirstChild("ObservationDateTime");
			if (H.Element()) {
				struct tm t0;
				t0.tm_hour = atoi(H.FirstChild("Hour").Element()->GetText());
				t0.tm_min  = atoi(H.FirstChild("Minute").Element()->GetText());
				t0.tm_sec  = atoi(H.FirstChild("Second").Element()->GetText());
				t0.tm_mday = atoi(H.FirstChild("Day").Element()->GetText());
				t0.tm_mon  = atoi(H.FirstChild("Month").Element()->GetText())-1;
				t0.tm_year = atoi(H.FirstChild("Year").Element()->GetText())-1900;
				hdr->T0    = tm_time2gdf_time(&t0);
	                } 

			H = geECG.FirstChild("Device-Type");
			if (H.Element()) {
				strncpy(hdr->ID.Manufacturer._field, H.Element()->GetText(),MAX_LENGTH_PID);
				hdr->ID.Manufacturer.Model = hdr->ID.Manufacturer._field;
			}				

			if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [413]\n"); 

			H = geECG.FirstChild("PatientInfo");
			if (H.Element()) {
				strncpy(hdr->Patient.Id, H.FirstChild("PID").Element()->GetText(),MAX_LENGTH_PID);
				const char *tmp = H.FirstChild("PID").Element()->GetText();
				hdr->Patient.Sex = (toupper(tmp[0])=='M') + 2*(toupper(tmp[0])=='F');
				if (!hdr->FLAG.ANONYMOUS) {
					const char *str1 = H.FirstChild("Name").FirstChild("FamilyName").Element()->GetText();
					const char *str2 = H.FirstChild("Name").FirstChild("GivenName").Element()->GetText();
					size_t l1 = str1 ? strlen(str1) : 0;
					size_t l2 = str2 ? strlen(str2) : 0;
					if (0 < l1 && l1 <= MAX_LENGTH_PID) strncpy(hdr->Patient.Name, str1, l1+1);
					if (l1+l2+1 < MAX_LENGTH_PID) {	
						hdr->Patient.Name[l1] = ' ';
						strncpy(hdr->Patient.Name+l1+1, str2, l2+1);
					}
				}
			}

			if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [413]\n"); 

			double Cal=0.0, LP=NAN, HP=NAN, Notch=0.0;
			hdr->NRec= 0;
			hdr->SPR = 1;
			hdr->NS  = 1;

			H = geECG.FirstChild("FilterSetting");
			if (H.Element()) {
				LP = atof(H.FirstChild("LowPass").Element()->GetText());
				HP = atof(H.FirstChild("HighPass").Element()->GetText());
				if (!strcasecmp("yes",H.FirstChild("Filter50Hz").Element()->GetText()))
					Notch = 50; 
				else if (!strcasecmp("yes",H.FirstChild("Filter60Hz").Element()->GetText()))
					Notch = 60; 
			}
			
			if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [413]\n"); 

			H = geECG.FirstChild("StripData");
			TiXmlElement *C = NULL;	
			if (H.Element()) {
				C = H.FirstChild("NumberOfLeads").Element();
				if (C != NULL) hdr->NS = atoi(C->GetText());

				C = H.FirstChild("ChannelSampleCountTotal").Element();
				if (C != NULL) hdr->NRec = atoi(C->GetText());

				hdr->SampleRate = atof(H.FirstChild("SampleRate").Element()->GetText());
				Cal = atof(H.FirstChild("Resolution").Element()->GetText());
			}
			
			uint16_t gdftyp = 3; 
			hdr->AS.bpb = 0; 
			
			int k=0, NCHAN = hdr->NS;
			hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, hdr->NS*sizeof(CHANNEL_TYPE));
			C = H.FirstChild("WaveformData").Element();
			while (C != NULL) {

				if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [413] %i\n",k); 

				if (k>=NCHAN) {
					NCHAN = max(12,(NCHAN+1)*2);
					hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, NCHAN*sizeof(CHANNEL_TYPE));
				}	

				CHANNEL_TYPE *hc = hdr->CHANNEL + k;
				// default values 
				hc->GDFTYP	= gdftyp;	
				hc->PhysDimCode	= 4275; //PhysDimCode("uV");	
				hc->DigMin 	= (double)(int16_t)0x8000;			
				hc->DigMax	= (double)(int16_t)0x7fff;	
				strncpy(hc->Label, C->Attribute("lead"), MAX_LENGTH_LABEL);

				hc->LeadIdCode	= 0;
				size_t j;
				for (j=0; strcasecmp(hc->Label, LEAD_ID_TABLE[j]) && LEAD_ID_TABLE[j][0]; j++) {}; 
				if (LEAD_ID_TABLE[j][0])	
					hc->LeadIdCode = j;

				hc->LowPass	= LP;
				hc->HighPass	= HP;
				hc->Notch	= Notch;
				hc->Impedance	= NAN;
				hc->XYZ[0] 	= 0.0;
				hc->XYZ[1] 	= 0.0;
				hc->XYZ[2] 	= 0.0;
				
				// defined 
				hc->Cal		= Cal;
				hc->Off		= 0.0;
				hc->SPR		= 1; 
				hc->OnOff	= 1;	

				C = C->NextSiblingElement();
				k++;
			}

			hdr->NS = k;	

		if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [417] %i\n",hdr->NS); 


			C = H.FirstChild("WaveformData").Element();
			size_t szRawData = 0; 
			size_t SPR = 0;	
			for (k=0; k<hdr->NS; k++) {

				if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [415] %i\n",k); 

				CHANNEL_TYPE *hc = hdr->CHANNEL + k;

				    /* read data samples */	
				hc->DigMax	= -1.0/0.0; 
				hc->DigMin	=  1.0/0.0; 
				// int16_t* data = (int16_t*)(hdr->AS.rawdata)+SPR;

				hc->bi	= hdr->AS.bpb; 
				char *s = (char*)C->GetText();
				const char *delim = ",";
				size_t spr = 0; 	
				while (s && *s) {
					s += strspn(s, delim);	//skip kommas
					if (SPR+spr+1 >= szRawData) {
						szRawData = max(5000,2*szRawData);
						hdr->AS.rawdata = (uint8_t*) realloc(hdr->AS.rawdata, szRawData * sizeof(int16_t));
					}	
					double d = strtod(s,&s);
					((int16_t*)hdr->AS.rawdata)[SPR+spr++] = d;				
					/* get Min/Max */
					if (d > hc->DigMax) hc->DigMax = d;
					if (d < hc->DigMin) hc->DigMin = d;
				}
				SPR += spr;		 	
				hc->SPR	= spr;
				hdr->SPR = lcm(hdr->SPR,spr); 

				hc->PhysMax	= hc->DigMax * hc->Cal + hc->Off; 
				hc->PhysMin	= hc->DigMin * hc->Cal + hc->Off; 

				C = C->NextSiblingElement();
			}
			hdr->AS.bpb    += hdr->SPR * 2;
			hdr->NRec = 1;
			hdr->AS.first = 0; 	
			hdr->AS.length = hdr->NRec; 	

		if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [497] %i\n",hdr->NS); 


	    }
	    else if (IHE.Element()) {

		fprintf(stderr,"XML IHE: support for IHE XML is experimental - some important features are not implmented yet \n"); 

		TiXmlHandle activityTime = IHE.FirstChild("activityTime");
		TiXmlHandle recordTarget = IHE.FirstChild("recordTarget");
		TiXmlHandle author = IHE.FirstChild("author");
		/* 
			an IHE file can contain several segments (i.e. components)
		 	need to implement TARGET_SEGMENT feature
		*/
		TiXmlHandle component = IHE.FirstChild("component");

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"IHE: [413] \n"); 

		if (author.FirstChild("assignedAuthor").Element()) {
			// TiXmlHandle noteText = author.FirstChild("noteText").Element();
			TiXmlHandle assignedAuthor = author.FirstChild("assignedAuthor").Element();
			if (assignedAuthor.FirstChild("assignedDevice").Element()) {
				TiXmlHandle assignedDevice = assignedAuthor.FirstChild("assignedDevice").Element();
				hdr->ID.Manufacturer.Name = hdr->ID.Manufacturer._field;
				
				if (assignedDevice.Element()) {
	 				strncpy(hdr->ID.Manufacturer._field, assignedDevice.FirstChild("manufacturerModelName").Element()->GetText(), MAX_LENGTH_MANUF);	
					int len = strlen(hdr->ID.Manufacturer._field)+1;
					hdr->ID.Manufacturer.Model = hdr->ID.Manufacturer._field+len;
					strncpy(hdr->ID.Manufacturer._field+len, assignedDevice.FirstChild("code").Element()->Attribute("code"),MAX_LENGTH_MANUF-len);
					len += strlen(hdr->ID.Manufacturer.Model)+1;
				}
			}	
		}	

		if (recordTarget.FirstChild("patient").Element()) {
			TiXmlHandle patient = recordTarget.FirstChild("patient").Element();

			TiXmlHandle id = patient.FirstChild("id").Element();
			TiXmlHandle patientPatient = patient.FirstChild("patientPatient").Element();
			TiXmlHandle providerOrganization = patient.FirstChild("providerOrganization").Element();
			
			if (VERBOSE_LEVEL>8)
				fprintf(stdout,"IHE: [414] %p %p %p\n",id.Element(),patientPatient.Element(),providerOrganization.Element()); 

			if (id.Element()) {	
			    	char *strtmp = strdup(id.Element()->Attribute("root"));
			    	size_t len = strlen(strtmp); 
				if (len <= MAX_LENGTH_RID) {
					strcpy(hdr->ID.Recording, strtmp);	// Flawfinder: ignore

					if (strtmp) free(strtmp);
				    	strtmp = strdup(id.Element()->Attribute("extension"));
					size_t l1 = strlen(strtmp); 
					if (len+1+l1 < MAX_LENGTH_RID) {
					    	len += 1 + l1;
						hdr->ID.Recording[len] = ' ';
						strncpy(hdr->ID.Recording+len+1,strtmp,l1+1);
					} 
					else 
						fprintf(stdout,"Warning HL7aECG(read): length of Recording ID exceeds maximum length %i>%i\n",(int)(len+1+l1),MAX_LENGTH_PID);
				}
				else 
					fprintf(stdout,"Warning HL7aECG(read): length of Recording ID exceeds maximum length %i>%i\n",(int)len,MAX_LENGTH_PID); 

				if (strtmp) free(strtmp); 
				if (VERBOSE_LEVEL>7)
					fprintf(stdout,"IHE (read): length of Recording ID %i,%i\n",(int)len,MAX_LENGTH_PID); 
			}	
			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"IHE: [414] RID= %s\n",hdr->ID.Recording); 
			
			if (providerOrganization.Element()) {
				hdr->ID.Hospital = strdup(providerOrganization.FirstChild("name").Element()->GetText());
			}	
			
			if (VERBOSE_LEVEL>7)
				fprintf(stdout,"IHE: [414] hospital %s\n",hdr->ID.Hospital); 

			if (patientPatient.Element()) {
				if (!hdr->FLAG.ANONYMOUS) {
				TiXmlHandle Name = patientPatient.FirstChild("name").Element();
				if (Name.Element()) {
					char *str1 = strdup(Name.FirstChild("family").Element()->GetText());
					char *str2 = strdup(Name.FirstChild("given").Element()->GetText());
					size_t l1 = str1 ? strlen(str1) : 0;
					size_t l2 = str2 ? strlen(str2) : 0;
					if (l1 <= MAX_LENGTH_NAME) 
						strcpy(hdr->Patient.Name, str1);		// Flawfinder: ignore
					if (l1+l2+2 <= MAX_LENGTH_NAME) {
						strcpy(hdr->Patient.Name, str1);		// Flawfinder: ignore
						strcpy(hdr->Patient.Name+l1, ", ");		// Flawfinder: ignore
						strcpy(hdr->Patient.Name+l1+2, str2);		// Flawfinder: ignore
					}
				}
				}
				TiXmlHandle Gender = patientPatient.FirstChild("administrativeGenderCode").Element();
				TiXmlHandle Birth = patientPatient.FirstChild("birthTime").Element();

				if (Gender.Element()) {
					const char *gender = Gender.Element()->Attribute("code");
					hdr->Patient.Sex = (tolower(gender[0])=='m') + (tolower(gender[0])=='f');
				}
				if (Birth.Element()) {
			 		hdr->Patient.Birthday = str_time2gdf_time(Birth.Element()->Attribute("value"));
				}
			}
		}
		
		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"IHE: [415] \n"); 

	    }
	    else if(aECG.Element()){

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"hl7r: [412]\n"); 

	    	size_t len = strlen(aECG.FirstChild("id").Element()->Attribute("root")); 

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"hl7r: [413]\n"); 

		strncpy(hdr->ID.Recording,aECG.FirstChild("id").Element()->Attribute("root"),MAX_LENGTH_RID);
	    	if (len>MAX_LENGTH_RID)	
			fprintf(stdout,"Warning HL7aECG(read): length of Recording ID exceeds maximum length %i>%i\n",(int)len,MAX_LENGTH_PID); 

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"hl7r: [414]\n"); 


		TiXmlHandle effectiveTime = aECG.FirstChild("effectiveTime");

		char *T0 = NULL;
		if(effectiveTime.FirstChild("low").Element())
		    T0 = (char *)effectiveTime.FirstChild("low").Element()->Attribute("value");
		else if(effectiveTime.FirstChild("center").Element())
		    T0 = (char *)effectiveTime.FirstChild("center").Element()->Attribute("value");

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"hl7r: [413 2] <%s>\n", T0); 

                if (T0 != NULL) 
                        hdr->T0 = str_time2gdf_time(T0);

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"hl7r: [413 4]\n"); 

		TiXmlHandle demographic = aECG.FirstChild("componentOf").FirstChild("timepointEvent").FirstChild("componentOf").FirstChild("subjectAssignment").FirstChild("subject").FirstChild("trialSubject");

		TiXmlElement *id = demographic.FirstChild("id").Element();
		if(id) {
			const char* tmpstr = id->Attribute("extension");
			size_t len = strlen(tmpstr); 
			if (len>MAX_LENGTH_PID)
				fprintf(stdout,"Warning HL7aECG(read): length of Patient Id exceeds maximum length %i>%i\n",(int)len,MAX_LENGTH_PID); 
		    	strncpy(hdr->Patient.Id,tmpstr,MAX_LENGTH_PID);
		}    

		if (VERBOSE_LEVEL>7) 
			fprintf(stdout,"hl7r: [413]\n"); 

		if (!hdr->FLAG.ANONYMOUS) 
		{
			demographic = demographic.FirstChild("subjectDemographicPerson");
			TiXmlElement *Name1 = demographic.FirstChild("name").Element();

			if (Name1 != NULL) {
				const char *name = Name1->GetText();
				if (name != NULL) {
					size_t len = strlen(name);
	
					if (len>MAX_LENGTH_NAME)
						fprintf(stdout,"Warning HL7aECG(read): length of Patient Name exceeds maximum length %i>%i\n",(int)len,MAX_LENGTH_PID); 
					strncpy(hdr->Patient.Name, name, MAX_LENGTH_NAME);
				}	
				else {
					fprintf(stderr,"Warning: composite subject name is not supported.\n");
                        		if (VERBOSE_LEVEL>7) {
                        			fprintf(stdout,"hl7r: [413++]<%s>\n",name);
        					for (int k=1;k<40;k++)
        						fprintf(stderr,"%c.",((char*)Name1)[k]);
                                        }
					//hdr->Patient.Name[0] = 0;
/*
				### FIXME: support of composite patient name.  

				const char *Name11 = Name1->Attribute("family");
				fprintf(stdout,"Patient Family Name %p\n", Name11);
				char *Name2 = Name.FirstChild("given")->GetText();

				if ((Name1!=NULL) || (Name2!=NULL)) {
					strncpy(hdr->Patient.Name, Name1, MAX_LENGTH_NAME);
				}
*/
				}
			}
			else {
				hdr->Patient.Name[0] = 0;
				fprintf(stderr,"Warning: Patient Name not available could not be read.\n");
			}
		}

		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"hl7r: [414]\n"); 

		/* non-standard fields height and weight */
		TiXmlElement *weight = demographic.FirstChild("weight").Element();
		if (weight) {
		    uint16_t code = PhysDimCode(weight->Attribute("unit"));
		    if ((code & 0xFFE0) != 1728) 
		    	fprintf(stderr,"Warning: incorrect weight unit (%s)\n",weight->Attribute("unit"));	
		    else 	// convert to kilogram
			hdr->Patient.Weight = (uint8_t)(atof(weight->Attribute("value"))*PhysDimScale(code)*1e-3);  
		}
		TiXmlElement *height = demographic.FirstChild("height").Element();

		if (VERBOSE_LEVEL>7)
			fprintf(stdout,"hl7r: [415]\n"); 

		if (height) {
		    uint16_t code = PhysDimCode(height->Attribute("unit"));
		    if ((code & 0xFFE0) != 1280)
			fprintf(stderr,"Warning: incorrect height unit (%s) %i \n",height->Attribute("unit"),code);
		    else	// convert to centimeter
			hdr->Patient.Height = (uint8_t)(atof(height->Attribute("value"))*PhysDimScale(code)*1e+2);
		}
		
		if (VERBOSE_LEVEL>7) 
		        fprintf(stdout,"hl7r: [416]\n"); 

		TiXmlElement *birthday = demographic.FirstChild("birthTime").Element();
		if(birthday){
		    T0 = (char *)birthday->Attribute("value");
		    if (T0==NULL) T0=(char *)birthday->GetText();  // workaround for reading two different formats 
		    hdr->Patient.Birthday = str_time2gdf_time(T0);
		}

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"hl7r: [417]\n"); 

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"hl7r: [418]\n"); 

		TiXmlElement *sex = demographic.FirstChild("administrativeGenderCode").Element();
		if(sex){

		    if (sex->Attribute("code")==NULL)
			hdr->Patient.Sex = 0;
		    else if(!strcmp(sex->Attribute("code"),"F"))
			hdr->Patient.Sex = 2;
		    else if(!strcmp(sex->Attribute("code"),"M"))
			hdr->Patient.Sex = 1;
		    else
			hdr->Patient.Sex = 0;
		} 
		else {
		    hdr->Patient.Sex = 0;
		}

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"hl7r: [419]\n"); 

		int LowPass=0, HighPass=0, Notch=0;
		TiXmlHandle channels = aECG.FirstChild("component").FirstChild("series").FirstChild("component").FirstChild("sequenceSet");
		TiXmlHandle variables = aECG.FirstChild("component").FirstChild("series");

		for(TiXmlElement *tmpvar = variables.FirstChild("controlVariable").Element(); tmpvar; tmpvar = tmpvar->NextSiblingElement("controlVariable")){
		    if(!strcmp(tmpvar->FirstChildElement("controlVariable")->FirstChildElement("code")->Attribute("code"), "MDC_ATTR_FILTER_NOTCH"))
			Notch = atoi(tmpvar->FirstChildElement("controlVariable")->FirstChildElement("component")->FirstChildElement("controlVariable")->FirstChildElement("value")->Attribute("value"));
		    else if(!strcmp(tmpvar->FirstChildElement("controlVariable")->FirstChildElement("code")->Attribute("code"), "MDC_ATTR_FILTER_LOW_PASS"))
			LowPass = atoi(tmpvar->FirstChildElement("controlVariable")->FirstChildElement("component")->FirstChildElement("controlVariable")->FirstChildElement("value")->Attribute("value"));
		    else if(!strcmp(tmpvar->FirstChildElement("controlVariable")->FirstChildElement("code")->Attribute("code"), "MDC_ATTR_FILTER_HIGH_PASS"))
			HighPass = atoi(tmpvar->FirstChildElement("controlVariable")->FirstChildElement("component")->FirstChildElement("controlVariable")->FirstChildElement("value")->Attribute("value"));
		}

		if (VERBOSE_LEVEL>8)
			fprintf(stdout,"hl7r: [421]\n"); 

		hdr->NRec = 1;
//		hdr->SPR = 1;
//		hdr->AS.rawdata = (uint8_t *)malloc(hdr->SPR);
//		int32_t *data;
		
		hdr->SampleRate = 1.0/atof(channels.FirstChild("component").FirstChild("sequence").FirstChild("value").FirstChild("increment").Element()->Attribute("value"));

		if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [517] %f\n",hdr->SampleRate); 
		
                /*************** Annotations **********************/
		TiXmlHandle AnnotationSet = aECG.FirstChild("component").FirstChild("series").FirstChild("subjectOf").FirstChild("annotationSet");
		TiXmlHandle Annotation = AnnotationSet.Child("component", 1).FirstChild("annotation").FirstChild("component").FirstChild("annotation"); 
		size_t N_Event = 0, N=0; 
		for(int i = 1; AnnotationSet.Child("component", i).FirstChild("annotation").Element(); ++i) {
        		for(int j = 0; j<3; ++j) {

		                Annotation = AnnotationSet.Child("component", i).FirstChild("annotation").Child("component",j).FirstChild("annotation");

				if (Annotation.FirstChild("value").Element() == NULL) break;
        		        const char *code = Annotation.FirstChild("value").Element()->Attribute("code");
				if (code==NULL) break;

                                uint16_t EventTyp1 = 0, EventTyp2 = 0;

                                if (!strcmp(code,"MDC_ECG_WAVC_PWAVE")) {
                                        EventTyp1 = 0x0502;        // start P-Wave
                                        EventTyp2 = 0x8502;        // end P-Wave
                                }
                                else if (!strcmp(code,"MDC_ECG_WAVC_QRSWAVE")) {
                                        EventTyp1 = 0x0503;        // start QRS
                                        EventTyp2 = 0x8505;        // end QRS
                                }
                                else if (!strcmp(code,"MDC_ECG_WAVC_TWAVE")) {
                                        EventTyp1 = 0x0506;        // start T-Wave
                                        EventTyp2 = 0x8506;        // end T-Wave
                                }    

                                if ((N+3) > N_Event) {
                                	N_Event = max(16,2*(N+2));
                                	hdr->EVENT.TYP = (typeof(hdr->EVENT.TYP)) realloc(hdr->EVENT.TYP,N_Event*sizeof(*hdr->EVENT.TYP));
                                	hdr->EVENT.POS = (typeof(hdr->EVENT.POS)) realloc(hdr->EVENT.POS,N_Event*sizeof(*hdr->EVENT.POS));
                                }

        		        TiXmlHandle Boundary = Annotation.FirstChild("support").FirstChild("supportingROI").FirstChild("component").FirstChild("boundary").FirstChild("value");

                                int64_t pos1=0, pos2=0;
        		        if (Boundary.FirstChild("low").Element()) {
                                        const char *tmpstr = (Boundary.FirstChild("low").Element()->Attribute("value"));
                                        pos1 = (ldexp((str_time2gdf_time(tmpstr)-hdr->T0)*86400*hdr->SampleRate,-32));
                                        hdr->EVENT.TYP[N] = EventTyp1;
                                        hdr->EVENT.POS[N] = pos1; 
                                        N++;        
                                }        

        		        if (Boundary.FirstChild("high").Element()) {
                                        const char *tmpstr = (Boundary.FirstChild("high").Element()->Attribute("value"));
                                        pos2 = (ldexp((str_time2gdf_time(tmpstr)-hdr->T0)*86400*hdr->SampleRate,-32));
                                        hdr->EVENT.TYP[N] = EventTyp2;
                                        hdr->EVENT.POS[N] = pos2;
                                        N++;        
                                }        
               		}
       		}
       		hdr->EVENT.N = N;

		TiXmlHandle channel = channels.Child("component", 1).FirstChild("sequence");
		for (hdr->NS = 0; channel.Element(); ++(hdr->NS), channel = channels.Child("component", hdr->NS+1).FirstChild("sequence")) {};
		hdr->CHANNEL = (CHANNEL_TYPE*) realloc(hdr->CHANNEL, hdr->NS * sizeof(CHANNEL_TYPE));

		channel = channels.Child("component", 1).FirstChild("sequence");
		hdr->AS.bpb = 0; 

		size_t szRawData = 0; 
		size_t SPR = 0;	
		hdr->SPR = 1;

		for(int i = 0; channel.Element(); ++i, channel = channels.Child("component", i+1).FirstChild("sequence")){

			const char *code = channel.FirstChild("code").Element()->Attribute("code");
		    
  			CHANNEL_TYPE *hc = hdr->CHANNEL+i;
                	hc->LeadIdCode = 0;
  			
			if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [420] %i\n",i); 

		    	strncpy(hc->Label, code, min(40, MAX_LENGTH_LABEL));
		    	hc->Label[MAX_LENGTH_LABEL] = '\0';
		    	hc->Transducer[0] = '\0';
		    	hc->GDFTYP = 16;	// float32
   		    	hc->DigMax	= -1.0/0.0; 
			hc->DigMin	=  1.0/0.0; 

			if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [420] %i\n",i); 

			char *s = (char*) channel.FirstChild("value").FirstChild("digits").Element()->GetText();
			size_t spr = 0; 	
			//char *ps = s ? s+strlen(s) : NULL; //end of s

			if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [420] %i %p <%s>\n",i,s,s); 

			while (s && *s) {
				if (SPR+spr+1 >= szRawData) {
					szRawData = max(5000, 2*szRawData);
					hdr->AS.rawdata = (uint8_t*) realloc(hdr->AS.rawdata, szRawData * sizeof(float));
				}	

				double d = strtod(s,&s);
				((float*)(hdr->AS.rawdata))[SPR + spr++] = d;				
				/* get Min/Max */
				if(d > hc->DigMax) hc->DigMax = d;
				if(d < hc->DigMin) hc->DigMin = d;
			}
			if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [420] %i %i\n",i,(int)spr); 

	   	    	hc->bi = hdr->AS.bpb;
			SPR += spr;		 	
			hc->SPR	= spr;
			if (spr>0) hdr->SPR = lcm(hdr->SPR,spr); 

			hc->OnOff = 1;
			hc->LeadIdCode = 0;
  			hdr->AS.bpb += hc->SPR * sizeof(float);

			if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [420+] %i\n",i); 

			/* scaling factors */ 
			const char *tmpchar; 
			tmpchar = channel.FirstChild("value").FirstChild("scale").Element()->Attribute("value");
			if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [420] <%s>\n",tmpchar); 
			hc->Cal = atof(tmpchar);
			tmpchar = channel.FirstChild("value").FirstChild("origin").Element()->Attribute("value");
			if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [420] <%s>\n",tmpchar); 
			hc->Off = atof(tmpchar);
			hc->DigMax += 1;
			hc->DigMin -= 1;
			if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [420] Cal: %f Off: %f\n",hc->Cal,hc->Off); 
			hc->PhysMax = hc->DigMax*hc->Cal + hc->Off;
			hc->PhysMin = hc->DigMin*hc->Cal + hc->Off;

			/* Physical units */ 
			strncpy(tmp, channel.FirstChild("value").FirstChild("origin").Element()->Attribute("unit"),20);
 			hc->PhysDimCode = PhysDimCode(tmp);
 		    
			if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7r: [420] %i\n",i); 

			hc->LowPass  = LowPass;
			hc->HighPass = HighPass;
			hc->Notch    = Notch;
// 			hc->XYZ[0]   = l_endian_f32( *(float*) (Header2+ 4*k + 224*hdr->NS) );
// 			hc->XYZ[1]   = l_endian_f32( *(float*) (Header2+ 4*k + 228*hdr->NS) );
// 			hc->XYZ[2]   = l_endian_f32( *(float*) (Header2+ 4*k + 232*hdr->NS) );
// 				//memcpy(&hdr->CHANNEL[k].XYZ,Header2 + 4*k + 224*hdr->NS,12);
// 			hc->Impedance= ldexp(1.0, (uint8_t)Header2[k + 236*hdr->NS]/8);

//		    hc->Impedance = INF;
//		    for(int k1=0; k1<3; hdr->CHANNEL[index].XYZ[k1++] = 0.0);


		}
		hdr->FLAG.OVERFLOWDETECTION = 0;

		if (VERBOSE_LEVEL>7) {
			fprintf(stdout,"hl7r: [430] %i\n",hdr->AS.B4C_ERRNUM); 
			hdr2ascii(hdr,stdout,3);
			fprintf(stdout,"hl7r: [431] %i\n",hdr->AS.B4C_ERRNUM); 
		}

	    } else {
		fprintf(stderr, "%s : failed to parse (2)\n", hdr->FileName);
	    }
	}
	else
	    fprintf(stderr, "%s : failed to parse (1)\n", hdr->FileName);

#endif

	return(0);

};

EXTERN_C void sopen_HL7aECG_write(HDRTYPE* hdr) {

	if (VERBOSE_LEVEL > 7) fprintf(stdout,"hl7w: [610] <%s>\n",hdr->FileName); 

	size_t k;
	for (k=0; k<hdr->NS; k++) {
		hdr->CHANNEL[k].GDFTYP = 16; //float32
		hdr->CHANNEL[k].SPR *= hdr->NRec;
	}
	hdr->SPR *= hdr->NRec;
	hdr->NRec = 1; 
	hdr->FILE.OPEN=2;

	return;
};

EXTERN_C int sclose_HL7aECG_write(HDRTYPE* hdr){
/*
	this function is a stub or placeholder and need to be defined in order to be useful.
	It will be called by the function SOPEN in "biosig.c"

	Input: HDR structure

	Output:
		char* HDR.AS.Header 	// contains the content which will be written to the file in SOPEN
*/	

	if (VERBOSE_LEVEL > 7) fprintf(stdout,"hl7c: [910] <%s>\n",hdr->FileName); 

    struct tm *t0;
    char tmp[80];	

#ifdef WITH_LIBXML2
	fprintf(stderr,"Warning: LIBXML2 is used instead of TinyXML - support for HL7aECG is very experimental and must not be used for production use! You are warned\n");

	xmlDoc *doc = xmlNewDoc("1.0");
	xmlNode *root = xmlNewNode(NULL, "root");
	xmlDocSetRootElement(doc, root);
 
	xmlNode *node = xmlNewNode(NULL, "element");
	xmlAddChild(node, xmlNewText("some text here"));
	xmlAddChild(root, node);
 
	if (ifopen(hdr, "w")) {
		biosigERROR(hdr, B4C_CANNOT_WRITE_FILE, "Cannot open file for writing");
	} 
	else if (hdr->FILE.COMPRESSION) 
		xmlElemDump(hdr->FILE.gzFID, doc, root);
	else
		xmlElemDump(hdr->FILE.FID, doc, root);
	
	ifclose(hdr);
 
	xmlFreeDoc(doc);
	xmlCleanupParser();
 

#else

    TiXmlDocument doc;
    
    TiXmlDeclaration* decl = new TiXmlDeclaration("1.0", "UTF-8", "");
    doc.LinkEndChild(decl);
    
	if (VERBOSE_LEVEL>7) fprintf(stdout,"910 %i\n",1);

    TiXmlElement *root = new TiXmlElement("AnnotatedECG");
    root->SetAttribute("xmlns", "urn:hl7-org:v3");
    root->SetAttribute("xmlns:voc", "urn:hl7-org:v3/voc");
    root->SetAttribute("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance");
    root->SetAttribute("xsi:schemaLocation", "urn:hl7-org:v3/HL7/aECG/2003-12/schema/PORT_MT020001.xsd");
    root->SetAttribute("classCode", "OBS");
    root->SetAttribute("moodCode", "EVN");
    doc.LinkEndChild(root);

    TiXmlElement *rootid = new TiXmlElement("id");
    rootid->SetAttribute("root", hdr->ID.Recording);
    root->LinkEndChild(rootid);
	
    TiXmlElement *rootCode = new TiXmlElement("code");
    rootCode->SetAttribute("code", "93000");
    rootCode->SetAttribute("codeSystem", "2.16.840.1.113883.6.12");
    rootCode->SetAttribute("codeSystemName", "CPT-4");
    root->LinkEndChild(rootCode);
    
	if (VERBOSE_LEVEL>7) fprintf(stdout,"910 %i\n",2);

	char timelow[24], timehigh[24];
	gdf_time t1,t2;
	t1 = hdr->T0;// + ldexp(timezone/(3600.0*24),32);	
	t0 = gdf_time2tm_time(t1);
	t2 = tm_time2gdf_time(t0);
	double dT;
	dT = ldexp(t1-t2,-32)*(3600*24);
	dT = round(dT*1000);
	sprintf(timelow, "%4d%2d%2d%2d%2d%2d.%3d", t0->tm_year+1900, t0->tm_mon+1, t0->tm_mday, t0->tm_hour, t0->tm_min, t0->tm_sec,(int)ceil(dT));

	t1 = hdr->T0 + ldexp((hdr->SPR/hdr->SampleRate)/(3600.0*24),32);	
	t0 = gdf_time2tm_time(t1);
	t2 = tm_time2gdf_time(t0);
	dT = ldexp(t1-t2,-32)*(3600*24);
	dT = floor(dT*1000);
	sprintf(timehigh, "%4d%2d%2d%2d%2d%2d.%3d", t0->tm_year+1900, t0->tm_mon+1, t0->tm_mday, t0->tm_hour, t0->tm_min, t0->tm_sec,(int)ceil(dT));
	for(int i=0; i<18; ++i) {
		if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7c 920 %i\n",i);
		if(timelow[i] == ' ')
			timelow[i] = '0';
		if(timehigh[i] == ' ')
			timehigh[i] = '0';
	}

	if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7c 930\n");

    TiXmlElement *effectiveTime = new TiXmlElement("effectiveTime");
    TiXmlElement *effectiveTimeLow = new TiXmlElement("low");
    effectiveTimeLow->SetAttribute("value", timelow);
    effectiveTime->LinkEndChild(effectiveTimeLow);
    TiXmlElement *effectiveTimeHigh = new TiXmlElement("high");
    effectiveTimeHigh->SetAttribute("value", timehigh);
    effectiveTime->LinkEndChild(effectiveTimeHigh);
    root->LinkEndChild(effectiveTime);

	if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7c 931\n");

    TiXmlElement *rootComponentOf = new TiXmlElement("componentOf");
    rootComponentOf->SetAttribute("typeCode", "COMP");
    rootComponentOf->SetAttribute("contextConductionInd", "true");
    root->LinkEndChild(rootComponentOf);

	if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7c 932\n");

    TiXmlElement *timePointEvent = new TiXmlElement("timepointEvent");
    timePointEvent->SetAttribute("classCode", "CTTEVENT");
    timePointEvent->SetAttribute("moodCode", "EVN");
    rootComponentOf->LinkEndChild(timePointEvent);
    
    TiXmlElement *timePointComponentOf = new TiXmlElement("componentOf");
    timePointComponentOf->SetAttribute("typeCode", "COMP");
    timePointComponentOf->SetAttribute("contextConductionInd", "true");
    timePointEvent->LinkEndChild(timePointComponentOf);

    TiXmlElement *subjectAssignment = new TiXmlElement("subjectAssignment");
    subjectAssignment->SetAttribute("classCode", "CLNTRL");
    subjectAssignment->SetAttribute("moodCode", "EVN");
    timePointComponentOf->LinkEndChild(subjectAssignment);

    TiXmlElement *subject = new TiXmlElement("subject");
    subject->SetAttribute("typeCode", "SBJ");
    subject->SetAttribute("contextControlCode", "OP");
    subjectAssignment->LinkEndChild(subject);

    TiXmlElement *trialSubject = new TiXmlElement("trialSubject");
    trialSubject->SetAttribute("classCode", "RESBJ");
    subject->LinkEndChild(trialSubject);
    
    if (strlen(hdr->Patient.Id)>0) {	
    	TiXmlElement *trialSubjectId = new TiXmlElement("id");
    	trialSubjectId->SetAttribute("extension", hdr->Patient.Id);
    	trialSubject->LinkEndChild(trialSubjectId);
    }

    TiXmlElement *trialSubjectDemographicPerson = new TiXmlElement("subjectDemographicPerson");
    trialSubjectDemographicPerson->SetAttribute("classCode", "PSN");
    trialSubjectDemographicPerson->SetAttribute("determinerCode", "INSTANCE");
    trialSubject->LinkEndChild(trialSubjectDemographicPerson);

	if (VERBOSE_LEVEL>7) fprintf(stdout,"933\n");

    if (strlen(hdr->Patient.Name)>0)
    if (!hdr->FLAG.ANONYMOUS) 
    {	
	TiXmlElement *subjectDemographicPersonName = new TiXmlElement("name");
    	TiXmlText *nameText = new TiXmlText(hdr->Patient.Name);
    	subjectDemographicPersonName->LinkEndChild(nameText);
    	trialSubjectDemographicPerson->LinkEndChild(subjectDemographicPersonName);
    }
    
    TiXmlElement *subjectDemographicPersonGender = new TiXmlElement("administrativeGenderCode");
    if(hdr->Patient.Sex == 1){
	subjectDemographicPersonGender->SetAttribute("code", "M");
	subjectDemographicPersonGender->SetAttribute("displayName", "Male");
    }
    else if(hdr->Patient.Sex == 2){
	subjectDemographicPersonGender->SetAttribute("code", "F");
	subjectDemographicPersonGender->SetAttribute("displayName", "Female");
    }
    else{
	subjectDemographicPersonGender->SetAttribute("code", "UN");
	subjectDemographicPersonGender->SetAttribute("displayName", "Undefined");
    }
    subjectDemographicPersonGender->SetAttribute("codeSystem", "2.16.840.1.113883.5.1");
    subjectDemographicPersonGender->SetAttribute("codeSystemName", "AdministrativeGender");
    trialSubjectDemographicPerson->LinkEndChild(subjectDemographicPersonGender);

	if (hdr->Patient.Birthday>0) {
		t0 = gdf_time2tm_time(hdr->Patient.Birthday);

		// TODO: fixme if "t0->tm_sec"
		sprintf(tmp, "%04d%02d%02d%02d%02d%02d.000", t0->tm_year+1900, t0->tm_mon+1, t0->tm_mday, t0->tm_hour, t0->tm_min, t0->tm_sec);

		TiXmlElement *subjectDemographicPersonBirthtime = new TiXmlElement("birthTime");
		subjectDemographicPersonBirthtime->SetAttribute("value", tmp);
		trialSubjectDemographicPerson->LinkEndChild(subjectDemographicPersonBirthtime);
	}	

	/* write non-standard fields height and weight */
    if (hdr->Patient.Weight) {
    	sprintf(tmp,"%i",hdr->Patient.Weight); 
    	TiXmlElement *subjectDemographicPersonWeight = new TiXmlElement("weight");
    	subjectDemographicPersonWeight->SetAttribute("value", tmp);
    	subjectDemographicPersonWeight->SetAttribute("unit", "kg");
    	trialSubjectDemographicPerson->LinkEndChild(subjectDemographicPersonWeight);
    }
    if (hdr->Patient.Height) {	
    	sprintf(tmp,"%i",hdr->Patient.Height); 
    	TiXmlElement *subjectDemographicPersonHeight = new TiXmlElement("height");
    	subjectDemographicPersonHeight->SetAttribute("value", tmp);
    	subjectDemographicPersonHeight->SetAttribute("unit", "cm");
    	trialSubjectDemographicPerson->LinkEndChild(subjectDemographicPersonHeight);
    }


	if (VERBOSE_LEVEL>7) fprintf(stdout,"937\n");

    TiXmlElement *subjectAssignmentComponentOf = new TiXmlElement("componentOf");
    subjectAssignmentComponentOf->SetAttribute("typeCode", "COMP");
    subjectAssignmentComponentOf->SetAttribute("contextConductionInd", "true");
    subjectAssignment->LinkEndChild(subjectAssignmentComponentOf);

    TiXmlElement *clinicalTrial = new TiXmlElement("clinicalTrial");
    clinicalTrial->SetAttribute("classCode", "CLNTRL");
    clinicalTrial->SetAttribute("moodCode", "EVN");
    subjectAssignmentComponentOf->LinkEndChild(clinicalTrial);

    TiXmlElement *clinicalTrialId = new TiXmlElement("id");
    clinicalTrialId->SetAttribute("root", "GRATZ");
    clinicalTrialId->SetAttribute("extension", "CLINICAL_TRIAL");
    clinicalTrial->LinkEndChild(clinicalTrialId);
    
    TiXmlElement *rootComponent = new TiXmlElement("component");
    rootComponent->SetAttribute("typeCode", "COMP");
    rootComponent->SetAttribute("contextConductionInd", "true");
    root->LinkEndChild(rootComponent);
    
	if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7c 939\n");

    TiXmlElement *series = new TiXmlElement("series");
    series->SetAttribute("classCode", "OBSSER");
    series->SetAttribute("moodCode", "EVN");
    rootComponent->LinkEndChild(series);
    
    TiXmlElement *seriesCode = new TiXmlElement("code");
    seriesCode->SetAttribute("code", "RHYTHM");
    seriesCode->SetAttribute("seriesCode", "2.16.840.1.113883.5.4");
    series->LinkEndChild(seriesCode);
    
    TiXmlElement *seriesEffectiveTime = new TiXmlElement("effectiveTime");
    TiXmlElement *seriesEffectiveTimeLow = new TiXmlElement("low");
    seriesEffectiveTimeLow->SetAttribute("value", timelow);
    seriesEffectiveTime->LinkEndChild(seriesEffectiveTimeLow);
    TiXmlElement *seriesEffectiveTimeHigh = new TiXmlElement("high");
    seriesEffectiveTimeHigh->SetAttribute("value", timehigh);
    seriesEffectiveTime->LinkEndChild(seriesEffectiveTimeHigh);
    series->LinkEndChild(seriesEffectiveTime);
    
    for(int i=3; i; --i){

	if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7c 950 %i\n",i);

	    TiXmlElement *seriesControlVariable = new TiXmlElement("controlVariable");
	    seriesControlVariable->SetAttribute("typeCode", "CTRLV");
	    series->LinkEndChild(seriesControlVariable);
	    
	    TiXmlElement *CTRLControlVariable = new TiXmlElement("controlVariable");
	    CTRLControlVariable->SetAttribute("classCode", "OBS");
	    seriesControlVariable->LinkEndChild(CTRLControlVariable);
	    
	    TiXmlElement *controlVariableCode = new TiXmlElement("code");
	    CTRLControlVariable->LinkEndChild(controlVariableCode);
    
	    TiXmlElement *controlVariableComponent = new TiXmlElement("component");
	    controlVariableComponent->SetAttribute("typeCode", "COMP");
	    CTRLControlVariable->LinkEndChild(controlVariableComponent);
	    
	    TiXmlElement *componentControlVariable = new TiXmlElement("controlVariable");
	    componentControlVariable->SetAttribute("classCode", "OBS");
	    controlVariableComponent->LinkEndChild(componentControlVariable);
	    
	    TiXmlElement *componentControlVariableCode = new TiXmlElement("code");
	    componentControlVariable->LinkEndChild(componentControlVariableCode);
	    
	    TiXmlElement *componentControlVariableValue = new TiXmlElement("value");
	    componentControlVariableValue->SetAttribute("xsi:type", "PQ");
	    componentControlVariable->LinkEndChild(componentControlVariableValue);
	    
	    switch(i){
		case 3:
		    controlVariableCode->SetAttribute("code", "MDC_ATTR_FILTER_NOTCH");
		    componentControlVariableCode->SetAttribute("code", "MDC_ATTR_NOTCH_FREQ");
		    componentControlVariableValue->SetDoubleAttribute("value", hdr->CHANNEL[0].Notch);
		    break;
		case 2:		    
		    controlVariableCode->SetAttribute("code", "MDC_ATTR_FILTER_LOW_PASS");
		    componentControlVariableCode->SetAttribute("code", "MDC_ATTR_FILTER_CUTOFF_FREQ");
		    componentControlVariableValue->SetDoubleAttribute("value", hdr->CHANNEL[0].LowPass);
		    break;
		case 1:
		    controlVariableCode->SetAttribute("code", "MDC_ATTR_FILTER_HIGH_PASS");
		    componentControlVariableCode->SetAttribute("code", "MDC_ATTR_FILTER_CUTOFF_FREQ");
		    componentControlVariableValue->SetDoubleAttribute("value", hdr->CHANNEL[0].HighPass);
		    break;
	    }
	    
	    controlVariableCode->SetAttribute("codeSystem", "2.16.840.1.113883.6.24");
	    controlVariableCode->SetAttribute("codeSystemName", "MDC");
	    componentControlVariableCode->SetAttribute("codeSystem", "2.16.840.1.113883.6.24");
	    componentControlVariableCode->SetAttribute("codeSystemName", "MDC");
	    componentControlVariableValue->SetAttribute("unit", "Hz");
	    
	    switch(i){
		case 3:
		    controlVariableCode->SetAttribute("displayName", "Notch Filter");
		    componentControlVariableCode->SetAttribute("displayName", "Notch Frequency");
		    break;
		case 2:		    
		    controlVariableCode->SetAttribute("displayName", "Low Pass Filter");
		    componentControlVariableCode->SetAttribute("displayName", "Cutoff Frequency");
		    break;
		case 1:
		    controlVariableCode->SetAttribute("displayName", "High Pass Filter");
		    componentControlVariableCode->SetAttribute("displayName", "Cutoff Frequency");
		    break;
	    }
    }
    
    TiXmlElement *seriesComponent = new TiXmlElement("component");
    seriesComponent->SetAttribute("typeCode", "COMP");
    seriesComponent->SetAttribute("contextConductionInd", "true");
    series->LinkEndChild(seriesComponent);
    
    TiXmlElement *sequenceSet = new TiXmlElement("sequenceSet");
    sequenceSet->SetAttribute("classCode", "OBSCOR");
    sequenceSet->SetAttribute("moodCode", "EVN");
    seriesComponent->LinkEndChild(sequenceSet);
    
    TiXmlElement *sequenceSetComponent = new TiXmlElement("component");
    sequenceSetComponent->SetAttribute("typeCode", "COMP");
    sequenceSetComponent->SetAttribute("contextConductionInd", "true");
    sequenceSet->LinkEndChild(sequenceSetComponent);
    
    TiXmlElement *sequence = new TiXmlElement("sequence");
    sequence->SetAttribute("classCode", "OBS");
    sequence->SetAttribute("moodCode", "EVN");    
    sequenceSetComponent->LinkEndChild(sequence);
    
    TiXmlElement *sequenceCode = new TiXmlElement("code");
    sequenceCode->SetAttribute("code", "TIME_ABSOLUTE");
    sequenceCode->SetAttribute("codeSystem", "2.16.840.1.113883.6.24");
    sequence->LinkEndChild(sequenceCode);

    TiXmlElement *sequenceValue = new TiXmlElement("value");
    sequenceValue->SetAttribute("xsi:type", "GLIST_TS");
    sequence->LinkEndChild(sequenceValue);

    TiXmlElement *valueHead = new TiXmlElement("head");
    valueHead->SetAttribute("value", timelow);
    valueHead->SetAttribute("unit", "s");   // TODO: value is date/time of the day - unit=[s] does not make sense 
    sequenceValue->LinkEndChild(valueHead);

    TiXmlElement *valueIncrement = new TiXmlElement("increment");
    valueIncrement->SetDoubleAttribute("value", 1/hdr->SampleRate);
    valueIncrement->SetAttribute("unit", "s");
    sequenceValue->LinkEndChild(valueIncrement);

    TiXmlText *digitsText;
	float*Dat;
	char *S;
	char *pS;

#ifdef NO_BI
    size_t bi = 0; 
#endif
    for(int i=0; i<hdr->NS; ++i)
    if (hdr->CHANNEL[i].OnOff)
    {

	if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7c 960 %i\n",i);


	sequenceSetComponent = new TiXmlElement("component");
	sequenceSetComponent->SetAttribute("typeCode", "COMP");
	sequenceSetComponent->SetAttribute("contextConductionInd", "true");
	sequenceSet->LinkEndChild(sequenceSetComponent);

	sequence = new TiXmlElement("sequence");
	sequence->SetAttribute("classCode", "OBS");
	sequence->SetAttribute("moodCode", "EVN");
	sequenceSetComponent->LinkEndChild(sequence);

	sequenceCode = new TiXmlElement("code");
	
	if (hdr->CHANNEL[i].LeadIdCode) {
		strcpy(tmp,"MDC_ECG_LEAD_");				// Flawfinder: ignore
		strcat(tmp,LEAD_ID_TABLE[hdr->CHANNEL[i].LeadIdCode]);	// Flawfinder: ignore
	}
	else 
		strcpy(tmp,hdr->CHANNEL[i].Label);			// Flawfinder: ignore

	sequenceCode->SetAttribute("code", tmp);

	sequenceCode->SetAttribute("codeSystem", "2.16.840.1.113883.6.24");
	sequenceCode->SetAttribute("codeSystemName", "MDC");
	sequence->LinkEndChild(sequenceCode);
    
	sequenceValue = new TiXmlElement("value");
	sequenceValue->SetAttribute("xsi:type", "SLIST_PQ");
	sequence->LinkEndChild(sequenceValue);

	valueHead = new TiXmlElement("origin");
	valueHead->SetDoubleAttribute("value", hdr->CHANNEL[i].Off);
	// valueHead->SetDoubleAttribute("value", 0);
	valueHead->SetAttribute("unit", PhysDim3(hdr->CHANNEL[i].PhysDimCode));
	sequenceValue->LinkEndChild(valueHead);

	valueIncrement = new TiXmlElement("scale");
	valueIncrement->SetDoubleAttribute("value", hdr->CHANNEL[i].Cal);
	//valueIncrement->SetDoubleAttribute("value", 1);
	valueIncrement->SetAttribute("unit", PhysDim3(hdr->CHANNEL[i].PhysDimCode));
	sequenceValue->LinkEndChild(valueIncrement);

	TiXmlElement *valueDigits = new TiXmlElement("digits");
	sequenceValue->LinkEndChild(valueDigits);

	if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7c [967] %i %f\n",i,*(float*)(hdr->AS.rawdata + hdr->CHANNEL[i].bi));

#ifndef NO_BI
	Dat=(float*)(hdr->AS.rawdata + hdr->CHANNEL[i].bi);
#else
	Dat=(float*)(hdr->AS.rawdata + bi);
#endif

	//size_t sz = GDFTYP_BITS[hdr->CHANNEL[i].GDFTYP]>>3;

	S=new char[32*hdr->CHANNEL[i].SPR];
	S[0]=0;
	pS=S;

	for (unsigned int j=0; j<hdr->CHANNEL[i].SPR; ++j) {
		if (VERBOSE_LEVEL>8) fprintf(stdout,"hl7c 969: %i %i %f \n",i, j, Dat [j]);
		pS+=sprintf(pS,"%g ",Dat[j]);
	}
#ifdef NO_BI
	bi += hdr->CHANNEL[i].SPR*sz;
	if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7c 970 %i %i \n%s \n",i, bi, S);
#else
	if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7c 970 %i %i \n%s \n",i, hdr->CHANNEL[i].bi, S);
#endif

	digitsText = new TiXmlText(S);
	valueDigits->LinkEndChild(digitsText);
	delete []S;
    }

	int status = doc.SaveFile(hdr->FileName, (char)hdr->FILE.COMPRESSION);
//	doc.SaveFile(hdr);
	if (VERBOSE_LEVEL>7) fprintf(stdout,"hl7c 989  (%i)\n",status);

#endif 

    return(0);
};
