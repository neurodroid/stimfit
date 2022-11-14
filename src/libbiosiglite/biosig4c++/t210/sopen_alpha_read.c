/*

    $Id: sopen_alpha_read.c,v 1.2 2009-02-12 16:15:17 schloegl Exp $
    Copyright (C) 2005,2006,2007,2008,2009 Alois Schloegl <a.schloegl@ieee.org>

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


#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "../biosig.h"

#define min(a,b)        (((a) < (b)) ? (a) : (b))
#define max(a,b)        (((a) > (b)) ? (a) : (b))

EXTERN_C void sopen_alpha_read(HDRTYPE* hdr) {	
/*
	this function will be called by the function SOPEN in "biosig.c"

	Input: 
		char* Header	// contains the file content
		
	Output: 
		HDRTYPE *hdr	// defines the HDR structure accoring to "biosig.h"
*/	
    	size_t	 	count;
    	unsigned int 	k;
    	const char	*FileName = hdr->FileName; 

		fprintf(stdout,"Warning: support for alpha format is just experimental.\n"); 
		
		char* fn = (char*)malloc(strlen(hdr->FileName)+15);
		strcpy(fn,hdr->FileName); 	// Flawfinder: ignore
		
		const size_t bufsiz = 4096; 
		char buf[bufsiz]; 

		// alpha.alp  cal_res digin  digvidtc  eog  marker  measure  mkdef  montage  rawdata  rawhead  report.txt  r_info  sleep
		
		const char *f2 = "alpha.alp";		
		char *tmpstr   = strrchr(fn,FILESEP); 
		if (tmpstr) 	strcpy(tmpstr+1,f2);  	// Flawfinder: ignore
		else 	    	strcpy(fn,f2);  	// Flawfinder: ignore

		FILE *fid = fopen(fn,"r"); count  = fread(buf,1,bufsiz-1,fid); fclose(fid); buf[count]=0;	// terminating 0 character 		
		char *t   = strtok(buf,"\xA\xD");
		while (t) {

if (VERBOSE_LEVEL>7) fprintf(stdout,"0: %s \n",t); 

			if (!strncmp(t,"Version = ",9))
				hdr->VERSION = atof(t+9);
			else if (!strncmp(t,"Id = ",4)) {
				strncpy(hdr->ID.Manufacturer._field,t+5,MAX_LENGTH_MANUF);
				hdr->ID.Manufacturer.Name = hdr->ID.Manufacturer._field;
			}
			t = strtok(NULL,"\xA\xD");
		}

		f2 = "rawhead";
		if (tmpstr) 	strcpy(tmpstr+1,f2); 	 	// Flawfinder: ignore
		else 	    	strcpy(fn,f2);  		// Flawfinder: ignore
				
		int Notch = 0; 		
		int Bits  = 0; 		
		double DigMax=0, DigMin=0; 
		uint16_t gdftyp = 0; 
		int ns = 0;
		fid = fopen(fn,"r"); count  = fread(buf,1,bufsiz-1,fid); fclose(fid); buf[count]=0;	// terminating 0 character 		
		t   = strtok(buf,"\xA\xD");
		char STATUS = 1; 
		uint32_t *ChanOrder=NULL; 
		char **ChanType = NULL;
		while ((t!=NULL) && (STATUS<9)) {
			char *t1 = strchr(t,'=');

if (VERBOSE_LEVEL>7) fprintf(stdout,"<%6.2f> %i- %s | %s\n",hdr->VERSION, STATUS,t,t1); 
				
			if (t1) {
				t1[-1] = 0; t1++;
				
				if (STATUS == 1) {
					if (!strcmp(t,"Version")) 
						hdr->VERSION = atof(t1);
					else if (!strcmp(t,"BitsPerValue")) {
						Bits = atoi(t1);
						switch (Bits) {
						case 12: gdftyp = 255+12; 
						//	hdr->FILE.LittleEndian = 0;
						        DigMax =  (1<<(Bits-1))-1;
						        DigMin = -(1<<(Bits-1));
							break;
						case 16: gdftyp = 3; 
						        DigMax =  32752.0;  //!!! that's the maximum value found in alpha-trace files 
						        DigMin = -32736.0;  //!!! that's the minimum value found in alpha-trace files
						        break; 
						case 32: gdftyp = 5; break; 
						        DigMax =  (1<<(Bits-1))-1;
						        DigMin = -(1<<(Bits-1));
						}
					}	
					else if (!strcmp(t,"ChanCount")) {
						hdr->NS = atoi(t1);
						hdr->CHANNEL = (CHANNEL_TYPE*)realloc(hdr->CHANNEL, hdr->NS*sizeof(CHANNEL_TYPE));
						ChanOrder = (uint32_t*)calloc(hdr->NS,sizeof(uint32_t)*2);
						ChanType  = (char**)calloc(hdr->NS,sizeof(char*));
						
					}
					else if (!strcmp(t,"SampleFreq"))
						hdr->SampleRate = atof(t1);
					else if (!strcmp(t,"SampleCount")) {
						hdr->NRec = atoi(t1);
						hdr->SPR  = 1; 
					}
					else if (!strcmp(t,"NotchFreq")) 
						Notch = atof(t1);
					else if (!strcmp(t,"DispFlags") && (hdr->VERSION < 411.89)) 
						STATUS = 2;
					else if (!strcmp(t,"Sec2Marker") && (hdr->VERSION > 411.89)) 
						STATUS = 2;
				}		

				else if (STATUS == 2) {
					if (ns>=hdr->NS) {
						ns = 0; 
						STATUS = 3;
					}
					else {	
						CHANNEL_TYPE *hc = hdr->CHANNEL+ns; 
						hc->GDFTYP  = gdftyp; 
						hc->Notch   = Notch; 
						hc->LeadIdCode = 0; 
						hc->SPR = hdr->SPR; 
						//hc->bi8 = GDFTYP_BITS[gdftyp]*ns; 
						hc->DigMax  = DigMax;
						hc->DigMin  = DigMin;
						hc->OnOff   = 1; 
						hc->Cal     = 1.0; 
						hc->Off     = 0.0; 
						hc->PhysMax = hc->DigMax; 
						hc->PhysMin = hc->DigMin; 
						hc->Transducer[0] = 0;
					
						strncpy(hc->Label, t, MAX_LENGTH_LABEL+1);
						char* t2= strchr(t1,',');
						t2[0] = 0; while (isspace((++t2)[0]));
						char* t3= strchr(t2,',');
						t3[0] = 0; while (isspace((++t3)[0]));
						char* t4= strchr(t3,',');
						t4[0] = 0; while (isspace((++t4)[0]));

						ChanOrder[ns*2] = atoi(t2); 
						ChanOrder[ns*2+1] = ns; 
						ChanType[ns] = t3; 
						ns++;
					}	
				}	
				else if (STATUS == 3) {
					// decode information (filters, PhysDim, etc.) and assign to corresponding channels. 
					char *pd = NULL;
					float tmp1, tmp2, HighPass, LowPass;
#if !defined __STDC_VERSION__ || __STDC_VERSION__ < 199901L
					sscanf(t1, "%f, %f, %f, %f, %as", &HighPass,&LowPass, &tmp1,&tmp2, &pd); 
#else
					sscanf(t1, "%f, %f, %f, %f, %ms", &HighPass,&LowPass, &tmp1,&tmp2, &pd);
#endif
					strrchr(pd,',')[0]=0;
					if (!strcmp(pd,"%%")) pd[1]=0; 
					uint16_t pdc = PhysDimCode(pd); 
					if (pd) free(pd); 

					char flag = 0; 
					for (k=0; k < hdr->NS; k++) {
						if ((ChanType[k]!=NULL) && !strcmp(t,ChanType[k])) {
							CHANNEL_TYPE *hc = hdr->CHANNEL+k; 
							hc->PhysDimCode = pdc; 
							//strcpy(hc->PhysDim,pd); 
							hc->LowPass = LowPass;
							hc->HighPass = HighPass;
							ChanType[k] = NULL; 
						}
						if (ChanType[k] != NULL) flag = 1;	// not done yet 
					}	
					if (!flag) STATUS = 99; 	// done with channel definition
				}	
			}	
			t = strtok(NULL,"\xA\xD");
		}
		hdr->AS.bpb8 = (GDFTYP_BITS[gdftyp]*hdr->NS);
//		hdr->AS.bpb = (GDFTYP_BITS[gdftyp]*hdr->NS)>>3;		// do not rely on this, because some bits can get lost

		// sort channels 
		qsort(ChanOrder,hdr->NS,2*sizeof(uint32_t),&u32cmp);
		for (k=0; k<hdr->NS; k++) {
			hdr->CHANNEL[ChanOrder[2*k+1]].bi8 = GDFTYP_BITS[gdftyp]*k; 
			hdr->CHANNEL[ChanOrder[2*k+1]].bi  = (GDFTYP_BITS[gdftyp]*k)>>3; 
		}
		free(ChanOrder); 		
		free(ChanType); 		

		
		f2 = "cal_res";
		if (tmpstr) 	strcpy(tmpstr+1,f2); 	// Flawfinder: ignore
		else 	    	strcpy(fn,f2); 		// Flawfinder: ignore
				
		fid = fopen(fn,"r"); 
		if (fid!=NULL) {
			if (VERBOSE_LEVEL>7) fprintf(stdout,"cal_res: \n"); 

			count  = fread(buf,1,bufsiz-1,fid); fclose(fid); buf[count]=0;	// terminating 0 character 		
			t   = strtok(buf,"\xA\xD");
			t   = strtok(NULL,"\xA\xD");	// skip lines 1 and 2 
			/*
			char label[MAX_LENGTH_LABEL+1];
			char flag[MAX_LENGTH_LABEL+1];
			double cal,off; 
			*/
			char *t0,*t1,*t2,*t3; 
			unsigned n=0; 	// 		
			for (k=0; max(k,n)<hdr->NS; k++) { 
				t = strtok(NULL,"\xA\xD");
			
				if (t==NULL) {	
					fprintf(stderr,"Warning SOPEN(alpha): scaling coefficients not defined for all channels\n");
					break;
				}

				// strncpy(hc->Label,t,min(strcspn(t," =,"),MAX_LENGTH_LABEL));
				t0 = strchr(t,'=');t0[0]=0;t0++;
				int ix = strlen(t)-1; 
				while ((ix>0) && isspace(t[ix])) t[ix--] = 0;	// remove trailing spaces
				t1 = strchr(t0,',');t1[0]=0;t1++;
				t2 = strchr(t1,',');t2[0]=0;t2++;
				t3 = strchr(t2,',');t3[0]=0;t3++;
	
				n  = atoi(t);	// n==0 if label is provided, n>0 if channel number is provided

/*				does not work because ambiguous labels are used in rawhead and cal_res (e.g. T3 and T7)
				if (!n) for (n=0; n<hdr->NS; n++) {
					if (!strcmp(hdr->CHANNEL[n].Label,t))
					{ 	n++; 
						break; 
					}
				}
*/				

				if (VERBOSE_LEVEL>7) fprintf(stdout,"cal_res: %i %i <%s> %s %s %s\n",k,n,t,t1,t2,t3); 

				CHANNEL_TYPE *hc = hdr->CHANNEL + (n>0 ? n-1 : k); // channel can be denoted by label or number  	
				hc->Cal = atof(t1);
				hc->Off = 0; 
						
				if (VERBOSE_LEVEL>7) fprintf(stdout,"  <%s>   %s = ###, %f, %f\n", t1,hc->Label,hc->Cal,hc->Off);

				hc->PhysMax = (hc->DigMax - hc->Off) * hc->Cal;
				hc->PhysMin = (hc->DigMin - hc->Off) * hc->Cal;
				hc->XYZ[0]=0;
				hc->XYZ[1]=0;
				hc->XYZ[2]=0;
			}
		}
		
		f2 = "r_info";
		if (tmpstr) 	strcpy(tmpstr+1,f2);	// Flawfinder: ignore
		else 	    	strcpy(fn,f2);		// Flawfinder: ignore
				
		fid = fopen(fn,"r"); 
		if (fid!=NULL) {

			if (VERBOSE_LEVEL>7) fprintf(stdout,"r_info: \n"); 

			count  = fread(buf,1,bufsiz-1,fid); fclose(fid); buf[count]=0;	// terminating 0 character 		
			struct tm T;
			t   = strtok(buf,"\xA\xD");
			t   = strtok(NULL,"\xA\xD");	// skip line 1 
			while (t!=NULL) {
				char *t1 = strchr(t,'=');
				t1[0] = 0;
				while (isspace((++t1)[0])); 
				for (k=strlen(t); (k>0) && isspace(t[--k]); t[k]=0);
				for (k=strlen(t1); (k>0) && isspace(t1[--k]); t1[k]=0);

			if (VERBOSE_LEVEL>7) fprintf(stdout,"r_info: %s = %s\n",t,t1); 

				if (0) {}
				else if (!strcmp(t,"RecId")) 
					strncpy(hdr->ID.Recording,t1,MAX_LENGTH_RID);
				else if (!strcmp(t,"RecDate")) {
					sscanf(t1,"%02d.%02d.%04d",&T.tm_mday,&T.tm_mon,&T.tm_year);
					T.tm_year -=1900;
					T.tm_mon -=1;
				}	
				else if (!strcmp(t,"RecTime"))
					sscanf(t1,"%02d.%02d.%02d",&T.tm_hour,&T.tm_min,&T.tm_sec);
				else if (!strcmp(t,"TechSal")) {
					if (hdr->ID.Technician) free(hdr->ID.Technician);
					hdr->ID.Technician = strdup(t1);
				}
				else if (!strcmp(t,"TechTitle") || !strcmp(t,"TechLast") || !strcmp(t,"TechFirst")) {
					size_t l0 = strlen(hdr->ID.Technician);
					size_t l1 = strlen(t1);
					hdr->ID.Technician = (char*)realloc(hdr->ID.Technician,l0+l1+2);
					hdr->ID.Technician[l0] = ' ';
					strcpy(hdr->ID.Technician+l0+1, t1);		// Flawfinder: ignore
				}	

				t = strtok(NULL,"\xA\xD");
			}
			hdr->T0 = tm_time2gdf_time(&T);
		}
		
		f2 = "marker";
		if (tmpstr) 	strcpy(tmpstr+1,f2); 		// Flawfinder: ignore
		else 	    	strcpy(fn,f2); 			// Flawfinder: ignore
		fid = fopen(fn,"r"); 
		if (fid != NULL) {
			size_t n,N;
			N=0; n=0;  
			while (!feof(fid)) {
				hdr->AS.auxBUF = (uint8_t*) realloc(hdr->AS.auxBUF,N+bufsiz+1);
				N += fread(hdr->AS.auxBUF+N, 1, bufsiz, fid); 
			}
			fclose(fid); 
			hdr->AS.auxBUF[N] = 0;	// terminating 0 character 		

			N = 0; 
			t = (char*)hdr->AS.auxBUF+strcspn((char*)hdr->AS.auxBUF,"\xA\xD");
			t = t+strspn(t,"\xA\xD");	// skip lines 1 and 2 
			while (t[0]) {
				char*  t1 = t;	
				size_t l1 = strcspn(t1,"="); 
				size_t p2 = strspn(t1+l1,"= ")+l1; 
				char*  t2 = t+p2;
				size_t l2 = strcspn(t2," ,"); 
				size_t p3 = strspn(t2+l2," ,")+l2; 
				char*  t3 = t2+p3;
				size_t l3 = strcspn(t3," ,"); 
				size_t p4 = strspn(t3+l3," ,")+l3; 
				char*  t4 = t3+p4;
				size_t l4 = strcspn(t4,"\xA\xD"); 
				size_t p5 = strspn(t4+l4,"\xA\xD")+l4;
				t1[l1] = 0;
				while (isspace(t1[--l1])) t1[l1]=0;
				t2[l2] = 0;
				t3[l3] = 0;
				t4[l4] = 0;
				t = t4 + p5;
				
				if (n+1 >= N) {
					const size_t sz = 100;
					hdr->EVENT.TYP = (typeof(hdr->EVENT.TYP)) realloc(hdr->EVENT.TYP,(N+sz)*sizeof(*hdr->EVENT.TYP));
					hdr->EVENT.POS = (typeof(hdr->EVENT.POS)) realloc(hdr->EVENT.POS,(N+sz)*sizeof(*hdr->EVENT.POS));
					N += sz; 
				}	

				hdr->EVENT.POS[n] = atol(t3);
				if (!strcmp(t1,"REC")) 
					hdr->EVENT.TYP[n] = 0x7ffe;
				else if (!strcmp(t1,"MON"))
					hdr->EVENT.TYP[n] = 0;
				else if (!strcmp(t1,"TXT"))
					FreeTextEvent(hdr, n, t4); 
				else
					FreeTextEvent(hdr, n, t1); 
						
				if (!strcmp(t2,"off")) 
					hdr->EVENT.TYP[n] |= 0x8000;	

//fprintf(stdout,"#%u, 0x%04x,%u | t1=<%s> = t2=<%s>, t3=<%s>, t4=<%s>\n",n,hdr->EVENT.TYP[n],hdr->EVENT.POS[n],t1,t2,t3,t4);

				n++;
//				t = strtok(NULL,"\xA\xD");

//fprintf(stdout," <%s>\n",t1);
			}	
			hdr->EVENT.N = n; 
			hdr->EVENT.SampleRate = hdr->SampleRate; 
//			convert2to4_eventtable(hdr);
		}

		tmpstr    = strrchr(fn,FILESEP); 
		tmpstr[0] = 0;	
		tmpstr    = strrchr(fn,FILESEP); 
		f2 = "s_info";
		if (tmpstr) 	strcpy(tmpstr+1,f2); 		// Flawfinder: ignore
		else 	    	strcpy(fn,f2); 			// Flawfinder: ignore
				
		fid = fopen(fn,"r"); 
		if (fid!=NULL) {
			count  = fread(buf,1,bufsiz-1,fid); fclose(fid); buf[count]=0;	// terminating 0 character 
			char *Lastname = NULL;		
			char *Firstname = NULL;		
			struct tm T;
			t   = strtok(buf,"\xA\xD");
			t   = strtok(NULL,"\xA\xD");	// skip line 1 
			while (t!=NULL) {
				char *t1 = strchr(t,'=');
				t1[0] = 0;
				while (isspace((++t1)[0])); 
				for (k=strlen(t); (k>0) && isspace(t[--k]); t[k]=0);
				for (k=strlen(t1); (k>0) && isspace(t1[--k]); t1[k]=0);

			if (VERBOSE_LEVEL>7) fprintf(stdout,"s_info: <%s> = <%s>\n",t,t1); 

				if (0) {}
				else if (!strcmp(t,"SubjId")) 
					strncpy(hdr->Patient.Id,t1,MAX_LENGTH_PID); 
				else if (!strcmp(t,"Gender")) 
					switch (t1[0]) {
					case 'm':
					case 'M':
						hdr->Patient.Sex = 1; break; 	
					case 'w':
					case 'W':
					case 'f':
					case 'F':
						hdr->Patient.Sex = 2; break; 	
					default: 	
						hdr->Patient.Sex = 0; break; 	
					}	
				else if (!strcmp(t,"Handedness")) 
					switch (t1[0]) {
					case 'r':
					case 'R':
						hdr->Patient.Handedness = 1; break; 	
					case 'l':
					case 'L':
						hdr->Patient.Handedness = 2; break; 	
					default: 	
						hdr->Patient.Handedness = 0; break; 	
					}	
				else if (!strcmp(t,"Size")) 
					hdr->Patient.Height = atof(t1); 
				else if (!strcmp(t,"Weight")) 
					hdr->Patient.Weight = atof(t1); 
				else if (!strcmp(t,"FirstName")) {
					Firstname = t1;
				}	
				else if (!strcmp(t,"LastName")) {
					Lastname = t1;
				}
				else if (!strcmp(t,"BirthDay")) {
					int c = sscanf(t1,"%02d.%02d.%04d",&T.tm_mday,&T.tm_mon,&T.tm_year);
					T.tm_year -=1900;
					T.tm_mon -=1;
					T.tm_hour =12;
					T.tm_min =0;
					T.tm_sec =0;
					if (c > 2) hdr->Patient.Birthday = tm_time2gdf_time(&T);
				}	
				t = strtok(NULL,"\xA\xD");
			}	

			size_t l0 = strlen(Firstname);
			size_t l1 = strlen(Lastname);
			if (l0+l1+1 <= MAX_LENGTH_NAME) {
				strcpy(hdr->Patient.Name, Firstname);			// Flawfinder: ignore
				hdr->Patient.Name[l0] = ' ';
				strcpy(hdr->Patient.Name + l0 + 1, Lastname);		// Flawfinder: ignore
			} else 
				strncpy(hdr->Patient.Name, Lastname, MAX_LENGTH_NAME+1); 	// Flawfinder: ignore
			
		}

		strcpy(fn,hdr->FileName); 		// Flawfinder: ignore
		tmpstr   = strrchr(fn,FILESEP);
		f2 = "rawdata";
		if (tmpstr) 	strcpy(tmpstr+1,f2);	// Flawfinder: ignore
		else 	    	strcpy(fn,f2);		// Flawfinder: ignore
				
		if (VERBOSE_LEVEL>7) fprintf(stdout,"rawdata11: %s \n",f2); 

		hdr->FileName = fn; 
		ifopen(hdr,"r"); 

		if (VERBOSE_LEVEL>7) fprintf(stdout,"rawdata22: %s open=%i\n",f2,hdr->FILE.OPEN); 
		if (hdr->FILE.OPEN) {
			int16_t a[3];
			ifread(a, 2, 3, hdr); 
			hdr->VERSION = a[0]; 
			hdr->HeadLen = 6; 
			
			switch (a[2]) {
			case 12: gdftyp = 255+12; break;
			case 16: gdftyp = 3; break; 
			case 32: gdftyp = 5; break; 
			}
			for (k=a[1]; k<hdr->NS; k++) 
				hdr->CHANNEL[k].OnOff = 0;  
			for (k=0; k<hdr->NS; k++) {
				hdr->CHANNEL[k].GDFTYP = gdftyp; 
			}

			hdr->AS.bpb = (GDFTYP_BITS[gdftyp]*a[1])>>3; 
			hdr->FILE.POS = 0;
			
			size_t len = (GDFTYP_BITS[gdftyp]*a[1]*hdr->NRec*hdr->SPR)>>3;	
			if ((GDFTYP_BITS[gdftyp]*a[1]) & 0x07) {
				/* hack: if SPR*NS*bits are not a multiple of bytes, 
				   hdr->AS.bpb would be non-integer causing some problems in SREAD reading the correct number of bytes. 
				   This hack makes sure that all data is loaded.  	
				*/
				len++;
			}	
			hdr->AS.rawdata = (uint8_t*) realloc(hdr->AS.rawdata, len);
			size_t count    = ifread(hdr->AS.rawdata,1,len,hdr);
			hdr->AS.first   = 0; 
			hdr->AS.length  = (count<<3)/(GDFTYP_BITS[gdftyp]*a[1]);
		}

		if (VERBOSE_LEVEL>7) fprintf(stdout,"rawdata55: %s c=%i [%i, %i] sizeof(CHAN)=%i\n",fn,(int)count,(int)hdr->AS.first,(int)hdr->AS.length,(int)sizeof(hdr->CHANNEL[0])); 

		free(fn);     
		hdr->FileName = FileName; 		
}

