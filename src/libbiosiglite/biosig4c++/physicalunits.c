/*

    Copyright (C) 2005-2019 Alois Schloegl <alois.schloegl@gmail.com>
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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "physicalunits.h"

#ifdef HAVE_PTHREAD
// This is optional, because so far there are no multi-threaded applications for libbiosig.
#include <pthread.h>
#endif

/* physical units are defined in
 prEN ISO 11073-10101 (Nov 2003)
 Health Informatics - Point-of-care medical device communications - Part 10101:Nomenclature
 (ISO/DIS 11073-10101:2003)
 Table A.6.1: Table of Decimal Factors

 CEN/TC251/PT40 2001
 File Exchange Format for Vital Signs - Annex A
 Table A.4.1: Table of Decimal Factors	const double scale[32] =
*/

const struct PhysDimIdx
	{
		const uint16_t	idx;
		const char*	PhysDimDesc;
	} _physdim[] = {
#include "units.i"
	{0xffff,  "end-of-table" },
} ;

/*
	compare strings, accept bit7=1
 */
int strcmp8(const char* str1, const char* str2)
{
	unsigned int k=0;
	int r;
	r = str1[k] - str2[k];
	while (r==0 && str1[k]!='\0' && str2[k]!='\0') {
		k++;
		r = str1[k] - str2[k];
	}
	return(r);
}

const char* PhysDimFactor[] = {
	"","da","h","k","M","G","T","P",	//  0..7
	"E","Z","Y","#","#","#","#","#",	//  8..15
	"d","c","m","u","n","p","f","a",	// 16..23
	"z","y","#","#","#","#","#","#",	// 24..31
	"\xB5"	//hack for "µ" = "u"		// 32
	};


#ifndef NAN
# define NAN (0.0/0.0)        /* used for encoding of missing values */
#endif
double PhysDimScale(uint16_t PhysDimCode)
{
// converting PhysDimCode -> scaling factor

	const double scale[] =
	{ 1e+0, 1e+1, 1e+2, 1e+3, 1e+6, 1e+9,  1e+12, 1e+15,	//  0..7
	  1e+18,1e+21,1e+24,NAN,  NAN,  NAN,   NAN,   NAN, 	//  8..15
	  1e-1, 1e-2, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18, 	// 16..23
	  1e-21,1e-24,NAN,  NAN,  NAN,  NAN,   NAN,   NAN,	// 24..31
	  1e-6	// hack for "µ" = "u" 				// 32
	  };

	return (scale[PhysDimCode & 0x001f]);
}

// DEPRECATED: USE INSTEAD  PhysDim3(uint16_t PhysDimCode)
__attribute__ ((deprecated)) char* PhysDim(uint16_t PhysDimCode, char *PhysDim)
{
#define MAX_LENGTH_PHYSDIM      20	// DEPRECATED - DO NOT USE
	// converting PhysDimCode -> PhysDim
	uint16_t k=0;
	size_t l2 = strlen(PhysDimFactor[PhysDimCode & 0x001F]);	
	memcpy(PhysDim,PhysDimFactor[PhysDimCode & 0x001F],l2);

	PhysDimCode &= ~0x001F;
	for (k=0; _physdim[k].idx<0xffff; k++)
	if (PhysDimCode == _physdim[k].idx) {
		strncpy(PhysDim+l2, _physdim[k].PhysDimDesc, MAX_LENGTH_PHYSDIM+1-l2);
		PhysDim[MAX_LENGTH_PHYSDIM]='\0';
		break;
	}
	return(PhysDim);
#undef MAX_LENGTH_PHYSDIM
}

char* PhysDim2(uint16_t PhysDimCode)
{
	// converting PhysDimCode -> PhysDim
	uint16_t k  = 0;
	uint16_t l2 = strlen(PhysDimFactor[PhysDimCode & 0x001F]);	

	for (k = 0; _physdim[k].idx < 0xffff; k++)
	if  ( (PhysDimCode & ~0x001F) == _physdim[k].idx) {
		char *PhysDim = (char*)malloc(l2 + 1 + strlen(_physdim[k].PhysDimDesc));
		if (PhysDim==NULL) return (NULL); 
		memcpy(PhysDim, PhysDimFactor[PhysDimCode & 0x001F], l2);
		strcpy(PhysDim+l2, _physdim[k].PhysDimDesc);
		return(PhysDim);
	}
	return(NULL);
}

uint16_t PhysDimCode(const char* PhysDim0)
{
// converting PhysDim -> PhysDimCode
	/* converts Physical dimension into 16 bit code */
	uint16_t k1, k2;
	char s[80];
	char *s1;

	if (PhysDim0==NULL) return(0);
	while (isspace(*PhysDim0)) PhysDim0++;	// remove leading whitespace
	if (strlen(PhysDim0)==0) return(0);

	// greedy search - check all codes 0..65535
	for (k1=0; k1<33; k1++)
	if (strncmp(PhysDimFactor[k1],PhysDim0,strlen(PhysDimFactor[k1]))==0 && (PhysDimScale(k1)>0.0))
	{ 	// exclude if beginning of PhysDim0 differs from PhysDimFactor and if NAN
		strncpy(s, PhysDimFactor[k1],3);
		s1 = s+strlen(s);
		for (k2=0; _physdim[k2].idx < 0xffff; k2++) {
			strncpy(s1, _physdim[k2].PhysDimDesc, 77);
			if (strcmp8(PhysDim0, s)==0) {
		 		if (k1==32) k1 = 19;		// hack for "Âµ" = "u"
				return(_physdim[k2].idx+k1);
			}
		}
	}
	return(0);
}

/*------------------------------------------------------------------------
 *	Table of Physical Units
 * 
 * This part can be better optimized with a more sophisticated hash table 
 
 * PhysDimTable depends only on constants, defined in units.csv/units.i;
   however, the table is only initialized upon usage. 

 * These functions are thread safe except for the call to PhysDim2 which updates
   the table (it does a malloc). Everything else is just read operation, and 
   the content is defined only by PhysDimFactor and PhysDimIdx, which are constant. 

 * The implementation does not seem straightforward, but it should be faster
   to store already computed strings in a table, rather then recomputing them, 
   again and again. 

 *------------------------------------------------------------------------*/

#define PHYS_DIM_TABLE_SIZE 0x10000
static char *PhysDimTable[PHYS_DIM_TABLE_SIZE];
static char FlagInit_PhysDimTable = 0; 
#ifdef _PTHREAD_H
pthread_mutex_t mutexPhysDimTable = PTHREAD_MUTEX_INITIALIZER;
#endif


/***** 
	Release allocated memory 
*****/
void ClearPhysDimTable(void) {
#ifdef _PTHREAD_H
	pthread_mutex_lock(&mutexPhysDimTable);
#endif
	unsigned k = 0;
	while (k < PHYS_DIM_TABLE_SIZE) {
		char *o = PhysDimTable[k++];
		if (o != NULL) free(o); 
	}
	FlagInit_PhysDimTable = 0; 
#ifdef _PTHREAD_H
	pthread_mutex_unlock(&mutexPhysDimTable);
#endif
}

/***** 
	PhysDim3 returns the text representation of the provided 16bit code 
 *****/
const char* PhysDim3(uint16_t PhysDimCode) {
#ifdef _PTHREAD_H
	pthread_mutex_lock(&mutexPhysDimTable); 
#endif 
	if (!FlagInit_PhysDimTable) {
		memset(PhysDimTable, 0, PHYS_DIM_TABLE_SIZE * sizeof(char*));
		atexit(&ClearPhysDimTable);
		FlagInit_PhysDimTable = 1; 
	}

	char **o = PhysDimTable+PhysDimCode; 

	if (*o==NULL) *o = PhysDim2(PhysDimCode);
#ifdef _PTHREAD_H
	pthread_mutex_unlock(&mutexPhysDimTable);
#endif
	return( (const char*) *o);
}



/****************************************************************************/
/**                                                                        **/
/**                               EOF                                      **/
/**                                                                        **/
/****************************************************************************/


#ifdef TEST_PHYSDIMTABLE_PERFORMANCE
/***********************************
       this is just for testing and is not part of the library
 ***********************************/

#include <sys/time.h>
#include <sys/resource.h>

int main() {

	int k; 
	char *s = NULL;
	struct rusage t[6];
	struct timeval r[3];
/*
#define PhysDim3(k) (PhysDim3(4275))
#define PhysDim2(k) (PhysDim2(4275))
*/	
	int c[6];
	memset(c,0,sizeof(c));

        getrusage(RUSAGE_SELF, &t[0]);

	
	// initialize PhysDimTable 
	for (k=0; k<0x10000; k++) 
		c[0] += (PhysDim3(k)!=NULL); 

        getrusage(RUSAGE_SELF, &t[1]);

	// recall PhysDimTable - many entries are Null, triggering a call to PhysDim2
	for (k=0; k<0x10000; k++)
		c[1] += (PhysDim3(k)!=NULL); 

        getrusage(RUSAGE_SELF, &t[2]);

	// recall PhysDimTable with a fixed code
	for (k=0; k<0x10000; k++)
		c[2] += (PhysDim3(4275)!=NULL); 

        getrusage(RUSAGE_SELF, &t[3]);


	// trivial implementation PhysDimTable 
	for (k=0; k<0x10000; k++) {
		s = PhysDim2(k); 
		if (s!=NULL) {
			free(s);
			c[3]++;
		}
	} 

        getrusage(RUSAGE_SELF, &t[4]);

	// trivial implementation PhysDimTable for a fixed code
	for (k=0; k<0x10000; k++) {
		s = PhysDim2(4275); 
		if (s!=NULL) {
			free(s);
			c[4]++;
		}
	} 

        getrusage(RUSAGE_SELF, &t[5]);

	// trivial implementation PhysDimTable 
	for (k=0; k<0x10000; k++) {
		if ( (k & ~0x001f)==65408) continue; // exclude user-defined code for Bel, because it was later added in the standard
		s = (char*)PhysDim3(k);
		int m = PhysDimCode(s);
		c[5] += (m==k); 
		if ((m!=k) && (s!=NULL) && (s[0]!='#')) {
			fprintf(stdout,"%s\t%d\t%d\t%s\n",PhysDimFactor[k & 0x1f],k,m,s);
		}
	}



	for (k=0; k<6; k++) {
		//fprintf(stdout,"=== [%i]: %d.%06d\t%d.%06d\n",k, t[k].ru_utime.tv_sec,t[k].ru_utime.tv_usec,t[k].ru_stime.tv_sec,t[k].ru_stime.tv_usec);

		if (!k) 
			continue;
	
		timersub(&(t[k].ru_utime), &(t[k-1].ru_utime), &r[0]);
		fprintf(stdout,"usr [%i]: %d.%06d\t",k, r[0].tv_sec,r[0].tv_usec);

		timersub(&(t[k].ru_stime), &(t[k-1].ru_stime) ,&r[1]);
		fprintf(stdout,"sys [%i]: %d.%06d\t",k, r[1].tv_sec,r[1].tv_usec);

		timeradd(&r[0],&r[1],&r[2]);
		fprintf(stdout,"tot [%i]: %d.%06d\t%i\n",k,r[2].tv_sec,r[2].tv_usec,c[k-1]);

	}
	return 0;
}

#endif 
