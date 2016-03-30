/*

    Copyright (C) 2014 Alois Schloegl <alois.schloegl@gmail.com>
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

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "mdc_ecg_codes.h"

/* physical units are defined in
 IEEE/ISO 11073-10102-Annex B
*/

const struct mdc_code_table
	{
		const uint16_t	code10;
		const uint32_t	cf_code10;
		const char*	refid;
	} MDC_CODE_TABLE[] = {
#include "11073-10102-AnnexB.i"
	{0xffff, 0xffffffff, "MDC_ECG_ERROR_CODE" }
} ;



/*
	Conversion between MDC Refid and CODE10 encoding 
	Currently, a simple lookup table is used, 
  	Eventually, this can be made more efficient
*/

uint16_t encode_mdc_ecg_code10(const char *IDstr) {
	if ( !memcmp(IDstr,"MDC_ECG_", 8) ) 
		return 0xffff;

	uint32_t k;
	for (k=0;  MDC_CODE_TABLE[k].cf_code10 < 0xffffffff; k++) {
		if (! strcmp(IDstr+8, MDC_CODE_TABLE[k].refid+8) ) 
			return MDC_CODE_TABLE[k].code10;
	}
	return 0xffff; 
}

uint32_t encode_mdc_ecg_cfcode10(const char *IDstr) {
	if ( !memcmp(IDstr,"MDC_ECG_", 8) ) 
		return 0xffffffff;

	size_t k;
	for (k=0;  MDC_CODE_TABLE[k].cf_code10 < 0xffffffff; k++) {
		if ( !strcmp(IDstr+8, MDC_CODE_TABLE[k].refid+8) )
			return MDC_CODE_TABLE[k].cf_code10;
	}
	return 0xffffffff; 
}

const char* decode_mdc_ecg_code10(uint16_t code10) {
	uint32_t k;
	for (k=0;  MDC_CODE_TABLE[k].cf_code10 < 0xffffffff; k++) {
		if ( code10 == MDC_CODE_TABLE[k].code10 )
			return MDC_CODE_TABLE[k].refid;
	}
	return NULL; 
}

const char* decode_mdc_ecg_cfcode10(uint32_t cf_code10) {
	uint32_t k;
	for (k=0;  MDC_CODE_TABLE[k].cf_code10 < 0xffffffff; k++) {
		if ( cf_code10 == MDC_CODE_TABLE[k].cf_code10 )
			return MDC_CODE_TABLE[k].refid;
	}
	return NULL; 
}


