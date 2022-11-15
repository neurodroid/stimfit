/*

    Copyright (C) 2022 Alois Schloegl <alois.schloegl@gmail.com>

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
	read BiosigDump file 
 ******************************************************************************/
void sopen_biosigdump_read(HDRTYPE* hdr) {

	if (VERBOSE_LEVEL > 8) fprintf(stdout,"%s (line %d) %s(hdr)\n", __FILE__, __LINE__, __func__);

	if (hdr->TYPE==BiosigDump) {
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format BiosigDump: not supported yet");
	}
	/*
		between 512 and 4096 bytes are already read into hdr->AS.Header, 
		the exact number of bytes read, is available in "hdr->HeadLen"	
		Currently this is 4096, but it might change in future. 
	*/
}

