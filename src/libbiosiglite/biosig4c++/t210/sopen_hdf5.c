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
#ifdef WITH_HDF5
#include <hdf5/serial/hdf5.h>
#include <hdf5/serial/hdf5_hl.h>
#endif
#include "../biosig.h"

int sopen_hdf5(HDRTYPE* hdr) {
#ifdef WITH_HDF5
        /*
                file hdr->FileName is already opened and hdr->HeadLen bytes are read
                These are available from hdr->AS.Header.

                ToDo: populate hdr
        */
	size_t count = hdr->HeadLen;
        fprintf(stdout,"Trying to read HDF data using \"%s\"\n",H5_VERS_INFO);

	// identify which type/origin of HDF5
	biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Error reading HDF5 file");

	ifclose(hdr);
	return(-1);
#else	// WITH_MATIO
	biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "SOPEN(HDF5): - HDF5 format not supported - libbiosig need to be recompiled with libhdf5 support.");
	return(-1);
#endif	// WITH_MATIO
}

