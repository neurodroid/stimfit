/*
    Copyright (C) 2005-2019 Alois Schloegl <alois.schloegl@ist.ac.at>
    This file is part of the "BioSig for C/C++" repository
    (biosig4c++) at http://biosig.sf.net/

    This program is free software; you can redistribute it and/or
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

/****************************************************************************
 **                                                                        **
 **    Conversion functions for encoded physical units according to        **
 **    ISO/IEEE 11073-10101:2004 Vital Signs Units of Measurement          **
 **                                                                        **
 ****************************************************************************/
#ifndef __PHYSICALUNITS_H__
#define __PHYSICALUNITS_H__

#if defined(_MSC_VER) && (_MSC_VER < 1600)
    typedef unsigned __int64	uint16_t;
#else
    #include <inttypes.h>
#endif

#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif

uint16_t PhysDimCode(const char* PhysDim);
/* Encodes  Physical Dimension as 16bit integer according to
   ISO/IEEE 11073-10101:2004 Vital Signs Units of Measurement
   Leading and trailing whitespace are skipped.
 --------------------------------------------------------------- */

const char* PhysDim3(uint16_t PhysDimCode);
/* converts PhysDimCode into a readable Physical Dimension
 --------------------------------------------------------------- */

double PhysDimScale(uint16_t PhysDimCode);
/* returns scaling factor of physical dimension
	e.g. 0.001 for milli, 1000 for kilo etc.
	for undefined codes, not-a-number (NAN) is returned
 --------------------------------------------------------------- */

#pragma GCC visibility pop

#ifdef __cplusplus
}
#endif


#endif	/* __PHYSICALUNITS_H__ */
