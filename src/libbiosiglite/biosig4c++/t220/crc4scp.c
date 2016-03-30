/*
---------------------------------------------------------------------------
Copyright (C) 2005-2006  Franco Chiarugi
Developed at the Foundation for Research and Technology - Hellas, Heraklion, Crete
Copyright (C) 2009 Alois Schloegl

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

$Id$
---------------------------------------------------------------------------
*/

#if defined(_MSC_VER) && (_MSC_VER < 1600)
typedef unsigned __int64	uint64_t;
typedef __int64			int64_t;
typedef unsigned __int32	uint32_t;
typedef __int32			int32_t;
typedef unsigned __int16	uint16_t;
typedef __int16			int16_t;
typedef unsigned __int8		uint8_t;
typedef __int8			int8_t;

#else

#include <inttypes.h>

#endif


#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************
*	CRCEvaluate							*
*									*
* Parameters: datablock is the buffer on which to evaluate the CRC.	*
*			  datalength is the length of the whole buffer	*
*									*
* Description:	Evaluate the SCP-ECG CRC on a data block		*
*				(all file or a section)			*
*									*
 ********************************************************************/

uint16_t CRCEvaluate(uint8_t* datablock, uint32_t datalength) {
	uint32_t	i;
	uint16_t	crc_tot;
	uint8_t		crchi, crclo;
	uint8_t		a, b;
	uint8_t		tmp1, tmp2;

	crchi = 0xFF;
	crclo = 0xFF;

	for (i = 0; i < datalength; i++) {
		a = datablock[i];
		a ^= crchi;
		crchi = a;
		a >>= 4;
		a &= 0x0F;
		a ^= crchi;
		crchi = crclo;
		crclo = a;
		tmp1 = ((a & 0x0F) << 4) & 0xF0;
		tmp2 = ((a & 0xF0) >> 4) & 0x0F;
		a = tmp1 | tmp2;
		b = a;
		tmp1 = ((a & 0x7F) << 1) & 0xFE;
		tmp2 = ((a & 0x80) >> 7) & 0x01;
		a = tmp1 | tmp2;
		a &= 0x1F;
		crchi ^= a;
		a = b & 0xF0;
		crchi ^= a;
		tmp1 = ((b & 0x7F) << 1) & 0xFE;
		tmp2 = ((b & 0x80) >> 7) & 0x01;
		b = tmp1 | tmp2;
		b &= 0xE0;
		crclo ^= b;
	}

	crc_tot = ((0x00FF & (uint16_t) crchi) << 8) & 0xFF00;
	crc_tot |= (0x00FF & (uint16_t) crclo);

	return (crc_tot);
}

/********************************************************************
*	CRCCheck							*
*									*
* Parameters: datablock is the buffer on which to verify the CRC.	*
*			  It starts with the two CRC-CCITT bytes.	*
*			  datalength is the length of the whole buffer	*
*			  (including the two CRC bytes)			*
*									*
* Description:	Check the SCP-ECG CRC on a data block			*
*				(all file or a section)			*
*									*
 ********************************************************************/

int16_t CRCCheck(uint8_t* datablock, uint32_t datalength)
{
	uint16_t crc;

	crc = 0;

	if (datalength <= 2)
		return (-1);

	// Evaluate CRC
	crc = CRCEvaluate((uint8_t*) (datablock + 2), (uint32_t) (datalength - 2));
	if (((uint8_t) ((crc & 0xFF00) >> 8) != (uint8_t) datablock[1]) ||
		((uint8_t) (crc & 0x00FF) != (uint8_t) datablock[0]))
		return (0);
	else
		return (1);
}


#ifdef __cplusplus
}
#endif

