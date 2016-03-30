/*

% Copyright (C) 2014 Alois Schloegl <alois.schloegl@gmail.com>
% This file is part of the "BioSig for C/C++" repository
% (biosig4c++) at http://biosig.sf.net/


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
 **    ISO/IEEE 11073-10102 Annex B                                        **
 **                                                                        **
 ****************************************************************************/
#ifndef __MDC_ECG_CODES_H__
#define __MDC_ECG_CODES_H__

#if defined(_MSC_VER) && (_MSC_VER < 1600)
    typedef unsigned __int16	uint16_t;
#else
    #include <inttypes.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif

uint16_t    encode_mdc_ecg_code10   (const char *IDstr);
uint32_t    encode_mdc_ecg_cfcode10 (const char *IDstr);
const char* decode_mdc_ecg_code10   (uint16_t code10);
const char* decode_mdc_ecg_cfcode10 (uint32_t cf_code10);

#ifdef __cplusplus
}
#endif


#endif	/* __PHYSICALUNITS_H__ */
