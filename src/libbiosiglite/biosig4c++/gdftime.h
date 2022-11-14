/*

% Copyright (C) 2005-2013,2020 Alois Schloegl <alois.schloegl@gmail.com>
% This file is part of the "BioSig for C/C++" repository
% (biosig4c++/libbiosig) at http://biosig.sf.net/


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



/****************************************************************************/
/**                                                                        **/
/**                 DEFINITIONS, TYPEDEFS AND MACROS                       **/
/**                                                                        **/
/****************************************************************************/

#ifndef __GDFTIME_H__
#define __GDFTIME_H__


#if defined(_MSC_VER) && (_MSC_VER < 1600)
    typedef __int64		int64_t;
#else
    #include <inttypes.h>
#endif
#include <stdio.h>
#include <string.h>
#include <time.h>
/* ensure that definition of "byte" does not conflict between
    rpcndr.h (Windows/mingw only) and bits/cpp_type_traits.h under g++
    see also https://github.com/mxe/mxe/issues/2759
 */
#define _GLIBCXX_INCLUDE_NEXT_C_HEADERS
#include <math.h>
#undef _GLIBCXX_INCLUDE_NEXT_C_HEADERS

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	Conversion of time formats between Unix and GDF format.

	The default time format in BIOSIG uses a 64-bit fixed point format with
	reference date 01-Jan-0000 00h00m00s (value=0).
	One unit indicates the 2^(-32) part of 1 day (ca 20 us). Accordingly,
	the higher 32 bits count the number of days, the lower 32 bits describe
	the fraction of a day.  01-Jan-1970 is the day 719529.

	time_t t0;
	t0 = time(NULL);
	T0 = (double)t0/86400.0;	// convert seconds in days since 1970-Jan-01
	floor(T0) + 719529;		// number of days since 01-Jan-0000
	floor(ldexp(T0-floor(T0),32));  // fraction x/2^32; one day is 2^32

	The following macros define the conversions between the unix time and the
	GDF format.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
typedef int64_t 		gdf_time, gdftime_t; /* gdf time is represented in 64 bits */

#define t_time2gdf_time(t)	((gdf_time)floor(ldexp(((double)(t))/86400.0 + 719529.0, 32)))
#define gdf_time2t_time(t)	((ldexp(((double)(t)),-32) - 719529.0)*86400.0)
#define	ntp_time2gdf_time(t)	((gdf_time)ldexp(ldexp(((double)(t)),-32)/86400 + 719529.0 - 70,32))
#define	gdf_time2ntp_time(t)	((int64_t)ldexp((ldexp(((double)(t)),-32) - 719529.0 + 70) * 86400,32))

#pragma GCC visibility push(default)

#ifdef __cplusplus
EXTERN_C {
#endif
/*
 * converts struct tm into gdf_time format
 */
gdftime_t   tm_time2gdf_time(struct tm *t);
/*
 * gdf_time2tm_time converts gdf-time into struct tm format,
   this is deprecated because time resolution (sub-seconds) are lost
 */
struct tm *gdf_time2tm_time(gdftime_t t);
/*
 * re-entrant version of gdf_time2tm_time, memory for t must be allocated
   this is deprecated because time resolution (sub-seconds) are lost
 */
int gdf_time2tm_time_r(gdftime_t t, struct tm *tm);

/*
   converts date and time to strings using this format %Y-%m-%d %H:%M:%S
   with microsecond resolution, if needed.
	%04d-%02d-%02d %02d:%02d:%02d
	%04d-%02d-%02d %02d:%02d:%09.6f
   strfgdftime resembles strftime(...), except for format
	%s and %S which will also present 6 digits the fraction of the second
*/

size_t snprintf_gdftime(char *out, size_t outbytesleft, gdftime_t T);
size_t snprintf_gdfdate(char *out, size_t outbytesleft, gdftime_t T);
size_t snprintf_gdfdatetime(char *out, size_t outbytesleft, gdftime_t T);
size_t strfgdftime(char *out, size_t outbytesleft, const char *FMT, gdftime_t T);


/*
gdftime_t string2gdftime(const char*)
gdftime_t string2gdfdate(const char*)
gdftime_t string2gdfdatetime(const char*)

gdftime_t time2gdftime(int,int,float)
gdftime_t date2gdftime(int,int,int)
gdftime_t datetime2gdftime(int,int,int,int,int,float)

void gdftime2datetime(&int,&int,&int,&int,&int,&float)
void gdftime2time(&int,&int,&float)
void gdftime2date(&int,&int,&int)
 */


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


/****************************************************************************/
/**                                                                        **/
/**                               EOF                                      **/
/**                                                                        **/
/****************************************************************************/

#endif	/* __BIOSIG_EXT_H__ */

