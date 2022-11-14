/*

    Copyright (C) 2005-2013,2020 Alois Schloegl <alois.schloegl@gmail.com>
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

/*

	Library function for conversion of gdf_time into other datetime formats.
	gdf_time is used in [1] and in Octave and Matlab. Also Python seems to use
        this format but with an offset of 366 days.

	References:
	[1] GDF - A general data format for biomedical signals.
		available online http://arxiv.org/abs/cs.DB/0608052

*/

#define _GNU_SOURCE

#include "gdftime.h"

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

#define fix(m)     	(m<0 ? ceil(m) : floor(m))

gdf_time tm_time2gdf_time(struct tm *t){
	/* based on Octave's datevec.m
	it referes Peter Baum's algorithm at http://vsg.cape.com/~pbaum/date/date0.htm
	but the link is not working anymore as of 2008-12-03.

	Other links to Peter Baum's algorithm are
	http://www.rexswain.com/b2mmddyy.rex
	http://www.dpwr.net/forums/index.php?s=ecfa72e38be61327403126e23aeea7e5&showtopic=4309
	*/

	if (t == NULL) return(0);

	int Y,M,s; // h,m,
	double D;
	gdf_time o;
	const int monthstart[] = {306, 337, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275};

	D = (double)t->tm_mday;
	M = t->tm_mon+1;
	Y = t->tm_year+1900;

	// Set start of year to March by moving Jan. and Feb. to previous year.
	// Correct for months > 12 by moving to subsequent years.
	Y += fix ((M-14.0)/12);

	// Lookup number of days since start of the current year.
	D += monthstart[t->tm_mon % 12] + 60;

	// Add number of days to the start of the current year. Correct
	// for leap year every 4 years except centuries not divisible by 400.
	D += 365*Y + floor (Y/4.0) - floor (Y/100.0) + floor (Y/400.0);

	// Add fraction representing current second of the day.
	s = t->tm_hour*3600 + t->tm_min*60 + t->tm_sec;

	// s -= timezone;
	o = (((uint64_t)D) << 32) + (((uint64_t)s) << 32)/86400;

	return(o);
}


struct tm *gdf_time2tm_time(gdf_time t) {
        // this is not re-entrant, use gdf_time2tm_time_r instead

	/* based Octave's datevec.m
	it referes Peter Baum's algorithm at http://vsg.cape.com/~pbaum/date/date0.htm
	but the link is not working anymore as of 2008-12-03.

	Other links to Peter Baum's algorithm are
	http://www.rexswain.com/b2mmddyy.rex
	http://www.dpwr.net/forums/index.php?s=ecfa72e38be61327403126e23aeea7e5&showtopic=4309
	*/

	static struct tm tt;	// allocate memory for t3;
        gdf_time2tm_time_r(t,&tt);
	return(&tt);
}

struct gdf_time_tm_t {
	int YEAR;
	int MONTH;
	int MDAY;
	int HOUR;
	int MINUTE;
	double SECOND;
};


int split_gdf_time(gdftime_t t, struct gdf_time_tm_t *gte) {
	/* based Octave's datevec.m
	it referes Peter Baum's algorithm at http://vsg.cape.com/~pbaum/date/date0.htm
	but the link is not working anymore as of 2008-12-03.

	Other links to Peter Baum's algorithm are
	http://www.rexswain.com/b2mmddyy.rex
	http://www.dpwr.net/forums/index.php?s=ecfa72e38be61327403126e23aeea7e5&showtopic=4309
	*/

	int32_t rd = (int32_t)floor(ldexp((double)t,-32)); // days since 0001-01-01
	double s = ldexp((t & 0x00000000ffffffff)*86400,-32); // seconds of the day
	// int32_t sec = round (s);
	// s += timezone;

	/* derived from datenum.m from Octave 3.0.0 */

	// Move day 0 from midnight -0001-12-31 to midnight 0000-3-1
	double z = floor (rd) - 60;
	// Calculate number of centuries; K1 = 0.25 is to avoid rounding problems.
	double a = floor ((z - 0.25) / 36524.25);
	// Days within century; K2 = 0.25 is to avoid rounding problems.
	double b = z - 0.25 + a - floor (a / 4);
	// Calculate the year (year starts on March 1).
	int y = (int)floor (b / 365.25);
	// Calculate day in year.
	double c = fix (b - floor (365.25 * y)) + 1;
	// Calculate month in year.
	double m = fix ((5 * c + 456) / 153);
	double d = c - fix ((153 * m - 457) / 5);

	// Move to Jan 1 as start of year.
	if (m>12) {y++; m-=12;}
	gte->YEAR = y;
	gte->MONTH  = (int)m;
	gte->MDAY = (int)d;

	int h = (int) s / 3600;
	s = s - (3600 * h);
	m = s / 60;	// !! reuse of m: is now minutes instead of month
	gte->HOUR = h;
	gte->MINUTE = (int) m;
	gte->SECOND = s - (60 * gte->MINUTE);
	//t3->tm_gmtoff = 3600;

        return(0);
}

int gdf_time2tm_time_r(gdftime_t t, struct tm *t3) {
	struct gdf_time_tm_t gte;

	split_gdf_time(t, &gte);

	t3->tm_year = gte.YEAR-1900;
	t3->tm_mon  = gte.MONTH-1;
	t3->tm_mday = gte.MDAY;

	t3->tm_hour = gte.HOUR;
	t3->tm_min  = gte.MINUTE;
	t3->tm_sec  = (int)gte.SECOND;
}

#if 0
gdftime_t string2gdftime(const char* str) {
	struct tm t;
	strptime(str,"%d %b %Y",&t);
	t.tm_hour = 0;
	t.tm_min  = 0;
	t.tm_sec  = 0;
	return tm_time2gdf_time(&t);
}

gdftime_t string2gdfdate(const char* str) {
	struct tm t;
	strptime(str,"%d %b %Y",&t);
	t.tm_hour = 0;
	t.tm_min  = 0;
	t.tm_sec  = 0;
	return tm_time2gdf_time(&t);
}

gdftime_t string2gdfdatetime(const char* str) {
	struct tm t;

	return tm_time2gdf_time(getdate(str));

}
#endif

size_t snprintf_gdftime(char *out, size_t outbytesleft, gdftime_t T) {
	struct gdf_time_tm_t gte;
	size_t len;

	split_gdf_time(T, &gte);
	len = snprintf(out, outbytesleft, "%02d:%02d:", gte.HOUR, gte.MINUTE);
	outbytesleft -= len;
	out += len;

	double intSec;
	double fracSec=modf(gte.SECOND, &intSec);
	if (fracSec == 0.0)
		len = snprintf(out, outbytesleft, "%02d", (int)gte.SECOND);
	else
		len = snprintf(out, outbytesleft, "%09.6f", gte.SECOND);

	outbytesleft -= len;
	out += len;
	return len;
}

size_t snprintf_gdfdate(char *out, size_t outbytesleft, gdftime_t T) {
	struct gdf_time_tm_t gte;
	size_t len;

	split_gdf_time(T, &gte);
	len = snprintf(out, outbytesleft, "%04d-%02d-%02d", gte.YEAR, gte.MONTH, gte.MDAY);

	outbytesleft -= len;
	out += len;
	return len;
}

size_t snprintf_gdfdatetime(char *out, size_t outbytesleft, gdftime_t T) {
	struct gdf_time_tm_t gte;
	size_t len;

	split_gdf_time(T, &gte);
	len = snprintf(out, outbytesleft, "%04d-%02d-%02d %02d:%02d:", gte.YEAR, gte.MONTH, gte.MDAY, gte.HOUR, gte.MINUTE);
	outbytesleft -= len;
	out += len;

	double intSec;
	double fracSec=modf(gte.SECOND, &intSec);
	if (fracSec == 0.0)
		len = snprintf(out, outbytesleft, "%02d", (int)gte.SECOND);
	else
		len = snprintf(out, outbytesleft, "%09.6f", gte.SECOND);

	outbytesleft -= len;
	out += len;
	return len;
}


size_t strfgdftime(char *out, size_t outbytesleft, const char *FMT, gdftime_t T) {
	struct gdf_time_tm_t gte;
	struct tm tm;
	size_t len;
	int cin=0;
	int cout=0;
	char FMT2[4]="%%\0\0";

	split_gdf_time(T, &gte);
	gdf_time2tm_time_r(T, &tm);

	while ( cout < outbytesleft && cin<strlen(FMT)) {
		switch (FMT[cin]) {
		case '%':
			FMT2[1] = FMT[cin+1];
			switch (FMT2[1]) {
			case 's':
				cout += snprintf(out+cout, outbytesleft-cout, "%f", gdf_time2t_time(T));
				cin += 2;
				break;
			case 'S':
				cout += snprintf(out+cout, outbytesleft-cout, "%09.6f", gte.SECOND);
				cin += 2;
				break;
			case 'E':
			case 'O':
				FMT2[2] = FMT[cin+2];
				cout += strftime(out+cout, outbytesleft-cout, FMT2, &tm);
				cin += 3;
				FMT2[2]=0;	// reset terminating \0
				break;
			default:
				cout += strftime(out+cout, outbytesleft-cout, FMT2, &tm);
				cin += 2;
			}
			break;
		default:
			out[cout++] = FMT[cin++];
		}
	}
	if (0 < outbytesleft) out[cout]=0;
	return cout;
}


/*
char *gdftime2string(gdftime_t)
char *gdfdate2string(gdftime_t)
char *gdfdatetime2string(gdftime_t)

gdftime_t time2gdftime(int,int,float)
gdftime_t date2gdftime(int,int,int)
gdftime_t datetime2gdftime(int,int,int,int,int,float)

void gdftime2datetime(&int,&int,&int,&int,&int,&float)
void gdftime2time(&int,&int,&float)
void gdftime2date(&int,&int,&int)


					strptime(line+p+1,"%H:%M:%S",&t);
					if (VERBOSE_LEVEL > 7) fprintf(stdout, "%s (line %i) %s\n", __FILE__, __LINE__, line);
					if (VERBOSE_LEVEL > 7) {
						char tmp[30];
						strftime(tmp,30,"%F %T",&t);
						fprintf(stdout, "%s (line %i) %s\n", __FILE__, __LINE__, tmp);
					}
				}
				else if (!strncmp(line,"Date",p)) {
					strptime(line+p+1,"%d %b %Y",&t);
					t.tm_hour = 0;
					t.tm_min  = 0;
					t.tm_sec  = 0;
					if (VERBOSE_LEVEL > 7) {
						char tmp[30];
						strftime(tmp,30,"%F %T",&t);
						fprintf(stdout, "%s (line %i) %s\n", __FILE__, __LINE__, tmp);
					}
				}
				else if (!strncmp(line,"Time Stamp",p)) {
					hdr->SampleRate *= hdr->SPR*hdr->NRec/strtod(line+p+1,NULL);
				}

				line = strtok(NULL, "\n\r\0");

 */




/****************************************************************************/
/**                                                                        **/
/**                               EOF                                      **/
/**                                                                        **/
/****************************************************************************/

