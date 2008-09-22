// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#include "wx/wxprec.h"
#include "wx/progdlg.h"

#include <boost/shared_ptr.hpp>

#include "./SONlib.h"

std::string stimfit::SON::SONError(short errorCode) {
	std::string errorMsg("");
	switch (errorCode) {
		case SON_NO_FILE:
			errorMsg="The file handle does not refer to an open SON file\n"
					 "or an attempt to open a file failed.";
			break;
		case SON_NO_HANDLES:
			errorMsg="There are too many files open in the system (a DOS problem).";
			break;
		case SON_NO_ACCESS:
			errorMsg="Access was denied, for example insufficient privilege,\n"
					 "attempt to search for a block in a FastWrite file,\n"
					 "attempt to change a file size failed.";
			break;
		case SON_BAD_HANDLE:
			errorMsg="The file you have referenced is not open in the library.";
			break;
		case SON_OUT_OF_MEMORY:
			errorMsg="The system could not allocate enough memory.";
			break;
		case SON_NO_CHANNEL:
			errorMsg="Data channel out of range 0 to SONMaxChans(fh)-1\n"
					 "or no free channel available.";
			break;
		case SON_CHANNEL_USED:
			errorMsg="The channel number supplied is already in use.";
			break;
		case SON_CHANNEL_UNUSED:
			errorMsg="The channel number supplied is not in use.";
			break;
		case SON_WRONG_FILE:
			errorMsg="File header doesn't match a SON file, unknown version,\n"
					 "wrong revision, wrong byte order.";
			break;
		case SON_NO_EXTRA:
			errorMsg="Read/write past end of extra data, or no extra data.";
			break;
		case SON_OUT_OF_HANDLES:
			errorMsg="The SON library has run out of file handles, are you\n"
					 "closing files after using them?";
			break;
		case SON_BAD_READ:
			errorMsg="A read from the disk resulted in an error.";
			break;
		case SON_BAD_WRITE:
			errorMsg="A write to the file resulted in an error.";
			break;
		case SON_CORRUPT_FILE:
			errorMsg="Internal file information is inconsistent, try SonFix.";
			break;
		case SON_READ_ONLY:
			errorMsg="Attempt to write to a file opened in read only mode.";
			break;
		case SON_BAD_PARAM:
			errorMsg="A function parameter value is illegal or inconsistent.";
			break;
		default:
			errorMsg="Unknown error code returned from SON library";
			break;
	}
	return errorMsg;
}

COREDLL_API
void stimfit::SON::importSONFile(const wxString& fName, Recording& ReturnData) {
	wxProgressDialog progDlg(
		wxT("SON file import"),
		wxT("Starting file import"),
		100,
		NULL,
		wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL
	);
	short sFh = SONOpenOldFile(fName.c_str(),0); // try to open file 
	if (sFh < 0) {
		std::string errorMsg(SONError(sFh));
		throw std::runtime_error(errorMsg);
	}
	WORD usPerTime = SONGetusPerTime(sFh); /* Get time information */
	double dTickLen = SONTimeBase(sFh, 0.0); /* base tick units, seconds */	
	std::vector<int> channel_indices;
	for (int i=0;i<SONMaxChans(sFh); ++i) { // for each data channel
		switch (SONChanKind(sFh, i)) {
			case Adc: //printf("%2d is Adc data\n",i); break;
			case RealWave: //printf("%2d RealWave data\n", i); break;
				channel_indices.push_back(i);
				break;
			case AdcMark: //printf("%2d AdcMark data\n",i); break;
			case ChanOff: break; /* do nothing if channel unused */
			case EventFall: break;//printf("%2d Event on falling edge\n",i); break;
			case EventRise: break;//printf("%2d Event on rising edge\n",i); break;
			case EventBoth: break;//printf("%2d Event on both edges\n",i); break;
			case Marker: break;//printf("%2d Marker data\n",i); break;
			case RealMark: break;//printf("%2d RealMark data\n",i); break;
			case TextMark: break;//printf("%2d TextMark data\n",i); break;
			default: 
				throw std::runtime_error("Unknown channel type in SON::importSONFile()");
		}
	}
	if (channel_indices.empty())
		throw std::runtime_error(
			"Error in SON::importSONFile():\n"
			"No Adc or RealWave data found\n"
		);
	ReturnData.resize(channel_indices.size());
	for (std::size_t n_c=0;n_c<channel_indices.size();++n_c) {
		// shorthand:
		WORD chan=(WORD)channel_indices[n_c];

		// read channel info:
		std::vector<char> szvComment(SON_CHANCOMSZ+1); /* SON_CHANCOMSZ is 71 */
		SONGetChanComment(sFh, chan, &szvComment[0], SON_CHANCOMSZ);
		std::string szComment(szvComment.begin(),szvComment.end()); /* the channel comment */
		std::vector<char> szvTitle(SON_TITLESZ+1); /* SON_TITLESZ is 9 */
		SONGetChanTitle(sFh, chan, &szvTitle[0]);
		std::string szTitle(szvTitle.begin(),szvTitle.end()); /* the channel title */
		float scale,offset;
		std::vector<char> szUnits(SON_UNITSZ+1);
		WORD points;
		short preTrig;
		SONGetADCInfo(sFh,chan,&scale,&offset,&szUnits[0],&points,&preTrig);
		// if this is AdcMark data, get duration from number of points:
		long bTime=0,eTime=0;
		Channel TempChannel;
		TempChannel.SetChannelName(szTitle);
		std::size_t n_s=0;
		while (points) {
			long actBTime; /* for returned time of first data point */
			int pow2=1 << 24;
			std::vector<float> afData(pow2); /* example floating point data */
			points=SONGetRealData(
				sFh, 
				chan, 
				&afData[0],
				(long)afData.size(), 
				bTime, 
				(long)afData.size()*SONChanDivide(sFh, chan),
				&actBTime, 
				NULL
			);
			if (points<0) {
				std::string errorMsg(SONError(points));
				throw std::runtime_error(errorMsg);
			}
			if (points>0) {
				std::ostringstream label;
				label << fName << ", Section #" << n_s+1;
				Section TempSection(points,label.str());
				std::copy(afData.begin(),afData.begin()+points,&TempSection[0]);
				TempChannel.resize(n_s+1);
				TempChannel.insert_section(TempSection,n_s++);
			}
			bTime = bTime + (points * SONChanDivide(sFh, chan));
		}
		// make a string:
		std::string units(szUnits.begin(),szUnits.begin()+SON_UNITSZ+1);
		ReturnData.SetGlobalYUnits(n_c,units);
		long lDivide = SONChanDivide(sFh, (long)n_c); /* get interval for channel 3 */
		ReturnData.SetXScale(lDivide*(usPerTime*dTickLen)*1e3); /* frequency in kHz */
		ReturnData.InsertChannel(TempChannel,n_c);
	}
	SONCloseFile(sFh);
}