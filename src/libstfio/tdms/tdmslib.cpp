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

/*! \file tdmslib.cpp
 *  \date 2016-04-23
 *  \brief Import TDMS files.
 */


#include "./tdmslib.h"
#include "../recording.h"

#include <sstream>

// TODO: Get recording time as a string
std::string get_time() {
    return "23:45:00";
}

// TODO: Get recording date as a string
std::string get_date() {
    return "2015-12-24";
}

// TODO: Get file description as a string
std::string get_file_description() {
    return "Recording from PAG";
}

// TODO: Get number of channels
std::size_t get_nchannels() {
    return 0;
}

// TODO: Get sampling interval in ms
double get_xscale() {
    return 1.0;
}

// TODO: Get number of sections (sweeps) for this channel
std::size_t get_nsections(std::size_t nchannel) {
    return 0;
}

// TODO: Get channel name
std::string get_channel_name(std::size_t nchannel) {
    return "DAC0";
}

// TODO: Get units as a string (e.g. "mV" or "pA")
std::string get_yunits(std::size_t nchannel) {
    return "mV";
}

// TODO: Get name of this section (sweep)
std::string get_section_name(std::size_t nchannel, std::size_t nsection) {
    return "DAC0 Sweep0";
}

// TODO: get data
Vector_double get_data(std::size_t nchannel, std::size_t nsection) {
    return Vector_double(0);
}

void stfio::importTDMSFile(const std::string &fName, Recording &ReturnData, ProgressInfo& progDlg) {
    std::string startStr("Reading TDMS file.");
    progDlg.Update(0, startStr);

    std::string errorMsg("Exception while calling importTDMSFile():\n");

    std::string time = get_time();
    std::string date = get_date();
    std::string file_description = get_file_description();
    std::size_t nchannels = get_nchannels();
    double xscale = get_xscale();

    ReturnData.resize(nchannels);

    for (std::size_t nchannel=0; nchannel < ReturnData.size(); ++nchannel) {

        std::size_t nsections = get_nsections(nchannel);
        std::string channel_name = get_channel_name(nchannel);
        std::string yunits = get_yunits(nchannel);

        Channel TempChannel(nsections);
        TempChannel.SetChannelName(channel_name);
        TempChannel.SetYUnits(yunits);

        for (std::size_t nsection=0; nsection < TempChannel.size(); ++nsection) {
            int progbar =
                // Channel contribution:
                (int)(((double)nchannel/(double)ReturnData.size())*100.0+
                    // Section contribution:
                       (double)nsection/(double)TempChannel.size()*(100.0/ReturnData.size()));
            std::ostringstream progStr;
            progStr << "Reading channel #" << nchannel + 1 << " of " << ReturnData.size()
                    << ", Section #" << nsection+1 << " of " << TempChannel.size();
            progDlg.Update(progbar, progStr.str());

            std::string section_name = get_section_name(nchannel, nsection);
            Vector_double data = get_data(nchannel, nsection);

            TempChannel.InsertSection(Section(data, section_name), nsection);
        }
        ReturnData.InsertChannel(TempChannel,nchannel);
    }

    ReturnData.SetXScale(xscale);
    ReturnData.SetFileDescription(file_description);
    ReturnData.SetTime(time);
    ReturnData.SetDate(date);
}

