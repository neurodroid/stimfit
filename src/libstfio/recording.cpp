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


#include "./stfio.h"
#include "./recording.h"

#include <sstream>

Recording::Recording(void)
    : ChannelArray(0)
{
    init();
}

Recording::Recording(const Channel& c_Channel)
    : ChannelArray(1,c_Channel)
{
    init();
}

Recording::Recording(const std::vector<Channel>& ChannelList)
    : ChannelArray(ChannelList)
{
    init();
}

Recording::Recording(std::size_t c_n_channels, std::size_t c_n_sections, std::size_t c_n_points)
  : ChannelArray(c_n_channels, Channel(c_n_sections, c_n_points))
{
    init();    
}

void Recording::init() {
    file_description = "\0";
    global_section_description = "\0";
    scaling = "\0";
    time = "\0";
    date = "\0";
    comment = "\0";
    xunits =  "ms" ;
    dt = 1.0;
}

Recording::~Recording() {
}

const Channel& Recording::at(std::size_t n_c) const {
    try {
        return ChannelArray.at(n_c);
    }
    catch (...) {
        throw;
    }
}

Channel& Recording::at(std::size_t n_c) {
    try {
        return ChannelArray.at(n_c);
    }
    catch (...) {
        throw;
    }
}

void Recording::InsertChannel(Channel& c_Channel, std::size_t pos) {
    try {
        if ( ChannelArray.at(pos).size() <= c_Channel.size() ) {
            ChannelArray.at(pos).resize( c_Channel.size() );
        }
        // Resize sections if necessary:
        std::size_t n_sec = 0;
        for ( sec_it sit = c_Channel.get().begin(); sit != c_Channel.get().end(); ++sit ) {
            if ( ChannelArray.at(pos).at(n_sec).size() <= sit->size() ) {
                ChannelArray.at(pos).at(n_sec).get_w().resize( sit->size() );
            }
            n_sec++;
        }
    }
    catch (...) {
        throw;
    }
    ChannelArray.at(pos) = c_Channel;
}

void Recording::CopyAttributes(const Recording& c_Recording) {
    file_description=c_Recording.file_description;
    global_section_description=c_Recording.global_section_description;
    scaling=c_Recording.scaling;
    time=c_Recording.time;
    date=c_Recording.date;
    comment=c_Recording.comment;
    for ( std::size_t n_ch = 0; n_ch < c_Recording.size(); ++n_ch ) {
        if ( size() > n_ch ) {
            ChannelArray[n_ch].SetYUnits( c_Recording[n_ch].GetYUnits() );
        }
    }
    dt=c_Recording.dt;
}

void Recording::resize(std::size_t c_n_channels) {
    ChannelArray.resize(c_n_channels);
}

size_t Recording::GetChannelSize(std::size_t n_channel) const {
    try {
        return ChannelArray.at(n_channel).size();
    }
    catch (...) {
        throw;
    }
}

void Recording::SetXScale(double value) {
    dt=value;
    for (ch_it it1 = ChannelArray.begin(); it1 != ChannelArray.end(); it1++) {
        for (sec_it it2 = it1->get().begin(); it2 != it1->get().end(); it2++) {
            it2->SetXScale(value);
        }
    }
}

void Recording::MakeAverage(Section& AverageReturn,
        Section& SigReturn,
        std::size_t channel,
        const std::vector<std::size_t>& section_index,
        bool isSig,
        const std::vector<int>& shift) const
{
    if (channel >= ChannelArray.size()) {
        throw std::out_of_range("Channel number out of range in Recording::MakeAverage");
    }
    unsigned int n_sections = section_index.size();
    if (shift.size() != n_sections) {
        throw std::out_of_range("Shift out of range in Recording::MakeAverage");
    }
    for (unsigned int l = 0; l < n_sections; ++l) {
        if (section_index[l] >= ChannelArray[channel].size()) {
            throw std::out_of_range("Section number out of range in Recording::MakeAverage");
        }
        if (AverageReturn.size() + shift[l] > ChannelArray[channel][section_index[l]].size()) {
            throw std::out_of_range("Sampling point out of range in Recording::MakeAverage");
        }
    }

    for (unsigned int k=0; k < AverageReturn.size(); ++k) {
        AverageReturn[k]=0.0;
        //Calculate average
        for (unsigned int l = 0; l < n_sections; ++l) {
            AverageReturn[k] += 
                ChannelArray[channel][section_index[l]][k+shift[l]];
        }
        AverageReturn[k] /= n_sections;

        // set sample interval of averaged traces
        AverageReturn.SetXScale(ChannelArray[channel][section_index[0]].GetXScale());

        if (isSig) {
            SigReturn[k]=0.0;
            //Calculate variance
            for (unsigned int l=0; l < n_sections; ++l) {
                SigReturn[k] += 
                    pow(ChannelArray[channel][section_index[l]][k+shift[l]] -
                            AverageReturn[k], 2);
            }
            SigReturn[k] /= (n_sections - 1);
            SigReturn[k]=sqrt(SigReturn[k]);
        }
    }
}

void Recording::AddRec(const Recording &toAdd) {
    // check number of channels:
    if (toAdd.size()!=size()) {
        throw std::runtime_error("Number of channels doesn't match");
    }
    // check dt:
    if (toAdd.GetXScale()!=dt) {
        throw std::runtime_error("Sampling interval doesn't match");
    }
    // add sections:
    std::vector< Channel >::iterator it;
    std::size_t n_c = 0;
    for (it = ChannelArray.begin();it != ChannelArray.end(); it++) {
        std::size_t old_size = it->size();
        it->resize(toAdd[n_c].size()+old_size);
        for (std::size_t n_s=old_size;n_s<toAdd[n_c].size()+old_size;++n_s) {
            try {
                it->InsertSection(toAdd[n_c].at(n_s-old_size),n_s);
            }
            catch (...) {
                throw;
            }
        }
        n_c++;
    }
}
