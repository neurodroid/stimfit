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

#include <stdio.h>
#include <ctime>
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

Recording::Recording(const std::deque<Channel>& ChannelList)
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
    comment = "\0";
    xunits =  "ms" ;
    dt = 1.0;

    // get current time
    time_t timer;
    timer = time(0);
    memcpy(&datetime, localtime(&timer), sizeof(datetime));

    cc = 0;
    sc = 1;
    cs = 0;
    selectedSections = std::vector<std::size_t>(0);
    selectBase = Vector_double(0);
	sectionMarker = std::vector<int>(0);
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

int Recording::SetDate(const std::string& value) {
    struct tm t = GetDateTime();
#if HAVE_STRPTIME_H
    strptime(value.c_str(), "%F", &t);
#else
    if ( sscanf(value.c_str(),"%i-%i-%i", &t.tm_year, &t.tm_mon, &t.tm_mday)==3
      || sscanf(value.c_str(),"%i.%i.%i", &t.tm_mday, &t.tm_mon, &t.tm_year)==3
      || sscanf(value.c_str(),"%i/%i/%i", &t.tm_mon, &t.tm_mday, &t.tm_year)==3
    ) {
	t.tm_mon--;
	if (t.tm_year < 50) t.tm_year += 100;
	else if (t.tm_year < 139);
	else if (t.tm_year > 1900) t.tm_year -= 1900;
    }
    else {
	// TODO: error handling
	fprintf(stderr,"SetDate(%s) failed\n",value.c_str());
        return(-1);
    }
#endif
    SetDateTime(t);
    return(0);
}

int Recording::SetTime(const std::string& value) {
    struct tm t = GetDateTime();
#if HAVE_STRPTIME_H
    strptime(value.c_str(), "%T", &t);
#else
    if ( sscanf(value.c_str(),"%i-%i-%i", &t.tm_hour, &t.tm_min, &t.tm_sec)==3
      || sscanf(value.c_str(),"%i.%i.%i", &t.tm_hour, &t.tm_min, &t.tm_sec)==3
      || sscanf(value.c_str(),"%i:%i:%i", &t.tm_hour, &t.tm_min, &t.tm_sec)==3
    ) {
	; // everthing is fine
    }
    else {
	// TODO: error handling
	fprintf(stderr,"SetTime(%s) failed\n",value.c_str());
	return(-1);
    }
#endif
    SetDateTime(t);
    return(0);
}

int Recording::SetTime(int hour, int minute, int sec) {
    datetime.tm_hour=hour;
    datetime.tm_min=minute;
    datetime.tm_sec=sec;
    return(0);
}

int Recording::SetDate(int year, int month, int mday) {
    datetime.tm_year=year;
    datetime.tm_mon=month;
    datetime.tm_mday=mday;
    return(0);
}

void Recording::SetDateTime(int year, int month, int mday,int hour, int minute, int sec) {
    datetime.tm_year=year;
    datetime.tm_mon=month;
    datetime.tm_mday=mday;
    datetime.tm_hour=hour;
    datetime.tm_min=minute;
    datetime.tm_sec=sec;
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
    datetime=c_Recording.datetime;
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

void Recording::SetCurChIndex(size_t value) {
    if (value>=get().size()) {
        throw std::out_of_range("channel out of range in Recording::SetCurChIndex()");
    }
    cc=value;
}

void Recording::SetSecChIndex(size_t value) {
    if (value>=get().size() || value==cc) {
        throw std::out_of_range("channel out of range in Recording::SetSecChIndex()");
    }
    sc=value;
}

void Recording::SetCurSecIndex( size_t value ) {
    if (value >= get()[cc].size()) {
        throw std::out_of_range("channel out of range in Recording::SetCurSecIndex()");
    }
    cs=value;
}

void Recording::SelectTrace(std::size_t sectionToSelect, std::size_t base_start, std::size_t base_end) {
    // Check range so that sectionToSelect can be used
    // without checking again:
    if (sectionToSelect>=curch().size()) {
        std::out_of_range e("subscript out of range in Recording::SelectTrace\n");
        throw e;
    }
    selectedSections.push_back(sectionToSelect);
    double sumY=0;
    if (curch()[sectionToSelect].size()==0) {
        selectBase.push_back(0);
    } else {
        int start = base_start;
        int end = base_end;
        if (start > (int)curch()[sectionToSelect].size()-1)
            start = curch()[sectionToSelect].size()-1;
        if (start < 0) start = 0;
        if (end > (int)curch()[sectionToSelect].size()-1)
            end = curch()[sectionToSelect].size()-1;
        if (end < 0) end = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sumY)
#endif
        for (int i=start; i<=end; i++) {
            sumY += curch()[sectionToSelect][i];
        }
        int n=(int)(end-start+1);
        selectBase.push_back(sumY/n);
    }
}

bool Recording::UnselectTrace(std::size_t sectionToUnselect) {

    //verify whether the trace has really been selected and find the 
    //number of the trace within the selectedTraces array:
    bool traceSelected=false;
    std::size_t traceToRemove=0;
    for (std::size_t n=0; n < selectedSections.size() && !traceSelected; ++n) { 
        if (selectedSections[n] == sectionToUnselect) traceSelected=true;
        if (traceSelected) traceToRemove=n;
    }
    //Shift the selectedTraces array by one position, beginning
    //with the trace to remove: 
    if (traceSelected) {
        //shift traces by one position:
        for (std::size_t k=traceToRemove; k < GetSelectedSections().size()-1; ++k) { 
            selectedSections[k]=selectedSections[k+1];
            selectBase[k]=selectBase[k+1];
        }
        // resize vectors:
        selectedSections.resize(selectedSections.size()-1);
        selectBase.resize(selectBase.size()-1);
        return true;
    } else {
        //msgbox
        return false;
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
    std::deque< Channel >::iterator it;
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


std::string Recording::GetEventDescription(int type) {
    return listOfMarkers[type];
}

void Recording::SetEventDescription(int type, const char* Description) {
    listOfMarkers[type] = Description;
}

void Recording::InitSectionMarkerList(size_t n) {
    sectionMarker.resize(n);
    return;
}

int Recording::GetSectionType(size_t section_number) {
    return sectionMarker[section_number];
}

void Recording::SetSectionType(size_t section_number, int type) {
    sectionMarker[section_number]=type;
    return;
}

