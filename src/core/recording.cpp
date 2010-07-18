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


#include "./core.h"
#include "./recording.h"

#ifndef MODULE_ONLY
    #include "./measlib.h"
#endif

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
    file_description = wxT("\0");
    global_section_description = wxT("\0");
    scaling = wxT("\0");
    time = wxT("\0");
    date = wxT("\0");
    comment = wxT("\0");
    xunits =  wxT("ms") ;
    dt = 1.0;

#ifndef MODULE_ONLY

    latencyStartMode = stf::riseMode;
    latencyEndMode = stf::footMode;
    latencyWindowMode = stf::defaultMode;
    direction = stf::both;    
    pslopeBegMode = stf::psBeg_manualMode;
    pslopeEndMode = stf::psEnd_manualMode;
    cc = 0;
    sc = 0;
    cs = 0;
    baseBeg = 0;
    baseEnd = 0;
    peakBeg = 0;
    peakEnd = 0;
    fitBeg = 0;
    fitEnd = 0;
    PSlopeBeg = 0;
    PSlopeEnd = 0;
    measCursor = 0;
    latencyStartCursor = 0.0;
    latencyEndCursor = 0.0;
    latency = 0.0;
    base = 0.0;
    APBase = 0.0;
    baseSD = 0.0;
    threshold = 0.0;
    slopeForThreshold = 20.0;
    peak = 0.0;
    APPeak = 0.0;
    t20Real = 0;
    t80Real = 0;
    t50LeftReal = 0;
    t50RightReal = 0;
    maxT = 0.0;
    thrT = -1.0;
    maxRiseY = 0.0;
    maxRiseT = 0.0;
    maxDecayY = 0.0;
    maxDecayT = 0.0;
    maxRise = 0.0;
    maxDecay = 0.0;
    t50Y = 0.0;
    APMaxT = 0.0;
    APMaxRise = 0.0;
    APMaxRiseT = 0.0;
    APt50LeftReal = 0.0;
    rt2080 = 0.0;
    halfDuration = 0.0;
    slopeRatio = 0.0;
    t0Real = 0.0;
    pM = 1;
    PSlope = 0.0;
    DeltaT = 0;
    selectedSections = std::vector<std::size_t>(0);
    selectBase = Vector_double(0);
    t20Index = 0;
    t80Index = 0;
    t50LeftIndex = 0;
    t50RightIndex = 0;
    fromBase = true;
    viewCrosshair = true;
    viewBaseline = true;
    viewBaseSD = true;
    viewThreshold = false;
    viewPeakzero = true;
    viewPeakbase = true;
    viewPeakthreshold = false;
    viewRT2080 = true;
    viewT50 = true;
    viewRD = true;
    viewSloperise = true;
    viewSlopedecay = true;
    viewLatency = true;
    viewPSlope = true;
    viewCursors = true;
    zoom = XZoom(0, 0.1, false);

#endif

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

#ifndef MODULE_ONLY

void Recording::correctRangeR(int& value) {
    if (value<0) {
        value=0;
        return;
    }
    if (value>=(int)cur().size()) {
        value=(int)cur().size()-1;
        return;
    }
}

void Recording::correctRangeR(std::size_t& value) {
    if (value<0) {
        value=0;
        return;
    }
    if (value>=cur().size()) {
        value=cur().size()-1;
        return;
    }
}

void Recording::CopyCursors(const Recording& c_Recording) {
    measCursor=c_Recording.measCursor;
    correctRangeR(measCursor);
    baseBeg=c_Recording.baseBeg;
    correctRangeR(baseBeg);
    baseEnd=c_Recording.baseEnd;
    correctRangeR(baseEnd);
    peakBeg=c_Recording.peakBeg;
    correctRangeR(peakBeg);
    peakEnd=c_Recording.peakEnd;
    correctRangeR(peakEnd);
    fitBeg=c_Recording.fitBeg;
    correctRangeR(fitBeg);
    fitEnd=c_Recording.fitEnd;
    correctRangeR(fitEnd);
    PSlopeBeg = c_Recording.PSlopeBeg; // PSlope left cursor
    correctRangeR(PSlopeBeg);
    PSlopeEnd = c_Recording.PSlopeEnd; // PSlope right cursor
    correctRangeR(PSlopeEnd);
 
    pM=c_Recording.pM;  //peakMean, number of points used for averaging
    DeltaT=c_Recording.DeltaT;  //distance (number of points) from first cursor 

}

void Recording::SetLatencyStartMode(int value) {
    switch (value) {
    case 1:
        latencyStartMode=stf::peakMode;
        break;
    case 2:
        latencyStartMode=stf::riseMode;
        break;
    case 3:
        latencyStartMode=stf::halfMode;
        break;
    case 4:
        latencyStartMode=stf::footMode;
    case 0:
    default:
        latencyStartMode=stf::manualMode;
    }
}

void Recording::SetLatencyEndMode(int value) {
    switch (value) {
    case 1:
        latencyEndMode=stf::peakMode;
        break;
    case 2:
        latencyEndMode=stf::riseMode;
        break;
    case 3:
        latencyEndMode=stf::halfMode;
        break;
    case 4:
        latencyEndMode=stf::footMode;
    case 0:
    default:
        latencyEndMode=stf::manualMode;
    }
}

void Recording::SetLatencyWindowMode(int value) {
    if ( value == 1 ) {
        latencyWindowMode = stf::windowMode;
    } else {
        latencyWindowMode = stf::defaultMode;
    }
}

void Recording::MakeAverage(Section& AverageReturn,
        Section& SigReturn,
        std::size_t channel,
        const std::vector<std::size_t>& section_index,
        bool isSig,
        const std::vector<int>& shift) const
{
    int n_sections=(int)section_index.size();

    for (int k=0; k < (int)AverageReturn.size(); ++k) {
        AverageReturn[k]=0.0;
        //Calculate average
        for (int l = 0; l < n_sections; ++l) {
            AverageReturn[k] += 
                ChannelArray[channel][section_index[l]][k+shift[l]];
        }
        AverageReturn[k] /= n_sections;
        if (isSig) {
            SigReturn[k]=0.0;
            //Calculate variance
            for (int l =0; l< n_sections; ++l) {
                SigReturn[k] += 
                    pow(ChannelArray[channel][section_index[l]][k+shift[l]] -
                            AverageReturn[k], 2);
            }
            SigReturn[k] /= (n_sections - 1);
            SigReturn[k]=sqrt(SigReturn[k]);
        }
    }
}

void Recording::SetCurCh(size_t value) {
    if (value<0 || value>=ChannelArray.size()) {
        throw std::out_of_range("channel out of range in Recording::set_cc()");
    }
    cc=value;
}

void Recording::SetSecCh(size_t value) {
    if (value<0 ||
            value>=ChannelArray.size() ||
            value==cc)
    {
        throw std::out_of_range("channel out of range in Recording::set_sc()");
    }
    sc=value;
}

void Recording::SetCurSec( size_t value ) {
    if (value<0 || value>=ChannelArray[cc].size()) {
        throw std::out_of_range("channel out of range in Recording::set_cs()");
    }
    cs=value;
}

void Recording::SetMeasCursor(int value) {
    correctRangeR(value);
    measCursor=value;
}

double Recording::GetMeasValue() const {
    try {
        return cur().at(measCursor);
    }
    catch (...) {
        throw;
    }
}

void Recording::SetBaseBeg(int value) {
    correctRangeR(value);
    baseBeg=value;
}

void Recording::SetBaseEnd(int value) {
    correctRangeR(value);
    baseEnd=value;
}

void Recording::SetPeakBeg(int value) {
    correctRangeR(value);
    peakBeg=value;
}

void Recording::SetPeakEnd(int value) {
    correctRangeR(value);
    peakEnd=value;
}

void Recording::SetFitBeg(int value) {
    correctRangeR(value);
    fitBeg=value;
}

void Recording::SetFitEnd(int value) {
    correctRangeR(value);
    fitEnd=value;
}

void Recording::SetLatencyBeg(double value) {
    if (value<0.0) {
        value=0.0;
    }
    if (value>=(double)cur().size()) {
        value=cur().size()-1.0;
    }
    latencyStartCursor=value;
}

void Recording::SetLatencyEnd(double value) {
    if (value<0.0) {
        value=0.0;
    }
    if (value>=(double)cur().size()) {
        value=cur().size()-1.0;
    }
    latencyEndCursor=value;
}

void Recording::SetPSlopeBeg(int value) {
    correctRangeR(value);
    PSlopeBeg = value;
}

void Recording::SetPSlopeEnd(int value) {
    correctRangeR(value);
    PSlopeEnd = value;
}

void Recording::SelectTrace(std::size_t sectionToSelect) {
    // Check range so that sectionToSelect can be used
    // without checking again:
    if (sectionToSelect<0 ||
            sectionToSelect>=ChannelArray[cc].size()) 
    {
        std::out_of_range e("subscript out of range in Recording::SelectTrace\n");
        throw e;
    }
    selectedSections.push_back(sectionToSelect);
    double sumY=0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sumY)
#endif
    for (int i=(int)baseBeg;i<=(int)baseEnd;i++) {
        sumY+=ChannelArray[cc][sectionToSelect][i];
    }
    int n=(int)(baseEnd-baseBeg+1);
    selectBase.push_back(sumY/n);
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

//Function calculates the peak and respective measures: base, 20/80 rise time
//half duration, ratio of rise/slope and maximum slope
void Recording::Measure( )
{
    double var=0.0;
    if (cur().get().size() == 0) return;
    try {
        cur().at(0);
    }
    catch (const std::out_of_range&) {
        return;
    }
    //Begin peak and base calculation
    //-------------------------------
    try {
        base=stf::base(var,cur().get(),baseBeg,baseEnd,peakBeg,peakEnd);
        baseSD=sqrt(var);
        peak=stf::peak(cur().get(),base,
                       peakBeg,peakEnd,pM,direction,maxT);
        threshold = stf::threshold( cur().get(), peakBeg, peakEnd, slopeForThreshold/GetSR(), thrT );
    }
    catch (const std::out_of_range& e) {
        base=0.0;
        baseSD=0.0;
        peak=0.0;
        throw e;
    }

    //Begin 20 to 80% Rise Time calculation
    //-------------------------------------
    // 2009-06-05: reference is either from baseline or from threshold
    double reference = base;
    if (!fromBase && thrT >= 0) {
        reference = threshold;
    }
    double ampl=peak-reference;
    
    t20Real=0.0;
    try {
        // 2008-04-27: changed limits to start from the beginning of the trace
        rt2080=stf::risetime(cur().get(),reference,ampl, (double)0/*(double)baseEnd*/,
                maxT,t20Index,t80Index,t20Real);
    }
    catch (const std::out_of_range& e) {
        rt2080=0.0;
        throw e;
    }

    t80Real=t20Real+rt2080;
    rt2080/=GetSR();

    //Begin Half Duration calculation
    //-------------------------------
    t50LeftReal=0.0;
    // 2008-04-27: changed limits to start from the beginning of the trace
    //             and to stop at the end of the trace
    halfDuration = stf::t_half(cur().get(), reference, ampl, (double)0 /*(double)baseBeg*/,
            (double)cur().size()-1 /*(double)peakEnd*/,maxT, t50LeftIndex,t50RightIndex,t50LeftReal);

    t50RightReal=t50LeftReal+halfDuration;
    halfDuration/=GetSR();
    t50Y=0.5*ampl + reference;

    //Calculate the beginning of the event by linear extrapolation:
    if (latencyEndMode==stf::footMode) {
        t0Real=t20Real-(t80Real-t20Real)/3.0;
    } else {
        t0Real=t50LeftReal;
    }

    //Begin Ratio of slopes rise/decay calculation
    //--------------------------------------------
    double left_rise = (peakBeg > t0Real-1 || !fromBase) ? peakBeg : t0Real-1;
    maxRise=stf::maxRise(cur().get(),left_rise,maxT,maxRiseT,maxRiseY);
    double t_half_3=t50RightIndex+2.0*(t50RightIndex-t50LeftIndex);
    double right_decay=peakEnd<=t_half_3 ? peakEnd : t_half_3+1;
    maxDecay=stf::maxDecay(cur().get(),maxT,right_decay,maxDecayT,maxDecayY);
    //Slope ratio
    if (maxDecay !=0) slopeRatio=maxRise/maxDecay;
    else slopeRatio=0.0;
    maxRise *= GetSR();
    maxDecay *= GetSR();

    if (size()>1) {
        //Calculate the absolute peak of the (AP) Ch2 inbetween the peak boundaries
        //A direction dependent evaluation of the peak as in Ch1 does NOT exist!!

        // endResting is set to 100 points arbitrarily in the pascal version
        // (see measlib.pas) assuming that the resting potential is stable
        // during the first 100 sampling points.
        const int endResting=100;
        const int searchRange=100;
        double APBase=0.0, APPeak=0.0, APVar=0.0;
        try {
            APBase=stf::base(APVar,sec().get(),0,endResting);
            APPeak=stf::peak(sec().get(),APBase,peakBeg,peakEnd,pM,stf::up,APMaxT);
        }
        catch (const std::out_of_range& e) {
            APBase=0.0;
            APPeak=0.0;
            throw e;
        }
        //-------------------------------
        //Maximal slope in the rise before the peak
        //----------------------------
        APMaxRiseT=0.0;
        double APMaxRiseY=0.0;
        double left_APRise = peakBeg; 
        if (GetLatencyWindowMode() == stf::defaultMode ) {
            left_APRise= APMaxT-searchRange>2.0 ? APMaxT-searchRange : 2.0;
        }
        stf::maxRise(sec().get(),left_APRise,APMaxT,APMaxRiseT,APMaxRiseY);
        //End determination of the region of maximal slope in the second channel
        //----------------------------

        //-------------------------------
        //Half-maximal amplitude
        //----------------------------
        std::size_t APt50LeftIndex,APt50RightIndex;
        stf::t_half(
                sec().get(),
                APBase,
                APPeak-APBase,
                left_APRise,
                (double)sec().get().size(),
                APMaxT,
                APt50LeftIndex,
                APt50RightIndex,
                APt50LeftReal
        );
        //End determination of the region of maximal slope in the second channel
        //----------------------------
    }

    // get and set start of latency measurement:
    double latStart=0.0;
    switch (latencyStartMode) {
    // Interestingly, latencyCursor is an int in pascal, although
    // the maxTs aren't. That's why there are double type casts
    // here.
    case stf::peakMode: //Latency cursor is set to the peak						
        latStart=APMaxT;
        break;
    case stf::riseMode:
        latStart=APMaxRiseT;
        break;
    case stf::halfMode:
        latStart=APt50LeftReal;
        break;
    case stf::manualMode:
    default:
        latStart=GetLatencyBeg();
        break;
    }
    SetLatencyBeg(latStart);

    // get and set end of latency measurement:
    double latEnd=0.0;
    switch (latencyEndMode) {
    // Interestingly, latencyCursor is an int in pascal, although
    // the maxTs aren't. That's why there are double type casts
    // here.
    case stf::footMode: //Latency cursor is set to the peak						
        latEnd=t20Real-(t80Real-t20Real)/3.0;
        break;
    case stf::riseMode:
        latEnd=maxRiseT;
        break;
    case stf::halfMode:
        latEnd=t50LeftReal;
        break;
    case stf::peakMode:
        latEnd=maxT;
        break;
    case stf::manualMode:
    default:
        latEnd=GetLatencyEnd();
        break;
    }
    SetLatencyEnd(latEnd);

    SetLatency(GetLatencyEnd()-GetLatencyBeg());

    //-------------------------------------
    // Begin PSlope calculation (PSP Slope)
    //-------------------------------------

    //
    int PSlopeBegVal; 
    switch (pslopeBegMode) {

        case stf::psBeg_footMode:   // Left PSlope to commencement
            PSlopeBegVal = (int)(t20Real-(t80Real-t20Real)/3.0);
            break;

        case stf::psBeg_thrMode:   // Left PSlope to threshold
            PSlopeBegVal = (int)thrT;
            break;

        case stf::psBeg_t50Mode:   // Left PSlope to the t50
            PSlopeBegVal = (int)t50LeftReal;
            break;

        case stf::psBeg_manualMode: // Left PSlope cursor manual
        default:
            PSlopeBegVal = PSlopeBeg;
    }
    SetPSlopeBeg(PSlopeBegVal);
    
    int PSlopeEndVal;
    switch (pslopeEndMode) {

        case stf::psEnd_t50Mode:    // Right PSlope to t50rigth
            PSlopeEndVal = (int)t50LeftReal;
            break;
        case stf::psEnd_peakMode:   // Right PSlope to peak
            PSlopeEndVal = (int)maxT;
            break;
        case stf::psEnd_DeltaTMode: // Right PSlope to DeltaT time from first peak
            PSlopeEndVal = (int)(PSlopeBeg + DeltaT);
            break;
        case stf::psEnd_manualMode:
        default:
            PSlopeEndVal = PSlopeEnd;
    }
    SetPSlopeEnd(PSlopeEndVal);

    try {
        PSlope = (stf::pslope(cur().get(), PSlopeBeg, PSlopeEnd))*GetSR();
    }
    catch (const std::out_of_range& e) {
        PSlope = 0.0;
        throw e;
    }

    //-----------------------------------
    // End PSlope calculation (PSP Slope)
    //-----------------------------------

    //--------------------------
}	//End of Measure(,,,,,)

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

stf::Table Recording::CurAsTable() const {
    stf::Table table(cur().size(),size());
    try {
        for (std::size_t nRow=0;nRow<table.nRows();++nRow) {
            wxString rLabel;
            rLabel << nRow*dt;
            table.SetRowLabel(nRow,rLabel);
            for (std::size_t nCol=0;nCol<table.nCols();++nCol) {
                table.at(nRow,nCol)=ChannelArray.at(nCol).at(cs).at(nRow);
            }
        }
        for (std::size_t nCol=0;nCol<table.nCols();++nCol) {
            table.SetColLabel(nCol,ChannelArray.at(nCol).GetChannelName());
        }
    }
    catch (const std::out_of_range& e) {
        throw e;
    }
    return table;
}

stf::Table Recording::CurResultsTable() const {
    // resize table:
    std::size_t n_cols=0;
    if (viewCrosshair) n_cols++;
    if (viewBaseline) n_cols++;
    if (viewBaseSD) n_cols++;
    if (viewThreshold) n_cols++;
    if (viewPeakzero) n_cols++;
    if (viewPeakbase) n_cols++;
    if (viewPeakthreshold) n_cols++;
    if (viewRT2080) n_cols++;
    if (viewT50) n_cols++;
    if (viewRD) n_cols++;
    if (viewSloperise) n_cols++;
    if (viewSlopedecay) n_cols++;
    if (viewLatency) n_cols++;
    if (viewPSlope) n_cols++;

    std::size_t n_rows=(viewCursors? 3:1);
    stf::Table table(n_rows,n_cols);

    // Labels
    table.SetRowLabel(0,wxT("Value"));
    if (viewCursors) {
        table.SetRowLabel(1,wxT("Cursor 1"));
        table.SetRowLabel(2,wxT("Cursor 2"));
    }
    int nCol=0;
    if (viewCrosshair) table.SetColLabel(nCol++, wxT("Crosshair"));
    if (viewBaseline) table.SetColLabel(nCol++,wxT("Baseline"));
    if (viewBaseSD) table.SetColLabel(nCol++,wxT("Base SD"));
    if (viewThreshold) table.SetColLabel(nCol++,wxT("Threshold"));
    if (viewPeakzero) table.SetColLabel(nCol++,wxT("Peak (from 0)"));
    if (viewPeakbase) table.SetColLabel(nCol++,wxT("Peak (from base)"));
    if (viewPeakthreshold) table.SetColLabel(nCol++,wxT("Peak (from threshold)"));
    if (viewRT2080) table.SetColLabel(nCol++,wxT("RT (20-80%)"));
    if (viewT50) table.SetColLabel(nCol++,wxT("t50"));
    if (viewRD) table.SetColLabel(nCol++,wxT("Rise/Decay"));
    if (viewSloperise) table.SetColLabel(nCol++,wxT("Slope (rise)"));
    if (viewSlopedecay) table.SetColLabel(nCol++,wxT("Slope (decay)"));
    if (viewLatency) table.SetColLabel(nCol++,wxT("Latency"));
    if (viewPSlope) table.SetColLabel(nCol++,wxT("Slope"));

    // Values
    nCol=0;
    // measurement cursor
    if (viewCrosshair) {
        table.at(0,nCol)=GetMeasValue();
        if (viewCursors) {
            table.at(1,nCol)=GetMeasCursor()*GetXScale();
            table.SetEmpty(2,nCol,true);
        }
        nCol++;
    }

    // baseline
    if (viewBaseline) {
        table.at(0,nCol)=GetBase();
        if (viewCursors) {
            table.at(1,nCol)=GetBaseBeg()*GetXScale();
            table.at(2,nCol)=GetBaseEnd()*GetXScale();
        }
        nCol++;
    }

    // base SD
    if (viewBaseSD) {
        table.at(0,nCol)=GetBaseSD();
        if (viewCursors) {
            table.at(1,nCol)=GetBaseBeg()*GetXScale();
            table.at(2,nCol)=GetBaseEnd()*GetXScale();
        }
        nCol++;
    }

    // threshold
    if (viewThreshold) {
        table.at(0,nCol)=GetThreshold();
        if (viewCursors) {
            table.at(1,nCol)=GetPeakBeg()*GetXScale();
            table.at(2,nCol)=GetPeakEnd()*GetXScale();
        }
        nCol++;
    }
    
    // peak
    if (viewPeakzero) {
        table.at(0,nCol)=GetPeak();
        if (viewCursors) {
            table.at(1,nCol)=GetPeakBeg()*GetXScale();
            table.at(2,nCol)=GetPeakEnd()*GetXScale();
        }
        nCol++;
    }

    if (viewPeakbase) {
        table.at(0,nCol)=GetPeak()-GetBase();
        if (viewCursors) {
            table.at(1,nCol)=GetPeakBeg()*GetXScale();
            table.at(2,nCol)=GetPeakEnd()*GetXScale();
        }
        nCol++;
    }
    if (viewPeakthreshold) {
        if (thrT >= 0) {
            table.at(0,nCol) = GetPeak()-GetThreshold();
        } else {
            table.at(0,nCol) = 0;
        }
        if (viewCursors) {
            table.at(1,nCol)=GetPeakBeg()*GetXScale();
            table.at(2,nCol)=GetPeakEnd()*GetXScale();
        }
        nCol++;
    }

    // RT (20-80%)
    if (viewRT2080) {table.at(0,nCol)=GetRT2080();
        if (viewCursors) {
            table.at(1,nCol)=GetT20Real()*GetXScale();
            table.at(2,nCol)=GetT80Real()*GetXScale();
        }
        nCol++;
    }

    // Half duration
    if (viewT50) {table.at(0,nCol)=GetHalfDuration();
        if (viewCursors) {
            table.at(1,nCol)=GetT50LeftReal()*GetXScale();
            table.at(2,nCol)=GetT50RightReal()*GetXScale();
        }
        nCol++;
    }

    // Rise/decay
    if (viewRD) {table.at(0,nCol)=GetSlopeRatio();
        if (viewCursors) {
            table.at(1,nCol)=GetMaxRiseT()*GetXScale();
            table.at(2,nCol)=GetMaxDecayT()*GetXScale();
        }
        nCol++;
    }

    // Max rise
    if (viewSloperise) {table.at(0,nCol)=GetMaxRise();
        if (viewCursors) {
            table.at(1,nCol)=GetMaxRiseT()*GetXScale();
            table.SetEmpty(2,nCol,true);
        }
        nCol++;
    }

    // Max decay
    if (viewSlopedecay) {table.at(0,nCol)=GetMaxDecay();
        if (viewCursors) {
            table.at(1,nCol)=GetMaxDecayT()*GetXScale();
            table.SetEmpty(2,nCol,true);
        }
        nCol++;
    }

    // Latency
    if (viewLatency) {table.at(0,nCol)=GetLatency()*GetXScale();
        if (viewCursors) {
            table.at(1,nCol)=GetLatencyBeg()*GetXScale();
            table.at(2,nCol)=GetLatencyEnd()*GetXScale();
        }
        nCol++;
    }

    // PSlope
    if (viewPSlope) {table.at(0,nCol)=GetPSlope();
        if (viewCursors) {
            table.at(1,nCol)=GetPSlopeBeg()*GetXScale();
            table.at(2,nCol)=GetPSlopeEnd()*GetXScale();
        }
        nCol++;
    }

    return table;
}
#endif
