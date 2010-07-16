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

/*! \file recording.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares the Recording class.
 */

#ifndef _RECORDING_H
#define _RECORDING_H

/*! \defgroup stfgen Generic stimfit classes and functions
 *  @{
 */

#include "./channel.h"
#include "./stimdefs.h"

//! Represents the data within a file.
/*! Contains an array of channels that can be accessed either via at() (range-checked,
 *  will throw an exception if out of range) or the []-operator (range unchecked). Moreover
 *  all the metadata such as time, date, samling rate and comments are stored here.
 */
class StfDll Recording {
 public:

    //ctor/dtor-------------------------------------------------------
    //! Default constuctor
    explicit Recording();

    //! Constructor
    /*! \param c_Channel The Channel from which to construct a new Recording.
     */
    explicit Recording(const Channel& c_Channel); 

    //! Constructor
    /*! Setting the number of channels and sections at construction time will avoid unnecessary 
     *  memory re-allocations.
     *  \param c_n_channels The number of channels.
     *  \param c_n_sections The number of sections.
     *  \param c_n_points The number of sampling points per section.
     */
    explicit Recording( std::size_t c_n_channels, std::size_t c_n_sections = 0, std::size_t c_n_points = 0 );

    //! Destructor
    virtual ~Recording();

    //member access functions: read-----------------------------------
    
    //! Retrieves the number of sections in a channel.
    /*! \param n_channel The index of the channel (range-checked).
     *  \return The number of sections in n_channel.
     */
    std::size_t GetChannelSize(std::size_t n_channel) const;

    //! Retrieves the channels (read-only).
    /*! \return A vector containing the channels.
     */
    const std::vector<Channel>& get() const { return ChannelArray; }
    
    //! Retrieves the channels (read and write).
    /*! \return A vector containing the channels.
     */
    std::vector<Channel>& get() { return ChannelArray; }
    
    //! Retrieves the file description.
    /*! \return The file description.
     */
    const wxString& GetFileDescription() const { return file_description; }

    //! Retrieves the common section description.
    /*! \return The common section description.
     */
    const wxString& GetGlobalSectionDescription() const { return global_section_description; }

    //! Retrieves the scaling as a string.
    /*! \return A string containing the description.
     */
    const wxString& GetScaling() const { return scaling; }

    //! Retrieves the time of recording as a string.
    /*! \return A string containing the time of recording.
     */
    const wxString& GetTime() const { return time; }

    //! Retrieves the date of recording as a string.
    /*! \return A string containing the date of recording.
     */
    const wxString& GetDate() const { return date; }

    //! Retrieves a comment string.
    /*! \return A string containing a comment.
     */
    const wxString& GetComment() const { return comment; }
    
    //! Retrieves the x units.
    /*! \return The x units. Currently hard-coded to be "ms".
     */
    const wxString& GetXUnits() const { return xunits; }

    //! Retrieves the size of the channel array.
    /*! \return The size of the channel array.
     */
    std::size_t size() const { return ChannelArray.size(); }
    
    //! Retrieves the x scaling (sampling interval).
    /*! \return The x scaling.
     */
    double GetXScale() const { return dt; }
    
    //! Retrieves the sampling rate ( 1 / x-scale )
    /*! \return The sampling rate.
     */
    double GetSR() const { return 1.0/dt; }

    //! Range-checked access to a channel (read-only).
    /*! Will throw std::out_of_range if out of range.
     *  \param n_c The index of the channel.
     *  \return The channel at index n_c.
     */
    const Channel& at(std::size_t n_c) const;

    //! Range-checked access to a channel (read and write).
    /*! Will throw std::out_of_range if out of range.
     *  \param n_c The index of the channel.
     *  \return The channel at index n_c.
     */
    Channel& at(std::size_t n_c);

#ifndef MODULE_ONLY

    //! Retrieves the index of the current channel.
    /*! \return The index of the current channel.
     */
    std::size_t GetCurCh() const { return cc; }

    //! Retrieves the index of the second channel.
    /*! \return The index of the second channel.
     */
    std::size_t GetSecCh() const { return sc; }

    //! Retrieves the index of the current section.
    /*! \return The index of the current section.
     */
    std::size_t GetCurSec() const { return cs; }

    //! Retrieves the position of the measurement cursor (crosshair).
    /*! \return The index of the measurement cursor within the current section.
     */
    std::size_t GetMeasCursor() const { return measCursor; }

    //! Retrieves the position of the left baseline cursor.
    /*! \return The index of the left baseline cursor within the current section.
     */
    std::size_t GetBaseBeg() const { return baseBeg; }

    //! Retrieves the position of the right baseline cursor
    /*! \return The index of the left baseline cursor within the current section.
     */
    std::size_t GetBaseEnd() const { return baseEnd; }

    //! Retrieves the position of the left peak cursor.
    /*! \return The index of the left peak cursor within the current section.
     */
    std::size_t GetPeakBeg() const { return peakBeg; }

    //! Retrieves the position of the right peak cursor.
    /*! \return The index of the right peak cursor within the current section.
     */
    std::size_t GetPeakEnd() const { return peakEnd; }

    //! Retrieves the position of the left fitting cursor.
    /*! \return The index of the left fitting cursor within the current section.
     */
    std::size_t GetFitBeg() const { return fitBeg; }

    //! Retrieves the position of the right fitting cursor.
    /*! \return The index of the right fitting cursor within the current section.
     */
    std::size_t GetFitEnd() const { return fitEnd; }

    //! Retrieves the position of the left PSlope cursor.
    /*! \return The index of the left PSlope cursor within the current section.
     */
    std::size_t GetPSlopeBeg() const { return PSlopeBeg; }

    //! Retrieves the position of the right PSlope cursor.
    /*! \return The index of the right PSlope cursor within the current section.
     */
    std::size_t GetPSlopeEnd() const { return PSlopeEnd; }

    //! Retrieves the number of points used for averaging during peak calculation.
    /*! \return The number of points to be used.
     */
    int GetPM() const { return pM; }

    //! Retrieves the position of the left latency cursor.
    /*! \return The index of the left latency cursor within the current section. Note that by contrast
     *  to the other cursors, this is a double because the latency cursor may be set to an interpolated
     *  position between two data points.
     */
    double GetLatencyBeg() const { return latencyStartCursor; }

    //! Retrieves the position of the right latency cursor.
    /*! \return The interpolated index of the right latency cursor within the current section. Note that
     *  by contrast to the other cursors, this is a double because the latency cursor may be set to an
     *  interpolated position between two data points.
     */
    double GetLatencyEnd() const { return latencyEndCursor; }
    
    //! Retrieves the latency.
    /*! \return The latency, expressed in units of data points.
     */
    double GetLatency() const { return latency; }

    //! Retrieves the time point at which 20% of the maximal amplitude have been reached.
    /*! \return The time point at which 20% of the maximal amplitude have been reached, expressed in
     *  units of data points.
     */
    double GetT20Real() const { return t20Real; }

    //! Retrieves the time point at which 80% of the maximal amplitude have been reached.
    /*! \return The time point at which 80% of the maximal amplitude have been reached, expressed in
     *  units of data points.
     */
    double GetT80Real() const { return t80Real; }

    //! Retrieves the time point at which 50% of the maximal amplitude have been reached from the left of the peak.
    /*! \return The time point at which 50% of the maximal amplitude have been reached from the left of the peak, 
     *  expressed in units of data points.
     */
    double GetT50LeftReal() const { return t50LeftReal; }

    //! Retrieves the time point at which 50% of the maximal amplitude have been reached from the right of the peak.
    /*! \return The time point at which 50% of the maximal amplitude have been reached from the right of the peak, 
     *  expressed in units of data points.
     */
    double GetT50RightReal() const { return t50RightReal; }

    //! Retrieves the y value at 50% of the maximal amplitude.
    /*! \return The y value at 50% of the maximal amplitude.
     */
    double GetT50Y() const { return t50Y; }

    //! Retrieves the maximal slope of the rising phase.
    /*! \return The maximal slope during the rising phase.
     */
    double GetMaxRise() const { return maxRise; }

    //! Retrieves the maximal slope of the decaying phase.
    /*! \return The maximal slope of rise.
     */
    double GetMaxDecay() const { return maxDecay; }

    //! Retrieves the time point of the maximal slope of the rising phase in the second channel.
    /*! This time point is needed as a reference for the latency calculation and for aligned averages.
     *  \return The time point at which the maximal slope of the rising phase is reached in the second channel, 
     *  expressed in units of data points..
     */
    double GetAPMaxRiseT() const { return APMaxRiseT; }

    //! Retrieves the time point of the peak in the second channel.
    /*! \return The time point at which the peak is found in the second channel, 
     *  expressed in units of data points.
     */
    double GetAPMaxT() const { return APMaxT; }

    //! Retrieves the time point at which 50% of the max. amplitude have been reached from the left of the peak in the inactive channel.
    /*! \return The time point at which 50% of the maximal amplitude have been reached from the left of the peak 
     *  in the inactive channel, expressed in units of data points.
     */
    double GetAPT50LeftReal() const { return APt50LeftReal; }

    //! Retrieves the time point of the maximal slope during the rising phase.
    /*! \return The time point of the maximal slope during the rising phase, expressed in units of data points.
     */
    double GetMaxRiseT() const { return maxRiseT; }

    //! Retrieves the y-value at the time point of the maximal slope during the rising phase.
    /*! \return The y-value at the time point of the maximal slope during the rising phase.
     */
    double GetMaxRiseY() const { return maxRiseY; }

    //! Retrieves the time point of the maximal slope during the decaying phase.
    /*! \return The time point of the maximal slope during the decaying phase, expressed in units of data points.
     */
    double GetMaxDecayT() const { return maxDecayT; }

    //! Retrieves the y-value at the time point of the maximal slope during the decaying phase.
    /*! \return The y-value at the time point of the maximal slope during the decaying phase.
     */
    double GetMaxDecayY() const { return maxDecayY; }
    
    //! Retrieves the y-value at the measurement cursor (crosshair).
    /*! \return The y-value at the measurement cursor.
     */
    double GetMeasValue() const;
    
    //! Retrieves the peak value.
    /*! \return The peak value.
     */
    double GetPeak() const { return peak; }
    
    //! Retrieves the baseline.
    /*! \return The baseline value.
     */
    double GetBase() const { return base; }

    //! Retrieves the baseline in the second channel.
    /*! \return The baseline value in the second channel.
     */
    double GetAPBase() const { return APBase; }
    
    //! Retrieves the standard deviation of the baseline.
    /*! \return The standard deviation of the baseline.
     */
    double GetBaseSD() const { return baseSD; }
    
    //! Retrieves the value at which the threshold slope is crossed.
    /*! \return The standard deviation of the baseline.
     */
    double GetThreshold() const { return threshold; }
    
    //! Retrieves the time point at which the peak is found.
    /*! \return The time point at which the peak is found, expressed in units of data points.
     */
    double GetMaxT() const { return maxT; }
    
    //! Retrieves the time point at which the threshold slope is crossed.
    /*! \return The time point at which the threshold slope is crossed, or
     *          a negative value if the threshold is not attained.
     */
    double GetThrT() const { return thrT; }
    
    //! Retrieves the 20 to 80% rise time.
    /*! \return The difference between GetT80Real() and GetT20Real(), expressed in units o data points.
     */
    double GetRT2080() const { return rt2080; }
    
    //! Retrieves the full width at half-maximal amplitude ("half duration").
    /*! \return The difference between GetT50RightReal() and GetT50LeftReal(), expressed in units of data points.
     */
    double GetHalfDuration() const { return halfDuration; }
    
    //! Retrieves ratio of the maximal slopes during the rising and decaying phase.
    /*! \return The ratio of GetMaxRise() and GetMaxDecay().
     */
    double GetSlopeRatio() const { return slopeRatio; }

    //! Retrieves the value of the Slope
    /*! \return slope value in y-units/x-units.
    */
    double GetPSlope() const { return PSlope; }

    //! Retrieves the mode of the latency start cursor.
    /*! \return The current mode of the latency start cursor..
     */
    stf::latency_mode GetLatencyStartMode() const { return latencyStartMode; }

    //! Retrieves the mode of the latency end cursor.
    /*! \return The current mode of the latency end cursor.
     */
    stf::latency_mode GetLatencyEndMode() const { return latencyEndMode; }
    
    //! Retrieves the mode of the latency window.
    /*! \return The current mode of the latency window.
     */
    stf::latency_window_mode GetLatencyWindowMode() const { return latencyWindowMode; }

    //! Retrieves the direction of peak calculations.
    /*! \return The current direction of peak calculations.
     */
    stf::direction GetDirection() const { return direction; }
    
    //! Retrieves the mode of the left PSlope cursor.
    /*! \return The current mode of the left PSlope cursor.
     */
    stf::pslope_mode_beg GetPSlopeBegMode() const { return pslopeBegMode; }

    //! Retrieves the mode of the right PSlope cursor.
    /*! \return The current mode of the right PSlope cursor.
     */
    stf::pslope_mode_end GetPSlopeEndMode() const { return pslopeEndMode; }

    //! Indicates whether to use the baseline as a reference for AP kinetics.
    /*! \return true if the baseline should be used, false if the threshold should be used.
     */
    bool GetFromBase() const { return fromBase; }

    //! Indicates whether the measurement cursor (crosshair) value should be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewCrosshair() const { return viewCrosshair; }

    //! Indicates whether the baseline value should be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewBaseline() const { return viewBaseline; }

    //! Indicates whether the baseline's standard deviation should be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewBaseSD() const { return viewBaseSD; }

    //! Indicates whether the threshold should be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewThreshold() const { return viewThreshold; }

    //! Indicates whether the peak value (measured from zero) should be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewPeakZero() const { return viewPeakzero; }

    //! Indicates whether the peak value (measured from baseline) should be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewPeakBase() const { return viewPeakbase; }

    //! Indicates whether the peak value (measured from threshold) should be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewPeakThreshold() const { return viewPeakthreshold; }

    //! Indicates whether the 20 to 80% rise time should be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewRT2080() const { return viewRT2080; }

    //! Indicates whether the half duration should be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewT50() const { return viewT50; }

    //! Indicates whether the ratio of the maximal slopes during rise and decay should be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewRD() const { return viewRD; }

    //! Indicates whether the maximal slope during the rising phase should be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewSlopeRise() const { return viewSloperise; }

    //! Indicates whether the maximal slope during the decaying phase should be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewSlopeDecay() const { return viewSlopedecay; }

    //! Indicates whether the latency should be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewLatency() const { return viewLatency; }

    //! Indicates whether the Slopeshould be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewPSlope() const { return viewPSlope; }

    //! Indicates whether two additional rows showing the positions of start and end cursors should be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewCursors() const { return viewCursors; }
    
    //! Returns the slope for threshold detection.
    /*! \return The slope value for threshold detection.
     */
    double GetSlopeForThreshold() const { return slopeForThreshold; }
    
    //! Retrieves the indices of the selected sections (read-only).
    /*! \return A vector containing the indices of the selected sections.
     */
    const std::vector<std::size_t>& GetSelectedSections() const { return selectedSections; } 

    //! Retrieves the indices of the selected sections (read and write).
    /*! \return A vector containing the indices of the selected sections.
     */
    std::vector<std::size_t>& GetSelectedSectionsW() { return selectedSections; } 

    //! Retrieves the stored baseline values of the selected sections (read-only).
    /*! \return A vector containing the stored baseline values of the selected sections.
     */
    const Vector_double& GetSelectBase() const { return selectBase; } 

    //! Retrieves the stored baseline values of the selected sections (read and write).
    /*! \return A vector containing the stored baseline values of the selected sections.
     */
    Vector_double& GetSelectBaseW() { return selectBase; }

    //! Retrieves the currently accessed section in the active channel (read-only)
    /*! \return The currently accessed section in the active channel.
     */
    const Section& cur() const { return ChannelArray[cc][cs]; }

    //! Retrieves the currently accessed section in the active channel (read and write)
    /*! \return The currently accessed section in the active channel.
     */
    Section& cur() { return ChannelArray[cc][cs]; }

    //! Retrieves the currently accessed section in the second (inactive) channel (read-only)
    /*! \return The currently accessed section in the second (inactive) channel.
     */
    const Section& sec() const { return ChannelArray[sc][cs]; }

#endif

    //member access functions: write---------------------------------

    //! Sets the file description.
    /*! \param value The file description.
     */
    void SetFileDescription(const wxString& value) { file_description=value; }

    //! Sets the common section description.
    /*! \param value The common section description.
     */
    void SetGlobalSectionDescription(const wxString& value) {
        global_section_description=value;
    }

    //! Sets the scaling as a string.
    /*! \param value A string containing the description.
     */
    void SetScaling(const wxString& value) { scaling=value; }
 
    //! Sets the time of recording as a string.
    /*! \param value A string containing the time of recording.
     */
    void SetTime(const wxString& value) { time=value; }

    //! Sets the date of recording as a string.
    /*! \param value A string containing the date of recording.
     */
    void SetDate(const wxString& value) { date=value; }

    //! Sets a comment string.
    /*! \param value A string containing a comment.
     */
    void SetComment(const wxString& value) { comment=value; }
    
    //! Sets the y units for a channel.
    /*! \param n_channel The channel index for which to set the units.
     *  \param value A string containing the y units.
     */
    void SetGlobalYUnits(std::size_t n_channel, const wxString& value);

    //! Sets the x units.
    /*! \param value A string containing the x units.
     */
    void SetXUnits(const wxString& value) { xunits=value; }

    //! Sets the x scaling.
    /*! Note that setting the global x-scale will set it for all sections
     *  \param value The x scaling.
     */
    void SetXScale(double value);
    
#ifndef MODULE_ONLY
    
    //! Sets the index of the current channel.
    /*! \param value The index of the current channel.
     */
    void SetCurCh(std::size_t value);

    //! Sets the index of the second channel.
    /*! \param value The index of the second channel.
     */
    void SetSecCh(std::size_t value);

    //! Sets the index of the current section.
    /*! \param value The index of the current section.
     */
    void SetCurSec(std::size_t value);

    //! Selects a section
    /*! \param sectionToSelect The index of the section to be selected.
     */
    void SelectTrace(std::size_t sectionToSelect);

    //! Unselects a section if it was selected before
    /*! \param sectionToUnselect The index of the section to be unselected.
     *  \return true if the section was previously selected, false otherwise.
     */
    bool UnselectTrace(std::size_t sectionToUnselect);

    //! Sets the position of the measurement cursor (crosshair).
    /*! \param value The index of the measurement cursor within the current section.
     */
    void SetMeasCursor(int value);

    //! Sets the position of the left baseline cursor.
    /*! \param value The index of the left baseline cursor within the current section.
     */
    void SetBaseBeg(int value);

    //! Sets the position of the right baseline cursor
    /*! \param value The index of the left baseline cursor within the current section.
     */
    void SetBaseEnd(int value);

    //! Sets the position of the left peak cursor.
    /*! \param value The index of the left peak cursor within the current section.
     */
    void SetPeakBeg(int value);

    //! Sets the position of the right peak cursor.
    /*! \param value The index of the right peak cursor within the current section.
     */
    void SetPeakEnd(int value);

    //! Sets the position of the left fitting cursor.
    /*! \param value The index of the left fitting cursor within the current section.
     */
    void SetFitBeg(int value);

    //! Sets the position of the right fitting cursor.
    /*! \param value The index of the right fitting cursor within the current section.
     */
    void SetFitEnd(int value);

    //! Sets the position of the left latency cursor.
    /*! \param value The index of the left latency cursor within the current section. Note that by contrast
     *  to the other cursors, this is a double because the latency cursor may be set to an interpolated
     *  position between two data points.
     */
    void SetLatencyBeg(double value);

    //! Sets the position of the right latency cursor.
    /*! \param value The index of the right latency cursor within the current section. Note that by contrast
     *  to the other cursors, this is a double because the latency cursor may be set to an interpolated
     *  position between two data points.
     */
    void SetLatencyEnd(double value);

    //! Sets the latency.
    /*! \param value The latency, expressed in units of data points.
     */
    void SetLatency(double value) { latency=value; }

    //! Sets the position of the left PSlope cursor.
    /*! \param value The index of the left PSlope cursor within the current section.
     */
    void SetPSlopeBeg(int value);

    //! Sets the position of the right PSlope cursor.
    /*! \param value The index of the right PSlope cursor within the current section.
     */
    void SetPSlopeEnd(int value);

    //! Sets the PSlope.
    /*! \param value The slope, expressed in y-units/x-units.
     */
    void SetPSlope(double value) { PSlope=value; }

    //! Set the position mode of the left PSlope cursor.
    /*! \param value The new mode of the left PSlope cursor.
     */
    void SetPSlopeBegMode(stf::pslope_mode_beg value) { pslopeBegMode=value; }

    //! Set the position mode of the right PSlope cursor.
    /*! \param value The new mode of the right PSlope cursor.
     */
    void SetPSlopeEndMode(stf::pslope_mode_end value) { pslopeEndMode=value; }

    //! Sets the number of points used for averaging during peak calculation.
    /*! \param value The number of points to be used.
     */
    void SetPM(int value) { pM=value; }

    //! Sets the mode of the latency start cursor.
    /*! \param value The new mode of the latency start cursor..
     */
    void SetLatencyStartMode(stf::latency_mode value) {
        latencyStartMode=value;
    }

    //! Sets the mode of the latency end cursor.
    /*! \param value The new mode of the latency end cursor..
     */
    void SetLatencyEndMode(stf::latency_mode value) {
        latencyEndMode=value;
    }

    //! Sets the mode of the latency end cursor.
    /*! \param value The new mode of the latency end cursor..
     */
    void SetLatencyWindowMode(stf::latency_window_mode value) {
        latencyWindowMode=value;
    }
    
    //! Sets the mode of the latency start cursor.
    /*! \param value The new mode of the latency start cursor..
     */
    void SetLatencyStartMode(int value);

    //! Sets the mode of the latency end cursor.
    /*! \param value The new mode of the latency end cursor..
     */
    void SetLatencyEndMode(int value);
    
    //! Sets the mode of the latency end cursor.
    /*! \param value The new mode of the latency end cursor..
     */
    void SetLatencyWindowMode(int value);

    //! Sets the direction of peak calculations.
    /*! \param value The new direction of peak calculations.
     */
    void SetDirection(stf::direction value) { direction=value; }

    //! Sets the reference for AP kinetics measurements.
    /*! \param frombase true if the baseline should be used, false if the threshold should be used.
     */
    void SetFromBase(bool frombase) { fromBase = frombase; }
    
    //! Determines whether the measurement cursor (crosshair) value should be shown in the results table.
    /*! \param value Set to true if it should be shown, false otherwise.
     */
    void SetViewCrosshair(bool value) { viewCrosshair=value; }

    //! Determines whether the baseline value should be shown in the results table.
    /*! \param value Set to true if it should be shown, false otherwise.
     */
    void SetViewBaseline(bool value) { viewBaseline=value; }

    //! Determines whether the baseline's standard deviation should be shown in the results table.
    /*! \param value Set to true if it should be shown, false otherwise.
     */
    void SetViewBaseSD(bool value) { viewBaseSD=value; }

    //! Determines whether the threshold should be shown in the results table.
    /*! \param value Set to true if it should be shown, false otherwise.
     */
    void SetViewThreshold(bool value) { viewThreshold=value; }

    //! Determines whether the peak value (measured from zero) should be shown in the results table.
    /*! \param value Set to true if it should be shown, false otherwise.
     */
    void SetViewPeakZero(bool value) { viewPeakzero=value; }

    //! Determines whether the peak value (measured from baseline) should be shown in the results table.
    /*! \param value Set to true if it should be shown, false otherwise.
     */
    void SetViewPeakBase(bool value) { viewPeakbase=value; }

    //! Determines whether the peak value (measured from threshold) should be shown in the results table.
    /*! \param value Set to true if it should be shown, false otherwise.
     */
    void SetViewPeakThreshold(bool value) { viewPeakthreshold=value; }

    //! Determines whether the 20 to 80% rise time should be shown in the results table.
    /*! \param value Set to true if it should be shown, false otherwise.
     */
    void SetViewRT2080(bool value) { viewRT2080=value; }

    //! Determines whether the half duration should be shown in the results table.
    /*! \param value Set to true if it should be shown, false otherwise.
     */
    void SetViewT50(bool value) { viewT50=value; }

    //! Determines whether the ratio of the maximal slopes during rise and decay should be shown in the results table.
    /*! \param value Set to true if it should be shown, false otherwise.
     */
    void SetViewRD(bool value) { viewRD=value; }

    //! Determines whether the maximal slope during the rising phase should be shown in the results table.
    /*! \param value Set to true if it should be shown, false otherwise.
     */
    void SetViewSlopeRise(bool value) { viewSloperise=value; }

    //! Determines whether the maximal slope during the decaying phase should be shown in the results table.
    /*! \param value Set to true if it should be shown, false otherwise.
     */
    void SetViewSlopeDecay(bool value) { viewSlopedecay=value; }

    //! Determines whether the latency should be shown in the results table.
    /*! \param value Set to true if it should be shown, false otherwise.
     */
    void SetViewLatency(bool value) { viewLatency=value; }

    //! Determines whether the slope should be shown in the results table.
    /*! \param value Set to true if it should be shown, false otherwise.
     */
    void SetViewPSlope(bool value) { viewPSlope=value; }

    //! Determines whether two additional rows showing the positions of start and end cursors should be shown in the results table.
    /*! \param value Set to true if they should be shown, false otherwise.
     */
    void SetViewCursors(bool value) { viewCursors=value; }

    //! Sets the slope where the baseline should be set.
    /*! \param value The slope value where the baseline shoudl be set.
     */
    void SetSlopeForThreshold(double value) { slopeForThreshold=value; }
#endif
    
    //misc-----------------------------------------------------------

    //! Resize the Recording to a new number of channels.
    /*! Resizes both the channel and the global y units arrays.
     *  \param c_n_channels The new number of channels.
     */
    void resize(std::size_t c_n_channels);

    //! Insert a Channel at a given position.
    /*! Will throw std::out_of_range if range check fails.
     *  \param c_Channel The Channel to be inserted.
     *  \param pos The position at which to insert the channel (0-based).
     */
    void InsertChannel(Channel& c_Channel, std::size_t pos);

    //! Copy descriptive attributes from another Recording to this Recording.
    /*! This will copy the file and global section decription, the scaling, time, date, 
     *  comment and global y units strings and the x-scale.
     *  \param c_Recording The Recording from which to copy the attributes.
     */
    void CopyAttributes(const Recording& c_Recording);

#ifndef MODULE_ONLY
    //! Copies the cursor positions from another Recording to this Recording.
    /*! This will copy the crosshair, base, peak and fit cursors positions as 
     *  well as the number of points for peak averaging from another Recording 
     *  and correct the new values if they are out of range. The latency cursors 
     *  will not be copied.
     *  \param c_Recording The Recording from which to copy the cursor positions.
     */
    void CopyCursors(const Recording& c_Recording);
    
    //! Measure everything using functions defined in measlib.h
    /*! This will measure the baseline, peak values, 20 to 80% rise time, 
     *  half duration, maximal slopes during rise and decay, the ratio of these slopes 
     *  and the latency.
     */
    void Measure();

    //! Calculates an average of several traces.
    /*! \param AverageReturn The average will be returned in this variable by passing 
     *         a reference. AverageReturn has to have the correct size upon entering 
     *         this function already, it won't be resized.
     *  \param SigReturn The standard deviation will be returned in this variable by 
     *         passing a reference if isSig == true. SigReturn has to have the correct 
     *         size upon entering this function already, it won't be resized.
     *  \param channel The index of the channel to be used.
     *  \param section_index A vector containing the indices of the sections to be
     *         used for the average.
     *  \param isSig Set to true if the standard deviation should be calculated as well.
     *  \param shift A vector indicating by how many data points each section should be
     *         shifted before averaging.
     */
    void MakeAverage( Section& AverageReturn, Section& SigReturn, std::size_t channel,
                      const std::vector<std::size_t>& section_index, bool isSig,
                      const std::vector<int>& shift) const;

    //! Add a Recording at the end of this Recording.
    /*! \param toAdd The Recording to be added.
     */
    void AddRec(const Recording& toAdd);
    
    //! Put the current trace into a text table.
    stf::Table CurAsTable() const;
    
    //! Put the current measurement results into a text table.
    stf::Table CurResultsTable() const;
    
    //! Returns the current zoom settings for this channel (read-only).
    /*! \return The current zoom settings.
     */
    const XZoom& GetXZoom() { return zoom; }

    //! Returns the current zoom settings for this channel (read & write).
    /*! \return The current zoom settings.
     */
    XZoom& GetXZoomW() { return zoom; }
#endif
    
    //operators------------------------------------------------------

    //! Unchecked channel access (read and write)
    /*! Use at() for range-checked access.
     *  \param at The channel index.
     */
    Channel& operator[](std::size_t at) { return ChannelArray[at]; }

    //! Unchecked channel access (read-only)
    /*! Use at() for range-checked access.
     *  \param at The channel index.
     */
    const Channel& operator[](std::size_t at) const { return ChannelArray[at]; }

 private:
    std::vector<Channel> ChannelArray;
    wxString file_description, global_section_description, scaling;

#ifdef MODULE_ONLY    
 public:
#endif
    
    double dt;
    wxString time, date, comment, xunits;

 private:

#ifndef MODULE_ONLY
    stf::latency_mode latencyStartMode, latencyEndMode;
    stf::latency_window_mode latencyWindowMode;
    stf::direction	direction; //of peak detection: UP, DOWN or BOTH
    stf::pslope_mode_beg pslopeBegMode; // for left mode PSlope cursor
    stf::pslope_mode_end pslopeEndMode; // for right mode PSlope cursor

    // currently accessed channel:
    std::size_t cc;
    // second channel:
    std::size_t sc;
    // currently accessed section:
    std::size_t cs;

    std::size_t baseBeg, baseEnd, peakBeg, peakEnd, fitBeg, fitEnd, measCursor, PSlopeBeg, PSlopeEnd;
    double latencyStartCursor,
        latencyEndCursor,
        latency,	 //time from latency cursor to beginning of event
        base, APBase, baseSD, threshold, slopeForThreshold, peak, APPeak, t20Real, t80Real, t50LeftReal, t50RightReal,
        maxT, thrT, maxRiseY, maxRiseT, maxDecayY, maxDecayT, maxRise, maxDecay,
        t50Y, APMaxT, APMaxRise, APMaxRiseT, APt50LeftReal, 
        rt2080, halfDuration, slopeRatio, t0Real, PSlope;
    // cursor windows:
    int pM;  //peakMean, number of points used for averaging

    // Indices of the selected sections
    std::vector<std::size_t> selectedSections;
    // Base line value for each selected trace
    Vector_double selectBase;
    
    std::size_t t20Index, t80Index, t50LeftIndex, t50RightIndex;

    bool fromBase, viewCrosshair,viewBaseline,viewBaseSD,viewThreshold, viewPeakzero,viewPeakbase,viewPeakthreshold, viewRT2080,
        viewT50,viewRD,viewSloperise,viewSlopedecay,viewLatency,viewPSlope, viewCursors;

    XZoom zoom;

    void correctRangeR(int& value);
    void correctRangeR(std::size_t& value);

#endif

    void init();

};

/*@}*/

#endif
