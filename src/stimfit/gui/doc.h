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

/*! \file doc.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares wxStfDoc.
 */

#ifndef _DOC_H
#define _DOC_H

/*! \addtogroup wxstf
 *  @{
 */

#include "./../stf.h"

//! The document class, derived from both wxDocument and Recording.
/*! The document class can be used to model an application’s file-based data.
 *  It is part of the document/view framework supported by wxWidgets.
 */
class StfDll wxStfDoc: public wxDocument, public Recording
{
#ifndef FROM_PYTHON
    DECLARE_DYNAMIC_CLASS(wxStfDoc)
#endif
private:
    bool peakAtEnd, initialized, progress;
    Recording Average;
    int InitCursors();
    void PostInit();
    bool ChannelSelDlg();
    void WriteToReg();
    bool outOfRange(std::size_t check) {
        return (check>=cur().size() || check<0);
    }
    void Focus();
    void OnNewfromselectedThisMenu( wxCommandEvent& event ) { OnNewfromselectedThis( ); }
    void Selectsome(wxCommandEvent& event);
    void Unselectsome(wxCommandEvent& event);
    void Concatenate(wxCommandEvent& event);
    void OnAnalysisBatch( wxCommandEvent& event );
    void OnAnalysisIntegrate( wxCommandEvent& event );
    void OnAnalysisDifferentiate( wxCommandEvent& event );
    void OnSwapChannels( wxCommandEvent& event );
    void Multiply(wxCommandEvent& event);
    void SubtractBaseMenu( wxCommandEvent& event ) { SubtractBase( ); }
    void LFit(wxCommandEvent& event);
    void LnTransform(wxCommandEvent& event);
    void Filter(wxCommandEvent& event);
    void Spectrum(wxCommandEvent& event);
    void P_over_N(wxCommandEvent& event);
    void Plotcriterion(wxCommandEvent& event);
    void Plotcorrelation(wxCommandEvent& event);
    void MarkEvents(wxCommandEvent& event);
    void Threshold(wxCommandEvent& event);
    void Viewtable(wxCommandEvent& event);
    void Fileinfo(wxCommandEvent& event);
    Recording ReorderChannels();

    wxMenu* doc_file_menu;


    stf::latency_mode latencyStartMode, latencyEndMode;
    stf::latency_window_mode latencyWindowMode;
    stf::direction	direction; //of peak detection: UP, DOWN or BOTH
#ifdef WITH_PSLOPE
    stf::pslope_mode_beg pslopeBegMode; // for left mode PSlope cursor
    stf::pslope_mode_end pslopeEndMode; // for right mode PSlope cursor
#endif 
    std::size_t baseBeg, baseEnd, peakBeg, peakEnd, fitBeg, fitEnd, 
#ifdef WITH_PSLOPE
    PSlopeBeg, PSlopeEnd,
#endif
    measCursor;
    double latencyStartCursor,
        latencyEndCursor,
        latency,	 //time from latency cursor to beginning of event
        base, APBase, baseSD, threshold, slopeForThreshold, peak, APPeak, t20Real, t80Real, t50LeftReal, t50RightReal,
        maxT, thrT, maxRiseY, maxRiseT, maxDecayY, maxDecayT, maxRise, maxDecay,
        t50Y, APMaxT, APMaxRise, APMaxRiseT, APt50LeftReal, 
//#ifdef WITH_PSLOPE
        PSlope,
//#endif
        rt2080, halfDuration, slopeRatio, t0Real;
    // cursor windows:
    int pM;  //peakMean, number of points used for averaging
#ifdef WITH_PSLOPE
    int DeltaT;  // distance (number of points) from the first cursor
#endif

    // currently accessed channel:
    std::size_t cc;
    // second channel:
    std::size_t sc;
    // currently accessed section:
    std::size_t cs;

    // Indices of the selected sections
    std::vector<std::size_t> selectedSections;
    // Base line value for each selected trace
    Vector_double selectBase;
    
    std::size_t t20Index, t80Index, t50LeftIndex, t50RightIndex;

    bool fromBase, viewCrosshair,viewBaseline,viewBaseSD,viewThreshold, viewPeakzero,viewPeakbase,viewPeakthreshold, viewRT2080,
        viewT50,viewRD,viewSloperise,viewSlopedecay,viewLatency,
#ifdef WITH_PSLOPE
        viewPSlope,
#endif
        viewCursors;

    XZoom xzoom;
    std::vector<YZoom> yzoom;

    std::vector< std::vector<stf::SectionAttributes> > sec_attr;
    
public:

    //! Constructor.
    /*! Does nothing but initialising the member list.
     */
    wxStfDoc();
    //! Destructor.
    ~wxStfDoc();

    //! Override default file opening.
    /*! Attempts to identify the file type from the filter extension (such as "*.dat")
     *  \param filename Full path of the file.
     *  \return true if successfully opened, false otherwise.
     */
    virtual bool OnOpenDocument(const wxString& filename);

    //! Open document without progress dialog.
    /*! Attempts to identify the file type from the filter extension (such as "*.dat")
     *  \param filename Full path of the file.
     *  \return true if successfully opened, false otherwise.
     */
    virtual bool OnOpenPyDocument(const wxString& filename);

    //! Override default file saving.
    /*! \return true if successfully saved, false otherwise.
     */
    virtual bool SaveAs();

    //! Override default file saving.
    /*! \param filename Full path of the file.
     *  \return true if successfully saved, false otherwise.
     */
    virtual bool DoSaveDocument(const wxString& filename);

    //! Override default file closing.
    /*! Writes settings to the config file or registry before closing.
     *  \return true if successfully closed, false otherwise.
     */
    virtual bool OnCloseDocument();

    //! Override default file creation.
    /*! \return true if successfully closed, false otherwise.
     */
    virtual bool OnNewDocument();

    //! Sets the content of a newly created file.
    /*! \param c_Data The data that is used for the new file.
     *  \param Sender Pointer to the document that generated this file.
     *  \param title Title of the new document.
     */
    void SetData( const Recording& c_Data, const wxStfDoc* Sender, const wxString& title );

    //! Indicates whether an average has been created.
    /*! \return true if an average has been created, false otherwise.
     */
    bool GetIsAverage() const { return !Average.get().empty(); }

    //! Indicates whether the right peak cursor should always be at the end of a trace.
    /*! \return true if the right peak cursor should be at the end, false otherwise.
     */
    bool GetPeakAtEnd() const { return peakAtEnd; }

    //! Indicates whether the the document is fully initialised.
    /*! The document has to be fully initialized before other parts of the
     *  program start accessing it; for example, the graph might start reading out values
     *  before they exist.
     *  \return true if the document is fully initialised, false otherwise.
     */
    bool IsInitialized() const { return initialized; }

    //! Sets the right peak cursor to the end of a trace.
    /*! \param value determines whether the peak cursor should be at the end of a trace.
     */
    void SetPeakAtEnd(bool value) { peakAtEnd=value; }

    //! Retrieves the average trace(s).
    /*! \return The average trace as a Recording object.
     */
    const Recording& GetAverage() const { return Average; }

    //! Checks whether any cursor is reversed or out of range and corrects it if required.
    void CheckBoundaries();

    //! Sets the current section to the specified value
    /*! Checks for out-of-range errors
     *  \param section The 0-based index of the new section
     */
    bool SetSection(std::size_t section);

    //! Creates a new window containing the selected sections of this file.
    /*! \return true upon success, false otherwise.
     */
    bool OnNewfromselectedThis( );

    //! Selects all sections
    /*! \param event The menu event that made the call.
     */
    void Selectall(wxCommandEvent& event);

    //! Unselects all sections
    /*! \param event The menu event that made the call.
     */
    void Deleteselected(wxCommandEvent& event);

    //! Updates the status of the selection button
    void UpdateSelectedButton();

    //! Creates an average trace from the selected sections
    /*! \param calcSD Set to true if the standard deviation should be calculated as well, false otherwise
     *  \param align Set to true if traces should be aligned to the point of steepest rise of the reference channel,
     *         false otherwise.
     */
    void CreateAverage( bool calcSD, bool align );

#if 0
    //! Applies a user-defined function to the current data set
    /*! \param id The id of the user-defined function
     */
    void Userdef(std::size_t id);
#endif

    //! Toggles the selection status of the current section
    void ToggleSelect( );

    //! Selects the current section if previously unselected
    void Select();

    //! Unselects the current section if previously selected
    void Remove();

    //! Creates a new document from the checked events
    /*! \param event The menu event that made the call.
     */
    void Extract(wxCommandEvent& event);

    //! Erases all events, independent of whether they are checked or not
    /*! \param event The menu event that made the call.
     */
    void InteractiveEraseEvents(wxCommandEvent& event);
    
    //! Adds an event at the current eventPos
    /*! \param event The menu event that made the call.
     */
    void AddEvent( wxCommandEvent& event );

    //! Subtracts the baseline of all selected traces.
    /*! \return true upon success, false otherwise.
     */
    bool SubtractBase( );

    //! Fit a function to the data.
    /*! \param event The menu event that made the call.
     */
    void FitDecay(wxCommandEvent& event);

    //! Sets a pointer to the file menu attached to this document.
    /*! \param menu The menu to be attached.
     */
    void SetFileMenu( wxMenu* menu ) { doc_file_menu = menu; }
    
    //! Measure everything using functions defined in measlib.h
    /*! This will measure the baseline, peak values, 20 to 80% rise time, 
     *  half duration, maximal slopes during rise and decay, the ratio of these slopes 
     *  and the latency.
     */
    void Measure();

    
    //! Put the current measurement results into a text table.
    stf::Table CurResultsTable();


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

#ifdef WITH_PSLOPE
    //! Retrieves the position of the left PSlope cursor.
    /*! \return The index of the left PSlope cursor within the current section.
     */
    std::size_t GetPSlopeBeg() const { return PSlopeBeg; }

    //! Retrieves the position of the right PSlope cursor.
    /*! \return The index of the right PSlope cursor within the current section.
     */
    std::size_t GetPSlopeEnd() const { return PSlopeEnd; }
#endif // WITH_PSLOPE

    //! Retrieves the number of points used for averaging during peak calculation.
    /*! \return The number of points to be used.
     */
    int GetPM() const { return pM; }

#ifdef WITH_PSLOPE
    //! Retrieves the number of points used for distance from the first cursor.
    /*! \return The number of points to be used.
     */
    int GetDeltaT() const { return DeltaT; }
#endif

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

    //! Retrieves the time point at which 50% of the max. amplitude have been reached from the left of the peak in the reference channel.
    /*! \return The time point at which 50% of the maximal amplitude have been reached from the left of the peak 
     *  in the reference channel, expressed in units of data points.
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
    
    //! Retrieves the y-value at the measurement cursor (crosshair). Will update measCursor if out of range.
    /*! \return The y-value at the measurement cursor.
     */
    double GetMeasValue();
    
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

//#ifdef WITH_PSLOPE
    //! Retrieves the value of the Slope
    /*! \return slope value in y-units/x-units.
    */
    double GetPSlope() const { return PSlope; }
//#endif

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
    

#ifdef WITH_PSLOPE
    //! Retrieves the mode of the left PSlope cursor.
    /*! \return The current mode of the left PSlope cursor.
     */
    stf::pslope_mode_beg GetPSlopeBegMode() const { return pslopeBegMode; }

    //! Retrieves the mode of the right PSlope cursor.
    /*! \return The current mode of the right PSlope cursor.
     */
    stf::pslope_mode_end GetPSlopeEndMode() const { return pslopeEndMode; }
#endif // WITH_PSLOPE

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

#ifdef WITH_PSLOPE
    //! Indicates whether the Slope should be shown in the results table.
    /*! \return true if it should be shown, false otherwise.
     */
    bool GetViewPSlope() const { return viewPSlope; }

#endif
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
    const Section& cur() const { return get()[cc][cs]; }

    //! Retrieves the currently accessed section in the active channel (read and write)
    /*! \return The currently accessed section in the active channel.
     */
    Section& cur() { return get()[cc][cs]; }

    //! Retrieves the currently accessed section in the second (reference) channel (read-only)
    /*! \return The currently accessed section in the second (reference) channel.
     */
    const Section& sec() const { return get()[sc][cs]; }
    
    //! Returns the current zoom settings for this channel (read-only).
    /*! \return The current zoom settings.
     */
    const XZoom& GetXZoom() { return xzoom; }

    //! Returns the current zoom settings for this channel (read & write).
    /*! \return The current zoom settings.
     */
    XZoom& GetXZoomW() { return xzoom; }
        
    //! Returns the current zoom settings for this channel (read-only).
    /*! \return The current zoom settings.
     */
    const YZoom& GetYZoom(int ch) { return yzoom.at(ch); }

    //! Returns the current zoom settings for this channel (read & write).
    /*! \return The current zoom settings.
     */
    YZoom& GetYZoomW(int ch) { return yzoom.at(ch); }

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

#ifdef WITH_PSLOPE
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

    //! Sets the number of points used for the distance from the first cursor.
    /*! \param value The number of points to be used.
     */
    void SetDeltaT(int value) { DeltaT=value; }

#endif // WITH_PSLOPE

    //! Sets the number of points used for averaging during peak calculation.
    /*! \param value The number of points to be used.
     */
    void SetPM(int value) { pM=value; }

    //! Sets the mode of the latency start cursor.
    /*! \param value The new mode of the latency start cursor..
     */
    void SetLatencyStartMode(stf::latency_mode value) { latencyStartMode=value; }

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

#ifdef WITH_PSLOPE
    //! Determines whether the slope should be shown in the results table.
    /*! \param value Set to true if it should be shown, false otherwise.
     */
    void SetViewPSlope(bool value) { viewPSlope=value; }
#endif

    //! Determines whether two additional rows showing the positions of start and end cursors should be shown in the results table.
    /*! \param value Set to true if they should be shown, false otherwise.
     */
    void SetViewCursors(bool value) { viewCursors=value; }

    //! Sets the slope where the baseline should be set.
    /*! \param value The slope value where the baseline shoudl be set.
     */
    void SetSlopeForThreshold(double value) { slopeForThreshold=value; }
    
    //! Put the current trace into a text table.
    stf::Table CurAsTable() const;
    
    //! Copies the cursor positions from another Recording to this Recording.
    /*! This will copy the crosshair, base, peak and fit cursors positions as 
     *  well as the number of points for peak averaging from another Recording 
     *  and correct the new values if they are out of range. The latency cursors 
     *  will not be copied.
     *  \param c_Recording The Recording from which to copy the cursor positions.
     */
    void CopyCursors(const wxStfDoc& c_Recording);

    //! Resize the Recording to a new number of channels.
    /*! Resizes both the channel and the global y units arrays.
     *  \param c_n_channels The new number of channels.
     */
    virtual void resize(std::size_t c_n_channels);

    //! Insert a Channel at a given position.
    /*! Will throw std::out_of_range if range check fails.
     *  \param c_Channel The Channel to be inserted.
     *  \param pos The position at which to insert the channel (0-based).
     */
    virtual void InsertChannel(Channel& c_Channel, std::size_t pos);

    stf::SectionAttributes GetSectionAttributes(std::size_t nchannel, std::size_t nsection);
    stf::SectionAttributes GetCurrentSectionAttributes();

    //! Deletes the current fit, sets isFitted to false;
    void DeleteFit(std::size_t nchannel, std::size_t nsection);
    
    //! Sets the best-fit parameters when a fit has been performed on this section.
    /*! \param bestFitP_ The best-fit parameters
        \param fitFunc_ The function used for fitting
        \param chisqr The sum of squared errors
        \param fitBeg Sampling point index where the fit starts
        \param fitEnd Sampling point index where the fit ends
     */
    void SetIsFitted( std::size_t nchannel, std::size_t nsection,
                      const Vector_double& bestFitP_, stf::storedFunc* fitFunc_,
                      double chisqr, std::size_t fitBeg, std::size_t fitEnd );


    //! Determines whether an integral has been calculated in this section.
    /*! \return true if an integral has been calculated, false otherwise.
     */
    void SetIsIntegrated(std::size_t nchannel, std::size_t nsection, bool value,
                         std::size_t begin, std::size_t end, const Vector_double& quad_p_);
    
    //! Erases all events.
    void ClearEvents(std::size_t nchannel, std::size_t nsection);

    void correctRangeR(int& value);
    void correctRangeR(std::size_t& value);
    
    DECLARE_EVENT_TABLE()
};

/*@}*/

#endif

