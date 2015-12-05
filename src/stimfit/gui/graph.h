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

/*! \file graph.h
 *  \author Christoph Schmidt-Hieber
 *  \date 2008-01-16
 *  \brief Declares wxStfGraph.
 */

#ifndef _GRAPH_H
#define _GRAPH_H

/*! \addtogroup wxstf
 *  @{
 */

// forward declarations:
class wxStfView;
class wxStfDoc;
class wxStfParentFrame;
class wxStfCheckBox;
class wxEnhMetaFile;

#include "./zoom.h"

enum plottype {
    active,
    reference,
    background
};
    
//! Handles drawing of traces and keyboard or mouse input.
/*! Derived from wxScrolledWindow, although no scrolling is implemented
 *  at this time. All the trace scaling and drawing happens here. Mouse
 *  and keyboard input is handled here as well.
 */
class StfDll wxStfGraph : public wxScrolledWindow
{
public:
    //! Constructor.
    /*! \param v is a pointer to the attached wxView.
     *  \param frame is a pointer to the attached child frame.
     *  \param pos and \param size indicate the initial position and size of this frame.
     *  \param style is the window style.
     */
    wxStfGraph(wxView *v, wxStfChildFrame *frame, const wxPoint& pos, const wxSize& size, long style);

    //! The central drawing function. Used for drawing to any output device, such as a printer or a screen.
    /*! \param dc is the device context used for drawing (can be a printer, a screen or a file).
     */ 
    virtual void OnDraw(wxDC& dc);
    
    //! Copies the drawing to the clipboard as a windows metafile.
    /*! Metafiles are only implemented in Windows. Some applications
     *  allow you to paste as an enhanced metafile (usually through
     *  "Edit -> Paste special..."); choose this option for best results.
     */
    void Snapshotwmf();

    //! Handles mouse events.
    /*! The different possibilities (e.g. left or right click) split up
     *  within this function.
     *  \param event The mouse event. Contains information such as whether 
     *  the left or right button was clicked.
     */
    void OnMouseEvent(wxMouseEvent& event);

    //! Handles keyboard input.
    /*! Key modifiers (e.g. Shift or Ctrl) ar handled within this function.
     *  \param event The keyboard event. Contains information about the key
     *  that was pressed.
     */
    void OnKeyDown(wxKeyEvent& event);
    
    //! Show and analyse next trace.
    /*! Called when either the "next trace"-button is clicked or the right
     *  arrow cursor key is pressed. Wraps around when last trace is reached.
     */
    void OnNext();

    //! Show and analyse previous trace.
    /*! Called when either the "previous trace"-button is clicked or the left
     *  arrow cursor key is pressed. Wraps around when first trace is reached.
     */
    void OnPrevious();

    //! Show and analyse last trace.
    /*! Called when the "last trace"-button is clicked.
     */
    void OnLast();

    //! Show and analyse first trace.
    /*! Called when the "first trace"-button is clicked.
     */
    void OnFirst();
    
    //! Moves the traces up by 20 px.
    /*! Called when either the up arrow cursor key is pressed
     *  or the "Move traces up"-button is clicked.
     */
    void OnUp();

    //! Moves the traces down by 20 px.
    /*! Called when either the down arrow cursor key is pressed
     *  or the "Move traces down"-button is clicked.
     */
    void OnDown();

    //! Moves the traces right by 20 px.
    /*! Called when either the right arrow cursor key and Ctrl are pressed
     *  at the same time or the "Move traces right"-button is clicked.
     */
    void OnRight();

    //! Moves the traces left by 20 px.
    /*! Called when either the left arrow cursor key and Ctrl are pressed
     *  at the same time or the "Move traces left"-button is clicked.
     */
    void OnLeft();

    //! Enlarges the x-scale by a factor of 2.
    /*! This is currently never called and might be removed in the future.
     */
    void OnXenlhi();

    //! Enlarges the x-scale by a factor of 1.1.
    /*! Called when either the "+" key and Ctrl are pressed
     *  at the same time or the "Enlarge x-scale"-button is clicked.
     */
    void OnXenllo();

    //! Shrinks the x-scale by a factor of 1.1.
    /*! Called when either the "-" key and Ctrl are pressed
     *  at the same time or the "Shrink x-scale"-button is clicked.
     */
    void OnXshrinklo();

    //! Shrinks the x-scale by a factor of 2.
    /*! This is currently never called and might be removed in the future.
     */
    void OnXshrinkhi();

    //! Enlarges the y-scale by a factor of 2.
    /*! This is currently never called and might be removed in the future.
     */
    void OnYenlhi();

    //! Enlarges the y-scale by a factor of 1.1.
    /*! Called when either the "+" key is pressed
     *  or the "Enlarge x-scale"-button is clicked.
     */
    void OnYenllo();

    //! Shrinks the y-scale by a factor of 1.1.
    /*! Called when either the "-" key is pressed
     *  or the "Shrink x-scale"-button is clicked.
     */
    void OnYshrinklo();

    //! Shrinks the y-scale by a factor of 2.
    /*! This is currently never called and might be removed in the future.
     */
    void OnYshrinkhi();

    //! Adjust y-positioning so that the baselines of channel 1 and 2 are at the same y-position.
    void Ch2base();

    //! Adjust y-positioning so that channel 1 and 2 are at the same absolute y-position.
    void Ch2pos();

    //! Adjust y-scale so that channel 1 and 2 have the same y-scale.
    void Ch2zoom();

    //! Combines Ch2zoom() and Ch2base().
    /*! This is a separate function so that the graph is not
     *  refreshed between adjusting the y-scale and the baseline.
     */
    void Ch2basezoom();

#if 0
    //! Swaps the active and the reference channel.
    void SwapChannels();
#endif
    
    //! Fits the graph to the window.
    /*! Fits the graph to 100% of the width and 50% of the height
     *  of the window and centers it.
     *  \param refresh Set to true if the graph should be refreshed after fitting it to the window.
     */
    void Fittowindow(bool refresh);

    //! Set to true if the graph is drawn on a printer.
    /*! \param value boolean determining whether the graph is printed.
     */
    void set_isPrinted(bool value);

    //! Sets the printing scale to the specified value.
    /*! \param value The new printing scale.
     */
    void set_printScale(double value) {printScale=value;}

    //! Sets the size of the printout to the epcified rectangle.
    /*! \param value The new printing rectangle.
     */
    void set_printRect(wxRect value) {printRect=value;}

    //! Set to true if the results table and the cursors should be printed.
    /*! \param value boolean determining whether everything should be printed.
     */
    void set_noGimmicks(bool value) {no_gimmicks=value;}

    //! Prints every n-th point.
    /*! \param value Determines that every n-th point should be printed.
     */
    void set_downsampling(int value) { downsampling = (value < 1 ? 1 : value); }

    //! Indicates whether everything (cursors, results table, etc.) is printed out.
    /*! \return true if everything is printed out.
     */
    bool get_noGimmicks() const {return no_gimmicks;}

    //! Returns the y-position of a right click when in event-detection mode.
    /*! \return the index of the trace that the right-click position corresponds to.
     */
    int get_eventPos() const { return eventPos; }

    //! Returns the current zoom struct.
    /*! \return the current zoom struct.
     */
//    Zoom get_zoom() { return Doc()->at(Doc()->GetCurChIndex()).GetZoom(); }

    //! Sets the current zoom struct.
    /*! \param zoom_ The current zoom struct.
     */
//    void set_zoom(const Zoom& zoom_) { Doc()->at(Doc()->GetCurChIndex()).GetZoomW()=zoom_; }

    //! The view attached to this wxStfGraph.
    wxStfView *view;

    //! Returns x value of the left screen border
    /*! \return x value of the left screen border
     */
    double get_plot_xmin() const;

    //! Returns x value of the right screen border
    /*! \return x value of the right screen border
     */
    double get_plot_xmax() const;

    //! Returns y value of the bottom screen border
    /*! \return y value of the bottom screen border
     */
    double get_plot_ymin() const;

    //! Returns y value of the top screen border
    /*! \return y value of the top screen border
     */
    double get_plot_ymax() const;
    
    //! Returns y value of the bottom screen border for the reference channel
    /*! \return y value of the bottom screen border for the reference channel
     */
    double get_plot_y2min() const;

    //! Returns y value of the top screen border for the reference channel
    /*! \return y value of the top screen border for the reference channel
     */
    double get_plot_y2max() const;

 private:
    wxStfChildFrame* pFrame;
    bool isZoomRect; //True if zoom window is set
    bool no_gimmicks; //True if no extra rulers/lines and circles shall be displayed
    bool isPrinted; //True when the View is drawn to a printer 
    bool isLatex;
    bool firstPass;
    bool isSyncx;

    //Zoom struct
//    Zoom zoom;

    //Zoom struct to retain StdOut
//    Zoom zoomOld;

    //Zoom struct for PrintOut
//    Zoom zoomPrint;

    //Variables for the scaling of the print out
    wxRect printRect;

    //Printout graphic variables
    static const int
        boebbelStd=6;//Size of circles for display output
    int
        boebbel, //Size of circles (for peak, 2080rise time, etc.)
        boebbelPrint; //Size of circles for scalable print out
    double printScale;
    int  printSizePen1,//Size of pens for scalable print out 
        printSizePen2,
        printSizePen4,
        downsampling,
        eventPos;

    // ll... means lower limit, ul... means upper limit
    double llz_x, ulz_x, llz_y, ulz_y, llz_y2,ulz_y2;

    //Three lines of text containing the results
    wxString results1, results2, results3,results4, results5, results6; 

    //Pens are declared here instead of locally to accelerate OnDraw()
    //Drawing (pen) styles for the different graphical standard output
    wxPen standardPen, standardPen2, standardPen3, scalePen, scalePen2, peakPen, peakLimitPen,
        basePen, baseLimitPen, decayLimitPen, ZoomRectPen, fitPen, fitSelectedPen,
        selectPen, averagePen, rtPen, hdPen, rdPen, slopePen, latencyPen,
        alignPen, measPen, eventPen, PSlopePen;						/*CSH*/

    //Drawing (pen) styles for the different graphical standard output
    wxPen standardPrintPen, standardPrintPen2, standardPrintPen3, scalePrintPen, scalePrintPen2,measPrintPen,
        peakPrintPen, peakLimitPrintPen, basePrintPen, baseLimitPrintPen,
        decayLimitPrintPen, fitPrintPen, fitSelectedPrintPen, selectPrintPen,
        averagePrintPen, rtPrintPen, hdPrintPen, rdPrintPen,
        slopePrintPen, resultsPrintPen, latencyPrintPen, PSlopePrintPen;

    wxBrush baseBrush, zeroBrush;
    
    wxPoint lastLDown;

    YZoom yzoombg;
    
#if (__cplusplus < 201103)
    boost::shared_ptr<wxMenu> m_zoomContext;
    boost::shared_ptr<wxMenu> m_eventContext;
#else
    std::shared_ptr<wxMenu> m_zoomContext;
    std::shared_ptr<wxMenu> m_eventContext;
#endif

    void InitPlot();
    void PlotSelected(wxDC& DC);
    void PlotAverage(wxDC& DC);
    void DrawZoomRect(wxDC& DC);
    void PlotGimmicks(wxDC& DC);
    void PlotEvents(wxDC& DC);
    void DrawCrosshair( wxDC& DC, const wxPen& pen, const wxPen& printPen, int crosshairSize, double xch, double ych);
    void PlotTrace( wxDC* pDC, const Vector_double& trace, plottype pt=active, int bgno=0 );
    void DoPlot( wxDC* pDC, const Vector_double& trace, int start, int end, int step, plottype pt=active, int bgno=0 );
    void PrintScale(wxRect& WindowRect);
    void PrintTrace( wxDC* pDC, const Vector_double& trace, plottype ptype=active);
    void DoPrint( wxDC* pDC, const Vector_double& trace, int start, int end, plottype ptype=active);
    void DrawCircle(wxDC* pDC, double x, double y, const wxPen& pen, const wxPen& printPen);
    void DrawVLine(wxDC* pDC, double x, const wxPen& pen, const wxPen& printPen);
    void DrawHLine(wxDC* pDC, double y, const wxPen& pen, const wxPen& printPen);
    void eventArrow(wxDC* pDC, int eventIndex);
    void DrawFit(wxDC* pDC);
    void PlotFit( wxDC* pDC, const stf::SectionPointer& Sec );
    void DrawIntegral(wxDC* pDC);
    void CreateScale(wxDC* pDC);

    // Function receives the x-coordinate of a point and returns 
    // its formatted value according to the current Zoom settings
    long xFormat(double);
    long xFormat(long); 
    long xFormat(int); 
    long xFormat(std::size_t); 
    // The same for the y coordinates
    long yFormat(double);
    long yFormat(long);
    long yFormat(int);
    long yFormatD(double f) { return yFormat(f); }
    // The same for the y coordinates of the second channel
    long yFormat2(double);
    long yFormat2(long);
    long yFormat2(int);
    long yFormatD2(double f) { return yFormat2(f); }
    // The same for the y coordinates of the background channel
    long yFormatB(double);
    long yFormatB(long);
    long yFormatB(int);
    long yFormatDB(double f) { return yFormatB(f); }

    void FittorectY(YZoom& yzoom, const wxRect& rect, double min, double max, double screen_part);
    void FitToWindowSecCh(bool refresh);

    void LButtonDown(wxMouseEvent& event);
    void RButtonDown(wxMouseEvent& event);
    void LButtonUp(wxMouseEvent& event);

    // shorthand:
    wxStfDoc* Doc() {
        if (view != NULL) 
            return view->Doc();
        else
            return NULL;
    }
    wxStfDoc* DocC() const {
        if (view != NULL) 
            return view->DocC();
        else
            return NULL;
    }
    void ChangeXScale(double factor);
    void ChangeYScale(double factor);
    wxStfParentFrame* ParentFrame();
    void OnZoomHV(wxCommandEvent& event);
    void OnZoomH(wxCommandEvent& event);
    void OnZoomV(wxCommandEvent& event);
#if defined __WXMAC__ && !(wxCHECK_VERSION(2, 9, 0))
    void OnPaint(wxPaintEvent &event);
#endif
    long SPX() const { return DocC()->GetXZoom().startPosX; }
    long& SPXW() { return DocC()->GetXZoomW().startPosX; } 
    long SPY() const { return DocC()->GetYZoom(DocC()->GetCurChIndex()).startPosY; }
    long& SPYW() { return DocC()->GetYZoomW(DocC()->GetCurChIndex()).startPosY; } 
    long SPY2() const { return DocC()->GetYZoom(DocC()->GetSecChIndex()).startPosY; }
    long& SPY2W() { return DocC()->GetYZoomW(DocC()->GetSecChIndex()).startPosY; }
    
    double XZ() const { return DocC()->GetXZoom().xZoom; }
    double& XZW() { return DocC()->GetXZoomW().xZoom; }
    double YZ() const { return DocC()->GetYZoom(DocC()->GetCurChIndex()).yZoom; }
    double& YZW() { return DocC()->GetYZoomW(DocC()->GetCurChIndex()).yZoom; }
    double YZ2() const { return DocC()->GetYZoom(DocC()->GetSecChIndex()).yZoom; }
    double& YZ2W() { return DocC()->GetYZoomW(DocC()->GetSecChIndex()).yZoom; }
    
    DECLARE_EVENT_TABLE()
};

/*@}*/

#endif
