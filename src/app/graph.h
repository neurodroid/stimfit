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

//! Handles drawing of traces and keyboard or mouse input.
/*! Derived from wxScrolledWindow, although no scrolling is implemented
 *  at this time. All the trace scaling and drawing happens here. Mouse
 *  and keyboard input is handled here as well.
 */
class wxStfGraph : public wxScrolledWindow
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

    //! Exports the drawing to a bitmap file (jpg, tiff or png at this time).
    void Exportimage();

    //! Exports the drawing to a postscript file
    /*! This function is only implemented in Windows. In Linux/Unix,
     *  the cairo library is used for postscript output from the printing
     *  dialog. It produces much better output than the generic postscript
     *  library.
     */ 
    void Exportps();

    //! Exports the drawing to a LaTeX file using PSTricks macros.
    /*! Uses wxLatexDC to create LaTeX-compatible postscipt macros
     *  with PSTricks.
     */
    void Exportlatex();

#if wxCHECK_VERSION(2, 9, 0)
    //! Exports the drawing to a SVG file
    /*! Uses wxSVGFileDC to create scalable vector graphics.
     */
    void Exportsvg();
#endif
    
    //! Copies the drawing to the clipboard as a bitmap.
    void Snapshot();

#ifdef _WINDOWS
    //! Copies the drawing to the clipboard as a windows metafile.
    /*! Metafiles are only implemented in Windows. Some applications
     *  allow you to paste as an enhanced metafile (usually through
     *  "Edit -> Paste special..."); choose this option for best results.
     */
    void Snapshotwmf();
#endif

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
    //! Swaps the active and the inactive channel.
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
//    Zoom get_zoom() { return Doc()->at(Doc()->GetCurCh()).GetZoom(); }

    //! Sets the current zoom struct.
    /*! \param zoom_ The current zoom struct.
     */
//    void set_zoom(const Zoom& zoom_) { Doc()->at(Doc()->GetCurCh()).GetZoomW()=zoom_; }

    //! The view attached to this wxStfGraph.
    wxStfView *view;

private:
    wxStfChildFrame* pFrame;
    bool isZoomRect; //True if zoom window is set
    bool no_gimmicks; //True if no extra rulers/lines and circles shall be displayed
    bool isPrinted; //True when the View is drawn to a printer 
    bool isLatex;
    bool firstPass;
    bool isSyncx;
    int resLimit;

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
    wxPen standardPen, standardPen2, scalePen, scalePen2, peakPen, peakLimitPen,
        basePen, baseLimitPen, decayLimitPen, ZoomRectPen, fitPen, fitSelectedPen,
        selectPen, averagePen, rtPen, hdPen, rdPen, slopePen, latencyPen,
        alignPen,measPen,eventPen;																/*CSH*/

    //Drawing (pen) styles for the different graphical standard output
    wxPen standardPrintPen, standardPrintPen2, scalePrintPen, scalePrintPen2,measPrintPen,
        peakPrintPen, peakLimitPrintPen, basePrintPen, baseLimitPrintPen,
        decayLimitPrintPen, fitPrintPen, fitSelectedPrintPen, selectPrintPen,
        averagePrintPen, rtPrintPen, hdPrintPen, rdPrintPen,
        slopePrintPen, resultsPrintPen, latencyPrintPen, alignPrintPen;

    wxBrush baseBrush, zeroBrush;
    
    wxPoint lastLDown;
    
    boost::shared_ptr<wxMenu> m_zoomContext;
    boost::shared_ptr<wxMenu> m_eventContext;
    std::vector<wxStfCheckBox*> cbList;
    void PlotTrace( wxDC* pDC, const std::valarray<double>& trace, bool is2=false );
    void DoPlot( wxDC* pDC, const std::valarray<double> trace, int start, int end, int step, bool is2 );
    void PrintTrace( wxDC* pDC, const std::valarray<double>& trace, bool is2=false );
    void DoPrint( wxDC* pDC, const std::valarray<double> trace, int start, int end, int downsampling, bool is2 );
    void DrawCircle(wxDC* pDC, double x, double y);
    void DrawVLine(wxDC* pDC, double x);
    void DrawHLine(wxDC* pDC, double y);
    void eventArrow(wxDC* pDC, int eventIndex);
    void DrawFit(wxDC* pDC);
    void PlotFit( wxDC* pDC, const Section& Sec );
    void DrawIntegral(wxDC* pDC);
    void CreateScale(wxDC* pDC);

    // Function receives the x-coordinate of a point and returns 
    // its formatted value according to the current Zoom settings
    int xFormat(double);
    int xFormat(int); 
    int xFormat(std::size_t); 
    // The same for the y coordinates
    int yFormat(double);
    int yFormat(int);
    int yFormatD(double f) { return yFormat(f); }
    // The same for the y coordinates of the second channel
    int yFormat2(double);
    int yFormat2(int);
    int yFormatD2(double f) { return yFormat2(f); }

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
    void ChangeXScale(double factor);
    void ChangeYScale(double factor);
    wxStfParentFrame* ParentFrame();
    void OnZoomHV(wxCommandEvent& event);
    void OnZoomH(wxCommandEvent& event);
    void OnZoomV(wxCommandEvent& event);
#if defined __WXMAC__ && !(wxCHECK_VERSION(2, 9, 0))
    void OnPaint(wxPaintEvent &event);
#endif
    int SPX() { return Doc()->GetXZoom().startPosX; }
    int& SPXW() { return Doc()->GetXZoomW().startPosX; } 
    int SPY() { return Doc()->at(Doc()->GetCurCh()).GetYZoom().startPosY; }
    int& SPYW() { return Doc()->at(Doc()->GetCurCh()).GetYZoomW().startPosY; } 
    int SPY2() { return Doc()->at(Doc()->GetSecCh()).GetYZoom().startPosY; }
    int& SPY2W() { return Doc()->at(Doc()->GetSecCh()).GetYZoomW().startPosY; }
    
    double XZ() { return Doc()->GetXZoom().xZoom; }
    double& XZW() { return Doc()->GetXZoomW().xZoom; }
    double YZ() { return Doc()->at(Doc()->GetCurCh()).GetYZoom().yZoom; }
    double& YZW() { return Doc()->at(Doc()->GetCurCh()).GetYZoomW().yZoom; }
    double YZ2() { return Doc()->at(Doc()->GetSecCh()).GetYZoom().yZoom; }
    double& YZ2W() { return Doc()->at(Doc()->GetSecCh()).GetYZoomW().yZoom; }
    
    DECLARE_EVENT_TABLE()
};

/*@}*/

#endif
