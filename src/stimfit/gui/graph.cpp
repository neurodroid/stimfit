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

// graph.cpp
// This is where the actual drawing happens.
// 2007-12-27, Christoph Schmidt-Hieber, University of Freiburg

#include <wx/wxprec.h>

#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#include <wx/image.h>
#include <wx/clipbrd.h>
#include <wx/metafile.h>
#include <wx/printdlg.h>
#include <wx/paper.h>

#include "./app.h"
#include "./doc.h"
#include "./view.h"
#include "./parentframe.h"
#include "./childframe.h"
#include "./printout.h"
#include "./dlgs/cursorsdlg.h"
#include "./dlgs/smalldlgs.h"
#include "./usrdlg/usrdlg.h"
#include "./graph.h"
#include "./../../libstfnum/measure.h"

#ifdef _STFDEBUG
#include <iostream>
#endif

#if defined (_WIN32)
#define isnan _isnan
#elif !defined(isnan)
#define isnan std::isnan
#endif
// #define BENCHMARK // uncomment to run benchmark

#ifdef BENCHMARK //def _STFDEBUG
#include <wx/utils.h>
#include <iostream>
#include <fstream>
#include <ctime>
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

/* From https://gist.github.com/jbenet/1087739 */
void current_utc_time(struct timespec *ts) {
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    ts->tv_sec = mts.tv_sec;
    ts->tv_nsec = mts.tv_nsec;
#else
    clock_gettime(CLOCK_REALTIME, ts);
#endif

}
static const double BILLION = 1000000000L;
double tdiff(timespec time1, timespec time0) {
    return ( time1.tv_sec - time0.tv_sec )
        + ( time1.tv_nsec - time0.tv_nsec ) / BILLION;
}

double t2d(timespec time1) {
    return time1.tv_sec + time1.tv_nsec / BILLION;
}
#endif

BEGIN_EVENT_TABLE(wxStfGraph, wxWindow)
EVT_MENU(ID_ZOOMHV,wxStfGraph::OnZoomHV)
EVT_MENU(ID_ZOOMH,wxStfGraph::OnZoomH)
EVT_MENU(ID_ZOOMV,wxStfGraph::OnZoomV)
EVT_MOUSE_EVENTS(wxStfGraph::OnMouseEvent)
EVT_KEY_DOWN( wxStfGraph::OnKeyDown )
#if defined __WXMAC__ && !(wxCHECK_VERSION(2, 9, 0))
EVT_PAINT( wxStfGraph::OnPaint )
#endif
END_EVENT_TABLE()

// Define a constructor for my canvas
wxStfGraph::wxStfGraph(wxView *v, wxStfChildFrame *frame, const wxPoint& pos, const wxSize& size, long style):
    wxScrolledWindow(frame, wxID_ANY, pos, size, style),pFrame(frame),
    isZoomRect(false),no_gimmicks(false),isPrinted(false),isLatex(false),firstPass(true),isSyncx(false),
    printRect(),boebbel(boebbelStd),boebbelPrint(boebbelStd),
#ifdef __WXGTK__
    printScale(1.0),printSizePen1(4),printSizePen2(8),printSizePen4(16),
#else
    printScale(1.0),printSizePen1(4),printSizePen2(8),printSizePen4(16),
#endif
    downsampling(1),eventPos(0),
    llz_x(0.0),ulz_x(1.0),llz_y(0.0),ulz_y(1.0),llz_y2(0.0),ulz_y2(1.0),
    results1(wxT("\0")),results2(wxT("\0")),results3(wxT("\0")),results4(wxT("\0")),results5(wxT("\0")),results6(wxT("\0")),
    standardPen(*wxBLACK,1,wxSOLID), //Solid black line
    standardPen2(*wxRED,1,wxSOLID), //Solid red line
    standardPen3(wxColour(255,192,192),1,wxSOLID), //Solid red line
    scalePen(*wxBLACK,2,wxSOLID), //Solid black line
    scalePen2(*wxRED,2,wxSOLID), //Solid red line
    peakPen(*wxRED,1,wxSHORT_DASH), //Dashed red line
    peakLimitPen(*wxRED,1,wxDOT), //Dotted red line
    basePen(*wxGREEN,1,wxSHORT_DASH), //Dashed green line
    baseLimitPen(*wxGREEN,1,wxDOT), //Dotted green line
    decayLimitPen(wxColour(127,127,127),1,wxDOT), //Dotted dark blue line
    ZoomRectPen(*wxLIGHT_GREY,1,wxDOT), //Dotted grey line
    fitPen(wxColour(127,127,127),4,wxSOLID), //Solid dark grey line
    fitSelectedPen(wxColour(192,192,192),2,wxSOLID), //Solid dark grey line
    selectPen(wxColour(127,127,127),1,wxSOLID), //Solid grey line
    averagePen(*wxBLUE,1,wxSOLID), //Solid light blue line
    rtPen(*wxGREEN,2,wxSOLID), //Solid green line
    hdPen(*wxCYAN,2,wxSOLID), //Solid violet line
    rdPen(*wxRED,2,wxSOLID), //Solid dark violet line
#ifdef WITH_PSLOPE
    slopePen(*wxBLUE,2,wxSOLID), //Solid blue line
#endif
    latencyPen(*wxBLUE,1,wxDOT),
    alignPen(*wxBLUE,1,wxSHORT_DASH),
    measPen(*wxBLACK,1,wxDOT),
    eventPen(*wxBLUE,2,wxSOLID),
#ifdef WITH_PSLOPE
    PSlopePen(wxColor(30,144,255), 1, wxDOT), // Dotted bright blue line
#endif
    standardPrintPen(*wxBLACK,printSizePen1,wxSOLID), //Solid black line
    standardPrintPen2(*wxRED,printSizePen1,wxSOLID), //Solid red line
    standardPrintPen3(wxColour(255,192,192),printSizePen1,wxSOLID), //Solid red line
    scalePrintPen(*wxBLACK,printSizePen2,wxSOLID), //Solid black line
    scalePrintPen2(*wxRED,printSizePen2,wxSOLID), //Solid red line
    measPrintPen(*wxBLACK,printSizePen1,wxDOT),
    peakPrintPen(*wxRED,printSizePen1,wxSHORT_DASH), //Dashed red line
    peakLimitPrintPen(*wxRED,printSizePen1,wxDOT), //Dotted red line
    basePrintPen(*wxGREEN,printSizePen1,wxSHORT_DASH), //Dashed green line
    baseLimitPrintPen(*wxGREEN,printSizePen1,wxDOT), //Dotted green line
    decayLimitPrintPen(wxColour(63,63,63),printSizePen1,wxDOT), //Dotted dark blue line
    fitPrintPen(wxColour(63,63,63),printSizePen2,wxSOLID), //Solid dark grey line
    fitSelectedPrintPen(wxColour(128,128,128),printSizePen2,wxSOLID), //Solid dark grey line
    selectPrintPen(wxColour(31,31,31),printSizePen1,wxSOLID), //Solid grey line
    averagePrintPen(*wxBLUE,printSizePen1,wxSOLID), //Solid light blue line
    rtPrintPen(*wxGREEN,printSizePen2,wxSOLID), //Solid green line
    hdPrintPen(*wxCYAN,printSizePen2,wxSOLID), //Solid violet line
    rdPrintPen(*wxRED,printSizePen2,wxSOLID), //Solid dark violet line
#ifdef WITH_PSLOPE
    slopePrintPen(*wxBLUE,printSizePen4,wxSOLID), //Solid blue line
#endif
    resultsPrintPen(*wxLIGHT_GREY,printSizePen2,wxSOLID),//Solid light grey line
    latencyPrintPen(*wxBLUE,printSizePen1,wxDOT),//Dotted violett line
    PSlopePrintPen(wxColour(30,144,255), printSizePen1, wxDOT), // Dotted bright blue line
    baseBrush(*wxLIGHT_GREY,wxBDIAGONAL_HATCH),
    zeroBrush(*wxLIGHT_GREY,wxFDIAGONAL_HATCH),
    lastLDown(0,0),
    yzoombg(),
    m_zoomContext( new wxMenu ),
    m_eventContext( new wxMenu )
{
    m_zoomContext->Append( ID_ZOOMHV, wxT("Expand zoom window horizontally && vertically") );
    m_zoomContext->Append( ID_ZOOMH, wxT("Expand zoom window horizontally") );
    m_zoomContext->Append( ID_ZOOMV, wxT("Expand zoom window vertically") );

    m_eventContext->Append( ID_EVENT_ADDEVENT, wxT("Add an event that starts here") );
    m_eventContext->Append( ID_EVENT_ERASE, wxT("Erase all events") );
    m_eventContext->Append( ID_EVENT_EXTRACT, wxT("Extract selected events") );

    SetBackgroundColour(*wxWHITE);
    view = (wxStfView*)v;
    wxString perspective=wxGetApp().wxGetProfileString(wxT("Settings"),wxT("Windows"),wxT(""));
/*    if (perspective != wxT("")) {
        // load the stored perspective:
        frame->GetMgr()->LoadPerspective(perspective);
    } else {
        // or the default:
        frame->GetMgr()->LoadPerspective(defaultPersp);
    }
*/
}

wxStfParentFrame* wxStfGraph::ParentFrame() {
    return (wxStfParentFrame*)wxGetApp().GetTopWindow();
}

// Defines the repainting behaviour
void wxStfGraph::OnDraw( wxDC& DC )
{

    if ( !view || Doc()->get().empty() || !Doc()->IsInitialized() )
        return;

    // ugly hack to force active document update:
#if defined(__WXGTK__) || defined(__WXMAC__)
    view->Activate(true);
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
    if (!HasFocus())
#else
    if (wxWindow::FindFocus()!=(wxWindow*)this)
#endif
        SetFocus();
#endif
    wxRect WindowRect(GetRect());

    if (isPrinted) {
        PrintScale(WindowRect);
    }

    if (firstPass) {
        firstPass = false;
        InitPlot();
    }
    
    //Creates scale bars and labelings for display or print out
    //Calculate scale bars and labelings
    CreateScale(&DC);

    //Create additional rulers/lines and circles on display
    if (!no_gimmicks) 	{
        PlotGimmicks(DC);
    }

    //Plot all selected traces and fitted functions if at least one trace ist selected
    //and 'is selected' is selected in the trace navigator/control box
    //Polyline() is used for printing to avoid separation of traces
    //in postscript files
    //LineTo()is used for display for performance reasons

    //Plot fit curves (including current trace)
    DrawFit(&DC);

    if (!Doc()->GetSelectedSections().empty() && pFrame->ShowSelected()) {
        PlotSelected(DC);
    }	//End plot all selected traces

    //Plot average
    if (Doc()->GetIsAverage()) {
        PlotAverage(DC);
    }	//End plot average


    // Plot integral boundaries
    try {
        if (Doc()->GetCurrentSectionAttributes().isIntegrated) {
            DrawIntegral(&DC);
        }
    }
    catch (const std::out_of_range& e) {
        /* Do nothing for now */
    }

    //Zoom window is displayed (see OnLeftButtonUp())
    if (isZoomRect) {
        DrawZoomRect(DC);
    }
    //End zoom

    //Plot of the second channel
    //Trace one when displayed first time
    if ((Doc()->size()>1) && pFrame->ShowSecond()) {
        if (!isPrinted) {
            //Draw current trace on display
            //For display use point to point drawing
            DC.SetPen(standardPen2);
            PlotTrace(&DC,Doc()->get()[Doc()->GetSecChIndex()][Doc()->GetCurSecIndex()].get(), reference);
        } else {	//Draw second channel for print out
            //For print out use polyline tool
            DC.SetPen(standardPrintPen2);
            PrintTrace(&DC,Doc()->get()[Doc()->GetSecChIndex()][Doc()->GetCurSecIndex()].get(), reference);
        }	// End display or print out
    }		//End plot of the second channel

    if ((Doc()->size()>1) && pFrame->ShowAll()) {
        for (std::size_t n=0; n < Doc()->size(); ++n) {
            if (!isPrinted) {
                //Draw current trace on display
                //For display use point to point drawing
                DC.SetPen(standardPen3);
                PlotTrace(&DC,Doc()->get()[n][Doc()->GetCurSecIndex()].get(), background, n);
            }
        }
    }		//End plot of the second channel
    
    //Standard plot of the current trace
    //Trace one when displayed first time
    if (!isPrinted) {
	//Draw current trace on display
        //For display use point to point drawing
        DC.SetPen(standardPen);
        PlotTrace(&DC,Doc()->get()[Doc()->GetCurChIndex()][Doc()->GetCurSecIndex()].get());
    } else {
        //For print out use polyline tool
        DC.SetPen(standardPrintPen);
        PrintTrace(&DC,Doc()->get()[Doc()->GetCurChIndex()][Doc()->GetCurSecIndex()].get());
    }	// End display or print out
    //End plot of the current trace

    //Ensure old scaling after print out
    if(isPrinted) {
        for (std::size_t n=0; n < Doc()->size(); ++n) {
            Doc()->GetYZoomW(n) = Doc()->GetYZoomW(n) * (1.0/printScale);
        }
        Doc()->GetXZoomW() = Doc()->GetXZoomW() * (1.0/printScale);
        WindowRect=printRect;
    }	//End ensure old scaling after print out

    view->OnDraw(& DC);
}

void wxStfGraph::InitPlot() {

    if (wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewScaleBars"),1)) {
        if (pFrame->GetMenuBar() && pFrame->GetMenuBar()->GetMenu(2)) {
            pFrame->GetMenuBar()->GetMenu(2)->Check(ID_SCALE,true);
        }
        wxGetApp().set_isBars(true);
    } else {
        if (pFrame->GetMenuBar() && pFrame->GetMenuBar()->GetMenu(2)) {
            pFrame->GetMenuBar()->GetMenu(2)->Check(ID_SCALE,false);
        }
        wxGetApp().set_isBars(false);
    }

    if (wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewSyncx"),1)) {
        isSyncx=true;
    } else {
        isSyncx=false;
    }

    // Ensure proper dimensioning
    // Determine scaling factors and Units
    // Zoom and offset variables are currently not part of the settings dialog =>
    // Read from registry
    // Return a negative value upon first program start so that the trace is
    // fit to the window dimensions
    YZW()=(double)(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("zoom.yZoom"), -1)
                   / 100000.0);
    SPYW()=wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("zoom.startPosY"), 0);
    XZW()=(double)(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("zoom.xZoom"), -1)
                   / 100000.0);
    SPXW()=wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("zoom.startPosX"), 0);


    if (XZ() <= 0 || YZ() <= 0 || fabs(double(SPY())) >= 1e15)
        Fittowindow(false);
    if ((Doc()->size()>1))
    {	//Second channel is not part of the settings dialog =>read from registry
        SPY2W() =
            wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("Zoom.startPosY2"), 1);
        YZ2W() =
            (double)(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("Zoom.yZoom2"), 1)
                     / 100000.0);
        //Ensure proper dimensioning
        if (YZ2() <=0)
            FitToWindowSecCh(false);
    }
}

void wxStfGraph::PlotSelected(wxDC& DC) {
    if (!isPrinted)
    {	//Draw traces on display
        DC.SetPen(selectPen);
        for (unsigned m=0; m < Doc()->GetSelectedSections().size(); ++m)
        {
            //For display use point to point drawing
            PlotTrace(
                      &DC,
                      Doc()->get()[Doc()->GetCurChIndex()][Doc()->GetSelectedSections()[m]].get()
                      );
        }
    }  //End draw traces on display
    else
    {  //Draw traces for print out
        DC.SetPen(selectPrintPen);
        for (unsigned m=0; m < Doc()->GetSelectedSections().size() && Doc()->GetSelectedSections().size()>0; ++m)
        {
            PrintTrace(&DC,Doc()->get()[Doc()->GetCurChIndex()][Doc()->GetSelectedSections()[m]].get());
        }	//End draw for print out
    }	//End if display or print out
}

void wxStfGraph::PlotAverage(wxDC& DC) {
    //Average is calculated but not plotted
    if (!isPrinted)
    {	//Draw Average on display
        //For display use point to point drawing
        DC.SetPen(averagePen);
        PlotTrace(&DC,Doc()->GetAverage()[0][0].get());
    }	//End draw Average on display
    else
    {	//Draw average for print out
        //For print out use polyline tool
        DC.SetPen(averagePrintPen);
        PrintTrace(&DC,Doc()->GetAverage()[0][0].get());
    }	//End draw average for print out
}

void wxStfGraph::DrawZoomRect(wxDC& DC) {
    DC.SetPen(ZoomRectPen);
    wxPoint ZoomPoints[4];
    wxPoint Ul_Corner((int)llz_x, (int)llz_y);
    wxPoint Ur_Corner((int)ulz_x, (int)llz_y);
    wxPoint Lr_Corner((int)ulz_x, (int)ulz_y);
    wxPoint Ll_Corner((int)llz_x, (int)ulz_y);
    ZoomPoints[0]=Ul_Corner;
    ZoomPoints[1]=Ur_Corner;
    ZoomPoints[2]=Lr_Corner;
    ZoomPoints[3]=Ll_Corner;
    DC.DrawPolygon(4,ZoomPoints);
}

void wxStfGraph::PlotGimmicks(wxDC& DC) {

    // crosshair through measurement cursor:
    int crosshairSize=20;
    DrawCrosshair(DC, measPen, measPrintPen, crosshairSize, Doc()->GetMeasCursor(), Doc()->GetMeasValue());

    // crosshair through threshold:
    DrawCrosshair(DC, peakPen, peakPrintPen, crosshairSize/2.0, Doc()->GetThrT(), Doc()->GetThreshold());

    // creates vertical ruler through measurement peak if needed
    if ( Doc()->GetMeasRuler() )
        DrawVLine(&DC, Doc()->GetMeasCursor(), measPen, measPrintPen);
    //creates red vertical and horizontal dashed lines through the peak
    DrawVLine(&DC,Doc()->GetMaxT(), peakPen, peakPrintPen);
    DrawHLine(&DC,Doc()->GetPeak(), peakPen, peakPrintPen);
    //and red dotted lines through peak calculation limits
    DrawVLine(&DC,Doc()->GetPeakBeg(), peakLimitPen, peakLimitPrintPen);
    DrawVLine(&DC,Doc()->GetPeakEnd(), peakLimitPen, peakLimitPrintPen);

    //creates a green horizontal dashed line through the base
    DrawHLine(&DC,Doc()->GetBase(), basePen, basePrintPen);
    //and green dotted lines through Doc()->GetBase() calculation limits
    DrawVLine(&DC,Doc()->GetBaseBeg(), baseLimitPen, baseLimitPrintPen);
    DrawVLine(&DC,Doc()->GetBaseEnd(), baseLimitPen, baseLimitPrintPen);

    //Create darkblue dotted lines through decay calculation limits
    DrawVLine(&DC,Doc()->GetFitBeg(), decayLimitPen, decayLimitPrintPen);
    DrawVLine(&DC,Doc()->GetFitEnd(), decayLimitPen, decayLimitPrintPen);

    // Create dotted line as a latency cursor
    DrawVLine(&DC,Doc()->GetLatencyBeg(), latencyPen, latencyPrintPen);
    DrawVLine(&DC,Doc()->GetLatencyEnd(), latencyPen, latencyPrintPen);

    // Create double-arrow between latency cursors:
    int latStart=xFormat(Doc()->GetLatencyBeg());
    int latEnd=xFormat(Doc()->GetLatencyEnd());

    if (latStart < 0) latStart = 0;
    if (latEnd > GetRect().width) latEnd = GetRect().width;

    DC.DrawLine(latStart,20,latEnd,20);
    // left arrowhead:
    DC.DrawLine(latStart+1,20,latStart+6,15);
    DC.DrawLine(latStart+1,20,latStart+6,25);
    // right arrowhead:
    DC.DrawLine(latEnd-1,20,latEnd-6,15);
    DC.DrawLine(latEnd-1,20,latEnd-6,25);

#ifdef WITH_PSLOPE
    // Create dotted bright blue line as slope cursor
    DrawVLine(&DC, Doc()->GetPSlopeBeg(), PSlopePen, PSlopePrintPen);
    DrawVLine(&DC, Doc()->GetPSlopeEnd(), PSlopePen, PSlopePrintPen);
#endif 

    //Set circle size depending on output
    if (!isPrinted)
        boebbel=boebbelStd;
    else
        boebbel=boebbelPrint;

    //draws green circles around the Lo% and the Hi% rise times
    double reference = Doc()->GetBase();
    if ( !Doc()->GetFromBase() && Doc()->GetThrT() >= 0 ) {
        reference = Doc()->GetThreshold();
    }
    double Low = Doc()->GetRTFactor()/100.;
    double High = 1-Low;
    DrawCircle( &DC,Doc()->GetTLoReal(), High*reference + Low*Doc()->GetPeak(), rtPen, rtPrintPen);
    DrawCircle( &DC,Doc()->GetTHiReal(), Low*reference + High*Doc()->GetPeak(), rtPen, rtPrintPen);

    //draws circles around the half duration limits
    DrawCircle(&DC,Doc()->GetT50LeftReal(),Doc()->GetT50Y(), hdPen, hdPrintPen);
    DrawCircle(&DC,Doc()->GetT50RightReal(),Doc()->GetT50Y(), hdPen, hdPrintPen);

    //draws dark violet circles around the points of steepest rise/decay
    DrawCircle(&DC,Doc()->GetMaxRiseT(),Doc()->GetMaxRiseY(), rdPen, rdPrintPen);
    DrawCircle(&DC,Doc()->GetMaxDecayT(),Doc()->GetMaxDecayY(), rdPen, rdPrintPen);
    
    try {
        stf::SectionAttributes sec_attr = Doc()->GetCurrentSectionAttributes();
        if (!sec_attr.eventList.empty()) {
            PlotEvents(DC);
        }
        if (!sec_attr.pyMarkers.empty()) {
            DC.SetPen(eventPen);
            for (c_marker_it it = sec_attr.pyMarkers.begin(); it != sec_attr.pyMarkers.end(); ++it) {
                // Create circles indicating the peak of an event:
                DC.DrawRectangle( xFormat(it->x), yFormat(it->y), boebbel*2.0, boebbel*2.0 );
            }
        }
    }
    catch (const std::out_of_range& e) {
        /* Do nothing for now */
    }


}

void wxStfGraph::PlotEvents(wxDC& DC) {
    const int MAX_EVENTS_PLOT = 200;

    stf::SectionAttributes sec_attr;
    try {
        sec_attr = Doc()->GetCurrentSectionAttributes();
    }
    catch (const std::out_of_range& e) {
        return;
    }
    DC.SetPen(eventPen);
    for (c_event_it it = sec_attr.eventList.begin(); it != sec_attr.eventList.end(); ++it) {
        // Create small arrows indicating the start of an event:
        eventArrow(&DC, (int)it->GetEventStartIndex());
        // Create circles indicating the peak of an event:
        try {
            DrawCircle( &DC, it->GetEventPeakIndex(), Doc()->cursec().at(it->GetEventPeakIndex()), eventPen, eventPen );
        }
        catch (const std::out_of_range& e) {
            wxGetApp().ExceptMsg( wxString( e.what(), wxConvLocal ) );
            return;
        }
    }

    // Only draw check boxes if there are less than 1000 events (takes too long
    // to draw them and it's impossible to check them anyway)
    wxRect WindowRect=GetRect();
    if (isPrinted) WindowRect=wxRect(printRect);
    int right=WindowRect.width;
    int nevents_plot = 0;
    for (event_it it2 = sec_attr.eventList.begin(); it2 != sec_attr.eventList.end(); ++it2) {
        nevents_plot += (xFormat(it2->GetEventStartIndex()) < right &&
                         xFormat(it2->GetEventStartIndex()) > 0);
    }

    if (nevents_plot < MAX_EVENTS_PLOT) {
        for (event_it it2 = sec_attr.eventList.begin(); it2 != sec_attr.eventList.end(); ++it2) {
            if (xFormat(it2->GetEventStartIndex()) < right &&
                xFormat(it2->GetEventStartIndex()) > 0)
            {
                it2->GetCheckBox()->Move(wxPoint(xFormat(it2->GetEventStartIndex()), 0));
                it2->GetCheckBox()->Show(true);
            } else {
                it2->GetCheckBox()->Show(false);
            }
        }
    } else {
        for (event_it it2 = sec_attr.eventList.begin(); it2 != sec_attr.eventList.end(); ++it2) {
            it2->GetCheckBox()->Show(false);
        }
    }

    // return focus to frame:
    SetFocus();
}

void wxStfGraph::ClearEvents() {
    stf::SectionAttributes sec_attr;
    try {
        sec_attr = Doc()->GetCurrentSectionAttributes();
    }
    catch (const std::out_of_range& e) {
        return;
    }
    for (event_it it2 = sec_attr.eventList.begin(); it2 != sec_attr.eventList.end(); ++it2) {
        it2->GetCheckBox()->Destroy();
    }
}

void wxStfGraph::DrawCrosshair( wxDC& DC, const wxPen& pen, const wxPen& printPen, int crosshairSize, double xch, double ych) {
    if (isnan(xch) || isnan(ych)) {
        return;
    }

    wxPen chpen = pen;
    if (isPrinted) {
        chpen = printPen;
        crosshairSize=(int)(crosshairSize*printScale);
    }
    DC.SetPen(chpen);
    try {
        // circle:
        wxRect frame(wxPoint( xFormat(xch)-crosshairSize,
                              yFormat(ych)-crosshairSize ),
                     wxPoint( xFormat(xch)+crosshairSize,
                              yFormat(ych)+crosshairSize ));
        DC.DrawEllipse(frame);
        // vertical part:
        DC.DrawLine( xFormat(xch),
                     yFormat(ych)-crosshairSize,
                     xFormat(xch),
                     yFormat(ych)+crosshairSize );
        if (wxGetApp().GetCursorsDialog()!=NULL &&
            wxGetApp().GetCursorsDialog()->IsShown())
        {
            //if (wxGetApp().GetCursorsDialog()->GetRuler())
            if ( Doc()->GetMeasRuler() )
            {
                DrawVLine(&DC,xch, pen, printPen);
            }
        }

        // horizontal part:
        DC.DrawLine( xFormat(xch)-crosshairSize, yFormat(ych),
                     xFormat(xch)+crosshairSize, yFormat(ych) );
    }
    catch (const std::out_of_range& e) {
        wxGetApp().ExceptMsg( wxString( e.what(), wxConvLocal ) );
        return;
    }

}

double wxStfGraph::get_plot_xmin() const {
    return -SPX()/XZ() * DocC()->GetXScale();
}

double wxStfGraph::get_plot_xmax() const {
    wxRect WindowRect=GetRect();
    int right=WindowRect.width;
    return (right-SPX())/XZ() * DocC()->GetXScale();
}

double wxStfGraph::get_plot_ymin() const {
    wxRect WindowRect=GetRect();
    int top=WindowRect.height;
    return (SPY()-top)/YZ();
}

double wxStfGraph::get_plot_ymax() const {
    return SPY()/YZ();
}

double wxStfGraph::get_plot_y2min() const {
    wxRect WindowRect=GetRect();
    int top=WindowRect.height;
    return (SPY2()-top)/YZ2();
}

double wxStfGraph::get_plot_y2max() const {
    return SPY2()/YZ2();
}

void wxStfGraph::PlotTrace( wxDC* pDC, const Vector_double& trace, plottype pt, int bgno ) {
    // speed up drawing by omitting points that are outside the window:

    // find point before left window border:
    // xFormat=toFormat * zoom.xZoom + zoom.startPosX
    // for xFormat==0:
    // toFormat=-zoom.startPosX/zoom.xZoom
    std::size_t start=0;
    int x0i=int(-SPX()/XZ());
    if (x0i>=0 && x0i<(int)trace.size()-1) start=x0i;
    // find point after right window border:
    // for xFormat==right:
    // toFormat=(right-zoom.startPosX)/zoom.xZoom
    std::size_t end=trace.size();
    wxRect WindowRect=GetRect();
    if (isPrinted) WindowRect=wxRect(printRect);
    int right=WindowRect.width;
    int xri = int((right-SPX())/XZ())+1;
    if (xri>=0 && xri<(int)trace.size()-1) end=xri;

    // apply filter at half the new sampling frequency:
    DoPlot(pDC, trace, start, end, 1, pt, bgno);
}

void wxStfGraph::DoPlot( wxDC* pDC, const Vector_double& trace, int start, int end, int step, plottype pt, int bgno) {
#if (__cplusplus < 201103)
    boost::function<int(double)> yFormatFunc;
#else
    std::function<int(double)> yFormatFunc;
#endif

    // NOTE: bind1st is removed in c++17.
    switch (pt) {
     case active:
         // yFormatFunc = std::bind1st( std::mem_fun(&wxStfGraph::yFormatD), this);
         yFormatFunc = std::bind(std::mem_fn(&wxStfGraph::yFormatD), this, std::placeholders::_1);
         break;
     case reference:
         // yFormatFunc = std::bind1st( std::mem_fun(&wxStfGraph::yFormatD2), this);
         yFormatFunc = std::bind(std::mem_fn(&wxStfGraph::yFormatD2), this, std::placeholders::_1);
         break;
     case background:
         Vector_double::const_iterator max_el = std::max_element(trace.begin(), trace.end());
         Vector_double::const_iterator min_el = std::min_element(trace.begin(), trace.end());
         double min = *min_el;
         if (min>1.0e12)  min= 1.0e12;
         if (min<-1.0e12) min=-1.0e12;
         double max = *max_el;
         if (max>1.0e12)  max= 1.0e12;
         if (max<-1.0e12) max=-1.0e12;
         wxRect WindowRect=GetRect();
         WindowRect.height /= Doc()->size();
         FittorectY(yzoombg, WindowRect, min, max, 1.0);
         yzoombg.startPosY += bgno*WindowRect.height;
         // yFormatFunc = std::bind1st( std::mem_fun(&wxStfGraph::yFormatDB), this);
         yFormatFunc = std::bind(std::mem_fn(&wxStfGraph::yFormatDB), this, std::placeholders::_1);
         break;
    }

    int x_last = xFormat(start);
    int y_last = yFormatFunc( trace[start] );
    int x_next = 0;
    int y_next = 0;
#ifdef BENCHMARK //def _STFDEBUG
    struct timespec time0, time1;
    current_utc_time(&time0);
#else
    wxRect WindowRect(GetRect());
    if (end-start < 2*WindowRect.width+2) {
#endif    
    for (int n=start; n<end-1; ++n) {
        x_next = xFormat(n+1);
        y_next = yFormatFunc( trace[n+1] );
        pDC->DrawLine( x_last, y_last, x_next, y_next );
        x_last = x_next;
        y_last = y_next;
    }
#ifdef BENCHMARK //def _STFDEBUG
    current_utc_time(&time1);
    double accum = tdiff(time1, time0)*1e3;
    std::string fn_platform = "plt_bench_" + stf::wx2std(wxGetOsDescription()) + ".txt";
    std::ofstream plt_bench;
    plt_bench.open(fn_platform.c_str(), std::ios::out | std::ios::app);
    plt_bench << end-start << "\t" << accum << "\t";
    current_utc_time(&time0);
    x_last = xFormat(start);
#else
    } else {
#endif
    double y_max = trace[start];
    double y_min = trace[start];
    for (int n=start; n<end-1; ++n) {
        x_next = xFormat(n+1);
        // if we are still in the same pixel column, find extrema:
        if (x_next == x_last) {
            if (trace[n+1] < y_min) {
                y_min = trace[n+1];
            }
            if (trace[n+1] > y_max) {
                y_max = trace[n+1];
            }
        } else {
            // plot line between extrema of previous column:
            pDC->DrawLine( x_last, yFormatFunc(y_min), x_last, yFormatFunc(y_max) );

            // plot line between last point of previous and first point of this column:
            pDC->DrawLine( x_last, yFormatFunc(trace[n]), x_next, yFormatFunc(trace[n+1]) );

            y_min = trace[n+1];
            y_max = trace[n+1];
            x_last = x_next;
        }
    }
#ifdef BENCHMARK //def _STFDEBUG
    current_utc_time(&time1);
    accum = tdiff(time1, time0)*1e3;
    plt_bench << accum << std::endl;
    plt_bench.close();
#else
    }
#endif
}

void wxStfGraph::PrintScale(wxRect& WindowRect) {
    //enhance resolution for printing - see OnPrint()
    //Ensures the scaling of all pixel dependent drawings
     
    for (std::size_t n=0; n < Doc()->size(); ++n) {
        Doc()->GetYZoomW(n) = Doc()->GetYZoomW(n) * printScale;
    }
    Doc()->GetXZoomW() = Doc()->GetXZoomW() * printScale;
    WindowRect=printRect;
    //Calculate scaling variables
    boebbelPrint=(int)(boebbelStd * printScale);
    if ( boebbelPrint < 1 ) boebbelPrint=2;
    printSizePen1=(int)(1 * printScale);
    if ( printSizePen1 < 1 ) boebbelPrint=1;
    printSizePen2=(int)(2 * printScale);
    if ( printSizePen2 < 1 ) boebbelPrint=2;
    printSizePen4=(int)(4 * printScale);
    if ( printSizePen4 < 1 ) boebbelPrint=4;
}

void wxStfGraph::PrintTrace( wxDC* pDC, const Vector_double& trace, plottype ptype ) {
    // speed up drawing by omitting points that are outside the window:

    // find point before left window border:
    // xFormat=toFormat * zoom.xZoom + zoom.startPosX
    // for xFormat==0:
    // toFormat=-zoom.startPosX/zoom.xZoom
    std::size_t start=0;
    int x0i=int(-SPX()/XZ());
    if (x0i>=0 && x0i<(int)trace.size()-1) start=x0i;
    // find point after right window border:
    // for xFormat==right:
    // toFormat=(right-zoom.startPosX)/zoom.xZoom
    std::size_t end=trace.size();
    wxRect WindowRect=GetRect();
    if (isPrinted)
        WindowRect=wxRect(printRect);
    int right=WindowRect.width;
    int xri=int((right-SPX())/XZ())+1;
    if (xri>=0 && xri<(int)trace.size()-1) end=xri;
    DoPrint(pDC, trace, start, end, ptype);
}

void wxStfGraph::DoPrint( wxDC* pDC, const Vector_double& trace, int start, int end, plottype ptype) {

#if (__cplusplus < 201103)
    boost::function<int(double)> yFormatFunc;
#else
    std::function<long(double)> yFormatFunc;
#endif
    
    switch (ptype) {
     case active:
         // yFormatFunc = std::bind1st( std::mem_fun(&wxStfGraph::yFormatD), this);
         yFormatFunc = std::bind( std::mem_fn(&wxStfGraph::yFormatD), this, std::placeholders::_1);
         break;
     default:
         // yFormatFunc = std::bind1st( std::mem_fun(&wxStfGraph::yFormatD2), this);
         yFormatFunc = std::bind(std::mem_fn(&wxStfGraph::yFormatD2), this, std::placeholders::_1);
         break;
    }

    std::vector<wxPoint> points;
    int x_last = xFormat(start);
    int y_last = yFormatFunc( trace[start] );
    int y_max = y_last;
    int y_min = y_last;
    int x_next = 0;
    int y_next = 0;
    points.push_back( wxPoint(x_last,y_last) );
    for (int n=start; n<end-downsampling; n+=downsampling) {
        x_next = xFormat(n+downsampling);
        y_next = yFormatFunc( trace[n+downsampling] );
        // if we are still at the same pixel column, only draw if this is an extremum:
        if (x_next == x_last) {
            if (y_next < y_min) {
                y_min = y_next;
            }
            if (y_next > y_max) {
                y_max = y_next;
            }
        } else {
            // else, always draw and reset extrema:
            if (y_min != y_next) {
                points.push_back( wxPoint(x_last, y_min) );
            }
            if (y_max != y_next) {
                points.push_back( wxPoint(x_last, y_max) );
            }
            points.push_back( wxPoint(x_next, y_next) );
            y_min = y_next;
            y_max = y_next;
            x_last = x_next;
        }
    }
    pDC->DrawLines((int)points.size(),&points[0]);
}

void wxStfGraph::DrawCircle(wxDC* pDC, double x, double y, const wxPen& pen, const wxPen& printPen) {
    if (isPrinted) {
        pDC->SetPen(printPen);
    } else {
        pDC->SetPen(pen);
    }
    wxRect Frame(
            wxPoint(xFormat(x)-boebbel,yFormat(y)-boebbel),
            wxPoint(xFormat(x)+boebbel,yFormat(y)+boebbel)
    );
    pDC->DrawEllipse(Frame);
}

void wxStfGraph::DrawVLine(wxDC* pDC, double x, const wxPen& pen, const wxPen& printPen) {
    wxRect WindowRect(GetRect());
    if (isPrinted)
    {   //Set WindowRect to print coordinates (page size)
        WindowRect=printRect;
        pDC->SetPen(printPen);
    } else {
        pDC->SetPen(pen);
    }
    pDC->DrawLine(xFormat(x),0,xFormat(x),WindowRect.height);
}

void wxStfGraph::DrawHLine(wxDC* pDC, double y, const wxPen& pen, const wxPen& printPen) {
    wxRect WindowRect(GetRect());
    if (isPrinted)
    {   //Set WindowRect to print coordinates (page size)
        WindowRect=printRect;
        pDC->SetPen(printPen);
    } else {
        pDC->SetPen(pen);
    }
    pDC->DrawLine(0, yFormat(y),WindowRect.width,yFormat(y));
}

void wxStfGraph::eventArrow(wxDC* pDC, int eventIndex) {
    // we only need that if it's within the screen:
    wxRect WindowRect(GetRect());
    if (xFormat(eventIndex)<0 || xFormat(eventIndex)>WindowRect.width) {
        return;
    }
    if (isPrinted)
    {   //Set WindowRect to print coordinates (page size)
        WindowRect=printRect;
    }

    pDC->DrawLine(xFormat(eventIndex), 20, xFormat(eventIndex), 0);

    // arrow head:
    pDC->DrawLine(xFormat(eventIndex)-5, 15, xFormat(eventIndex), 20);
    pDC->DrawLine(xFormat(eventIndex)+5, 15, xFormat(eventIndex), 20);
}

void wxStfGraph::DrawFit(wxDC* pDC) {

    try {
        // go through selected traces:
        if ( isPrinted )
            pDC->SetPen(fitSelectedPrintPen);
        else
            pDC->SetPen(fitSelectedPen);
        for ( std::size_t n_sel = 0; n_sel < Doc()->GetSelectedSections().size(); ++n_sel ) {
            std::size_t sel_index = Doc()->GetSelectedSections()[ n_sel ];
            // Check whether this section contains a fit:
            try {
                stf::SectionAttributes sec_attr = Doc()->GetSectionAttributes(Doc()->GetCurChIndex(), sel_index);
                if ( sec_attr.isFitted && pFrame->ShowSelected() ) {
                    PlotFit( pDC, stf::SectionPointer( &((*Doc())[Doc()->GetCurChIndex()][sel_index]), sec_attr ) );
                }
            } catch (const std::out_of_range& e) {
                /* Do nothing */
            }
        }

        // Active trace
        if ( isPrinted )
            pDC->SetPen(fitPrintPen);
        else
            pDC->SetPen(fitPen);
        stf::SectionAttributes sec_attr = Doc()->GetCurrentSectionAttributes();
        if (sec_attr.isFitted) {
            PlotFit( pDC, stf::SectionPointer( &((*Doc())[Doc()->GetCurChIndex()][Doc()->GetCurSecIndex()]),
                                               sec_attr) );
        }
    }
    catch (const std::out_of_range& e) {
        
    }
}

void wxStfGraph::PlotFit( wxDC* pDC, const stf::SectionPointer& Sec ) {

    wxRect WindowRect = GetRect();
    if (isPrinted)
    {   //Set WindowRect to print coordinates (page size)
        WindowRect=printRect;
    }

    int firstPixel = xFormat( Sec.sec_attr.storeFitBeg );
    if ( firstPixel < 0 ) firstPixel = 0;
    int lastPixel = xFormat( Sec.sec_attr.storeFitEnd );
    if ( lastPixel > WindowRect.width + 1 ) lastPixel = WindowRect.width + 1;

    if (!isPrinted) {
        //Draw Fit on display
        //For display use point to point drawing
        double fit_time_1 =
            ( ((double)firstPixel - (double)SPX()) / XZ() -
                    (double)Sec.sec_attr.storeFitBeg )* Doc()->GetXScale();
        for ( int n_px = firstPixel; n_px < lastPixel-1; n_px++ ) {
            // Calculate pixel back to time (GetStoreFitBeg() is t=0)
            double fit_time_2 =
                ( ((double)n_px+1.0 - (double)SPX()) / XZ() -
                        (double)Sec.sec_attr.storeFitBeg )
                        * Doc()->GetXScale(); // undo xFormat = (int)(toFormat * XZ() + SPX());
            pDC->DrawLine( n_px,
                    yFormat(Sec.sec_attr.fitFunc->func( fit_time_1, Sec.sec_attr.bestFitP)),
                            n_px + 1, yFormat(Sec.sec_attr.fitFunc->func(fit_time_2, Sec.sec_attr.bestFitP))
            );
            fit_time_1 = fit_time_2;
        }
    } else {    //Draw Fit for print out
        // For print out use polyline
        std::vector<wxPoint> f_print( lastPixel - firstPixel );
        for ( int n_px = firstPixel; n_px < lastPixel; n_px++ ) {
            // Calculate pixel back to time (GetStoreFitBeg() is t=0)
            double fit_time =
                ( ((double)n_px - (double)SPX()) / XZ() -(double)Sec.sec_attr.storeFitBeg )
                        * Doc()->GetXScale(); // undo xFormat = (int)(toFormat * XZ() + SPX());
            f_print[n_px-firstPixel].x = n_px;
            f_print[n_px-firstPixel].y = yFormat( Sec.sec_attr.fitFunc->func(
                            fit_time, Sec.sec_attr.bestFitP) );
        }
        pDC->DrawLines( f_print.size(), &f_print[0] );
    }   //End if display or print out
}

void wxStfGraph::DrawIntegral(wxDC* pDC) {
    // Draws a polygon around the integral. Note that the polygon will be drawn
    // out of screen as well.
    stf::SectionAttributes sec_attr;
    try {
        sec_attr = Doc()->GetCurrentSectionAttributes();
    }
    catch (const std::out_of_range& e) {
        return;
    }
    
    if (!isPrinted) {
        pDC->SetPen(scalePen);
    } else {
        pDC->SetPen(scalePrintPen);
    }
    bool even = std::div((int)sec_attr.storeIntEnd-(int)sec_attr.storeIntBeg, 2).rem==0;
    int firstPixel=xFormat(sec_attr.storeIntBeg);
    // last pixel:
    int lastPixel= even ? xFormat(sec_attr.storeIntEnd) : xFormat(sec_attr.storeIntEnd-1);
    std::size_t qt_size=
        lastPixel-firstPixel + // part that covers the trace
        2; // straight line through base or 0
    if (!even)
        qt_size++; //straight line for trapezoidal part
    std::vector<wxPoint> quadTrace;
    quadTrace.reserve(qt_size);
    quadTrace.push_back(wxPoint(firstPixel,yFormat(Doc()->GetBase())));
    // "Simpson part" (piecewise quadratic functions through three adjacent points):
    for (int n_pixel=firstPixel; n_pixel < lastPixel; ++n_pixel) {
        // (lower) index corresponding to pixel:
        int n_relIndex =
            (int)(((double)n_pixel-(double)SPX())/(double)XZ()-sec_attr.storeIntBeg);
        if (n_relIndex >= 0 && (unsigned int)(n_relIndex/2)*3+2 < sec_attr.quad_p.size()) {
            double n_absIndex = ((double)n_pixel-(double)SPX())/(double)XZ();
            // quadratic parameters at this point:
            double a = sec_attr.quad_p[(int)(n_relIndex/2)*3];
            double b = sec_attr.quad_p[(int)(n_relIndex/2)*3+1];
            double c = sec_attr.quad_p[(int)(n_relIndex/2)*3+2];
            double y = a*n_absIndex*n_absIndex + b*n_absIndex + c;
            quadTrace.push_back(wxPoint(n_pixel,yFormat(y)));
        }
    }

    // add trapezoidal integration part if uneven:
    if (!even) {
    // draw a straight line:
        quadTrace.push_back(
            wxPoint(
                    xFormat(sec_attr.storeIntEnd),
                    yFormat(Doc()->cursec()[sec_attr.storeIntEnd])
                    ));
    }
    quadTrace.push_back(
        wxPoint(
                xFormat(sec_attr.storeIntEnd),
                yFormat(Doc()->GetBase())
        ));

    // Polygon from base:
    pDC->SetBrush(baseBrush);
    pDC->DrawPolygon((int)quadTrace.size(),&quadTrace[0]);
    // Polygon from 0:
    quadTrace[0]=wxPoint(firstPixel,yFormat(0L));
    quadTrace[quadTrace.size()-1]=
        wxPoint(
                xFormat(sec_attr.storeIntEnd),
                yFormat(0L)
        );
    pDC->SetBrush(zeroBrush);
    pDC->DrawPolygon((int)quadTrace.size(),&quadTrace[0]);
    pDC->SetBrush(*wxTRANSPARENT_BRUSH);
}

void wxStfGraph::Snapshotwmf() {
    wxStfPreprintDlg myDlg(this,true);
    if (myDlg.ShowModal()!=wxID_OK) return;
    set_downsampling(myDlg.GetDownSampling());

    // Get size of Graph, in pixels:
    wxRect screenRect(GetRect());

    // Get size of page, in pixels:
    // assuming the screen is ~ 96 dpi, but we want ~ 720:
    printRect = wxRect(wxPoint(0,0), wxSize(GetRect().GetSize()*4));

    double scale=(double)printRect.width/(double)screenRect.width;

#if _WINDOWS
    // FIXME: for non-Windows platforms	
    wxMetafileDC wmfDC;
    if (!wmfDC.IsOk()) 
#endif 
    {
        wxGetApp().ErrorMsg(wxT("Error while creating clipboard data"));
        return;
    }
    set_noGimmicks(true);
    set_isPrinted(true);
    printScale=scale;
#if _WINDOWS
    // FIXME: for non-Windows platforms	
    OnDraw(wmfDC);
#endif
    set_isPrinted(false);
    no_gimmicks=false;
#if _WINDOWS
    // FIXME: for non-Windows platforms	
    wxMetafile* mf = wmfDC.Close();
    if (mf && mf->IsOk()) {
        mf->SetClipboard();
        delete mf;
    } else 
#endif 
    {
        wxGetApp().ErrorMsg(wxT("Error while copying to clipboard"));
    }
}

void wxStfGraph::OnMouseEvent(wxMouseEvent& event) {
    // event.Skip();
    
    if (!view) return;

    if (event.LeftDown()) LButtonDown(event);
    if (event.RightDown()) RButtonDown(event);
    if (event.LeftUp()) LButtonUp(event);

}

void wxStfGraph::LButtonDown(wxMouseEvent& event) {
    // event.Skip();
    if (!view) return;
    view->Activate(true);
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
    if (!HasFocus())
#else
    if (wxWindow::FindFocus()!=(wxWindow*)this)
#endif
        SetFocusIgnoringChildren();

    wxClientDC dc(this);
    PrepareDC(dc);
    lastLDown = event.GetLogicalPosition(dc);
    switch (ParentFrame()->GetMouseQual())
    {	//Depending on the radio buttons (Mouse field)
    //in the (trace navigator) control box
    case stf::measure_cursor:
        //conversion of pixel on screen to time (inversion of xFormat())
        Doc()->SetMeasCursor( stf::round( ((double)lastLDown.x - (double)SPX())/XZ() ) ); //second 'double' added
        // in this case, update results string without waiting for "Return":
        pFrame->UpdateResults();
        break;
    case stf::peak_cursor:
        //conversion of pixel on screen to time (inversion of xFormat())
        Doc()->SetPeakBeg( stf::round( ((double)lastLDown.x - (double)SPX())/XZ() ) ); //second 'double' added
        //Set x-value as lower limit of the peak calculation dialog box
        break;
    case stf::base_cursor:
        //conversion of pixel on screen to time (inversion of xFormat())
        Doc()->SetBaseBeg( stf::round( ((double)lastLDown.x - (double)SPX())/XZ() ) ); //second 'double' added
        break;
    case stf::decay_cursor:
        //conversion of pixel on screen to time (inversion of xFormat())
        if (wxGetApp().GetCursorsDialog() != NULL && wxGetApp().GetCursorsDialog()->GetStartFitAtPeak()) {
            wxGetApp().ErrorMsg(
                    wxT("Fit will start at the peak. Change cursor settings (Edit->Cursor settings) to set manually.")
            );
            break;
        }
        Doc()->SetFitBeg( stf::round( ((double)lastLDown.x - (double)SPX())/XZ() ) ); //second 'double' added
        break;
    case stf::latency_cursor:
        if (Doc()->GetLatencyStartMode() != stf::manualMode) {
            Doc()->SetLatencyStartMode( stf::manualMode );
            wxGetApp().ErrorMsg(
                    wxT("The first latency cursor is set to manual mode")
            );
        }
        Doc()->SetLatencyBeg(((double)lastLDown.x-(double)SPX())/XZ());
        Refresh();
        break;
    case stf::zoom_cursor:
        llz_x=(double)lastLDown.x;
        llz_y=(double)lastLDown.y;
        llz_y2=llz_y;
        break;
#ifdef WITH_PSLOPE
    case stf::pslope_cursor:
        Doc()->SetPSlopeBegMode(stf::psBeg_manualMode); // set left cursor to manual
        // conversion of pixel on screen to time (inversion of xFormat())
        Doc()->SetPSlopeBeg( stf::round( ((double)lastLDown.x - (double)SPX())/XZ() ) ); // second 'double' added
        break;
#endif
    default: break;
    }	//End switch TraceNav->GetMouseQual()
    if (wxGetApp().GetCursorsDialog()!=NULL && wxGetApp().GetCursorsDialog()->IsShown()) {
        try {
            wxGetApp().GetCursorsDialog()->UpdateCursors();
        }
        catch (const std::runtime_error& e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal) );
        }
    }
}

void wxStfGraph::RButtonDown(wxMouseEvent& event) {
    // event.Skip();

    if (!view) return;
    view->Activate(true);
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
    if (!HasFocus())
#else
    if (wxWindow::FindFocus()!=(wxWindow*)this)
#endif
        SetFocusIgnoringChildren();

    wxClientDC dc(this);
    PrepareDC(dc);
    wxPoint point(event.GetLogicalPosition(dc));
    switch (ParentFrame()->GetMouseQual()) {
    case stf::peak_cursor:
        //conversion of pixel on screen to time (inversion of xFormat())
        Doc()->SetPeakEnd( stf::round( ((double)point.x - (double)SPX())/XZ() ) );
        break;
    case stf::base_cursor:
        //conversion of pixel on screen to time (inversion of xFormat())
        Doc()->SetBaseEnd( stf::round( ((double)point.x - (double)SPX())/XZ() ) );
        break;
    case stf::decay_cursor:
        //conversion of pixel on screen to time (inversion of xFormat())
        Doc()->SetFitEnd( stf::round( ((double)point.x - (double)SPX())/XZ() ) );
        break;
    case stf::latency_cursor:
        if (Doc()->GetLatencyEndMode() != stf::manualMode) {
            Doc()->SetLatencyEndMode( stf::manualMode ); 
            wxGetApp().ErrorMsg(
                wxT("The second latency cursor is set to manual mode")
            );
        }
        Doc()->SetLatencyEnd(((double)point.x-(double)SPX())/XZ());
        Refresh();
        break;
    case stf::zoom_cursor:
        if (isZoomRect) {
            PopupMenu(m_zoomContext.get());
        } else {
            wxGetApp().ErrorMsg(wxT("Draw a zoom window with the left mouse button first"));
        }
        break;
    case stf::event_cursor:
        try {
            if (!Doc()->GetCurrentSectionAttributes().eventList.empty()) {
                // store the position that has been clicked:
                eventPos = stf::round( ((double)point.x - (double)SPX())/XZ() );
                PopupMenu(m_eventContext.get());
            } else {
                wxGetApp().ErrorMsg(wxT("No events have been detected yet"));
            }
        } catch (const std::out_of_range& e) {

        }
        break;
#ifdef WITH_PSLOPE
    case stf::pslope_cursor:
        Doc()->SetPSlopeEndMode(stf::psEnd_manualMode); // set right cursor to manual mode
        Doc()->SetPSlopeEnd( stf::round( ((double)point.x - (double)SPX())/XZ() ) );
        break;
#endif
    default: ;
    }	//End switch TraceNav->GetMouseQual()
    if (wxGetApp().GetCursorsDialog()!=NULL && wxGetApp().GetCursorsDialog()->IsShown()) {
        try {
            wxGetApp().GetCursorsDialog()->UpdateCursors();
        }
        catch (const std::runtime_error& e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal) );
        }
    }
    Refresh();
}

void wxStfGraph::LButtonUp(wxMouseEvent& event) {
    // event.Skip();
    wxClientDC dc(this);
    PrepareDC(dc);
    wxPoint point(event.GetLogicalPosition(dc));
    if (point == lastLDown) {
        Refresh();
        return;
    }
    switch (ParentFrame()->GetMouseQual()) {
    case stf::peak_cursor:
        //conversion of pixel on screen to time (inversion of xFormat())
        Doc()->SetPeakEnd( stf::round( ((double)point.x - (double)SPX())/XZ() ) );
        break;
    case stf::base_cursor:
        //conversion of pixel on screen to time (inversion of xFormat())
        Doc()->SetBaseEnd( stf::round( ((double)point.x - (double)SPX())/XZ() ) );
        break;
    case stf::decay_cursor:
        //conversion of pixel on screen to time (inversion of xFormat())
        Doc()->SetFitEnd( stf::round( ((double)point.x - (double)SPX())/XZ() ) );
        break;
#ifdef WITH_PSLOPE
    case stf::pslope_cursor:
        // conversion of pixel on screen to time (inversion of xFormat())
        Doc()->SetPSlopeEnd( stf::round( ((double)point.x - (double)SPX())/XZ() ) );
#endif
    case stf::latency_cursor:
        if (Doc()->GetLatencyEndMode() != stf::manualMode) {
            wxGetApp().ErrorMsg(
                    wxT("The latency cursor can not be set in the current mode\n \
                    Choose manual mode to set the latency cursor manually")
            );
            break;
        }
        Doc()->SetLatencyEnd(((double)point.x-(double)SPX())/XZ());
        break;
    case stf::zoom_cursor:
        ulz_x=(double)point.x;
        ulz_y=(double)point.y;
        ulz_y2=ulz_y;
        if (llz_x>ulz_x) std::swap(llz_x,ulz_x);
        if (llz_y>ulz_y) std::swap(llz_y,ulz_y);
        if (llz_y2>ulz_y2) std::swap(llz_y2,ulz_y2);
        isZoomRect=true;
        break;
     default: break;
         
    }
    Refresh();
}

void wxStfGraph::OnKeyDown(wxKeyEvent& event) {
    // event.Skip();
    if (!view)
        return;
    view->Activate(true);
    int kc = event.GetKeyCode();
#ifdef _STFDEBUG
    std::cout << "User pressed " << char(kc) << ", corresponding keycode is " << kc << std::endl;
    std::cout << "Mouse Cursor Mode " << ParentFrame()->GetMouseQual() << std::endl;
#endif

    wxRect WindowRect(GetRect());
    switch (kc) {
    case WXK_LEFT:	//left cursor
        if (event.ControlDown()) {
            OnLeft();
            return;
        }
        if (event.ShiftDown()) {
            SPXW() = SPX()-WindowRect.width;
            Refresh();
            return;
        }
        OnPrevious();
        return;
    case WXK_RIGHT:	{//right cursor
        if (event.ControlDown()) {
            OnRight();
            return;
        }
        if (event.ShiftDown()) {
            SPXW() = SPX()+WindowRect.width;
            Refresh();
            return;
        }
        OnNext();
        return;
    }
    case WXK_DOWN:   //down cursor
        if (event.ControlDown()) {
            ChanScroll(-1);
        }
        else {
            OnDown();
        }
        return;
     case WXK_UP:     //up cursor
        if (event.ControlDown()) {
            ChanScroll(1);
        }
        else {
            OnUp();
        }
        return;
     case 49: //1
         ParentFrame()->SetZoomQual(stf::zoomch1);
         return;
     case 50:  //2
         if (Doc()->size()>1)
             ParentFrame()->SetZoomQual(stf::zoomch2);
         return;
     case 51: //3
         if (Doc()->size()>1)
             ParentFrame()->SetZoomQual(stf::zoomboth);
         return;
     case 69: // e
     case 101:
         ParentFrame()->SetMouseQual(stf::event_cursor);
         return;
     case 70:
     case 102: // f
         Fittowindow(true);
         return;
     case 77:  // m
     case 109:
         ParentFrame()->SetMouseQual(stf::measure_cursor);
         return;
     case 80: // p
     case 112:
         ParentFrame()->SetMouseQual(stf::peak_cursor);
         return;
     case 65: // 'a'
     case 97:
         // Select all traces:
         if (event.ControlDown()) {
             wxCommandEvent com;
             Doc()->Selectall(com);
             return;
         }
         return;
     case 66:  // b
     case 98:
         ParentFrame()->SetMouseQual(stf::base_cursor);
         return;
#ifdef WITH_PSLOPE
     case 79:  // key 'o' to activate PSlope cursors
     case 111:
         ParentFrame()->SetMouseQual(stf::pslope_cursor);
         return;
#endif
     case 68:  // d
     case 100:
         ParentFrame()->SetMouseQual(stf::decay_cursor);
         return;
     case 90:  // z
     case 122:
         ParentFrame()->SetMouseQual(stf::zoom_cursor);
         return;
     case 76:  // l
     case 108:
         ParentFrame()->SetMouseQual(stf::latency_cursor);
         return;
     case WXK_RETURN:    //Enter or Return
     {
         wxGetApp().OnPeakcalcexecMsg();
         pFrame->UpdateResults();
         return;
     }
     case 83: // Invalidate();//s
     case 115: {
         Doc()->Select();
         return;
     }
     case 88: // x
     case 120: 
     {
        wxCommandEvent foo;
        Doc()->OnSwapChannels(foo);
        return;
     }
    
     case 82: // Invalidate();//r
     case 114: {
         Doc()->Remove();
         return;
     }
    }

    switch (char(kc)) {
    case '0':
    case '=':
    case '+':
        if (event.ControlDown()) {
            OnXenllo();
            return;
        }
        OnYenllo();
        return;
    case '-':
        if (event.ControlDown()) {
            OnXshrinklo();
            return;
        }
        OnYshrinklo();
        return;
    }
}

void wxStfGraph::OnZoomHV(wxCommandEvent& event) {
    OnZoomH(event);
    OnZoomV(event);
}

void wxStfGraph::OnZoomH(wxCommandEvent& WXUNUSED(event)) {
    wxRect WindowRect=GetRect();
    llz_x=(llz_x - SPX()) / XZ();
    ulz_x=(ulz_x - SPX()) / XZ();
    int points=(int)(ulz_x - llz_x);
    XZW()=(double)WindowRect.width / points;
    SPXW()=(int)(-llz_x * XZ());
    isZoomRect=false;
}

void wxStfGraph::OnZoomV(wxCommandEvent& WXUNUSED(event)) {
    wxRect WindowRect=GetRect();
    llz_y=(SPY() - llz_y) / YZ();
    ulz_y=(SPY() - ulz_y) / YZ();
    YZW()=WindowRect.height/fabs(ulz_y-llz_y);
    SPYW()=(int)(WindowRect.height + ulz_y * YZ());

    if (Doc()->size() > 1) {
        llz_y2=(SPY2()-llz_y2)/YZ2();
        ulz_y2=(SPY2()-ulz_y2)/YZ2();
        YZ2W()=WindowRect.height/fabs(ulz_y2-llz_y2);
        SPY2W()=(int)(WindowRect.height + ulz_y2 * YZ2());
    }
    isZoomRect=false;
}

#if defined __WXMAC__ && !(wxCHECK_VERSION(2, 9, 0))
void wxStfGraph::OnPaint(wxPaintEvent &WXUNUSED(event)) {
    wxPaintDC PDC(this);
    OnDraw(PDC);
}
#endif

double prettyNumber( double fDistance, double pixelDistance, int limit ) {
    double fScaled = 1.0;
    for (;;)
    {
        //set stepsize
        int nZeros = (int)log10(fScaled);
        int prev10e = (int)(pow(10.0, nZeros));
        int next10e = prev10e * 10;
        int step = prev10e < 1 ? 1 : prev10e;
        
        if ( fScaled / prev10e  > 5 ) {
            fScaled = next10e;
            step = next10e;
        }
        
        //check whether f scale is ok
        if ((fScaled/fDistance) * pixelDistance > limit || fScaled>1e9)
            break;
        else {
            //suggest a new f scale:
            fScaled += step;
        }
    }
    return fScaled;
}

void wxStfGraph::CreateScale(wxDC* pDC)
{
    // catch bizarre y-Zooms:
    double fstartPosY=(double)SPY();
    if (fabs(fstartPosY)>(double)1.0e15)
        SPYW()=0;
    if (fabs(YZ())>1e15)
        YZW()=1.0;

    if (!isPrinted) {
        wxFont font((int)(8*printScale), wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL,
                    wxFONTWEIGHT_NORMAL);
        pDC->SetFont(font);
    }

    //Copy main window coordinates to 'WindowRect'
    wxRect WindowRect(GetRect());
    if (isPrinted)
    {	//Set WindowRect to print coordinates (page size)
        WindowRect=printRect;
    }

    //1. Creation of x-(time-)scale:
    //distance between two neigboured time steps in pixels:
    XZW()=XZ()>0 ? XZ() : 1.0;

    double pixelDistance=XZ();

    //distance between time steps in msec:
    //(it might be more elegant to read out the equalspaced step size
    // directly from the Doc())
    double timeDistance=1.0/Doc()->GetSR();
    //i.e., 1 timeDistance corresponds to 1 pixelDistance

    //get an integer time value which comes close to 150 pixels:
    int limit=(int)(100*printScale);
    double timeScaled = prettyNumber(timeDistance, pixelDistance, limit);
    int barLength=(int)((timeScaled/timeDistance) * pixelDistance);

    //2. creation of y-(voltage- or current-)scale
    //distance between two neigboured yValues in pixels:
    YZW()=YZ()>1e-9 ? YZ() : 1.0;
    double pixelDistanceY=YZ();
    //real distance (difference) between two neighboured Values:
    double realDistanceY=1.0;

    //get an integer y-value which comes close to 150 pixels:
    double yScaled = prettyNumber(realDistanceY, pixelDistanceY, limit);

    int barLengthY=(int)((yScaled/realDistanceY) * pixelDistanceY);

    //3. creation of y-scale for the second channel
    //Fit scale of second channel to window
    //distance between two neigboured yValues in pixels:
    int barLengthY2=100;
    double yScaled2 =1.0;
    double pixelDistanceY2= 1.0;
    //real distance (difference) between two neighboured Values:
    double realDistanceY2 = 1.0;
    if ((Doc()->size()>1))
    {
        pixelDistanceY2= YZ2();
        //get an entire y-value which comes close to 150 pixels:
        yScaled2 = prettyNumber(realDistanceY2, pixelDistanceY2, limit);
        barLengthY2=(int)((yScaled2/realDistanceY2) * pixelDistanceY2);
    }	//End creation y-scale of the 2nd Channel

    if (wxGetApp().get_isBars()) {
        // Use scale bars
        std::vector<wxPoint> Scale(5);
        // Distance of scale bar from bottom and right border of window:
        int bottomDist=(int)(50*printScale);
        int rightDist=(int)(60*printScale);
        // leave space for a second scale bar:
        if ((Doc()->size()>1)) rightDist*=2;
        // Set end points for the scale bar
        Scale[0]=wxPoint(WindowRect.width-rightDist-barLength,
                WindowRect.height-bottomDist);
        Scale[1]=wxPoint(WindowRect.width-rightDist,
                WindowRect.height-bottomDist);
        Scale[2]=wxPoint(WindowRect.width-rightDist,
                WindowRect.height-bottomDist-barLengthY);
        if (Doc()->size()>1 && pFrame->ShowSecond())
        {	//Set end points for the second channel y-bar
            Scale[3]=wxPoint(WindowRect.width-rightDist/2,
                    WindowRect.height-bottomDist);
            Scale[4]=wxPoint(WindowRect.width-rightDist/2,
                    WindowRect.height-bottomDist-barLengthY2);
        }

        // Set scalebar labels
        wxString scaleXString;
        scaleXString << (int)timeScaled << wxT(" ms");
        // Center of x-scalebar:
        int xCenter=WindowRect.width-(Scale[1].x-Scale[0].x)/2-rightDist;
        wxRect TextFrameX(
                wxPoint(xCenter-(int)(40*printScale),WindowRect.height-bottomDist+(int)(5.0*(double)printScale)),
                wxPoint(xCenter+(int)(40*printScale),WindowRect.height-bottomDist+(int)(25.0*(double)printScale))
        );
        if (!isLatex) {
            pDC->DrawLabel( scaleXString,TextFrameX,wxALIGN_CENTRE_HORIZONTAL | wxALIGN_TOP );
        } else {
#if 0
            wxLatexDC* pLatexDC = (wxLatexDC*)pDC;
            pLatexDC->DrawLabelLatex( scaleXString, TextFrameX,wxALIGN_CENTRE_HORIZONTAL | wxALIGN_TOP );
#endif
        }
        wxString scaleYString;
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
        scaleYString <<  (int)yScaled << wxT(" ") << Doc()->at(Doc()->GetCurChIndex()).GetYUnits() << wxT("\0");
#else
        scaleYString <<  (int)yScaled << wxT(" ") << wxString(Doc()->at(Doc()->GetCurChIndex()).GetYUnits().c_str(), wxConvUTF8) << wxT("\0");
#endif        
        // Center of y-scalebar:
        int yCenter=WindowRect.height-bottomDist-(Scale[1].y-Scale[2].y)/2;
        wxRect TextFrameY(
                wxPoint(WindowRect.width-rightDist+(int)(5*printScale),yCenter-(int)(10*printScale)),
                wxPoint(WindowRect.width,yCenter+(int)(10*printScale))
        );
        if (!isLatex) {
            pDC->DrawLabel(scaleYString,TextFrameY,wxALIGN_LEFT | wxALIGN_CENTRE_VERTICAL);
        } else {
#if 0
            wxLatexDC* pLatexDC = (wxLatexDC*)pDC;
            pLatexDC->DrawLabelLatex(scaleYString,TextFrameY,wxALIGN_LEFT | wxALIGN_CENTRE_VERTICAL);
#endif
        }
        if (Doc()->size()>1  && pFrame->ShowSecond())	{
            wxString scaleYString2;
            scaleYString2 << (int)yScaled2 << wxT(" ")
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
                          << Doc()->at(Doc()->GetSecChIndex()).GetYUnits();
#else
            << wxString(Doc()->at(Doc()->GetSecChIndex()).GetYUnits().c_str(), wxConvUTF8);
#endif
            // Center of y2-scalebar:
            int y2Center=WindowRect.height-bottomDist-(Scale[3].y-Scale[4].y)/2;
            wxRect TextFrameY2(
                    wxPoint(WindowRect.width-rightDist/2+(int)(5*printScale),y2Center-(int)(10*printScale)),
                    wxPoint(WindowRect.width,y2Center+(int)(10*printScale))
            );
            pDC->SetTextForeground(*wxRED);
            if (!isLatex) {
                pDC->DrawLabel(scaleYString2,TextFrameY2,wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL);
            } else {
#if 0
                wxLatexDC* pLatexDC = (wxLatexDC*)pDC;
                pLatexDC->DrawLabelLatex(scaleYString2,TextFrameY2,wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL);
#endif
            }
            pDC->SetTextForeground(*wxBLACK);
        }
        //Set PenStyle
        if (!isPrinted)
            pDC->SetPen(scalePen);
        else
            pDC->SetPen(scalePrintPen);
        //Plot them
        pDC->DrawLine(Scale[0],Scale[1]);
        pDC->DrawLine(Scale[1],Scale[2]);
        if (Doc()->size()>1  && pFrame->ShowSecond()) {
            if (!isPrinted)
                pDC->SetPen(scalePen2);
            else
                pDC->SetPen(scalePrintPen2);
            pDC->DrawLine(Scale[3],Scale[4]);
        }

    } else {
        // Use grid
        // Added 11/02/2006, CSH
        // Distance of coordinates from bottom, left, top and right border of window:
        int bottomDist=(int)(50*printScale);
        int leftDist=(int)(50*printScale);
        int topDist=(int)(20*printScale);
        int rightDist=(int)(20*printScale);
        // upper left corner:
        pDC->DrawLine(leftDist,topDist,leftDist,WindowRect.height-bottomDist);
        // lower right corner:
        pDC->DrawLine(leftDist,WindowRect.height-bottomDist,
                WindowRect.width-rightDist,WindowRect.height-bottomDist);
        // second y-axis:
        if (Doc()->size()>1  && pFrame->ShowSecond()) {
            pDC->SetPen(scalePen2);
            // upper left corner:
            pDC->DrawLine(leftDist*2,topDist,leftDist*2,WindowRect.height-bottomDist);
        }
        //Set PenStyle
        if (!isPrinted)
            pDC->SetPen(scalePen);
        else
            pDC->SetPen(scalePrintPen);
        // Set ticks:
        int tickLength=(int)(10*printScale);
        // Find first y-axis tick:
        // Get y-value of bottomDist:
        double yBottom=(SPY()-(WindowRect.height-bottomDist))/YZ();
        // Find next-higher integer multiple of barLengthY:
        int nextTickMult=(int)(yBottom/yScaled);
        // nextTickMult is truncated; hence, negative and positive values
        // have to be treated separately:
        if (yBottom>0) {
            nextTickMult++;
        }
        // pixel position of this tick:
        double yFirst=nextTickMult*yScaled;
        int yFirstTick=yFormat(yFirst);
        // How many times does the y-scale bar fit into the window?
        int yScaleInWindow=(yFirstTick-topDist)/barLengthY;
        // y-Axis ticks:
        for (int n_tick_y=0;n_tick_y<=yScaleInWindow;++n_tick_y) {
            pDC->DrawLine(leftDist-tickLength,
                    yFirstTick-n_tick_y*barLengthY,
                    leftDist,
                    yFirstTick-n_tick_y*barLengthY);
            // Create a rectangle from the left window border to the tick:
            wxRect TextFrame(
                    wxPoint(0,yFirstTick-n_tick_y*barLengthY-(int)(10*printScale)),
                    wxPoint(leftDist-tickLength-1,yFirstTick-n_tick_y*barLengthY+(int)(10*printScale))
            );
            // Draw Text:
            int y=(int)(yScaled*n_tick_y+yFirst);
            wxString yLabel;yLabel << y;
            pDC->DrawLabel(yLabel,TextFrame,wxALIGN_RIGHT | wxALIGN_CENTER_VERTICAL);
        }
        // Write y units:
        // Length of y-axis:
        int yLength=WindowRect.height-topDist-bottomDist;
        // position of vertical center:
        int vCenter=topDist+yLength/2;
        wxRect TextFrame(
                wxPoint(2,vCenter-(int)(10*printScale)),
                wxPoint(leftDist-tickLength-1,vCenter+(int)(10*printScale))
        );
        pDC->DrawLabel(
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
                       Doc()->at(Doc()->GetCurChIndex()).GetYUnits(),
#else
                       wxString(Doc()->at(Doc()->GetCurChIndex()).GetYUnits().c_str(), wxConvUTF8),
#endif
                TextFrame,
                wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL
        );

        // y-Axis of second channel:
        if (Doc()->size()>1  && pFrame->ShowSecond()) {
            pDC->SetPen(scalePen2);
            // Find first y-axis tick:
            // Get y-value of bottomDist:
            double y2Bottom=(SPY2()-(WindowRect.height-bottomDist))/YZ2();
            // Find next-higher integer multiple of barLengthY:
            int nextTickMult=(int)(y2Bottom/yScaled2);
            // nextTickMult is truncated; hence, negative and positive values
            // have to be treated separately:
            if (y2Bottom>0) {
                nextTickMult++;
            }
            // pixel position of this tick:
            double y2First=nextTickMult*yScaled2;
            int y2FirstTick=yFormat2(y2First);
            // How many times does the y-scale bar fit into the window?
            int y2ScaleInWindow = 1;
            if (barLengthY2 > 0) {
                y2ScaleInWindow = (y2FirstTick-topDist)/barLengthY2;
            } else {
                y2ScaleInWindow = (y2FirstTick-topDist)/1e-15;
            }
            // y-Axis ticks:
            for (int n_tick_y=0;n_tick_y<=y2ScaleInWindow;++n_tick_y) {
                pDC->DrawLine(leftDist*2-tickLength,
                        y2FirstTick-n_tick_y*barLengthY2,
                        leftDist*2,
                        y2FirstTick-n_tick_y*barLengthY2);
                // Create a rectangle from the left window border to the tick:
                wxRect TextFrame2(
                        wxPoint(0,y2FirstTick-n_tick_y*barLengthY2-(int)(10*printScale)),
                        wxPoint(leftDist*2-tickLength-1,y2FirstTick-n_tick_y*barLengthY2+(int)(10*printScale))
                );
                // Draw Text:
                int y2=(int)(yScaled2*n_tick_y+y2First);
                wxString y2Label;
                y2Label << y2;
                pDC->SetTextForeground(*wxRED);
                pDC->DrawLabel(y2Label,TextFrame2,wxALIGN_RIGHT | wxALIGN_CENTER_VERTICAL);
                pDC->SetTextForeground(*wxBLACK);
            }
            // Write y units:
            // Length of y-axis:
            int y2Length=WindowRect.height-topDist-bottomDist;
            // position of vertical center:
            int v2Center=topDist+y2Length/2;
            wxRect TextFrame2(
                    wxPoint(2+leftDist,v2Center-(int)(10*printScale)),
                    wxPoint(leftDist*2-tickLength-1,v2Center+(int)(10*printScale))
            );
            pDC->SetTextForeground(*wxRED);
            pDC->DrawLabel(
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
                           Doc()->at(Doc()->GetSecChIndex()).GetYUnits(),
#else
                           wxString(Doc()->at(Doc()->GetSecChIndex()).GetYUnits().c_str(), wxConvUTF8),
#endif                           
                    TextFrame2,
                    wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL
            );
            pDC->SetTextForeground(*wxBLACK);
        }
        // x-Axis ticks:
        // if x axis starts with the beginning of the trace, find first tick:
        int xFirstTick=leftDist;
        double xFirst=0.0;
        if (isSyncx) {
            // Find first x-axis tick:
            // Get x-value of leftDist:
            double xLeft=(leftDist-SPX())/XZ()*Doc()->GetXScale();
            // Find next-higher integer multiple of barLengthX:
            int nextTickMult=(int)(xLeft/timeScaled);
            // nextTickMult is truncated; hence, negative and positive values
            // have to be treated separately:
            if (xLeft>0) {
                nextTickMult++;
            }
            // pixel position of this tick:
            xFirst=nextTickMult*timeScaled;
            double xFirstSamplingPoint=xFirst/Doc()->GetXScale(); // units of sampling points
            xFirstTick=xFormat(xFirstSamplingPoint);
        }
        // How many times does the x-scale bar fit into the window?
        int xScaleInWindow=(WindowRect.width-xFirstTick-rightDist)/barLength;
        pDC->SetPen(scalePen);
        for (int n_tick_x=0;n_tick_x<=xScaleInWindow;++n_tick_x) {
            pDC->DrawLine(xFirstTick+n_tick_x*barLength,
                    WindowRect.height-bottomDist+tickLength,
                    xFirstTick+n_tick_x*barLength,
                    WindowRect.height-bottomDist);
            // Create a rectangle:
            wxRect TextFrame(
                    wxPoint(
                            xFirstTick+n_tick_x*barLength-(int)(40*printScale),
                            WindowRect.height-bottomDist+tickLength
                    ),
                    wxPoint(
                            xFirstTick+n_tick_x*barLength+(int)(40*printScale),
                            WindowRect.height-bottomDist+tickLength+(int)(20*printScale)
                    )
            );
            // Draw Text:
            int x=(int)(timeScaled*n_tick_x+xFirst);
            wxString xLabel;
            xLabel << x;
            pDC->DrawLabel(xLabel,TextFrame,wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL);
        }
        // Draw x-units:
        // Length of x-axis:
        int xLength=WindowRect.width-leftDist-rightDist;
        // position of horizontal center:
        int hCenter=leftDist+xLength/2;
        wxRect xTextFrame(
                wxPoint(
                        hCenter-(int)(40*printScale),
                        WindowRect.height-bottomDist+tickLength+(int)(20*printScale)
                ),
                wxPoint(
                        hCenter+(int)(40*printScale),
                        WindowRect.height-bottomDist+tickLength+(int)(40*printScale)
                )
        );
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
        pDC->DrawLabel(Doc()->GetXUnits(),xTextFrame,wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL);
#else
        pDC->DrawLabel(wxString(Doc()->GetXUnits().c_str(), wxConvUTF8),xTextFrame,wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL);
#endif        
    }
}

inline long wxStfGraph::xFormat(double toFormat) {
    return (int)(toFormat * XZ() + SPX());
}

inline long wxStfGraph::xFormat(long toFormat) {
    return (long)(toFormat * XZ() + SPX());
}

inline long wxStfGraph::xFormat(int toFormat) {
    return (long)(toFormat * XZ() + SPX());
}

inline long wxStfGraph::xFormat(std::size_t toFormat) {
    return (long)(toFormat * XZ() + SPX());
}

inline long wxStfGraph::yFormat(double toFormat) {
    return (long)(SPY() - toFormat * YZ());
}

inline long wxStfGraph::yFormat(long toFormat) {
    return (long)(SPY() - toFormat * YZ());
}

inline long wxStfGraph::yFormat(int toFormat) {
    return (long)(SPY() - toFormat * YZ());
}

inline long wxStfGraph::yFormat2(double toFormat) {
    return (long)(SPY2() - toFormat * YZ2());
}

inline long wxStfGraph::yFormat2(long toFormat){
    return (long)(SPY2() - toFormat * YZ2());
}

inline long wxStfGraph::yFormat2(int toFormat){
    return (long)(SPY2() - toFormat * YZ2());
}

inline long wxStfGraph::yFormatB(double toFormat) {
    return (long)(yzoombg.startPosY - toFormat * yzoombg.yZoom);
}

inline long wxStfGraph::yFormatB(long toFormat){
    return (long)(yzoombg.startPosY - toFormat * yzoombg.yZoom);
}

inline long wxStfGraph::yFormatB(int toFormat){
    return (long)(yzoombg.startPosY - toFormat * yzoombg.yZoom);
}

void wxStfGraph::FittorectY(YZoom& yzoom, const wxRect& rect, double min, double max, double screen_part) {
    
    yzoom.yZoom = (rect.height/fabs(max-min))*screen_part;
    yzoom.startPosY = (long)(((screen_part+1.0)/2.0)*rect.height
                            + min * yzoom.yZoom);
}

void wxStfGraph::Fittowindow(bool refresh)
{
    const double screen_part=0.5; //part of the window to be filled
    std::size_t points=Doc()->cursec().size();
    if (points==0) {
        wxGetApp().ErrorMsg(wxT("Array of size zero in wxGraph::Fittowindow()"));
        return;
    }
#if (__cplusplus < 201103)
    Vector_double::const_iterator max_el = std::max_element(Doc()->cursec().get().begin(), Doc()->cursec().get().end());
    Vector_double::const_iterator min_el = std::min_element(Doc()->cursec().get().begin(), Doc()->cursec().get().end());
#else
    Vector_double::const_iterator max_el = std::max(Doc()->cursec().get().begin(), Doc()->cursec().get().end());
    Vector_double::const_iterator min_el = std::min(Doc()->cursec().get().begin(), Doc()->cursec().get().end());
#endif
    double min = *min_el;
    if (min>1.0e12)  min= 1.0e12;
    if (min<-1.0e12) min=-1.0e12;
    double max = *max_el;
    if (max>1.0e12)  max= 1.0e12;
    if (max<-1.0e12) max=-1.0e12;
    wxRect WindowRect(GetRect());
    switch (ParentFrame()->GetZoomQual())
    {	//Depending on the zoom radio buttons (Mouse field)
    //in the (trace navigator) control box
    case stf::zoomboth:
        if(!(Doc()->size()>1))
            return;

        //Fit to window Ch2
        FitToWindowSecCh(false);
        //Fit to window Ch1
        XZW()=(double)WindowRect.width /points;
        SPXW()=0;
        FittorectY(Doc()->GetYZoomW(Doc()->GetCurChIndex()), WindowRect, min, max, screen_part);
        break;
    case stf::zoomch2:
        //ErrorMsg if no second channel available
        if(!(Doc()->size()>1))
            return;

        //Fit to window Ch2
        FitToWindowSecCh(false);
        break;
    default:
        //ErrorMsg if no second channel available
        //			Invalidate();
        //Fit to window Ch1
        XZW()=(double)WindowRect.width /points;
        SPXW()=0;
        FittorectY(Doc()->GetYZoomW(Doc()->GetCurChIndex()), WindowRect, min, max, screen_part);
        break;
    }
    if (refresh) Refresh();
}

void wxStfGraph::FitToWindowSecCh(bool refresh)
{

    if (Doc()->size()>1) {
        //Get coordinates of the main window
        wxRect WindowRect(GetRect());

        const double screen_part=0.5; //part of the window to be filled
        std::size_t secCh=Doc()->GetSecChIndex();
    #undef min
    #undef max
#if (__cplusplus < 201103)
        Vector_double::const_iterator max_el = std::max_element(Doc()->get()[secCh][Doc()->GetCurSecIndex()].get().begin(),
                                                                Doc()->get()[secCh][Doc()->GetCurSecIndex()].get().end());
        Vector_double::const_iterator min_el = std::min_element(Doc()->get()[secCh][Doc()->GetCurSecIndex()].get().begin(),
                                                                Doc()->get()[secCh][Doc()->GetCurSecIndex()].get().end());
#else
        Vector_double::const_iterator max_el = std::max(Doc()->get()[secCh][Doc()->GetCurSecIndex()].get().begin(),
                                                               Doc()->get()[secCh][Doc()->GetCurSecIndex()].get().end());
        Vector_double::const_iterator min_el = std::min(Doc()->get()[secCh][Doc()->GetCurSecIndex()].get().begin(),
                                                                Doc()->get()[secCh][Doc()->GetCurSecIndex()].get().end());
#endif

        double min=*min_el;
        double max=*max_el;
        FittorectY(Doc()->GetYZoomW(Doc()->GetSecChIndex()), WindowRect, min, max, screen_part);
        if (refresh) Refresh();
    }
}	//End FitToWindowSecCh()

void wxStfGraph::ChangeTrace(int trace) {
    stf::SectionAttributes sec_attr = Doc()->GetCurrentSectionAttributes();
    if (!sec_attr.eventList.empty() && trace != Doc()->GetCurSecIndex()) {
        for (event_it it2 = sec_attr.eventList.begin(); it2 != sec_attr.eventList.end(); ++it2) {
            it2->GetCheckBox()->Show(false);
        }
    }

    Doc()->SetSection(trace);
    wxGetApp().OnPeakcalcexecMsg();
    pFrame->SetCurTrace(trace);
    Refresh();
}

void wxStfGraph::OnPrevious() {
    if (Doc()->get()[Doc()->GetCurChIndex()].size()==1) return;
    std::size_t curSection=Doc()->GetCurSecIndex();
    if (Doc()->GetCurSecIndex() > 0) curSection--;
    else curSection=Doc()->get()[Doc()->GetCurChIndex()].size()-1;
    ChangeTrace(curSection);
}

void wxStfGraph::OnFirst() {
    if (Doc()->GetCurSecIndex()==0) return;
    ChangeTrace(0);
}

void wxStfGraph::OnLast() {
    if (Doc()->GetCurSecIndex()==Doc()->get()[Doc()->GetCurChIndex()].size()-1) return;
    std::size_t curSection=Doc()->get()[Doc()->GetCurChIndex()].size()-1;
    ChangeTrace(curSection);
}

void wxStfGraph::OnNext() {
    if (Doc()->get()[Doc()->GetCurChIndex()].size()==1) return;
    std::size_t curSection=Doc()->GetCurSecIndex();
    if (curSection < Doc()->get()[Doc()->GetCurChIndex()].size()-1) curSection++;
    else curSection=0;
    ChangeTrace(curSection);
}

void wxStfGraph::OnUp() {
    switch (ParentFrame()->GetZoomQual())
    {	//Depending on the zoom radio buttons (Mouse field)
    //in the (trace navigator) control box
    case stf::zoomboth:
        //ErrorMsg if no second channel available
        //yZooms of Ch1 are performed keeping the base constant
        SPYW()=SPY() - 20;
        if(!(Doc()->size()>1)) break;
        //Ymove of Ch2 is performed
        SPY2W()=SPY2() - 20;
        break;
    case stf::zoomch2:
        if(!(Doc()->size()>1)) break;
        //Ymove of Ch2 is performed
        SPY2W()=SPY2() - 20;
        break;
    default:
        //Ymove of Ch1 is performed
        SPYW()=SPY() - 20;
        break;
    }
    Refresh();
}

void wxStfGraph::OnDown() {
    switch (ParentFrame()->GetZoomQual())
    {	//Depending on the zoom radio buttons (Mouse field)
    //in the (trace navigator) control box
    case stf::zoomboth:
        //yZooms of Ch1 are performed keeping the base constant
        SPYW()=SPY() + 20;
        if(!(Doc()->size()>1)) break;
        //Ymove of Ch2 is performed
        SPY2W()=SPY2() + 20;
        break;
    case stf::zoomch2:
        if(!(Doc()->size()>1)) break;
        //Ymove of Ch2 is performed
        SPY2W()=SPY2() + 20;
        break;
    default:
        //Ymove of Ch1 is performed
        SPYW()=SPY() + 20;
        break;
    }
    Refresh();
}

void wxStfGraph::ChanScroll(int direction) {
    /* on Control + cursor press, adjust the active channel
       up or down. 
    */
    
    // direction is either +1, or -1 
    
    int ref_chan = Doc()->GetSecChIndex();  
    int new_chan = Doc()->GetCurChIndex() + direction;
    int last_chan = Doc()->size()-1;
    
    // Exit early if there is only one channel    
    if (Doc()->size() == 1) {
        return;
    }
    
    /*Rollover conditions
      -------------------
      I ended up resorting to ternery operators because I need to
      check both that we haven't gone over the document range and
      that we aren't hitting the reference channel
    */
    
    if (new_chan == ref_chan) {
        // Skip the reference channel
        new_chan += direction;
        // move one unit past the ref channel.
    }
    
    if (new_chan > last_chan) {
    	// Rollover to start if channel out of range
    	// making sure to skip the reference channel
        new_chan = (ref_chan == 0)? 1 : 0;
    }
    else if (new_chan < 0) {
        // Rollover to end if channel out of range
        // making sure to skip the reference channel
        new_chan = (ref_chan == last_chan)? last_chan-1 : last_chan;
    } 
    
    /*Update the window
      -----------------
    */
    
    // Pointer to wxStfChildFrame to access Channel selection combo
    wxStfChildFrame* pFrame = (wxStfChildFrame*)Doc()->GetDocumentWindow();
    if (!pFrame) {
        return;
    }
    // set the channel selection combo 
    //pFrame->SetChannels( actDoc()->GetCurChIndex(), actDoc()->GetSecChIndex()); 
    pFrame->SetChannels(new_chan, ref_chan); 
    pFrame->UpdateChannels(); // update according to the combo
    Refresh();
}

void wxStfGraph::OnRight() {
    SPXW()=SPX() + 20;
    Refresh();
}

void wxStfGraph::OnLeft() {
    SPXW()=SPX() - 20;
    Refresh();
}

void wxStfGraph::OnXenlhi() {
    ChangeXScale(2.0);
}

void wxStfGraph::OnXenllo() {
    ChangeXScale(1.1);
}

void wxStfGraph::OnXshrinklo() {
    ChangeXScale(1.0/1.1);
}

void wxStfGraph::OnXshrinkhi() {
    ChangeXScale(0.5);
}

void wxStfGraph::ChangeXScale(double factor) {
    wxRect WindowRect(GetRect());
    //point in the middle:
    double middle=(WindowRect.width/2.0 - SPX()) / XZ();

    //new setting for xZoom
    XZW()=XZ() * factor;

    //calculation of new start position
    SPXW()=(int)(WindowRect.width/2.0 - middle * XZ());
    Refresh();
}

void wxStfGraph::OnYenlhi() {
    ChangeYScale(2.0);
}

void wxStfGraph::OnYenllo() {
    ChangeYScale(1.1);
}

void wxStfGraph::OnYshrinklo() {
    ChangeYScale(1/1.1);
}

void wxStfGraph::OnYshrinkhi() {
    ChangeYScale(0.5);
}

void wxStfGraph::ChangeYScale(double factor) {
    switch (ParentFrame()->GetZoomQual()) {
    // Depending on the zoom radio buttons (Mouse field)
    // in the (trace navigator) control box
    case stf::zoomboth:
        //yZooms of Ch1 are performed keeping the base constant
        SPYW()=(int)(SPY() + Doc()->GetBase() * (YZ() * factor
                - YZ()));
        YZW()=YZ() * factor;
        //ErrorMsg if no second channel available
        if (Doc()->size()<=1) break;
        //yZooms of Ch2 are performed keeping the base constant
        SPY2W()=(int)(SPY2()
                + Doc()->GetBase() * (YZ2() * factor
                        - YZ2()));
        YZ2W()=YZ2() * factor;
        break;
    case stf::zoomch2:
        if (Doc()->size()<=1) break;
        //yZooms of Ch2 are performed keeping the base constant
        SPY2W()=(int)(SPY2()
                + Doc()->GetBase() * (YZ2() * factor
                        - YZ2()));
        YZ2W()=YZ2() * factor;
        break;
    default:
        //yZooms of Ch1 are performed keeping the base constant
        SPYW()=(int)(SPY() + Doc()->GetBase() * (YZ() * factor
                - YZ()));
        YZW()=YZ() * factor;
        break;
    }
    Refresh();
}

void wxStfGraph::Ch2base() {
    if ((Doc()->size()>1)) {
        double base2=0.0;
        try {
            double var2=0.0;
            base2=stfnum::base(Doc()->GetBaselineMethod(),var2,Doc()->get()[Doc()->GetSecChIndex()][Doc()->GetCurSecIndex()].get(),
                    Doc()->GetBaseBeg(),Doc()->GetBaseEnd());
        }
        catch (const std::out_of_range& e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ) );
            return;
        }
        double base1=Doc()->GetBase();
        int base1_onScreen=yFormat(base1);
        // Adjust startPosY2 so that base2 is the same as base1 on the screen;
        // i.e. yFormat2(base2) == yFormat(base1)
        // this is what yFormat2(toFormat) does:
        // return (int)(zoom.startPosY2 - toFormat * zoom.yZoom2);
        // Solved for startPosY2, this gets:
        SPY2W()=(int)(base1_onScreen+base2*YZ2());
        Refresh();
    }
}

void wxStfGraph::Ch2pos() {
    if ((Doc()->size()>1)) {
        SPY2W()=SPY();
        Refresh();
    }
}

void wxStfGraph::Ch2zoom() {
    if ((Doc()->size()>1)) {
        YZ2W()=YZ();
        Refresh();
    }
}

void wxStfGraph::Ch2basezoom() {
    if ((Doc()->size()>1)) {
        // Adjust y-scale without refreshing:
        YZ2W()=YZ();
        // Adjust baseline:
        double base2=0.0;
        try {
            double var2=0.0;
            base2=stfnum::base(Doc()->GetBaselineMethod(),var2,Doc()->get()[Doc()->GetSecChIndex()][Doc()->GetCurSecIndex()].get(),
                    Doc()->GetBaseBeg(),Doc()->GetBaseEnd());
        }
        catch (const std::out_of_range& e) {
            wxGetApp().ExceptMsg( wxString( e.what(), wxConvLocal ) );
            return;
        }
        double base1=Doc()->GetBase();
        int base1_onScreen=yFormat(base1);
        // Adjust startPosY2 so that base2 is the same as base1 on the screen;
        // i.e. yFormat2(base2) == yFormat(base1)
        // this is what yFormat2(toFormat) does:
        // return (int)(zoom.startPosY2 - toFormat * zoom.yZoom2);
        // Solved for startPosY2, this gets:
        SPY2W()=(int)(base1_onScreen+base2*YZ2());
        Refresh();
    }
}

void wxStfGraph::set_isPrinted(bool value) {
    if (value==false) {
        printScale=1.0;
        no_gimmicks=false;
    } else {
#if defined __WXGTK__ || defined __APPLE__
        printScale=0.25;
#endif        
        // store zoom settings upon switching from normal to print view:
        if (isPrinted==false) {
            //            zoomOld=zoom;
        }
    }
    isPrinted=value;
}
