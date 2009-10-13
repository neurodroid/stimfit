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
#include <wx/dcps.h>
#if 0
#include "wx/dclatex.h"
#endif
#include <wx/metafile.h>
#include <wx/printdlg.h>
#include <wx/paper.h>
#if wxCHECK_VERSION(2, 9, 0)
#include <wx/dcsvg.h>
#endif

#ifdef _STFDEBUG
#include <iostream>
#endif


#include "./app.h"
#include "./doc.h"
#include "./view.h"
#include "./parentframe.h"
#include "./childframe.h"
#include "./printout.h"
#include "./stfcheckbox.h"
#include "./dlgs/cursorsdlg.h"
#include "./dlgs/smalldlgs.h"
#include "./../core/measlib.h"
#include "./usrdlg/usrdlg.h"
#include "./graph.h"

BEGIN_EVENT_TABLE(wxStfGraph, wxWindow)
EVT_MENU(wxID_ZOOMHV,wxStfGraph::OnZoomHV)
EVT_MENU(wxID_ZOOMH,wxStfGraph::OnZoomH)
EVT_MENU(wxID_ZOOMV,wxStfGraph::OnZoomV)
EVT_MOUSE_EVENTS(wxStfGraph::OnMouseEvent)
EVT_KEY_DOWN( wxStfGraph::OnKeyDown )
#if defined __WXMAC__ && !(wxCHECK_VERSION(2, 9, 0))
EVT_PAINT( wxStfGraph::OnPaint )
#endif
END_EVENT_TABLE()

// Define a constructor for my canvas
wxStfGraph::wxStfGraph(wxView *v, wxStfChildFrame *frame, const wxPoint& pos, const wxSize& size, long style):
    wxScrolledWindow(frame, wxID_ANY, pos, size, style),pFrame(frame),
    isZoomRect(false),no_gimmicks(false),isPrinted(false),isLatex(false),firstPass(true),isSyncx(false),resLimit(100000),
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
    slopePen(*wxBLUE,2,wxSOLID), //Solid blue line
    latencyPen(*wxBLUE,1,wxDOT),
    alignPen(*wxBLUE,1,wxSHORT_DASH),
    measPen(*wxBLACK,1,wxDOT),
    eventPen(*wxBLUE,2,wxSOLID),

    standardPrintPen(*wxBLACK,printSizePen1,wxSOLID), //Solid black line
    standardPrintPen2(*wxRED,printSizePen1,wxSOLID), //Solid red line
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
    slopePrintPen(*wxBLUE,printSizePen4,wxSOLID), //Solid blue line
    resultsPrintPen(*wxLIGHT_GREY,printSizePen2,wxSOLID),//Solid light grey line
    latencyPrintPen(*wxBLUE,printSizePen1,wxDOT),//Dotted violett line
    baseBrush(*wxLIGHT_GREY,wxBDIAGONAL_HATCH),
    zeroBrush(*wxLIGHT_GREY,wxFDIAGONAL_HATCH),
    lastLDown(0,0),
    m_zoomContext( new wxMenu ),
    m_eventContext( new wxMenu )
{
    m_zoomContext->Append( wxID_ZOOMHV, wxT("Expand zoom window horizontally && vertically") );
    m_zoomContext->Append( wxID_ZOOMH, wxT("Expand zoom window horizontally") );
    m_zoomContext->Append( wxID_ZOOMV, wxT("Expand zoom window vertically") );

    m_eventContext->Append( wxID_EVENT_ADDEVENT, wxT("Add an event that starts here") );
    m_eventContext->Append( wxID_EVENT_ERASE, wxT("Erase all events") );
    m_eventContext->Append( wxID_EVENT_EXTRACT, wxT("Extract selected events") );

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
    wxRect WindowRect(GetRect());

    if (isPrinted)
    {	//enhance resolution for printing - see OnPrint()
        //Ensures the scaling of all pixel dependent drawings

        //Calculate scaling

        for ( ch_it cit = Doc()->get().begin(); cit != Doc()->get().end(); ++cit ) {
            cit->GetYZoomW() = cit->GetYZoomW() * printScale;
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


    }	//End if print out
    if (firstPass) {
        firstPass = false;

/*        wxAuiPaneInfo graphInfo;
        pFrame->GetMgr()->AddPane( this, graphInfo.Caption(wxT("Traces")).Name(wxT("Traces")).CaptionVisible(true).
                CloseButton(false).Centre().PaneBorder(true) );
        pFrame->GetMgr()->Update();
        pFrame->ActivateGraph();
*/
        if (wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewScaleBars"),1)) {
            pFrame->GetMenuBar()->GetMenu(2)->Check(wxID_SCALE,true);
            wxGetApp().set_isBars(true);
        } else {
            pFrame->GetMenuBar()->GetMenu(2)->Check(wxID_SCALE,false);
            wxGetApp().set_isBars(false);
        }

        if (wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewSyncx"),1)) {
            // ParentFrame()->GetMenu()->CheckMenuItem(ID_VIEW_SYNCX,MF_CHECKED);
            isSyncx=true;
        } else {
            // ParentFrame()->GetMenu()->CheckMenuItem(ID_VIEW_SYNCX,MF_UNCHECKED);
            isSyncx=false;
        }

        if (wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewHiRes"),1)) {
            pFrame->GetMenuBar()->GetMenu(2)->Check(wxID_HIRES,true);
#ifndef __APPLE__
            wxGetApp().set_isHires(true);
#else
            wxGetApp().set_isHires(false);
#endif
        } else {
            pFrame->GetMenuBar()->GetMenu(2)->Check(wxID_HIRES,false);
            wxGetApp().set_isHires(false);
        }
        //Ensure proper dimensioning
        //Determine scaling factors and Units
        //Zoom and offset variables are currently not part of the settings dialog =>
        //Read from registry
        YZW()=(double)(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("zoom.yZoom"), 1)
                / 100000.0);
        SPYW()=wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("zoom.startPosY"), 1);
        XZW()=(double)(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("zoom.xZoom"), 1)
                / 100000.0);
        SPXW()=wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("zoom.startPosX"), 1);


        if (XZ() <= 0 || YZ() <= 0)
            Fittowindow(false);
        if ((Doc()->get().size()>1))
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
        //Peak, Base, 20/80 rise time, half duration, ratio of rise/slope,
        //maximum slope and latencycursor calculation and update of result box
        // wxGetApp().OnPeakcalcexecMsg();
    }
    //Creates scale bars and labelings for display or print out
    //Calculate scale bars and labelings
    CreateScale(&DC);

    //Create additional rulers/lines and circles on display
    if (!no_gimmicks) 	{

        // crosshair through measurement cursor:
        int crosshairSize=20;
        if (!isPrinted) {
            DC.SetPen(measPen);
        } else {
            DC.SetPen(measPrintPen);
            crosshairSize=(int)(crosshairSize*printScale);
        }
        try {
            // circle:
            wxPoint corner(xFormat(Doc()->GetMeasCursor())-crosshairSize,
                    yFormat(Doc()->GetMeasValue())-crosshairSize);
            wxRect frame(
                    wxPoint( xFormat(Doc()->GetMeasCursor())-crosshairSize,
                             yFormat(Doc()->GetMeasValue())-crosshairSize ),
                    wxPoint( xFormat(Doc()->GetMeasCursor())+crosshairSize,
                             yFormat(Doc()->GetMeasValue())+crosshairSize )
            );
            DC.DrawEllipse(frame);//, corner, corner);
            // vertical part:
            DC.DrawLine( xFormat(Doc()->GetMeasCursor()),
                         yFormat(Doc()->GetMeasValue())-crosshairSize,
                         xFormat(Doc()->GetMeasCursor()),
                         yFormat(Doc()->GetMeasValue())+crosshairSize );
            if (wxGetApp().GetCursorsDialog()!=NULL &&
                    wxGetApp().GetCursorsDialog()->IsShown())
            {
                if (wxGetApp().GetCursorsDialog()->GetRuler())
                {
                    DrawVLine(&DC,Doc()->GetMeasCursor());
                }
            }

            // horizontal part:
            DC.DrawLine( xFormat(Doc()->GetMeasCursor())-crosshairSize,
                         yFormat(Doc()->GetMeasValue()),
                         xFormat(Doc()->GetMeasCursor())+crosshairSize,
                         yFormat(Doc()->GetMeasValue()) );
        }
        catch (const std::out_of_range& e) {
            wxGetApp().ExceptMsg( wxString( e.what(), wxConvLocal ) );
            return;
        }

        // crosshair through threshold:
        if (Doc()->GetThrT() >= 0) {
            int thrCrosshairSize = crosshairSize / 2.0;        
            if (!isPrinted) {
                DC.SetPen(peakPen);
            } else {
                DC.SetPen(peakPrintPen);
                thrCrosshairSize=(int)(thrCrosshairSize*printScale);
            }
            try {
                // circle:
                wxPoint corner(xFormat(Doc()->GetThrT())-thrCrosshairSize,
                               yFormat(Doc()->GetThreshold())-thrCrosshairSize);
                wxRect frame(
                             wxPoint( xFormat(Doc()->GetThrT())-thrCrosshairSize,
                                      yFormat(Doc()->GetThreshold())-thrCrosshairSize ),
                             wxPoint( xFormat(Doc()->GetThrT())+thrCrosshairSize,
                                      yFormat(Doc()->GetThreshold())+thrCrosshairSize )
                             );
                DC.DrawEllipse(frame);//, corner, corner);
                // vertical part:
                DC.DrawLine( xFormat(Doc()->GetThrT()),
                             yFormat(Doc()->GetThreshold())-thrCrosshairSize,
                             xFormat(Doc()->GetThrT()),
                             yFormat(Doc()->GetThreshold())+thrCrosshairSize );
                if (wxGetApp().GetCursorsDialog()!=NULL &&
                    wxGetApp().GetCursorsDialog()->IsShown())
                {
                    if (wxGetApp().GetCursorsDialog()->GetRuler())
                    {
                        DrawVLine(&DC,Doc()->GetThrT());
                    }
                }
                
                // horizontal part:
                DC.DrawLine( xFormat(Doc()->GetThrT())-thrCrosshairSize,
                             yFormat(Doc()->GetThreshold()),
                             xFormat(Doc()->GetThrT())+thrCrosshairSize,
                             yFormat(Doc()->GetThreshold()) );
            }
            catch (const std::out_of_range& e) {
                wxGetApp().ExceptMsg( wxString( e.what(), wxConvLocal ) );
                return;
            }
        }

        //creates red vertical and horizontal dashed lines through the peak
        if (!isPrinted)
            DC.SetPen(peakPen);
        else
            DC.SetPen(peakPrintPen);
        DrawVLine(&DC,Doc()->GetMaxT());
        DrawHLine(&DC,Doc()->GetPeak());
        //and red dotted lines through peak calculation limits
        if (!isPrinted)
            DC.SetPen(peakLimitPen);
        else
            DC.SetPen(peakLimitPrintPen);
        DrawVLine(&DC,Doc()->GetPeakBeg());
        DrawVLine(&DC,Doc()->GetPeakEnd());

        //creates a green horizontal dashed line through the base
        if (!isPrinted)
            DC.SetPen(basePen);
        else
            DC.SetPen(basePrintPen);
        DrawHLine(&DC,Doc()->GetBase());
        //and green dotted lines through Doc()->GetBase() calculation limits
        if (!isPrinted)
            DC.SetPen(baseLimitPen);
        else
            DC.SetPen(baseLimitPrintPen);
        DrawVLine(&DC,Doc()->GetBaseBeg());
        DrawVLine(&DC,Doc()->GetBaseEnd());

        //Create darkblue dotted lines through decay calculation limits
        if (!isPrinted)
            DC.SetPen(decayLimitPen);
        else
            DC.SetPen(decayLimitPrintPen);
        DrawVLine(&DC,Doc()->GetFitBeg());
        DrawVLine(&DC,Doc()->GetFitEnd());

        // Create dotted line as a latency cursor
        if (!isPrinted)
            DC.SetPen(latencyPen);
        else
            DC.SetPen(latencyPrintPen);
        DrawVLine(&DC,Doc()->GetLatencyBeg());
        DrawVLine(&DC,Doc()->GetLatencyEnd());
        // Create double-arrow between latency cursors:
        int latStart=xFormat(Doc()->GetLatencyBeg());
        int latEnd=xFormat(Doc()->GetLatencyEnd());
        DC.DrawLine(latStart,20,latEnd,20);
        // left arrowhead:
        DC.DrawLine(latStart+1,20,latStart+6,15);
        DC.DrawLine(latStart+1,20,latStart+6,25);
        // right arrowhead:
        DC.DrawLine(latEnd-1,20,latEnd-6,15);
        DC.DrawLine(latEnd-1,20,latEnd-6,25);


        // Created dashed line to indicate the alignment cursor
        /*		if (!isPrinted && (Doc()->get().size()>1)) {
        DC.SetPen(alignPen);
        DrawVLine(&DC,Doc()->GetAPMaxSlopeT());
        }
         */
        //Set circle size depending on output
        if (!isPrinted)
            boebbel=boebbelStd;
        else
            boebbel=boebbelPrint;

        if (Doc()->cur().HasEvents()) {
            DC.SetPen(eventPen);
            for (c_event_it it = Doc()->cur().GetEvents().begin(); it != Doc()->cur().GetEvents().end(); ++it) {
                // Create small arrows indicating the start of an event:
                eventArrow(&DC, (int)it->GetEventStartIndex());
                // Create circles indicating the peak of an event:
                try {
                    DrawCircle( &DC, it->GetEventPeakIndex(), Doc()->cur().at(it->GetEventPeakIndex()) );
                }
                catch (const std::out_of_range& e) {
                    wxGetApp().ExceptMsg( wxString( e.what(), wxConvLocal ) );
                    return;
                }
            }

            // resize list if necessary:
            if (cbList.size() != Doc()->cur().GetEvents().size()) {
                // destroy checkboxes that are not needed:
                for (std::size_t n_cbl = Doc()->cur().GetEvents().size();
                n_cbl < cbList.size();
                ++n_cbl)
                {
                    cbList[n_cbl]->Destroy();
                }
                cbList.resize(Doc()->cur().GetEvents().size());
            }
            std::size_t n_cb = 0;
            for (event_it it2 = Doc()->cur().GetEventsW().begin(); it2 != Doc()->cur().GetEventsW().end(); ++it2) {
                try {
                    if (cbList.at(n_cb) == NULL) {
                        cbList.at(n_cb) =
                            new wxStfCheckBox(
                                    this, -1, wxEmptyString, &*it2,
                                    wxPoint(xFormat(it2->GetEventStartIndex()), 0));
                    }
                    cbList.at(n_cb)->ResetEvent( &*it2 );
                    cbList.at(n_cb++)->Move(
                            wxPoint(xFormat(it2->GetEventStartIndex()), 0));
                }
                catch (const std::out_of_range& e) {
                    wxGetApp().ExceptMsg( wxString( e.what(), wxConvLocal ) );
                    return;
                }
            }
            // return focus to frame:
            SetFocus();
        } else { // no events
            // Destroy checkboxes (if any)
            std::vector<wxStfCheckBox*>::iterator it2;
            for (it2 = cbList.begin(); it2 != cbList.end(); ++it2) {
                if (*it2 != NULL)
                    (*it2)->Destroy();
            }
            if (!cbList.empty())
                cbList.clear();
        }

        if (Doc()->cur().HasPyMarkers()) {
            DC.SetPen(eventPen);
            for (c_marker_it it = Doc()->cur().GetPyMarkers().begin(); it != Doc()->cur().GetPyMarkers().end(); ++it) {
                // Create circles indicating the peak of an event:
                DC.DrawRectangle( xFormat(it->x), yFormat(it->y), boebbel*2.0, boebbel*2.0 );
            }
        }

        //draws green circles around the 20% and the 80% rise times
        if (!isPrinted)
            DC.SetPen(rtPen);
        else
            DC.SetPen(rtPrintPen);

        double reference = Doc()->GetBase();
        if ( !Doc()->GetFromBase() && Doc()->GetThrT() >= 0 ) {
            reference = Doc()->GetThreshold();
        }
        DrawCircle(&DC,Doc()->GetT20Real(),0.8*reference+0.2*Doc()->GetPeak());
        DrawCircle(&DC,Doc()->GetT80Real(),0.2*reference+0.8*Doc()->GetPeak());

        //draws circles around the half duration limits
        if (!isPrinted)
            DC.SetPen(hdPen);
        else
            DC.SetPen(hdPrintPen);
        DrawCircle(&DC,Doc()->GetT50LeftReal(),Doc()->GetT50Y());
        DrawCircle(&DC,Doc()->GetT50RightReal(),Doc()->GetT50Y());

        //draws dark violet circles around the points of steepest rise/decay
        if (!isPrinted)
            DC.SetPen(rdPen);
        else
            DC.SetPen(rdPrintPen);
        DrawCircle(&DC,Doc()->GetMaxRiseT(),Doc()->GetMaxRiseY());
        DrawCircle(&DC,Doc()->GetMaxDecayT(),Doc()->GetMaxDecayY());
    }	//End create additional rulers/lines and circles

    //Plot all selected traces and fitted functions if at least one trace ist selected
    //and 'is selected' is selected in the trace navigator/control box
    //Polyline() is used for printing to avoid separation of traces
    //in postscript files
    //LineTo()is used for display for performance reasons

    //Plot fit curves (including current trace)
    DrawFit(&DC);

    if (!Doc()->GetSelectedSections().empty() && pFrame->PlotSelected())
    {
        if (!isPrinted)
        {	//Draw traces on display
            DC.SetPen(selectPen);
            for (unsigned m=0; m < Doc()->GetSelectedSections().size(); ++m)
            {
                //For display use point to point drawing
                PlotTrace(
                        &DC,
                        Doc()->get()[Doc()->GetCurCh()][Doc()->GetSelectedSections()[m]].get()
                );
            }
        }  //End draw traces on display
        else
        {  //Draw traces for print out
            DC.SetPen(selectPrintPen);
            for (unsigned m=0; m < Doc()->GetSelectedSections().size() && Doc()->GetSelectedSections().size()>0; ++m)
            {
                PrintTrace(&DC,Doc()->get()[Doc()->GetCurCh()][Doc()->GetSelectedSections()[m]].get());
            }	//End draw for print out
        }	//End if display or print out
    }	//End plot all selected traces

    //Plot average
    if (Doc()->GetIsAverage())
    {	//Average is calculated but not plotted
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
    }	//End plot average


    // Plot integral boundaries
    if (Doc()->cur().IsIntegrated()) {
        DrawIntegral(&DC);
    }

    //Zoom window is displayed (see OnLeftButtonUp()) ...
    if (isZoomRect)
    {
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
    //End zoom

    //Plot of the second channel
    //Trace one when displayed first time
    if ((Doc()->get().size()>1) && pFrame->ShowSecond())
    {
        if (!isPrinted)
        {	//Draw current trace on display
            //For display use point to point drawing
            DC.SetPen(standardPen2);
            PlotTrace(&DC,Doc()->get()[Doc()->GetSecCh()][Doc()->GetCurSec()].get(), true);
        } else {	//Draw second channel for print out
            //For print out use polyline tool
            DC.SetPen(standardPrintPen2);
            PrintTrace(&DC,Doc()->get()[Doc()->GetSecCh()][Doc()->GetCurSec()].get(), true);
        }	// End display or print out
    }		//End plot of the second channel

    //Standard plot of the current trace
    //Trace one when displayed first time
    if (!isPrinted)
    {	//Draw current trace on display
        //For display use point to point drawing
        DC.SetPen(standardPen);
        PlotTrace(&DC,Doc()->get()[Doc()->GetCurCh()][Doc()->GetCurSec()].get());
    }
    else
    {	//Draw average for print out
        //For print out use polyline tool
        DC.SetPen(standardPrintPen);
        PrintTrace(&DC,Doc()->get()[Doc()->GetCurCh()][Doc()->GetCurSec()].get());
    }	// End display or print out
    //End plot of the current trace

    //Ensure old scaling after print out
    if(isPrinted)
    {
        for ( ch_it cit = Doc()->get().begin(); cit != Doc()->get().end(); ++cit ) {
            cit->GetYZoomW() = cit->GetYZoomW() * (1.0/printScale);
        }
        Doc()->GetXZoomW() = Doc()->GetXZoomW() * (1.0/printScale);
        WindowRect=printRect;
    }	//End ensure old scaling after print out

    view->OnDraw(& DC);
}

void wxStfGraph::PlotTrace( wxDC* pDC, const std::valarray<double>& trace, bool is2 ) {
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

    // speed up drawing by down-sampling large traces:
    int step=1;
    if ((int)(end-start)>resLimit && !wxGetApp().get_isHires()) {
        // truncate:
        step=div(int(end-start),resLimit).quot;
    }

    // apply filter at half the new sampling frequency:
    DoPlot(pDC, trace, start, end, step, is2);
}

void wxStfGraph::DoPlot( wxDC* pDC, const std::valarray<double>& trace, int start, int end, int step, bool is2 ) {
    boost::function<int(double)> yFormatFunc;
    
    if (!is2) {
        yFormatFunc = std::bind1st( std::mem_fun(&wxStfGraph::yFormatD), this);
    } else {
        yFormatFunc = std::bind1st( std::mem_fun(&wxStfGraph::yFormatD2), this);
    }

    int x_last = xFormat(start);
    int y_last = yFormatFunc( trace[start] );
    int y_max = y_last;
    int y_min = y_last;
    int x_next = 0;
    int y_next = 0;
    for (int n=start; n<end-1; ++n) {
        x_next = xFormat(n+1);
        y_next = yFormatFunc( trace[n+1] );
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
                pDC->DrawLine( x_last, y_last, x_last, y_min );
                y_last = y_min;
            }
            if (y_max != y_next) {
                pDC->DrawLine( x_last, y_last, x_last, y_max );
                y_last = y_max;
            }
            pDC->DrawLine( x_last, y_last, x_next, y_next );
            y_min = y_next;
            y_max = y_next;
            x_last = x_next;
            y_last = y_next;
        }
    }
}

void wxStfGraph::PrintTrace( wxDC* pDC, const std::valarray<double>& trace, bool is2 ) {
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
    DoPrint(pDC, trace, start, end, downsampling, is2);
}

void wxStfGraph::DoPrint( wxDC* pDC, const std::valarray<double> trace, int start, int end, int downsampling, bool is2 ) {
    boost::function<int(double)> yFormatFunc;
    
    if (!is2) {
        yFormatFunc = std::bind1st( std::mem_fun(&wxStfGraph::yFormatD), this);
    } else {
        yFormatFunc = std::bind1st( std::mem_fun(&wxStfGraph::yFormatD2), this);
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

void wxStfGraph::DrawCircle(wxDC* pDC,double x,double y) {
    wxPoint corner(xFormat(x)-boebbel,yFormat(y)-boebbel);
    wxRect Frame(
            wxPoint(xFormat(x)-boebbel,yFormat(y)-boebbel),
            wxPoint(xFormat(x)+boebbel,yFormat(y)+boebbel)
    );
    pDC->DrawEllipse(Frame);//, corner, corner);
}

void wxStfGraph::DrawVLine(wxDC* pDC, double x) {
    wxRect WindowRect(GetRect());
    if (isPrinted)
    {   //Set WindowRect to print coordinates (page size)
        WindowRect=printRect;
    }
    pDC->DrawLine(xFormat(x),0,xFormat(x),WindowRect.height);
}

void wxStfGraph::DrawHLine(wxDC* pDC, double y) {
    wxRect WindowRect(GetRect());
    if (isPrinted)
    {   //Set WindowRect to print coordinates (page size)
        WindowRect=printRect;
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

    pDC->DrawLine(
            xFormat(eventIndex),
            20,
            xFormat(eventIndex),
            0
    );
    // arrow head:
    pDC->DrawLine(
            xFormat(eventIndex)-5,
            15,
            xFormat(eventIndex),
            20
    );
    pDC->DrawLine(
            xFormat(eventIndex)+5,
            15,
            xFormat(eventIndex),
            20
    );
}

void wxStfGraph::DrawFit(wxDC* pDC) {

    // go through selected traces:
    if ( isPrinted )
        pDC->SetPen(fitSelectedPrintPen);
    else
        pDC->SetPen(fitSelectedPen);
    for ( std::size_t n_sel = 0; n_sel < Doc()->GetSelectedSections().size(); ++n_sel ) {
        std::size_t sel_index = Doc()->GetSelectedSections()[ n_sel ];
        // Check whether this section contains a fit:
        if ( (*Doc())[Doc()->GetCurCh()][sel_index].IsFitted() && pFrame->PlotSelected() ) {
            PlotFit( pDC, (*Doc())[Doc()->GetCurCh()][sel_index] );
        }
    }

    // go through selected traces:
    if ( isPrinted )
        pDC->SetPen(fitPrintPen);
    else
        pDC->SetPen(fitPen);
    if ( (*Doc())[Doc()->GetCurCh()][Doc()->GetCurSec()].IsFitted() ) {
        PlotFit( pDC, (*Doc())[Doc()->GetCurCh()][Doc()->GetCurSec()] );
    }
}

void wxStfGraph::PlotFit( wxDC* pDC, const Section& Sec ) {

    wxRect WindowRect = GetRect();
    if (isPrinted)
    {   //Set WindowRect to print coordinates (page size)
        WindowRect=printRect;
    }

    int firstPixel = xFormat( Sec.GetStoreFitBeg() );
    if ( firstPixel < 0 ) firstPixel = 0;
    int lastPixel = xFormat( Sec.GetStoreFitEnd() );
    if ( lastPixel > WindowRect.width + 1 ) lastPixel = WindowRect.width + 1;

    if (!isPrinted) {
        //Draw Fit on display
        //For display use point to point drawing
        double fit_time_1 =
            ( ((double)firstPixel - (double)SPX()) / XZ() -
                    (double)Sec.GetStoreFitBeg() )* Doc()->GetXScale();
        for ( int n_px = firstPixel; n_px < lastPixel-1; n_px++ ) {
            // Calculate pixel back to time (GetStoreFitBeg() is t=0)
            double fit_time_2 =
                ( ((double)n_px+1.0 - (double)SPX()) / XZ() -
                        (double)Sec.GetStoreFitBeg() )
                        * Doc()->GetXScale(); // undo xFormat = (int)(toFormat * XZ() + SPX());
            pDC->DrawLine( n_px,
                    yFormat(Sec.GetFitFunc()->func( fit_time_1, Sec.GetBestFitP())),
                            n_px + 1, yFormat(Sec.GetFitFunc()->func(fit_time_2, Sec.GetBestFitP()))
            );
            fit_time_1 = fit_time_2;
        }
    } else {    //Draw Fit for print out
        // For print out use polyline
        std::vector<wxPoint> f_print( lastPixel - firstPixel );
        for ( int n_px = firstPixel; n_px < lastPixel; n_px++ ) {
            // Calculate pixel back to time (GetStoreFitBeg() is t=0)
            double fit_time =
                ( ((double)n_px - (double)SPX()) / XZ() -(double)Sec.GetStoreFitBeg() )
                        * Doc()->GetXScale(); // undo xFormat = (int)(toFormat * XZ() + SPX());
            f_print[n_px-firstPixel].x = n_px;
            f_print[n_px-firstPixel].y = yFormat( Sec.GetFitFunc()->func(
                            fit_time, Sec.GetBestFitP()) );
        }
        pDC->DrawLines( f_print.size(), &f_print[0] );
    }   //End if display or print out
}

void wxStfGraph::DrawIntegral(wxDC* pDC) {
    // Draws a polygon around the integral. Note that the polygon will be drawn
    // out of screen as well.

    if (!isPrinted) {
        pDC->SetPen(scalePen);
    } else {
        pDC->SetPen(scalePrintPen);
    }
    bool even = std::div((int)Doc()->cur().GetStoreIntEnd()-(int)Doc()->cur().GetStoreIntBeg(), 2).rem==0;
    int firstPixel=xFormat(Doc()->cur().GetStoreIntBeg());
    // last pixel:
    int lastPixel= even ? xFormat(Doc()->cur().GetStoreIntEnd()) : xFormat(Doc()->cur().GetStoreIntEnd()-1);
    std::size_t qt_size=
        lastPixel-firstPixel + // part that covers the trace
        2; // straight line through base or 0
    if (!even)
        qt_size++; //straight line for trapezoidal part
    std::vector<wxPoint> quadTrace(qt_size);
    std::size_t n_qt=0;
    quadTrace[n_qt++]=wxPoint(firstPixel,yFormat(Doc()->GetBase()));
    // "Simpson part" (piecewise quadratic functions through three adjacent points):
    for (int n_pixel=firstPixel; n_pixel < lastPixel; ++n_pixel) {
        // (lower) index corresponding to pixel:
        int n_relIndex =
            (int)(((double)n_pixel-(double)SPX())/(double)XZ()-Doc()->cur().GetStoreIntBeg());
        double n_absIndex = ((double)n_pixel-(double)SPX())/(double)XZ();
        // quadratic parameters at this point:
        double a = Doc()->cur().GetQuadP()[(int)(n_relIndex/2)*3];
        double b = Doc()->cur().GetQuadP()[(int)(n_relIndex/2)*3+1];
        double c = Doc()->cur().GetQuadP()[(int)(n_relIndex/2)*3+2];
        double y = a*n_absIndex*n_absIndex + b*n_absIndex + c;

        quadTrace[n_qt++]=wxPoint(n_pixel,yFormat(y));
    }

    // add trapezoidal integration part if uneven:
    if (!even) {
        // draw a straight line:
        quadTrace[n_qt++]=
            wxPoint(
                    xFormat(Doc()->cur().GetStoreIntEnd()),
                    yFormat(Doc()->cur()[Doc()->cur().GetStoreIntEnd()])
            );
    }
    quadTrace[n_qt]=
        wxPoint(
                xFormat(Doc()->cur().GetStoreIntEnd()),
                yFormat(Doc()->GetBase())
        );

    // Polygon from base:
    pDC->SetBrush(baseBrush);
    pDC->DrawPolygon((int)quadTrace.size(),&quadTrace[0]);
    // Polygon from 0:
    quadTrace[0]=wxPoint(firstPixel,yFormat(0));
    quadTrace[n_qt]=
        wxPoint(
                xFormat(Doc()->cur().GetStoreIntEnd()),
                yFormat(0)
        );
    pDC->SetBrush(zeroBrush);
    pDC->DrawPolygon((int)quadTrace.size(),&quadTrace[0]);
    pDC->SetBrush(*wxTRANSPARENT_BRUSH);
}

void wxStfGraph::Exportimage() {
    wxFileDialog bmpDialog(
            ParentFrame(),
            wxT("Save image"),
            wxT(""),
            wxT(""),
            wxT("TIFF image (*.tif)|*.tif|")
            wxT("JPG image (*.jpg)|*.jpg|")
            wxT("PNG image (*.png)|*.png"),
            wxFD_SAVE
    );
    if (bmpDialog.ShowModal()!=wxID_OK) return;
    std::vector<wxString> sizeStr(1);
    sizeStr[0] = wxT("Set width (in pixels):");
    std::vector<double> defaultWidth(1);
    defaultWidth[0]=GetRect().width;
    stf::UserInput(sizeStr,defaultWidth,wxT("Change image size"));
    wxStfUsrDlg sizeDlg(
            ParentFrame(),
            stf::UserInput(sizeStr,defaultWidth,wxT("Change image size"))
    );
    if (sizeDlg.ShowModal()!=wxID_OK) return;

    double prop=(double)GetRect().height/(double)GetRect().width;
    int newWidth=(int)sizeDlg.readInput()[0];
    int newHeight=(int)(prop*newWidth);
    if (newWidth<=0 || newHeight <=0) {
        wxGetApp().ErrorMsg(wxT("Check size settings"));
        return;
    }
    // Create a bitmap:
    wxBitmap tempBmp;
    tempBmp.Create(newWidth,newHeight,-1);
    wxMemoryDC bmpDC;
    bmpDC.SelectObject(tempBmp);
    bmpDC.SetBrush(*wxWHITE_BRUSH);
    bmpDC.SetPen(*wxWHITE_PEN);
    bmpDC.DrawRectangle(0,0,bmpDC.GetSize().x,bmpDC.GetSize().y);
    bmpDC.SetPen(*wxBLACK_PEN);
    wxRect bmpRect(0,0,newWidth,newHeight);
    double scale=(double)bmpRect.width/(double)GetRect().width;
    set_isPrinted(true);
    printScale=scale;
    printRect=bmpRect;
    no_gimmicks=true;
    OnDraw(bmpDC);
    set_isPrinted(false);
    no_gimmicks=false;
    switch (bmpDialog.GetFilterIndex()) {
    case 0:
        tempBmp.SaveFile(bmpDialog.GetPath(),wxBITMAP_TYPE_TIF );
        break;
    case 1:
        tempBmp.SaveFile(bmpDialog.GetPath(),wxBITMAP_TYPE_JPEG );
        break;
    case 2:
        tempBmp.SaveFile(bmpDialog.GetPath(),wxBITMAP_TYPE_PNG );
        break;
    default:
        tempBmp.SaveFile(bmpDialog.GetPath(),wxBITMAP_TYPE_TIF );
        break;
    }
}

void wxStfGraph::Exportps() {
    wxStfPreprintDlg myDlg(this,true);
    if (myDlg.ShowModal()!=wxID_OK) return;

    set_downsampling(myDlg.GetDownSampling());
    set_noGimmicks(true);

    wxFileDialog psFileDialog( this, wxT("Save postscript file"), wxT(""), wxT(""),
            wxT("Postscript file (*.ps)|*.ps"), wxFD_SAVE );
    if (psFileDialog.ShowModal()!=wxID_OK) return;

    // Get size of Graph, in pixels:
    wxRect screenRect(GetRect());

    // Get size of page, in pixels:
    // assuming the screen is ~ 96 dpi, but we want ~ 720:
    printRect=wxRect( 0,0, (int)(GetRect().GetSize().GetWidth()*7.5),
            (int)(GetRect().GetSize().GetHeight()*7.5) );

    // Fit to width or fit to height?
    // If the screenRect's proportion is wider than the printRect's,
    // fit to width:

    double scale=(double)screenRect.width/(double)printRect.width;

    set_isPrinted(true);
    printScale=scale;

    wxPrintData print_d;
    print_d.SetPaperSize(GetRect().GetSize());
    print_d.SetOrientation(wxLANDSCAPE);

    print_d.SetFilename(psFileDialog.GetPath());
    print_d.SetPrintMode(wxPRINT_MODE_FILE);

    wxPostScriptDC psdc(print_d);

    psdc.StartDoc(wxT("printing..."));

    // Tell it where to find the AFM files
    // psdc.GetPrintData().SetFontMetricPath(wxT("afm/"));

    OnDraw(psdc);
    psdc.EndDoc();

    set_isPrinted(false);
    no_gimmicks=false;
}

void wxStfGraph::Exportlatex() {
#if 0
    wxStfPreprintDlg myDlg(this,true);
    if (myDlg.ShowModal()!=wxID_OK) return;

    set_downsampling(myDlg.GetDownSampling());
    set_noGimmicks(true);

    wxFileDialog latexFileDialog(
            this,
            wxT("Save LaTeX file"),
            wxT(""),
            wxT(""),
            wxT("LaTeX file (*.tex)|*.tex"),
            wxFD_SAVE
    );
    if (latexFileDialog.ShowModal()!=wxID_OK) return;

    // Get size of Graph, in pixels:
    wxRect screenRect(GetRect());

    // Get size of page, in pixels:
    // assuming the screen is ~ 96 dpi, but we want ~ 720:
    double upscale = 4.0;
    printRect=wxRect(
            0,0,
            (int)(GetRect().GetSize().GetWidth()*upscale),
            (int)(GetRect().GetSize().GetHeight()*upscale)
    );

    // Fit to width or fit to height?
    // If the screenRect's proportion is wider than the printRect's,
    // fit to width:

    double scale=(double)printRect.width/(double)screenRect.width;

    set_isPrinted(true);
    printScale=scale;

    wxLatexDC latexDC(latexFileDialog.GetPath(),printRect.width,printRect.height,96.0*upscale);
    isLatex = true;
    OnDraw (latexDC);
    isLatex = false;
    set_isPrinted(false);

    no_gimmicks=false;
#endif
}

#if wxCHECK_VERSION(2, 9, 0)
void wxStfGraph::Exportsvg() {
    wxStfPreprintDlg myDlg(this,true);
    if (myDlg.ShowModal()!=wxID_OK) return;

    set_downsampling(myDlg.GetDownSampling());
    set_noGimmicks(true);

    wxFileDialog svgFileDialog(
            this,
            wxT("Save svg file"),
            wxT(""),
            wxT(""),
            wxT("SVG file (*.svg)|*.svg"),
            wxFD_SAVE
    );
    if (svgFileDialog.ShowModal()!=wxID_OK) return;

    // Get size of Graph, in pixels:
    wxRect screenRect(GetRect());

    // Get size of page, in pixels:
    // assuming the screen is ~ 96 dpi, but we want ~ 720:
    double upscale = 7.5;
    printRect=wxRect(
            0,0,
            (int)(GetRect().GetSize().GetWidth()*upscale),
            (int)(GetRect().GetSize().GetHeight()*upscale)
    );

    // Fit to width or fit to height?
    // If the screenRect's proportion is wider than the printRect's,
    // fit to width:

    double scale=(double)printRect.width/(double)screenRect.width;

    set_isPrinted(true);
    printScale=scale;

    wxSVGFileDC svgDC(svgFileDialog.GetPath(),printRect.width,printRect.height,96.0*upscale);
    OnDraw (svgDC);

    set_isPrinted(false);

    no_gimmicks=false;
}
#endif

void wxStfGraph::Snapshot() {
    std::vector<wxString> sizeStr(1);
    sizeStr[0]=wxT("Set width (in pixels):");
    std::vector<double> defaultWidth(1);
    defaultWidth[0]=GetRect().width;
    stf::UserInput(sizeStr,defaultWidth,wxT("Change image size"));
    wxStfUsrDlg sizeDlg(
            ParentFrame(),
            stf::UserInput(sizeStr,defaultWidth,wxT("Change image size"))
    );
    if (sizeDlg.ShowModal()!=wxID_OK) return;

    double prop=(double)GetRect().height/(double)GetRect().width;
    int newWidth=(int)sizeDlg.readInput()[0];
    int newHeight=(int)(prop*newWidth);
    if (newWidth<=0 || newHeight <=0) {
        wxGetApp().ErrorMsg(wxT("Check size settings"));
        return;
    }
    // Create a bitmap:
    wxBitmap tempBmp;
    tempBmp.Create(newWidth,newHeight,-1);
    wxMemoryDC bmpDC;
    bmpDC.SelectObject(tempBmp);
    bmpDC.SetPen(*wxWHITE_PEN);
    bmpDC.SetBrush(*wxWHITE_BRUSH);
    bmpDC.DrawRectangle(0,0,bmpDC.GetSize().x,bmpDC.GetSize().y);
    bmpDC.SetPen(*wxBLACK_PEN);
    wxRect bmpRect(0,0,newWidth,newHeight);
    double scale=(double)bmpRect.width/(double)GetRect().width;
    set_isPrinted(true);
    printScale=scale;
    printRect=bmpRect;
    no_gimmicks=true;
    OnDraw(bmpDC);
    set_isPrinted(false);
    no_gimmicks=false;
    if (wxTheClipboard->Open()) {
        wxTheClipboard->SetData(
                new wxBitmapDataObject(tempBmp)
        );
        wxTheClipboard->Close();
    }
}

#ifdef _WINDOWS
void wxStfGraph::Snapshotwmf() {
    wxStfPreprintDlg myDlg(this,true);
    if (myDlg.ShowModal()!=wxID_OK) return;
    set_downsampling(myDlg.GetDownSampling());

    // Get size of Graph, in pixels:
    wxRect screenRect(GetRect());

    // Get size of page, in pixels:
    // assuming the screen is ~ 96 dpi, but we want ~ 720:
    printRect=(0,0,wxSize(GetRect().GetSize()*4));

    double scale=(double)printRect.width/(double)screenRect.width;

    wxMetafileDC wmfDC;
    if (!wmfDC.IsOk()) {
        wxGetApp().ErrorMsg(wxT("Error while creating clipboard data"));
        return;
    }
    set_noGimmicks(true);
    set_isPrinted(true);
    printScale=scale;
    OnDraw(wmfDC);
    set_isPrinted(false);
    no_gimmicks=false;
    wxMetafile* mf = wmfDC.Close();
    if (mf && mf->IsOk()) {
        mf->SetClipboard();
        delete mf;
    } else {
        wxGetApp().ErrorMsg(wxT("Error while copying to clipboard"));
    }
}
#endif

void wxStfGraph::OnMouseEvent(wxMouseEvent& event) {
    event.Skip();
    if (!view) return;
    if (event.LeftDown()) LButtonDown(event);
    if (event.RightDown()) RButtonDown(event);
    if (event.LeftUp()) LButtonUp(event);
    wxClientDC dc(this);
    PrepareDC(dc);

    wxPoint pt(event.GetLogicalPosition(dc));
}

void wxStfGraph::LButtonDown(wxMouseEvent& event) {
    event.Skip();
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
            wxGetApp().ErrorMsg(
                    wxT("The latency cursor can not be set in the current mode\nChoose manual mode to set the latency cursor manually")
            );
            break;
        }
        Doc()->SetLatencyBeg(((double)lastLDown.x-(double)SPX())/XZ());
        break;
    case stf::zoom_cursor:
        llz_x=(double)lastLDown.x;
        llz_y=(double)lastLDown.y;
        llz_y2=llz_y;
        break;
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
    event.Skip();
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
            wxGetApp().ErrorMsg(
                    wxT("The latency cursor can not be set in the current mode\n \
                    Choose manual mode to set the latency cursor manually")
            );
            break;
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
        if (Doc()->cur().HasEvents()) {
            // store the position that has been clicked:
            eventPos = stf::round( ((double)point.x - (double)SPX())/XZ() );
            PopupMenu(m_eventContext.get());
        } else {
            wxGetApp().ErrorMsg(wxT("No events have been detected yet"));
        }
        break;
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
    event.Skip();
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
    event.Skip();
    wxRect WindowRect(GetRect());
    int kc = event.GetKeyCode();
#ifdef _STFDEBUG
    std::cout << "User pressed " << char(kc) << ", corresponding keycode is " << kc << std::endl;
#endif
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
        OnDown();
        return;
     case WXK_UP:     //up cursor
        OnUp();
        return;
     case 49: //1
         ParentFrame()->SetZoomQual(stf::zoomch1);
         //Child()->SetActiveWindow();
         return;
     case 50:  //2
         if (Doc()->size()>1)
             ParentFrame()->SetZoomQual(stf::zoomch2);
         //Child()->SetActiveWindow();
         return;
     case 51: //3
         if (Doc()->size()>1)
             ParentFrame()->SetZoomQual(stf::zoomboth);
         //Child()->SetActiveWindow();
         return;
     case 69: // e
     case 101:
         ParentFrame()->SetMouseQual(stf::event_cursor);
         //Child()->SetActiveWindow();
         return;
     case 70:
     case 102: // f
         Fittowindow(true);
         //Child()->SetActiveWindow();
         return;
     case 77:  // m
     case 109:
         ParentFrame()->SetMouseQual(stf::measure_cursor);
         //Child()->SetActiveWindow();
         return;
     case 80: // p
     case 112:
         ParentFrame()->SetMouseQual(stf::peak_cursor);
         //Child()->SetActiveWindow();
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
         //Child()->SetActiveWindow();
         return;
     case 68:  // d
     case 100:
         ParentFrame()->SetMouseQual(stf::decay_cursor);
         //Child()->SetActiveWindow();
         return;
     case 90:  // z
     case 122:
         ParentFrame()->SetMouseQual(stf::zoom_cursor);
         //Child()->SetActiveWindow();
         return;
     case 76:  // l
     case 108:
         ParentFrame()->SetMouseQual(stf::latency_cursor);
         //Child()->SetActiveWindow();
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
    llz_y2=(SPY2()-llz_y2)/YZ2();
    ulz_y=(SPY() - ulz_y) / YZ();
    ulz_y2=(SPY2()-ulz_y2)/YZ2();
    YZW()=WindowRect.height/fabs(ulz_y-llz_y);
    YZ2W()=WindowRect.height/fabs(ulz_y2-llz_y2);
    SPYW()=(int)(WindowRect.height + ulz_y * YZ());
    SPY2W()=(int)(WindowRect.height + ulz_y2 * YZ2());
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
        if ((fScaled/fDistance) * pixelDistance > limit)
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
    if (fabs(fstartPosY)>(double)stf::pow2(16))
        SPYW()=0;
    if (fabs(YZ())>1e15)
        YZW()=1.0;

    if (!isPrinted) {
        wxFont font(
                (int)(8*printScale),
                wxFONTFAMILY_SWISS,
                wxFONTSTYLE_NORMAL,
                wxFONTWEIGHT_NORMAL
        );
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
    double pixelDistanceY2=YZ2();
    //real distance (difference) between two neighboured Values:
    double realDistanceY2 = 1.0;
    if ((Doc()->get().size()>1))
    {
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
        if ((Doc()->get().size()>1)) rightDist*=2;
        // Set end points for the scale bar
        Scale[0]=wxPoint(WindowRect.width-rightDist-barLength,
                WindowRect.height-bottomDist);
        Scale[1]=wxPoint(WindowRect.width-rightDist,
                WindowRect.height-bottomDist);
        Scale[2]=wxPoint(WindowRect.width-rightDist,
                WindowRect.height-bottomDist-barLengthY);
        if ((Doc()->get().size()>1))
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
        scaleYString <<  (int)yScaled << wxT(" ") << Doc()->at(Doc()->GetCurCh()).GetYUnits() << wxT("\0");
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
        if ((Doc()->get().size()>1))	{
            wxString scaleYString2;
            scaleYString2 << (int)yScaled2 << wxT(" ")
            << Doc()->at(Doc()->GetSecCh()).GetYUnits();
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
        if ((Doc()->get().size()>1))	{
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
        if ((Doc()->get().size()>1)) {
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
                Doc()->at(Doc()->GetCurCh()).GetYUnits(),
                TextFrame,
                wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL
        );

        // y-Axis of second channel:
        if ((Doc()->get().size()>1)) {
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
            int y2ScaleInWindow=(y2FirstTick-topDist)/barLengthY2;
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
                    Doc()->at(Doc()->GetSecCh()).GetYUnits(),
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
        pDC->DrawLabel(Doc()->GetXUnits(),xTextFrame,wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL);
    }
}

inline int wxStfGraph::xFormat(double toFormat) {
    return (int)(toFormat * XZ() + SPX());
}

inline int wxStfGraph::xFormat(int toFormat) {
    return (int)(toFormat * XZ() + SPX());
}

inline int wxStfGraph::xFormat(std::size_t toFormat) {
    return (int)(toFormat * XZ() + SPX());
}

inline int wxStfGraph::yFormat(double toFormat) {
    return (int)(SPY() - toFormat * YZ());
}

inline int wxStfGraph::yFormat(int toFormat) {
    return (int)(SPY() - toFormat * YZ());
}

inline int wxStfGraph::yFormat2(double toFormat) {
    return (int)(SPY2() - toFormat * YZ2());
}

inline int wxStfGraph::yFormat2(int toFormat){
    return (int)(SPY2() - toFormat * YZ2());
}


void wxStfGraph::Fittowindow(bool refresh)
{
    const double screen_part=0.5; //part of the window to be filled
    std::size_t points=Doc()->cur().size();
    if (points==0) {
        wxGetApp().ErrorMsg(wxT("Array of size zero in wxGraph::Fittowindow()"));
        return;
    }
    double min=Doc()->cur().get().min();
    if (min>1.0e12)  min= 1.0e12;
    if (min<-1.0e12) min=-1.0e12;
    double max=Doc()->cur().get().max();
    if (max>1.0e12)  max= 1.0e12;
    if (max<-1.0e12) max=-1.0e12;
    wxRect WindowRect(GetRect());
    switch (ParentFrame()->GetZoomQual())
    {	//Depending on the zoom radio buttons (Mouse field)
    //in the (trace navigator) control box
    case stf::zoomboth:
        if(!(Doc()->get().size()>1))
        {
            wxGetApp().ErrorMsg(wxT("No second channel available! Choose Channel 1!"));
            return;
        }
        //Fit to window Ch2
        FitToWindowSecCh(false);
        //Fit to window Ch1
        XZW()=(double)WindowRect.width /points;
        YZW()=(WindowRect.height/fabs(max-min))*screen_part;
        SPYW()=(int)(((screen_part+1)/2)*WindowRect.height
                + min * YZ());
        SPXW()=0;
        break;
    case stf::zoomch2:
        //ErrorMsg if no second channel available
        if(!(Doc()->get().size()>1))
        {
            wxGetApp().ErrorMsg(wxT("No second channel available! Choose Channel 1!"));
            return;
        }
        //Fit to window Ch2
        FitToWindowSecCh(false);
        break;
    default:
        //ErrorMsg if no second channel available
        //			Invalidate();
        //Fit to window Ch1
        XZW()=(double)WindowRect.width /points;
        YZW()=(WindowRect.height/fabs(max-min))*screen_part;
        SPYW()=(int)(((screen_part+1)/2)*WindowRect.height
                + min * YZ());
        SPXW()=0;
        break;
    }
    if (refresh) Refresh();
}

void wxStfGraph::FitToWindowSecCh(bool refresh)
{

    //Get coordinates of the main window
    wxRect WindowRect(GetRect());

    const double screen_part=0.5; //part of the window to be filled
    std::size_t secCh=Doc()->GetSecCh();
#undef min
#undef max
    double min=Doc()->get()[secCh][Doc()->GetCurSec()].get().min();
    double max=Doc()->get()[secCh][Doc()->GetCurSec()].get().max();
    YZ2W()=(WindowRect.height/fabs(max-min))*screen_part;
    SPY2W()=(int)(((screen_part+1)/2)*WindowRect.height
            + min * YZ2());
    if (refresh) Refresh();
}	//End FitToWindowSecCh()

void wxStfGraph::OnPrevious() {
    if (Doc()->get()[Doc()->GetCurCh()].size()==1) return;
    std::size_t curSection=Doc()->GetCurSec();
    if (Doc()->GetCurSec() > 0) curSection--;
    else curSection=Doc()->get()[Doc()->GetCurCh()].size()-1;
    Doc()->SetSection(curSection);
    wxGetApp().OnPeakcalcexecMsg();
    pFrame->SetCurTrace(curSection);
    Refresh();
}

void wxStfGraph::OnFirst() {
    if (Doc()->GetCurSec()==0) return;
    Doc()->SetSection(0);
    wxGetApp().OnPeakcalcexecMsg();
    pFrame->SetCurTrace(0);
    Refresh();
}

void wxStfGraph::OnLast() {
    if (Doc()->GetCurSec()==Doc()->get()[Doc()->GetCurCh()].size()-1) return;
    std::size_t curSection=Doc()->get()[Doc()->GetCurCh()].size()-1;
    Doc()->SetSection(curSection);
    wxGetApp().OnPeakcalcexecMsg();
    pFrame->SetCurTrace(curSection);
    Refresh();
}

void wxStfGraph::OnNext() {
    if (Doc()->get()[Doc()->GetCurCh()].size()==1) return;
    std::size_t curSection=Doc()->GetCurSec();
    if (curSection < Doc()->get()[Doc()->GetCurCh()].size()-1) curSection++;
    else curSection=0;
    Doc()->SetSection(curSection);
    wxGetApp().OnPeakcalcexecMsg();
    pFrame->SetCurTrace(curSection);
    Refresh();
}

void wxStfGraph::OnUp() {
    switch (ParentFrame()->GetZoomQual())
    {	//Depending on the zoom radio buttons (Mouse field)
    //in the (trace navigator) control box
    case stf::zoomboth:
        //ErrorMsg if no second channel available
        //yZooms of Ch1 are performed keeping the base constant
        SPYW()=SPY() - 20;
        if(!(Doc()->get().size()>1)) break;
        //Ymove of Ch2 is performed
        SPY2W()=SPY2() - 20;
        break;
    case stf::zoomch2:
        if(!(Doc()->get().size()>1)) break;
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
        if(!(Doc()->get().size()>1)) break;
        //Ymove of Ch2 is performed
        SPY2W()=SPY2() + 20;
        break;
    case stf::zoomch2:
        if(!(Doc()->get().size()>1)) break;
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
        if(!(Doc()->get().size()>1)) break;
        //yZooms of Ch2 are performed keeping the base constant
        SPY2W()=(int)(SPY2()
                + Doc()->GetBase() * (YZ2() * factor
                        - YZ2()));
        YZ2W()=YZ2() * factor;
        break;
    case stf::zoomch2:
        if(!(Doc()->get().size()>1)) break;
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
    if ((Doc()->get().size()>1)) {
        double base2=0.0;
        try {
            double var2=0.0;
            base2=stf::base(var2,Doc()->get()[Doc()->GetSecCh()][Doc()->GetCurSec()].get(),
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
    if ((Doc()->get().size()>1)) {
        SPY2W()=SPY();
        Refresh();
    }
}

void wxStfGraph::Ch2zoom() {
    if ((Doc()->get().size()>1)) {
        YZ2W()=YZ();
        Refresh();
    }
}

void wxStfGraph::Ch2basezoom() {
    if ((Doc()->get().size()>1)) {
        // Adjust y-scale without refreshing:
        YZ2W()=YZ();
        // Adjust baseline:
        double base2=0.0;
        try {
            double var2=0.0;
            base2=stf::base(var2,Doc()->get()[Doc()->GetSecCh()][Doc()->GetCurSec()].get(),
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
