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

// printout.cpp
// Controls printing of traces.
// 2007-12-27, Christoph Schmidt-Hieber, University of Freiburg

#include <wx/wxprec.h>

#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#include <wx/print.h>

#include "./app.h"
#include "./view.h"
#include "./parentframe.h"
#include "./doc.h"
#include "./printout.h"
#include "./graph.h"

extern wxStfParentFrame* frame;

wxStfPrintout::wxStfPrintout(const wxChar *title) :
wxPrintout(title) ,
store_noGimmicks(false)
{
    wxStfDoc* pDoc = wxGetApp().GetActiveDoc();
    if (!pDoc) {
        wxGetApp().ErrorMsg(wxT("Null pointer (pDoc) in wxStfPrintout::wxStfPrintout()\nAborting printing"));
        return;
    }
    wxStfView* pView = (wxStfView*)pDoc->GetFirstView();
    if (!pView) {
        wxGetApp().ErrorMsg(wxT("Null pointer (pView) in wxStfPrintout::wxStfPrintout()\nAborting printing"));
        return;
    }
    wxStfGraph* pGraph = pView->GetGraph();
    if (!pGraph) {
        wxGetApp().ErrorMsg(wxT("Null pointer (pGraph) in wxStfPrintout::wxStfPrintout()\nAborting printing"));
        return;
    }
    store_noGimmicks=wxGetApp().GetActiveView()->GetGraph()->get_noGimmicks();
}

bool wxStfPrintout::OnPrintPage(int WXUNUSED(page))
{
    wxDC* dc=GetDC();
    if (dc)
    {
        DrawPageOne();

        return true;
    }
    else
        return false;
}

bool wxStfPrintout::OnBeginDocument(int startPage, int endPage)
{
    if (!wxPrintout::OnBeginDocument(startPage, endPage))
        return false;

    return true;
}

void wxStfPrintout::GetPageInfo(int *minPage, int *maxPage, int *selPageFrom, int *selPageTo)
{
    *minPage = 1;
    *maxPage = 1;
    *selPageFrom = 1;
    *selPageTo = 1;
}

bool wxStfPrintout::HasPage(int pageNum)
{
    return (pageNum == 1);
}

void wxStfPrintout::DrawPageOne()
{
    int x,y;
    GetPPIPrinter(&x,&y);
    // Get size of Graph, in pixels:
    wxStfDoc* pDoc = wxGetApp().GetActiveDoc();
    if (!pDoc) {
        wxGetApp().ErrorMsg(wxT("Null pointer (pDoc) in wxStfPrintout::DrawPageOne()\nAborting printing"));
        return;
    }
    wxStfView* pView = (wxStfView*)pDoc->GetFirstView();
    if (!pView) {
        wxGetApp().ErrorMsg(wxT("Null pointer (pView) in wxStfPrintout::DrawPageOne()\nAborting printing"));
        return;
    }
    wxStfGraph* pGraph = pView->GetGraph();
    if (!pGraph) {
        wxGetApp().ErrorMsg(wxT("Null pointer (pGraph) in wxStfPrintout::DrawPageOne()\nAborting printing"));
        return;
    }
    wxRect screenRect(pGraph->GetRect());
    // Get size of page, in pixels:
    wxRect printRect=GetLogicalPageMarginsRect(*(frame->GetPageSetup()));

    // A first guess at the scale:
    double hScale=(double)printRect.height/(double)screenRect.height;
    double headerSizeY=0.0;
    // Space needed for the header:
    if (!store_noGimmicks) {
        headerSizeY=30.0*hScale;
    } else {
        pGraph->set_noGimmicks(true);
    }

    // Fit to width or fit to height?
    // If the screenRect's proportion is wider than the printRect's,
    // fit to width:
    double scale=1.0;
    wxRect propPrintRect(printRect);
    double prop=(double)screenRect.width/(double)screenRect.height;
    if (prop > (printRect.height-headerSizeY)/printRect.width) {
        scale=(double)printRect.width/(double)(screenRect.width);
        // keep width:
        propPrintRect.height=(int)((double)propPrintRect.width/prop);
    } else {
        scale=(double)(printRect.height-headerSizeY)/(double)(screenRect.height);
        propPrintRect.width=(int)((double)propPrintRect.height*prop);
    }
    // maximal extent of the Graph on paper:
    wxCoord maxX = (int)((double)(screenRect.width)*scale);
    wxCoord maxY = (int)((double)(screenRect.height)*scale);

    wxCoord xoff =(printRect.width - maxX) / 2.0;
    wxCoord yoff =(printRect.height - maxY) / 2.0;
#ifdef __WXGTK__
    xoff = -printRect.width*0.8;
    OffsetLogicalOrigin(xoff, 0);
    xoff = 0;
#endif

    pGraph->set_isPrinted(true);

    pGraph->set_printScale(scale);
    // construct a rectangle with the same proportion as the graph on screen:
    pGraph->set_printRect(propPrintRect);

    if (!store_noGimmicks) {
        PrintHeader(GetDC(),hScale);
    }

    // create a font that looks similar to the screen font:
    wxFont font( (int)(6.0 * (double)x/72.0), wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL );
    GetDC()->SetFont(font);

    OffsetLogicalOrigin(xoff, (int)(yoff+headerSizeY));
    pGraph->OnDraw(*GetDC());
    pGraph->set_isPrinted(false);
}

void wxStfPrintout::PrintHeader(wxDC* pDC, double scale) {
    int ppiX,ppiY;
    GetPPIPrinter(&ppiX,&ppiY);
    double resScale = ppiX / 72.0;
#ifdef _WINDOWS
    int fontScale=(int)(6.0 * resScale);
#else
    int fontScale=(int)(10.0 * resScale);
#endif
    int xstart=0;
    int ystart=0;

    // create a font that looks similar to the screen font:
    wxFont font( fontScale, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD );
    GetDC()->SetFont(font);

    // File name and section number:
    wxString description;
    description << Doc()->GetFilename()
        << wxT(", Trace ") << (int)Doc()->GetCurSecIndex()+1
        << wxT(" of ") << (int)Doc()->get()[Doc()->GetCurChIndex()].size();
    pDC->DrawText(description,xstart,ystart);

    // Results:
    stfnum::Table table(Doc()->CurResultsTable());

    font.SetWeight(wxFONTWEIGHT_NORMAL);
    pDC->SetFont(font);

    int xpos=xstart;
    for (std::size_t nRow=0;nRow<1;/*table.nRows()*/++nRow) {
        // row label:
        for (std::size_t nCol=0;nCol<table.nCols();++nCol) {
            int colSize=(int)(40.0*resScale+table.GetColLabel(nCol).length()*4.0*resScale);
            if (nRow==0) pDC->DrawText(stf::std2wx(table.GetColLabel(nCol)),xpos,(int)(14.0*resScale)+ystart);
            if (!table.IsEmpty(nRow,nCol)) {
                wxString entry; entry << table.at(nRow,nCol);
                pDC->DrawText(entry,xpos,(int)(24.0*resScale)+ystart);
            }
            xpos+=colSize;
        }
    }
    try {
        stf::SectionAttributes sec_attr = Doc()->GetCurrentSectionAttributes();
        if (sec_attr.isFitted) {
            wxRect WindowRect(GetLogicalPageMarginsRect(*(frame->GetPageSetup())));
            int increment=WindowRect.height/50;
            int yPos=(int)(WindowRect.height*0.5);
            int xPos=(int)(WindowRect.width*0.75);
            // print fit info line by line:
            for (std::size_t n=0;n < sec_attr.bestFit.nRows();++n) {
                pDC->DrawText(stf::std2wx(sec_attr.bestFit.GetRowLabel(n)),xPos,yPos);
                wxString value;
                value << sec_attr.bestFit.at(n,0);
                pDC->DrawText(value,(int)(xPos+40.0*resScale),yPos);
                yPos+=increment;
            }
        }
    } catch (const std::out_of_range& e) {
        
    }
}
