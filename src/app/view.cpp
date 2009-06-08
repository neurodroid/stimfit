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

// view.cpp
// Derived from wxView to satisfy the doc/view architecture.
// Doesn't do a lot in stimfit.
// 2007-12-27, Christoph Schmidt-Hieber, University of Freiburg

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include "wx/filename.h"

#if !wxUSE_DOC_VIEW_ARCHITECTURE
#error You must set wxUSE_DOC_VIEW_ARCHITECTURE to 1 in setup.h!
#endif

#include "./app.h"
#include "./doc.h"
#include "./view.h"
#include "./parentframe.h"
#include "./childframe.h"
#include "./graph.h"
#include "./dlgs/cursorsdlg.h"

IMPLEMENT_DYNAMIC_CLASS(wxStfView, wxView)

BEGIN_EVENT_TABLE(wxStfView, wxView)
END_EVENT_TABLE()

extern wxStfParentFrame* frame;

wxStfView::wxStfView() :
    graph((wxStfGraph *) NULL),
    childFrame((wxStfChildFrame *) NULL)
{
}

// What to do when a view is created. Creates actual
// windows for displaying the view.
bool wxStfView::OnCreate(wxDocument *doc, long WXUNUSED(flags) )
{
    childFrame = wxGetApp().CreateChildFrame(doc, this);
    if (childFrame==NULL) {
        return false;
    }
    // extract file name:
    wxFileName fn(doc->GetFilename());
    childFrame->SetTitle(fn.GetName());
    graph = GetMainFrame()->CreateGraph(this, childFrame);
    if (graph==NULL) {
        return false;
    }
    childFrame->GetMgr()->AddPane( graph, wxAuiPaneInfo().Caption(wxT("Traces")).Name(wxT("Traces")).CaptionVisible(true).
            CloseButton(false).Centre().PaneBorder(true)  );
    childFrame->GetMgr()->Update();

    // childFrame->ActivateGraph();
#if defined(__X__) || defined(__WXMAC__)
    // X seems to require a forced resize
    // childFrame->SetClientSize(800,600);
#endif
    childFrame->Show(true);
    Activate(true);
    return true;
}

wxStfDoc* wxStfView::Doc() {
    return (wxStfDoc*)GetDocument();
}

// Sneakily gets used for default print/preview
// as well as drawing on the screen.
void wxStfView::OnDraw(wxDC *WXUNUSED(pDC)) {

}

void wxStfView::OnUpdate(wxView *WXUNUSED(sender), wxObject *WXUNUSED(hint))
{
    if (graph) {
        graph->Refresh();
    }
}

// Clean up windows used for displaying the view.
bool wxStfView::OnClose(bool deleteWindow)
{
	if ( !GetDocument()->Close() )
        return false;

    Activate(false);

    if ( deleteWindow )
        wxDELETE(childFrame);

    SetFrame(NULL);

    return true;

}

void wxStfView::OnActivateView(bool activate, wxView *activeView, wxView *deactiveView) {
    //this function will be called whenever the view is activated
    if (activate) {
        if (wxGetApp().GetCursorsDialog()!=NULL && wxGetApp().GetCursorsDialog()->IsShown()) {
            wxGetApp().GetCursorsDialog()->SetActiveDoc(Doc());
            try {
                wxGetApp().GetCursorsDialog()->UpdateCursors();
            }
            catch (const std::runtime_error& e) {
                wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
            }
        }

        // Update menu checks:
        Doc()->UpdateMenuCheckmarks();
        frame->SetSingleChannel(Doc()->size()<2);
        if (graph)
            graph->SetFocus();
    }
    wxView::OnActivateView(activate,activeView,deactiveView);
}
