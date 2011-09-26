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

// checkbox.cpp
// Derived from wxCheckBox to pipe keyboard input to the graph
// 2007-12-27, Christoph Schmidt-Hieber, University of Freiburg

#include "wx/wxprec.h"

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include "wx/checkbox.h"

#include "./app.h"
#include "./doc.h"
#include "./view.h"
#include "./graph.h"
#include "./stfcheckbox.h"

IMPLEMENT_CLASS(wxStfCheckBox, wxCheckBox)

BEGIN_EVENT_TABLE(wxStfCheckBox, wxCheckBox)
    EVT_KEY_DOWN( wxStfCheckBox::OnKeyDown )
    EVT_MOUSE_EVENTS( wxStfCheckBox::OnStfClicked )
END_EVENT_TABLE()

wxStfCheckBox::wxStfCheckBox(
    wxWindow* parent,
    wxWindowID id,
    const wxString& label,
    stfio::Event* pEvent,
    const wxPoint& pos,
    const wxSize& size,
    long style,
    const wxValidator& val,
    const wxString& name) : wxCheckBox(parent,id,label,pos,size,style,val,name), m_pEvent(pEvent)
{
    
}
    
void wxStfCheckBox::OnKeyDown(wxKeyEvent& event) {
    // Do nothing here:
    event.Skip();
    // pipe the key input to the graph:
    ((wxStfGraph*)GetParent())->OnKeyDown(event);
}

void wxStfCheckBox::OnStfClicked(wxMouseEvent& event) {
    // Toggle discard status of event:
    event.Skip();
    if (event.LeftDown())
        m_pEvent->ToggleStatus();
}

