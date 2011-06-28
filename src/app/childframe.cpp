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

// frame.cpp
// These are the top-level and child windows of the application.
// 2007-12-27, Christoph Schmidt-Hieber, University of Freiburg

#ifdef _STFDEBUG
#include <iostream>
#endif

// For compilers that support precompilation, includes "wx/wx.h".
#include <wx/wxprec.h>
#include <wx/grid.h>
#include <wx/artprov.h>
#include <wx/printdlg.h>
#include <wx/file.h>
#include <wx/filename.h>
#include <wx/progdlg.h>
#include <wx/splitter.h>
#include <wx/choicdlg.h>
#include <wx/aboutdlg.h>

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#if !wxUSE_DOC_VIEW_ARCHITECTURE
#error You must set wxUSE_DOC_VIEW_ARCHITECTURE to 1 in setup.h!
#endif

#if !wxUSE_MDI_ARCHITECTURE
#error You must set wxUSE_MDI_ARCHITECTURE to 1 in setup.h!
#endif

#include "wx/spinctrl.h"
#include "./app.h"
#include "./doc.h"
#include "./view.h"
#include "./graph.h"
#include "./table.h"
#include "./printout.h"
#include "./dlgs/smalldlgs.h"
#include "./copygrid.h"
#ifdef _WINDOWS
#include "./../core/filelib/atflib.h"
#include "./../core/filelib/igorlib.h"
#endif

#include "./childframe.h"

IMPLEMENT_CLASS(wxStfChildFrame, wxStfChildType)

BEGIN_EVENT_TABLE(wxStfChildFrame, wxStfChildType)
EVT_SPINCTRL( ID_SPINCTRLTRACES, wxStfChildFrame::OnSpinCtrlTraces )
EVT_COMBOBOX( ID_COMBOACTCHANNEL, wxStfChildFrame::OnComboActChannel )
EVT_COMBOBOX( ID_COMBOINACTCHANNEL, wxStfChildFrame::OnComboInactChannel )
EVT_CHECKBOX( ID_ZERO_INDEX, wxStfChildFrame::OnZeroIndex)
EVT_CHECKBOX( ID_PLOTSELECTED, wxStfChildFrame::OnShowselected )
// workaround for status bar:
EVT_MENU_HIGHLIGHT_ALL( wxStfChildFrame::OnMenuHighlight )
END_EVENT_TABLE()

wxStfChildFrame::wxStfChildFrame(wxDocument* doc, wxView* view, wxStfParentType* parent,
                                 wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size,
                                 long style, const wxString& name) :
wxStfChildType(doc,view,parent,id,title,pos,size,style,name), m_parent(parent),
    m_notebook(NULL)
{
    m_mgr.SetManagedWindow(this);
    m_mgr.SetFlags( wxAUI_MGR_ALLOW_FLOATING | wxAUI_MGR_TRANSPARENT_DRAG |
                    wxAUI_MGR_VENETIAN_BLINDS_HINT | wxAUI_MGR_ALLOW_ACTIVE_PANE );
}

wxStfChildFrame::~wxStfChildFrame() {
    // deinitialize the frame manager
    m_mgr.UnInit();
}

wxStfGrid* wxStfChildFrame::CreateTable() {
    // create the notebook off-window to avoid flicker
    wxSize client_size = GetClientSize();

    wxStfGrid* ctrl = new wxStfGrid( this, wxID_ANY,
                                     wxDefaultPosition, wxDefaultSize,
                                     wxVSCROLL | wxHSCROLL );
#ifndef __APPLE__
    wxFont font( 10, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL );
    ctrl->SetDefaultCellFont(font);
#endif
    ctrl->SetDefaultColSize(108);
    ctrl->SetColLabelSize(20);
    ctrl->SetDefaultCellAlignment(wxALIGN_RIGHT,wxALIGN_CENTRE);
    ctrl->CreateGrid(3,10);

    ctrl->EnableEditing(false);
    return ctrl;
}

wxAuiNotebook* wxStfChildFrame::CreateNotebook() {
    // create the notebook off-window to avoid flicker
    wxSize client_size = GetClientSize();
    m_notebook_style =
        wxAUI_NB_SCROLL_BUTTONS |
        wxAUI_NB_CLOSE_ON_ACTIVE_TAB |/*wxAUI_NB_DEFAULT_STYLE | */
        wxNO_BORDER;

    wxAuiNotebook* ctrl = new wxAuiNotebook( this, wxID_ANY,
                                             wxPoint(client_size.x, client_size.y), wxSize(200,200),
                                             m_notebook_style );

    return ctrl;
}

wxPanel* wxStfChildFrame::CreateTraceCounter() {
    wxSize client_size = GetClientSize();
    wxPanel* ctrl = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize
                                 );// , wxSize(165,88) );

    return ctrl;
}

wxPanel* wxStfChildFrame::CreateChannelCounter() {
    wxPanel* ctrl = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize
                                 ); //, wxSize(256,88) );


    return ctrl;
}

void wxStfChildFrame::CreateMenuTraces(const std::size_t value) {
    sizemax = value;

    m_traceCounter = CreateTraceCounter(); // this is wxPanel

    wxBoxSizer* pTracesBoxSizer; // top-level Sizer
    pTracesBoxSizer = new wxBoxSizer(wxVERTICAL);

    wxGridSizer* TracesGridSizer; // top-level GridSizer
    TracesGridSizer = new wxGridSizer(3,1,0,0);

    // Grid for spin control
    wxFlexGridSizer* pSpinCtrlTraceSizer;
    pSpinCtrlTraceSizer = new wxFlexGridSizer(1,3,0,0); // 1 row, 3 columns for the SpinCtrl + text

    // 1) the wxSpinCtrl object 
    trace_spinctrl = new wxSpinCtrl( m_traceCounter, ID_SPINCTRLTRACES, wxEmptyString, wxDefaultPosition,
                     wxSize(64, wxDefaultCoord), wxSP_WRAP);

    // the "of n", where n is the number of traces
    // n is zero-based in zero-based check box is selected
    wxStaticText* pIndexText;
    pIndexText = new wxStaticText(m_traceCounter, wxID_ANY, wxT("Index: "));
    pSize=new wxStaticText( m_traceCounter, wxID_ANY, wxEmptyString);
    wxString sizeStr;

    pSpinCtrlTraceSizer->Add( pIndexText,     0, wxALIGN_CENTER_VERTICAL  | wxALL, 1) ;
    pSpinCtrlTraceSizer->Add( trace_spinctrl, 0, wxALIGN_LEFT  | wxALL, 1) ;
    pSpinCtrlTraceSizer->Add( pSize,          0, wxALIGN_LEFT  | wxALIGN_CENTER | wxALL, 1) ;

    // 2) Show zero-based index? Read from Stimfit registry
    pZeroIndex = new wxCheckBox( m_traceCounter, ID_ZERO_INDEX, wxT("Zero-based index ") );
    pZeroIndex->SetValue(wxGetApp().wxGetProfileInt(wxT("Settings"), wxT("Zeroindex"), 0));

    // If true set the starting value to zero
    if (pZeroIndex->GetValue()){
        sizemax--;
        trace_spinctrl->SetValue(0);
        trace_spinctrl->SetRange(0, (int)sizemax);
    }
    else {
        trace_spinctrl->SetValue(1);
        trace_spinctrl->SetRange(1, (int)sizemax);
    
    }

    sizeStr << wxT("of ") << wxString::Format(wxT("%3d"),(int)sizemax);
    pSize->SetLabel(sizeStr);
    // Show selected
    pShowSelected = new wxCheckBox( m_traceCounter, ID_PLOTSELECTED, wxT("Show selected       "));
    pShowSelected->SetValue(false);

    // Add everything to top-level GridSizer
    TracesGridSizer->Add(pSpinCtrlTraceSizer, 0, wxALIGN_LEFT | wxALIGN_TOP    | wxALL, 3);
    TracesGridSizer->Add(pZeroIndex,          0, wxALIGN_LEFT | wxALIGN_BOTTOM | wxALL, 3);
    TracesGridSizer->Add(pShowSelected,       0, wxALIGN_LEFT | wxALIGN_BOTTOM | wxALL, 3);

    pTracesBoxSizer->Add(TracesGridSizer, 0, wxALIGN_CENTER | wxALL, 1);

    pTracesBoxSizer->SetSizeHints(m_traceCounter);
    m_traceCounter->SetSizer( TracesGridSizer );
    m_traceCounter->Layout();
    wxSize size = m_traceCounter->GetSize();
    wxStfDoc* pDoc=(wxStfDoc*)GetDocument();
    m_mgr.AddPane( m_traceCounter, wxAuiPaneInfo().Caption(wxT("Trace selection")).Fixed().BestSize(size.x, size.y).
                   Position(pDoc->size()-1).CloseButton(false).Floatable().Dock().Top().Name(wxT("SelectionT")) );
    m_table=CreateTable();

    m_mgr.AddPane( m_table, wxAuiPaneInfo().Caption(wxT("Results")).Position(pDoc->size()).
                   CloseButton(false).Floatable().Dock().Top().Name(wxT("Results")) );
    m_mgr.Update();
    Refresh();
}
// Channel Selection childframe
void wxStfChildFrame::CreateComboChannels(const wxArrayString& channelStrings) {

    m_channelCounter = CreateChannelCounter();
    
    wxBoxSizer* pChannelsBoxSizer; // top-level Sizer
    pChannelsBoxSizer = new wxBoxSizer(wxVERTICAL);

    wxGridSizer* ChannelsGridSizer; // top-level GridSizer
    ChannelsGridSizer = new wxGridSizer(3,1,0,0);

    // Grid for Active comboBox
    wxBoxSizer* pComboActSizer;
    pComboActSizer = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* pActIndex  = new wxStaticText( m_channelCounter, wxID_ANY, wxT("Active channel:        ") );

    pActChannel = new wxComboBox( m_channelCounter, ID_COMBOACTCHANNEL, wxT("0"),
                                  wxDefaultPosition, wxSize(92, wxDefaultCoord), channelStrings, wxCB_DROPDOWN | wxCB_READONLY );

    pComboActSizer->Add( pActIndex,   0,  wxALIGN_CENTER_VERTICAL| wxALIGN_LEFT,  1);
    pComboActSizer->Add( pActChannel, 0,  wxALIGN_CENTER_VERTICAL| wxALIGN_RIGHT, 1);

    
    // Grid for reference comboBox
    wxBoxSizer* pComboRefSizer;
    pComboRefSizer = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* pInactIndex = new wxStaticText( m_channelCounter, wxID_ANY, wxT("Reference channel: ") );
    pInactIndex->SetForegroundColour( *wxRED );

    pInactChannel = new wxComboBox( m_channelCounter, ID_COMBOINACTCHANNEL, wxT("1"),
                                    wxDefaultPosition, wxSize(92,wxDefaultCoord), channelStrings, wxCB_DROPDOWN | wxCB_READONLY );

    pComboRefSizer->Add( pInactIndex,   0,  wxALIGN_CENTER_VERTICAL | wxALIGN_LEFT , 1);
    pComboRefSizer->Add( pInactChannel, 0,  wxALIGN_CENTER_VERTICAL | wxALIGN_RIGHT, 1 );
    

    wxBoxSizer *pShowChannelSizer;
    pShowChannelSizer = new wxBoxSizer(wxHORIZONTAL);

    // Checkbox to hide reference channel:
    pShowSecond = new wxCheckBox( m_channelCounter, ID_PLOTSELECTED, wxT("Show reference") );
    pShowSecond->SetForegroundColour( *wxRED );
    pShowSecond->SetValue(true);
    
    pShowAll = new wxCheckBox( m_channelCounter, ID_PLOTSELECTED, wxT("Show all  ") );
    pShowAll->SetValue(false);
    pShowChannelSizer->Add( pShowAll );
    pShowChannelSizer->Add( pShowSecond );
    
    // Add everything to top-level GridSizer
    ChannelsGridSizer->Add(pComboActSizer,    0, wxALIGN_LEFT | wxALIGN_TOP    | wxALL, 3);
    ChannelsGridSizer->Add(pComboRefSizer,    0, wxALIGN_LEFT | wxALIGN_BOTTOM | wxALL, 3);
    ChannelsGridSizer->Add(pShowChannelSizer, 0, wxALIGN_LEFT | wxALIGN_BOTTOM | wxALL, 3);
    

    pChannelsBoxSizer->Add(ChannelsGridSizer, 0, wxALIGN_CENTER | wxALL, 1);

    pChannelsBoxSizer->SetSizeHints(m_channelCounter);

    m_channelCounter->SetSizer( ChannelsGridSizer );
    m_channelCounter->Layout();
    wxSize size = m_channelCounter->GetSize();
    m_mgr.AddPane( m_channelCounter, wxAuiPaneInfo().Caption(wxT("Channel selection")).Fixed().BestSize(size.x, size.y).
                   Position(0).CloseButton(false).Floatable().Dock().Top().Name(wxT("SelectionC")) );
    m_mgr.Update();

    Refresh();
}
// Trace selection childframe
void wxStfChildFrame::SetSelected(std::size_t value) {
    wxString selStr;
    selStr << wxT("Show ") << wxString::Format(wxT("%3d"),(int)value) << wxT(" selected");

    pShowSelected->SetLabel(selStr);
}

void wxStfChildFrame::SetChannels( std::size_t act, std::size_t inact ) {
    pActChannel->SetSelection( act );
    pInactChannel->SetSelection( inact );
}

std::size_t wxStfChildFrame::GetCurTrace() const {

    // if zero-based is True
    if ( pZeroIndex->GetValue() )
        return trace_spinctrl->GetValue();
    else 
        return trace_spinctrl->GetValue()-1;
}

void wxStfChildFrame::SetCurTrace(std::size_t n) {

    // if zero-based is True
    if ( pZeroIndex->GetValue() )
        trace_spinctrl->SetValue((int)n);
    else
        trace_spinctrl->SetValue((int)n+1);
}

void wxStfChildFrame::OnSpinCtrlTraces( wxSpinEvent& event ){
    event.Skip();


    wxStfView* pView=(wxStfView*)GetView();
    wxStfDoc* pDoc=(wxStfDoc*)GetDocument();

    if (pDoc == NULL || pView == NULL) {
        wxGetApp().ErrorMsg(wxT("Null pointer in wxStfChildFrame::OnSpinCtrlTraces()"));
        return;
    }

    pDoc->SetSection(GetCurTrace()); 
    wxGetApp().OnPeakcalcexecMsg();

    if (pView->GetGraph() != NULL) {
        pView->GetGraph()->Refresh();
        pView->GetGraph()->Enable();
        pView->GetGraph()->SetFocus();
    }
}

void wxStfChildFrame::OnActivate(wxActivateEvent &event) {
    wxStfView* pView=(wxStfView*)GetView();
    if (pView)
        pView->Activate(true);
}

void wxStfChildFrame::OnComboActChannel(wxCommandEvent& WXUNUSED(event)) {
    if ( pActChannel->GetCurrentSelection() == pInactChannel->GetCurrentSelection()) {
        // correct selection:
        for (int n_c=0;n_c<(int)pActChannel->GetCount();++n_c) {
            if (n_c!=pActChannel->GetCurrentSelection()) {
                pInactChannel->SetSelection(n_c);
                break;
            }
        }
    }

    UpdateChannels();
}

void wxStfChildFrame::OnComboInactChannel(wxCommandEvent& WXUNUSED(event)) {
    if (pInactChannel->GetCurrentSelection()==pActChannel->GetCurrentSelection()) {
        // correct selection:
        for (int n_c=0;n_c<(int)pInactChannel->GetCount();++n_c) {
            if (n_c!=pInactChannel->GetCurrentSelection()) {
                pActChannel->SetSelection(n_c);
                break;
            }
        }
    }

    UpdateChannels();
}

void wxStfChildFrame::UpdateChannels( ) {

    wxStfDoc* pDoc=(wxStfDoc*)GetDocument();

    if ( pDoc != NULL && pDoc->size() > 1) {
        try {
            if (pActChannel->GetCurrentSelection() >= 0 ||
                pActChannel->GetCurrentSelection() <  (int)pDoc->size())
            {
                pDoc->SetCurCh( pActChannel->GetCurrentSelection() );
                if (pInactChannel->GetCurrentSelection() >= 0 ||
                    pInactChannel->GetCurrentSelection() <  (int)pDoc->size())
                {
                    pDoc->SetSecCh( pInactChannel->GetCurrentSelection() );
                } else {
                    pDoc->SetCurCh(0);
                    pDoc->SetSecCh(1);
                }
            } else {
                pDoc->SetCurCh(0);
                pDoc->SetSecCh(1);
            }
        }
        catch (const std::out_of_range& e) {
            wxString msg(wxT("Error while changing channels\nPlease close file\n"));
            msg += wxString( e.what(), wxConvLocal );
            wxGetApp().ExceptMsg(msg);
            return;
        }

        // Update measurements:
        wxGetApp().OnPeakcalcexecMsg();
        UpdateResults();
        wxStfView* pView=(wxStfView*)GetView();
        if ( pView == NULL ) {
            wxGetApp().ErrorMsg( wxT("View is zero in wxStfDoc::SwapChannels"));
            return;
        }
        if (pView->GetGraph() != NULL) {
            pView->GetGraph()->Refresh();
            pView->GetGraph()->Enable();
            pView->GetGraph()->SetFocus();
        }
    }
}

void wxStfChildFrame::OnZeroIndex( wxCommandEvent& event) {
    event.Skip();
    
    wxSpinCtrl* pTraceCtrl = (wxSpinCtrl*)FindWindow(ID_SPINCTRLTRACES);
    wxCheckBox* pZeroIndex = (wxCheckBox*)FindWindow(ID_ZERO_INDEX);

    if (pZeroIndex == NULL || pTraceCtrl == NULL){
        wxGetApp().ErrorMsg(wxT("Null pointer in wxStfChildFrame::OnZeroIndex"));
        return;
    }
    
    // If  Zero-index is ON (selected) 
    if (pZeroIndex->GetValue()){
        wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("Zeroindex"), 1); // write config
        if (pTraceCtrl->GetValue()==1){
            sizemax--;
            pTraceCtrl->SetRange(0, sizemax); // first set new range
            pTraceCtrl->SetValue(pTraceCtrl->GetValue()-1); // now you can move one less 
        }
        else if (pTraceCtrl->GetValue()==(int)sizemax){
            sizemax--;
            pTraceCtrl->SetValue(pTraceCtrl->GetValue()-1); // move one less
            pTraceCtrl->SetRange(0, sizemax); // next set new range
        }
        else {
            sizemax--;
            pTraceCtrl->SetRange(0, sizemax); // first set new range
            pTraceCtrl->SetValue(pTraceCtrl->GetValue()-1); // now you can move one less 
        }
        
    }
    // If Zero-index is OFF (unselected) 
    else {
        wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("Zeroindex"), 0); 
        if (pTraceCtrl->GetValue()==0){
            sizemax++;
            pTraceCtrl->SetValue(pTraceCtrl->GetValue()+1); 
            pTraceCtrl->SetRange(1, (int)sizemax); 
        }
        else if (pTraceCtrl->GetValue()==(int)sizemax){
            sizemax++;
            pTraceCtrl->SetRange(1, (int)sizemax); // first set new range
            pTraceCtrl->SetValue(pTraceCtrl->GetValue()+1); // now you can move one more 
        }
        else { // now the order does not matter
            sizemax++;
            pTraceCtrl->SetRange(1, (int)sizemax); // first set new range
            pTraceCtrl->SetValue(pTraceCtrl->GetValue()+1); // now you can move one more 
        }
        
    }

    //wxString sizeStr;
    //sizeStr << wxT("of ") << wxString::Format(wxT("%3d"),(int)sizemax);
    //pSize->SetLabel(sizeStr);
}

void wxStfChildFrame::OnShowselected(wxCommandEvent& WXUNUSED(event)) {
    wxStfView* pView=(wxStfView*)GetView();
    if (pView != NULL && pView->GetGraph()!= NULL) { 
        pView->GetGraph()->Refresh();
        pView->GetGraph()->Enable();
        pView->GetGraph()->SetFocus();
    }
}

void wxStfChildFrame::ActivateGraph() {
    wxStfView* pView=(wxStfView*)GetView();
    // Set the focus somewhere else:
    if (m_traceCounter != NULL) 
        m_traceCounter->SetFocus();
    if (pView != NULL && pView->GetGraph()!= NULL) { 
        pView->GetGraph()->Enable();
        pView->GetGraph()->SetFocus();
    }
}

void wxStfChildFrame::ShowTable(const stf::Table &table,const wxString& caption) {

    // Create and show notebook if necessary:
    if (m_notebook==NULL && !m_mgr.GetPane(m_notebook).IsOk()) {
        m_notebook=CreateNotebook();
        m_mgr.AddPane( m_notebook, wxAuiPaneInfo().Caption(wxT("Analysis results")).
                       Floatable().Dock().Left().Name( wxT("Notebook") ) );
    } else {
        // Re-open notebook if it has been closed:
        if (!m_mgr.GetPane(m_notebook).IsShown()) {
            m_mgr.GetPane(m_notebook).Show();
        }
    }
    wxStfGrid* pGrid = new wxStfGrid( m_notebook, wxID_ANY, wxPoint(0,20), wxDefaultSize );
    wxStfTable* pTable(new wxStfTable(table));
    pGrid->SetTable(pTable,true); // the grid will take care of the deletion
    pGrid->SetEditable(false);
    pGrid->SetDefaultCellAlignment(wxALIGN_RIGHT,wxALIGN_CENTRE);
    for (std::size_t n_row=0; n_row<=table.nRows()+1; ++n_row) {
        pGrid->SetCellAlignment(wxALIGN_LEFT,(int)n_row,0);
    }
    m_notebook->AddPage( pGrid, caption, true );

    // "commit" all changes made to wxAuiManager
    m_mgr.Update();
    wxStfView* pView=(wxStfView*)GetView();
    if (pView != NULL && pView->GetGraph()!= NULL) { 
        pView->GetGraph()->Enable();
        pView->GetGraph()->SetFocus();
    }
}

void wxStfChildFrame::UpdateResults() {
    wxStfDoc* pDoc=(wxStfDoc*)GetDocument();
    stf::Table table(pDoc->CurResultsTable());
    
    // Delete or append columns:
    if (m_table->GetNumberCols()<(int)table.nCols()) {
        m_table->AppendCols((int)table.nCols()-(int)m_table->GetNumberCols());
    } else {
        if (m_table->GetNumberCols()>(int)table.nCols()) {
            m_table->DeleteCols(0,(int)m_table->GetNumberCols()-(int)table.nCols());
        }
    }

    // Delete or append row:
    if (m_table->GetNumberRows()<(int)table.nRows()) {
        m_table->AppendRows((int)table.nRows()-(int)m_table->GetNumberRows());
    } else {
        if (m_table->GetNumberRows()>(int)table.nRows()) {
            m_table->DeleteRows(0,(int)m_table->GetNumberRows()-(int)table.nRows());
        }
    }

    for (std::size_t nRow=0;nRow<table.nRows();++nRow) {
        // set row label:
        m_table->SetRowLabelValue((int)nRow,table.GetRowLabel(nRow));
        for (std::size_t nCol=0;nCol<table.nCols();++nCol) {
            if (nRow==0) m_table->SetColLabelValue((int)nCol,table.GetColLabel(nCol));
            if (!table.IsEmpty(nRow,nCol)) {
                wxString entry; entry << table.at(nRow,nCol);
                m_table->SetCellValue((int)nRow,(int)nCol,entry);
            } else {
                m_table->SetCellValue((int)nRow,(int)nCol,wxT("n.a."));
            }
        }
    }
}

void wxStfChildFrame::Saveperspective() {
    wxString perspective = m_mgr.SavePerspective();
    // Save to wxConfig:
    wxGetApp().wxWriteProfileString(wxT("Settings"),wxT("Windows"),perspective);
#ifdef _STFDEBUG
    wxFile persp(wxT("perspective.txt"), wxFile::write);
    persp.Write(perspective);
    persp.Close();
#endif
}

void wxStfChildFrame::Loadperspective() {
    wxString perspective = wxGetApp().wxGetProfileString(wxT("Settings"),wxT("Windows"),wxT(""));
    if (perspective!=wxT("")) {
        m_mgr.LoadPerspective(perspective);
    } else {
        wxGetApp().ErrorMsg(wxT("Couldn't find saved windows settings"));
    }
}


void wxStfChildFrame::Restoreperspective() {
    m_mgr.LoadPerspective(defaultPersp);
    m_mgr.Update();
}

void wxStfChildFrame::OnMenuHighlight(wxMenuEvent& event) {
    if (this->GetMenuBar()) {
        wxMenuItem *item = this->GetMenuBar()->FindItem(event.GetId());
        if(item) {
            wxLogStatus(item->GetHelp());
        }
    }
    event.Skip();

}

#if wxUSE_DRAG_AND_DROP
bool wxStfFileDrop::OnDropFiles(wxCoord WXUNUSED(x), wxCoord WXUNUSED(y), const wxArrayString& filenames) {
    int nFiles=(int)filenames.GetCount();
    if (nFiles>0) {
        return wxGetApp().OpenFileSeries(filenames);
    } else {
        return false;
    }
}
#endif
