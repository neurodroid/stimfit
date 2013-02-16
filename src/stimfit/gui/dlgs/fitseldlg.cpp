#include "wx/wxprec.h"

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include "./../app.h"
#include "./../doc.h"
#include "./../view.h"
#include "./../graph.h"
#include "./fitseldlg.h"

#define wxID_LIST    1001
#define wxID_PREVIEW 1002

BEGIN_EVENT_TABLE( wxStfFitSelDlg, wxDialog )
EVT_LIST_ITEM_SELECTED( wxID_LIST, wxStfFitSelDlg::OnListItemSelected )
EVT_BUTTON( wxID_PREVIEW, wxStfFitSelDlg::OnButtonClick )
END_EVENT_TABLE()

wxStfFitSelDlg::wxStfFitSelDlg(wxWindow* parent, wxStfDoc* doc, int id, wxString title, wxPoint pos,
                               wxSize size, int style)
: wxDialog( parent, id, title, pos, size, style ),
    m_fselect(18), init_p(0), opts(6), noInput(false), use_scaling(false),
    paramDescArray(MAXPAR),
    paramEntryArray(MAXPAR), pDoc(doc)
{
    // Respectively the scale factor for initial \mu,
    // stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2,
    // maxIter, maxPass
    opts[0]=5*1E-3; //default: 1E-03;
    opts[1]=1E-17; //default: 1E-17;
    opts[2]=1E-17; //default: 1E-17;
    opts[3]=1E-32; //default: 1E-17;
    opts[4]=64; //default: 64;
    opts[5]=16;

    wxBoxSizer* topSizer;
    topSizer = new wxBoxSizer( wxVERTICAL );

    // 2-column sizer for funcs (left) and settings (right)
    wxFlexGridSizer* mainGrid = new wxFlexGridSizer(1,2,0,5);

    wxStaticBoxSizer* m_listSizer = new wxStaticBoxSizer(
        wxVERTICAL, this, wxT("Available functions") );

    m_listCtrl = new wxListCtrl( this, wxID_LIST, wxDefaultPosition, wxSize(550,300),
            wxLC_LIST );
    int n_f = 0;
    for (c_stfunc_it cit = wxGetApp().GetFuncLib().begin(); cit != wxGetApp().GetFuncLib().end(); cit++) {
        wxString funcName;
        funcName << wxString::Format(wxT("%2d: "), n_f) << stf::std2wx(cit->name);
        m_listCtrl->InsertItem( n_f++, funcName );
    }

    m_listSizer->Add( m_listCtrl, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 2 );
    mainGrid->Add( m_listSizer, 0, wxALIGN_CENTER_HORIZONTAL, 2 );

    // vertical sizer for initial parameters (top) and options (bottom)
    wxBoxSizer* settingsSizer;
    settingsSizer=new wxBoxSizer(wxVERTICAL);
    wxStaticBoxSizer* paramSizer = new wxStaticBoxSizer(
        wxVERTICAL, this, wxT("Initial parameters") );

    // grid for parameters:
    wxFlexGridSizer* paramGrid;
    paramGrid=new wxFlexGridSizer(0,4,0,4);

    // add parameter boxes:
    std::vector< wxStaticText* >::iterator it1;
    std::vector< wxTextCtrl* >::iterator it2 = paramEntryArray.begin();
    for (it1 = paramDescArray.begin();
         it1 != paramDescArray.end() && it2 != paramEntryArray.end();
         it1++) {
        *it1 = new wxStaticText( this, wxID_ANY, wxT(" "), wxDefaultPosition,
                wxSize(74,20), wxTE_LEFT );
        paramGrid->Add( *it1, 0, wxALIGN_CENTER_VERTICAL | wxALL, 2 );
        *it2 = new wxTextCtrl( this, wxID_ANY, wxT(" "), wxDefaultPosition,
                wxSize(74,20), wxTE_RIGHT );
        paramGrid->SetFlexibleDirection(wxHORIZONTAL);
        paramGrid->Add( *it2, 0, wxALIGN_CENTER_VERTICAL | wxALL, 2 );
        it2++;
    }

    //settingsSizer->Add( paramGrid, 0, wxALIGN_CENTER_HORIZONTAL, 2 );
    paramSizer->Add( paramGrid, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 2 );
    settingsSizer->Add( paramSizer, 0, wxALIGN_LEFT | wxALIGN_TOP | wxALL, 2 );

    // Fit options:
    // grid for parameters:
    wxFlexGridSizer* optionsGrid;
    optionsGrid=new wxFlexGridSizer(opts.size()+1, 2, 0, 0);

    wxStaticBoxSizer* fitoptSizer = new wxStaticBoxSizer(
        wxVERTICAL, this, wxT("Fitting options") );

    InitOptions(optionsGrid);
    // add the options grid to the settings sizer:
    
    fitoptSizer->Add( optionsGrid, 0, wxEXPAND | wxALL, 2 );
    settingsSizer->Add( fitoptSizer, 0, wxALIGN_CENTER_HORIZONTAL | wxALIGN_BOTTOM, 2 );
    //settingsSizer->Add( optionsGrid, 0, wxALIGN_CENTER_HORIZONTAL | wxALIGN_BOTTOM, 2 );
    // add the settings sizer to the main grid:

    mainGrid->Add( settingsSizer, 0, wxALIGN_CENTER_HORIZONTAL, 2 );
    // add the main grid to the dialog:
    topSizer->Add( mainGrid, 0, wxALIGN_CENTER_HORIZONTAL| wxALL, 5 );

    // Ok / Cancel / Preview:
    wxButton* previewButton;
    previewButton = new wxButton( this, wxID_PREVIEW, wxT("Preview"), wxDefaultPosition,
            wxDefaultSize, 0 );
    topSizer->Add( previewButton, 0, wxALIGN_CENTER | wxALL, 2 );

    m_sdbSizer = new wxStdDialogButtonSizer();
    m_sdbSizer->AddButton( new wxButton( this, wxID_OK ) );
    m_sdbSizer->AddButton( new wxButton( this, wxID_CANCEL ) );
    m_sdbSizer->Realize();
    topSizer->Add( m_sdbSizer, 0, wxALIGN_CENTER| wxALL, 2 );
    topSizer->SetSizeHints(this);
    this->SetSizer( topSizer );

    this->Layout();
    // select first function:
    if (m_listCtrl->GetItemCount()>0) {
        m_listCtrl->SetItemState(0,wxLIST_STATE_SELECTED,wxLIST_STATE_SELECTED);
    }
}

void wxStfFitSelDlg::EndModal(int retCode) {
    // similar to overriding OnOK in MFC (I hope...)
    switch (retCode) {
    case wxID_OK:
        if (!OnOK()) {
            wxLogMessage(wxT("Please select a valid function"));
            return;
        }
        break;
     case wxID_CANCEL:
         try {
             pDoc->DeleteFit(pDoc->GetCurCh(), pDoc->GetCurSec());
         } catch (const std::out_of_range& e) {

         }
        break;
    default:
        ;
    }
    wxDialog::EndModal(retCode);
}

bool wxStfFitSelDlg::OnOK() {
    Update_fselect();
    read_init_p();
    read_opts();
//    wxStfDoc* pDoc=pDoc;
//    pDoc->cur().SetIsFitted(false);
//    pDoc->cur().SetFit(Vector_double(0));
    return true;
}

void wxStfFitSelDlg::InitOptions(wxFlexGridSizer* optionsGrid) {
    // Number of passes--------------------------------------------------
    wxStaticText* staticTextNPasses;
    staticTextNPasses=new wxStaticText( this, wxID_ANY, wxT("Max. number of passes:"),
            wxDefaultPosition, wxDefaultSize, 0 );
    optionsGrid->Add( staticTextNPasses, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    wxString strNPasses; strNPasses << opts[5];
    m_textCtrlMaxpasses = new wxTextCtrl( this, wxID_ANY, strNPasses,
            wxDefaultPosition, wxSize(74,20), wxTE_RIGHT );
    optionsGrid->Add( m_textCtrlMaxpasses, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    // Number of iterations----------------------------------------------
    wxStaticText* staticTextNIter;
    staticTextNIter=new wxStaticText( this, wxID_ANY, wxT("Max. number of iterations per pass:"),
            wxDefaultPosition, wxDefaultSize, 0 );
    optionsGrid->Add( staticTextNIter, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    wxString strNIter; strNIter << opts[4];
    m_textCtrlMaxiter=new wxTextCtrl( this, wxID_ANY, strNIter,
            wxDefaultPosition, wxSize(74,20), wxTE_RIGHT );
    optionsGrid->Add( m_textCtrlMaxiter, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    // Initial scaling factor--------------------------------------------
    wxStaticText* staticTextMu;
    staticTextMu=new wxStaticText( this, wxID_ANY, wxT("Initial scaling factor:"),
            wxDefaultPosition, wxDefaultSize, 0 );
    optionsGrid->Add( staticTextMu, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    wxString strMu; strMu << opts[0];
    m_textCtrlMu=new wxTextCtrl( this, wxID_ANY, strMu, wxDefaultPosition, wxSize(74,20),
            wxTE_RIGHT );
    optionsGrid->Add( m_textCtrlMu, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    // Gradient of squared error-----------------------------------
    wxStaticText* staticTextJTE;
    staticTextJTE=new wxStaticText( this, wxID_ANY, wxT("Stop. thresh. for gradient of squared error:"),
            wxDefaultPosition, wxDefaultSize, 0 );
    optionsGrid->Add( staticTextJTE, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    wxString strJTE; strJTE << opts[1];
    m_textCtrlJTE=new wxTextCtrl( this, wxID_ANY, strJTE, wxDefaultPosition,
            wxSize(74,20), wxTE_RIGHT );
    optionsGrid->Add( m_textCtrlJTE, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    // Parameter gradient------------------------------------------------
    wxStaticText* staticTextDP;
    staticTextDP=new wxStaticText( this, wxID_ANY, wxT("Stop. thresh. for rel. parameter change:"),
            wxDefaultPosition, wxDefaultSize, 0 );
    optionsGrid->Add( staticTextDP, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    wxString strDP; strDP << opts[2];
    m_textCtrlDP=new wxTextCtrl( this, wxID_ANY, strDP, wxDefaultPosition,
            wxSize(74,20), wxTE_RIGHT );
    optionsGrid->Add( m_textCtrlDP, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    // Squared error-----------------------------------------------------
    wxStaticText* staticTextE2;
    staticTextE2=new wxStaticText( this, wxID_ANY, wxT("Stop. thresh. for squared error:"),
            wxDefaultPosition, wxDefaultSize, 0 );
    optionsGrid->Add( staticTextE2, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    wxString strE2; strE2 << opts[3];
    m_textCtrlE2=new wxTextCtrl( this, wxID_ANY, strE2, wxDefaultPosition,
            wxSize(74,20), wxTE_RIGHT );
    optionsGrid->Add( m_textCtrlE2, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    // Use scaling-------------------------------------------------------
    m_checkBox = new wxCheckBox(this, wxID_ANY, wxT("Scale data amplitude to 1.0"),
                                         wxDefaultPosition, wxDefaultSize, 0); 
    m_checkBox->SetValue(true);
    optionsGrid->Add( m_checkBox, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    
}

void wxStfFitSelDlg::OnButtonClick( wxCommandEvent& event ) {
    event.Skip();
    // Make sure we are up-to-date:
    Update_fselect();
    // read in parameters:
    read_init_p();
    // tell the document that a fit has been performed:
    if (pDoc==0) {
        wxGetApp().ErrorMsg(wxT("Couldn't connect to document"));
        return;
    }
    // calculate a graph from the current parameters:
    std::size_t fitSize=
        pDoc->GetFitEnd()-pDoc->GetFitBeg();
    Vector_double fit(fitSize);
    for (std::size_t n_f=0;n_f<fit.size();++n_f) {
        try {
            fit[n_f]=
                wxGetApp().GetFuncLib().at(m_fselect).func(
                        pDoc->GetXScale()*n_f,init_p
                );
        }
        catch (const std::out_of_range& e) {
            wxString msg(wxT("Could not retrieve selected function from library:\n"));
            msg += wxString( e.what(), wxConvLocal );
            wxGetApp().ExceptMsg(msg);
            m_fselect=-1;
            return;
        }
    }
    try {
        pDoc->SetIsFitted(pDoc->GetCurCh(), pDoc->GetCurSec(), init_p,
                          wxGetApp().GetFuncLibPtr(m_fselect), 0,
                          pDoc->GetFitBeg(), pDoc->GetFitEnd() );
    } catch (const std::out_of_range& e) {
        
    }
    // tell the view to draw the fit:
    wxStfView* pView = (wxStfView*)pDoc->GetFirstView();
    if (pView != NULL)
        if (pView->GetGraph() != NULL)
            pView->GetGraph()->Refresh();
}

void wxStfFitSelDlg::SetPars() {
    Update_fselect();
    // get parameter names from selected function:
    try {
        // fill a temporary array:
        if (pDoc==NULL) return;
        std::size_t fitSize=
            pDoc->GetFitEnd()-pDoc->GetFitBeg();
        if (fitSize<=0) {
            wxGetApp().ErrorMsg(wxT("Check fit cursor settings"));
            return;
        }
        Vector_double x(fitSize);
        //fill array:
        std::copy(&pDoc->cur()[pDoc->GetFitBeg()],
                &pDoc->cur()[pDoc->GetFitBeg()+fitSize],
                &x[0]);
        Vector_double initPars(wxGetApp().GetFuncLib().at(m_fselect).pInfo.size());
        wxGetApp().GetFuncLib().at(m_fselect).init( x, pDoc->GetBase(),
                pDoc->GetPeak(), pDoc->GetXScale(), initPars);
        std::vector< wxStaticText* >::iterator it1;
        std::vector< wxTextCtrl* >::iterator it2 = paramEntryArray.begin();
        std::size_t n_p = 0;
        for (it1 = paramDescArray.begin();
             it1 != paramDescArray.end() && it2 != paramEntryArray.end();
             it1++) {
            if (n_p < wxGetApp().GetFuncLib().at(m_fselect).pInfo.size()) {
                (*it1)->Show();
                (*it2)->Show();
                // Parameter label:
                (*it1)->SetLabel(stf::std2wx(wxGetApp().GetFuncLib().at(m_fselect).pInfo[n_p].desc));
                // Initial parameter values:
                wxString strInit; strInit << initPars[n_p];
                (*it2)->SetValue(strInit);
                (*it2)->Enable(!noInput);
            } else {
                (*it1)->Show(false);
                (*it2)->Show(false);
            }
            it2++;
            n_p++;
        }
    }
    catch (const std::out_of_range& e) {
        wxString msg(wxT("Could not retrieve selected function from library:\n"));
        msg += wxString( e.what(), wxConvLocal );
        wxLogMessage(msg);
        m_fselect = -1;
        return;
    }
    this->Layout();
}

void wxStfFitSelDlg::Update_fselect() {
    // Update currently selected function:
    if (m_listCtrl->GetSelectedItemCount()>0) {
        // Get first selected item:
        long item = -1;
        item=m_listCtrl->GetNextItem(item,wxLIST_NEXT_ALL,wxLIST_STATE_SELECTED);
        if (item==-1) return;
        m_fselect = item;
    }
}

void wxStfFitSelDlg::OnListItemSelected( wxListEvent& event ) {
    event.Skip();
    SetPars();
}

void wxStfFitSelDlg::read_init_p() {
    init_p.resize(wxGetApp().GetFuncLib().at(m_fselect).pInfo.size());
    for (std::size_t n_p=0;n_p<init_p.size();++n_p) {
        wxString entryInit = paramEntryArray[n_p]->GetValue();
        entryInit.ToDouble( &init_p[n_p] );
    }
}

void wxStfFitSelDlg::read_opts() {
    // Read entry to string:
    wxString entryMu = m_textCtrlMu->GetValue();
    entryMu.ToDouble( &opts[0] );
    wxString entryJTE = m_textCtrlJTE->GetValue();
    entryJTE.ToDouble( &opts[1] );
    wxString entryDP = m_textCtrlDP->GetValue();
    entryDP.ToDouble( &opts[2] );
    wxString entryE2 = m_textCtrlE2->GetValue();
    entryE2.ToDouble( &opts[3] );
    wxString entryMaxiter = m_textCtrlMaxiter->GetValue();
    entryMaxiter.ToDouble( &opts[4] );
    wxString entryMaxpasses = m_textCtrlMaxpasses->GetValue();
    entryMaxpasses.ToDouble( &opts[5] );

    use_scaling = m_checkBox->GetValue();
}
