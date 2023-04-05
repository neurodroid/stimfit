#include "wx/wxprec.h"

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif


#include "./../../stf.h"
#include "./eventdlg.h"
#include "./../app.h"

enum {
    wxID_COMBOTEMPLATES,
    wxID_CRITERIA,
    wxDETECTIONCLEMENTS,
    wxDETECTIONJONAS,
    wxDETECTIONPERNIA
};

BEGIN_EVENT_TABLE( wxStfEventDlg, wxDialog )
EVT_RADIOBUTTON( wxDETECTIONCLEMENTS, wxStfEventDlg::OnClements )
EVT_RADIOBUTTON( wxDETECTIONJONAS,   wxStfEventDlg::OnJonas )
EVT_RADIOBUTTON( wxDETECTIONPERNIA,  wxStfEventDlg::OnPernia )
END_EVENT_TABLE()

wxStfEventDlg::wxStfEventDlg(wxWindow* parent, const std::vector<stf::SectionPointer>& templateSections,
                             bool isExtract_, int id, wxString title, wxPoint pos, wxSize size, int style) :
wxDialog( parent, id, title, pos, size, style ), m_threshold(4.0), m_mode(stf::criterion),
    isExtract(isExtract_), m_minDistance(150), m_template(-1)
{
    wxBoxSizer* topSizer;
    topSizer = new wxBoxSizer( wxVERTICAL );

    wxFlexGridSizer* templateSizer = new wxFlexGridSizer(2,1,0,0);
    wxStaticText* staticTextTempl =
        new wxStaticText( this, wxID_ANY, wxT("Select template fit from section:"),
                          wxDefaultPosition, wxDefaultSize, 0 );
    templateSizer->Add( staticTextTempl, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    wxArrayString templateNames;
    templateNames.Alloc(templateSections.size());
    int max_w = 0;
    for (std::size_t n_templ=0;n_templ<templateSections.size();++n_templ) {
        if (templateSections[n_templ].pSection != NULL) {
            wxString sec_desc = stf::std2wx(templateSections[n_templ].pSection->GetSectionDescription());
            if (sec_desc.empty()) {
                sec_desc = wxT("Section ");
                sec_desc << n_templ;
            }
            int w, h;
            GetTextExtent( sec_desc, &w, &h );
            if ( w > max_w )
                max_w = w;
            templateNames.Add( sec_desc );
        }
    }
    m_comboBoxTemplates =
        new wxComboBox( this, wxID_COMBOTEMPLATES, wxT("1"), wxDefaultPosition,
                        wxSize( max_w + 36, 24 ), templateNames, wxCB_DROPDOWN | wxCB_READONLY );
    if (templateSections.size()>0) {
        m_comboBoxTemplates->SetSelection(0);
    }
    templateSizer->Add( m_comboBoxTemplates, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    topSizer->Add( templateSizer, 0, wxALIGN_CENTER | wxALL, 5 );

    if (isExtract) {
        wxFlexGridSizer* gridSizer;
        gridSizer=new wxFlexGridSizer(2,2,0,0);

        wxStaticText* staticTextThr;
        staticTextThr =
            new wxStaticText( this, wxID_CRITERIA, wxT("Threshold:"), wxDefaultPosition, wxDefaultSize, 0 );
        gridSizer->Add( staticTextThr, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
        wxString def; def << m_threshold;
        m_textCtrlThr =
            new wxTextCtrl( this, wxID_ANY, def, wxDefaultPosition, wxSize(40,20), wxTE_RIGHT );
        gridSizer->Add( m_textCtrlThr, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

        wxStaticText* staticTextDist;
        staticTextDist =
            new wxStaticText( this, wxID_ANY, wxT("Min. distance between events (# points):"),
                wxDefaultPosition, wxDefaultSize, 0 );
        gridSizer->Add( staticTextDist, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
        wxString def2; def2 << m_minDistance;
        m_textCtrlDist =
            new wxTextCtrl( this, wxID_ANY, def2, wxDefaultPosition, wxSize(40,20), wxTE_RIGHT );
        gridSizer->Add( m_textCtrlDist, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

        topSizer->Add( gridSizer, 0, wxALIGN_CENTER | wxALL, 5 );

        //*** Radio options for event detection methods ***//
        m_radioBox = new wxStaticBoxSizer( wxVERTICAL, this, wxT("Detection method"));

        wxString m_radioBoxChoices[] = {
                wxT("Template scaling (Clements && Bekkers)"),
                wxT("Correlation coefficient (Jonas et al.)"),
                wxT("Deconvolution (Pernia-Andrade et al.)")
        };


        wxRadioButton* wxRadioClements = new wxRadioButton(this, wxDETECTIONCLEMENTS, m_radioBoxChoices[0],
            wxDefaultPosition, wxDefaultSize, wxRB_GROUP);

        wxRadioButton* wxRadioJonas = new wxRadioButton(this, wxDETECTIONJONAS, m_radioBoxChoices[1],
            wxDefaultPosition, wxDefaultSize);

        wxRadioButton* wxRadioPernia = new wxRadioButton(this, wxDETECTIONPERNIA, m_radioBoxChoices[2],
            wxDefaultPosition, wxDefaultSize);

        m_radioBox->Add(wxRadioClements, 0, wxALIGN_LEFT | wxALL, 2);
        m_radioBox->Add(wxRadioJonas,   0, wxALIGN_LEFT | wxALL, 2);
        m_radioBox->Add(wxRadioPernia,  0, wxALIGN_LEFT | wxALL, 2);
        //m_radioBox->SetSelection(0);

        topSizer->Add( m_radioBox, 0, wxALIGN_CENTER | wxALL, 5 );
    }

    m_sdbSizer = new wxStdDialogButtonSizer();
    m_sdbSizer->AddButton( new wxButton( this, wxID_OK ) );
    m_sdbSizer->AddButton( new wxButton( this, wxID_CANCEL ) );
    m_sdbSizer->Realize();
    topSizer->Add( m_sdbSizer, 0, wxALIGN_CENTER | wxALL, 5 );

    topSizer->SetSizeHints(this);
    this->SetSizer( topSizer );

    this->Layout();
}

void wxStfEventDlg::EndModal(int retCode) {
    wxCommandEvent unusedEvent;
    // similar to overriding OnOK in MFC (I hope...)
    switch( retCode) {

    case wxID_OK:
        if (!OnOK()) {
            wxLogMessage(wxT("Select a detection method"));
            return;
        }
        break;

    case wxID_CANCEL:
        break;

    default:
        return;
    }
    wxDialog::EndModal(retCode);
}

bool wxStfEventDlg::OnOK() {
    // Read template:
    m_template = m_comboBoxTemplates->GetCurrentSelection();
    if (m_template<0) {
        wxLogMessage(wxT("Please select a valid template"));
        return false;
    }
    if (isExtract) {
        // Read entry to string:
        m_textCtrlThr->GetValue().ToDouble( &m_threshold );
        long tempLong;
        m_textCtrlDist->GetValue().ToLong( &tempLong );
        m_minDistance = (int)tempLong;

        wxRadioButton* wxRadioClements = (wxRadioButton*)FindWindow(wxDETECTIONCLEMENTS);
        wxRadioButton* wxRadioJonas = (wxRadioButton*)FindWindow(wxDETECTIONJONAS);
        wxRadioButton* wxRadioPernia = (wxRadioButton*)FindWindow(wxDETECTIONPERNIA);

        if ( wxRadioClements->GetValue() ) 
            m_mode = stf::criterion;
        else if ( wxRadioJonas->GetValue() ) 
            m_mode = stf::correlation;
        else if ( wxRadioPernia->GetValue() )
            m_mode = stf::deconvolution;
        else
            return false;

        
        /*switch (m_radioBox->GetSelection()) {
         case 0:
             m_mode = stf::criterion;
             break;
         case 1:
             m_mode = stf::correlation;
             break;
         case 2:
             m_mode = stf::deconvolution;
             break;
        }*/
        if (m_mode==stf::correlation && (m_threshold<0 || m_threshold>1)) {
            wxLogMessage(wxT("Please select a value between 0 and 1 for the correlation coefficient"));
            return false;
        }
    }
    return true;
}

void wxStfEventDlg::OnClements( wxCommandEvent& event) {
    event.Skip();
    
    wxRadioButton* wxRadioClements = (wxRadioButton*)FindWindow(wxDETECTIONCLEMENTS);
    wxStaticText* staticTextThr = (wxStaticText*)FindWindow(wxID_CRITERIA);

    if (wxRadioClements == NULL || staticTextThr == NULL){
        wxGetApp().ErrorMsg(wxT("Null pointer in wxStfEvenDlg::OnClements()"));
        return;
    }
    staticTextThr->SetLabel(wxT("Threshold:"));
    
}

void wxStfEventDlg::OnJonas( wxCommandEvent& event) {
    event.Skip();
    
    wxRadioButton* wxRadioJonas = (wxRadioButton*)FindWindow(wxDETECTIONJONAS);
    wxStaticText* staticTextThr = (wxStaticText*)FindWindow(wxID_CRITERIA);

    if (wxRadioJonas == NULL || staticTextThr == NULL){
        wxGetApp().ErrorMsg(wxT("Null pointer in wxStfEvenDlg::OnJonas()"));
        return;
    }
    staticTextThr->SetLabel(wxT("Correlation:"));
    
}

void wxStfEventDlg::OnPernia( wxCommandEvent& event) {
    event.Skip();
    
    wxRadioButton* wxRadioPernia = (wxRadioButton*)FindWindow(wxDETECTIONPERNIA);
    wxStaticText* staticTextThr = (wxStaticText*)FindWindow(wxID_CRITERIA);

    if (wxRadioPernia == NULL || staticTextThr == NULL){
        wxGetApp().ErrorMsg(wxT("Null pointer in wxStfEvenDlg::OnPernia()"));
        return;
    }
    staticTextThr->SetLabel( wxT("Standard deviation:") );
    
}
