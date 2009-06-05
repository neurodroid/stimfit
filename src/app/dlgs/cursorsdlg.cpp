#include "wx/wxprec.h"

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include "./../app.h"
#include "./cursorsdlg.h"
#include "./../doc.h"

enum {
    wxCOMBOUM,
    wxCOMBOU1P,
    wxCOMBOU2P,
    wxCOMBOU1B,
    wxCOMBOU2B,
    wxCOMBOU1D,
    wxCOMBOU2D,
    wxTEXTM,
    wxTEXT1P,
    wxTEXT2P,
    wxTEXT1B,
    wxTEXT2B,
    wxTEXT1D,
    wxTEXT2D,
    wxTEXTPM,
    wxRADIOALL,
    wxRADIOMEAN,
    wxMEASCURSOR,
    wxPEAKATEND,
    wxPEAKMEAN,
    wxDIRECTION,
    wxSLOPE,
    wxSLOPEUNITS,
    wxREFERENCE,
    wxSTARTFITATPEAK,
    wxID_STARTFITATPEAK,
    wxIDNOTEBOOK
};

BEGIN_EVENT_TABLE( wxStfCursorsDlg, wxDialog )
EVT_NOTEBOOK_PAGE_CHANGED(wxIDNOTEBOOK, wxStfCursorsDlg::OnPageChanged)
EVT_COMBOBOX( wxCOMBOUM, wxStfCursorsDlg::OnComboBoxUM )
EVT_COMBOBOX( wxCOMBOU1P, wxStfCursorsDlg::OnComboBoxU1P )
EVT_COMBOBOX( wxCOMBOU2P, wxStfCursorsDlg::OnComboBoxU2P )
EVT_COMBOBOX( wxCOMBOU1B, wxStfCursorsDlg::OnComboBoxU1B )
EVT_COMBOBOX( wxCOMBOU2B, wxStfCursorsDlg::OnComboBoxU2B )
EVT_COMBOBOX( wxCOMBOU1D, wxStfCursorsDlg::OnComboBoxU1D )
EVT_COMBOBOX( wxCOMBOU2D, wxStfCursorsDlg::OnComboBoxU2D )
EVT_BUTTON( wxID_APPLY, wxStfCursorsDlg::OnPeakcalcexec )
EVT_RADIOBUTTON( wxRADIOALL, wxStfCursorsDlg::OnRadioAll )
EVT_RADIOBUTTON( wxRADIOMEAN, wxStfCursorsDlg::OnRadioMean )
END_EVENT_TABLE()

wxStfCursorsDlg::wxStfCursorsDlg(wxWindow* parent, wxStfDoc* initDoc, int id, wxString title, wxPoint pos,
                                 wxSize size, int style)
: wxDialog( parent, id, title, pos, size, style ), cursorMIsTime(true),
    cursor1PIsTime(true),cursor2PIsTime(true), cursor1BIsTime(true),cursor2BIsTime(true),
    cursor1DIsTime(true),cursor2DIsTime(true), actDoc(initDoc)
{
    wxBoxSizer* topSizer;
    topSizer = new wxBoxSizer( wxVERTICAL );

    m_notebook = new wxNotebook( this, wxIDNOTEBOOK, wxDefaultPosition, wxDefaultSize, 0 );
    m_notebook->AddPage( CreateMeasurePage(), wxT("Measure"));
    m_notebook->AddPage( CreatePeakPage(), wxT("Peak"));
    m_notebook->AddPage( CreateBasePage(), wxT("Base"));
    m_notebook->AddPage( CreateDecayPage(), wxT("Decay"));
    topSizer->Add( m_notebook, 1, wxEXPAND | wxALL, 5 );

    wxStdDialogButtonSizer* pSdbSizer = new wxStdDialogButtonSizer();
    pSdbSizer->AddButton( new wxButton( this, wxID_OK ) );
    pSdbSizer->AddButton( new wxButton( this, wxID_APPLY ) );
    pSdbSizer->AddButton( new wxButton( this, wxID_CANCEL ) );
    pSdbSizer->Realize();
    topSizer->Add( pSdbSizer, 0, wxALIGN_CENTER | wxALL, 5 );
    topSizer->SetSizeHints(this);
    this->SetSizer( topSizer );

    this->Layout();
    if (actDoc!=NULL) {
        try {
            UpdateCursors();
        }
        catch (const std::runtime_error& e) {
            wxGetApp().ExceptMsg( wxString( e.what(), wxConvLocal ));
        }
    }
}

bool wxStfCursorsDlg::TransferDataFromWindow() {
    // Apply settings before closing dialog:
    wxCommandEvent unusedEvent;
    OnPeakcalcexec(unusedEvent);
    return wxWindow::TransferDataFromWindow();
}

void wxStfCursorsDlg::EndModal(int retCode) {
    // similar to overriding OnOK in MFC (I hope...)
    switch (retCode) {
    case wxID_OK:
        if (!OnOK()) {
            wxLogMessage(wxT("Please select a valid function"));
            return;
        }
        break;
    case wxID_CANCEL:
        break;
    default:
        ;
    }
    wxDialog::EndModal(retCode);
}

bool wxStfCursorsDlg::OnOK() {
    return true;
}

wxNotebookPage* wxStfCursorsDlg::CreateMeasurePage() {
    wxPanel* nbPage;
    nbPage=new wxPanel(m_notebook);
    wxBoxSizer* pageSizer;
    pageSizer=new wxBoxSizer(wxVERTICAL);
    pageSizer->Add( CreateCursorInput( nbPage, wxTEXTM, -1, wxCOMBOUM,
            -1, 1, 10 ), 0, wxALIGN_CENTER | wxALL, 2 );
    wxCheckBox* pMeasCursor=new wxCheckBox( nbPage, wxMEASCURSOR,
            wxT("Vertical ruler through cursor"), wxDefaultPosition,
            wxDefaultSize, 0 );
    pageSizer->Add( pMeasCursor, 0, wxALIGN_CENTER | wxALL, 2);

    pageSizer->SetSizeHints( nbPage );
    nbPage->SetSizer( pageSizer );
    nbPage->Layout();
    return nbPage;
}

wxNotebookPage* wxStfCursorsDlg::CreatePeakPage() {
    wxPanel* nbPage;
    nbPage=new wxPanel(m_notebook);
    wxBoxSizer* pageSizer;
    pageSizer=new wxBoxSizer(wxVERTICAL);
    pageSizer->Add( CreateCursorInput( nbPage, wxTEXT1P, wxTEXT2P, wxCOMBOU1P,
            wxCOMBOU2P, 1, 10 ), 0, wxALIGN_CENTER | wxALL, 2 );

    wxCheckBox* pPeakAtEnd=new wxCheckBox( nbPage, wxPEAKATEND,
            wxT("Peak window ends at end of trace"), wxDefaultPosition,
            wxDefaultSize, 0 );
    pPeakAtEnd->SetValue(false);
    pageSizer->Add( pPeakAtEnd, 0, wxALIGN_CENTER | wxALL, 2);

    // Number of points for peak calculation:
    wxFlexGridSizer* peakSettingsGrid;

    peakSettingsGrid=new wxFlexGridSizer(1,3,0,0);
    wxStaticBoxSizer* peakPointsSizer = new wxStaticBoxSizer(
            wxVERTICAL, nbPage, wxT("Number of points for peak") );
    // Number of points for peak calculation:
    wxRadioButton* pAllPoints = new wxRadioButton( nbPage, wxRADIOALL,
            wxT("All points within peak window"), wxDefaultPosition,
            wxDefaultSize, wxRB_GROUP );
    peakPointsSizer->Add( pAllPoints, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    pAllPoints->SetValue(false);
    wxRadioButton* pMeanPoints = new wxRadioButton( nbPage, wxRADIOMEAN, wxT("User-defined:"),
            wxDefaultPosition, wxDefaultSize );
    pMeanPoints->SetValue(true);
    peakPointsSizer->Add( pMeanPoints, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    wxTextCtrl* textMeanPoints=new wxTextCtrl( nbPage, wxTEXTPM, wxT("1"),
            wxDefaultPosition, wxSize(48,20), wxTE_RIGHT );
    peakPointsSizer->Add( textMeanPoints, 0, wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    peakSettingsGrid->Add( peakPointsSizer, 0, wxALIGN_LEFT | wxALIGN_TOP | wxALL, 2 );

    // Direction of peak calculation:
    wxString directionChoices[] = { wxT("Up"), wxT("Down"), wxT("Both") };
    int directionNChoices = sizeof( directionChoices ) / sizeof( wxString );
    wxRadioBox* pDirection = new wxRadioBox( nbPage, wxDIRECTION, wxT("Peak direction"),
            wxDefaultPosition, wxDefaultSize, directionNChoices, directionChoices,
            0, wxRA_SPECIFY_ROWS );
    pDirection->SetSelection(1);
    peakSettingsGrid->Add( pDirection, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    pageSizer->Add(peakSettingsGrid, 0, wxALIGN_CENTER | wxALL, 2);
    
    wxFlexGridSizer* slopeSettingsGrid = new wxFlexGridSizer(1,2,0,0);
    
    // Threshold slope
    wxStaticBoxSizer* slopeSizer =
        new wxStaticBoxSizer( wxVERTICAL, nbPage, wxT("Threshold slope") );

    wxFlexGridSizer* slopeGrid = new wxFlexGridSizer(1,2,0,0);
    // user entry
    wxTextCtrl* pSlope=new wxTextCtrl( nbPage, wxSLOPE, wxT(""), wxDefaultPosition,
            wxSize(64,20), wxTE_RIGHT );
    slopeGrid->Add( pSlope, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    // Units
    wxStaticText* pSlopeUnits=new wxStaticText( nbPage, wxSLOPEUNITS, wxT("      "),
            wxDefaultPosition, wxDefaultSize, wxTE_LEFT );
    slopeGrid->Add( pSlopeUnits, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    slopeSizer->Add( slopeGrid, 0, wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    slopeSettingsGrid->Add( slopeSizer, 0, wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    
    // Ap kinetics reference
    wxString referenceChoices[] = { wxT("From baseline"), wxT("From threshold") };
    int referenceNChoices = sizeof( referenceChoices ) / sizeof( wxString );
    wxRadioBox* pReference = new wxRadioBox( nbPage, wxREFERENCE, wxT("Measure AP kinetics:"),
            wxDefaultPosition, wxDefaultSize, referenceNChoices, referenceChoices,
            0, wxRA_SPECIFY_ROWS );
    pReference->SetSelection(0);
    slopeSettingsGrid->Add( pReference, 0, wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    
    pageSizer->Add( slopeSettingsGrid, 0, wxALIGN_CENTER | wxALL, 2 );

    pageSizer->SetSizeHints(nbPage);
    nbPage->SetSizer( pageSizer );
    nbPage->Layout();
    return nbPage;
}

wxNotebookPage* wxStfCursorsDlg::CreateBasePage() {
    wxPanel* nbPage;
    nbPage=new wxPanel(m_notebook);
    wxBoxSizer* pageSizer;
    pageSizer=new wxBoxSizer(wxVERTICAL);
    pageSizer->Add( CreateCursorInput( nbPage, wxTEXT1B, wxTEXT2B, wxCOMBOU1B,
            wxCOMBOU2B, 1, 10 ), 0, wxALIGN_CENTER | wxALL, 2 );

    pageSizer->SetSizeHints(nbPage);
    nbPage->SetSizer( pageSizer );
    nbPage->Layout();
    return nbPage;
}

wxNotebookPage* wxStfCursorsDlg::CreateDecayPage() {
    wxPanel* nbPage;
    nbPage=new wxPanel(m_notebook);
    wxBoxSizer* pageSizer;
    pageSizer=new wxBoxSizer(wxVERTICAL);
    pageSizer->Add( CreateCursorInput( nbPage, wxTEXT1D, wxTEXT2D, wxCOMBOU1D,
            wxCOMBOU2D, 1, 10 ), 0, wxALIGN_CENTER | wxALL, 2 );

    wxFlexGridSizer* decaySettingsGrid = new wxFlexGridSizer(1,3,0,0);
    wxCheckBox* pStartFitAtPeak = new wxCheckBox( nbPage, wxSTARTFITATPEAK,
            wxT("Start fit at peak"),  wxDefaultPosition,  wxDefaultSize, 0  );
    decaySettingsGrid->Add( pStartFitAtPeak, 0, wxALIGN_CENTER | wxALL, 2);

    pageSizer->Add( decaySettingsGrid, 0, wxALIGN_CENTER | wxALL, 2 );

    pageSizer->SetSizeHints(nbPage);
    nbPage->SetSizer( pageSizer );
    nbPage->Layout();
    return nbPage;
}

wxFlexGridSizer* wxStfCursorsDlg::CreateCursorInput( wxPanel* nbPage, wxWindowID textC1id,
        wxWindowID textC2id, wxWindowID comboU1id, wxWindowID comboU2id, std::size_t c1,
        std::size_t c2 )
{
    wxFlexGridSizer* cursorGrid=new wxFlexGridSizer(2,3,0,0);

    // Cursor 1:

    // Description
    wxStaticText *Cursor1;
    Cursor1 = new wxStaticText( nbPage, wxID_ANY, wxT("First cursor:"),
            wxDefaultPosition, wxDefaultSize, wxTE_LEFT );
    cursorGrid->Add( Cursor1, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    // user entry
    wxString strc1,strc2;
    strc1 << (int)c1;
    wxTextCtrl* textC1 = new wxTextCtrl( nbPage, textC1id, strc1, wxDefaultPosition,
            wxSize(64,20), wxTE_RIGHT );
    cursorGrid->Add( textC1, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    // units
    wxString szUnits[] = { actDoc->GetXUnits(), wxT("pts") };
    int szUnitsSize = sizeof( szUnits ) / sizeof( wxString );
    wxComboBox* comboU1 = new wxComboBox( nbPage, comboU1id,  actDoc->GetXUnits(), wxDefaultPosition,
                wxSize(64,20), szUnitsSize, szUnits, wxCB_DROPDOWN | wxCB_READONLY );
    cursorGrid->Add( comboU1, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    // Cursor 2:
    if (textC2id >= 0) {
        wxStaticText *Cursor2;
        // Description
        Cursor2 = new wxStaticText( nbPage, wxID_ANY, wxT("Second cursor:"),
                wxDefaultPosition, wxDefaultSize, wxTE_LEFT );
        cursorGrid->Add( Cursor2, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

        // user entry
        strc2 << (int)c2;
        wxTextCtrl* textC2 = new wxTextCtrl( nbPage, textC2id, strc2, wxDefaultPosition,
                wxSize(64,20), wxTE_RIGHT );
        cursorGrid->Add( textC2, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
        // units
        wxComboBox* comboU2 = new wxComboBox( nbPage, comboU2id, actDoc->GetXUnits(), wxDefaultPosition,
                    wxSize(64,20), szUnitsSize, szUnits, wxCB_DROPDOWN | wxCB_READONLY );
        cursorGrid->Add( comboU2, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    }
    return cursorGrid;
}


void wxStfCursorsDlg::OnPeakcalcexec( wxCommandEvent& event )
{
    event.Skip();
    wxGetApp().OnPeakcalcexecMsg(actDoc);
}

int wxStfCursorsDlg::ReadCursor(wxWindowID textId, bool isTime) const {
    // always returns in units of sampling points,
    // conversion is necessary if it's in units of time:

    long cursor;
    wxString strEdit;
    wxTextCtrl *pText = (wxTextCtrl*)FindWindow(textId);
    if (pText == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::ReadCursor()"));
        return 0;
    }
    strEdit << pText->GetValue();
    if (isTime) {
        double fEdit;
        strEdit.ToDouble( &fEdit );
        cursor=stf::round(fEdit/actDoc->GetXScale());
    } else {
        strEdit.ToLong ( &cursor );
    }
    return (int)cursor;

}

int wxStfCursorsDlg::GetCursorM() const {
    return ReadCursor(wxTEXTM,cursorMIsTime);
}

int wxStfCursorsDlg::GetCursor1P() const {
    return ReadCursor(wxTEXT1P,cursor1PIsTime);
}

int wxStfCursorsDlg::GetCursor2P() const {
    return ReadCursor(wxTEXT2P,cursor2PIsTime);
}

int wxStfCursorsDlg::GetCursor1B() const {
    return ReadCursor(wxTEXT1B,cursor1BIsTime);
}

int wxStfCursorsDlg::GetCursor2B() const {
    return ReadCursor(wxTEXT2B,cursor2BIsTime);
}

int wxStfCursorsDlg::GetCursor1D() const {
    return ReadCursor(wxTEXT1D,cursor1DIsTime);
}

int wxStfCursorsDlg::GetCursor2D() const {
    return ReadCursor(wxTEXT2D,cursor2DIsTime);
}

int wxStfCursorsDlg::GetPeakPoints() const
{
    wxRadioButton* pRadioButtonAll = (wxRadioButton*)FindWindow(wxRADIOALL);
    wxRadioButton* pRadioButtonMean = (wxRadioButton*)FindWindow(wxRADIOMEAN);
    if (pRadioButtonAll==NULL || pRadioButtonMean==NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::GetPeakPoints()"));
        return 0;
    }
    if (pRadioButtonAll->GetValue()) {
        return -1;
    } else {
        if (pRadioButtonMean->GetValue()) {
            return ReadCursor(wxTEXTPM,false);
        } else {
            wxGetApp().ErrorMsg(wxT("nothing selected in wxCursorsDlg::GetPeakPoints()"));
            return 0;
        }
    }
}

void wxStfCursorsDlg::SetPeakPoints(int peakPoints)
{
    wxRadioButton* pRadioButtonAll = (wxRadioButton*)FindWindow(wxRADIOALL);
    wxRadioButton* pRadioButtonMean = (wxRadioButton*)FindWindow(wxRADIOMEAN);
    wxTextCtrl* pTextPM = (wxTextCtrl*)FindWindow(wxTEXTPM);
    if (pRadioButtonAll==NULL || pRadioButtonMean==NULL || pTextPM==NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::SetPeakPoints()"));
        return;
    }
    if (peakPoints == -1) {
        pRadioButtonAll->SetValue(true);
        pRadioButtonMean->SetValue(false);
        pTextPM->Enable(false);
        return;
    }
    if (peakPoints==0 || peakPoints<-1) {
        throw std::runtime_error("peak points out of range in wxCursorsDlg::SetPeakPoints()");
    }
    wxString entry;
    entry << peakPoints;
    pRadioButtonAll->SetValue(false);
    pRadioButtonMean->SetValue(true);
    pTextPM->Enable();
    pTextPM->SetValue( entry );
}

stf::direction wxStfCursorsDlg::GetDirection() const {
    wxRadioBox* pDirection = (wxRadioBox*)FindWindow(wxDIRECTION);
    if (pDirection == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::GetDirection()"));
        return stf::undefined_direction;
    }
    switch (pDirection->GetSelection()) {
    case 0: return stf::up;
    case 1: return stf::down;
    case 2: return stf::both;
    default: return stf::undefined_direction;
    }
}

void wxStfCursorsDlg::SetDirection(stf::direction direction) {
    wxRadioBox* pDirection = (wxRadioBox*)FindWindow(wxDIRECTION);
    if (pDirection == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::GetDirection()"));
        return;
    }
    switch (direction) {
    case stf::up:
        pDirection->SetSelection(0);
        break;
    case stf::down:
        pDirection->SetSelection(1);
        break;
    case stf::both:
    case stf::undefined_direction:
        pDirection->SetSelection(2);
        break;
    }
}

bool wxStfCursorsDlg::GetFromBase() const {
    wxRadioBox* pReference = (wxRadioBox*)FindWindow(wxREFERENCE);
    if (pReference == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::GetFromBase()"));
        return true;
    }
    switch (pReference->GetSelection()) {
     case 0: return true;
     case 1: return false;
     default: return true;
    }
}

void wxStfCursorsDlg::SetFromBase(bool fromBase) {
    wxRadioBox* pReference = (wxRadioBox*)FindWindow(wxREFERENCE);
    if (pReference == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::SetFromBase()"));
        return;
    }
    if (fromBase) {
        pReference->SetSelection(0);
    } else {
        pReference->SetSelection(1);
    }
}

bool wxStfCursorsDlg::GetPeakAtEnd() const
{	//Check if 'Upper limit at end of trace' is selected
    wxCheckBox* pPeakAtEnd = (wxCheckBox*)FindWindow(wxPEAKATEND);
    if (pPeakAtEnd == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxStfCursorsDlg::GetPeakAtEnd()"));
        return false;
    }
    return pPeakAtEnd->IsChecked();
}

bool wxStfCursorsDlg::GetStartFitAtPeak() const
{   //Check if 'Upper limit at end of trace' is selected
    wxCheckBox* pStartFitAtPeak = (wxCheckBox*)FindWindow(wxSTARTFITATPEAK);
    if (pStartFitAtPeak == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxStfCursorsDlg::GetStartFitAtPeak()"));
        return false;
    }
    return pStartFitAtPeak->IsChecked();
}

void wxStfCursorsDlg::OnPageChanged(wxNotebookEvent& event) {
    event.Skip();
    if (actDoc!=NULL) {
        try {
            UpdateCursors();
        }
        catch (const std::runtime_error& e) {
            wxGetApp().ExceptMsg( wxString( e.what(), wxConvLocal ));
        }
    }
}

void wxStfCursorsDlg::OnComboBoxUM( wxCommandEvent& event ) {
    event.Skip();
    UpdateUnits(wxCOMBOUM,cursorMIsTime,wxTEXTM);
}

void wxStfCursorsDlg::OnComboBoxU1P( wxCommandEvent& event ) {
    event.Skip();
    UpdateUnits(wxCOMBOU1P,cursor1PIsTime,wxTEXT1P);
}

void wxStfCursorsDlg::OnComboBoxU2P( wxCommandEvent& event ) {
    event.Skip();
    UpdateUnits(wxCOMBOU2P,cursor2PIsTime,wxTEXT2P);
}

void wxStfCursorsDlg::OnComboBoxU1B( wxCommandEvent& event ) {
    event.Skip();
    UpdateUnits(wxCOMBOU1B,cursor1BIsTime,wxTEXT1B);
}

void wxStfCursorsDlg::OnComboBoxU2B( wxCommandEvent& event ) {
    event.Skip();
    UpdateUnits(wxCOMBOU2B,cursor2BIsTime,wxTEXT2B);
}

void wxStfCursorsDlg::OnComboBoxU1D( wxCommandEvent& event ) {
    event.Skip();
    UpdateUnits(wxCOMBOU1D,cursor1DIsTime,wxTEXT1D);
}

void wxStfCursorsDlg::OnComboBoxU2D( wxCommandEvent& event ) {
    event.Skip();
    UpdateUnits(wxCOMBOU2D,cursor2DIsTime,wxTEXT2D);
}

void wxStfCursorsDlg::OnRadioAll( wxCommandEvent& event ) {
    event.Skip();
    wxRadioButton* pRadioAll = (wxRadioButton*)FindWindow(wxRADIOALL);
    wxRadioButton* pRadioMean = (wxRadioButton*)FindWindow(wxRADIOMEAN);
    wxTextCtrl* pTextPM = (wxTextCtrl*)FindWindow(wxTEXTPM);
    if (pTextPM==NULL || pRadioMean==NULL || pRadioAll==NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::OnRadioAll()"));
        return;
    }
    pTextPM->Enable(false);
    pRadioMean->SetValue(false);
}

void wxStfCursorsDlg::OnRadioMean( wxCommandEvent& event ) {
    event.Skip();
    wxRadioButton* pRadioAll = (wxRadioButton*)FindWindow(wxRADIOALL);
    wxRadioButton* pRadioMean = (wxRadioButton*)FindWindow(wxRADIOMEAN);
    wxTextCtrl* pTextPM = (wxTextCtrl*)FindWindow(wxTEXTPM);
    if (pTextPM==NULL || pRadioMean==NULL || pRadioAll==NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::OnRadioMean()"));
        return;
    }
    pTextPM->Enable(true);
    pRadioAll->SetValue(false);
}

void wxStfCursorsDlg::UpdateUnits(wxWindowID comboId, bool& setTime, wxWindowID textId) {
    // Read current entry:
    wxString strRead;
    wxTextCtrl* pText = (wxTextCtrl*)FindWindow(textId);
    if (pText == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::UpdateUnits()"));
        return;
    }
    strRead << pText->GetValue();
    double fEntry=0.0;
    strRead.ToDouble( &fEntry );
    wxComboBox* pCombo = (wxComboBox*)FindWindow(comboId);
    if (pCombo == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::UpdateUnits()"));
        return;
    }
    bool isTimeNow=(pCombo->GetCurrentSelection()==0);
    // switched from pts to time:
    if (!setTime&&isTimeNow) {
        // switched from pts to time:
        double fNewValue=fEntry*actDoc->GetXScale();
        wxString strNewValue;strNewValue << fNewValue;
        pText->SetValue( strNewValue );
        setTime=true;
    }
    if (setTime&&!isTimeNow) {
        // switched from time to pts:
        int iNewValue=stf::round(fEntry/actDoc->GetXScale());
        wxString strNewValue; strNewValue << iNewValue;
        pText->SetValue( strNewValue );
        setTime=false;
    }
}

void wxStfCursorsDlg::UpdateCursors() {
    stf::cursor_type select = CurrentCursor();
    int iNewValue1=0, iNewValue2=0;
    bool cursor2isTime=true, cursor1isTime=true;
    wxTextCtrl* pText1=NULL, *pText2=NULL;
    if (actDoc == NULL) {
        throw std::runtime_error("No active document found");
    }
    switch (select) {
    case stf::measure_cursor:	// Measure
        iNewValue1=(int)actDoc->GetMeasCursor();
        cursor1isTime=cursorMIsTime;
        pText1=(wxTextCtrl*)FindWindow(wxTEXTM);
        break;
    case stf::peak_cursor: // Peak
        iNewValue1=(int)actDoc->GetPeakBeg();
        iNewValue2=(int)actDoc->GetPeakEnd();
        cursor1isTime=cursor1PIsTime;
        cursor2isTime=cursor2PIsTime;
        pText1=(wxTextCtrl*)FindWindow(wxTEXT1P);
        pText2=(wxTextCtrl*)FindWindow(wxTEXT2P);
        // Update the mean peak points and direction:
        SetPeakPoints( actDoc->GetPM() );
        SetDirection( actDoc->GetDirection() );
        SetFromBase( actDoc->GetFromBase() );
        break;
    case stf::base_cursor: // Base
        iNewValue1=(int)actDoc->GetBaseBeg();
        iNewValue2=(int)actDoc->GetBaseEnd();
        cursor1isTime=cursor1BIsTime;
        cursor2isTime=cursor2BIsTime;
        pText1=(wxTextCtrl*)FindWindow(wxTEXT1B);
        pText2=(wxTextCtrl*)FindWindow(wxTEXT2B);
        break;
    case stf::decay_cursor: // Decay
        iNewValue1=(int)actDoc->GetFitBeg();
        iNewValue2=(int)actDoc->GetFitEnd();
        cursor1isTime=cursor1DIsTime;
        cursor2isTime=cursor2DIsTime;
        pText1=(wxTextCtrl*)FindWindow(wxTEXT1D);
        pText2=(wxTextCtrl*)FindWindow(wxTEXT2D);
        break;
    default:
        break;
    }

    double fNewValue1=iNewValue1*actDoc->GetXScale();
    double fNewValue2=iNewValue2*actDoc->GetXScale();

    wxString strNewValue;
    if (cursor1isTime) {
        strNewValue << fNewValue1;
    } else {
        strNewValue << iNewValue1;
    }
    if (pText1 != NULL) {
        pText1->SetValue( strNewValue );
    }

    if (select!=stf::measure_cursor && pText2 != NULL) {
        wxString strNewValue2;
        if (cursor2isTime) {
            strNewValue2 << fNewValue2;
        } else {
            strNewValue2 << iNewValue2;
        }
        pText2->SetValue( strNewValue2 );
    }
    
    SetSlope( actDoc->GetSlopeForThreshold() );
    
    wxString slopeUnits;
    slopeUnits += actDoc->at(actDoc->GetCurCh()).GetYUnits();
    slopeUnits += wxT("/");
    slopeUnits += actDoc->GetXUnits();
    SetSlopeUnits(slopeUnits);
}

stf::cursor_type wxStfCursorsDlg::CurrentCursor() const {
    if (m_notebook == NULL)
        return stf::undefined_cursor;
    switch (m_notebook->GetSelection()) {
    case 0:	return stf::measure_cursor;
    case 1: return stf::peak_cursor;
    case 2: return stf::base_cursor;
    case 3: return stf::decay_cursor;
    default: return stf::undefined_cursor;
    }
}

double wxStfCursorsDlg::GetSlope() const {
    double f=0.0;
    wxTextCtrl* pSlope =(wxTextCtrl*) FindWindow(wxSLOPE);
    if (pSlope == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::GetSlope()"));
        return 0;
    }
    wxString entry;
    entry << pSlope->GetValue();
    entry.ToDouble( &f );
    return f;
}

void wxStfCursorsDlg::SetSlope( double fSlope ) {
    wxTextCtrl* pSlope = (wxTextCtrl*)FindWindow(wxSLOPE);
    wxString wxsSlope;
    wxsSlope << fSlope;
    if ( pSlope != NULL )
        pSlope->SetValue( wxsSlope );
}

void wxStfCursorsDlg::SetSlopeUnits(const wxString& units) {
    wxStaticText* pSlopeUnits = (wxStaticText*)FindWindow(wxSLOPEUNITS);
    if (pSlopeUnits != NULL) {
        pSlopeUnits->SetLabel(units);
    }
}

bool wxStfCursorsDlg::GetRuler() const {
    wxCheckBox* pMeasCursor = (wxCheckBox*)FindWindow(wxMEASCURSOR);
    if (pMeasCursor == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::GetRuler()"));
        return false;
    }
    return pMeasCursor->IsChecked();
}
