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
    wxCOMBOU1L,
    wxCOMBOU2L,
#ifdef WITH_PSLOPE
    wxCOMBOU1PS,
    wxCOMBOU2PS,
#endif 
    wxTEXTM,
    wxTEXT1P,
    wxTEXT2P,
    wxTEXT1B,
    wxTEXT2B,
    wxTEXT1D,
    wxTEXT2D,
    wxTEXT1L,
    wxTEXT2L,
#ifdef WITH_PSLOPE
    wxTEXT1PS,
    wxTEXT2PS,
    wxTEXT_PSDELTAT,
#endif
    wxTEXTPM,
    wxRADIOALL,
    wxRADIOMEAN,
    wxRADIO_LAT_MAXSLOPE1,
    wxRADIO_LAT_HALFWIDTH1,
    wxRADIO_LAT_PEAK1,
    wxRADIO_LAT_MANUAL1,
    
    wxRADIO_LAT_EVENT2,
    wxRADIO_LAT_MAXSLOPE2,
    wxRADIO_LAT_HALFWIDTH2,
    wxRADIO_LAT_PEAK2,
    wxRADIO_LAT_MANUAL2,
    wxLATENCYWINDOW,
#ifdef WITH_PSLOPE
    // Slope radio boxes 
    wxRADIO_PSManBeg,
    wxRADIO_PSEventBeg,
    wxRADIO_PSThrBeg,
    wxRADIO_PSt50Beg,

    wxRADIO_PSManEnd,
    wxRADIO_PSt50End,
    wxRADIO_PSDeltaT,
    wxRADIO_PSPeakEnd,
#endif //WITH_PSLOPE
    wxMEASCURSOR,
    wxPEAKATEND,
    wxPEAKMEAN,
    wxDIRECTION,
    wxSLOPE,
#ifdef WITH_PSLOPE
    wxSLOPE_FROM_PSLOPE,
#endif
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
EVT_COMBOBOX( wxCOMBOU1L, wxStfCursorsDlg::OnComboBoxU1L )
EVT_COMBOBOX( wxCOMBOU2L, wxStfCursorsDlg::OnComboBoxU2L )
#ifdef WITH_PSLOPE
EVT_COMBOBOX( wxCOMBOU1PS, wxStfCursorsDlg::OnComboBoxU1PS )
EVT_COMBOBOX( wxCOMBOU2PS, wxStfCursorsDlg::OnComboBoxU2PS )
#endif

// bindings radio buttons
EVT_BUTTON( wxID_APPLY, wxStfCursorsDlg::OnPeakcalcexec )
EVT_RADIOBUTTON( wxRADIOALL, wxStfCursorsDlg::OnRadioAll )
EVT_RADIOBUTTON( wxRADIOMEAN, wxStfCursorsDlg::OnRadioMean )

EVT_RADIOBUTTON( wxRADIO_LAT_MANUAL1,    wxStfCursorsDlg::OnRadioLatManualBeg )
EVT_RADIOBUTTON( wxRADIO_LAT_MAXSLOPE1,  wxStfCursorsDlg::OnRadioLatNonManualBeg )
EVT_RADIOBUTTON( wxRADIO_LAT_HALFWIDTH1, wxStfCursorsDlg::OnRadioLatNonManualBeg )
EVT_RADIOBUTTON( wxRADIO_LAT_PEAK1,      wxStfCursorsDlg::OnRadioLatNonManualBeg )

EVT_RADIOBUTTON( wxRADIO_LAT_MANUAL2,    wxStfCursorsDlg::OnRadioLatManualEnd )
EVT_RADIOBUTTON( wxRADIO_LAT_HALFWIDTH2, wxStfCursorsDlg::OnRadioLatNonManualEnd )
EVT_RADIOBUTTON( wxRADIO_LAT_PEAK2,      wxStfCursorsDlg::OnRadioLatNonManualEnd )
EVT_RADIOBUTTON( wxRADIO_LAT_MAXSLOPE2,  wxStfCursorsDlg::OnRadioLatNonManualEnd )
EVT_RADIOBUTTON( wxRADIO_LAT_EVENT2,     wxStfCursorsDlg::OnRadioLatNonManualEnd )
#ifdef WITH_PSLOPE
EVT_RADIOBUTTON( wxRADIO_PSManBeg, wxStfCursorsDlg::OnRadioPSManBeg )
EVT_RADIOBUTTON( wxRADIO_PSEventBeg, wxStfCursorsDlg::OnRadioPSEventBeg )
EVT_RADIOBUTTON( wxRADIO_PSThrBeg, wxStfCursorsDlg::OnRadioPSThrBeg )
EVT_RADIOBUTTON( wxRADIO_PSt50Beg, wxStfCursorsDlg::OnRadioPSt50Beg )

EVT_RADIOBUTTON( wxRADIO_PSManEnd, wxStfCursorsDlg::OnRadioPSManEnd )
EVT_RADIOBUTTON( wxRADIO_PSt50End, wxStfCursorsDlg::OnRadioPSt50End )
EVT_RADIOBUTTON( wxRADIO_PSDeltaT, wxStfCursorsDlg::OnRadioPSDeltaT )
EVT_RADIOBUTTON( wxRADIO_PSPeakEnd, wxStfCursorsDlg::OnRadioPSPeakEnd )
#endif

END_EVENT_TABLE()

wxStfCursorsDlg::wxStfCursorsDlg(wxWindow* parent, wxStfDoc* initDoc, int id, wxString title, wxPoint pos,
                                 wxSize size, int style)
: wxDialog( parent, id, title, pos, size, style ), cursorMIsTime(true),
    cursor1PIsTime(true), cursor2PIsTime(true), cursor1BIsTime(true), cursor2BIsTime(true), cursor1DIsTime(true), cursor2DIsTime(true),
#ifdef WITH_PSLOPE
    cursor1PSIsTime(true), cursor2PSIsTime(true), 
#endif
    cursor1LIsTime(true), cursor2LIsTime(true),
    actDoc(initDoc)
{
    wxBoxSizer* topSizer;
    topSizer = new wxBoxSizer( wxVERTICAL );

    m_notebook = new wxNotebook( this, wxIDNOTEBOOK, wxDefaultPosition, wxDefaultSize, 0 );
    m_notebook->AddPage( CreateMeasurePage(), wxT("Measure"));
    m_notebook->AddPage( CreatePeakPage(), wxT("Peak"));
    m_notebook->AddPage( CreateBasePage(), wxT("Base"));
    m_notebook->AddPage( CreateDecayPage(), wxT("Decay"));
    m_notebook->AddPage( CreateLatencyPage(), wxT("Latency"));
#ifdef WITH_PSLOPE
    m_notebook->AddPage( CreatePSlopePage(), wxT("PSlope"));
#endif
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
    /* how did this sneak in?!
       UpdateCursors();
       OnPeakcalcexec(unusedEvent);
    */
    return wxWindow::TransferDataFromWindow();
}

void wxStfCursorsDlg::EndModal(int retCode) {
    wxCommandEvent unusedEvent;
    // similar to overriding OnOK in MFC (I hope...)
    switch (retCode) {
    case wxID_OK:
        if (!OnOK()) {
            wxLogMessage(wxT("Please select a valid function"));
            return;
        }
        OnPeakcalcexec(unusedEvent);
        break;
    case wxID_CANCEL:
        break;
    default:
        ;
    }
    wxDialog::EndModal(retCode);
}

bool wxStfCursorsDlg::OnOK() {
    //wxCommandEvent unusedEvent;
    //OnPeakcalcexec(unusedEvent);
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

    wxFlexGridSizer* peakSettingsGrid;
    peakSettingsGrid=new wxFlexGridSizer(1,3,0,0);

    // Number of points for peak calculation:
    wxStaticBoxSizer* peakPointsSizer = new wxStaticBoxSizer(
            wxVERTICAL, nbPage, wxT("Number of points for peak") );

    wxRadioButton* pAllPoints = new wxRadioButton( nbPage, wxRADIOALL,
            wxT("All points within peak window"), wxDefaultPosition,
            wxDefaultSize, wxRB_GROUP );
    peakPointsSizer->Add( pAllPoints, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    pAllPoints->SetValue(false);

    wxFlexGridSizer* usrdefGrid;
    usrdefGrid = new wxFlexGridSizer(1,2,0,0);

    wxRadioButton* pMeanPoints = new wxRadioButton( nbPage, wxRADIOMEAN, wxT("User-defined:"),
            wxDefaultPosition, wxDefaultSize );
    pMeanPoints->SetValue(true);

    usrdefGrid->Add(pMeanPoints, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);

    //peakPointsSizer->Add( pMeanPoints, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    wxTextCtrl* textMeanPoints=new wxTextCtrl( nbPage, wxTEXTPM, wxT("1"),
            wxDefaultPosition, wxSize(48,20), wxTE_RIGHT );
    peakPointsSizer->Add( textMeanPoints, 0, wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL | wxALL, 2 );

    usrdefGrid->Add(textMeanPoints, 0, wxALIGN_RIGHT | wxALIGN_CENTER_VERTICAL | wxALL, 2);

    //peakSettingsGrid->Add( peakPointsSizer, 0, wxALIGN_LEFT | wxALIGN_TOP | wxALL, 2 );
    peakPointsSizer->Add( usrdefGrid, 0, wxALIGN_LEFT | wxALIGN_TOP | wxALL, 2 );
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
        new wxStaticBoxSizer( wxVERTICAL, nbPage, wxT("Threshold slope   ") );

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
    wxRadioBox* pReference = new wxRadioBox( nbPage, wxREFERENCE, wxT("Measure AP kinetics "),
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

wxNotebookPage* wxStfCursorsDlg:: CreateLatencyPage(){
    wxPanel* nbPage;
    nbPage = new wxPanel(m_notebook);

    wxBoxSizer* pageSizer;
    pageSizer = new wxBoxSizer(wxVERTICAL);

    pageSizer->Add(CreateCursorInput(nbPage, wxTEXT1L, wxTEXT2L, wxCOMBOU1L,
        wxCOMBOU2L, 1, 10), 0, wxALIGN_CENTER | wxALL, 2);

    // Checkbox for using peak window for latency cursors
    wxCheckBox *pUsePeak = new wxCheckBox(nbPage, wxLATENCYWINDOW,
        wxT("Measure latencies within peak cursors"), wxDefaultPosition,
        wxDefaultSize, 0);
    pageSizer->Add(pUsePeak, 0 , wxALIGN_CENTER | wxALL, 2);

    // Grid
    wxFlexGridSizer* LatBegEndGrid;
    LatBegEndGrid = new wxFlexGridSizer(1,2,0,0); // rows, cols

    //**** Radio options "Measure from" ****
    wxStaticBoxSizer* LeftBoxSizer = new wxStaticBoxSizer(
        wxVERTICAL, nbPage, wxT("Latency from") );

    // Latency from: Manual
    wxRadioButton* wxRadio_Lat_Manual1 = new wxRadioButton( nbPage, wxRADIO_LAT_MANUAL1, wxT("Manual"),
            wxDefaultPosition, wxDefaultSize, wxRB_GROUP);
    //wxRadio_Lat_Manual1->SetValue(true);
    
    // Latency from: Peak
    wxRadioButton* wxRadio_Lat_Peak1 = new wxRadioButton( nbPage, wxRADIO_LAT_PEAK1, wxT("Peak"),
            wxDefaultPosition, wxDefaultSize);

    // Latency from: Maximal slope
    wxRadioButton* wxRadio_Lat_MaxSlope1 = new wxRadioButton( nbPage, wxRADIO_LAT_MAXSLOPE1, wxT("Maximal slope"),
            wxDefaultPosition, wxDefaultSize );

    // Latency from: Half-maximal amplitude
    wxRadioButton* wxRadio_Lat_HalfWidth1 = new wxRadioButton( nbPage, wxRADIO_LAT_HALFWIDTH1, wxT("Half-width (t50)"),
            wxDefaultPosition, wxDefaultSize );
     

    // Sizer to group the radio options
    LeftBoxSizer->Add( wxRadio_Lat_Manual1,    0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    LeftBoxSizer->Add( wxRadio_Lat_Peak1,      0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    LeftBoxSizer->Add( wxRadio_Lat_MaxSlope1,  0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    LeftBoxSizer->Add( wxRadio_Lat_HalfWidth1, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    // Add to LatBegEndGrid
    LatBegEndGrid->Add(LeftBoxSizer, 0, wxALIGN_LEFT | wxALIGN_TOP | wxALL, 2);

    //**** Radio options "Latency to" ****
    wxStaticBoxSizer* RightBoxSizer = new wxStaticBoxSizer(
        wxVERTICAL, nbPage, wxT("Latency to") );

    // Latency to: Manual
    wxRadioButton* wxRadio_Lat_Manual2 = new wxRadioButton( nbPage, wxRADIO_LAT_MANUAL2, wxT("Manual"),
            wxDefaultPosition, wxDefaultSize, wxRB_GROUP);
    //wxRadio_Lat_Manual2->SetValue(true);

    // Latency to: Peak
    wxRadioButton* wxRadio_Lat_Peak2 = new wxRadioButton( nbPage, wxRADIO_LAT_PEAK2, wxT("Peak"),
            wxDefaultPosition, wxDefaultSize);

    // Latency to: Half-maximal amplitude
    wxRadioButton* wxRadio_Lat_HalfWidth2 = new wxRadioButton( nbPage, wxRADIO_LAT_HALFWIDTH2, wxT("Half-width (t50)"),
            wxDefaultPosition, wxDefaultSize);

    // Latency to: Maximal slope
    wxRadioButton* wxRadio_Lat_MaxSlope2 = new wxRadioButton( nbPage, wxRADIO_LAT_MAXSLOPE2, wxT("Maximal slope"),
            wxDefaultPosition, wxDefaultSize);

    // Latency to: Beginning of event
    wxRadioButton* wxRadio_Lat_Event2 = new wxRadioButton( nbPage, wxRADIO_LAT_EVENT2, wxT("Beginning of event"),
            wxDefaultPosition, wxDefaultSize );


    // Sizer to group the radio options
    RightBoxSizer->Add( wxRadio_Lat_Manual2,    0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    RightBoxSizer->Add( wxRadio_Lat_Peak2,      0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    RightBoxSizer->Add( wxRadio_Lat_MaxSlope2,  0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    RightBoxSizer->Add( wxRadio_Lat_HalfWidth2, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    RightBoxSizer->Add( wxRadio_Lat_Event2,     0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);

    // Add to LatBegEndGrid
    LatBegEndGrid->Add(RightBoxSizer, 0, wxALIGN_LEFT | wxALIGN_TOP | wxALL, 2);

    pageSizer->Add(LatBegEndGrid, 0, wxALIGN_CENTER | wxALL, 2);

    nbPage->SetSizer(pageSizer);
    nbPage->Layout();
    return nbPage;


}
#ifdef WITH_PSLOPE
wxNotebookPage* wxStfCursorsDlg::CreatePSlopePage() {
    wxPanel* nbPage;
    nbPage=new wxPanel(m_notebook);

    // Sizer
    wxBoxSizer* pageSizer;
    pageSizer=new wxBoxSizer(wxVERTICAL);

    pageSizer->Add( CreateCursorInput( nbPage, wxTEXT1PS, wxTEXT2PS, wxCOMBOU1PS,
            wxCOMBOU2PS, 1, 10 ), 0, wxALIGN_CENTER | wxALL, 2 );

    // Grid
    wxFlexGridSizer* PSBegEndGrid;
    PSBegEndGrid = new wxFlexGridSizer(1,2,0,0); // rows, cols

    //**** Radio options "Slope from" ****
    wxStaticBoxSizer* LeftBoxSizer = new wxStaticBoxSizer(
        wxVERTICAL, nbPage, wxT("Slope from") );

    // Slope from: Manual
    wxRadioButton* pPSManBeg = new wxRadioButton( nbPage, wxRADIO_PSManBeg, wxT("Manual"),
            wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
    pPSManBeg->SetValue(true);

    // Slope from: Commencement
    wxRadioButton* pPSEventBeg = new wxRadioButton( nbPage, wxRADIO_PSEventBeg, wxT("Commencement"),
            wxDefaultPosition, wxDefaultSize );
     
    // Slope from: Threshold slope
    wxFlexGridSizer* thrGrid;
    thrGrid = new wxFlexGridSizer(1,2,0,0);

    wxRadioButton* pPSThrBeg = new wxRadioButton( nbPage, wxRADIO_PSThrBeg, wxT("Threshold slope"),
            wxDefaultPosition, wxDefaultSize);
    thrGrid->Add(pPSThrBeg, 0, wxALIGN_RIGHT | wxALIGN_CENTER_VERTICAL | wxALL, 2);

    wxTextCtrl* pPStextThr = new wxTextCtrl(nbPage, wxSLOPE_FROM_PSLOPE, wxT(" "), wxDefaultPosition, 
            wxSize(44,20), wxTE_RIGHT);
    pPStextThr->Enable(false);
    thrGrid->Add(pPStextThr, 0, wxALIGN_RIGHT | wxALIGN_CENTER_VERTICAL | wxALL, 2);

    // Slope from: t50
    wxRadioButton* pPSt50Beg = new wxRadioButton( nbPage, wxRADIO_PSt50Beg, wxT("Half-width (t50)"),
            wxDefaultPosition, wxDefaultSize);


    // Sizer to group the radio options
    LeftBoxSizer->Add( pPSManBeg, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    LeftBoxSizer->Add( pPSEventBeg, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    //LeftBoxSizer->Add( pPSThrBeg, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    LeftBoxSizer->Add( thrGrid, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    LeftBoxSizer->Add( pPSt50Beg, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    // Add to PSBegEndGrid
    PSBegEndGrid->Add(LeftBoxSizer, 0, wxALIGN_LEFT | wxALIGN_TOP | wxALL, 2);

    //**** Radio options "Slope to" ****
    wxStaticBoxSizer* RightBoxSizer = new wxStaticBoxSizer(
        wxVERTICAL, nbPage, wxT("Slope to") );

    // Slope to: Manual
    wxRadioButton* pPSManEnd = new wxRadioButton( nbPage, wxRADIO_PSManEnd, wxT("Manual"),
            wxDefaultPosition, wxDefaultSize, wxRB_GROUP);
    pPSManEnd->SetValue(true);

    // Slope to: Half-width
    wxRadioButton* pPSt50End = new wxRadioButton( nbPage, wxRADIO_PSt50End, wxT("Half-width (t50)"),
            wxDefaultPosition, wxDefaultSize);

    // Slope to: DeltaT
    wxFlexGridSizer* DeltaTGrid;
    DeltaTGrid = new wxFlexGridSizer(1,2,0,0);

    wxRadioButton* pPSDeltaT = new wxRadioButton( nbPage, wxRADIO_PSDeltaT, wxT("Delta t"),
            wxDefaultPosition, wxDefaultSize);
    DeltaTGrid->Add(pPSDeltaT, 0, wxALIGN_RIGHT | wxALIGN_CENTER_VERTICAL | wxALL, 2);

    wxTextCtrl* pTextPSDeltaT = new wxTextCtrl(nbPage, wxTEXT_PSDELTAT, wxT(""), wxDefaultPosition, 
            wxSize(44,20), wxTE_RIGHT);
    pTextPSDeltaT->Enable(false);

    DeltaTGrid->Add(pTextPSDeltaT, 0, wxALIGN_RIGHT | wxALIGN_CENTER_VERTICAL | wxALL, 2);

    // Slope to: Peak
    wxRadioButton* pPSPeakEnd = new wxRadioButton( nbPage, wxRADIO_PSPeakEnd, wxT("Peak"),
            wxDefaultPosition, wxDefaultSize);


    // Sizer to group the radio options
    RightBoxSizer->Add( pPSManEnd, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    RightBoxSizer->Add( pPSt50End, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    RightBoxSizer->Add( DeltaTGrid, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    RightBoxSizer->Add( pPSPeakEnd, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2);
    // Add to PSBegEndGrid
    PSBegEndGrid->Add(RightBoxSizer, 0, wxALIGN_LEFT | wxALIGN_TOP | wxALL, 2);

    pageSizer->Add(PSBegEndGrid, 0, wxALIGN_CENTER | wxALL, 2);

    nbPage->SetSizer(pageSizer);
    nbPage->Layout();
    return nbPage;
}
#endif // WITH_PSLOPE

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
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
    wxString szUnits[] = { actDoc->GetXUnits(), wxT("pts") };
    int szUnitsSize = sizeof( szUnits ) / sizeof( wxString );
    wxComboBox* comboU1 = new wxComboBox( nbPage, comboU1id,  actDoc->GetXUnits(), wxDefaultPosition,
#else
    wxString szUnits[] = { wxString(actDoc->GetXUnits().c_str(), wxConvUTF8), wxT("pts") };
    int szUnitsSize = sizeof( szUnits ) / sizeof( wxString );
    wxComboBox* comboU1 = new wxComboBox( nbPage, comboU1id,
                                                                                wxString(actDoc->GetXUnits().c_str(), wxConvUTF8), wxDefaultPosition,
#endif
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
        
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
        wxComboBox* comboU2 = new wxComboBox( nbPage, comboU2id, actDoc->GetXUnits(), wxDefaultPosition,
#else
                                              wxComboBox* comboU2 = new wxComboBox( nbPage, comboU2id, wxString(actDoc->GetXUnits().c_str(), wxConvUTF8), wxDefaultPosition,
#endif
                    wxSize(64,20), szUnitsSize, szUnits, wxCB_DROPDOWN | wxCB_READONLY );
        cursorGrid->Add( comboU2, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    }
    return cursorGrid;
}


void wxStfCursorsDlg::OnPeakcalcexec( wxCommandEvent& event )
{
    event.Skip();
    // Update the results table (see wxStfApp in app.cpp)
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

#ifdef WITH_PSLOPE
int wxStfCursorsDlg::ReadDeltaT(wxWindowID textID) const {
    // returns DeltaT entered in the textBox in units of sampling points

    long cursorpos=0;
    wxString strDeltaT;
    wxTextCtrl *pText = (wxTextCtrl*)FindWindow(textID);
    if (pText == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::ReadDeltaT()"));
        return 0;
    }
    
    strDeltaT << pText->GetValue();
    double DeltaTval;
    strDeltaT.ToDouble(&DeltaTval);
    cursorpos = stf::round(DeltaTval/actDoc->GetXScale());

    return (int)cursorpos;

}
#endif // WITH_PSLOPE

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

int wxStfCursorsDlg::GetCursor1L() const {
    return ReadCursor(wxTEXT1L, cursor1LIsTime);
}

int wxStfCursorsDlg::GetCursor2L() const {
    return ReadCursor(wxTEXT2L, cursor2LIsTime);
}

#ifdef WITH_PSLOPE
int wxStfCursorsDlg::GetCursor1PS() const {
    return ReadCursor(wxTEXT1PS, cursor1PSIsTime);
}

int wxStfCursorsDlg::GetCursor2PS() const {
    return ReadCursor(wxTEXT2PS, cursor2PSIsTime);
}

#endif

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

#ifdef WITH_PSLOPE
int wxStfCursorsDlg::GetDeltaT() const {
    return ReadDeltaT(wxTEXT_PSDELTAT);
}
#endif

#ifdef WITH_PSLOPE
void wxStfCursorsDlg::SetDeltaT (int DeltaT) {
    wxRadioButton* pRadPSDeltaT = (wxRadioButton*)FindWindow(wxRADIO_PSDeltaT);
    wxTextCtrl* pTextPSDeltaT = (wxTextCtrl*)FindWindow(wxTEXT_PSDELTAT);
    if (pRadPSDeltaT == NULL || pTextPSDeltaT == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxSetDeltaT()"));
        return;
    }
    
    // transform sampling points into x-units
    double fDeltaT;
    fDeltaT =DeltaT*actDoc->GetXScale();
    wxString strDeltaTval;
    strDeltaTval << fDeltaT;
    //pRadPSDeltaT->Enable(true);
    //pTextPSDeltaT->Enable(true);
    pTextPSDeltaT->SetValue(strDeltaTval);
}

#endif // WITH_PSLOPE

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

void wxStfCursorsDlg::OnComboBoxU1L( wxCommandEvent& event ) {
    event.Skip();
    wxRadioButton* wxRadio_Lat_Manual1 = (wxRadioButton*)FindWindow(wxRADIO_LAT_MANUAL1);
    if (wxRadio_Lat_Manual1 == NULL){
        wxGetApp().ErrorMsg(wxT("Null pointer in wxCursorsDlg::OnComboBoxU1LS()"));
        return;
    }
    else
        wxRadio_Lat_Manual1->SetValue(true);

    UpdateUnits(wxCOMBOU1L,cursor1LIsTime,wxTEXT1L);
}

void wxStfCursorsDlg::OnComboBoxU2L( wxCommandEvent& event ) {
    event.Skip();
    wxRadioButton* wxRadio_Lat_Manual2 = (wxRadioButton*)FindWindow(wxRADIO_LAT_MANUAL2);
    if (wxRadio_Lat_Manual2 == NULL){
        wxGetApp().ErrorMsg(wxT("Null pointer in wxCursorsDlg::OnComboBoxU2LS()"));
        return;
    }
    else
        wxRadio_Lat_Manual2->SetValue(true);
    UpdateUnits(wxCOMBOU2L,cursor2LIsTime,wxTEXT2L);
}

void wxStfCursorsDlg::OnRadioLatManualBeg( wxCommandEvent& event ) {
    event.Skip();
    wxTextCtrl* pCursor1L = (wxTextCtrl*)FindWindow(wxTEXT1L);
    if (pCursor1L == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::OnRadioLatManBeg()"));
        return;
    }
    // if cursor wxTextCtrl is NOT enabled
    if (!pCursor1L->IsEnabled())
        pCursor1L->Enable(true);

}

void wxStfCursorsDlg::OnRadioLatManualEnd( wxCommandEvent& event ) {
    event.Skip();
    wxTextCtrl* pCursor2L = (wxTextCtrl*)FindWindow(wxTEXT2L);
    if (pCursor2L == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::OnRadioLatManEnd()"));
        return;
    }
    // if cursor wxTextCtrl is NOT enabled
    if (!pCursor2L->IsEnabled())
        pCursor2L->Enable(true);

}

void wxStfCursorsDlg::OnRadioLatNonManualBeg( wxCommandEvent& event ) {
    event.Skip();
    wxTextCtrl* pCursor1L = (wxTextCtrl*)FindWindow(wxTEXT1L);
    if (pCursor1L == NULL ) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::OnRadioLatt50Beg()"));
        return;
    }
    // disable cursor wxTextCtrl if it is enabled 
    if (pCursor1L->IsEnabled())
        pCursor1L->Enable(false);

}

void wxStfCursorsDlg::OnRadioLatNonManualEnd( wxCommandEvent& event ) {
    event.Skip();
    wxTextCtrl* pCursor2L = (wxTextCtrl*)FindWindow(wxTEXT2L);
    if (pCursor2L == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::OnRadioNonManualEnd()"));
        return;
    }
    // disable cursor wxTextCtrl if it is enabled 
    if (pCursor2L->IsEnabled()) 
        pCursor2L->Enable(false);

}


#ifdef WITH_PSLOPE

void wxStfCursorsDlg::OnComboBoxU1PS( wxCommandEvent& event ) {

    event.Skip();
    // select manual option in "measurement from" box
    wxRadioButton* pPSManBeg   = (wxRadioButton*)FindWindow(wxRADIO_PSManBeg);

    if (pPSManBeg == NULL) {
        wxGetApp().ErrorMsg(wxT("Null pointer in wxCursorsDlg::OnComboBoxU1PS()"));
        return;
    }
    else 
        pPSManBeg->SetValue(true);

    UpdateUnits(wxCOMBOU1PS,cursor1PSIsTime,wxTEXT1PS);
}

void wxStfCursorsDlg::OnComboBoxU2PS( wxCommandEvent& event ) {

    event.Skip();
    // select manual option in "measurement to" box
    wxRadioButton* pPSManEnd   = (wxRadioButton*)FindWindow(wxRADIO_PSManEnd);

    if (pPSManEnd == NULL) {
        wxGetApp().ErrorMsg(wxT("Null pointer in wxCursorsDlg::OnComboBoxU2PS()"));
        return;
    }
    else 
        pPSManEnd->SetValue(true);

    UpdateUnits(wxCOMBOU2PS,cursor2PSIsTime,wxTEXT2PS);
}

void wxStfCursorsDlg::OnRadioPSManBeg( wxCommandEvent& event ) {
    event.Skip();
    wxTextCtrl* pCursor1PS = (wxTextCtrl*)FindWindow(wxTEXT1PS);
    wxTextCtrl* pPStextThr = (wxTextCtrl*)FindWindow(wxSLOPE_FROM_PSLOPE);
    if (pCursor1PS == NULL || pPStextThr == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::OnRadioManBeg()"));
        return;
    }
    // if cursor wxTextCtrl is NOT enabled
    if (!pCursor1PS->IsEnabled())
        pCursor1PS->Enable(true);

    // disable threshold text if is enabled
    if (pPStextThr->IsEnabled())
        pPStextThr->Enable(false);

}

void wxStfCursorsDlg::OnRadioPSEventBeg( wxCommandEvent& event ) {
    event.Skip();
    wxTextCtrl* pCursor1PS = (wxTextCtrl*)FindWindow(wxTEXT1PS);
    wxTextCtrl* pPStextThr = (wxTextCtrl*)FindWindow(wxSLOPE_FROM_PSLOPE);
    if (pCursor1PS == NULL || pPStextThr == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::OnRadioPSEventBeg()"));
        return;
    }
    // disable cursor wxTextCtrl if it is enabled 
    if (pCursor1PS->IsEnabled()) 
        pCursor1PS->Enable(false);

    // disable threshold text if is enabled
    if (pPStextThr->IsEnabled())
        pPStextThr->Enable(false);
}

void wxStfCursorsDlg::OnRadioPSThrBeg( wxCommandEvent& event ) {
    event.Skip();
    wxTextCtrl* pCursor1PS = (wxTextCtrl*)FindWindow(wxTEXT1PS);
    wxTextCtrl* pPStextThr = (wxTextCtrl*)FindWindow(wxSLOPE_FROM_PSLOPE);
    if (pCursor1PS == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::OnRadioThrBeg()"));
        return;
    }
    // disable cursor wxTextCtrl if it is enabled 
    if (pCursor1PS->IsEnabled()) 
        pCursor1PS->Enable(false);

    // enable threshold text if is not enabled
    if (!pPStextThr->IsEnabled())
        pPStextThr->Enable(true);
}

void wxStfCursorsDlg::OnRadioPSt50Beg( wxCommandEvent& event ) {
    event.Skip();
    wxTextCtrl* pCursor1PS = (wxTextCtrl*)FindWindow(wxTEXT1PS);
    wxTextCtrl* pPStextThr = (wxTextCtrl*)FindWindow(wxSLOPE_FROM_PSLOPE);
    if (pCursor1PS == NULL || pPStextThr == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::OnRadioPSt50Beg()"));
        return;
    }
    // disable cursor wxTextCtrl if it is enabled 
    if (pCursor1PS->IsEnabled())
        pCursor1PS->Enable(false);

    // disable threshold text if is enabled
    if (pPStextThr->IsEnabled())
        pPStextThr->Enable(false);
}

void wxStfCursorsDlg::OnRadioPSManEnd( wxCommandEvent& event ) {
    event.Skip();
    wxTextCtrl* pCursor2PS    = (wxTextCtrl*)FindWindow(wxTEXT2PS);
    wxTextCtrl* pTextPSDeltaT = (wxTextCtrl*)FindWindow(wxTEXT_PSDELTAT);
    if (pCursor2PS == NULL || pTextPSDeltaT == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::OnRadioManEnd()"));
        return;
    }
    // if cursor wxTextCtrl is NOT enabled 
    if (!pCursor2PS->IsEnabled())
        pCursor2PS->Enable(true);
    
    if (pTextPSDeltaT->IsEnabled())
        pTextPSDeltaT->Enable(false);

//    SetPSlopeEndMode(stf::psEnd_manualMode);
}

void wxStfCursorsDlg::OnRadioPSt50End( wxCommandEvent& event ) {
    event.Skip();
    wxTextCtrl* pCursor2PS = (wxTextCtrl*)FindWindow(wxTEXT2PS);
    wxTextCtrl* pTextPSDeltaT = (wxTextCtrl*)FindWindow(wxTEXT_PSDELTAT);
    if (pCursor2PS == NULL || pTextPSDeltaT == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::OnRadioPSt50End()"));
        return;
    }
    // disable cursor wxTextCtrl if it is enabled 
    if (pCursor2PS->IsEnabled())
        pCursor2PS->Enable(false);

    if (pTextPSDeltaT->IsEnabled())
        pTextPSDeltaT->Enable(false);

    SetPSlopeEndMode(stf::psEnd_t50Mode);
}

void wxStfCursorsDlg::OnRadioPSDeltaT( wxCommandEvent& event) {
    event.Skip();
    wxTextCtrl* pCursor2PS = (wxTextCtrl*)FindWindow(wxTEXT2PS);
    wxTextCtrl* pTextPSDeltaT = (wxTextCtrl*)FindWindow(wxTEXT_PSDELTAT);
    if (pCursor2PS == NULL || pTextPSDeltaT == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::OnRadioPSDeltaT"));
        return;
    }

    // disable cursor wxTextCtrl if it is enabled
    if (pCursor2PS->IsEnabled())
        pCursor2PS->Enable(false);

    // enable text control
    if (!pTextPSDeltaT->IsEnabled())
        pTextPSDeltaT->Enable(true);

    SetPSlopeEndMode(stf::psEnd_DeltaTMode);
}

void wxStfCursorsDlg::OnRadioPSPeakEnd( wxCommandEvent& event ) {
    event.Skip();
    wxTextCtrl* pCursor2PS = (wxTextCtrl*)FindWindow(wxTEXT2PS);
    wxTextCtrl* pTextPSDeltaT = (wxTextCtrl*)FindWindow(wxTEXT_PSDELTAT);
    if (pCursor2PS == NULL || pTextPSDeltaT == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxCursorsDlg::OnRadioPeakEnd()"));
        return;
    }
    // disable cursor wxTextCtrl if is enabled
    if (pCursor2PS->IsEnabled())
        pCursor2PS->Enable(false);

    if (pTextPSDeltaT->IsEnabled())
        pTextPSDeltaT->Enable(false);

    SetPSlopeEndMode(stf::psEnd_peakMode);
}

#endif // WITH_PSLOPE

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

stf::latency_mode wxStfCursorsDlg::GetLatencyStartMode() const {

    wxRadioButton* pManual   = (wxRadioButton*)FindWindow(wxRADIO_LAT_MANUAL1);
    wxRadioButton* pPeak     = (wxRadioButton*)FindWindow(wxRADIO_LAT_PEAK1);
    wxRadioButton* pMaxSlope = (wxRadioButton*)FindWindow(wxRADIO_LAT_MAXSLOPE1);
    wxRadioButton* pt50      = (wxRadioButton*)FindWindow(wxRADIO_LAT_HALFWIDTH1);

    if (pManual == NULL || pPeak == NULL
            || pMaxSlope == NULL || pt50 == NULL ) {
        wxGetApp().ErrorMsg(wxT("Null pointer in wxCursorsDlg::GetLatencyStartMode()"));
        return stf::undefinedMode;
    }

    if (pManual->GetValue() )
        return stf::manualMode;
    else if (pPeak->GetValue())
        return stf::peakMode;
    else if (pMaxSlope->GetValue())
        return stf::riseMode;
    else if (pt50->GetValue())
        return stf::halfMode;
    else
        return stf::undefinedMode;

}

stf::latency_mode wxStfCursorsDlg::GetLatencyEndMode() const {

    wxRadioButton* pEvent    = (wxRadioButton*)FindWindow(wxRADIO_LAT_EVENT2);
    wxRadioButton* pManual   = (wxRadioButton*)FindWindow(wxRADIO_LAT_MANUAL2);
    wxRadioButton* pPeak     = (wxRadioButton*)FindWindow(wxRADIO_LAT_PEAK2);
    wxRadioButton* pMaxSlope = (wxRadioButton*)FindWindow(wxRADIO_LAT_MAXSLOPE2);
    wxRadioButton* pt50      = (wxRadioButton*)FindWindow(wxRADIO_LAT_HALFWIDTH2);

    if (pEvent == NULL || pManual == NULL || pPeak == NULL
            || pMaxSlope == NULL || pt50 == NULL ) {
        wxGetApp().ErrorMsg(wxT("Null pointer in wxCursorsDlg::GetLatencyEndMode()"));
        return stf::undefinedMode;
    }

    if (pManual->GetValue() ) {
        return stf::manualMode;
    } else {
        if (pEvent->GetValue()) {
            return stf::footMode;
        } else {
            if (pPeak->GetValue()) {
                return stf::peakMode;
            } else {
                if (pMaxSlope->GetValue()) {
                    return stf::riseMode;
                } else {
                    if (pt50->GetValue()) {
                        return stf::halfMode;
                    } else {
                        return stf::undefinedMode;
                    }
                }
            }
        }
    }
}

void wxStfCursorsDlg::SetLatencyStartMode(stf::latency_mode latencyBegMode){

    wxRadioButton* pManual   = (wxRadioButton*)FindWindow(wxRADIO_LAT_MANUAL1);
    wxRadioButton* pPeak     = (wxRadioButton*)FindWindow(wxRADIO_LAT_PEAK1);
    wxRadioButton* pMaxSlope = (wxRadioButton*)FindWindow(wxRADIO_LAT_MAXSLOPE1);
    wxRadioButton* pt50      = (wxRadioButton*)FindWindow(wxRADIO_LAT_HALFWIDTH1);
    
    if (pManual == NULL || pPeak == NULL
        || pMaxSlope == NULL || pt50 == NULL) {
        wxGetApp().ErrorMsg(wxT("Null pointer in wxCursorsDlg::SetLatencyStartMode()"));
    }

    switch (latencyBegMode) {
        case stf::manualMode:
            pManual->SetValue(true);
            break;
        case stf::peakMode:
            pPeak->SetValue(true);
            break;
        case stf::riseMode:
            pMaxSlope->SetValue(true);
            break;
        case stf::halfMode:
            pt50->SetValue(true);
            break;
        default:
            break;
        }
}


void wxStfCursorsDlg::SetLatencyEndMode(stf::latency_mode latencyEndMode){

    wxRadioButton* pManual   = (wxRadioButton*)FindWindow(wxRADIO_LAT_MANUAL2);
    wxRadioButton* pPeak     = (wxRadioButton*)FindWindow(wxRADIO_LAT_PEAK2);
    wxRadioButton* pMaxSlope = (wxRadioButton*)FindWindow(wxRADIO_LAT_MAXSLOPE2);
    wxRadioButton* pt50      = (wxRadioButton*)FindWindow(wxRADIO_LAT_HALFWIDTH2);
    wxRadioButton* pEvent    = (wxRadioButton*)FindWindow(wxRADIO_LAT_EVENT2);
    
    if (pManual == NULL || pPeak == NULL
        || pMaxSlope == NULL || pt50 == NULL || pEvent == NULL) {
        wxGetApp().ErrorMsg(wxT("Null pointer in wxCursorsDlg::SetLatencyEndtMode()"));
    }

    switch (latencyEndMode) {
        case stf::manualMode:
            pManual->SetValue(true);
            break;
        case stf::peakMode:
            pPeak->SetValue(true);
            break;
        case stf::riseMode:
            pMaxSlope->SetValue(true);
            break;
        case stf::halfMode:
            pt50->SetValue(true);
            break;
        case stf::footMode:
            pEvent->SetValue(true);
            break;
        default:
            break;
        }
}
void wxStfCursorsDlg::SetPeak4Latency(int val){

    wxCheckBox* pUsePeak = (wxCheckBox*)FindWindow(wxLATENCYWINDOW);
    if (pUsePeak == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxStfCursorsDlg::SetUsePeak4Latency()"));
        return;
    }

    pUsePeak->SetValue(val);
}

bool wxStfCursorsDlg::UsePeak4Latency() const
{   
    wxCheckBox* pUsePeak = (wxCheckBox*)FindWindow(wxLATENCYWINDOW);
    if (pUsePeak == NULL) {
        wxGetApp().ErrorMsg(wxT("null pointer in wxStfCursorsDlg::GetUsePeak4Latency()"));
        return false;
    }
    return pUsePeak->IsChecked(); 
    
}


#ifdef WITH_PSLOPE
stf::pslope_mode_beg wxStfCursorsDlg::GetPSlopeBegMode() const {

    wxRadioButton* pPSManBeg   = (wxRadioButton*)FindWindow(wxRADIO_PSManBeg);
    wxRadioButton* pPSEventBeg = (wxRadioButton*)FindWindow(wxRADIO_PSEventBeg);
    wxRadioButton* pPSThrBeg   = (wxRadioButton*)FindWindow(wxRADIO_PSThrBeg);
    wxRadioButton* pPSt50Beg   = (wxRadioButton*)FindWindow(wxRADIO_PSt50Beg);

    if (pPSManBeg == NULL || pPSEventBeg == NULL 
            || pPSThrBeg == NULL || pPSt50Beg == NULL) {
        wxGetApp().ErrorMsg(wxT("Null pointer in wxCursorsDlg::OnRadioPSBeg()"));
        return stf::psBeg_undefined;
    }

    if ( pPSManBeg->GetValue() )
        return stf::psBeg_manualMode;
    else if ( pPSEventBeg->GetValue() )
        return stf::psBeg_footMode;
    else  if( pPSThrBeg->GetValue() )
        return stf::psBeg_thrMode;
    else if ( pPSt50Beg->GetValue() )
        return stf::psBeg_t50Mode;
    else
        return stf::psBeg_undefined;
}

stf::pslope_mode_end wxStfCursorsDlg::GetPSlopeEndMode() const {
  
    wxRadioButton* pPSManEnd   = (wxRadioButton*)FindWindow(wxRADIO_PSManEnd);
    wxRadioButton* pPSt50End = (wxRadioButton*)FindWindow(wxRADIO_PSt50End);
    wxRadioButton* pPSDeltaT   = (wxRadioButton*)FindWindow(wxRADIO_PSDeltaT);
    wxRadioButton* pPSPeakEnd   = (wxRadioButton*)FindWindow(wxRADIO_PSPeakEnd);

    if (pPSManEnd == NULL || pPSt50End == NULL 
            || pPSDeltaT == NULL || pPSPeakEnd == NULL) {
        wxGetApp().ErrorMsg(wxT("Null pointer in wxCursorsDlg::GetPSlopeEndMode()"));
    }

    if ( pPSManEnd->GetValue() )
        return stf::psEnd_manualMode;
    else if ( pPSt50End->GetValue() )
        return stf::psEnd_t50Mode;
    else  if( pPSDeltaT->GetValue() )
        return stf::psEnd_DeltaTMode;
    else if ( pPSPeakEnd->GetValue() )
        return stf::psEnd_peakMode;
    else
        return stf::psEnd_undefined;
//   return dlgPSlopeModeEnd;
}

void wxStfCursorsDlg::SetPSlopeEndMode(stf::pslope_mode_end pslopeEndMode) {
 
    wxRadioButton* pPSManBeg  = (wxRadioButton*)FindWindow(wxRADIO_PSManBeg);
    wxRadioButton* pPSEventBeg  = (wxRadioButton*)FindWindow(wxRADIO_PSEventBeg);
    wxRadioButton* pPSThrBeg = (wxRadioButton*)FindWindow(wxRADIO_PSThrBeg);
    wxRadioButton* pPSt50Beg   = (wxRadioButton*)FindWindow(wxRADIO_PSThrBeg);

    if (pPSManBeg == NULL || pPSEventBeg == NULL 
        || pPSThrBeg == NULL || pPSt50Beg == NULL){
        wxGetApp().ErrorMsg(wxT("Null pointer in wxCursorsDlg::SetPSlopeEndMode()"));
        return;
    }

    switch (pslopeEndMode) {
    case stf::psBeg_manualMode:
        pPSManBeg->Enable(true);
        break;
    case stf::psBeg_footMode:
        pPSEventBeg->Enable(true);
        break;
    case stf::psBeg_thrMode:
        pPSThrBeg->Enable(true);
        break;
    case stf::psBeg_t50Mode:
        pPSt50Beg->Enable(true);
    default:
        break;
    
}


#ifdef _STFDEBUG
    std::cout << "wxStfCursorsDlg: PSlopeEndMode is " << pslopeEndMode << std::endl;
#endif
}

void wxStfCursorsDlg::SetPSlopeBegMode(stf::pslope_mode_beg pslopeBegMode) {

    wxRadioButton* pPSManBeg  = (wxRadioButton*)FindWindow(wxRADIO_PSManBeg);
    wxRadioButton* pPSEventBeg  = (wxRadioButton*)FindWindow(wxRADIO_PSEventBeg);
    wxRadioButton* pPSThrBeg = (wxRadioButton*)FindWindow(wxRADIO_PSThrBeg);
    wxRadioButton* pPSt50Beg   = (wxRadioButton*)FindWindow(wxRADIO_PSThrBeg);

    if (pPSManBeg == NULL || pPSEventBeg == NULL 
        || pPSThrBeg == NULL || pPSt50Beg == NULL){
        wxGetApp().ErrorMsg(wxT("Null pointer in wxCursorsDlg::SetPSlopeBegMode()"));
        return;
    }

    switch (pslopeBegMode) {
    case stf::psBeg_manualMode:
        pPSManBeg->Enable(true);
        break;
    case stf::psBeg_footMode:
        pPSEventBeg->Enable(true);
        break;
    case stf::psBeg_thrMode:
        pPSThrBeg->Enable(true);
        break;
    case stf::psBeg_t50Mode:
        pPSt50Beg->Enable(true);
    default:
        break;
    }
}

#endif // WITH_PSLOPE

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
        int iNewValue = stf::round(fEntry/actDoc->GetXScale());
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

    case stf::latency_cursor: // Latency
        iNewValue1= (int)actDoc->GetLatencyBeg();
        iNewValue2= (int)actDoc->GetLatencyEnd();
        cursor1isTime=cursor1LIsTime;
        cursor2isTime=cursor2LIsTime;

        // if GetLatencyStartmode() is zero, textbox is enabled
        pText1=(wxTextCtrl*)FindWindow(wxTEXT1L);
        pText1->Enable(!actDoc->GetLatencyStartMode()); 

        // if GetLatencyEndmode() is zero, textbox is enabled
        pText2=(wxTextCtrl*)FindWindow(wxTEXT2L);
        pText2->Enable(!actDoc->GetLatencyEndMode());

        // use peak for latency measurements?
        SetPeak4Latency ( actDoc->GetLatencyWindowMode() );

        // Update RadioButton options
        SetLatencyStartMode( actDoc->GetLatencyStartMode() );
        SetLatencyEndMode(   actDoc->GetLatencyEndMode() );
        break;

#ifdef WITH_PSLOPE
    case stf::pslope_cursor: // PSlope
        iNewValue1=(int)actDoc->GetPSlopeBeg();
        iNewValue2=(int)actDoc->GetPSlopeEnd();
        cursor1isTime=cursor1PSIsTime;
        cursor2isTime=cursor2PSIsTime;
        pText1=(wxTextCtrl*)FindWindow(wxTEXT1PS);
        pText2=(wxTextCtrl*)FindWindow(wxTEXT2PS);
        // Update PSlope Beg and End options
        SetPSlopeBegMode( actDoc->GetPSlopeBegMode() );
        SetPSlopeEndMode( actDoc->GetPSlopeEndMode() );
        SetDeltaT( actDoc->GetDeltaT() );
        break;
#endif

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
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
    slopeUnits += actDoc->at(actDoc->GetCurCh()).GetYUnits();
    slopeUnits += wxT("/");
    slopeUnits += actDoc->GetXUnits();
#else
    slopeUnits += wxString(actDoc->at(actDoc->GetCurCh()).GetYUnits().c_str(), wxConvUTF8);
    slopeUnits += wxT("/");
    slopeUnits += wxString(actDoc->GetXUnits().c_str(), wxConvUTF8);
#endif    
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
    case 4: return stf::latency_cursor;
#ifdef WITH_PSLOPE
    case 5: return stf::pslope_cursor;
#endif 
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

#ifdef WITH_PSLOPE
    wxTextCtrl* pSlope2 = (wxTextCtrl*)FindWindow(wxSLOPE_FROM_PSLOPE);
#endif 

    wxString wxsSlope;
    wxsSlope << fSlope;
    if ( pSlope != NULL ) {
        pSlope->SetValue( wxsSlope );

#ifdef WITH_PSLOPE
        pSlope2->SetValue( wxsSlope );
#endif 
    }
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
