#include "wx/wxprec.h"

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include <wx/stdpaths.h>
#include <wx/dir.h>
#include <wx/checkbox.h>

#include "convertdlg.h"
#include "./../app.h"

enum {
    wxCOMBOBOX_SRC,
    wxCOMBOBOX_DEST,
    wxGENERICDIRCTRL_SRC,
    wxGENERICDIRCTRL_DEST
};

BEGIN_EVENT_TABLE( wxStfConvertDlg, wxDialog )
EVT_COMBOBOX( wxCOMBOBOX_SRC,  wxStfConvertDlg::OnComboBoxSrcExt)
EVT_COMBOBOX( wxCOMBOBOX_DEST, wxStfConvertDlg::OnComboBoxDestExt)
END_EVENT_TABLE()

// wxStfConvertDlg constructor 
wxStfConvertDlg::wxStfConvertDlg(wxWindow* parent, int id, wxString title, wxPoint pos,
        wxSize size, int style)
: wxDialog( parent, id, title, pos, size, style ),
    srcDir(wxT("")),
    destDir(wxT("")),
    srcFilter(wxT("")), myCheckBoxSubdirs(NULL),
    srcFilterExt(stfio::cfs), destFilterExt(stfio::igor),
    srcFileNames(0)

{
    if (srcDir == wxT("")) {
        srcDir = wxGetApp().wxGetProfileString(
            wxT("Settings"), wxT("Most recent batch source directory"), wxT(""));
        if (srcDir == wxT("") || !wxFileName::DirExists(srcDir)) {
            srcDir = wxStandardPaths::Get().GetDocumentsDir();
        }
    }

    if (destDir == wxT("")) {
        destDir = wxGetApp().wxGetProfileString(
            wxT("Settings"), wxT("Most recent batch target directory"), wxT(""));
        if (destDir == wxT("") || !wxFileName::DirExists(destDir)) {
            destDir = wxStandardPaths::Get().GetDocumentsDir();
        }
    }

    wxBoxSizer* topSizer;
    topSizer = new wxBoxSizer( wxVERTICAL );

    //wxFlexGridSizer *gridSizer; 
    //gridSizer = new wxFlexGridSizer(2,2,0,10);

    wxFlexGridSizer *gridSizer; 
    gridSizer = new wxFlexGridSizer(1,2,0,0);

    // SOURCE dir ------------------------------------------------------
    // wxFlexGridSizer to place a 1) combo + 2) directory listing
    wxFlexGridSizer *myLeftSizer; // this is a sizer for the left side 
    myLeftSizer = new wxFlexGridSizer(3, 1, 0, 0);

    // SOURCE 1.- wxComboBox to select the source file extension
    wxFlexGridSizer *mySrcComboSizer; // a sizer for my Combo
    mySrcComboSizer = new wxFlexGridSizer(1, 2, 0, 0); 

    wxStaticText* staticTextExt;
    staticTextExt = new wxStaticText( this, wxID_ANY, wxT("Origin filetype:"),
            wxDefaultPosition, wxDefaultSize, 0 );

    wxArrayString myextensions; 
    myextensions.Add(wxT("CFS binary    [*.dat ]"));
    myextensions.Add(wxT("Axon binary   [*.abf ]"));
    myextensions.Add(wxT("Axograph      [*.axgd]"));
    myextensions.Add(wxT("Axon textfile [*.atf ]"));
    myextensions.Add(wxT("ASCII         [*.*   ]"));
    myextensions.Add(wxT("HDF5          [*.h5  ]"));
    myextensions.Add(wxT("HEKA files    [*.dat ]"));
#if (BIOSIG_VERSION >= 10404)
    myextensions.Add(wxT("Igor files    [*.ibw ]"));
#endif

    wxComboBox* myComboBoxExt;
    myComboBoxExt = new wxComboBox(this, wxCOMBOBOX_SRC, myextensions[0], 
        wxDefaultPosition, wxDefaultSize, myextensions, wxCB_READONLY);
    // add to mySrcComboSizer
    mySrcComboSizer->Add( staticTextExt, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    mySrcComboSizer->Add( myComboBoxExt, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    // add to myLeftSizer
    myLeftSizer->Add( mySrcComboSizer, 0, wxEXPAND | wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    // ---- wxComboBox to select the source file extension

    
    // SOURCE 2.- A wxGenericDirCtrl to select the source directory:

    //wxGenericDirCtrl *mySrcDirCtrl; 
    mySrcDirCtrl = new wxGenericDirCtrl(this, wxGENERICDIRCTRL_SRC, srcDir,
        wxDefaultPosition, wxSize(300,300), wxDIRCTRL_DIR_ONLY);
    // add to myLeftSizer
    myLeftSizer->Add( mySrcDirCtrl, 0, wxEXPAND | wxALL , 2 );
    // ---- A wxGenericDirCtrl to select the source directory:

    myCheckBoxSubdirs = new wxCheckBox(
                this,
                wxID_ANY,
                wxT("Include subdirectories"),
                wxDefaultPosition,
                wxDefaultSize,
                0
        );
    myCheckBoxSubdirs->SetValue(false);
    myLeftSizer->Add( myCheckBoxSubdirs, 0, wxALIGN_LEFT | wxALL, 2 );

    // Finally add myLeftSizer to the gridSizer
    gridSizer->Add( myLeftSizer, 0, wxALIGN_LEFT, 5 );
    //topSizer->Add( gridSizer, 0, wxALIGN_CENTER, 5 );
    
    // DESTINATION dir ----------------------------------------------------------
    // wxFlexGridSizer to place a 1) combo + 2) directory listing
    wxFlexGridSizer *myRightSizer; // this is a sizer for the right side
    myRightSizer = new wxFlexGridSizer(2, 1, 0, 0);

    
    // DESTINATION 1.- wxComboBox to select the destiny file extension
    wxFlexGridSizer *myDestComboSizer; // a sizer for my Combo
    myDestComboSizer = new wxFlexGridSizer(1, 2, 0, 0); 

    wxStaticText* staticTextDestExt;
    staticTextDestExt = new wxStaticText( this, wxID_ANY, wxT("Destination filetype:"),
            wxDefaultPosition, wxDefaultSize, 0 );

    wxArrayString mydestextensions; //ordered by importance 
    mydestextensions.Add(wxT("Igor binary   [*.ibw ]"));
    mydestextensions.Add(wxT("Axon textfile [*.atf ]"));
#if (defined(WITH_BIOSIG) || defined(WITH_BIOSIG2))
    mydestextensions.Add(wxT("GDF (Biosig) [*.gdf ]"));
#endif


    wxComboBox* myComboBoxDestExt;
    myComboBoxDestExt = new wxComboBox(this, wxCOMBOBOX_DEST, mydestextensions[0], 
        wxDefaultPosition, wxDefaultSize, mydestextensions, wxCB_READONLY);
    // add to mySrcComboSizer
    myDestComboSizer->Add( staticTextDestExt, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    myDestComboSizer->Add( myComboBoxDestExt, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    // add to myRightSizer
    myRightSizer->Add( myDestComboSizer, 0, wxEXPAND | wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL | wxALL, 2 );
    // ---- wxComboBox to select the source file extension

    // DESTINATION 2.- A wxGenericDirCtrl to select the destiny directory:

    myDestDirCtrl = new wxGenericDirCtrl(this, wxGENERICDIRCTRL_DEST, destDir,
        wxDefaultPosition, wxSize(300,300), wxDIRCTRL_DIR_ONLY);
    // add to myLeftSizer
    myRightSizer->Add( myDestDirCtrl, 0, wxEXPAND | wxALL, 2 );
    // ---- A wxGenericDirCtrl to select the source directory:

    // Finally add myRightSizer to gridSizer and this to topSizer
    gridSizer->Add( myRightSizer, 0, wxALIGN_RIGHT, 5);
    topSizer->Add( gridSizer, 0, wxALIGN_CENTER, 5 );


    // OK / Cancel buttons-----------------------------------------------
    wxStdDialogButtonSizer* sdbSizer = new wxStdDialogButtonSizer();
    wxButton *myConvertButton;
    myConvertButton = new wxButton( this, wxID_OK, wxT("C&onvert"));
    // this for wxWidgets 2.9.1
    //myConvertButton->SetBitmap(wxBitmap(wxT("icon_cross.png"), wxBITMAP_TYPE_PNG));

    sdbSizer->AddButton(myConvertButton);
    sdbSizer->AddButton( new wxButton( this, wxID_CANCEL ) );
    sdbSizer->Realize();
    topSizer->Add( sdbSizer, 0, wxALIGN_CENTER | wxALL, 5 );

    topSizer->SetSizeHints(this);
    this->SetSizer( topSizer );

    this->Layout();
}
void wxStfConvertDlg::OnComboBoxDestExt(wxCommandEvent& event){
    event.Skip();

    wxComboBox* pComboBox = (wxComboBox*)FindWindow(wxCOMBOBOX_DEST);
    if (pComboBox == NULL) {
        wxGetApp().ErrorMsg(wxT("Null pointer in wxStfConvertDlg::OnComboBoxDestExt()"));
        return;
    }
    // update destFilterExt 
    switch(pComboBox->GetSelection()){
        case 0:
            destFilterExt =  stfio::igor;
            break;
        case 1:
            destFilterExt = stfio::atf;
            break;
#if (defined(WITH_BIOSIG) || defined(WITH_BIOSIG2))
        case 2:
            destFilterExt = stfio::biosig;
            break;
#endif
        default:
            destFilterExt = stfio::igor;
    }


}

void wxStfConvertDlg::OnComboBoxSrcExt(wxCommandEvent& event){

    event.Skip();
    wxComboBox* pComboBox = (wxComboBox*)FindWindow(wxCOMBOBOX_SRC);
    if (pComboBox == NULL) {
        wxGetApp().ErrorMsg(wxT("Null pointer in wxStfConvertDlg::OnComboBoxSrcExt()"));
        return;
    }

    // update srcFilterExt and srcFilter
    // see index of wxArrayString myextensions to evaluate case
    switch(pComboBox->GetSelection()){
        case 0:
            srcFilterExt =  stfio::cfs;
            break;
        case 1:
            srcFilterExt =  stfio::abf;
            break;
        case 2:
            srcFilterExt = stfio::axg;
            break;
        case 3: 
            srcFilterExt =  stfio::atf;
            break;
        case 4: 
            break;
        case 5: 
            srcFilterExt =  stfio::hdf5;
            break;
        case 6: 
            srcFilterExt =  stfio::heka;
            break;
#if (BIOSIG_VERSION >= 10404)
        case 7:
            srcFilterExt =  stfio::igor;
            break;
#endif
        default:   
            srcFilterExt =  stfio::none;
    }
    srcFilter = "*" + stfio::findExtension(srcFilterExt);

}

void wxStfConvertDlg::EndModal(int retCode) {
    // similar to overriding OnOK in MFC (I hope...)
    if (retCode==wxID_OK) {
        if (!OnOK()) {
            return;
        }
    }
    wxDialog::EndModal(retCode);
}

bool wxStfConvertDlg::OnOK() {

    srcDir  = mySrcDirCtrl->GetPath();
    destDir = myDestDirCtrl->GetPath();

    if (!wxDir::Exists(srcDir)) {
        wxString msg;
        msg << srcDir << wxT(" doesn't exist");
        wxLogMessage(msg);
        return false;
    }
    if (!wxDir::Exists(destDir)) {
        wxString msg;
        msg << destDir << wxT(" doesn't exist");
        wxLogMessage(msg);
        return false;
    }

    if (!ReadPath(srcDir)) {
        wxString msg;
        msg << srcFilter << wxT(" not found in ") << srcDir;
        wxLogMessage(msg);
        return false;
    }

    wxGetApp().wxWriteProfileString(
        wxT("Settings"), wxT("Most recent batch source directory"), srcDir);

    wxGetApp().wxWriteProfileString(
        wxT("Settings"), wxT("Most recent batch target directory"), destDir);

    return true;
}

bool wxStfConvertDlg::ReadPath(const wxString& path) {
    // Walk through path:
    wxDir dir(path);

    if ( !dir.IsOpened() )
    {
        return false;
    }

    if (!dir.HasFiles(srcFilter)) {
        return false;
    }

    int dir_flags = myCheckBoxSubdirs->IsChecked() ?
        wxDIR_FILES | wxDIR_DIRS | wxDIR_HIDDEN :
        wxDIR_FILES | wxDIR_HIDDEN;

    wxDir::GetAllFiles(path, &srcFileNames, srcFilter, dir_flags);
    return true;
}
