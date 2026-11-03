#include <wx/wx.h>
#include <wx/filename.h>
#include "readerdlg.h"

wxStfReaderDlg::wxStfReaderDlg(wxWindow* parent, const wxString& filename,
                             const stfio::ReaderInfo& info, stfio::fileReader preference)
    : wxDialog(parent, wxID_ANY, wxT("Open with reader")) {
    auto* layout = new wxBoxSizer(wxVERTICAL);
    layout->Add(new wxStaticText(this, wxID_ANY, wxFileName(filename).GetFullName()),
                0, wxALL, 12);
    const wxString recommended = info.recommended == stfio::fileReader::biosig
        ? wxT("BioSig") : wxT("Native");
    wxArrayString labels;
    labels.Add(wxT("Automatic (") + recommended + wxT(" for this file)"));
    readers.push_back(stfio::fileReader::automatic);
    if (info.biosigAvailable) {
        labels.Add(wxT("BioSig"));
        readers.push_back(stfio::fileReader::biosig);
    }
    if (info.nativeAvailable) {
        labels.Add(wxT("Native (Stimfit)"));
        readers.push_back(stfio::fileReader::native);
    }
    auto* row = new wxBoxSizer(wxHORIZONTAL);
    row->Add(new wxStaticText(this, wxID_ANY, wxT("Reader:")), 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 12);
    choice = new wxChoice(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, labels);
    choice->SetSelection(0);
    for (std::size_t n = 0; n < readers.size(); ++n)
        if (readers[n] == preference) choice->SetSelection(static_cast<int>(n));
    row->Add(choice, 1);
    layout->Add(row, 0, wxEXPAND | wxLEFT | wxRIGHT, 12);
    wxString explanation = wxT("Automatic prefers BioSig and checks for known import problems.");
    if (!info.biosigAvailable) explanation = wxT("Native is the available reader for this file.");
    else if (!info.nativeAvailable) explanation = wxT("BioSig is the available reader for this file.");
    if (!info.reason.empty()) explanation = wxString::FromUTF8(info.reason.c_str());
    auto* note = new wxStaticText(this, wxID_ANY, explanation);
    note->Wrap(440);
    layout->Add(note, 0, wxEXPAND | wxALL, 12);
    remember = new wxCheckBox(this, wxID_ANY, wxT("Remember this choice for this file"));
    remember->SetValue(preference != stfio::fileReader::automatic);
    layout->Add(remember, 0, wxLEFT | wxRIGHT | wxBOTTOM, 12);
    auto* buttons = CreateSeparatedButtonSizer(wxOK | wxCANCEL);
    static_cast<wxButton*>(FindWindow(wxID_OK))->SetLabel(wxT("Open"));
    layout->Add(buttons, 0, wxEXPAND | wxALL, 12);
    SetSizerAndFit(layout);
    CentreOnParent();
}

stfio::fileReader wxStfReaderDlg::GetReader() const {
    return readers.at(choice->GetSelection());
}

bool wxStfReaderDlg::RememberChoice() const {
    return remember->GetValue();
}
