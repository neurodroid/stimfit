#include <wx/wx.h>
#include <wx/filename.h>
#include "../../stf.h"
#include "../app.h"
#include "opendlg.h"

wxStfOpenDlg::wxStfOpenDlg(wxWindow* parent, const wxString& directory, const wxString& wildcard)
    : wxDialog(parent, wxID_ANY, wxT("Open"), wxDefaultPosition, wxDefaultSize,
               wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER), probeTimer(this) {
    auto* layout = new wxBoxSizer(wxVERTICAL);
    auto* navigation = new wxBoxSizer(wxHORIZONTAL);
    navigation->Add(new wxStaticText(this, wxID_ANY, wxT("Location:")),
                    0, wxALIGN_CENTER_VERTICAL | wxRIGHT, FromDIP(8));
    location = new wxTextCtrl(this, wxID_ANY, directory, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER);
    navigation->Add(location, 1, wxRIGHT, FromDIP(8));
    auto* up = new wxButton(this, wxID_ANY, wxT("Up"));
    auto* home = new wxButton(this, wxID_ANY, wxT("Home"));
    navigation->Add(up, 0, wxRIGHT, FromDIP(8));
    navigation->Add(home);
    layout->Add(navigation, 0, wxEXPAND | wxALL, FromDIP(12));

    auto* content = new wxBoxSizer(wxHORIZONTAL);
    files = new wxFileCtrl(this, wxID_ANY, directory, wxEmptyString, wildcard, wxFC_OPEN);
    content->Add(files, 1, wxEXPAND | wxRIGHT, FromDIP(16));
    auto* options = new wxBoxSizer(wxVERTICAL);
    auto* heading = new wxStaticText(this, wxID_ANY, wxT("File reader"));
    heading->SetFont(heading->GetFont().Bold());
    options->Add(heading, 0, wxBOTTOM, FromDIP(12));
    filename = new wxStaticText(this, wxID_ANY, wxT("No file selected"),
        wxDefaultPosition, wxSize(FromDIP(280), -1), wxST_ELLIPSIZE_MIDDLE);
    options->Add(filename, 0, wxEXPAND | wxBOTTOM, FromDIP(12));
    choice = new wxChoice(this, wxID_ANY);
    options->Add(choice, 0, wxEXPAND | wxBOTTOM, FromDIP(12));
    explanation = new wxStaticText(this, wxID_ANY, wxEmptyString,
        wxDefaultPosition, wxSize(FromDIP(280), FromDIP(110)));
    options->Add(explanation, 0, wxEXPAND | wxBOTTOM, FromDIP(12));
    remember = new wxCheckBox(this, wxID_ANY, wxT("Remember for this file"));
    options->Add(remember, 0, wxEXPAND);
    options->SetMinSize(FromDIP(280), -1);
    content->Add(options, 0, wxEXPAND);
    layout->Add(content, 1, wxEXPAND | wxLEFT | wxRIGHT, FromDIP(12));
    auto* buttons = CreateSeparatedButtonSizer(wxOK | wxCANCEL);
    static_cast<wxButton*>(FindWindow(wxID_OK))->SetLabel(wxT("Open"));
    layout->Add(buttons, 0, wxEXPAND | wxALL, FromDIP(12));
    SetSizer(layout);
    SetMinSize(FromDIP(wxSize(760, 460)));
    SetSize(FromDIP(wxSize(1040, 660)));
    CentreOnParent();

    files->Bind(wxEVT_FILECTRL_SELECTIONCHANGED, &wxStfOpenDlg::OnSelection, this);
    files->Bind(wxEVT_FILECTRL_FOLDERCHANGED, &wxStfOpenDlg::OnFolder, this);
    files->Bind(wxEVT_FILECTRL_FILEACTIVATED, &wxStfOpenDlg::OnActivate, this);
    Bind(wxEVT_TIMER, [this](wxTimerEvent&) { InspectSelection(); });
    Bind(wxEVT_BUTTON, [this](wxCommandEvent&) { AcceptSelection(); }, wxID_OK);
    location->Bind(wxEVT_TEXT_ENTER, [this](wxCommandEvent&) { ChangeLocation(); });
    location->Bind(wxEVT_TEXT, [this](wxCommandEvent&) { SelectPath(wxEmptyString); });
    up->Bind(wxEVT_BUTTON, [this](wxCommandEvent&) {
        wxFileName folder = wxFileName::DirName(files->GetDirectory());
        if (folder.GetDirCount() == 0) return;
        folder.RemoveLastDir();
        files->SetDirectory(folder.GetPath());
        location->ChangeValue(folder.GetPath());
        SelectPath(wxEmptyString);
    });
    home->Bind(wxEVT_BUTTON, [this](wxCommandEvent&) {
        files->SetDirectory(wxGetHomeDir());
        location->ChangeValue(wxGetHomeDir());
        SelectPath(wxEmptyString);
    });
    wxAcceleratorEntry entry(wxACCEL_CMD, 'L', wxID_JUMP_TO);
    SetAcceleratorTable(wxAcceleratorTable(1, &entry));
    Bind(wxEVT_MENU, [this](wxCommandEvent&) { location->SetFocus(); location->SelectAll(); }, wxID_JUMP_TO);
    SelectPath(wxEmptyString);
}

stfio::fileReader wxStfOpenDlg::GetReader() const {
    if (!multipleReaders || readers.empty()) return stfio::fileReader::automatic;
    return readers.at(choice->GetSelection());
}

bool wxStfOpenDlg::RememberChoice() const {
    return multipleReaders && remember->GetValue();
}

bool wxStfOpenDlg::SetPath(const wxString& path) {
    if (!wxFileName::FileExists(path)) return false;
    files->SetPath(path);
    location->ChangeValue(path);
    SelectPath(path);
    return true;
}

void wxStfOpenDlg::SaveSelection() {
    if (multipleReaders && !inspected.empty())
        selections[inspected] = {GetReader(), remember->GetValue()};
}

void wxStfOpenDlg::SelectPath(const wxString& path) {
    if (candidate == path && choice->GetCount()) return;
    SaveSelection();
    probeTimer.Stop();
    candidate = path;
    inspected.clear();
    multipleReaders = false;
    canOpen = false;
    readers.clear();
    choice->Clear();
    choice->Append(wxT("Automatic"));
    choice->SetSelection(0);
    choice->Disable();
    remember->SetValue(false);
    remember->Disable();
    FindWindow(wxID_OK)->Disable();
    const bool isFile = !path.empty() && wxFileName::FileExists(path);
    filename->SetLabel(isFile ? wxFileName(path).GetFullName() : wxT("No file selected"));
    explanation->SetLabel(isFile ? wxT("Checking available readers...")
                                : wxT("Select a file to see the available readers."));
    explanation->Wrap(FromDIP(280));
    if (isFile) probeTimer.StartOnce(150);
}

void wxStfOpenDlg::InspectSelection() {
    if (candidate.empty() || !wxFileName::FileExists(candidate)) return;
    try {
        auto info = stfio::getReaderInfo(stf::wx2std(candidate));
#ifdef WITH_PYTHON
        if (info.type == stfio::tdms) {
            info.nativeAvailable = true;
            info.recommended = stfio::fileReader::native;
        }
#endif
        inspected = candidate;
        multipleReaders = info.biosigAvailable && info.nativeAvailable;
        canOpen = info.biosigAvailable || info.nativeAvailable;
        choice->Clear();
        readers.clear();
        if (multipleReaders) {
            const wxString recommended = info.recommended == stfio::fileReader::biosig
                ? wxT("BioSig") : wxT("Native");
            choice->Append(wxT("Automatic (") + recommended + wxT(")"));
            choice->Append(wxT("BioSig"));
            choice->Append(wxT("Native (Stimfit)"));
            readers = {stfio::fileReader::automatic, stfio::fileReader::biosig, stfio::fileReader::native};
            const auto found = selections.find(candidate);
            const auto preference = wxGetApp().GetFileReaderPreference(candidate);
            const Selection selected = found == selections.end()
                ? Selection{preference, preference != stfio::fileReader::automatic} : found->second;
            choice->SetSelection(0);
            for (std::size_t n = 0; n < readers.size(); ++n)
                if (readers[n] == selected.reader) choice->SetSelection(static_cast<int>(n));
            remember->SetValue(selected.remember);
        } else {
            choice->Append(info.biosigAvailable ? wxT("BioSig")
                : info.nativeAvailable ? wxT("Native (Stimfit)") : wxT("No reader available"));
            choice->SetSelection(0);
        }
        choice->Enable(multipleReaders);
        remember->Enable(multipleReaders);
        wxString note;
        if (!canOpen) note = wxT("No supported reader is available for this file.");
        else if (!multipleReaders) note = wxT("This file is supported by a single reader.");
        else if (!info.reason.empty()) note = wxString::FromUTF8(info.reason.c_str());
        else note = wxT("Automatic prefers BioSig and checks for import problems.");
        explanation->SetLabel(note);
        explanation->Wrap(FromDIP(280));
        FindWindow(wxID_OK)->Enable(canOpen);
    } catch (const std::exception& error) {
        explanation->SetLabel(wxString::FromUTF8(error.what()));
        explanation->Wrap(FromDIP(280));
    }
}

void wxStfOpenDlg::AcceptSelection() {
    probeTimer.Stop();
    if (inspected != candidate) InspectSelection();
    if (canOpen && wxFileName::FileExists(candidate)) EndModal(wxID_OK);
}

void wxStfOpenDlg::ChangeLocation() {
    wxFileName path(location->GetValue());
    path.Normalize(wxPATH_NORM_DOTS | wxPATH_NORM_TILDE | wxPATH_NORM_ABSOLUTE,
                   files->GetDirectory());
    if (wxFileName::DirExists(path.GetFullPath())) {
        files->SetDirectory(path.GetFullPath());
        location->ChangeValue(path.GetFullPath());
        SelectPath(wxEmptyString);
    } else {
        SetPath(path.GetFullPath());
    }
}

void wxStfOpenDlg::OnSelection(wxFileCtrlEvent& event) {
    const auto selected = event.GetFiles();
    const wxString path = selected.size() == 1
        ? wxFileName(event.GetDirectory(), selected[0]).GetFullPath() : wxString();
    if (!path.empty()) location->ChangeValue(path);
    SelectPath(path);
}

void wxStfOpenDlg::OnFolder(wxFileCtrlEvent& event) {
    location->ChangeValue(event.GetDirectory());
    SelectPath(wxEmptyString);
}

void wxStfOpenDlg::OnActivate(wxFileCtrlEvent& event) {
    OnSelection(event);
    AcceptSelection();
}
