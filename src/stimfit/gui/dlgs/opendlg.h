#ifndef STIMFIT_OPENDLG_H
#define STIMFIT_OPENDLG_H

#include <wx/dialog.h>
#include <wx/filectrl.h>
#include <wx/timer.h>
#include <map>
#include <vector>
#include "../../../libstfio/stfio.h"

class wxChoice;
class wxCheckBox;
class wxStaticText;
class wxTextCtrl;

//! File browser with reader options for the selected recording.
class wxStfOpenDlg : public wxDialog {
public:
    wxStfOpenDlg(wxWindow* parent, const wxString& directory, const wxString& wildcard);
    wxString GetPath() const { return candidate; }
    stfio::fileReader GetReader() const;
    bool RememberChoice() const;
    bool SetPath(const wxString& path);

private:
    struct Selection {
        stfio::fileReader reader;
        bool remember;
    };
    wxFileCtrl* files;
    wxTextCtrl* location;
    wxChoice* choice;
    wxCheckBox* remember;
    wxStaticText* filename;
    wxStaticText* explanation;
    wxTimer probeTimer;
    wxString candidate, inspected;
    std::vector<stfio::fileReader> readers;
    std::map<wxString, Selection> selections;
    bool multipleReaders = false;
    bool canOpen = false;

    void SelectPath(const wxString& path);
    void InspectSelection();
    void SaveSelection();
    void AcceptSelection();
    void ChangeLocation();
    void OnSelection(wxFileCtrlEvent& event);
    void OnFolder(wxFileCtrlEvent& event);
    void OnActivate(wxFileCtrlEvent& event);
};

#endif
