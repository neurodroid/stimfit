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

// doc.cpp
// The document class, derived from both wxDocument and recording
// 2007-12-27, Christoph Schmidt-Hieber, University of Freiburg

// For compilers that support precompilation, includes "wx/wx.h".
#include <wx/wxprec.h>
#include <wx/progdlg.h>
#include <wx/filename.h>
#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#if !wxUSE_DOC_VIEW_ARCHITECTURE
#error You must set wxUSE_DOC_VIEW_ARCHITECTURE to 1 in setup.h!
#endif

#include "./app.h"
#include "./view.h"
#include "./parentframe.h"
#include "./childframe.h"
#include "./dlgs/smalldlgs.h"
#include "./dlgs/fitseldlg.h"
#include "./dlgs/eventdlg.h"
#include "./dlgs/cursorsdlg.h"
#include "./../core/fitlib.h"
#include "./../core/measlib.h"
#include "./../core/filelib/cfslib.h"
#include "./../core/filelib/atflib.h"
#include "./../core/filelib/hdf5lib.h"
#include "./../core/filelib/asciilib.h"
#ifdef _WINDOWS
#include "./../core/filelib/igorlib.h"
#endif
#include "./usrdlg/usrdlg.h"
#include "./doc.h"
#include "./graph.h"

IMPLEMENT_DYNAMIC_CLASS(wxStfDoc, wxDocument)

BEGIN_EVENT_TABLE( wxStfDoc, wxDocument )
EVT_MENU( wxID_SWAPCHANNELS, wxStfDoc::OnSwapChannels )
// EVT_MENU( wxID_TOOL_REMOVE, wxStfDoc::OnRemove)
EVT_MENU( wxID_FILEINFO, wxStfDoc::Fileinfo)
EVT_MENU( wxID_NEWFROMSELECTEDTHIS, wxStfDoc::OnNewfromselectedThisMenu  )
EVT_MENU( wxID_MYSELECTALL, wxStfDoc::Selectall )
EVT_MENU( wxID_UNSELECTALL, wxStfDoc::Deleteselected )
EVT_MENU( wxID_SELECTSOME, wxStfDoc::Selectsome )
EVT_MENU( wxID_UNSELECTSOME, wxStfDoc::Unselectsome )
EVT_MENU( wxID_CONCATENATE, wxStfDoc::Concatenate )
EVT_MENU( wxID_BATCH, wxStfDoc::OnAnalysisBatch )
EVT_MENU( wxID_INTEGRATE, wxStfDoc::OnAnalysisIntegrate )
EVT_MENU( wxID_DIFFERENTIATE, wxStfDoc::OnAnalysisDifferentiate )
EVT_MENU( wxID_MULTIPLY, wxStfDoc::Multiply)
EVT_MENU( wxID_SUBTRACTBASE, wxStfDoc::SubtractBaseMenu )
EVT_MENU( wxID_FIT, wxStfDoc::FitDecay)
EVT_MENU( wxID_LFIT, wxStfDoc::LFit)
EVT_MENU( wxID_LOG, wxStfDoc::LnTransform)
EVT_MENU( wxID_FILTER,wxStfDoc::Filter)
EVT_MENU( wxID_SPECTRUM,wxStfDoc::Spectrum)
EVT_MENU( wxID_POVERN,wxStfDoc::P_over_N)
EVT_MENU( wxID_PLOTCRITERION,wxStfDoc::Plotcriterion)
EVT_MENU( wxID_PLOTCORRELATION,wxStfDoc::Plotcorrelation)
EVT_MENU( wxID_EXTRACT,wxStfDoc::MarkEvents )
EVT_MENU( wxID_THRESHOLD,wxStfDoc::Threshold)
EVT_MENU( wxID_VIEWTABLE, wxStfDoc::Viewtable)
EVT_MENU( wxID_EVENT_EXTRACT, wxStfDoc::Extract )
EVT_MENU( wxID_EVENT_ERASE, wxStfDoc::EraseEvents )
EVT_MENU( wxID_EVENT_ADDEVENT, wxStfDoc::AddEvent )
#ifdef WITH_PYTHON
EVT_MENU( wxID_IMPORTPYTHON, wxStfApp::OnPythonImport)
#endif // WITH_PYTHON
END_EVENT_TABLE()

static const int baseline=100;

wxStfDoc::wxStfDoc() :
    Recording(),peakAtEnd(false),initialized(false),progress(true),Average(0)
{

}

wxStfDoc::~wxStfDoc()
{}

bool wxStfDoc::OnOpenPyDocument(const wxString& filename) {
    progress = false;
    bool success = OnOpenDocument( filename );
    progress = true;
    return success;
}

bool wxStfDoc::OnOpenDocument(const wxString& filename) {
    // Check whether the file exists:
    if ( !wxFileName::FileExists( filename ) ) {
        wxString msg;
        msg << wxT("Couldn't find ") << filename;
        wxGetApp().ErrorMsg( msg );
        return false;
    }
    // Store directory: 
    wxFileName wxfFilename( filename );
    wxGetApp().wxWriteProfileString( wxT("Settings"), wxT("Last directory"), wxfFilename.GetPath() );
    if (wxDocument::OnOpenDocument(filename)) { //calls base class function
        // Detect type of file according to filter:
#ifndef __APPLE__        
        wxString filter(GetDocumentTemplate()->GetFileFilter());
#else
        wxString filter = wxT("*.");
        filter << wxfFilename.GetExt();
#endif
        stf::filetype type = stf::findType(filter);
        if (type==stf::ascii) {
            if (!wxGetApp().get_directTxtImport()) {
                wxStfTextImportDlg ImportDlg( GetDocumentWindow(),
                        stf::CreatePreview(filename), 1, false );
                if (ImportDlg.ShowModal()!=wxID_OK) {
                    get().clear();
                    return false;
                }
                // store settings in application:
                wxGetApp().set_txtImportSettings(ImportDlg.GetTxtImport());
            }
        }
        try {
            stf::importFile(filename,type,*this,wxGetApp().GetTxtImport(),progress);
        }
        catch (const std::runtime_error& e) {
            wxString errorMsg(wxT("Error opening file\n"));
            errorMsg += wxString( e.what(),wxConvLocal );
            wxGetApp().ExceptMsg(errorMsg);
            get().clear();
            return false;
        }
        catch (const std::exception& e) {
            wxString errorMsg(wxT("Error opening file\n"));
            errorMsg += wxString( e.what(), wxConvLocal );
            wxGetApp().ExceptMsg(errorMsg);
            get().clear();
            return false;
        }
        if (get().empty()) {
            wxGetApp().ErrorMsg(wxT("File is probably empty\n"));
            get().clear();
            return false;
        }
        wxStfParentFrame* pFrame = GetMainFrame();
        if (pFrame == NULL) {
            throw std::runtime_error("pFrame is 0 in wxStfDoc::OnOpenDocument");
        }

        // pFrame->SetSingleChannel( size() <= 1 );

        if (InitCursors()!=wxID_OK) {
            get().clear();
            wxGetApp().ErrorMsg(wxT( "Couldn't initialize cursors\n" ));
            return false;
        }
        //Select active channel to be displayed
        if (get().size()>1) {
            try {
                if (ChannelSelDlg() != true) {
                    wxGetApp().ErrorMsg(wxT( "File is probably empty\n" ));
                    get().clear();
                    return false;
                }
            }
            catch (const std::out_of_range& e) {
                wxString msg(wxT( "Channel could not be selected:" ));
                msg += wxString( e.what(), wxConvLocal );
                wxGetApp().ExceptMsg(msg);
                get().clear();
                return false;
            }
        }
    } else {
        wxGetApp().ErrorMsg(wxT( "Failure in wxDocument::OnOpenDocument\n" ));
        get().clear();
        return false;
    }
    // Make sure curChannel and secondChannel are not out of range
    // so that we can use them safely without range checking:
    wxString msg(wxT( "Error while checking range:\nParts of the file might be empty\nClosing file now" ));
    if (!(get().size()>1)) {
        if (cur().size()==0) {
            wxGetApp().ErrorMsg(msg);
            get().clear();
            return false;
        }
    } else {
        if (cur().size()==0 ||
                sec().size()==0)
        {
            wxGetApp().ErrorMsg(msg);
            get().clear();
            return false;
        }
    }
    wxFileName fn(GetFilename());
    SetTitle(fn.GetFullName());
    PostInit();
    return true;
}

void wxStfDoc::SetData( const Recording& c_Data, const wxStfDoc* Sender, const wxString& title )
{
    this->resize(c_Data.size());
    std::copy(c_Data.get().begin(),c_Data.get().end(),get().begin());
    CopyAttributes(c_Data);

    // Make sure curChannel and curSection are not out of range:
    std::out_of_range e("Data empty in wxStimfitDoc::SetData()");
    if (get().empty()) {
        throw e;
    }

    wxStfParentFrame* pFrame = GetMainFrame();
    if (pFrame == NULL) {
        throw std::runtime_error("pFrame is 0 in wxStfDoc::SetData");
    }
    // pFrame->SetSingleChannel( size() <= 1 );

    // If the title is not a zero string...
    if (title != wxT("\0")) {
        // ... reset its title ...
        SetTitle(title);
    }

    //Read object variables and ensure correct and appropriate values:
    if (Sender!=NULL) {
        CopyCursors(*Sender);
        SetLatencyBeg( Sender->GetLatencyBeg() );
        SetLatencyEnd( Sender->GetLatencyEnd() );
        //Get value of the reset latency cursor box
        //0=Off, 1=Peak, 2=Rise
        SetLatencyStartMode( Sender->GetLatencyStartMode() );
        SetLatencyEndMode( Sender->GetLatencyEndMode() );
        SetLatencyWindowMode( Sender->GetLatencyWindowMode() );
        // Update menu checks:
        UpdateMenuCheckmarks();
        //Get value of the peak direction dialog box
        SetDirection( Sender->GetDirection() );
        SetFromBase( Sender->GetFromBase() );
        CheckBoundaries();
    } else {
        if (InitCursors()!=wxID_OK) {
            get().clear();
            return;
        }
    }

    //Number of channels to display (1 or 2 only!)
    if (get().size()>1) {
        //Select active channel to be displayed
        try {
            if (ChannelSelDlg()!=true) {
                get().clear();
                throw std::runtime_error("Couldn't select channels");
            }
        }
        catch (...) {
            throw;
        }
    }

    //Latency Cursor: OFF-Mode only if one channel is selected!
    if (!(get().size()>1) &&
            GetLatencyStartMode()!=stf::manualMode &&
            GetLatencyEndMode()!=stf::manualMode)
    {
        SetLatencyStartMode(stf::manualMode);
        SetLatencyEndMode(stf::manualMode);
        UpdateMenuCheckmarks();
    }

    // Make sure once again curChannel and curSection are not out of range:
    if (!(get().size()>1)) {
        if (cur().size()==0) {
            throw e;
        }
    } else {
        if (cur().size()==0 ||
                sec().size()==0)
        {
            throw e;
        }
    }
    PostInit();
}

//Dialog box to display the specific settings of the current CFS file.
int wxStfDoc::InitCursors() {
    //Get values from .ini and ensure proper settings
    SetBaseBeg(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("BaseBegin"), 1));
    SetBaseEnd(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("BaseEnd"), 20));
    SetPeakBeg(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("PeakBegin"), (int)cur().size()-100));
    SetPeakEnd(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("PeakEnd"), (int)cur().size()-50));
    int iDirection=wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("Direction"),2);
    switch (iDirection) {
    case 0: SetDirection(stf::up); break;
    case 1: SetDirection(stf::down); break;
    case 2: SetDirection(stf::both); break;
    default: SetDirection(stf::undefined_direction);
    }
    SetFromBase( true ); // reset at every program start   wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("FromBase"),1) );
    SetFitBeg(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("FitBegin"), 10));
    SetFitEnd(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("FitEnd"), 100));
    SetLatencyBeg(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("LatencyStartCursor"), 0));	/*CSH*/
    SetLatencyEnd(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("LatencyEndCursor"), 2));	/*CSH*/
    SetLatencyStartMode(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("LatencyStartMode"),0));
    SetLatencyEndMode(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("LatencyEndMode"),0));
    SetLatencyWindowMode(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("LatencyWindowMode"),1));
    // Set corresponding menu checkmarks:
    UpdateMenuCheckmarks();
    SetPM(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("PeakMean"),1));
    wxString wxsSlope = wxGetApp().wxGetProfileString(wxT("Settings"),wxT("Slope"),wxT("20.0"));
    double fSlope = 0.0;
    wxsSlope.ToDouble(&fSlope);
    SetSlopeForThreshold( fSlope );
    
    if (!(get().size()>1) &&
            GetLatencyStartMode()!=stf::manualMode &&
            GetLatencyEndMode()!=stf::manualMode)
    {
        wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("LatencyStartMode"),stf::manualMode);
        wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("LatencyEndMode"),stf::manualMode);
        SetLatencyStartMode(stf::manualMode);
        SetLatencyEndMode(stf::manualMode);
    }
    CheckBoundaries();
    return wxID_OK;
}	//End SettingsDlg()

void wxStfDoc::UpdateMenuCheckmarks() {
    // get menu bar:
    wxStfChildFrame *pChildFrame = (wxStfChildFrame*)GetDocumentWindow();
	if (pChildFrame) {
        wxMenuBar *pBar=pChildFrame->GetMenuBar();

        if (pBar) {
            pBar->FindItem(wxID_LATENCYSTART_MAXSLOPE)->Check(GetLatencyStartMode()==stf::riseMode);
            pBar->FindItem(wxID_LATENCYSTART_HALFRISE)->Check(GetLatencyStartMode()==stf::halfMode);
            pBar->FindItem(wxID_LATENCYSTART_PEAK)->Check(GetLatencyStartMode()==stf::peakMode);
            pBar->FindItem(wxID_LATENCYSTART_MANUAL)->Check(GetLatencyStartMode()==stf::manualMode);
            pBar->FindItem(wxID_LATENCYEND_FOOT)->Check(GetLatencyEndMode()==stf::footMode);
            pBar->FindItem(wxID_LATENCYEND_MAXSLOPE)->Check(GetLatencyEndMode()==stf::riseMode);
            pBar->FindItem(wxID_LATENCYEND_HALFRISE)->Check(GetLatencyEndMode()==stf::halfMode);
            pBar->FindItem(wxID_LATENCYEND_PEAK)->Check(GetLatencyEndMode()==stf::peakMode);
            pBar->FindItem(wxID_LATENCYEND_MANUAL)->Check(GetLatencyEndMode()==stf::manualMode);
            pBar->FindItem(wxID_LATENCYWINDOW)->Check(GetLatencyWindowMode()==stf::windowMode);
	    }
    }
}

void wxStfDoc::PostInit() {
    wxStfChildFrame *pFrame = (wxStfChildFrame*)GetDocumentWindow();
    if ( pFrame == NULL ) {
        wxGetApp().ErrorMsg( wxT("Zero pointer in wxStfDoc::PostInit") );
        return;
    }
    try {
        pFrame->CreateComboTraces(get().at(GetCurCh()).size());
        if ( size() > 1 ) {
            wxArrayString channelNames;
            channelNames.Alloc( size() );
            for (std::size_t n_c=0; n_c < size(); ++n_c) {
                wxString channelStream;
                channelStream << n_c << wxT(" (") << at(n_c).GetChannelName() << wxT(")");
                channelNames.Add( channelStream );
            }
            pFrame->CreateComboChannels( channelNames );
            pFrame->SetChannels( GetCurCh(), GetSecCh() );
        }
    }
    catch (const std::out_of_range& e) {
        wxGetApp().ExceptMsg( wxString( e.what(), wxConvLocal ) );
        return;
    }
    if (GetSR()>1000) {
        wxString highSampling;
        highSampling << wxT("Sampling rate seems very high (") << GetSR() << wxT(" kHz).\n")
        << wxT("Divide by 1000?");
        if (wxMessageDialog(
                GetDocumentWindow(),
                highSampling,
                wxT("Adjust sampling rate"),
                wxYES_NO
        ).ShowModal()==wxID_YES)
        {
            SetXScale(GetXScale()*1000.0);
        }
    }
    if (size()>1) {
        wxMenu* ch2Sub=new wxMenu;
        ch2Sub->Append(wxID_CH2BASE, wxT("Match &baseline"));
        ch2Sub->Append(wxID_CH2POS, wxT("Match &abs. position"));
        ch2Sub->Append(wxID_CH2ZOOM, wxT("Match &y-scale"));
        ch2Sub->Append(wxID_CH2BASEZOOM, wxT("Match baseline a&nd y-scale"));
        wxStfChildFrame *pChildFrame = (wxStfChildFrame*)GetDocumentWindow();
        wxMenuBar* pBar=pChildFrame->GetMenuBar();

        if (pBar!=NULL) {
            pBar->GetMenu(2)->AppendSeparator();
            pBar->GetMenu(2)->AppendSubMenu(ch2Sub, wxT("&Channel 2 scaling"));
            pBar->GetMenu(2)->Append(wxID_SWAPCHANNELS, wxT("&Swap channels"));
        }
    }
    // Read results table settings from registry:
    SetViewCrosshair(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewCrosshair"),1)==1);
    SetViewBaseline(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewBaseline"),1)==1);
    SetViewBaseSD(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewBaseSD"),1)==1);
    SetViewThreshold(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewThreshold"),1)==1);
    SetViewPeakZero(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewPeakzero"),1)==1);
    SetViewPeakBase(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewPeakbase"),1)==1);
    SetViewPeakThreshold(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewPeakthreshold"),1)==1);
    SetViewRT2080(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewRT2080"),1)==1);
    SetViewT50(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewT50"),1)==1);
    SetViewRD(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewRD"),1)==1);
    SetViewSlopeRise(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewSloperise"),1)==1);
    SetViewSlopeDecay(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewSlopedecay"),1)==1);
    SetViewLatency(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewLatency"),1)==1);
    SetViewCursors(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewCursors"),1)==1);

    // refresh the view once we are through:
    initialized=true;
    wxStfView* pView=(wxStfView*)GetFirstView();
    if (pView != NULL) {
        wxStfGraph* pGraph = pView->GetGraph();
        if (pGraph != NULL) {
            pGraph->Refresh();
            // Set the focus:
            pGraph->SetFocus();
        }
    }
    wxGetApp().OnPeakcalcexecMsg();
    pFrame->SetCurTrace(0);
    UpdateSelectedButton();
}

//Dialog box to select channel to be displayed
bool wxStfDoc::ChannelSelDlg() {
    // Set default channels:
    if ( size() < 2 ) {
        return false;
    }
    SetCurCh( 0 );
    SetSecCh( 1 );
    return true;
}	//End ChannelSelDlg()

void wxStfDoc::CheckBoundaries()
{
    //Security check base
    if (GetBaseBeg() > GetBaseEnd())
    {
        std::size_t aux=GetBaseBeg();
        SetBaseBeg((int)GetBaseEnd());
        SetBaseEnd((int)aux);
        wxGetApp().ErrorMsg(wxT("Base cursors are reversed,\nthey will be exchanged"));
    }
    //Security check peak
    if (GetPeakBeg() > GetPeakEnd())
    {
        std::size_t aux=GetPeakBeg();
        SetPeakBeg((int)GetPeakEnd());
        SetPeakEnd((int)aux);
        wxGetApp().ErrorMsg(wxT("Peak cursors are reversed,\nthey will be exchanged"));
    }
    //Security check decay
    if (GetFitBeg() > GetFitEnd())
    {
        std::size_t aux=GetFitBeg();
        SetFitBeg((int)GetFitEnd());
        SetFitEnd((int)aux);
        wxGetApp().ErrorMsg(wxT("Decay cursors are reversed,\nthey will be exchanged"));
    }

    if (GetPM() > (int)cur().size()) {
        SetPM((int)cur().size()-1);
    }
    if (GetPM() == 0) {
        SetPM(1);
    }
}

bool wxStfDoc::OnNewDocument() {
    // correct caption:
    wxString title(GetTitle());
    wxStfChildFrame* wnd=(wxStfChildFrame*)GetDocumentWindow();
    wnd->SetLabel(title);
    // call base class member:
    return true;
    //	return wxDocument::OnNewDocument();
}

void wxStfDoc::Fileinfo(wxCommandEvent& WXUNUSED(event)) {
    //Create CFileOpenDlg object 'dlg'
    wxString oss1, oss2;
    oss1 << wxT("Number of Channels: ") << static_cast<unsigned int>(get().size());
    oss2 << wxT("Number of Sweeps: ") << static_cast<unsigned int>(get()[GetCurCh()].size());
    wxString general;
    general+=wxT("Date:\n") + GetDate() + wxT("\n")+
    wxT("Time:\n") + GetTime() + wxT("\n")+
    oss1+wxT("\n") + oss2 + wxT("\n")+
    wxT("Comment:\n") + GetComment();

    wxStfFileInfoDlg dlg( GetDocumentWindow(), general, GetFileDescription(),
            GetGlobalSectionDescription() );
    dlg.ShowModal();
}

bool wxStfDoc::OnCloseDocument() {
    if (!get().empty()) {
        WriteToReg();
    }
    // Remove file menu from file menu list:
    wxGetApp().GetDocManager()->GetFileHistory()->RemoveMenu( doc_file_menu );

    // Tell the App:
    wxGetApp().OnCloseDocument();
    return wxDocument::OnCloseDocument();
    //Note that the base class version will delete all the document's data
}

bool wxStfDoc::SaveAs() {
    // Override file save dialog to display only writeable
    // file types
    wxString filters;
    filters += wxT("hdf5 file (*.h5)|*.h5|");
    filters += wxT("CED filing system (*.dat;*.cfs)|*.dat;*.cfs|");
    filters += wxT("Axon text file (*.atf)|*.atf|");
    filters += wxT("Igor binary wave (*.*)|*.*|");
    filters += wxT("Text file series (*.*)|*.*");
    wxFileDialog SelectFileDialog( GetDocumentWindow(), wxT("Save file"), wxT(""), wxT(""), filters,
            wxFD_SAVE | wxFD_OVERWRITE_PROMPT | wxFD_PREVIEW );
    if(SelectFileDialog.ShowModal()==wxID_OK) {
        wxString filename = SelectFileDialog.GetPath();
        Recording writeRec(ReorderChannels());
        if (writeRec.size() == 0) return false;
        try {
            switch (SelectFileDialog.GetFilterIndex()) {
            case 1:
                return stf::exportCFSFile(filename, writeRec);
            case 2:
                return stf::exportATFFile(filename, writeRec);
            case 3:
#ifdef _WINDOWS
                return stf::exportIGORFile(filename, writeRec);
#else
                wxGetApp().ErrorMsg( wxT("Igor file export only implemented on Windows platforms") );
                return false;
#endif
            case 4:
                return stf::exportASCIIFile(filename, get()[GetCurCh()]);
            case 0:
            default:
                return stf::exportHDF5File(filename, writeRec);
            }
        }
        catch (const std::runtime_error& e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
            return false;
        }
    } else {
        return false;
    }
}

Recording wxStfDoc::ReorderChannels() {
    // Re-order channels?
    std::vector< wxString > channelNames(size());
    wxs_it it = channelNames.begin();
    for (c_ch_it cit = get().begin();
         cit != get().end() && it != channelNames.end();
         cit++)
    {
        *it = cit->GetChannelName();
        it++;
    }
    std::vector<int> channelOrder(size());
    if (size()>1) {
        wxStfOrderChannelsDlg orderDlg(GetDocumentWindow(),channelNames);
        if (orderDlg.ShowModal() != wxID_OK) {
            return Recording(0);
        }
        channelOrder=orderDlg.GetChannelOrder();
    } else {
        int n_c = 0;
        for (int_it it = channelOrder.begin(); it != channelOrder.end(); it++) {
            *it = n_c++;
        }
    }

    Recording writeRec(size());
    writeRec.CopyAttributes(*this);
    std::size_t n_c = 0;
    for (c_int_it cit2 = channelOrder.begin(); cit2 != channelOrder.end(); cit2++) {
        writeRec.InsertChannel(get()[*cit2],n_c);
        // correct units:
        writeRec[n_c++].SetYUnits( at(*cit2).GetYUnits() );
    }
    return writeRec;
}

bool wxStfDoc::DoSaveDocument(const wxString& filename) {
    Recording writeRec(ReorderChannels());
    if (writeRec.size() == 0) return false;
    try {
        if (stf::exportHDF5File(filename, writeRec))
            return true;
        else
            return false;
    }
    catch (const std::runtime_error& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
        return false;
    }
}

void wxStfDoc::WriteToReg() {
    //Write file length
    wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("FirstPoint"), 1);
    wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("LastPoint"), (int)cur().size()-1);
    
    //Write cursors
    if (!outOfRange(GetBaseBeg()))
        wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("BaseBegin"), (int)GetBaseBeg());
    if (!outOfRange(GetBaseEnd()))
        wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("BaseEnd"), (int)GetBaseEnd());
    if (!outOfRange(GetPeakBeg()))
        wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("PeakBegin"), (int)GetPeakBeg());
    if (!outOfRange(GetPeakEnd()))
        wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("PeakEnd"), (int)GetPeakEnd());
    wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("PeakMean"),(int)GetPM());
    wxString wxsSlope;
    wxsSlope << GetSlopeForThreshold();
    wxGetApp().wxWriteProfileString(wxT("Settings"),wxT("Slope"),wxsSlope);
    if (wxGetApp().GetCursorsDialog() != NULL) {
        wxGetApp().wxWriteProfileInt(
                wxT("Settings"),wxT("StartFitAtPeak"),(int)wxGetApp().GetCursorsDialog()->GetStartFitAtPeak()
        );
    }
    if (!outOfRange(GetFitBeg()))
        wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("FitBegin"), (int)GetFitBeg());
    if (!outOfRange(GetFitEnd()))
        wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("FitEnd"), (int)GetFitEnd());
    if (!outOfRange((size_t)GetLatencyBeg()))
        wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("LatencyCursor"), (int)GetLatencyBeg());
    if (!outOfRange((size_t)GetLatencyEnd()))
        wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("LatencyCursor"), (int)GetLatencyEnd());

    // Write Zoom
    wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("Zoom.xZoom"),(int)GetXZoom().xZoom*100000);
    wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("Zoom.yZoom"),at(GetCurCh()).GetYZoom().yZoom*100000);
    wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("Zoom.startPosX"),(int)GetXZoom().startPosX);
    wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("Zoom.startPosY"),at(GetCurCh()).GetYZoom().startPosY);
    if ((get().size()>1))
    {
        wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("Zoom.yZoom2"),(int)at(GetSecCh()).GetYZoom().yZoom*100000);
        wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("Zoom.startPosY2"),at(GetSecCh()).GetYZoom().startPosY);
    }
}

bool wxStfDoc::SetSection(std::size_t section){
    // Check range:
    if (!(get().size()>1)) {
        if (section<0 ||
                section>=get()[GetCurCh()].size())
        {
            wxGetApp().ErrorMsg(wxT("subscript out of range\nwhile calling CStimfitDoc::SetSection()"));
            return false;
        }
        if (get()[GetCurCh()][section].size()==0) {
            wxGetApp().ErrorMsg(wxT("Section is empty"));
            return false;
        }
    } else {
        if (section<0 ||
                section>=get()[GetCurCh()].size() ||
                section>=get()[GetSecCh()].size())
        {
            wxGetApp().ErrorMsg(wxT("subscript out of range\nwhile calling CStimfitDoc::SetSection()"));
            return false;
        }
        if (get()[GetCurCh()][section].size()==0 ||
                get()[GetSecCh()][section].size()==0) {
            wxGetApp().ErrorMsg(wxT("Section is empty"));
            return false;
        }
    }
    CheckBoundaries();
    SetCurSec(section);
    UpdateSelectedButton();

    return true;
}

void wxStfDoc::OnSwapChannels(wxCommandEvent& WXUNUSED(event)) {
    if ( size() > 1) {
        // Update combo boxes:
        wxStfChildFrame* pFrame=(wxStfChildFrame*)GetDocumentWindow();
        if ( pFrame == NULL ) {
            wxGetApp().ErrorMsg( wxT("Frame is zero in wxStfDoc::SwapChannels"));
            return;
        }
        pFrame->SetChannels( GetSecCh(), GetCurCh() );
        pFrame->UpdateChannels();
    }
}

void wxStfDoc::ToggleSelect() {
    // get current selection status of this trace:

    bool selected = false;
    for (c_st_it cit = GetSelectedSections().begin();
         cit != GetSelectedSections().end() && !selected;
         ++cit) {
        if (*cit == GetCurSec()) {
            selected = true;
        }
    }

    if (selected) {
        Remove();
    } else {
        Select();
    }

}

void wxStfDoc::Select() {
    if (GetSelectedSections().size() == get()[GetCurCh()].size()) {
        wxGetApp().ErrorMsg(wxT("No more traces can be selected\nAll traces are selected"));
        return;
    }
    //control whether trace has already been selected:
    bool already=false;
    for (c_st_it cit = GetSelectedSections().begin();
         cit != GetSelectedSections().end() && !already;
         ++cit) {
        if (*cit == GetCurSec()) {
            already = true;
        }
    }

    //add trace number to selected numbers, print number of selected traces
    if (!already) {
        SelectTrace(GetCurSec());
        //String output in the trace navigator
        wxStfChildFrame* pFrame=(wxStfChildFrame*)GetDocumentWindow();
        pFrame->SetSelected(GetSelectedSections().size());
    } else {
        wxGetApp().ErrorMsg(wxT("Trace is already selected"));
        return;
    }

    Focus();
}

void wxStfDoc::Remove() {
    if (UnselectTrace(GetCurSec())) {
        //Message update in the trace navigator
        wxStfChildFrame* pFrame = (wxStfChildFrame*)GetDocumentWindow();
        if (pFrame)
            pFrame->SetSelected(GetSelectedSections().size());
    } else {
        wxGetApp().ErrorMsg(wxT("Trace is not selected"));
    }

    Focus();

}

void wxStfDoc::Concatenate(wxCommandEvent &WXUNUSED(event)) {
    if (GetSelectedSections().empty()) {
        wxGetApp().ErrorMsg(wxT("Select traces first"));
        return;
    }
    wxProgressDialog progDlg( wxT("Concatenating traces"), wxT("Starting..."),
            100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );
    int new_size=0;
    for (c_st_it cit = GetSelectedSections().begin(); cit != GetSelectedSections().end(); cit++) {
        new_size+=(int)get()[GetCurCh()][*cit].size();
    }
    Section TempSection(new_size);
    std::size_t n_new=0;
    std::size_t n_s=0;
    for (c_st_it cit = GetSelectedSections().begin(); cit != GetSelectedSections().end(); cit++) {
        wxString progStr;
        progStr << wxT("Adding section #") << (int)n_s+1 << wxT(" of ") << (int)GetSelectedSections().size();
        progDlg.Update(
                (int)((double)n_s/(double)GetSelectedSections().size()*100.0),
                progStr
        );
        // copy using iterators (somewhat unusual for a valarray):
        std::size_t secSize=get()[GetCurCh()][*cit].size();
        std::copy(&get()[GetCurCh()][*cit][0],
                &get()[GetCurCh()][*cit][secSize],
                &TempSection[n_new]);
        n_new+=secSize;
        n_s++;
    }
    TempSection.SetSectionDescription(
            wxString(GetTitle()+wxT(", concatenated"))
    );
    Channel TempChannel(TempSection);
    Recording Concatenated(TempChannel);
    Concatenated.CopyAttributes(*this);
    wxGetApp().NewChild(Concatenated,this,wxString(GetTitle()+wxT(", concatenated")));
}

void wxStfDoc::CreateAverage(
        bool calcSD,
        bool align	       //align to steepest rise of other channel?
) {
    if(GetSelectedSections().empty()) {
        wxGetApp().ErrorMsg(wxT("Select traces first"));
        return;
    }
    wxBusyCursor wc;
    //array indicating how many indices to shift when aligning,
    //has to be filled with zeros:
    std::vector<int> shift(GetSelectedSections().size(),0);
    //number of points in average:
    int average_size;

    //find alignment points in the inactive (==second) channel:
    if (align) {
        wxStfAlignDlg AlignDlg(GetDocumentWindow());
        if (AlignDlg.ShowModal()!=wxID_OK) return;
        //store current section and channel index:
        std::size_t section_old=GetCurSec();
        std::size_t channel_old=GetCurCh();
        //initialize the lowest and the highest index:
        std::size_t min_index=0;
        try {
            min_index=get()[GetSecCh()].at(GetSelectedSections().at(0)).size()-1;
        }
        catch (const std::out_of_range& e) {
            wxString msg(wxT("Error while aligning\nIt is safer to re-start the program\n"));
            msg+=wxString( e.what(), wxConvLocal );
            wxGetApp().ExceptMsg(msg);
            return;
        }
        // swap channels temporarily:
        SetCurCh(GetSecCh());
        std::size_t max_index=0, n=0;
        int_it it = shift.begin();
        //loop through all selected sections:
        for (c_st_it cit = GetSelectedSections().begin();
             cit != GetSelectedSections().end() && it != shift.end();
             cit++)
        {
            //Set the selected section as the current section temporarily:
            SetSection(*cit);
            if (peakAtEnd) {
                SetPeakEnd((int)get()[GetSecCh()][*cit].size()-1);
            }
            // Calculate all variables for the current settings
            // APMaxSlopeT will be calculated for the second (==inactive)
            // channel, so channels may not be changed!
            try {
                Measure();
            }
            catch (const std::out_of_range& e) {
                Average.resize(0);
                wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
                return;
            }

            //check whether the current index is a max or a min,
            //and if so, store it:
            std::size_t alignIndex= AlignDlg.AlignRise()? (int)GetAPMaxRiseT():(int)GetMaxT();
            *it = int(alignIndex);
            if (alignIndex > max_index) {
                max_index=alignIndex;
            }
            if (alignIndex < min_index) {
                min_index=alignIndex;
            }
            n++;
            it++;
        }
        //now that max and min indices are known, calculate the number of
        //points that need to be shifted:
        for (int_it it = shift.begin(); it != shift.end(); it++) {
            (*it) -= (int)min_index;
        }
        //restore section and channel settings:
        SetSection(section_old);
        SetCurCh(channel_old);
        average_size=(int)(get()[0][GetSelectedSections()[0]].size()-(max_index-min_index));
    } else {
        average_size=(int)get()[0][GetSelectedSections()[0]].size();
    }
    //initialize temporary sections and channels:
    Average.resize(size());
    std::size_t n_c = 0;
    for (c_ch_it cit = get().begin(); cit != get().end(); cit++) {
        Section TempSection(average_size), TempSig(average_size);
        MakeAverage(TempSection, TempSig, n_c, GetSelectedSections(), calcSD, shift);
        TempSection.SetSectionDescription( wxString(GetTitle()+wxT(", average")) );
        Channel TempChannel(TempSection);
        TempChannel.SetChannelName(cit->GetChannelName());
        try {
            Average.InsertChannel(TempChannel,n_c);
        }
        catch (const std::out_of_range& e) {
            Average.resize(0);
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
            return;
        }
        n_c++;
    }
    Average.CopyAttributes(*this);

    wxString title;
    title << GetFilename() << wxT(", average of ") << (int)GetSelectedSections().size() << wxT(" traces");
    wxGetApp().NewChild(Average,this,title);
}	//End of CreateAverage(.,.,.)

void wxStfDoc::FitDecay(wxCommandEvent& WXUNUSED(event)) {
    int fselect=-2;
    wxStfFitSelDlg FitSelDialog(GetDocumentWindow(), this);
    if (FitSelDialog.ShowModal() != wxID_OK) return;
    wxBeginBusyCursor();
    fselect=FitSelDialog.GetFSelect();
    if (outOfRange(GetFitBeg()) || outOfRange(GetFitEnd())) {
        wxGetApp().ErrorMsg(wxT("Subscript out of range in wxStfDoc::FitDecay()"));
        return;
    }
    //number of parameters to be fitted:
    std::size_t n_params=0;

    //number of points:
    std::size_t n_points = GetFitEnd()-GetFitBeg();
    if (n_points<=1) {
        wxGetApp().ErrorMsg(wxT("Check fit limits"));
        return;
    }
    wxString fitInfo;

    try {
        n_params=(int)wxGetApp().GetFuncLib().at(fselect).pInfo.size();
    }
    catch (const std::out_of_range& e) {
        wxString msg(wxT("Could not retrieve function from library:\n"));
        msg+=wxString( e.what(), wxConvLocal );
        wxGetApp().ExceptMsg(msg);
        return;
    }
    std::valarray<double> params ( FitSelDialog.GetInitP() );
    int warning = 0;
    try {
        std::size_t fitSize = GetFitEnd() - GetFitBeg();
        std::valarray<double> x( fitSize );
        //fill array:
        std::copy(&cur()[GetFitBeg()], &cur()[GetFitBeg()+fitSize], &x[0]);
        if (params.size() != n_params) {
            throw std::runtime_error("Wrong size of params in Recording::lmFit()");
        }
        double chisqr = stf::lmFit( x, GetXScale(), wxGetApp().GetFuncLib()[fselect],
                FitSelDialog.GetOpts(), params, fitInfo, warning );
        cur().SetIsFitted( params, wxGetApp().GetFuncLibPtr(fselect),
                chisqr, GetFitBeg(), GetFitEnd() );
    }
    catch (const std::out_of_range& e) {
        wxGetApp().ExceptMsg( wxString(e.what(), wxConvLocal) );
        return;
    }
    catch (const std::runtime_error& e) {
        wxGetApp().ExceptMsg( wxString(e.what(), wxConvLocal) );
        return;
    }

    // Refresh the graph to show the fit before
    // the dialog pops up:
    wxStfView* pView=(wxStfView*)GetFirstView();
    pView->GetGraph()->Refresh();
    wxStfFitInfoDlg InfoDialog(GetDocumentWindow(),fitInfo);
    wxEndBusyCursor();
    InfoDialog.ShowModal();
    wxStfChildFrame* pFrame=(wxStfChildFrame*)GetDocumentWindow();
    wxString label; label << wxT("Fit, Section #") << (int)GetCurSec()+1;
    pFrame->ShowTable(cur().GetBestFit(),label);
}

void wxStfDoc::LFit(wxCommandEvent& WXUNUSED(event)) {
    wxBusyCursor wc;
    if (outOfRange(GetFitBeg()) || outOfRange(GetFitEnd())) {
        wxGetApp().ErrorMsg(wxT("Subscript out of range in CStimfitDoc::FitDecay()"));
        return;
    }
    //number of parameters to be fitted:
    std::size_t n_params=0;

    //number of points:
    std::size_t n_points=GetFitEnd()-GetFitBeg();
    if (n_points<=1) {
        wxGetApp().ErrorMsg(wxT("Check fit limits"));
        return;
    }
    wxString fitInfo;
    n_params=2;
    std::valarray<double> params( n_params );

    //fill array:
    std::valarray<double> x(n_points);
    std::copy(&cur()[GetFitBeg()], &cur()[GetFitBeg()+n_points], &x[0]);
    std::valarray<double> t(x.size());
    for (std::size_t n_t=0;n_t<x.size();++n_t) t[n_t]=n_t*GetXScale();

    // Perform the fit:
    double chisqr = stf::linFit(t,x,params[0],params[1]);

	cur().SetIsFitted( params, wxGetApp().GetLinFuncPtr(), chisqr, GetFitBeg(), GetFitEnd() );

    // Refresh the graph to show the fit before
    // the dialog pops up:
    wxStfView* pView=(wxStfView*)GetFirstView();
    pView->GetGraph()->Refresh();
    fitInfo << wxT("slope = ") << params[0] << wxT("\n1/slope = ") << 1.0/params[0]
            << wxT("\ny-intercept = ") << params[1];
    wxStfFitInfoDlg InfoDialog(GetDocumentWindow(),fitInfo);
    InfoDialog.ShowModal();
    wxStfChildFrame* pFrame=(wxStfChildFrame*)GetDocumentWindow();
    wxString label; label << wxT("Fit, Section #") << (int)GetCurSec();
    pFrame->ShowTable(cur().GetBestFit(),label);
}

void wxStfDoc::LnTransform(wxCommandEvent& WXUNUSED(event)) {
    Channel TempChannel(GetSelectedSections().size(), get()[GetCurCh()][GetSelectedSections()[0]].size());
    std::size_t n = 0;
    for (c_st_it cit = GetSelectedSections().begin(); cit != GetSelectedSections().end(); cit++) {
        Section TempSection(log(get()[GetCurCh()][*cit].get()));
        TempSection.SetSectionDescription( get()[GetCurCh()][*cit].GetSectionDescription()+
                wxT(", transformed (ln)") );
        try {
            TempChannel.InsertSection(TempSection,n);
        }
        catch (const std::out_of_range e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
        }
        n++;
    }
    if (TempChannel.size()>0) {
        Recording Transformed(TempChannel);
        Transformed.CopyAttributes(*this);
        wxString title(GetTitle());
        title+=wxT(", transformed (ln)");
        wxGetApp().NewChild(Transformed,this,title);
    }
}

void wxStfDoc::Viewtable(wxCommandEvent& WXUNUSED(event)) {
    wxBusyCursor wc;
    try {
        wxStfChildFrame* pFrame=(wxStfChildFrame*)GetDocumentWindow();
        pFrame->ShowTable(CurAsTable(),cur().GetSectionDescription());
    }
    catch (const std::out_of_range& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
        return;
    }
}

void wxStfDoc::Multiply(wxCommandEvent& WXUNUSED(event)) {
    if (GetSelectedSections().empty()) {
        wxGetApp().ErrorMsg(wxT("Select traces first"));
        return;
    }
    //insert standard values:
    std::vector<wxString> labels(1);
    std::vector<double> defaults(labels.size());
    labels[0]=wxT("Multiply with:");defaults[0]=1;
    stf::UserInput init(labels,defaults,wxT("Set factor"));

    wxStfUsrDlg MultDialog(GetDocumentWindow(),init);
    if (MultDialog.ShowModal()!=wxID_OK) return;
    std::vector<double> input(MultDialog.readInput());
    if (input.size()!=1) return;

    double factor=input[0];
    Channel TempChannel(GetSelectedSections().size(), get()[GetCurCh()][GetSelectedSections()[0]].size());
    std::size_t n = 0;
    for (c_st_it cit = GetSelectedSections().begin(); cit != GetSelectedSections().end(); cit++) {
        // Multiply the valarray in Data:
        Section TempSection(get()[GetCurCh()][*cit].get()*factor);
        TempSection.SetSectionDescription(
                get()[GetCurCh()][*cit].GetSectionDescription()+
                wxT(", multiplied")
        );
        try {
            TempChannel.InsertSection(TempSection,n);
        }
        catch (const std::out_of_range e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
        }
        n++;
    }
    if (TempChannel.size()>0) {
        Recording Multiplied(TempChannel);
        Multiplied.CopyAttributes(*this);
        Multiplied[0].SetYUnits( at( GetCurCh() ).GetYUnits() );
        wxString title(GetTitle());
        title+=wxT(", multiplied");
        wxGetApp().NewChild(Multiplied,this,title);
    }
}

bool wxStfDoc::SubtractBase( ) {
    if (GetSelectedSections().empty()) {
        wxGetApp().ErrorMsg(wxT("Select traces first"));
        return false;
    }
    Channel TempChannel(GetSelectedSections().size(), get()[GetCurCh()][GetSelectedSections()[0]].size());
    std::size_t n = 0;
    for (c_st_it cit = GetSelectedSections().begin(); cit != GetSelectedSections().end(); cit++) {
        Section TempSection(get()[GetCurCh()][*cit].get()-GetSelectBase()[n]);
        TempSection.SetSectionDescription( get()[GetCurCh()][*cit].GetSectionDescription()+
                wxT(", baseline subtracted") );
        try {
            TempChannel.InsertSection(TempSection,n);
        }
        catch (const std::out_of_range& e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
            return false;
        }
        n++;
    }
    if (TempChannel.size()>0) {
        Recording SubBase(TempChannel);
        SubBase.CopyAttributes(*this);
        wxString title(GetTitle());
        title+=wxT(", baseline subtracted");
        wxGetApp().NewChild(SubBase,this,title);
    } else {
        wxGetApp().ErrorMsg( wxT("Channel is empty.") );
        return false;
    }

    return true;
}


void wxStfDoc::OnAnalysisBatch(wxCommandEvent &WXUNUSED(event)) {
    //	event.Skip();
    if (GetSelectedSections().empty())
    {
        wxGetApp().ErrorMsg(wxT("No selected traces"));
        return;
    }

    std::size_t section_old=GetCurSec(); //
    wxStfBatchDlg SaveYtDialog(GetDocumentWindow());
    if (SaveYtDialog.ShowModal()!=wxID_OK) return;
    std::vector<wxString> colTitles;
    //Write the header of the SaveYt file in a string
    if (SaveYtDialog.PrintBase()) {
        colTitles.push_back(wxT("Base"));
    }
    if (SaveYtDialog.PrintBaseSD()) {
        colTitles.push_back(wxT("Base SD"));
    }
    if (SaveYtDialog.PrintThreshold()) {
        colTitles.push_back(wxT("Threshold"));
    }
    if (SaveYtDialog.PrintPeakZero()) {
        colTitles.push_back(wxT("Peak (from 0)"));
    }
    if (SaveYtDialog.PrintPeakBase()) {
        colTitles.push_back(wxT("Peak (from baseline)"));
    }
    if (SaveYtDialog.PrintPeakThreshold()) {
        colTitles.push_back(wxT("Peak (from threshold)"));
    }
    if (SaveYtDialog.PrintRT2080()) {
        colTitles.push_back(wxT("RT 20-80%"));
    }
    if (SaveYtDialog.PrintT50()) {
        colTitles.push_back(wxT("t 1/2"));
    }
    if (SaveYtDialog.PrintSlopes()) {
        colTitles.push_back(wxT("Max. slope rise"));
        colTitles.push_back(wxT("Max. slope decay"));
    }
    if (SaveYtDialog.PrintLatencies()) {
        colTitles.push_back(wxT("Latency"));
    }

    int fselect=-2;
    std::size_t n_params=0;
    wxStfFitSelDlg FitSelDialog(GetDocumentWindow(), this);
    if (SaveYtDialog.PrintFitResults()) {
        while (fselect<0) {
            FitSelDialog.SetNoInput(true);
            if (FitSelDialog.ShowModal()!=wxID_OK) {
                SetSection(section_old);
                return;
            }
            fselect=FitSelDialog.GetFSelect();
        }
        try {
            n_params=(int)wxGetApp().GetFuncLib().at(fselect).pInfo.size();
        }
        catch (const std::out_of_range& e) {
            wxString msg(wxT("Error while retrieving function from library:\n"));
            msg+=wxString( e.what(), wxConvLocal );
            wxGetApp().ExceptMsg(msg);
            SetSection(section_old);
            return;
        }
        for (std::size_t n_pf=0;n_pf<n_params;++n_pf) {
            colTitles.push_back( wxGetApp().GetFuncLib()[fselect].pInfo[n_pf].desc );
        }
        colTitles.push_back(wxT("Fit warning code"));
    }
    if (SaveYtDialog.PrintThr()) {
        colTitles.push_back(wxT("# of thr. crossings"));
    }
    double threshold=0.0;
    if (SaveYtDialog.PrintThr()) {
        // Get threshold from user:
        stf::UserInput Input( std::vector<wxString>(1,wxT("Threshold")),
                std::vector<double> (1,0.0), wxT("Set threshold") );
        wxStfUsrDlg myDlg( GetDocumentWindow(), Input );
        if (myDlg.ShowModal()!=wxID_OK) {
            return;
        }
        threshold=myDlg.readInput()[0];
    }
    wxProgressDialog progDlg( wxT("Batch analysis in progress"), wxT("Starting batch analysis"),
            100, GetDocumentWindow(), wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );

    stf::Table table(GetSelectedSections().size(),colTitles.size());
    for (std::size_t nCol=0;nCol<colTitles.size();++nCol) {
        try {
            table.SetColLabel(nCol,colTitles[nCol]);
        }
        catch (const std::out_of_range& e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
            SetSection(section_old);
            return;
        }
    }
    std::size_t n_s = 0;
    for (c_st_it cit = GetSelectedSections().begin(); cit != GetSelectedSections().end(); cit++) {
        wxString progStr;
        progStr << wxT("Processing trace # ") << (int)n_s+1 << wxT(" of ") << (int)GetSelectedSections().size();
        progDlg.Update( (int)((double)n_s/ (double)GetSelectedSections().size()*100.0), progStr );
        SetSection(*cit);
        if (peakAtEnd)
            SetPeakEnd((int)get()[GetCurCh()][*cit].size()-1);

        //Calculate all variables for the current settings
        try {
            Measure();
        }
        catch (const std::out_of_range& e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
            SetSection(section_old);
            return;
        }

        // Set fit start cursor to new peak if necessary:
        if (wxGetApp().GetCursorsDialog() != NULL && wxGetApp().GetCursorsDialog()->GetStartFitAtPeak())
            SetFitBeg(GetMaxT());

        std::valarray<double> params;
        int fitWarning = 0;
        if (SaveYtDialog.PrintFitResults()) {
            try {
                n_params=(int)wxGetApp().GetFuncLib().at(fselect).pInfo.size();
            }
            catch (const std::out_of_range& e) {
                wxString msg(wxT("Could not retrieve function from library:\n"));
                msg+=wxString( e.what(), wxConvLocal );
                wxGetApp().ExceptMsg(msg);
                return;
            }
            // in this case, initialize parameters from init function,
            // not from user input:
            std::valarray<double> x(GetFitEnd()-GetFitBeg());
            //fill array:
            std::copy(&cur()[GetFitBeg()], &cur()[GetFitEnd()], &x[0]);
            params.resize(n_params);
            wxGetApp().GetFuncLib().at(fselect).init( x, GetBase(), GetPeak(),
                    GetXScale(), params );
            wxString fitInfo;

            try {
                double chisqr = stf::lmFit( x, GetXScale(), wxGetApp().GetFuncLib()[fselect],
                        FitSelDialog.GetOpts(), params, fitInfo, fitWarning );
                cur().SetIsFitted( params, wxGetApp().GetFuncLibPtr(fselect),
                        chisqr, GetFitBeg(), GetFitEnd() );
            }

            catch (const std::out_of_range& e) {
                wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
                SetSection(section_old);
                return;
            }
            catch (const std::runtime_error& e) {
                wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
                SetSection(section_old);
                return;
            }
            catch (const std::exception& e) {
                wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
                SetSection(section_old);
                return;
            }
        }

        // count number of threshold crossings if needed:
        std::size_t n_crossings=0;
        if (SaveYtDialog.PrintThr()) {
            n_crossings= stf::peakIndices( cur().get(), threshold, 0 ).size();
        }
        std::size_t nCol=0;
        //Write the variables of the current channel in a string
        try {
            table.SetRowLabel(n_s,cur().GetSectionDescription());
            if (SaveYtDialog.PrintBase())
                table.at(n_s,nCol++)=GetBase();
            if (SaveYtDialog.PrintBaseSD())
                table.at(n_s,nCol++)=GetBaseSD();
            if (SaveYtDialog.PrintThreshold())
                table.at(n_s,nCol++)=GetThreshold();
            if (SaveYtDialog.PrintPeakZero())
                table.at(n_s,nCol++)=GetPeak();
            if (SaveYtDialog.PrintPeakBase())
                table.at(n_s,nCol++)=GetPeak()-GetBase();
            if (SaveYtDialog.PrintPeakThreshold())
                table.at(n_s,nCol++)=GetPeak()-GetThreshold();
            if (SaveYtDialog.PrintRT2080())
                table.at(n_s,nCol++)=GetRT2080();
            if (SaveYtDialog.PrintT50())
                table.at(n_s,nCol++)=GetHalfDuration();
            if (SaveYtDialog.PrintSlopes()) {
                table.at(n_s,nCol++)=GetMaxRise();
                table.at(n_s,nCol++)=GetMaxDecay();
            }
            if (SaveYtDialog.PrintLatencies()) {
                table.at(n_s,nCol++)=GetLatency()*GetXScale();
            }
            if (SaveYtDialog.PrintFitResults()) {
                for (std::size_t n_pf=0;n_pf<n_params;++n_pf) {
                    table.at(n_s,nCol++)=params[n_pf];
                }
                if (fitWarning != 0) {
                    table.at(n_s,nCol++) = (double)fitWarning;
                } else {
                    table.SetEmpty(n_s,nCol++);
                }
            }
            if (SaveYtDialog.PrintThr()) {
                table.at(n_s,nCol++)=n_crossings;
            }
        }
        catch (const std::out_of_range& e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
            SetSection(section_old);
            return;
        }
        n_s++;
    }
    progDlg.Update(100,wxT("Finished"));
    SetSection(section_old);
    wxStfChildFrame* pFrame=(wxStfChildFrame*)GetDocumentWindow();
    pFrame->ShowTable(table,wxT("Batch analysis results"));
}

void wxStfDoc::OnAnalysisIntegrate(wxCommandEvent &WXUNUSED(event)) {
    double integral_s = stf::integrate_simpson(cur().get(),GetFitBeg(),GetFitEnd(),GetXScale());
    double integral_t = stf::integrate_trapezium(cur().get(),GetFitBeg(),GetFitEnd(),GetXScale());
    stf::Table integralTable(6,1);
    try {
        integralTable.SetRowLabel(0, wxT("Trapezium (linear)"));
        integralTable.SetRowLabel(1, wxT("Integral (from 0)"));
        integralTable.SetRowLabel(2, wxT("Integral (from base)"));
        integralTable.SetRowLabel(3, wxT("Simpson (quadratic)"));
        integralTable.SetRowLabel(4, wxT("Integral (from 0)"));
        integralTable.SetRowLabel(5, wxT("Integral (from base)"));
        integralTable.SetColLabel(0, wxT("Result"));
        integralTable.SetEmpty(0,0);
        integralTable.at(1,0) = integral_t;
        integralTable.at(2,0) =
            integral_t - (GetFitEnd()-GetFitBeg())*GetXScale()*GetBase();
        integralTable.SetEmpty(3,0);
        integralTable.at(4,0) = integral_s;
        integralTable.at(5,0) =
            integral_s - (GetFitEnd()-GetFitBeg())*GetXScale()*GetBase();
    }
    catch (const std::out_of_range& e) {
        wxGetApp().ErrorMsg(wxString( e.what(), wxConvLocal ));
        return;
    }
    wxStfChildFrame* pFrame=(wxStfChildFrame*)GetDocumentWindow();
    pFrame->ShowTable(integralTable,wxT("Integral"));
    try {
        cur().SetIsIntegrated(true,GetFitBeg(),GetFitEnd());
    }
    catch (const std::runtime_error& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
    catch (const std::out_of_range& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
}

void wxStfDoc::OnAnalysisDifferentiate(wxCommandEvent &WXUNUSED(event)) {
    if (GetSelectedSections().empty()) {
        wxGetApp().ErrorMsg(wxT("Select traces first"));
        return;
    }
    Channel TempChannel(GetSelectedSections().size(), get()[GetCurCh()][GetSelectedSections()[0]].size());
    std::size_t n = 0;
    for (c_st_it cit = GetSelectedSections().begin(); cit != GetSelectedSections().end(); cit++) {
        Section TempSection( stf::diff( get()[GetCurCh()][*cit].get(), GetXScale() ) );
        TempSection.SetSectionDescription( get()[GetCurCh()][*cit].GetSectionDescription()+
                wxT(", differentiated") );
        try {
            TempChannel.InsertSection(TempSection,n);
        }
        catch (const std::out_of_range& e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
        }
        n++;
    }
    if (TempChannel.size()>0) {
        Recording Diff(TempChannel);
        Diff.CopyAttributes(*this);
        wxString title(GetTitle());
        title+=wxT(", differentiated");
        wxGetApp().NewChild(Diff,this,title);
    }

}

bool wxStfDoc::OnNewfromselectedThis( ) {
    if (GetSelectedSections().empty()) {
        wxGetApp().ErrorMsg(wxT("Select traces first"));
        return false;
    }

    Channel TempChannel(GetSelectedSections().size(), get()[GetCurCh()][GetSelectedSections()[0]].size());
    std::size_t n = 0;
    for (c_st_it cit = GetSelectedSections().begin(); cit != GetSelectedSections().end(); cit++) {
        // Multiply the valarray in Data:
        Section TempSection(get()[GetCurCh()][*cit].get());
        TempSection.SetSectionDescription( get()[GetCurCh()][*cit].GetSectionDescription()+
                wxT(", new from selected") );
        try {
            TempChannel.InsertSection(TempSection,n);
        }
        catch (const std::out_of_range e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
            return false;
        }
        n++;
    }
    if (TempChannel.size()>0) {
        Recording Selected(TempChannel);
        Selected.CopyAttributes(*this);
        Selected[0].SetYUnits( at(GetCurCh()).GetYUnits() );
        wxString title(GetTitle());
        title+=wxT(", new from selected");
        wxGetApp().NewChild(Selected,this,title);
    } else {
        wxGetApp().ErrorMsg( wxT("Channel is empty.") );
        return false;
    }
    return true;
}

void wxStfDoc::Selectsome(wxCommandEvent &WXUNUSED(event)) {
    if (GetSelectedSections().size()>0) {
        wxGetApp().ErrorMsg(wxT("Unselect all"));
        return;
    }
    //insert standard values:
    std::vector<wxString> labels(2);
    std::vector<double> defaults(labels.size());
    labels[0]=wxT("Select every x-th trace:");defaults[0]=1;
    labels[1]=wxT("Starting with the y-th:");defaults[1]=1;
    stf::UserInput init(labels,defaults,wxT("Select every n-th (1-based)"));

    wxStfUsrDlg EveryDialog(GetDocumentWindow(),init);
    if (EveryDialog.ShowModal()!=wxID_OK) return;
    std::vector<double> input(EveryDialog.readInput());
    if (input.size()!=2) return;
    int everynth=(int)input[0];
    int everystart=(int)input[1];
    div_t n_selected=div((int)get()[GetCurCh()].size(),everynth);
    for (int n=0; n*everynth+everystart-1 < (int)get()[GetCurCh()].size(); ++n) {
        try {
            SelectTrace(n*everynth+everystart-1);
        }
        catch (const std::out_of_range& e) {
            wxGetApp().ExceptMsg( wxString::FromAscii(e.what()) );
        }
    }
    wxStfChildFrame* pFrame=(wxStfChildFrame*)GetDocumentWindow();
    pFrame->SetSelected(GetSelectedSections().size());
    Focus();
}

void wxStfDoc::Unselectsome(wxCommandEvent &WXUNUSED(event)) {
    if (GetSelectedSections().size() < get()[GetCurCh()].size()) {
        wxGetApp().ErrorMsg(wxT("Select all traces first"));
        return;
    }
    //insert standard values:
    std::vector<wxString> labels(2);
    std::vector<double> defaults(labels.size());
    labels[0]=wxT("Unselect every x-th trace:");defaults[0]=1;
    labels[1]=wxT("Starting with the y-th:");defaults[1]=1;
    stf::UserInput init(labels,defaults,wxT("Unselect every n-th (1-based)"));

    wxStfUsrDlg EveryDialog(GetDocumentWindow(),init);
    if (EveryDialog.ShowModal()!=wxID_OK) return;
    std::vector<double> input(EveryDialog.readInput());
    if (input.size()!=2) return;
    int everynth=(int)input[0];
    int everystart=(int)input[1];
    div_t n_unselected=div((int)get()[GetCurCh()].size(),everynth);
    for (int n=0; n*everynth+everystart-1 < (int)get()[GetCurCh()].size(); ++n) {
        UnselectTrace(n*everynth+everystart-1);
    }
    wxStfChildFrame* pFrame=(wxStfChildFrame*)GetDocumentWindow();
    pFrame->SetSelected(GetSelectedSections().size());
    Focus();
}

void wxStfDoc::Selectall(wxCommandEvent& event) {
    //Make sure all traces are unselected prior to selecting them all:
    if ( !GetSelectedSections().empty() )
        Deleteselected(event);
    for (int n_s=0; n_s<(int)get()[GetCurCh()].size(); ++n_s) {
        SelectTrace(n_s);
    }
    wxStfChildFrame* pFrame=(wxStfChildFrame*)GetDocumentWindow();
    pFrame->SetSelected(GetSelectedSections().size());
    Focus();
}

void wxStfDoc::Deleteselected(wxCommandEvent &WXUNUSED(event)) {
    wxStfChildFrame* pFrame=(wxStfChildFrame*)GetDocumentWindow();
    if( !GetSelectedSections().empty() ) {
        GetSelectedSectionsW().clear();
        GetSelectBaseW().clear();
        //Update selected traces string in the trace navigator
        pFrame->SetSelected(GetSelectedSections().size());
    } else {
        wxGetApp().ErrorMsg(wxT("No selected trace to remove"));
        return;
    }
    // refresh the view once we are through:
    if (pFrame->PlotSelected()) {
        wxStfView* pView=(wxStfView*)GetFirstView();
        pView->GetGraph()->Refresh();
    }
    Focus();
}

void wxStfDoc::Focus() {
    // refresh the view once we are through:
    wxStfView* pView=(wxStfView*)GetFirstView();
    if (pView != NULL && pView->GetGraph() != NULL) {
        pView->GetGraph()->SetFocus();
    }
    
    UpdateSelectedButton();
}

void wxStfDoc::UpdateSelectedButton() {
    //control whether trace has selected been selected:
    bool selected=false;
    for (c_st_it cit = GetSelectedSections().begin();
         cit != GetSelectedSections().end() && !selected;
         ++cit) {
        if (*cit == GetCurSec()) {
            selected = true;
        }
    }

    // Set status of selection button:
    wxStfParentFrame* parentFrame = GetMainFrame();
    if (parentFrame) {
        parentFrame->SetSelectedButton( selected );
    }

}

void wxStfDoc::Filter(wxCommandEvent& WXUNUSED(event)) {
#ifndef TEST_MINIMAL
    if (GetSelectedSections().empty()) {
        wxGetApp().ErrorMsg(wxT("No traces selected"));
        return;
    }

    //--For details on the Fast Fourier Transform see NR in C++, chapters 12 and 13
    std::vector<wxString> windowLabels(2);
    std::vector<double> windowDefaults(windowLabels.size());
    windowLabels[0]=wxT("From point #:");windowDefaults[0]=0;
    windowLabels[1]=wxT("To point #:");windowDefaults[1]=(int)cur().size()-1;
    stf::UserInput initWindow(windowLabels,windowDefaults,wxT("Filter window"));

    wxStfUsrDlg FilterWindowDialog(GetDocumentWindow(),initWindow);
    if (FilterWindowDialog.ShowModal()!=wxID_OK) return;
    std::vector<double> windowInput(FilterWindowDialog.readInput());
    if (windowInput.size()!=2) return;
    int llf=(int)windowInput[0];
    int ulf=(int)windowInput[1];

    wxStfFilterSelDlg FilterSelectDialog(GetDocumentWindow());
    if (FilterSelectDialog.ShowModal()!=wxID_OK) return;
    int fselect=FilterSelectDialog.GetFilterSelect();
    int size=0;
    bool inverse=true;
    switch (fselect) {
    case 1:
        size=3; break;
    case 2:
    case 3:
        size=1;
        break;
    }
    wxStfGaussianDlg FftDialog(GetDocumentWindow());

    std::valarray<double> a(size);
    switch (fselect) {
    case 1:
        if (FftDialog.ShowModal()!=wxID_OK) return;
        a[0]=(int)(FftDialog.Amp()*100000.0)/100000.0;	/*amplitude from 0 to 1*/
        a[1]=(int)(FftDialog.Center()*100000.0)/100000.0;	/*center in kHz*/
        a[2]=(int)(FftDialog.Width()*100000.0)/100000.0;	/*width in kHz*/
        break;
    case 2:
    case 3: {
        //insert standard values:
        std::vector<wxString> labels(1);
        std::vector<double> defaults(labels.size());
        labels[0]=wxT("Cutoff frequency (kHz):");
        defaults[0]=10;
        stf::UserInput init(labels,defaults,wxT("Set frequency"));

        wxStfUsrDlg FilterHighLowDialog(GetDocumentWindow(),init);
        if (FilterHighLowDialog.ShowModal()!=wxID_OK) return;
        std::vector<double> input(FilterHighLowDialog.readInput());
        if (input.size()!=1) return;
        a[0]=(int)(input[0]*100000.0)/100000.0;    /*midpoint of sigmoid curve in kHz*/
        break;
    }
    }

    wxBusyCursor wc;

    //--I. Defining the parameters of the filter function

    /*sampling interval in ms*/

    Channel TempChannel(GetSelectedSections().size(), get()[GetCurCh()][GetSelectedSections()[0]].size());
    std::size_t n = 0;
    for (c_st_it cit = GetSelectedSections().begin(); cit != GetSelectedSections().end(); cit++) {
        try {
            switch (fselect) {
                case 3: {
                    Section FftTemp(stf::filter(get()[GetCurCh()][*cit].get(),
                            llf,ulf,a,(int)GetSR(),stf::fgaussColqu,false));
                    FftTemp.SetSectionDescription( get()[GetCurCh()][*cit].GetSectionDescription()+
                                                   wxT(", filtered") );
                    TempChannel.InsertSection(FftTemp, n);
                    break;
                }
                case 2: {
                    Section FftTemp(stf::filter(get()[GetCurCh()][*cit].get(),
                            llf,ulf,a,(int)GetSR(),stf::fbessel4,false));
                    FftTemp.SetSectionDescription( get()[GetCurCh()][*cit].GetSectionDescription()+
                                                   wxT(", filtered") );
                    TempChannel.InsertSection(FftTemp, n);
                    break;
                }
                case 1: {
                    Section FftTemp(stf::filter(get()[GetCurCh()][*cit].get(),
                            llf,ulf,a,(int)GetSR(),stf::fgauss,inverse));
                    FftTemp.SetSectionDescription( get()[GetCurCh()][*cit].GetSectionDescription()+
                                                   wxT(", filtered") );
                    TempChannel.InsertSection(FftTemp, n);
                    break;
                }
            }
        }
        catch (const std::exception& e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
        }
        n++;
    }
    if (TempChannel.size()>0) {
        Recording Fft(TempChannel);
        Fft.CopyAttributes(*this);
        wxGetApp().NewChild(Fft, this,GetTitle()+wxT(", filtered"));
    }
#endif
}

void wxStfDoc::Spectrum(wxCommandEvent& WXUNUSED(event)) {
#ifndef TEST_MINIMAL
    if (GetSelectedSections().empty()) {
        wxGetApp().ErrorMsg(wxT("No traces selected"));
        return;
    }
    //insert standard values:
    std::vector<wxString> labels(1);
    std::vector<double> defaults(labels.size());
    labels[0]=wxT("Number of periodograms:");defaults[0]=10;
    stf::UserInput init(labels,defaults,wxT("Settings for Welch's method"));

    wxStfUsrDlg SegDialog(GetDocumentWindow(),init);
    if (SegDialog.ShowModal()!=wxID_OK) return;
    std::vector<double> input(SegDialog.readInput());
    if (input.size()!=1) return;

    int n_seg=(int)SegDialog.readInput()[0];
    wxBusyCursor wc;

    Channel TempChannel(GetSelectedSections().size(),
            get()[GetCurCh()][GetSelectedSections()[0]].size());
    double f_s=1.0; // frequency stepsize of the spectrum
    std::size_t n = 0;
    for (c_st_it cit = GetSelectedSections().begin(); cit != GetSelectedSections().end(); cit++) {
        std::valarray< std::complex<double> > temp(std::complex<double>(0.0,0.0),
                get()[GetCurCh()][*cit].size());
        for (int i=0;i<(int)get()[GetCurCh()][*cit].size();++i) {
            temp[i]=get()[GetCurCh()][*cit][i];
        }
        try {
            Section TempSection(stf::spectrum(temp,n_seg,f_s));
            TempSection.SetSectionDescription(
                    get()[GetCurCh()][*cit].GetSectionDescription()+
                    wxT(", spectrum") );
            TempChannel.InsertSection(TempSection,n);
        }
        catch (const std::runtime_error& e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
            return;
        }
        catch (const std::out_of_range& e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
        }
        n++;
    }
    if (TempChannel.size()>0) {
        Recording Fft(TempChannel);
        Fft.CopyAttributes(*this);
        double unit_f=f_s/GetXScale();
        Fft[0].SetYUnits( at( GetCurCh() ).GetYUnits()+wxChar(-78));
        Fft.SetXScale(unit_f);
        wxGetApp().NewChild(Fft,this,GetTitle()+wxT(", spectrum"));
    }
#endif
}

void wxStfDoc::P_over_N(wxCommandEvent& WXUNUSED(event)){
    //insert standard values:
    std::vector<wxString> labels(1);
    std::vector<double> defaults(labels.size());
    labels[0]=wxT("N = (mind polarity!)");defaults[0]=-4;
    stf::UserInput init(labels,defaults,wxT("P over N"));

    wxStfUsrDlg PonDialog(GetDocumentWindow(),init);
    if (PonDialog.ShowModal()!=wxID_OK) return;
    std::vector<double> input(PonDialog.readInput());
    if (input.size()!=1) return;
    int PoN=(int)fabs(input[0]);
    int ponDirection=input[0]<0? -1:1;
    int new_sections=(int)get()[GetCurCh()].size()/(PoN+1);
    if (new_sections<1) {
        wxGetApp().ErrorMsg(wxT("Not enough traces for P/n correction"));
        return;
    }

    //File dialog box
    wxBusyCursor wc;
    Channel TempChannel(new_sections);

    //read and PoN
    for (int n_section=0; n_section < new_sections; n_section++) {
        //Section loop
        Section TempSection(get()[GetCurCh()][n_section].size());
        for (int n_point=0; n_point < (int)get()[GetCurCh()][n_section].size(); n_point++)
            TempSection[n_point]=0.0;

        //Addition of the PoN-values:
        for (int n_PoN=1; n_PoN < PoN+1; n_PoN++)
            for (int n_point=0; n_point < (int)get()[GetCurCh()][n_section].size(); n_point++)
                TempSection[n_point] += get()[GetCurCh()][n_PoN+(n_section*(PoN+1))][n_point];

        //Subtraction from the original values:
        for (int n_point=0; n_point < (int)get()[GetCurCh()][n_section].size(); n_point++)
            TempSection[n_point] = get()[GetCurCh()][n_section*(PoN+1)][n_point]-
                    TempSection[n_point]*ponDirection;
        wxString povernLabel;
        povernLabel << GetTitle() << wxT(", #") << n_section << wxT(", P over N");
        TempSection.SetSectionDescription( povernLabel );
        try {
            TempChannel.InsertSection(TempSection,n_section);
        }
        catch (const std::out_of_range& e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
        }
    }
    if (TempChannel.size()>0) {
        Recording DataPoN(TempChannel);
        DataPoN.CopyAttributes(*this);
        wxGetApp().NewChild(DataPoN,this,GetTitle()+wxT(", p over n subtracted"));
    }

}

void wxStfDoc::Plotcriterion(wxCommandEvent& WXUNUSED(event)) {
    std::vector<Section*> sectionList(wxGetApp().GetSectionsWithFits());
    if (sectionList.empty()) {
        wxGetApp().ErrorMsg(
                wxT("You have to create a template first\nby fitting a function to an event") );
        return;
    }
    wxStfEventDlg MiniDialog(GetDocumentWindow(),wxGetApp().GetSectionsWithFits(),false);
    if (MiniDialog.ShowModal()!=wxID_OK)  {
        return;
    }
    int nTemplate=MiniDialog.GetTemplate();
    try {
        std::valarray<double> templateWave(
                sectionList.at(nTemplate)->GetStoreFitEnd() -
                sectionList.at(nTemplate)->GetStoreFitBeg());
        for ( std::size_t n_p=0; n_p < templateWave.size(); n_p++ ) {
            templateWave[n_p] = sectionList.at(nTemplate)->GetFitFunc()->func(
                    n_p*GetXScale(), sectionList.at(nTemplate)->GetBestFitP());
        }
        wxBusyCursor wc;
#undef min
#undef max
        // subtract offset and normalize:
        templateWave-=templateWave.max();
        double minim=fabs(templateWave.min());
        templateWave /= minim;
        Section TempSection(
                stf::detectionCriterion( cur().get(), templateWave ) );
        if (TempSection.size()==0) return;
        TempSection.SetSectionDescription(
                wxT("Detection criterion of ")+cur().GetSectionDescription()
        );
        Channel TempChannel(TempSection);
        Recording detCrit(TempChannel);
        detCrit.CopyAttributes(*this);
        wxGetApp().NewChild(detCrit,this,GetTitle()+wxT(", detection criterion"));
    }
    catch (const std::runtime_error& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
    catch (const std::exception& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
}

void wxStfDoc::Plotcorrelation(wxCommandEvent& WXUNUSED(event)) {
    std::vector<Section*> sectionList(wxGetApp().GetSectionsWithFits());
    if (sectionList.empty()) {
        wxGetApp().ErrorMsg(
                wxT("You have to create a template first\nby fitting a function to an event")
        );
        return;
    }
    wxStfEventDlg MiniDialog(GetDocumentWindow(),wxGetApp().GetSectionsWithFits(),false);
    if (MiniDialog.ShowModal()!=wxID_OK)  {
        return;
    }
    int nTemplate=MiniDialog.GetTemplate();
    try {
        std::valarray<double> templateWave(
                sectionList.at(nTemplate)->GetStoreFitEnd() -
                sectionList.at(nTemplate)->GetStoreFitBeg());
        for ( std::size_t n_p=0; n_p < templateWave.size(); n_p++ ) {
            templateWave[n_p] = sectionList.at(nTemplate)->GetFitFunc()->func(
                    n_p*GetXScale(), sectionList.at(nTemplate)->GetBestFitP());
        }
        wxBusyCursor wc;
#undef min
#undef max
        // subtract offset and normalize:
        templateWave-=templateWave.max();
        double minim=fabs(templateWave.min());
        templateWave /= minim;
        Section TempSection( stf::linCorr( cur().get(), templateWave ) );
        if (TempSection.size()==0) return;
        TempSection.SetSectionDescription(
                wxT("Template correlation of ")+cur().GetSectionDescription() );
        Channel TempChannel(TempSection);
        Recording detCrit(TempChannel);
        detCrit.CopyAttributes(*this);
        wxGetApp().NewChild(detCrit,this,GetTitle()+wxT(", linear correlation"));
    }
    catch (const std::runtime_error& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
    catch (const std::exception& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
}

void wxStfDoc::MarkEvents(wxCommandEvent& WXUNUSED(event)) {
    std::vector<Section*> sectionList(wxGetApp().GetSectionsWithFits());
    if (sectionList.empty()) {
        wxGetApp().ErrorMsg(
                wxT( "You have to create a template first\nby fitting a function to an event" ) );
        return;
    }
    wxStfEventDlg MiniDialog( GetDocumentWindow(), wxGetApp().GetSectionsWithFits(), true );
    if ( MiniDialog.ShowModal()!=wxID_OK ) {
        return;
    }
    int nTemplate=MiniDialog.GetTemplate();
    try {
        std::valarray<double> templateWave(
                sectionList.at(nTemplate)->GetStoreFitEnd() -
                sectionList.at(nTemplate)->GetStoreFitBeg());
        for ( std::size_t n_p=0; n_p < templateWave.size(); n_p++ ) {
            templateWave[n_p] = sectionList.at(nTemplate)->GetFitFunc()->func(
                    n_p*GetXScale(), sectionList.at(nTemplate)->GetBestFitP());
        }
        wxBusyCursor wc;
#undef min
#undef max
        // subtract offset and normalize:
        templateWave-=templateWave.max();
        double minim=fabs(templateWave.min());
        templateWave /= minim;
        std::valarray<double> detect( cur().get().size() - templateWave.size() );
        if (MiniDialog.GetScaling()) {
            detect=stf::detectionCriterion( cur().get(),templateWave );
        } else {
            detect=stf::linCorr(
                    cur().get(),templateWave
            );
        }
        if (detect.size()==0) return;
        std::vector<int> startIndices(
                stf::peakIndices( detect, MiniDialog.GetThreshold(),
                        MiniDialog.GetMinDistance() ) );
        if (startIndices.empty()) {
            wxGetApp().ErrorMsg( wxT( "No events were found. Try to lower the threshold." ) );
            return;
        }
        // erase old events:
        cur().EraseEvents();
        for (c_int_it cit = startIndices.begin(); cit != startIndices.end(); ++cit ) {
            cur().CreateEvent( stf::Event( *cit, 0, templateWave.size() ) );
            // Find peak in this event:
            double baselineMean=0;
            for ( std::size_t n_mean = (std::size_t)*cit-baseline;
            n_mean < (std::size_t)(*cit);
            ++n_mean )
            {
                if (n_mean < 0) {
                    baselineMean += cur().at(0);
                } else {
                    baselineMean += cur().at(n_mean);
                }
            }
            baselineMean /= baseline;
            double peakIndex=0;
            stf::peak( cur().get(), baselineMean, *cit, *cit + templateWave.size(),
                    1, stf::both, peakIndex );
            // set peak index of last event:
            cur().GetEventsW()[cur().GetEvents().size()-1].SetEventPeakIndex((int)peakIndex);
        }
    }
    catch (const std::runtime_error& e) {
        wxGetApp().ExceptMsg( wxString( e.what(), wxConvLocal ));
    }
    catch (const std::exception& e) {
        wxGetApp().ExceptMsg( wxString( e.what(), wxConvLocal ));
    }
}

void wxStfDoc::Extract( wxCommandEvent& WXUNUSED(event) ) {
    try {
        stf::Table events(cur().GetEvents().size(),2);
        events.SetColLabel(0,wxT("Time of event onset"));
        events.SetColLabel(1,wxT("Inter-event interval"));
        // using the peak indices (these are the locations of the beginning of an optimal
        // template matching), new sections are created:

        // count non-discarded events:
        std::size_t n_real = 0;
        for (c_event_it cit = cur().GetEvents().begin(); cit != cur().GetEvents().end(); ++cit) {
            n_real += (int)(!cit->GetDiscard());
        }
        Channel TempChannel2(n_real);
        std::vector<int> peakIndices(n_real);
        n_real = 0;
        c_event_it lastEventIt = cur().GetEvents().begin();
        for (event_it it = cur().GetEventsW().begin(); it != cur().GetEventsW().end(); ++it) {
            if (!it->GetDiscard()) {
                wxString miniName; miniName << wxT( "Event #" ) << (int)n_real+1;
                events.SetRowLabel(n_real, miniName);
                events.at(n_real,0) = (double)it->GetEventStartIndex() / GetSR();
                events.at(n_real,1)=
                    ((double)(it->GetEventStartIndex() -
                            lastEventIt->GetEventStartIndex())) / GetSR();
                // add some baseline at the beginning and end:
                std::size_t eventSize = it->GetEventSize() + 2*baseline;
                Section TempSection2( eventSize );
                for ( std::size_t n_new = 0; n_new < eventSize; ++n_new ) {
                    // make sure index is not out of range:
                    int index = it->GetEventStartIndex() + n_new - baseline;
                    if (index < 0)
                        index = 0;
                    if (index >= (int)cur().size())
                        index = cur().size()-1;
                    TempSection2[n_new] = cur()[index];
                }
                wxString eventDesc;
                eventDesc << wxT( "Extracted event #" ) << (int)n_real;
                TempSection2.SetSectionDescription(eventDesc);
                TempChannel2.InsertSection( TempSection2, n_real );
                n_real++;
                lastEventIt = it;
            }
        }
        if (TempChannel2.size()>0) {
            Recording Minis( TempChannel2 );
            Minis.CopyAttributes( *this );
            wxStfDoc* pDoc=wxGetApp().NewChild( Minis, this,
                    GetTitle()+wxT(", extracted events") );
            if (pDoc != NULL) {
                wxStfChildFrame* pChild=(wxStfChildFrame*)pDoc->GetDocumentWindow();
                if (pChild!=NULL) {
                    pChild->ShowTable(events,wxT("Extracted events"));
                }
            }
        }
    }
    catch (const std::runtime_error& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
    catch (const std::exception& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
}

void wxStfDoc::EraseEvents( wxCommandEvent& WXUNUSED(event) ) {
    if (wxMessageDialog( GetDocumentWindow(), wxT("Do you really want to erase all events?"),
                         wxT("Erase all events"), wxYES_NO ).ShowModal()==wxID_YES)
    {
        cur().EraseEvents();
    }
}

void wxStfDoc::AddEvent( wxCommandEvent& WXUNUSED(event) ) {
    try {
        // retrieve the position where to add the event:
        wxStfView* pView = (wxStfView*)GetFirstView();
        wxStfGraph* pGraph = pView->GetGraph();
        int newStartPos = pGraph->get_eventPos();
        stf::Event newEvent(newStartPos, 0, cur().GetEvent(0).GetEventSize());
        // Find peak in this event:
        double baselineMean=0;
        for ( std::size_t n_mean = (std::size_t)newStartPos - baseline;
        n_mean < (std::size_t)newStartPos;
        ++n_mean )
        {
            if (n_mean < 0) {
                baselineMean += cur().at(0);
            } else {
                baselineMean += cur().at(n_mean);
            }
        }
        baselineMean /= baseline;
        double peakIndex=0;
        stf::peak( cur().get(), baselineMean, newStartPos,
                newStartPos + cur().GetEvent(0).GetEventSize(), 1,
                stf::both, peakIndex );
        // set peak index of last event:
        newEvent.SetEventPeakIndex( (int)peakIndex );
        // find the position in the current event list where the new
        // event should be inserted:
        bool found = false;
        for (event_it it = cur().GetEventsW().begin(); it != cur().GetEventsW().end(); ++it) {
            if ( (int)(it->GetEventStartIndex()) > newStartPos ) {
                // insert new event before this event, then break:
                cur().GetEventsW().insert( it, newEvent );
                found = true;
                break;
            }
        }
        // if we are at the end of the list, append the event:
        if (!found)
            cur().GetEventsW().push_back( newEvent );
    }
    catch (const std::runtime_error& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
    catch (const std::exception& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
}

void wxStfDoc::Threshold(wxCommandEvent& WXUNUSED(event)) {
    // get threshold from user input:
    std::vector<double> threshold(0);
    stf::UserInput Input( std::vector<wxString>(1,wxT("Threshold")),
            std::vector<double> (1,0.0), wxT("Set threshold") );
    wxStfUsrDlg myDlg( GetDocumentWindow(), Input );
    if (myDlg.ShowModal()!=wxID_OK) {
        return;
    }
    threshold=myDlg.readInput();

    std::vector<int> startIndices(
            stf::peakIndices( cur().get(), threshold[0], 0 )
    );
    if (startIndices.empty()) {
        wxGetApp().ErrorMsg(
                wxT("Couldn't find any events;\ntry again with lower threshold")
        );
    }
    for (c_int_it cit = startIndices.begin(); cit != startIndices.end(); ++cit) {
        cur().CreateEvent( stf::Event( *cit, 0, baseline ) );
    }
    // show results in a table:
    stf::Table events(cur().GetEvents().size(),2);
    events.SetColLabel( 0, wxT( "Time of event peak" ) );
    events.SetColLabel( 1, wxT( "Inter-event interval" ) );
    std::size_t n_event = 0;
    c_event_it lastEventCit = cur().GetEvents().begin();
    for (c_event_it cit2 = cur().GetEvents().begin(); cit2 != cur().GetEvents().end(); ++cit2) {
        wxString eventName; eventName << wxT("Event #") << (int)n_event+1;
        events.SetRowLabel(n_event,eventName);
        events.at(n_event,0)= (double)cit2->GetEventStartIndex() / GetSR();
        events.at(n_event,1)=
            ((double)(cit2->GetEventStartIndex() -
                    lastEventCit->GetEventStartIndex()) ) / GetSR();
        n_event++;
        lastEventCit = cit2;
    }
    wxStfChildFrame* pChild=(wxStfChildFrame*)GetDocumentWindow();
    if (pChild!=NULL) {
        pChild->ShowTable(events,wxT("Extracted events"));
    }
}

void wxStfDoc::Userdef(std::size_t id) {
    wxBusyCursor wc;
    int fselect=(int)id;
    Recording newR;
    std::vector<double> init(0);
    // get user input if necessary:
    if (!wxGetApp().GetPluginLib().at(fselect).input.labels.empty()) {
        wxStfUsrDlg myDlg( GetDocumentWindow(),
                wxGetApp().GetPluginLib().at(fselect).input );
        if (myDlg.ShowModal()!=wxID_OK) {
            return;
        }
        init=myDlg.readInput();
    }
    // Apply function to current valarray
    std::map< wxString, double > resultsMap;
    try {
        newR=wxGetApp().GetPluginLib().at(fselect).pluginFunc( *this, init, resultsMap );
    }
    catch (const std::out_of_range& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
        return;
    }
    catch (const std::runtime_error& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
        return;
    }
    catch (const std::exception& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
        return;
    }
    if (newR.size()==0) {
        return;
    }
    wxString newTitle(GetTitle());
    newTitle += wxT(", ");
    newTitle += wxGetApp().GetPluginLib().at(fselect).menuEntry;
    wxStfDoc* pDoc = wxGetApp().NewChild(newR,this,newTitle);
    ((wxStfChildFrame*)pDoc->GetDocumentWindow())->ShowTable(
            stf::Table(resultsMap), wxGetApp().GetPluginLib().at(fselect).menuEntry
    );
}
