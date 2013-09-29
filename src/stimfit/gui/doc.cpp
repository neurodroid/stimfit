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
#include "./../math/stfmath.h"
#include "./../math/fit.h"
#include "./../math/funclib.h"
#include "./../math/measure.h"
#include "./../../libstfio/stfio.h"
#include "./usrdlg/usrdlg.h"
#include "./doc.h"
#include "./graph.h"

IMPLEMENT_DYNAMIC_CLASS(wxStfDoc, wxDocument)

BEGIN_EVENT_TABLE( wxStfDoc, wxDocument )
EVT_MENU( ID_SWAPCHANNELS, wxStfDoc::OnSwapChannels )
EVT_MENU( ID_FILEINFO, wxStfDoc::Fileinfo)
EVT_MENU( ID_NEWFROMSELECTEDTHIS, wxStfDoc::OnNewfromselectedThisMenu  )
EVT_MENU( ID_MYSELECTALL, wxStfDoc::Selectall )
EVT_MENU( ID_UNSELECTALL, wxStfDoc::Deleteselected )
EVT_MENU( ID_SELECTSOME, wxStfDoc::Selectsome )
EVT_MENU( ID_UNSELECTSOME, wxStfDoc::Unselectsome )
EVT_MENU( ID_CONCATENATE, wxStfDoc::Concatenate )
EVT_MENU( ID_BATCH, wxStfDoc::OnAnalysisBatch )
EVT_MENU( ID_INTEGRATE, wxStfDoc::OnAnalysisIntegrate )
EVT_MENU( ID_DIFFERENTIATE, wxStfDoc::OnAnalysisDifferentiate )
EVT_MENU( ID_MULTIPLY, wxStfDoc::Multiply)
EVT_MENU( ID_SUBTRACTBASE, wxStfDoc::SubtractBaseMenu )
EVT_MENU( ID_FIT, wxStfDoc::FitDecay)
EVT_MENU( ID_LFIT, wxStfDoc::LFit)
EVT_MENU( ID_LOG, wxStfDoc::LnTransform)
EVT_MENU( ID_FILTER,wxStfDoc::Filter)
EVT_MENU( ID_POVERN,wxStfDoc::P_over_N)
EVT_MENU( ID_PLOTCRITERION,wxStfDoc::Plotcriterion)
EVT_MENU( ID_PLOTCORRELATION,wxStfDoc::Plotcorrelation)
EVT_MENU( ID_PLOTDECONVOLUTION,wxStfDoc::Plotdeconvolution)
EVT_MENU( ID_EXTRACT,wxStfDoc::MarkEvents )
EVT_MENU( ID_THRESHOLD,wxStfDoc::Threshold)
EVT_MENU( ID_VIEWTABLE, wxStfDoc::Viewtable)
EVT_MENU( ID_EVENT_EXTRACT, wxStfDoc::Extract )
EVT_MENU( ID_EVENT_ERASE, wxStfDoc::InteractiveEraseEvents )
EVT_MENU( ID_EVENT_ADDEVENT, wxStfDoc::AddEvent )
END_EVENT_TABLE()

static const int baseline=100;
// static const double rtfrac = 0.2; // now expressed in percentage, see RTFactor

wxStfDoc::wxStfDoc() :
    Recording(),peakAtEnd(false),initialized(false),progress(true), Average(0),
    latencyStartMode(stf::riseMode),
    latencyEndMode(stf::footMode),
    latencyWindowMode(stf::defaultMode),
    direction(stf::both),    
#ifdef WITH_PSLOPE
    pslopeBegMode(stf::psBeg_manualMode),
    pslopeEndMode(stf::psEnd_manualMode),
#endif
    baseBeg(0),
    baseEnd(0),
    peakBeg(0),
    peakEnd(0),
    fitBeg(0),
    fitEnd(0),
#ifdef WITH_PSLOPE
    PSlopeBeg(0),
    PSlopeEnd(0),
    DeltaT(0),
    viewPSlope(true),
#endif
    measCursor(0),
    latencyStartCursor(0.0),
    latencyEndCursor(0.0),
    latency(0.0),
    base(0.0),
    APBase(0.0),
    baseSD(0.0),
    threshold(0.0),
    slopeForThreshold(20.0),
    peak(0.0),
    APPeak(0.0),
    tLoReal(0),
    tHiReal(0),
    t50LeftReal(0),
    t50RightReal(0),
    maxT(0.0),
    thrT(-1.0),
    maxRiseY(0.0),
    maxRiseT(0.0),
    maxDecayY(0.0),
    maxDecayT(0.0),
    maxRise(0.0),
    maxDecay(0.0),
    t50Y(0.0),
    APMaxT(0.0),
    APMaxRise(0.0),
    APMaxRiseT(0.0),
    APt50LeftReal(0.0),
    PSlope(0.0),
    rtLoHi(0.0),
    InnerLoRT(NAN),
    InnerHiRT(NAN),
    OuterLoRT(NAN),
    OuterHiRT(NAN),
    halfDuration(0.0),
    slopeRatio(0.0),
    t0Real(0.0),
    pM(1),
    RTFactor(20),
    cc(0),
    sc(1),
    cs(0),
    selectedSections(std::vector<std::size_t>(0)),
    selectBase(Vector_double(0)),
    tLoIndex(0),
    tHiIndex(0),
    t50LeftIndex(0),
    t50RightIndex(0),
    APt50LeftIndex(0),
    APt50RightIndex(0),
    fromBase(true),
    viewCrosshair(true),
    viewBaseline(true),
    viewBaseSD(true),
    viewThreshold(false),
    viewPeakzero(true),
    viewPeakbase(true),
    viewPeakthreshold(false),
    viewRTLoHi(true),
    viewInnerRiseTime(false),
    viewOuterRiseTime(false),
    viewT50(true),
    viewRD(true),
    viewSloperise(true),
    viewSlopedecay(true),
    viewLatency(true),
    viewCursors(true),
    xzoom(XZoom(0, 0.1, false)),
    yzoom(size(), YZoom(500,0.1,false)),
    sec_attr(size())
{
    for (std::size_t nchannel=0; nchannel < sec_attr.size(); ++nchannel) {
        sec_attr[nchannel].resize(at(nchannel).size());
    }
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

#ifndef TEST_MINIMAL
        // Detect type of file according to filter:
        wxString filter(GetDocumentTemplate()->GetFileFilter());
        stfio::filetype type = stfio::findType(stf::wx2std(filter));
#else
        stfio::filetype type = stfio::none;
#endif
#if 0 // TODO: backport ascii
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
#endif
        try {
            if (progress) {
                stf::wxProgressInfo progDlg("Reading file", "Opening file", 100);
                stfio::importFile(stf::wx2std(filename), type, *this, wxGetApp().GetTxtImport(), progDlg);
            } else {
                stfio::StdoutProgressInfo progDlg("Reading file", "Opening file", 100, true);
                stfio::importFile(stf::wx2std(filename), type, *this, wxGetApp().GetTxtImport(), progDlg);
            }
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
        catch (...) {
            wxString errorMsg(wxT("Error opening file\n"));
            wxGetApp().ExceptMsg(errorMsg);
            get().clear();
            return false;
        }
        if (get().empty()) {
            wxGetApp().ErrorMsg(wxT("File is probably empty\n"));
            get().clear();
            return false;
        }
        if (get()[0].get().empty()) {
            wxGetApp().ErrorMsg(wxT("File is probably empty\n"));
            get().clear();
            return false;
        }
        if (get()[0][0].get().empty()) {
            wxGetApp().ErrorMsg(wxT("File is probably empty\n"));
            get().clear();
            return false;
        }
        wxStfParentFrame* pFrame = GetMainFrame();
        if (pFrame == NULL) {
            throw std::runtime_error("pFrame is 0 in wxStfDoc::OnOpenDocument");
        }
        
        pFrame->SetSingleChannel( size() <= 1 );

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
    resize(c_Data.size());
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
    pFrame->SetSingleChannel( size() <= 1 );

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
#ifdef WITH_PSLOPE
        SetPSlopeBegMode ( Sender->GetPSlopeBegMode() );
        SetPSlopeEndMode ( Sender->GetPSlopeEndMode() );
#endif
        // Update menu checks:
        // UpdateMenuCheckmarks();
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
        // UpdateMenuCheckmarks();
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
    //Get values from .Stimfit and ensure proper settings
    SetMeasCursor(wxGetApp().wxGetProfileInt(wxT("Settings"), wxT("MeasureCursor"), 1));
    SetBaseBeg(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("BaseBegin"), 1));
    SetBaseEnd(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("BaseEnd"), 20));
    SetPeakBeg(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("PeakBegin"), (int)cur().size()-100));
    SetPeakEnd(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("PeakEnd"), (int)cur().size()-50));
    int iDirection = wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("Direction"),2);
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
    //SetLatencyStartMode( wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("LatencyStartMode"),0) );
    //SetLatencyEndMode( wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("LatencyEndMode"),0) );
    SetLatencyWindowMode(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("LatencyWindowMode"),1));
    // Set corresponding menu checkmarks:
    // UpdateMenuCheckmarks();
    SetPM(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("PeakMean"),1));
    SetRTFactor(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("RTFactor"),20));
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

#ifdef WITH_PSLOPE
    // read PSlope Beg mode from Stimfit registry
    int iPSlopeMode = wxGetApp().wxGetProfileInt( wxT("Settings"), wxT("PSlopeStartMode"), stf::psBeg_manualMode );
    switch (iPSlopeMode) {
        case 0:
            SetPSlopeBegMode( stf::psBeg_manualMode );
            break;
        case 1:
            SetPSlopeBegMode( stf::psBeg_footMode );
            break;
        case 2:
            SetPSlopeBegMode( stf::psBeg_thrMode );
            break;
        case 3:
            SetPSlopeBegMode( stf::psBeg_t50Mode );
            break;
        default:
            SetPSlopeBegMode( stf::psBeg_undefined );
    }
    // read PSlope End mode from Stimfit registry
    iPSlopeMode = wxGetApp().wxGetProfileInt( wxT("Settings"), wxT("PSlopeEndMode"), stf::psEnd_manualMode );
    switch (iPSlopeMode) {
        case 0:
            SetPSlopeEndMode( stf::psEnd_manualMode );
            break;
        case 1:
            SetPSlopeEndMode( stf::psEnd_t50Mode );
            break;
        case 2:
            SetPSlopeEndMode( stf::psEnd_DeltaTMode );
            break;
        case 3:
            SetPSlopeEndMode( stf::psEnd_peakMode );
            break;
        default:
            SetPSlopeEndMode( stf::psEnd_undefined );
    }
            
#endif

    CheckBoundaries();
    return wxID_OK;
}	//End SettingsDlg()

void wxStfDoc::PostInit() {
    wxStfChildFrame *pFrame = (wxStfChildFrame*)GetDocumentWindow();
    if ( pFrame == NULL ) {
        wxGetApp().ErrorMsg( wxT("Zero pointer in wxStfDoc::PostInit") );
        return;
    }

    // Update some vector sizes
    sec_attr.resize(size());
    for (std::size_t nchannel=0; nchannel < sec_attr.size(); ++nchannel) {
        sec_attr[nchannel].resize(at(nchannel).size());
    }
    yzoom.resize(size());
    
    try {
        pFrame->CreateMenuTraces(get().at(GetCurCh()).size());
        if ( size() > 1 ) {
            wxArrayString channelNames;
            channelNames.Alloc( size() );
            for (std::size_t n_c=0; n_c < size(); ++n_c) {
                wxString channelStream;
                channelStream << n_c << wxT(" (") << stf::std2wx( at(n_c).GetChannelName() ) << wxT(")");
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

    // Read results table settings from registry:
    SetViewCrosshair(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewCrosshair"),1)==1);
    SetViewBaseline(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewBaseline"),1)==1);
    SetViewBaseSD(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewBaseSD"),1)==1);
    SetViewThreshold(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewThreshold"),1)==1);
    SetViewPeakZero(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewPeakzero"),1)==1);
    SetViewPeakBase(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewPeakbase"),1)==1);
    SetViewPeakThreshold(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewPeakthreshold"),1)==1);
    SetViewRTLoHi(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewRTLoHi"),1)==1);
    SetViewInnerRiseTime(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewInnerRiseTime"),1)==1);
    SetViewOuterRiseTime(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewOuterRiseTime"),1)==1);
    SetViewT50(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewT50"),1)==1);
    SetViewRD(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewRD"),1)==1);
    SetViewSlopeRise(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewSloperise"),1)==1);
    SetViewSlopeDecay(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewSlopedecay"),1)==1);
#ifdef WITH_PSLOPE
    //SetViewPSlope(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewPSlope"),1)==1);
    SetViewPSlope(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewPSlope"),1)==1);
    //SetPSlopeBegMode( wxGetApp().wxGetProfileInt( wxT("Settings"), wxT("PSlopeStartMode"), 1)==1);
#endif
    SetViewLatency(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewLatency"),1)==1);
    SetViewCursors(wxGetApp().wxGetProfileInt(wxT("Settings"),wxT("ViewCursors"),1)==1);

    // refresh the view once we are through:
    initialized=true;
    pFrame->SetCurTrace(0);
    UpdateSelectedButton();
    wxGetApp().OnPeakcalcexecMsg();
    wxStfParentFrame* parentFrame = GetMainFrame();
    if (parentFrame) {
        parentFrame->SetFocus();
    }
    wxStfView* pView=(wxStfView*)GetFirstView();
    if (pView != NULL) {
        wxStfGraph* pGraph = pView->GetGraph();
        if (pGraph != NULL) {
            pGraph->Refresh();
            pGraph->Enable();
            // Set the focus:
            pGraph->SetFocus();
        }
    }
}

//Dialog box to select channel to be displayed
bool wxStfDoc::ChannelSelDlg() {
    // Set default channels:
    if ( size() < 2 ) {
        return false;
    }
    // SetCurCh(); done in Recording constructor
    // SetSecCh( 1 );
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
    std::ostringstream oss1, oss2;
    oss1 << "Number of Channels: " << static_cast<unsigned int>(get().size());
    oss2 << "Number of Sweeps: " << static_cast<unsigned int>(get()[GetCurCh()].size());
    std::string general =
        "Date:\t" + GetDate() + "\n"
        + "Time:\t" + GetTime() + "\n"
        + oss1.str() + "\n" + oss2.str() + "\n"
        + "Comment:\n" + GetComment();
    wxStfFileInfoDlg dlg( GetDocumentWindow(), general, GetFileDescription(),
            GetGlobalSectionDescription() );
    dlg.ShowModal();
}

bool wxStfDoc::OnCloseDocument() {
    if (!get().empty()) {
        WriteToReg();
    }
    // Remove file menu from file menu list:
#ifndef __WXGTK__
    wxGetApp().GetDocManager()->GetFileHistory()->RemoveMenu( doc_file_menu );
#endif
    
    // Tell the App:
    wxGetApp().CleanupDocument(this);
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
    filters += wxT("Igor binary wave (*.ibw)|*.ibw|");
    filters += wxT("Text file series (*.txt)|*.txt|");
#if (defined(WITH_BIOSIG) || defined(WITH_BIOSIG2))
    filters += wxT("GDF file (*.gdf)|*.gdf");
#endif

    wxFileDialog SelectFileDialog( GetDocumentWindow(), wxT("Save file"), wxT(""), wxT(""), filters,
            wxFD_SAVE | wxFD_OVERWRITE_PROMPT | wxFD_PREVIEW );
    if(SelectFileDialog.ShowModal()==wxID_OK) {
        wxString filename = SelectFileDialog.GetPath();
        Recording writeRec(ReorderChannels());
        if (writeRec.size() == 0) return false;
        try {
            stf::wxProgressInfo progDlg("Reading file", "Opening file", 100);
			stfio::filetype type;
            switch (SelectFileDialog.GetFilterIndex()) {
            case 0: type=stfio::hdf5; break;
            case 1: type=stfio::cfs; break;
            case 2: type=stfio::atf; break;
            case 3: type=stfio::igor; break;
            case 4: type=stfio::ascii; break;
#if (defined(WITH_BIOSIG) || defined(WITH_BIOSIG2))
            default: type=stfio::biosig;
#else
            default: type=stfio::hdf5;
#endif
            }
            return stfio::exportFile(stf::wx2std(filename), type, writeRec, progDlg);
        }
        catch (const std::runtime_error& e) {
            wxGetApp().ExceptMsg(stf::std2wx(e.what()));
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
        *it = stf::std2wx( cit->GetChannelName() );
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

#ifndef TEST_MINIMAL
bool wxStfDoc::DoSaveDocument(const wxString& filename) {
    Recording writeRec(ReorderChannels());
    if (writeRec.size() == 0) return false;
    try {
        stf::wxProgressInfo progDlg("Reading file", "Opening file", 100);
        if (stfio::exportFile(stf::wx2std(filename), stfio::hdf5, writeRec, progDlg))
            return true;
        else
            return false;
    }
    catch (const std::runtime_error& e) {
        wxGetApp().ExceptMsg(stf::std2wx(e.what()));
        return false;
    }
}
#endif

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
    wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("RTFactor"),(int)GetRTFactor());
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
        wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("LatencyStartCursor"), (int)GetLatencyBeg());
    if (!outOfRange((size_t)GetLatencyEnd()))
        wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("LatencyEndCursor"), (int)GetLatencyEnd());

#ifdef WITH_PSLOPE
    if (!outOfRange((size_t)GetPSlopeBeg()))
        wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("PSlopeStartCursor"), GetPSlopeBeg() );
    if (!outOfRange((size_t)GetPSlopeEnd()))
        wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("PSlopeEndCursor"), GetPSlopeEnd() );

    wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("PSlopeStartMode"), (int)GetPSlopeBegMode());
    wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("PSlopeEndMode"), (int)GetPSlopeEndMode());
    wxGetApp().wxWriteProfileInt(wxT("Settings"), wxT("DeltaT"), GetDeltaT() );
    
#endif

    // Write Zoom
    wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("Zoom.xZoom"), (int)GetXZoom().xZoom*100000);
    wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("Zoom.yZoom"), GetYZoom(GetCurCh()).yZoom*100000);
    wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("Zoom.startPosX"), (int)GetXZoom().startPosX);
    wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("Zoom.startPosY"), GetYZoom(GetCurCh()).startPosY);
    if ((get().size()>1))
    {
        wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("Zoom.yZoom2"), (int)GetYZoom(GetSecCh()).yZoom*100000);
        wxGetApp().wxWriteProfileInt(wxT("Settings"),wxT("Zoom.startPosY2"), GetYZoom(GetSecCh()).startPosY);
    }
}

bool wxStfDoc::SetSection(std::size_t section){
    // Check range:
    if (!(get().size()>1)) {
        if (section>=get()[GetCurCh()].size())
        {
            wxGetApp().ErrorMsg(wxT("subscript out of range\nwhile calling CStimfitDoc::SetSection()"));
            return false;
        }
        if (get()[GetCurCh()][section].size()==0) {
            wxGetApp().ErrorMsg(wxT("Section is empty"));
            return false;
        }
    } else {
        if (section>=get()[GetCurCh()].size() ||
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
        std::size_t secSize=get()[GetCurCh()][*cit].size();
        if (n_new+secSize>TempSection.size()) {
            wxGetApp().ErrorMsg(wxT("Memory allocation error"));
            return;
        }
        std::copy(get()[GetCurCh()][*cit].get().begin(),
                  get()[GetCurCh()][*cit].get().end(),
                  &TempSection[n_new]);
        n_new+=secSize;
        n_s++;
    }
    ///TempSection.SetXScale(###FIXME###);
    TempSection.SetSectionDescription(
                                      stf::wx2std(GetTitle())+
                                      ", concatenated"
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
    int shift_size = 0;

    /* Aligned average */
    //find alignment points in the reference (==second) channel:

    if (align) {
        // check that we have more than one channel
        if (size()==1){
            wxGetApp().ErrorMsg(wxT("Aligned average requires more than one channel"));
            return;
        }
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
            // APMaxSlopeT will be calculated for the second (==reference)
            // channel, so channels may not be changed!
            try {
                Measure();
            }
            catch (const std::out_of_range& e) {
                Average.resize(0);
                wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
                return;
            }

            std::size_t alignIndex;
            //check whether the current index is a max or a min,
            //and if so, store it:
            switch (AlignDlg.AlignRise()) {
            case 0 :	// align to peak time
                alignIndex = lround(GetMaxT());
                break;
            case 1 :	// align to steepest slope time
                alignIndex = lround(GetAPMaxRiseT());
                break;
            case 2 :	// align to half amplitude time 
                alignIndex = lround(GetAPT50LeftReal());
                break;
            default:
                wxGetApp().ExceptMsg(wxT("Invalid alignment method"));
                return;
            }

            *it = alignIndex;
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
        shift_size = (max_index-min_index);
    }

    //number of points in average:
    size_t average_size = get()[cc].get()[cs].size();
    for (c_st_it sit = GetSelectedSections().begin(); sit != GetSelectedSections().end(); sit++) {
        if (get()[cc].get()[*sit].size() < average_size) {
            average_size = get()[cc].get()[*sit].size();
        }
    }
    average_size -= shift_size;

    //initialize temporary sections and channels:
    Average.resize(size());
    std::size_t n_c = 0;
    for (c_ch_it cit = get().begin(); cit != get().end(); cit++) {
        Section TempSection(average_size), TempSig(average_size);
        try {
            MakeAverage(TempSection, TempSig, n_c, GetSelectedSections(), calcSD, shift);
        }
        catch (const std::out_of_range& e) {
            Average.resize(0);
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
            return;
        }
        ///TempSection.SetXScale(###FIXME###);
        TempSection.SetSectionDescription(stf::wx2std(GetTitle())
                                          +std::string(", average"));
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
    std::string fitInfo;

    try {
        n_params=(int)wxGetApp().GetFuncLib().at(fselect).pInfo.size();
    }
    catch (const std::out_of_range& e) {
        wxString msg(wxT("Could not retrieve function from library:\n"));
        msg+=wxString( e.what(), wxConvLocal );
        wxGetApp().ExceptMsg(msg);
        return;
    }
    Vector_double params ( FitSelDialog.GetInitP() );
    int warning = 0;
    try {
        std::size_t fitSize = GetFitEnd() - GetFitBeg();
        Vector_double x( fitSize );
        //fill array:
        std::copy(&cur()[GetFitBeg()], &cur()[GetFitBeg()+fitSize], &x[0]);
        if (params.size() != n_params) {
            throw std::runtime_error("Wrong size of params in wxStfDoc::lmFit()");
        }
        double chisqr = stf::lmFit( x, GetXScale(), wxGetApp().GetFuncLib()[fselect],
                                    FitSelDialog.GetOpts(), FitSelDialog.UseScaling(),
                                    params, fitInfo, warning );
        SetIsFitted( GetCurCh(), GetCurSec(), params, wxGetApp().GetFuncLibPtr(fselect),
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
    if (pView!=NULL && pView->GetGraph()!=NULL)
        pView->GetGraph()->Refresh();
    wxStfFitInfoDlg InfoDialog(GetDocumentWindow(), stf::std2wx(fitInfo));
    wxEndBusyCursor();
    InfoDialog.ShowModal();
    wxStfChildFrame* pFrame=(wxStfChildFrame*)GetDocumentWindow();
    wxString label; label << wxT("Fit, Section #") << (int)GetCurSec()+1;
    try {
        pFrame->ShowTable(sec_attr.at(GetCurCh()).at(GetCurSec()).bestFit, label);
    }
    catch (const std::out_of_range e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
}

void wxStfDoc::LFit(wxCommandEvent& WXUNUSED(event)) {
    wxBusyCursor wc;
    if (outOfRange(GetFitBeg()) || outOfRange(GetFitEnd())) {
        wxGetApp().ErrorMsg(wxT("Subscript out of range in wxStfDoc::FitDecay()"));
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
    std::string fitInfo;
    n_params=2;
    Vector_double params( n_params );

    //fill array:
    Vector_double x(n_points);
    std::copy(&cur()[GetFitBeg()], &cur()[GetFitBeg()+n_points], &x[0]);
    Vector_double t(x.size());
    for (std::size_t n_t=0;n_t<x.size();++n_t) t[n_t]=n_t*GetXScale();

    // Perform the fit:
    double chisqr = stf::linFit(t,x,params[0],params[1]);
    try {
        SetIsFitted( GetCurCh(), GetCurSec(), params, wxGetApp().GetLinFuncPtr(), chisqr, GetFitBeg(), GetFitEnd() );
    }
    catch (const std::out_of_range e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
        return;
    }

    // Refresh the graph to show the fit before
    // the dialog pops up:
    wxStfView* pView=(wxStfView*)GetFirstView();
    if (pView!=NULL && pView->GetGraph()!=NULL)
        pView->GetGraph()->Refresh();
    std::ostringstream fitInfoStr;
    fitInfoStr << wxT("slope = ") << params[0] << wxT("\n1/slope = ") << 1.0/params[0]
            << wxT("\ny-intercept = ") << params[1];
    fitInfo += fitInfoStr.str();
    wxStfFitInfoDlg InfoDialog(GetDocumentWindow(), stf::std2wx(fitInfo));
    InfoDialog.ShowModal();
    wxStfChildFrame* pFrame=(wxStfChildFrame*)GetDocumentWindow();
    wxString label; label << wxT("Fit, Section #") << (int)GetCurSec();
    try {
        pFrame->ShowTable(sec_attr.at(GetCurCh()).at(GetCurSec()).bestFit, label);
    }
    catch (const std::out_of_range e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
}

void wxStfDoc::LnTransform(wxCommandEvent& WXUNUSED(event)) {
    Channel TempChannel(GetSelectedSections().size(), get()[GetCurCh()][GetSelectedSections()[0]].size());
    std::size_t n = 0;
    for (c_st_it cit = GetSelectedSections().begin(); cit != GetSelectedSections().end(); cit++) {
        Section TempSection(size());
        std::transform(get()[GetCurCh()][*cit].get().begin(), 
                       get()[GetCurCh()][*cit].get().end(), 
                       TempSection.get_w().begin(),
#if defined(_WINDOWS) && !defined(__MINGW32__)
                       std::logl);
#else
                       log);
#endif
        TempSection.SetXScale(get()[GetCurCh()][*cit].GetXScale());
        TempSection.SetSectionDescription( get()[GetCurCh()][*cit].GetSectionDescription()+
                                           ", transformed (ln)");
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
        pFrame->ShowTable( CurAsTable(), stf::std2wx( cur().GetSectionDescription() ) );
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
    std::vector<std::string> labels(1);
    Vector_double defaults(labels.size());
    labels[0]="Multiply with:";defaults[0]=1;
    stf::UserInput init(labels,defaults,"Set factor");

    wxStfUsrDlg MultDialog(GetDocumentWindow(),init);
    if (MultDialog.ShowModal()!=wxID_OK) return;
    Vector_double input(MultDialog.readInput());
    if (input.size()!=1) return;

    double factor=input[0];
    Channel TempChannel(GetSelectedSections().size(), get()[GetCurCh()][GetSelectedSections()[0]].size());
    std::size_t n = 0;
    for (c_st_it cit = GetSelectedSections().begin(); cit != GetSelectedSections().end(); cit++) {
        // Multiply the valarray in Data:
        Section TempSection(stfio::vec_scal_mul(get()[GetCurCh()][*cit].get(),factor));
        TempSection.SetXScale(get()[GetCurCh()][*cit].GetXScale());
        TempSection.SetSectionDescription(
                get()[GetCurCh()][*cit].GetSectionDescription()+
                ", multiplied"
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
        Section TempSection(stfio::vec_scal_minus(get()[GetCurCh()][*cit].get(), GetSelectBase()[n]));
        TempSection.SetXScale(get()[GetCurCh()][*cit].GetXScale());
        TempSection.SetSectionDescription( get()[GetCurCh()][*cit].GetSectionDescription()+
                                           ", baseline subtracted");
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
    std::vector<std::string> colTitles;
    //Write the header of the SaveYt file in a string
    if (SaveYtDialog.PrintBase()) {
        colTitles.push_back("Base");
    }
    if (SaveYtDialog.PrintBaseSD()) {
        colTitles.push_back("Base SD");
    }
    if (SaveYtDialog.PrintThreshold()) {
        colTitles.push_back("Slope threshold");
    }
    if (SaveYtDialog.PrintPeakZero()) {
        colTitles.push_back("Peak (from 0)");
    }
    if (SaveYtDialog.PrintPeakBase()) {
        colTitles.push_back("Peak (from baseline)");
    }
    if (SaveYtDialog.PrintPeakThreshold()) {
        colTitles.push_back("Peak (from threshold)");
    }
    if (SaveYtDialog.PrintRTLoHi()) {
        colTitles.push_back("RT Lo-Hi%");
    }
    if (SaveYtDialog.PrintInnerRTLoHi()) {
        colTitles.push_back("inner Rise Time Lo-Hi%");
    }
    if (SaveYtDialog.PrintOuterRTLoHi()) {
        colTitles.push_back("Outer Rise Time Lo-Hi%");
    }
    if (SaveYtDialog.PrintT50()) {
        colTitles.push_back("t 1/2");
    }
    if (SaveYtDialog.PrintSlopes()) {
        colTitles.push_back("Max. slope rise");
        colTitles.push_back("Max. slope decay");
    }
    if (SaveYtDialog.PrintLatencies()) {
        colTitles.push_back("Latency");
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
            msg += stf::std2wx(e.what());
            wxGetApp().ExceptMsg(msg);
            SetSection(section_old);
            return;
        }
        for (std::size_t n_pf=0;n_pf<n_params;++n_pf) {
            colTitles.push_back( wxGetApp().GetFuncLib()[fselect].pInfo[n_pf].desc);
        }
        colTitles.push_back("Fit warning code");
    }
    if (SaveYtDialog.PrintThr()) {
        colTitles.push_back("# of thr. crossings");
    }
    double threshold=0.0;
    if (SaveYtDialog.PrintThr()) {
        // Get threshold from user:
        std::ostringstream thrS;
        thrS << "Threshold (" << at(GetCurCh()).GetYUnits() << ")";
        stf::UserInput Input( std::vector<std::string>(1, thrS.str()),
                Vector_double (1,0.0), "Set threshold");
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

        Vector_double params;
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
            Vector_double x(GetFitEnd()-GetFitBeg());
            //fill array:
            std::copy(&cur()[GetFitBeg()], &cur()[GetFitEnd()], &x[0]);
            params.resize(n_params);
            wxGetApp().GetFuncLib().at(fselect).init( x, GetBase(), GetPeak(), GetRTLoHi(),
                    GetHalfDuration(), GetXScale(), params );

            std::string fitInfo;
            try {
                double chisqr = stf::lmFit( x, GetXScale(), wxGetApp().GetFuncLib()[fselect],
                                            FitSelDialog.GetOpts(), FitSelDialog.UseScaling(),
                                            params, fitInfo, fitWarning );
                SetIsFitted( GetCurCh(), GetCurSec(), params, wxGetApp().GetFuncLibPtr(fselect),
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
            table.SetRowLabel(n_s, cur().GetSectionDescription());

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
            if (SaveYtDialog.PrintRTLoHi())
                table.at(n_s,nCol++)=GetRTLoHi();
            if (SaveYtDialog.PrintInnerRTLoHi())
                table.at(n_s,nCol++)=GetInnerRiseTime();
            if (SaveYtDialog.PrintOuterRTLoHi())
                table.at(n_s,nCol++)=GetOuterRiseTime();
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
    double integral_s = 0.0, integral_t = 0.0;
    try {
        integral_s = stf::integrate_simpson(cur().get(),GetFitBeg(),GetFitEnd(),GetXScale());
        integral_t = stf::integrate_trapezium(cur().get(),GetFitBeg(),GetFitEnd(),GetXScale());
    }
    catch (const std::exception& e) {
        wxGetApp().ErrorMsg(wxString( e.what(), wxConvLocal ));
        return;
    }
    stf::Table integralTable(6,1);
    try {
        integralTable.SetRowLabel(0, "Trapezium (linear)");
        integralTable.SetRowLabel(1, "Integral (from 0)");
        integralTable.SetRowLabel(2, "Integral (from base)");
        integralTable.SetRowLabel(3, "Simpson (quadratic)");
        integralTable.SetRowLabel(4, "Integral (from 0)");
        integralTable.SetRowLabel(5, "Integral (from base)");
        integralTable.SetColLabel(0, "Result");
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
        Vector_double quad_p = stf::quad(cur().get(), GetFitBeg(), GetFitEnd());
        SetIsIntegrated(GetCurCh(), GetCurSec(), true,GetFitBeg(),GetFitEnd(), quad_p);
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
        TempSection.SetXScale(get()[GetCurCh()][*cit].GetXScale());
        TempSection.SetSectionDescription( get()[GetCurCh()][*cit].GetSectionDescription()+
                ", differentiated");
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
        TempSection.SetXScale(get()[GetCurCh()][*cit].GetXScale());
        TempSection.SetSectionDescription( get()[GetCurCh()][*cit].GetSectionDescription()+
                ", new from selected");
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
        Selected[0].SetChannelName( at(GetCurCh()).GetChannelName() );
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
    std::vector<std::string> labels(2);
    Vector_double defaults(labels.size());
    labels[0]="Select every x-th trace:";defaults[0]=1;
    labels[1]="Starting with the y-th:";defaults[1]=1;
    stf::UserInput init(labels,defaults,"Select every n-th (1-based)");

    wxStfUsrDlg EveryDialog(GetDocumentWindow(),init);
    if (EveryDialog.ShowModal()!=wxID_OK) return;
    Vector_double input(EveryDialog.readInput());
    if (input.size()!=2) return;
    int everynth=(int)input[0];
    int everystart=(int)input[1];
    //div_t n_selected=div((int)get()[GetCurCh()].size(),everynth);
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
    std::vector<std::string> labels(2);
    Vector_double defaults(labels.size());
    labels[0]="Unselect every x-th trace:";defaults[0]=1;
    labels[1]="Starting with the y-th:";defaults[1]=1;
    stf::UserInput init(labels,defaults,"Unselect every n-th (1-based)");

    wxStfUsrDlg EveryDialog(GetDocumentWindow(),init);
    if (EveryDialog.ShowModal()!=wxID_OK) return;
    Vector_double input(EveryDialog.readInput());
    if (input.size()!=2) return;
    int everynth=(int)input[0];
    int everystart=(int)input[1];
    //div_t n_unselected=div((int)get()[GetCurCh()].size(),everynth);
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
    if (pFrame->ShowSelected()) {
        wxStfView* pView=(wxStfView*)GetFirstView();
        if (pView!=NULL && pView->GetGraph()!=NULL)
            pView->GetGraph()->Refresh();
    }
    Focus();
}

void wxStfDoc::Focus() {

    UpdateSelectedButton();

    // refresh the view once we are through:
    wxStfView* pView=(wxStfView*)GetFirstView();
    if (pView != NULL && pView->GetGraph() != NULL) {
        pView->GetGraph()->Enable();
        pView->GetGraph()->SetFocus();
    }
    
}

void wxStfDoc::UpdateSelectedButton() {
    // control whether trace has been selected:
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
    std::vector<std::string> windowLabels(2);
    Vector_double windowDefaults(windowLabels.size());
    windowLabels[0]="From point #:";windowDefaults[0]=0;
    windowLabels[1]="To point #:";windowDefaults[1]=(int)cur().size()-1;
    stf::UserInput initWindow(windowLabels,windowDefaults,"Filter window");

    wxStfUsrDlg FilterWindowDialog(GetDocumentWindow(),initWindow);
    if (FilterWindowDialog.ShowModal()!=wxID_OK) return;
    Vector_double windowInput(FilterWindowDialog.readInput());
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

    Vector_double a(size);
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
        std::vector<std::string> labels(1);
        Vector_double defaults(labels.size());
        labels[0]="Cutoff frequency (kHz):";
        defaults[0]=10;
        stf::UserInput init(labels,defaults,"Set frequency");

        wxStfUsrDlg FilterHighLowDialog(GetDocumentWindow(),init);
        if (FilterHighLowDialog.ShowModal()!=wxID_OK) return;
        Vector_double input(FilterHighLowDialog.readInput());
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
		    FftTemp.SetXScale(get()[GetCurCh()][*cit].GetXScale());
                    FftTemp.SetSectionDescription( get()[GetCurCh()][*cit].GetSectionDescription()+
                                                   ", filtered");
                    TempChannel.InsertSection(FftTemp, n);
                    break;
                }
                case 2: {
                    Section FftTemp(stf::filter(get()[GetCurCh()][*cit].get(),
                            llf,ulf,a,(int)GetSR(),stf::fbessel4,false));
		    FftTemp.SetXScale(get()[GetCurCh()][*cit].GetXScale());
                    FftTemp.SetSectionDescription( get()[GetCurCh()][*cit].GetSectionDescription()+
                                                   ", filtered" );
                    TempChannel.InsertSection(FftTemp, n);
                    break;
                }
                case 1: {
                    Section FftTemp(stf::filter(get()[GetCurCh()][*cit].get(),
                            llf,ulf,a,(int)GetSR(),stf::fgauss,inverse));
		    FftTemp.SetXScale(get()[GetCurCh()][*cit].GetXScale());
                    FftTemp.SetSectionDescription( get()[GetCurCh()][*cit].GetSectionDescription()+
                                                   std::string(", filtered") );
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

void wxStfDoc::P_over_N(wxCommandEvent& WXUNUSED(event)){
    //insert standard values:
    std::vector<std::string> labels(1);
    Vector_double defaults(labels.size());
    labels[0]="N = (mind polarity!)";defaults[0]=-4;
    stf::UserInput init(labels,defaults,"P over N");

    wxStfUsrDlg PonDialog(GetDocumentWindow(),init);
    if (PonDialog.ShowModal()!=wxID_OK) return;
    Vector_double input(PonDialog.readInput());
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
        TempSection.SetXScale(get()[GetCurCh()][n_section].GetXScale());
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
        std::ostringstream povernLabel;
        povernLabel << GetTitle() << ", #" << n_section << ", P over N";
        TempSection.SetSectionDescription(povernLabel.str());
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

void wxStfDoc::Plotextraction(stf::extraction_mode mode) {
    std::vector<stf::SectionPointer> sectionList(wxGetApp().GetSectionsWithFits());
    if (sectionList.empty()) {
        wxGetApp().ErrorMsg(
                wxT("You have to create a template first\nby fitting a function to an event") );
        return;
    }
    wxStfEventDlg MiniDialog(GetDocumentWindow(), sectionList, false);
    if (MiniDialog.ShowModal()!=wxID_OK)  {
        return;
    }
    int nTemplate=MiniDialog.GetTemplate();
    try {
        Vector_double templateWave(
                sectionList.at(nTemplate).sec_attr.storeFitEnd -
                sectionList.at(nTemplate).sec_attr.storeFitBeg);
        for ( std::size_t n_p=0; n_p < templateWave.size(); n_p++ ) {
            templateWave[n_p] = sectionList.at(nTemplate).sec_attr.fitFunc->func(
                n_p*GetXScale(), sectionList.at(nTemplate).sec_attr.bestFitP);
        }
        wxBusyCursor wc;
#undef min
#undef max
        // subtract offset and normalize:

        double fmax = *std::max_element(templateWave.begin(), templateWave.end());
        double fmin = *std::min_element(templateWave.begin(), templateWave.end());
        templateWave = stfio::vec_scal_minus(templateWave, fmax);
        double minim=fabs(fmin);
        templateWave = stfio::vec_scal_div(templateWave, minim);
        std::string section_description, window_title;
        Section TempSection(cur().get().size());
        switch (mode) {
         case stf::criterion: {
             stf::wxProgressInfo progDlg("Computing detection criterion...", "Computing detection criterion...", 100);
             TempSection = Section(stf::detectionCriterion( cur().get(), templateWave, progDlg));
             section_description = "Detection criterion of ";
             window_title = ", detection criterion";
             break;
         }
         case stf::correlation: {
             stf::wxProgressInfo progDlg("Computing linear correlation...", "Computing linear correlation...", 100);
             TempSection = Section(stf::linCorr(cur().get(), templateWave, progDlg));
             section_description = "Template correlation of ";
             window_title = ", linear correlation";
             break;
         }
         case stf::deconvolution:
             std::string usrInStr[2] = {"Lowpass (kHz)", "Highpass (kHz)"};
             double usrInDbl[2] = {0.5, 0.0001};
             stf::UserInput Input( std::vector<std::string>(usrInStr, usrInStr+2),
                                   Vector_double (usrInDbl, usrInDbl+2), "Filter settings" );
             wxStfUsrDlg myDlg( GetDocumentWindow(), Input );
             if (myDlg.ShowModal()!=wxID_OK) return;
             Vector_double filter = myDlg.readInput();
             stf::wxProgressInfo progDlg("Computing deconvolution...", "Starting deconvolution...", 100);
             TempSection = Section(stf::deconvolve(cur().get(), templateWave,
                                                   (int)GetSR(), filter[1], filter[0], progDlg));
             section_description = "Template deconvolution from ";
             window_title = ", deconvolution";
             break;
        }
        if (TempSection.size()==0) return;
        TempSection.SetXScale(cur().GetXScale());
        TempSection.SetSectionDescription(section_description +
                                          cur().GetSectionDescription());
        Channel TempChannel(TempSection);
        Recording detCrit(TempChannel);
        detCrit.CopyAttributes(*this);

        wxGetApp().NewChild(detCrit, this, GetTitle() + stf::std2wx(window_title));
    }
    catch (const std::runtime_error& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
    catch (const std::exception& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
}

void wxStfDoc::Plotcriterion(wxCommandEvent& WXUNUSED(event)) {
    Plotextraction(stf::criterion);
}

void wxStfDoc::Plotcorrelation(wxCommandEvent& WXUNUSED(event)) {
    Plotextraction(stf::correlation);
}

void wxStfDoc::Plotdeconvolution(wxCommandEvent& WXUNUSED(event)) {
    Plotextraction(stf::deconvolution);
}

void wxStfDoc::MarkEvents(wxCommandEvent& WXUNUSED(event)) {
    std::vector<stf::SectionPointer> sectionList(wxGetApp().GetSectionsWithFits());
    if (sectionList.empty()) {
        wxGetApp().ErrorMsg(
                wxT( "You have to create a template first\nby fitting a function to an event" ) );
        return;
    }
    wxStfEventDlg MiniDialog( GetDocumentWindow(), sectionList, true );
    if ( MiniDialog.ShowModal()!=wxID_OK ) {
        return;
    }
    int nTemplate=MiniDialog.GetTemplate();
    try {
        Vector_double templateWave(
                sectionList.at(nTemplate).sec_attr.storeFitEnd -
                sectionList.at(nTemplate).sec_attr.storeFitBeg);
        for ( std::size_t n_p=0; n_p < templateWave.size(); n_p++ ) {
            templateWave[n_p] = sectionList.at(nTemplate).sec_attr.fitFunc->func(
                    n_p*GetXScale(), sectionList.at(nTemplate).sec_attr.bestFitP);
        }
        wxBusyCursor wc;
#undef min
#undef max
        // subtract offset and normalize:
        double fmax = *std::max_element(templateWave.begin(), templateWave.end());
        double fmin = *std::min_element(templateWave.begin(), templateWave.end());
        templateWave = stfio::vec_scal_minus(templateWave, fmax);
        double minim=fabs(fmin);
        templateWave = stfio::vec_scal_div(templateWave, minim);
        Vector_double detect( cur().get().size() - templateWave.size() );
        switch (MiniDialog.GetMode()) {
         case stf::criterion: {
             stf::wxProgressInfo progDlg("Computing detection criterion...", "Computing detection criterion...", 100);
             detect=stf::detectionCriterion(cur().get(), templateWave, progDlg);
             break;
         }
         case stf::correlation: {
             stf::wxProgressInfo progDlg("Computing linear correlation...", "Computing linear correlation...", 100);
             detect=stf::linCorr(cur().get(), templateWave, progDlg);
             break;
         }
         case stf::deconvolution:
             std::string usrInStr[2] = {"Lowpass (kHz)", "Highpass (kHz)"};
             double usrInDbl[2] = {0.5, 0.0001};
             stf::UserInput Input( std::vector<std::string>(usrInStr, usrInStr+2),
                                   Vector_double (usrInDbl, usrInDbl+2), "Filter settings" );
             wxStfUsrDlg myDlg( GetDocumentWindow(), Input );
             if (myDlg.ShowModal()!=wxID_OK) return;
             Vector_double filter = myDlg.readInput();
             stf::wxProgressInfo progDlg("Computing deconvolution...", "Starting deconvolution...", 100);
             detect=stf::deconvolve(cur().get(), templateWave, (int)GetSR(), filter[1], filter[0], progDlg);
             break;
        }
        if (detect.empty()) {
            wxGetApp().ErrorMsg(wxT("Error: Detection criterion is empty."));
            return;
        }
        std::vector<int> startIndices(
                stf::peakIndices( detect, MiniDialog.GetThreshold(),
                        MiniDialog.GetMinDistance() ) );
        if (startIndices.empty()) {
            wxGetApp().ErrorMsg( wxT( "No events were found. Try to lower the threshold." ) );
            return;
        }
        // erase old events:
        ClearEvents(GetCurCh(), GetCurSec());
        for (c_int_it cit = startIndices.begin(); cit != startIndices.end(); ++cit ) {
            sec_attr.at(GetCurCh()).at(GetCurSec()).eventList.push_back( stf::Event( *cit, 0, templateWave.size() ) );
            // Find peak in this event:
            double baselineMean=0;
            for ( int n_mean = *cit-baseline;
                  n_mean < *cit;
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
            std::size_t last_idx = GetCurrentSectionAttributes().eventList.size()-1;
            sec_attr.at(GetCurCh()).at(GetCurSec()).eventList.at(last_idx).SetEventPeakIndex((int)peakIndex);
        }
    }
    catch (const std::out_of_range& e) {
        wxGetApp().ExceptMsg( wxString( e.what(), wxConvLocal ));
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
        stf::Table events(GetCurrentSectionAttributes().eventList.size(), 2);
        events.SetColLabel(0, "Time of event onset");
        events.SetColLabel(1, "Inter-event interval");
        // using the peak indices (these are the locations of the beginning of an optimal
        // template matching), new sections are created:

        // count non-discarded events:
        std::size_t n_real = 0;
        for (c_event_it cit = GetCurrentSectionAttributes().eventList.begin();
             cit != GetCurrentSectionAttributes().eventList.end(); ++cit) {
            n_real += (int)(!cit->GetDiscard());
        }
        Channel TempChannel2(n_real);
        std::vector<int> peakIndices(n_real);
        n_real = 0;
        c_event_it lastEventIt = GetCurrentSectionAttributes().eventList.begin();
        for (c_event_it it = GetCurrentSectionAttributes().eventList.begin();
             it != GetCurrentSectionAttributes().eventList.end(); ++it) {
            if (!it->GetDiscard()) {
                wxString miniName; miniName << wxT( "Event #" ) << (int)n_real+1;
                events.SetRowLabel(n_real, stf::wx2std(miniName));
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
                std::ostringstream eventDesc;
                eventDesc << "Extracted event #" << (int)n_real;
                //?TempSection2.SetXScale(###FIXME###);
                TempSection2.SetSectionDescription(eventDesc.str());
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
    catch (const std::out_of_range& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
    catch (const std::runtime_error& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
    catch (const std::exception& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
    }
}

void wxStfDoc::InteractiveEraseEvents( wxCommandEvent& WXUNUSED(event) ) {
    if (wxMessageDialog( GetDocumentWindow(), wxT("Do you really want to erase all events?"),
                         wxT("Erase all events"), wxYES_NO ).ShowModal()==wxID_YES)
    {
        try {
            ClearEvents(GetCurCh(), GetCurSec());
        }
        catch (const std::out_of_range& e) {
            wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
        }
    }
}

void wxStfDoc::AddEvent( wxCommandEvent& WXUNUSED(event) ) {
    try {
        // retrieve the position where to add the event:
        wxStfView* pView = (wxStfView*)GetFirstView();
        wxStfGraph* pGraph = pView->GetGraph();
        int newStartPos = pGraph->get_eventPos();
        stf::Event newEvent(newStartPos, 0, GetCurrentSectionAttributes().eventList.at(0).GetEventSize());
        // Find peak in this event:
        double baselineMean=0;
        for ( int n_mean = newStartPos - baseline;
              n_mean < newStartPos;
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
                newStartPos + GetCurrentSectionAttributes().eventList.at(0).GetEventSize(), 1,
                stf::both, peakIndex );
        // set peak index of last event:
        newEvent.SetEventPeakIndex( (int)peakIndex );
        // find the position in the current event list where the new
        // event should be inserted:
        bool found = false;
        for (event_it it = sec_attr.at(GetCurCh()).at(GetCurSec()).eventList.begin();
             it != sec_attr.at(GetCurCh()).at(GetCurSec()).eventList.end(); ++it) {
            if ( (int)(it->GetEventStartIndex()) > newStartPos ) {
                // insert new event before this event, then break:
                sec_attr.at(GetCurCh()).at(GetCurSec()).eventList.insert( it, newEvent );
                found = true;
                break;
            }
        }
        // if we are at the end of the list, append the event:
        if (!found)
            sec_attr.at(GetCurCh()).at(GetCurSec()).eventList.push_back( newEvent );
    }
    catch (const std::out_of_range& e) {
        wxGetApp().ExceptMsg(wxString( e.what(), wxConvLocal ));
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
    Vector_double threshold(0);
    std::ostringstream thrS;
    thrS << "Threshold (" << at(GetCurCh()).GetYUnits() << wxT(")");
    stf::UserInput Input( std::vector<std::string>(1, thrS.str()),
                          Vector_double (1,0.0), "Set threshold" );
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
        sec_attr.at(GetCurCh()).at(GetCurSec()).eventList.push_back( stf::Event( *cit, 0, baseline ) );
    }
    // show results in a table:
    stf::Table events(GetCurrentSectionAttributes().eventList.size(),2);
    events.SetColLabel( 0, "Time of event peak");
    events.SetColLabel( 1, "Inter-event interval");
    std::size_t n_event = 0;
    c_event_it lastEventCit = GetCurrentSectionAttributes().eventList.begin();
    for (c_event_it cit2 = GetCurrentSectionAttributes().eventList.begin();
         cit2 != GetCurrentSectionAttributes().eventList.end(); ++cit2) {
        wxString eventName; eventName << wxT("Event #") << (int)n_event+1;
        events.SetRowLabel(n_event, stf::wx2std(eventName));
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

//Function calculates the peak and respective measures: base, Lo/Hi rise time
//half duration, ratio of rise/slope and maximum slope
void wxStfDoc::Measure( )
{
    double var=0.0;
    if (cur().get().size() == 0) return;
    try {
        cur().at(0);
    }
    catch (const std::out_of_range&) {
        return;
    }

    long windowLength = 1;
    /*
       windowLength (defined in samples) determines the size of the window for computing slopes.
       if the window length larger than 1 is used, a kind of smoothing and low pass filtering is applied.
       If slope estimates from data with different sampling rates should be compared, the
       window should be choosen in such a way that the length in milliseconds is approximately the same.
       This reduces some variability, the slope estimates are more robust and comparible.

       Set window length to 0.05 ms, with a minimum of 1 sample. In this way, all data
       sampled with 20 kHz or lower, will use a 1 sample window, data with a larger sampling rate
       use a window of 0.05 ms for computing the slope.
    */
    windowLength = lround(0.05 * GetSR());    // use window length of about 0.05 ms.
    if (windowLength < 1) windowLength = 1;   // use a minimum window length of 1 sample


    //Begin peak and base calculation
    //-------------------------------
    try {
        base=stf::base(var,cur().get(),baseBeg,baseEnd);
        baseSD=sqrt(var);
        peak=stf::peak(cur().get(),base,
                       peakBeg,peakEnd,pM,direction,maxT);
    }
    catch (const std::out_of_range& e) {
        base=0.0;
        baseSD=0.0;
        peak=0.0;
        throw e;
    }
    try {
        threshold = stf::threshold( cur().get(), peakBeg, peakEnd, slopeForThreshold/GetSR(), thrT, windowLength );
    } catch (const std::out_of_range& e) {
        threshold = 0;
        throw e;
    }
    //Begin Lo to Hi% Rise Time calculation
    //-------------------------------------
    // 2009-06-05: reference is either from baseline or from threshold
    double reference = base;
    if (!fromBase && thrT >= 0) {
        reference = threshold;
    }
    double ampl=peak-reference;
    
    tLoReal=0.0;
    double factor= RTFactor*0.01; /* normalized value */
    InnerLoRT=NAN;
    InnerHiRT=NAN;
    OuterLoRT=NAN;
    OuterHiRT=NAN;

    try {
        // 2008-04-27: changed limits to start from the beginning of the trace
        // 2013-06-16: changed to accept different rise-time proportions
        rtLoHi=stf::risetime2(cur().get(),reference,ampl, (double)0/*(double)baseEnd*/,
                             maxT, factor/*0.2*/, InnerLoRT, InnerHiRT, OuterLoRT, OuterHiRT);
        InnerLoRT/=GetSR();
        InnerHiRT/=GetSR();
        OuterLoRT/=GetSR();
        OuterHiRT/=GetSR();
    }
    catch (const std::out_of_range& e) {
        throw e;
    }


    try {
        // 2008-04-27: changed limits to start from the beginning of the trace
        // 2013-06-16: changed to accept different rise-time proportions 
        rtLoHi=stf::risetime(cur().get(),reference,ampl, (double)0/*(double)baseEnd*/,
                             maxT, factor/*0.2*/, tLoIndex, tHiIndex, tLoReal);
    }
    catch (const std::out_of_range& e) {
        rtLoHi=0.0;
        throw e;
    }

    tHiReal=tLoReal+rtLoHi;
    rtLoHi/=GetSR();

    //Begin Half Duration calculation
    //-------------------------------
    //t50LeftReal=0.0;
    // 2008-04-27: changed limits to start from the beginning of the trace
    //             and to stop at the end of the trace
    halfDuration = stf::t_half(cur().get(), reference, ampl, (double)0 /*(double)baseBeg*/,
            (double)cur().size()-1 /*(double)peakEnd*/,maxT, t50LeftIndex,t50RightIndex,t50LeftReal);

    t50RightReal=t50LeftReal+halfDuration;
    halfDuration/=GetSR();
    t50Y=0.5*ampl + reference;

    //Calculate the beginning of the event by linear extrapolation:
    if (latencyEndMode==stf::footMode) {
        t0Real=tLoReal-(tHiReal-tLoReal)/3.0;
    } else {
        t0Real=t50LeftReal;
    }

    //Begin Ratio of slopes rise/decay calculation
    //--------------------------------------------
    double left_rise = peakBeg;
    maxRise=stf::maxRise(cur().get(),left_rise,maxT,maxRiseT,maxRiseY,windowLength);
    double t_half_3=t50RightIndex+2.0*(t50RightIndex-t50LeftIndex);
    double right_decay=peakEnd<=t_half_3 ? peakEnd : t_half_3+1;
    maxDecay=stf::maxDecay(cur().get(),maxT,right_decay,maxDecayT,maxDecayY,windowLength);

    //Slope ratio
    if (maxDecay !=0) slopeRatio=maxRise/maxDecay;
    else slopeRatio=0.0;
    maxRise *= GetSR();
    maxDecay *= GetSR();

    if (size()>1) {
        //Calculate the absolute peak of the (AP) Ch2 inbetween the peak boundaries
        //A direction dependent evaluation of the peak as in Ch1 does NOT exist!!

        // endResting is set to 100 points arbitrarily in the pascal version
        // (see measlib.pas) assuming that the resting potential is stable
        // during the first 100 sampling points.
        // const int endResting=100;
        const int searchRange=100;
        double APBase=0.0, APPeak=0.0, APVar=0.0;
        try {
            // in 2012-11-02: use baseline cursors and not arbitrarily 100 points
            //APBase=stf::base(APVar,sec().get(),0,endResting);
            APBase=stf::base( APVar,sec().get(), baseBeg, baseEnd ); // use baseline cursors 
            //APPeak=stf::peak(sec().get(),APBase,peakBeg,peakEnd,pM,stf::up,APMaxT);
            APPeak=stf::peak( sec().get(),APBase ,peakBeg ,peakEnd ,pM,direction ,APMaxT );
        }
        catch (const std::out_of_range& e) {
            APBase=0.0;
            APPeak=0.0;
            throw e;
        }
        //-------------------------------
        //Maximal slope in the rise before the peak
        //----------------------------
        APMaxRiseT=0.0;
        double APMaxRiseY=0.0;
        double left_APRise = peakBeg; 
        if (GetLatencyWindowMode() == stf::defaultMode ) {
            left_APRise= APMaxT-searchRange>2.0 ? APMaxT-searchRange : 2.0;
        }
        try {
            stf::maxRise(sec().get(),left_APRise,APMaxT,APMaxRiseT,APMaxRiseY,windowLength);
        }
        catch (const std::out_of_range&) {
            APMaxRiseT=0.0;
            APMaxRiseY=0.0;
            left_APRise = peakBeg; 
        }

        //End determination of the region of maximal slope in the second channel
        //----------------------------

        //-------------------------------
        //Half-maximal amplitude
        //----------------------------
        //APt50LeftReal=0.0;
        //std::size_t APt50LeftIndex,APt50RightIndex;
        stf::t_half(sec().get(), APBase, APPeak-APBase, left_APRise,
                      (double)sec().get().size(), APMaxT, APt50LeftIndex,
                      APt50RightIndex, APt50LeftReal);
        //End determination of the region of maximal slope in the second channel
        //----------------------------
    }

    // get and set start of latency measurement:
    double latStart=0.0;
    switch (latencyStartMode) {
    // Interestingly, latencyCursor is an int in pascal, although
    // the maxTs aren't. That's why there are double type casts
    // here.
    case stf::peakMode: //Latency cursor is set to the peak						
        latStart=APMaxT;
        break;
    case stf::riseMode:
        latStart=APMaxRiseT;
        break;
    case stf::halfMode:
        latStart=APt50LeftReal;
        break;
    case stf::manualMode:
    default:
        latStart=GetLatencyBeg();
        break;
    }
    SetLatencyBeg(latStart);

    // get and set end of latency measurement:
    double latEnd=0.0;
    switch (latencyEndMode) {
    // Interestingly, latencyCursor is an int in pascal, although
    // the maxTs aren't. That's why there are double type casts
    // here.
    case stf::footMode: //Latency cursor is set to the peak						
        latEnd=tLoReal-(tHiReal-tLoReal)/3.0;
        break;
    case stf::riseMode:
        latEnd=maxRiseT;
        break;
    case stf::halfMode:
        latEnd=t50LeftReal;
        break;
    case stf::peakMode:
        latEnd=maxT;
        break;
    case stf::manualMode:
    default:
        latEnd=GetLatencyEnd();
        break;
    }
    SetLatencyEnd(latEnd);

    SetLatency(GetLatencyEnd()-GetLatencyBeg());

#ifdef WITH_PSLOPE
    //-------------------------------------
    // Begin PSlope calculation (PSP Slope)
    //-------------------------------------

    //
    int PSlopeBegVal; 
    switch (pslopeBegMode) {

        case stf::psBeg_footMode:   // Left PSlope to commencement
            PSlopeBegVal = (int)(tLoReal-(tHiReal-tLoReal)/3.0);
            break;

        case stf::psBeg_thrMode:   // Left PSlope to threshold
            PSlopeBegVal = (int)thrT;
            break;

        case stf::psBeg_t50Mode:   // Left PSlope to the t50
            PSlopeBegVal = (int)t50LeftReal;
            break;

        case stf::psBeg_manualMode: // Left PSlope cursor manual
        default:
            PSlopeBegVal = PSlopeBeg;
    }
    SetPSlopeBeg(PSlopeBegVal);
    
    int PSlopeEndVal;
    switch (pslopeEndMode) {

        case stf::psEnd_t50Mode:    // Right PSlope to t50rigth
            PSlopeEndVal = (int)t50LeftReal;
            break;
        case stf::psEnd_peakMode:   // Right PSlope to peak
            PSlopeEndVal = (int)maxT;
            break;
        case stf::psEnd_DeltaTMode: // Right PSlope to DeltaT time from first peak
            PSlopeEndVal = (int)(PSlopeBeg + DeltaT);
            break;
        case stf::psEnd_manualMode:
        default:
            PSlopeEndVal = PSlopeEnd;
    }
    SetPSlopeEnd(PSlopeEndVal);

    try {
        PSlope = (stf::pslope(cur().get(), PSlopeBeg, PSlopeEnd))*GetSR();
    }
    catch (const std::out_of_range& e) {
        PSlope = 0.0;
        throw e;
    }

    //-----------------------------------
    // End PSlope calculation (PSP Slope)
    //-----------------------------------

#endif // WITH_PSLOPE
    //--------------------------

}	//End of Measure(,,,,,)


void wxStfDoc::CopyCursors(const wxStfDoc& c_Recording) {
    measCursor=c_Recording.measCursor;
    correctRangeR(measCursor);
    baseBeg=c_Recording.baseBeg;
    correctRangeR(baseBeg);
    baseEnd=c_Recording.baseEnd;
    correctRangeR(baseEnd);
    peakBeg=c_Recording.peakBeg;
    correctRangeR(peakBeg);
    peakEnd=c_Recording.peakEnd;
    correctRangeR(peakEnd);
    fitBeg=c_Recording.fitBeg;
    correctRangeR(fitBeg);
    fitEnd=c_Recording.fitEnd;
    correctRangeR(fitEnd);

#ifdef WITH_PSLOPE
    PSlopeBeg = c_Recording.PSlopeBeg; // PSlope left cursor
    correctRangeR(PSlopeBeg);
    PSlopeEnd = c_Recording.PSlopeEnd; // PSlope right cursor
    correctRangeR(PSlopeEnd);
    DeltaT=c_Recording.DeltaT;  //distance (number of points) from first cursor 
#endif
 
    pM=c_Recording.pM;  //peakMean, number of points used for averaging

}

void wxStfDoc::SetLatencyStartMode(int value) {
    switch (value) {
    case 1:
        latencyStartMode=stf::peakMode;
        break;
    case 2:
        latencyStartMode=stf::riseMode;
        break;
    case 3:
        latencyStartMode=stf::halfMode;
        break;
    case 4:
        latencyStartMode=stf::footMode;
        break;
    case 0:
    default:
        latencyStartMode=stf::manualMode;
    }
}

void wxStfDoc::SetLatencyEndMode(int value) {
    switch (value) {
    case 1:
        latencyEndMode=stf::peakMode;
        break;
    case 2:
        latencyEndMode=stf::riseMode;
        break;
    case 3:
        latencyEndMode=stf::halfMode;
        break;
    case 4:
        latencyEndMode=stf::footMode;
        break;
    case 0:
    default:
        latencyEndMode=stf::manualMode;
    }
}

void wxStfDoc::SetLatencyWindowMode(int value) {
    if ( value == 1 ) {
        latencyWindowMode = stf::windowMode;
    } else {
        latencyWindowMode = stf::defaultMode;
    }
}

void wxStfDoc::correctRangeR(int& value) {
    if (value<0) {
        value=0;
        return;
    }
    if (value>=(int)cur().size()) {
        value=(int)cur().size()-1;
        return;
    }
}

void wxStfDoc::correctRangeR(std::size_t& value) {
    if (value>=cur().size()) {
        value=cur().size()-1;
        return;
    }
}


void wxStfDoc::SetCurCh(size_t value) {
    if (value>=get().size()) {
        throw std::out_of_range("channel out of range in wxStfDoc::SetCurCh()");
    }
    cc=value;
}

void wxStfDoc::SetSecCh(size_t value) {
    if (value>=get().size() || value==cc) {
        throw std::out_of_range("channel out of range in wxStfDoc::SetSecCh()");
    }
    sc=value;
}

void wxStfDoc::SetCurSec( size_t value ) {
    if (value >= get()[cc].size()) {
        throw std::out_of_range("channel out of range in wxStfDoc::SetCurSec()");
    }
    cs=value;
}

void wxStfDoc::SetMeasCursor(int value) {
    correctRangeR(value);
    measCursor=value;
}

double wxStfDoc::GetMeasValue() {
    if (measCursor>=get()[cc].size()) {
        correctRangeR(measCursor);
    }
    return cur().at(measCursor);
}

void wxStfDoc::SetBaseBeg(int value) {
    correctRangeR(value);
    baseBeg=value;
}

void wxStfDoc::SetBaseEnd(int value) {
    correctRangeR(value);
    baseEnd=value;
}

void wxStfDoc::SetPeakBeg(int value) {
    correctRangeR(value);
    peakBeg=value;
}

void wxStfDoc::SetPeakEnd(int value) {
    correctRangeR(value);
    peakEnd=value;
}

void wxStfDoc::SetFitBeg(int value) {
    correctRangeR(value);
    fitBeg=value;
}

void wxStfDoc::SetFitEnd(int value) {
    correctRangeR(value);
    fitEnd=value;
}

void wxStfDoc::SetLatencyBeg(double value) {
    if (value<0.0) {
        value=0.0;
    }
    if (value>=(double)cur().size()) {
        value=cur().size()-1.0;
    }
    latencyStartCursor=value;
}

void wxStfDoc::SetLatencyEnd(double value) {
    if (value<0.0) {
        value=0.0;
    }
    if (value>=(double)cur().size()) {
        value=cur().size()-1.0;
    }
    latencyEndCursor=value;
}

void wxStfDoc::SetRTFactor(int value) {
    if (value < 0){
        value = 5;  
    }
    else if (value > 50) {
        value = 45;  
    }
    
    RTFactor = value;
}

#ifdef WITH_PSLOPE
void wxStfDoc::SetPSlopeBeg(int value) {
    correctRangeR(value);
    PSlopeBeg = value;
}


void wxStfDoc::SetPSlopeEnd(int value) {
    correctRangeR(value);
    PSlopeEnd = value;
}
#endif 

void wxStfDoc::SelectTrace(std::size_t sectionToSelect) {
    // Check range so that sectionToSelect can be used
    // without checking again:
    if (sectionToSelect>=get()[cc].size()) {
        std::out_of_range e("subscript out of range in wxStfDoc::SelectTrace\n");
        throw e;
    }
    selectedSections.push_back(sectionToSelect);
    double sumY=0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sumY)
#endif
    if (get()[cc][sectionToSelect].size()==0) {
        selectBase.push_back(0);
    } else {
        int start = baseBeg;
        int end = baseEnd;
        if (start > (int)get()[cc][sectionToSelect].size()-1)
            start = get()[cc][sectionToSelect].size()-1;
        if (start < 0) start = 0;
        if (end > (int)get()[cc][sectionToSelect].size()-1)
            end = get()[cc][sectionToSelect].size()-1;
        if (end < 0) end = 0;
        for (int i=start; i<=end; i++) {
            sumY += get()[cc][sectionToSelect][i];
        }
        int n=(int)(end-start+1);
        selectBase.push_back(sumY/n);
    }
}

bool wxStfDoc::UnselectTrace(std::size_t sectionToUnselect) {

    //verify whether the trace has really been selected and find the 
    //number of the trace within the selectedTraces array:
    bool traceSelected=false;
    std::size_t traceToRemove=0;
    for (std::size_t n=0; n < selectedSections.size() && !traceSelected; ++n) { 
        if (selectedSections[n] == sectionToUnselect) traceSelected=true;
        if (traceSelected) traceToRemove=n;
    }
    //Shift the selectedTraces array by one position, beginning
    //with the trace to remove: 
    if (traceSelected) {
        //shift traces by one position:
        for (std::size_t k=traceToRemove; k < GetSelectedSections().size()-1; ++k) { 
            selectedSections[k]=selectedSections[k+1];
            selectBase[k]=selectBase[k+1];
        }
        // resize vectors:
        selectedSections.resize(selectedSections.size()-1);
        selectBase.resize(selectBase.size()-1);
        return true;
    } else {
        //msgbox
        return false;
    }
}


stf::Table wxStfDoc::CurAsTable() const {
    stf::Table table(cur().size(),size());
    try {
        for (std::size_t nRow=0;nRow<table.nRows();++nRow) {
            std::ostringstream rLabel;
            rLabel << nRow*GetXScale();
            table.SetRowLabel(nRow,rLabel.str());
            for (std::size_t nCol=0;nCol<table.nCols();++nCol) {
                table.at(nRow,nCol)=get().at(nCol).at(cs).at(nRow);
            }
        }
        for (std::size_t nCol=0;nCol<table.nCols();++nCol) {
            table.SetColLabel(nCol, get().at(nCol).GetChannelName());
        }
    }
    catch (const std::out_of_range& e) {
        throw e;
    }
    return table;
}

stf::Table wxStfDoc::CurResultsTable() {
    // resize table:
    std::size_t n_cols=0;
    if (viewCrosshair) n_cols++;
    if (viewBaseline) n_cols++;
    if (viewBaseSD) n_cols++;
    if (viewThreshold) n_cols++;
    if (viewPeakzero) n_cols++;
    if (viewPeakbase) n_cols++;
    if (viewPeakthreshold) n_cols++;
    if (viewRTLoHi) n_cols++;
    if (viewInnerRiseTime) n_cols++;
    if (viewOuterRiseTime) n_cols++;
    if (viewT50) n_cols++;
    if (viewRD) n_cols++;
    if (viewSloperise) n_cols++;
    if (viewSlopedecay) n_cols++;
    if (viewLatency) n_cols++;
#ifdef WITH_PSLOPE
    if (viewPSlope) n_cols++;
#endif

    std::size_t n_rows=(viewCursors? 3:1);
    stf::Table table(n_rows,n_cols);

    // Labels
    table.SetRowLabel(0, "Value");
    if (viewCursors) {
        table.SetRowLabel(1, "Cursor 1");
        table.SetRowLabel(2, "Cursor 2");
    }
    int nCol=0;
    if (viewCrosshair) table.SetColLabel(nCol++, "Crosshair");
    if (viewBaseline) table.SetColLabel(nCol++,"Baseline");
    if (viewBaseSD) table.SetColLabel(nCol++,"Base SD");
    if (viewThreshold) table.SetColLabel(nCol++,"Threshold");
    if (viewPeakzero) table.SetColLabel(nCol++,"Peak (from 0)");
    if (viewPeakbase) table.SetColLabel(nCol++,"Peak (from base)");
    if (viewPeakthreshold) table.SetColLabel(nCol++,"Peak (from threshold)");
    if (viewRTLoHi) table.SetColLabel(nCol++,"RT (Lo-Hi%)");
    if (viewInnerRiseTime) table.SetColLabel(nCol++,"inner rise time");
    if (viewOuterRiseTime) table.SetColLabel(nCol++,"outer rise time");
    if (viewT50) table.SetColLabel(nCol++,"t50");
    if (viewRD) table.SetColLabel(nCol++,"Rise/Decay");
    if (viewSloperise) table.SetColLabel(nCol++,"Slope (rise)");
    if (viewSlopedecay) table.SetColLabel(nCol++,"Slope (decay)");
    if (viewLatency) table.SetColLabel(nCol++,"Latency");
#ifdef WITH_PSLOPE
    if (viewPSlope) table.SetColLabel(nCol++,"PSlope");
#endif

    // Values
    nCol=0;
    // measurement cursor
    if (viewCrosshair) {
        table.at(0,nCol)=GetMeasValue();
        if (viewCursors) {
            table.at(1,nCol)=GetMeasCursor()*GetXScale();
            table.SetEmpty(2,nCol,true);
        }
        nCol++;
    }

    // baseline
    if (viewBaseline) {
        table.at(0,nCol)=GetBase();
        if (viewCursors) {
            table.at(1,nCol)=GetBaseBeg()*GetXScale();
            table.at(2,nCol)=GetBaseEnd()*GetXScale();
        }
        nCol++;
    }

    // base SD
    if (viewBaseSD) {
        table.at(0,nCol)=GetBaseSD();
        if (viewCursors) {
            table.at(1,nCol)=GetBaseBeg()*GetXScale();
            table.at(2,nCol)=GetBaseEnd()*GetXScale();
        }
        nCol++;
    }

    // threshold
    if (viewThreshold) {
        table.at(0,nCol)=GetThreshold();
        if (viewCursors) {
            table.at(1,nCol)=GetPeakBeg()*GetXScale();
            table.at(2,nCol)=GetPeakEnd()*GetXScale();
        }
        nCol++;
    }
    
    // peak
    if (viewPeakzero) {
        table.at(0,nCol)=GetPeak();
        if (viewCursors) {
            table.at(1,nCol)=GetPeakBeg()*GetXScale();
            table.at(2,nCol)=GetPeakEnd()*GetXScale();
        }
        nCol++;
    }

    if (viewPeakbase) {
        table.at(0,nCol)=GetPeak()-GetBase();
        if (viewCursors) {
            table.at(1,nCol)=GetPeakBeg()*GetXScale();
            table.at(2,nCol)=GetPeakEnd()*GetXScale();
        }
        nCol++;
    }
    if (viewPeakthreshold) {
        if (thrT >= 0) {
            table.at(0,nCol) = GetPeak()-GetThreshold();
        } else {
            table.at(0,nCol) = 0;
        }
        if (viewCursors) {
            table.at(1,nCol)=GetPeakBeg()*GetXScale();
            table.at(2,nCol)=GetPeakEnd()*GetXScale();
        }
        nCol++;
    }

    // RT (Lo-Hi%)
    if (viewRTLoHi) {table.at(0,nCol)=GetRTLoHi();
        if (viewCursors) {
            table.at(1,nCol)=GetTLoReal()*GetXScale();
            table.at(2,nCol)=GetTHiReal()*GetXScale();
        }
        nCol++;
    }

    if (viewInnerRiseTime) { table.at(0,nCol)=GetInnerRiseTime();
        if (viewCursors) {
            table.at(1,nCol)=GetInnerLoRT();
            table.at(2,nCol)=GetInnerHiRT();
        }
        nCol++;
    }

    if (viewOuterRiseTime) { table.at(0,nCol)=GetOuterRiseTime();
        if (viewCursors) {
            table.at(1,nCol)=GetOuterLoRT();
            table.at(2,nCol)=GetOuterHiRT();
        }
        nCol++;
    }

    // Half duration
    if (viewT50) {table.at(0,nCol)=GetHalfDuration();
        if (viewCursors) {
            table.at(1,nCol)=GetT50LeftReal()*GetXScale();
            table.at(2,nCol)=GetT50RightReal()*GetXScale();
        }
        nCol++;
    }

    // Rise/decay
    if (viewRD) {table.at(0,nCol)=GetSlopeRatio();
        if (viewCursors) {
            table.at(1,nCol)=GetMaxRiseT()*GetXScale();
            table.at(2,nCol)=GetMaxDecayT()*GetXScale();
        }
        nCol++;
    }

    // Max rise
    if (viewSloperise) {table.at(0,nCol)=GetMaxRise();
        if (viewCursors) {
            table.at(1,nCol)=GetMaxRiseT()*GetXScale();
            table.SetEmpty(2,nCol,true);
        }
        nCol++;
    }

    // Max decay
    if (viewSlopedecay) {table.at(0,nCol)=GetMaxDecay();
        if (viewCursors) {
            table.at(1,nCol)=GetMaxDecayT()*GetXScale();
            table.SetEmpty(2,nCol,true);
        }
        nCol++;
    }

    // Latency
    if (viewLatency) {table.at(0,nCol)=GetLatency()*GetXScale();
        if (viewCursors) {
            table.at(1,nCol)=GetLatencyBeg()*GetXScale();
            table.at(2,nCol)=GetLatencyEnd()*GetXScale();
        }
        nCol++;
    }

#ifdef WITH_PSLOPE
    // PSlope
    if (viewPSlope) {table.at(0,nCol)=GetPSlope();
        if (viewCursors) {
            table.at(1,nCol)=GetPSlopeBeg()*GetXScale();
            table.at(2,nCol)=GetPSlopeEnd()*GetXScale();
        }
        nCol++;
    }
#endif // WITH_PSLOPE
    return table;
}


void wxStfDoc::resize(std::size_t c_n_channels) {
    Recording::resize(c_n_channels);
    yzoom.resize(size());
    sec_attr.resize(size());
    for (std::size_t nchannel = 0; nchannel < size(); ++nchannel) {
        sec_attr[nchannel].resize(at(nchannel).size());
    }
}

void wxStfDoc::InsertChannel(Channel& c_Channel, std::size_t pos) {
    Recording::InsertChannel(c_Channel, pos);
    yzoom.resize(size());
    sec_attr.resize(size());
    for (std::size_t nchannel = 0; nchannel < size(); ++nchannel) {
        sec_attr[nchannel].resize(at(nchannel).size());
    }
}

void wxStfDoc::SetIsFitted( std::size_t nchannel, std::size_t nsection,
                            const Vector_double& bestFitP_, stf::storedFunc* fitFunc_,
                            double chisqr, std::size_t fitBeg, std::size_t fitEnd )
{
    if (nchannel >= sec_attr.size() || nsection >= sec_attr[nchannel].size()) {
        throw std::out_of_range("Index out of range in wxStfDoc::SetIsFitted");
    }
    if ( !fitFunc_ ) {
        throw std::runtime_error("Function pointer is zero in wxStfDoc::SetIsFitted");
    }
    if ( fitFunc_->pInfo.size() != bestFitP_.size() ) {
        throw std::runtime_error("Number of best-fit parameters doesn't match number\n \
                                 of function parameters in wxStfDoc::SetIsFitted");
    }
    sec_attr[nchannel][nsection].fitFunc = fitFunc_;
    if ( sec_attr[nchannel][nsection].bestFitP.size() != bestFitP_.size() )
        sec_attr[nchannel][nsection].bestFitP.resize(bestFitP_.size()); 
    sec_attr[nchannel][nsection].bestFitP = bestFitP_;
    sec_attr[nchannel][nsection].bestFit =
        sec_attr[nchannel][nsection].fitFunc->output(sec_attr[nchannel][nsection].bestFitP,
                                                     sec_attr[nchannel][nsection].fitFunc->pInfo, chisqr );
    sec_attr[nchannel][nsection].storeFitBeg = fitBeg;
    sec_attr[nchannel][nsection].storeFitEnd = fitEnd;
    sec_attr[nchannel][nsection].isFitted = true;
}

void wxStfDoc::DeleteFit(std::size_t nchannel, std::size_t nsection) {
    if (nchannel >= sec_attr.size() || nsection >= sec_attr[nchannel].size()) {
        throw std::out_of_range("Index out of range in wxStfDoc::DeleteFit");
    }
    sec_attr[nchannel][nsection].fitFunc = NULL;
    sec_attr[nchannel][nsection].bestFitP.resize( 0 );
    sec_attr[nchannel][nsection].bestFit = stf::Table( 0, 0 );
    sec_attr[nchannel][nsection].isFitted = false;
}


void wxStfDoc::SetIsIntegrated(std::size_t nchannel, std::size_t nsection, bool value,
                               std::size_t begin, std::size_t end, const Vector_double& quad_p_)
{
    if (nchannel >= sec_attr.size() || nsection >= sec_attr[nchannel].size()) {
        throw std::out_of_range("Index out of range in wxStfDoc::SetIsIntegrated");
    }
    if (value==false) {
        sec_attr[nchannel][nsection].isIntegrated=value;
        return;
    }
    if (end<=begin) {
        throw std::out_of_range("integration limits out of range in Section::SetIsIntegrated");
    }
    int n_intervals=std::div((int)end-(int)begin,2).quot;
    if ((int)quad_p_.size() != n_intervals*3) {
        throw std::out_of_range("Wrong number of parameters for quadratic equations in Section::SetIsIntegrated");
    }
    sec_attr[nchannel][nsection].quad_p = quad_p_;
    sec_attr[nchannel][nsection].isIntegrated=value;
    sec_attr[nchannel][nsection].storeIntBeg=begin;
    sec_attr[nchannel][nsection].storeIntEnd=end;
}

void wxStfDoc::ClearEvents(std::size_t nchannel, std::size_t nsection) {
    try {
        sec_attr.at(nchannel).at(nsection).eventList.clear();
    }
    catch(const std::out_of_range& e) {
        throw e;
    }
}

const stf::SectionAttributes& wxStfDoc::GetSectionAttributes(std::size_t nchannel, std::size_t nsection) const {
    try {
        return sec_attr.at(nchannel).at(nsection);
    }
    catch(const std::out_of_range& e) {
        throw e;
    }
}

const stf::SectionAttributes& wxStfDoc::GetCurrentSectionAttributes() const {
    try {
        return sec_attr.at(GetCurCh()).at(GetCurSec());
    }
    catch(const std::out_of_range& e) {
        throw e;
    }
}

stf::SectionAttributes& wxStfDoc::GetCurrentSectionAttributesW() {
    try {
        return sec_attr.at(GetCurCh()).at(GetCurSec());
    }
    catch(const std::out_of_range& e) {
        throw e;
    }
}

#if 0
void wxStfDoc::Userdef(std::size_t id) {
    wxBusyCursor wc;
    int fselect=(int)id;
    Recording newR;
    Vector_double init(0);
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
#endif
