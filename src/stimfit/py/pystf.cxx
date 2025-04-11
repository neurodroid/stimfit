#include <stdlib.h>
#include <math.h>

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#ifdef _WINDOWS
    #ifdef _DEBUG
        #undef _DEBUG
        #define _UNDEBUG
    #endif
#endif

#ifdef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE_WAS_DEF
#undef _POSIX_C_SOURCE
#endif
#ifdef _XOPEN_SOURCE
#define _XOPEN_SOURCE_WAS_DEF
#undef _XOPEN_SOURCE
#endif

#ifdef WITH_PYTHON
    #include <Python.h>
#endif

#ifdef _POSIX_C_SOURCE_WAS_DEF
  #ifndef _POSIX_C_SOURCE
    #define _POSIX_C_SOURCE
  #endif
#endif
#ifdef _XOPEN_SOURCE_WAS_DEF
  #ifndef _XOPEN_SOURCE
    #define _XOPEN_SOURCE
  #endif
#endif

#if defined(__WXMAC__) || defined(__WXGTK__)
  #pragma GCC diagnostic ignored "-Wwrite-strings"
#endif

#ifdef WITH_PYTHON
    #if PY_MAJOR_VERSION >= 3
        #include "sip.h"
        #include <wxPython/wxpy_api.h>
        #define PyInt_FromLong PyLong_FromLong
        #define PyString_AsString PyBytes_AsString
    #else
        #include <wx/wxPython/wxPython.h>
    #endif
#endif

// revert to previous behaviour
#if defined(__WXMAC__) || defined(__WXGTK__)
  #pragma GCC diagnostic warning "-Wwrite-strings"
#endif

#ifdef _WINDOWS
    #ifdef _UNDEBUG
        #define _DEBUG
    #endif
#endif

#ifdef WITH_PYTHON
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#include <numpy/arrayobject.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#endif

#include "pystf.h"

#include "./../gui/app.h"
#include "./../gui/doc.h"
#include "./../gui/view.h"
#include "./../gui/graph.h"
#include "./../gui/parentframe.h"
#include "./../gui/childframe.h"
#include "./../gui/dlgs/cursorsdlg.h"
#include "./../../libstfnum/fit.h"

#ifdef WITH_PYTHON
#define array_data(a)          (((PyArrayObject *)a)->data)
#endif

std::vector< std::vector< Vector_double > > gMatrix;
std::vector< std::string > gNames;
double _figsize[] = {8.0,6.0};

#if PY_MAJOR_VERSION >= 3
int wrap_array() {
#ifdef WITH_PYTHON
    import_array1(-1);
#endif
    return 0;
}
#else
void wrap_array() {
#ifdef WITH_PYTHON
    import_array();
#endif
}
#endif

void ShowExcept(const std::exception& e) {
    wxString msg;
    msg << wxT("Caught an exception in the python module:\n")
        << wxString( e.what(), wxConvLocal );
    wxGetApp().ExceptMsg( msg );
    return;
}

void ShowError( const wxString& msg ) {
    wxString fullmsg;
    fullmsg << wxT("Error in the python module:\n")
            << msg;
    wxGetApp().ErrorMsg( msg );
    return;
}

wxStfDoc* actDoc() {
    return wxGetApp().GetActiveDoc();
}

wxStfGraph* actGraph() {
    if ( !check_doc() ) return NULL;

    wxStfView* pView=(wxStfView*)actDoc()->GetFirstView();
    if ( !pView )
        return NULL;

    return pView->GetGraph();
}

bool refresh_graph() {
    wxStfGraph* pGraph = actGraph();
    if ( !pGraph ) {
        ShowError( wxT("Pointer to graph is zero") );
        return false;
    }
    pGraph->Refresh();
    return true;
}

// update data and labels in the results box
bool update_results_table(){
    wxStfChildFrame* pFrame = (wxStfChildFrame*)actDoc()->GetDocumentWindow();
    if (!pFrame) {
        ShowError( wxT("Error in update_results_table()") );
        return false;
    }

    wxGetApp().OnPeakcalcexecMsg();
    pFrame->UpdateResults();
    return true;
}

void write_stf_registry(const wxString& item, int value){
    wxGetApp().wxWriteProfileInt(wxT("Settings"), item, value);
}

bool update_cursor_dialog( ) {
    if (wxGetApp().GetCursorsDialog()!=NULL && wxGetApp().GetCursorsDialog()->IsShown()) {
        try {
            wxGetApp().GetCursorsDialog()->UpdateCursors();
        }
        catch (const std::runtime_error& e) {
            ShowExcept( e );
            // We don't necessarily need to return false here.
        }
    }

    return refresh_graph();
}

bool check_doc( bool show_dialog ) {
    if (actDoc() == NULL)  {
        if (show_dialog)
            ShowError( wxT("Couldn't find open file") );
        return false;
    }
    return true;
}

std::string get_filename( ) {
    if ( !check_doc() ) return 0;

#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
    return std::string(actDoc()->GetFilename());
#else
    return std::string(actDoc()->GetFilename().mb_str());
#endif
}


std::string get_versionstring() {
#if (wxCHECK_VERSION(2, 9, 0) || defined(MODULE_ONLY))
    return std::string(wxGetApp().GetVersionString());
#else
    return std::string(wxGetApp().GetVersionString().mb_str());
#endif
}

#ifdef WITH_PYTHON
PyObject* get_trace(int trace, int channel) {
    wrap_array();

    if ( !check_doc() ) return NULL;

    if ( trace == -1 ) {
        trace = actDoc()->GetCurSecIndex();
    }
    if ( channel == -1 ) {
        channel = actDoc()->GetCurChIndex();
    }

    npy_intp dims[1] = {(int)actDoc()->at(channel).at(trace).size()};
    PyObject* np_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double* gDataP = (double*)PyArray_DATA((PyArrayObject*)np_array);

    /* fill */
    std::copy( (*actDoc())[channel][trace].get().begin(),
               (*actDoc())[channel][trace].get().end(),
               gDataP);

    return np_array;
}
#endif

bool new_window( double* invec, int size ) {
    bool open_doc = actDoc() != NULL;

    std::vector< double > va(size);
    std::copy( &invec[0], &invec[size], va.begin() );
    Section sec(va);
    Channel ch(sec);
    if (open_doc) {
        ch.SetYUnits( actDoc()->at( actDoc()->GetCurChIndex() ).GetYUnits() );
    }
    Recording new_rec( ch );
    if (open_doc) {
        new_rec.SetXScale( actDoc()->GetXScale() );
    }
    wxStfDoc* testDoc = wxGetApp().NewChild( new_rec, actDoc(), wxT("From python") );
    if ( testDoc == NULL ) {
        ShowError( wxT("Failed to create a new window.") );
        return false;
    }
    return true;
}

bool _new_window_gMatrix( ) {
    bool open_doc = actDoc() != NULL;

    Recording new_rec( gMatrix.size() );
    for (std::size_t n_c=0; n_c < new_rec.size(); ++n_c) {
        Channel ch( gMatrix[n_c].size() );
        for ( std::size_t n_s = 0; n_s < ch.size(); ++n_s ) {
            ch.InsertSection( Section(gMatrix[n_c][n_s]), n_s );
        }
        std::string yunits = "";
        if (open_doc) {
            yunits = actDoc()->at( actDoc()->GetCurChIndex() ).GetYUnits();
        }
        ch.SetYUnits( yunits );
        if ( !gNames.empty() ) {
            ch.SetChannelName(gNames[n_c]);
        }
        new_rec.InsertChannel( ch, n_c );
    }
    gNames.resize(0);
    double xscale = 1.0;
    if (open_doc) {
        xscale =  actDoc()->GetXScale();
    }
    new_rec.SetXScale( xscale );
    wxStfDoc* pDoc = NULL;
    if ( open_doc ) {
        pDoc = actDoc();
    }
    wxStfDoc* testDoc = wxGetApp().NewChild( new_rec, pDoc, wxT("From python") );
    if ( testDoc == NULL ) {
        ShowError( wxT("Failed to create a new window.") );
        return false;
    }
    return true;
}

bool new_window_matrix( double* invec, int traces, int size ) {
    bool open_doc = actDoc() != NULL;

    Channel ch( traces );
    for (int n = 0; n < traces; ++n) {
        std::size_t offset = n * size;
        std::vector< double > va(size);
        std::copy( &invec[offset], &invec[offset+size], &va[0] );
        Section sec(va);
        ch.InsertSection(sec, n);
    }
    if (open_doc) {
        ch.SetYUnits( actDoc()->at( actDoc()->GetCurChIndex() ).GetYUnits() );
    }
    Recording new_rec( ch );
    if (open_doc) {
        new_rec.SetXScale( actDoc()->GetXScale() );
    }
    wxStfDoc* testDoc = wxGetApp().NewChild( new_rec, actDoc(), wxT("From python") );
    if ( testDoc == NULL ) {
        ShowError( wxT("Failed to create a new window.") );
        return false;
    }
    return true;
}

bool new_window_selected_this( ) {
    if ( !check_doc() ) return false;

    if ( !actDoc()->OnNewfromselectedThis( ) ) {
        return false;
    }
    return true;
}

bool new_window_selected_all( ) {
    if ( !check_doc() ) return false;

    try {
        wxCommandEvent wce;
        wxGetApp().OnNewfromselected( wce );
    }
    catch ( const std::exception& e) {
        ShowExcept( e );
        return false;
    }
    return true;
}

int get_size_trace( int trace, int channel ) {
    if ( !check_doc() ) return 0;

    if ( trace == -1 ) {
        trace = actDoc()->GetCurSecIndex();
    }
    if ( channel == -1 ) {
        channel = actDoc()->GetCurChIndex();
    }

    int size = 0;
    try {
        size = actDoc()->at(channel).at(trace).size();
    }
    catch ( const std::out_of_range& e) {
        ShowExcept( e );
        return 0;
    }
    return size;
}

int get_size_channel( int channel ) {
    if ( !check_doc() ) return 0;

    if ( channel == -1 ) {
        channel = actDoc()->GetCurChIndex();
    }

    int size = 0;
    try {
        size = actDoc()->at(channel).size();
    }
    catch ( const std::out_of_range& e) {
        ShowExcept( e );
        return 0;
    }
    return size;
}

int get_size_recording( ) {
    if ( !check_doc() ) return 0;
    return actDoc()->size();
}

double get_maxdecay() {
    if ( !check_doc() ) return -1.0;

    return actDoc()->GetMaxDecay();
}

double get_maxrise() {
    if ( !check_doc() ) return -1.0;

    return actDoc()->GetMaxRise();
}

const char* get_recording_time( ) {
    if ( !check_doc() ) return 0;
    static char buffer[9];
    struct tm dt = actDoc()->GetDateTime();
    // Use strftime to format the time into the buffer.
    if (strftime(buffer, sizeof(buffer), "%H:%M:%S", &dt) == 0) {
        return NULL;
    }
    return buffer;
}

const char* get_recording_date( ) {
    if ( !check_doc() ) return 0;
    static char buffer[11];
    struct tm dt = actDoc()->GetDateTime();
    // Use strftime to format the time into the buffer.
    if (strftime(buffer, sizeof(buffer), "%Y:%m:%d", &dt) == 0) {
        return NULL;
    }
    return buffer;
}

std::string get_recording_comment( ) {
    if ( !check_doc() ) return "";
    std::ostringstream comment;
    comment << actDoc()->GetFileDescription() << actDoc()->GetGlobalSectionDescription();
    return comment.str();
}

bool set_recording_comment( const char* comment ) {
    if ( !check_doc() ) return false;
    actDoc()->SetFileDescription(comment);
    return true;
}

bool set_recording_date( const char* date ) {
    if ( !check_doc() ) return false;
    actDoc()->SetDate(date);
    return true;
}

bool set_recording_time( const char* time ) {
    if ( !check_doc() ) return false;
    actDoc()->SetTime(time);
    return true;
}

bool select_trace( int trace ) {
    if ( !check_doc() ) return false;
    int max_size = (int)actDoc()->at(actDoc()->GetCurChIndex()).size();
    if (trace < -1 || trace >= max_size) {
        wxString msg;
        msg << wxT("Select a trace with a zero-based index between 0 and ") << max_size-1;
        ShowError( msg );
        return false;
    }
    if ((int)actDoc()->GetSelectedSections().size() == max_size) {
        ShowError(wxT("No more traces can be selected\nAll traces are selected"));
        return false;
    }
    if ( trace == -1 ) {
        trace = actDoc()->GetCurSecIndex();
    }

    // control whether trace has already been selected:
    bool already=false;
    for (c_st_it cit = actDoc()->GetSelectedSections().begin();
         cit != actDoc()->GetSelectedSections().end() && !already;
         ++cit) {
        if ((int)*cit == trace) {
            already = true;
        }
    }

    // add trace number to selected numbers, print number of selected traces
    if (!already) {
        actDoc()->SelectTrace(trace, actDoc()->GetBaseBeg(), actDoc()->GetBaseEnd());
        //String output in the trace navigator
        wxStfChildFrame* pFrame = (wxStfChildFrame*)actDoc()->GetDocumentWindow();
        if ( !pFrame ) {
            ShowError( wxT("Pointer to frame is zero") );
            return false;
        }
        pFrame->SetSelected(actDoc()->GetSelectedSections().size());
    } else {
        ShowError( wxT("Trace is already selected") );
        return false;
    }
    return true;
}

void select_all( ) {
    if ( !check_doc() ) return;

    wxCommandEvent wce;
    actDoc()->Selectall( wce );
}

void unselect_all( ) {
    if ( !check_doc() ) return;

    wxCommandEvent wce;
    actDoc()->Deleteselected( wce );
}

#ifdef WITH_PYTHON
PyObject* get_selected_indices() {
    if ( !check_doc() ) return NULL;

    PyObject* retObj = PyTuple_New( (int)actDoc()->GetSelectedSections().size() );
    c_st_it cit;
    int n=0;
    for ( cit = actDoc()->GetSelectedSections().begin(); cit != actDoc()->GetSelectedSections().end(); ++cit ) {
        PyTuple_SetItem(retObj, n++, PyInt_FromLong( (long)*cit ) );
    }

    // The main program apparently takes the ownership of the tuple;
    // no reference count decrement should be performed here.
    return retObj;
}
#endif

bool set_trace( int trace ) {
    if ( !check_doc() ) return false; // use only with open document

    if ( !actDoc()->SetSection( trace ) ) {
        return false;
    }
    wxGetApp().OnPeakcalcexecMsg();

    wxStfChildFrame* pFrame = (wxStfChildFrame*)actDoc()->GetDocumentWindow();
    if ( !pFrame ) {
        ShowError( wxT("Pointer to frame is zero") );
        return false;
    }
    pFrame->SetCurTrace( trace );

    return refresh_graph();
}

int get_trace_index() {
    if ( !check_doc() )
        return -1;

    return actDoc()->GetCurSecIndex();
}

int get_channel_index( bool active ) {
    if ( !check_doc() )
        return -1;

    if ( active )
        return actDoc()->GetCurChIndex();
    else
        return actDoc()->GetSecChIndex();
}

bool set_channel(int channel) {
    if ( !check_doc() ) return false; // use only with open document

    // channel negative
    if (channel<0) {
        ShowError( wxT("Negative value is not allowed") );
        return false;
    }

    // only if we want to change the active channel
    if ((unsigned int)channel == actDoc()->GetCurChIndex() ) {
        return true;
    }

    int reference_ch = actDoc()->GetCurChIndex();

    // catch exceptions (i.e out of range)
    try {
        actDoc()->SetCurChIndex(channel);
    }
    catch (const std::out_of_range& e) {
        ShowError( wxT("Value exceeds the number of available channels") );
        return false;
    }

    // Pointer to wxStfChildFrame to access Channel selection combo
    wxStfChildFrame* pFrame = (wxStfChildFrame*)actDoc()->GetDocumentWindow();
    if (!pFrame) {
        ShowError( wxT("Pointer to frame is zero") );
        return false;
    }
    // set the channel selection combo
    //pFrame->SetChannels( actDoc()->GetCurChIndex(), actDoc()->GetSecChIndex());
    pFrame->SetChannels( actDoc()->GetCurChIndex(), reference_ch);
    pFrame->UpdateChannels(); // update according to the combo
    return refresh_graph();
}


const char* get_channel_name( int index ) {
    if ( !check_doc() ) return "";

    if (index < 0) {
        index = actDoc()->GetCurChIndex();
    }
    try {
        return actDoc()->at( index ).GetChannelName().c_str();
    }
    catch (const std::out_of_range& e) {
        wxString msg(wxT("Index out of range in get_channel_name:\n"));
        msg+=wxString( e.what(), wxConvLocal );
        ShowError( msg );
        return "";
    }
}

bool set_channel_name( const char* name, int index ) {
    if ( !check_doc() ) return "";

    if (index < 0) {
        index = actDoc()->GetCurChIndex();
    }
    try {
        actDoc()->at( index ).SetChannelName(name);
    }
    catch (const std::out_of_range& e) {
        wxString msg(wxT("Index out of range in get_channel_name:\n"));
        msg+=wxString( e.what(), wxConvLocal );
        ShowError( msg );
        return false;
    }
    return true;
}

const char* get_trace_name( int trace, int channel ) {
    if ( !check_doc() ) return "";

    if (channel < 0) {
        channel = actDoc()->GetCurChIndex();
    }
    if (trace < 0) {
        trace = actDoc()->GetCurSecIndex();
    }
    try {
        return actDoc()->at( channel ).at( trace ).GetSectionDescription().c_str();
    }
    catch (const std::out_of_range& e) {
        wxString msg(wxT("Index out of range in get_trace_name:\n"));
        msg+=wxString( e.what(), wxConvLocal );
        ShowError( msg );
        return "";
    }
}

bool subtract_base( ) {
    if ( !check_doc() ) return false;

    return actDoc()->SubtractBase();
}

bool file_open( const char* filename ) {
    wxString wxFilename( filename, wxConvLocal );
    return wxGetApp().OpenFilePy( wxFilename );
}

bool file_save( const char* filename ) {
    if ( !check_doc() ) return false;

    wxString wxFilename( filename, wxConvLocal );
    return actDoc()->OnSaveDocument( wxFilename );
}

bool close_all( ) {
    return wxGetApp().CloseAll();
}

bool close_this( ) {
    if ( !check_doc() ) return false;
    return actDoc()->DeleteAllViews( );
}

double peak_index( bool active ) {
    if ( !check_doc() ) return -1.0;

    if ( active ) {
        return actDoc()->GetMaxT();
    } else {
        // Test whether a second channel is available at all:
        if ( actDoc()->size() < 2 ) {
            ShowError( wxT("No second channel found") );
            return -1.0;
        }
        return actDoc()->GetAPMaxT();
    }
}

double maxrise_index( bool active ) {
    if ( !check_doc() ) return -1.0;

    if ( active ) {
        return actDoc()->GetMaxRiseT();
    } else {
        // Test whether a second channel is available at all:
        if ( actDoc()->size() < 2 ) {
            ShowError( wxT("No second channel found") );
            return -1.0;
        }
        return actDoc()->GetAPMaxRiseT();
    }
}

double maxdecay_index( ) {
    if ( !check_doc() ) return -1.0;

    return actDoc()->GetMaxDecayT();
}

double foot_index( bool active ) {
    if ( !check_doc() ) return -1.0;

    if ( active ) {
        return  actDoc()->GetTLoReal() - (actDoc()->GetTHiReal() - actDoc()->GetTLoReal()) / 3.0;
    } else {
        ShowError( wxT("At this time, foot_index() is only implemented for the active channel") );
        return -1.0;
    }
}

double t50left_index( bool active ) {
    if ( !check_doc() ) return -1.0;

    if ( active ) {
        return actDoc()->GetT50LeftReal();
    } else {
        // Test whether a second channel is available at all:
        if ( actDoc()->size() < 2 ) {
            ShowError( wxT("No second channel found") );
            return -1.0;
        }
        return actDoc()->GetAPT50LeftReal();
    }
}

double t50right_index( bool active ) {
    if ( !check_doc() ) return -1.0;

    if ( active ) {
        return actDoc()->GetT50RightReal();
    } else {
        ShowError( wxT("At this time, t50right_index() is only implemented for the active channel") );
        return -1.0;
    }
}

double get_halfwidth( bool active ) {
    if ( !check_doc() ) return -1.0;

    if ( active ) {
        double dt = actDoc()->GetXScale();
        double t50left = actDoc()->GetT50LeftReal();
        double t50right = actDoc()->GetT50RightReal();
        return ( t50right-t50left )*dt;
    } else {
        ShowError( wxT("At this time, halfwidth is only implemented for the active channel") );
        return -1.0;
    }

}

double rtlow_index( bool active ) {
    if ( !check_doc() ) return -1.0;

    if ( active ) {
        return actDoc()->GetTLoReal();
    } else {
        ShowError( wxT("At this time, rtlow_index() is only implemented for the active channel") );
        return -1.0;
    }
}

double rthigh_index( bool active ) {
    if ( !check_doc() ) return -1.0;

    if ( active ) {
        return actDoc()->GetTHiReal();
    } else {
        ShowError( wxT("At this time, rthigh_index() is only implemented for the active channel") );
        return -1.0;
    }
}

double get_threshold_time( bool is_time ) {
    if ( !check_doc() ) return -1;

    if ( !is_time )
        return actDoc()->GetThrT();
    else
        return (double)actDoc()->GetThrT() * actDoc()->GetXScale();
}

double get_threshold_value( ) {
    if ( !check_doc() ) return -1;

    return actDoc()->GetThreshold();

}

double get_latency( ) {
    if ( !check_doc() ) return -1.0;

        double dt = actDoc()->GetXScale();
        return ( actDoc()->GetLatency() )*dt;

}

double get_risetime( ) {
    if ( !check_doc() ) return -1.0;

    double dt = actDoc()->GetXScale();
    return ( actDoc()->GetTHiReal()-actDoc()->GetTLoReal() )*dt;

}

double get_risetime_factor() {

    if ( !check_doc() ) return -1.0;
    return actDoc()->GetRTFactor()/100.;

}

double get_fit_start( bool is_time ) {
    if ( !check_doc() ) return -1;

    if ( !is_time )
        return actDoc()->GetFitBeg();
    else
        return (double)actDoc()->GetFitBeg() * actDoc()->GetXScale();
}
bool set_risetime_factor(double factor) {

    if ( !check_doc() ) return false;

    if (factor > 0.45 || factor < 0.05) {
        ShowError( wxT("Value out of range (0.05-0.45) in set_risetime_factor()") );
        return false;
    }

    int RTFactor = (int)(factor*100);
    actDoc()->SetRTFactor(RTFactor); // defined in wxStfApp::OnPeakcalcexecMsg
    update_cursor_dialog();
    update_results_table();
    write_stf_registry(wxT("RTFactor"), RTFactor);
    return true;


}
bool set_fit_start( double pos, bool is_time ) {
    if ( !check_doc() ) return false;

    if ( is_time )
        pos /= actDoc()->GetXScale();

    int posInt = stf::round( pos );
    // range check:
    if ( posInt < 0 || posInt >= (int)actDoc()->cursec().size() ) {
        ShowError( wxT("Value out of range in set_fit_start()") );
        return false;
    }
    //conversion of pixel on screen to time (inversion of xFormat())
    if (wxGetApp().GetCursorsDialog() != NULL && wxGetApp().GetCursorsDialog()->GetStartFitAtPeak()) {
        ShowError(
                wxT("Fit will start at the peak. Change cursor settings (Edit->Cursor settings) to set manually.") );
        return false;
    }

    actDoc()->SetFitBeg( posInt );

    return update_cursor_dialog();
}

double get_fit_end( bool is_time ) {
    if ( !check_doc() ) return -1;

    if ( !is_time )
        return actDoc()->GetFitEnd();
    else
        return (double)actDoc()->GetFitEnd() * actDoc()->GetXScale();
}

bool set_fit_end( double pos, bool is_time ) {
    if ( !check_doc() ) return false;

    if ( is_time )
        pos /= actDoc()->GetXScale();

    int posInt = stf::round( pos );

    // range check:
    if ( posInt < 0 || posInt >= (int)actDoc()->cursec().size() ) {
        ShowError( wxT("Value out of range in set_fit_end()") );
        return false;
    }
    //conversion of pixel on screen to time (inversion of xFormat())
    if (wxGetApp().GetCursorsDialog() != NULL && wxGetApp().GetCursorsDialog()->GetStartFitAtPeak()) {
        ShowError(
                wxT("Fit will start at the peak. Change cursor settings (Edit->Cursor settings) to set manually.") );
        return false;
    }

    actDoc()->SetFitEnd( posInt );

    return update_cursor_dialog();
}

double get_peak_start( bool is_time ) {
    if ( !check_doc() ) return -1;

    if ( !is_time )
        return actDoc()->GetPeakBeg();
    else
        return (double)actDoc()->GetPeakBeg() * actDoc()->GetXScale();
}

bool set_peak_start( double pos, bool is_time ) {
    if ( !check_doc() ) return false;

    if ( is_time )
        pos /= actDoc()->GetXScale();

    int posInt = stf::round( pos );

    // range check:
    if ( posInt < 0 || posInt >= (int)actDoc()->cursec().size() ) {
        ShowError( wxT("Value out of range in set_peak_start()") );
        return false;
    }

    actDoc()->SetPeakBeg( posInt );

    return update_cursor_dialog();
}

double get_peak_end( bool is_time ) {
    if ( !check_doc() ) return -1;

    if ( !is_time )
        return actDoc()->GetPeakEnd();
    else
        return (double)actDoc()->GetPeakEnd() * actDoc()->GetXScale();
}

bool set_peak_end( double pos, bool is_time ) {
    if ( !check_doc() ) return false;

    if ( is_time )
        pos /= actDoc()->GetXScale();

    int posInt = stf::round( pos );

    // range check:
    if ( posInt < 0 || posInt >= (int)actDoc()->cursec().size() ) {
        ShowError( wxT("Value out of range in set_peak_end()") );
        return false;
    }

    actDoc()->SetPeakEnd( posInt );

    return update_cursor_dialog();
}

bool set_peak_mean( int pts ) {
    if ( !check_doc() ) return false;

    // range check (-1 is a legal value!):
    if ( pts == 0 || pts < -1 ) {
        ShowError( wxT("Value out of range in set_peak_mean()") );
        return false;
    }

    actDoc()->SetPM( pts );

    return update_cursor_dialog();
}

int get_peak_mean() {
    if ( !check_doc() ) return 0;

    return (int)actDoc()->GetPM();
}

const char* get_peak_direction( ) {
    if ( !check_doc() ) return "";

    const char *direction = "both";
    if ( actDoc()->GetDirection() == stfnum::up )
        direction = "up";
    else if ( actDoc()->GetDirection() == stfnum::down )
        direction = "down";
    else if ( actDoc()->GetDirection() == stfnum::both )
        direction = "both";

    return direction;
}

bool set_peak_direction( const char* direction ) {
    if ( !check_doc() ) return false;

    if ( strcmp( direction, "up" ) == 0 ) {
        actDoc()->SetDirection( stfnum::up );
        return update_cursor_dialog();
    }

    if ( strcmp( direction, "down" ) == 0 ) {
        actDoc()->SetDirection( stfnum::down );
        return update_cursor_dialog();
    }

    if ( strcmp( direction, "both" ) == 0 ) {
        actDoc()->SetDirection( stfnum::both );
        return update_cursor_dialog();
    }

    wxString msg;
    msg << wxT("\"") << wxString::FromAscii(direction) << wxT("\" is not a valid direction\n");
    msg << wxT("Use \"up\", \"down\" or \"both\"");
    ShowError( msg );
    return false;

}
const char* get_baseline_method() {
    if ( !check_doc() ) return "";

    const char *method=" ";
    if ( actDoc()->GetBaselineMethod() == stfnum::mean_sd )
        method = "mean";
    else if ( actDoc()->GetBaselineMethod() == stfnum::median_iqr )
        method = "median";

    return method;
}

bool set_baseline_method( const char* method ) {
    if ( !check_doc() ) return false;

    const wxString myitem = wxT("BaselineMethod");
    if ( strcmp( method, "mean" ) == 0 ) {
        actDoc()->SetBaselineMethod( stfnum::mean_sd );
        update_cursor_dialog();
        update_results_table();
        write_stf_registry(myitem, stfnum::mean_sd);
        return true;
    }
    else if ( strcmp( method, "median" ) == 0 ) {
        actDoc()->SetBaselineMethod( stfnum::median_iqr );
        update_cursor_dialog(); // update wxStfCursorsDlg
        update_results_table(); // update results and labels in the table
        write_stf_registry(myitem, stfnum::median_iqr); // write in .Stimfit
        return true;
    }
    else {
        wxString msg;
        msg << wxT("\"") << wxString::FromAscii(method) << wxT("\" is not a valid method\n");
        msg << wxT("Use \"mean\" or \"median\"");
        ShowError( msg );
        return false;
    }
}


const char* get_latency_start_mode( ) {
    if ( !check_doc() ) return "";

    const char *mode = "undefined";
    if ( actDoc()->GetLatencyStartMode() == stf::manualMode )
        mode = "manual";
    else if ( actDoc()->GetLatencyStartMode() == stf::peakMode )
        mode = "peak";
    else if ( actDoc()->GetLatencyStartMode() == stf::riseMode )
        mode = "rise";
    else if ( actDoc()->GetLatencyStartMode() == stf::halfMode )
        mode = "half";

    return mode;
}

bool set_latency_start(double pos, bool is_time) {
    if ( !check_doc() ) return false;

    if (is_time) pos /= actDoc()->GetXScale();

    int posInt = stf::round( pos );

    /* range check */
    if ( posInt < 0 || posInt >= (int)actDoc()->cursec().size() ){
        ShowError( wxT("Value out of range in set_latency_start()") );
        return false;
    }

    actDoc()->SetLatencyStartMode( stf::manualMode );
    actDoc()->SetLatencyBeg( posInt );

    /* set start latency mode to manual and write in registry */
    const wxString myitem = wxT("LatencyStartMode");
    bool dlg_OK, result_OK;
    dlg_OK = update_cursor_dialog();
    result_OK = update_results_table();
    if (dlg_OK && result_OK) {
        write_stf_registry(myitem, stf::manualMode);
        return true;
        }
    return false;

}

bool set_latency_start_mode( const char* mode ) {
    if ( !check_doc() ) return false;

    const wxString myitem = wxT("LatencyStartMode");
    bool dlg_OK, result_OK;
    if ( strcmp( mode, "manual" ) == 0 ) {
        actDoc()->SetLatencyStartMode( stf::manualMode );
        dlg_OK = update_cursor_dialog();
        result_OK = update_results_table();
        if (dlg_OK && result_OK) {
            write_stf_registry(myitem, stf::manualMode);
            return true;
        }
        return false;
    }
    else if ( strcmp( mode, "peak" ) == 0 ) {
        actDoc()->SetLatencyStartMode( stf::peakMode );
        dlg_OK = update_cursor_dialog();
        result_OK = update_results_table();
        if (dlg_OK && result_OK) {
            write_stf_registry(myitem, stf::peakMode);
            return true;
        }
        return false;
    }
    else if ( strcmp( mode, "rise" ) == 0 ) {
        actDoc()->SetLatencyStartMode( stf::riseMode );
        dlg_OK = update_cursor_dialog();
        result_OK = update_results_table();
        if (dlg_OK && result_OK) {
            write_stf_registry(myitem, stf::riseMode);
            return true;
        }
        return false;
    }
    else if ( strcmp( mode, "half" ) == 0 ) {
        actDoc()->SetLatencyStartMode( stf::halfMode );
        dlg_OK = update_cursor_dialog();
        result_OK = update_results_table();
        if (dlg_OK && result_OK) {
            write_stf_registry(myitem, stf::halfMode);
            return true;
        }
        return false;
    }
    else {
        wxString msg;
        msg << wxT("\"") << wxString::FromAscii(mode) << wxT("\" is not a valid start latency mode\n");
        msg << wxT("Use \"manual\", \"peak\", \"rise\" or \"half\"");
        ShowError( msg );
        return false;
    }
}

const char* get_latency_end_mode( ) {
    if ( !check_doc() ) return "";

    const char *mode = "undefined";
    if ( actDoc()->GetLatencyEndMode() == stf::manualMode )
        mode = "manual";
    else if ( actDoc()->GetLatencyEndMode() == stf::peakMode )
        mode = "peak";
    else if ( actDoc()->GetLatencyEndMode() == stf::riseMode )
        mode = "rise";
    else if ( actDoc()->GetLatencyEndMode() == stf::halfMode )
        mode = "half";
    else if ( actDoc()->GetLatencyEndMode() == stf::footMode )
        mode = "foot";

    return mode;
}

bool set_latency_end(double pos, bool is_time) {
    if ( !check_doc() ) return false;

    if (is_time) pos /= actDoc()->GetXScale();

    int posInt = stf::round( pos );

    /* range check */
    if ( posInt < 0 || posInt >= (int)actDoc()->cursec().size() ){
        ShowError( wxT("Value out of range in set_latency_start()") );
        return false;
    }

    actDoc()->SetLatencyEndMode( stf::manualMode );
    actDoc()->SetLatencyEnd( posInt );

    /* set start latency mode to manual and write in registry */
    const wxString myitem = wxT("LatencyEndMode");
    bool dlg_OK, result_OK;
    dlg_OK = update_cursor_dialog();
    result_OK = update_results_table();
    if (dlg_OK && result_OK) {
        write_stf_registry(myitem, stf::manualMode);
        return true;
        }
    return false;

}
bool set_latency_end_mode( const char* mode ) {
    if ( !check_doc() ) return false;

    const wxString myitem = wxT("LatencyEndMode");
    bool dlg_OK, result_OK;

    if ( strcmp( mode, "manual" ) == 0 ) {
        actDoc()->SetLatencyEndMode( stf::manualMode );
        dlg_OK = update_cursor_dialog();
        result_OK = update_results_table();
        if (dlg_OK && result_OK) {
            write_stf_registry(myitem, stf::manualMode);
            return true;
        }
        return false;
    }
    else if ( strcmp( mode, "peak" ) == 0 ) {
        actDoc()->SetLatencyEndMode( stf::peakMode );
        dlg_OK = update_cursor_dialog();
        result_OK = update_results_table();
        if (dlg_OK && result_OK) {
            write_stf_registry(myitem, stf::peakMode);
            return true;
        }
        return false;
    }
    else if ( strcmp( mode, "rise" ) == 0 ) {
        actDoc()->SetLatencyEndMode( stf::riseMode );
        dlg_OK = update_cursor_dialog();
        result_OK = update_results_table();
        if (dlg_OK && result_OK) {
            write_stf_registry(myitem, stf::riseMode);
            return true;
        }
        return false;
    }
    else if ( strcmp( mode, "half" ) == 0 ) {
        actDoc()->SetLatencyEndMode( stf::halfMode );
        dlg_OK = update_cursor_dialog();
        result_OK = update_results_table();
        if (dlg_OK && result_OK) {
            write_stf_registry(myitem, stf::halfMode);
            return true;
        }
        return false;
    }
    else if ( strcmp( mode, "foot" ) == 0 ) {
        actDoc()->SetLatencyEndMode( stf::footMode );
        dlg_OK = update_cursor_dialog();
        result_OK = update_results_table();
        if (dlg_OK && result_OK) {
            write_stf_registry(myitem, stf::footMode);
            return true;
        }
        return false;
    }
    else {
        wxString msg;
        msg << wxT("\"") << wxString::FromAscii(mode) << wxT("\" is not a valid end latency mode\n");
        msg << wxT("Use \"manual\", \"peak\", \"rise\", \"half\" or \"foot\"");
        ShowError( msg );
        return false;
    }

}

double get_latency_start( bool is_time ) {
    if ( !check_doc() ) return -1;

    if ( !is_time )
        return actDoc()->GetLatencyBeg();
    else
        return (double)actDoc()->GetLatencyBeg() * actDoc()->GetXScale();
}

double get_latency_end( bool is_time ) {
    if ( !check_doc() ) return -1;

    if ( !is_time )
        return actDoc()->GetLatencyEnd();
    else
        return (double)actDoc()->GetLatencyEnd() * actDoc()->GetXScale();
}

double get_base_SD() {
    if ( !check_doc() ) return 0.0;

    return actDoc()->GetBaseSD();
}

double get_base_start( bool is_time ) {
    if ( !check_doc() ) return -1;

    if ( !is_time )
        return actDoc()->GetBaseBeg();
    else
        return (double)actDoc()->GetBaseBeg() * actDoc()->GetXScale();
}

bool set_base_start( double pos, bool is_time ) {
    if ( !check_doc() ) return false;

    if ( is_time )
        pos /= actDoc()->GetXScale();

    int posInt = stf::round( pos );

    // range check:
    if ( posInt < 0 || posInt >= (int)actDoc()->cursec().size() ) {
        ShowError( wxT("Value out of range in set_base_start()") );
        return false;
    }

    actDoc()->SetBaseBeg( posInt );

    return update_cursor_dialog();
}

double get_base_end( bool is_time ) {
    if ( !check_doc() ) return -1;

    if ( !is_time )
        return actDoc()->GetBaseEnd();
    else
        return (double)actDoc()->GetBaseEnd() * actDoc()->GetXScale();
}

bool set_base_end( double pos, bool is_time ) {
    if ( !check_doc() ) return false;

    if ( is_time )
        pos /= actDoc()->GetXScale();

    int posInt = stf::round( pos );

    // range check:
    if ( posInt < 0 || posInt >= (int)actDoc()->cursec().size() ) {
        ShowError( wxT("Value out of range in set_base_end()") );
        return false;
    }

    actDoc()->SetBaseEnd( posInt );

    return update_cursor_dialog();
}

bool set_slope(double slope) {
    if ( !check_doc() ) return false;

    actDoc()->SetSlopeForThreshold( slope );
    actDoc()->SetFromBase( false );
    return update_cursor_dialog();
}

double get_sampling_interval( ) {
    if ( !check_doc() ) return -1.0;

    return actDoc()->GetXScale();
}

const char* get_xunits( ) {
    if ( !check_doc() ) return "";

    return actDoc()->GetXUnits().c_str();
}

const char* get_yunits( int trace, int channel ) {
    if ( !check_doc() ) return "";

    if (channel < 0) {
        channel = actDoc()->GetCurChIndex();
    }
    if (trace < 0) {
        trace = actDoc()->GetCurSecIndex();
    }
    try {
        return actDoc()->at( channel ).GetYUnits().c_str();
    }
    catch (const std::out_of_range& e) {
        wxString msg(wxT("Index out of range in get_yunits:\n"));
        msg+=wxString( e.what(), wxConvLocal );
        ShowError( msg );
        return "";
    }
}

bool set_xunits( const char* units ) {
    if ( !check_doc() ) return false;

    actDoc()->SetXUnits(units);
    return true;
}

bool set_yunits( const char* units, int trace, int channel ) {
    if ( !check_doc() ) return false;

    if (channel < 0) {
        channel = actDoc()->GetCurChIndex();
    }
    if (trace < 0) {
        trace = actDoc()->GetCurSecIndex();
    }
    try {
        actDoc()->at( channel ).SetYUnits(units);
    }
    catch (const std::out_of_range& e) {
        wxString msg(wxT("Index out of range in set_yunits:\n"));
        msg+=wxString( e.what(), wxConvLocal );
        ShowError( msg );
        return false;
    }
    return true;
}

bool set_sampling_interval( double si ) {
    if ( !check_doc() ) return false;

    if (si <= 0) {
        ShowError( wxT("New sampling interval needs to be greater than 0.") );
        return false;
    }

    actDoc()->SetXScale( si );

    return refresh_graph();
}

bool measure( ) {
    if ( !check_doc() )
        return false;

    // check cursor positions:
    if ( actDoc()->GetPeakBeg() > actDoc()->GetPeakEnd() ) {
        ShowError( wxT("Peak window cursors are reversed; will abort now.") );
        return false;
    }

    if ( actDoc()->GetBaseBeg() > actDoc()->GetBaseEnd() ) {
        ShowError( wxT("Base window cursors are reversed; will abort now.") );
        return false;
    }

    if ( actDoc()->GetFitBeg() > actDoc()->GetFitEnd() ) {
        ShowError( wxT("Fit window cursors are reversed; will abort now.") );
        return false;
    }

    wxStfChildFrame* pFrame = (wxStfChildFrame*)actDoc()->GetDocumentWindow();
    if ( !pFrame ) {
        ShowError( wxT("Pointer to frame is zero") );
        return false;
    }

    wxGetApp().OnPeakcalcexecMsg();
    pFrame->UpdateResults();
    return true;
}

double get_base( bool active ) {

    if ( !check_doc() ) return -1.0;

    if ( active ) {
        return actDoc()->GetBase();
    } else {
        // Test wheter a second channel is available at all:
        if ( actDoc()->size() < 2) {
            ShowError( wxT("No second channel was found") );
            return -1.0;
        }
        return actDoc()->GetAPBase();
    }
}

#ifdef WITH_PSLOPE
double get_pslope() {

    if (!check_doc() ) return 0.0;

    return actDoc()->GetPSlope();
}
#endif

double get_peak( ) {

    if ( !check_doc() ) return 0.0;

    return actDoc()->GetPeak();

}

void _gMatrix_resize( std::size_t channels, std::size_t sections ) {
    gMatrix.resize( channels );
    std::vector< std::vector< Vector_double > >::iterator it;
    for (it = gMatrix.begin(); it != gMatrix.end(); ++it) {
        it->resize( sections );
    }
}

void _gMatrix_at( double* invec, int size, int channel, int section ) {
    std::vector< double > va(size);
    std::copy( &invec[0], &invec[size], va.begin() );

    try{
        gMatrix.at(channel).at(section).resize( va.size() );
        gMatrix[channel][section] = va;
    }
    catch (const std::out_of_range& e) {
        wxString msg(wxT("Out of range exception in _gMatrix_at:\n"));
        msg+=wxString( e.what(), wxConvLocal );
        ShowError( msg );
        return;
    }
}

void _gNames_resize( std::size_t channels ) {
    gNames.resize( channels );
}

void _gNames_at( const char* name, int channel ) {
    try{
        gNames.at(channel) = std::string(name);
    }
    catch (const std::out_of_range& e) {
        wxString msg(wxT("Out of range exception in _gNames_at:\n"));
        msg+=wxString( e.what(), wxConvLocal );
        ShowError( msg );
        return;
    }
}

void align_selected(  double (*alignment)( bool ), bool active ) {
    if ( !check_doc() ) return;
    wxStfDoc* pDoc = actDoc();

    //store current section:
    std::size_t section_old = pDoc->GetCurSecIndex();

    if ( pDoc->GetSelectedSections().empty() ) {
        ShowError( wxT("No selected traces") );
        return;
    }
    // check for a second channel
    if ( actDoc()->size() < 2 ) {
        ShowError( wxT("No second channel found") );
        return;
    }

    //initialize the lowest and the highest index:
    std::size_t min_index=0;
    try {
        min_index=pDoc->get()[pDoc->GetSecChIndex()].at(pDoc->GetSelectedSections().at(0)).size()-1;
    }
    catch (const std::out_of_range& e) {
        wxString msg(wxT("Error while aligning\nIt is safer to re-start the program\n"));
        msg+=wxString( e.what(), wxConvLocal );
        ShowError( msg );
        return;
    }

    std::size_t max_index=0;
    std::vector<int> shift( pDoc->GetSelectedSections().size(), 0 );
    int_it it = shift.begin();
    //loop through all selected sections:
    for (c_st_it cit = pDoc->GetSelectedSections().begin();
         cit != pDoc->GetSelectedSections().end() && it != shift.end();
         cit++)
    {
        //Set the selected section as the current section temporarily:
        pDoc->SetSection(*cit);
        if ( pDoc->GetPeakAtEnd() ) {
            pDoc->SetPeakEnd((int)pDoc->get()[pDoc->GetSecChIndex()][*cit].size()-1);
        }
        // Calculate all variables for the current settings
        // APMaxSlopeT will be calculated for the second (==reference)
        // channel, so channels may not be changed!
        try {
            pDoc->Measure();
        }
        catch (const std::out_of_range& e) {
            ShowExcept( e );
            return;
        }

        //check whether the current index is a max or a min,
        //and if so, store it:
        double alignIndex = alignment( active );
        *it = stf::round( alignIndex );
        if (alignIndex > max_index) {
            max_index=alignIndex;
        }
        if (alignIndex < min_index) {
            min_index=alignIndex;
        }
        it++;
    }
    // now that max and min indices are known, calculate the number of
    // points that need to be shifted:
    for (int_it it2 = shift.begin(); it2 != shift.end(); it2++) {
        (*it2) -= (int)min_index;
    }
    //restore section and channel settings:
    pDoc->SetSection( section_old );

    int new_size=(int)(pDoc->get()[0][pDoc->GetSelectedSections()[0]].size()-(max_index-min_index));

    Recording Aligned( pDoc->size(), pDoc->GetSelectedSections().size(), new_size );

    ch_it chan_it;       // Channel iterator
    std::size_t n_ch = 0;
    for ( ch_it chan_it = pDoc->get().begin();
          chan_it != pDoc->get().end();
          ++chan_it )
    {
        Channel ch( pDoc->GetSelectedSections().size() );
        ch.SetChannelName( pDoc->at(n_ch).GetChannelName() );
        ch.SetYUnits(  pDoc->at(n_ch).GetYUnits() );
        std::size_t n_sec = 0;
        int_it it3 = shift.begin();
        for ( c_st_it sel_it = pDoc->GetSelectedSections().begin();
              sel_it != pDoc->GetSelectedSections().end() && it3 != shift.end();
              ++sel_it )
        {
            Vector_double va( new_size );
            std::copy( &(chan_it->at( *sel_it ).get_w()[ 0 + (*it3) ]),
                       &(chan_it->at( *sel_it ).get_w()[ (*it3) + new_size ]),
                       &va[0] );
            Section sec(va);
            ch.InsertSection(sec, n_sec++);
            ++it3;
        }
        Aligned.InsertChannel( ch, n_ch++ );
    }

    wxString title( pDoc->GetTitle() );
    title += wxT(", aligned");
    Aligned.CopyAttributes( *pDoc );
    wxStfDoc* testDoc = wxGetApp().NewChild(Aligned, pDoc, title);
    if ( testDoc == NULL ) {
        ShowError( wxT("Failed to create a new window.") );
    }

    return;
}

int leastsq_param_size( int fselect ) {
    int npar = 0;
    try {
        npar = (int)wxGetApp().GetFuncLib().at(fselect).pInfo.size();
    }
    catch (const std::out_of_range& e) {
        wxString msg( wxT("Could not retrieve function from library:\n") );
        msg << wxString( e.what(), wxConvLocal );
        ShowError(msg);
        return -1;
    }
    return npar;
}

PyObject* leastsq( int fselect, bool refresh ) {
    if ( !check_doc() ) return NULL;

    wxStfDoc* pDoc = actDoc();

    wxCommandEvent wce;

    int n_params = 0;
    try {
        n_params=(int)wxGetApp().GetFuncLib().at(fselect).pInfo.size();
    }
    catch (const std::out_of_range& e) {
        wxString msg( wxT("Could not retrieve function from library:\n") );
        msg << wxString( e.what(), wxConvLocal );
        ShowError(msg);
        return NULL;
    }

    std::vector< double > x( pDoc->GetFitEnd() - pDoc->GetFitBeg() );
    //fill array:
    std::copy(&pDoc->cursec()[pDoc->GetFitBeg()], &pDoc->cursec()[pDoc->GetFitEnd()], &x[0]);

    std::vector< double > params( n_params );

    // initialize parameters from init function,
    wxGetApp().GetFuncLib().at(fselect).init( x, pDoc->GetBase(), pDoc->GetPeak(),
            pDoc->GetRTLoHi(), pDoc->GetHalfDuration(), pDoc->GetXScale(), params );
    std::string fitInfo;
    int fitWarning = 0;
    std::vector< double > opts( 6 );
    // check values in src/stimfit/gui/dlgs/fitseldlg.cpp
    // Respectively the scale factor for initial damping term \mu,
    // stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2,
    // maxIter, maxPass
    //opts[0]=5*1E-3; //default: 1E-03;
    opts[0] = 1E-05; //default: 1E-03;
    opts[1] = 1E-17; //default: 1E-17;
    opts[2] = 1E-17; //default: 1E-17;
    opts[3] = 1E-32; //default: 1E-17;
    opts[4] = 64; //default: 64;
    opts[5] = 16;
    double chisqr = 0.0;
    try {
        chisqr = stfnum::lmFit( x, pDoc->GetXScale(), wxGetApp().GetFuncLib().at(fselect),
                             opts, true, params, fitInfo, fitWarning );
        pDoc->SetIsFitted( pDoc->GetCurChIndex(), pDoc->GetCurSecIndex(), params,
                           wxGetApp().GetFuncLibPtr(fselect),
                           chisqr, pDoc->GetFitBeg(), pDoc->GetFitEnd() );
    }

    catch (const std::out_of_range& e) {
        ShowExcept( e );
        return NULL;
    }
    catch (const std::runtime_error& e) {
        ShowExcept( e );
        return NULL;
    }
    catch (const std::exception& e) {
        ShowExcept( e );
        return NULL;
    }

    if ( refresh ) {
        if ( !refresh_graph() ) return NULL;
    }

    // Dictionaries apparently grow as needed; no initial size is required.
    PyObject* retDict = PyDict_New( );
    for ( std::size_t n_dict = 0; n_dict < params.size(); ++n_dict ) {
         PyDict_SetItemString( retDict, wxGetApp().GetFuncLib()[fselect].pInfo.at(n_dict).desc.c_str(),
                PyFloat_FromDouble( params[n_dict] ) );
    }
    PyDict_SetItemString( retDict, "SSE", PyFloat_FromDouble( chisqr ) );

    return retDict;
}

#ifdef WITH_PYTHON
PyObject* get_fit( int trace, int channel ) {
    wrap_array();

    if ( !check_doc() ) return NULL;

    if ( trace == -1 ) {
        trace = actDoc()->GetCurSecIndex();
    }
    if ( channel == -1 ) {
        channel = actDoc()->GetCurChIndex();
    }

    /* Does the specified section contain a fit at all? */
    stf::SectionAttributes sec_attr;
    try {
        sec_attr = actDoc()->GetSectionAttributes(channel, trace);
    } catch (const std::out_of_range& e) {
        return NULL;
    }
    if (!sec_attr.isFitted) {
        return Py_None;
    }

    unsigned int size = sec_attr.storeFitEnd - sec_attr.storeFitBeg;
    Vector_double xy_fit(2*size);
    for (unsigned int x = 0; x < size; ++x) {
        xy_fit[x] = (x+sec_attr.storeFitBeg) * actDoc()->GetXScale();
        xy_fit[x+size] = sec_attr.fitFunc->func(x*actDoc()->GetXScale(), sec_attr.bestFitP);
    }

    npy_intp dims[2] = {2, size};
    PyObject* np_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    double* gDataP = (double*)PyArray_DATA((PyArrayObject*)np_array);

    /* fill */
    std::copy(xy_fit.begin(), xy_fit.end(), gDataP);

    return np_array;
}
#endif

bool show_table( PyObject* dict, const char* caption ) {
    if ( !check_doc() ) return false;

    // Check whether the dictionary is intact:
    if ( !PyDict_Check( dict ) ) {
        ShowError( wxT("First argument to ShowTable() is not a dictionary.") );
        return false;
    }
    std::map< std::string, double > pyMap;
    Py_ssize_t n_dict = 0;
    PyObject *pkey = NULL, *pvalue = NULL;
    while ( PyDict_Next( dict, &n_dict, &pkey, &pvalue ) ) {
        if ( !pkey || !pvalue ) {
            ShowError( wxT("Couldn't read from dictionary in ShowTable().") );
            return false;
        }
        std::string key = PyString_AsString( pkey );
        double value = PyFloat_AsDouble( pvalue );
        pyMap[key] = value;
    }
    stfnum::Table pyTable( pyMap );

    wxStfChildFrame* pFrame = (wxStfChildFrame*)actDoc()->GetDocumentWindow();
    if ( !pFrame ) {
        ShowError( wxT("Pointer to frame is zero") );
        return false;
    }
    pFrame->ShowTable( pyTable, wxString( caption, wxConvLocal ) );
    return true;
}

bool show_table_dictlist( PyObject* dict, const char* caption, bool reverse ) {
    if ( !check_doc() ) return false;

    if ( !reverse ) {
        ShowError( wxT("Row-major order (reverse = False) has not been implemented yet.") );
        return false;
    }

    // Check whether the dictionary is intact:
    if ( !PyDict_Check( dict ) ) {
        ShowError( wxT("First argument to ShowTable() is not a dictionary.") );
        return false;
    }
    Py_ssize_t n_dict = 0;
    PyObject *pkey = NULL, *pvalue = NULL;
    std::vector< Vector_double > pyVector;
    std::vector< std::string > pyStrings;
    while ( PyDict_Next( dict, &n_dict, &pkey, &pvalue ) ) {
        if ( !pkey || !pvalue ) {
            ShowError( wxT("Couldn't read from dictionary in ShowTable().") );
            return false;
        }
        pyStrings.push_back(PyString_AsString( pkey ));
        if ( !PyList_Check( pvalue ) ) {
            ShowError( wxT("Dictionary values are not (consistently) lists.") );
            return false;
        }
        Vector_double values( PyList_Size( pvalue ) );
        for (int n_list = 0; n_list < (int)values.size(); ++n_list ) {
            PyObject* plistvalue = PyList_GetItem( pvalue, n_list );
            if ( !plistvalue ) {
                ShowError( wxT("Can't read list elements in show_table().") );
                return false;
            }
            values[n_list] = PyFloat_AsDouble( plistvalue );
        }
        pyVector.push_back( values );
    }
    if ( pyVector.empty() ) {
        ShowError( wxT("Dictionary was empty in show_table().") );
        return false;
    }
    stfnum::Table pyTable( pyVector[0].size(), pyVector.size() );
    std::vector< std::vector< double > >::const_iterator c_va_it;
    std::size_t n_col = 0;
    for (  c_va_it = pyVector.begin(); c_va_it != pyVector.end(); ++c_va_it ) {
        try {
            pyTable.SetColLabel( n_col, pyStrings[n_col] );
            for ( std::size_t n_va=0; n_va < (*c_va_it).size(); ++n_va ) {
                pyTable.at( n_va, n_col ) = (*c_va_it)[n_va];
            }
        }
        catch ( const std::out_of_range& e ) {
            ShowExcept( e );
            return false;
        }
        n_col++;
    }
    wxStfChildFrame* pFrame = (wxStfChildFrame*)actDoc()->GetDocumentWindow();
    if ( !pFrame ) {
        ShowError( wxT("Pointer to frame is zero") );
        return false;
    }
    pFrame->ShowTable( pyTable, wxString( caption, wxConvLocal ) );
    return true;
}

bool set_marker(double x, double y) {
    if ( !check_doc() )
        return false;
    try {
        actDoc()->GetCurrentSectionAttributesW().pyMarkers.
            push_back(stf::PyMarker(x,y));
    }
    catch (const std::out_of_range& e) {
        wxString msg( wxT("Could not set the marker:\n") );
        msg << wxString( e.what(), wxConvLocal );
        ShowError(msg);
        return false;
    }

    return refresh_graph();
}

bool erase_markers() {
    if ( !check_doc() )
        return false;

    try {
        actDoc()->GetCurrentSectionAttributesW().pyMarkers.clear();
    }
    catch (const std::out_of_range& e) {
        wxString msg( wxT("Could not erase markers:\n") );
        msg << wxString( e.what(), wxConvLocal );
        ShowError(msg);
        return false;
    }

    return refresh_graph();
}

double plot_xmin() {
    wxStfGraph* pGraph = actGraph();
    if ( !pGraph ) {
        ShowError( wxT("Pointer to graph is zero") );
        return 0;
    }

    return pGraph->get_plot_xmin();
}

double plot_xmax() {
    wxStfGraph* pGraph = actGraph();
    if ( !pGraph ) {
        ShowError( wxT("Pointer to graph is zero") );
        return 0;
    }

    return pGraph->get_plot_xmax();
}

double plot_ymin() {
    wxStfGraph* pGraph = actGraph();
    if ( !pGraph ) {
        ShowError( wxT("Pointer to graph is zero") );
        return 0;
    }

    return pGraph->get_plot_ymin();
}

double plot_ymax() {
    wxStfGraph* pGraph = actGraph();
    if ( !pGraph ) {
        ShowError( wxT("Pointer to graph is zero") );
        return 0;
    }

    return pGraph->get_plot_ymax();
}

double plot_y2min() {
    wxStfGraph* pGraph = actGraph();
    if ( !pGraph ) {
        ShowError( wxT("Pointer to graph is zero") );
        return 0;
    }

    return pGraph->get_plot_y2min();
}

double plot_y2max() {
    wxStfGraph* pGraph = actGraph();
    if ( !pGraph ) {
        ShowError( wxT("Pointer to graph is zero") );
        return 0;
    }

    return pGraph->get_plot_y2max();
}

PyObject* mpl_panel(const std::vector<double>& figsize) {
    if ( !check_doc() ) return NULL;

    if (figsize.size() < 2) {
        ShowError( wxT("figsize has to have length 2") );
    }

    wxStfParentFrame* parent = wxGetApp().GetMainFrame();
    if ( !parent ) {
        ShowError( wxT("Parent window is NULL") );
        return 0;
    }

    std::ostringstream mpl_name;
    mpl_name << "mpl" << parent->GetMplFigNo();

    int width = 800 * figsize[0]/8.0;
    int height = 600 * figsize[1]/6.0;
    PyObject* result = parent->MakePythonWindow("makeWindowMpl", mpl_name.str(), "Matplotlib",
                                                true, false, true, width, height, figsize[0], figsize[1]).pyWindow;

    return result;
}
