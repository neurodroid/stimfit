#ifndef _PYSTF_H
#define _PYSTF_H
#undef _DEBUG

#include <wx/wx.h>
#undef _DEBUG

std::string get_versionstring( );

PyObject* get_trace(int trace=-1, int channel=-1);

bool new_window( double* invec, int size );
bool new_window_matrix( double* inarr, int traces, int size );
bool new_window_selected_this( );
bool new_window_selected_all( );
bool show_table( PyObject* dict, const char* caption = "Python table" );
bool show_table_dictlist( PyObject* dict, const char* caption  = "Python table", bool reverse = true );

int get_size_trace( int trace = -1, int channel = -1 );
int get_size_channel( int channel = -1 );
int get_size_recording( );

double get_sampling_interval( );
bool set_sampling_interval( double si );

const char* get_xunits( );
const char* get_yunits( int trace = -1, int channel = -1 );
bool set_xunits( const char* units );
bool set_yunits( const char* units, int trace = -1, int channel = -1 );

double get_maxdecay();
double get_maxrise();

const char* get_recording_time( );
const char* get_recording_date( );
std::string get_recording_comment( );
bool set_recording_comment( const char* comment );
bool set_recording_time( const char* time );
bool set_recording_date( const char* date );

bool select_trace( int trace = -1 );
void select_all( );
void unselect_all( );
PyObject* get_selected_indices( );

bool set_trace( int trace );
int get_trace_index();
const char* get_trace_name( int trace = -1, int channel = -1 );

bool set_channel( int channel);
int get_channel_index( bool active = true );
const char* get_channel_name( int index = -1 );
bool set_channel_name( const char* name, int index = -1 );

void align_selected(  double (*alignment)( bool ), bool active = false );

bool subtract_base( );

int leastsq_param_size( int fselect );
PyObject* leastsq( int fselect, bool refresh = true );

bool check_doc( );
std::string get_filename( );
bool file_open( const char* filename );
bool file_save( const char* filename );
bool close_all( );
bool close_this( );

bool measure( );

double get_base( bool active = true );
double get_peak( );
double get_pslope();
double get_threshold_time( bool is_time = false );
double get_threshold_value( );
double get_latency();
double get_risetime();

double peak_index( bool active = true );
double maxrise_index( bool active = true );
double maxdecay_index( );
double foot_index( bool active = true );
double rtlow_index( bool active = true );
double rthigh_index( bool active = true );
double t50left_index( bool active = true );
double t50right_index( bool active = true );

bool set_marker(double x, double y);
bool erase_markers();

double get_fit_start( bool is_time = false );
bool set_fit_start( double pos, bool is_time = false );
double get_fit_end( bool is_time = false );
bool set_fit_end( double pos, bool is_time = false );

double get_peak_start( bool is_time = false );
bool set_peak_start( double pos, bool is_time = false );
double get_peak_end( bool is_time = false );
bool set_peak_end( double pos, bool is_time = false );
int get_peak_mean( );
bool set_peak_mean( int pts );
const char* get_peak_direction( );
bool set_peak_direction( const char* direction );

double get_base_start( bool is_time = false );
bool set_base_start( double pos, bool is_time = false );
double get_base_end( bool is_time = false );
bool set_base_end( double pos, bool is_time = false );

const char* get_latency_start_mode( );
bool set_latency_start_mode( const char* direction );
const char* get_latency_end_mode( );
bool set_latency_end_mode( const char* direction );
double get_latency_start( bool is_time = false );
double get_latency_end( bool is_time = false );

bool set_slope(double slope);

double plot_xmin();
double plot_xmax();
double plot_ymin();
double plot_ymax();
double plot_y2min();
double plot_y2max();

void _gMatrix_resize( std::size_t channels, std::size_t sections );
void _gNames_resize( std::size_t channels );
void _gMatrix_at( double* invec, int size, int channel, int section );
void _gNames_at( const char* name, int channel );
bool _new_window_gMatrix( );

PyObject* mpl_panel();

PyObject* template_matching(double* invec, int size, bool correlate=false);

#endif
