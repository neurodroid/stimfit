#ifndef _STFSWIG_H
#define _STFSWIG_H

void _get_trace_fixedsize( double* outvec, int size, int trace = -1, int channel = -1 );
void new_window( double* invec, int size );
void new_window_matrix( double* inarr, int traces, int size );
bool new_window_selected_this( );
bool new_window_selected_all( );
bool show_table( PyObject* dict, const char* caption = "Python table" );
bool show_table_dictlist( PyObject* dict, const char* caption  = "Python table", bool reverse = true );

int get_size_trace( int trace = -1, int channel = -1 );
int get_size_channel( int channel = -1 );
double get_sampling_interval( );
bool set_sampling_interval( double si );
const char* get_recording_time( );
const char* get_recording_date( );

bool select_trace( int trace = -1 );
void select_all( );
void unselect_all( );
PyObject* get_selected_indices( );

bool set_trace( int trace );
int get_trace_index();
int get_channel_index( bool active = true );

void align_selected(  double (*alignment)( bool ), bool active = false );

bool subtract_base( );

int leastsq_param_size( int fselect );
PyObject* leastsq( int fselect, bool refresh = true );

bool check_doc( );
bool file_open( const char* filename );
bool file_save( const char* filename );
bool close_all( );
bool close_this( );

bool measure( );

double get_base( );
double get_peak( );

double peak_index( bool active = true );
double maxrise_index( bool active = true );
double foot_index( bool active = true );
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
bool set_peak_mean( int pts );
bool set_peak_direction( const char* direction );

double get_base_start( bool is_time = false );
bool set_base_start( double pos, bool is_time = false );
double get_base_end( bool is_time = false );
bool set_base_end( double pos, bool is_time = false );

void _gVector_resize( std::size_t size );
void _gVector_at( double* invec, int size, int at );
void _new_window_gVector( );

#endif
