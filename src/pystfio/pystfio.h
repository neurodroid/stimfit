#ifndef _PYSTFIO_H
#define _PYSTFIO_H

#include "../libstfio/stfio.h"


#include <numpy/arrayobject.h>

#define array_data(a)          PyArray_DATA((PyArrayObject*)a)

#if PY_MAJOR_VERSION >= 3
int
#else
void
#endif
wrap_array();

stfio::filetype gettype(const std::string& ftype);
bool _read(const std::string& filename, const std::string& ftype, bool verbose, Recording& Data);
PyObject* detect_events(double* data, int size_data, double* templ, int size_templ, double dt,
                        const std::string& mode="criterion",
                        bool norm=true, double lowpass=0.5, double highpass=0.0001);
PyObject* peak_detection(double* invec, int size, double threshold, int min_distance);
double risetime(double* invec, int size, double base, double amp, double frac=0.2);

#endif
