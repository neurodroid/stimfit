#include <stdlib.h>
#include <math.h>

#if 0 //def _WINDOWS
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
#include <Python.h>
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

#include <iostream>
#include <iomanip>

#if 0//def _WINDOWS
    #ifdef _UNDEBUG
        #define _DEBUG
    #endif
#endif

#include "./../libstfnum/fit.h"

#include "pystfio.h"

#if PY_MAJOR_VERSION >= 3
int wrap_array() {
    import_array();
    return 0;
}
#else
void wrap_array() {
    import_array();
}
#endif

stfio::filetype gettype(const std::string& ftype) {
    stfio::filetype stftype = stfio::none;
    if (ftype == "cfs") {
        stftype = stfio::cfs;
    } else if (ftype == "hdf5") {
        stftype = stfio::hdf5;
    } else if (ftype == "abf") {
        stftype = stfio::abf;
    } else if (ftype == "atf") {
        stftype = stfio::atf;
    } else if (ftype == "axg") {
        stftype = stfio::axg;
    } else if (ftype == "biosig") {
        stftype = stfio::biosig;
    } else if (ftype == "gdf") {
        stftype = stfio::biosig;
    } else if (ftype == "heka") {
        stftype = stfio::heka;
    } else if (ftype == "igor") {
        stftype = stfio::igor;
    } else {
        stftype = stfio::none;
    }
    return stftype;
}

bool _read(const std::string& filename, const std::string& ftype, bool verbose, Recording& Data) {

#ifndef TEST_MINIMAL
    stfio::filetype stftype = gettype(ftype);
#else
    const stfio::filetype stftype = stfio::none;
#endif // TEST_MINIMAL

    stfio::txtImportSettings tis;
    stfio::StdoutProgressInfo progDlg("File import", "Starting file import", 100, verbose);
    
    try {
        if (!stfio::importFile(filename, stftype, Data, tis, progDlg)) {
            std::cerr << "Error importing file\n";
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error importing file:\n"
                  << e.what() << std::endl;
        return false;
    }
        
    return true;
}

PyObject* detect_events(double* data, int size_data, double* templ, int size_templ,
                        double dt, const std::string& mode, bool norm, double lowpass, double highpass)
{
    wrap_array();

    Vector_double vtempl(templ, &templ[size_templ]);
    if (norm) {
        double fmin = *std::min_element(vtempl.begin(), vtempl.end());
        double fmax = *std::max_element(vtempl.begin(), vtempl.end());
        double basel = 0;
        double normval = 1.0;
        if (fabs(fmin) > fabs(fmax)) {
            basel = fmax;
        } else {
            basel = fmin;
        }
        vtempl = stfio::vec_scal_minus(vtempl, basel);
        fmin = *std::min_element(vtempl.begin(), vtempl.end());
        fmax = *std::max_element(vtempl.begin(), vtempl.end());
        if (fabs(fmin) > fabs(fmax)) {
            normval = fabs(fmin);
        } else {
            normval = fabs(fmax);
        }
        vtempl = stfio::vec_scal_div(vtempl, normval);
    }
    Vector_double trace(data, &data[size_data]);
    Vector_double detect(size_data);
    if (mode=="criterion") {
        stfio::StdoutProgressInfo progDlg("Computing detection criterion...", "Computing detection criterion...", 100, true);
        detect = stfnum::detectionCriterion(trace, vtempl, progDlg);
    } else if (mode=="correlation") {
        stfio::StdoutProgressInfo progDlg("Computing linear correlation...", "Computing linear correlation...", 100, true);
        detect = stfnum::linCorr(trace, vtempl, progDlg);
    } else if (mode=="deconvolution") {
        stfio::StdoutProgressInfo progDlg("Computing detection criterion...", "Computing detection criterion...", 100, true);
        detect = stfnum::deconvolve(trace, vtempl, 1.0/dt, highpass, lowpass, progDlg);
    }
    npy_intp dims[1] = {(int)detect.size()};
    PyObject* np_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double* gDataP = (double*)array_data(np_array);

    /* fill */
    std::copy(detect.begin(), detect.end(), gDataP);
    
    return np_array;
}

PyObject* peak_detection(double* invec, int size, double threshold, int min_distance) {
    wrap_array();

    Vector_double data(invec, &invec[size]);

    std::vector<int> peak_idcs = stfnum::peakIndices(data, threshold, min_distance);

    npy_intp dims[1] = {(int)peak_idcs.size()};
    PyObject* np_array = PyArray_SimpleNew(1, dims, NPY_INT);
    if (sizeof(int) == 4) {
        int* gDataP = (int*)array_data(np_array);
        /* fill */
        std::copy(peak_idcs.begin(), peak_idcs.end(), gDataP);
    
        return np_array;
    } else if (sizeof(short) == 4) {
        short* gDataP = (short*)array_data(np_array);
        
        /* fill */
        std::copy(peak_idcs.begin(), peak_idcs.end(), gDataP);
    
        return np_array;
    } else {
        std::cerr << "Couldn't find 4-byte integer type\n";
        return NULL;
    }
        
}
