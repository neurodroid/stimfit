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

/*! \file stfio.cpp
 *  \author Christoph Schmidt-Hieber
 *  \date 2011-09-25
 *  \brief General functions for libstfio
 * 
 * 
 *  Implements some general functions for libstfio
 */

#include <sstream>

#include "stfio.h"

// TODO #include "./ascii/asciilib.h"
#include "./cfs/cfslib.h"
#include "./hdf5/hdf5lib.h"
#include "./abf/abflib.h"
#include "./atf/atflib.h"
#include "./axg/axglib.h"
#include "./heka/hekalib.h"
#include "./igor/igorlib.h"
#ifdef WITH_BIOSIG
#include "./biosig/biosiglib.h"
#endif
#if 0
#include "./son/sonlib.h"
#endif


stfio::filetype
stfio::findType(const std::string& ext) {
    
    if (ext=="*.dat;*.cfs") return stfio::cfs;
    else if (ext=="*.abf") return stfio::abf;
    else if (ext=="*.axgd;*.axgx") return stfio::axg;
    else if (ext=="*.h5") return stfio::hdf5;
    else if (ext=="*.atf") return stfio::atf;
    else if (ext=="*.dat") return stfio::heka;
    else if (ext=="*.smr") return stfio::son;

#ifdef WITH_BIOSIG
    else if (ext=="*.bs") return stfio::biosig;
#endif
    else return stfio::none;
}

bool stfio::importFile(
        const std::string& fName,
        stfio::filetype type,
        Recording& ReturnData,
        const stfio::txtImportSettings& txtImport,
        ProgressInfo& progDlg
) {
    try {
        switch (type) {
        case stfio::cfs: {
            int res = stfio::importCFSFile(fName, ReturnData, progDlg);
            if (res==-7) {
                stfio::importHEKAFile(fName, ReturnData, progDlg);
            }
            break;
        }
        case stfio::hdf5: {
            stfio::importHDF5File(fName, ReturnData, progDlg);
            break;
        }
        case stfio::abf: {
            stfio::importABFFile(fName, ReturnData, progDlg);
            break;
        }
        case stfio::atf: {
            stfio::importATFFile(fName, ReturnData, progDlg);
            break;
        }
        case stfio::axg: {
            stfio::importAXGFile(fName, ReturnData, progDlg);
            break;
        }
        case stfio::heka: {
            stfio::importHEKAFile(fName, ReturnData, progDlg);
            break;
        }
#ifdef WITH_BIOSIG
        case stfio::biosig: {
            stfio::importBSFile(fName, ReturnData, progDlg);
            break;
        }
#endif

#if 0
        case stfio::son: {
            stfio::SON::importSONFile(fName,ReturnData);
            break;
        }
        case stfio::ascii: {
            stfio::importASCIIFile( fName, txtImport.hLines, txtImport.ncolumns,
                    txtImport.firstIsTime, txtImport.toSection, ReturnData );
            if (!txtImport.firstIsTime) {
                ReturnData.SetXScale(1.0/txtImport.sr);
            }
            if (ReturnData.size()>0)
                ReturnData[0].SetYUnits(txtImport.yUnits);
            if (ReturnData.size()>1)
                ReturnData[1].SetYUnits(txtImport.yUnitsCh2);
            ReturnData.SetXUnits(txtImport.xUnits);
            break;
        }
#endif

        default:
            throw std::runtime_error("Unknown file type");
        }
    }
    catch (...) {
        throw;
    }
    return true;
}

bool stfio::exportFile(const std::string& fName, stfio::filetype type, const Recording& Data,
                       ProgressInfo& progDlg)
{
    try {
        switch (type) {
        case stfio::hdf5: {
            stfio::exportHDF5File(fName, Data, progDlg);
            break;
        }
        case stfio::igor: {
            stfio::exportIGORFile(fName, Data, progDlg);
            break;
        }
        default:
            throw std::runtime_error("Only hdf5 and IGOR are supported for writing at present.");
        }
    }
    catch (...) {
        throw;
    }
    return true;
}

Vector_double stfio::vec_scal_plus(const Vector_double& vec, double scalar) {
    Vector_double ret_vec(vec.size(), scalar);
    std::transform(vec.begin(), vec.end(), ret_vec.begin(), ret_vec.begin(), std::plus<double>());
    return ret_vec;
}

Vector_double stfio::vec_scal_minus(const Vector_double& vec, double scalar) {
    Vector_double ret_vec(vec.size(), scalar);
    std::transform(vec.begin(), vec.end(), ret_vec.begin(), ret_vec.begin(), std::minus<double>());
    return ret_vec;
}

Vector_double stfio::vec_scal_mul(const Vector_double& vec, double scalar) {
    Vector_double ret_vec(vec.size(), scalar);
    std::transform(vec.begin(), vec.end(), ret_vec.begin(), ret_vec.begin(), std::multiplies<double>());
    return ret_vec;
}

Vector_double stfio::vec_scal_div(const Vector_double& vec, double scalar) {
    Vector_double ret_vec(vec.size(), scalar);
    std::transform(vec.begin(), vec.end(), ret_vec.begin(), ret_vec.begin(), std::divides<double>());
    return ret_vec;
}

Vector_double stfio::vec_vec_plus(const Vector_double& vec1, const Vector_double& vec2) {
    Vector_double ret_vec(vec1.size());
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), ret_vec.begin(), std::plus<double>());
    return ret_vec;
}

Vector_double stfio::vec_vec_minus(const Vector_double& vec1, const Vector_double& vec2) {
    Vector_double ret_vec(vec1.size());
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), ret_vec.begin(), std::minus<double>());
    return ret_vec;
}

Vector_double stfio::vec_vec_mul(const Vector_double& vec1, const Vector_double& vec2) {
    Vector_double ret_vec(vec1.size());
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), ret_vec.begin(), std::multiplies<double>());
    return ret_vec;
}

Vector_double stfio::vec_vec_div(const Vector_double& vec1, const Vector_double& vec2) {
    Vector_double ret_vec(vec1.size());
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), ret_vec.begin(), std::divides<double>());
    return ret_vec;
}
