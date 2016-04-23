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

// Copyright 2012 Alois Schloegl, IST Austria <alois.schloegl@ist.ac.at>


#include <sstream>

#include "stfio.h"

// TODO #include "./ascii/asciilib.h"
#include "./hdf5/hdf5lib.h"
#include "./abf/abflib.h"
#include "./atf/atflib.h"
#include "./axg/axglib.h"
#include "./tdms/tdmslib.h"
#include "./igor/igorlib.h"
#if (defined(WITH_BIOSIG) || defined(WITH_BIOSIG2))
  #include "./biosig/biosiglib.h"
#endif
#include "./cfs/cfslib.h"
#ifndef TEST_MINIMAL
  #include "./heka/hekalib.h"
#else
  #if (!defined(WITH_BIOSIG) && !defined(WITH_BIOSIG2))
    #error -DTEST_MINIMAL requires -DWITH_BIOSIG or -DWITH_BIOSIG2
  #endif
#endif
#if 0
#include "./son/sonlib.h"
#endif

#ifdef _MSC_VER
    StfioDll long int lround(double x) {
        int i = (long int) x;
        if (x >= 0.0) {
            return ((x-i) >= 0.5) ? (i + 1) : (i);
        } else {
            return (-x+i >= 0.5) ? (i - 1) : (i);
        }
    }
#endif

stfio::StdoutProgressInfo::StdoutProgressInfo(const std::string& title, const std::string& message, int maximum, bool verbose)
    : ProgressInfo(title, message, maximum, verbose),
      verbosity(verbose)
{
    if (verbosity) {
        std::cout << title << std::endl;
        std::cout << message << std::endl;
    }
}

bool stfio::StdoutProgressInfo::Update(int value, const std::string& newmsg, bool* skip) {
    if (verbosity) {
        std::cout << "\r";
        std::cout.width(3);
        std::cout << value << "% " << newmsg
                  << std::flush;
    }
    return true;
}

#ifndef TEST_MINIMAL
stfio::filetype
stfio::findType(const std::string& ext) {
    
    if (ext=="*.dat;*.cfs") return stfio::cfs;
    else if (ext=="*.cfs") return stfio::cfs;
    else if (ext=="*.abf") return stfio::abf;
    else if (ext=="*.axgd") return stfio::axg;
    else if (ext=="*.axgx") return stfio::axg;
    else if (ext=="*.axgd;*.axgx") return stfio::axg;
    else if (ext=="*.h5")  return stfio::hdf5;
    else if (ext=="*.atf") return stfio::atf;
    else if (ext=="*.dat") return stfio::heka;
    else if (ext=="*.smr") return stfio::son;
    else if (ext=="*.tdms") return stfio::tdms;
#if (defined(WITH_BIOSIG) || defined(WITH_BIOSIG2))
    else if (ext=="*.dat;*.cfs;*.gdf;*.ibw") return stfio::biosig;
    else if (ext=="*.*")   return stfio::biosig;
#endif
    else return stfio::none;
}
#endif // TEST_MINIMAL

std::string
stfio::findExtension(stfio::filetype ftype) {

    switch (ftype) {
     case stfio::cfs:
         return ".dat";
     case stfio::abf:
         return ".abf";
     case stfio::axg:
         return ".axg*";
     case stfio::igor:
         return ".ibw";
     case stfio::hdf5:
         return ".h5";
     case stfio::atf:
         return ".atf";
     case stfio::heka:
         return ".dat";
     case stfio::son:
         return ".smr";
     case stfio::tdms:
         return ".tdms";
#if (defined(WITH_BIOSIG) || defined(WITH_BIOSIG2))
     case stfio::biosig:
         return ".gdf";
#endif
     default:
         return ".*";
    }
}

bool stfio::importFile(
        const std::string& fName,
        stfio::filetype type,
        Recording& ReturnData,
        const stfio::txtImportSettings& txtImport,
        ProgressInfo& progDlg
) {
    try {

#if (defined(WITH_BIOSIG) || defined(WITH_BIOSIG2))
       // make use of automated file type identification

#ifndef WITHOUT_ABF
        if (!check_biosig_version(1,6,3)) {
            try {
                // workaround for older versions of libbiosig
                stfio::importABFFile(fName, ReturnData, progDlg);
                return true;
            }
            catch (...) {
#ifndef NDEBUG
                fprintf(stdout,"%s (line %i): importABF attempted\n",__FILE__,__LINE__);
#endif
            };
       }
#endif // WITHOUT_ABF

       // if this point is reached, import ABF was not applied or not successful
        try {
            stfio::filetype type1 = stfio::importBiosigFile(fName, ReturnData, progDlg);
            switch (type1) {
            case stfio::biosig:
                return true;    // succeeded
            case stfio::none:
                break;          // do nothing, use input argument for deciding on type
            default:
                type = type1;   // filetype is recognized and should be used below
            }
        }
        catch (...) {
                // this should never occur, importBiosigFile should always return without exception
                std::cout << "importBiosigFile failed with an exception - this is a bug";
        }
#endif

        switch (type) {
        case stfio::hdf5: {
            stfio::importHDF5File(fName, ReturnData, progDlg);
            break;
        }
        case stfio::tdms: {
            stfio::importTDMSFile(fName, ReturnData, progDlg);
            break;
        }
#ifndef WITHOUT_ABF
        case stfio::abf: {
            stfio::importABFFile(fName, ReturnData, progDlg);
            break;
        }
        case stfio::atf: {
            stfio::importATFFile(fName, ReturnData, progDlg);
            break;
        }
#endif
#ifndef WITHOUT_AXG
        case stfio::axg: {
            stfio::importAXGFile(fName, ReturnData, progDlg);
            break;
        }
#endif

#ifndef TEST_MINIMAL
        case stfio::cfs: {
            {
            int res = stfio::importCFSFile(fName, ReturnData, progDlg);
         /*
            // disable old Heka import - its broken and will not be fixed, use biosig instead
            if (res==-7) {
                stfio::importHEKAFile(fName, ReturnData, progDlg);
            }
         */
          break;
            }
        }
        /*
	// disable old Heka import - its broken and will not be fixed, use biosig instead
        case stfio::heka: {
            {
                try {
                    stfio::importHEKAFile(fName, ReturnData, progDlg);
                } catch (const std::runtime_error& e) {
                    stfio::importCFSFile(fName, ReturnData, progDlg);
                }
                break;
            }
        }
        */
#endif // TEST_MINIMAL

        default:
            throw std::runtime_error("Unknown or unsupported file type");
	}

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
#ifndef WITHOUT_ABF
        case stfio::atf: {
            stfio::exportATFFile(fName, Data);
            break;
        }
#endif
#if (defined(WITH_BIOSIG) || defined(WITH_BIOSIG2))
        case stfio::biosig: {
            stfio::exportBiosigFile(fName, Data, progDlg);
            break;
        }
#endif
        case stfio::cfs: {
            stfio::exportCFSFile(fName, Data, progDlg);
            break;
        }
        case stfio::hdf5: {
            stfio::exportHDF5File(fName, Data, progDlg);
            break;
        }
        case stfio::igor: {
            stfio::exportIGORFile(fName, Data, progDlg);
            break;
        }
        default:
            throw std::runtime_error("Trying to write an unsupported dataformat.");
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

Recording
stfio::concatenate(const Recording& src, const std::vector<std::size_t>& sections,
                   ProgressInfo& progDlg)
{
    size_t nc, NC = src.size();
    Recording Concatenated(NC, 1);

    for (nc = 0; nc < NC; nc++) {
        int new_size=0;
        for (c_st_it cit = sections.begin(); cit != sections.end(); cit++) {
            new_size += (int)src[nc][*cit].size();
        }
        Section TempSection(new_size);
        std::size_t n_new=0;
        std::size_t n_s=0;
        for (c_st_it cit = sections.begin(); cit != sections.end(); cit++) {
            std::ostringstream progStr;
            progStr << "Adding section #" << (int)n_s+1 << " of " << (int)sections.size();
            progDlg.Update(
                (int)((double)n_s/(double)sections.size()*100.0),
                progStr.str()
            );

            if (cit == sections.begin()) {
                TempSection.SetXScale(src[nc][*cit].GetXScale());
            }
            else if (TempSection.GetXScale() != src[nc][*cit].GetXScale()) {
                Concatenated.resize(0);
                throw std::runtime_error("can not concatanate because sampling frequency differs");
            }

            std::size_t secSize=src[nc][*cit].size();
            if (n_new+secSize>TempSection.size()) {
                Concatenated.resize(0);
                throw std::runtime_error("memory allocation error");
            }
            std::copy(src[nc][*cit].get().begin(),
                      src[nc][*cit].get().end(),
                      &TempSection[n_new]);
            n_new += secSize;
            n_s++;
        }
        TempSection.SetSectionDescription(src[nc][0].GetSectionDescription() + ", concatenated");
        Channel TempChannel(TempSection);
	TempChannel.SetChannelName(src[nc].GetChannelName());
	TempChannel.SetYUnits(src[nc].GetYUnits());
	Concatenated.InsertChannel(TempChannel, nc);
    }

    // Recording Concatenated(TempChannel);
    Concatenated.CopyAttributes(src);

    return Concatenated;
}

Recording
stfio::multiply(const Recording& src, const std::vector<std::size_t>& sections,
                std::size_t channel, double factor)
{
    Channel TempChannel(sections.size(), src[channel][sections[0]].size());
    std::size_t n = 0;
    for (c_st_it cit = sections.begin(); cit != sections.end(); cit++) {
        // Multiply the valarray in Data:
        Section TempSection(stfio::vec_scal_mul(src[channel][*cit].get(),factor));
        TempSection.SetXScale(src[channel][*cit].GetXScale());
        TempSection.SetSectionDescription(
                src[channel][*cit].GetSectionDescription()+
                ", multiplied"
        );
        try {
            TempChannel.InsertSection(TempSection, n);
        }
        catch (const std::out_of_range e) {
            throw e;
        }
        n++;
    }
    if (TempChannel.size()>0) {
        Recording Multiplied(TempChannel);
        Multiplied.CopyAttributes(src);
        Multiplied[0].SetYUnits( src.at( channel ).GetYUnits() );
        return Multiplied;
    } else {
        throw std::runtime_error("Channel empty in stfio::multiply");
    }
}
