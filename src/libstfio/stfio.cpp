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
#include <fstream>
#include <algorithm>
#include <cctype>

#include "stfio.h"

// TODO #include "./ascii/asciilib.h"
#include "./hdf5/hdf5lib.h"
#include "./abf/abflib.h"
#include "./atf/atflib.h"
#include "./axg/axglib.h"
#include "./igor/igorlib.h"
#if (defined(WITH_BIOSIG))
  #include "./biosig/biosiglib.h"
#else
  #include "./heka/hekalib.h"
#endif
#include "./cfs/cfslib.h"
#include "./intan/intanlib.h"

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
    else if (ext=="*.clp") return stfio::intan;
#if defined(WITH_BIOSIG)
    else if (ext=="*.dat;*.cfs;*.gdf;*.ibw;*.wcp") return stfio::biosig;
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
     case stfio::intan:
         return ".clp";
#if defined(WITH_BIOSIG)
     case stfio::biosig:
         return ".gdf";
#endif
     default:
         return ".*";
    }
}

namespace {

stfio::filetype NativeFileType(const std::string& filename, stfio::filetype hint) {
    char signature[8] = {};
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open file: " + filename);
    file.read(signature, sizeof(signature));
    if (std::string(signature, 4) == "ABF2" || std::string(signature, 4) == "ABF ")
        return stfio::abf;
    if (std::string(signature, 3) == "ATF") return stfio::atf;
    if (std::string(signature, 7) == "CEDFILE") return stfio::cfs;
    if (std::string(signature, 8) == std::string("\211HDF\r\n\032\n", 8)) return stfio::hdf5;
    if (hint != stfio::none && hint != stfio::biosig) return hint;
    const auto dot = filename.find_last_of('.');
    if (dot != std::string::npos) {
        std::string extension = filename.substr(dot);
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#ifndef TEST_MINIMAL
        const auto type = stfio::findType("*" + extension);
        if (type != stfio::none) return type;
#endif
    }
    return hint;
}

void ImportNative(const std::string& filename, stfio::filetype type,
                  Recording& data, stfio::ProgressInfo& progress) {
    switch (type) {
    case stfio::abf: stfio::importABFFile(filename, data, progress); break;
    case stfio::atf: stfio::importATFFile(filename, data, progress); break;
    case stfio::hdf5: stfio::importHDF5File(filename, data, progress); break;
    case stfio::cfs: stfio::importCFSFile(filename, data, progress); break;
    case stfio::intan: stfio::importIntanFile(filename, data, progress); break;
#if !defined(WITH_BIOSIG)
    case stfio::axg: stfio::importAXGFile(filename, data, progress); break;
    case stfio::heka: stfio::importHEKAFile(filename, data, progress); break;
#endif
    default: throw std::runtime_error("No native reader is available for this file format.");
    }
}

} // namespace

bool stfio::hasNativeReader(filetype type) {
    switch (type) {
    case abf: case atf: case hdf5: case cfs: case intan:
#if !defined(WITH_BIOSIG)
    case axg: case heka:
#endif
        return true;
    default: return false;
    }
}

stfio::fileReader stfio::parseReader(const std::string& name) {
    if (name == "auto" || name == "automatic") return fileReader::automatic;
    if (name == "biosig") return fileReader::biosig;
    if (name == "native") return fileReader::native;
    throw std::invalid_argument("Reader must be 'auto', 'biosig' or 'native'.");
}

stfio::ReaderInfo stfio::getReaderInfo(const std::string& filename, filetype hint) {
    ReaderInfo info;
    info.type = NativeFileType(filename, hint);
#if defined(WITH_BIOSIG)
    inspectBiosigFile(filename, info);
#endif
    info.nativeAvailable = hasNativeReader(info.type);
    if (!info.biosigAvailable || (info.recommended == fileReader::native && info.nativeAvailable)) {
        info.recommended = fileReader::native;
    } else {
        info.recommended = fileReader::biosig;
    }
    return info;
}

bool stfio::importFile(
        const std::string& fName,
        stfio::filetype type,
        Recording& ReturnData,
        const stfio::txtImportSettings& txtImport,
        ProgressInfo& progDlg,
        fileReader reader,
        fileReader* usedReader
) {
    (void)txtImport;
    ReaderInfo info;
    if (reader == fileReader::native) {
        // An explicit native choice bypasses BioSig completely.
        info.type = NativeFileType(fName, type);
        info.nativeAvailable = hasNativeReader(info.type);
    } else if (reader == fileReader::automatic) {
        info = getReaderInfo(fName, type);
    }
    const auto selected = reader == fileReader::automatic ? info.recommended : reader;
    if (selected == fileReader::biosig) {
#if defined(WITH_BIOSIG)
        try {
            if (importBiosigFile(fName, ReturnData, progDlg, true) != stfio::biosig
                || ReturnData.size() == 0 || ReturnData[0].size() == 0)
                throw std::runtime_error("BioSig could not read this file. Choose Automatic or Native.");
            if (usedReader) *usedReader = fileReader::biosig;
            return true;
        } catch (const std::exception&) {
            if (reader != fileReader::automatic || !info.nativeAvailable) throw;
            ReturnData.resize(0);
        }
#else
        throw std::runtime_error("This build does not include the BioSig reader.");
#endif
    }
    ImportNative(fName, info.type, ReturnData, progDlg);
    if (usedReader) *usedReader = fileReader::native;
    return true;
}

bool stfio::exportFile(const std::string& fName, stfio::filetype type, const Recording& Data,
                       ProgressInfo& progDlg)
{
    try {
        switch (type) {
        case stfio::atf: {
            stfio::exportATFFile(fName, Data);
            break;
        }
#if defined(WITH_BIOSIG)
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
        catch (const std::out_of_range&) {
            throw;
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
