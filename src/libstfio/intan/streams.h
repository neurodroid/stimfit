//----------------------------------------------------------------------------------
// streams.h
// Downloaded from http://www.intantech.com/files/CLAMP_source_code_v1_0.zip
// as of 2017-03-13
//
// Copyright information added according to license directory in the original
// source code package:
//
// Intan Technologies
//
// Copyright (c) 2016 Intan Technologies LLC
//
// This is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This file is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this file.  If not, see <http://www.gnu.org/licenses/>.
//----------------------------------------------------------------------------------

#ifndef STREAMS_H
#define STREAMS_H

#include <memory>
#include <string>
#if __cplusplus > 199711L
#include <cstdint>
using std::unique_ptr;
using std::move;
#else
#include <boost/cstdint.hpp>
#include <boost/move/unique_ptr.hpp>
using boost::movelib::unique_ptr;
using boost::move;
#define nullptr NULL
#endif
#include <iosfwd>
#include <istream>


#if defined(_WIN32) && defined(_UNICODE)
    typedef std::wstring FILENAME;
#else
    typedef std::string FILENAME;
#endif

//  ------------------------------------------------------------------------
class InStream {
public:
    virtual ~InStream() {}

    virtual int read(char* data, int len) = 0;
    virtual uint64_t bytesRemaining() = 0;
    virtual std::istream::pos_type currentPos() = 0;
};

//  ------------------------------------------------------------------------
class FileInStream : public InStream {
public:
    FileInStream();
    ~FileInStream();

    virtual bool open(const FILENAME& filename); // Opens with new name
    virtual int read(char* data, int len) override;
    uint64_t bytesRemaining() override;
    std::istream::pos_type currentPos() override;

private:
    unique_ptr<std::ifstream> filestream;
    std::istream::pos_type filesize;

    virtual void close();
};

//  ------------------------------------------------------------------------
class BinaryReader {
public:
#if __cplusplus > 199711L
    BinaryReader(unique_ptr<FileInStream>&& other_);
#else
    BinaryReader(BOOST_RV_REF(unique_ptr<FileInStream>) other_);
#endif
    virtual ~BinaryReader();

    uint64_t bytesRemaining() { return other->bytesRemaining();  }
    std::istream::pos_type currentPos() { return other->currentPos(); }

protected:
    friend BinaryReader& operator>>(BinaryReader& istream, int32_t& value);
    friend BinaryReader& operator>>(BinaryReader& istream, uint32_t& value);
    friend BinaryReader& operator>>(BinaryReader& istream, int16_t& value);
    friend BinaryReader& operator>>(BinaryReader& istream, uint16_t& value);
    friend BinaryReader& operator>>(BinaryReader& istream, int8_t& value);
    friend BinaryReader& operator>>(BinaryReader& istream, uint8_t& value);
    friend BinaryReader& operator>>(BinaryReader& istream, float& value);
    friend BinaryReader& operator>>(BinaryReader& istream, double& value);
    friend BinaryReader& operator>>(BinaryReader& istream, std::wstring& value);

private:
    unique_ptr<FileInStream> other;
};

FILENAME toFileName(const std::string& s);
FILENAME toFileName(const std::wstring& ws);

#endif // STREAMS_H
