//----------------------------------------------------------------------------------
// streams.cpp
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
//
// You should have received a copy of the GNU General Public License
// along with this file.  If not, see <http://www.gnu.org/licenses/>.
//----------------------------------------------------------------------------------

#include "streams.h"
#include <iostream>
#include <fstream>
#include <cerrno>
#include <vector>
#include "common.h"
#include <string.h>

using std::cerr;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::wstring;
using std::vector;
using std::exception;
using std::runtime_error;

#define IS_BIG_ENDIAN (*(uint16_t *)"\0\xff" < 0x100)

//  ------------------------------------------------------------------------
FileInStream::FileInStream() : filestream(nullptr) {

}

FileInStream::~FileInStream() {
    close();
}

bool FileInStream::open(const FILENAME& filename) {
    unique_ptr<ifstream> tmp(new ifstream(filename.c_str(), ios::binary | ios::in));
    if (tmp->is_open()) {
        filestream.reset(tmp.release());
        filestream->seekg(0, filestream->end);
        filesize = filestream->tellg();
        filestream->seekg(0, filestream->beg);

        return true;
    } else {
        char buffer[100];
#ifdef _WIN32
        if (strerror_s(buffer, sizeof(buffer), errno) == 0) {
#else
        if (strerror_r(errno, buffer, sizeof(buffer)) == 0) {
#endif
            cerr << "Cannot open file for reading: " << buffer << endl;
        } else {
            cerr << "Cannot open file for reading: reason unknown" << endl;
        }
        return false;
    }
}

uint64_t FileInStream::bytesRemaining() {
    std::istream::pos_type pos = filestream->tellg();
    return filesize - pos;
}

std::istream::pos_type FileInStream::currentPos() {
     return filestream->tellg();
}

void FileInStream::close() {
    filestream.reset();
}

int FileInStream::read(char* data, int len) {
    filestream->read(data, len);
    if (filestream->fail()) {
        throw runtime_error("No more data");
    }
    return static_cast<int>(filestream->gcount());
}

//  ------------------------------------------------------------------------
#if __cplusplus > 199711L
    BinaryReader::BinaryReader(unique_ptr<FileInStream>&& other_) :
#else
    BinaryReader::BinaryReader(BOOST_RV_REF(unique_ptr<FileInStream>) other_) :
#endif
    other(move(other_))
{
}

BinaryReader::~BinaryReader() {
}

BinaryReader& operator>>(BinaryReader& istream, int32_t& value) {
    unsigned char data[4];
    istream.other->read(reinterpret_cast<char*>(data), 4);
    value = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
    return istream;
}

BinaryReader& operator>>(BinaryReader& istream, uint32_t& value) {
    unsigned char data[4];
    istream.other->read(reinterpret_cast<char*>(data), 4);
    value = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
    return istream;
}


BinaryReader& operator>>(BinaryReader& istream, uint16_t& value) {
    unsigned char data[2];
    istream.other->read(reinterpret_cast<char*>(data), 2);
    value = data[0] | (data[1] << 8);
    return istream;
}


BinaryReader& operator>>(BinaryReader& istream, int16_t& value) {
    unsigned char data[2];
    istream.other->read(reinterpret_cast<char*>(data), 2);
    value = data[0] | (data[1] << 8);
    return istream;
}

BinaryReader& operator>>(BinaryReader& istream, uint8_t& value) {
    unsigned char data[1];
    istream.other->read(reinterpret_cast<char*>(data), 1);
    value = data[0];
    return istream;
}


BinaryReader& operator>>(BinaryReader& istream, int8_t& value) {
    unsigned char data[1];
    istream.other->read(reinterpret_cast<char*>(data), 1);
    value = data[0];
    return istream;
}

BinaryReader& operator>>(BinaryReader& istream, float& value) {
    char* tmp = reinterpret_cast<char*>(&value);
    if (IS_BIG_ENDIAN) {
        char data[4];
        istream.other->read(data, sizeof(value));
        tmp[0] = data[3];
        tmp[1] = data[2];
        tmp[2] = data[1];
        tmp[3] = data[0];
    } else {
        istream.other->read(tmp, sizeof(value));
    }
    return istream;
}

BinaryReader& operator>>(BinaryReader& istream, double& value) {
    float f;
    istream >> f;
    value = f;
    return istream;
}

BinaryReader& operator>>(BinaryReader& istream, wstring& value) {
    uint32_t size;
    istream >> size;
    value.clear();
    if (size > 0) {
        vector<char> tmp(size + 2);
#ifndef _WINDOWS
        istream.other->read(tmp.data(), size);
        tmp[size] = 0;
        tmp[size + 1] = 0;
        value = reinterpret_cast<wchar_t*>(tmp.data());
#else
        istream.other->read(&tmp[0], size);
        tmp[size] = 0;
        tmp[size + 1] = 0;
        value = reinterpret_cast<wchar_t*>(&tmp[0]);
#endif
    }
    return istream;
}

//  ------------------------------------------------------------------------
FILENAME toFileName(const std::string& s) {
#if defined(_WIN32) && defined(_UNICODE)
    return toWString(s);
#else
    return s;
#endif
}

FILENAME toFileName(const std::wstring& ws) {
#if defined(_WIN32) && defined(_UNICODE)
    return ws;
#else
    return toString(ws);
#endif
}
