#include "streams.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "common.h"
#include <system_error>
#include <string.h>

using std::cerr;
using std::endl;
using std::unique_ptr;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::wstring;
using std::vector;
using std::exception;
using std::runtime_error;

#define IS_BIG_ENDIAN (*(uint16_t *)"\0\xff" < 0x100)

//  ------------------------------------------------------------------------
FileOutStream::FileOutStream() : filestream(nullptr) {

}

FileOutStream::~FileOutStream() {
    close();
}

void FileOutStream::open(const FILENAME& filename) {
    unique_ptr<ofstream> tmp(new ofstream(filename, ios::binary | ios::out));
    if (tmp->is_open()) {
        filestream.reset(tmp.release());
        return;
    } else {
        throw std::system_error(errno, std::system_category());;
    }
}

void FileOutStream::close() {
    filestream.reset();
}

int FileOutStream::write(const char* data, int len) {
    filestream->write(data, len);
    if (filestream->fail()) {
        throw std::system_error(errno, std::system_category());
    }
    return len;
}

//  ------------------------------------------------------------------------
//const unsigned int BUFFERSIZE = ;
BufferedOutStream::BufferedOutStream(unique_ptr<FileOutStream>&& other_, unsigned int bufferSize_) :
    other(std::move(other_)),
    dataStreamBuffer(nullptr),
    bufferIndex(0),
    bufferSize(bufferSize_)
{
    dataStreamBuffer = new char[bufferSize_ + 4 * KILO];
}

BufferedOutStream::~BufferedOutStream() {
    flush();
    delete[] dataStreamBuffer;
}

int BufferedOutStream::write(const char* data, int len) {
    memcpy(&dataStreamBuffer[bufferIndex], data, len);
    bufferIndex += len;
    flushIfNecessary();
    return len;
}

void BufferedOutStream::flushIfNecessary() {
    if (bufferIndex > bufferSize) {
        other->write(dataStreamBuffer, bufferSize);
        bufferIndex -= bufferSize;
        memmove(dataStreamBuffer, dataStreamBuffer + bufferSize, bufferIndex);
    }
}

void BufferedOutStream::flush() {
    if (bufferIndex > 0) {
        other->write(dataStreamBuffer, bufferIndex);
        bufferIndex = 0;
    }
}

//  ------------------------------------------------------------------------
BinaryWriter::BinaryWriter(unique_ptr<FileOutStream>&& other_, unsigned int bufferSize_) :
    other(std::move(other_), bufferSize_)
{
}

BinaryWriter::~BinaryWriter() {
}

BinaryWriter& operator<<(BinaryWriter& ostream, int32_t value) {
    char data[4];
    data[0] = value & 0x000000ff;
    data[1] = (value & 0x0000ff00) >> 8;
    data[2] = (value & 0x00ff0000) >> 16;
    data[3] = (value & 0xff000000) >> 24;
    ostream.other.write(data, 4);
    return ostream;
}

BinaryWriter& operator<<(BinaryWriter& ostream, uint32_t value) {
    char data[4];
    data[0] = value & 0x000000ff;
    data[1] = (value & 0x0000ff00) >> 8;
    data[2] = (value & 0x00ff0000) >> 16;
    data[3] = (value & 0xff000000) >> 24;
    ostream.other.write(data, 4);
    return ostream;
}


BinaryWriter& operator<<(BinaryWriter& ostream, uint16_t value) {
    char data[2];
    data[0] = value & 0x00ff;
    data[1] = (value & 0xff00) >> 8;
    ostream.other.write(data, 2);
    return ostream;
}


BinaryWriter& operator<<(BinaryWriter& ostream, int16_t value) {
    char data[2];
    data[0] = value & 0x00ff;
    data[1] = (value & 0xff00) >> 8;
    ostream.other.write(data, 2);
    return ostream;
}

BinaryWriter& operator<<(BinaryWriter& ostream, uint8_t value) {
    char data[1];
    data[0] = value;
    ostream.other.write(data, 1);
    return ostream;
}


BinaryWriter& operator<<(BinaryWriter& ostream, int8_t value) {
    char data[1];
    data[0] = value;
    ostream.other.write(data, 1);
    return ostream;
}

BinaryWriter& operator<<(BinaryWriter& ostream, float value) {
    char* tmp = reinterpret_cast<char*>(&value);
    if (IS_BIG_ENDIAN) {
        char data[4];
        data[0] = tmp[3];
        data[1] = tmp[2];
        data[2] = tmp[1];
        data[3] = tmp[0];
        ostream.other.write(data, sizeof(value));
    } else {
        ostream.other.write(tmp, sizeof(value));
    }
    return ostream;
}

BinaryWriter& operator<<(BinaryWriter& ostream, double value) {
    float f = static_cast<float>(value);
    return ostream << f;
}

BinaryWriter& operator<<(BinaryWriter& ostream, const wstring& value) {
    if (value.empty()) {
        ostream << (uint32_t)0;
    } else {
        uint32_t size = static_cast<uint32_t>(2 * value.length());
        ostream << size;
        ostream.other.write(reinterpret_cast<const char*>(value.c_str()), size);
    }
    return ostream;
}

//  ------------------------------------------------------------------------
FileInStream::FileInStream() : filestream(nullptr) {

}

FileInStream::~FileInStream() {
    close();
}

bool FileInStream::open(const FILENAME& filename) {
    unique_ptr<ifstream> tmp(new ifstream(filename, ios::binary | ios::in));
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
BinaryReader::BinaryReader(unique_ptr<FileInStream>&& other_) :
    other(std::move(other_))
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
        istream.other->read(tmp.data(), size);
        tmp[size] = 0;
        tmp[size + 1] = 0;
        value = reinterpret_cast<wchar_t*>(tmp.data());
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
