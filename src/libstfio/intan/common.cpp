//----------------------------------------------------------------------------------
// common.h
// Downloaded from http://www.intantech.com/files/CLAMP_source_code_v1_0.zip
// as of 2017-03-13
//
// Original comment and license information for the header file common.h:
// Intan Technologies
// Common Source File
// Version 2.0 (25 August 2015)
//
// Provides common functionality for RHD and CLAMP source code.
//
// Copyright (c) 2014-2015 Intan Technologies LLC
//
// This software is provided 'as-is', without any express or implied warranty.
// In no event will the authors be held liable for any damages arising from the
// use of this software.
//
// Permission is granted to anyone to use this software for any applications that
// use Intan Technologies integrated circuits, and to alter it and redistribute it
// freely.
//
// See http://www.intantech.com for documentation and product information.
//----------------------------------------------------------------------------------

#include "common.h"
#include <iostream>

using std::ostream;
using std::string;
using std::wstring;

#if __cplusplus <= 199711L
#define nullptr NULL
#endif

class nullbuf : public std::streambuf
{
protected:
    virtual int overflow(int c) { return c; }
};

class nullstream : public std::ostream {
    nullbuf _streambuf;
public:
    nullstream() : std::ostream(&_streambuf)
    {
        clear();
    }
};

// A stream that ouputs nowhere
nullstream dev_null;

ostream* logger = &dev_null;
ostream* nulllogger = &dev_null;

ostream* SetLogger(std::ostream* logger_) {
    ostream* prev = logger;
    if (logger_ == nullptr) {
        logger = &dev_null;
    }
    else {
        logger = logger_;
    }
    return prev;
}

wstring toWString(const string& s) {
    wstring ws;
    ws.insert(ws.begin(), s.begin(), s.end());
    return ws;
}

string toString(const wstring& ws) {
    string s;
    s.insert(s.begin(), ws.begin(), ws.end());
    return s;
}

#if defined WIN32 && defined _DEBUG
bool _trace(TCHAR *format, ...)
{
    TCHAR buffer[1000];

    va_list argptr;
    va_start(argptr, format);
    vswprintf_s(buffer, sizeof(buffer)/sizeof(buffer[0]), format, argptr);
    va_end(argptr);

    OutputDebugString(buffer);

    return true;
}
#endif
