//----------------------------------------------------------------------------------
// common.h
//
// Intan Technologies
// Common Header File
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

#pragma once

#include <iosfwd>
#include <string>
#include <stdexcept>
#include <sstream>

using std::endl;

// Debug output window printing macro
#define DEBUGOUT( s )            \
{                             \
   std::ostringstream os_;    \
   os_ << "  DEBUG: " << s;                   \
   OutputDebugStringA( os_.str().c_str() );  \
}

// Logging -----------------------------------------------------------------
// Logging goes to this, which may point to dev_null
extern std::ostream* logger;

extern std::ostream* nulllogger; // Always points to dev_null

// To 
#define LOG(x) ((x) ? (*logger) : (*nulllogger))

// Use SetLogger(&std::cerr), for example, or SetLogger(nullptr).
std::ostream* SetLogger(std::ostream* logger_);

// _T macro for unicode support ---------------------------------------------
#ifndef _T
    #if defined(_WIN32) && defined(_UNICODE)
	    #define _T(x) L ##x
    #else
	    #define _T(x) x
    #endif
#endif

std::wstring toWString(const std::string& s);
std::string toString(const std::wstring& ws);

// Throws an exception if value has more than numBits set.  For example, if you're trying to put something into 3 bits, and you specify 15.
template <typename T>
T CheckBits(T value, unsigned int numBits) {
    T mask = (-1 << numBits);
    if ((value & mask) != 0) {
        throw std::invalid_argument("Invalid value with too many bits set.");
    }
    return value;
}

#define CALL_MEMBER_FN(object,ptrToMember)  ((object).*(ptrToMember))

#if defined WIN32
    #define NOMINMAX
    #include <windows.h>
    bool _trace(TCHAR *format, ...);
    #ifdef _DEBUG
        #define TRACE _trace
    #else
        #define TRACE false && _trace
    #endif
#endif