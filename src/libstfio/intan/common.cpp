#include "common.h"
#include <iostream>

using std::ostream;
using std::string;
using std::wstring;

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