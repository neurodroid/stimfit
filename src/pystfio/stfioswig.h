#ifndef _STFIOSWIG_H
#define _STFIOSWIG_H

#include "../libstfio/stfio.h"

stfio::filetype gettype(const std::string& ftype);
bool _read(const std::string& filename, const std::string& ftype, Recording& Data);

/* Progress Info interface adapter; does nothing at present */
class StdoutProgressInfo : public stfio::ProgressInfo {
 public:
    StdoutProgressInfo(const std::string& title, const std::string& message, int maximum);
    bool Update(int value, const std::string& newmsg="", bool* skip=NULL);
};

#endif
