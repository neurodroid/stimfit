#ifndef _STFIOSWIG_H
#define _STFIOSWIG_H

#include "../core/stimdefs.h"

stf::filetype gettype(const std::string& ftype);
bool _read(const std::string& filename, const std::string& ftype, Recording& Data);

#endif
