#ifndef _PYSTFIO_H
#define _PYSTFIO_H

#include "../libstfio/stfio.h"

stfio::filetype gettype(const std::string& ftype);
bool _read(const std::string& filename, const std::string& ftype, bool verbose, Recording& Data);

#endif
