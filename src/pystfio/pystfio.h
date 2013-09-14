#ifndef _PYSTFIO_H
#define _PYSTFIO_H

#include "../libstfio/stfio.h"

#if (defined(WITH_BIOSIG) || defined(WITH_BIOSIG2))
  #define TEST_MINIMAL
#endif

stfio::filetype gettype(const std::string& ftype);
bool _read(const std::string& filename, const std::string& ftype, bool verbose, Recording& Data);

#endif
