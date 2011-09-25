//***********************************************************************************************
//
//    Copyright (c) 1999 - 2002 Axon Instruments, Inc.
//    All rights reserved.
//
//***********************************************************************************************
// MODULE:  SimpleStringCache.HPP
// PURPOSE: 
// AUTHOR:  BHI  Nov 1999
//          PRC  May 2002

// Simple String Cache class, based on StringCache.hpp / cpp
//

#ifndef INC_SIMPLESTRINGCACHE_HPP
#define INC_SIMPLESTRINGCACHE_HPP

#pragma once
#include <vector>
#pragma pack(push, 1)

#pragma pack(pop)

class CSimpleStringCache
{
private:   // Attributes
   // Typedefs to simplify code.

   std::vector<LPCSTR> m_Cache;

   UINT     m_uMaxSize;

private:   // Unimplemented copy functions.
   CSimpleStringCache(const CSimpleStringCache &);
   const CSimpleStringCache &operator=(const CSimpleStringCache &);

public:    // Public interface
   CSimpleStringCache();
   ~CSimpleStringCache();

   void   Clear();
   UINT   Add(LPCSTR psz);
   LPCSTR Get(UINT uIndex) const;

   BOOL   Write(HANDLE hFile, UINT &uOffset) const;
   BOOL   Read(HANDLE hFile, UINT uOffset);

   UINT   GetNumStrings() const;
   UINT   GetMaxSize() const { return m_uMaxSize; };
   UINT   GetTotalSize() const;

};

#endif      // INC_SIMPLESTRINGCACHE_HPP

