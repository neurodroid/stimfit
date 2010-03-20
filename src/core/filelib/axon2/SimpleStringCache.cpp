//***********************************************************************************************
//
//    Copyright (c) 1999 - 2002 Axon Instruments, Inc.
//    All rights reserved.
//
//***********************************************************************************************
// MODULE:  SimpleStringCache.CPP
// PURPOSE: Cache of strings stored in a vector.
// AUTHOR:  BHI  Nov 1999
//          PRC  May 2002
//
#include "../axon/Common/wincpp.hpp"
#include "SimpleStringCache.hpp"
#include "../axon/Common/ArrayPtr.hpp"
#include "../axon/Common/FileIO.hpp"

#ifdef _WINDOWS
#pragma warning(disable : 4201)
#include <mmsystem.h>
#endif

#include "../axon/Common/FileIO.hpp"

#include <iostream>

#define MAKEFOURCC(ch0, ch1, ch2, ch3)                  \
    ((DWORD)(BYTE)(ch0) | ((DWORD)(BYTE)(ch1) << 8) |   \
    ((DWORD)(BYTE)(ch2) << 16) | ((DWORD)(BYTE)(ch3) << 24 ))

const DWORD c_dwSIGNATURE         = MAKEFOURCC('S','S','C','H');   // Simple String Cache Header
const DWORD c_dwCURRENT_VERSION   = MAKEFOURCC(1,0,0,0);           // 1.0.0.0

#if defined(__linux__) || defined(__STF__) || defined(__APPLE__)
    #define max(a,b)   (((a) > (b)) ? (a) : (b))
    #define min(a,b)   (((a) < (b)) ? (a) : (b))
#endif

struct SimpleStringCacheHeader
{
   DWORD dwSignature;
   DWORD dwVersion;
   UINT  uNumStrings;
   UINT  uMaxSize;
   ABFLONG  lTotalBytes;
   UINT  uUnused[6];

   SimpleStringCacheHeader()
   {
      memset(this, 0, sizeof(*this));
      dwSignature = c_dwSIGNATURE;
      dwVersion   = c_dwCURRENT_VERSION;
   }
};

#if 0
//#define SHOW_STRUCT_SIZES
#ifdef SHOW_STRUCT_SIZES

AXODBG_SHOW_SIZE(SimpleStringCacheHeader);
#else
ASSERT( sizeof(SimpleStringCacheHeader) == 44 );
#endif

#endif

//###############################################################################################
//###############################################################################################
//###############################################################################################
//###############################################################################################

//===============================================================================================
// FUNCTION: Constructor
// PURPOSE:  Object initialization.
//
CSimpleStringCache::CSimpleStringCache()
{
   MEMBERASSERT();

   m_uMaxSize = 0;
}

//===============================================================================================
// FUNCTION: Destructor
// PURPOSE:  Object cleanup.
//
CSimpleStringCache::~CSimpleStringCache()
{
   MEMBERASSERT();

   Clear();
}

//===============================================================================================
// FUNCTION: Clear
// PURPOSE:  Clear the cache.
//
void CSimpleStringCache::Clear()
{
   MEMBERASSERT();

   // Delete the strings.
   for( UINT i=0; i<m_Cache.size(); i++ )
   {
      LPSTR pszItem = const_cast<LPSTR>( m_Cache[i] );
      delete pszItem;
      pszItem = NULL;
   }

   // Now  clear the vector
   m_Cache.clear();
}

//===============================================================================================
// FUNCTION: Add
// PURPOSE:  Add a new string into the cache.
//
UINT CSimpleStringCache::Add(LPCSTR psz)
{
   MEMBERASSERT();

   std::size_t uLen = strlen(psz);
   LPSTR pszText = new char[uLen+1];
   strcpy( pszText, psz );

   m_Cache.push_back( pszText );

   m_uMaxSize = max( m_uMaxSize, uLen );

   return GetNumStrings();
}

//===============================================================================================
// FUNCTION: Get
// PURPOSE:  Get the string pointer that corresponds to the index.
//
LPCSTR CSimpleStringCache::Get(UINT uIndex) const
{
   MEMBERASSERT();

   if( uIndex < m_Cache.size() )
   {
      LPCSTR pszText = m_Cache[uIndex];
      return pszText;
   }
#ifndef __APPLE__
   std::cerr << "Bad index passed to CSimpleStringCache (" << uIndex << ")";
#endif
   return NULL;
}


//===============================================================================================
// FUNCTION: GetTotalSize
// PURPOSE:  Returns to total size (in bytes) required to write out all the strings (including the header).
//
UINT CSimpleStringCache::GetTotalSize() const
{
   MEMBERASSERT();

   UINT uSize = sizeof(SimpleStringCacheHeader);
   for( std::size_t i=0; i<m_Cache.size(); i++ )
   {
      LPCSTR pszText = m_Cache[i];
      uSize += (UINT)strlen( pszText ) + 1;     // Allow for the the terminator.
   }

   return uSize;
}



//#################################################################################################
//#################################################################################################
//###
//###  File I/O code.
//###
//#################################################################################################
//#################################################################################################

#if 0
//===============================================================================================
// FUNCTION: Write
// PURPOSE:  Write the cache to a file.
//
BOOL CSimpleStringCache::Write(HANDLE hFile, UINT &uOffset) const
{
   MEMBERASSERT();

   // Object wrapper for the file handle.
   CFileIO_NoClose File(hFile);

   // Get the current position.
   LONGLONG lHeaderPos = 0;
   if (!File.GetCurrentPosition(&lHeaderPos))
      return false;

   // Build the header and write it to the file.
   SimpleStringCacheHeader Header;
   Header.uNumStrings = GetNumStrings();
   Header.uMaxSize = m_uMaxSize;

   if (!File.Write(&Header, sizeof(Header)))
      return false;
   
   // Get the position after writing the header.
   LONGLONG lPostHeaderPos = 0;
   if (!File.GetCurrentPosition(&lPostHeaderPos))
      return false;

   // Go through the blocks, writing them to the file.
   for( UINT i=0; i<m_Cache.size(); i++ )
   {
      LPCSTR pszText = m_Cache[i];
      UINT uLen = strlen( pszText ) + 1;     // Write out the NULL terminator as well.
      if( !File.Write( pszText, uLen ) )
         return false;
   }

   LONGLONG lSavePos = 0;
   File.GetCurrentPosition(&lSavePos);

   Header.lTotalBytes = ABFLONG( lSavePos - lPostHeaderPos );
   Header.uNumStrings = m_Cache.size();
   File.Seek(lHeaderPos);
   File.Write(&Header, sizeof(Header));
   File.Seek(lSavePos);

   uOffset = UINT(lHeaderPos);
   return true;
}
#endif

//===============================================================================================
// FUNCTION: Read
// PURPOSE:  Read the cache from a file.
//
BOOL CSimpleStringCache::Read(HANDLE hFile, UINT uOffset)
{
   MEMBERASSERT();
   Clear();

   // Object wrapper for the file handle.
   CFileIO_NoClose File(hFile);

   // Get the current position.
   if (!File.Seek(uOffset))
      return false;

   // Read the header from the file.
   SimpleStringCacheHeader Header;
   if (!File.Read(&Header, sizeof(Header)))
      return false;

   if ((Header.dwSignature != c_dwSIGNATURE) || (Header.dwVersion != c_dwCURRENT_VERSION))
      return false;
   
   m_uMaxSize = Header.uMaxSize;

   // Read everything into a buffer.
   CArrayPtr<char> pszBuffer( Header.lTotalBytes );
   if( !File.Read( pszBuffer, Header.lTotalBytes ) )
      return false;

   // Copy each string into the cache.
   LPCSTR pszText = pszBuffer;
   for (UINT i=0; i<Header.uNumStrings; i++)
   {
      if( !pszText )
         return false;
      Add( pszText );
      
      // Move the pointer over the NULL terminator.
      pszText += strlen( pszText ) + 1;
   }

   return true;
}


//===============================================================================================
// FUNCTION: GetNumStrings
// PURPOSE:  .
//
UINT CSimpleStringCache::GetNumStrings() const
{
   MEMBERASSERT();
   
   return (UINT)m_Cache.size();
}

