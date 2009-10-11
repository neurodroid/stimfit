//***********************************************************************************************
//
//    Copyright (c) 2005 Molecular Devices Corporation.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************

#ifndef INC_ABF2PROTOCOLREADER_H
#define INC_ABF2PROTOCOLREADER_H

#include "abf2headr.h"
#include "SimpleStringCache.hpp"
#include "ProtocolStructs.h"            // Struct definitions for actual file contents
#include "../axon/AxAbfFio32/AxAbffio32.h"
#include "../axon/AxAbfFio32/filedesc.hpp"
#include <boost/shared_ptr.hpp>

//===============================================================================================
class CABF2ProtocolReader
{
  private:
    ABF2_FileInfo         m_FileInfo;
    CSimpleStringCache   m_Strings;  // The string writing object.
    CFileDescriptor* m_pFI;
    int nFile;
    boost::shared_ptr<ABF2FileHeader> m_pFH;

    BOOL ReadFileInfo();
    BOOL ReadProtocolInfo();
    BOOL ReadADCInfo();
    BOOL ReadDACInfo();
    BOOL ReadEpochs();
    BOOL ReadStats();
    BOOL ReadUserList();
    BOOL ReadMathInfo();

    BOOL GetString( UINT uIndex, LPSTR pszText, UINT uBufSize );

  public:
    CABF2ProtocolReader( );
    virtual ~CABF2ProtocolReader();

    virtual BOOL Open( LPCTSTR fName );
    virtual BOOL Close( );
    
    static BOOL CanOpen( const void *pFirstBlock, UINT uBytes );
   
    virtual BOOL Read( int* pnError);
    virtual const ABF2_FileInfo *GetFileInfo() const      { return &m_FileInfo; }
    virtual const ABF2FileHeader* GetFileHeader() const      { return m_pFH.get(); }
    virtual ABF2FileHeader* GetFileHeaderW() { return m_pFH.get(); }
    virtual int GetFileNumber() const { return nFile; }
    // virtual BOOL ValidateCRC();
};

#endif   // INC_ABF2PROTOCOLREADER_H
