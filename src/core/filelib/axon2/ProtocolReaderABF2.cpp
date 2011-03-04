//***********************************************************************************************
//
//    Copyright (c) 2005 Molecular Devices.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
//
// MODULE:  ProtocolReader.CPP
// PURPOSE: ReadsABF 2 protocols from an ABF file.
// 

#include "../axon/Common/wincpp.hpp"
#include "../axon/Common/axodefn.h"
#include "ProtocolReaderABF2.hpp"
#include "../axon/AxAbfFio32/abfutil.h"
#include "../axon/AxAbfFio32/abffiles.h"
#include <math.h>

#if defined(__linux__) || defined(__STF__) || defined(__APPLE__)
#define max(a,b)   (((a) > (b)) ? (a) : (b))
#define min(a,b)   (((a) < (b)) ? (a) : (b))
#endif

extern  BOOL GetNewFileDescriptor(CFileDescriptor **ppFI, int *pnFile, int *pnError);
extern  BOOL GetFileDescriptor(CFileDescriptor **ppFI, int nFile, int *pnError);
extern  void ReleaseFileDescriptor(int nFile);
static BOOL ErrorReturn(int *pnError, int nErrorNum)
{
   if (pnError)
      *pnError = nErrorNum;
   return FALSE;
}

//===============================================================================================

//===============================================================================================
// FUNCTION: FlattenGearShift
// PURPOSE:  Converts a header with gear shift enabled to the equivalent with the gear shift flattened.
//
static BOOL FlattenGearShift(ABF2FileHeader *pFH)
{
    ASSERT( pFH );

    if( pFH->nOperationMode != ABF_WAVEFORMFILE )
        return FALSE;

    if( pFH->uFileCompressionRatio == 1 )
        return FALSE;

    return TRUE;
}

#if 0
//===============================================================================================
// FUNCTION: RemoveExtraChannels
// PURPOSE:  Removes "extra" channels when P/N is enabled and files are opened as protocols.
// NOTES:    This function allows Clampex 10 data files (with P/N enabled) to be opened as a protocol.
//           These files have an extra channel that stores the raw data (in addition to the corrected data).
//           Therefore there is some tweaking and scaling of header parameters.
//
static BOOL RemoveExtraChannels( ABF2FileHeader *pFH, UINT uFlags )
{
    WPTRASSERT( pFH );

    // Must be episodic stimulation mode.
    if( pFH->nOperationMode != ABF_WAVEFORMFILE )
        return FALSE;

    // P/N must be enabled.
    if( !ABF2H_IsPNEnabled( pFH ) )
        return FALSE;

    // This fix only applies to files with one ADC channel and one P/N channel.
    if( pFH->nADCNumChannels != 2 )
        return FALSE;

    // Must be a data file (i.e. not a protocol).
    if( pFH->lActualAcqLength == 0 )
        return FALSE;

    bool bFudge = false;
    if( uFlags & ABF_PARAMFILE )
        bFudge = true;
    else
    {
        ABFLONG lActualSamplesPerEpisode = pFH->lActualAcqLength / pFH->lActualEpisodes;
        if( lActualSamplesPerEpisode != pFH->lNumSamplesPerEpisode )
            bFudge = true;
    }
    if( !bFudge )
        return FALSE;

    int nIndex = ABF_UNUSED_CHANNEL;
    for( UINT i=0; i<ABF_ADCCOUNT; i++ )
    {
        if( pFH->nADCSamplingSeq[i] == ABF_UNUSED_CHANNEL )
        {
            nIndex = i - 1;  // i.e. the previous channel is the last valid channel.
            break;
        }
    }

    if( nIndex == ABF_UNUSED_CHANNEL )
        return FALSE;

    // There are extra channels to be stripped.
    short nOldChans        = pFH->nADCNumChannels;
    short nNewChans        = nOldChans - 1;
    pFH->nADCNumChannels   = nNewChans;
    pFH->lNumSamplesPerEpisode = MulDiv( pFH->lNumSamplesPerEpisode, nNewChans, nOldChans );

    // Adjust the statistics search regions.
    pFH->lStatsBaselineStart = MulDiv( pFH->lStatsBaselineStart, nNewChans, nOldChans );
    pFH->lStatsBaselineEnd   = MulDiv( pFH->lStatsBaselineEnd, nNewChans, nOldChans );
   
    for( UINT i=0; i<ABF_STATS_REGIONS; i++ )
    {
        pFH->lStatsStart[i] = MulDiv( pFH->lStatsStart[i], nNewChans, nOldChans );
        pFH->lStatsEnd[i]   = MulDiv( pFH->lStatsEnd[i], nNewChans, nOldChans );
    }

    return TRUE;
}
#endif

//===============================================================================================
//  CABF2ProtocolReader implementation
//
// NOTE:   Any changes made here will require complementary changes to ProtocolWriter.cpp


//===============================================================================================
// Constructor.
//
CABF2ProtocolReader::CABF2ProtocolReader( ) : m_pFI( NULL )
{
    m_pFH.reset( new ABF2FileHeader );
    ABF2H_Initialize( m_pFH.get() );
    MEMBERASSERT();

}


//===============================================================================================
// Destructor.
//
CABF2ProtocolReader::~CABF2ProtocolReader()
{
    if (m_pFI != NULL ) {
        Close();
    }
    MEMBERASSERT();

}


//===============================================================================================
// FUNCTION: Read
// PURPOSE:  Reads the complete protocol from the data file.
//
BOOL CABF2ProtocolReader::Read( int* pnError )
{
    MEMBERASSERT();

    if (m_pFI == NULL)
        return FALSE;
           
    BOOL bOK = TRUE;
    bOK &= m_pFI->Seek( 0L, FILE_BEGIN);
    if( !bOK )
        return FALSE;

    bOK &= m_pFI->Read( &m_FileInfo, sizeof( m_FileInfo ) );

    if( m_FileInfo.StringsSection.uBlockIndex )
    {
        // Read the protocol strings into the cache.
        UINT uSeekPos = m_FileInfo.StringsSection.uBlockIndex * ABF_BLOCKSIZE;
        if( !m_Strings.Read( m_pFI->GetFileHandle(), uSeekPos ) )
            return FALSE;     //SetLastError( ABF_ENOSTRINGS );
    }

    bOK &= ReadFileInfo();
    bOK &= ReadProtocolInfo();
    bOK &= ReadADCInfo();
    bOK &= ReadDACInfo();
    bOK &= ReadEpochs();
    bOK &= ReadStats();
    bOK &= ReadUserList();
    bOK &= ReadMathInfo();
    
    // modified from ABF_ReadOpen
    int nError = 0;
    
    // Check that the data file actually contains data.
    if ((m_pFH->lActualAcqLength <= 0) || (m_pFH->nADCNumChannels <= 0))
    {
        nError = ABF_EBADPARAMETERS;
        Close();
        nFile = (int)ABF_INVALID_HANDLE;
        ERRORRETURN(pnError, nError);
    }

    // Set header variable for the number of episodes in the file.
    if( m_pFH->nOperationMode == ABF2_GAPFREEFILE ) {
        double fdiv = (double)m_pFH->lActualAcqLength / m_pFH->lNumSamplesPerEpisode;
        DWORD dwMaxEpi = ceil(fdiv);
#ifdef _STFDEBUG
        std::cout << "Total number of samples " << m_pFH->lActualAcqLength << std::endl;
#endif
        m_pFH->lActualEpisodes = dwMaxEpi;
    }
    
    m_pFI->SetAcquiredEpisodes(m_pFH->lActualEpisodes);
    m_pFI->SetAcquiredSamples(m_pFH->lActualAcqLength);

    // CSH   RemoveExtraChannels( m_pFH, fFlags );
    FlattenGearShift( m_pFH.get() );

    return bOK;
}

//===============================================================================================
// FUNCTION: GetString
// PURPOSE:  Read a single ProtocolString into the buffer.
//
BOOL CABF2ProtocolReader::GetString( UINT uIndex, LPSTR pszText, UINT uBufSize )
{
    MEMBERASSERT();
    // LPSZASSERT( pszText );
    WARRAYASSERT( pszText, uBufSize );
   
    ABFU_SetABFString( pszText, "", uBufSize );
   
    // Just return an empty string if the index is invalid.
    if( uIndex == 0 )
        return TRUE;

    // or if we do not have the requested string available.
    if( uIndex > m_Strings.GetNumStrings() )
        return TRUE;

    LPCSTR pszString = m_Strings.Get( uIndex - 1 );
    if( pszString )
    {
        UINT uLen = strlen( pszString );
        if( uLen > uBufSize )
            return FALSE;

        ABFU_SetABFString( pszText, pszString, uLen );
        return TRUE;
    }

    return FALSE;
}

#define MAJOR( n )  HIBYTE( HIWORD( (n) ) )
#define MINOR( n )  LOBYTE( HIWORD( (n) ) )
#define BUGFIX( n ) HIBYTE( LOWORD( (n) ) )
#define BUILD( n )  LOBYTE( LOWORD( (n) ) )

//===============================================================================================
// FUNCTION: ReadFileInfo
// PURPOSE:  Reads the file info from the data file.
//
BOOL CABF2ProtocolReader::ReadFileInfo()
{
    MEMBERASSERT();

    BOOL bOK = TRUE;

    short nMajor = MAJOR( m_FileInfo.uFileVersionNumber );
    short nMinor = MINOR( m_FileInfo.uFileVersionNumber );
    m_pFH->fFileVersionNumber     = nMajor + nMinor/100.0F;
    m_pFH->fHeaderVersionNumber   = ABF_CURRENTVERSION;
    m_pFH->nFileType              = m_FileInfo.nFileType;
    m_pFH->nDataFormat            = m_FileInfo.nDataFormat;
    m_pFH->nSimultaneousScan      = m_FileInfo.nSimultaneousScan;
    m_pFH->FileGUID               = m_FileInfo.FileGUID;
    m_pFH->ulFileCRC              = m_FileInfo.uFileCRC;
    m_pFH->nCRCEnable             = m_FileInfo.nCRCEnable;
    m_pFH->nCreatorMajorVersion   = MAJOR ( m_FileInfo.uCreatorVersion );
    m_pFH->nCreatorMinorVersion   = MINOR ( m_FileInfo.uCreatorVersion );
    m_pFH->nCreatorBugfixVersion  = BUGFIX( m_FileInfo.uCreatorVersion );
    m_pFH->nCreatorBuildVersion   = BUILD ( m_FileInfo.uCreatorVersion );
    bOK &= GetString( m_FileInfo.uCreatorNameIndex, m_pFH->sCreatorInfo,  ELEMENTS_IN( m_pFH->sCreatorInfo ) );

    m_pFH->nModifierMajorVersion  = MAJOR ( m_FileInfo.uModifierVersion );
    m_pFH->nModifierMinorVersion  = MINOR ( m_FileInfo.uModifierVersion );
    m_pFH->nModifierBugfixVersion = BUGFIX( m_FileInfo.uModifierVersion );
    m_pFH->nModifierBuildVersion  = BUILD ( m_FileInfo.uModifierVersion );
    bOK &= GetString( m_FileInfo.uModifierNameIndex, m_pFH->sModifierInfo, ELEMENTS_IN( m_pFH->sModifierInfo ) );

    m_pFH->nNumPointsIgnored      = 0;
    m_pFH->uFileStartDate         = m_FileInfo.uFileStartDate;
    m_pFH->uFileStartTimeMS       = m_FileInfo.uFileStartTimeMS;
    m_pFH->lStopwatchTime         = m_FileInfo.uStopwatchTime;

    m_pFH->lActualEpisodes        = m_FileInfo.uActualEpisodes;
    m_pFH->lActualAcqLength       = m_FileInfo.DataSection.GetNumEntries();
    m_pFH->lDataSectionPtr        = m_FileInfo.DataSection.uBlockIndex;

    m_pFH->lScopeConfigPtr        = m_FileInfo.ScopeSection.uBlockIndex;
    m_pFH->lNumScopes             = m_FileInfo.ScopeSection.GetNumEntries();
    m_pFH->lStatisticsConfigPtr   = m_FileInfo.StatsSection.uBlockIndex;
    m_pFH->lTagSectionPtr         = m_FileInfo.TagSection.uBlockIndex;
    m_pFH->lNumTagEntries         = m_FileInfo.TagSection.GetNumEntries();
    m_pFH->lDeltaArrayPtr         = m_FileInfo.DeltaSection.uBlockIndex;
    m_pFH->lNumDeltas             = m_FileInfo.DeltaSection.GetNumEntries();
    m_pFH->lVoiceTagPtr           = m_FileInfo.VoiceTagSection.uBlockIndex;
    m_pFH->lVoiceTagEntries       = m_FileInfo.VoiceTagSection.GetNumEntries();
    m_pFH->lSynchArrayPtr         = m_FileInfo.SynchArraySection.uBlockIndex;
    m_pFH->lSynchArraySize        = m_FileInfo.SynchArraySection.GetNumEntries();
    m_pFH->lAnnotationSectionPtr  = m_FileInfo.AnnotationSection.uBlockIndex;
    m_pFH->lNumAnnotations        = m_FileInfo.AnnotationSection.GetNumEntries();
   
    bOK &= GetString( m_FileInfo.uProtocolPathIndex, m_pFH->sProtocolPath,  ELEMENTS_IN( m_pFH->sProtocolPath ) );

    return bOK;
}

//===============================================================================================
// FUNCTION: ReadProtocolInfo
// PURPOSE:  Reads the protocol info from the data file.
//
BOOL CABF2ProtocolReader::ReadProtocolInfo()
{
    MEMBERASSERT();

    BOOL bOK = TRUE;
    ABF_ProtocolInfo Protocol;
    bOK &= m_pFI->Seek( LONGLONG(m_FileInfo.ProtocolSection.uBlockIndex) * ABF_BLOCKSIZE, FILE_BEGIN );
    if( !bOK )
        return FALSE;

    bOK &= m_pFI->Read( &Protocol, sizeof( Protocol ) );
    ASSERT( bOK );

    m_pFH->nADCNumChannels          = short( m_FileInfo.ADCSection.llNumEntries );
    m_pFH->nOperationMode           = Protocol.nOperationMode;
    m_pFH->fADCSequenceInterval     = Protocol.fADCSequenceInterval;
    m_pFH->uFileCompressionRatio    = max( Protocol.uFileCompressionRatio, 1 );
    m_pFH->bEnableFileCompression   = Protocol.bEnableFileCompression;

    m_pFH->fSynchTimeUnit           = Protocol.fSynchTimeUnit;
    m_pFH->fSecondsPerRun           = Protocol.fSecondsPerRun;
    m_pFH->lNumSamplesPerEpisode    = Protocol.lNumSamplesPerEpisode;
    m_pFH->lPreTriggerSamples       = Protocol.lPreTriggerSamples;
    m_pFH->lEpisodesPerRun          = Protocol.lEpisodesPerRun;
    m_pFH->lRunsPerTrial            = Protocol.lRunsPerTrial;
    m_pFH->lNumberOfTrials          = Protocol.lNumberOfTrials;
    m_pFH->nAveragingMode           = Protocol.nAveragingMode;
    m_pFH->nUndoRunCount            = Protocol.nUndoRunCount;
    m_pFH->nFirstEpisodeInRun       = Protocol.nFirstEpisodeInRun;
    m_pFH->fTriggerThreshold        = Protocol.fTriggerThreshold;
    m_pFH->nTriggerSource           = Protocol.nTriggerSource;
    m_pFH->nTriggerAction           = Protocol.nTriggerAction;
    m_pFH->nTriggerPolarity         = Protocol.nTriggerPolarity;
    m_pFH->fScopeOutputInterval     = Protocol.fScopeOutputInterval;
    m_pFH->fEpisodeStartToStart     = Protocol.fEpisodeStartToStart;
    m_pFH->fRunStartToStart         = Protocol.fRunStartToStart;
    m_pFH->lAverageCount            = Protocol.lAverageCount;
    m_pFH->fTrialStartToStart       = Protocol.fTrialStartToStart;
    m_pFH->nAutoTriggerStrategy     = Protocol.nAutoTriggerStrategy;
    m_pFH->fFirstRunDelayS          = Protocol.fFirstRunDelayS;
    m_pFH->nChannelStatsStrategy    = Protocol.nChannelStatsStrategy;
    m_pFH->lSamplesPerTrace         = Protocol.lSamplesPerTrace;
    m_pFH->lStartDisplayNum         = Protocol.lStartDisplayNum;
    m_pFH->lFinishDisplayNum        = Protocol.lFinishDisplayNum;
    m_pFH->nShowPNRawData           = Protocol.nShowPNRawData;
    m_pFH->fStatisticsPeriod        = Protocol.fStatisticsPeriod;
    m_pFH->lStatisticsMeasurements  = Protocol.lStatisticsMeasurements;
    m_pFH->nStatisticsSaveStrategy  = Protocol.nStatisticsSaveStrategy;
    m_pFH->fADCRange                = Protocol.fADCRange;
    m_pFH->fDACRange                = Protocol.fDACRange;
    m_pFH->lADCResolution           = Protocol.lADCResolution;
    m_pFH->lDACResolution           = Protocol.lDACResolution;
    m_pFH->nDigitizerADCs           = Protocol.nDigitizerADCs;
    m_pFH->nDigitizerDACs           = Protocol.nDigitizerDACs;
    m_pFH->nDigitizerTotalDigitalOuts = Protocol.nDigitizerTotalDigitalOuts;
    m_pFH->nDigitizerSynchDigitalOuts = Protocol.nDigitizerSynchDigitalOuts;
    m_pFH->nDigitizerType           = Protocol.nDigitizerType;

    m_pFH->nExperimentType          = Protocol.nExperimentType;
    m_pFH->nManualInfoStrategy      = Protocol.nManualInfoStrategy;
    m_pFH->fCellID1                 = Protocol.fCellID[0];
    m_pFH->fCellID2                 = Protocol.fCellID[1];
    m_pFH->fCellID3                 = Protocol.fCellID[2];
    m_pFH->nCommentsEnable          = Protocol.nCommentsEnable;
    m_pFH->nAutoAnalyseEnable       = Protocol.nAutoAnalyseEnable;
    m_pFH->nSignalType              = Protocol.nSignalType;
    m_pFH->nDigitalEnable           = Protocol.nDigitalEnable;
    m_pFH->nActiveDACChannel        = Protocol.nActiveDACChannel;
    m_pFH->nDigitalHolding          = Protocol.nDigitalHolding;
    m_pFH->nDigitalInterEpisode     = Protocol.nDigitalInterEpisode;
    m_pFH->nDigitalDACChannel       = Protocol.nDigitalDACChannel;
    m_pFH->nDigitalTrainActiveLogic = Protocol.nDigitalTrainActiveLogic;
    m_pFH->nStatsEnable             = Protocol.nStatsEnable;
    m_pFH->nLevelHysteresis         = Protocol.nLevelHysteresis;
    m_pFH->lTimeHysteresis          = Protocol.lTimeHysteresis;
    m_pFH->nAllowExternalTags       = Protocol.nAllowExternalTags;
    m_pFH->nAverageAlgorithm        = Protocol.nAverageAlgorithm;
    m_pFH->fAverageWeighting        = Protocol.fAverageWeighting;
    m_pFH->nUndoPromptStrategy      = Protocol.nUndoPromptStrategy;
    m_pFH->nTrialTriggerSource      = Protocol.nTrialTriggerSource;
    m_pFH->nStatisticsDisplayStrategy = Protocol.nStatisticsDisplayStrategy;
    m_pFH->nExternalTagType         = Protocol.nExternalTagType;
    m_pFH->nStatisticsClearStrategy = Protocol.nStatisticsClearStrategy;
    m_pFH->nLTPType                 = Protocol.nLTPType;
    m_pFH->nScopeTriggerOut         = Protocol.nScopeTriggerOut;
    m_pFH->nAlternateDACOutputState = Protocol.nAlternateDACOutputState;
    m_pFH->nAlternateDigitalOutputState = Protocol.nAlternateDigitalOutputState;

    bOK &= GetString( Protocol.lFileCommentIndex,        m_pFH->sFileComment,        ELEMENTS_IN( m_pFH->sFileComment ) );

    return bOK;
}

//===============================================================================================
// FUNCTION: ReadADCInfo
// PURPOSE:  Reads the ADC info from the data file.
//
BOOL CABF2ProtocolReader::ReadADCInfo()
{
    MEMBERASSERT();

    short ch  = 0;
    BOOL bOK = TRUE;
    ABF_ADCInfo ADCInfo;
    ASSERT( m_FileInfo.ADCSection.llNumEntries );
    ASSERT( m_FileInfo.ADCSection.uBytes == sizeof( ADCInfo ) );
    bOK &= m_pFI->Seek( LONGLONG(m_FileInfo.ADCSection.uBlockIndex) * ABF_BLOCKSIZE, FILE_BEGIN );
    if( !bOK )
        return FALSE;

    for( int a=0; a<m_FileInfo.ADCSection.llNumEntries; a++ )
    {
        m_pFI->Read( &ADCInfo, sizeof( ADCInfo ) );

      
        // Read the channel.
        ch = ADCInfo.nADCNum;

        if( ADCInfo.nADCNum >= 0 )
        {
            // Setup the sampling sequence array, using the channel sequence index.
            m_pFH->nADCSamplingSeq[a]                = ADCInfo.nADCNum; 

            // Everything else use the channel index.
            m_pFH->nTelegraphEnable[ch]              = ADCInfo.nTelegraphEnable; 
            m_pFH->nTelegraphInstrument[ch]          = ADCInfo.nTelegraphInstrument; 
            m_pFH->fTelegraphAdditGain[ch]           = ADCInfo.fTelegraphAdditGain; 
            m_pFH->fTelegraphFilter[ch]              = ADCInfo.fTelegraphFilter; 
            m_pFH->fTelegraphMembraneCap[ch]         = ADCInfo.fTelegraphMembraneCap; 
            m_pFH->nTelegraphMode[ch]                = ADCInfo.nTelegraphMode; 
            m_pFH->fTelegraphAccessResistance[ch]    = ADCInfo.fTelegraphAccessResistance; 
            m_pFH->nADCPtoLChannelMap[ch]            = ADCInfo.nADCPtoLChannelMap; 
            m_pFH->fADCProgrammableGain[ch]          = ADCInfo.fADCProgrammableGain; 
            m_pFH->fADCDisplayAmplification[ch]      = ADCInfo.fADCDisplayAmplification; 
            m_pFH->fADCDisplayOffset[ch]             = ADCInfo.fADCDisplayOffset; 
            m_pFH->fInstrumentScaleFactor[ch]        = ADCInfo.fInstrumentScaleFactor; 
            m_pFH->fInstrumentOffset[ch]             = ADCInfo.fInstrumentOffset; 
            m_pFH->fSignalGain[ch]                   = ADCInfo.fSignalGain; 
            m_pFH->fSignalOffset[ch]                 = ADCInfo.fSignalOffset; 
            m_pFH->fSignalLowpassFilter[ch]          = ADCInfo.fSignalLowpassFilter; 
            m_pFH->fSignalHighpassFilter[ch]         = ADCInfo.fSignalHighpassFilter; 
            m_pFH->nLowpassFilterType[ch]            = ADCInfo.nLowpassFilterType; 
            m_pFH->nHighpassFilterType[ch]           = ADCInfo.nHighpassFilterType; 
            m_pFH->fPostProcessLowpassFilter[ch]     = ADCInfo.fPostProcessLowpassFilter; 
            m_pFH->nPostProcessLowpassFilterType[ch] = ADCInfo.nPostProcessLowpassFilterType; 
            m_pFH->nStatsChannelPolarity[ch]         = ADCInfo.nStatsChannelPolarity; 
         
            // Set the DAC index if an old P/N file is read.
            // CSH if( ADCInfo.bEnabledDuringPN )
            // CSH    m_pFH->nLeakSubtractADCIndex[0] = ch;

            bOK &= GetString( ADCInfo.lADCChannelNameIndex, m_pFH->sADCChannelName[ADCInfo.nADCNum], ABF_ADCNAMELEN );
            bOK &= GetString( ADCInfo.lADCUnitsIndex,       m_pFH->sADCUnits[ADCInfo.nADCNum],       ABF_ADCUNITLEN );
        }
    }

    return bOK;
}

//===============================================================================================
// FUNCTION: ReadDACInfo
// PURPOSE:  Reads the DAC info from the data file.
//
BOOL CABF2ProtocolReader::ReadDACInfo()
{
    MEMBERASSERT();

    BOOL bOK = TRUE;
    ABF_DACInfo DACInfo;
    ASSERT( m_FileInfo.DACSection.llNumEntries <= ABF_DACCOUNT );
    ASSERT( m_FileInfo.DACSection.uBytes == sizeof( DACInfo ) );
    bOK &= m_pFI->Seek( LONGLONG(m_FileInfo.DACSection.uBlockIndex) * ABF_BLOCKSIZE, FILE_BEGIN );
    if( !bOK )
        return FALSE;

    for( UINT d=0; d<m_FileInfo.DACSection.llNumEntries; d++ )
    {
        bOK &= m_pFI->Read( &DACInfo, sizeof( DACInfo ) );

        //DACInfo.nDACNum                        = d;
        m_pFH->nTelegraphDACScaleFactorEnable[d] = DACInfo.nTelegraphDACScaleFactorEnable;
        m_pFH->fInstrumentHoldingLevel[d]        = DACInfo.fInstrumentHoldingLevel;
        m_pFH->fDACScaleFactor[d]                = DACInfo.fDACScaleFactor;
        m_pFH->fDACHoldingLevel[d]               = DACInfo.fDACHoldingLevel;
        m_pFH->fDACCalibrationFactor[d]          = DACInfo.fDACCalibrationFactor;
        m_pFH->fDACCalibrationOffset[d]          = DACInfo.fDACCalibrationOffset;
        m_pFH->lDACFilePtr[d]                    = DACInfo.lDACFilePtr;
        m_pFH->lDACFileNumEpisodes[d]            = DACInfo.lDACFileNumEpisodes;
        m_pFH->nWaveformEnable[d]                = DACInfo.nWaveformEnable;
        m_pFH->nWaveformSource[d]                = DACInfo.nWaveformSource;
        m_pFH->nInterEpisodeLevel[d]             = DACInfo.nInterEpisodeLevel;
        m_pFH->fDACFileScale[d]                  = DACInfo.fDACFileScale;
        m_pFH->fDACFileOffset[d]                 = DACInfo.fDACFileOffset;
        m_pFH->lDACFileEpisodeNum[d]             = DACInfo.lDACFileEpisodeNum;
        m_pFH->nDACFileADCNum[d]                 = DACInfo.nDACFileADCNum;
        m_pFH->nConditEnable[d]                  = DACInfo.nConditEnable;
        m_pFH->lConditNumPulses[d]               = DACInfo.lConditNumPulses;
        m_pFH->fBaselineDuration[d]              = DACInfo.fBaselineDuration;
        m_pFH->fBaselineLevel[d]                 = DACInfo.fBaselineLevel;
        m_pFH->fStepDuration[d]                  = DACInfo.fStepDuration;
        m_pFH->fStepLevel[d]                     = DACInfo.fStepLevel;
        m_pFH->fPostTrainPeriod[d]               = DACInfo.fPostTrainPeriod;
        m_pFH->fPostTrainLevel[d]                = DACInfo.fPostTrainLevel;
        m_pFH->nMembTestEnable[d]                = DACInfo.nMembTestEnable;
        m_pFH->fMembTestPreSettlingTimeMS[d]     = DACInfo.fMembTestPreSettlingTimeMS;
        m_pFH->fMembTestPostSettlingTimeMS[d]    = DACInfo.fMembTestPostSettlingTimeMS;
        m_pFH->nLeakSubtractType[d]              = DACInfo.nLeakSubtractType;
        m_pFH->nPNPosition                       = DACInfo.nPNPosition;
        m_pFH->nPNNumPulses                      = DACInfo.nPNNumPulses;
        m_pFH->fPNSettlingTime                   = DACInfo.fPNSettlingTime;
        m_pFH->fPNInterpulse                     = DACInfo.fPNInterpulse;
        m_pFH->nPNPolarity                       = DACInfo.nPNPolarity;
        m_pFH->fPNHoldingLevel[d]                = DACInfo.fPNHoldingLevel;
        // CSH m_pFH->nLeakSubtractADCIndex[d]          = DACInfo.nLeakSubtractADCIndex;
        m_pFH->nLTPUsageOfDAC[d]                 = DACInfo.nLTPUsageOfDAC;
        m_pFH->nLTPPresynapticPulses[d]          = DACInfo.nLTPPresynapticPulses;

        bOK &= GetString( DACInfo.lDACChannelNameIndex, m_pFH->sDACChannelName[d], ABF_DACNAMELEN );
        bOK &= GetString( DACInfo.lDACChannelUnitsIndex, m_pFH->sDACChannelUnits[d], ABF_DACUNITLEN );
        bOK &= GetString( DACInfo.lDACFilePathIndex, m_pFH->sDACFilePath[d], ABF_PATHLEN );
    }

    return bOK;
}

//===============================================================================================
// FUNCTION: ReadEpochs
// PURPOSE:  Reads the epochs from the data file.
//
BOOL CABF2ProtocolReader::ReadEpochs()
{
    MEMBERASSERT();

    BOOL bOK = TRUE;

    // Analog Epochs ... one set for each DAC in use.
    if( m_FileInfo.EpochPerDACSection.uBlockIndex )
    {
        ABF_EpochInfoPerDAC Epoch;
        ASSERT( m_FileInfo.EpochPerDACSection.uBytes == sizeof( Epoch ) );
        ASSERT( m_FileInfo.EpochPerDACSection.llNumEntries );
        bOK &= m_pFI->Seek( LONGLONG(m_FileInfo.EpochPerDACSection.uBlockIndex) * ABF_BLOCKSIZE, FILE_BEGIN );
        if( !bOK )
            return FALSE;

        for( long i=0; i<m_FileInfo.EpochPerDACSection.llNumEntries; i++ )
        {
            bOK &= m_pFI->Read( &Epoch, sizeof( Epoch ) );
            ASSERT( Epoch.nEpochType != ABF_EPOCHDISABLED );
            short e = Epoch.nEpochNum;         
            short d = Epoch.nDACNum; 

            m_pFH->nEpochType[d][e]         = Epoch.nEpochType;          
            m_pFH->fEpochInitLevel[d][e]    = Epoch.fEpochInitLevel;     
            m_pFH->fEpochLevelInc[d][e]     = Epoch.fEpochLevelInc;      
            m_pFH->lEpochInitDuration[d][e] = Epoch.lEpochInitDuration;
            m_pFH->lEpochDurationInc[d][e]  = Epoch.lEpochDurationInc;   
            m_pFH->lEpochPulsePeriod[d][e]  = Epoch.lEpochPulsePeriod;   
            m_pFH->lEpochPulseWidth[d][e]   = Epoch.lEpochPulseWidth;    
        }
    }

    // Digital Epochs ... one set only.
    if( m_FileInfo.EpochSection.uBlockIndex )
    {
        ABF_EpochInfo Epoch;
        ASSERT( m_FileInfo.EpochSection.uBytes == sizeof( Epoch ) );
        ASSERT( m_FileInfo.EpochSection.llNumEntries );
        bOK &= m_pFI->Seek( LONGLONG(m_FileInfo.EpochSection.uBlockIndex) * ABF_BLOCKSIZE, FILE_BEGIN );
        if( !bOK )
            return FALSE;

        for( long i=0; i<m_FileInfo.EpochSection.llNumEntries; i++ )
        {
            bOK &= m_pFI->Read( &Epoch, sizeof( Epoch ) );
            short e = Epoch.nEpochNum;         

            m_pFH->nDigitalValue[e]               = Epoch.nDigitalValue;          
            m_pFH->nDigitalTrainValue[e]          = Epoch.nDigitalTrainValue;      
            m_pFH->nAlternateDigitalValue[e]      = Epoch.nAlternateDigitalValue;
            m_pFH->nAlternateDigitalTrainValue[e] = Epoch.nAlternateDigitalTrainValue;   
            m_pFH->bEpochCompression[e]           = Epoch.bEpochCompression;    
        }
    }
    return bOK;
}

//===============================================================================================
// FUNCTION: ReadStats
// PURPOSE:  Reads the Stats regions from the data file.
//
BOOL CABF2ProtocolReader::ReadStats()
{
    MEMBERASSERT();

    BOOL bOK = TRUE;
    if( m_FileInfo.StatsRegionSection.uBlockIndex )
    {
        bOK &= m_pFI->Seek( LONGLONG(m_FileInfo.StatsRegionSection.uBlockIndex) * ABF_BLOCKSIZE, FILE_BEGIN );
        if( !bOK )
            return FALSE;

        for( long i=0; i<m_FileInfo.StatsRegionSection.llNumEntries; i++ )
        {
            ABF_StatsRegionInfo Stats = ABF_StatsRegionInfo();
            ASSERT( m_FileInfo.StatsRegionSection.uBytes == sizeof( Stats ) );
            ASSERT( m_FileInfo.StatsRegionSection.llNumEntries );

            bOK &= m_pFI->Read( &Stats, sizeof( Stats ) );

            short r = Stats.nRegionNum;         
            UINT uBitMask = 0x01 << r;
            m_pFH->nStatsSearchRegionFlags |= uBitMask;

            m_pFH->lStatsMeasurements[r]     = Stats.lStatsMeasurements;    
            m_pFH->lStatsStart[r]            = Stats.lStatsStart;           
            m_pFH->lStatsEnd[r]              = Stats.lStatsEnd;             
            m_pFH->nRiseTopPercentile[r]     = Stats.nRiseTopPercentile;    
            m_pFH->nRiseBottomPercentile[r]  = Stats.nRiseBottomPercentile; 
            m_pFH->nDecayBottomPercentile[r] = Stats.nDecayBottomPercentile;
            m_pFH->nDecayTopPercentile[r]    = Stats.nDecayTopPercentile;   
            m_pFH->nStatsSearchMode[r]       = Stats.nStatsSearchMode; 
            m_pFH->nStatsSearchDAC[r]        = Stats.nStatsSearchDAC; 

            m_pFH->nStatsActiveChannels      = Stats.nStatsActiveChannels;
            m_pFH->nStatsSearchRegionFlags   = Stats.nStatsSearchRegionFlags;
            m_pFH->nStatsSmoothing           = Stats.nStatsSmoothing;
            m_pFH->nStatsSmoothingEnable     = Stats.nStatsSmoothingEnable;
            m_pFH->nStatsBaseline            = Stats.nStatsBaseline;
            m_pFH->nStatsBaselineDAC         = Stats.nStatsBaselineDAC;
            m_pFH->lStatsBaselineStart       = Stats.lStatsBaselineStart;
            m_pFH->lStatsBaselineEnd         = Stats.lStatsBaselineEnd;

            // Some early ABF 2 protocols did not use the "DAC" field, so coerce these.
            if( Stats.nStatsSearchMode >= ABF_EPOCHCOUNT )
            {
                m_pFH->nStatsSearchMode[r] = Stats.nStatsSearchMode % ABF_EPOCHCOUNT;
                m_pFH->nStatsSearchDAC[r]  = Stats.nStatsSearchMode / ABF_EPOCHCOUNT;
            }

            if( Stats.nStatsBaseline >= ABF_EPOCHCOUNT )
            {
                m_pFH->nStatsBaseline    = Stats.nStatsBaseline % ABF_EPOCHCOUNT;
                m_pFH->nStatsBaselineDAC = Stats.nStatsBaseline / ABF_EPOCHCOUNT;
            }
        }
    }
    return bOK;
}

//===============================================================================================
// FUNCTION: ReadUserList
// PURPOSE:  Reads the user list from the data file.
//
BOOL CABF2ProtocolReader::ReadUserList()
{
    MEMBERASSERT();

    BOOL bOK = TRUE;
    if( m_FileInfo.UserListSection.uBlockIndex )
    {
        ABF_UserListInfo UserList;
        ASSERT( m_FileInfo.UserListSection.uBytes == sizeof( UserList ) );
        ASSERT( m_FileInfo.UserListSection.llNumEntries );
        bOK &= m_pFI->Seek( LONGLONG(m_FileInfo.UserListSection.uBlockIndex) * ABF_BLOCKSIZE, FILE_BEGIN );
        if( !bOK )
            return FALSE;

        for( long i=0; i<m_FileInfo.UserListSection.llNumEntries; i++ )
        {
            bOK &= m_pFI->Read( &UserList, sizeof( UserList ) );
            short u = UserList.nListNum;         

            m_pFH->nULEnable[u]      = 1;    
            m_pFH->nULParamToVary[u] = UserList.nULParamToVary;           
            m_pFH->nULRepeat[u]      = UserList.nULRepeat; 

            bOK &= GetString( UserList.lULParamValueListIndex, m_pFH->sULParamValueList[u], ABF_USERLISTLEN );
        }
    }
    return bOK;
}

//===============================================================================================
// FUNCTION: ReadMathInfo
// PURPOSE:  Read the math channel info to the data file.
// NOTES:    We currently only support one math channel, but the file can support any number.
//
BOOL CABF2ProtocolReader::ReadMathInfo()
{
    MEMBERASSERT();

    BOOL bOK = TRUE;
    if( m_FileInfo.MathSection.uBlockIndex )
    {
        ABF_MathInfo Math;
        ASSERT( m_FileInfo.MathSection.uBytes == sizeof( ABF_MathInfo ) );
        ASSERT( m_FileInfo.MathSection.llNumEntries );
        bOK &= m_pFI->Seek( LONGLONG(m_FileInfo.MathSection.uBlockIndex) * ABF_BLOCKSIZE, FILE_BEGIN );
        if( !bOK )
            return FALSE;

        bOK &= m_pFI->Read( &Math, sizeof( Math ) );

        m_pFH->nArithmeticEnable     = Math.nMathEnable;
        m_pFH->nArithmeticExpression = Math.nMathExpression;   
        m_pFH->fArithmeticUpperLimit = Math.fMathUpperLimit;   
        m_pFH->fArithmeticLowerLimit = Math.fMathLowerLimit; 

        m_pFH->nArithmeticADCNumA    = Math.nMathADCNum[0];   
        m_pFH->nArithmeticADCNumB    = Math.nMathADCNum[1];   

        m_pFH->fArithmeticK1         = Math.fMathK[0];     
        m_pFH->fArithmeticK2         = Math.fMathK[1];     
        m_pFH->fArithmeticK3         = Math.fMathK[2];     
        m_pFH->fArithmeticK4         = Math.fMathK[3];     
        m_pFH->fArithmeticK5         = Math.fMathK[4];     
        m_pFH->fArithmeticK6         = Math.fMathK[5];    

        GetString( Math.uMathOperatorIndex, m_pFH->sArithmeticOperator, sizeof( m_pFH->sArithmeticOperator ) );
        GetString( Math.uMathUnitsIndex, m_pFH->sArithmeticUnits, sizeof( m_pFH->sArithmeticUnits ) );
    }

    return bOK;
}

#if 0
//===============================================================================================
// FUNCTION: ValidateCRC
// PURPOSE:  Validates the CRC in the FileInfo matches the CRC of the file.
//
BOOL CABF2ProtocolReader::ValidateCRC()
{
    MEMBERASSERT();

    if( m_pFH->nCRCEnable != ABF_CRC_ENABLED )
        return TRUE;
   
    // CRC checking required.
    Read( 0 );
   
#if _DEBUG
    // Get the total length of the file.
#ifdef _DEBUG
    LONGLONG llFileLength = m_pFI->GetFileSize();
   
    UINT uHeaderSize = sizeof( m_FileInfo );
    ASSERT( llFileLength > uHeaderSize );
#endif   // _DEBUG
#endif

    // Keep expected CRC value from header and Zero the lFileCRC.
    UINT uExpectedCRC = m_FileInfo.uFileCRC;
    m_FileInfo.uFileCRC = 0;

    UINT uFileCRC = CalculateCRC( &m_FileInfo, sizeof( m_FileInfo ) );

    // Restore the original CRC.
    m_FileInfo.uFileCRC = uExpectedCRC;

    // Compare expected CRC with file CRC.
    if ( uFileCRC != uExpectedCRC )
    {
        TRACE2( "File CRC Validation Failed: Expected %X, Calculated %X\n", uExpectedCRC, uFileCRC );
        return FALSE;
    }

    TRACE1( "File CRC Validation OK: %X\n", uFileCRC);
    return TRUE;
}
#endif

BOOL CABF2ProtocolReader::Open( LPCTSTR fName ) {
    
    int nError = 0;
    
    // Get a new file descriptor if available.
    if (!GetNewFileDescriptor(&m_pFI, &nFile, &nError))
        return FALSE;
    
    // Now open the file for reading.
    if (!m_pFI->Open(fName, TRUE)) {
        return FALSE;
    }
    return TRUE;
}

BOOL CABF2ProtocolReader::Close( ) {
    int nError = 0;
    CFileDescriptor *pFI = NULL;
    if (!GetFileDescriptor(&pFI, nFile, &nError))
    {
        return FALSE;
    }
    
    ReleaseFileDescriptor( nFile );
    return TRUE;
}

//===============================================================================================
// FUNCTION: CanOpen
// PURPOSE:  Returns TRUE if this reader can open the file
//
BOOL CABF2ProtocolReader::CanOpen( const void *pFirstBlock, UINT uBytes )
{
   ASSERT( pFirstBlock );
   ASSERT( uBytes >= sizeof( ABF2_FileInfo ) );

   ABF2_FileInfo *pInfo = (ABF2_FileInfo *)pFirstBlock;

   // Check if it has the correct signature.
   if( pInfo->uFileSignature != ABF2_FILESIGNATURE )
      return FALSE;

   // Check the major file version
   BYTE byMajorVersion = HIBYTE( HIWORD(  pInfo->uFileVersionNumber ) );
   if( byMajorVersion == 2 )
      return TRUE;

   return FALSE;
}
