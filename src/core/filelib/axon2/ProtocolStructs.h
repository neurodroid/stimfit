//***********************************************************************************************
//
//    Copyright (c) 1993-2005 Molecular Devices.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
//
// MODULE:     ProtocolStructs.HPP
// ABF structs: used to describe the actual file contents.
//
// Added 64bit support according to Jakub Nowacki's implementation in libaxon:
// http://libaxon.sourceforge.net

#ifndef INC_PROTOCOLSTRUCTS_HPP
#define INC_PROTOCOLSTRUCTS_HPP

#include "../axon/Common/axodebug.h"
#include "../axon/AxAbfFio32/AxAbffio32.h"

#include <iostream>

#pragma once
#pragma pack(push, 1)

// GUID is normally defined in the Windows Platform SDK 
#ifndef GUID_DEFINED
#define GUID_DEFINED
typedef struct _GUID
{
   akxjsbasd
    unsigned ABFLONG  Data1;
    unsigned short Data2;
    unsigned short Data3;
    unsigned char  Data4[8];
} GUID;
#endif /* GUID_DEFINED */


// All these structs are persisted to file -> their sizes must NOT be changed without careful 
// attention to versioning issues in order to maintain compatibility.

struct ABF_Section
{
   UINT     uBlockIndex;            // ABF block number of the first entry
   UINT     uBytes;                 // size in bytes of of each entry
   long long llNumEntries;           // number of entries in this section

   ABF_Section();
   ABFLONG GetNumEntries();
   void Set( const UINT p_uBlockIndex, const UINT p_uBytes, const long long p_llNumEntries );

};

#define MEMSET_CTOR

inline ABF_Section::ABF_Section() { MEMSET_CTOR; }

inline void ABF_Section::Set( const UINT p_uBlockIndex, const UINT p_uBytes, const long long p_llNumEntries )
{
   uBytes       = 0;
   llNumEntries = 0;
   uBlockIndex  = p_uBlockIndex;
   if( uBlockIndex )
   {
      uBytes       = p_uBytes;
      llNumEntries = p_llNumEntries;
   }
}

inline ABFLONG ABF_Section::GetNumEntries()
{
   // If this assert goes off, then files longer than 2 gigasamples need to be handled.
   if( llNumEntries > LONG_MAX )
   {
       std::cerr << "File contains" << (int)(llNumEntries / 1000000L) 
                 << "megasamples which exceeds current limit (" << (int)(LONG_MAX / 1000000L) << ").";
   }

   return ABFLONG(llNumEntries);
}

#define ABF2_FILESIGNATURE   0x32464241      // PC="ABF2", MAC="2FBA"

struct ABF2_FileInfo
{
   UINT  uFileSignature;
   UINT  uFileVersionNumber;

   // After this point there is no need to be the same as the ABF 1 equivalent.
   UINT  uFileInfoSize;

   UINT  uActualEpisodes;
   UINT  uFileStartDate;
   UINT  uFileStartTimeMS;
   UINT  uStopwatchTime;
   short nFileType;
   short nDataFormat;
   short nSimultaneousScan;
   short nCRCEnable;
   UINT  uFileCRC;
   GUID  FileGUID;
   UINT  uCreatorVersion;
   UINT  uCreatorNameIndex;
   UINT  uModifierVersion;
   UINT  uModifierNameIndex;
   UINT  uProtocolPathIndex;   

   // New sections in ABF 2 - protocol stuff ...
   ABF_Section ProtocolSection;           // the protocol
   ABF_Section ADCSection;                // one for each ADC channel
   ABF_Section DACSection;                // one for each DAC channel
   ABF_Section EpochSection;              // one for each epoch
   ABF_Section ADCPerDACSection;          // one for each ADC for each DAC
   ABF_Section EpochPerDACSection;        // one for each epoch for each DAC
   ABF_Section UserListSection;           // one for each user list
   ABF_Section StatsRegionSection;        // one for each stats region
   ABF_Section MathSection;
   ABF_Section StringsSection;

   // ABF 1 sections ...
   ABF_Section DataSection;            // Data
   ABF_Section TagSection;             // Tags
   ABF_Section ScopeSection;           // Scope config
   ABF_Section DeltaSection;           // Deltas
   ABF_Section VoiceTagSection;        // Voice Tags
   ABF_Section SynchArraySection;      // Synch Array
   ABF_Section AnnotationSection;      // Annotations
   ABF_Section StatsSection;           // Stats config
   
   char  sUnused[148];     // size = 512 bytes
   
   ABF2_FileInfo() 
   { 
      MEMSET_CTOR;
      ASSERT( sizeof( ABF2_FileInfo ) == 512 );

      uFileSignature = ABF2_FILESIGNATURE;
      uFileInfoSize  = sizeof( ABF2_FileInfo);
   }

};

struct ABF_ProtocolInfo
{
   short nOperationMode;
   float fADCSequenceInterval;
   bool  bEnableFileCompression;
   char  sUnused1[3];
   UINT  uFileCompressionRatio;

   float fSynchTimeUnit;
   float fSecondsPerRun;
   ABFLONG  lNumSamplesPerEpisode;
   ABFLONG  lPreTriggerSamples;
   ABFLONG  lEpisodesPerRun;
   ABFLONG  lRunsPerTrial;
   ABFLONG  lNumberOfTrials;
   short nAveragingMode;
   short nUndoRunCount;
   short nFirstEpisodeInRun;
   float fTriggerThreshold;
   short nTriggerSource;
   short nTriggerAction;
   short nTriggerPolarity;
   float fScopeOutputInterval;
   float fEpisodeStartToStart;
   float fRunStartToStart;
   ABFLONG  lAverageCount;
   float fTrialStartToStart;
   short nAutoTriggerStrategy;
   float fFirstRunDelayS;

   short nChannelStatsStrategy;
   ABFLONG  lSamplesPerTrace;
   ABFLONG  lStartDisplayNum;
   ABFLONG  lFinishDisplayNum;
   short nShowPNRawData;
   float fStatisticsPeriod;
   ABFLONG  lStatisticsMeasurements;
   short nStatisticsSaveStrategy;

   float fADCRange;
   float fDACRange;
   ABFLONG  lADCResolution;
   ABFLONG  lDACResolution;
   
   short nExperimentType;
   short nManualInfoStrategy;
   short nCommentsEnable;
   ABFLONG  lFileCommentIndex;            
   short nAutoAnalyseEnable;
   short nSignalType;

   short nDigitalEnable;
   short nActiveDACChannel;
   short nDigitalHolding;
   short nDigitalInterEpisode;
   short nDigitalDACChannel;
   short nDigitalTrainActiveLogic;

   short nStatsEnable;
   short nStatisticsClearStrategy;

   short nLevelHysteresis;
   ABFLONG  lTimeHysteresis;
   short nAllowExternalTags;
   short nAverageAlgorithm;
   float fAverageWeighting;
   short nUndoPromptStrategy;
   short nTrialTriggerSource;
   short nStatisticsDisplayStrategy;
   short nExternalTagType;
   short nScopeTriggerOut;

   short nLTPType;
   short nAlternateDACOutputState;
   short nAlternateDigitalOutputState;

   float fCellID[3];

   short nDigitizerADCs;
   short nDigitizerDACs;
   short nDigitizerTotalDigitalOuts;
   short nDigitizerSynchDigitalOuts;
   short nDigitizerType;

   char  sUnused[304];     // size = 512 bytes
   
   ABF_ProtocolInfo() 
   { 
      MEMSET_CTOR; 
      ASSERT( sizeof( ABF_ProtocolInfo ) == 512 );
   }
};

struct ABF_MathInfo
{
   short nMathEnable;
   short nMathExpression;
   UINT  uMathOperatorIndex;     
   UINT  uMathUnitsIndex;        
   float fMathUpperLimit;
   float fMathLowerLimit;
   short nMathADCNum[2];
   char  sUnused[16];
   float fMathK[6];

   char  sUnused2[64];     // size = 128 bytes
   
   ABF_MathInfo()
   { 
      MEMSET_CTOR; 
      ASSERT( sizeof( ABF_MathInfo ) == 128 );
   }
};

struct ABF_ADCInfo
{
   // The ADC this struct is describing.
   short nADCNum;

   short nTelegraphEnable;
   short nTelegraphInstrument;
   float fTelegraphAdditGain;
   float fTelegraphFilter;
   float fTelegraphMembraneCap;
   short nTelegraphMode;
   float fTelegraphAccessResistance;

   short nADCPtoLChannelMap;
   short nADCSamplingSeq;

   float fADCProgrammableGain;
   float fADCDisplayAmplification;
   float fADCDisplayOffset;
   float fInstrumentScaleFactor;
   float fInstrumentOffset;
   float fSignalGain;
   float fSignalOffset;
   float fSignalLowpassFilter;
   float fSignalHighpassFilter;

   char  nLowpassFilterType;
   char  nHighpassFilterType;
   float fPostProcessLowpassFilter;
   char  nPostProcessLowpassFilterType;
   bool  bEnabledDuringPN;

   short nStatsChannelPolarity;

   ABFLONG  lADCChannelNameIndex;
   ABFLONG  lADCUnitsIndex;

   char  sUnused[46];         // size = 128 bytes
   
   ABF_ADCInfo()
   { 
      MEMSET_CTOR; 
      ASSERT( sizeof( ABF_ADCInfo ) == 128 );
   }
};

struct ABF_DACInfo
{
   // The DAC this struct is describing.
   short nDACNum;

   short nTelegraphDACScaleFactorEnable;
   float fInstrumentHoldingLevel;

   float fDACScaleFactor;
   float fDACHoldingLevel;
   float fDACCalibrationFactor;
   float fDACCalibrationOffset;

   ABFLONG  lDACChannelNameIndex;
   ABFLONG  lDACChannelUnitsIndex;

   ABFLONG  lDACFilePtr;
   ABFLONG  lDACFileNumEpisodes;

   short nWaveformEnable;
   short nWaveformSource;
   short nInterEpisodeLevel;

   float fDACFileScale;
   float fDACFileOffset;
   ABFLONG  lDACFileEpisodeNum;
   short nDACFileADCNum;

   short nConditEnable;
   ABFLONG  lConditNumPulses;
   float fBaselineDuration;
   float fBaselineLevel;
   float fStepDuration;
   float fStepLevel;
   float fPostTrainPeriod;
   float fPostTrainLevel;
   short nMembTestEnable;

   short nLeakSubtractType;
   short nPNPolarity;
   float fPNHoldingLevel;
   short nPNNumADCChannels;
   short nPNPosition;
   short nPNNumPulses;
   float fPNSettlingTime;
   float fPNInterpulse;

   short nLTPUsageOfDAC;
   short nLTPPresynapticPulses;

   ABFLONG  lDACFilePathIndex;

   float fMembTestPreSettlingTimeMS;
   float fMembTestPostSettlingTimeMS;

   short nLeakSubtractADCIndex;

   char  sUnused[124];     // size = 256 bytes
   
   ABF_DACInfo()
   { 
      MEMSET_CTOR; 
      ASSERT( sizeof( ABF_DACInfo ) == 256 );
   }
};

struct ABF_EpochInfoPerDAC
{
   // The Epoch / DAC this struct is describing.
   short nEpochNum;
   short nDACNum;

   // One full set of epochs (ABF_EPOCHCOUNT) for each DAC channel ...
   short nEpochType;
   float fEpochInitLevel;
   float fEpochLevelInc;
   ABFLONG  lEpochInitDuration;  
   ABFLONG  lEpochDurationInc;
   ABFLONG  lEpochPulsePeriod;
   ABFLONG  lEpochPulseWidth;

   char  sUnused[18];      // size = 48 bytes
   
   ABF_EpochInfoPerDAC()
   { 
      MEMSET_CTOR; 
      ASSERT( sizeof( ABF_EpochInfoPerDAC ) == 48 );
   }
};

struct ABF_EpochInfo
{
   // The Epoch this struct is describing.
   short nEpochNum;

   // Describes one epoch
   short nDigitalValue;
   short nDigitalTrainValue;
   short nAlternateDigitalValue;
   short nAlternateDigitalTrainValue;
   bool  bEpochCompression;   // Compress the data from this epoch using uFileCompressionRatio

   char  sUnused[21];      // size = 32 bytes
   
   ABF_EpochInfo()
   { 
      MEMSET_CTOR; 
      ASSERT( sizeof( ABF_EpochInfo ) == 32 );
   }
};

struct ABF_StatsRegionInfo
{ 
   // The stats region this struct is describing.
   short nRegionNum;
   short nADCNum;

   short nStatsActiveChannels;
   short nStatsSearchRegionFlags;
   short nStatsSelectedRegion;
   short nStatsSmoothing;
   short nStatsSmoothingEnable;
   short nStatsBaseline;
   ABFLONG  lStatsBaselineStart;
   ABFLONG  lStatsBaselineEnd;

   // Describes one stats region
   ABFLONG  lStatsMeasurements;
   ABFLONG  lStatsStart;
   ABFLONG  lStatsEnd;
   short nRiseBottomPercentile;
   short nRiseTopPercentile;
   short nDecayBottomPercentile;
   short nDecayTopPercentile;
   short nStatsSearchMode;
   short nStatsSearchDAC;
   short nStatsBaselineDAC;

   char  sUnused[78];   // size = 128 bytes
   
   ABF_StatsRegionInfo()
   { 
      MEMSET_CTOR; 
      ASSERT( sizeof( ABF_StatsRegionInfo ) == 128 );
   }
};

struct ABF_UserListInfo
{
   // The user list this struct is describing.
   short nListNum;

   // Describes one user list
   short nULEnable;
   short nULParamToVary;
   short nULRepeat;
   ABFLONG  lULParamValueListIndex;

   char  sUnused[52];   // size = 64 bytes
   
   ABF_UserListInfo()
   { 
      MEMSET_CTOR; 
      ASSERT( sizeof( ABF_UserListInfo ) == 64 );
   }
};

#pragma pack(pop)                      // return to default packing

#endif   // INC_PROTOCOLSTRUCTS_HPP
