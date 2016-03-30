//
// This file is part of the Axon Library.
//
// Copyright (c) 2008-2009 Jakub Nowacki
//
// The Axon Binary Format is property of Molecular Devices.
// All rights to the Axon Binary Format are reserved to Molecular Devices.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301, USA
//

/*
    modified for use with biosig 
    Copyright (C) 2013 Alois Schlögl, 
    This file is part of the "BioSig for C/C++" repository 
    (biosig4c++) at http://biosig.sf.net/ 

    BioSig is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>. 
*/    

/*! \file
 *  \brief Header containing all structures of ABF
 */

#ifndef INC_PROTOCOLSTRUCTS_HPP
#define INC_PROTOCOLSTRUCTS_HPP

#pragma once
#pragma pack(push, 1)

//#define UINT unsigned int
//#define LONGLONG long long
#define bool int


// GUID is normally defined in the Windows Platform SDK
struct GUID
{
    uint32_t Data1;
    unsigned short Data2;
    unsigned short Data3;
    unsigned char  Data4[8];
};


// All these structs are persisted to file -> their sizes must NOT be changed without careful
// attention to versioning issues in order to maintain compatibility.

typedef struct ABF_Section
{
   unsigned int     uBlockIndex;            // ABF block number of the first entry
   unsigned int     uBytes;                 // size in bytes of of each entry
   long long llNumEntries;           // number of entries in this section
} ABF_Section; 

#define ABF_FILESIGNATURE   0x32464241      // PC="ABF2", MAC="2FBA"

struct ABF_FileInfo
{
   unsigned int  uFileSignature;
   unsigned int  uFileVersionNumber;

   // After this point there is no need to be the same as the ABF 1 equivalent.
   unsigned int  uFileInfoSize;

   unsigned int  uActualEpisodes;
   unsigned int  uFileStartDate;
   unsigned int  uFileStartTimeMS;
   unsigned int  uStopwatchTime;
   short nFileType;
   short nDataFormat;
   short nSimultaneousScan;
   short nCRCEnable;
   unsigned int  uFileCRC;
   struct GUID  FileGUID;
   unsigned int  uCreatorVersion;
   unsigned int  uCreatorNameIndex;
   unsigned int  uModifierVersion;
   unsigned int  uModifierNameIndex;
   unsigned int  uProtocolPathIndex;

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
};

struct ABF_ProtocolInfo
{
   short nOperationMode;
   float fADCSequenceInterval;
   bool  bEnableFileCompression;
   char  sUnused1[3];
   unsigned int  uFileCompressionRatio;

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
};

struct ABF_MathInfo
{
   short nMathEnable;
   short nMathExpression;
   unsigned int  uMathOperatorIndex;
   unsigned int  uMathUnitsIndex;
   float fMathUpperLimit;
   float fMathLowerLimit;
   short nMathADCNum[2];
   char  sUnused[16];
   float fMathK[6];

   char  sUnused2[64];     // size = 128 bytes
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
};

struct ABF_SynchArray
{
	ABFLONG lStart;
	ABFLONG lLength;
};

// Strings section structure not defined by Axon

#pragma pack(pop)                      // return to default packing

#endif   // INC_PROTOCOLSTRUCTS_HPP
