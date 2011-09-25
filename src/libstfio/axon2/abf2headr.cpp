//***********************************************************************************************
//
//    Copyright (c) 1993-2000 Axon Instruments.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
// This is ABF2HEADR.CPP; the routines that cope with reading the data file
// parameters block for all AXON pCLAMP binary file formats.
//
// An ANSI C compiler should be used for compilation.
// Compile with the large memory model option.
// (e.g. CL -c -AL ABFHEADR.C)

#include "../axon/Common/wincpp.hpp"
#include "abf2headr.h"
#include "../axon/AxAbfFio32/abfheadr.h"
#include "../axon/AxAbfFio32/abfutil.h"

//===============================================================================================
// FUNCTION: ABFH_Initialize
// PURPOSE:  Initialize an ABFFileHeader structure to a consistent set of parameters
//
void WINAPI ABF2H_Initialize( ABF2FileHeader *pFH )
{
   int i;

   // Zero fill all to start with.
   memset(pFH, '\0', sizeof(*pFH));
   
   // Blank fill all strings.
   ABF_BLANK_FILL(pFH->sADCChannelName);
   ABF_BLANK_FILL(pFH->sADCUnits);
   ABF_BLANK_FILL(pFH->sDACChannelName);
   ABF_BLANK_FILL(pFH->sDACChannelUnits);
   ABF_BLANK_FILL(pFH->sDACFilePath[0]);
   ABF_BLANK_FILL(pFH->sDACFilePath[1]);
   ABF_SET_STRING(pFH->sArithmeticOperator, "+");
   ABF_BLANK_FILL(pFH->sArithmeticUnits);
      
   pFH->fFileVersionNumber    = ABF2_CURRENTVERSION;
   pFH->fHeaderVersionNumber  = ABF2_CURRENTVERSION;
   pFH->nOperationMode        = ABF2_GAPFREEFILE;
   pFH->nADCNumChannels       = 1;
   pFH->fADCSequenceInterval  = 100.0F;
   pFH->lNumSamplesPerEpisode = 512;
   pFH->lEpisodesPerRun       = 1;
   pFH->lDataSectionPtr       = sizeof(ABFFileHeader) / ABF2_BLOCKSIZE;
      
   pFH->nDataDisplayMode      = ABF2_DRAW_LINES;
   pFH->nFileType             = ABF2_ABFFILE;
   pFH->nAutoTriggerStrategy  = 1;   // Allow auto triggering.
   pFH->nChannelStatsStrategy = 0;   // Don't calculate channel statistics.
   pFH->fStatisticsPeriod     = 1.0F;
   pFH->lStatisticsMeasurements = ABF2_STATISTICS_ABOVETHRESHOLD | ABF2_STATISTICS_MEANOPENTIME;
   
   pFH->lSamplesPerTrace      = 16384;
   pFH->lPreTriggerSamples    = 16;    // default to 16

   pFH->fADCRange             = 10.24F;
   pFH->fDACRange             = 10.24F;
   pFH->lADCResolution        = 32768L;
   pFH->lDACResolution        = 32768L;
   pFH->nExperimentType       = ABF2_SIMPLEACQUISITION;
   
      
   ABF_BLANK_FILL(pFH->sCreatorInfo);
   ABF_BLANK_FILL(pFH->sModifierInfo);
   ABF_BLANK_FILL(pFH->sFileComment);
      
   // ADC channel data
   for (i=0; i<ABF2_ADCCOUNT; i++)
   {
      char szName[13];      
      sprintf(szName, "AI #%-8d", i);
      strncpy(pFH->sADCChannelName[i], szName, ABF2_ADCNAMELEN);
      strncpy(pFH->sADCUnits[i], "pA        ", ABF2_ADCUNITLEN);
      
      pFH->nADCPtoLChannelMap[i]       = short(i);
      pFH->nADCSamplingSeq[i]          = ABF2_UNUSED_CHANNEL;
      pFH->fADCProgrammableGain[i]     = 1.0F;
      pFH->fADCDisplayAmplification[i] = 1.0F;
      pFH->fInstrumentScaleFactor[i]   = 0.1F;
      pFH->fSignalGain[i]              = 1.0F;
      pFH->fSignalLowpassFilter[i]     = ABF2_FILTERDISABLED;

// FIX FIX FIX PRC DEBUG Telegraph changes - check !
      pFH->fTelegraphAdditGain[i]      = 1.0F;
      pFH->fTelegraphFilter[i]         = 100000.0F;
   }
   pFH->nADCSamplingSeq[0] = 0;
      
   // DAC channel data
   for (i=0; i<ABF2_DACCOUNT; i++)
   {
      char szName[13];
      sprintf(szName, "AO #%-8d", i);
      strncpy(pFH->sDACChannelName[i], szName, ABF2_DACNAMELEN);
      strncpy(pFH->sDACChannelUnits[i], "mV        ", ABF2_ADCUNITLEN);
      pFH->fDACScaleFactor[i] = 20.0F;
   }
   
   // DAC file settings
   for (i=0; i<ABF2_DACCOUNT; i++)
   {
      pFH->fDACFileScale[i] = 1.0F;
   }
   pFH->nPNPolarity   = ABF2_PN_SAME_POLARITY;

   pFH->nPNNumPulses        = 2;
   pFH->fPNInterpulse       = 0;

   // Initialize as non-zero to avoid glitch in first holding
   pFH->fPNSettlingTime     = 10;
   for (i=0; i<ABF2_DACCOUNT; i++)
      pFH->fPostTrainPeriod[i] = 10;

   // Initialize statistics variables.
   pFH->nStatsSearchRegionFlags = ABF2_PEAK_SEARCH_REGION0;
   pFH->nStatsBaseline          = ABF2_PEAK_BASELINE_SPECIFIED;
   pFH->nStatsSmoothing         = 1;
   pFH->nStatsActiveChannels    = 0;
   for( int nStatsRegionID = 0; nStatsRegionID < ABF2_STATS_REGIONS; nStatsRegionID++ )
   {
      pFH->nStatsSearchMode[ nStatsRegionID ]       = ABF2_PEAK_SEARCH_SPECIFIED;
      pFH->lStatsMeasurements[ nStatsRegionID ]     = ABF2_PEAK_MEASURE_PEAK | ABF2_PEAK_MEASURE_PEAKTIME;
      pFH->nRiseBottomPercentile[ nStatsRegionID ]  = 10;
      pFH->nRiseTopPercentile[ nStatsRegionID ]     = 90;
      pFH->nDecayBottomPercentile[ nStatsRegionID ] = 10;
      pFH->nDecayTopPercentile[ nStatsRegionID ]    = 90;   
   }

   for ( UINT uChannel = 0; uChannel < ABF2_ADCCOUNT; uChannel++ )
      pFH->nStatsChannelPolarity[uChannel] = ABF2_PEAK_ABSOLUTE;

   pFH->fArithmeticUpperLimit = 100.0F;
   pFH->fArithmeticLowerLimit = -100.0F;
   pFH->fArithmeticK1         = 1.0F;
   pFH->fArithmeticK3         = 1.0F;

   pFH->nLevelHysteresis    = 64;   // Two LSBits of level hysteresis.
   pFH->lTimeHysteresis     = 1;    // Two sequences of time hysteresis.
   pFH->fAverageWeighting   = 0.1F;                       // Add 10% of trace to 90% of average.
   pFH->nTrialTriggerSource = ABF2_TRIALTRIGGER_NONE;
   pFH->nExternalTagType    = ABF2_EXTERNALTAG;

   pFH->nAutoAnalyseEnable  = ABF2_AUTOANALYSE_DEFAULT;

   for( i=0; i<ABF2_USERLISTCOUNT; i++ )
      ABF_BLANK_FILL( pFH->sULParamValueList[i] );

   // DAC Calibration Factors.
   for( i=0; i<ABF2_DACCOUNT; i++ )
   {
      pFH->fDACCalibrationFactor[i] = 1.0F;
      pFH->fDACCalibrationOffset[i] = 0.0F;
   }

   // Digital train params.
   pFH->nDigitalTrainActiveLogic = 1;
   for( i = 0; i < ABF2_EPOCHCOUNT; i ++ )
   {
      pFH->nDigitalTrainValue[ i ] = 0;
   }

   // Initialize LTP type.
   pFH->nLTPType = ABF2_LTP_TYPE_NONE;
   for( i=0; i<ABF2_DACCOUNT; i++ )
   {
      pFH->nLTPUsageOfDAC[ i ] = ABF2_LTP_DAC_USAGE_NONE;
      pFH->nLTPPresynapticPulses[ i ] = 0;
   }

   // Alternating Outputs 
   pFH->nAlternateDACOutputState = 0;
   pFH->nAlternateDigitalOutputState = 0;
   for( int nEpoch = 0; nEpoch  < ABF2_EPOCHCOUNT; nEpoch ++ )
   {
      pFH->nAlternateDigitalValue[ nEpoch ] = 0;
      pFH->nAlternateDigitalTrainValue[ nEpoch ] = 0;
   }

   //Post-processing values.
   for( i=0; i<ABF2_ADCCOUNT; i++)
   {
      pFH->fPostProcessLowpassFilter[i] = ABF2_FILTERDISABLED;
      pFH->nPostProcessLowpassFilterType[i] = ABF2_POSTPROCESS_FILTER_NONE;
   }

}

//===============================================================================================
// FUNCTION: ABFH_GetChannelOffset
// PURPOSE:  Get the offset in the sampling sequence for the given physical channel.
//
BOOL WINAPI ABF2H_GetChannelOffset( const ABF2FileHeader *pFH, int nChannel, UINT *puChannelOffset )
{
//   ABFH_ASSERT(pFH);
//   WPTRASSERT(puChannelOffset);

   int nOffset;

   // check the ADC channel number, -1 refers to the math channel

   if (nChannel < 0)
   {
      if (!pFH->nArithmeticEnable)
      {
         if (puChannelOffset)
            *puChannelOffset = 0;   // return the offset to this channel
         return FALSE;              // channel not found in sampling sequence
      }
      nChannel = pFH->nArithmeticADCNumA;
   }

   for (nOffset = 0; nOffset < pFH->nADCNumChannels; nOffset++)
   {
      if (pFH->nADCSamplingSeq[nOffset] == nChannel)
      {
         if (puChannelOffset)
            *puChannelOffset = UINT(nOffset);  // return the offset to this channel
         return TRUE;
      }
   }

   if (puChannelOffset)
      *puChannelOffset = 0;  // return the offset to this channel
   return FALSE;
}

//==============================================================================================
// FUNCTION:   GetADCtoUUFactors
// PURPOSE:    Calculates the scaling factors used to convert ADC values to UserUnits.
// PARAMETERS:
//    nChannel        - The physical channel number to get the factors for.
//    pfADCToUUFactor - Pointers to return locations for scale and offset.
//    pfADCToUUShift    UserUnits = ADCValue * fADCToUUFactor + fADCToUUShift;
//
void WINAPI ABF2H_GetADCtoUUFactors( const ABF2FileHeader *pFH, int nChannel, 
                                    float *pfADCToUUFactor, float *pfADCToUUShift )
{
   ASSERT(nChannel < ABF2_ADCCOUNT);

   float fTotalScaleFactor = pFH->fInstrumentScaleFactor[nChannel] *
                             pFH->fADCProgrammableGain[nChannel];
   if (pFH->nSignalType != 0)
      fTotalScaleFactor *= pFH->fSignalGain[nChannel];

   // Adjust for the telegraphed gain.
   if( pFH->nTelegraphEnable[nChannel] )
      fTotalScaleFactor *= pFH->fTelegraphAdditGain[nChannel];

   ASSERT(fTotalScaleFactor != 0.0F);
   if (fTotalScaleFactor==0.0F)
      fTotalScaleFactor = 1.0F;

   // InputRange and InputOffset is the range and offset of the signal in
   // user units when it hits the Analog-to-Digital converter

   float fInputRange = pFH->fADCRange / fTotalScaleFactor;
   float fInputOffset= -pFH->fInstrumentOffset[nChannel];
   if (pFH->nSignalType != 0)
      fInputOffset += pFH->fSignalOffset[nChannel];

   *pfADCToUUFactor = fInputRange / pFH->lADCResolution;
   *pfADCToUUShift  = -fInputOffset;
}

#define AVERYBIGNUMBER 3.402823466E+38
//===============================================================================================
// FUNCTION: ABFH_GetMathValue
// PURPOSE:  Evaluate the Math expression for the given UU values.
// RETURNS:  TRUE if the expression could be evaluated OK.
//           FALSE if a divide by zero occurred.
//
BOOL WINAPI ABF2H_GetMathValue(const ABF2FileHeader *pFH, float fA, float fB, float *pfRval)
{
//   ABFH_ASSERT(pFH);
//   WPTRASSERT(pfRval);
   double dResult = 0.0;          // default return response
   double dLeftVal, dRightVal;
   BOOL bRval = TRUE;

   if (pFH->nArithmeticExpression == ABF2_SIMPLE_EXPRESSION)
   {
      dLeftVal  = pFH->fArithmeticK1 * fA + pFH->fArithmeticK2;
      dRightVal = pFH->fArithmeticK3 * fB + pFH->fArithmeticK4;
   }
   else
   {
      double dRatio;
      if (fB + pFH->fArithmeticK6 != 0.0F)
         dRatio = (fA + pFH->fArithmeticK5) / (fB + pFH->fArithmeticK6);
      else if (fA + pFH->fArithmeticK5 > 0.0F)
      {
         dRatio = AVERYBIGNUMBER;
         bRval = FALSE;
      }
      else
      {
         dRatio = -AVERYBIGNUMBER;
         bRval = FALSE;
      }
      dLeftVal  = pFH->fArithmeticK1 * dRatio + pFH->fArithmeticK2;
      dRightVal = pFH->fArithmeticK3 * dRatio + pFH->fArithmeticK4;
   }

   switch (pFH->sArithmeticOperator[0])
   {
      case '+':
         dResult = dLeftVal + dRightVal;
         break;
      case '-':
         dResult = dLeftVal - dRightVal;
         break;
      case '*':
         dResult = dLeftVal * dRightVal;
         break;
      case '/':
         if (dRightVal != 0.0)
            dResult = dLeftVal / dRightVal;
         else if (dLeftVal > 0)
         {
            dResult = pFH->fArithmeticUpperLimit;
            bRval = FALSE;
         }
         else
         {
            dResult = pFH->fArithmeticLowerLimit;
            bRval = FALSE;
         }
         break;

      default:
         //ERRORMSG1("Unexpected operator '%c'.", pFH->sArithmeticOperator[0]);
         break;
   }

   if (dResult < pFH->fArithmeticLowerLimit)
      dResult = pFH->fArithmeticLowerLimit;
   else if (dResult > pFH->fArithmeticUpperLimit)
      dResult = pFH->fArithmeticUpperLimit;

   if (pfRval)
      *pfRval = (float)dResult;
   return bRval;
}

//===============================================================================================
// FUNCTION: GetSampleInterval
// PURPOSE:  Gets the sample interval expressed as a double.
//           This prevents round off errors in modifiable ABF files, 
//           where sample intervals are not constrained to be in multiples of 0.5 us.
//
static double GetSampleInterval( const ABF2FileHeader *pFH, const UINT uInterval )
{
//   ABFH_ASSERT( pFH );
   ASSERT( uInterval == 1 ||
           uInterval == 2 );

   float fInterval = 0;
   if( uInterval == 1 )
      fInterval = pFH->fLegacyADCSequenceInterval;
   else if( uInterval == 2 ) 
      fInterval = pFH->fLegacyADCSecondSequenceInterval;
   else ;
      //ERRORMSG( "ABFH_GetSampleInterval called with invalid parameters !\n" );

   
   // Modifiable ABF allows sample intervals which are not multiples of 0.5 us
   // Attempt to reconstruct the original sample interval to 0.1 us resolution
   // This has no adverse effect for acquisition files and prevents rounding errors in modifable ABF files.
   double dInterval = int((fInterval * pFH->nADCNumChannels) * 10 + 0.5);
   dInterval /= 10 * pFH->nADCNumChannels;

   return dInterval;
}

//===============================================================================================
// FUNCTION: ABFH_GetFirstSampleInterval
// PURPOSE:  Gets the first sample interval expressed as a double.
double WINAPI ABF2H_GetFirstSampleInterval( const ABF2FileHeader *pFH )
{
   return GetSampleInterval( pFH, 1 );
}
