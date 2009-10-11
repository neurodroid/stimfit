//***********************************************************************************************
//
//    Copyright (c) 1993-2004 Molecular Devices Corporation.
//    All rights reserved.
//    Permission is granted to freely use, modify and copy the code in this file.
//
//***********************************************************************************************
// HEADER:  ABFHEADR.H.
// PURPOSE: Defines the ABFFileHeader structure, and provides prototypes for
//          functions implemented in ABFHEADR.CPP for reading and writing
//          ABFFileHeader's.
// REVISIONS:
//   2.0  - This version separates the data in the file from the struct passed around within the application.

#ifndef INC_ABFHEADR2_H
#define INC_ABFHEADR2_H

#include "../axon/AxAbfFio32/AxAbffio32.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// Constants used in defining the ABF file header
//
#define ABF2_ADCCOUNT           16    // number of ADC channels supported.
#define ABF2_DACCOUNT           4     // number of DAC channels supported.
#define ABF2_EPOCHCOUNT         10    // number of waveform epochs supported.
#define ABF2_ADCUNITLEN         8     // length of ADC units strings
#define ABF2_ADCNAMELEN         10    // length of ADC channel name strings
#define ABF2_DACUNITLEN         8     // length of DAC units strings
#define ABF2_DACNAMELEN         10    // length of DAC channel name strings
#define ABF2_USERLISTLEN        256   // length of the user list (V1.6)
#define ABF2_USERLISTCOUNT      4     // number of independent user lists (V1.6)
#define ABF2_OLDFILECOMMENTLEN  56    // length of file comment string (pre V1.6)
#define ABF2_FILECOMMENTLEN     128   // length of file comment string (V1.6)
#define ABF2_PATHLEN            256   // length of full path, used for DACFile and Protocol name.
#define ABF2_CREATORINFOLEN     16    // length of file creator info string
#define ABF2_ARITHMETICOPLEN    2     // length of the Arithmetic operator field
#define ABF2_ARITHMETICUNITSLEN 8     // length of arithmetic units string
#define ABF2_TAGCOMMENTLEN      56    // length of tag comment string
#define ABF2_BLOCKSIZE          512   // Size of block alignment in ABF files.
#define PCLAMP6_MAXSWEEPLENGTH 16384   // Maximum multiplexed sweep length supported by pCLAMP6 apps.
#define PCLAMP7_MAXSWEEPLEN_PERCHAN    1032258  // Maximum per channel sweep length supported by pCLAMP7 apps.
#define ABF2_MAX_SWEEPS_PER_AVERAGE 65500     // The maximum number of sweeps that can be combined into a
                                             // cumulative average (nAverageAlgorithm=ABF2_INFINITEAVERAGE).
#define ABF2_MAX_TRIAL_SAMPLES  0x7FFFFFFF    // Maximum length of acquisition supported (samples)
                                             // INT_MAX is used instead of UINT_MAX because of the signed 
                                             // values in the ABF header.

//
// Constants for nDigitizerType
//
#define ABF2_DIGI_UNKNOWN   0
#define ABF2_DIGI_DEMO      1
#define ABF2_DIGI_MINIDIGI  2
#define ABF2_DIGI_DD132X    3
#define ABF2_DIGI_OPUS      4
#define ABF2_DIGI_PATCH     5
#define ABF2_DIGI_DD1440    6

//
// Constants for nDrawingStrategy
//
#define ABF2_DRAW_NONE            0
#define ABF2_DRAW_REALTIME        1
#define ABF2_DRAW_FULLSCREEN      2
#define ABF2_DRAW_ENDOFRUN        3

//
// Constants for nTiledDisplay
//
#define ABF2_DISPLAY_SUPERIMPOSED 0
#define ABF2_DISPLAY_TILED        1

//
// Constants for nDataDisplayMode
//
#define ABF2_DRAW_POINTS       0
#define ABF2_DRAW_LINES        1

// Constants for the ABF2_ReadOpen and ABF2_WriteOpen functions
#define ABF2_DATAFILE          0     
#define ABF2_PARAMFILE         1     
#define ABF2_ALLOWOVERLAP      2     // If this flag is not set, overlapping data in fixed-length 
                                    // event-detected data will be edited out by adjustment of
                                    // the synch array. (ABF2_ReadOpen only!)
#define ABF2_DATAFILE_ABF1     4 
#define ABF2_PARAMFILE_ABF1    8 

//
// Constants for lParameterID in the ABFDelta structure.
//
// NOTE: If any changes are made to this list, the code in ABF2_UpdateHeader must
//       be updated to include the new items.
#define ABF2_DELTA_HOLDING0          0
#define ABF2_DELTA_HOLDING1          1
#define ABF2_DELTA_HOLDING2          2
#define ABF2_DELTA_HOLDING3          3
#define ABF2_DELTA_DIGITALOUTS       4
#define ABF2_DELTA_THRESHOLD         5
#define ABF2_DELTA_PRETRIGGER        6

// Because of lack of space, the Autosample Gain ID also contains the ADC number.
#define ABF2_DELTA_AUTOSAMPLE_GAIN   100   // +ADC channel.

// Because of lack of space, the Signal Gain ID also contains the ADC number.
#define ABF2_DELTA_SIGNAL_GAIN       200   // +ADC channel.


//
// Constants for nAveragingMode
//
#define ABF2_NOAVERAGING       0
#define ABF2_SAVEAVERAGEONLY   1
#define ABF2_AVERAGESAVEALL    2

//
// Constants for nAverageAlgorithm
//
#define ABF2_INFINITEAVERAGE   0
#define ABF2_SLIDINGAVERAGE    1

//
// Constants for nUndoPromptStrategy
//
#define ABF2_UNDOPROMPT_ONABORT   0
#define ABF2_UNDOPROMPT_ALWAYS    1

//
// Constants for nTriggerAction
//
#define ABF2_TRIGGER_STARTEPISODE 0
#define ABF2_TRIGGER_STARTRUN     1
#define ABF2_TRIGGER_STARTTRIAL   2    // N.B. Discontinued in favor of nTrialTriggerSource

//
// Constants for nTriggerPolarity.
//
#define ABF2_TRIGGER_RISINGEDGE  0
#define ABF2_TRIGGER_FALLINGEDGE 1

//
// Constants for nDACFileEpisodeNum
//
#define ABF2_DACFILE_SKIPFIRSTSWEEP -1
#define ABF2_DACFILE_USEALLSWEEPS    0
// >0 = The specific sweep number.

//
// Constants for nInterEpisodeLevel & nDigitalInterEpisode
//
#define ABF2_INTEREPI_USEHOLDING    0
#define ABF2_INTEREPI_USELASTEPOCH  1

//
// Constants for nArithmeticExpression
//
#define ABF2_SIMPLE_EXPRESSION    0
#define ABF2_RATIO_EXPRESSION     1

//
// Constants for nLowpassFilterType & nHighpassFilterType
//
#define ABF2_FILTER_NONE          0
#define ABF2_FILTER_EXTERNAL      1
#define ABF2_FILTER_SIMPLE_RC     2
#define ABF2_FILTER_BESSEL        3
#define ABF2_FILTER_BUTTERWORTH   4

//
// Constants for post nPostprocessLowpassFilterType
//
#define ABF2_POSTPROCESS_FILTER_NONE          0
#define ABF2_POSTPROCESS_FILTER_ADAPTIVE      1
#define ABF2_POSTPROCESS_FILTER_BESSEL        2
#define ABF2_POSTPROCESS_FILTER_BOXCAR        3
#define ABF2_POSTPROCESS_FILTER_BUTTERWORTH   4
#define ABF2_POSTPROCESS_FILTER_CHEBYSHEV     5
#define ABF2_POSTPROCESS_FILTER_GAUSSIAN      6
#define ABF2_POSTPROCESS_FILTER_RC            7
#define ABF2_POSTPROCESS_FILTER_RC8           8
#define ABF2_POSTPROCESS_FILTER_NOTCH         9

//
// The output sampling sequence identifier for a separate digital out channel.
//
#define ABF2_DIGITAL_OUT_CHANNEL -1
#define ABF2_PADDING_OUT_CHANNEL -2

//
// Constants for nAutoAnalyseEnable
//
#define ABF2_AUTOANALYSE_DISABLED   0
#define ABF2_AUTOANALYSE_DEFAULT    1
#define ABF2_AUTOANALYSE_RUNMACRO   2

//
// Constants for nAutopeakSearchMode
//
#define ABF2_PEAK_SEARCH_SPECIFIED       -2
#define ABF2_PEAK_SEARCH_ALL             -1
// nAutopeakSearchMode 0..9   = epoch in waveform 0's epoch table
// nAutopeakSearchMode 10..19 = epoch in waveform 1's epoch table

//
// Constants for nAutopeakBaseline
//
#define ABF2_PEAK_BASELINE_SPECIFIED    -3
#define ABF2_PEAK_BASELINE_NONE 	      -2
#define ABF2_PEAK_BASELINE_FIRSTHOLDING -1
#define ABF2_PEAK_BASELINE_LASTHOLDING  -4

// Bit flag settings for nStatsSearchRegionFlags
//
#define ABF2_PEAK_SEARCH_REGION0           0x01
#define ABF2_PEAK_SEARCH_REGION1           0x02
#define ABF2_PEAK_SEARCH_REGION2           0x04
#define ABF2_PEAK_SEARCH_REGION3           0x08
#define ABF2_PEAK_SEARCH_REGION4           0x10
#define ABF2_PEAK_SEARCH_REGION5           0x20
#define ABF2_PEAK_SEARCH_REGION6           0x40
#define ABF2_PEAK_SEARCH_REGION7           0x80
#define ABF2_PEAK_SEARCH_REGIONALL         0xFF        // All of the above OR'd together.

//
// Constants for nStatsActiveChannels
//
#define ABF2_PEAK_SEARCH_CHANNEL0          0x0001
#define ABF2_PEAK_SEARCH_CHANNEL1          0x0002
#define ABF2_PEAK_SEARCH_CHANNEL2          0x0004
#define ABF2_PEAK_SEARCH_CHANNEL3          0x0008
#define ABF2_PEAK_SEARCH_CHANNEL4          0x0010
#define ABF2_PEAK_SEARCH_CHANNEL5          0x0020
#define ABF2_PEAK_SEARCH_CHANNEL6          0x0040
#define ABF2_PEAK_SEARCH_CHANNEL7          0x0080
#define ABF2_PEAK_SEARCH_CHANNEL8          0x0100
#define ABF2_PEAK_SEARCH_CHANNEL9          0x0200
#define ABF2_PEAK_SEARCH_CHANNEL10         0x0400
#define ABF2_PEAK_SEARCH_CHANNEL11         0x0800
#define ABF2_PEAK_SEARCH_CHANNEL12         0x1000
#define ABF2_PEAK_SEARCH_CHANNEL13         0x2000
#define ABF2_PEAK_SEARCH_CHANNEL14         0x4000
#define ABF2_PEAK_SEARCH_CHANNEL15         0x8000
#define ABF2_PEAK_SEARCH_CHANNELSALL       0xFFFF      // All of the above OR'd together.

//
// Constants for nLeakSubtractType
//
#define ABF2_LEAKSUBTRACT_NONE       0
#define ABF2_LEAKSUBTRACT_PN         1
#define ABF2_LEAKSUBTRACT_RESISTIVE  2

//
// Constants for nPNPolarity
//
#define ABF2_PN_OPPOSITE_POLARITY -1
#define ABF2_PN_SAME_POLARITY     1

//
// Constants for nPNPosition
//
#define ABF2_PN_BEFORE_EPISODE    0
#define ABF2_PN_AFTER_EPISODE     1

//
// Constants for nAutosampleEnable
//
#define ABF2_AUTOSAMPLEDISABLED   0
#define ABF2_AUTOSAMPLEAUTOMATIC  1
#define ABF2_AUTOSAMPLEMANUAL     2

//
// Constants for nAutosampleInstrument
//
#define ABF2_INST_UNKNOWN         0   // Unknown instrument (manual or user defined telegraph table).
#define ABF2_INST_AXOPATCH1       1   // Axopatch-1 with CV-4-1/100
#define ABF2_INST_AXOPATCH1_1     2   // Axopatch-1 with CV-4-0.1/100
#define ABF2_INST_AXOPATCH1B      3   // Axopatch-1B(inv.) CV-4-1/100
#define ABF2_INST_AXOPATCH1B_1    4   // Axopatch-1B(inv) CV-4-0.1/100
#define ABF2_INST_AXOPATCH201     5   // Axopatch 200 with CV 201
#define ABF2_INST_AXOPATCH202     6   // Axopatch 200 with CV 202
#define ABF2_INST_GENECLAMP       7   // GeneClamp
#define ABF2_INST_DAGAN3900       8   // Dagan 3900
#define ABF2_INST_DAGAN3900A      9   // Dagan 3900A
#define ABF2_INST_DAGANCA1_1      10  // Dagan CA-1  Im=0.1
#define ABF2_INST_DAGANCA1        11  // Dagan CA-1  Im=1.0
#define ABF2_INST_DAGANCA10       12  // Dagan CA-1  Im=10
#define ABF2_INST_WARNER_OC725    13  // Warner OC-725
#define ABF2_INST_WARNER_OC725C   14  // Warner OC-725
#define ABF2_INST_AXOPATCH200B    15  // Axopatch 200B
#define ABF2_INST_DAGANPCONE0_1   16  // Dagan PC-ONE  Im=0.1
#define ABF2_INST_DAGANPCONE1     17  // Dagan PC-ONE  Im=1.0
#define ABF2_INST_DAGANPCONE10    18  // Dagan PC-ONE  Im=10
#define ABF2_INST_DAGANPCONE100   19  // Dagan PC-ONE  Im=100
#define ABF2_INST_WARNER_BC525C   20  // Warner BC-525C
#define ABF2_INST_WARNER_PC505    21  // Warner PC-505
#define ABF2_INST_WARNER_PC501    22  // Warner PC-501
#define ABF2_INST_DAGANCA1_05     23  // Dagan CA-1  Im=0.05
#define ABF2_INST_MULTICLAMP700   24  // MultiClamp 700
#define ABF2_INST_TURBO_TEC       25  // Turbo Tec
#define ABF2_INST_OPUSXPRESS6000  26  // OpusXpress 6000A
#define ABF2_INST_AXOCLAMP900     27  // Axoclamp 900

//
// Constants for nTagType in the ABFTag structure.
//
#define ABF2_TIMETAG              0
#define ABF2_COMMENTTAG           1
#define ABF2_EXTERNALTAG          2
#define ABF2_VOICETAG             3
#define ABF2_NEWFILETAG           4
#define ABF2_ANNOTATIONTAG        5        // Same as a comment tag except that nAnnotationIndex holds 
                                          // the index of the annotation that holds extra information.

// Comment inserted for externally acquired tags (expanded with spaces to ABF2_TAGCOMMENTLEN).
#define ABF2_EXTERNALTAGCOMMENT   "<External>"
#define ABF2_VOICETAGCOMMENT      "<Voice Tag>"

//
// Constants for nManualInfoStrategy
//
#define ABF2_ENV_DONOTWRITE      0
#define ABF2_ENV_WRITEEACHTRIAL  1
#define ABF2_ENV_PROMPTEACHTRIAL 2

//
// Constants for nAutopeakPolarity
//
#define ABF2_PEAK_NEGATIVE       -1
#define ABF2_PEAK_ABSOLUTE        0
#define ABF2_PEAK_POSITIVE        1

//
// LTP Types - Reflects whether the header is used for LTP as baseline or induction.
//
#define ABF2_LTP_TYPE_NONE              0
#define ABF2_LTP_TYPE_BASELINE          1
#define ABF2_LTP_TYPE_INDUCTION         2

//
// LTP Usage of DAC - Reflects whether the analog output will be used presynaptically or postsynaptically.
//
#define ABF2_LTP_DAC_USAGE_NONE         0
#define ABF2_LTP_DAC_USAGE_PRESYNAPTIC  1
#define ABF2_LTP_DAC_USAGE_POSTSYNAPTIC 2

// Values for the wScopeMode field in ABFScopeConfig.
#define ABF2_EPISODICMODE    0
#define ABF2_CONTINUOUSMODE  1
//#define ABF2_XYMODE          2

//
// Constants for nExperimentType
//
#define ABF2_VOLTAGECLAMP         0
#define ABF2_CURRENTCLAMP         1
#define ABF2_SIMPLEACQUISITION    2

//
// Miscellaneous constants
//
#define ABF2_FILTERDISABLED  100000.0F     // Large frequency to disable lowpass filters
#define ABF2_UNUSED_CHANNEL  -1            // Unused ADC and DAC channels.
#define ABF2_ANY_CHANNEL     (UINT)-1      // Any ADC or DAC channel.

//
// Constant definitions for nDataFormat
//
#define ABF2_INTEGERDATA      0
#define ABF2_FLOATDATA        1

//
// Constant definitions for nOperationMode
//
#define ABF2_VARLENEVENTS     1
#define ABF2_FIXLENEVENTS     2     // (ABF2_FIXLENEVENTS == ABF2_LOSSFREEOSC)
#define ABF2_LOSSFREEOSC      2
#define ABF2_GAPFREEFILE      3
#define ABF2_HIGHSPEEDOSC     4
#define ABF2_WAVEFORMFILE     5

//
// Constants for nEpochType
//
#define ABF2_EPOCHDISABLED           0     // disabled epoch
#define ABF2_EPOCHSTEPPED            1     // stepped waveform
#define ABF2_EPOCHRAMPED             2     // ramp waveform
#define ABF2_EPOCH_TYPE_RECTANGLE    3     // rectangular pulse train
#define ABF2_EPOCH_TYPE_TRIANGLE     4     // triangular waveform
#define ABF2_EPOCH_TYPE_COSINE       5     // cosinusoidal waveform
#define ABF2_EPOCH_TYPE_UNUSED       6     // was ABF2_EPOCH_TYPE_RESISTANCE
#define ABF2_EPOCH_TYPE_BIPHASIC     7     // biphasic pulse train

//
// Constants for epoch resistance
//
#define ABF2_MIN_EPOCH_RESISTANCE_DURATION 8

//
// Constants for nWaveformSource
//
#define ABF2_WAVEFORMDISABLED     0               // disabled waveform
#define ABF2_EPOCHTABLEWAVEFORM   1
#define ABF2_DACFILEWAVEFORM      2

//
// Constant definitions for nFileType
//
#define ABF2_ABFFILE          1
#define ABF2_FETCHEX          2
#define ABF2_CLAMPEX          3

//
// maximum values for various parameters (used by ABFH1_CheckUserList).
//
#define ABF2_CTPULSECOUNT_MAX           10000
#define ABF2_CTBASELINEDURATION_MAX     1000000.0F
#define ABF2_CTSTEPDURATION_MAX         1000000.0F
#define ABF2_CTPOSTTRAINDURATION_MAX    1000000.0F
#define ABF2_SWEEPSTARTTOSTARTTIME_MAX  1000000.0F 
#define ABF2_PNPULSECOUNT_MAX           8
#define ABF2_DIGITALVALUE_MAX           0xFF
#define ABF2_EPOCHDIGITALVALUE_MAX      0x0F

//
// Constants for nTriggerSource
//
#define ABF2_TRIGGERLINEINPUT           -5   // Start on line trigger (DD1320 only)
#define ABF2_TRIGGERTAGINPUT            -4
#define ABF2_TRIGGERFIRSTCHANNEL        -3
#define ABF2_TRIGGEREXTERNAL            -2
#define ABF2_TRIGGERSPACEBAR            -1
// >=0 = ADC channel to trigger off.

//
// Constants for nTrialTriggerSource
//
#define ABF2_TRIALTRIGGER_SWSTARTONLY   -6   // Start on software message, end when protocol ends.
#define ABF2_TRIALTRIGGER_SWSTARTSTOP   -5   // Start and end on software messages.
#define ABF2_TRIALTRIGGER_LINEINPUT     -4   // Start on line trigger (DD1320 only)
#define ABF2_TRIALTRIGGER_SPACEBAR      -3   // Start on spacebar press.
#define ABF2_TRIALTRIGGER_EXTERNAL      -2   // Start on external trigger high
#define ABF2_TRIALTRIGGER_NONE          -1   // Start immediately (default).
// >=0 = ADC channel to trigger off.    // Not implemented as yet...

//
// Constants for lStatisticsMeasurements
//
#define ABF2_STATISTICS_ABOVETHRESHOLD     0x00000001
#define ABF2_STATISTICS_EVENTFREQUENCY     0x00000002
#define ABF2_STATISTICS_MEANOPENTIME       0x00000004
#define ABF2_STATISTICS_MEANCLOSEDTIME     0x00000008
#define ABF2_STATISTICS_ALL                0x0000000F     // All the above OR'd together.

//
// Constants for nStatisticsSaveStrategy
//
#define ABF2_STATISTICS_NOAUTOSAVE            0
#define ABF2_STATISTICS_AUTOSAVE              1
#define ABF2_STATISTICS_AUTOSAVE_AUTOCLEAR    2

//
// Constants for nStatisticsDisplayStrategy
//
#define ABF2_STATISTICS_DISPLAY      0
#define ABF2_STATISTICS_NODISPLAY    1

//
// Constants for nStatisticsClearStrategy
// determines whether to clear statistics after saving.
//
#define ABF2_STATISTICS_NOCLEAR      0
#define ABF2_STATISTICS_CLEAR        1

#define ABF2_STATS_REGIONS     8              // The number of independent statistics regions.
#define ABF2_BASELINE_REGIONS  1              // The number of independent baseline regions.
#define ABF2_STATS_NUM_MEASUREMENTS 18        // The total number of supported statistcs measurements.

//
// Constants for lAutopeakMeasurements
//
#define ABF2_PEAK_MEASURE_PEAK                0x00000001
#define ABF2_PEAK_MEASURE_PEAKTIME            0x00000002
#define ABF2_PEAK_MEASURE_ANTIPEAK            0x00000004
#define ABF2_PEAK_MEASURE_ANTIPEAKTIME        0x00000008
#define ABF2_PEAK_MEASURE_MEAN                0x00000010
#define ABF2_PEAK_MEASURE_STDDEV              0x00000020
#define ABF2_PEAK_MEASURE_INTEGRAL            0x00000040
#define ABF2_PEAK_MEASURE_MAXRISESLOPE        0x00000080
#define ABF2_PEAK_MEASURE_MAXRISESLOPETIME    0x00000100
#define ABF2_PEAK_MEASURE_MAXDECAYSLOPE       0x00000200
#define ABF2_PEAK_MEASURE_MAXDECAYSLOPETIME   0x00000400
#define ABF2_PEAK_MEASURE_RISETIME            0x00000800
#define ABF2_PEAK_MEASURE_DECAYTIME           0x00001000
#define ABF2_PEAK_MEASURE_HALFWIDTH           0x00002000
#define ABF2_PEAK_MEASURE_BASELINE            0x00004000
#define ABF2_PEAK_MEASURE_RISESLOPE           0x00008000
#define ABF2_PEAK_MEASURE_DECAYSLOPE          0x00010000
#define ABF2_PEAK_MEASURE_REGIONSLOPE         0x00020000

#define ABF2_PEAK_NORMAL_PEAK                 0x00100000
#define ABF2_PEAK_NORMAL_ANTIPEAK             0x00400000
#define ABF2_PEAK_NORMAL_MEAN                 0x01000000
#define ABF2_PEAK_NORMAL_STDDEV               0x02000000
#define ABF2_PEAK_NORMAL_INTEGRAL             0x04000000

#define ABF2_PEAK_NORMALISABLE                0x00000075
#define ABF2_PEAK_NORMALISED                  0x07500000

#define ABF2_PEAK_MEASURE_ALL                 0x0752FFFF    // All of the above OR'd together.

//
// Constant definitions for nParamToVary
//
#define ABF2_CONDITNUMPULSES         0
#define ABF2_CONDITBASELINEDURATION  1
#define ABF2_CONDITBASELINELEVEL     2
#define ABF2_CONDITSTEPDURATION      3
#define ABF2_CONDITSTEPLEVEL         4
#define ABF2_CONDITPOSTTRAINDURATION 5
#define ABF2_CONDITPOSTTRAINLEVEL    6
#define ABF2_EPISODESTARTTOSTART     7
#define ABF2_INACTIVEHOLDING         8
#define ABF2_DIGITALHOLDING          9
#define ABF2_PNNUMPULSES             10
#define ABF2_PARALLELVALUE           11
#define ABF2_EPOCHINITLEVEL          (ABF2_PARALLELVALUE + ABF2_EPOCHCOUNT)
#define ABF2_EPOCHINITDURATION       (ABF2_EPOCHINITLEVEL + ABF2_EPOCHCOUNT)
#define ABF2_EPOCHTRAINPERIOD        (ABF2_EPOCHINITDURATION + ABF2_EPOCHCOUNT)
#define ABF2_EPOCHTRAINPULSEWIDTH    (ABF2_EPOCHTRAINPERIOD + ABF2_EPOCHCOUNT)
// Next value is (ABF2_EPOCHINITDURATION + ABF2_EPOCHCOUNT)

// Values for the nEraseStrategy field in ABFScopeConfig.
#define ABF2_ERASE_EACHSWEEP   0
#define ABF2_ERASE_EACHRUN     1
#define ABF2_ERASE_EACHTRIAL   2
#define ABF2_ERASE_DONTERASE   3

// Indexes into the rgbColor field of ABFScopeConfig.
#define ABF2_BACKGROUNDCOLOR   0
#define ABF2_GRIDCOLOR         1
#define ABF2_THRESHOLDCOLOR    2
#define ABF2_EVENTMARKERCOLOR  3
#define ABF2_SEPARATORCOLOR    4
#define ABF2_AVERAGECOLOR      5
#define ABF2_OLDDATACOLOR      6
#define ABF2_TEXTCOLOR         7
#define ABF2_AXISCOLOR         8
#define ABF2_ACTIVEAXISCOLOR   9
#define ABF2_LASTCOLOR         ABF2_ACTIVEAXISCOLOR
#define ABF2_SCOPECOLORS       (ABF2_LASTCOLOR+1)

// Extended colors for rgbColorEx field in ABFScopeConfig
#define ABF2_STATISTICS_REGION0 0
#define ABF2_STATISTICS_REGION1 1
#define ABF2_STATISTICS_REGION2 2
#define ABF2_STATISTICS_REGION3 3
#define ABF2_STATISTICS_REGION4 4
#define ABF2_STATISTICS_REGION5 5
#define ABF2_STATISTICS_REGION6 6
#define ABF2_STATISTICS_REGION7 7
#define ABF2_BASELINE_REGION    8
#define ABF2_STOREDSWEEPCOLOR   9
#define ABF2_LASTCOLOR_EX       ABF2_STOREDSWEEPCOLOR
#define ABF2_SCOPECOLORS_EX     (ABF2_LASTCOLOR+1)

//
// Constants for nCompressionType in the ABFVoiceTagInfo structure.
//
#define ABF2_COMPRESSION_NONE     0
#define ABF2_COMPRESSION_PKWARE   1

#define ABF2_CURRENTVERSION    ABF2_V200        // Current file format version number
//
// Header Version Numbers
//
#define ABF2_V200  2.00F                       // Alpha versions of pCLAMP 10 and DataXpress 2
#define ABF2_V201  2.01F                       // DataXpress 2.0.0.16 and later
                                              // pCLAMP 10.0.0.6 and later

// Retired constants.
#undef ABF2_AUTOANALYSE_RUNMACRO
#undef ABF2_MACRONAMELEN

//
// pack structure on byte boundaries
//
#ifndef RC_INVOKED
#pragma pack(push, 1)
#endif

//
// Definition of the ABF header structure.
//
struct ABF2FileHeader
{
public:
   // GROUP #1 - File ID and size information
   float    fFileVersionNumber;
   short    nOperationMode;
   ABFLONG     lActualAcqLength;
   short    nNumPointsIgnored;
   ABFLONG     lActualEpisodes;
   UINT     uFileStartDate;         // YYYYMMDD
   UINT     uFileStartTimeMS;
   ABFLONG     lStopwatchTime;
   float    fHeaderVersionNumber;
   short    nFileType;

   // GROUP #2 - File Structure
   ABFLONG     lDataSectionPtr;
   ABFLONG     lTagSectionPtr;
   ABFLONG     lNumTagEntries;
   ABFLONG     lScopeConfigPtr;
   ABFLONG     lNumScopes;
   ABFLONG     lDeltaArrayPtr;
   ABFLONG     lNumDeltas;
   ABFLONG     lVoiceTagPtr;
   ABFLONG     lVoiceTagEntries;
   ABFLONG     lSynchArrayPtr;
   ABFLONG     lSynchArraySize;
   short    nDataFormat;
   short    nSimultaneousScan;
   ABFLONG     lStatisticsConfigPtr;
   ABFLONG     lAnnotationSectionPtr;
   ABFLONG     lNumAnnotations;
   ABFLONG     lDACFilePtr[ABF2_DACCOUNT];
   ABFLONG     lDACFileNumEpisodes[ABF2_DACCOUNT];

   // GROUP #3 - Trial hierarchy information
   short    nADCNumChannels;
   float    fADCSequenceInterval;
   UINT     uFileCompressionRatio;
   bool     bEnableFileCompression;
   float    fSynchTimeUnit;
   float    fSecondsPerRun;
   ABFLONG     lNumSamplesPerEpisode;
   ABFLONG     lPreTriggerSamples;
   ABFLONG     lEpisodesPerRun;
   ABFLONG     lRunsPerTrial;
   ABFLONG     lNumberOfTrials;
   short    nAveragingMode;
   short    nUndoRunCount;
   short    nFirstEpisodeInRun;
   float    fTriggerThreshold;
   short    nTriggerSource;
   short    nTriggerAction;
   short    nTriggerPolarity;
   float    fScopeOutputInterval;
   float    fEpisodeStartToStart;
   float    fRunStartToStart;
   float    fTrialStartToStart;
   ABFLONG     lAverageCount;
   short    nAutoTriggerStrategy;
   float    fFirstRunDelayS;

   // GROUP #4 - Display Parameters
   short    nDataDisplayMode;
   short    nChannelStatsStrategy;
   ABFLONG     lSamplesPerTrace;
   ABFLONG     lStartDisplayNum;
   ABFLONG     lFinishDisplayNum;
   short    nShowPNRawData;
   float    fStatisticsPeriod;
   ABFLONG     lStatisticsMeasurements;
   short    nStatisticsSaveStrategy;

   // GROUP #5 - Hardware information
   float    fADCRange;
   float    fDACRange;
   ABFLONG     lADCResolution;
   ABFLONG     lDACResolution;
   short    nDigitizerADCs;
   short    nDigitizerDACs;
   short    nDigitizerTotalDigitalOuts;
   short    nDigitizerSynchDigitalOuts;
   short    nDigitizerType;

   // GROUP #6 Environmental Information
   short    nExperimentType;
   short    nManualInfoStrategy;
   float    fCellID1;
   float    fCellID2;
   float    fCellID3;
   char     sProtocolPath[ABF2_PATHLEN];
   char     sCreatorInfo[ABF2_CREATORINFOLEN];
   char     sModifierInfo[ABF2_CREATORINFOLEN];
   short    nCommentsEnable;
   char     sFileComment[ABF2_FILECOMMENTLEN];
   short    nTelegraphEnable[ABF2_ADCCOUNT];
   short    nTelegraphInstrument[ABF2_ADCCOUNT];
   float    fTelegraphAdditGain[ABF2_ADCCOUNT];
   float    fTelegraphFilter[ABF2_ADCCOUNT];
   float    fTelegraphMembraneCap[ABF2_ADCCOUNT];
   float    fTelegraphAccessResistance[ABF2_ADCCOUNT];
   short    nTelegraphMode[ABF2_ADCCOUNT];
   short    nTelegraphDACScaleFactorEnable[ABF2_DACCOUNT];

   short    nAutoAnalyseEnable;

   GUID     FileGUID;
   float    fInstrumentHoldingLevel[ABF2_DACCOUNT];
   unsigned ABFLONG ulFileCRC;
   short    nCRCEnable;

   // GROUP #7 - Multi-channel information
   short    nSignalType;                        // why is this only single channel ?
   short    nADCPtoLChannelMap[ABF2_ADCCOUNT];
   short    nADCSamplingSeq[ABF2_ADCCOUNT];
   float    fADCProgrammableGain[ABF2_ADCCOUNT];
   float    fADCDisplayAmplification[ABF2_ADCCOUNT];
   float    fADCDisplayOffset[ABF2_ADCCOUNT];       
   float    fInstrumentScaleFactor[ABF2_ADCCOUNT];  
   float    fInstrumentOffset[ABF2_ADCCOUNT];       
   float    fSignalGain[ABF2_ADCCOUNT];
   float    fSignalOffset[ABF2_ADCCOUNT];
   float    fSignalLowpassFilter[ABF2_ADCCOUNT];
   float    fSignalHighpassFilter[ABF2_ADCCOUNT];
   char     nLowpassFilterType[ABF2_ADCCOUNT];
   char     nHighpassFilterType[ABF2_ADCCOUNT];

   char     sADCChannelName[ABF2_ADCCOUNT][ABF2_ADCNAMELEN];
   char     sADCUnits[ABF2_ADCCOUNT][ABF2_ADCUNITLEN];
   float    fDACScaleFactor[ABF2_DACCOUNT];
   float    fDACHoldingLevel[ABF2_DACCOUNT];
   float    fDACCalibrationFactor[ABF2_DACCOUNT];
   float    fDACCalibrationOffset[ABF2_DACCOUNT];
   char     sDACChannelName[ABF2_DACCOUNT][ABF2_DACNAMELEN];
   char     sDACChannelUnits[ABF2_DACCOUNT][ABF2_DACUNITLEN];

   // GROUP #9 - Epoch Waveform and Pulses
   short    nDigitalEnable;
   short    nActiveDACChannel;                     // should retire !
   short    nDigitalDACChannel;
   short    nDigitalHolding;
   short    nDigitalInterEpisode;
   short    nDigitalTrainActiveLogic;                                   
   short    nDigitalValue[ABF2_EPOCHCOUNT];
   short    nDigitalTrainValue[ABF2_EPOCHCOUNT];                         
   bool     bEpochCompression[ABF2_EPOCHCOUNT];
   short    nWaveformEnable[ABF2_DACCOUNT];
   short    nWaveformSource[ABF2_DACCOUNT];
   short    nInterEpisodeLevel[ABF2_DACCOUNT];
   short    nEpochType[ABF2_DACCOUNT][ABF2_EPOCHCOUNT];
   float    fEpochInitLevel[ABF2_DACCOUNT][ABF2_EPOCHCOUNT];
   float    fEpochLevelInc[ABF2_DACCOUNT][ABF2_EPOCHCOUNT];
   ABFLONG     lEpochInitDuration[ABF2_DACCOUNT][ABF2_EPOCHCOUNT];
   ABFLONG     lEpochDurationInc[ABF2_DACCOUNT][ABF2_EPOCHCOUNT];

   // GROUP #10 - DAC Output File
   float    fDACFileScale[ABF2_DACCOUNT];
   float    fDACFileOffset[ABF2_DACCOUNT];
   ABFLONG     lDACFileEpisodeNum[ABF2_DACCOUNT];
   short    nDACFileADCNum[ABF2_DACCOUNT];
   char     sDACFilePath[ABF2_DACCOUNT][ABF2_PATHLEN];

   // GROUP #11 - Presweep (conditioning) pulse train
   short    nConditEnable[ABF2_DACCOUNT];
   ABFLONG     lConditNumPulses[ABF2_DACCOUNT];
   float    fBaselineDuration[ABF2_DACCOUNT];
   float    fBaselineLevel[ABF2_DACCOUNT];
   float    fStepDuration[ABF2_DACCOUNT];
   float    fStepLevel[ABF2_DACCOUNT];
   float    fPostTrainPeriod[ABF2_DACCOUNT];
   float    fPostTrainLevel[ABF2_DACCOUNT];
   short    nMembTestEnable[ABF2_DACCOUNT];
   float    fMembTestPreSettlingTimeMS[ABF2_DACCOUNT];
   float    fMembTestPostSettlingTimeMS[ABF2_DACCOUNT];

   // GROUP #12 - Variable parameter user list
   short    nULEnable[ABF2_USERLISTCOUNT];
   short    nULParamToVary[ABF2_USERLISTCOUNT];
   short    nULRepeat[ABF2_USERLISTCOUNT];
   char     sULParamValueList[ABF2_USERLISTCOUNT][ABF2_USERLISTLEN];

   // GROUP #13 - Statistics measurements
   short    nStatsEnable;
   unsigned short nStatsActiveChannels;             // Active stats channel bit flag
   unsigned short nStatsSearchRegionFlags;          // Active stats region bit flag
   short    nStatsSmoothing;
   short    nStatsSmoothingEnable;
   short    nStatsBaseline;
   short    nStatsBaselineDAC;                      // If mode is epoch, then this holds the DAC
   ABFLONG     lStatsBaselineStart;
   ABFLONG     lStatsBaselineEnd;
   ABFLONG     lStatsMeasurements[ABF2_STATS_REGIONS];  // Measurement bit flag for each region
   ABFLONG     lStatsStart[ABF2_STATS_REGIONS];
   ABFLONG     lStatsEnd[ABF2_STATS_REGIONS];
   short    nRiseBottomPercentile[ABF2_STATS_REGIONS];
   short    nRiseTopPercentile[ABF2_STATS_REGIONS];
   short    nDecayBottomPercentile[ABF2_STATS_REGIONS];
   short    nDecayTopPercentile[ABF2_STATS_REGIONS];
   short    nStatsChannelPolarity[ABF2_ADCCOUNT];
   short    nStatsSearchMode[ABF2_STATS_REGIONS];    // Stats mode per region: mode is cursor region, epoch etc 
   short    nStatsSearchDAC[ABF2_STATS_REGIONS];     // If mode is epoch, then this holds the DAC

   // GROUP #14 - Channel Arithmetic
   short    nArithmeticEnable;
   short    nArithmeticExpression;
   float    fArithmeticUpperLimit;
   float    fArithmeticLowerLimit;
   short    nArithmeticADCNumA;
   short    nArithmeticADCNumB;
   float    fArithmeticK1;
   float    fArithmeticK2;
   float    fArithmeticK3;
   float    fArithmeticK4;
   float    fArithmeticK5;
   float    fArithmeticK6;
   char     sArithmeticOperator[ABF2_ARITHMETICOPLEN];
   char     sArithmeticUnits[ABF2_ARITHMETICUNITSLEN];

   // GROUP #15 - Leak subtraction
   short    nPNPosition;
   short    nPNNumPulses;
   short    nPNPolarity;
   float    fPNSettlingTime;
   float    fPNInterpulse;
   short    nLeakSubtractType[ABF2_DACCOUNT];
   float    fPNHoldingLevel[ABF2_DACCOUNT];
   bool     bEnabledDuringPN[ABF2_ADCCOUNT];

   // GROUP #16 - Miscellaneous variables
   short    nLevelHysteresis;
   ABFLONG     lTimeHysteresis;
   short    nAllowExternalTags;
   short    nAverageAlgorithm;
   float    fAverageWeighting;
   short    nUndoPromptStrategy;
   short    nTrialTriggerSource;
   short    nStatisticsDisplayStrategy;
   short    nExternalTagType;
   ABFLONG     lHeaderSize;
   short    nStatisticsClearStrategy;
   
   // GROUP #17 - Trains parameters
   ABFLONG     lEpochPulsePeriod[ABF2_DACCOUNT][ABF2_EPOCHCOUNT];
   ABFLONG     lEpochPulseWidth [ABF2_DACCOUNT][ABF2_EPOCHCOUNT];

   // GROUP #18 - Application version data
   short    nCreatorMajorVersion;
   short    nCreatorMinorVersion;
   short    nCreatorBugfixVersion;
   short    nCreatorBuildVersion;
   short    nModifierMajorVersion;
   short    nModifierMinorVersion;
   short    nModifierBugfixVersion;
   short    nModifierBuildVersion;

   // GROUP #19 - LTP protocol
   short    nLTPType;
   short    nLTPUsageOfDAC[ABF2_DACCOUNT];
   short    nLTPPresynapticPulses[ABF2_DACCOUNT];

   // GROUP #20 - Digidata 132x Trigger out flag
   short    nScopeTriggerOut;

   // GROUP #22 - Alternating episodic mode
   short    nAlternateDACOutputState;
   short    nAlternateDigitalOutputState;
   short    nAlternateDigitalValue[ABF2_EPOCHCOUNT];
   short    nAlternateDigitalTrainValue[ABF2_EPOCHCOUNT];

   // GROUP #23 - Post-processing actions
   float    fPostProcessLowpassFilter[ABF2_ADCCOUNT];
   char     nPostProcessLowpassFilterType[ABF2_ADCCOUNT];

   // GROUP #24 - Legacy gear shift info
   float    fLegacyADCSequenceInterval;
   float    fLegacyADCSecondSequenceInterval;
   ABFLONG     lLegacyClockChange;
   ABFLONG     lLegacyNumSamplesPerEpisode;

   ABF2FileHeader();
};   

inline ABF2FileHeader::ABF2FileHeader()
{
   // Set everything to 0.
   memset( this, 0, sizeof(ABF2FileHeader) );
   
   // Set critical parameters so we can determine the version.
   fFileVersionNumber   = ABF2_CURRENTVERSION;
   fHeaderVersionNumber = ABF2_CURRENTVERSION;
   lHeaderSize          = sizeof(ABF2FileHeader);
}

//
// Scope descriptor format.
//
#define ABF2_FACESIZE 32
struct ABFLogFont
{
   short nHeight;                // Height of the font in pixels.
//   short lWidth;               // use 0
//   short lEscapement;          // use 0
//   short lOrientation;         // use 0
   short nWeight;                // MSWindows font weight value.
//   char bItalic;               // use 0
//   char bUnderline;            // use 0
//   char bStrikeOut;            // use 0
//   char cCharSet;              // use ANSI_CHARSET (0)
//   char cOutPrecision;         // use OUT_TT_PRECIS
//   char cClipPrecision;        // use CLIP_DEFAULT_PRECIS
//   char cQuality;              // use PROOF_QUALITY
   char cPitchAndFamily;         // MSWindows pitch and family mask.
   char Unused[3];               // Unused space to maintain 4-byte packing.
   char szFaceName[ABF2_FACESIZE];// Face name of the font.
};     // Size = 40

struct ABFSignal
{
   char     szName[ABF2_ADCNAMELEN+2];        // ABF name length + '\0' + 1 for alignment.
   short    nMxOffset;                       // Offset of the signal in the sampling sequence.
   DWORD    rgbColor;                        // Pen color used to draw trace.
   char     nPenWidth;                       // Pen width in pixels.
   char     bDrawPoints;                     // TRUE = Draw disconnected points
   char     bHidden;                         // TRUE = Hide the trace.
   char     bFloatData;                      // TRUE = Floating point pseudo channel
   float    fVertProportion;                 // Relative proportion of client area to use
   float    fDisplayGain;                    // Display gain of trace in UserUnits
   float    fDisplayOffset;                  // Display offset of trace in UserUnits

//   float    fUUTop;                          // Top of window in UserUnits
//   float    fUUBottom;                       // Bottom of window in UserUnits
};      // Size = 34

struct ABFScopeConfig
{
   // Section 1 scope configurations
   DWORD       dwFlags;                   // Flags that are meaningful to the scope.
   DWORD       rgbColor[ABF2_SCOPECOLORS]; // Colors for the components of the scope.
   float       fDisplayStart;             // Start of the display area in ms.
   float       fDisplayEnd;               // End of the display area in ms.
   WORD        wScopeMode;                // Mode that the scope is in.
   char        bMaximized;                // TRUE = Scope parent is maximized.
   char        bMinimized;                // TRUE = Scope parent is minimized.
   short       xLeft;                     // Coordinate of the left edge.
   short       yTop;                      // Coordinate of the top edge.
   short       xRight;                    // Coordinate of the right edge.
   short       yBottom;                   // Coordinate of the bottom edge.
   ABFLogFont  LogFont;                   // Description of current font.
   ABFSignal   TraceList[ABF2_ADCCOUNT];   // List of traces in current use.
   short       nYAxisWidth;               // Width of the YAxis region.
   short       nTraceCount;               // Number of traces described in TraceList.
   short       nEraseStrategy;            // Erase strategy.
   short       nDockState;                // Docked position.
   // Size 656
   // * Do not insert any new members above this point! *
   // Section 2 scope configurations for file version 1.68.
   short       nSizeofOldStructure;              // Unused byte to determine the offset of the version 2 data.
   DWORD       rgbColorEx[ ABF2_SCOPECOLORS_EX ]; // New color settings for stored sweep and cursors.
   short       nAutoZeroState;                   // Status of the autozero selection.
   DWORD       dwCursorsVisibleState;            // Flag for visible status of cursors.
   DWORD       dwCursorsLockedState;             // Flag for enabled status of cursors.
   char        sUnasigned[61];
   // Size 113
   ABFScopeConfig();
}; // Size = 769


inline ABFScopeConfig::ABFScopeConfig()
{
   // Set everything to 0.
   memset( this, 0, sizeof(ABFScopeConfig) );
   
   // Set critical parameters so we can determine the version.
   nSizeofOldStructure = 656;
}

//
// Definition of the ABF Tag structure
//
struct ABFTag
{
   ABFLONG    lTagTime;          // Time at which the tag was entered in fSynchTimeUnit units.
   char    sComment[ABF2_TAGCOMMENTLEN];   // Optional tag comment.
   short   nTagType;          // Type of tag ABF2_TIMETAG, ABF2_COMMENTTAG, ABF2_EXTERNALTAG, ABF2_VOICETAG, ABF2_NEWFILETAG or ABF2_ANNOTATIONTAG
   union 
   {
      short   nVoiceTagNumber;   // If nTagType=ABF2_VOICETAG, this is the number of this voice tag.
      short   nAnnotationIndex;  // If nTagType=ABF2_ANNOTATIONTAG, this is the index of the corresponding annotation.
   };
}; // Size = 64

//
// Definition of the ABFVoiceTagInfo structure.
//
struct ABFVoiceTagInfo
{
   ABFLONG  lTagNumber;          // The tag number that corresponds to this VoiceTag
   ABFLONG  lFileOffset;         // Offset to this tag within the VoiceTag block
   ABFLONG  lUncompressedSize;   // Size of the voice tag expanded.
   ABFLONG  lCompressedSize;     // Compressed size of the tag.
   short nCompressionType;    // Compression method used.
   short nSampleSize;         // Size of the samples acquired.
   ABFLONG  lSamplesPerSecond;   // Rate at which the sound was acquired.
   DWORD dwCRC;               // CRC used to check data integrity.
   WORD  wChannels;           // Number of channels in the tag (usually 1).
   WORD  wUnused;             // Unused space.
}; // Size 32

//
// Definition of the ABF Delta structure.
//
struct ABFDelta
{
   ABFLONG    lDeltaTime;        // Time at which the parameter was changed in fSynchTimeUnit units.
   ABFLONG    lParameterID;      // Identifier for the parameter changed
   union
   {
      ABFLONG  lNewParamValue;   // Depending on the value of lParameterID
      float fNewParamValue;   // this entry may be either a float or a long.
   };
}; // Size = 12

//
// Definition of the ABF synch array structure
//
struct ABF2Synch
{
   ABFLONG    lStart;            // Start of the episode/event in fSynchTimeUnit units.
   ABFLONG    lLength;           // Length of the episode/event in multiplexed samples.
}; // Size = 8

#ifndef RC_INVOKED
#pragma pack(pop)                      // return to default packing
#endif

// ============================================================================================
// Function prototypes for functions in ABFHEADR.C
// ============================================================================================
    
void WINAPI ABF2H_Initialize( ABF2FileHeader *pFH );

#if 0
void WINAPI ABF2H_InitializeScopeConfig(const ABF2FileHeader *pFH, ABFScopeConfig *pCfg);

BOOL WINAPI ABF2H_CheckScopeConfig(const ABF2FileHeader *pFH, ABFScopeConfig *pCfg);

void WINAPI ABF2H_GetADCDisplayRange( const ABF2FileHeader *pFH, int nChannel, 
                                     float *pfUUTop, float *pfUUBottom);
#endif                                     
void WINAPI ABF2H_GetADCtoUUFactors( const ABF2FileHeader *pFH, int nChannel, 
                                    float *pfADCToUUFactor, float *pfADCToUUShift );
#if 0    
void WINAPI ABF2H_ClipADCUUValue(const ABF2FileHeader *pFH, int nChannel, float *pfUUValue);
                                           
void WINAPI ABF2H_GetDACtoUUFactors( const ABF2FileHeader *pFH, int nChannel, 
                                    float *pfDACToUUFactor, float *pfDACToUUShift );
void WINAPI ABF2H_ClipDACUUValue(const ABF2FileHeader *pFH, int nChannel, float *pfUUValue);
#endif
BOOL WINAPI ABF2H_GetMathValue(const ABF2FileHeader *pFH, float fA, float fB, float *pfRval);
#if 0
int WINAPI ABF2H_GetMathChannelName(LPSTR psz, UINT uLen);

BOOL WINAPI ABF2H_ParamReader( HANDLE hFile, ABF2FileHeader *pFH, int *pnError );
BOOL WINAPI ABF2H_ParamWriter( HANDLE hFile, ABF2FileHeader *pFH, int *pnError );

BOOL WINAPI ABF2H_GetErrorText( int nError, char *pszBuffer, UINT nBufferSize );

BOOL WINAPI ABF2H_GetCreatorInfo(const ABF2FileHeader *pFH, char *pszName, UINT uNameSize, char *pszVersion, UINT uVersionSize);
BOOL WINAPI ABF2H_GetModifierInfo(const ABF2FileHeader *pFH, char *pszName, UINT uNameSize, char *pszVersion, UINT uVersionSize);

// ABF 1 conversion functions - use with care.
struct ABF2FileHeader1;
BOOL WINAPI ABF2H_ConvertFromABF1( const ABF2FileHeader1 *pIn, ABF2FileHeader *pOut, int *pnError );
BOOL WINAPI ABF2H_ConvertABF2ToABF1Header( const ABF2FileHeader *pNewFH, ABF2FileHeader1 *pOldFH, int *pnError );


// ABFHWAVE.CPP

// Constants for ABF2H_GetEpochLimits
#define ABF2H_FIRSTHOLDING  -1
#define ABF2H_LASTHOLDING   ABF2_EPOCHCOUNT

// Return the bounds of a given epoch in a given episode. Values returned are ZERO relative.
BOOL WINAPI ABF2H_GetEpochLimits(const ABF2FileHeader *pFH, int nADCChannel, UINT uDACChannel, DWORD dwEpisode, 
                                int nEpoch, UINT *puEpochStart, UINT *puEpochEnd,
                                int *pnError);
    
#endif
// Get the offset in the sampling sequence for the given physical channel.
BOOL WINAPI ABF2H_GetChannelOffset( const ABF2FileHeader *pFH, int nChannel, UINT *puChannelOffset );

// Gets the first sample interval, expressed as a double.
double WINAPI ABF2H_GetFirstSampleInterval( const ABF2FileHeader *pFH );

#if 0
// This function forms the de-multiplexed DAC output waveform for the
// particular channel in the pfBuffer, in DAC UserUnits.
BOOL WINAPI ABF2H_GetWaveform( const ABF2FileHeader *pFH, UINT uDACChannel, DWORD dwEpisode, 
                                float *pfBuffer, int *pnError);

// This function forms the de-multiplexed Digital output waveform for the
// particular channel in the pdwBuffer, as a bit mask. Digital OUT 0 is in bit 0.
BOOL WINAPI ABF2H_GetDigitalWaveform( const ABF2FileHeader *pFH, int nChannel, DWORD dwEpisode, 
                                     DWORD *pdwBuffer, int *pnError);

// Calculates the timebase array for the file.
void WINAPI ABF2H_GetTimebase(const ABF2FileHeader *pFH, double dTimeOffset, double *pdBuffer, UINT uBufferSize);

// Constant for ABF2H_GetHoldingDuration
#define ABF2H_HOLDINGFRACTION 64

// Get the duration of the first holding period.
UINT WINAPI ABF2H_GetHoldingDuration(const ABF2FileHeader *pFH);

// Checks whether the waveform varies from episode to episode.
BOOL WINAPI ABF2H_IsConstantWaveform(const ABF2FileHeader *pFH, UINT uDACChannel);

// Get the full sweep length given the length available to epochs or vice-versa.
int WINAPI ABF2H_SweepLenFromUserLen(int nUserLength, int nNumChannels);
int WINAPI ABF2H_UserLenFromSweepLen(int nSweepLength, int nNumChannels);

// Converts a display range to the equivalent gain and offset factors.
void WINAPI ABF2H_GainOffsetToDisplayRange( const ABF2FileHeader *pFH, int nChannel, 
                                           float fDisplayGain, float fDisplayOffset,
                                           float *pfUUTop, float *pfUUBottom);

// Converts a display range to the equivalent gain and offset factors.
void WINAPI ABF2H_DisplayRangeToGainOffset( const ABF2FileHeader *pFH, int nChannel, 
                                           float fUUTop, float fUUBottom,
                                           float *pfDisplayGain, float *pfDisplayOffset);

// Converts a time value to a synch time count or vice-versa.
void WINAPI ABF2H_SynchCountToMS(const ABF2FileHeader *pFH, UINT uCount, double *pdTimeMS);
UINT WINAPI ABF2H_MSToSynchCount(const ABF2FileHeader *pFH, double dTimeMS);

// Gets the duration of the Waveform Episode (in us), allowing for split clock etc.
void WINAPI ABF2H_GetEpisodeDuration(const ABF2FileHeader *pFH, double *pdEpisodeDuration);

// Returns TRUE is P/N is enabled on any output channel.
BOOL WINAPI ABF2H_IsPNEnabled(const ABF2FileHeader *pFH, UINT uDAC=ABF2_ANY_CHANNEL);

// Gets the duration of a P/N sequence (in us), including settling times.
void WINAPI ABF2H_GetPNDuration(const ABF2FileHeader *pFH, double *pdPNDuration);

// Gets the duration of a pre-sweep train in us.
void WINAPI ABF2H_GetTrainDuration (const ABF2FileHeader *pFH, UINT uDAC, double *pdTrainDuration);

// Gets the duration of a post-train portion of the pre-sweep train in us.
void WINAPI ABF2H_GetPostTrainDuration (const ABF2FileHeader *pFH, UINT uDAC, UINT uEpisode, double *pdDuration);

// Gets the level of a post-train portion of the pre-sweep train.
void WINAPI ABF2H_GetPostTrainLevel (const ABF2FileHeader *pFH, UINT uDAC, UINT uEpisode, double *pdLevel);

// Gets the duration of a whole meta-episode (in us).
void WINAPI ABF2H_GetMetaEpisodeDuration(const ABF2FileHeader *pFH, double *pdMetaEpisodeDuration);

// Gets the start to start period for the episode in us.
void WINAPI ABF2H_GetEpisodeStartToStart(const ABF2FileHeader *pFH, double *pdEpisodeStartToStart);

// Checks that the user list contains valid entries for the protocol.
BOOL WINAPI ABF2H_CheckUserList(const ABF2FileHeader *pFH, UINT uListNum, int *pnError);

// Counts the number of changing sweeps.
UINT WINAPI ABF2H_GetNumberOfChangingSweeps( const ABF2FileHeader *pFH );

// // Checks whether the digital output varies from episode to episode.
BOOL WINAPI ABF2H_IsConstantDigitalOutput(const ABF2FileHeader *pFH, UINT uDACChannel);

int WINAPI ABF2H_GetEpochDuration(const ABF2FileHeader *pFH, UINT uDACChannel, UINT uEpisode, int nEpoch);

float WINAPI ABF2H_GetEpochLevel(const ABF2FileHeader *pFH, UINT uDACChannel, UINT uEpisode, int nEpoch);
BOOL WINAPI ABF2H_GetEpochLevelRange(const ABF2FileHeader *pFH, UINT uDACChannel, int nEpoch, float *pfMin, float *pfMax);
UINT WINAPI ABF2H_GetMaxPNSubsweeps(const ABF2FileHeader *pFH, UINT uDACChannel);
#endif

//
// Error return values that may be returned by the ABF2H_xxx functions.
//

#define ABF2H_FIRSTERRORNUMBER          2001
#define ABF2H_EHEADERREAD               2001
#define ABF2H_EHEADERWRITE              2002
#define ABF2H_EINVALIDFILE              2003
#define ABF2H_EUNKNOWNFILETYPE          2004
#define ABF2H_CHANNELNOTSAMPLED         2005
#define ABF2H_EPOCHNOTPRESENT           2006
#define ABF2H_ENOWAVEFORM               2007
#define ABF2H_EDACFILEWAVEFORM          2008
#define ABF2H_ENOMEMORY                 2009
#define ABF2H_BADSAMPLEINTERVAL         2010
#define ABF2H_BADSECONDSAMPLEINTERVAL   2011
#define ABF2H_BADSAMPLEINTERVALS        2012
#define ABF2H_ENOCONDITTRAINS           2013
#define ABF2H_EMETADURATION             2014
#define ABF2H_ECONDITNUMPULSES          2015
#define ABF2H_ECONDITBASEDUR            2016
#define ABF2H_ECONDITBASELEVEL          2017
#define ABF2H_ECONDITPOSTTRAINDUR       2018
#define ABF2H_ECONDITPOSTTRAINLEVEL     2019
#define ABF2H_ESTART2START              2020
#define ABF2H_EINACTIVEHOLDING          2021
#define ABF2H_EINVALIDCHARS             2022
#define ABF2H_ENODIG                    2023
#define ABF2H_EDIGHOLDLEVEL             2024
#define ABF2H_ENOPNPULSES               2025
#define ABF2H_EPNNUMPULSES              2026
#define ABF2H_ENOEPOCH                  2027
#define ABF2H_EEPOCHLEN                 2028
#define ABF2H_EEPOCHINITLEVEL           2029
#define ABF2H_EDIGLEVEL                 2030
#define ABF2H_ECONDITSTEPDUR            2031
#define ABF2H_ECONDITSTEPLEVEL          2032
#define ABF2H_EINVALIDBINARYCHARS       2033
#define ABF2H_EBADWAVEFORM              2034

#ifdef __cplusplus
}
#endif

#endif   /* INC_ABFHEADR2_H */
