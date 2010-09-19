// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

// Inspired by sigTOOL
// http://sigtool.sourceforge.net/
// Original comment for the Matlab file:
//--------------------------------------------------------------------------
// Author: Malcolm Lidierth 12/09
// Copyright (c) The Author & King's College London 2009-
//--------------------------------------------------------------------------

#include <string>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm> //required for std::swap
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#ifndef MODULE_ONLY
#include <wx/wxprec.h>
#include <wx/progdlg.h>
#include <wx/msgdlg.h>
#endif

#include "./../core.h"
#include "./hekalib.h"

#define C_ASSERT(e) extern void __C_ASSERT__(int [(e)?1:-1])
#define ByteSwap16(x) ByteSwap((unsigned char *) &x,sizeof(x))
#define ByteSwap32(x) ByteSwap((unsigned char *) &x,sizeof(x))

enum Level {
    root = 0,
    group,
    series,
    sweep,
    trace
};

Level int2Level(int n) {
    switch (n) {
     case root:
         return root;
     case group:
         return group;
     case series:
         return series;
     case sweep:
         return sweep;
     case trace:
         return trace;
     default:
         return root;
    }
}

struct TreeEntry {
    TreeEntry(Level l, int c, int i) :
        level(l), counter(c), idx(i)
    {}
        
    Level level;
    int counter;
    int idx;
};

//
// pack structure on byte boundaries
//
#ifndef RC_INVOKED
#pragma pack(push, 1)
#endif

struct BundleItem {
   int oStart;            /* INT32 */
   int oLength;           /* INT32 */
   char oExtension[8];    /* ARRAY[0..7] OF CHAR */
};
C_ASSERT(sizeof(BundleItem) == 16);

struct BundleHeader {
    char oSignature[8];    /*   8 ARRAY[0..7] OF CHAR */
    char oVersion[32];     /*  40 ARRAY[0..31] OF CHAR */
    double oTime;        /*  48 LONGREAL */
    int oItems;            /*  52 INT32 */
    char oIsLittleEndian[12];   /*  64 BOOLEAN */
    BundleItem oBundleItems[12];    /* 256 ARRAY[0..11] OF BundleItem */
};
C_ASSERT(sizeof(BundleHeader) == 256);

struct TraceRecord {
    int TrMark; /* INT32 */
    char TrLabel[32]; /* String32Type */
    int TrTraceCount; /* INT32 */
    int TrData; /* INT32 */
    int TrDataPoints; /* INT32 */
    int TrInternalSolution; /* INT32 */
    int TrAverageCount; /* INT32 */
    int TrLeakCount; /* INT32 */
    int TrLeakTraces; /* INT32 */
    short TrDataKind; /* SET16 */
    short TrFiller1; /* SET16 */
    char TrRecordingMode; /* BYTE */
    char TrAmplIndex; /* CHAR */
    char TrDataFormat; /* BYTE */
    char TrDataAbscissa; /* BYTE */
    double TrDataScaler; /* LONGREAL */
    double TrTimeOffset; /* LONGREAL */
    double TrZeroData; /* LONGREAL */
    char TrYUnit[8]; /* String8Type */
    double TrXInterval; /* LONGREAL */
    double TrXStart; /* LONGREAL */
    char TrXUnit[8]; /* String8Type */
    double TrYRange; /* LONGREAL */
    double TrYOffset; /* LONGREAL */
    double TrBandwidth; /* LONGREAL */
    double TrPipetteResistance; /* LONGREAL */
    double TrCellPotential; /* LONGREAL */
    double TrSealResistance; /* LONGREAL */
    double TrCSlow; /* LONGREAL */
    double TrGSeries; /* LONGREAL */
    double TrRsValue; /* LONGREAL */
    double TrGLeak; /* LONGREAL */
    double TrMConductance; /* LONGREAL */
    int TrLinkDAChannel; /* INT32 */
    char TrValidYrange; /* BOOLEAN */
    char TrAdcMode; /* CHAR */
    short TrAdcChannel; /* INT16 */
    double TrYmin; /* LONGREAL */
    double TrYmax; /* LONGREAL */
    int TrSourceChannel; /* INT32 */
    int TrExternalSolution; /* INT32 */
    double TrCM; /* LONGREAL */
    double TrGM; /* LONGREAL */
    double TrPhase; /* LONGREAL */
    int TrDataCRC; /* CARD32 */
    int TrCRC; /* CARD32 */
    double TrGS; /* LONGREAL */
    int TrSelfChannel; /* INT32 */
    int TrFiller2; /* SET16 */
};
C_ASSERT(sizeof(TraceRecord) == 296);

struct SweepRecord {
    int SwMark; /* INT32 */
    char SwLabel[32]; /* String32Type */
    int SwAuxDataFileOffset; /* INT32 */
    int SwStimCount; /* INT32 */
    int SwSweepCount; /* INT32 */
    double SwTime; /* LONGREAL */
    double SwTimer; /* LONGREAL */
    double SwSwUserParams[4]; /* ARRAY[0..3] OF LONGREAL */
    double SwTemperature; /* LONGREAL */
    int SwOldIntSol; /* INT32 */
    int SwOldExtSol; /* INT32 */
    short SwDigitalIn; /* SET16 */
    short SwSweepKind; /* SET16 */
    int SwFiller1; /* INT32 */
    double SwMarkers[4]; /* ARRAY[0..3] OF LONGREAL */
    int SwFiller2; /* INT32 */
    int SwCRC; /* CARD32 */
};
C_ASSERT(sizeof(SweepRecord) == 160);

struct UserParamDescrType  {
    char Name[32];
    char Unit[8];
};
C_ASSERT(sizeof(UserParamDescrType) == 40);

struct AmplifierState {
    char E9StateVersion[8]; /* 8 = SizeStateVersion */
    double E9RealCurrentGain; /* LONGREAL */
    double E9RealF2Bandwidth; /* LONGREAL */
    double E9F2Frequency; /* LONGREAL */
    double E9RsValue; /* LONGREAL */
    double E9RsFraction; /* LONGREAL */
    double E9GLeak; /* LONGREAL */
    double E9CFastAmp1; /* LONGREAL */
    double E9CFastAmp2; /* LONGREAL */
    double E9CFastTau; /* LONGREAL */
    double E9CSlow              ; /* LONGREAL */
    double E9GSeries            ; /* LONGREAL */
    double E9StimDacScale       ; /* LONGREAL */
    double E9CCStimScale        ; /* LONGREAL */
    double E9VHold              ; /* LONGREAL */
    double E9LastVHold          ; /* LONGREAL */
    double E9VpOffset           ; /* LONGREAL */
    double E9VLiquidJunction    ; /* LONGREAL */
    double E9CCIHold            ; /* LONGREAL */
    double E9CSlowStimVolts     ; /* LONGREAL */
    double E9CCTrackVHold       ; /* LONGREAL */
    double E9TimeoutLength      ; /* LONGREAL */
    double E9SearchDelay        ; /* LONGREAL */
    double E9MConductance       ; /* LONGREAL */
    double E9MCapacitance       ; /* LONGREAL */
    char E9SerialNumber[8]      ; /* 8 = SizeSerialNumber */
    short E9E9Boards           ; /* INT16 */
    short E9CSlowCycles        ; /* INT16 */
    short E9IMonAdc            ; /* INT16 */
    short E9VMonAdc            ; /* INT16 */
    short E9MuxAdc             ; /* INT16 */
    short E9TstDac             ; /* INT16 */
    short E9StimDac            ; /* INT16 */
    short E9StimDacOffset      ; /* INT16 */
    short E9MaxDigitalBit      ; /* INT16 */
    short E9SpareInt1       ; /* INT16 */
    short E9SpareInt2       ; /* INT16 */
    short E9SpareInt3       ; /* INT16 */

    char E9AmplKind           ; /* BYTE */
    char E9IsEpc9N            ; /* BYTE */
    char E9ADBoard            ; /* BYTE */
    char E9BoardVersion       ; /* BYTE */
    char E9ActiveE9Board      ; /* BYTE */
    char E9Mode               ; /* BYTE */
    char E9Range              ; /* BYTE */
    char E9F2Response         ; /* BYTE */

    char E9RsOn               ; /* BYTE */
    char E9CSlowRange         ; /* BYTE */
    char E9CCRange            ; /* BYTE */
    char E9CCGain             ; /* BYTE */
    char E9CSlowToTstDac      ; /* BYTE */
    char E9StimPath           ; /* BYTE */
    char E9CCTrackTau         ; /* BYTE */
    char E9WasClipping        ; /* BYTE */

    char E9RepetitiveCSlow    ; /* BYTE */
    char E9LastCSlowRange     ; /* BYTE */
    char E9Locked             ; /* BYTE */
    char E9CanCCFast          ; /* BYTE */
    char E9CanLowCCRange      ; /* BYTE */
    char E9CanHighCCRange     ; /* BYTE */
    char E9CanCCTracking      ; /* BYTE */
    char E9HasVmonPath        ; /* BYTE */

    char E9HasNewCCMode       ; /* BYTE */
    char E9Selector           ; /* CHAR */
    char E9HoldInverted       ; /* BYTE */
    char E9AutoCFast          ; /* BYTE */
    char E9AutoCSlow          ; /* BYTE */
    char E9HasVmonX100        ; /* BYTE */
    char E9TestDacOn          ; /* BYTE */
    char E9QMuxAdcOn          ; /* BYTE */

    double E9RealImon1Bandwidth ; /* LONGREAL */
    double E9StimScale          ; /* LONGREAL */

    char E9Gain               ; /* BYTE */
    char E9Filter1            ; /* BYTE */
    char E9StimFilterOn       ; /* BYTE */
    char E9RsSlow             ; /* BYTE */
    char E9Old1            ; /* BYTE */
    char E9CCCFastOn          ; /* BYTE */
    char E9CCFastSpeed        ; /* BYTE */
    char E9F2Source           ; /* BYTE */

    char E9TestRange          ; /* BYTE */
    char E9TestDacPath        ; /* BYTE */
    char E9MuxChannel         ; /* BYTE */
    char E9MuxGain64          ; /* BYTE */
    char E9VmonX100           ; /* BYTE */
    char E9IsQuadro           ; /* BYTE */
    char E9SpareBool4      ; /* BYTE */
    char E9SpareBool5      ; /* BYTE */

    double E9StimFilterHz       ; /* LONGREAL */
    double E9RsTau              ; /* LONGREAL */
    short E9FilterOffsetDac    ; /* INT16 */
    short E9ReferenceDac       ; /* INT16 */
    short E9SpareInt6       ; /* INT16 */
    short E9SpareInt7       ; /* INT16 */
    char E9Spares1[24]         ;

    char E9CalibDate[16]; /* 16 = SizeCalibDate */
    double E9SelHold; /* LONGREAL */
    char E9Spares2[32]; /* remaining */
};
C_ASSERT(sizeof(AmplifierState) == 400);

struct LockInParams {
    /* see definition in AmplTreeFile_v9.txt */
    double loExtCalPhase        ; /* LONGREAL */
    double loExtCalAtten        ; /* LONGREAL */
    double loPLPhase            ; /* LONGREAL */
    double loPLPhaseY1          ; /* LONGREAL */
    double loPLPhaseY2          ; /* LONGREAL */
    double loUsedPhaseShift     ; /* LONGREAL */
    double loUsedAttenuation    ; /* LONGREAL */
    char loSpares2[8]         ;
    char loExtCalValid        ; /* BOOLEAN */
    char loPLPhaseValid       ; /* BOOLEAN */
    char loLockInMode         ; /* BYTE */
    char loCalMode            ; /* BYTE */
    char loSpares[28]         ; /* remaining */
};
C_ASSERT(sizeof(LockInParams) == 96);

struct SeriesRecord {
    int SeMark; /* INT32 */
    char SeLabel[32]; /* String32Type */
    char SeComment[80]; /* String80Type */
    int SeSeriesCount; /* INT32 */
    int SeNumberSweeps; /* INT32 */
    int SeAmplStateOffset; /* INT32 */
    int SeAmplStateSeries; /* INT32 */
    char SeSeriesType; /* BYTE */
    char SeFiller1; /* BYTE */
    char SeFiller2; /* BYTE */
    char SeFiller3; /* BYTE */
    double SeTime; /* LONGREAL */
    double SePageWidth; /* LONGREAL */
    UserParamDescrType SeSwUserParamDescr[4]; /* ARRAY[0..3] OF UserParamDescrType = 4*40 */
    char SeFiller4[32]; /* 32 BYTE */
    double SeSeUserParams[4]; /* ARRAY[0..3] OF LONGREAL */
    LockInParams SeLockInParams; /* SeLockInSize = 96, see "Pulsed.de" */
    AmplifierState SeAmplifierState; /* AmplifierStateSize = 400 */
    char SeUsername[80]; /* String80Type */
    UserParamDescrType SeSeUserParamDescr[4]; /* ARRAY[0..3] OF UserParamDescrType = 4*40 */
    int SeFiller5; /* INT32 */
    int SeCRC; /* CARD32 */
};
C_ASSERT(sizeof(SeriesRecord) == 1120);

struct GroupRecord {
    int GrMark; /* INT32 */
    char GrLabel[32]; /* String32Size */
    char GrText[80]; /* String80Size */
    int GrExperimentNumber; /* INT32 */
    int GrGroupCount      ; /* INT32 */
    int GrCRC             ; /* CARD32 */
};
C_ASSERT(sizeof(GroupRecord) == 128);

struct RootRecord {
      /*
         NOTE: The "Version" field must be at offset zero in the file
               while the "Mark" field must be at offset zero in RAM!
       */
    int RoVersion         ; /* INT32 */
    int RoMark            ; /* INT32 */
    char RoVersionName[32]; /* String32Type */
    char RoAuxFileName[80]        ; /* String80Type */
    char RoRootText[400]           ; /* String400Type */
    double RoStartTime          ; /* LONGREAL */
    int RoMaxSamples         ; /* INT32 */
    int RoCRC                ; /* CARD32 */
    short RoFeatures           ; /* SET16 */
    short RoFiller1         ; /* INT16 */
    int RoFiller2         ; /* INT32 */
};
C_ASSERT(sizeof(RootRecord) == 544);


#ifndef RC_INVOKED
#pragma pack(pop)                      // return to default packing
#endif

struct Tree {
    std::vector<RootRecord> RootList;
    std::vector<GroupRecord> GroupList;
    std::vector<SeriesRecord> SeriesList;
    std::vector<SweepRecord> SweepList;
    std::vector<TraceRecord> TraceList;
    std::vector<TreeEntry> entries;
    bool needsByteSwap;
};
    
void printHeader(const BundleHeader& header) {
   
    std::cout << header.oSignature << std::endl;

    std::string strsig(header.oSignature);
    if (strsig == "DATA") {
        throw std::runtime_error("DATA file format not supported at present");
    } else if (strsig=="DAT1" || strsig=="DAT2") {
        // Newer format
        std::cout << header.oVersion << std::endl;
        std::cout << header.oTime << std::endl;
        std::cout << header.oItems << std::endl;
        std::cout << int(header.oIsLittleEndian[0]) << std::endl;
        if (strsig=="DAT1") {
            
        } else {
            // "DAT2"
            for (int k=0; k<12; ++k) {
                std::cout <<  header.oBundleItems[k].oStart << std::endl
                          <<  header.oBundleItems[k].oLength << std::endl
                          <<  header.oBundleItems[k].oExtension << std::endl;
            }
        }
    }
}

void ByteSwap(unsigned char * b, int n)
{
    register int i = 0;
    register int j = n-1;
    while (i<j)
    {
        std::swap(b[i], b[j]);
        i++, j--;
    }
}

#if 0
void ByteSwap16(void *ptr16) {
    unsigned short *uShort = (unsigned short *)ptr16;
    *uShort = ((*uShort >> 8) | (*uShort << 8));
}

void ByteSwap32(void* ptr32) {
    unsigned int *uInt = (unsigned int *)ptr32;
    *uInt = (((*uInt & 0x000000FF)<<24) + ((*uInt & 0x0000FF00)<<8) +
             ((*uInt & 0x00FF0000)>>8) +  ((*uInt & 0xFF000000)>>24));
}
#endif

void ShortByteSwap(short& s) {
    ByteSwap((unsigned char *) &s,sizeof(s));
}

void FloatByteSwap(float& s) {
    ByteSwap((unsigned char *) &s,sizeof(s));
}

void IntByteSwap(int& s) {
    ByteSwap((unsigned char *) &s,sizeof(s));
}

void DoubleByteSwap(double& s) {
    ByteSwap((unsigned char *) &s,sizeof(s));
}

void SwapItem(BundleItem& item) {
    ByteSwap32(item.oStart);
    ByteSwap32(item.oLength);
}

void SwapHeader(BundleHeader& header) {
    std::string strsig(header.oSignature);
    if (strsig == "DATA") {
        throw std::runtime_error("DATA file format not supported at present");
    } else if (strsig=="DAT1" || strsig=="DAT2") {
        // Newer format

        ByteSwap32(header.oTime);
        ByteSwap32(header.oItems);
        if (strsig=="DAT1") {
            
        } else {
            // "DAT2"
            for (int k=0; k<12; ++k) {
                SwapItem(header.oBundleItems[k]);
            }
        }
    }

}

void SwapRoot(RootRecord& root) {
    ByteSwap32(root.RoVersion);
    ByteSwap32(root.RoMark);
    ByteSwap32(root.RoStartTime);
    ByteSwap32(root.RoMaxSamples);
    ByteSwap32(root.RoCRC);
    ByteSwap16(root.RoFeatures);
    ByteSwap16(root.RoFiller1);
    ByteSwap32(root.RoFiller2);
}

void SwapGroup(GroupRecord& group) {
    ByteSwap32(group.GrMark); /* INT32 */
    ByteSwap32(group.GrExperimentNumber); /* INT32 */
    ByteSwap32(group.GrGroupCount      ); /* INT32 */
    ByteSwap32(group.GrCRC             ); /* CARD32 */
}

void SwapSeries(SeriesRecord& series) {
    ByteSwap32(series.SeMark); /* INT32 */
    ByteSwap32(series.SeSeriesCount); /* INT32 */
    ByteSwap32(series.SeNumberSweeps); /* INT32 */
    ByteSwap32(series.SeAmplStateOffset); /* INT32 */
    ByteSwap32(series.SeAmplStateSeries); /* INT32 */
    ByteSwap32(series.SeTime); /* LONGREAL */
    ByteSwap32(series.SePageWidth); /* LONGREAL */
    // TODO UserParamDescrType SeSwUserParamDescr[4]); /* ARRAY[0..3] OF UserParamDescrType = 4*40 */
    // TODO SeSeUserParams[4]); /* ARRAY[0..3] OF LONGREAL */
    // TODO LockInParams SeLockInParams); /* SeLockInSize = 96, see "Pulsed.de" */
    // TODO AmplifierState SeAmplifierState); /* AmplifierStateSize = 400 */
    // TODO UserParamDescrType SeSeUserParamDescr[4]); /* ARRAY[0..3] OF UserParamDescrType = 4*40 */
    ByteSwap32(series.SeFiller5); /* INT32 */
    ByteSwap32(series.SeCRC); /* CARD32 */
}

void SwapSweep(SweepRecord& sweep) {
    ByteSwap32(sweep.SwMark); /* INT32 */
    ByteSwap32(sweep.SwAuxDataFileOffset); /* INT32 */
    ByteSwap32(sweep.SwStimCount); /* INT32 */
    ByteSwap32(sweep.SwSweepCount); /* INT32 */
    ByteSwap32(sweep.SwTime); /* LONGREAL */
    ByteSwap32(sweep.SwTimer); /* LONGREAL */
    // TODO SwSwUserParams[4]); /* ARRAY[0..3] OF LONGREAL */
    ByteSwap32(sweep.SwTemperature); /* LONGREAL */
    ByteSwap32(sweep.SwOldIntSol); /* INT32 */
    ByteSwap32(sweep.SwOldExtSol); /* INT32 */
    ByteSwap16(sweep.SwDigitalIn); /* SET16 */
    ByteSwap16(sweep.SwSweepKind); /* SET16 */
    ByteSwap32(sweep.SwFiller1); /* INT32 */
    // TODO ByteSwap32(sweep.SwMarkers[4]); /* ARRAY[0..3] OF LONGREAL */
    ByteSwap32(sweep.SwFiller2); /* INT32 */
    ByteSwap32(sweep.SwCRC); /* CARD32 */
}

void SwapTrace(TraceRecord& trace) {
    ByteSwap32(trace.TrMark); /* INT32 */
    ByteSwap32(trace.TrTraceCount); /* INT32 */
    ByteSwap32(trace.TrData); /* INT32 */
    ByteSwap32(trace.TrDataPoints); /* INT32 */
    ByteSwap32(trace.TrInternalSolution); /* INT32 */
    ByteSwap32(trace.TrAverageCount); /* INT32 */
    ByteSwap32(trace.TrLeakCount); /* INT32 */
    ByteSwap32(trace.TrLeakTraces); /* INT32 */
    ByteSwap16(trace.TrDataKind); /* SET16 */
    ByteSwap16(trace.TrFiller1); /* SET16 */
    ByteSwap32(trace.TrDataScaler); /* LONGREAL */
    ByteSwap32(trace.TrTimeOffset); /* LONGREAL */
    ByteSwap32(trace.TrZeroData); /* LONGREAL */
    ByteSwap32(trace.TrXInterval); /* LONGREAL */
    ByteSwap32(trace.TrXStart); /* LONGREAL */
    ByteSwap32(trace.TrYRange); /* LONGREAL */
    ByteSwap32(trace.TrYOffset); /* LONGREAL */
    ByteSwap32(trace.TrBandwidth); /* LONGREAL */
    ByteSwap32(trace.TrPipetteResistance); /* LONGREAL */
    ByteSwap32(trace.TrCellPotential); /* LONGREAL */
    ByteSwap32(trace.TrSealResistance); /* LONGREAL */
    ByteSwap32(trace.TrCSlow); /* LONGREAL */
    ByteSwap32(trace.TrGSeries); /* LONGREAL */
    ByteSwap32(trace.TrRsValue); /* LONGREAL */
    ByteSwap32(trace.TrGLeak); /* LONGREAL */
    ByteSwap32(trace.TrMConductance); /* LONGREAL */
    ByteSwap32(trace.TrLinkDAChannel); /* INT32 */
    ByteSwap16(trace.TrAdcChannel); /* INT16 */
    ByteSwap32(trace.TrYmin); /* LONGREAL */
    ByteSwap32(trace.TrYmax); /* LONGREAL */
    ByteSwap32(trace.TrSourceChannel); /* INT32 */
    ByteSwap32(trace.TrExternalSolution); /* INT32 */
    ByteSwap32(trace.TrCM); /* LONGREAL */
    ByteSwap32(trace.TrGM); /* LONGREAL */
    ByteSwap32(trace.TrPhase); /* LONGREAL */
    ByteSwap32(trace.TrDataCRC); /* CARD32 */
    ByteSwap32(trace.TrCRC); /* CARD32 */
    ByteSwap32(trace.TrGS); /* LONGREAL */
    ByteSwap32(trace.TrSelfChannel); /* INT32 */
    ByteSwap32(trace.TrFiller2); /* SET16 */
}

BundleHeader getBundleHeader(FILE* fh) {
    BundleHeader header;

    int res = 0;
    res = fseek(fh, 0, SEEK_SET);
    res = fread(&header, sizeof(BundleHeader), 1, fh);
    return header;
}

RootRecord getRoot(FILE* fh, bool needsByteSwap) {
    int res = 0;
    RootRecord rec;
    res = fread(&rec, sizeof(RootRecord), 1, fh);
    if (needsByteSwap) {
        SwapRoot(rec);
    }
    return rec;
}

GroupRecord getGroup(FILE* fh, bool needsByteSwap) {
    int res = 0;
    GroupRecord rec;
    res = fread(&rec, sizeof(GroupRecord), 1, fh);
    if (needsByteSwap) {
        SwapGroup(rec);
    }
    return rec;
}

SeriesRecord getSeries(FILE* fh, bool needsByteSwap) {
    int res = 0;
    SeriesRecord rec;
    res = fread(&rec, sizeof(SeriesRecord), 1, fh);
    if (needsByteSwap) {
        SwapSeries(rec);
    }
    return rec;
}

SweepRecord getSweep(FILE* fh, bool needsByteSwap) {
    int res = 0;
    SweepRecord rec;
    res = fread(&rec, sizeof(SweepRecord), 1, fh);
    if (needsByteSwap) {
        SwapSweep(rec);
    }
    return rec;
}

TraceRecord getTrace(FILE* fh, bool needsByteSwap) {
    int res = 0;
    TraceRecord rec;
    res = fread(&rec, sizeof(TraceRecord), 1, fh);
    if (needsByteSwap) {
        SwapTrace(rec);
    }
    return rec;
}

int findExt(const BundleHeader& header, const std::string& ext) {
    int extNo = -1;
    for (int k=0; k<12; ++k) {
        if (header.oBundleItems[k].oExtension == ext) {
            extNo = k;
        }
    }
    return extNo;
}
void getOneRecord(FILE* fh, Level level, Tree& TreeInOut, int& CounterInOut) {
    // Gets one record
    int idx = -1;
    switch (level) {
     case root:
         idx = TreeInOut.RootList.size();
         TreeInOut.RootList.push_back(getRoot(fh, TreeInOut.needsByteSwap));
         break;
     case group:
         idx = TreeInOut.GroupList.size();
         TreeInOut.GroupList.push_back(getGroup(fh, TreeInOut.needsByteSwap));
         break;
     case series:
         idx = TreeInOut.SeriesList.size();
         TreeInOut.SeriesList.push_back(getSeries(fh, TreeInOut.needsByteSwap));
         break;
     case sweep:
         idx = TreeInOut.SweepList.size();
         TreeInOut.SweepList.push_back(getSweep(fh, TreeInOut.needsByteSwap));
         break;
     case trace:
         idx = TreeInOut.TraceList.size();
         TreeInOut.TraceList.push_back(getTrace(fh, TreeInOut.needsByteSwap));
         break;
     default:
         throw std::runtime_error("Couldn't read record");
    }

    TreeInOut.entries.push_back(TreeEntry(level, CounterInOut, idx));
    CounterInOut++;
}

int getOneLevel(FILE* fh, const std::vector<int>& Sizes, Level level, Tree& TreeInOut, int& PositionInOut, int& CounterInOut) {
    // Gets one record of the tree and the number of children
    getOneRecord(fh, level, TreeInOut, CounterInOut);
    PositionInOut += Sizes[level];
    fseek(fh, PositionInOut, SEEK_SET);
    int nchild = 0;
    int res = 0;
    res = fread(&nchild, sizeof(int), 1, fh);
    if (TreeInOut.needsByteSwap) {
        ByteSwap32(nchild);
    }
    PositionInOut = ftell(fh);
    return nchild;
}

void getTreeReentrant(FILE* fh, const std::vector<int>& Sizes, Level level, Tree& TreeInOut, int& PositionInOut, int& CounterInOut) {
    // Recursive routine called from LoadTree
    int nchild = getOneLevel(fh, Sizes, level, TreeInOut, PositionInOut, CounterInOut);
    for (int k=0; k<nchild; ++k) {
        getTreeReentrant(fh, Sizes, int2Level(level+1), TreeInOut, PositionInOut, CounterInOut);
    }
}

Tree getTree(FILE* fh, const std::vector<int>& Sizes, int& PositionInOut, bool needsByteSwap) {
    Tree tree;
    tree.needsByteSwap = needsByteSwap;
    // Main entry point for loading tree
    int Counter = 0;
    getTreeReentrant(fh, Sizes, int2Level(0), tree, PositionInOut, Counter);
    return tree;
}

std::string time2date(double t) {
    long time = (long)t - 1580970496;
    if (time<0) {
        time += 4294967296;
    }
    time += 9561652096;
    time_t timer(time);
    std::string datestr(ctime(&timer)); 
    return datestr;
}

void ReadData(FILE* fh, const Tree& tree, bool progress, Recording& RecordingInOut
#ifndef MODULE_ONLY
                   , wxProgressDialog& progDlg
#endif
                  ) {

    int nsweeps = tree.SweepList.size();
    int ntraces = tree.TraceList.size();

    int nchannels = ntraces/nsweeps;
    RecordingInOut.resize(nchannels);
    int res = 0;
    for (int nc=0; nc<nchannels; ++nc) {
        RecordingInOut[nc].resize(nsweeps);
        for (int ns=0; ns<nsweeps; ++ns) {
            // nstree=nc; nstree<ntraces; nstree += nchannels) {
            // int ns = nstree/nchannels;
            int nstree = (ns*nchannels)+nc;
            if (progress) {
                int progbar =
                    // Channel contribution:
                    (int)(((double)nc/(double)nchannels)*100.0+
                          // Section contribution:
                          (double)ns/(double)nsweeps*(100.0/nchannels));
#ifndef MODULE_ONLY
                wxString progStr;
                progStr << "Reading channel #" << nc + 1 << " of " << nchannels
                        << ", Section #" << ns + 1 << " of " << nsweeps;
                bool skip = false;
                progDlg.Update(progbar, progStr, &skip);
                if (skip) {
                    RecordingInOut.resize(0);
                    return;
                }
#else
                std::cout << "\r";
                std::cout << progbar << "%" << std::flush;
#endif
            }
            int npoints = tree.TraceList[nstree].TrDataPoints;
            RecordingInOut[nc][ns].resize(npoints);

            fseek(fh, tree.TraceList[nstree].TrData, SEEK_SET);
            switch (int(tree.TraceList[nstree].TrDataFormat)) {
             case 0: {
                 /*int16*/
                 std::vector<short> tmpSection(npoints);
                 res = fread(&tmpSection[0], sizeof(short), npoints, fh);
                 if (tree.needsByteSwap) 
                     std::for_each(tmpSection.begin(), tmpSection.end(), ShortByteSwap);

                 std::copy(tmpSection.begin(), tmpSection.end(), RecordingInOut[nc][ns].get_w().begin());
                 break;
             }
             case 1: {
                 /*int32*/
                 std::vector<int> tmpSection(npoints);
                 res = fread(&tmpSection[0], sizeof(int), npoints, fh);
                 if (tree.needsByteSwap) 
                     std::for_each(tmpSection.begin(), tmpSection.end(), IntByteSwap);
                 std::copy(tmpSection.begin(), tmpSection.end(), RecordingInOut[nc][ns].get_w().begin());
                 break;
             }
             case 2: {
                 /*double16*/
                 std::vector<float> tmpSection(npoints);
                 res = fread(&tmpSection[0], sizeof(float), npoints, fh);
                 if (tree.needsByteSwap) 
                     std::for_each(tmpSection.begin(), tmpSection.end(), FloatByteSwap);
                 std::copy(tmpSection.begin(), tmpSection.end(), RecordingInOut[nc][ns].get_w().begin());
                 break;
             }
             case 3: {
                 /*double32*/
                 std::vector<double> tmpSection(npoints);
                 res = fread(&tmpSection[0], sizeof(double), npoints, fh);
                 if (tree.needsByteSwap) 
                     std::for_each(tmpSection.begin(), tmpSection.end(), DoubleByteSwap);
                 std::copy(tmpSection.begin(), tmpSection.end(), RecordingInOut[nc][ns].get_w().begin());
                 break;
             }
             default:
                 throw std::runtime_error("Unknown data format while reading heka file");
            }
            double factor = 1.0;
            if (std::string(tree.TraceList[nc].TrYUnit) == "V") {
                RecordingInOut[nc].SetYUnits("mV");
                factor = 1.0e3;
            } else if (std::string(tree.TraceList[nc].TrYUnit) == "A") {
                RecordingInOut[nc].SetYUnits("pA");
                factor = 1.0e12;
            } else {
                RecordingInOut[nc].SetYUnits(tree.TraceList[nc].TrYUnit);
            }
            factor *=  tree.TraceList[nc].TrDataScaler;
            RecordingInOut[nc][ns].get_w() = stf::vec_scal_mul(RecordingInOut[nc][ns].get(), factor);
            RecordingInOut[nc][ns].get_w() = stf::vec_scal_plus(RecordingInOut[nc][ns].get(), tree.TraceList[nc].TrZeroData);
        }
        RecordingInOut[nc].SetChannelName(tree.TraceList[nc].TrLabel);
        
    }
    double tsc = 1.0;
    std::string xunits(tree.TraceList[0].TrXUnit);
    if (xunits == "s") {
        tsc=1.0e3;
    } else if (xunits == "ms") {
        tsc=1.0;
    } else if (xunits == "µs") {
        tsc=1.0e-3;
    } else {
        throw std::runtime_error("Unsupported time units");
    }
    RecordingInOut.SetXScale(tree.TraceList[0].TrXInterval*tsc);

}

void stf::importHEKAFile(const wxString &fName, Recording &ReturnData, bool progress) {
#ifndef MODULE_ONLY
    wxProgressDialog progDlg(wxT("HEKA binary file import"), wxT("Starting file import"),
                             100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL | wxPD_CAN_SKIP );
#endif
    wxString errorMsg(wxT("Exception while calling importHEKAFile():\n"));
    wxString yunits;
    int res = 0;
    
    // Open file
    FILE* dat_fh = fopen(fName.c_str(), "rb");
    if (dat_fh==NULL) {
        return;
    }

    BundleHeader header = getBundleHeader(dat_fh);
    bool needsByteSwap = (int(header.oIsLittleEndian[0]) == 0);
    if (needsByteSwap) {
        SwapHeader(header);
    }

    int start = 0;
    bool isBundled = false;
    if (std::string(header.oSignature)=="DAT2") {
        // find the pulse data
        isBundled = true;
        int extNo = findExt(header, ".pul");
        if (extNo < 0) {
            throw std::runtime_error("Couldn't find .pul file in bundle");
        }
        start = header.oBundleItems[extNo].oStart;
    } else {
        throw std::runtime_error("Can only deal with bundled data at present");
    }

    // Base of tree
    fseek(dat_fh, start, SEEK_SET);
    char cMagic[4];
    res = fread(&cMagic[0], sizeof(char), 4, dat_fh);
    std::string magic(cMagic);
    int levels = 0;
    res = fread(&levels, sizeof(int), 1, dat_fh);
    if (needsByteSwap) {
        ByteSwap32(levels);
    }

    std::vector<int> sizes(levels);
    if (levels!=0)
        res = fread(&sizes[0], sizeof(int), levels, dat_fh);

    if (needsByteSwap)
        std::for_each(sizes.begin(), sizes.end(), IntByteSwap);

    // Get the tree from the pulse file
    int pos = ftell(dat_fh);
    Tree tree = getTree(dat_fh, sizes, pos, needsByteSwap);
    std::string date = time2date(tree.RootList[0].RoStartTime);

    if (isBundled) {
        // find the data
        int extNo = findExt(header, ".dat");
        if (extNo < 0) {
            throw std::runtime_error("Couldn't find .dat file in bundle");
        }
        start = header.oBundleItems[extNo].oStart;
    } else {
        throw std::runtime_error("Can only deal with bundled data at present");
    }

    // Now set pointer to the start of the data
    fseek(dat_fh, start, SEEK_SET);

    // NOW IMPORT
    ReadData(dat_fh, tree, progress, ReturnData
#ifndef MODULE_ONLY
                          , progDlg
#endif
                         );

#ifdef MODULE_ONLY
    if (progress) {
        std::cout << "\r";
        std::cout << "100%" << std::endl;
    }
#endif

    // Close file
    fclose(dat_fh);
}

