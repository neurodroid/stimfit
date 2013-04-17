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

// Copyright 2012,2013 Alois Schloegl, IST Austria <alois.schloegl@ist.ac.at>

#include <string>
#include <iomanip>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <sstream>
#include <string.h>

#include "../stfio.h"

#if 1
#include <biosig.h>
/* these are internal biosig functions, defined in biosig-dev.h which is not always available */
extern "C" size_t ifwrite(void* buf, size_t size, size_t nmemb, HDRTYPE* hdr);
extern "C" uint32_t lcm(uint32_t A, uint32_t B);
#if !defined(__MINGW32__) && !defined(_MSC_VER)
#include <endian.h>	// not available on mingw
#endif
#endif

#ifdef _MSC_VER
    #ifndef NAN
        static const unsigned long __nan[2] = {0xffffffff, 0x7fffffff};
        #define NAN (*(const float *) __nan)
    #endif
    long int lround(double x) {
        int i = (long int) x;
        if (x >= 0.0) {
            return ((x-i) >= 0.5) ? (i + 1) : (i);
        } else {
            return (-x+i >= 0.5) ? (i - 1) : (i);
        }
    }
#endif

#include "./biosiglib.h"

/* Redefine BIOSIG_VERSION for versions < 1 */
#if (BIOSIG_VERSION_MAJOR < 1)
#undef BIOSIG_VERSION
#define BIOSIG_VERSION (BIOSIG_VERSION_MAJOR * 10000 + BIOSIG_VERSION_MINOR * 100 + BIOSIG_PATCHLEVEL)
#endif

void stfio::importBSFile(const std::string &fName, Recording &ReturnData, ProgressInfo& progDlg) {

    std::string errorMsg("Exception while calling std::importBSFile():\n");
    std::string yunits;
    // =====================================================================================================================
    //
    // Open an AxoGraph file and read in the data
    //
    // =====================================================================================================================
    
    HDRTYPE* hdr =  sopen( fName.c_str(), "r", NULL );
    if (hdr==NULL) {
        errorMsg += "\nBiosig header is empty";
        ReturnData.resize(0);
        throw std::runtime_error(errorMsg.c_str());
    }
#if !defined(BIOSIG_VERSION) || (BIOSIG_VERSION < 10501)
    if (hdr->TYPE==ABF) {
        /*
           biosig v1.5.0 and earlier does not always return
           with a proper error message for ABF files.
           This causes problems with the ABF fallback mechanism
        */
#else
    if (hdr->TYPE==ABF && hdr->AS.B4C_ERRNUM) {
        /* this triggers the fall back mechanims w/o reporting an error message */
#endif
        ReturnData.resize(0);
        destructHDR(hdr);	// free allocated memory
        throw std::runtime_error(errorMsg.c_str());
    }
    errorMsg += "\n";
#if defined(BIOSIG_VERSION) && (BIOSIG_VERSION > 10400)
    if (serror2(hdr)) {
        errorMsg += std::string(hdr->AS.B4C_ERRMSG);
#else
    if (serror()) {
	errorMsg += std::string(B4C_ERRMSG);
#endif
        ReturnData.resize(0);
        destructHDR(hdr);	// free allocated memory
        throw std::runtime_error(errorMsg.c_str());
    }

    // ensure the event table is in chronological order	
    sort_eventtable(hdr);

    /*
	count sections and generate list of indices indicating start and end of sweeps
     */	
    size_t LenIndexList = 256; 
    if (LenIndexList > hdr->EVENT.N) LenIndexList = hdr->EVENT.N + 2;
    size_t *SegIndexList = (size_t*)malloc(LenIndexList*sizeof(size_t));
    uint32_t nsections = 0;
    SegIndexList[nsections] = 0;
    size_t MaxSectionLength = 0;
    for (size_t k=0; k <= hdr->EVENT.N; k++) {
        if (LenIndexList <= nsections+2) {
            // allocate more memory as needed
		    LenIndexList *=2;
		    SegIndexList = (size_t*)realloc(SegIndexList, LenIndexList*sizeof(size_t));
	    }
        /*
            count number of sections and stores it in nsections;
            EVENT.TYP==0x7ffe indicate number of breaks between sweeps
	        SegIndexList includes index to first sample and index to last sample,
	        thus, the effective length of SegIndexList is the number of 0x7ffe plus two.
	    */
        if (0)                              ;
        else if (k >= hdr->EVENT.N)         SegIndexList[++nsections] = hdr->NRec*hdr->SPR;
        else if (hdr->EVENT.TYP[k]==0x7ffe) SegIndexList[++nsections] = hdr->EVENT.POS[k];
        else                                continue;

        size_t SPS = SegIndexList[nsections]-SegIndexList[nsections-1];	// length of segment, samples per segment
	    if (MaxSectionLength < SPS) MaxSectionLength = SPS;
    }

    /*************************************************************************
        Date and time conversion
     *************************************************************************/
    struct tm t;
#if (BIOSIG_VERSION_MAJOR > 0)
    gdf_time2tm_time_r(hdr->T0, &t);
#else
    struct tm* tp;
    tp = gdf_time2tm_time(hdr->T0);
    t = *tp;
#endif
    // allocate local memory for intermediate results;
    const int strSize=100;
    char str[strSize];

    strftime(str,100,"%F",&t);
    ReturnData.SetDate(str);
    strftime(str,100,"%D",&t);
    ReturnData.SetTime(str);

    /*************************************************************************
        rescale data to mV and pA
     *************************************************************************/    
    for (int ch=0; ch < hdr->NS; ++ch) {
        CHANNEL_TYPE *hc = hdr->CHANNEL+ch;
        double scale = PhysDimScale(hc->PhysDimCode); 
        switch (hc->PhysDimCode & 0xffe0) {
        case 4256:  // Volt
                hc->PhysDimCode = 4274; // = PhysDimCode("mV");
                scale *=1e3;   // V->mV
                hc->PhysMax *= scale;         
                hc->PhysMin *= scale;         
                hc->Cal *= scale;         
                hc->Off *= scale;         
                break; 
        case 4160:  // Ampere
                hc->PhysDimCode = 4181; // = PhysDimCode("pA");
                scale *=1e12;   // A->pA
                hc->PhysMax *= scale;         
                hc->PhysMin *= scale;         
                hc->Cal *= scale;         
                hc->Off *= scale;         
                break; 
        }     
    }

    /*************************************************************************
        read bulk data 
     *************************************************************************/    
    hdr->FLAG.ROW_BASED_CHANNELS = 0;
    /* size_t blks = */ sread(NULL, 0, hdr->NRec, hdr);

    int numberOfChannels = 0;
    for (int k=0; k < hdr->NS; k++)
        if (hdr->CHANNEL[k].OnOff==1)
            numberOfChannels++;

#ifdef _STFDEBUG
    std::cout << "Number of channels: " << hdr->NS << std::endl;
    std::cout << "Number of records per channel: " << hdr->NRec << std::endl;
    std::cout << "Number of samples per record: " << hdr->SPR << std::endl;
    std::cout << "Data size: " << hdr->data.size[0] << "x" << hdr->data.size[1] << std::endl;
    std::cout << "Sampling rate: " << hdr->SampleRate << std::endl;
    std::cout << "Number of events: " << hdr->EVENT.N << std::endl;
    /*int res = */ hdr2ascii(hdr, stdout, 4);
#endif

    int NS = 0;   // number of non-empty channels
    for (size_t nc=0; nc<hdr->NS; ++nc) {

        if (hdr->CHANNEL[nc].OnOff == 0) continue;

        Channel TempChannel(nsections);
        TempChannel.SetChannelName(hdr->CHANNEL[nc].Label);
#if defined(BIOSIG_VERSION) && (BIOSIG_VERSION > 10301)
        TempChannel.SetYUnits(PhysDim3(hdr->CHANNEL[nc].PhysDimCode));
#else
        PhysDim(hdr->CHANNEL[nc].PhysDimCode,str);
        TempChannel.SetYUnits(str);
#endif

        for (size_t ns=1; ns<=nsections; ns++) {
	        size_t SPS = SegIndexList[ns]-SegIndexList[ns-1];	// length of segment, samples per segment

		int progbar = 100.0*(1.0*ns/nsections + nc)/(hdr->NS);
		std::ostringstream progStr;
		progStr << "Reading channel #" << nc + 1 << " of " << hdr->NS
			<< ", Section #" << ns << " of " << nsections;
		progDlg.Update(progbar, progStr.str());

		/* unused //
		char sweepname[20];
		sprintf(sweepname,"sweep %i",(int)ns);		
		*/
		Section TempSection(
                                SPS, // TODO: hdr->nsamplingpoints[nc][ns]
                                "" // TODO: hdr->sectionname[nc][ns]
            	);

		std::copy(&(hdr->data.block[nc*hdr->SPR*hdr->NRec + SegIndexList[ns-1]]), 
			  &(hdr->data.block[nc*hdr->SPR*hdr->NRec + SegIndexList[ns]]), 
			  TempSection.get_w().begin() );

		try {
			TempChannel.InsertSection(TempSection, ns-1);
		}
		catch (...) {
			ReturnData.resize(0);
			destructHDR(hdr);
			throw;
		}
	}        
    try {
        if ((int)ReturnData.size() < numberOfChannels) {
			ReturnData.resize(numberOfChannels);
		}
		ReturnData.InsertChannel(TempChannel, NS++);
    }
    catch (...) {
		ReturnData.resize(0);
		destructHDR(hdr);
		throw;
        }
    }

    free(SegIndexList); 	

    ReturnData.SetComment ( hdr->ID.Recording );

    sprintf(str,"v%i.%i.%i (compiled on %s %s)",BIOSIG_VERSION_MAJOR,BIOSIG_VERSION_MINOR,BIOSIG_PATCHLEVEL,__DATE__,__TIME__);
    std::string Desc = std::string("importBiosig with libbiosig ")+std::string(str);

    if (hdr->ID.Technician)
            Desc += std::string ("\nTechnician:\t") + std::string (hdr->ID.Technician);
    Desc += std::string( "\nCreated with: ");
    if (hdr->ID.Manufacturer.Name)
        Desc += std::string( hdr->ID.Manufacturer.Name );
    if (hdr->ID.Manufacturer.Model)
        Desc += std::string( hdr->ID.Manufacturer.Model );
    if (hdr->ID.Manufacturer.Version)
        Desc += std::string( hdr->ID.Manufacturer.Version );
    if (hdr->ID.Manufacturer.SerialNumber)
        Desc += std::string( hdr->ID.Manufacturer.SerialNumber );

    Desc += std::string ("\nUser specified Annotations:\n");
    for (size_t k=0; k < hdr->EVENT.N; k++) {
        if (hdr->EVENT.TYP[k] < 256) {
            sprintf(str,"%f s:\t",hdr->EVENT.POS[k]/hdr->EVENT.SampleRate);
            Desc += std::string( str );
            Desc += std::string( hdr->EVENT.CodeDesc[hdr->EVENT.TYP[k]] ) + "\n";
        }
    }
    ReturnData.SetFileDescription(Desc);
    //ReturnData.SetGlobalSectionDescription(Desc);

    ReturnData.SetXScale(1000.0/hdr->SampleRate);
    ReturnData.SetXUnits("ms");
    ReturnData.SetScaling("biosig scaling factor");

    struct tm T; 
#if (BIOSIG_VERSION_MAJOR > 0)
    gdf_time2tm_time_r(hdr->T0, &T);
#else
    struct tm* Tp;
    Tp = gdf_time2tm_time(hdr->T0);
    T = *Tp;
#endif
    strftime(str,strSize,"%Y-%m-%d",&T);
    ReturnData.SetDate(str);
    strftime(str,strSize,"%H:%M:%S",&T);
    ReturnData.SetTime(str);

#ifdef MODULE_ONLY
    if (progress) {
        std::cout << "\r";
        std::cout << "100%" << std::endl;
    }
#endif

    destructHDR(hdr);
}


bool stfio::exportBiosigFile(const std::string& fName, const Recording& Data, stfio::ProgressInfo& progDlg) {
/*
    converts the internal data structure to libbiosig's internal structure
    and saves the file as gdf file.

    The data in converted into the raw data format, and not into the common
    data matrix.
*/

    HDRTYPE* hdr = constructHDR(Data.size(), 0);
    assert(hdr->NS == Data.size());

	/* Initialize all header parameters */
    hdr->TYPE = GDF;
    hdr->VERSION = 3;   // latest version

    struct tm t;
    char *str;
    str = strdup(Data.GetDate().c_str());
    t.tm_year = strtol(str,&str,10)-1900;
    t.tm_mon  = strtol(str+1,&str,10)-1;
    t.tm_mday = strtol(str+1,&str,10);

    str = strdup(Data.GetTime().c_str());
    t.tm_hour = strtol(str,&str,10);
    t.tm_min = strtol(str+1,&str,10);
    t.tm_sec = strtol(str+1,&str,10);

    hdr->T0   = tm_time2gdf_time(&t);

    const char *xunits = Data.GetXUnits().c_str();
#if (BIOSIG_VERSION_MAJOR > 0)
    uint16_t pdc = PhysDimCode(xunits);
#else
    uint16_t pdc = PhysDimCode((char*)xunits);
#endif
    if ((pdc & 0xffe0) == PhysDimCode("s")) {
        fprintf(stderr,"Stimfit exportBiosigFile: xunits [%s] has not proper units, assume [ms]\n",Data.GetXUnits().c_str());
        pdc = PhysDimCode("ms");
    }
    hdr->SampleRate = 1.0/(PhysDimScale(pdc) * Data.GetXScale());
    hdr->SPR  = 1;

    hdr->FLAG.UCAL = 0;
    hdr->FLAG.OVERFLOWDETECTION = 0;

    hdr->FILE.COMPRESSION = 0;

	/* Initialize all channel parameters */
    size_t k, m;
    for (k = 0; k < hdr->NS; ++k) {
        CHANNEL_TYPE *hc = hdr->CHANNEL+k;

        hc->PhysMin = -1e9;
        hc->PhysMax = 1e9;
        hc->DigMin  = -1e9;
        hc->DigMax  = 1e9;
        hc->Cal    = 1.0;
        hc->Off    = 0.0;

        /* Channel descriptions. */
        strncpy(hc->Label, Data[k].GetChannelName().c_str(), MAX_LENGTH_LABEL);
#if (BIOSIG_VERSION_MAJOR > 0)
        hc->PhysDimCode = PhysDimCode(Data[k].GetYUnits().c_str());
#else
        hc->PhysDimCode = PhysDimCode((char*)Data[k].GetYUnits().c_str());
#endif
        hc->OnOff      = 1;
        hc->LeadIdCode = 0;

        hc->TOffset  = 0.0;
        hc->Notch    = NAN;
        hc->LowPass  = NAN;
        hc->HighPass = NAN;
        hc->Impedance= NAN;

        hc->SPR    = hdr->SPR;
        hc->GDFTYP = 17; 	// double

        // each segment gets one marker, roughly
        hdr->EVENT.N += Data[k].size();

        size_t m,len = 0;
        for (len=0, m = 0; m < Data[k].size(); ++m) {
            unsigned div = lround(Data[k][m].GetXScale()/Data.GetXScale());
            hc->SPR = lcm(hc->SPR,div);  // sampling interval of m-th segment in k-th channel
            len += div*Data[k][m].size();
        }
        hdr->SPR = lcm(hdr->SPR, hc->SPR);

        if (k==0) {
            hdr->NRec = len;
        }
        else if ((size_t)hdr->NRec != len) {
            destructHDR(hdr);
            throw std::runtime_error("File can't be exported:\n"
                "No data or traces have different sizes" );

            return false;
        }
    }

    hdr->AS.bpb = 0;
    for (k = 0; k < hdr->NS; ++k) {
        CHANNEL_TYPE *hc = hdr->CHANNEL+k;
        hc->SPR = hdr->SPR / hc->SPR;
        hc->bi  = hdr->AS.bpb;
        hdr->AS.bpb += hc->SPR * 8; /* its always double */
    }

	/***
	    build Event table for storing segment information
	 ***/
	size_t N = hdr->EVENT.N * 2;    // about two events per segment
	hdr->EVENT.POS = (uint32_t*)realloc(hdr->EVENT.POS, N * sizeof(*hdr->EVENT.POS));
	hdr->EVENT.DUR = (uint32_t*)realloc(hdr->EVENT.DUR, N * sizeof(*hdr->EVENT.DUR));
	hdr->EVENT.TYP = (uint16_t*)realloc(hdr->EVENT.TYP, N * sizeof(*hdr->EVENT.TYP));
	hdr->EVENT.CHN = (uint16_t*)realloc(hdr->EVENT.CHN, N * sizeof(*hdr->EVENT.CHN));
#if (BIOSIG_VERSION >= 10500)
	hdr->EVENT.TimeStamp = (gdf_time*)realloc(hdr->EVENT.TimeStamp, N * sizeof(gdf_time));
#endif

    /* check whether all segments have same size */
    {
        char flag = (hdr->NS>0);
        size_t m, POS, pos;
        for (k=0; k < hdr->NS; ++k) {
            pos = Data[k].size();
            if (k==0)
                POS = pos;
            else
                flag &= (POS == pos);
        }
        for (m=0; flag && (m < Data[(size_t)0].size()); ++m) {
            for (k=0; k < hdr->NS; ++k) {
                pos = Data[k][m].size() * lround(Data[k][m].GetXScale()/Data.GetXScale());
                if (k==0)
                    POS = pos;
                else
                    flag &= (POS == pos);
            }
        }
        if (!flag) {
            destructHDR(hdr);
            throw std::runtime_error(
                    "File can't be exported:\n"
                    "Traces have different sizes or no channels found"
            );
            return false;
        }
    }

        N=0;
        k=0;
        size_t pos = 0;
        for (m=0; m < (Data[k].size()); ++m) {
            if (pos > 0) {
                // start of new segment after break
                hdr->EVENT.POS[N] = pos;
                hdr->EVENT.TYP[N] = 0x7ffe;
                hdr->EVENT.CHN[N] = 0;
                hdr->EVENT.DUR[N] = 0;
                N++;
            }
#if 0
            // event description
            hdr->EVENT.POS[N] = pos;
            FreeTextEvent(hdr, N, "myevent");
            //FreeTextEvent(hdr, N, Data[k][m].GetSectionDescription().c_str()); // TODO
            hdr->EVENT.CHN[N] = k;
            hdr->EVENT.DUR[N] = 0;
            N++;
#endif
            pos += Data[k][m].size() * lround(Data[k][m].GetXScale()/Data.GetXScale());
        }

        hdr->EVENT.N = N;
        hdr->EVENT.SampleRate = hdr->SampleRate;

        sort_eventtable(hdr);

	/* convert data into GDF rawdata from  */
	hdr->AS.rawdata = (uint8_t*)realloc(hdr->AS.rawdata, hdr->AS.bpb*hdr->NRec);
	for (k=0; k < hdr->NS; ++k) {
        CHANNEL_TYPE *hc = hdr->CHANNEL+k;

        size_t m,n,len=0;
        for (m=0; m < Data[k].size(); ++m) {
            size_t div = lround(Data[k][m].GetXScale()/Data.GetXScale());
            size_t div2 = hdr->SPR/div;

            // fprintf(stdout,"k,m,div,div2: %i,%i,%i,%i\n",(int)k,(int)m,(int)div,(int)div2);  //
            for (n=0; n < Data[k][m].size(); ++n) {
                uint64_t val;
                double d = Data[k][m][n];
#if !defined(__MINGW32__) && !defined(_MSC_VER)
                val = htole64(*(uint64_t*)&d);
#else
                val = *(uint64_t*)&d;
#endif
                size_t p, spr = (len + n*div) / hdr->SPR;
                for (p=0; p < div2; p++)
                   *(uint64_t*)(hdr->AS.rawdata + hc->bi + hdr->AS.bpb * spr + p*8) = val;
            }
            len += div*Data[k][m].size();
        }
    }

    /******************************
        write to file
    *******************************/
    std::string errorMsg("Exception while calling std::exportBiosigFile():\n");

    hdr = sopen( fName.c_str(), "w", hdr );
#if (BIOSIG_VERSION > 10400)
    if (serror2(hdr)) {
        errorMsg += hdr->AS.B4C_ERRMSG;
#else
    if (serror()) {
	    errorMsg += B4C_ERRMSG;
#endif
        destructHDR(hdr);
        throw std::runtime_error(errorMsg.c_str());
        return false;
    }

    ifwrite(hdr->AS.rawdata, hdr->AS.bpb, hdr->NRec, hdr);

    sclose(hdr);
    destructHDR(hdr);

    return true;
}

