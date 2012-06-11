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

// Copyright 2012 Alois Schloegl, IST Austria <alois.schloegl@ist.ac.at>

#include <string>
#include <iomanip>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <sstream>
#include <biosig.h>

#include "../stfio.h"
#include "./biosiglib.h"

class BiosigHDR {
  public:
    BiosigHDR(unsigned int NS, unsigned int N_EVENT) {
        pHDR = constructHDR(NS, N_EVENT);
    }
    ~BiosigHDR() {
        destructHDR(pHDR);
    }

  private:
    HDRTYPE* pHDR;
};

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
	sclose(hdr);
        throw std::runtime_error(errorMsg.c_str());
    }
    hdr->FLAG.ROW_BASED_CHANNELS = 0;
    /* size_t blks = */ sread(NULL, 0, hdr->NRec, hdr);

#ifdef _STFDEBUG
    std::cout << "Number of channels: " << hdr->NS << std::endl;
    std::cout << "Number of records per channel: " << hdr->NRec << std::endl;
    std::cout << "Number of samples per record: " << hdr->SPR << std::endl;
    std::cout << "Data size: " << hdr->data.size[0] << "x" << hdr->data.size[1] << std::endl;
    std::cout << "Sampling rate: " << hdr->SampleRate << std::endl;
    std::cout << "Number of events: " << hdr->EVENT.N << std::endl;
    /*int res = */ hdr2ascii(hdr, stdout, 4);
#endif

    // ensure the event table is in chronological order	
    sort_eventtable(hdr);

    /*
	count sections and generate list of indeces indicating start and end of sweeps
     */	
    size_t LenIndexList = 256; 
    if (LenIndexList > hdr->EVENT.N) LenIndexList = hdr->EVENT.N + 2;
    size_t *SegIndexList = (size_t*)malloc(LenIndexList*sizeof(size_t)); 
    uint32_t nsections = 0; 
    SegIndexList[nsections] = 0; 
    size_t MaxSectionLength = 0; 
    for (size_t k=0; k <= hdr->EVENT.N; k++) {
	if (LenIndexList <= nsections+2) {
		LenIndexList *=2; 
		SegIndexList = (size_t*)realloc(SegIndexList, LenIndexList*sizeof(size_t)); 
	}
	/* 
           count number of sections and stores it in nsections; 
  	   EVENT.TYP==0x7ffe indicate number of breaks between sweeps
	   SegIndexList includes index to first sample and index to last sample,
	   thus, the effective length of SegIndexList is the number of 0x7ffe plus two.
	*/
	if (0) ; 
        else if (k>=hdr->EVENT.N)		SegIndexList[++nsections] = hdr->NRec*hdr->SPR; 
	else if (hdr->EVENT.TYP[k]==0x7ffe)	SegIndexList[++nsections] = hdr->EVENT.POS[k]; 
	else					continue; 

        size_t SPS = SegIndexList[nsections]-SegIndexList[nsections-1];	// length of segment, samples per segment
	if (MaxSectionLength < SPS) MaxSectionLength = SPS;
    }

    // allocate local memory for intermediate results;    
    const int strSize=100;     
    char str[strSize];

    for (size_t nc=0; nc<hdr->NS; ++nc) {
	Channel TempChannel(nsections);
	TempChannel.SetChannelName(hdr->CHANNEL[nc].Label);
#if defined(BIOSIG_VERSION) && (BIOSIG_VERSION > 10301)
        TempChannel.SetYUnits(PhysDim(hdr->CHANNEL[nc].PhysDimCode));
#else
        PhysDim(hdr->CHANNEL[nc].PhysDimCode,str);
        TempChannel.SetYUnits(str);
#endif

        for (size_t ns=1; ns<=nsections; ns++) {
	        size_t SPS = SegIndexList[ns]-SegIndexList[ns-1];	// length of segment, samples per segment

		int progbar = 100.0*(1.0*ns/(nsections+1) + nc)/(hdr->NS); 
		std::ostringstream progStr;
		progStr << "Reading channel #" << nc + 1 << " of " << hdr->NS
			<< ", Section #" << ns + 1 << " of " << nsections;
		progDlg.Update(progbar, progStr.str());

		char sweepname[20];
		sprintf(sweepname,"sweep %i",(int)ns);		
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
		if (ReturnData.size() < hdr->NS) {
			ReturnData.resize(hdr->NS);
		}
		ReturnData.InsertChannel(TempChannel, nc);
        }
        catch (...) {
		ReturnData.resize(0);
		destructHDR(hdr);
		throw;
        }
    }

    free(SegIndexList); 	

    ReturnData.SetXScale(1000.0/hdr->SampleRate);
    ReturnData.SetComment(hdr->FileName);

    struct tm T; 
    gdf_time2tm_time_r(hdr->T0, &T); 
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
