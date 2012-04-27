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
    hdr->FLAG.ROW_BASED_CHANNELS = 1;
    /* size_t blks = */ sread(NULL, 0, hdr->NRec, hdr);

#ifdef _STFDEBUG
    std::cout << "Number of channels: " << hdr->NS << std::endl;
    std::cout << "Number of records per channel: " << hdr->NRec << std::endl;
    std::cout << "Number of samples per record: " << hdr->SPR << std::endl;
    std::cout << "Data size: " << hdr->data.size[0] << "x" << hdr->data.size[1] << std::endl;
    std::cout << "Sampling rate: " << hdr->SampleRate << std::endl;
    std::cout << "Number of events: " << hdr->EVENT.N << std::endl;
    /*int res = */ hdr2ascii(hdr, stdout, 4);
   #define VERBOSE_LEVEL 8
#endif


if (VERBOSE_LEVEL>7) fprintf(stdout,"stimfit:biosiglib:ImportBSfile; 110\n");  

    // make sure that the event table is in chronological order	
    sort_eventtable(hdr);

if (VERBOSE_LEVEL>7) fprintf(stdout,"stimfit:biosiglib:ImportBSfile; 120\n");  

    /* 
       memory for only Events of TYP==0x7ffe is needed, superfluous memory is allocated 
       in order to avoid loop + malloc + a 2nd loop
    */ 	
    size_t *SegIndexList = (size_t*)malloc((hdr->EVENT.N+1)*sizeof(size_t)); 
    uint32_t nsections = 0; 
    SegIndexList[nsections] = 0; 
    size_t MaxSectionLength = 0; 
    for (size_t k=0; k<=hdr->EVENT.N; k++) {

if (VERBOSE_LEVEL>7) fprintf(stdout,"stimfit:biosiglib:ImportBSfile; 130 %i %i\n",(int)k, (int)nsections);  

	if (0) ; 
        else if (k>=hdr->EVENT.N)		SegIndexList[++nsections] = hdr->NRec*hdr->SPR; 
	else if (hdr->EVENT.TYP[k]==0x7ffe)	SegIndexList[++nsections] = hdr->EVENT.POS[k]; 
	else					continue; 

        size_t SPS = SegIndexList[nsections]-SegIndexList[nsections-1];	// length of segment, samples per segment
	if (MaxSectionLength < SPS) MaxSectionLength = SPS;
    }

/***********

TODO: 
Data of Segment ns and channel nc is available from this memory range:

	hdr->data.block[nc*hdr->SPR*hdr->NRec + SegIndexList[ns-1]]
	hdr->data.block[nc*hdr->SPR*hdr->NRec + SegIndexList[ns]-1], 


The length of the data (i.e. number of samples) can be obtain in the following way:
        size_t SPS = SegIndexList[ns]-SegIndexList[ns-1];

Thus, the data can be accessed in this way 
	
	for (nc=0; nc<hdr->NS; nc++) {
		for (ns=0; ns<nsections; nc++) {
			for (k=0; k<SPS; k++) {
				// value of k-th sample, of channel nc in segment ns 
				value = hdr->data.block[nc*hdr->SPR*hdr->NRec + SegIndexList[ns-1] + k];
			}
		}
	}

Don't forget 
    free(SegIndexList); 	

************/

    
#ifdef NON_WORKING_ATTEMPT

    for (size_t nc=0; nc<hdr->NS; ++nc) {

if (VERBOSE_LEVEL>7) fprintf(stdout,"stimfit:biosiglib:ImportBSfile; 150 %i [%i %i]\n",(int)nc, (int)nc, (int)nsections);  

	        Channel TempChannel(nsections);
	        TempChannel.SetChannelName(""); // TODO: hdr->channelname[nc];
        	TempChannel.SetYUnits(""); // TODO: hdr->yunits[nc];

	        //Channel TempChannel(nsections, MaxSectionLength);
	        //TempChannel.SetChannelName(hdr->CHANNEL[nc].Label); // TODO: hdr->channelname[nc];
	        //TempChannel.SetYUnits(hdr->CHANNEL[nc].PhysDim); // TODO: hdr->yunits[nc];


        for (size_t ns=1; ns<=nsections; ns++) {

	        size_t SPS = SegIndexList[ns]-SegIndexList[ns-1];	// length of segment, samples per segment

if (VERBOSE_LEVEL>7) fprintf(stdout,"stimfit:biosiglib:ImportBSfile; 140 %i %i %i %i\n",(int)nc, (int)SPS, (int)SegIndexList[ns], (int)SegIndexList[ns-1]);  


		int progbar = 100.0*(1.0*ns/(nsections+1) + nc)/(hdr->NS); 
		std::ostringstream progStr;
		progStr << "Reading event #" << nc + 1 << " of " << hdr->NS ;
		progDlg.Update(progbar, progStr.str());

//		memcpy(&(TempChannel.at(ns-1).get_w()),hdr->data.block+nc*hdr->SPR*hdr->NRec + SegIndexList[ns-1],SPS*sizeof(double)); 


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

if (VERBOSE_LEVEL>7) fprintf(stdout,"stimfit:biosiglib:ImportBSfile; 160 Insert section %i in channel %i failed\n",int(ns-1),(int)nc);  

			ReturnData.resize(0);
			sclose(hdr);
			destructHDR(hdr);
			throw;
		}

	}        

        try {
            ReturnData.InsertChannel(TempChannel, nc);
        }
        catch (...) {

if (VERBOSE_LEVEL>7) fprintf(stdout,"stimfit:biosiglib:ImportBSfile; 170 Insert channel %i failed\n",(int)nc);  

            ReturnData.resize(0);
            sclose(hdr);
            destructHDR(hdr);
            throw;
        }
    }

#endif 


    free(SegIndexList); 	

if (VERBOSE_LEVEL>7) fprintf(stdout,"stimfit:biosiglib:ImportBSfile; 180 %i\n",(int)nsections);  



/*
    // This seems to be wrong?
    if (hdr->NRec > hdr->SPR) {
        int tmp = hdr->NRec;
        hdr->NRec = hdr->SPR;
        hdr->SPR = tmp;
    }

    for (typeof(hdr->NS) nc=0; nc<nchannels; ++nc) {

        Channel TempChannel(nsections);
        TempChannel.SetChannelName(""); // TODO: hdr->channelname[nc];
        TempChannel.SetYUnits(""); // TODO: hdr->yunits[nc];
        
        for (uint32_t ns=0; ns<nsections; ++ns) {
            int progbar =
                // Channel contribution:
                (int)(((double)nc/(double)nchannels)*100.0+
                      // Section contribution:
                      (double)ns/(double)nsections*(100.0/nchannels));
            std::ostringstream progStr;
            progStr << "Reading channel #" << nc + 1 << " of " << nchannels
                    << ", Section #" << ns + 1 << " of " << nsections;
            progDlg.Update(progbar, progStr.str());
            Section TempSection(
                                hdr->SPR, // TODO: hdr->nsamplingpoints[nc][ns]
                                "" // TODO: hdr->sectionname[nc][ns]
            );
            std::copy(&(hdr->data.block[nc*hdr->SPR]), &(hdr->data.block[(nc+1)*hdr->SPR]), TempSection.get_w().begin());
            try {
                TempChannel.InsertSection(TempSection, ns);
            }
            catch (...) {
                ReturnData.resize(0);
                sclose(hdr);
                destructHDR(hdr);
                throw;
            }
        }
        try {
            if ((int)ReturnData.size() < nchannels) {
                ReturnData.resize(nchannels);
            }
            ReturnData.InsertChannel(TempChannel, nc);
        }
        catch (...) {
            ReturnData.resize(0);
            sclose(hdr);
            destructHDR(hdr);
            throw;
        }
    }
*/
    ReturnData.SetXScale(hdr->SampleRate);
    ReturnData.SetComment(""); // TODO: hdr->comment
    ReturnData.SetDate(""); // TODO: hdr->datestring
    ReturnData.SetTime(""); // TODO: hdr->timestring

#ifdef MODULE_ONLY
    if (progress) {
        std::cout << "\r";
        std::cout << "100%" << std::endl;
    }
#endif

    sclose(hdr);
    destructHDR(hdr);
}
