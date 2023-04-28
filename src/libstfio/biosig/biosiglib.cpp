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

// Copyright 2012,2013,2017 Alois Schloegl, IST Austria

#include <sstream>

#include "../stfio.h"

    #if defined(WITH_BIOSIGLITE)
        #include "../../libbiosiglite/biosig4c++/biosig2.h"
    #else
        #include <biosig.h>
    #endif
    #if (BIOSIG_VERSION < 10902)
        #error libbiosig v1.9.2 or later is required
    #endif
    #define DONOTUSE_DYNAMIC_ALLOCATION_FOR_CHANSPR


  /* these are internal biosig functions, defined in biosig-dev.h which is not always available */
extern "C" size_t ifwrite(void* buf, size_t size, size_t nmemb, HDRTYPE* hdr);
extern "C" uint32_t lcm(uint32_t A, uint32_t B);
#if !defined(__MINGW32__) && !defined(_MSC_VER)
    #if defined (__APPLE__)
        #include <machine/endian.h>
    #else
        #include <endian.h>
    #endif
#endif


#include "./biosiglib.h"

stfio::filetype stfio_file_type(HDRTYPE* hdr) {
        switch (biosig_get_filetype(hdr)) {

        case ABF2:	return stfio::abf;
        case ABF:	return stfio::abf;
        case ATF:	return stfio::atf;
        case CFS:	return stfio::cfs;
        case HEKA:	return stfio::heka;
        case HDF:	return stfio::hdf5;
        case AXG:	return stfio::axg;
        case IBW:	return stfio::igor;
        case SMR:	return stfio::son;
        default:	return stfio::none;
        }
}

#if defined(WITH_BIOSIG)
bool stfio::check_biosig_version(int a, int b, int c) {
	return (BIOSIG_VERSION >= 10000*a + 100*b + c);
}
#endif

stfio::filetype stfio::importBiosigFile(const std::string &fName, Recording &ReturnData, ProgressInfo& progDlg) {

    std::string errorMsg("Exception while calling std::importBSFile():\n");
    std::string yunits;
    stfio::filetype type;

    // =====================================================================================================================
    //
    // importBiosig opens file with libbiosig
    //	- performs an automated identification of the file format
    //  - and decides whether the data is imported through importBiosig (currently CFS, HEKA, ABF1, GDF, and others)
    //  - or handed back to other import*File functions (currently ABF2, AXG, HDF5)
    //
    // =====================================================================================================================


#ifdef __LIBBIOSIG2_H__

    HDRTYPE* hdr =  sopen( fName.c_str(), "r", NULL );
    if (hdr==NULL) {
        ReturnData.resize(0);
        return stfio::none;
    }

    type = stfio_file_type(hdr);
    if (biosig_check_error(hdr)) {
        ReturnData.resize(0);
        destructHDR(hdr);
        return type;
    }
    enum FileFormat biosig_filetype=biosig_get_filetype(hdr);
    if ( (biosig_filetype==ATF  && get_biosig_version() < 0x030001) \
      || (biosig_filetype==ABF2 && get_biosig_version() < 0xf30001) \
      ||  biosig_filetype==HDF ) {
        // ATF, ABF2 HDF5 support should be handled by importATF, and importABF, and importHDF5 not importBiosig
        // with libbiosig v3.0.1 (release 2.3.1) and later, ATF and ABF2 should be handled by Biosig
        ReturnData.resize(0);
        destructHDR(hdr);
        return type;
    }

    // ensure the event table is in chronological order	
    sort_eventtable(hdr);

    // allocate local memory for intermediate results;
    const int strSize=100;
    char str[strSize];

    /*
	count sections and generate list of indices indicating start and end of sweeps
     */	

    int numberOfChannels = biosig_get_number_of_channels(hdr);
    double fs = biosig_get_eventtable_samplerate(hdr);
    size_t numberOfEvents = biosig_get_number_of_events(hdr);
    size_t nsections = biosig_get_number_of_segments(hdr);
    ReturnData.InitSectionMarkerList(nsections);
    std::vector<size_t> SegIndexList(nsections+1);
    SegIndexList[0] = 0;
    SegIndexList[nsections] = biosig_get_number_of_samples(hdr);
    std::string annotationTableDesc = std::string();
    for (size_t k=0, n=0; k < numberOfEvents; k++) {
        uint32_t pos;
        uint16_t typ;
        uint32_t dur;
        uint16_t chn;
        const char *desc;
        /*
        gdftype  timestamp;
        */

        biosig_get_nth_event(hdr, k, &typ, &pos, &chn, &dur, NULL, &desc);
        if (typ == 0x7ffe) {
            SegIndexList[++n] = pos;
        }
        else if (typ < 256) {
            sprintf(str,"%f s:\t%s\n", pos/fs, desc);
            annotationTableDesc += std::string( str );

            size_t currentSectionNumber = (pos < SegIndexList[n]) ? n : (n+1);
            ReturnData.SetSectionType(currentSectionNumber-1, typ);
            // TODO: Description of EvenTypes
            // ReturnData.SetEventDescription( currentSectionNumber-1, desc);
        }
    }

    /*************************************************************************
        rescale data to mV and pA
     *************************************************************************/
    for (int ch=0; ch < numberOfChannels; ++ch) {
        CHANNEL_TYPE *hc = biosig_get_channel(hdr, ch);
        switch (biosig_channel_get_physdimcode(hc) & 0xffe0) {
        case 4256:  // Volt
		//biosig_channel_scale_to_unit(hc, "mV");
		biosig_channel_change_scale_to_physdimcode(hc, 4274);
		break;
        case 4160:  // Ampere
		//biosig_channel_scale_to_unit(hc, "pA");
		biosig_channel_change_scale_to_physdimcode(hc, 4181);
		break;
	    }
    }

    /*************************************************************************
        read bulk data
     *************************************************************************/
    biosig_data_type *data = biosig_get_data(hdr, 0);
    size_t SPR = biosig_get_number_of_samples(hdr);

#ifdef _STFDEBUG
    std::cout << "Number of events: " << numberOfEvents << std::endl;
    /*int res = */ hdr2ascii(hdr, stdout, 4);
#endif

    for (int NS=0; NS < numberOfChannels; ) {
        CHANNEL_TYPE *hc = biosig_get_channel(hdr, NS);
        Channel TempChannel(nsections);
        TempChannel.SetChannelName(biosig_channel_get_label(hc));
        TempChannel.SetYUnits(biosig_channel_get_physdim(hc));

        for (size_t ns=1; ns<=nsections; ns++) {
            if (SegIndexList[ns]-SegIndexList[ns-1] < 0) {
                ReturnData.resize(0);
                destructHDR(hdr);
                return type;
            }
            size_t SPS = SegIndexList[ns]-SegIndexList[ns-1];	// length of segment, samples per segment

            int progbar = int(100.0 * (1.0 * ns / nsections + NS) / numberOfChannels);
            std::ostringstream progStr;
            progStr << "Reading channel #" << NS + 1 << " of " << numberOfChannels
                << ", Section #" << ns << " of " << nsections;
            progDlg.Update(progbar, progStr.str());

            Section TempSection(SPS, "");

            std::copy(&(data[NS*SPR + SegIndexList[ns-1]]),
			    &(data[NS*SPR + SegIndexList[ns]]),
			    TempSection.get_w().begin() );

            try {
                TempChannel.InsertSection(TempSection, ns-1);
            }
            catch (...) {
                ReturnData.resize(0);
                destructHDR(hdr);
                return type;
            }
        }   // end for sections
        try {
            if ((int)ReturnData.size() < numberOfChannels)
                ReturnData.resize(numberOfChannels);
            ReturnData.InsertChannel(TempChannel, NS++);
        }
        catch (...) {
            ReturnData.resize(0);
            destructHDR(hdr);
            return type;
        }
    }   // end for channels

    ReturnData.SetComment ( biosig_get_recording_id(hdr) );

    sprintf(str,"v%i.%i.%i (compiled on %s %s)",BIOSIG_VERSION_MAJOR,BIOSIG_VERSION_MINOR,BIOSIG_PATCHLEVEL,__DATE__,__TIME__);
    std::string Desc = std::string("importBiosig with libbiosig ")+std::string(str) + " ";
    const char* tmpstr;
    if ((tmpstr=biosig_get_technician(hdr)))
            Desc += std::string ("\nTechnician:\t") + std::string (tmpstr) + " ";
    Desc += std::string( "\nCreated with: ");
    if ((tmpstr=biosig_get_manufacturer_name(hdr)))
        Desc += std::string( tmpstr ) + " ";
    if ((tmpstr=biosig_get_manufacturer_model(hdr)))
        Desc += std::string( tmpstr ) + " ";
    if ((tmpstr=biosig_get_manufacturer_version(hdr)))
        Desc += std::string( tmpstr ) + " ";
    if ((tmpstr=biosig_get_manufacturer_serial_number(hdr)))
        Desc += std::string( tmpstr ) + " ";

    Desc += std::string ("\nUser specified Annotations:\n")+annotationTableDesc;

    ReturnData.SetFileDescription(Desc);

    tmpstr = biosig_get_application_specific_information(hdr);
    if (tmpstr != NULL) /* MSVC2008 can not properly handle std::string( (char*)NULL ) */
        ReturnData.SetGlobalSectionDescription(tmpstr);

    ReturnData.SetXScale(1000.0/biosig_get_samplerate(hdr));
    ReturnData.SetXUnits("ms");
    ReturnData.SetScaling("biosig scaling factor");

    /*************************************************************************
        Date and time conversion
     *************************************************************************/
    struct tm T;
    biosig_get_startdatetime(hdr, &T);
    ReturnData.SetDateTime(T);

    destructHDR(hdr);


#endif	//ifdef __LIBBIOSIG2_H__
    return stfio::biosig;
}


    // =====================================================================================================================
    //
    // Save file with libbiosig into GDF format
    //
    // =====================================================================================================================

bool stfio::exportBiosigFile(const std::string& fName, const Recording& Data, stfio::ProgressInfo& progDlg) {
/*
    converts the internal data structure to libbiosig's internal structure
    and saves the file as gdf file.

    The data in converted into the raw data format, and not into the common
    data matrix.
*/

#ifdef __LIBBIOSIG2_H__

    size_t numberOfChannels = Data.size();
    HDRTYPE* hdr = constructHDR(numberOfChannels, 0);

	/* Initialize all header parameters */
    biosig_set_filetype(hdr, GDF);

    biosig_set_startdatetime(hdr, Data.GetDateTime());

    const char *xunits = Data.GetXUnits().c_str();
    uint16_t pdc = PhysDimCode(xunits);

    if ((pdc & 0xffe0) != PhysDimCode("s")) {
        fprintf(stderr,"Stimfit exportBiosigFile: xunits [%s] has not proper units, assume [ms]\n",Data.GetXUnits().c_str());
        pdc = PhysDimCode("ms");
    }

    double fs = 1.0/(PhysDimScale(pdc) * Data.GetXScale());
    biosig_set_samplerate(hdr, fs);

    biosig_reset_flag(hdr, BIOSIG_FLAG_COMPRESSION | BIOSIG_FLAG_UCAL | BIOSIG_FLAG_OVERFLOWDETECTION | BIOSIG_FLAG_ROW_BASED_CHANNELS );

    size_t k, m, numberOfEvents=0;
    size_t NRec=0;	// corresponds to hdr->NRec
    size_t SPR=1;	// corresponds to hdr->SPR
    size_t chSPR=0;	// corresponds to hc->SPR

    /* Initialize all channel parameters */
#ifndef DONOTUSE_DYNAMIC_ALLOCATION_FOR_CHANSPR
	size_t *chanSPR = (size_t*)malloc(numberOfChannels*sizeof(size_t));
#endif
    for (k = 0; k < numberOfChannels; ++k) {
        CHANNEL_TYPE *hc = biosig_get_channel(hdr, k);

		biosig_channel_set_datatype_to_double(hc);
		biosig_channel_set_scaling(hc, 1e9, -1e9, 1e9, -1e9);
		biosig_channel_set_label(hc, Data[k].GetChannelName().c_str());
		biosig_channel_set_physdim(hc, Data[k].GetYUnits().c_str());

		biosig_channel_set_filter(hc, NAN, NAN, NAN);
		biosig_channel_set_timing_offset(hc, 0.0);
		biosig_channel_set_impedance(hc, NAN);

        chSPR    = SPR;

        // each segment gets one marker, roughly
        numberOfEvents += Data[k].size();

        size_t m,len = 0;
        for (len=0, m = 0; m < Data[k].size(); ++m) {
            unsigned div = lround(Data[k][m].GetXScale()/Data.GetXScale());
            chSPR = lcm(chSPR,div);  // sampling interval of m-th segment in k-th channel
            len += div*Data[k][m].size();
        }
        SPR = lcm(SPR, chSPR);

        /*
            hc->SPR (i.e. chSPR) is 'abused' to store final hdr->SPR/hc->SPR, this is corrected in the loop below
            its a hack to avoid the need for another malloc().
        */
#ifdef DONOTUSE_DYNAMIC_ALLOCATION_FOR_CHANSPR
        biosig_channel_set_samples_per_record(hc, chSPR);
#else
        chanSPR[k]=chSPR;
#endif

        if (k==0) {
            NRec = len;
        }
        else if ((size_t)NRec != len) {
            destructHDR(hdr);
            throw std::runtime_error("File can't be exported:\n"
                "No data or traces have different sizes" );

            return false;
        }
    }

    biosig_set_number_of_samples(hdr, NRec, SPR);
    size_t bpb = 0;
    for (k = 0; k < numberOfChannels; ++k) {
        CHANNEL_TYPE *hc = biosig_get_channel(hdr, k);
        // the 'abuse' of hc->SPR described above is corrected
#ifdef DONOTUSE_DYNAMIC_ALLOCATION_FOR_CHANSPR
        size_t spr = biosig_channel_get_samples_per_record(hc);
        spr = SPR / spr;
        biosig_channel_set_samples_per_record(hc, spr);
#else
        size_t spr = SPR/chanSPR[k];
        chanSPR[k] = spr;
#endif
        bpb += spr * 8; /* its always double */
    }

	/***
	    build Event table for storing segment information
	    pre-allocate memory for even table
         ***/

        numberOfEvents *= 2;    // about two events per segment
        biosig_set_number_of_events(hdr, numberOfEvents);

    /* check whether all segments have same size */
    {
        char flag = (numberOfChannels > 0);
        size_t m, POS, pos;
        for (k=0; k < numberOfChannels; ++k) {
            pos = Data[k].size();
            if (k==0)
                POS = pos;
            else
                flag &= (POS == pos);
        }
        for (m=0; flag && (m < Data[(size_t)0].size()); ++m) {
            for (k=0; k < biosig_get_number_of_channels(hdr); ++k) {
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

        size_t N=0;
        k=0;
        size_t pos = 0;
        for (m=0; m < (Data[k].size()); ++m) {
            if (pos > 0) {
                uint16_t typ=0x7ffe;
                uint32_t pos32=pos;
                uint16_t chn=0;
                uint32_t dur=0;
                // set break marker
                biosig_set_nth_event(hdr, N++, &typ, &pos32, &chn, &dur, NULL, NULL);
                /*
                // set annotation
                const char *Desc = Data[k][m].GetSectionDescription().c_str();
                 if (Desc != NULL && strlen(Desc)>0)
                    biosig_set_nth_event(hdr, N++, NULL, &pos32, &chn, &dur, NULL, Desc);   // TODO
                */
            }
            pos += Data[k][m].size() * lround(Data[k][m].GetXScale()/Data.GetXScale());
        }

        biosig_set_number_of_events(hdr, N);
        biosig_set_eventtable_samplerate(hdr, fs);
        sort_eventtable(hdr);

        /* convert data into GDF rawdata from  */
        uint8_t *rawdata = (uint8_t*)malloc(bpb * NRec);

        size_t bi=0;
	for (k=0; k < numberOfChannels; ++k) {
        CHANNEL_TYPE *hc = biosig_get_channel(hdr, k);
#ifdef DONOTUSE_DYNAMIC_ALLOCATION_FOR_CHANSPR
        size_t chSPR = biosig_channel_get_samples_per_record(hc);
#else
        size_t chSPR = chanSPR[k];
#endif
        size_t m,n,len=0;
        for (m=0; m < Data[k].size(); ++m) {

            size_t div = lround(Data[k][m].GetXScale()/Data.GetXScale());
            size_t div2 = SPR/div;		  // TODO: avoid using hdr->SPR

            // fprintf(stdout,"k,m,div,div2: %i,%i,%i,%i\n",(int)k,(int)m,(int)div,(int)div2);  //
            for (n=0; n < Data[k][m].size(); ++n) {
                uint64_t val;
                double d = Data[k][m][n];
#if !defined(__MINGW32__) && !defined(_MSC_VER) && !defined(__APPLE__)
                val = htole64(*(uint64_t*)&d);
#else
                val = *(uint64_t*)&d;
#endif
                size_t p, spr = (len + n*div) / SPR;

                for (p=0; p < div2; p++)
                   *(uint64_t*)(rawdata + bi + bpb * spr + p*8) = val;
            }
            len += div*Data[k][m].size();
        }
		bi += chSPR*8;
    }

#ifndef DONOTUSE_DYNAMIC_ALLOCATION_FOR_CHANSPR
	if (chanSPR) free(chanSPR);
#endif

    /******************************
        write to file
    *******************************/
    std::string errorMsg("Exception while calling std::exportBiosigFile():\n");

    hdr = sopen( fName.c_str(), "w", hdr );
    if (serror2(hdr)) {
        errorMsg += biosig_get_errormsg(hdr);
        destructHDR(hdr);
        throw std::runtime_error(errorMsg.c_str());
        return false;
    }

    ifwrite(rawdata, bpb, NRec, hdr);

    sclose(hdr);
    destructHDR(hdr);
    free(rawdata);

#endif
    return true;
}

