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

#include "hdf5.h"
#if H5_VERS_MINOR > 6
  #include "hdf5_hl.h"
#else
  #include "H5TA.h"
#endif
#include <boost/shared_ptr.hpp>
#include <cmath>
#include <sstream>
#include <iostream>

#include "./hdf5lib.h"
#include "../recording.h"

const static unsigned int DATELEN = 128;
const static unsigned int TIMELEN = 128;
const static unsigned int UNITLEN = 16;

typedef struct rt {
    int channels;
    char date[DATELEN];
    char time[TIMELEN];
} rt;

typedef struct ct {
    int n_sections;
} ct;

typedef struct st {
    double dt;
    char xunits[UNITLEN];
    char yunits[UNITLEN];
} st;

bool stfio::exportHDF5File(const std::string& fName, const Recording& WData, ProgressInfo& progDlg) {
    
    hid_t file_id = H5Fcreate(fName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    const int NRECORDS = 1;
    const int NFIELDS = 3;

    /* Calculate the size and the offsets of our struct members in memory */
    size_t rt_offset[NFIELDS] = {  HOFFSET( rt, channels ),
                                   HOFFSET( rt, date ),
                                   HOFFSET( rt, time )};

    /* Define an array of root tables */
    rt p_data;
    p_data.channels = WData.size();
    if (WData.GetDate().length() < DATELEN)
        strcpy( p_data.date, (WData.GetDate()).c_str());
    if (WData.GetTime().length() < TIMELEN)
        strcpy( p_data.time, (WData.GetTime()).c_str());

    /* Define field information */
    const char *field_names[NFIELDS]  =  { "channels", "date", "time" };
    hid_t      field_type[NFIELDS];

    /* Initialize the field field_type */
    hid_t string_type1 = H5Tcopy( H5T_C_S1 );
    hid_t string_type2 = H5Tcopy( H5T_C_S1 );
    std::size_t date_length = WData.GetDate().length();
    std::size_t time_length = WData.GetTime().length();
    if (date_length <= 0) date_length = 1; 
    if (time_length <= 0) time_length = 1; 
    H5Tset_size( string_type1,  date_length);
    H5Tset_size( string_type2,  time_length);
    field_type[0] = H5T_NATIVE_INT;
    field_type[1] = string_type1;
    field_type[2] = string_type2;
    
    std::ostringstream desc;
    desc << "Description of " << fName;
    
    herr_t status = H5TBmake_table( desc.str().c_str(), file_id, "description", (hsize_t)NFIELDS, (hsize_t)NRECORDS, sizeof(rt),
                                    field_names, rt_offset, field_type, 10, NULL, 0, &p_data  );

    if (status < 0) {
        std::string errorMsg("Exception while writing description in stfio::exportHDF5File");
        throw std::runtime_error(errorMsg);
    }

    H5Gcreate(file_id, "/comment", 0 );

    /* File comment. */
    std::string description(WData.GetFileDescription());
    if (description.length() <= 0) {
        description = "No description";
    }

    status = H5LTmake_dataset_string(file_id, "/comment/description", description.c_str());
    if (status < 0) {
        std::string errorMsg("Exception while writing description in stfio::exportHDF5File");
        throw std::runtime_error(errorMsg);
    }

    std::string comment(WData.GetComment());
    if (comment.length() <= 0) {
        comment = "No comment";
    }

    status = H5LTmake_dataset_string(file_id, "/comment/comment", comment.c_str());
    if (status < 0) {
        std::string errorMsg("Exception while writing comment in stfio::exportHDF5File");
        throw std::runtime_error(errorMsg);
    }

    std::vector<std::string> channel_name(WData.size());
    H5Gcreate(file_id, "/channels", 0 );
    for ( std::size_t n_c=0; n_c < WData.size(); ++n_c) {
        /* Channel descriptions. */
        std::ostringstream ossname;
        ossname << WData[n_c].GetChannelName();
        if ( ossname.str() == "" ) {
            ossname << "ch" << (n_c);
        }
        channel_name[n_c] = ossname.str();
        hsize_t dimsc[1] = { 1 };
        hid_t string_typec = H5Tcopy( H5T_C_S1 );
        std::size_t cn_length = channel_name[n_c].length();
        if (cn_length <= 0) cn_length = 1;
        H5Tset_size( string_typec, cn_length );

        std::vector<char> datac(channel_name[n_c].length());
        std::copy(channel_name[n_c].begin(),channel_name[n_c].end(), datac.begin());
        std::ostringstream desc_path; desc_path << "/channels/ch" << (n_c);
        status = H5LTmake_dataset(file_id, desc_path.str().c_str(), 1, dimsc, string_typec, &datac[0]);
        if (status < 0) {
            std::string errorMsg("Exception while writing channel name in stfio::exportHDF5File");
            throw std::runtime_error(errorMsg);
        }

        std::ostringstream channel_path; channel_path << "/" << channel_name[n_c];
        hid_t channel_group = H5Gcreate( file_id, channel_path.str().c_str(), 0 );

        /* Calculate the size and the offsets of our struct members in memory */
        size_t ct_size =  sizeof( ct );
        size_t ct_offset[1] = { HOFFSET( rt, channels ) };
        /* Define an array of channel tables */
        ct c_data = { WData[n_c].size() };

        /* Define field information */
        const char *cfield_names[1]  =  { "n_sections" };
        hid_t      cfield_type[1] = {H5T_NATIVE_INT};
        std::ostringstream c_desc;
        c_desc << "Description of channel " << n_c;
        status = H5TBmake_table( c_desc.str().c_str(), channel_group, "description", (hsize_t)1, (hsize_t)1, ct_size,
                                 cfield_names, ct_offset, cfield_type, 10, NULL, 0, &c_data  );
        if (status < 0) {
            std::string errorMsg("Exception while writing channel description in stfio::exportHDF5File");
            throw std::runtime_error(errorMsg);
        }

        int max_log10 = 0;
        if (WData[n_c].size() > 1) {
            max_log10 = int(log10((double)WData[n_c].size()-1.0));
        }

        for (std::size_t n_s=0; n_s < WData[n_c].size(); ++n_s) {
            int progbar = 
                // Channel contribution:
                (int)(((double)n_c/(double)WData.size())*100.0+
                      // Section contribution:
                      (double)(n_s)/(double)WData[n_c].size()*(100.0/WData.size()));
            std::ostringstream progStr;
            progStr << "Writing channel #" << n_c + 1 << " of " << WData.size()
                    << ", Section #" << n_s << " of " << WData[n_c].size();
            progDlg.Update(progbar, progStr.str());
            
            // construct a number with leading zeros:
            int n10 = 0;
            if (n_s > 0) {
                n10 = int(log10((double)n_s));
            }
            std::ostringstream strZero; strZero << "";
            for (int n_z=n10; n_z < max_log10; ++n_z) {
                strZero << "0";
            }

            // construct a section name:
            std::ostringstream section_name; section_name << WData[n_c][n_s].GetSectionDescription();
            if ( section_name.str() == "" ) {
                section_name << "sec" << n_s;
            }

            // create a child group in the channel:
            std::ostringstream section_path;
            section_path << channel_path.str() << "/" << "section_" << strZero.str() << n_s;
            hid_t section_group = H5Gcreate( file_id, section_path.str().c_str(), 0 );

            // add data and description, store as 32 bit little endian independent of machine:
            hsize_t dims[1] = { WData[n_c][n_s].size() };
            std::ostringstream data_path;
            data_path << section_path.str() << "/data";
            Vector_float data_cp(WData[n_c][n_s].get().size()); /* 32 bit */
            for (std::size_t n_cp = 0; n_cp < WData[n_c][n_s].get().size(); ++n_cp) {
                data_cp[n_cp] = float(WData[n_c][n_s][n_cp]);
            }
            status = H5LTmake_dataset(file_id, data_path.str().c_str(), 1, dims, H5T_IEEE_F32LE, &data_cp[0]);
            if (status < 0) {
                std::string errorMsg("Exception while writing data in stfio::exportHDF5File");
                throw std::runtime_error(errorMsg);
            }

            const int NSRECORDS = 1;
            const int NSFIELDS = 3;

            /* Calculate the size and the offsets of our struct members in memory */
            size_t st_size =  sizeof( st );
            size_t st_offset[NSFIELDS] = {  HOFFSET( st, dt ),
                                            HOFFSET( st, xunits ),
                                            HOFFSET( st, yunits )};

            /* Define an array of root tables */
            st s_data;
            s_data.dt = WData.GetXScale();
            if (WData.GetXUnits().length() < UNITLEN)
                strcpy( s_data.xunits, WData.GetXUnits().c_str() );
            if (WData[n_c].GetYUnits().length() < UNITLEN)
                strcpy( s_data.yunits, WData[n_c].GetYUnits().c_str() );

            /* Define field information */
            const char *sfield_names[NSFIELDS]  =  { "dt", "xunits", "yunits" };
            hid_t      sfield_type[NSFIELDS];

            /* Initialize the field field_type */
            hid_t string_type4 = H5Tcopy( H5T_C_S1 );
            hid_t string_type5 = H5Tcopy( H5T_C_S1 );
            H5Tset_size( string_type4,  2);
            std::size_t yu_length = WData[n_c].GetYUnits().length();
            if (yu_length <= 0) yu_length = 1;

            H5Tset_size( string_type5, yu_length );
            sfield_type[0] = H5T_NATIVE_DOUBLE;
            sfield_type[1] = string_type4;
            sfield_type[2] = string_type5;

            std::ostringstream sdesc;
            sdesc << "Description of " << section_name.str();
            status = H5TBmake_table( sdesc.str().c_str(), section_group, "description", (hsize_t)NSFIELDS, (hsize_t)NSRECORDS, st_size,
                                     sfield_names, st_offset, sfield_type, 10, NULL, 0, &s_data  );
            if (status < 0) {
                std::string errorMsg("Exception while writing section description in stfio::exportHDF5File");
                throw std::runtime_error(errorMsg);
            }
        }
    }

    /* Terminate access to the file. */
    status = H5Fclose(file_id);
    if (status < 0) {
        std::string errorMsg("Exception while closing file in stfio::exportHDF5File");
        throw std::runtime_error(errorMsg);
    }
#ifdef MODULE_ONLY
    std::cout << "\r";
    std::cout << "100%" << std::endl;
#endif
    
    return (status >= 0);
}

void stfio::importHDF5File(const std::string& fName, Recording& ReturnData, ProgressInfo& progDlg) {
    /* Create a new file using default properties. */
    hid_t file_id = H5Fopen(fName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    
    /* H5TBread_table
       const int NRECORDS = 1;*/
    const int NFIELDS    = 3;

    /* Calculate the size and the offsets of our struct members in memory */
    size_t rt_offset[NFIELDS] = {  HOFFSET( rt, channels ),
                                   HOFFSET( rt, date ),
                                   HOFFSET( rt, time )};
    rt rt_buf[1];
    size_t rt_sizes[NFIELDS] = { sizeof( rt_buf[0].channels),
                                 sizeof( rt_buf[0].date),
                                 sizeof( rt_buf[0].time)};
    herr_t status=H5TBread_table( file_id, "description", sizeof(rt), rt_offset, rt_sizes, rt_buf );
    if (status < 0) {
        std::string errorMsg("Exception while reading description in stfio::importHDF5File");
        throw std::runtime_error(errorMsg);
    }
    int numberChannels =rt_buf[0].channels;
    ReturnData.SetDate( rt_buf[0].date );
    ReturnData.SetTime( rt_buf[0].time );

    /* Create the data space for the dataset. */
    hsize_t dims;
    H5T_class_t class_id;
    size_t type_size;

    std::string description, comment;
    hid_t group_id = H5Gopen(file_id, "/comment");
    status = H5Lexists(group_id, "/comment/description", 0);
    if (status==1) {
        status = H5LTget_dataset_info( file_id, "/comment/description", &dims, &class_id, &type_size );
        if (status >= 0) {
            description.resize( type_size );
            status = H5LTread_dataset_string (file_id, "/comment/description", &description[0]);
            if (status < 0) {
                std::string errorMsg("Exception while reading description in stfio::importHDF5File");
                throw std::runtime_error(errorMsg);
            }
        }
    }
    ReturnData.SetFileDescription(description);
    
    status = H5Lexists(group_id, "/comment/comment", 0);
    if (status==1) {
        status = H5LTget_dataset_info( file_id, "/comment/comment", &dims, &class_id, &type_size );
        if (status >= 0) {
            comment.resize( type_size );
            status = H5LTread_dataset_string (file_id, "/comment/comment", &comment[0]);
            if (status < 0) {
                std::string errorMsg("Exception while reading comment in stfio::importHDF5File");
                throw std::runtime_error(errorMsg);
            }
        }
    }
    ReturnData.SetComment(comment);

    double dt = 1.0;
    std::string yunits = "";
    for (int n_c=0;n_c<numberChannels;++n_c) {
        /* Calculate the size and the offsets of our struct members in memory */
        size_t ct_offset[NFIELDS] = { HOFFSET( ct, n_sections ) };
        ct ct_buf[1];
        size_t ct_sizes[NFIELDS] = { sizeof( ct_buf[0].n_sections) };

        /* Read channel name */
        hsize_t cdims;
        H5T_class_t cclass_id;
        size_t ctype_size;
        std::ostringstream desc_path;
        desc_path << "/channels/ch" << (n_c);
        status = H5LTget_dataset_info( file_id, desc_path.str().c_str(), &cdims, &cclass_id, &ctype_size );
        if (status < 0) {
            std::string errorMsg("Exception while reading channel in stfio::importHDF5File");
            throw std::runtime_error(errorMsg);
        }
        hid_t string_typec= H5Tcopy( H5T_C_S1 );
        H5Tset_size( string_typec,  ctype_size );
        std::vector<char> szchannel_name(ctype_size);
        // szchannel_name.reset( new char[ctype_size] );
        status = H5LTread_dataset(file_id, desc_path.str().c_str(), string_typec, &szchannel_name[0] );
        if (status < 0) {
            std::string errorMsg("Exception while reading channel name in stfio::importHDF5File");
            throw std::runtime_error(errorMsg);
        }
        std::ostringstream channel_name;
        for (std::size_t c=0; c<ctype_size; ++c) {
            channel_name << szchannel_name[c];
        }

        std::ostringstream channel_path;
        channel_path << "/" << channel_name.str();

        hid_t channel_group = H5Gopen(file_id, channel_path.str().c_str() );
        status=H5TBread_table( channel_group, "description", sizeof(ct), ct_offset, ct_sizes, ct_buf );
        if (status < 0) {
            std::string errorMsg("Exception while reading channel description in stfio::importHDF5File");
            throw std::runtime_error(errorMsg);
        }
        Channel TempChannel(ct_buf[0].n_sections);
        TempChannel.SetChannelName( channel_name.str() );
        int max_log10 = 0;
        if (ct_buf[0].n_sections > 1) {
            max_log10 = int(log10((double)ct_buf[0].n_sections-1.0));
        }

        for (int n_s=0; n_s < ct_buf[0].n_sections; ++n_s) {
            int progbar =
                // Channel contribution:
                (int)(((double)n_c/(double)numberChannels)*100.0+
                      // Section contribution:
                      (double)(n_s)/(double)ct_buf[0].n_sections*(100.0/numberChannels));
            std::ostringstream progStr;
            progStr << "Reading channel #" << n_c + 1 << " of " << numberChannels
                    << ", Section #" << n_s+1 << " of " << ct_buf[0].n_sections;
            progDlg.Update(progbar, progStr.str());
            
            // construct a number with leading zeros:
            int n10 = 0;
            if (n_s > 0) {
                n10 = int(log10((double)n_s));
            }
            std::ostringstream strZero; strZero << "";
            for (int n_z=n10; n_z < max_log10; ++n_z) {
                strZero << "0";
            }

            // construct a section name:
            std::ostringstream section_name;
            section_name << "sec" << n_s;

            // create a child group in the channel:
            std::ostringstream section_path;
            section_path << channel_path.str() << "/" << "section_" << strZero.str() << n_s;
            hid_t section_group = H5Gopen(file_id, section_path.str().c_str() );

            std::ostringstream data_path; data_path << section_path.str() << "/data";
            hsize_t sdims;
            H5T_class_t sclass_id;
            size_t stype_size;
            status = H5LTget_dataset_info( file_id, data_path.str().c_str(), &sdims, &sclass_id, &stype_size );
            if (status < 0) {
                std::string errorMsg("Exception while reading data information in stfio::importHDF5File");
                throw std::runtime_error(errorMsg);
            }
            Vector_float TempSection(sdims);
            status = H5LTread_dataset(file_id, data_path.str().c_str(), H5T_IEEE_F32LE, &TempSection[0]);
            if (status < 0) {
                std::string errorMsg("Exception while reading data in stfio::importHDF5File");
                throw std::runtime_error(errorMsg);
            }

            Section TempSectionT(TempSection.size(), section_name.str());
            for (std::size_t cp = 0; cp < TempSectionT.size(); ++cp) {
                TempSectionT[cp] = double(TempSection[cp]);
            }
            // std::copy(TempSection.begin(),TempSection.end(),&TempSectionT[0]);
            try {
                TempChannel.InsertSection(TempSectionT,n_s);
            }
            catch (...) {
                throw;
            }


            /* H5TBread_table
               const int NSRECORDS = 1; */
            const int NSFIELDS    = 3;

            /* Calculate the size and the offsets of our struct members in memory */
            size_t st_offset[NSFIELDS] = {  HOFFSET( st, dt ),
                                            HOFFSET( st, xunits ),
                                            HOFFSET( st, yunits )};
            st st_buf[1];
            size_t st_sizes[NSFIELDS] = { sizeof( st_buf[0].dt),
                                          sizeof( st_buf[0].xunits),
                                          sizeof( st_buf[0].yunits)};
            status=H5TBread_table( section_group, "description", sizeof(st), st_offset, st_sizes, st_buf );
            if (status < 0) {
                std::string errorMsg("Exception while reading data description in stfio::importHDF5File");
                throw std::runtime_error(errorMsg);
            }
            dt = st_buf[0].dt;
            yunits = st_buf[0].yunits;
            H5Gclose( section_group );
        }
        try {
            if ((int)ReturnData.size()<numberChannels) {
                ReturnData.resize(numberChannels);
            }
            ReturnData.InsertChannel(TempChannel,n_c);
            ReturnData[n_c].SetYUnits( yunits );
        }
        catch (...) {
            ReturnData.resize(0);
            throw;
        }
        H5Gclose( channel_group );
    }
    ReturnData.SetXScale(dt);
    /* Terminate access to the file. */
    status = H5Fclose(file_id);
    if (status < 0) {
        std::string errorMsg("Exception while closing file in stfio::importHDF5File");
        throw std::runtime_error(errorMsg);
    }
#ifdef MODULE_ONLY
    if (progress) {
        std::cout << "\r";
        std::cout << "100%" << std::endl;
    }
#endif
    
}
