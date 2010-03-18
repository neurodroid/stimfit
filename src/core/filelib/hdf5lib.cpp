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

#include "wx/wxprec.h"
#include "wx/progdlg.h"
#include "wx/filename.h"
#include <wx/msgdlg.h>
#include "hdf5.h"
#if H5_VERS_MINOR > 6
  #include "hdf5_hl.h"
#else
  #include "H5TA.h"
#endif
#include <boost/shared_ptr.hpp>
#include <cmath>

#include "./hdf5lib.h"

typedef struct rt {
    int channels;
    char date[128];
    char time[128];
} rt;

typedef struct ct {
    int n_sections;
} ct;

typedef struct st {
    double dt;
    char xunits[16];
    char yunits[16];
} st;

bool stf::exportHDF5File(const wxString& fName, const Recording& WData) {
    wxProgressDialog progDlg( wxT("HDF5 export"), wxT("Starting file export"),
                              100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );

    /* Create a new file using default properties. */
    hid_t file_id = H5Fcreate(fName.utf8_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    const int NRECORDS = 1;
    const int NFIELDS = 3;

    /* Calculate the size and the offsets of our struct members in memory */
    size_t rt_offset[NFIELDS] = {  HOFFSET( rt, channels ),
                                   HOFFSET( rt, date ),
                                   HOFFSET( rt, time )};

    /* Define an array of root tables */
    rt p_data;
    p_data.channels = WData.size();
    strcpy( p_data.date, (WData.GetDate()).utf8_str());
    strcpy( p_data.time, (WData.GetTime()).utf8_str());

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
    wxFileName file(fName);
    wxString desc; desc << wxT("Description of ") << file.GetFullName();
    herr_t status = H5TBmake_table( desc.utf8_str(), file_id, "description", (hsize_t)NFIELDS, (hsize_t)NRECORDS, sizeof(rt),
                                    field_names, rt_offset, field_type, 10, NULL, 0, &p_data  );
    if (status < 0) {
        wxString errorMsg(wxT("Exception while writing description in stf::exportHDF5File"));
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }

    hid_t comment_group = H5Gcreate(file_id, "/comment", 0 );

    /* File comment. */
    hsize_t dims[1] = { 1 };
    hid_t string_type3 = H5Tcopy( H5T_C_S1 );
    std::size_t file_desc_length = WData.GetFileDescription().length();
    if (file_desc_length <= 0) file_desc_length = 1;
    H5Tset_size( string_type3, file_desc_length);

    std::vector<char> data(WData.GetFileDescription().length());
    std::copy(WData.GetFileDescription().begin(), WData.GetFileDescription().end(), data.begin());
    status = H5LTmake_dataset(file_id, "/comment/comment", 1, dims, string_type3, &data[0]);
    if (status < 0) {
        wxString errorMsg(wxT("Exception while writing comment in stf::exportHDF5File"));
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }

    std::vector<wxString> channel_name(WData.size());
    hid_t channel_group = H5Gcreate(file_id, "/channels", 0 );
    for ( std::size_t n_c=0; n_c < WData.size(); ++n_c) {
        /* Channel descriptions. */
        channel_name[n_c] = WData[n_c].GetChannelName();
        if ( channel_name[n_c] == wxT("") ) {
            channel_name[n_c] << wxT("ch") << (n_c);
        }
        hsize_t dimsc[1] = { 1 };
        hid_t string_typec = H5Tcopy( H5T_C_S1 );
        std::size_t cn_length = channel_name[n_c].length();
        if (cn_length <= 0) cn_length = 1;
        H5Tset_size( string_typec, cn_length );

        std::vector<char> datac(channel_name[n_c].length());
        std::copy(channel_name[n_c].begin(),channel_name[n_c].end(), datac.begin());
        wxString desc_path; desc_path << wxT("/channels/ch") << (n_c);
        status = H5LTmake_dataset(file_id, desc_path.utf8_str(), 1, dimsc, string_typec, &datac[0]);
        if (status < 0) {
            wxString errorMsg(wxT("Exception while writing channel name in stf::exportHDF5File"));
            throw std::runtime_error(std::string(errorMsg.char_str()));
        }

        wxString channel_path; channel_path << wxT("/") << channel_name[n_c];
        hid_t channel_group = H5Gcreate( file_id, channel_path.utf8_str(), 0 );

        /* Calculate the size and the offsets of our struct members in memory */
        size_t ct_size =  sizeof( ct );
        size_t ct_offset[1] = { HOFFSET( rt, channels ) };
        /* Define an array of channel tables */
        ct c_data = { WData[n_c].size() };

        /* Define field information */
        const char *cfield_names[1]  =  { "n_sections" };
        hid_t      cfield_type[1] = {H5T_NATIVE_INT};
        wxString c_desc;
        c_desc << wxT("Description of channel ") << n_c;
        status = H5TBmake_table( c_desc.utf8_str(), channel_group, "description", (hsize_t)1, (hsize_t)1, ct_size,
                                 cfield_names, ct_offset, cfield_type, 10, NULL, 0, &c_data  );
        if (status < 0) {
            wxString errorMsg(wxT("Exception while writing channel description in stf::exportHDF5File"));
            throw std::runtime_error(std::string(errorMsg.char_str()));
        }

        int max_log10 = 0;
        if (WData[n_c].size() > 1) {
            max_log10 = int(log10((double)WData[n_c].size()-1.0));
        }

        for (std::size_t n_s=0; n_s < WData[n_c].size(); ++n_s) {
            wxString progStr;
            progStr << wxT("Writing channel #") << n_c + 1 << wxT(" of ") << WData.size()
                    << wxT(", Section #") << n_s << wxT(" of ") << WData[n_c].size();
            progDlg.Update(
                           // Channel contribution:
                           (int)(((double)n_c/(double)WData.size())*100.0+
                                 // Section contribution:
                                 (double)(n_s)/(double)WData[n_c].size()*(100.0/WData.size())),
                           progStr
                           );

            // construct a number with leading zeros:
            int n10 = 0;
            if (n_s > 0) {
                n10 = int(log10((double)n_s));
            }
            wxString strZero = wxT("");
            for (int n_z=n10; n_z < max_log10; ++n_z) {
                strZero << wxT("0");
            }

            // construct a section name:
            wxString section_name = WData[n_c][n_s].GetSectionDescription();
            if ( section_name == wxT("") ) {
                section_name << wxT("sec") << n_s;
            }

            // create a child group in the channel:
            wxString section_path; section_path << channel_path << wxT("/") << wxT("section_") << strZero << n_s;
            hid_t section_group = H5Gcreate( file_id, section_path.utf8_str(), 0 );

            // add data and description, store as 32 bit little endian independent of machine:
            hsize_t dims[1] = { WData[n_c][n_s].size() };
            wxString data_path; data_path << section_path << wxT("/data");
            Vector_float data_cp(WData[n_c][n_s].get().size()); /* 32 bit */
            for (std::size_t n_cp = 0; n_cp < WData[n_c][n_s].get().size(); ++n_cp) {
                data_cp[n_cp] = float(WData[n_c][n_s][n_cp]);
            }
            status = H5LTmake_dataset(file_id, data_path.utf8_str(), 1, dims, H5T_IEEE_F32LE, &data_cp[0]);
            if (status < 0) {
                wxString errorMsg(wxT("Exception while writing data in stf::exportHDF5File"));
                throw std::runtime_error(std::string(errorMsg.char_str()));
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
            strcpy( s_data.xunits, WData.GetXUnits().utf8_str() );
            strcpy( s_data.yunits, WData[n_c].GetYUnits().utf8_str() );

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

            wxString sdesc; sdesc << wxT("Description of ") << section_name;
            status = H5TBmake_table( sdesc.utf8_str(), section_group, "description", (hsize_t)NSFIELDS, (hsize_t)NSRECORDS, st_size,
                                     sfield_names, st_offset, sfield_type, 10, NULL, 0, &s_data  );
            if (status < 0) {
                wxString errorMsg(wxT("Exception while writing section description in stf::exportHDF5File"));
                throw std::runtime_error(std::string(errorMsg.char_str()));
            }
        }
    }

    /* Terminate access to the file. */
    status = H5Fclose(file_id);
    if (status < 0) {
        wxString errorMsg(wxT("Exception while closing file in stf::exportHDF5File"));
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }
    return (status >= 0);

}

void stf::importHDF5File(const wxString& fName, Recording& ReturnData, bool progress) {
    wxProgressDialog progDlg( wxT("HDF5 import"), wxT("Starting file import"),
                              100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL );

    /* Create a new file using default properties. */
    hid_t file_id = H5Fopen(fName.utf8_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    /* H5TBread_table */
    const int NRECORDS = 1;
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
        wxString errorMsg(wxT("Exception while reading description in stf::importHDF5File"));
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }
    int numberChannels =rt_buf[0].channels;
    ReturnData.SetDate( wxString(rt_buf[0].date,wxConvLocal) );
    ReturnData.SetTime( wxString(rt_buf[0].time,wxConvLocal) );

    /* Create the data space for the dataset. */
    hsize_t dims;
    H5T_class_t class_id;
    size_t type_size;

    status = H5LTget_dataset_info( file_id, "/comment/comment", &dims, &class_id, &type_size );
    if (status < 0) {
        wxString errorMsg(wxT("Exception while reading comment in stf::importHDF5File"));
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }
    hid_t string_type3 = H5Tcopy( H5T_C_S1 );
    H5Tset_size( string_type3,  type_size );
    std::vector<char> comment( type_size );
    status = H5LTread_dataset (file_id, "/comment/comment", string_type3, &comment[0]);
    if (status < 0) {
        wxString errorMsg(wxT("Exception while reading comment in stf::importHDF5File"));
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }
    wxString wxComment;
	for (std::size_t c=0; c<type_size; ++c) {
        wxComment << wxChar(comment[c]);
    }
    ReturnData.SetFileDescription( wxComment );
    double dt = 1.0;
    wxString yunits = wxT("");
    for (int n_c=0;n_c<numberChannels;++n_c) {
        /* Calculate the size and the offsets of our struct members in memory */
        size_t ct_offset[NFIELDS] = { HOFFSET( ct, n_sections ) };
        ct ct_buf[1];
        size_t ct_sizes[NFIELDS] = { sizeof( ct_buf[0].n_sections) };

        /* Read channel name */
        hsize_t cdims;
        H5T_class_t cclass_id;
        size_t ctype_size;
        wxString desc_path; desc_path << wxT("/channels/ch") << (n_c);
        status = H5LTget_dataset_info( file_id, desc_path.utf8_str(), &cdims, &cclass_id, &ctype_size );
        if (status < 0) {
            wxString errorMsg(wxT("Exception while reading channel in stf::importHDF5File"));
            throw std::runtime_error(std::string(errorMsg.char_str()));
        }
        hid_t string_typec= H5Tcopy( H5T_C_S1 );
        H5Tset_size( string_typec,  ctype_size );
        boost::shared_ptr<char> szchannel_name;
        szchannel_name.reset( new char[ctype_size] );
        status = H5LTread_dataset(file_id, desc_path.utf8_str(), string_typec, szchannel_name.get() );
        if (status < 0) {
            wxString errorMsg(wxT("Exception while reading channel name in stf::importHDF5File"));
            throw std::runtime_error(std::string(errorMsg.char_str()));
        }
        wxString channel_name;
		for (std::size_t c=0; c<ctype_size; ++c) {
            channel_name << wxChar(szchannel_name.get()[c]);
        }
        wxString channel_path; channel_path << wxT("/") << channel_name;
        hid_t channel_group = H5Gopen(file_id, channel_path.utf8_str() );
        status=H5TBread_table( channel_group, "description", sizeof(ct), ct_offset, ct_sizes, ct_buf );
        if (status < 0) {
            wxString errorMsg(wxT("Exception while reading channel description in stf::importHDF5File"));
            throw std::runtime_error(std::string(errorMsg.char_str()));
        }
        Channel TempChannel(ct_buf[0].n_sections);
        TempChannel.SetChannelName( channel_name );
        int max_log10 = 0;
        if (ct_buf[0].n_sections > 1) {
            max_log10 = int(log10((double)ct_buf[0].n_sections-1.0));
        }

        for (int n_s=0; n_s < ct_buf[0].n_sections; ++n_s) {
            if (progress) {
                wxString progStr;
                progStr << wxT("Reading channel #") << n_c + 1 << wxT(" of ") << numberChannels
                        << wxT(", Section #") << n_s+1 << wxT(" of ") << ct_buf[0].n_sections;
                progDlg.Update(
                               // Channel contribution:
                               (int)(((double)n_c/(double)numberChannels)*100.0+
                                     // Section contribution:
                                     (double)(n_s)/(double)ct_buf[0].n_sections*(100.0/numberChannels)),
                               progStr
                               );
            }
            // construct a number with leading zeros:
            int n10 = 0;
            if (n_s > 0) {
                n10 = int(log10((double)n_s));
            }
            wxString strZero = wxT("");
            for (int n_z=n10; n_z < max_log10; ++n_z) {
                strZero << wxT("0");
            }

            // construct a section name:
            wxString section_name; section_name << wxT("sec") << n_s;

            // create a child group in the channel:
            wxString section_path; section_path << channel_path << wxT("/") << wxT("section_") << strZero << n_s;
            hid_t section_group = H5Gopen(file_id, section_path.utf8_str() );

            wxString data_path; data_path << section_path << wxT("/data");
            hsize_t sdims;
            H5T_class_t sclass_id;
            size_t stype_size;
            status = H5LTget_dataset_info( file_id, data_path.utf8_str(), &sdims, &sclass_id, &stype_size );
            if (status < 0) {
                wxString errorMsg(wxT("Exception while reading data information in stf::importHDF5File"));
                throw std::runtime_error(std::string(errorMsg.char_str()));
            }
            Vector_float TempSection(sdims);
            status = H5LTread_dataset(file_id, data_path.utf8_str(), H5T_IEEE_F32LE, &TempSection[0]);
            if (status < 0) {
                wxString errorMsg(wxT("Exception while reading data in stf::importHDF5File"));
                throw std::runtime_error(std::string(errorMsg.char_str()));
            }

            Section TempSectionT(TempSection.size(), section_name);
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


            /* H5TBread_table */
            const int NSRECORDS = 1;
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
                wxString errorMsg(wxT("Exception while reading data description in stf::importHDF5File"));
                throw std::runtime_error(std::string(errorMsg.char_str()));
            }
            dt = st_buf[0].dt;
            yunits = wxString(st_buf[0].yunits, wxConvLocal);
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
        wxString errorMsg(wxT("Exception while closing file in stf::importHDF5File"));
        throw std::runtime_error(std::string(errorMsg.char_str()));
    }

}
