/*
---------------------------------------------------------------------------
Copyright (C) 2014  Alois Schloegl
Copyright (C) 2003  Eugenio Cervesato & Giorgio De Odorico.
Developed at the Associazione per la Ricerca in Cardiologia - Pordenone - Italy.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
---------------------------------------------------------------------------
*/
// structures.h          header file 

#ifndef __STRUCTURES_H__
#define __STRUCTURES_H__
#include <time.h>

/*
// obsolete definition
#define int_S	int8_t
#define int_M	int16_t
#define int_L	int32_t
#define U_int_S	uint8_t
#define U_int_M	uint16_t
#define U_int_L	uint32_t
#define dec_S	float
#define dec_M	double
#define dec_L	long double
 */
#define str	char

#define bool char
#define true 1
#define false 0
#define TRUE 1
#define FALSE 0

struct alfabetic
{
	uint16_t number;
	const char *sentence;
};

struct numeric
{
	uint16_t value;
	uint8_t unit;
};

struct section_header
{
	uint16_t CRC;
	uint16_t ID;
	uint32_t length;
	uint8_t version;
	uint8_t protocol_version;
	char *word;
};

struct file_header
{
	uint16_t CRC;
	uint32_t length;
};

struct pointer_section
{
	uint32_t index;
	 int16_t ID;
	uint32_t length;
};

struct device_info
{
	uint16_t institution_number;
	uint16_t department_number;
	uint16_t ID;
	uint8_t type;
	uint8_t manifacturer;
	char 	*model_description;
	uint8_t protocol_revision_number;
	uint8_t category;
	uint8_t language;
	uint8_t capability[4];
	uint8_t AC;
	char	*analysing_program_revision_number;
	char	*serial_number_device;
	char	*device_system_software;
	char	*device_SCP_implementation_software;
	char	*manifacturer_trade_name;
};

struct info_drug
{
	uint8_t table;
	uint8_t classes;
	uint8_t drug_code;
	uint16_t length;
};	

struct Time_Zone
{
	int16_t offset;
	uint16_t index;
	const char *description;
};

struct demographic
{
	char 	 *first_name;
	char 	 *last_name;
	char 	 *ID;
	char 	 *second_last_name;
	struct numeric  age;
        time_t   date_birth2;    // by E.C. feb 2006
	struct numeric  height;
	struct numeric  weight;
	uint8_t  sex;
	uint8_t  race;
	uint16_t  systolic_pressure;
	uint16_t  diastolic_pressure;
};

struct clinic
{
	uint16_t	number_drug;
	struct info_drug	*drug;
	char		*text_drug;

	uint16_t	number_diagnose;
	struct numeric	*diagnose;
	char		*text_diagnose;

	char		*referring_physician;
	char 		*latest_confirming_physician;
	char 		*technician_description;

	uint16_t	number_text;
	struct numeric	*free_text;
	char		*text_free_text;

	uint16_t	number_hystory;
	struct numeric	*medical_hystory;

	uint16_t	number_free_hystory;
	struct numeric	*free_medical_hystory;
	char		*text_free_medical_hystory;
};

struct descriptive
{
	struct device_info 	acquiring;
	struct device_info 	analyzing;
	char 		*acquiring_institution;
	char 		*analyzing_institution;
	char 		*acquiring_department;
	char 		*analyzing_department;
	char 		*room;
	uint8_t 	stat_code;
};

struct device
{
        time_t    date_acquisition2;       // by E.C. feb 2006
        time_t    time_acquisition2;       // by E.C. feb 2006
	uint16_t   baseline_filter;
	uint16_t   lowpass_filter;
	uint8_t   other_filter[4];
	char 	  *sequence_number;
	struct numeric   electrode_configuration;
	struct Time_Zone TZ;
};

struct table_H
{
	uint8_t bit_prefix;
	uint8_t bit_code;
	uint8_t TMS;
	int16_t	 base_value;
	uint32_t base_code;
};

struct f_lead
{
	uint8_t number;
	bool 	 subtraction;
	bool 	 all_simultaneously;
	uint8_t number_simultaneously;
};

struct lead
{
	uint8_t ID;
	uint32_t start;
	uint32_t end;
};

struct Subtraction_Zone
{
	uint16_t beat_type;
	uint32_t SB;
	uint32_t fc;
	uint32_t SE;
};

struct Protected_Area
{
	uint32_t QB;
	uint32_t QE;
};

struct f_BdR0
{
	uint16_t length;
	uint16_t fcM;
	uint16_t AVM;
	uint16_t STM;
	uint16_t number_samples;
	uint8_t encoding;
};

struct f_Res
{
	uint16_t AVM;
	uint16_t STM;
	uint16_t number;
	uint16_t number_samples;
	uint8_t encoding;
	bool bimodal;
	uint8_t decimation_factor;
};

struct spike
{
	uint16_t time;
	int16_t amplitude;
	uint8_t type;
	uint8_t source;
	uint8_t index;
	uint16_t pulse_width;
};

struct global_measurement
{
	uint8_t number;
	uint16_t number_QRS;
	uint8_t number_spike;
	uint16_t average_RR;
	uint16_t average_PP;
	uint16_t ventricular_rate;
	uint16_t atrial_rate;
	uint16_t QT_corrected;
	uint8_t formula_type;
	uint16_t number_tag;
};

struct additional_measurement
{
	uint8_t ID;
	uint8_t byte[5];
};

struct BdR_measurement
{
	uint16_t P_onset;
	uint16_t P_offset;
	uint16_t QRS_onset;
	uint16_t QRS_offset;
	uint16_t T_offset;
	int16_t P_axis;
	int16_t QRS_axis;
	int16_t T_axis;
};

struct info
{
	uint8_t type;
	char *date;
	char *time;
	uint8_t number;
};

struct header_lead_measurement
{
	uint16_t number_lead;
	uint16_t number_lead_measurement;
};

struct lead_measurement_block
{
	uint16_t ID;
	int16_t P_duration;
	int16_t PR_interval;
	int16_t QRS_duration;
	int16_t QT_interval;
	int16_t Q_duration;
	int16_t R_duration;
	int16_t S_duration;
	int16_t R1_duration;
	int16_t S1_duration;
	int16_t Q_amplitude;
	int16_t R_amplitude;
	int16_t S_amplitude;
	int16_t R1_amplitude;
	int16_t S1_amplitude;
	int16_t J_point_amplitude;
	int16_t Pp_amplitude;
	int16_t Pm_amplitude;
	int16_t Tp_amplitude;
	int16_t Tm_amplitude;
	int16_t ST_slope;
	int16_t P_morphology;
	int16_t T_morphology;
	int16_t iso_electric_segment_onset_QRS;
	int16_t iso_electric_segment_offset_QRS;
	int16_t intrinsicoid_deflection;
	uint16_t quality_recording[8];
	int16_t ST_amplitude_Jplus20;
	int16_t ST_amplitude_Jplus60;
	int16_t ST_amplitude_Jplus80;
	int16_t ST_amplitude_JplusRR16;
	int16_t ST_amplitude_JplusRR8;
};

struct statement_coded
{
	uint8_t sequence_number;
	uint16_t length;
	uint8_t type;
	uint16_t number_field;
};

//_____________________________________
//structs for sections: 2, 3, 4, 5, 6
//_____________________________________
struct DATA_DECODE
{
	struct table_H *t_Huffman;
	uint16_t *flag_Huffman;

	struct lead *data_lead;
	struct f_lead flag_lead;

	struct Protected_Area *data_protected;
	struct Subtraction_Zone *data_subtraction;

	struct f_BdR0 flag_BdR0;
	uint16_t *length_BdR0;
	uint8_t *samples_BdR0;
	int32_t *Median;

	struct f_Res flag_Res;
	uint16_t *length_Res;
	uint8_t *samples_Res;
	int32_t *Residual;

	int32_t *Reconstructed;
};

struct TREE_NODE
//struttura di un nodo dell'albero
{
	struct TREE_NODE *next_0;
	struct TREE_NODE *next_1;
	int16_t row;
};

//_____________________________________
//structs for sections: 7, 10
//_____________________________________
struct DATA_RECORD
{
	struct global_measurement data_global;
	struct spike *data_spike;
	uint8_t *type_BdR;
	struct BdR_measurement *data_BdR;
	struct additional_measurement *data_additional;

	struct header_lead_measurement header_lead;
	struct lead_measurement_block *lead_block;
};

//_____________________________________
//structs for sections: 1, 8, 11
//_____________________________________
struct DATA_INFO
{
	struct demographic ana;
	struct clinic cli;
	struct descriptive des;
	struct device dev;

	struct info flag_report;
	struct numeric *text_dim;
	char *text_report;

	struct info flag_statement;
	struct statement_coded *data_statement;
	char *text_statement;
};

#endif /*__STRUCTURES_H__*/
//_____________________________________
