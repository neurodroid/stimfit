/*
---------------------------------------------------------------------------
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
// codes.h             consts included in the protocol
#ifndef __CODES_H__
#define __CODES_H__

static const char STR_END[]={(char)-1,'\0'};
static char STR_NULL[]=" unspecified/unknown ";

static alfabetic _special[]={
	{ 29999	, "measurements not computed" },
	{ 29998	, "measurements not found due to rejection of the lead" },
	{ 19999	, "measurements not found because wave not present" },
	{ 999	, "undefined" }
};

static alfabetic _age[]={
	{ 0	, " unspecified/unknown " } ,
	{ 1	, " years " } ,
	{ 2	, " months " } ,
	{ 3	, " weeks " } ,
	{ 4	, " days " } ,
	{ 5	, " hours " } 
};

static alfabetic _height[]={
	{ 0	, " unspecified/unknown " } ,
	{ 1	, " cm " } ,
	{ 2	, " inch " } ,
	{ 3	, " mm " } 
};

static alfabetic _weight[]={
	{ 0	, " unspecified/unknown " } ,
	{ 1	, " Kg " } ,
	{ 2	, " g " } ,
	{ 3	, " lb " } ,
	{ 4	, " oz " } 
};

static alfabetic _sex[]={
	{ 0	, " ? " } ,
	{ 1	, " M " } ,
	{ 2	, " F " } ,
	{ 3	, " unspecified/unknown " } 
};

static alfabetic _race[]={
	{ 0	, " unspecified/unknown " } ,
	{ 1	, " caucasian " } ,
	{ 2	, " black " } ,
	{ 3	, " oriental " } 
};

static alfabetic class_drug[]={
	{ 0	, " unspecified/unknown " } ,
	{ 1	, " digitalis preparation " } ,
	{ 2	, " antiarrhythmic " } ,
	{ 3	, " diuretics " } ,
	{ 4	, " antihypertensive " } ,
	{ 5	, " antianginal " } ,
	{ 6	, " antithrombotic agents " } ,
	{ 7	, " beta blockers " } ,
	{ 8	, " psychotropic " } ,
	{ 9	, " calcium blockers " } ,
	{ 10	, " antihypotensive " } ,
	{ 11	, " anticholesterol " } ,
	{ 12	, " ACE-inhibitors " } ,
	{ 100	, " not taking drugs " } ,
	{ 101	, " drugs, but unknown type " } ,
	{ 102	, " other medication " }, 
	{ 256	, " unspecified/unknown " } ,
	{ 257	, " digoxin-lanoxin " } ,
	{ 258	, " digitoxin-digitalis " } ,
	{ 265	, " other " },
	{ 512	, " unspecified/unknown " } ,
	{ 513	, " dysopyramide " } ,
	{ 514	, " quinidine " } ,
	{ 515	, " procainamide " } ,
	{ 516	, " lidocaine " } ,
	{ 517	, " phenytoin " } ,
	{ 518	, " dilantin " } ,
	{ 519	, " amiodarone " } ,
	{ 520	, " tocainide " } ,
	{ 521	, " other " } ,
	{ 522	, " encainide " } ,
	{ 523	, " mexitil/mexilitine " } ,
	{ 524	, " flecainide " } ,
	{ 525	, " lorcainide " } ,
	{ 768	, " unspecified/unknown " } ,
	{ 769	, " thiazide " } ,
	{ 770	, " furosemide (lasix) " } ,
	{ 771	, " potassium cloride " } ,
	{ 777	, " other " } ,
	{ 1024	, " unspecified/unknown " } ,
	{ 1025	, " clonidine " } ,
	{ 1026	, " prasozin " } ,
	{ 1027	, " hydralazine " } ,
	{ 1033	, " other " },
	{ 1280	, " unspecified/unknown " } ,
	{ 1281	, " isosorbide " } ,
	{ 1282	, " calcium blockers " } ,
	{ 1283	, " diuretics " } ,
	{ 1284	, " nitrates " } ,
	{ 1289	, " other " },
	{ 1536	, " unspecified/unknown " } ,
	{ 1537	, " aspirin " } ,
	{ 1538	, " coumarin " } ,
	{ 1539	, " heparin " } ,
	{ 1540	, " warfarin " } ,
	{ 1541	, " streptokinase " } ,
	{ 1542	, " t-PA " } ,
	{ 1545	, " other " },
	{ 1792	, " unspecified/unknown " } ,
	{ 1793	, " propanolol " } ,
	{ 1794	, " corgard " } ,
	{ 1795	, " atenolol " } ,
	{ 1796	, " metoprolol " } ,
	{ 1797	, " pindolol " } ,
	{ 1798	, " acebutolol " } ,
	{ 1801	, " other " }, 
	{ 2048	, " unspecified/unknown " } ,
	{ 2049	, " tricyclic antidepressant " } ,
	{ 2050	, " phenothiazide " } ,
	{ 2051	, " barbiturate " } ,
	{ 2057	, " other " },
	{ 2304	, " unspecified/unknown " } ,
	{ 2305	, " nifedipine " } ,
	{ 2306	, " verapamil " } ,
	{ 2313	, " other " },
	{ 2560	, " unspecified/unknown " } ,
	{ 2561	, " asthmatic drug " } ,
	{ 2562	, " aminophyline " } ,
	{ 2563	, " isuprel " } ,
	{ 2569	, " other " },
	{ 2816	, " unspecified/unknown " } ,
	{ 2817	, " colestid " } ,
	{ 2818	, " lovastatin " } ,
	{ 2819	, " simvastatin " } ,
	{ 2820	, " fibrates " } ,
	{ 2825	, " other " },
	{ 3071	, " unspecified/unknown " } ,
	{ 3072	, " captopril " } ,
	{ 3081	, " other " } 
};

static alfabetic device_type[]={
	{ 0	, " Cart " },
	{ 1	, " host " },
	{ 2	, " unspecified/unknown "}
};

static alfabetic legacy_device[]={
	{ 0	, " unspecified/unknown " } ,
	{ 1	, " Burdick " } ,
	{ 2	, " Cambridge " } ,
	{ 3	, " Comprumed " } ,
	{ 4	, " Datamed " } ,
	{ 5	, " Fukuda " } ,
	{ 6	, " Hewlett-Packard " } ,
	{ 7	, " Marquette Electronics " } ,
	{ 8	, " Moratara Instruments " } ,
	{ 9	, " Nihon Kohden " } ,
	{ 10	, " Okin " } ,
	{ 11	, " Quinton " } ,
	{ 12	, " Siemens " } ,
	{ 13	, " Spacelabs " } ,
	{ 14	, " Telemed " } ,
	{ 15	, " Hellige " } ,
	{ 16	, " ESA-OTE " } ,
	{ 17	, " Schiller " } ,
	{ 18	, " Picker-Schwarzer " } ,
	{ 19	, " Elettronica-Trentina " } ,
	{ 20	, " Zwonitz " }
};

static alfabetic compatibility[]={
	{ 72	, " I " } ,
	{ 160	, " II " } ,
	{ 176	, " III " } ,
	{ 192	, " IV " },
	{ 255	, " unspecified/unknown " }
};

static alfabetic language_code[]={
	{ 0	, " 8 bit ASCII only " } ,
	{ 1	, " ISO-8859-1 latin-1 " } ,
	{ 192	, " ISO-8859-2 latin-2 (central and estern european) " } ,
	{ 208	, " ISO-8859-4 latin-4 (Baltic) " } ,
	{ 200	, " ISO-8859-5 (Cyrillic) " } ,
	{ 216	, " ISO-8859-6 (Arabic) " } ,
	{ 196	, " ISO-8859-7 (Greek) " } ,
	{ 212	, " ISO-8859-8 (Hebrew) " } ,
	{ 204	, " ISO-8859-11 (Thai) " } ,
	{ 220	, " ISO-8859-15 latin-9 (latin-0) " } ,
	{ 224	, " Unicode (ISO-60646) " } ,
	{ 240	, " JIS X0201-1976 (Japanese) " } ,
	{ 232	, " JIS X0208-1977 (Japanese) " } ,
	{ 248	, " JIS X0212-1990 (Japanese) " } ,
	{ 228	, " GB 2312-80 (Chinese) " } ,
	{ 244	, " KS C5601-1987 (Korean) " } ,
	{ 255	, " unspecified/unknown " } 
};

static alfabetic capability_device[]={
	{ 1	, " No printing " } ,
	{ 2	, " No analysis " } ,
	{ 3	, " No storage " } ,
	{ 4	, " No acquisition " } ,
	{ 5	, " can print ECG reports " } ,
	{ 6	, " can interpret ECG " } ,
	{ 7	, " can store ECG records " } ,
	{ 8	, " can acquire ECG data " } 
};

static alfabetic frequency_AC[]={
	{ 0	, " unspecified/unknown " } ,
	{ 1	, " 50 Hz " } ,
	{ 2	, " 60 Hz " } 
};

static alfabetic filter_bitmap[]={
	{ 0	, " unspecified/unknown " }, 
	{ 1	, " 60 Hz notch filter " } ,
	{ 2	, " 50 Hz notch filter " } ,
	{ 3	, " artifact filter " } ,
	{ 4	, " baseline filter " }
};

static alfabetic _hystory[]={
	{ 0	, " diagnoses or clinical problems " } ,
	{ 1	, " apparently healty " } ,
	{ 10	, " acute myocardial infarction " } ,
	{ 11	, " myocardial infarction " } ,
	{ 12	, " previous myocardial infarction " } ,
	{ 15	, " ischemic heart disease " } ,
	{ 18	, " peripheral vascular disease " } ,
	{ 20	, " cyanotic congenital heart disease " } ,
	{ 21	, " acyanotic congenital heart disease " } ,
	{ 22	, " valvular heart disease " } ,
	{ 25	, " hypertension " } ,
	{ 27	, " cerebrovascular accident " } ,
	{ 30	, " cardiomyopathy " } ,
	{ 35	, " pericardits " } ,
	{ 36	, " myocardits " } ,
	{ 40	, " post-operative cardiac surgery " } ,
	{ 42	, " implanted cardiac pacemaker " } ,
	{ 45	, " pulmonary embolism " } ,
	{ 50	, " respiratory disease " } ,
	{ 55	, " endocrine disease " } ,
	{ 60	, " neurological disease " } ,
	{ 65	, " alimentary disease " } ,
	{ 70	, " renal disease " } ,
	{ 80	, " pre-operative general surgery " } ,
	{ 81	, " post-operative general surgery " } ,
	{ 90	, " general medical " } ,
	{ 100	, " unspecified/unknown " } 
};

static alfabetic electrode_configuration_standard[]={
	{ 0	, " unspecified/unknown " } ,
	{ 1	, " 12-lead positions: RA, RL, LA, and LL at limb extremities. V1 to V6 at standard positions on the chest. Individually " } ,
	{ 2	, " RA, RL, LA, and LL are placed on the torso. V1 to V6 are placed at standard positions on the chest. Individually " } ,
	{ 3	, " RA, RL, LA, and LL are individually placed on the torso. V1 to V6 on the chest as part of a single electrode pad " } ,
	{ 4	, " RA, RL, LA, LL, and V1 to V6 (all electrodes) are on the chest in a single electrode pad " } ,
	{ 5	, " 12-lead ECG is derived from Frank XYZ leads " } ,
	{ 6	, " 12-lead ECG is derived from non-standard leads " }
};

static alfabetic electrode_configuration_XYZ[]={
	{ 0	, " unspecified/unknown " } ,
	{ 1	, " Frank " } ,
	{ 2	, " McFee-Parungao " } ,
	{ 3	, " Cube " } ,
	{ 4	, " XYZ bipolar uncorrected " } ,
	{ 5	, " pseudo-orthogonal XYZ (as used in Holter) " } ,
	{ 6	, " XYZ derived from standard 12 leads " }
};

static alfabetic lead_identification[]={
	{ 0	, " unspecified/unknown " } ,
	{ 1	, " I " } ,
	{ 2	, " II " } ,
	{ 3	, " V1 " } ,
	{ 4	, " V2 " } ,
	{ 5	, " V3 " } ,
	{ 6	, " V4 " } ,
	{ 7	, " V5 " } ,
	{ 8	, " V6 " } ,
	{ 9	, " V7 " } ,
	{ 10	, " V2R " } ,
	{ 11	, " V3R " } ,
	{ 12	, " V4R " } ,
	{ 13	, " V5R " } ,
	{ 14	, " V6R " } ,
	{ 15	, " V7R " } ,
	{ 16	, " X " } ,
	{ 17	, " Y " } ,
	{ 18	, " Z " } ,
	{ 19	, " CC5 " } ,
	{ 20	, " CM5 " } ,
	{ 21	, " left arm " } ,
	{ 22	, " right arm " } ,
	{ 23	, " left leg " } ,
	{ 24	, " I " } ,
	{ 25	, " R " } ,
	{ 26	, " C " } , 
	{ 27	, " A " } ,
	{ 28	, " M " } ,
	{ 29	, " F " } ,
	{ 30	, " H " } ,
	{ 31	, " I-cal " } ,
	{ 32	, " II-cal " } ,
	{ 33	, " V1-cal " } ,
	{ 34	, " V2-cal " } ,
	{ 35	, " V3-cal " } ,
	{ 36	, " V4-cal " } ,
	{ 37	, " V5-cal " } ,
	{ 38	, " V6-cal " } ,
	{ 39	, " V7-cal " } ,
	{ 40	, " V2R-cal " } ,
	{ 41	, " V3R-cal " } ,
	{ 42	, " V4R-cal " } ,
	{ 43	, " V5R-cal " } ,
	{ 44	, " V6R-cal " } ,
	{ 45	, " V7R-cal " } ,
	{ 46	, " X-cal " } ,
	{ 47	, " Y-cal " } ,
	{ 48	, " Z-cal " } ,
	{ 49	, " CC5-cal " } ,
	{ 50	, " CM5-cal " } ,
	{ 51	, " left arm-cal " } ,
	{ 52	, " right arm-cal " } ,
	{ 53	, " left leg-cal " } ,
	{ 54	, " I-cal " } ,
	{ 55	, " R-cal " } ,
	{ 56	, " C-cal " } ,
	{ 57	, " A-cal " } ,
	{ 58	, " M-cal " } ,
	{ 59	, " F-cal " } ,
	{ 60	, " H-cal " } ,
	{ 61	, " III " } ,
	{ 62	, " aVR " } ,
	{ 63	, " aVL " } ,
	{ 64	, " aVF " } ,
	{ 65	, " -aVR " } ,
	{ 66	, " V8 " } ,
	{ 67	, " V9 " } ,
	{ 68	, " V8R " } ,
	{ 69	, " V9R " } ,
	{ 70	, " D (Nehb-dorsal) " } ,
	{ 71	, " A (Nehb-anterior) " } ,
	{ 72	, " J (Nehb-inferior) " } ,
	{ 73	, " defibrillator anterior-lateral " } ,
	{ 74	, " external pacing anterior-posterior " } ,
	{ 75	, " A1 (auxiliary unipolar lead 1) " } ,
	{ 76	, " A2 (auxiliary unipolar lead 2) " } ,
	{ 77	, " A3 (auxiliary unipolar lead 3) " } ,
	{ 78	, " A4 (auxiliary unipolar lead 4) " } ,
	{ 79	, " V8-cal " } ,
	{ 80	, " V9-cal " } ,
	{ 81	, " V8R-cal " } ,
	{ 82	, " V9R-cal " } ,
	{ 83	, " D-cal (Nehb-dorsal) " } ,
	{ 84	, " A-cal (Nehb-anterior) " } ,
	{ 85	, " J-cal (Nehb-inferior) " } 
};

static alfabetic _encode[]={
	{ 0	, " real " } ,
	{ 1	, " first difference " } ,
	{ 2	, " second difference " }
};

static alfabetic _compression[]={
	{ 0	, " bimodal compression not used " } ,
	{ 1	, " bimodal compression used " }
};

static alfabetic spike_type[]={
	{ 0	, " unspecified/unknown " } ,
	{ 1	, " spike triggers neither P-wave nor QRS " } ,
	{ 2	, " spike triggers a QRS " } ,
	{ 3	, " spike triggers a P-wave " } 
};

static alfabetic source_pacemaker[]={
	{ 0	, " unspecified/unknown " } ,
	{ 1	, " internal " } ,
	{ 2	, " external " } 
};

static alfabetic triggered_spike[]={
	{ 0	, " spike does not trigger a QRS " } ,
	{ 1	, " spike triggers a QRS " } 
};

static alfabetic _formula_type[]={
	{ 0	, " unspecified/unknown " } ,
	{ 1	, " Bazett " } ,
	{ 2	, " Hodges " } 
};

static alfabetic ID_tag[]={
	{ 0	, " QTend all-lead dispersion " } ,
	{ 1	, " QTpeak all-lead dispersion " } ,
	{ 2	, " QTend precordial dispersion " } ,
	{ 3	, " QTpeak precordial dispersion " } ,
	{ 4	, " unspecified/unknown " } 
};

static alfabetic value_tag[]={
	{ 0	, " Dispersion = maximum QT interval ? minimum QT interval " } ,
	{ 1	, " Heart rate corrected Dispersion: Max?Min " } ,
	{ 2	, " Dispersion = standard deviation of the QT intervals " } ,
	{ 3	, " Heart rate corrected Dispersion: standard deviation " } ,
	{ 4	, " Heart rate correction formula. (See definition of byte 7 for valid values) " } 
};

static alfabetic type_confirm[]={
	{ 0	, " original report (not overread) " } ,
	{ 1	, " confirmed report " } ,
	{ 2	, " overread report, but not confirmed " } ,
	{ 3	, " unspecified/unknown " } 
};

static alfabetic morphology_description[]={
	{ 0	, " unspecified/unknown " } ,
	{ 1	, " positive " } ,
	{ 2	, " negative " } ,
	{ 3	, " positive/negative " } ,
	{ 4	, " negative/positive " } ,
	{ 5	, " positive/negative/positive " } ,
	{ 6	, " negative/positive/negative " } ,
	{ 7	, " notched M-shaped " } ,
	{ 8	, " notched W-shaped " } 
};

static alfabetic quality_code[]={
	{ 0	, " AC (mains) noise " } ,
	{ 1	, " overrange " } ,
	{ 2	, " wander " } ,
	{ 3	, " tremor or muscle artifact " } ,
	{ 4	, " spike or sudden jumps " } ,
	{ 5	, " electrode loose or off " } ,
	{ 6	, " pacemaker " } ,
	{ 7	, " interchanged lead " } 
};

static alfabetic noise_level[]={
	{ 0	, " none/no " } ,
	{ 1	, " moderate/yes " } ,
	{ 2	, " severe " } ,
	{ 3	, " unknown " }
};

static alfabetic type_statement[]={
	{ 1	, " coded statement type " } ,
	{ 2	, " full text type " } ,
	{ 3	, " statement logic type " } 
};

#endif /*__CODES_H__*/
