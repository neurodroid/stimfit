/*
    $Id: scp-decode.cpp,v 1.24 2008-07-12 20:46:58 schloegl Exp $
   Copyright (C) 2011,2014 Alois Schloegl <alois.schloegl@gmail.com>
   Copyright (C) 2011 Stoyan Mihaylov
   This function is part of the "BioSig for C/C++" repository 
    (biosig4c++) at http://biosig.sf.net/ 

Modifications by Alois Schloegl 
    Jul 2011: get rid of warnings for unitialized variables and signed/unsigned comparison
    Jun 2007: replaced ultoa with sprintf	
    Aug 2007: On-The-Fly-Decompression using ZLIB
    Oct 2007: Consider SunOS/SPARC platform 
    	      obsolete code sections marked, this reduced SegFault from 18 to 1.
    	      

---------------------------------------------------------------------------
Copyright (C) 2006  Eugenio Cervesato.
Developed at the Associazione per la Ricerca in Cardiologia - Pordenone - Italy,
based on the work of Eugenio Cervesato & Giorgio De Odorico. The original
Copyright and comments follow.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
---------------------------------------------------------------------------

______________________________________________________________________________

   scp-decode.cpp       This is the "decode" module of the program SCP-AV.
                        It opens an SCP-ECG v1.0 to v2.0 test file and
                        extracts all the informations.

                         Release 2.3 - feb 2006
---------------------------------------------------------------------------
**************************************************************************

************************* original Copyright *****************************
Copyright (C) 2003-2004  Eugenio Cervesato & Giorgio De Odorico.
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
//______________________________________________________________________________
/*
   scp-decode.cpp       This is the "decode" module of the program SCP-AV.
                        It opens an SCP-ECG v1.3 or v2.0 test file and
                        extracts all the informations.

                	Developed by ING. GIORGIO DE ODORICO (giorgio.deodorico@tiscali.it)

                        Documentation of the standard comes mainly from:
                        http://www.centc251.org/TCMeet/doclist/TCdoc02/N02-015-prEN1064.pdf

                        Internationalization, test, bug fix by Eugenio Cervesato (eugenio.cervesato@aopn.fvg.it)

                        Release 2.1 - february 2004

*/
// ************************* end of the original Copyright *****************************

// contribution of Michael Breuss <m.breuss@tma-medical.com>. see 'by MB' in the sources
// contribution of Stelios Sfakianakis <ssfak@ics.forth.gr>. see 'by SS' in the sources
// contribution of Federico cantini <cantini@ifc.cnr.it>. see 'by FeC' in the sources

//void remark(char *string);

// 
#define WITH_OBSOLETE_PARTS 


//      by E.C. 13.10.2003   part nedded to compile with gcc (Linux).
//                           To compile with Borland C++ add the conditional define: WIN32.
//                           In porting, I nedded to adapt fseek() and write a custom ultoa()

#define COMPAT

//______________________________________________________________________________

//#include <iostream>
#include <cstring>    //strcat, strcpy
#include <cstdio>
#include <stdlib.h>
using namespace std;

/*
   error handling should use error variables local to each HDR
   otherwise, sopen() etc. is not re-entrant.

   Therefore, use of variables B4C_ERRNUM and B4C_ERRMSG is deprecated;
   Instead, use biosigERROR for setting error status, and
   serror2(hdr), hdr->AS.B4C_ERRNUM, hdr->AS.B4C_ERRMSG for error handling.

 */
__attribute__ ((deprecated)) extern int B4C_ERRNUM;
__attribute__ ((deprecated)) extern const char *B4C_ERRMSG;


//______________________________________________________________________________
//               FILE POINTERS

#include "../biosig-dev.h"
#include "structures.h"
#include "codes.h"
//     the following define is private of Eugenio Cervesato. Please other readers ignore it!
#ifdef CPPBUILDER3
#include "CPPBUILDER3.h"  // inside are definitions needed to run on C++Builder GUI as a standalone module and bypass ZLIB
#endif
//     end of private define.
HDRTYPE* in;

//---------------------------------------------------------------------------
static uint32_t _COUNT_BYTE=1UL;                  // counter of bytes read
static uint32_t _DIM_FILE;                      // file length in byte
static const uint8_t _NUM_SECTION=12U;     // sections over 11 are not considered

//______________________________________________________________________________
//                      section 2
uint8_t         Input_Bit(uint8_t*,uint16_t&,uint16_t,uint8_t&,bool&);
int16_t         Input_Bits(uint8_t*,uint16_t&,uint16_t,uint8_t&,uint8_t,bool&);
void            decompress(TREE_NODE*,int16_t*,uint8_t&,uint16_t&,int32_t*,uint16_t,uint16_t&,table_H*,uint16_t*,uint16_t&);
void            Tree_Destroy(TREE_NODE*);
TREE_NODE       *Tree_Create(TREE_NODE*,uint16_t,table_H*,uint16_t);
void            Huffman(int32_t*,uint16_t*,uint8_t*,uint16_t&,uint16_t,table_H*,uint16_t*);
void            InitHuffman(table_H*);                          //inizialize default Huffman table

//______________________________________________________________________________
//                      sections 3, 4, 5 and 6
template<class t1>
void            Differences(int32_t*,t1,uint8_t);
void            Multiply(int32_t*,uint32_t,uint16_t);
void            Interpolate(int32_t*,int32_t*,f_lead,lead*,f_Res,Protected_Area*,uint32_t);
void            ExecFilter(int32_t*,int32_t*,uint32_t&,uint16_t);
void            DoFilter(int32_t*,int32_t*,f_Res,f_lead,lead*,Protected_Area*,Subtraction_Zone*);
void            DoAdd(int32_t*,int32_t*,f_Res,int32_t*,f_BdR0,Subtraction_Zone*,f_lead,lead*);
void            Opt_Filter(int32_t*, int32_t*, f_Res, f_lead, lead*, Protected_Area*);

//______________________________________________________________________________
//                             INTERNAL FUNCTIONS
char*           ReadString(char*,uint16_t);                          //read a string
char            *FindString(char*,uint16_t);                         // calculate the length of a string and write it down
int16_t         Look(alfabetic*,uint16_t,uint16_t,uint16_t);      //look at a number in alfabetic and give the position of the array

//______________________________________________________________________________
template<class t1>
void            ReadByte(t1&);         //read a byte from stream
void            Skip(uint16_t);        //skip some bytes

//______________________________________________________________________________
uint16_t        ReadCRC();                             //read first 6 bytes of the file
bool            Check_CRC(uint16_t,uint32_t,uint32_t);     // CRC check
//______________________________________________________________________________

uint32_t        ID_section(uint32_t, int8_t &version);              //read section ID header
void            sectionsOptional(pointer_section*,DATA_DECODE &,DATA_RECORD&,DATA_INFO&);       //handles optional sections

#ifdef WITH_OBSOLETE_PARTS
void            section_0(pointer_section*, int size_max);                    //read section 0
void            Init_S1(DATA_INFO &inf);
void            section_1(pointer_section,DATA_INFO&);    //read section 1 data
void            section_1_0(demographic&);                        //read tag 0 of section 1
void            section_1_1(demographic&);                        //read tag 1 of section 1
void            section_1_2(demographic&);                        // ... and so on ...
void            section_1_3(demographic&);
void            section_1_4(demographic&);
void            section_1_5(demographic&);
void            section_1_6(demographic&);
void            section_1_7(demographic&);
void            section_1_8(demographic&);
void            section_1_9(demographic&);
void            section_1_10(clinic&,uint16_t&);
void            section_1_11(demographic&);
void            section_1_12(demographic&);
void            section_1_13(clinic&,uint16_t&);
void            section_1_14(descriptive&);
void            section_1_15(descriptive&);
void            section_1_16(descriptive&);
void            section_1_17(descriptive&);
void            section_1_18(descriptive&);
void            section_1_19(descriptive&);
void            section_1_20(clinic&);
void            section_1_21(clinic&);
void            section_1_22(clinic&);
void            section_1_23(descriptive&);
void            section_1_24(descriptive&);
void            section_1_25(device&);
void            section_1_26(device&);
void            section_1_27(device&);
void            section_1_28(device&);
void            section_1_29(device&);
void            section_1_30(clinic&,uint16_t&);
void            section_1_31(device&);
void            section_1_32(clinic&,uint16_t&, int8_t version);
void            section_1_33(device&);
void            section_1_34(device&);
void            section_1_35(clinic&,uint16_t&);
void            section_1_();                                   //skip tags of the manufacturer of the section 1
void            section_1_255();                                //read tag 255 of section 1
void            section_7(pointer_section,DATA_RECORD&, int8_t version); //read section 7
void            section_8(pointer_section,DATA_INFO&);          //read section 8
void            section_10(pointer_section,DATA_RECORD&, int8_t version); //read section 10
void            section_11(pointer_section,DATA_INFO&);         //read section 11
#endif
void            section_2(pointer_section,DATA_DECODE&);        //read section 2
void            section_3(pointer_section,DATA_DECODE&, int8_t version); //read section 3
void            section_4(pointer_section,DATA_DECODE&, int8_t version); //read section 4
bool            section_5(pointer_section,DATA_DECODE&,bool);   //read section 5
void            section_6(pointer_section,DATA_DECODE&,bool);   //read section 6

//______________________________________________________________________________
void            Decode_Data(pointer_section*,DATA_DECODE&,bool&);

//______________________________________________________________________________
#define STR_NULL StrNull()
void *FreeWithCare(void*P){
//Stoyan - else I got problems with some scp files - prestandart or so
	if(P) 
		free(P);
	return NULL;
}

void *mymalloc(size_t size)            // by E.C. 07.11.2003    this is a workaround for a bug
{                                      // present somewhere in memory allocation.
//        char buff[30];               // this problem should be fixed next!
//        ultoa(size, buff, 10);       // used for debug purposes, shows the size
//        remark(buff);
//	fprintf(stdout,"MYMEMALLOC: %i\n",size);
//        void *res=malloc(size*2);      // this way each time a doubled memory is requested. And it works!!
	void *res=malloc(size);
	return res;
}
const int StrNullLen=strlen(" unspecified/unknown ");
char * StrNull(){
//Stoyan this way we can release everything 
	char*Ret=(char*)mymalloc(StrNullLen+4);
	strcpy(Ret," unspecified/unknown ");
	return Ret;
}

/* moved by MB
   must be declared before first call (otherwise compiler error)
*/
//--------------------------------BYTE & BIT------------------------------------
template<class t1>
void ReadByte(t1 &number)
{
//read the requested number of bytes and
//convert in decimal, taking into account that the first byte is the LSB.
//the sign of the number is kept
	uint8_t *num, dim=sizeof(t1);
	uint8_t mask=0xFF;


	if(dim!=0 && (num=(uint8_t*)mymalloc(dim))==NULL)
	{
		B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
		B4C_ERRMSG = "SCP-DECODE: Not enough memory";
		return; 
	}
	ifread(num,dim,1,in);
	// *num = *(uint8_t*)(in->AS.Header+_COUNT_BYTE);
	number=0;
	_COUNT_BYTE+=dim;

	while((dim--)>0)
	{
		number<<=8;
		number+=num[dim]&mask;
	}
	free(num);
}//end ReadByte

//                      MAIN

EXTERN_C void sopen_SCP_clean(struct DATA_DECODE *decode, struct DATA_RECORD *record, struct DATA_INFO *textual) {
	FreeWithCare(decode->length_BdR0);
	FreeWithCare(decode->samples_BdR0);
	FreeWithCare(decode->length_Res);
	FreeWithCare(decode->samples_Res);
	FreeWithCare(decode->t_Huffman);
	FreeWithCare(decode->flag_Huffman);
	FreeWithCare(decode->data_lead);
	FreeWithCare(decode->data_protected);
	FreeWithCare(decode->data_subtraction);
	FreeWithCare(decode->Median);
	FreeWithCare(decode->Residual);
	FreeWithCare(decode->Reconstructed);

	FreeWithCare(record->data_spike);
	FreeWithCare(record->type_BdR);
	FreeWithCare(record->data_BdR);
	FreeWithCare(record->data_additional);
	FreeWithCare(record->lead_block);
	FreeWithCare(textual->text_dim);
	FreeWithCare(textual->data_statement);
	FreeWithCare(textual->text_statement);

	
	FreeWithCare(textual->ana.last_name);
	FreeWithCare(textual->ana.first_name);
	FreeWithCare(textual->ana.ID);
	FreeWithCare(textual->ana.second_last_name);
	FreeWithCare(textual->cli.text_drug);
	FreeWithCare(textual->cli.text_diagnose);
	FreeWithCare(textual->cli.referring_physician);
	FreeWithCare(textual->cli.latest_confirming_physician);
	FreeWithCare(textual->cli.technician_description);
	FreeWithCare(textual->cli.text_free_text);
	FreeWithCare(textual->cli.text_free_medical_hystory);
	FreeWithCare(textual->cli.medical_hystory);
	FreeWithCare(textual->cli.free_text);
	FreeWithCare(textual->cli.drug);

	FreeWithCare(textual->des.acquiring.model_description);
	FreeWithCare(textual->des.acquiring.analysing_program_revision_number);
	FreeWithCare(textual->des.acquiring.serial_number_device);
	FreeWithCare(textual->des.acquiring.device_system_software);
	FreeWithCare(textual->des.acquiring.device_SCP_implementation_software);
	FreeWithCare(textual->des.acquiring.manifacturer_trade_name);
	FreeWithCare(textual->des.analyzing.model_description);
	FreeWithCare(textual->des.analyzing.analysing_program_revision_number);
	FreeWithCare(textual->des.analyzing.serial_number_device);
	FreeWithCare(textual->des.analyzing.device_system_software);
	FreeWithCare(textual->des.analyzing.device_SCP_implementation_software);
	FreeWithCare(textual->des.analyzing.manifacturer_trade_name);
	FreeWithCare(textual->des.acquiring_institution);
	FreeWithCare(textual->des.analyzing_institution);
	FreeWithCare(textual->des.acquiring_department);
	FreeWithCare(textual->des.analyzing_department);
	FreeWithCare(textual->des.room);
	FreeWithCare(textual->dev.sequence_number);
	FreeWithCare((char*)textual->dev.TZ.description);

}
//There is serious problem if we try to transfer whole structures between c and c++. I am not sure were exactly it is, but using pointers - solve it. extern C is not enough. Stoyan
EXTERN_C int scp_decode(HDRTYPE* hdr, pointer_section *section, struct DATA_DECODE *decode, struct DATA_RECORD *info_recording, struct DATA_INFO *info_textual, bool add_filter)
{
	uint16_t CRC;
	uint32_t pos;
	if (hdr->FILE.OPEN) {
		ifseek(hdr,0,SEEK_SET);
	}
	else 	
		hdr = ifopen(hdr,"rb");

	if (!hdr->FILE.OPEN)
	{
		fprintf(stdout,"Cannot open the file %s.\n",hdr->FileName);
		return FALSE;              // by E.C. 15.10.2003    now return FALSE
	}

	in = hdr;
	_COUNT_BYTE=1UL;
	CRC=ReadCRC();
	pos=_COUNT_BYTE;
	ReadByte(_DIM_FILE);
//	if (CRC != 0xFFFF) Check_CRC(CRC,pos,_DIM_FILE-2U);  // by E.C. may 2004 CARDIOLINE 1.0
	ifseek(in, 0L, SEEK_SET);

//mandatory sections
#ifdef WITH_OBSOLETE_PARTS
	section_0(section, _DIM_FILE);                 // by E.C. may 2004 check file size
	section_1(section[1],*info_textual);
	sectionsOptional(section,*decode,*info_recording,*info_textual);
#else 

	if (section[2].length>0)	
		section_2(section[2],*decode);       //HUFFMAN
	if (section[3].length>0)	
		section_3(section[3],*decode,hdr->aECG->Section1.Tag14.VERSION);      //lead
	if (section[4].length) 
		section_4(section[4],*decode,hdr->aECG->Section1.Tag15.VERSION);       // fiducial locations
	if (section[5].length)
		if (!section_5(section[5],*decode,section[2].length)) 
			section[5].length=0 ;       //type 0 median beat
	if (section[6].length)
		section_6(section[6],*decode,section[2].length);       //rhythm compressed data

#endif
 
	ifclose(in);

	Decode_Data(section,*decode,add_filter);
	return TRUE;              // by E.C. 15.10.2003    now return TRUE
}
//______________________________________________________________________________
//                           COMPUTATIONAL FUNCTIONS

#ifdef WITH_OBSOLETE_PARTS
//------------------------------STRINGS----------------------------------------
char *ReadString(char *temp_string, uint16_t num)
//read a string from the stream.
//the first extracted byte is written for fist.
//each byte read from the stream is first "transformed" in char.
{
	if(temp_string)
		free(temp_string);
	if(!num)
		return NULL;//before alocating memory, which will be loosed in case of num=0
	if((temp_string=(char*)mymalloc(sizeof(char)*(num+2)))==NULL)    // by E.C. 26.02.2004 one more byte
	{
		B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
		B4C_ERRMSG = "SCP-DECODE: Not enough memory";
		return NULL; 
	}
	_COUNT_BYTE+=num;

	ifread(temp_string,sizeof(char),num,in);
	if (temp_string[num-1]!='\0')
		temp_string[num]='\0';

	return temp_string;
}//end ReadString

int16_t Look(alfabetic *code_, uint16_t a, uint16_t b, uint16_t key_)
// look num in code_.number and give the element position
{
	uint16_t middle=(a+b)/2U;

	if(code_[middle].number==key_)
		return middle;
	if(a>=b)
		return -1;
	if(code_[middle].number<key_)
		return Look(code_,middle+1,b,key_);
	else
		return Look(code_,a,middle-1,key_);
}//end Look

char *FindString(char *temp_string,uint16_t max)
//read bytes until NULL
//Stoyan - there were memory leaks
{
	if(temp_string)
		free(temp_string);
	char c;
	uint16_t num=0;
	//fpos_t
	long filepos;

	if(!max)
		return NULL;

	filepos = iftell(in); //FGETPOS(in,&filepos);
	do
	{
		c=ifgetc(in);
		++num;
	}
	while(c!='\0' && num<max);

	ifseek(in,filepos COMPAT,0);

	if((temp_string=(char*)mymalloc(sizeof(char)*(num+2)))==NULL)   // by E.C. one extra byte nedded
	{                                                               // for later str_cat()
		B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
		B4C_ERRMSG = "SCP-DECODE: Not enough memory";
		return NULL; 
	}

	if(!num)
		return NULL;

	_COUNT_BYTE+=num;

	ifread(temp_string,sizeof(char),num,in);
	if (temp_string[num-1]!='\0')
		temp_string[num]='\0';

	return temp_string;
}//end FindString
#endif

void Skip(uint16_t num)
//skip num bytes from the stream
{
	if(num>0U)
		ifseek(in,num,1U);
	_COUNT_BYTE+=num;
}//end Skip

//______________________________________________________________________________
//                         INITIALIZATION FUNCTIONS

void InitHuffman(table_H *riga)
//build the default Huffman table
{
/*
  The table is contructed as stated in the protocol SCP-ECG; each structure in each line.
  Columns are:
        bit_prefix = number of bits in the prefix
        bit_code   = number of bits in the code
        TMS        = table mode
        base_value = base value (decoded)
        cp         = prefix code (in bits)
        base_code  = decimal value of reversed cp

  From the stream I take a bit at a time until I find a correspondent value in the Huffman table.
  If:
        nbp=nbc the decode value is vb
        nbp!=nbc
           if m=1 I take nbc-nbp bits next in the stream
           if m=0 change to table vb
  Remark: Huffman tables stored in the stream contain vb in 2 complemented bytes.
  The decimal value of cp is included. Furthermore, always the MSB comes first!
  In my Huffman tables I set cp as a decimal value. Infact, I read a group of bits
  from the stream and convert them by multipying by 2^pos (where pos is
  0 for bit 0, 1 for bit 1 and so on). So I read bits as they are.

              DEFAULT HUFFMAN TABLE
             nbp nbc  m  vb  cp10    cp
               1,  1, 1,  0,    0,    0,
               3,  3, 1,  1,    1,    4,
               3,  3, 1, -1,    5,    5,
               4,  4, 1,  2,    3,   12,
               4,  4, 1, -2,   11,   13,
               5,  5, 1,  3,    7,   28,
               5,  5, 1, -3,   23,   29,
               6,  6, 1,  4,   15,   60,
               6,  6, 1, -4,   47,   61,
               7,  7, 1,  5,   31,  124,
               7,  7, 1, -5,   95,  125,
               8,  8, 1,  6,   63,  252,
               8,  8, 1, -6,  191,  253,
               9,  9, 1,  7,  127,  508,
               9,  9, 1, -7,  383,  509,
              10, 10, 1,  8,  255, 1020,
              10, 10, 1, -8,  767, 1021,
              10, 18, 1,  0,  511, 1022,
              10, 26, 1,  0, 1023, 1023,
*/

	uint8_t i;

	i= 0U;  riga[i].bit_prefix= 1U; riga[i].bit_code= 1U; riga[i].TMS=1U; riga[i].base_value= 0; riga[i].base_code=   0UL;
	i= 1U;  riga[i].bit_prefix= 3U; riga[i].bit_code= 3U; riga[i].TMS=1U; riga[i].base_value= 1; riga[i].base_code=   1UL;
	i= 2U;  riga[i].bit_prefix= 3U; riga[i].bit_code= 3U; riga[i].TMS=1U; riga[i].base_value=-1; riga[i].base_code=   5UL;
	i= 3U;  riga[i].bit_prefix= 4U; riga[i].bit_code= 4U; riga[i].TMS=1U; riga[i].base_value= 2; riga[i].base_code=   3UL;
	i= 4U;  riga[i].bit_prefix= 4U; riga[i].bit_code= 4U; riga[i].TMS=1U; riga[i].base_value=-2; riga[i].base_code=  11UL;
	i= 5U;  riga[i].bit_prefix= 5U; riga[i].bit_code= 5U; riga[i].TMS=1U; riga[i].base_value= 3; riga[i].base_code=   7UL;
	i= 6U;  riga[i].bit_prefix= 5U; riga[i].bit_code= 5U; riga[i].TMS=1U; riga[i].base_value=-3; riga[i].base_code=  23UL;
	i= 7U;  riga[i].bit_prefix= 6U; riga[i].bit_code= 6U; riga[i].TMS=1U; riga[i].base_value= 4; riga[i].base_code=  15UL;
	i= 8U;  riga[i].bit_prefix= 6U; riga[i].bit_code= 6U; riga[i].TMS=1U; riga[i].base_value=-4; riga[i].base_code=  47UL;
	i= 9U;  riga[i].bit_prefix= 7U; riga[i].bit_code= 7U; riga[i].TMS=1U; riga[i].base_value= 5; riga[i].base_code=  31UL;
	i=10U;  riga[i].bit_prefix= 7U; riga[i].bit_code= 7U; riga[i].TMS=1U; riga[i].base_value=-5; riga[i].base_code=  95UL;
	i=11U;  riga[i].bit_prefix= 8U; riga[i].bit_code= 8U; riga[i].TMS=1U; riga[i].base_value= 6; riga[i].base_code=  63UL;
	i=12U;  riga[i].bit_prefix= 8U; riga[i].bit_code= 8U; riga[i].TMS=1U; riga[i].base_value=-6; riga[i].base_code= 191UL;
	i=13U;  riga[i].bit_prefix= 9U; riga[i].bit_code= 9U; riga[i].TMS=1U; riga[i].base_value= 7; riga[i].base_code= 127UL;
	i=14U;  riga[i].bit_prefix= 9U; riga[i].bit_code= 9U; riga[i].TMS=1U; riga[i].base_value=-7; riga[i].base_code= 383UL;
	i=15U;  riga[i].bit_prefix=10U; riga[i].bit_code=10U; riga[i].TMS=1U; riga[i].base_value= 8; riga[i].base_code= 255UL;
	i=16U;  riga[i].bit_prefix=10U; riga[i].bit_code=10U; riga[i].TMS=1U; riga[i].base_value=-8; riga[i].base_code= 767UL;
	i=17U;  riga[i].bit_prefix=10U; riga[i].bit_code=18U; riga[i].TMS=1U; riga[i].base_value= 0; riga[i].base_code= 511UL;
	i=18U;  riga[i].bit_prefix=10U; riga[i].bit_code=26U; riga[i].TMS=1U; riga[i].base_value= 0; riga[i].base_code=1023UL;
}//end InitHuffman

//______________________________________________________________________________
//                           handle sections

uint16_t ReadCRC()
// read the CRC of the entire file or of a section and convert it to decimal.
{
	uint16_t dim;

	ReadByte(dim);

	return dim;
}//end ReadCRC

bool Check_CRC(uint16_t CRC, uint32_t pos, uint32_t length)
/* CRC check starting from pos for Length

Remark: all computations are in byte.
A	= new byte
B	= temp byte
CRCHI	= MSB of the CRC (16 bits)
CRCLO	= LSB of the CRC

START:
	for A=first_byte to last_byte in block do:
	        A = A xor CRCHI
	        CRCHI = A
		shift A right 4 times    	{fulfill with zeroes}
		A = A xor CRCHI			{I J K L M N O P}
		CRCHI = CRCLO			{swap CRCHI, CRCLO}
		CRCLO = A
		rotate A left 4 times     	{M N O P I J K L}
                B = A 				{ temp save }
                rotate A left once       	{ N O P I J K L M }
                A = A and $1F 			{ 0 0 0 I J L L M }
                CRCHI = A xor CRCHI
                A = B and $F0 			{ M N O P 0 0 0 0 }
                CRCHI = A xor CRCHI		{ CRCHI complete }
                rotate B left once        	{ N O P 0 0 0 0 M }
                B = B and $E0 			{ N O P 0 0 0 0 0 }
                CRCLO = B xor CRCLO 		{ CRCLO complete }
        end

Final check on the CRC is accomplished by adding or concatenating CRCHI and CRCLO at
the end of the data stream. Calculating the checksum of the resulting data stream shall result in
a zero CRC if the data was correctly received.

*/
{
	uint32_t i;
	uint8_t A, B;
	uint8_t CRCLO, CRCHI;

	CRCLO=0xFF;
	CRCHI=0xFF;
	ifseek(in,pos-1,0U);
	for(i=1;i<=length;i++)
	{
		A=ifgetc(in);
		A^=CRCHI;
		A^=(A>>4);
		CRCHI=CRCLO;
		CRCLO=A;
		A=(A<<4)|(A>>4U);
		B=A;
		A=(A<<1)|(A>>7U);
		A&=0x1F;
		CRCHI^=A;
		A=B&0xF0;
		CRCHI^=A;
		B=(B<<1)|(B>>7U);
		B&=0xE0;
		CRCLO^=B;
	}//end for

	CRCLO-=CRC%256UL;
	CRCHI-=CRC/256UL;
	if ((CRCLO==CRCHI) && (CRCLO==0))
		return 1;
	else
	{
		fprintf(stderr,"Cannot read the file: BAD CRC.\n");
//		exit(2);
		return 0;
	}
}//end Check_CRC

uint32_t ID_section(uint32_t pos, int8_t &version)
//read section Header
{
	uint32_t dim;
	uint16_t CRC;

	CRC=ReadCRC();
	Skip(2U);
	ReadByte(dim);
//	if (CRC != 0xFFFF) Check_CRC(CRC,pos+2,dim-2);  // by E.C. may 2004 CARDIOLINE 1.0
	ifseek(in,pos+7L,0);
	ReadByte(version);                   // by E.C. may 2004  store the version number
	Skip(7U);

	return dim;
}//end ID_section


void sectionsOptional(pointer_section *section, DATA_DECODE &block1, DATA_RECORD &block2, DATA_INFO &block3)
//handles optional sections
{
	uint8_t i=0, bimodal;
//initialization
	block1.t_Huffman=NULL;
	block1.flag_Huffman=NULL;
	block1.data_lead=NULL;
	block1.data_protected=NULL;
	block1.data_subtraction=NULL;
	block1.length_BdR0=NULL;
	block1.samples_BdR0=NULL;
	block1.Median=NULL;
	block1.length_Res=NULL;
	block1.samples_Res=NULL;
	block1.Residual=NULL;
//	block1.Reconstructed=NULL;
	block2.data_spike=NULL;
	block2.type_BdR=NULL;
	block2.data_BdR=NULL;
	block2.data_additional=NULL;
	block2.lead_block=NULL;
	block3.text_dim=NULL;
	block3.text_report=NULL;
	block3.data_statement=NULL;
	block3.text_statement=NULL;

	//variables inizialization
	block1.flag_lead.number=0;
	block1.flag_lead.subtraction=0;
	block1.flag_lead.all_simultaneously=0;
	block1.flag_lead.number_simultaneously=0;

	block1.flag_BdR0.length=0;
	block1.flag_BdR0.fcM=0;
	block1.flag_BdR0.AVM=0;
	block1.flag_BdR0.STM=0;
	block1.flag_BdR0.number_samples=0;
	block1.flag_BdR0.encoding=0;

	block1.flag_Res.AVM=0;
	block1.flag_Res.STM=0;
	block1.flag_Res.number=0;
	block1.flag_Res.number_samples=0;
	block1.flag_Res.encoding=0;
	block1.flag_Res.bimodal=0;
	block1.flag_Res.decimation_factor=0;

	block2.data_global.number=0;
	block2.data_global.number_QRS=0;
	block2.data_global.number_spike=0;
	block2.data_global.average_RR=0;
	block2.data_global.average_PP=0;
	block2.data_global.ventricular_rate=0;
	block2.data_global.atrial_rate=0;
	block2.data_global.QT_corrected=0;
	block2.data_global.formula_type=0;
	block2.data_global.number_tag=0;

	block2.header_lead.number_lead=0;
	block2.header_lead.number_lead_measurement=0;

	while(i<_NUM_SECTION)
	{
		if(section[i].ID)
			switch(section[i].ID)
			{
				case 2: if(section[i].length)
							section_2(section[i],block1);       //HUFFMAN
						break;
				case 3: if(section[i].length)
							section_3(section[i],block1,block3.des.acquiring.protocol_revision_number);      //lead
						break;
				case 4: if(section[i].length)
						{
							if((block3.des.acquiring.protocol_revision_number>10) && section[6].length)      // by E.C. 27.02.2004 whole section to be included in {} !
							{
								ifseek(in,section[6].index+22,0);
								ReadByte(bimodal);
								block1.flag_Res.bimodal=bimodal;
							}
							else
								block1.flag_Res.bimodal=0;
							section_4(section[i],block1,block3.des.analyzing.protocol_revision_number);       // fiducial locations
						}
						break;
				case 5: if(section[i].length)
							if (!section_5(section[i],block1,section[2].length)) section[i].length=0 ;       //type 0 median beat
						break;
				case 6: if(section[i].length)
							section_6(section[i],block1,section[2].length);       //rhythm compressed data
						break;

#ifdef WITH_OBSOLETE_PARTS
/*
				case 7: if(section[i].length)
							section_7(section[i],block2,block3.des.acquiring.protocol_revision_number);       //global measurements
						break;
				case 8: if(section[i].length)
							section_8(section[i],block3);       //full text interpretative statements
						break;
				case 10:if(section[i].length)
							section_10(section[i],block2,block3.des.acquiring.protocol_revision_number);      //lead measurement block
						break;
				case 11:if(section[i].length)          //universal ECG interpretative statements
				//			section_11(section[i],block3);
						break;
*/
#endif
			}//end switch
		++i;
	}//end while
}//end sectionsOptional

//______________________________________________________________________________
//                                sections
//______________________________________________________________________________

#ifdef WITH_OBSOLETE_PARTS

//______________________________________________________________________________
//                              section 0
//______________________________________________________________________________

void section_0(pointer_section *info, int size_max)
// section 0
//build info_sections with ID, offset and length of each section
{
	uint32_t pos, dim, ini;
	uint16_t ind;
	uint8_t i;
	int8_t version;

	ifseek(in,6L,0);
	pos=ID_section(7L, version)+7L; //length + offset
	_COUNT_BYTE=7L+16L;

	for(i=0;i<_NUM_SECTION;i++)
	{
		info[i].ID=0;
		info[i].length=0L;
		info[i].index=0L;
	}
	while((_COUNT_BYTE+10)<=pos)
	{
		ReadByte(ind);
		if(ind>11U)
			Skip(8U);
		else
		{
			ReadByte(dim);
			if(dim)
			{
				ReadByte(ini);
				if (ini<(unsigned)size_max) {            // by E.C. may 2004 check overflow of file
					info[ind].ID=ind;
					info[ind].length=dim;
					info[ind].index=ini;
				}
			}//end if dim
			else
				Skip(4U);
		}//end else
	}//end while
}//end section_0

//______________________________________________________________________________
//                              section 1
//______________________________________________________________________________
void Init_S1(DATA_INFO &inf)
{
	inf.ana.last_name=STR_NULL;
	inf.ana.first_name=STR_NULL;
	inf.ana.ID=STR_NULL;
	inf.ana.second_last_name=STR_NULL;
	inf.ana.age.value=0;
	inf.ana.age.unit=0;
	inf.ana.height.value=0;
	inf.ana.height.unit=0;
	inf.ana.weight.value=0;
	inf.ana.weight.unit=0;
	inf.ana.sex=0;
	inf.ana.race=0;
	inf.ana.systolic_pressure=0;
	inf.ana.diastolic_pressure=0;

	inf.cli.number_drug=0;
	inf.cli.text_drug=STR_NULL;
	inf.cli.number_diagnose=0;
	inf.cli.text_diagnose=STR_NULL;
	inf.cli.referring_physician=STR_NULL;
	inf.cli.latest_confirming_physician=STR_NULL;
	inf.cli.technician_description=STR_NULL;
	inf.cli.number_text=0;
	inf.cli.text_free_text=STR_NULL;
	inf.cli.number_hystory=0;
	inf.cli.number_free_hystory=0;
	inf.cli.text_free_medical_hystory=STR_NULL;
	inf.cli.free_text=NULL;
	inf.cli.medical_hystory=NULL;
	inf.cli.drug=NULL;

	inf.des.acquiring.institution_number=0;
	inf.des.acquiring.department_number=0;
	inf.des.acquiring.ID=0;
	inf.des.acquiring.type=2;
	inf.des.acquiring.manifacturer=0;
	inf.des.acquiring.model_description=STR_NULL;
	inf.des.acquiring.protocol_revision_number=0;
	inf.des.acquiring.category=255;
	inf.des.acquiring.language=255;
	inf.des.acquiring.capability[0]=1;
	inf.des.acquiring.capability[1]=2;
	inf.des.acquiring.capability[2]=3;
	inf.des.acquiring.capability[3]=4;
	inf.des.acquiring.AC=0;
	inf.des.acquiring.analysing_program_revision_number=STR_NULL;
	inf.des.acquiring.serial_number_device=STR_NULL;
	inf.des.acquiring.device_system_software=STR_NULL;
	inf.des.acquiring.device_SCP_implementation_software=STR_NULL;
	inf.des.acquiring.manifacturer_trade_name=STR_NULL;
	inf.des.analyzing.institution_number=0;
	inf.des.analyzing.department_number=0;
	inf.des.analyzing.ID=0;
	inf.des.analyzing.type=2;
	inf.des.analyzing.manifacturer=0;
	inf.des.analyzing.model_description=STR_NULL;
	inf.des.analyzing.protocol_revision_number=0;
	inf.des.analyzing.category=255;
	inf.des.analyzing.language=255;
	inf.des.analyzing.capability[0]=1;
	inf.des.analyzing.capability[1]=2;
	inf.des.analyzing.capability[2]=3;
	inf.des.analyzing.capability[3]=4;
	inf.des.analyzing.AC=0;
	inf.des.analyzing.analysing_program_revision_number=STR_NULL;
	inf.des.analyzing.serial_number_device=STR_NULL;
	inf.des.analyzing.device_system_software=STR_NULL;
	inf.des.analyzing.device_SCP_implementation_software=STR_NULL;
	inf.des.analyzing.manifacturer_trade_name=STR_NULL;
	inf.des.acquiring_institution=STR_NULL;
	inf.des.analyzing_institution=STR_NULL;
	inf.des.acquiring_department=STR_NULL;
	inf.des.analyzing_department=STR_NULL;
	inf.des.room=STR_NULL;
	inf.des.stat_code=0;

	inf.dev.baseline_filter=0;
	inf.dev.lowpass_filter=0;
	inf.dev.other_filter[0]=0;
	inf.dev.other_filter[1]=0;
	inf.dev.other_filter[2]=0;
	inf.dev.other_filter[3]=0;
	inf.dev.sequence_number=STR_NULL;
	inf.dev.electrode_configuration.value=0;
	inf.dev.electrode_configuration.unit=0;
	inf.dev.TZ.offset=0;
	inf.dev.TZ.index=0;
	inf.dev.TZ.description=STR_NULL;
}

void section_1(pointer_section info_sections, DATA_INFO &inf)
// section 1
{
	uint8_t tag;
	uint32_t num=info_sections.length+_COUNT_BYTE;
	uint16_t dim=0U;
	int8_t version;

	_COUNT_BYTE=info_sections.index;
	ifseek(in,info_sections.index-1,0);
	ID_section(info_sections.index, version);

	Init_S1(inf);

	do
	{
		ReadByte(tag);
		switch(tag)
		{
			case 0:
				section_1_0(inf.ana); break;
			case 1:
				section_1_1(inf.ana); break;
			case 2:
				section_1_2(inf.ana); break;
			case 3:
				section_1_3(inf.ana); break;
			case 4:
				section_1_4(inf.ana); break;
			case 5:
				section_1_5(inf.ana); break;
			case 6:
				section_1_6(inf.ana); break;
			case 7:
				section_1_7(inf.ana); break;
			case 8:
				section_1_8(inf.ana); break;
			case 9:
				section_1_9(inf.ana); break;
			case 10:
				if(!inf.cli.number_drug)
				{
					inf.cli.drug=NULL;
					inf.cli.text_drug=(char*)FreeWithCare(inf.cli.text_drug);
					dim=0;
				}
				section_1_10(inf.cli,dim); break;
			case 11:
				section_1_11(inf.ana); break;
			case 12:
				section_1_12(inf.ana); break;
			case 13:
				if(!inf.cli.number_diagnose)
				{
					inf.cli.diagnose=NULL;
					inf.cli.text_diagnose=(char*)FreeWithCare(inf.cli.text_diagnose);
					dim=0;
				}
				section_1_13(inf.cli,dim); break;
			case 14:
				section_1_14(inf.des); break;
			case 15:
				section_1_15(inf.des); break;
			case 16:
				section_1_16(inf.des); break;
			case 17:
				section_1_17(inf.des); break;
			case 18:
				section_1_18(inf.des); break;
			case 19:
				section_1_19(inf.des); break;
			case 20:
				section_1_20(inf.cli); break;
			case 21:
				section_1_21(inf.cli); break;
			case 22:
				section_1_22(inf.cli); break;
			case 23:
				section_1_23(inf.des); break;
			case 24:
				section_1_24(inf.des); break;
			case 25:
				section_1_25(inf.dev); break;
			case 26:
				section_1_26(inf.dev); break;
			case 27:
				section_1_27(inf.dev); break;
			case 28:
				section_1_28(inf.dev); break;
			case 29:
				section_1_29(inf.dev); break;
			case 30:
				if(!inf.cli.number_text)
				{
					inf.cli.text_free_text=(char*)FreeWithCare(inf.cli.text_free_text);
					dim=0;
				}
				section_1_30(inf.cli,dim); break;
			case 31:
				section_1_31(inf.dev); break;
			case 32:
				if(!inf.cli.number_hystory)
				{
					inf.cli.medical_hystory=NULL;
					dim=0;
				}
				section_1_32(inf.cli,dim,inf.des.acquiring.protocol_revision_number); break;
			case 33:
				section_1_33(inf.dev); break;
			case 34:
				section_1_34(inf.dev); break;
			case 35:
				if(!inf.cli.number_free_hystory)
				{
					inf.cli.free_medical_hystory=NULL;
					inf.cli.text_free_medical_hystory=(char*)FreeWithCare(inf.cli.text_free_medical_hystory);
					dim=0;
				}
				section_1_35(inf.cli,dim); break;
			case 255:
				section_1_255(); break;
			default:
				section_1_();
				break;
		}//end switch
	}//end do
	while((tag!=255) && (_COUNT_BYTE<num));
	if ((inf.des.analyzing.protocol_revision_number==0) && (version>0))
		inf.des.analyzing.protocol_revision_number=version;  // by E.C. may 2004 CARDIOLINE 1.0
}//end section_1

void section_1_0(demographic &ana)
// section 1 tag 0
{
	uint16_t dim;

	ReadByte(dim);
	ana.last_name=ReadString(ana.last_name,dim);
}//end section_1_0

void section_1_1(demographic &ana)
// section 1 tag 1
{
	uint16_t dim;

	ReadByte(dim);
	ana.first_name=ReadString(ana.first_name,dim);
}//end section_1_1

void section_1_2(demographic &ana)
//section 1 tag 2
{
	uint16_t dim;

	ReadByte(dim);
	ana.ID=ReadString(ana.ID,dim);
}//end section_1_2

void section_1_3(demographic &ana)
// section 1 tag 3
{
	uint16_t dim;

	ReadByte(dim);
	ana.second_last_name=ReadString(ana.second_last_name,dim);
}//end section_1_3

void section_1_4(demographic &ana)
// section 1 tag 4
{
	uint16_t dim;

	ReadByte(dim);
	ReadByte(ana.age.value);
	ReadByte(ana.age.unit);
	if(ana.age.unit>5)
		ana.age.unit=0;
}//end section_1_4

void section_1_5(demographic &ana)
// section 1 tag 5
{
	uint16_t dim;

	uint8_t m, g;
	uint16_t a;

	ReadByte(dim);
	ReadByte(a);
	ReadByte(m);
	ReadByte(g);

        struct tm tmf;                      // by E.C. feb 2006
        tmf.tm_year = a - 1900;
        tmf.tm_mon  = m - 1;
        tmf.tm_mday = g;
        tmf.tm_hour = 0;
        tmf.tm_min  = 0;
        tmf.tm_sec  = 0;
        tmf.tm_isdst = 0;
        ana.date_birth2 = mktime(&tmf);    // store date in native format

}//end section_1_5

void section_1_6(demographic &ana)
// section 1 tag 6
{
	uint16_t dim;

	ReadByte(dim);
	ReadByte(ana.height.value);
	ReadByte(ana.height.unit);
	if(ana.height.unit>3)
		ana.height.unit=0;
}//end section_1_6

void section_1_7(demographic &ana)
// section 1 tag 7
{
	uint16_t dim;

	ReadByte(dim);
	ReadByte(ana.weight.value);
	ReadByte(ana.weight.unit);
	if(ana.weight.unit>4)
		ana.weight.unit=0;
}//end section_1_7

void section_1_8(demographic &ana)
// section 1 tag 8
{
	uint16_t dim;

	ReadByte(dim);
	ReadByte(ana.sex);
	if(ana.sex>2)
		ana.sex=3;
}//end section_1_8

void section_1_9(demographic &ana)
// section 1 tag 9
{
	uint16_t dim;

	ReadByte(dim);
	ReadByte(ana.race);
	if(ana.race>3)
		ana.race=0;
}//end section_1_9

void section_1_10(clinic &cli, uint16_t &dim)
// section 1 tag 10
{
	uint16_t val;
	uint8_t code_;
	char *temp_string=NULL, *pos_char;
	int16_t pos;

	ReadByte(val);
	/*
		this tag may have more instances; each instance have a: table code (1 byte), class code (1 byte)
		drug code (1 byte), text description of the drug (at least 1 byte with NULL).
	*/
	if(val)
	{
		if((cli.drug=(info_drug*)realloc(cli.drug,sizeof(info_drug)*(cli.number_drug+1)))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
		}
		ReadByte(cli.drug[cli.number_drug].table);
		ReadByte(code_);
		if(!cli.drug[cli.number_drug].table)
		{
			pos=Look(class_drug,0,15,code_);
			if(pos<=0)
				cli.drug[cli.number_drug].classes=0;
			else
				cli.drug[cli.number_drug].classes=pos;
		}
		else
			cli.drug[cli.number_drug].classes=code_;
		ReadByte(cli.drug[cli.number_drug].drug_code );
		if(!cli.drug[cli.number_drug].table)
		{
			code_=cli.drug[cli.number_drug].drug_code +256*cli.drug[cli.number_drug].classes;
			pos=Look(class_drug,16,88,code_);
			if(pos<0)
				pos=0;
			cli.drug[cli.number_drug].drug_code =pos;
		}
		cli.drug[cli.number_drug].length=val-3;        //string length + NULL
		if(cli.drug[cli.number_drug].length)
		{
			temp_string=ReadString(temp_string,cli.drug[cli.number_drug].length);
			strcat(temp_string,STR_END);
			dim+=strlen(temp_string);
			if((cli.text_drug=(char*)realloc(cli.text_drug,sizeof(char)*(dim+1)))==NULL)
			{
				B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
				B4C_ERRMSG = "SCP-DECODE: Not enough memory";
				return; 
			}
			pos_char=cli.text_drug;
			pos_char+=dim-strlen(temp_string);
			strcpy(pos_char,temp_string);
			free(temp_string);
		}
		cli.number_drug++;
	}
}//end section_1_10

void section_1_11(demographic &ana)
// section 1 tag 11
{
	uint16_t dim;

	ReadByte(dim);
	if(dim)
		ReadByte(ana.systolic_pressure);
	else
		ana.systolic_pressure=0;
}//end section_1_11

void section_1_12(demographic &ana)
// section 1 tag 12
{
	uint16_t dim;

	ReadByte(dim);
	if(dim)
		ReadByte(ana.diastolic_pressure);
	else
		ana.diastolic_pressure=0;
}//end section_1_12

void section_1_13(clinic &cli, uint16_t &dim)
// section 1 tag 13
{
	uint16_t val;
	char *temp_string=NULL, *pos_char;

	ReadByte(val);
	if(val)
	{
		if((cli.diagnose=(numeric*)realloc(cli.diagnose,sizeof(numeric)*(cli.number_diagnose+1)))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
		}
		cli.diagnose[cli.number_diagnose].unit=cli.number_diagnose+1;
		cli.diagnose[cli.number_diagnose].value=val;
		temp_string=ReadString(temp_string,cli.diagnose[cli.number_diagnose].value);
		strcat(temp_string,STR_END);
		dim+=strlen(temp_string);
		if((cli.text_diagnose=(char*)realloc(cli.text_diagnose,dim+1))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
		}
		pos_char=cli.text_diagnose;
		pos_char+=dim-strlen(temp_string);
		strcpy(pos_char,temp_string);
		free(temp_string);
		cli.number_diagnose++;
	}
}//end section_1_13

void section_1_14(descriptive &des)
// section 1 tag 14
{
	uint16_t dim, dim_to_skip;
	uint8_t i, mask, code_;
	int16_t pos;
	//fpos_t filepos, filepos_iniz;
	long filepos, filepos_iniz;

	ReadByte(dim);
	filepos = iftell(in); //FGETPOS(in,&filepos);
	//FGETPOS(in,&filepos_iniz);    // by E.C. may 2004 ESAOTE    save to reposition at the end of this section
	filepos_iniz=filepos;
	dim_to_skip=dim;
	dim+=filepos COMPAT;
	ReadByte(des.acquiring.institution_number);
	ReadByte(des.acquiring.department_number);
	ReadByte(des.acquiring.ID);
	ReadByte(des.acquiring.type);
	if(des.acquiring.type>1)
		des.acquiring.type=2;
	ReadByte(des.acquiring.manifacturer);
	if(des.acquiring.manifacturer>20 && des.acquiring.manifacturer!=255)
		des.acquiring.manifacturer=0;
	des.acquiring.model_description=ReadString(des.acquiring.model_description,6);
	ReadByte(des.acquiring.protocol_revision_number);
	ReadByte(des.acquiring.category);
	pos=Look(compatibility,0,3,des.acquiring.category);
	if(pos<0)
		pos=4;
	des.acquiring.category=pos;
	ReadByte(code_);
	if(code_<128U)
		pos=0;
	else if((code_<192U) && (code_>=128U))
		pos=1;
	else
	{
		pos=Look(language_code,2,15,code_);
		if(pos<0)
			pos=16;
	}
	des.acquiring.language=pos;
	ReadByte(code_);
	mask=0x10;
	for(i=0;i<4;i++)
	{
		if(code_&mask)
			des.acquiring.capability[i]=i+4;
		else
			des.acquiring.capability[i]=i;
		mask<<=1;
	}
	ReadByte(des.acquiring.AC);
	if(des.acquiring.AC>2)
		des.acquiring.AC=0;
	Skip(16);
	des.acquiring.analysing_program_revision_number=(char*)FreeWithCare(des.acquiring.analysing_program_revision_number);
	des.acquiring.serial_number_device=(char*)FreeWithCare(des.acquiring.serial_number_device);
	des.acquiring.device_system_software=(char*)FreeWithCare(des.acquiring.device_system_software);
	des.acquiring.device_SCP_implementation_software=(char*)FreeWithCare(des.acquiring.device_SCP_implementation_software);
	des.acquiring.manifacturer_trade_name=(char*)FreeWithCare(des.acquiring.manifacturer_trade_name);
	ReadByte(i);
	if(!i)
		des.acquiring.analysing_program_revision_number=(char*)FreeWithCare(des.acquiring.analysing_program_revision_number);
	else
		des.acquiring.analysing_program_revision_number=ReadString(des.acquiring.analysing_program_revision_number,i);

	filepos = iftell(in); //FGETPOS(in,&filepos);
	des.acquiring.serial_number_device=FindString(des.acquiring.serial_number_device,dim-filepos COMPAT);
	if ((des.acquiring.protocol_revision_number==10) || (des.acquiring.protocol_revision_number==11))
													 // by E.C. may 2004 CARDIOLINE 1.0 & ESAOTE 1.1
		ifseek(in,filepos_iniz COMPAT +dim_to_skip,0);   //  reposition file pointer
	else {
		filepos = iftell(in); //FGETPOS(in,&filepos);
		des.acquiring.device_system_software=FindString(des.acquiring.device_system_software,dim-filepos COMPAT);
		filepos = iftell(in); //FGETPOS(in,&filepos);
		des.acquiring.device_SCP_implementation_software=FindString(des.acquiring.device_SCP_implementation_software,dim-filepos COMPAT);
		filepos = iftell(in); //FGETPOS(in,&filepos);
		des.acquiring.manifacturer_trade_name=FindString(des.acquiring.manifacturer_trade_name,dim-filepos COMPAT);
	}
}//end section_1_14

void section_1_15(descriptive &des)
// section 1 tag 15
{
	uint16_t dim;
	uint8_t i, mask, code_;
	int16_t pos;
	//fpos_t filepos;
	long filepos;

	ReadByte(dim);
	filepos = iftell(in); //FGETPOS(in,&filepos);
	dim+=filepos COMPAT;
	ReadByte(des.analyzing.institution_number);
	ReadByte(des.analyzing.department_number);
	ReadByte(des.analyzing.ID);
	ReadByte(des.analyzing.type);
	if(des.analyzing.type>1)
		des.analyzing.type=2;
	ReadByte(des.analyzing.manifacturer);
	if(des.analyzing.manifacturer>20 && des.analyzing.manifacturer!=255)
		des.analyzing.manifacturer=0;
	des.analyzing.model_description=ReadString(des.analyzing.model_description,6);
	ReadByte(des.analyzing.protocol_revision_number);
	ReadByte(des.analyzing.category);
	pos=Look(compatibility,0,3,des.analyzing.category);
	if(pos<0)
		pos=4;
	des.analyzing.category=pos;
	ReadByte(code_);
	if(code_<128U)
		pos=0;
	else if((code_<192U) && (code_>=128U))
		pos=1;
	else
	{
		pos=Look(language_code,2,15,code_);
		if(pos<0)
			pos=16;
	}
	des.analyzing.language=pos;
	ReadByte(code_);
	mask=0x10;
	for(i=0;i<4;i++)
	{
		if(code_&mask)
			des.analyzing.capability[i]=i+4;
		else
			des.analyzing.capability[i]=i;
		mask<<=1;
	}
	ReadByte(des.analyzing.AC);
	if(des.analyzing.AC>2)
		des.analyzing.AC=0;
	Skip(16);
	des.analyzing.analysing_program_revision_number=(char*)FreeWithCare(des.analyzing.analysing_program_revision_number);
	des.analyzing.serial_number_device=(char*)FreeWithCare(des.analyzing.serial_number_device);
	des.analyzing.device_system_software=(char*)FreeWithCare(des.analyzing.device_system_software);
	des.analyzing.device_SCP_implementation_software=(char*)FreeWithCare(des.analyzing.device_SCP_implementation_software);
	des.analyzing.manifacturer_trade_name=(char*)FreeWithCare(des.analyzing.manifacturer_trade_name);

	ReadByte(i);
	if(!i)
		des.analyzing.analysing_program_revision_number=(char*)FreeWithCare(des.analyzing.analysing_program_revision_number);
	else
		des.analyzing.analysing_program_revision_number=ReadString(des.analyzing.analysing_program_revision_number,i);

	filepos = iftell(in); //FGETPOS(in,&filepos);
	des.analyzing.serial_number_device=FindString(des.analyzing.serial_number_device,dim-filepos COMPAT);
	filepos = iftell(in); //FGETPOS(in,&filepos);
	des.analyzing.device_system_software=FindString(des.analyzing.device_system_software,dim-filepos COMPAT);
	filepos = iftell(in); //FGETPOS(in,&filepos);
	des.analyzing.device_SCP_implementation_software=FindString(des.analyzing.device_SCP_implementation_software,dim-filepos COMPAT);
	filepos = iftell(in); //FGETPOS(in,&filepos);
	des.analyzing.manifacturer_trade_name=FindString(des.analyzing.manifacturer_trade_name,dim-filepos COMPAT);
}//end section_1_15

void section_1_16(descriptive &des)
// section 1 tag 16
{
	uint16_t dim;

	ReadByte(dim);
	des.acquiring_institution=ReadString(des.acquiring_institution,dim);
}//end section_1_16

void section_1_17(descriptive &des)
// section 1 tag 17
{
	uint16_t dim;

	ReadByte(dim);
	des.analyzing_institution=ReadString(des.analyzing_institution,dim);
}//end section_1_17

void section_1_18(descriptive &des)
// section 1 tag 18
{
	uint16_t dim;

	ReadByte(dim);
	des.acquiring_department=ReadString(des.acquiring_department,dim);
}//end section_1_18

void section_1_19(descriptive &des)
// section 1 tag 19
{
	uint16_t dim;

	ReadByte(dim);
	des.analyzing_department=ReadString(des.analyzing_department,dim);
}//end section_1_19

void section_1_20(clinic &cli)
// section 1 tag 20
{
	uint16_t dim;

	ReadByte(dim);
	cli.referring_physician=ReadString(cli.referring_physician,dim);
}//end section_1_20

void section_1_21(clinic &cli)
// section 1 tag 21
{
	uint16_t dim;

	ReadByte(dim);
	cli.latest_confirming_physician=ReadString(cli.latest_confirming_physician,dim);
}//end section_1_21

void section_1_22(clinic &cli)
// section 1 tag 22
{
	uint16_t dim;

	ReadByte(dim);
	cli.technician_description=ReadString(cli.technician_description,dim);
}//end section_1_22

void section_1_23(descriptive &des)
// section 1 tag 23
{
	uint16_t dim;

	ReadByte(dim);
	des.room=ReadString(des.room,dim);
}//end section_1_23

void section_1_24(descriptive &des)
// section 1 tag 24
{
	uint16_t dim;

	ReadByte(dim);
	ReadByte(des.stat_code);
}//end section_1_24

void section_1_25(device &dev)
// section 1 tag 25
{
	uint16_t dim;
	uint8_t m, g;
	uint16_t a;

	ReadByte(dim);
	ReadByte(a);
	ReadByte(m);
	ReadByte(g);

        struct tm tmf;                      // by E.C. feb 2006
        tmf.tm_year = a - 1900;
        tmf.tm_mon  = m - 1;
        tmf.tm_mday = g;
        tmf.tm_hour = 0;
        tmf.tm_min  = 0;
        tmf.tm_sec  = 0;
        tmf.tm_isdst = 0;
        dev.date_acquisition2 = mktime(&tmf);    // store date in native format

}//end section_1_25

void section_1_26(device &dev)
// section 1 tag 26
{
	uint16_t dim;
	uint8_t h, m, s;

	ReadByte(dim);
	ReadByte(h);
	ReadByte(m);
	ReadByte(s);
	
        dev.time_acquisition2 = (time_t) (s + m*(60 + h*24));   // by E.C. feb 2006 time in seconds
}//end section_1_26

void section_1_27(device &dev)
// section 1 tag 27
{
	uint16_t dim;

	ReadByte(dim);
	ReadByte(dev.baseline_filter);
}//end section_1_27

void section_1_28(device &dev)
// section 1 tag 28
{
	uint16_t dim;

	ReadByte(dim);
	ReadByte(dev.lowpass_filter);
}//end section_1_28

void section_1_29(device &dev)
// section 1 tag 29
{
	uint16_t dim;
	uint8_t mask=0x1, val, i, max=4, ris=0;

	ReadByte(dim);
	ReadByte(val);
	for(i=0;i<max;i++)
	{
		if(val&mask)
			dev.other_filter[i]=i+1;
		else
			dev.other_filter[i]=0;
		ris|=val&mask;
		mask<<=1;
	}
	int w=dim;
	while (--w) ReadByte(val);        // by E.C. may 2004 CARDIOLINE 1.0
}//end section_1_29

void section_1_30(clinic &cli, uint16_t &dim)
// section 1 tag 30
{
	uint16_t val;
	char *temp_string=0, *pos_char;

	ReadByte(val);
	if(val)
	{
		if((cli.free_text=(numeric*)realloc(cli.free_text,sizeof(numeric)*(cli.number_text+1)))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
		}
		cli.free_text[cli.number_text].unit=cli.number_text+1;
		cli.free_text[cli.number_text].value=val;
		temp_string=ReadString(temp_string,cli.free_text[cli.number_text].value);
		strcat(temp_string,STR_END);
		dim+=strlen(temp_string);
		if((cli.text_free_text=(char*)realloc(cli.text_free_text,dim+1))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
		}
		pos_char=cli.text_free_text;
		pos_char+=dim-strlen(temp_string);
		strcpy(pos_char,temp_string);
		free(temp_string);
		cli.number_text++;
	}
}//end section_1_30

void section_1_31(device &dev)
// section 1 tag 31
{
	uint16_t dim;

	ReadByte(dim);
	if(dim)
		dev.sequence_number=ReadString(dev.sequence_number,dim);
	else{
		dev.sequence_number=(char*)FreeWithCare(dev.sequence_number);
		dev.sequence_number=STR_NULL;
	}
}//end section_1_31

void section_1_32(clinic &cli, uint16_t &dim, int8_t version)
// section 1 tag 32
{
	uint16_t val;
	uint8_t pos;
	int16_t ris;

	ReadByte(val);
	if(val)
	{
		dim+=val;
		//first byte designates the Diagnostic Code Table which is applied; if 0 then the codes used are documented in the standard book.
		// second byte is the code
		if((cli.medical_hystory=(numeric*)realloc(cli.medical_hystory,sizeof(numeric)*(cli.number_hystory+1)))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
		}
		ReadByte(pos);
		cli.medical_hystory[cli.number_hystory].value=pos;
		if (version != 10) {
			ReadByte(pos);                 // 1 byte vers. 1.0; 2 byte vers. 2.0 by E.C. may 2004
			ris=Look(_hystory,0,26,pos);
			if(ris<0)
				pos=26;
			else
				pos=ris;
			cli.medical_hystory[cli.number_hystory].unit=pos;
		}
		cli.number_hystory++;
	}
}//end section_1_32

void section_1_33(device &dev)
// section 1 tag 33
{
	uint16_t dim;
	uint8_t pos;

	ReadByte(dim);
	ReadByte(pos);
	if(pos>6)
		pos=0;
	dev.electrode_configuration.value=pos;

	ReadByte(pos);
	if(pos>6)
		pos=0;
	dev.electrode_configuration.unit=pos;
}//end section_1_33

void section_1_34(device &dev)
// section 1 tag 34
{
	uint16_t dim;

	ReadByte(dim);
	ReadByte(dev.TZ.offset); //complemented if negative
	ReadByte(dev.TZ.index);
	if(dim-4)
		dev.TZ.description = FindString((char*)dev.TZ.description,dim-4);
	else{
		dev.TZ.description = (const char*)realloc((char*)dev.TZ.description,4);
		strcpy((char*)dev.TZ.description,"-");
	}
}//end section_1_34

void section_1_35(clinic &cli, uint16_t &dim)
// section 1 tag 35
{
	uint16_t val;
	char *temp_string=NULL, *pos_char;

	ReadByte(val);
	if(val)
	{
		if((cli.free_medical_hystory=(numeric*)realloc(cli.free_medical_hystory,sizeof(numeric)*(cli.number_free_hystory+1)))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
		}
		cli.free_medical_hystory[cli.number_free_hystory].unit=cli.number_free_hystory+1;
		cli.free_medical_hystory[cli.number_free_hystory].value=val;
		temp_string=ReadString(temp_string,cli.free_medical_hystory[cli.number_free_hystory].value);
		strcat(temp_string,STR_END);
		dim+=strlen(temp_string);
		if((cli.text_free_medical_hystory=(char*)realloc(cli.text_free_medical_hystory,dim+1))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
		}
		pos_char=cli.text_free_medical_hystory;
		pos_char+=dim-strlen(temp_string);
		strcpy(pos_char,temp_string);
		free(temp_string);
		cli.number_free_hystory++;
	}
}//end section_1_35

void section_1_()
// section 1 tag 36..254 are manufacturer specifics and are not utilized
{
	uint16_t dim;

	ReadByte(dim);
	Skip(dim);
}//end section_1_

void section_1_255()
// section 1 tag 255
{
	uint16_t dim;

	ReadByte(dim);
}//end section_1_255
#endif 

//______________________________________________________________________________
//                              section 2
//______________________________________________________________________________

void section_2(pointer_section info_sections,DATA_DECODE &data)
//build Huffman tables if included in the file; if none then use del default one
//cannot read the dummy Huffman table
{
	uint16_t nt, i, j, ns=0, pos, dim;
	//fpos_t filepos;
	long filepos;
	int8_t version;

	_COUNT_BYTE=info_sections.index;
	ifseek(in,info_sections.index-1,0);
	ID_section(info_sections.index, version);
	dim=info_sections.length-16;

	ReadByte(nt);
	if(nt!=19999U)
	{
		if((data.flag_Huffman=(uint16_t*)mymalloc(sizeof(uint16_t)*(nt+1)))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
		}
		data.flag_Huffman[0]=nt;
		filepos = iftell(in); //FGETPOS(in,&filepos);
		for(i=1;i<=data.flag_Huffman[0];i++)
		{
			ReadByte(data.flag_Huffman[i]);
			ns+=data.flag_Huffman[i];
			Skip(9*data.flag_Huffman[i]);
		}
		ifseek(in,filepos COMPAT,0);
		if((ns*9)>dim || !ns)
		{
			B4C_ERRNUM = B4C_UNSPECIFIC_ERROR;
			B4C_ERRMSG = "SCP-DECODE: Cannot read data";
			return; 
		}
		if(ns!=0 && (data.t_Huffman=(table_H*)mymalloc(sizeof(table_H)*ns))==NULL)         //array of 5 columns and ns rows
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
		}
		pos=0;
		for(j=0;j<data.flag_Huffman[0];j++)
		{
			Skip(2);
			for(i=0;i<data.flag_Huffman[j+1];i++)
			{
				ReadByte(data.t_Huffman[pos+i].bit_prefix);
				ReadByte(data.t_Huffman[pos+i].bit_code);
				ReadByte(data.t_Huffman[pos+i].TMS);
				ReadByte(data.t_Huffman[pos+i].base_value);
				ReadByte(data.t_Huffman[pos+i].base_code);
			}
			pos+=data.flag_Huffman[j+1]*9;         // by E.C. may 2004
		}
	}
	else
	{
		if((data.flag_Huffman=(uint16_t*)mymalloc(sizeof(uint16_t)*2))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
		}
		data.flag_Huffman[0]=1;
		data.flag_Huffman[1]=19;      //number of rows of the default Huffman table
		if((data.t_Huffman=(table_H*)mymalloc(sizeof(table_H)*data.flag_Huffman[1]))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
		}
		InitHuffman(data.t_Huffman);
	}
}//end section_2

//______________________________________________________________________________
//                              section 3
//______________________________________________________________________________

void section_3(pointer_section info_sections,DATA_DECODE &data, int8_t version)
{
	uint8_t val, mask=0x1, i;
	int8_t version_loc;

	_COUNT_BYTE=info_sections.index;
	ifseek(in,info_sections.index-1,0);
	ID_section(info_sections.index, version_loc);

	ReadByte(data.flag_lead.number);
	ReadByte(val);
	if(val&mask)
		data.flag_lead.subtraction=1;
	else
		data.flag_lead.subtraction=0;
	mask<<=2;
	if(val&mask)
		data.flag_lead.all_simultaneously=1;
	else
		data.flag_lead.all_simultaneously=0;
	data.flag_lead.number_simultaneously=(val&0xF8)>>3;

	if (version==11)                        // by E.C. may 2004 ESAOTE
		data.flag_lead.number_simultaneously=8;

	if(data.flag_lead.number!=0 && (data.data_lead=(lead*)mymalloc(sizeof(lead)*data.flag_lead.number))==NULL)
	{
		B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
		B4C_ERRMSG = "SCP-DECODE: Not enough memory";
		return; 
	}
	for(i=0;i<data.flag_lead.number;i++)
	{
		ReadByte(data.data_lead[i].start);
		ReadByte(data.data_lead[i].end);
		ReadByte(data.data_lead[i].ID);
		if(data.data_lead[i].ID>85)
			data.data_lead[i].ID=0;
	}
}//end section_3

//______________________________________________________________________________
//                              section 4
//______________________________________________________________________________

void section_4(pointer_section info_sections,DATA_DECODE &data,int8_t version)
{
	uint16_t i;
	int8_t version_loc;

	_COUNT_BYTE=info_sections.index;
	ifseek(in,info_sections.index-1,0);
	ID_section(info_sections.index, version_loc);

	ReadByte(data.flag_BdR0.length);
	ReadByte(data.flag_BdR0.fcM);
	ReadByte(data.flag_Res.number);
	if(data.flag_Res.bimodal || data.flag_lead.subtraction)     // by E.C. may 2004
	{
		if(data.flag_Res.number!=0 && (data.data_subtraction=(Subtraction_Zone*)mymalloc(sizeof(Subtraction_Zone)*data.flag_Res.number))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
		}
		for(i=0;i<data.flag_Res.number;i++)
		{
			ReadByte(data.data_subtraction[i].beat_type);
			ReadByte(data.data_subtraction[i].SB);
			ReadByte(data.data_subtraction[i].fc);
			ReadByte(data.data_subtraction[i].SE);
		}//end for
	}//end if
	if(data.flag_Res.bimodal || data.flag_lead.subtraction)     // by E.C. may 2004
	{
		if(data.flag_Res.number!=0 && (data.data_protected=(Protected_Area*)mymalloc(sizeof(Protected_Area)*data.flag_Res.number))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
		}
		for(i=0;i<data.flag_Res.number;i++)
			if ((version==10) || (version==11)) {               // by E.C. may 2004 SCP 1.0 no data 
				data.data_protected[i].QB=data.data_subtraction[i].SB;
				data.data_protected[i].QE=data.data_subtraction[i].SE;
			} else {
				ReadByte(data.data_protected[i].QB);
				ReadByte(data.data_protected[i].QE);
			}//end if/for
	}//end if
}//end section_4

//______________________________________________________________________________
//                              section 5
//______________________________________________________________________________

bool section_5(pointer_section info_sections,DATA_DECODE &data, bool sez2)
{
	uint16_t i;
	uint32_t t, dim;
	uint16_t value;
	int8_t version;

	_COUNT_BYTE=info_sections.index;
	ifseek(in,info_sections.index-1,0);
	ID_section(info_sections.index, version);

	ReadByte(data.flag_BdR0.AVM);
	ReadByte(data.flag_BdR0.STM);
	ReadByte(data.flag_BdR0.encoding);
	if(data.flag_BdR0.encoding>2)
		data.flag_BdR0.encoding=0;
	Skip(1);
	if(data.flag_lead.number!=0 && (data.length_BdR0=(uint16_t*)mymalloc(sizeof(uint16_t)*data.flag_lead.number))==NULL)
	{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return false; 
	}
	dim=0;
	for(i=0;i<data.flag_lead.number;i++)
	{
		ReadByte(data.length_BdR0[i]);     //number of samples (2 bytes each) for each lead
		dim+=data.length_BdR0[i];
	}
        if (data.flag_BdR0.length==0) return false;      // by E.C. 12/09/2007
	if(sez2)
	{
		data.flag_BdR0.number_samples=(uint32_t)data.flag_BdR0.length*1000L/(uint32_t)data.flag_BdR0.STM;           //number di campioni per elettrodo
		dim*=sizeof(uint8_t);
		if(dim!=0 && (data.samples_BdR0=(uint8_t*)mymalloc(dim))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return false; 
		}
		ifread(data.samples_BdR0,sizeof(uint8_t),dim,in);
	}
	else
	{
		data.flag_BdR0.number_samples=dim/(sizeof(int16_t)*data.flag_lead.number);       //number of samples per lead
		dim>>=1;
		dim*=sizeof(int32_t);
		if(dim!=0 && (data.Median=(int32_t*)mymalloc(dim))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return false; 
		}
		dim/=sizeof(int32_t);
		for(t=0;t<dim;t++)
		{
			ReadByte(value);
			data.Median[t]=value;
			if(value>0x7FFF)
				data.Median[t]|=0xFFFF0000;
		}
	}
        return true;
}//end section_5

//______________________________________________________________________________
//                              section 6
//______________________________________________________________________________

void section_6(pointer_section info_sections,DATA_DECODE &data, bool sez2)
{
	uint16_t i;
	uint32_t t, dim;
	uint16_t value;
	int8_t version;

	_COUNT_BYTE=info_sections.index;
	ifseek(in,info_sections.index-1,0);
	ID_section(info_sections.index, version);

	ReadByte(data.flag_Res.AVM);
	ReadByte(data.flag_Res.STM);
	ReadByte(data.flag_Res.encoding);
	if(data.flag_Res.encoding>2)
		data.flag_Res.encoding=0;
	Skip(1);
	if(data.flag_lead.number!=0 && (data.length_Res=(uint16_t*)mymalloc(sizeof(uint16_t)*data.flag_lead.number))==NULL)
	{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
	}
	dim=0;
	for(i=0;i<data.flag_lead.number;i++)
	{
		ReadByte(data.length_Res[i]);     //number of bytes for each lead
		dim+=data.length_Res[i];
	}
	if(sez2)
	{
		data.flag_Res.number_samples=data.data_lead[1].end-data.data_lead[1].start+1; //number of samples after interpolation
		dim*=sizeof(uint8_t);
		if(dim!=0 && (data.samples_Res=(uint8_t*)mymalloc(dim))==NULL)
		{
			B4C_ERRNUM = B4C_INSUFFICIENT_MEMORY;
			B4C_ERRMSG = "SCP-DECODE: Not enough memory";
			return; 
		}
		ifread(data.samples_Res,sizeof(uint8_t),dim,in);
	}
	else
	{
		data.flag_Res.number_samples=dim/(sizeof(int16_t)*data.flag_lead.number);       //number of samples per lead
		dim>>=1;
		dim*=sizeof(int32_t);
		if(dim!=0 && (data.Residual=(int32_t*)mymalloc(dim))==NULL)
		{
			fprintf(stderr,"Not enough memory");  // no, exit //
			exit(2);
		}
		dim/=sizeof(int32_t);
		for(t=0;t<dim;t++)
		{
			ReadByte(value);
			data.Residual[t]=value;
			if(value>0x7FFF)
				data.Residual[t]|=0xFFFF0000;
		}
	}
}//end section_6

#ifdef WITH_OBSOLETE_PARTS

//______________________________________________________________________________
//                              section 7
//______________________________________________________________________________

void section_7(pointer_section info_sections ,DATA_RECORD &data, int8_t version)
{
	uint16_t i, j, dim;
	uint8_t lung;
	//fpos_t filepos;
	long filepos;
	int8_t version_loc;
	uint32_t length_eval;

	_COUNT_BYTE=info_sections.index;
	ifseek(in,info_sections.index-1,0);
	ID_section(info_sections.index, version_loc);

	ReadByte(data.data_global.number);
	//each value should be checked in _special!
	ReadByte(data.data_global.number_spike);
	if (version==11) ReadByte(data.data_global.number_spike);       // by E.C. may 2004 x ESAOTE!! This is an implementation error, for sure!
	ReadByte(data.data_global.average_RR);
	ReadByte(data.data_global.average_PP);
	if(Look(_special,0,3,data.data_global.number)<0)
	{
		if(data.data_global.number!=0 && (data.data_BdR=(BdR_measurement*)mymalloc(sizeof(BdR_measurement)*data.data_global.number))==NULL)
		{
			fprintf(stderr,"Not enough memory");  // no, exit //
			exit(2);
		}
		for(i=0;i<data.data_global.number;i++)
		{
			ReadByte(data.data_BdR[i].P_onset);
			ReadByte(data.data_BdR[i].P_offset);
			ReadByte(data.data_BdR[i].QRS_onset);
			ReadByte(data.data_BdR[i].QRS_offset);
			ReadByte(data.data_BdR[i].T_offset);
			ReadByte(data.data_BdR[i].P_axis);
			ReadByte(data.data_BdR[i].QRS_axis);
			ReadByte(data.data_BdR[i].T_axis);
		}//end for
	}
	if(Look(_special,0,3,data.data_global.number_spike)<0)
	{
		if(data.data_global.number_spike!=0 && (data.data_spike=(spike*)mymalloc(sizeof(spike)*data.data_global.number_spike))==NULL)
		{
			fprintf(stderr,"Not enough memory");  // no, exit //
			exit(2);
		}
		//spike time is in ms from the start of recording
		//amplitude is in signed uV
		for(i=0;i<data.data_global.number_spike;i++)
		{
			ReadByte(data.data_spike[i].time);
			ReadByte(data.data_spike[i].amplitude);
		}//end for
		for(i=0;i<data.data_global.number_spike;i++)
		{
			ReadByte(data.data_spike[i].type);
			if(data.data_spike[i].type>3)
				data.data_spike[i].type=0;
			ReadByte(data.data_spike[i].source);
			if(data.data_spike[i].source>2)
				data.data_spike[i].source=0;
			ReadByte(data.data_spike[i].index);
			ReadByte(data.data_spike[i].pulse_width);
		}//end for
	}//end if
	if (version<13) {          // by E.C. may 2004    CARDIOLINE & ESAOTE missing!!
		if ((data.data_global.average_RR>0) &&
			(data.data_global.average_RR<10000))
			data.data_global.ventricular_rate=60000.0/data.data_global.average_RR+0.5;
		return;
	}
// Insert by F.C.
	if (version>=13) {
		length_eval = 16 + 6 + data.data_global.number * 16 + data.data_global.number_spike * 4 + data.data_global.number_spike * 6;
		if (length_eval >= info_sections.length)
			return;
	}
// End of F.C. insertion
	ReadByte(data.data_global.number_QRS);
        if (data.data_global.number_QRS==29999) return;    // by E.C.  12/09/2007
	if(Look(_special,0,3,data.data_global.number_QRS)<0)
	{
		filepos = iftell(in); //FGETPOS(in,&filepos);                         //necessary for ESAOTE and CARDIOLINE test files
		dim=info_sections.index+info_sections.length-filepos COMPAT+1;
		if(data.data_global.number_QRS>dim)
		{
			fprintf(stderr,"Error: Cannot extract these data!!!");
			exit(2);                                      //necessary for ESAOTE and CARDIOLINE test files
		} 
		if(data.data_global.number_QRS!=0 && (data.type_BdR=(uint8_t*)mymalloc(sizeof(uint8_t)*data.data_global.number_QRS))==NULL)
		{
			fprintf(stderr,"Not enough memory");  // no, exit //
			exit(2);
		}
		for(i=0;i<data.data_global.number_QRS;i++)
			ReadByte(data.type_BdR[i]);
	}
// Insert by F.C.
	if (version>=13) {
		length_eval += (2 + data.data_global.number_QRS);
		if (length_eval >= info_sections.length)
			return;
	}
// End of F.C. insertion
	ReadByte(data.data_global.ventricular_rate);
	ReadByte(data.data_global.atrial_rate);
	ReadByte(data.data_global.QT_corrected);
	ReadByte(data.data_global.formula_type);
	if(data.data_global.formula_type>2)
		data.data_global.formula_type=0;
	ReadByte(data.data_global.number_tag);
	if(data.data_global.number_tag)
	{
		data.data_global.number_tag-=2;
		data.data_global.number_tag/=7; // tag number 
		//warnig: this calculation is relative to the structure of STANDARD v2.0!
		if(data.data_global.number_tag!=0 && (data.data_additional=(additional_measurement*)mymalloc(sizeof(additional_measurement)*data.data_global.number_tag))==NULL)
		{
			fprintf(stderr,"Not enough memory");  // no, exit //
			exit(2);
		}
		for(i=0;i<data.data_global.number_tag;i++)
		{
			ReadByte(data.data_additional[i].ID);
			if(data.data_additional[i].ID==255)
				break;
			if(data.data_additional[i].ID>3)
				data.data_additional[i].ID=4;
			ReadByte(lung);
			if(lung)
			{
				//warning:(255 is undefined)
				for(j=0;j<5;j++)
					ReadByte(data.data_additional[i].byte[j]);
			}//end if
		}//end for
	}//end if
}//end section_7

//______________________________________________________________________________
//                              section 8
//______________________________________________________________________________

void section_8(pointer_section info_sections,DATA_INFO &data)
{
	uint8_t m, g, h, s, i;
	uint16_t a, dim;
	char *c;
	//fpos_t filepos;
	long filepos;
	int8_t version;

	_COUNT_BYTE=info_sections.index;
	ifseek(in,info_sections.index-1,0);
	ID_section(info_sections.index, version);

	ReadByte(data.flag_report.type);
	if(data.flag_report.type>2)
		data.flag_report.type=3;
	ReadByte(a);
	ReadByte(m);
	ReadByte(g);

	ReadByte(h);
	ReadByte(m);
	ReadByte(s);
	
	struct tm tmf; 
	tmf.tm_year = a; 
	tmf.tm_mon  = m; 
	tmf.tm_mday = g; 
	tmf.tm_hour = h; 
	tmf.tm_min  = m;
	tmf.tm_sec  = s; 

	data.flag_report.date = (char*)mymalloc(18);
	strftime(data.flag_report.date,18,"%d %b %Y", &tmf);

	data.flag_report.time = (char*)mymalloc(18);
	strftime(data.flag_report.date,18,"%H:%M:%S", &tmf);

	ReadByte(data.flag_report.number);
	if(data.flag_report.number)
	{
		filepos = iftell(in); //FGETPOS(in,&filepos);
		if(data.flag_report.number!=0 && (data.text_dim=(numeric*)mymalloc(data.flag_report.number*sizeof(numeric)))==NULL)
		{
			fprintf(stderr,"Not enough memory");  // no, exit //
			exit(2);
		}
		dim=0;
		for(i=0;i<data.flag_report.number;i++)
		{
			//this should start with 1, but I documented 0!!
			ReadByte(data.text_dim[i].unit);   //sequence number
			ReadByte(data.text_dim[i].value);  //length
			dim+=data.text_dim[i].value;
			Skip(data.text_dim[i].value);
		}
		ifseek(in,filepos COMPAT,0);
		if(dim!=0 && (data.text_report=(char*)mymalloc((dim+1)*sizeof(char)))==NULL)
		{
			fprintf(stderr,"Not enough memory");  // no, exit //
			exit(2);
		}
		c=data.text_report;
		for(i=0;i<data.flag_report.number;i++)
		{
			Skip(3);
			char *temp_string=ReadString(NULL,data.text_dim[i].value);
			strcat(temp_string,STR_END);
			strcpy(c,temp_string);
			c+=strlen(temp_string);
			free(temp_string);
		}//end for
	}
}//end section_8

//______________________________________________________________________________
//                              section 10
//______________________________________________________________________________

void section_10(pointer_section info_sections, DATA_RECORD &data, int8_t version)
{
	uint16_t dim, n1, n2, i, j, k, id, val, mask;
	int16_t skip;
	int8_t version_loc;

	_COUNT_BYTE=info_sections.index;
	ifseek(in,info_sections.index-1,0);
	ID_section(info_sections.index, version_loc);

	ReadByte(data.header_lead.number_lead);
	ReadByte(dim);
	if(dim<6)      //no measures
	{
		if (version != 10) {
			fprintf(stderr,"Error: no measures or cannot extract section 10 data!!!");
			return;       // by E.C. 12/09/2007
		}
	}
	n1=(dim>>1)-2;
	if(n1>31)     // for now 33 are defined
		n1=31;       //ignore next bytes
	data.header_lead.number_lead_measurement=n1; //max number of statements by the manufacturer (2 bytes each)
	if(data.header_lead.number_lead)                // by E.C. 17.11.2003   deleted "!" in the if
	{
		if((data.lead_block=(lead_measurement_block*)mymalloc(data.header_lead.number_lead*sizeof(lead_measurement_block)))==NULL)
		{
			fprintf(stderr,"Not enough memory");  // no, exit //
			exit(2);
		}
		for(i=0;i<data.header_lead.number_lead;i++)
		{
			if(data.header_lead.number_lead_measurement)        // by E.C. 17.11.2003   deleted "!" in the if
			{
				ReadByte(id);
				if(id>85)
					id=0;
				ReadByte(dim);
				n2=(dim>>1);
				if(n2>n1)
					skip=(n2-n1)<<1;     // bytes to skip
				else
					skip=0;
				data.lead_block[i].ID=id;
				//warnig: values relative to _SPECIAL
				for(j=1;j<=data.header_lead.number_lead_measurement;j++)
				{
					switch(j)
					{
						case  1:	ReadByte(data.lead_block[i].P_duration);
									break;
						case  2:	ReadByte(data.lead_block[i].PR_interval);
									break;
						case  3:	ReadByte(data.lead_block[i].QRS_duration);
									break;
						case  4:	ReadByte(data.lead_block[i].QT_interval);
									break;
						case  5:	ReadByte(data.lead_block[i].Q_duration);
									break;
						case  6:	ReadByte(data.lead_block[i].R_duration);
									break;
						case  7:	ReadByte(data.lead_block[i].S_duration);
									break;
						case  8:	ReadByte(data.lead_block[i].R1_duration);
									break;
						case  9:	ReadByte(data.lead_block[i].S1_duration);
									break;
						case 10:	ReadByte(data.lead_block[i].Q_amplitude);
									break;
						case 11:	ReadByte(data.lead_block[i].R_amplitude);
									break;
						case 12:	ReadByte(data.lead_block[i].S_amplitude);
									break;
						case 13:	ReadByte(data.lead_block[i].R1_amplitude);
									break;
						case 14:	ReadByte(data.lead_block[i].S1_amplitude);
									break;
						case 15:	ReadByte(data.lead_block[i].J_point_amplitude);
									break;
						case 16:	ReadByte(data.lead_block[i].Pp_amplitude);
									break;
						case 17:	ReadByte(data.lead_block[i].Pm_amplitude);
									break;
						case 18:	ReadByte(data.lead_block[i].Tp_amplitude);
									break;
						case 19:	ReadByte(data.lead_block[i].Tm_amplitude);
									break;
						case 20:	ReadByte(data.lead_block[i].ST_slope);
									break;
						case 21:	ReadByte(data.lead_block[i].P_morphology);
									if(data.lead_block[i].P_morphology)
										data.lead_block[i].P_morphology=0;
									break;
						case 22:	ReadByte(data.lead_block[i].T_morphology);
									if(data.lead_block[i].T_morphology)
										data.lead_block[i].T_morphology=0;
									break;
						case 23:	ReadByte(data.lead_block[i].iso_electric_segment_onset_QRS);
									break;
						case 24:	ReadByte(data.lead_block[i].iso_electric_segment_offset_QRS);
									break;
						case 25:	ReadByte(data.lead_block[i].intrinsicoid_deflection);
									break;
						case 26:	ReadByte(val);
									mask=0x3;
									for(k=0;k<8;k++)
									{
										data.lead_block[i].quality_recording[k]=mask&val;
										// TODO: code has no effect 
										mask<<2;
									}
									break;
						case 27:	ReadByte(data.lead_block[i].ST_amplitude_Jplus20);
									break;
						case 28:	ReadByte(data.lead_block[i].ST_amplitude_Jplus60);
									break;
						case 29:	ReadByte(data.lead_block[i].ST_amplitude_Jplus80);
									break;
						case 30:	ReadByte(data.lead_block[i].ST_amplitude_JplusRR16);
									break;
						case 31:	ReadByte(data.lead_block[i].ST_amplitude_JplusRR8);
									break;
					}//end switch
				}//end for
				if(skip)
					Skip(skip);
			}//end if
		}//end for
	}//end if
}//end section_10

//______________________________________________________________________________
//                              section 11
//______________________________________________________________________________

void section_11(pointer_section info_sections,DATA_INFO &data)
/*
	expressions (ASCII) should be either:
 		1) diagnostic statement_probability_modifiers;
		2) diagnostic statement_probability_modifier_conjunctive term_diagnostic statement_probability_modifier...;
	in the test files I found only 1 diagnostic statement per expression, ending with a NULL.
*/
{
	uint8_t m, g, h, s, i, j;
	uint16_t a, dim;
	char *temp_string=0, *punt, c;
	//fpos_t filepos;
	long filepos;
	int8_t version;

	_COUNT_BYTE=info_sections.index;
	ifseek(in,info_sections.index-1,0);
	ID_section(info_sections.index, version);

	ReadByte(data.flag_statement.type);
	if(data.flag_statement.type>2)
		data.flag_statement.type=3;
	ReadByte(a);
	ReadByte(m);
	ReadByte(g);
	ReadByte(h);
	ReadByte(m);
	ReadByte(s);
	
	struct tm tmf; 
	tmf.tm_year = a; 
	tmf.tm_mon  = m; 
	tmf.tm_mday = g; 
	tmf.tm_hour = h; 
	tmf.tm_min  = m;
	tmf.tm_sec  = s; 

	data.flag_statement.date = (char*)mymalloc(18);
	strftime(data.flag_statement.date,18,"%d %b %Y", &tmf);

	data.flag_statement.time = (char*)mymalloc(18);
	strftime(data.flag_statement.time,18,"%H:%M:%S", &tmf);

	ReadByte(data.flag_statement.number); //number of expressions
	if(!data.flag_statement.number)
	{
		filepos = iftell(in); //FGETPOS(in,&filepos);
		if(data.flag_statement.number!=0 && (data.data_statement=(statement_coded*)mymalloc(data.flag_statement.number*sizeof(statement_coded)))==NULL)
		{
			fprintf(stderr,"Not enough memory");  // no, exit //
			exit(2);
		}
		dim=0;
		for(i=0;i<data.flag_statement.number;i++)
		{
			ReadByte(data.data_statement[i].sequence_number);
			ReadByte(data.data_statement[i].length);
			dim+=data.data_statement[i].length-1;
			ReadByte(data.data_statement[i].type);
			data.data_statement[i].number_field=1;
			if(data.data_statement[i].type==1)
			{
				for(j=1;j<(data.data_statement[i].length-1);j++)
				{
					c=ifgetc(in);
					if(c=='\0')
						++data.data_statement[i].number_field;
				}
			}
			else
				Skip(data.data_statement[i].length-1);
		}
		ifseek(in,filepos COMPAT,0);
		if(dim!=0 && (data.text_statement=(char*)mymalloc(dim))==NULL)
		{
			fprintf(stderr,"Not enough memory");  // no, exit //
			exit(2);
		}
		punt=data.text_statement;
		for(i=0;i<data.flag_statement.number;i++)
		{
			Skip(4);
			if(data.data_statement[i].type==1)
			{
				dim=data.data_statement[i].length;
				for(j=0;j<data.data_statement[i].number_field;j++)
				{
					temp_string=FindString(temp_string,dim);
					strcat(temp_string,STR_END);
					strcpy(punt,temp_string);
					punt+=strlen(temp_string);
					dim-=strlen(temp_string);
					free(temp_string);
				}
			}
			else
			{
				temp_string=ReadString(temp_string,data.data_statement[i].length);
				strcat(temp_string,STR_END);
				strcpy(punt,temp_string);
				punt+=strlen(temp_string);
				free(temp_string);
			}
		}//end for
	}//end if
}//end section_11

#endif 

//______________________________________________________________________________
//                              CALCULATIONS
//______________________________________________________________________________

//______________________________________________________________________________
//                      MULTIPLICATION BY AVM
void Multiply(int32_t *dati, uint32_t num, uint16_t AVM)
{
	uint32_t i;


	for(i=0;i<num;i++)
		dati[i]=dati[i]*AVM;   // AS 2007: scale in nV - no round of errors
}//end Multiply

//______________________________________________________________________________
//                      FIRST AND SECOND DIFFERENCES
template<class t1>
void Differences(int32_t *dati, t1 flag, uint8_t num)
{
	uint8_t i;
	uint16_t j;

	for(i=0;i<num;i++)
		for(j=flag.encoding;j<flag.number_samples;j++)
			if(flag.encoding==1)
				dati[(j+flag.number_samples*i)]+=dati[(j+flag.number_samples*i)-1];
			else
				dati[(j+flag.number_samples*i)]+=(dati[(j+flag.number_samples*i)-1]<<1)-dati[(j+flag.number_samples*i)-2];
}//end Differences

//______________________________________________________________________________
//                              HUFFMAN DECODING
void Tree_Destroy(TREE_NODE *radix)
{
	if (radix!=NULL)
	{
		Tree_Destroy(radix->next_0);
		Tree_Destroy(radix->next_1);
		free(radix);
	}
	return;
}

TREE_NODE *Tree_Create(TREE_NODE *tree, uint16_t n_of_struct, table_H *table, uint16_t pos)
//build a tree
{
	uint8_t i,j;
	uint32_t mask;
	TREE_NODE *temp;

	//build the root
	if((tree=(TREE_NODE *)mymalloc(sizeof(TREE_NODE)))==NULL)
	{
		fprintf(stderr,"Not enough memory");  // no, exit //
		exit(2);
	}
	tree->next_0=NULL;
	tree->next_1=NULL;
	tree->row=-1;	//-1 means no row in the table

	for (j=0;j<n_of_struct;j++)
	{
		temp=tree;
		mask=0x00000001;
		for (i=0;i<table[j+pos].bit_prefix;i++)
		{
			if (table[j+pos].base_code & mask)
			{
				if (temp->next_1==NULL)
				{
					if((temp->next_1=(TREE_NODE *)mymalloc(sizeof(TREE_NODE)))==NULL)
					{
						fprintf(stderr,"Not enough memory");  // no, exit //
						exit(2);
					}
					temp->next_1->next_0=NULL;
					temp->next_1->next_1=NULL;
					temp->next_1->row=-1;
				}
				temp=temp->next_1;
			}//end if
			else
			{
				if (temp->next_0==NULL)
				{
					if((temp->next_0=(TREE_NODE *)mymalloc(sizeof(TREE_NODE)))==NULL)
					{
						fprintf(stderr,"Not enough memory");  // no, exit //
						exit(2);
					}
					temp->next_0->next_0=NULL;
					temp->next_0->next_1=NULL;
					temp->next_0->row=-1;
				}
				temp=temp->next_0;
			}//end else
			mask <<=1;
		}//end for i
		temp->row=j;		//marks the j row in the table
	}//end for j

	return tree;
}//end Tree_Create

uint8_t Input_Bit(uint8_t *raw, uint16_t &pos, uint16_t max, uint8_t &mask, bool &err)
{
	uint8_t value;

	if(pos==max)
	{
		err=1;
		return 0;
	}

	value=raw[pos]&mask;
	mask>>=1;
	if(!mask)
	{
		mask=0x80;
		++pos;
	}

	return( value ? 1 : 0 );
}//end Input_Bit

int16_t Input_Bits(uint8_t *raw, uint16_t &pos, uint16_t max, uint8_t &mask, uint8_t bit_count, bool &err)
{
	uint16_t temp;
	int16_t value;

	if(pos==max)
	{
		err=1;
		return 0;
	}

	temp=1<<(bit_count-1);
	value=0;
	do
	{
		if(raw[pos]&mask)
			value |= temp;
		temp>>=1;
		mask>>=1;
		if(!mask)
		{
			mask=0x80;
			++pos;
			if(pos==max) {
				mask=0;                // by E.C. may 2004
				break;
			}
		}
	} while(temp);
	if(temp)
		err=1;

	if (value & (1<<(bit_count-1)))       // by E.C. may 2004 negative value. extend to the left
		value|=(-1<<bit_count);        // this approach is good for any number of bits

	return value;
}//end InputBits

void decompress(TREE_NODE *tree, uint8_t *raw_in, uint16_t &pos_in, uint16_t max_in, int32_t *raw_out, uint16_t &pos_out, uint16_t &max_out, table_H *table ,uint16_t *flag, uint16_t &pos_tH)
{
	uint16_t i, j;
	uint8_t nbits;
	TREE_NODE *temp;
	uint8_t mask=0x80;
	bool err;
	uint16_t maxIN=max_in+pos_in;

	j=0;
	while(pos_in<maxIN)    //check number of bytes
	{
		temp=tree;
		for (;;)
		{
			err=0;
			if (Input_Bit(raw_in,pos_in,maxIN,mask,err))    // for each bit read follow the tree
				temp=temp->next_1;
			else
				temp=temp->next_0;
			if (temp==NULL)
			{
				fprintf(stderr,"Tree overflow");
				err=1;
				break;
			}
			if (temp->row > -1)          // row found in the table:  exit for
				break;
			if(pos_in==maxIN)
				err=1;
			if(err)
				break;
		}//end for
		if(err)       // check for error conditions and if exit while
		{
			break;            // by E.C. may 2004
		}
		if(table[temp->row+pos_tH].TMS!=1)    // switch to another table
		{
			Tree_Destroy(tree);                  // destroy the tree and rebuild with the new table
			pos_tH=0;
			for(i=1;i<table[temp->row+pos_tH].base_value;i++)
				pos_tH+=flag[i];    // offset of the table
			tree=Tree_Create(tree,flag[table[temp->row+pos_tH].base_value],table,pos_tH);
			continue;
		}//end if table
		else
		{
			nbits=table[temp->row+pos_tH].bit_code-table[temp->row+pos_tH].bit_prefix;
			if (nbits)           // bit of the code != bit of the prefix
			{
				if(pos_in==maxIN)
				{
					err=1;
					break;            // by E.C. may 2004
				}
				raw_out[pos_out]=Input_Bits(raw_in,pos_in,maxIN,mask,nbits,err);  // take value from the stream
				if(err)       // check for error conditions and if exit while
				{
					break;          // by E.C. may 2004
				}
				++pos_out;
			}//end if nbits
			else         // bit of the code = bit of the prefix
			{
				raw_out[pos_out]=table[temp->row+pos_tH].base_value;   // take value from the table
				++pos_out;
			}
		}//end else if table
		++j;
		if (j==max_out) break;        // by E.C. may 2004
	} //end while
	pos_in=maxIN;          // by E.C. 23.02.2004: align for safety
	max_out=j;             //                     flows here anyhow!
	if (max_out>4900) {
		max_out=5000;                           // by E.C. may 2004 ESAOTE
		pos_out=(pos_out+100)/max_out*max_out;  // align pointer
	}
}//end decompress

//data.data_BdR0 , data.length_BdR0 , data.samples_BdR0 , data.flag_BdR0.number_samples , data.flag_lead.number , data.t_Huffman , data.flag_Huffman
//out_data       , length           , in_data           , n_samples                     , n_lead                , t_Huffman      , flag_Huffman
void Huffman(int32_t *out_data, uint16_t *length, uint8_t *in_data, uint16_t &n_samples, uint16_t n_lead, table_H *t_Huffman, uint16_t *flag_Huffman)
{
	TREE_NODE *tree = NULL;
	uint16_t pos_in, pos_out, pos_tH;
	uint8_t i;

	pos_in=0;
	pos_out=0;
	pos_tH=0;
	tree=Tree_Create(tree,flag_Huffman[1],t_Huffman,pos_tH);
	for(i=0;i<n_lead;i++)
	{
		pos_tH=0;
		if(pos_tH)
		{
			Tree_Destroy(tree);
			tree=Tree_Create(tree,flag_Huffman[1],t_Huffman,pos_tH);
		}
		decompress(tree,in_data,pos_in,length[i],out_data,pos_out,n_samples,t_Huffman,flag_Huffman,pos_tH);
	}
	Tree_Destroy(tree);
}//end Huffman

void Decode_Data(pointer_section *section, DATA_DECODE &data, bool &add_filter)
{
	uint32_t t;
	uint32_t dim_B, dim_R, dim_R_, number_samples_;

	int32_t *dati_Res_ = NULL;
	if(!data.flag_Huffman){//Or we will get crash 
		return;
	}
	//Decode the reference beat
	if(section[5].length)
	{
		if(section[2].length && data.flag_Huffman[0])
		{
			dim_B=data.flag_BdR0.number_samples*sizeof(int32_t)*data.flag_lead.number;   //whole number of bytes
			if(dim_B!=0 && (data.Median=(int32_t*)mymalloc(dim_B))==NULL)
			{
				fprintf(stderr,"Not enough memory");  // no, exit //
				exit(2);
			}
			Huffman(data.Median,data.length_BdR0,data.samples_BdR0,data.flag_BdR0.number_samples,data.flag_lead.number,data.t_Huffman,data.flag_Huffman);
			free(data.samples_BdR0);
			data.samples_BdR0=NULL;
		}
		else
		{
			dim_B=data.flag_BdR0.number_samples*sizeof(int16_t)*data.flag_lead.number;   //number of samples of all leads
			//they shuld be equal!!!
		}
		free(data.length_BdR0);
		data.length_BdR0=NULL;
		if(data.flag_BdR0.encoding)
			Differences(data.Median,data.flag_BdR0,data.flag_lead.number);

		Multiply(data.Median,data.flag_BdR0.number_samples*data.flag_lead.number,data.flag_BdR0.AVM);
	}//end if

	//Decode rhythm data
	if(section[6].length)
	{
		dim_R_=data.flag_Res.number_samples*data.flag_lead.number*sizeof(int32_t);	//whole number of bytes
		number_samples_=data.flag_Res.number_samples;                //number of samples per lead after interpolation
		if(dim_R_!=0 && (data.Residual=(int32_t*)mymalloc(dim_R_))==NULL)
		{
			fprintf(stderr,"Not enough memory");  // no, exit //
			exit(2);
		}
		if(section[2].length && data.flag_Huffman[0])
			Huffman(data.Residual,data.length_Res,data.samples_Res,data.flag_Res.number_samples,data.flag_lead.number,data.t_Huffman,data.flag_Huffman);

		dim_R=data.flag_Res.number_samples*sizeof(int32_t)*data.flag_lead.number;   //number of bytes of all leads before interpolation (also, data.flag_Res.number_samples has been changed to the number of samples from HUffman)
		free(data.samples_Res);
		free(data.length_Res);
		data.samples_Res=NULL;
		data.length_Res=NULL;

		if(data.flag_Res.encoding)
		Differences(data.Residual,data.flag_Res,data.flag_lead.number);

		//calculate resampling rate
		if(data.flag_Res.bimodal || data.flag_lead.subtraction)     // by E.C. 19.02.2004  (i.e. to open p120n00)
		{
			if(section[5].length)
				data.flag_Res.decimation_factor=data.flag_Res.STM/data.flag_BdR0.STM;
			else
				data.flag_Res.decimation_factor=0; //no interpolation!!
			//exec interpolation
			if(data.flag_Res.decimation_factor>1)                // by E.C. 19.02.2004  (i.e. to open pd3471)
			{
//dim_R = number of bytes of decimated signal (4 bytes (int32_t) )
//data.flag_Res.number_samples = samples of the decimated signal
//dim_R_ = number of bytes of the reconstructed signal (4 bytes (int32_t) )
//number_samples_ = samples of the reconstructed signal

				data.flag_Res.number_samples=number_samples_;     //number of samples per lead
				number_samples_=dim_R/(sizeof(int32_t)*data.flag_lead.number);
				dim_R=dim_R_;
				dim_R_=(dim_R/sizeof(int32_t))*sizeof(float)*2;

//dim_R_ = number of bytes of decimated signal (4 bytes (int32_t) )
//number_samples_ = samples of the decimated signal
//dim_R = number of bytes of the reconstructed signal (4 bytes (int32_t) )
//data.flag_Res.number_samples = samples of the reconstructed signal

				if(dim_R_!=0 && (dati_Res_=(int32_t*)mymalloc(dim_R_))==NULL)
				{
					fprintf(stderr,"Not enough memory");  // no, exit //
					exit(2);
				}
				Interpolate(dati_Res_,data.Residual,data.flag_lead,data.data_lead,data.flag_Res,data.data_protected,number_samples_);
				DoFilter(data.Residual,dati_Res_,data.flag_Res,data.flag_lead,data.data_lead,data.data_protected,data.data_subtraction);
				free(dati_Res_);
				//dim_R modifies
			}
		}
		Multiply(data.Residual,data.flag_Res.number_samples*data.flag_lead.number,data.flag_Res.AVM);

/* AS 2007-10-23: for some (unknown) reason, sometimes the memory allocation has the wrong size
	doing the memory allocation in sopen_scp_read does it correctly. 
	Number of files with SegFaults is reduced by 3.  
	
		if(dim_R!=0 && (data.Reconstructed=(int32_t*)mymalloc(dim_R))==NULL)
		{
			fprintf(stderr,"Not enough memory");  // no, exit //
			exit(2);
		}
*/
		unsigned dim_RR=dim_R/sizeof(int32_t);       // by E.C. 15.10.2003   This to correct a trivial error
		for(t=0;t<dim_RR;t++)                 // of array overflow!!
			data.Reconstructed[t]=data.Residual[t];   // by E.C. 19.02.2004: first copy rhythm then add the reference beat

		if(section[3].length && section[5].length && data.flag_lead.subtraction)
		{
			DoAdd(data.Reconstructed,data.Residual,data.flag_Res,data.Median,data.flag_BdR0,data.data_subtraction,data.flag_lead,data.data_lead);
			if (add_filter && (data.flag_Res.decimation_factor>1)){                  // by E.C. 25.02.2004
					//  now that the signal is completely reconstructructed, and if decimation was performed
					//  during the compression phase, do an extra low-pass filter outside protected zones !
					//  This shows lower RMS values of about 0.5 - 1.0
					//  If you don't like this extra filtering please uncheck the menu option and reopen
				for(t=0;t<dim_RR;t++)
					data.Residual[t]=data.Reconstructed[t];       // reuse array as temp storage
				Opt_Filter(data.Reconstructed,data.Residual,data.flag_Res,data.flag_lead,data.data_lead,data.data_protected);
				for(t=0;t<dim_RR;t++)
					data.Residual[t]=data.Reconstructed[t];       // reuse array as temp storage
				Opt_Filter(data.Reconstructed,data.Residual,data.flag_Res,data.flag_lead,data.data_lead,data.data_protected);
			}
			else add_filter=false;
		}
		else add_filter=false;
	}
}

//______________________________________________________________________________
//                      sections 3, 4, 5, and 6
//                  dati_Res_ ,data.Residual     ,dim_R   ,data.flag_lead.number  ,data.flag_Res, data.data_lead ,data.data_protected)

void Interpolate(int32_t *raw_out, int32_t *raw_in, f_lead flag_L, lead *marker_A, f_Res flag, Protected_Area *marker_Z, uint32_t sample_Huff)
//dim_out  = n byte in raw_out
//dim_in   =    "      raw_in
//marker_Z = no-subtraction zones
//marker_A =  start/end for each lead
//dec      = decimation factor
//
//for each lead, copy the first and the last two samples, then insert max 3 samples
// between each couple (this means that starting from the first and excluded the last,
// I create 4 samples).
//marker_Z contains the start (marker_Z->info.start=QB) and the end (marker_Z->info..end=QE)
//of all the no-subtraction zones.
//marker_A contains the start (marker_A.start = fist sample) and the end (marker_A.end = last sample)
//of lead data.
/*  given the interval [a,b] where to perform the interpolation, assumed that
    b>=a, the algorithm does:
        - mINS=((QB(k)-1)-(QE(k-1)+1)+1)/dec=(b-a+1)/dec (# of samples to expand)
        - mCOPY=(b-a+1)%dec (# of samples left)
        - in a "normal" situation where (b-a+1) is a multiple of dec (and mCOPY is 0) the algorithm does:
        - insert two samples as found in a and a+1
        - insert dec samples according to the formula:
                Z[i]=Va+j*D with D=(Vb-Va)/dec and i=0,1,2,3
                (where Va=a and Vb=a+1)
          i.e. between two samples I insert 3 samples.
          Finally, I insert two samples b-1 and b.
        - in "not normal" conditions, I do:
          for num=(b-a+1)<4: mCOPY may be from 1 to 3.
          samples are copied as they are;

  If a>b : may be
        - the first sample is the first of the first protected zone (a=b+1=1)
        - empty no-protection zone (a=b+1)
        - the last sample is the last of the last protected zone (a=b+1)
        in these situations samples are copied as they are.
*/
{
	uint16_t a, b;          //interpolation range
	float v;               //interpolation value
	uint8_t j;             //working variable
	uint16_t dim;           //number of samples (to skip) into a protected zone
	uint16_t mINS, mCOPY;   //number of samples to insert, copy
	int16_t num;             //number of samples to interpolate
	uint32_t pos_in, pos_out;
	uint16_t nz;            //index of protected zones, i.e. index of QRS
	uint8_t ne;            //index of lead

	pos_in=0;
	pos_out=0;
	for(ne=0;ne<flag_L.number;ne++)      // by E.C. 19.02.2004  loop on leads
	{
		for (nz=0;nz<=flag.number;nz++)    //  loop on QRS for each lead
		{
		//set limits for interpolation
			if(nz==0)
			{
				a=marker_A[ne].start;       // first interval  1 - QB[0]
				b=marker_Z[nz].QB-1;
			}
			else if (nz==flag.number)
			{
				a=marker_Z[nz-1].QE+1;      // last interval QE[flag.number-1] - 5000
				b=marker_A[ne].end;
			}
			else
			{
				a=marker_Z[nz-1].QE+1;       // interval between two QRS
				b=marker_Z[nz].QB-1;
			}
			//number of samples
			num=b-a+1;
			if(num>0)
			{
				mINS=num/flag.decimation_factor;  //number to interpolate
				mCOPY=num%flag.decimation_factor;  //if, residual number to copy
				if(mINS)
				{
					//store first two samples equal to the first value in the list ...
					raw_out[pos_out++]=raw_in[pos_in];
					raw_out[pos_out++]=raw_in[pos_in];
				}
				// ... then proceed with interpolation
				dim=mINS;
				while((dim--)>1)
				{
					v=1.0*(raw_in[pos_in+1]-raw_in[pos_in])/flag.decimation_factor;   // by E.C. 23.02.2004 (float v)
																					// rounding improves RMS!
					for(j=0;j<flag.decimation_factor;j++)
						raw_out[pos_out++]=raw_in[pos_in]+j*v;
					if(pos_in<(sample_Huff*(ne+1)))        // by E.C. 19.02.2004  check overflow
						++pos_in;
				}//end while
				if(pos_in>=(sample_Huff*(ne+1)))       //check overflow
					break;
				if(mINS)
				{
					raw_out[pos_out++]=raw_in[pos_in];
					raw_out[pos_out++]=raw_in[pos_in];
					if(pos_in<(sample_Huff*(ne+1)))        // by E.C. 19.02.2004  check overflow
						++pos_in;
				}
				//not normal situation? insert mCOPY samples
				while((mCOPY--)>0)
					if(pos_in<(sample_Huff*(ne+1)))              // by E.C. 19.02.2004  check overflow
						raw_out[pos_out++]=raw_in[pos_in++];
					else
						raw_out[pos_out++]=0;
			}//end if num>0

			if (nz<flag.number) {
				dim=marker_Z[nz].QE-marker_Z[nz].QB+1;    // copy protected zone
				while((dim--)>0)
					raw_out[pos_out++]=raw_in[pos_in++];
			}
		}//for nz
		pos_in=sample_Huff*(ne+1);      // by E.C. 19.02.2004  align, never mind!!
		pos_out=(pos_out+100)/5000*5000;     // by E.C. may 2004  for testing purposes only
	}//for ne
}//end Interpolate

void ExecFilter(int32_t *raw_in, int32_t *raw_out, uint32_t &pos, uint16_t dim)
//filter from pos for dim samples
{
	int32_t v;               //value
	uint16_t i;

	if (dim>0)
	{
		//fist sample = unchanged
		raw_out[pos]=raw_in[pos];  pos++;
		if(dim>2)
			for(i=2;i<dim;i++)                   // do filter
			{                                    // as mean of three samples
				v=raw_in[pos-1]+raw_in[pos]+raw_in[pos+1];

				if(v>=0)         // as suggested by the standard for rounding
					v+=1;
				else
					v-=1;

				raw_out[pos++]=v/3;    // by E.C. 24.02.2004  in this case, declaring v as float doesn't change results
			}
	}
	//last sample = unchanged
	if(dim>1) {
		raw_out[pos]=raw_in[pos];  pos++; }
}//end ExecFilter

void DoFilter(int32_t *raw_out, int32_t *raw_in, f_Res flag, f_lead flag_L, lead *marker_A, Protected_Area *marker_P, Subtraction_Zone *marker_S)
//filter low-pass outside the proteced zones (marker_Z) for each lead (marker_A)
// but taking into account transients at the boundaries of the subtraction zones (marker_S)
//It's included rounding.
{
	uint16_t a, b=0;         //interval
	int16_t num;
	uint32_t pos;
	uint16_t nz;
	uint8_t ne;            //index of lead

	pos=0;                               // by E.C. 19.02.2004  function redefined such as Interpolate()
	for(ne=0;ne<flag_L.number;ne++)      // loop on leads
	{
		for (nz=0;nz<=flag.number;nz++)     // loop on QRS of each lead
		{
			//set interval to filter
			if(nz==0)
			{
				a=marker_A[ne].start;
				if (marker_S[nz].beat_type==0) {
					b=marker_S[nz].SB-1;
					num=b-a+1;
					ExecFilter(raw_in,raw_out,pos,num);      // if, 1 - SB[0]-1
					a=marker_S[nz].SB;                       // by E.C. may 2004 (handle open and close intervals)
				}
				b=marker_P[nz].QB;
				num=b-a+1;
				ExecFilter(raw_in,raw_out,pos,num);      // (1 or) SB[0] - QB[0]
			}
			else if (nz==flag.number)
			{
				a=marker_P[nz-1].QE;
				if (marker_S[nz-1].beat_type==0) {
					b=marker_S[nz-1].SE;
					num=b-a+1;
					ExecFilter(raw_in,raw_out,pos,num);      // if, QE[last] - SE[last]
					a=marker_S[nz-1].SE+1;
				}
				b=marker_A[ne].end;
				num=b-a+1;
				ExecFilter(raw_in,raw_out,pos,num);      // (QE[last] or) SE[last]+1 - 5000
			}
			else
			{
				a=b+1;
				if (marker_S[nz-1].beat_type==0) {
					b=marker_S[nz-1].SE;
					num=b-a+1;
					ExecFilter(raw_in,raw_out,pos,num);      // in between
					a=marker_S[nz-1].SE+1;
				}
				if (marker_S[nz].beat_type==0) {
					b=marker_S[nz].SB-1;
					num=b-a+1;
					ExecFilter(raw_in,raw_out,pos,num);
					a=marker_S[nz].SB;
				}
				b=marker_P[nz].QB;
				num=b-a+1;
				ExecFilter(raw_in,raw_out,pos,num);
			}

			if (nz<flag.number) {
			a=marker_P[nz].QB+1;
			b=marker_P[nz].QE-1;
			num=b-a+1;    // copy protected zone
			while((num--)>0) {
				raw_out[pos]=raw_in[pos]; pos++; }
			}
		}//for nz
	}  // for ne/ng
}//end DoFilter

void DoAdd(int32_t *raw_out, int32_t *raw_R, f_Res flag_R, int32_t *raw_B, f_BdR0 flag_B, Subtraction_Zone *marker_S, f_lead flag_L, lead *marker_A)
//add BdR0 with the rhythm data for all leads
{
	uint16_t pos_B;
	uint32_t pos_R, pos_out;
	uint16_t ns, a , b, num;
	uint8_t ne;            //index of leads

	pos_R=0;                             // by E.C. 19.02.2004  add reference beat to rhythm
	for(ne=0;ne<flag_L.number;ne++)      //  loop on lead
		for (ns=0;ns<flag_R.number;ns++)   //  loop on QRS
			if(marker_S[ns].beat_type==0)      //  only reference beat!
			{
				pos_B=(flag_B.fcM-1)-(marker_S[ns].fc-marker_S[ns].SB);  //initial position in reference beat
				a= marker_S[ns].SB;
				b= marker_S[ns].SE;
				num=b-a+1;     //number of samples to add
				pos_out= a-1;
				pos_out+= ne*flag_R.number_samples;
				pos_B+= ne*flag_B.number_samples;
				while((num--)>0)
				{
					raw_out[pos_out]+=raw_B[pos_B];
					pos_out++;
					pos_B++;
					pos_R++;
				}
			}//end for/if
}//end DoAdd

void Opt_Filter(int32_t *raw_out, int32_t *raw_in, f_Res flag, f_lead flag_L, lead *marker_A, Protected_Area *marker_P)
// by E.C. 25.02.2004
// do an extra low-pass filter outside protected zones (marker_P)
// in the range of the signal (marker_A) for each lead
// this is simpler than DoFilter()
{
	uint16_t a, b=0;         //interval for filtering
	int16_t num;
	uint32_t pos;
	uint16_t nz;
	uint8_t ne;            //lead index

	pos=0;
	for(ne=0;ne<flag_L.number;ne++)      // loop on leads
	{
		for (nz=0;nz<=flag.number;nz++)     // loop on QRS of each lead
		{
			// establish the interval
			if(nz==0)
			{
				a=marker_A[ne].start;
				b=marker_P[nz].QB;
				num=b-a+1;
				ExecFilter(raw_in,raw_out,pos,num);      // from start to first protected zone
			}
			else if (nz==flag.number)
			{
				a=marker_P[nz-1].QE;
				b=marker_A[ne].end;
				num=b-a+1;
				ExecFilter(raw_in,raw_out,pos,num);      // from last protected zone to end
			}
			else
			{
				a=b+1;
				b=marker_P[nz].QB;
				num=b-a+1;
				ExecFilter(raw_in,raw_out,pos,num);      // between two protected zones
			}

			if (nz<flag.number) {
				a=marker_P[nz].QB+1;
				b=marker_P[nz].QE-1;
				num=b-a+1;                              // copy the protected zone as is
				while((num--)>0) {
					raw_out[pos]=raw_in[pos]; pos++; }
			}
		}//for nz
	}  // for ne
}//end Opt_Filter
