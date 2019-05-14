/*

    Copyright (C) 2005,2006,2007,2008,2009,2016 Alois Schloegl <alois.schloegl@gmail.com>
    This file is part of the "BioSig for C/C++" repository
    (biosig4c++) at http://biosig.sf.net/


    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>. 
    
 */

#define IPv4

#ifndef __BIOSIG_NETWORK_H__
#define __BIOSIG_NETWORK_H__

#include "biosig-dev.h"


#define SERVER_PORT 54321

#if defined(__MINGW32__) 
#include <winsock2.h>
#include <ws2tcpip.h>

#ifndef socklen_t
#define socklen_t int
#endif

/* mingw/include/errno.h */
#ifndef _INC_ERRNO
#define EALREADY      WSAEALREADY    
#define ECONNABORTED  WSAECONNABORTED
#define ECONNREFUSED  WSAECONNREFUSED
#define ECONNRESET    WSAECONNRESET  
#define EHOSTDOWN     WSAEHOSTDOWN   
#define EHOSTUNREACH  WSAEHOSTUNREACH
#define EINPROGRESS   WSAEINPROGRESS 
#define EISCONN       WSAEISCONN     
#define ENETDOWN      WSAENETDOWN    
#define ENETRESET     WSAENETRESET   
#define ENETUNREACH   WSAENETUNREACH 
#define EWOULDBLOCK   WSAEWOULDBLOCK 
#define EADDRINUSE    WSAEADDRINUSE
#define ENOTSUP       ENOSYS
#define ETIMEDOUT     WSAETIMEDOUT
#define ENOTSOCK      WSAENOTSOCK
#define ENOBUFS       WSAENOBUFS
#define EMSGSIZE      WSAEMSGSIZE
#define EADDRNOTAVAIL WSAEADDRNOTAVAIL
#define EPROTONOSUPPORT WSAEPROTONOSUPPORT
#endif

#if 0 //!__linux__
// needed by MinGW on Windows
#define creat(a, c)    OpenFile(a, O_WRONLY|O_CREAT|O_TRUNC, c)
#define write(a,b,c)   WriteFile(a,b,c,0,0)
#define close(a)       CloseFile(a)
#endif

#else 
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#endif


/* External API definitions */

/****************************************************************************/
/**                                                                        **/
/**                 DEFINITIONS, TYPEDEFS AND MACROS                       **/
/**                                                                        **/
/****************************************************************************/


	/* client server commands*/
#define BSCS_ID_BITLEN 64 	
#define BSCS_MAX_BUFSIZ_LOG2 14 
#define BSCS_MAX_BUFSIZ (1<<BSCS_MAX_BUFSIZ_LOG2) 
#define BSCS_VERSION (htobe32(0x01000000)) 	// version 1

#define	VER_MASK   (htobe32(0xff000000))
#define	CMD_MASK   (htobe32(0x00ff0000))
#define	STATE_MASK (htobe32(0x0000ff00))
#define	ERR_MASK   (htobe32(0x000000ff))

#define	BSCS_VERSION_0 (htobe32(0x00000000)) 		// Version 0
#define	BSCS_VERSION_01 (htobe32(0x01000000)) 		// Version 0.1
#define	BSCS_VERSION_02 (htobe32(0x02000000)) 		// Version 0.2

#define	BSCS_NOP       (htobe32(0x00000000))	// no operation
#define	BSCS_OPEN      (htobe32(0x00010000))	// open
#define	BSCS_OPEN_R    (htobe32(0x00010000))	// open read
#define	BSCS_OPEN_W    (htobe32(0x00010000))	// open write
#define	BSCS_CLOSE     (htobe32(0x00020000))	// close
#define	BSCS_SEND_MSG  (htobe32(0x00030000))	// send message
#define	BSCS_SEND_HDR  (htobe32(0x00040000))	// send header information
#define	BSCS_SEND_DAT  (htobe32(0x00050000))	// send data block
#define	BSCS_SEND_EVT  (htobe32(0x00060000))	// send event information
#define	BSCS_REQU_HDR  (htobe32(0x00070000))	// reqest header info
#define	BSCS_REQU_DAT  (htobe32(0x00080000))	// request data block
#define	BSCS_REQU_EVT  (htobe32(0x00090000))	// request event table
#define	BSCS_PUT_FILE  (htobe32(0x000a0000))	// request event table
#define	BSCS_GET_FILE  (htobe32(0x000b0000))	// request event table
#define	BSCS_REPLY     (htobe32(0x00800000))	// replay flag: can be combined with any of the above codes

#define	STATE_INIT    	      (htobe32(0x00000000)) 		// initial state
#define	STATE_OPEN_READ       (htobe32(0x00000a00)) 	// connection opened for reading
#define	STATE_OPEN_WRITE_HDR  (htobe32(0x00000b00)) 	// connection opened for writing header
#define	STATE_OPEN_WRITE      (htobe32(0x00000c00)) 	// connection opened for writing data and events #define

#define	BSCS_NO_ERROR    			 (htobe32(0x00000000))	// no error
#define	BSCS_ERROR_CANNOT_OPEN_FILE 		 (htobe32(0x00000001))	// writing error
#define	BSCS_ERROR_INCORRECT_PACKET_LENGTH 	 (htobe32(0x00000002))	// writing error
#define	BSCS_ERROR_CLOSE_FILE 			 (htobe32(0x00000003))	// any error
#define	BSCS_ERROR_COULD_NOT_WRITE_HDR 		 (htobe32(0x00000004))	// any error
#define	BSCS_ERROR_COULD_NOT_WRITE_DAT 		 (htobe32(0x00000005))	// any error
#define	BSCS_ERROR_COULD_NOT_WRITE_EVT 		 (htobe32(0x00000006))	// any error
#define	BSCS_INCORRECT_REPLY_PACKET_LENGTH 	 (htobe32(0x00000007))	// writing error
#define	BSCS_ERROR_MEMORY_OVERFLOW 		 (htobe32(0x00000008))	// any error
#define	BSCS_ERROR       			 (htobe32(0x000000ff))	// any error

// error code for connecting to server: must be negative numbers
#define	BSCS_UNKNOWN_HOST		 	 (-1)	//  
#define	BSCS_CANNOT_OPEN_SOCKET		 	 (-2)	//  
#define	BSCS_CANNOT_BIND_PORT		 	 (-3)	//  
#define	BSCS_CANNOT_CONNECT		 	 (-4)	//  
#define	BSCS_SERVER_NOT_SUPPORTED	 	 (-5)	//  

#define max(a,b)        (((a) > (b)) ? (a) : (b))

typedef struct {
	uint32_t STATE; 
	uint32_t LEN; 
	uint8_t  LOAD[max(8,BSCS_ID_BITLEN>>3)]  __attribute__ ((aligned (8))); 	// must fit at least ID length
} mesg_t  __attribute__ ((aligned (8)));

extern uint32_t SERVER_STATE; 

/****************************************************************************/
/**                                                                        **/
/**                     EXPORTED FUNCTIONS                                 **/
/**                                                                        **/
/****************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif 


int c64ta(uint64_t ID, char* txt);	// convert 64bit to ascii
int cat64(char* txt, uint64_t *ID);	// convert ascii to 64bit

void *get_in_addr(struct sockaddr *sa);

/*
	biosig client-server functions 
*/

int bscs_connect(const char* hostname); 
/*  opens a connection to the server <hostname>
	on success, the socket file descriptor (a positive integer) is returned
	in case of failure, a negative integer is returned 	
-------------------------------------------------------------- */

int bscs_disconnect(int sd); 
/*  disconnects the socket file descriptor 	
-------------------------------------------------------------- */

int send_packet(int sd, uint32_t state, uint32_t len, void* load);
/* send a single packet including header and load 	
  -------------------------------------------------------------- */

int bscs_open(int sd, uint64_t *ID); // read-open
/* ID = 0 : write access, new identifier is returned in ID
   ID > 0 : read access to the file with known ID
  -------------------------------------------------------------- */

int bscs_close(int sd);
/* close current connection 
  -------------------------------------------------------------- */

int bscs_send_hdr(int sd, HDRTYPE *hdr);
/* hdr->AS.Header must contain GDF header information 
   hdr->HeadLen   must contain header length 
  -------------------------------------------------------------- */

int bscs_send_dat(int sd, void* buf, size_t len );
/* buf must contain the data block as in hdr->AS.rawdata 
  -------------------------------------------------------------- */

int bscs_send_evt(int sd, HDRTYPE *hdr);
/* hdr->EVENT defines the event table 
  -------------------------------------------------------------- */

int bscs_send_msg(int sd, char* msg);
/* msg is string 
   -------------------------------------------------------------- */

int bscs_error(int sd, int ERRNUM, char* ERRMSG);
/* ERRNUM contains the error number 
   ERRMSG is string 
   -------------------------------------------------------------- */

int bscs_requ_hdr(int sd, HDRTYPE *hdr);
/* request header information 
   -------------------------------------------------------------- */

ssize_t bscs_requ_dat(int sd, size_t start, size_t nblocks, HDRTYPE *hdr);
/* request data blocks 
	bufsiz is maximum number of bytes, typically it must be nblocks*hdr->AS.bpb
   -------------------------------------------------------------- */

int bscs_requ_evt(int sd, HDRTYPE *hdr);
/* request event information 
   -------------------------------------------------------------- */

int bscs_put_file(int sd, char *filename);
/* put raw data file on server 
   -------------------------------------------------------------- */

int bscs_get_file(int sd, uint64_t ID, char *filename);
/* put raw data file on server 
   -------------------------------------------------------------- */

int bscs_nop(int sd);
/* no operation 
   -------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif 

/****************************************************************************/
/**                                                                        **/
/**                               EOF                                      **/
/**                                                                        **/
/****************************************************************************/

#endif	/* __BIOSIG_NETWORK_H__ */
