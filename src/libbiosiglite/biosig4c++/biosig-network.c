/*

    Copyright (C) 2009,2015,2016 Alois Schloegl <alois.schloegl@gmail.com>
    This file is part of the "BioSig for C/C++" repository 
    (biosig4c++) at http://biosig.sf.net/ 

    BioSig is free software; you can redistribute it and/or
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

#include <ctype.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>

#include "biosig-network.h"

#ifdef _WIN32
#define TC (char*)	// WINSOCK's send and recv require (char*)buf 

#if 0 
/* 
	these functions have a different name under windows, 
	MinGW für windows requires these 
*/
int creat(const char *path, mode_t mode) {
//	return(OpenFile(path, O_WRONLY|O_CREAT|O_TRUNC, mode));
	return(OpenFile(path, , OF_WRITE|OF_CREAT ));
}
ssize_t write(int fildes, const void *buf, size_t nbyte) {
	ssize_t sz;
	WriteFile(fildes, buf, nbyte, &sz, NULL)
	return(sz);
}
int close(int fildes) {
	return(CloseFile(fildes)); {
}
// the following ones are not used 
/*
int open(const char *path, int oflag) {
	return(OpenFile(path, oflag));
}
int open(const char *path, int oflag, mode_t mode ) {
	return(OpenFile(path, oflag, mode));
}
ssize_t read(int fildes, const void *buf, size_t nbyte) {
	return(ReadFile(fildes, buf, nbyte));
}
*/
#endif 

#else	
#define TC
#endif 

uint64_t B4C_ID=0;	// ID of currently open file 
const char *B4C_HOSTNAME = NULL; 
uint32_t SERVER_STATE; // state of server, useful for preliminary error checking 

/*
	converts 64bit integer into hex string
*/
int c64ta(uint64_t ID, char *txt) {
	const char hex[] = "0123456789abcdef";
	int k=(BSCS_ID_BITLEN>>2)-1;
	for (; k>=0; k--) {
		txt[k] = hex[ID & 0x0f]; 
		ID>>=4;
	}
	txt[BSCS_ID_BITLEN>>2] = 0;
	if (VERBOSE_LEVEL>8) fprintf(stdout,"c64ta: ID=%016"PRIx64" TXT=%s\n",ID,txt);
	return 0;
}


/*
	converts hex string into 64bit integer 
*/
int cat64(char* txt, uint64_t *id) {
	uint64_t ID = 0; 
	int k = 0;
	for (; txt[k] && (k<(BSCS_ID_BITLEN>>2));k++) {
		ID<<=4; 
		if (isdigit(txt[k]))
			ID += txt[k]-'0';
		else if (isxdigit(txt[k]))
			ID += toupper(txt[k])-'A'+10;
		else {
			*id = -1;
			return(-1);
		}	 	
	}
	*id = ID;
	if (VERBOSE_LEVEL>8) fprintf(stdout,"cat64: ID=%016"PRIx64" TXT=%s\n",ID,txt);
	return(0);
}


// get sockaddr, IPv4 or IPv6:
void *get_in_addr(struct sockaddr *sa)
{
	if (sa->sa_family == AF_INET) {
		return &(((struct sockaddr_in*)sa)->sin_addr);
	}
#ifndef _WIN32
	return &(((struct sockaddr_in6*)sa)->sin6_addr);
#else 
	return(NULL); 
#endif     
}


/****************************************************************************************
	OPEN CONNECTION TO SERVER
 ****************************************************************************************/
int bscs_connect(const char* hostname) {

	int sd; 
    	struct sockaddr_in localAddr;

	if (hostname==NULL) hostname = "129.27.3.99"; 
	B4C_HOSTNAME = hostname; 

#ifdef _WIN32
	WSADATA wsadata;
	if (WSAStartup(MAKEWORD(1,1), &wsadata) == SOCKET_ERROR) {
		fprintf(stderr,"Error creating socket.");
		return(BSCS_CANNOT_CONNECT);
	}
#endif
	
	int status;

#if 1	// IPv4 and IPv6
	struct addrinfo hints;
	struct addrinfo *result, *p;

	memset(&hints, 0, sizeof(struct addrinfo));
	hints.ai_family   = AF_UNSPEC;    /* Allow IPv4 or IPv6 */
	hints.ai_socktype = SOCK_STREAM; // TCP stream sockets
	hints.ai_flags    = 0;
	hints.ai_protocol = 0;          /* Any protocol */

	status = getaddrinfo(hostname, NULL, &hints, &result);
	if (status != 0) {
	        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(status));
	   	return(BSCS_UNKNOWN_HOST);
   	}

	// loop through all the results and connect to the first we can
#ifndef NI_MAXHOST
#define NI_MAXHOST 1025
#endif
	for(p = result; p != NULL; p = p->ai_next) {
        	char hostname1[NI_MAXHOST] = "";
        	int error = getnameinfo(p->ai_addr, p->ai_addrlen, hostname1, NI_MAXHOST, NULL, 0, 0); 
        	if (*hostname1)
        		printf("hostname: %s\n", hostname1);

		if ((sd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1) {
			perror("client: socket");
			continue;
		}
		if (connect(sd, p->ai_addr, p->ai_addrlen) != -1)
			break;

		close(sd);
    	}
	if (p == NULL) {
		fprintf(stderr, "client: failed to connect\n");
#ifdef _WIN32
		WSACleanup();
#endif
		return(BSCS_CANNOT_CONNECT);
	}

	char s[INET6_ADDRSTRLEN];
#if !defined(_WIN32)
	// FIXME: does not compile with MXE for TARGET "i686-w64-mingw32.static"
	inet_ntop(p->ai_family, get_in_addr((struct sockaddr *)p->ai_addr), s, sizeof(s));
#else
	strcpy(s," [Icannot get server name - NET_NTOP() not available]");
#endif
    	printf("client: connecting to %s\n", s);

	freeaddrinfo(result);   // all done with this structure

#else 

	// IPv4 only 	
    	struct hostent *h;
    	struct sockaddr_in sain;

	
   	h = gethostbyname(hostname);
   	if(h==NULL) return(BSCS_UNKNOWN_HOST);

   	sain.sin_family = h->h_addrtype;
   	memcpy((char *) &sain.sin_addr.s_addr, h->h_addr_list[0],h->h_length);
   	sain.sin_port = htons(SERVER_PORT);

   	/* create socket */
   	sd = socket(AF_INET, SOCK_STREAM, 0);
   	if(sd<0) return(BSCS_CANNOT_OPEN_SOCKET);

   	/* bind any port number */
   	localAddr.sin_family = AF_INET;
   	localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
   	localAddr.sin_port = htons(0);
            
   	/* connect to server */
   	int rc = connect(sd, (struct sockaddr *) &sain, sizeof(sain));
   	if(rc<0) return(BSCS_CANNOT_CONNECT);

#endif 
   
	/* identification */	
	mesg_t msg; 
	recv(sd,TC&msg,8,0);
	int len = be32toh(msg.LEN);
	if ((msg.STATE & (VER_MASK|CMD_MASK)) != (BSCS_VERSION_01 | BSCS_SEND_MSG))	// currently only V0.1 is supported 
	{
		close(sd);
		return(BSCS_SERVER_NOT_SUPPORTED);
	}
	
	char *greeting = (char*)malloc(len+1); 
	recv(sd,greeting,len,0);
	greeting[len]=0; 
	//fprintf(stdout,"%s",greeting);
	free(greeting);
   	return(sd);
}



/****************************************************************************************
	CLOSE CONNECTION TO SERVER
 ****************************************************************************************/
int bscs_disconnect(int sd) {
	close(sd); 
#ifdef _WIN32		
	WSACleanup();
#endif
}


int send_packet(int sd, uint32_t state, uint32_t len, void* load) {
	mesg_t msg; 
	msg.STATE = state; 
	msg.LEN = htobe32(len);
	send(sd, TC &msg, 8, 0);
	if (len>0) send(sd, TC load, len, 0);
}


/*
	stores a bscs link file on client side for the 
	currently open network file is stored in the directory "/tmp/"
*/
int savelink(const char* filename) {

	if ((SERVER_STATE & STATE_MASK)==STATE_INIT) return(-1);	

	char *logpath = "/tmp/";		// perhaps some other directory 
	const char *ext = ".bscs";
	const char *fn = strrchr(filename,FILESEP);
	if (fn==NULL) fn = filename; 
	else fn++;

	size_t l1 = strlen(logpath);
	size_t l2 = strlen(fn);
	size_t l3 = strlen(ext);
	size_t l4 = 10;
	char *logfile = (char*)malloc(l1+l2+l3+l4+1);
	memcpy(logfile, logpath, l1); 
	memcpy(logfile+l1, fn, l2); 
	strcpy(logfile+l1+l2, ext); 
	int k=0; 
	size_t sl = strlen(logfile);
	
	FILE *fid;
	// check whether file already exists 
	while ((fid=fopen(logfile,"r")) != NULL) {
		fclose(fid);
		snprintf(logfile+sl, l4, ".%i", k);
		k++;
	}
	errno = 0; 	// fopen caused errno=2; reset errno 

fprintf(stdout,"savelink %s\n",logfile); 
	
	fid = fopen(logfile,"w");
	fprintf(fid,"bscs://%s/%016"PRIx64"\n",B4C_HOSTNAME,B4C_ID);
	fclose(fid);
	free(logfile);

}


/****************************************************************************************
	OPEN FILE with ID 
 ****************************************************************************************/
int bscs_open(int sd, uint64_t* ID) {

	if (SERVER_STATE != STATE_INIT) return(BSCS_ERROR);

	size_t LEN;
	mesg_t msg; 
	
	if (*ID==0) {
		msg.STATE = BSCS_VERSION_01 | BSCS_OPEN_W | STATE_INIT | BSCS_NO_ERROR;
		LEN   = 0;
	}
	else {
		msg.STATE = BSCS_VERSION_01 | BSCS_OPEN_R | STATE_INIT | BSCS_NO_ERROR;
		*(uint64_t*)&msg.LOAD  = leu64p(ID);
		LEN = BSCS_ID_BITLEN>>3;
	}	

	if (VERBOSE_LEVEL>8) fprintf(stdout,"open: %16"PRIx64" %016"PRIx64"\n",*ID,*(uint64_t*)&(msg.LOAD));

	msg.LEN   = htobe32(LEN);
	int s = send(sd, TC &msg, LEN+8, 0);	

	// wait for reply 
	ssize_t count = 0; 
	count = recv(sd, TC &msg, 8, 0);
	LEN = be32toh(msg.LEN);
	SERVER_STATE = msg.STATE & STATE_MASK;

	if (VERBOSE_LEVEL>8) fprintf(stdout,"BSCS_OPEN %i:%"PRIiPTR": ID=%16"PRIx64" LEN=%"PRIx32" STATE=0x%08"PRIx32"\n",s,count,*ID,msg.LEN,be32toh(msg.STATE));

	if ((*ID==0) && (LEN==8) && (msg.STATE==(BSCS_VERSION_01 | BSCS_OPEN_W | BSCS_REPLY | STATE_OPEN_WRITE_HDR | BSCS_NO_ERROR)) ) 
	{
		// write access: get new ID
		count += recv(sd, TC ID, LEN, 0);
		
		*ID = htole64(*ID);
		B4C_ID = *ID;	
		return(0);
	}
	
	if ((*ID != 0) && (LEN==0) && (msg.STATE==(BSCS_VERSION_01 | BSCS_OPEN_R | BSCS_REPLY | STATE_OPEN_READ | BSCS_NO_ERROR)) ) 
	{
		; // read access 
		return(0); 
	}
	
	uint8_t buf[8];
	count = 0;
	while (LEN > (size_t)count) { 	
		count += recv(sd, TC &buf, min(8,LEN-count), 0);
	 	// invalid packet or error opening file 
	}
	
	if (VERBOSE_LEVEL>7) 	fprintf(stdout,"ERR: state= %08"PRIx32" %08"PRIx32" len=%"PRIiPTR"\n",htobe32(msg.STATE),BSCS_VERSION_01 | BSCS_OPEN_R | BSCS_REPLY | STATE_OPEN_READ | BSCS_NO_ERROR,LEN);
	
	return(msg.STATE);
}

/****************************************************************************************
	CLOSE FILE 
 ****************************************************************************************/
int bscs_close(int sd) {
	int s;
	size_t LEN;
	mesg_t msg; 
	
	msg.STATE = BSCS_VERSION_01 | BSCS_CLOSE | BSCS_NO_ERROR | SERVER_STATE;
if (VERBOSE_LEVEL>8) fprintf(stdout,"close1: %08"PRIx32" \n",msg.STATE);
	msg.LEN   = htobe32(0);
if (VERBOSE_LEVEL>8) fprintf(stdout,"close2: %08"PRIx32" %"PRIiPTR" %"PRIi32"\n",msg.STATE,sizeof(msg),msg.LEN);

	s = send(sd, TC &msg, 8, 0);	

if (VERBOSE_LEVEL>8) fprintf(stdout,"close3: %08"PRIx32" %i\n",msg.STATE,s);

	// wait for reply 
	s = recv(sd, TC &msg, 8, 0);
	LEN = be32toh(msg.LEN);
	SERVER_STATE = msg.STATE & STATE_MASK;

if (VERBOSE_LEVEL>8) fprintf(stdout,"s=%i state= %08"PRIx32" len=%"PRIiPTR" %i  %08"PRIx32"\n",s,msg.STATE & ~STATE_MASK,LEN,s,(BSCS_VERSION_01 | BSCS_CLOSE | BSCS_REPLY));

	if ((LEN==0) && ((msg.STATE & ~STATE_MASK)==(BSCS_VERSION_01 | BSCS_CLOSE | BSCS_REPLY | BSCS_NO_ERROR)) ) 
		// close without error 
		return(0);

	if ((LEN==0) && ((msg.STATE & ~STATE_MASK & ~ERR_MASK)==(BSCS_VERSION_01 | BSCS_CLOSE | BSCS_REPLY)) ) 
		// close with error 
		return(msg.STATE & ERR_MASK);

	 	// invalid packet or error opening file 
	if (VERBOSE_LEVEL>8) 	fprintf(stdout,"ERR: state= %08"PRIx32" len=%"PRIiPTR"\n",msg.STATE,LEN);
	return(msg.STATE);
}

/****************************************************************************************
	SEND HEADER
 ****************************************************************************************/
int bscs_send_hdr(int sd, HDRTYPE *hdr) {
/* hdr->AS.Header must contain GDF header information 
   hdr->HeadLen   must contain header length 
  -------------------------------------------------------------- */
	// ToDo: convert HDR into AS.Header

	if (SERVER_STATE != STATE_OPEN_WRITE_HDR) return(BSCS_ERROR);

	mesg_t msg; 

	hdr->TYPE = GDF;
	hdr->FLAG.ANONYMOUS = 1; 	// do not store name 
	struct2gdfbin(hdr); 

	msg.STATE = BSCS_VERSION_01 | BSCS_SEND_HDR | STATE_OPEN_WRITE_HDR | BSCS_NO_ERROR;
	msg.LEN   = htobe32(hdr->HeadLen);
	int s = send(sd, TC &msg, 8, 0);	
if (VERBOSE_LEVEL>8) fprintf(stdout,"SND HDR  %i %i\n",hdr->HeadLen,s); 
	s = send(sd, TC hdr->AS.Header, hdr->HeadLen, 0);	

if (VERBOSE_LEVEL>8) fprintf(stdout,"SND HDR %i %i\n",hdr->HeadLen,s); 

	// wait for reply 
	ssize_t count = recv(sd, TC &msg, 8, 0);

if (VERBOSE_LEVEL>8) fprintf(stdout,"SND HDR %i %i %"PRIiPTR" %08"PRIx32"\n",hdr->HeadLen,s,count,msg.STATE);

	size_t LEN = be32toh(msg.LEN);
	SERVER_STATE = msg.STATE & STATE_MASK;

	if ((LEN==0) && (msg.STATE==(BSCS_VERSION_01 | BSCS_SEND_HDR | BSCS_REPLY | STATE_OPEN_WRITE | BSCS_NO_ERROR)) ) 
		// close without error 
		return(0);
	if ((LEN==0) && ((msg.STATE & ~ERR_MASK)==(BSCS_VERSION_01 | BSCS_SEND_HDR | BSCS_REPLY | STATE_OPEN_WRITE_HDR)) ) 
		// could not write header  
		return((msg.STATE & ~ERR_MASK) | BSCS_ERROR_COULD_NOT_WRITE_HDR);

	 	// invalid packet or error opening file 
	return(msg.STATE);
}

/****************************************************************************************
	SEND DATA
 ****************************************************************************************/
int bscs_send_dat(int sd, void* buf, size_t len ) {

	if (SERVER_STATE != STATE_OPEN_WRITE) return(BSCS_ERROR);

	size_t LEN;
	mesg_t msg; 
	msg.STATE = BSCS_VERSION_01 | BSCS_SEND_DAT | STATE_OPEN_WRITE | BSCS_NO_ERROR;
	msg.LEN   = htobe32(len);

if (VERBOSE_LEVEL>8) fprintf(stdout,"SND DAT %"PRIiPTR" %08"PRIx32"\n",len,msg.STATE);

	ssize_t s;
	s = send(sd, TC &msg, 8, 0);	
	s = send(sd, TC buf, len, 0);	
	if (errno) fprintf(stdout,"SND DAT ERR=%i %s\n",errno,strerror(errno));

if (VERBOSE_LEVEL>8) fprintf(stdout,"SND DAT %"PRIiPTR" %08"PRIx32" %"PRIiPTR" \n",len,msg.STATE,s);

	// wait for reply 
	s = recv(sd, TC &msg, 8, 0);
	LEN = be32toh(msg.LEN);
	SERVER_STATE = msg.STATE & STATE_MASK;

if (VERBOSE_LEVEL>8) fprintf(stdout,"SND DAT RPLY %"PRIiPTR" %08"PRIx32" \n",s,msg.STATE);

	if ((LEN==0) && (msg.STATE==(BSCS_VERSION_01 | BSCS_SEND_DAT | BSCS_REPLY | STATE_OPEN_WRITE | BSCS_NO_ERROR)) ) 
		// end without error 
		return(0);

	if (LEN>0) 
		return (BSCS_VERSION_01 | BSCS_SEND_DAT | BSCS_REPLY | STATE_OPEN_WRITE | BSCS_INCORRECT_REPLY_PACKET_LENGTH);

	if ((msg.STATE & ~ERR_MASK)==(BSCS_VERSION_01 | BSCS_SEND_DAT | BSCS_REPLY | STATE_OPEN_WRITE)) 
		// could not write header  
		return ((msg.STATE & ~ERR_MASK) | BSCS_ERROR_COULD_NOT_WRITE_DAT);

 	// invalid packet or error opening file 
	return(msg.STATE);
}

/****************************************************************************************
	SEND EVENTS
 ****************************************************************************************/
int bscs_send_evt(int sd, HDRTYPE *hdr) {

	if (SERVER_STATE != STATE_OPEN_WRITE) return(BSCS_ERROR);

	int sze;
	char flag;

	size_t LEN;
	mesg_t msg;

	if ((hdr->EVENT.DUR!=NULL) && (hdr->EVENT.CHN!=NULL)) {
		sze = 12; 
		flag = 3;
	} else {
		sze = 6; 
		flag = 1;
	}	

	size_t len = hdrEVT2rawEVT(hdr); 

if (VERBOSE_LEVEL>8) fprintf(stdout,"write evt: len=%"PRIiPTR"\n",len);

	msg.STATE = BSCS_VERSION_01 | BSCS_SEND_EVT | STATE_OPEN_WRITE | BSCS_NO_ERROR;
	msg.LEN   = htobe32(len);
	int s1 = send(sd, TC &msg, 8, 0);	
	int s2 = send(sd, TC hdr->AS.rawEventData, len, 0);	

if (VERBOSE_LEVEL>8) fprintf(stdout,"write evt2: %08"PRIx32" len=%"PRIiPTR"\n",msg.STATE,len);
	
	// wait for reply 
	ssize_t count = recv(sd, TC &msg, 8, 0);
	LEN = be32toh(msg.LEN);
	SERVER_STATE = msg.STATE & STATE_MASK;

if (VERBOSE_LEVEL>8) fprintf(stdout,"write evt2: %08"PRIx32" len=%"PRIiPTR" count=%"PRIiPTR"\n",msg.STATE,LEN,count);
	
	if ((LEN==0) && (msg.STATE==(BSCS_VERSION_01 | BSCS_SEND_EVT | BSCS_REPLY | STATE_OPEN_WRITE | BSCS_NO_ERROR)) ) 
		// close without error 
		return(0);

	if (LEN>0) 
		return (BSCS_VERSION_01 | BSCS_SEND_EVT | BSCS_REPLY | STATE_OPEN_WRITE | BSCS_INCORRECT_REPLY_PACKET_LENGTH);

	if (msg.STATE & ERR_MASK) 
		// could not write evt  
		return (BSCS_ERROR_COULD_NOT_WRITE_EVT);

 	// invalid packet or error opening file 
	return(msg.STATE);
}

/****************************************************************************************
	REQUEST HEADER
 ****************************************************************************************/
int bscs_requ_hdr(int sd, HDRTYPE *hdr) {
	mesg_t msg; 
	int count; 
	

	if (SERVER_STATE != STATE_OPEN_READ) return(BSCS_ERROR);
	
	msg.STATE = BSCS_VERSION_01 | BSCS_REQU_HDR | BSCS_NO_ERROR | (SERVER_STATE & STATE_MASK);
	msg.LEN   = 0;
	count = send(sd, TC &msg, 8, 0);	

   	count = recv(sd, TC &msg, 8, 0);
	hdr->HeadLen   = be32toh(msg.LEN);
    	hdr->AS.Header = (uint8_t*)realloc(hdr->AS.Header,hdr->HeadLen);
    	hdr->TYPE      = GDF; 
	count = 0; 
	while (hdr->HeadLen > (size_t)count) {
		count += recv(sd, TC hdr->AS.Header+count, hdr->HeadLen-count, 0);
	}

if (VERBOSE_LEVEL>8) fprintf(stdout,"REQ HDR: %i %s\n",count,GetFileTypeString(hdr->TYPE));
	
	hdr->FLAG.ANONYMOUS = 1; 	// do not store name 
	gdfbin2struct(hdr); 
	
	
   	return(count-hdr->HeadLen); 
}

/****************************************************************************************
	REQUEST DATA
 ****************************************************************************************/
ssize_t bscs_requ_dat(int sd, size_t start, size_t length, HDRTYPE *hdr) {
/*
	bufsiz should be equal to hdr->AS.bpb*length	
*/	
	mesg_t msg; 
	size_t LEN;
	
	if (SERVER_STATE != STATE_OPEN_READ) return(BSCS_ERROR);
	
	msg.STATE = BSCS_VERSION_01 | BSCS_REQU_DAT | BSCS_NO_ERROR | (SERVER_STATE & STATE_MASK);
	msg.LEN   = htobe32(8);


	leu32a(length,msg.LOAD+0);
	leu32a(start, msg.LOAD+4);
	int s = send(sd, TC &msg, 16, 0);	
	
	ssize_t count = recv(sd, TC &msg, 8, 0);
	LEN = be32toh(msg.LEN);
	
	hdr->AS.rawdata = (uint8_t*) realloc(hdr->AS.rawdata,LEN);
	count = 0; 
	while (LEN > (size_t)count) {
		count += recv(sd, TC hdr->AS.rawdata+count, LEN-count, 0);
	}	
	
	hdr->AS.first = start;
if (VERBOSE_LEVEL>8) fprintf(stdout,"REQ DAT: %"PRIiPTR" %i\n",count,hdr->AS.bpb);
	hdr->AS.length= (hdr->AS.bpb ? (size_t)count/hdr->AS.bpb : length);  
if (VERBOSE_LEVEL>8) fprintf(stdout,"REQ DAT: %"PRIiPTR" %"PRIiPTR"\n",hdr->AS.first,hdr->AS.length);

	return(0);
}


/****************************************************************************************
	REQUEST EVENT TABLE 
 ****************************************************************************************/
int bscs_requ_evt(int sd, HDRTYPE *hdr) {
	mesg_t msg; 
	size_t LEN;

if (VERBOSE_LEVEL>8) fprintf(stdout,"REQ EVT %08x %08x\n",SERVER_STATE, STATE_OPEN_READ);

	if (SERVER_STATE != STATE_OPEN_READ) return(BSCS_ERROR);

	msg.STATE = BSCS_VERSION_01 | BSCS_REQU_EVT | BSCS_NO_ERROR | (SERVER_STATE & STATE_MASK);
	msg.LEN   = htobe32(0);
	int s = send(sd, TC &msg, 8, 0);

	// wait for reply 
	s = recv(sd, TC &msg, 8, 0);
	LEN = be32toh(msg.LEN);

if (VERBOSE_LEVEL>8) fprintf(stdout,"REQ EVT: %i %"PRIiPTR" \n",s,LEN);

	if (LEN>0) {
	    	hdr->AS.rawEventData = (uint8_t*)realloc(hdr->AS.rawEventData,LEN);
		int count = 0; 
		while (LEN > (size_t)count) {
			count += recv(sd, TC hdr->AS.rawEventData+count, LEN-count, 0);
		}
		rawEVT2hdrEVT(hdr, count); // TODO: replace this function because it is inefficient
   	}

if (VERBOSE_LEVEL>8) fprintf(stdout,"REQ EVT: %i %"PRIiPTR" \n",s,LEN);
#if 0 
			uint8_t *buf = hdr->AS.rawEventData; 
			if (hdr->VERSION < 1.94) {
				if (buf[1] | buf[2] | buf[3])
					hdr->EVENT.SampleRate = buf[1] + (buf[2] + buf[3]*256.0)*256.0; 
				else {
					fprintf(stdout,"Warning GDF v1: SampleRate in Eventtable is not set in %s !!!\n",hdr->FileName);
					hdr->EVENT.SampleRate = hdr->SampleRate; 
				}	
				hdr->EVENT.N = leu32p(buf + 4);
			}	
			else	{
				hdr->EVENT.N = buf[1] + (buf[2] + buf[3]*256)*256; 
				hdr->EVENT.SampleRate = lef32p(buf + 4);
			}	
			int sze = (buf[0]>1) ? 12 : 6;
	
	 		hdr->EVENT.POS = (uint32_t*) realloc(hdr->EVENT.POS, hdr->EVENT.N*sizeof(*hdr->EVENT.POS) );
			hdr->EVENT.TYP = (uint16_t*) realloc(hdr->EVENT.TYP, hdr->EVENT.N*sizeof(*hdr->EVENT.TYP) );
#endif 


	return(0);    	
}

/****************************************************************************************
	PUT FILE 
 ****************************************************************************************/
int bscs_put_file(int sd, char *filename) {

	size_t LEN;
	mesg_t msg;


	if (SERVER_STATE != STATE_OPEN_WRITE_HDR) return(BSCS_ERROR);

	struct stat FileBuf;
	stat(filename,&FileBuf);
	LEN = FileBuf.st_size;

if (VERBOSE_LEVEL>8) fprintf(stdout,"PUT FILE(1) %s\n",filename);

//	int sdi = open(filename,O_RDONLY);
//	if (sdi<=0) return(BSCS_ERROR);

	FILE *fid = fopen(filename,"r");
	if (fid==NULL) return(BSCS_ERROR);


	msg.STATE = BSCS_VERSION_01 | BSCS_PUT_FILE | STATE_OPEN_WRITE_HDR | BSCS_NO_ERROR;
	msg.LEN   = htobe32(LEN);
	int s1 = send(sd, TC &msg, 8, 0);	

//if (VERBOSE_LEVEL>8) fprintf(stdout,"PUT FILE(2) %i %i\n",LEN,sdi);

	const unsigned BUFLEN = 1024;
	char buf[BUFLEN]; 
	size_t count = 0;
	while (count<LEN) {
		//size_t len  = read(sdi, buf, min(LEN-count,BUFLEN));
		size_t len  = fread( buf, 1, min(LEN - count, BUFLEN), fid);
		count += send(sd, buf, len, 0);
//		fprintf(stdout,"\b\b\b\b%02i%% ",100.0*count/LEN);
	}
	//close(sdi); 
	fclose(fid); 

	if (LEN-count) fprintf(stdout,"file size and number of sent bytes do not fit");

	// wait for reply 
	count = recv(sd, TC &msg, 8, 0);
	LEN = be32toh(msg.LEN);
	SERVER_STATE = msg.STATE & STATE_MASK;

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%"PRIiPTR" LEN=%"PRIiPTR" %08x %08x %08x %08x\n",count,LEN,msg.STATE, BSCS_PUT_FILE | BSCS_REPLY , STATE_INIT, (BSCS_VERSION_01 | BSCS_PUT_FILE | BSCS_REPLY | STATE_INIT | BSCS_NO_ERROR));

	if ((LEN==0) && (msg.STATE==(BSCS_VERSION_01 | BSCS_PUT_FILE | BSCS_REPLY | STATE_INIT | BSCS_NO_ERROR)) ) 
		// close without error 
		return(0);

	if (LEN>0) 
		return (BSCS_VERSION_01 | BSCS_PUT_FILE | BSCS_REPLY | STATE_INIT | BSCS_ERROR);

 	// invalid packet or error opening file 
	return(msg.STATE);
}

/****************************************************************************************
	GET FILE 
 ****************************************************************************************/
int bscs_get_file(int sd, uint64_t ID, char *filename) {

	size_t LEN;
	mesg_t msg;

#if _WIN32 
	mode_t mode = S_IRUSR | S_IWUSR;
#else
	mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
#endif 	

	int sdo = creat(filename, mode);
	
	if (VERBOSE_LEVEL>7) fprintf(stdout,"get file (1) %i\n",sdo);

	msg.STATE = BSCS_VERSION_01 | BSCS_GET_FILE | STATE_INIT | BSCS_NO_ERROR;
	leu64a(ID, &msg.LOAD);
	LEN = BSCS_ID_BITLEN>>3;
	msg.LEN   = htobe32(LEN);
	int s = send(sd, TC &msg, LEN+8, 0);	
	
	if (VERBOSE_LEVEL>7) fprintf(stdout,"get file (1)\n");

	// wait for reply 
	ssize_t count = recv(sd, TC &msg, 8, 0);
	LEN = be32toh(msg.LEN);
	SERVER_STATE = msg.STATE & STATE_MASK;

	if (VERBOSE_LEVEL>7) fprintf(stdout,"get file (3) %"PRIiPTR"\n",LEN);

	const unsigned BUFLEN = 1024;
	char buf[BUFLEN]; 
	count = 0; 
	size_t len = 0; 
	while ((size_t)count < LEN) {
		size_t len = recv(sd, buf, min(LEN - count,BUFLEN), 0);
		count+=write(sdo,buf,len);
	}
	if (VERBOSE_LEVEL>7) fprintf(stdout,"get file (1) %"PRIiPTR"\n",count);

	close(sdo); 
	if (LEN-count)	
		return(msg.STATE);
	else
		// close without error 
		return(0);

}

/****************************************************************************************
	NO OPERATION 
 ****************************************************************************************/
int bscs_nop(int sd) {

	size_t LEN;
	mesg_t msg;
	
	msg.STATE = BSCS_NOP | SERVER_STATE;
	msg.LEN = htobe32(0);
	int s = send(sd, TC &msg, 8, 0);
	

}

