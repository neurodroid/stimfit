/*

    Copyright (C) 2021 Alois Schloegl <alois.schloegl@gmail.com>

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


#include <assert.h>
#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef WITH_SQLITE3
#include <sqlite3.h>
#endif
#include "../biosig.h"


#ifdef WITH_SQLITE3
static int callback(void *NotUsed, int argc, char **argv, char **azColName) {
	for(int i=0; i<argc; i++)
		printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
	printf("\n");
	return 0;
}

void printColumnValue(sqlite3_stmt* stmt, int col) {

  int colType = sqlite3_column_type(stmt, col);

  switch(colType) {

    case SQLITE_INTEGER:
         printf("  %3d   ", sqlite3_column_int(stmt, col));
         break;

    case SQLITE_FLOAT:
         printf("  %5.2f", sqlite3_column_double(stmt, col));
         break;

    case SQLITE_TEXT:
         printf("  %-5s", sqlite3_column_text(stmt, col));
         break;

    case SQLITE_NULL:
         printf("  null");
         break;

    case SQLITE_BLOB:
         printf("  blob");
         break;
    }

}
#endif

int sopen_sqlite(HDRTYPE* hdr) {
#ifdef WITH_SQLITE3
	sclose(hdr);

	/* refererence to libsqlite3:
		https://www.sqlite.org/cintro.html
	*/
	const SIZE_zSql=10000;
	char zErrMsg[SIZE_zSql];*zErrMsg=0;
	int rc;
	sqlite3 *db;		/* Database handle */
	sqlite3_blob *Blob;

	if (VERBOSE_LEVEL>6) fprintf(stdout,"%s (line %d): %s(...) libsqlite v%s is used\n", __FILE__,__LINE__,__func__,sqlite3_libversion());

	//sqlite3_initialize();
	rc = sqlite3_open(hdr->FileName, &db);
	if (rc) {
		biosigERROR(hdr, B4C_CANNOT_OPEN_FILE, "can not open (sqlite) file.");
		sqlite3_close(db);
		return;
	}

	if (VERBOSE_LEVEL>6) fprintf(stdout,"%s (line %d): %s(...)rc=%d\n", __FILE__,__LINE__,__func__,rc);

	/*
		Identify whether it is an Cadwell EZDATA file or not
		Cadwell has these tables:
	*/
	char zSql[SIZE_zSql];
	strcpy(zSql, "SELECT COUNT(*) AS x FROM FrameInfo;"
	"SELECT COUNT(*) AS a FROM MiscInfo;"
	"SELECT COUNT(*) AS b FROM TrackInfoSyncRowData;"
	"SELECT COUNT(*) AS c FROM MaxUpdateTick;"
	"SELECT COUNT(*) AS d FROM MiscInfoSyncRowData;"
	"SELECT COUNT(*) AS e FROM syncrowdata;"
	"SELECT COUNT(*) AS f FROM MediaHeader;"
	"SELECT COUNT(*) AS g FROM SchemaUpdateLog;"
	"SELECT COUNT(*) AS h FROM synctable;"
	"SELECT COUNT(*) AS i FROM MediaHeaderSyncRowData;"
	"SELECT COUNT(*) AS j FROM TrackInfo;");

	rc = sqlite3_exec(db, zSql, callback, 0, zErrMsg);
	if( rc != SQLITE_OK ) {
		biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format SQLite: not supported yet - not an ARC(Cadwell) file.");
		fprintf(stderr, "SQL error: %s\n", zErrMsg);
		sqlite3_free(zErrMsg);
	}

	if (VERBOSE_LEVEL>6) fprintf(stdout,"%s (line %d) %s(...) most likely a Cadwell file, because all tables are available.\n",__FILE__,__LINE__,__func__);
	const char *ARC_TableList[]={"FrameInfo", "MiscInfo", "TrackInfoSyncRowData", "MaxUpdateTick", "MiscInfoSyncRowData", "syncrowdata", "MediaHeader", "SchemaUpdateLog", "synctable", "MediaHeaderSyncRowData", "TrackInfo", NULL};

	//  https://www.sqlite.org/c3ref/prepare.html
		/* SQL statement, UTF-8 encoded */
	unsigned int prepFlags = 0;
	sqlite3_stmt *pStmt    = NULL;		/* OUT: Statement handle */
	const char *zTail      = NULL;		/* OUT: Pointer to unused portion of zSql */

	if (VERBOSE_LEVEL>6) fprintf(stdout,"%s (line %d): %s(...)rc=%d\n", __FILE__, __LINE__, __func__, rc);

	for (int k=0; ARC_TableList[k]!=NULL; k++) {
		snprintf(zSql, SIZE_zSql, "SELECT * FROM %s", ARC_TableList[k]);
		rc = sqlite3_prepare_v2(db, zSql, sizeof(zSql), &pStmt, &zTail);

	if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d): %s(...)rc=%d <-prepare(db,\"%s\",...)\n", __FILE__,__LINE__,__func__, rc, zSql);

		int M = 0;
		while (SQLITE_ROW==sqlite3_step(pStmt)) {
			M++;

		if (M==1) for (int col = 0; ; col++) {
			int typ   = sqlite3_column_type(pStmt, col);
			if (typ==SQLITE_NULL) break;
			char *DatabaseName = sqlite3_column_database_name(pStmt,col);
			char *TableName    = sqlite3_column_table_name(pStmt,col);
			char *ColumnName   = sqlite3_column_origin_name(pStmt,col);
			if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d): %s(...)  col%d: %s.%s.%s\n", \
				__FILE__,__LINE__,__func__, col+1, DatabaseName, TableName, ColumnName);
		}

		for (int col = 0; ; col++) {
			int T,I;
			int64_t I64;
			char *S;

		if (M > 10) break;  // this breaks only the inner loop and stopping the output/fprintf, only up to M rows are displayed

			size_t sz = sqlite3_column_bytes(pStmt, col);
			int   typ = sqlite3_column_type(pStmt, col);

		if (typ==SQLITE_NULL) break;

			switch (typ) {
			case SQLITE_INTEGER: {	// 1
				double V = sqlite3_column_int64(pStmt, col);
				if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d): %s(...) r%d,c%d sz%d I<%ld>\n", \
					__FILE__,__LINE__,__func__, M, col+1, sz, V);
				break;
			}
			case SQLITE_FLOAT: {	// 2
				double V = sqlite3_column_double(pStmt, col);
				if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d): %s(...) r%d,c%d sz%d F<%g>\n", \
					__FILE__,__LINE__,__func__, M, col+1, sz, V);
				break;
			}
			case SQLITE_TEXT: {	// 3
				char* V = sqlite3_column_text(pStmt, col);
				if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d): %s(...) r%d,c%d sz%d T<%s>\n", \
					__FILE__,__LINE__,__func__, M, col+1, sz, V);
				break;
			}
			case SQLITE_BLOB: {	// 4
				void* V = sqlite3_column_blob(pStmt, col);
				if (VERBOSE_LEVEL>7)
					fprintf(stdout,"%s (line %d): %s(...) r%d,c%d sz%d 0x%016x\n", \
						__FILE__,__LINE__,__func__, M, col+1, sz, leu64p(V));
				if (VERBOSE_LEVEL>8) {
					char A[33]; A[32]=0;
					for (int k=0; k<sz; k++) {
						int ks = k%32;
						if (ks==0) fprintf(stdout,"\n%08x ",k);
						fprintf(stdout,"%02x ",*(uint8_t*)(V+k));
						A[ks] = *(uint8_t*)(V+k);
						if (!isprint(A[ks])) A[ks]='.';
						if (ks==31) fprintf(stdout,"  %s",A);
					}
					fprintf(stdout,"\n");
				}
				break;
			}
			case SQLITE_NULL: 	// 5
				break;
			otherwise:
				;
			}
//		if (VERBOSE_LEVEL>7) fprintf(stdout,"%s (line %d): %s(...) rows=%d T%d %d %ld %s\n", __FILE__,__LINE__,__func__, M,T,I,I64,S);
		}  // for
		}  // while

		if (VERBOSE_LEVEL>6) fprintf(stdout,"%s (line %d): %s(...) rows=%d\n", __FILE__,__LINE__,__func__, M);

		// https://www.sqlite.org/c3ref/finalize.html
		rc = sqlite3_finalize(pStmt);
		if (VERBOSE_LEVEL>6) fprintf(stdout,"%s (line %d): %s(...)rc=%d <-finalize(...) \n", __FILE__,__LINE__,__func__, rc);
	}

	if (VERBOSE_LEVEL > 6) fprintf(stdout,"%s (line %d) %s(...) rc=%d\n", __FILE__,__LINE__,__func__,rc);

	sqlite3_close(db);
	sqlite3_shutdown();

	/* TODO: 
		- check whether it is a recognized and supported data set (i.e. Cadwell/ARC format)
		- fill in HDRstruct* hdr
		- extract data samples and fill hdr->data
		- fill event table if applicable
		Once these steps are completed, the following line can be removed
	 */
	biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "Format SQLite: not supported yet.");

#else	// WITH_SQLITE3

	biosigERROR(hdr, B4C_FORMAT_UNSUPPORTED, "SOPEN(SQLite): - sqlite format not supported - libbiosig need to be recompiled with libsqlite3 support.");
#endif	// WITH_SQLITE3
}

