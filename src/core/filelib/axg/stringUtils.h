
#ifndef STRINGUTILS_H
#define STRINGUTILS_H

typedef unsigned char axgchar;

void PascalToCString( axgchar *string );
void CToPascalString( axgchar *string );
void UnicodeToCString( axgchar *string, const int stringBytes );
void CStringToUnicode( axgchar *string, const int stringBytes );

#endif
