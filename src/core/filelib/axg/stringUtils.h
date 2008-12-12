
#ifndef STRINGUTILS_H
#define STRINGUTILS_H

void PascalToCString( unsigned char *string );
void CToPascalString( unsigned char *string );
void UnicodeToCString( unsigned char *string, const int stringBytes );
void CStringToUnicode( unsigned char *string, const int stringBytes );

#endif
