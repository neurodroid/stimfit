
#include "stringUtils.h"

// In place string converstion functions

void PascalToCString( axgchar *string )
{
	// First byte of pascal string contains string length
	short stringLength = string[0];
	
	// Shift string left
	for ( short i = 0; i < stringLength; i++ )
		string[i] = string[i+1];
	
	// Append null byte
	string[stringLength] = 0;
}


void CToPascalString( axgchar *string )
{
	// Find first null byte (determine string length)
	short i = 0;
	while ( string[i++] );
	short stringLength = i - 1;
	
	// Shift string right
	for ( short j = stringLength-1; j >= 0; j-- )
		string[j+1] = string[j];
	
	// Insert length byte
	string[0] = stringLength;
}


void UnicodeToCString( axgchar *string, const int stringBytes )
{
	// Construct C string from every second byte
	int stringLength = stringBytes / 2;
	for ( int i = 0; i < stringLength; i++ )
		string[i] = string[i*2+1];
	
	// Append null byte
	string[stringLength] = 0;
}


void CStringToUnicode( axgchar *string, const int stringBytes )
{
	// Expand C string to every second byte
	// Set first byte to null
	int stringLength = stringBytes / 2;
	for ( int i = stringLength-1; i >= 0; i-- )
	{
		string[i*2+1] = string[i];
		string[i*2] = 0;
	}
}


