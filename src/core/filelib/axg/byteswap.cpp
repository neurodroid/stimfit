// ******************************************************************************************
// Graph Document Routines 
// Copyright © 1996 Dr. John Clements.  All rights reserved. 
// ******************************************************************************************

#include "byteswap.h"


void ByteSwapShort( short *shortNumber )
{
	unsigned short *uShortNumber = ( unsigned short * )shortNumber;
	
	*uShortNumber = ( ( *uShortNumber >> 8 ) | ( *uShortNumber << 8 ) );
}


void ByteSwapLong( AXGLONG *longNumber )
{
	unsigned int *uLongNumber = ( unsigned int * )longNumber;
	
	*uLongNumber = ( ( ( *uLongNumber & 0x000000FF )<<24 ) + ( ( *uLongNumber & 0x0000FF00 )<<8 ) +
					 ( ( *uLongNumber & 0x00FF0000 )>>8 ) +  ( ( *uLongNumber & 0xFF000000 )>>24 ) );
}


void ByteSwapFloat( float *floatNumber )
{
	unsigned int *uLongNumber = ( unsigned int * )floatNumber;
	
	*uLongNumber = ( ( ( *uLongNumber & 0x000000FF )<<24 ) + ( ( *uLongNumber & 0x0000FF00 )<<8 ) +
					 ( ( *uLongNumber & 0x00FF0000 )>>8 ) +  ( ( *uLongNumber & 0xFF000000 )>>24 ) );
}


void ByteSwapDouble( double *doubleNumber )
{
	// cast the double to an array of two unsigned ints
	unsigned int *uLongArray = ( unsigned int * )doubleNumber;
	
	// swap the bytes in each long
	ByteSwapLong( ( AXGLONG * )&uLongArray[0] );
	ByteSwapLong( ( AXGLONG * )&uLongArray[1] );
	
	// swap the two longs
	unsigned int saveLong0 = uLongArray[0];
	uLongArray[0] = uLongArray[1];
	uLongArray[1] = saveLong0;
}


void ByteSwapShortArray( short *shortArray, int arraySize )
{
	for ( int i = 0; i < arraySize; i++ )
	{
		ByteSwapShort( shortArray++ );
	}
}


void ByteSwapLongArray( AXGLONG *longArray, int arraySize )
{
	for ( int i = 0; i < arraySize; i++ )
	{
		ByteSwapLong( longArray++ );
	}
}


void ByteSwapFloatArray( float *floatArray, int arraySize )
{
	for ( int i = 0; i < arraySize; i++ )
	{
		ByteSwapFloat( floatArray++ );
	}
}


void ByteSwapDoubleArray( double *doubleArray, int arraySize )
{
	for ( int i = 0; i < arraySize; i++ )
	{
		ByteSwapDouble( doubleArray++ );
	}
}


