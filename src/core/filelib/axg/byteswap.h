
#ifndef BYTESWAP_H
#define BYTESWAP_H

#include "longdef.h"

//------------------------ Byte Swap Routines  -------------------------

// swap bytes in a short (2 byte) variable
void ByteSwapShort( short *shortNumber );

// swap bytes in an int (4 byte) variable
void ByteSwapLong( AXGLONG *longNumber );

// swap bytes in a float (4 byte) variable
void ByteSwapFloat( float *floatNumber );

// swap bytes in a double (8 byte) variable
void ByteSwapDouble( double *doubleNumber );

// swap bytes in a short (2 byte) array
void ByteSwapShortArray( short *shortArray, int arraySize );

// swap bytes in an int (4 byte) array
void ByteSwapLongArray( AXGLONG *longArray, int arraySize );

// swap bytes in a float (4 byte) array
void ByteSwapFloatArray( float *floatArray, int arraySize );

// swap bytes in a double (8 byte) array
void ByteSwapDoubleArray( double *doubleArray, int arraySize );


#endif
