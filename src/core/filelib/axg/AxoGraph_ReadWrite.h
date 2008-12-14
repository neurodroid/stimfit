#ifndef AXOGRAPH_READEWRITE_H
#define AXOGRAPH_READEWRITE_H

/* ----------------------------------------------------------------------------------

	AxoGraph_ReadWrite : functions for reading and writing AxoGraph data files.

	See also : the example programs which uses these functions, Demo_AxoGraph_ReadWrite
				and Simple_AxoGraph_ReadWrite

	This source code and the AxoGraph data file format are in the public domain.

	To run on little endian hardware (Intel, etc.) __LITTLE_ENDIAN__ must be defined
	This is done automatically under OS X / XCode

----------------------------------------------------------------------------------

The three AxoGraph file formats are descibed here for completeness, but this
information is not needed in order to use the supplied AxoGraph file functions
to read and write binary data. For information about reading and writing graph
display information, see the end of the section on the AxoGraph X file format.


AxoGraph Data File Format
=========================

Header
------
Byte	Type		Contents
0		char[4]		AxoGraph file header identifier = 'AxGr' - same as document type ID
4		short		AxoGraph graph file format ID = 1
6		short		Number of columns to follow


Each column
-----------
Byte	Type		Contents
0		long		Number of points in the column ( columnPoints )
4		char[80]	Column title (Pascal 'String' format) - S.I. units should be in brackets e.g. 'Current (pA)'
84		float		1st Data point
88		float		2nd Data point
..		..			....
..		..			etc.


----------------------------------------------------------------------------------

AxoGraph Digitized Data File Format
===================================

Header
------
Byte	Type		Contents
0		char[4]		AxoGraph file header identifier = 'AxGr' - same as document type ID
4		short		AxoGraph file format ID = 2
6		short		Number of columns to follow


Each column
----------------------
Byte	Type		Contents
0		long		Number of points in the column ( columnPoints )
4		long		Data type
8		char[80]	Column title (Pascal 'String' format) - S.I. units should be in brackets e.g. 'Current (pA)'
84		float		Scaling Factor
88		short		1st Data point
90		short		2nd Data point
..		...			....
..		...			etc.


----------------------------------------------------------------------------------

AxoGraph X Data File Format
===================================

Header
------
Byte	Type		Contents
0		char[4]		AxoGraph file header identifier = 'AxGx' - same as filename extension
4		long		AxoGraph X file format ID = a number between 3 (earliest version) and 6 (latest version)
8		long		Number of columns to follow


Each column
----------------------
Byte	Type		Contents
0		long		Number of points in the column ( columnPoints )
4		long		Column type
8		long		Length of column title in bytes (Unicode - 2 bytes per character)
12		char*		Column title (Unicode 2 byte per char) - S.I. units should be in brackets e.g. 'Current (pA)'
??		??			Byte offset depends on length of column title string.
..		...			Numeric type and layout depend on the column type
..		...			....
..		...			etc.


Six column types are supported...
	4: short
	5: long
	6: float
	7: double
	9: 'series'
	10: 'scaled short'

In the first four column types, data is stored as a simple array of the corresponding type.
The 'scaled short' column type stores data as a 'double' scaling factor and offset, and a 'short' array.
The 'series' column type stores data as a 'double' first value and a 'double' increment.

Prior to AxoGraph X, all graph display information was stored in the 'resource fork' of the file,
and the resource fork format was not documented. In contrast, AxoGraph X has a 'flat' format
with all display information stored immediately following the data columns.
It is safe to simply leave out this information. AxoGraph X will use default parameters
when the file is read in. For greater control of graph appearance when creating a file
it may be necessary to add display format information. When reading in a file,
it may be necessary to access the 'Notes' string. The following is a preliminary description
of the file format used to store important elements of graph display information.
It is not supported in the AxoGraph_ReadWrite example functions.

The Comment and Notes strings are stored immediately after the last data column.
Both are stored in Unicode string format..


Unicode string format
----------------------
	long		Length of string in bytes
	char*		Notes string (Unicode 2 byte per char)
				For Latin1 strings, every second byte is an ASCII character code

Each trace consists of a pair of columns. The trace header specifies the
X and Y column numbers, and other trace-specific information.
'bool' header fields are stored as long int: false = 0, true = 1

The number of traces is stored immediately after the comment and notes strings.

	long		Number of trace headers to follow

Header for each trace
----------------------
	long		X column number
	long		Y column number
	long		Error bar column number or -1 if no error bars

	long		Group number that this column belongs to
	bool		Trace shown? False if trace is hidden

	double		Minimum X data point in this trace
	double		Maximum X data point in this trace (if both are zero, they will be recalculated)
	double		Minimum positive X data point in this trace (used in log-axis format)

	bool		True if X axis data is regularly spaced
	bool		True if X axis data is monotonic (each point > previous point)
	double		Interval between points for regular X axis data

	double		Minimum Y data point in this trace
	double		Maximum Y data point in this trace (if both are zero, they will be recalculated)
	double		Minimum positive Y data point in this trace (used in log-axis format)

	long		Trace color with RGB values serialized into a long int

	bool		True if a line plot joining the data points is displayed
	double		Thickness of the line plot (can be less than 1.0 for fine lines)
	long		Pen style (zero for solid line, non zero for dashed lines)

	bool		True if symbols are displayed
	long		Symbol type
	long		Symbol size (radius in pixels)

	bool		True if some symbols are to be skipped
	bool		True if symbols are to be skipped by distance instead of number of points
	long		Minimum separation of symbols in pixes is previous parameter is true

	bool		True for a histogram plot
	long		Type of histogram (zero for standard solid fill)
	long		Separation between adjacent histogram bars expressed as a percentage of bar width

	bool		True if error bars are displayed
	bool		True if a positive error bar is displayed
	bool		True if a negative error bar is displayed
	long		Error bar width in pixels

---------------------------------------------------------------------------------- */

// uncomment the following line to run on little endian hardware ( byte swaps data before reading or writing )
#define __LITTLE_ENDIAN__

#include "fileUtils.h"

// errors numbers
const short kAG_MemoryErr = -21;
const short kAG_FormatErr = -23;
const short kAG_VersionErr = -24;

// file format id's
const short kAxoGraph_Graph_Format = 1;
const short kAxoGraph_Digitized_Format = 2;
const short kAxoGraph_X_Format = 6;
const short kAxoGraph_X_Digitized_Format = 6;

const unsigned char kAxoGraph4DocType[4] = { 'A', 'x', 'G', 'r' };
const unsigned char kAxoGraphXDocType[4] = { 'a', 'x', 'g', 'x' };

// column header for AxoGraph graph files
struct ColumnHeader
{
	long points;
	unsigned char title[80];
};

// x-axis column header for AxoGraph digitized files
struct DigitizedFirstColumnHeader
{
	long points;
	unsigned char title[80];
	float firstPoint;
	float sampleInterval;
};

// y-axis column header for AxoGraph digitized files
struct DigitizedColumnHeader
{
	long points;
	unsigned char title[80];
	float scalingFactor;
};

// column header for AxoGraph X files
struct AxoGraphXColumnHeader
{
	long points;
	long dataType;
	long titleLength;
};

//============= ColumnData structure ======================

// This enum is copied from AxoGraph X source code
// The only types used for data file columns are...
//   ShortArrayType = 4		IntArrayType = 5
//	 FloatArrayType = 6		DoubleArrayType = 7
//   SeriesArrayType = 9	ScaledShortArrayType = 10
enum ColumnType {
	IntType,
	DoubleType,
	BoolType,
	StringType,
	ShortArrayType,
	IntArrayType,
	FloatArrayType,
	DoubleArrayType,
	BoolArrayType,
	SeriesArrayType,
	ScaledShortArrayType,
	StringArrayType,
	ReferenceType
};

struct SeriesArray {
	double firstValue;
	double increment;
};

struct ScaledShortArray {
	double scale;
	double offset;
	short *shortArray;
};


struct ColumnData {
	ColumnType type;
	long points;
	long titleLength;
	unsigned char *title;
	union {
		short *shortArray;
		int *intArray;
		float *floatArray;
		double *doubleArray;
		SeriesArray seriesArray;
		ScaledShortArray scaledShortArray;
	};
};


// ----------- AxoGraph Read and Write functions -------------

int AG_GetFileFormat( filehandle refNum, int *fileFormat );

//	Check that the file referenced by refNum is an AxoGraph data file
//	and read in the file format. Legal values are 1, 2, or 3,
//  corresponding to AxoGraph Graph, Digitized, or AxoGraph X formats.
//	Called once per file. Returns 0 if all goes well.
//	If an error occurs, returns the result from the file access functions,
//	or kAG_FormatErr if file is not in AxoGraph format,
//	or kAG_VersionErr if the file is of a more recent version than supported by this code.


int AG_GetNumberOfColumns( filehandle refNum, const int fileFormat, long *numberOfColumns );

//	Read in the number of columns to follow in this file.
//	Called once per file. Returns 0 if all goes well.
//	If an error occurs, returns the result from the file access functions,


int AG_ReadColumn( filehandle refNum, const int fileFormat, const int columnNumber, ColumnData *columnData );

//	Read in a column from any AxoGraph data file.
//	Called once for each column in the file.
//	Returns data in a pointer in structure that contains the number of points,
//  the column title, and the column data.
//	This function allocates new pointers of the appropriate size, reads the data into
//	them and returns it in columnData.

int AG_ReadFloatColumn( filehandle refNum, const int fileFormat, const int columnNumber, ColumnData *columnData );

//	Read in a column from any AxoGraph data file.
//  Convert the column data to a float array, regardless of the input column format
//	Called once for each column in the file.
//	Returns data in a pointer in structure that contains the number of points,
//  the column title, and the column data.
//	This function allocates new pointers of the appropriate size, reads the data into
//	them and returns it in columnData.

#endif

