/* ----------------------------------------------------------------------------------

   AxoGraph_ReadWrite : functions for reading and writing AxoGraph data files.

   See also : AxoGraph_ReadWrite.h

   This source code and the AxoGraph data file format are in the public domain.

   ---------------------------------------------------------------------------------- */

#include <string.h>
#include <stdlib.h>
#include <sstream>

#include "stringUtils.h"
#include "byteswap.h"

#include "AxoGraph_ReadWrite.h"

int AG_GetFileFormat( filehandle refNum, int *fileFormat )
{
    *fileFormat = 0;

    // Read the file header
    int posn = 0;
    int result = SetFilePosition( refNum, posn );		// Position the mark at start
    if ( result )
        return result;

    // Read the 4-byte prefix present in all AxoGraph file formats
    unsigned char AxoGraphFileID[4];
    AXGLONG bytes = 4;	// 4 byte identifier
    result = ReadFromFile( refNum, bytes, AxoGraphFileID );
    if ( result )
        return result;

    // Check the prefix
    if ( memcmp( AxoGraphFileID, kAxoGraph4DocType, 4 ) == 0 )
    {
        // Got an AxoGraph version 4 format file. Read the file type.
        short version;
        bytes = sizeof( short );
        result = ReadFromFile( refNum, bytes, &version );
        if ( result )
            return result;

#ifdef __LITTLE_ENDIAN__
        ByteSwapShort( &version );
#endif

        if ( version != kAxoGraph_Graph_Format &&
             version != kAxoGraph_Digitized_Format  )
            return kAG_VersionErr;

        // Return the file format
        *fileFormat = version;
    }
    else if ( memcmp( AxoGraphFileID, kAxoGraphXDocType, 4 ) == 0 )
    {
        // Got an AxoGraph X format file. Check the file version.
        AXGLONG version;
        bytes = sizeof( AXGLONG );
        result = ReadFromFile( refNum, bytes, &version );
        if ( result )
            return result;

#ifdef __LITTLE_ENDIAN__
        ByteSwapLong( &version );
#endif

        if ( version < 3 || version > kAxoGraph_X_Format )
        {
            return kAG_VersionErr;
        }

        // update to latest version number
        version = kAxoGraph_X_Format;

        // Return the file format
        *fileFormat = version;
    }
    else
    {
        result = kAG_FormatErr;
    }

    // pass back the result ( = 0 if all went well)
    return result;
}



int AG_GetNumberOfColumns( filehandle refNum, const int fileFormat, AXGLONG *numberOfColumns )
{
    *numberOfColumns = 0;

    if ( fileFormat == kAxoGraph_Digitized_Format || fileFormat == kAxoGraph_Graph_Format )
    {
        // Read the number of columns (short integer in AxoGraph 4 files)
        short nColumns;
        AXGLONG bytes = 2;
        int result = ReadFromFile( refNum, bytes, &nColumns);
        if ( result )
            return result;

#ifdef __LITTLE_ENDIAN__
        ByteSwapShort( &nColumns );
#endif

        *numberOfColumns = nColumns;
        return result;
    }
    else if ( fileFormat == kAxoGraph_X_Format )
    {
        // Read the number of columns (long integer in AxoGraph X files)
        AXGLONG nColumns;
        AXGLONG bytes = 4;
        int result = ReadFromFile( refNum, bytes, &nColumns);
        if ( result )
            return result;

#ifdef __LITTLE_ENDIAN__
        ByteSwapLong( &nColumns );
#endif

        *numberOfColumns = nColumns;
        return result;
    }
    else
    {
        return -1;
    }
}


int AG_ReadColumn( filehandle refNum, const int fileFormat, const int columnNumber, ColumnData *columnData )
{
    // Initialize in case of error during read
    columnData->points = 0;
    columnData->title = "";

    switch ( fileFormat )
    {
     case kAxoGraph_Graph_Format:
         {
             // Read the standard column header
             ColumnHeader columnHeader;
             AXGLONG bytes = sizeof( ColumnHeader );
             int result = ReadFromFile( refNum, bytes, &columnHeader );
             if ( result )
                 return result;

#ifdef __LITTLE_ENDIAN__
             ByteSwapLong( &columnHeader.points );
#endif

             // Retrieve the title and number of points in the column
             columnData->type = FloatArrayType;
             columnData->points = columnHeader.points;
             columnData->title.resize( 80 );
             PascalToCString( columnHeader.title );
             columnData->title = std::string( (char*)columnHeader.title );

             // create a new pointer to receive the data
             AXGLONG columnBytes = columnHeader.points * sizeof( float );
             columnData->floatArray.resize( columnHeader.points );
             if ( columnData->floatArray.empty() )
                 return kAG_MemoryErr;

             // Read in the column's data
             result = ReadFromFile( refNum, columnBytes, &(columnData->floatArray[0]) );

#ifdef __LITTLE_ENDIAN__
             ByteSwapFloatArray( &(columnData->floatArray[0]), columnHeader.points );
#endif

             break;
         }

     case kAxoGraph_Digitized_Format:
         {
             if ( columnNumber == 0 )
             {
                 // Read the column header
                 DigitizedFirstColumnHeader columnHeader;
                 AXGLONG bytes = sizeof( DigitizedFirstColumnHeader );
                 int result = ReadFromFile( refNum, bytes, &columnHeader );
                 if ( result )
                     return result;

#ifdef __LITTLE_ENDIAN__
                 ByteSwapLong( &columnHeader.points );
                 ByteSwapFloat( &columnHeader.firstPoint );
                 ByteSwapFloat( &columnHeader.sampleInterval );
#endif

                 // Retrieve the title, number of points in the column, and sample interval
                 columnData->type = SeriesArrayType;
                 columnData->points = columnHeader.points;
                 columnData->title.resize( 80 );
                 PascalToCString( columnHeader.title );
                 columnData->title = std::string( (char*)columnHeader.title );

                 columnData->seriesArray.firstValue = columnHeader.firstPoint;
                 columnData->seriesArray.increment = columnHeader.sampleInterval;
             }
             else
             {
                 // Read the column header
                 DigitizedColumnHeader columnHeader;
                 AXGLONG bytes = sizeof( DigitizedColumnHeader );
                 int result = ReadFromFile( refNum, bytes, &columnHeader );
                 if ( result )
                     return result;

#ifdef __LITTLE_ENDIAN__
                 ByteSwapLong( &columnHeader.points );
                 ByteSwapFloat( &columnHeader.scalingFactor );
#endif

                 // Retrieve the title and number of points in the column
                 columnData->type = ScaledShortArrayType;
                 columnData->points = columnHeader.points;
                 columnData->title.resize( 80 );
                 PascalToCString( columnHeader.title );
                 columnData->title = std::string( (char*)columnHeader.title );

                 columnData->scaledShortArray.scale = columnHeader.scalingFactor;
                 columnData->scaledShortArray.offset = 0;

                 // create a new pointer to receive the data
                 AXGLONG columnBytes = columnHeader.points * sizeof( short );
                 columnData->scaledShortArray.shortArray.resize( columnHeader.points );
                 if ( columnData->scaledShortArray.shortArray.empty() )
                     return kAG_MemoryErr;

                 // Read in the column's data
                 result = ReadFromFile( refNum, columnBytes, &(columnData->scaledShortArray.shortArray[0]) );

#ifdef __LITTLE_ENDIAN__
                 ByteSwapShortArray( &(columnData->scaledShortArray.shortArray[0]), columnHeader.points );
#endif

             }
             break;
         }

     case kAxoGraph_X_Format:
         {
             // Read the column header
             AxoGraphXColumnHeader columnHeader;
             AXGLONG bytes = sizeof( AxoGraphXColumnHeader );
             int result = ReadFromFile( refNum, bytes, &columnHeader );
             if ( result )
                 return result;

#ifdef __LITTLE_ENDIAN__
             ByteSwapLong( &columnHeader.points );
             ByteSwapLong( &columnHeader.dataType );
             ByteSwapLong( &columnHeader.titleLength );
#endif

             // Retrieve the column type and number of points in the column
             columnData->type = (ColumnType)columnHeader.dataType;
             columnData->points = columnHeader.points;

             // sanity check on column type
             if ( columnData->type < 0 || columnData->type > 14 )
                 return -1;

             // Read the column title
             columnData->titleLength = columnHeader.titleLength;
             // columnData->title.resize( columnHeader.titleLength );
             std::vector< unsigned char > charBuffer( columnHeader.titleLength, '\0' );
             result = ReadFromFile( refNum, columnHeader.titleLength, &charBuffer[0] );
             if ( result )
                 return result;
             // Copy characters one by one into title (tedious but safe)
             for (std::vector< unsigned char >::const_iterator c = charBuffer.begin()+1; c < charBuffer.end(); c += 2) {
                 columnData->title += char(*c);
             }
             // UnicodeToCString( columnData->title, columnData->titleLength );

             switch ( columnHeader.dataType )
             {
              case ShortArrayType:
                  {
                      // create a new pointer to receive the data
                      AXGLONG columnBytes = columnHeader.points * sizeof( short );
                      columnData->shortArray.resize( columnHeader.points );
                      if ( columnData->shortArray.empty() )
                          return kAG_MemoryErr;

                      // Read in the column's data
                      result = ReadFromFile( refNum, columnBytes, &(columnData->shortArray[0]) );

#ifdef __LITTLE_ENDIAN__
                      ByteSwapShortArray( &(columnData->shortArray[0]), columnHeader.points );
#endif

                      break;
                  }
              case IntArrayType:
                  {
                      // create a new pointer to receive the data
                      AXGLONG columnBytes = columnHeader.points * sizeof( int );
                      columnData->intArray.resize( columnHeader.points );
                      if ( columnData->intArray.empty() )
                          return kAG_MemoryErr;

                      // Read in the column's data
                      result = ReadFromFile( refNum, columnBytes, &(columnData->intArray[0]) );

#ifdef __LITTLE_ENDIAN__
                      ByteSwapLongArray( (AXGLONG *)&(columnData->intArray[0]), columnHeader.points );
#endif

                      break;
                  }
              case FloatArrayType:
                  {
                      // create a new pointer to receive the data
                      AXGLONG columnBytes = columnHeader.points * sizeof( float );
                      columnData->floatArray.resize( columnHeader.points );
                      if ( columnData->floatArray.empty() )
                          return kAG_MemoryErr;

                      // Read in the column's data
                      result = ReadFromFile( refNum, columnBytes, &(columnData->floatArray[0]) );

#ifdef __LITTLE_ENDIAN__
                      ByteSwapFloatArray( &(columnData->floatArray[0]), columnHeader.points );
#endif

                      break;
                  }
              case DoubleArrayType:
                  {
                      // create a new pointer to receive the data
                      AXGLONG columnBytes = columnHeader.points * sizeof( double );
                      columnData->doubleArray.resize( columnHeader.points );
                      if ( columnData->doubleArray.empty() )
                          return kAG_MemoryErr;

                      // Read in the column's data
                      result = ReadFromFile( refNum, columnBytes, &(columnData->doubleArray[0]) );

#ifdef __LITTLE_ENDIAN__
                      ByteSwapDoubleArray( &(columnData->doubleArray[0]), columnHeader.points );
#endif

                      break;
                  }
              case SeriesArrayType:
                  {
                      SeriesArray seriesParameters;
                      AXGLONG bytes = sizeof( SeriesArray );
                      result = ReadFromFile( refNum, bytes, &seriesParameters );

#ifdef __LITTLE_ENDIAN__
                      ByteSwapDouble( &seriesParameters.firstValue );
                      ByteSwapDouble( &seriesParameters.increment );
#endif

                      columnData->seriesArray.firstValue = seriesParameters.firstValue;
                      columnData->seriesArray.increment = seriesParameters.increment;
                      break;
                  }
              case ScaledShortArrayType:
                  {
                      double scale, offset;
                      AXGLONG bytes = sizeof( double );
                      result = ReadFromFile( refNum, bytes, &scale );
                      result = ReadFromFile( refNum, bytes, &offset );
#ifdef __LITTLE_ENDIAN__
                      ByteSwapDouble( &scale );
                      ByteSwapDouble( &offset );
#endif

                      columnData->scaledShortArray.scale = scale;
                      columnData->scaledShortArray.offset = offset;

                      // create a new pointer to receive the data
                      AXGLONG columnBytes = columnHeader.points * sizeof( short );
                      columnData->scaledShortArray.shortArray.resize( columnHeader.points );
                      if ( columnData->scaledShortArray.shortArray.empty() )
                          return kAG_MemoryErr;

                      // Read in the column's data
                      result = ReadFromFile( refNum, columnBytes, &(columnData->scaledShortArray.shortArray[0]) );

#ifdef __LITTLE_ENDIAN__
                      ByteSwapShortArray( &(columnData->scaledShortArray.shortArray[0]), columnHeader.points );
#endif

                      break;
                  }
             }
         }
         break;
     default:
         {
             return -1;
         }
    }

    return 0;

}

std::string AG_ReadComment( filehandle refNum )
{
    // File comment
    std::ostringstream comment; comment << "\0";

    AXGLONG comment_size = 0;
    int result = ReadFromFile( refNum, sizeof(AXGLONG), &comment_size );
    if ( result )
        return comment.str();
#ifdef __LITTLE_ENDIAN__
    ByteSwapLong( &comment_size );
#endif

    if (comment_size > 0) {
        std::vector< unsigned char > charBuffer( comment_size, '\0' );
        result = ReadFromFile( refNum, comment_size, &charBuffer[0] );
        if ( result )
            return comment.str();
        // Copy characters one by one into title (tedious but safe)
        for (std::vector< unsigned char >::const_iterator c = charBuffer.begin()+1; c < charBuffer.end(); c += 2) {
            comment << char(*c);
        }
    }
    

    return comment.str();
}

std::string AG_ParseDate( const std::string& notes ) {
    int datepos = notes.find("Created on ");
    if (datepos+11 < notes.length()) {
        std::string full = notes.substr(datepos+11);
        return full.substr(0, full.find('\n'));
    } else {
        return std::string();
    }
}

std::string AG_ParseTime( const std::string& notes ) {
    int datepos = notes.find("acquisition at ");
    if (datepos+15 < notes.length()) {
        std::string full = notes.substr(datepos+15);
        return full.substr(0, full.find('\n'));
    } else {
        return std::string();
    }
}

std::string AG_ReadNotes( filehandle refNum )
{
    
    // File notes
    std::ostringstream notes; notes << "\0";
    AXGLONG notes_size = 0;
    int result = ReadFromFile( refNum, sizeof(AXGLONG), &notes_size );
    if ( result )
        return notes.str();
#ifdef __LITTLE_ENDIAN__
    ByteSwapLong( &notes_size );
#endif

    if (notes_size > 0) {
        std::vector< unsigned char > charBuffer( notes_size, '\0' );
        result = ReadFromFile( refNum, notes_size, &charBuffer[0] );
        if ( result )
            return notes.str();
        // Copy characters one by one into title (tedious but safe)
        for (std::vector< unsigned char >::const_iterator c = charBuffer.begin()+1; c < charBuffer.end(); c += 2) {
            notes << char(*c);
        }
    }
    return notes.str();
}

std::string AG_ReadTraceHeaders( filehandle refNum ) {

    std::string headers = "\0";
    AXGLONG num_headers = 0;
    int result = ReadFromFile( refNum, sizeof(AXGLONG), &num_headers );
    if ( result )
        return headers;
#ifdef __LITTLE_ENDIAN__
    ByteSwapLong( &num_headers );
#endif

    for (AXGLONG nh=0; nh<num_headers; ++nh) {
        AxoGraphXTraceHeader trace_header;
        result = ReadFromFile( refNum, sizeof(AxoGraphXTraceHeader), &trace_header );
        if ( result ) {
            return headers;
        }
    }

    return headers;
}

int AG_ReadFloatColumn( filehandle refNum, const int fileFormat, const int columnNumber, ColumnData *columnData )
{
    int result = AG_ReadColumn( refNum, fileFormat, columnNumber, columnData );

    // If necessary, convert the columnData to FloatArrayType
    switch ( columnData->type )
    {
     case ShortArrayType:
         {
             // Convert in the column data
             columnData->floatArray.resize( columnData->shortArray.size() );
             std::copy( columnData->shortArray.begin(), columnData->shortArray.end(), columnData->floatArray.begin() );
             columnData->shortArray.resize(0);
             columnData->type = FloatArrayType;
             return result;
         }
     case IntArrayType:
         {
             // Convert in the column data
             columnData->floatArray.resize( columnData->intArray.size() );
             std::copy( columnData->intArray.begin(), columnData->intArray.end(), columnData->floatArray.begin() );
             columnData->intArray.resize(0);
             columnData->type = FloatArrayType;
             return result;
         }
     case FloatArrayType:
         {
             // Don't need to convert
             return result;
         }
     case DoubleArrayType:
         {
             // Convert in the column data
             columnData->floatArray.resize( columnData->doubleArray.size() );
             std::copy( columnData->doubleArray.begin(), columnData->doubleArray.end(), columnData->floatArray.begin() );
             columnData->doubleArray.resize(0);
             columnData->type = FloatArrayType;
             return result;
         }
     case SeriesArrayType:
         {
             // create a new pointer to receive the converted data
             double firstValue = columnData->seriesArray.firstValue;
             double increment = columnData->seriesArray.increment;
             columnData->floatArray.resize( columnData->points );

             // Convert in the column data
             for ( AXGLONG i = 0; i < columnData->points; i++ )
             {
                 columnData->floatArray[i] = firstValue + i * increment;
             }

             columnData->type = FloatArrayType;
             return result;
         }
     case ScaledShortArrayType:
         {
             // create a new pointer to receive the converted data
             double scale = columnData->scaledShortArray.scale;
             double offset = columnData->scaledShortArray.offset;
             columnData->floatArray.resize( columnData->points );

             // Convert in the column data
             for ( AXGLONG i = 0; i < columnData->points; i++ )
             {
                 columnData->floatArray[i] = columnData->scaledShortArray.shortArray[i] * scale + offset;
             }

             // free old short array
             columnData->scaledShortArray.shortArray.resize(0);

             // pass in new float array
             columnData->type = FloatArrayType;
             return result;
         }
     default:
         {
             return result;
         }
    }
}
