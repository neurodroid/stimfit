#include "./zoom.h"

YZoom YZoom::operator*( double factor ) {
   return YZoom( int(startPosY*(double)factor), yZoom*factor, isLogScaleY ); 
}


XZoom XZoom::operator*( double factor ) {
   return XZoom( int(startPosX*(double)factor), xZoom*factor, isLogScaleX ); 
}
