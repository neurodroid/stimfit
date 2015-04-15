#include "./zoom.h"

YZoom YZoom::operator*( double factor ) {
   return YZoom( long(startPosY*(double)factor), yZoom*factor, isLogScaleY ); 
}


XZoom XZoom::operator*( double factor ) {
   return XZoom( long(startPosX*(double)factor), xZoom*factor, isLogScaleX ); 
}
