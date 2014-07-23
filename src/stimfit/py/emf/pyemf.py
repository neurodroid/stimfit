#!/usr/bin/env python

"""

pyemf is a pure python module that provides a cross-platform ability
to generate enhanced metafiles (.emf files), a vector graphics format
defined by the ECMA-234 standard.  Enhanced metafiles are a natively
supported image and scalable clip-art format in the OpenOffice suite
of tools and in Windows applications.

U{ECMA-234<http://www.ecma-international.org/publications/standards/Ecma-234.htm>}
is the published interface for enhanced metafiles, which is also a
file-based representation of the Windows graphics device interface.
This API follows most of the naming conventions of ECMA-234, and most
of the parameter lists of methods are the same as their ECMA-234
equivalents.  The primary difference is that pyemf has extended the
API to be object-oriented based on the class L{EMF}.  So, while in
ECMA-234 the first argument is generally the device context, here in
pyemf it is implicit in the class instance.

ECMA-234 defines a lot of constants (mostly integers that are used as
flags to various functions) that pyemf defines as module level
variables.  So, rather than pollute your global namespace, it is
recommended that you use C{import pyemf} rather than C{from pyemf
import *}.

Introduction
============

To use pyemf in your programs, you L{instantiate<EMF.__init__>} an
L{EMF} object, draw some stuff using its methods, and save the file.
An example::

  #!/usr/bin/env python

  import pyemf

  width=8.0
  height=6.0
  dpi=300

  emf=pyemf.EMF(width,height,dpi)
  thin=emf.CreatePen(pyemf.PS_SOLID,1,(0x01,0x02,0x03))
  emf.SelectObject(thin)
  emf.Polyline([(0,0),(width*dpi,height*dpi)])
  emf.Polyline([(0,height*dpi),(width*dpi,0)])
  emf.save("test-1.emf")

This small program creates a 8in x 6in EMF at 300 dots per inch, and
draws two lines connecting the opposite corners.  This simple test is
available as C{test-1.py} in the C{examples} directory of the pyemf
distribution.  There are many other small test programs to demonstrate
other features of the EMF class.


Naming Conventions in pyemf
===========================

Methods that belong to ECMA-234 are C{CamelCased} starting with a
capital letter.  Methods that apply to the operation of the L{EMF}
class itself (i.e. L{load<EMF.load>} and L{save<EMF.save>}) are
C{lower} cased.  Constants described in L{pyemf} that are used as
parameters are C{ALL_UPPER_CASE}.


Coordinate System
=================

Coordinates are addressed a coordinate system called B{page space} by
integer pixels in a horizontal range (increasing to the right) from C{0}
to C{width*density}, and vertically (from the top down) C{0} to
C{height*density}.  Density is either dots per inch if working in
english units, or dots per millimeter if working in metric.

World and Page Space
--------------------

Note that there are four coordinate spaces used by GDI: world, page,
device, and physical device.  World and page are the same, unless a
world transform (L{SetWorldTransform<EMF.SetWorldTransform>},
L{ModifyWorldTransform<EMF.ModifyWorldTransform>}) is used.  In that
case, you operate in world space (that is transformed into page space
by multiplying by the transformation matrix), and it could be scaled
differently.


Experimental Coordinate System
------------------------------

Experimental support for device coordinates is available through
L{SetMapMode<EMF.SetMapMode>} and the various Window and Viewport
methods.  Device coordinates are referenced by physical dimensions
corresponding to the mapping mode currently used.  [The methods work
correctly (in the sense that they generate the correct records in the
metafile) and the API won't change, but it's not clear to me what the
parameters should do.]


Drawing Characteristics
=======================

GDI has a concept of the B{current object} for the each of the three
drawing characteristics: line style, fill style, and font.  Once a
characteristic is made current using
L{SelectObject<EMF.SelectObject>}, it remains current until it is
replaced by another call to SelectObject.  Note that a call to
SelectObject only affects that characteristic, and not the other two,
so changing the line style doesn't effect the fill style or the font.

Additionally, there is a set of B{stock objects} retrievable with
L{GetStockObject<EMF.GetStockObject>} that should be available on any
system capable of rendering an EMF.


Colors
------

A quick note about color.  Colors in pyemf are specified one of three
ways:

  - (r,g,b) tuple, where each component is a integer between 0 and 255 inclusive.

  - (r,g,b) tuple, where each component is a float between 0.0 and 1.0 inclusive.

  - packed integer created by a call to L{RGB}


Line Styles
-----------

Line styles are created by L{CreatePen<EMF.CreatePen>} and specify the
style, width, and color.

Note that there is a NULL_PEN stock object if you don't actually want
to draw a line with a drawing primitive.


Fill Styles
-----------

Polygon fill styles are created by
L{CreateSolidBrush<EMF.CreateSolidBrush>} and theoretically
L{CreateHatchBrush<EMF.CreateHatchBrush>}, although the latter doesn't
seem to be supported currently in OpenOffice.  So, reliably we can only
use CreateSolidBrush and thus can only specify a fill color and not a
fill pattern.

Note that there is a stock object NULL_BRUSH that doesn't fill, useful
if you want to only draw an outline of a primitive that is normally
filled.

An interesting side-note is that there is no direct support for
gradients in EMF.  Examining some .emfs that do have gradients shows
that Windows produces them using clipping regions and subdividing the
object into areas of a single color an drawing slices of the
individual color.  Better support for clipping regions is the subject
of a future release of pyemf, but they also don't seem to work well in
OpenOffice, so it hasn't been a high priority.


Fonts
-----

L{CreateFont<EMF.CreateFont>} requires a large number of parameters,
the most important being the height, the rotation, and the name.  Note
that the height can either be specified as a positive or negative
integer, where negative means use that value as the average I{glyph}
height and positive means use the value as the average I{cell} height.
Since a glyph is contained within a cell, the negative value will
yield a slightly larger font when rendered on screen.

Note that the two rotation values must specify the same angle.

Also note that font color is not part of a
L{SelectObject<EMF.SelectObject>} characteristic.  It is specified
using the separate method L{SetTextColor<EMF.SetTextColor>}.
L{SetBkMode<EMF.SetBkMode>} and L{SetBkColor<EMF.SetBkColor>} are
supposed to work with text, but in my testing with OpenOffice it hasn't been
consistent.  I tend to just C{SetBkMode(pyemf.TRANSPARENT)} and leave
it at that.


Drawing
=======

The methods listed under B{Drawing Primitives} below use either the
current line style or the current fill style (or both).  Any primitive
that creates a closed figure (L{Polygon<EMF.Polygon>},
L{PolyPolygon<EMF.PolyPolygon>}, L{Rectangle<EMF.Rectangle>},
L{RoundRect<EMF.RoundRect>}, L{Ellipse<EMF.Ellipse>},
L{Chord<EMF.Chord>}, and L{Pie<EMF.Pie>}) will use both the line and
fill style.  Others (L{Polyline<EMF.Polyline>},
L{PolyPolyline<EMF.PolyPolyline>} and L{Arc<EMF.Arc>}) will only use
the line style, excepting L{SetPixel<EMF.SetPixel>} which doesn't use either.


Paths
=====

To create more complicated shapes, the B{Path Primitives} are used.  A
path is started with a call to L{BeginPath<EMF.BeginPath>} and the
initial point should be set with L{MoveTo<EMF.MoveTo>}.  Calls to
L{LineTo<EMF.LineTo>}, L{PolylineTo<EMF.PolylineTo>},
L{ArcTo<EMF.ArcTo>}, and L{PolyBezierTo<EMF.PolyBezierTo>} extend the
path.  L{CloseFigure<EMF.CloseFigure>} should be used to connect the
final point to the starting point, otherwise the path may be filled
incorrectly.  L{EndPath<EMF.EndPath>} then completes the path, and it
may be outlined with L{StrokePath<EMF.StrokePath>}, filled with
L{FillPath<EMF.FillPath>} or both with
L{StrokeAndFillPath<EMF.StrokeAndFillPath>}.

Note that OpenOffice ignores L{ArcTo<EMF.ArcTo>} in terms of path
continuity -- the arc is drawn, but it is not connected to the path.

Note that L{SelectClipPath<EMF.SelectClipPath>} is broken in OpenOffice.


Coordinate System Transformation
================================

You might have noticed that methods like L{Ellipse<EMF.Ellipse>} and
L{Rectangle<EMF.Rectangle>} can only create objects that are aligned
with the X-Y axis.  This would be a real limitation without some way
to rotate the figures.  L{SetWorldTransform<EMF.SetWorldTransform>}
and L{ModifyWorldTransform<EMF.ModifyWorldTransform>} provide this.
These methods provide a generalized linear transformation that can
translate, rotate, scale and shear subsequent graphics operations.

These methods aren't required by the ECMA-234 spec, which may explain
why their support in OpenOffice is mixed.  Drawing primitives and
paths seem to be supported and are transformed, but text is not
(though it should be).


@author: $author
@version: $version
"""

__extra_epydoc_fields__ = [
                           ('gdi', 'GDI Command', 'GDI Commands'),
                           ('oo', 'OpenOffice Support'),
                          ]


import os,sys,re
import struct
from cStringIO import StringIO
import copy

# setup.py requires that these be defined, and the OnceAndOnlyOnce
# principle is used here.  This is the only place where these values
# are defined in the source distribution, and everything else that
# needs this should grab it from here.
__version__ = "2.0.0"
__author__ = "Rob McMullen"
__author_email__ = "robm@users.sourceforge.net"
__url__ = "http://pyemf.sourceforge.net"
__download_url__ = "http://sourceforge.net/project/showfiles.php?group_id=148144"
__description__ = "Pure Python Enhanced Metafile Library"
__keywords__ = "graphics, scalable, vector, image, clipart, emf"
__license__ = "LGPL"


# Reference: libemf.h
# and also wine: http://cvs.winehq.org/cvsweb/wine/include/wingdi.h

# Brush styles
BS_SOLID	    = 0
BS_NULL		    = 1
BS_HOLLOW	    = 1
BS_HATCHED	    = 2
BS_PATTERN	    = 3
BS_INDEXED	    = 4
BS_DIBPATTERN	    = 5
BS_DIBPATTERNPT	    = 6
BS_PATTERN8X8	    = 7
BS_DIBPATTERN8X8    = 8
BS_MONOPATTERN      = 9

# Hatch styles
HS_HORIZONTAL       = 0
HS_VERTICAL         = 1
HS_FDIAGONAL        = 2
HS_BDIAGONAL        = 3
HS_CROSS            = 4
HS_DIAGCROSS        = 5

# mapping modes
MM_TEXT = 1
MM_LOMETRIC = 2
MM_HIMETRIC = 3
MM_LOENGLISH = 4
MM_HIENGLISH = 5
MM_TWIPS = 6
MM_ISOTROPIC = 7
MM_ANISOTROPIC = 8
MM_MAX = MM_ANISOTROPIC

# background modes
TRANSPARENT = 1
OPAQUE = 2
BKMODE_LAST = 2

# polyfill modes
ALTERNATE = 1
WINDING = 2
POLYFILL_LAST = 2

# line styles and options
PS_SOLID         = 0x00000000
PS_DASH          = 0x00000001
PS_DOT           = 0x00000002
PS_DASHDOT       = 0x00000003
PS_DASHDOTDOT    = 0x00000004
PS_NULL          = 0x00000005
PS_INSIDEFRAME   = 0x00000006
PS_USERSTYLE     = 0x00000007
PS_ALTERNATE     = 0x00000008
PS_STYLE_MASK    = 0x0000000f

PS_ENDCAP_ROUND  = 0x00000000
PS_ENDCAP_SQUARE = 0x00000100
PS_ENDCAP_FLAT   = 0x00000200
PS_ENDCAP_MASK   = 0x00000f00

PS_JOIN_ROUND    = 0x00000000
PS_JOIN_BEVEL    = 0x00001000
PS_JOIN_MITER    = 0x00002000
PS_JOIN_MASK     = 0x0000f000

PS_COSMETIC      = 0x00000000
PS_GEOMETRIC     = 0x00010000
PS_TYPE_MASK     = 0x000f0000
 
# Stock GDI objects for GetStockObject()
WHITE_BRUSH         = 0
LTGRAY_BRUSH        = 1
GRAY_BRUSH          = 2
DKGRAY_BRUSH        = 3
BLACK_BRUSH         = 4
NULL_BRUSH          = 5
HOLLOW_BRUSH        = 5
WHITE_PEN           = 6
BLACK_PEN           = 7
NULL_PEN            = 8
OEM_FIXED_FONT      = 10
ANSI_FIXED_FONT     = 11
ANSI_VAR_FONT       = 12
SYSTEM_FONT         = 13
DEVICE_DEFAULT_FONT = 14
DEFAULT_PALETTE     = 15
SYSTEM_FIXED_FONT   = 16
DEFAULT_GUI_FONT    = 17

STOCK_LAST          = 17

# Text alignment
TA_NOUPDATECP       = 0x00
TA_UPDATECP         = 0x01
TA_LEFT             = 0x00
TA_RIGHT            = 0x02
TA_CENTER           = 0x06
TA_TOP              = 0x00
TA_BOTTOM           = 0x08
TA_BASELINE         = 0x18
TA_RTLREADING       = 0x100
TA_MASK             = TA_BASELINE+TA_CENTER+TA_UPDATECP+TA_RTLREADING

# lfWeight values
FW_DONTCARE         = 0
FW_THIN             = 100
FW_EXTRALIGHT       = 200
FW_ULTRALIGHT       = 200
FW_LIGHT            = 300
FW_NORMAL           = 400
FW_REGULAR          = 400
FW_MEDIUM           = 500
FW_SEMIBOLD         = 600
FW_DEMIBOLD         = 600
FW_BOLD             = 700
FW_EXTRABOLD        = 800
FW_ULTRABOLD        = 800
FW_HEAVY            = 900
FW_BLACK            = 900

# lfCharSet values
ANSI_CHARSET          = 0   # CP1252, ansi-0, iso8859-{1,15}
DEFAULT_CHARSET       = 1
SYMBOL_CHARSET        = 2
SHIFTJIS_CHARSET      = 128 # CP932
HANGEUL_CHARSET       = 129 # CP949, ksc5601.1987-0
HANGUL_CHARSET        = HANGEUL_CHARSET
GB2312_CHARSET        = 134 # CP936, gb2312.1980-0
CHINESEBIG5_CHARSET   = 136 # CP950, big5.et-0
GREEK_CHARSET         = 161 # CP1253
TURKISH_CHARSET       = 162 # CP1254, -iso8859-9
HEBREW_CHARSET        = 177 # CP1255, -iso8859-8
ARABIC_CHARSET        = 178 # CP1256, -iso8859-6
BALTIC_CHARSET        = 186 # CP1257, -iso8859-13
RUSSIAN_CHARSET       = 204 # CP1251, -iso8859-5
EE_CHARSET            = 238 # CP1250, -iso8859-2
EASTEUROPE_CHARSET    = EE_CHARSET
THAI_CHARSET          = 222 # CP874, iso8859-11, tis620
JOHAB_CHARSET         = 130 # korean (johab) CP1361
MAC_CHARSET           = 77
OEM_CHARSET           = 255

VISCII_CHARSET        = 240 # viscii1.1-1
TCVN_CHARSET          = 241 # tcvn-0
KOI8_CHARSET          = 242 # koi8-{r,u,ru}
ISO3_CHARSET          = 243 # iso8859-3
ISO4_CHARSET          = 244 # iso8859-4
ISO10_CHARSET         = 245 # iso8859-10
CELTIC_CHARSET        = 246 # iso8859-14

FS_LATIN1              = 0x00000001L
FS_LATIN2              = 0x00000002L
FS_CYRILLIC            = 0x00000004L
FS_GREEK               = 0x00000008L
FS_TURKISH             = 0x00000010L
FS_HEBREW              = 0x00000020L
FS_ARABIC              = 0x00000040L
FS_BALTIC              = 0x00000080L
FS_VIETNAMESE          = 0x00000100L
FS_THAI                = 0x00010000L
FS_JISJAPAN            = 0x00020000L
FS_CHINESESIMP         = 0x00040000L
FS_WANSUNG             = 0x00080000L
FS_CHINESETRAD         = 0x00100000L
FS_JOHAB               = 0x00200000L
FS_SYMBOL              = 0x80000000L

# lfOutPrecision values
OUT_DEFAULT_PRECIS      = 0
OUT_STRING_PRECIS       = 1
OUT_CHARACTER_PRECIS    = 2
OUT_STROKE_PRECIS       = 3
OUT_TT_PRECIS           = 4
OUT_DEVICE_PRECIS       = 5
OUT_RASTER_PRECIS       = 6
OUT_TT_ONLY_PRECIS      = 7
OUT_OUTLINE_PRECIS      = 8

# lfClipPrecision values
CLIP_DEFAULT_PRECIS     = 0x00
CLIP_CHARACTER_PRECIS   = 0x01
CLIP_STROKE_PRECIS      = 0x02
CLIP_MASK               = 0x0F
CLIP_LH_ANGLES          = 0x10
CLIP_TT_ALWAYS          = 0x20
CLIP_EMBEDDED           = 0x80

# lfQuality values
DEFAULT_QUALITY        = 0
DRAFT_QUALITY          = 1
PROOF_QUALITY          = 2
NONANTIALIASED_QUALITY = 3
ANTIALIASED_QUALITY    = 4

# lfPitchAndFamily pitch values
DEFAULT_PITCH       = 0x00
FIXED_PITCH         = 0x01
VARIABLE_PITCH      = 0x02
MONO_FONT           = 0x08

FF_DONTCARE         = 0x00
FF_ROMAN            = 0x10
FF_SWISS            = 0x20
FF_MODERN           = 0x30
FF_SCRIPT           = 0x40
FF_DECORATIVE       = 0x50

# Graphics Modes
GM_COMPATIBLE     = 1
GM_ADVANCED       = 2
GM_LAST           = 2

# Arc direction modes
AD_COUNTERCLOCKWISE = 1
AD_CLOCKWISE        = 2

# Clipping paths
RGN_ERROR         = 0
RGN_AND           = 1
RGN_OR            = 2
RGN_XOR           = 3
RGN_DIFF          = 4
RGN_COPY          = 5
RGN_MIN           = RGN_AND
RGN_MAX           = RGN_COPY

# Color management
ICM_OFF   = 1
ICM_ON    = 2
ICM_QUERY = 3
ICM_MIN   = 1
ICM_MAX   = 3

# World coordinate system transformation
MWT_IDENTITY      = 1
MWT_LEFTMULTIPLY  = 2
MWT_RIGHTMULTIPLY = 3


def _round4(num):
    """Round to the nearest multiple of 4 greater than or equal to the
    given number.  EMF records are required to be aligned to 4 byte
    boundaries."""
    return ((num+3)/4)*4

def RGB(r,g,b):
    """
Pack integer color values into a 32-bit integer format.

@param r: 0 - 255 or 0.0 - 1.0 specifying red
@param g: 0 - 255 or 0.0 - 1.0 specifying green
@param b: 0 - 255 or 0.0 - 1.0 specifying blue
@return: single integer that should be used when any function needs a color value
@rtype: int
@type r: int or float
@type g: int or float
@type b: int or float

"""
    if isinstance(r,float):
        r=int(255*r)
    if r>255: r=255
    elif r<0: r=0

    if isinstance(g,float):
        g=int(255*g)
    if g>255: g=255
    elif g<0: g=0

    if isinstance(b,float):
        b=int(255*b)
    if b>255: b=255
    elif b<0: b=0

    return ((b<<16)|(g<<8)|r)

def _normalizeColor(c):
    """
Normalize the input into a packed integer.  If the input is a tuple,
pass it through L{RGB} to generate the color value.

@param c: color
@type c: int or (r,g,b) tuple
@return: packed integer color from L{RGB}
@rtype: int
"""
    if isinstance(c,int):
        return c
    if isinstance(c,tuple) or isinstance(c,list):
        return RGB(*c)
    raise TypeError("Color must be specified as packed integer or 3-tuple (r,g,b)")



# FIXME: do I need DPtoLP and LPtoDP?

class _DC:
    """Device Context state machine.  This is used to simulate the
    state of the GDI buffer so that some user commands can return
    information.  In a real GDI implementation, there'd be lots of
    error checking done, but here we can't do a whole bunch because
    we're outputting to a metafile.  So, in general, we assume
    success.

    Here's Microsoft's explanation of units: http://msdn.microsoft.com/library/default.asp?url=/library/en-us/gdi/cordspac_3qsz.asp

    Window <=> Logical units <=> page space or user addressable
    integer pixel units.

    Viewport <=> Physical units <=> device units and are measured in actual
    dimensions, like .01 mm units.

    There are four coordinate spaces used by GDI: world, page, device,
    and physical device.  World and page are the same, unless a world
    transform is used.  These are addressed by integer pixels.  Device
    coordinates are referenced by physical dimensions corresponding to
    the mapping mode currently used.

    """
    
    def __init__(self,width='6.0',height='4.0',density='72',units='in'):
        self.x=0
        self.y=0

        # list of objects that can be referenced by their index
        # number, called "handle"
        self.objects=[]
        self.objects.append(None) # handle 0 is reserved
        
        # Maintain a stack that contains list of empty slots in object
        # list resulting from deletes
        self.objectholes=[]

        # Reference device size in logical units (pixels)
        self.ref_pixelwidth=1024
        self.ref_pixelheight=768

        # Reference device size in mm
        self.ref_width=320 
        self.ref_height=240

        # physical dimensions are in .01 mm units
        self.width=0
        self.height=0
        if units=='mm':
            self.setPhysicalSize(0,0,int(width*100),int(height*100))
        else:
            self.setPhysicalSize(0,0,int(width*2540),int(height*2540))

        # addressable pixel sizes
        self.pixelwidth=0
        self.pixelheight=0
        self.setPixelSize(0,0,int(width*density),int(height*density))
            
        #self.text_alignment = TA_BASELINE;
        self.text_color = RGB(0,0,0);

        #self.bk_mode = OPAQUE;
        #self.polyfill_mode = ALTERNATE;
        #self.map_mode = MM_TEXT;

        # Viewport origin.  A pixel drawn at (x,y) after the viewport
        # origin has been set to (xv,yv) will be displayed at
        # (x+xv,y+yv).
        self.viewport_x=0
        self.viewport_y=0

        # Viewport extents.  Should density be replaced by
        # self.ref_pixelwidth/self.ref_width?
        self.viewport_ext_x=self.width/100*density
        self.viewport_ext_y=self.height/100*density

        # Window origin.  A pixel drawn at (x,y) after the window
        # origin has been set to (xw,yw) will be displayed at
        # (x-xw,y-yw).
        
        # If both window and viewport origins are set, a pixel drawn
        # at (x,y) will be displayed at (x-xw+xv,y-yw+yv)
        self.window_x=0
        self.window_y=0

        # Window extents
        self.window_ext_x=self.pixelwidth
        self.window_ext_y=self.pixelheight



    def getBounds(self,header):
        """Extract the dimensions from an _EMR._HEADER record."""
        
        self.setPhysicalSize(header.rclFrame_left,header.rclFrame_top,
                             header.rclFrame_right,header.rclFrame_bottom)
        if header.szlMicrometers_cx>0:
            self.ref_width=header.szlMicrometers_cx/10
            self.ref_height=header.szlMicrometers_cy/10
        else:
            self.ref_width=header.szlMillimeters_cx*100
            self.ref_height=header.szlMillimeters_cy*100

        self.setPixelSize(header.rclBounds_left,header.rclBounds_top,
                           header.rclBounds_right,header.rclBounds_bottom)
        self.ref_pixelwidth=header.szlDevice_cx
        self.ref_pixelheight=header.szlDevice_cy

    def setPhysicalSize(self,left,top,right,bottom):
        """Set the physical (i.e. stuff you could measure with a
        meterstick) dimensions."""
        self.width=right-left
        self.height=bottom-top
        self.frame_left=left
        self.frame_top=top
        self.frame_right=right
        self.frame_bottom=bottom

    def setPixelSize(self,left,top,right,bottom):
        """Set the pixel-addressable dimensions."""
        self.pixelwidth=right-left
        self.pixelheight=bottom-top
        self.bounds_left=left
        self.bounds_top=top
        self.bounds_right=right
        self.bounds_bottom=bottom
                

    def addObject(self,emr,handle=-1):
        """Add an object to the handle list, so it can be retrieved
        later or deleted."""
        count=len(self.objects)
        if handle>0:
            # print "Adding handle %s (%s)" % (handle,emr.__class__.__name__.lstrip('_'))
            if handle>=count:
                self.objects+=[None]*(handle-count+1)
            self.objects[handle]=emr
        elif self.objectholes:
            handle=self.objectholes.pop()
            self.objects[handle]=emr
        else:
            handle=count
            self.objects.append(emr)
        return handle

    def removeObject(self,handle):
        """Remove an object by its handle.  Handles can be reused, and
        are reused from lowest available handle number."""
        if handle<1 or handle>=len(self.objects):
            raise IndexError("Invalid handle")
        # print "removing handle %d (%s)" % (handle,self.objects[handle].__class__.__name__.lstrip('_'))
        self.objects[handle]=None
        found=False

        # insert handle in objectholes list, but keep object holes
        # list in sorted order
        i=0
        while i<len(self.objectholes):
            if handle<self.objectholes[i]:
                self.objectholes.insert(i,handle)
                break
            i+=1
        else:
            self.objectholes.append(handle)
        # print self.objectholes

    def popObject(self):
        """Remove last object.  Used mainly in case of error."""
        self.objects.pop()



class _EMR_FORMAT:
    def __init__(self,emr_id,typedef):
        self.typedef=typedef
        self.id=emr_id
        self.fmtlist=[] # list of typecodes
        self.defaults=[] # list of default values
        self.fmt="<" # string for pack/unpack.  little endian
        self.structsize=0

        self.names=[]
        self.namepos={}
        
        self.debug=0

        self.setFormat(typedef)

    def setFormat(self,typedef,default=None):
        if self.debug: print "typedef=%s" % str(typedef)
        if isinstance(typedef,list) or isinstance(typedef,tuple):
            for item in typedef:
                if len(item)==3:
                    typecode,name,default=item
                else:
                    typecode,name=item
                self.appendFormat(typecode,name,default)
        elif typedef:
            raise AttributeError("format must be a list")
        self.structsize=struct.calcsize(self.fmt)
        if self.debug: print "current struct=%s size=%d\n  names=%s" % (self.fmt,self.structsize,self.names)

    def appendFormat(self,typecode,name,default):
        self.fmt+=typecode
        self.fmtlist.append(typecode)
        self.defaults.append(default)
        self.namepos[name]=len(self.names)
        self.names.append(name)





class _EMR_UNKNOWN(object): # extend from new-style class, or __getattr__ doesn't work
    """baseclass for EMR objects"""
    emr_id=0
    emr_typedef=()
    format=None

    twobytepadding='\0'*2
    
    def __init__(self):
        self.iType=self.__class__.emr_id
        self.nSize=0

        self.verbose=0
        
        self.datasize=0
        self.data=None
        self.unhandleddata=None

        # number of padding zeros we had to add because the format was
        # expecting more data
        self.zerofill=0

        # if we've never seen this class before, create a new format.
        # Note that subclasses of classes that we have already seen
        # pick up any undefined class attributes from their
        # superclasses, so we have to check if this is a subclass with
        # a different typedef
        if self.__class__.format==None or self.__class__.emr_typedef != self.format.typedef:
            if self.verbose: print "creating format for %d" % self.__class__.emr_id
            self.__class__.format=_EMR_FORMAT(self.__class__.emr_id,self.__class__.emr_typedef)

        # list of values parsed from the input stream
        self.values=copy.copy(self.__class__.format.defaults)

        # error code.  Currently just used as a boolean
        self.error=0


    def __getattr__(self,name):
        """Return EMR attribute if the name exists in the typedef list
        of the object.  This is only called when the standard
        attribute lookup fails on this object, so we don't have to
        handle the case where name is an actual attribute of self."""
        f=_EMR_UNKNOWN.__getattribute__(self,'format')
        try:
            if name in f.names:
                v=_EMR_UNKNOWN.__getattribute__(self,'values')
                index=f.namepos[name]
                return v[index]
        except IndexError:
            raise IndexError("name=%s index=%d values=%s" % (name,index,str(v)))
        raise AttributeError("%s not defined in EMR object" % name)

    def __setattr__(self,name,value):
        """Set a value in the object, propagating through to
        self.values[] if the name is in the typedef list."""
        f=_EMR_UNKNOWN.__getattribute__(self,'format')
        try:
            if f and name in f.names:
                v=_EMR_UNKNOWN.__getattribute__(self,'values')
                index=f.namepos[name]
                v[index]=value
            else:
                # it's not an automatically serializable item, so store it.
                self.__dict__[name]=value
        except IndexError:
            raise IndexError("name=%s index=%d values=%s" % (name,index,str(v)))

    def hasHandle(self):
        """Return true if this object has a handle that needs to be
        saved in the object array for later recall by SelectObject."""
        return False

    def setBounds(self,bounds):
        """Set bounds of object.  Depends on naming convention always
        defining the bounding rectangle as
        rclBounds_[left|top|right|bottom]."""
        self.rclBounds_left=bounds[0]
        self.rclBounds_top=bounds[1]
        self.rclBounds_right=bounds[2]
        self.rclBounds_bottom=bounds[3]

    def getBounds(self):
        """Return bounds of object, or None if not applicable."""
        return None

    def unserialize(self,fh,itype=-1,nsize=-1):
        """Read data from the file object and, using the format
        structure defined by the subclass, parse the data and store it
        in self.values[] list."""
        if itype>0:
            self.iType=itype
            self.nSize=nsize
        else:
            (self.iType,self.nSize)=struct.unpack("<ii",8)
        if self.nSize>8:
            self.datasize=self.nSize-8
            self.data=fh.read(self.datasize)
            if self.format.structsize>0:
                if self.format.structsize>len(self.data):
                    # we have a problem.  More stuff to unparse than
                    # we have data.  Hmmm.  Fill with binary zeros
                    # till I think of a better idea.
                    self.zerofill=self.format.structsize-len(self.data)
                    self.data+="\0"*self.zerofill
                self.values=list(struct.unpack(self.format.fmt,self.data[0:self.format.structsize]))
            if self.datasize>self.format.structsize:
                self.unserializeExtra(self.data[self.format.structsize:])

    def unserializeOffset(self,offset):
        """Adjust offset to point to correct location in extra data.
        Offsets in the EMR record are from the start of the record, so
        we must subtract 8 bytes for iType and nSize, and also
        subtract the size of the format structure."""
        return offset-8-self.format.structsize-self.zerofill

    def unserializeExtra(self,data):
        """Hook for subclasses to handle extra data in the record that
        isn't specified by the format statement."""
        self.unhandleddata=data
        pass

    def unserializeList(self,fmt,count,data,start):
        fmt="<%d%s" % (count,fmt)
        size=struct.calcsize(fmt)
        vals=list(struct.unpack(fmt,data[start:start+size]))
        #print "vals fmt=%s size=%d: %s" % (fmt,len(vals),str(vals))
        start+=size
        return (start,vals)

    def unserializePoints(self,fmt,count,data,start):
        fmt="<%d%s" % ((2*count),fmt)
        size=struct.calcsize(fmt)
        vals=struct.unpack(fmt,data[start:start+size])
        pairs=[(vals[i],vals[i+1]) for i in range(0,len(vals),2)]
        #print "points size=%d: %s" % (len(pairs),pairs)
        start+=size
        return (start,pairs)
            
    def serialize(self,fh):
        fh.write(struct.pack("<ii",self.iType,self.nSize))
        try:
            fh.write(struct.pack(self.format.fmt,*self.values))
        except struct.error:
            print "!!!!!Struct error:",
            print self
            raise
        self.serializeExtra(fh)

    def serializeOffset(self):
        """Return the initial offset for any extra data that must be
        written to the record.  See L{unserializeOffset}."""
        return 8+self.format.structsize

    def serializeExtra(self,fh):
        """This is for special cases, like writing text or lists.  If
        this is not overridden by a subclass method, it will write out
        anything in the self.unhandleddata string."""
        if self.unhandleddata:
            fh.write(self.unhandleddata)
            

    def serializeList(self,fh,fmt,vals):
        fmt="<%s" % fmt
        for val in vals:
            fh.write(struct.pack(fmt,val))

    def serializePoints(self,fh,fmt,pairs):
        fmt="<2%s" % fmt
        for pair in pairs:
            fh.write(struct.pack(fmt,pair[0],pair[1]))

    def serializeString(self,fh,txt):
        if isinstance(txt,unicode):
            txt=txt.encode('utf-16le')
        fh.write(txt)
        extra=_round4(len(txt))-len(txt)
        if extra>0:
            fh.write('\0'*extra)

    def resize(self):
        before=self.nSize
        self.nSize=8+self.format.structsize+self.sizeExtra()
        if self.verbose and before!=self.nSize:
            print "resize: before=%d after=%d" % (before,self.nSize),
            print self
        if self.nSize%4 != 0:
            print "size error.  Must be divisible by 4"
            print self
            raise TypeError

    def sizeExtra(self):
        """Hook for subclasses before anything is serialized.  This is
        used to return the size of any extra components not in the
        format string, and also provide the opportunity to recalculate
        any changes in size that should be reflected in self.nSize
        before the record is written out."""
        if self.unhandleddata:
            return len(self.unhandleddata)
        return 0

    def str_extra(self):
        """Hook to print out extra data that isn't in the format"""
        return ""

    def str_color(self,val):
        return "red=0x%02x green=0x%02x blue=0x%02x" % ((val&0xff),((val&0xff00)>>8),((val&0xff0000)>>16))

    def str_decode(self,typecode,name):
        val=_EMR_UNKNOWN.__getattr__(self,name)
        if name.endswith("olor"):
            val=self.str_color(val)
        elif typecode.endswith("s"):
            val=val.decode('utf-16le')
        return val
    
    def str_details(self):
        txt=StringIO()

        # _EMR_UNKNOWN objects don't have a typedef, so only process
        # those that do
        if self.format.typedef:
            #print "typedef=%s" % str(self.format.typedef)
            for item in self.format.typedef:
                typecode=item[0]
                name=item[1]
                val=self.str_decode(typecode,name)
                try:
                    txt.write("\t%-20s: %s\n" % (name,val))
                except UnicodeEncodeError:
                    txt.write("\t%-20s: <<<BAD UNICODE STRING>>>\n" % name)
        txt.write(self.str_extra())
        return txt.getvalue()

    def __str__(self):
        ret=""
        details=self.str_details()
        if details:
            ret=os.linesep
        return "**%s: iType=%s nSize=%s  struct='%s' size=%d\n%s%s" % (self.__class__.__name__.lstrip('_'),self.iType,self.nSize,self.format.fmt,self.format.structsize,details,ret)
        return 


# Collection of classes

class _EMR:

    class _HEADER(_EMR_UNKNOWN):
        """Header has different fields depending on the version of
        windows.  Also note that if offDescription==0, there is no
        description string."""

        emr_id=1
        emr_typedef=[('i','rclBounds_left'),
                     ('i','rclBounds_top'),
                     ('i','rclBounds_right'),
                     ('i','rclBounds_bottom'),
                     ('i','rclFrame_left'),
                     ('i','rclFrame_top'),
                     ('i','rclFrame_right'),
                     ('i','rclFrame_bottom'),
                     ('i','dSignature',1179469088),
                     ('i','nVersion',0x10000),
                     ('i','nBytes',0),
                     ('i','nRecords',0),
                     ('h','nHandles',0),
                     ('h','sReserved',0),
                     ('i','nDescription',0),
                     ('i','offDescription',0),
                     ('i','nPalEntries',0),
                     ('i','szlDevice_cx',1024),
                     ('i','szlDevice_cy',768),
                     ('i','szlMillimeters_cx',320),
                     ('i','szlMillimeters_cy',240),
                     ('i','cbPixelFormat',0),
                     ('i','offPixelFormat',0),
                     ('i','bOpenGL',0),
                     ('i','szlMicrometers_cx'),
                     ('i','szlMicrometers_cy')]
        
        def __init__(self,description=''):
            _EMR_UNKNOWN.__init__(self)

            # NOTE: rclBounds and rclFrame will be determined at
            # serialize time

            if isinstance(description,str):
                # turn it into a unicode string
                # print "description=%s" % description
                self.description=description.decode('utf-8')
                # print "self.description=%s" % self.description
                # print isinstance(self.description,unicode)
            if len(description)>0:
                self.description=u'pyemf '+__version__.decode('utf-8')+u'\0'+description+u'\0\0'
            self.nDescription=len(self.description)

        def unserializeExtra(self,data):
            if self.verbose: print "found %d extra bytes." % len(data)

            # FIXME: descriptionStart could potentially be negative if
            # we have an old format metafile without stuff after
            # szlMillimeters AND we have a description.
            if self.offDescription>0:
                start=self.unserializeOffset(self.offDescription)
                # unicode is always stored in little endian format
                txt=data[start:start+(2*self.nDescription)]
                self.description=txt.decode('utf-16le')
                if self.verbose: print "str: %s" % self.description

        def str_extra(self):
            txt=StringIO()
            txt.write("\tunicode string: %s\n" % str(self.description))
            txt.write("%s\n" % (struct.pack('16s',self.description.encode('utf-16le'))))
            return txt.getvalue()

        def setBounds(self,dc,scaleheader):
            self.rclBounds_left=dc.bounds_left
            self.rclBounds_top=dc.bounds_top
            self.rclBounds_right=dc.bounds_right
            self.rclBounds_bottom=dc.bounds_bottom
            
            self.rclFrame_left=dc.frame_left
            self.rclFrame_top=dc.frame_top
            self.rclFrame_right=dc.frame_right
            self.rclFrame_bottom=dc.frame_bottom

            if scaleheader:
                self.szlDevice_cx=dc.pixelwidth
                self.szlDevice_cy=dc.pixelheight
                self.szlMicrometers_cx=dc.width*10
                self.szlMicrometers_cy=dc.height*10
            else:
                self.szlDevice_cx=dc.ref_pixelwidth
                self.szlDevice_cy=dc.ref_pixelheight
                self.szlMicrometers_cx=dc.ref_width*10
                self.szlMicrometers_cy=dc.ref_height*10
                
            self.szlMillimeters_cx=self.szlMicrometers_cx/1000
            self.szlMillimeters_cy=self.szlMicrometers_cy/1000
                


        def sizeExtra(self):
            if self.szlMicrometers_cx==0:
                self.szlMicrometers_cx=self.szlMillimeters_cx*1000
                self.szlMicrometers_cy=self.szlMillimeters_cy*1000

            self.nDescription=len(self.description)
            if self.nDescription>0:
                self.offDescription=self.serializeOffset()
            else:
                self.offDescription=0
            sizestring=_round4(self.nDescription*2) # always unicode
            
            return sizestring

        def serializeExtra(self,fh):
            self.serializeString(fh,self.description)



    class _POLYBEZIER(_EMR_UNKNOWN):
        emr_id=2
        emr_typedef=[('i','rclBounds_left'),
                     ('i','rclBounds_top'),
                     ('i','rclBounds_right'),
                     ('i','rclBounds_bottom'),
                     ('i','cptl')]
        emr_point_type='i'
        
        def __init__(self,points=[],bounds=(0,0,0,0)):
            _EMR_UNKNOWN.__init__(self)
            self.setBounds(bounds)
            self.cptl=len(points)
            self.aptl=points

        def unserializeExtra(self,data):
            # print "found %d extra bytes." % len(data)

            start=0
            start,self.aptl=self.unserializePoints(self.emr_point_type,
                                                   self.cptl,data,start)
            # print "apts size=%d: %s" % (len(self.apts),self.apts)

        def sizeExtra(self):
            return struct.calcsize(self.emr_point_type)*2*self.cptl

        def serializeExtra(self,fh):
            self.serializePoints(fh,self.emr_point_type,self.aptl)

        def str_extra(self):
            txt=StringIO()
            start=0
            txt.write("\tpoints: %s\n" % str(self.aptl))
                    
            return txt.getvalue()

    class _POLYGON(_POLYBEZIER):
        emr_id=3
        pass

    class _POLYLINE(_POLYBEZIER):
        emr_id=4
        pass

    class _POLYBEZIERTO(_POLYBEZIER):
        emr_id=5

        def getBounds(self):
            return (self.rclBounds_left,self.rclBounds_top,
                    self.rclBounds_right,self.rclBounds_bottom)

    class _POLYLINETO(_POLYBEZIERTO):
        emr_id=6
        pass
    


    class _POLYPOLYLINE(_EMR_UNKNOWN):
        emr_id=7
        emr_typedef=[('i','rclBounds_left'),
                     ('i','rclBounds_top'),
                     ('i','rclBounds_right'),
                     ('i','rclBounds_bottom'),
                     ('i','nPolys'),
                     ('i','cptl')]
        emr_point_type='i'
        
        def __init__(self,points=[],polycounts=[],bounds=(0,0,0,0)):
            _EMR_UNKNOWN.__init__(self)
            self.setBounds(bounds)
            self.cptl=len(points)
            self.aptl=points
            self.nPolys=len(polycounts)
            self.aPolyCounts=polycounts

        def unserializeExtra(self,data):
            # print "found %d extra bytes." % len(data)

            start=0
            start,self.aPolyCounts=self.unserializeList("i",self.nPolys,data,start)
            # print "aPolyCounts start=%d size=%d: %s" % (start,len(self.aPolyCounts),str(self.aPolyCounts))

            start,self.aptl=self.unserializePoints(self.emr_point_type,self.cptl,data,start)
            # print "apts size=%d: %s" % (len(self.apts),self.apts)

        def sizeExtra(self):
            return (struct.calcsize("i")*self.nPolys +
                    struct.calcsize(self.emr_point_type)*2*self.cptl)

        def serializeExtra(self,fh):
            self.serializeList(fh,"i",self.aPolyCounts)
            self.serializePoints(fh,self.emr_point_type,self.aptl)

        def str_extra(self):
            txt=StringIO()
            start=0
            for n in range(self.nPolys):
                txt.write("\tPolygon %d: %d points\n" % (n,self.aPolyCounts[n]))
                txt.write("\t\t%s\n" % str(self.aptl[start:start+self.aPolyCounts[n]]))
                start+=self.aPolyCounts[n]
                    
            return txt.getvalue()

    class _POLYPOLYGON(_POLYPOLYLINE):
        emr_id=8
        pass




    class _SETWINDOWEXTEX(_EMR_UNKNOWN):
        emr_id=9
        emr_typedef=[('i','szlExtent_cx'),
                     ('i','szlExtent_cy')]
        
        def __init__(self,cx=0,cy=0):
            _EMR_UNKNOWN.__init__(self)
            self.szlExtent_cx=cx
            self.szlExtent_cy=cy


    class _SETWINDOWORGEX(_EMR_UNKNOWN):
        emr_id=10
        emr_typedef=[('i','ptlOrigin_x'),
                     ('i','ptlOrigin_y')]
        
        def __init__(self,x=0,y=0):
            _EMR_UNKNOWN.__init__(self)
            self.ptlOrigin_x=x
            self.ptlOrigin_y=y


    class _SETVIEWPORTEXTEX(_SETWINDOWEXTEX):
        emr_id=11
        pass


    class _SETVIEWPORTORGEX(_SETWINDOWORGEX):
        emr_id=12
        pass


    class _SETBRUSHORGEX(_SETWINDOWORGEX):
        emr_id=13
        pass


    class _EOF(_EMR_UNKNOWN):
        """End of file marker.  Usually 20 bytes long, but I have a
        Windows generated .emf file that only has a 12 byte long EOF
        record.  I don't know if that's a broken example or what, but
        both Windows progs and OpenOffice seem to handle it."""
        emr_id=14
        emr_typedef=[
                ('i','nPalEntries',0),
                ('i','offPalEntries',0),
                ('i','nSizeLast',0)]
        
        def __init__(self):
            _EMR_UNKNOWN.__init__(self)


    class _SETPIXELV(_EMR_UNKNOWN):
        emr_id=15
        emr_typedef=[
                ('i','ptlPixel_x'),
                ('i','ptlPixel_y'),
                ('i','crColor')]
        
        def __init__(self,x=0,y=0,color=0):
            _EMR_UNKNOWN.__init__(self)
            self.ptlPixel_x=x
            self.ptlPixel_y=y
            self.crColor=color


    class _SETMAPPERFLAGS(_EMR_UNKNOWN):
        emr_id=16
        emr_format=[('i','dwFlags',0)]
        
        def __init__(self):
            _EMR_UNKNOWN.__init__(self)


    class _SETMAPMODE(_EMR_UNKNOWN):
        emr_id=17
        emr_typedef=[('i','iMode',MM_ANISOTROPIC)]
        
        def __init__(self,mode=MM_ANISOTROPIC,first=0,last=MM_MAX):
            _EMR_UNKNOWN.__init__(self)
            if mode<first or mode>last:
                self.error=1
            else:
                self.iMode=mode
            
    class _SETBKMODE(_SETMAPMODE):
        emr_id=18
        def __init__(self,mode=OPAQUE):
            _EMR._SETMAPMODE.__init__(self,mode,last=BKMODE_LAST)


    class _SETPOLYFILLMODE(_SETMAPMODE):
        emr_id=19
        def __init__(self,mode=ALTERNATE):
            _EMR._SETMAPMODE.__init__(self,mode,last=POLYFILL_LAST)

            
    class _SETROP2(_SETMAPMODE):
        emr_id=20
        pass

            
    class _SETSTRETCHBLTMODE(_SETMAPMODE):
        emr_id=21
        pass

            
    class _SETTEXTALIGN(_SETMAPMODE):
        emr_id=22
        def __init__(self,mode=TA_BASELINE):
            _EMR._SETMAPMODE.__init__(self,mode,last=TA_MASK)

            
#define EMR_SETCOLORADJUSTMENT	23

    class _SETTEXTCOLOR(_EMR_UNKNOWN):
        emr_id=24
        emr_typedef=[('i','crColor',0)]
        
        def __init__(self,color=0):
            _EMR_UNKNOWN.__init__(self)
            self.crColor=color


    class _SETBKCOLOR(_SETTEXTCOLOR):
        emr_id=25
        pass


#define EMR_OFFSETCLIPRGN	26


    class _MOVETOEX(_EMR_UNKNOWN):
        emr_id=27
        emr_typedef=[
                ('i','ptl_x'),
                ('i','ptl_y')]
        
        def __init__(self,x=0,y=0):
            _EMR_UNKNOWN.__init__(self)
            self.ptl_x=x
            self.ptl_y=y

        def getBounds(self):
            return (self.ptl_x,self.ptl_y,self.ptl_x,self.ptl_y)
            

#define EMR_SETMETARGN	28
#define EMR_EXCLUDECLIPRECT	29
#define EMR_INTERSECTCLIPRECT	30

    class _SCALEVIEWPORTEXTEX(_EMR_UNKNOWN):
        emr_id=31
        emr_typedef=[
                ('i','xNum',1),
                ('i','xDenom',1),
                ('i','yNum',1),
                ('i','yDenom',1)]
        
        def __init__(self,xn=1,xd=1,yn=1,yd=1):
            _EMR_UNKNOWN.__init__(self)
            self.xNum=xn
            self.xDenom=xd
            self.yNum=yn
            self.yDenom=yd

    class _SCALEWINDOWEXTEX(_SCALEVIEWPORTEXTEX):
        emr_id=32
        pass


    class _SAVEDC(_EMR_UNKNOWN):
        emr_id=33
        pass

    class _RESTOREDC(_EMR_UNKNOWN):
        emr_id=34
        emr_typedef=[('i','iRelative')]
        
        def __init__(self,rel=-1):
            _EMR_UNKNOWN.__init__(self)
            self.iRelative=rel


    class _SETWORLDTRANSFORM(_EMR_UNKNOWN):
        emr_id=35
        emr_typedef=[
                ('f','eM11'),
                ('f','eM12'),
                ('f','eM21'),
                ('f','eM22'),
                ('f','eDx'),
                ('f','eDy')]
        
        def __init__(self,em11=1.0,em12=0.0,em21=0.0,em22=1.0,edx=0.0,edy=0.0):
            _EMR_UNKNOWN.__init__(self)
            self.eM11=em11
            self.eM12=em12
            self.eM21=em21
            self.eM22=em22
            self.eDx=edx
            self.eDy=edy

    class _MODIFYWORLDTRANSFORM(_EMR_UNKNOWN):
        emr_id=36
        emr_typedef=[
                ('f','eM11'),
                ('f','eM12'),
                ('f','eM21'),
                ('f','eM22'),
                ('f','eDx'),
                ('f','eDy'),
                ('i','iMode')]
        
        def __init__(self,em11=1.0,em12=0.0,em21=0.0,em22=1.0,edx=0.0,edy=0.0,mode=MWT_IDENTITY):
            _EMR_UNKNOWN.__init__(self)
            self.eM11=em11
            self.eM12=em12
            self.eM21=em21
            self.eM22=em22
            self.eDx=edx
            self.eDy=edy
            self.iMode=mode


    class _SELECTOBJECT(_EMR_UNKNOWN):
        """Select a brush, pen, font (or bitmap or region but there is
        no current user interface for those) object to be current and
        replace the previous item of that class.  Note that stock
        objects have their high order bit set, so the handle must be
        an unsigned int."""
        emr_id=37
        emr_typedef=[('I','handle')]
        
        def __init__(self,dc=None,handle=0):
            _EMR_UNKNOWN.__init__(self)
            self.handle=handle


    # Note: a line will still be drawn when the linewidth==0.  To force an
    # invisible line, use style=PS_NULL
    class _CREATEPEN(_EMR_UNKNOWN):
        emr_id=38
        emr_typedef=[
                ('i','handle',0),
                ('i','lopn_style'),
                ('i','lopn_width'),
                ('i','lopn_unused',0),
                ('i','lopn_color')]
        
        def __init__(self,style=PS_SOLID,width=1,color=0):
            _EMR_UNKNOWN.__init__(self)
            self.lopn_style=style
            self.lopn_width=width
            self.lopn_color=color

        def hasHandle(self):
            return True


    class _CREATEBRUSHINDIRECT(_EMR_UNKNOWN):
        emr_id=39
        emr_typedef=[
                ('i','handle',0),
                ('I','lbStyle'),
                ('i','lbColor'),
                ('I','lbHatch')]
        
        def __init__(self,style=BS_SOLID,hatch=HS_HORIZONTAL,color=0):
            _EMR_UNKNOWN.__init__(self)
            self.lbStyle = style
            self.lbColor = color
            self.lbHatch = hatch

        def hasHandle(self):
            return True


    class _DELETEOBJECT(_SELECTOBJECT):
        emr_id=40
        pass


    class _ANGLEARC(_EMR_UNKNOWN):
        emr_id=41
        emr_typedef=[
                ('i','ptlCenter_x'),
                ('i','ptlCenter_y'),
                ('i','nRadius'),
                ('f','eStartAngle'),
                ('f','eSweepAngle')]
        
        def __init__(self):
            _EMR_UNKNOWN.__init__(self)


    class _ELLIPSE(_EMR_UNKNOWN):
        emr_id=42
        emr_typedef=[
                ('i','rclBox_left'),
                ('i','rclBox_top'),
                ('i','rclBox_right'),
                ('i','rclBox_bottom')]
        
        def __init__(self,box=(0,0,0,0)):
            _EMR_UNKNOWN.__init__(self)
            self.rclBox_left=box[0]
            self.rclBox_top=box[1]
            self.rclBox_right=box[2]
            self.rclBox_bottom=box[3]


    class _RECTANGLE(_ELLIPSE):
        emr_id=43
        pass


    class _ROUNDRECT(_EMR_UNKNOWN):
        emr_id=44
        emr_typedef=[
                ('i','rclBox_left'),
                ('i','rclBox_top'),
                ('i','rclBox_right'),
                ('i','rclBox_bottom'),
                ('i','szlCorner_cx'),
                ('i','szlCorner_cy')]
        
        def __init__(self,box=(0,0,0,0),cx=0,cy=0):
            _EMR_UNKNOWN.__init__(self)
            self.rclBox_left=box[0]
            self.rclBox_top=box[1]
            self.rclBox_right=box[2]
            self.rclBox_bottom=box[3]
            self.szlCorner_cx=cx
            self.szlCorner_cy=cy


    class _ARC(_EMR_UNKNOWN):
        emr_id=45
        emr_typedef=[
                ('i','rclBox_left'),
                ('i','rclBox_top'),
                ('i','rclBox_right'),
                ('i','rclBox_bottom'),
                ('i','ptlStart_x'),
                ('i','ptlStart_y'),
                ('i','ptlEnd_x'),
                ('i','ptlEnd_y')]
        
        def __init__(self,left=0,top=0,right=0,bottom=0,
                     xstart=0,ystart=0,xend=0,yend=0):
            _EMR_UNKNOWN.__init__(self)
            self.rclBox_left=left
            self.rclBox_top=top
            self.rclBox_right=right
            self.rclBox_bottom=bottom
            self.ptlStart_x=xstart
            self.ptlStart_y=ystart
            self.ptlEnd_x=xend
            self.ptlEnd_y=yend
            

    class _CHORD(_ARC):
        emr_id=46
        pass


    class _PIE(_ARC):
        emr_id=47
        pass


    class _SELECTPALETTE(_EMR_UNKNOWN):
        emr_id=48
        emr_typedef=[('i','handle')]
        
        def __init__(self):
            _EMR_UNKNOWN.__init__(self)
        

    # Stub class for palette
    class _CREATEPALETTE(_EMR_UNKNOWN):
        emr_id=49
        emr_typedef=[('i','handle',0)]
        
        def __init__(self):
            _EMR_UNKNOWN.__init__(self)
            
        def hasHandle(self):
            return True
        
            
#define EMR_SETPALETTEENTRIES	50
#define EMR_RESIZEPALETTE	51
#define EMR_REALIZEPALETTE	52
#define EMR_EXTFLOODFILL	53


    class _LINETO(_MOVETOEX):
        emr_id=54
        pass


    class _ARCTO(_ARC):
        emr_id=55

        def getBounds(self):
            # not exactly the bounds, because the arc may actually use
            # less of the ellipse than is specified by the bounds.
            # But at least the actual bounds aren't outside these
            # bounds.
            return (self.rclBox_left,self.rclBox_top,
                    self.rclBox_right,self.rclBox_bottom)



#define EMR_POLYDRAW	56


    class _SETARCDIRECTION(_EMR_UNKNOWN):
        emr_id=57
        emr_typedef=[('i','iArcDirection')]
        def __init__(self):
            _EMR_UNKNOWN.__init__(self)
        


#define EMR_SETMITERLIMIT	58


    class _BEGINPATH(_EMR_UNKNOWN):
        emr_id=59
        pass


    class _ENDPATH(_EMR_UNKNOWN):
        emr_id=60
        pass


    class _CLOSEFIGURE(_EMR_UNKNOWN):
        emr_id=61
        pass


    class _FILLPATH(_EMR_UNKNOWN):
        emr_id=62
        emr_typedef=[
                ('i','rclBounds_left'),
                ('i','rclBounds_top'),
                ('i','rclBounds_right'),
                ('i','rclBounds_bottom')]
        def __init__(self,bounds=(0,0,0,0)):
            _EMR_UNKNOWN.__init__(self)
            self.setBounds(bounds)

    class _STROKEANDFILLPATH(_FILLPATH):
        emr_id=63
        pass


    class _STROKEPATH(_FILLPATH):
        emr_id=64
        pass


    class _FLATTENPATH(_EMR_UNKNOWN):
        emr_id=65
        pass


    class _WIDENPATH(_EMR_UNKNOWN):
        emr_id=66
        pass


    class _SELECTCLIPPATH(_SETMAPMODE):
        """Select the current path and make it the clipping region.
        Must be a closed path.

        @gdi: SelectClipPath
        """
        emr_id=67
        def __init__(self,mode=RGN_COPY):
            _EMR._SETMAPMODE.__init__(self,mode,first=RGN_MIN,last=RGN_MAX)


    class _ABORTPATH(_EMR_UNKNOWN):
        """Discards any current path, whether open or closed.

        @gdi: AbortPath"""
        emr_id=68
        pass


#define EMR_GDICOMMENT	70
#define EMR_FILLRGN	71
#define EMR_FRAMERGN	72
#define EMR_INVERTRGN	73
#define EMR_PAINTRGN	74
#define EMR_EXTSELECTCLIPRGN	75
#define EMR_BITBLT	76
#define EMR_STRETCHBLT	77
#define EMR_MASKBLT	78
#define EMR_PLGBLT	79
#define EMR_SETDIBITSTODEVICE	80

    class _STRETCHDIBITS(_EMR_UNKNOWN):
        """Copies the image from the source image to the destination
        image.  DIB is currently an opaque format to me, but
        apparently it has been extented recently to allow JPG and PNG
        images...

        @gdi: StretchDIBits
        """
        emr_id=81
        emr_typedef=[
                ('i','rclBounds_left'),
                ('i','rclBounds_top'),
                ('i','rclBounds_right'),
                ('i','rclBounds_bottom'),
                ('i','xDest'), 
                ('i','yDest'), 
                ('i','xSrc'), 
                ('i','ySrc'), 
                ('i','cxSrc'), 
                ('i','cySrc'),
                ('i','offBmiSrc'),
                ('i','cbBmiSrc'), 
                ('i','offBitsSrc'), 
                ('i','cbBitsSrc'), 
                ('i','iUsageSrc'), 
                ('i','dwRop'), 
                ('i','cxDest'), 
                ('i','cyDest')]
        
        def __init__(self):
            _EMR_UNKNOWN.__init__(self)


    class _EXTCREATEFONTINDIRECTW(_EMR_UNKNOWN):
        # Note: all the strings here (font names, etc.) are unicode
        # strings.
        
        emr_id=82
        emr_typedef=[
                ('i','handle'),
                ('i','lfHeight'),
                ('i','lfWidth'),
                ('i','lfEscapement'),
                ('i','lfOrientation'),
                ('i','lfWeight'),
                ('B','lfItalic'),
                ('B','lfUnderline'),
                ('B','lfStrikeOut'),
                ('B','lfCharSet'),
                ('B','lfOutPrecision'),
                ('B','lfClipPrecision'),
                ('B','lfQuality'),
                ('B','lfPitchAndFamily'),
                ('64s','lfFaceName',), # really a 32 char unicode string
                ('128s','elfFullName','\0'*128), # really 64 char unicode str
                ('64s','elfStyle','\0'*64), # really 32 char unicode str
                ('i','elfVersion',0),
                ('i','elfStyleSize',0),
                ('i','elfMatch',0),
                ('i','elfReserved',0),
                ('i','elfVendorId',0),
                ('i','elfCulture',0),
                ('B','elfPanose_bFamilyType',1),
                ('B','elfPanose_bSerifStyle',1),
                ('B','elfPanose_bWeight',1),
                ('B','elfPanose_bProportion',1),
                ('B','elfPanose_bContrast',1),
                ('B','elfPanose_bStrokeVariation',1),
                ('B','elfPanose_bArmStyle',1),
                ('B','elfPanose_bLetterform',1),
                ('B','elfPanose_bMidline',1),
                ('B','elfPanose_bXHeight',1)]

        def __init__(self,height=0,width=0,escapement=0,orientation=0,
                     weight=FW_NORMAL,italic=0,underline=0,strike_out=0,
                     charset=ANSI_CHARSET,out_precision=OUT_DEFAULT_PRECIS,
                     clip_precision=CLIP_DEFAULT_PRECIS,
                     quality=DEFAULT_QUALITY,
                     pitch_family=DEFAULT_PITCH|FF_DONTCARE,name='Times New Roman'):
            _EMR_UNKNOWN.__init__(self)
            self.lfHeight=height
            self.lfWidth=width
            self.lfEscapement=escapement
            self.lfOrientation=orientation
            self.lfWeight=weight
            self.lfItalic=italic
            self.lfUnderline=underline
            self.lfStrikeOut=strike_out
            self.lfCharSet=charset
            self.lfOutPrecision=out_precision
            self.lfClipPrecision=clip_precision
            self.lfQuality=quality
            self.lfPitchAndFamily=pitch_family

            # pad the structure out to 4 byte boundary
            self.unhandleddata=_EMR_UNKNOWN.twobytepadding

            # truncate or pad to exactly 32 characters
            if len(name)>32:
                name=name[0:32]
            else:
                name+='\0'*(32-len(name))
            self.lfFaceName=name.decode('utf-8').encode('utf-16le')
            # print "lfFaceName=%s" % self.lfFaceName

        def hasHandle(self):
            return True


    class _EXTTEXTOUTA(_EMR_UNKNOWN):
        emr_id=83
        emr_typedef=[
                ('i','rclBounds_left',0),
                ('i','rclBounds_top',0),
                ('i','rclBounds_right',-1),
                ('i','rclBounds_bottom',-1),
                ('i','iGraphicsMode',GM_COMPATIBLE),
                ('f','exScale',1.0),
                ('f','eyScale',1.0),
                ('i','ptlReference_x'),
                ('i','ptlReference_y'),
                ('i','nChars'),
                ('i','offString'),
                ('i','fOptions',0),
                ('i','rcl_left',0),
                ('i','rcl_top',0),
                ('i','rcl_right',-1),
                ('i','rcl_bottom',-1),
                ('i','offDx',0)]
        def __init__(self,x=0,y=0,txt=""):
            _EMR_UNKNOWN.__init__(self)
            self.ptlReference_x=x
            self.ptlReference_y=y
            if isinstance(txt,unicode):
                self.string=txt.encode('utf-16le')
            else:
                self.string=txt
            self.charsize=1
            self.dx=[]

        def unserializeExtra(self,data):
            # print "found %d extra bytes.  nChars=%d" % (len(data),self.nChars)

            start=0
            # print "offDx=%d offString=%d" % (self.offDx,self.offString)

            # Note: offsets may appear before OR after string.  Don't
            # assume they will appear first.
            if self.offDx>0:
                start=self.unserializeOffset(self.offDx)
                start,self.dx=self.unserializeList("i",self.nChars,data,start)
            else:
                self.dx=[]
                
            if self.offString>0:
                start=self.unserializeOffset(self.offString)
                self.string=data[start:start+(self.charsize*self.nChars)]
            else:
                self.string=""

        def sizeExtra(self):
            offset=self.serializeOffset()
            sizedx=0
            sizestring=0

            if len(self.string)>0:
                self.nChars=len(self.string)/self.charsize
                self.offString=offset
                sizestring=_round4(self.charsize*self.nChars)
                offset+=sizestring
            if len(self.dx)>0:
                self.offDx=offset
                sizedx=struct.calcsize("i")*self.nChars
                offset+=sizedx
                
            return (sizedx+sizestring)

        def serializeExtra(self,fh):
            # apparently the preferred way is to store the string
            # first, then the offsets
            if self.offString>0:
                self.serializeString(fh,self.string)
            if self.offDx>0:
                self.serializeList(fh,"i",self.dx)

        def str_extra(self):
            txt=StringIO()
            txt.write("\tdx: %s\n" % str(self.dx))
            if self.charsize==2:
                txt.write("\tunicode string: %s\n" % str(self.string.decode('utf-16le')))
            else:
                txt.write("\tascii string: %s\n" % str(self.string))
                    
            return txt.getvalue()


    class _EXTTEXTOUTW(_EXTTEXTOUTA):
        emr_id=84

        def __init__(self,x=0,y=0,txt=u''):
            _EMR._EXTTEXTOUTA.__init__(self,x,y,txt)
            self.charsize=2




    class _POLYBEZIER16(_POLYBEZIER):
        emr_id=85
        emr_point_type='h'

    class _POLYGON16(_POLYBEZIER16):
        emr_id=86
        pass

    class _POLYLINE16(_POLYBEZIER16):
        emr_id=87
        pass

    class _POLYBEZIERTO16(_POLYBEZIERTO):
        emr_id=88
        emr_point_type='h'
        pass

    class _POLYLINETO16(_POLYBEZIERTO16):
        emr_id=89
        pass

    class _POLYPOLYLINE16(_POLYPOLYLINE):
        emr_id=90
        emr_point_type='h'
        pass
    
    class _POLYPOLYGON16(_POLYPOLYLINE16):
        emr_id=91
        pass

#define EMR_POLYDRAW16	92

    # Stub class for storage of brush with monochrome bitmap or DIB
    class _CREATEMONOBRUSH(_CREATEPALETTE):
        emr_id=93
        pass

    # Stub class for device independent bitmap brush
    class _CREATEDIBPATTERNBRUSHPT(_CREATEPALETTE):
        emr_id=94
        pass

    # Stub class for extended pen
    class _EXTCREATEPEN(_CREATEPALETTE):
        emr_id=95
        pass


#define EMR_POLYTEXTOUTA	96
#define EMR_POLYTEXTOUTW	97


    class _SETICMMODE(_SETMAPMODE):
        """Set or query the current color management mode.

        @gdi: SetICMMode
        """
        emr_id=98
        def __init__(self,mode=ICM_OFF):
            _EMR._SETMAPMODE.__init__(self,mode,first=ICM_MIN,last=ICM_MAX)


#define EMR_CREATECOLORSPACE	99
#define EMR_SETCOLORSPACE	100
#define EMR_DELETECOLORSPACE	101
#define EMR_GLSRECORD	102
#define EMR_GLSBOUNDEDRECORD	103
#define EMR_PIXELFORMAT 104
#define EMR_DRAWESCAPE    105
#define EMR_EXTESCAPE     106
#define EMR_STARTDOC      107
#define EMR_SMALLTEXTOUT  108
#define EMR_FORCEUFIMAPPING       109
#define EMR_NAMEDESCAPE   110
#define EMR_COLORCORRECTPALETTE   111
#define EMR_SETICMPROFILEA        112
#define EMR_SETICMPROFILEW        113
#define EMR_ALPHABLEND    114
#define EMR_SETLAYOUT     115
#define EMR_TRANSPARENTBLT        116
#define EMR_RESERVED_117  117
#define EMR_GRADIENTFILL  118
#define EMR_SETLINKEDUFI  119
#define EMR_SETTEXTJUSTIFICATION  120
#define EMR_COLORMATCHTOTARGETW   121
#define EMR_CREATECOLORSPACEW     122


# Set up the mapping of ids to classes for all of the record types in
# the EMR class.
_emrmap={}

for name in dir(_EMR):
    #print name
    cls=getattr(_EMR,name,None)
    if cls and callable(cls) and issubclass(cls,_EMR_UNKNOWN):
        #print "subclass! id=%d %s" % (cls.emr_id,str(cls))
        _emrmap[cls.emr_id]=cls



class EMF:
    """
Reference page of the public API for enhanced metafile creation.  See
L{pyemf} for an overview / mini tutorial.

@group Creating Metafiles: __init__, load, save
@group Drawing Parameters: GetStockObject, SelectObject, DeleteObject, CreatePen, CreateSolidBrush, CreateHatchBrush, SetBkColor, SetBkMode, SetPolyFillMode
@group Drawing Primitives: SetPixel, Polyline, PolyPolyline, Polygon, PolyPolygon, Rectangle, RoundRect, Ellipse, Arc, Chord, Pie, PolyBezier
@group Path Primatives: BeginPath, EndPath, MoveTo, LineTo, PolylineTo, ArcTo,
 PolyBezierTo, CloseFigure, FillPath, StrokePath, StrokeAndFillPath
@group Clipping: SelectClipPath
@group Text: CreateFont, SetTextAlign, SetTextColor, TextOut
@group Coordinate System Transformation: SaveDC, RestoreDC, SetWorldTransform, ModifyWorldTransform
@group **Experimental** -- Viewport Manipulation: SetMapMode, SetViewportOrgEx, GetViewportOrgEx, SetWindowOrgEx, GetWindowOrgEx, SetViewportExtEx, ScaleViewportExtEx, GetViewportExtEx, SetWindowExtEx, ScaleWindowExtEx, GetWindowExtEx 

"""

    def __init__(self,width=6.0,height=4.0,density=300,units="in",
                 description="pyemf.sf.net",verbose=False):
        """
Create an EMF structure in memory.  The size of the resulting image is
specified in either inches or millimeters depending on the value of
L{units}.  Width and height are floating point values, but density
must be an integer because this becomes the basis for the coordinate
system in the image.  Density is the number of individually
addressible pixels per unit measurement (dots per inch or dots per
millimeter, depending on the units system) in the image.  A
consequence of this is that each pixel is specified by a pair of
integer coordinates.

@param width: width of EMF image in inches or millimeters
@param height: height of EMF image in inches or millimeters
@param density: dots (pixels) per unit measurement
@param units: string indicating the unit measurement, one of:
 - 'in'
 - 'mm'
@type width: float
@type height: float
@type density: int
@type units: string
@param description: optional string to specify a description of the image
@type description: string

"""
        self.filename=None
        self.dc=_DC(width,height,density,units)
        self.records=[]

        # path recordkeeping
        self.pathstart=0

        self.verbose=verbose

        # if True, scale the image using only the header, and not
        # using MapMode or SetWindow/SetViewport.
        self.scaleheader=True

        emr=_EMR._HEADER(description)
        self._append(emr)
        if not self.scaleheader:
            self.SetMapMode(MM_ANISOTROPIC)
            self.SetWindowExtEx(self.dc.pixelwidth,self.dc.pixelheight)
            self.SetViewportExtEx(
                int(self.dc.width/100.0*self.dc.ref_pixelwidth/self.dc.ref_width),
                int(self.dc.height/100.0*self.dc.ref_pixelheight/self.dc.ref_height))


    def load(self,filename=None):
        """
Read an existing EMF file.  If any records exist in the current
object, they will be overwritten by the records from this file.

@param filename: filename to load
@type filename: string
@returns: True for success, False for failure.
@rtype: Boolean
        """
        if filename:
            self.filename=filename

        if self.filename:
            fh=open(self.filename)
            self.records=[]
            self._unserialize(fh)
            self.scaleheader=False
            # get DC from header record
            self.dc.getBounds(self.records[0])


    def _unserialize(self,fh):
        try:
            count=1
            while count>0:
                data=fh.read(8)
                count=len(data)
                if count>0:
                    (iType,nSize)=struct.unpack("<ii",data)
                    if self.verbose: print "EMF:  iType=%d nSize=%d" % (iType,nSize)

                    if iType in _emrmap:
                        e=_emrmap[iType]()
                    else:
                        e=_EMR_UNKNOWN()

                    e.unserialize(fh,iType,nSize)
                    self.records.append(e)
                    
                    if e.hasHandle():
                        self.dc.addObject(e,e.handle)
                    elif isinstance(e,_EMR._DELETEOBJECT):
                        self.dc.removeObject(e.handle)
                        
                    if self.verbose:
                        print "Unserializing: ",
                        print e
                
        except EOFError:
            pass

    def _append(self,e):
        """Append an EMR to the record list, unless the record has
        been flagged as having an error."""
        if not e.error:
            if self.verbose:
                print "Appending: ",
                print e
            self.records.append(e)
            return 1
        return 0

    def _end(self):
        """
Append an EOF record and compute header information.  The header needs
to know the number of records, number of handles, bounds, and size of
the entire metafile before it can be written out, so we have to march
through all the records and gather info.
        """
        
        end=self.records[-1]
        if not isinstance(end,_EMR._EOF):
            if self.verbose: print "adding EOF record"
            e=_EMR._EOF()
            self._append(e)
        header=self.records[0]
        header.setBounds(self.dc,self.scaleheader)
        header.nRecords=len(self.records)
        header.nHandles=len(self.dc.objects)
        size=0
        for e in self.records:
            e.resize()
            size+=e.nSize
        header.nBytes=size
        
    def save(self,filename=None):
        """
Write the EMF to disk.

@param filename: filename to write
@type filename: string
@returns: True for success, False for failure.
@rtype: Boolean
        """

        self._end()
    
        if filename:
            self.filename=filename
            
        if self.filename:
            try:
                fh=open(self.filename,"wb")
                self._serialize(fh)
                fh.close()
                return True
            except:
                raise
                return False
        return False
        
    def _serialize(self,fh):
        for e in self.records:
            if self.verbose: print e
            e.serialize(fh)

    def _create(self,width,height,dots_per_unit,units):
        pass

    def _getBounds(self,points):
        """Get the bounding rectangle for this list of 2-tuples."""
        left=points[0][0]
        right=left
        top=points[0][1]
        bottom=top
        for x,y in points[1:]:
            if x<left:
                left=x
            elif x>right:
                right=x
            if y<top:
                top=y
            elif y>bottom:
                bottom=y
        return (left,top,right,bottom)

    def _mergeBounds(self,bounds,itembounds):
        if itembounds:
            if itembounds[0]<bounds[0]: bounds[0]=itembounds[0]
            if itembounds[1]<bounds[1]: bounds[1]=itembounds[1]
            if itembounds[2]>bounds[2]: bounds[2]=itembounds[2]
            if itembounds[3]>bounds[3]: bounds[3]=itembounds[3]

    def _getPathBounds(self):
        """Get the bounding rectangle for the list of EMR records
        starting from the last saved path start to the current record."""
        big=1000000
        bounds=[big,big,-1,-1]
        for i in range(self.pathstart,len(self.records)):
            # print "FIXME: checking record %d" % i
            e=self.records[i]
            # print e
            # print "bounds=%s" % str(e.getBounds())
            self._mergeBounds(bounds,e.getBounds())

        if bounds[0]==big: bounds[0]=0
        if bounds[1]==big: bounds[1]=0
        return bounds

    def _useShort(self,bounds):
        """Determine if we can use the shorter 16-bit EMR structures.
        If all the numbers can fit within 16 bit integers, return
        true.  The bounds 4-tuple is (left,top,right,bottom)."""

        SHRT_MIN=-32768
        SHRT_MAX=32767
        if bounds[0]>=SHRT_MIN and bounds[1]>=SHRT_MIN and bounds[2]<=SHRT_MAX and bounds[3]<=SHRT_MAX:
            return True
        return False

    def _appendOptimize16(self,points,cls16,cls):
        bounds=self._getBounds(points)
        if self._useShort(bounds):
            e=cls16(points,bounds)
        else:
            e=cls(points,bounds)
        if not self._append(e):
            return 0
        return 1

    def _appendOptimizePoly16(self,polylist,cls16,cls):
        """polylist is a list of lists of points, where each inner
        list represents a single polygon or line.  The number of
        polygons is the size of the outer list."""
        points=[]
        polycounts=[]
        for polygon in polylist:
            count=0
            for point in polygon:
                points.append(point)
                count+=1
            polycounts.append(count)
        
        bounds=self._getBounds(points)
        if self._useShort(bounds):
            e=cls16(points,polycounts,bounds)
        else:
            e=cls(points,polycounts,bounds)
        if not self._append(e):
            return 0
        return 1

    def _appendHandle(self,e):
        handle=self.dc.addObject(e)
        if not self._append(e):
            self.dc.popObject()
            return 0
        e.handle=handle
        return handle

    def GetStockObject(self,obj):
        """

Retrieve the handle for a predefined graphics object. Stock objects
include (at least) the following:

 - WHITE_BRUSH
 - LTGRAY_BRUSH
 - GRAY_BRUSH
 - DKGRAY_BRUSH
 - BLACK_BRUSH
 - NULL_BRUSH
 - HOLLOW_BRUSH
 - WHITE_PEN
 - BLACK_PEN
 - NULL_PEN
 - OEM_FIXED_FONT
 - ANSI_FIXED_FONT
 - ANSI_VAR_FONT
 - SYSTEM_FONT
 - DEVICE_DEFAULT_FONT
 - DEFAULT_PALETTE
 - SYSTEM_FIXED_FONT
 - DEFAULT_GUI_FONT

@param    obj:  	number of stock object.

@return:    handle of stock graphics object.
@rtype: int
@type obj: int

        """
        if obj>=0 and obj<=STOCK_LAST:
            return obj|0x80000000
        raise IndexError("Undefined stock object.")

    def SelectObject(self,handle):
        """

Make the given graphics object current.

@param handle: handle of graphics object to make current.

@return:
    the handle of the current graphics object which obj replaces.

@rtype: int
@type handle: int

        """
        return self._append(_EMR._SELECTOBJECT(self.dc,handle))

    def DeleteObject(self,handle):
        """

Delete the given graphics object. Note that, now, only those contexts
into which the object has been selected get a delete object
records.

@param    handle:  	handle of graphics object to delete.

@return:    true if the object was successfully deleted.
@rtype: int
@type handle: int

        """
        e=_EMR._DELETEOBJECT(self.dc,handle)
        self.dc.removeObject(handle)
        return self._append(e)

    def CreatePen(self,style,width,color):
        """

Create a pen, used to draw lines and path outlines.


@param    style:  	the style of the new pen, one of:
 - PS_SOLID
 - PS_DASH
 - PS_DOT
 - PS_DASHDOT
 - PS_DASHDOTDOT
 - PS_NULL
 - PS_INSIDEFRAME
 - PS_USERSTYLE
 - PS_ALTERNATE
@param    width:  	the width of the new pen.
@param    color:  	(r,g,b) tuple or the packed integer L{color<RGB>} of the new pen.

@return:    handle to the new pen graphics object.
@rtype: int
@type style: int
@type width: int
@type color: int

        """
        return self._appendHandle(_EMR._CREATEPEN(style,width,_normalizeColor(color)))
        
    def CreateSolidBrush(self,color):
        """

Create a solid brush used to fill polygons.
@param color: the L{color<RGB>} of the solid brush.
@return: handle to brush graphics object.

@rtype: int
@type color: int

        """
        return self._appendHandle(_EMR._CREATEBRUSHINDIRECT(color=_normalizeColor(color)))

    def CreateHatchBrush(self,hatch,color):
        """

Create a hatched brush used to fill polygons.

B{Note:} Currently appears unsupported in OpenOffice.

@param hatch: integer representing type of fill:
 - HS_HORIZONTAL
 - HS_VERTICAL  
 - HS_FDIAGONAL 
 - HS_BDIAGONAL 
 - HS_CROSS     
 - HS_DIAGCROSS 
@type hatch: int
@param color: the L{color<RGB>} of the 'on' pixels of the brush.
@return: handle to brush graphics object.

@rtype: int
@type color: int

        """
        return self._appendHandle(_EMR._CREATEBRUSHINDIRECT(hatch=hatch,color=_normalizeColor(color)))

    def SetBkColor(self,color):
        """

Set the background color used for any transparent regions in fills or
hatched brushes.

B{Note:} Currently appears sporadically supported in OpenOffice.

@param color: background L{color<RGB>}.
@return: previous background L{color<RGB>}.
@rtype: int
@type color: int

        """
        e=_EMR._SETBKCOLOR(_normalizeColor(color))
        if not self._append(e):
            return 0
        return 1

    def SetBkMode(self,mode):
        """

Set the background mode for interaction between transparent areas in
the region to be drawn and the existing background.

The choices for mode are:
 - TRANSPARENT
 - OPAQUE

B{Note:} Currently appears sporadically supported in OpenOffice.

@param mode: background mode.
@return: previous background mode.
@rtype: int
@type mode: int

        """
        e=_EMR._SETBKMODE(mode)
        if not self._append(e):
            return 0
        return 1

    def SetPolyFillMode(self,mode):
        """

Set the polygon fill mode.  Generally these modes produce
different results only when the edges of the polygons overlap
other edges.

@param mode: fill mode with the following options:
 - ALTERNATE - fills area between odd and even numbered sides
 - WINDING - fills all area as long as a point is between any two sides
@return: previous fill mode.
@rtype: int
@type mode: int

        """
        e=_EMR._SETPOLYFILLMODE(mode)
        if not self._append(e):
            return 0
        return 1

    def SetMapMode(self,mode):
        """

Set the window mapping mode.  This is the mapping between pixels in page space to pixels in device space.  Page space is the coordinate system that is used for all the drawing commands -- it is how pixels are identified and figures are placed in the metafile.  They are integer units.

Device space is the coordinate system of the final output, measured in physical dimensions such as mm, inches, or twips.  It is this coordinate system that provides the scaling that makes metafiles into a scalable graphics format.
 - MM_TEXT: each unit in page space is mapped to one pixel
 - MM_LOMETRIC: 1 page unit = .1 mm in device space
 - MM_HIMETRIC: 1 page unit = .01 mm in device space
 - MM_LOENGLISH: 1 page unit = .01 inch in device space
 - MM_HIENGLISH: 1 page unit = .001 inch in device space
 - MM_TWIPS: 1 page unit = 1/20 point (or 1/1440 inch)
 - MM_ISOTROPIC: 1 page unit = user defined ratio, but axes equally scaled
 - MM_ANISOTROPIC: 1 page unit = user defined ratio, axes may be independently scaled
@param mode: window mapping mode.
@return: previous window mapping mode, or zero if error.
@rtype: int
@type mode: int
        """
        e=_EMR._SETMAPMODE(mode)
        if not self._append(e):
            return 0
        return 1
        
    def SetViewportOrgEx(self,xv,yv):
        """

Set the origin of the viewport, which translates the origin of the
coordinate system by (xv,yv).  A pixel drawn at (x,y) in the new
coordinate system will be displayed at (x+xv,y+yv) in terms of the
previous coordinate system.

Contrast this with L{SetWindowOrgEx}, which seems to be the opposite
translation.  So, if in addition, the window origin is set to (xw,yw)
using L{SetWindowOrgEx}, a pixel drawn at (x,y) will be displayed at
(x-xw+xv,y-yw+yv) in terms of the original coordinate system.
        

@param xv: new x position of the viewport origin.
@param yv: new y position of the viewport origin.
@return: previous viewport origin
@rtype: 2-tuple (x,y) if successful, or None if unsuccessful
@type xv: int
@type yv: int
        """
        e=_EMR._SETVIEWPORTORGEX(xv,yv)
        if not self._append(e):
            return None
        old=(self.dc.viewport_x,self.dc.viewport_y)
        self.dc.viewport_x=xv
        self.dc.viewport_y=yv
        return old

    def GetViewportOrgEx(self):
        """

Get the origin of the viewport.
@return: returns the current viewport origin.
@rtype: 2-tuple (x,y)
        """
        return (self.dc.viewport_x,self.dc.viewport_y)
    
    def SetWindowOrgEx(self,xw,yw):
        """

Set the origin of the window, which translates the origin of the
coordinate system by (-xw,-yw).  A pixel drawn at (x,y) in the new
coordinate system will be displayed at (x-xw,y-yw) in terms of the
previous coordinate system.

Contrast this with L{SetViewportOrgEx}, which seems to be the opposite
translation.  So, if in addition, the viewport origin is set to
(xv,yv) using L{SetViewportOrgEx}, a pixel drawn at (x,y) will be
displayed at (x-xw+xv,y-yw+yv) in terms of the original coordinate
system.

@param xw: new x position of the window origin.
@param yw: new y position of the window origin.
@return: previous window origin
@rtype: 2-tuple (x,y) if successful, or None if unsuccessful
@type xw: int
@type yw: int
        """
        e=_EMR._SETWINDOWORGEX(xw,yw)
        if not self._append(e):
            return None
        old=(self.dc.window_x,self.dc.window_y)
        self.dc.window_x=xw
        self.dc.window_y=yw
        return old

    def GetWindowOrgEx(self):
        """

Get the origin of the window.
@return: returns the current window  origin.
@rtype: 2-tuple (x,y)

        """
        return (self.dc.window_x,self.dc.window_y)

    def SetViewportExtEx(self,x,y):
        """
Set the dimensions of the viewport in device units.  Device units are
physical dimensions, in millimeters.  The total extent is equal to the
width is millimeters multiplied by the density of pixels per
millimeter in that dimension.

Note: this is only usable when L{SetMapMode} has been set to
MM_ISOTROPIC or MM_ANISOTROPIC.

@param x: new width of the viewport.
@param y: new height of the viewport.
@return: returns the previous size of the viewport.
@rtype: 2-tuple (width,height) if successful, or None if unsuccessful
@type x: int
@type y: int
        """
        e=_EMR._SETVIEWPORTEXTEX(x,y)
        if not self._append(e):
            return None
        old=(self.dc.viewport_ext_x,self.dc.viewport_ext_y)
        self.dc.viewport_ext_x=xv
        self.dc.viewport_ext_y=yv
        return old

    def ScaleViewportExtEx(self,x_num,x_den,y_num,y_den):
        """

Scale the dimensions of the viewport.
@param x_num: numerator of x scale
@param x_den: denominator of x scale
@param y_num: numerator of y scale
@param y_den: denominator of y scale
@return: returns the previous size of the viewport.
@rtype: 2-tuple (width,height) if successful, or None if unsuccessful
@type x_num: int
@type x_den: int
@type y_num: int
@type y_den: int
        """
        e=_EMR._EMR._SCALEVIEWPORTEXTEX(x_num,x_den,y_num,y_den)
        if not self._append(e):
            return None
        old=(self.dc.viewport_ext_x,self.dc.viewport_ext_y)
        self.dc.viewport_ext_x=old[0]*x_num/x_den
        self.dc.viewport_ext_y=old[1]*y_num/y_den
        return old

    def GetViewportExtEx(self):
        """

Get the dimensions of the viewport in device units (i.e. physical dimensions).
@return: returns the size of the viewport.
@rtype: 2-tuple (width,height)

        """
        old=(self.dc.viewport_ext_x,self.dc.viewport_ext_y)
        return old

    def SetWindowExtEx(self,x,y):
        """

Set the dimensions of the window.  Window size is measured in integer
numbers of pixels (logical units).

Note: this is only usable when L{SetMapMode} has been set to
MM_ISOTROPIC or MM_ANISOTROPIC.

@param x: new width of the window.
@param y: new height of the window.
@return: returns the previous size of the window.
@rtype: 2-tuple (width,height) if successful, or None if unsuccessful
@type x: int
@type y: int
        """
        e=_EMR._SETWINDOWEXTEX(x,y)
        if not self._append(e):
            return None
        old=(self.dc.window_ext_x,self.dc.window_ext_y)
        self.dc.window_ext_x=x
        self.dc.window_ext_y=y
        return old

    def ScaleWindowExtEx(self,x_num,x_den,y_num,y_den):
        """

Scale the dimensions of the window.
@param x_num: numerator of x scale
@param x_den: denominator of x scale
@param y_num: numerator of y scale
@param y_den: denominator of y scale
@return: returns the previous size of the window.
@rtype: 2-tuple (width,height) if successful, or None if unsuccessful
@type x_num: int
@type x_den: int
@type y_num: int
@type y_den: int
        """
        e=_EMR._SCALEWINDOWEXTEX(x_num,x_den,y_num,y_den)
        if not self._append(e):
            return None
        old=(self.dc.window_ext_x,self.dc.window_ext_y)
        self.dc.window_ext_x=old[0]*x_num/x_den
        self.dc.window_ext_y=old[1]*y_num/y_den
        return old

    def GetWindowExtEx(self):
        """

Get the dimensions of the window in logical units (integer numbers of pixels).
@return: returns the size of the window.
@rtype: 2-tuple (width,height)
        """
        old=(self.dc.window_ext_x,self.dc.window_ext_y)
        return old


    def SetWorldTransform(self,m11=1.0,m12=0.0,m21=0.0,m22=1.0,dx=0.0,dy=0.0):
        """
Set the world coordinate to logical coordinate linear transform for
subsequent operations.  With this matrix operation, you can translate,
rotate, scale, shear, or a combination of all four.  The matrix
operation is defined as follows where (x,y) are the original
coordinates and (x',y') are the transformed coordinates::

 | x |   | m11 m12 0 |   | x' |
 | y | * | m21 m22 0 | = | y' |
 | 0 |   | dx  dy  1 |   | 0  |
 
or, the same thing defined as a system of linear equations::

 x' = x*m11 + y*m21 + dx
 y' = x*m12 + y*m22 + dy

http://msdn.microsoft.com/library/en-us/gdi/cordspac_0inn.asp
says that the offsets are in device coordinates, not pixel
coordinates.

B{Note:} Currently partially supported in OpenOffice.

@param m11: matrix entry
@type m11: float
@param m12: matrix entry
@type m12: float
@param m21: matrix entry
@type m21: float
@param m22: matrix entry
@type m22: float
@param dx: x shift
@type dx: float
@param dy: y shift
@type dy: float
@return: status
@rtype: boolean

        """
        return self._append(_EMR._SETWORLDTRANSFORM(m11,m12,m21,m22,dx,dy))
        
    def ModifyWorldTransform(self,mode,m11=1.0,m12=0.0,m21=0.0,m22=1.0,dx=0.0,dy=0.0):
        """
Change the current linear transform.  See L{SetWorldTransform} for a
description of the matrix parameters.  The new transform may be
modified in one of three ways, set by the mode parameter:

 - MWT_IDENTITY: reset the transform to the identity matrix (the matrix parameters are ignored).
 - MWT_LEFTMULTIPLY: multiply the matrix represented by these parameters by the current world transform to get the new transform.
 - MWT_RIGHTMULTIPLY: multiply the current world tranform by the matrix represented here to get the new transform.
 
The reason that there are two different multiplication types is that
matrix multiplication is not commutative, which means the order of
multiplication makes a difference.

B{Note:} The parameter order was changed from GDI standard so that I
could make the matrix parameters optional in the case of MWT_IDENTITY.

B{Note:} Currently appears unsupported in OpenOffice.

@param mode: MWT_IDENTITY, MWT_LEFTMULTIPLY, or MWT_RIGHTMULTIPLY
@type mode: int
@param m11: matrix entry
@type m11: float
@param m12: matrix entry
@type m12: float
@param m21: matrix entry
@type m21: float
@param m22: matrix entry
@type m22: float
@param dx: x shift
@type dx: float
@param dy: y shift
@type dy: float
@return: status
@rtype: boolean

        """
        return self._append(_EMR._MODIFYWORLDTRANSFORM(m11,m12,m21,m22,dx,dy,mode))
        

    def SetPixel(self,x,y,color):
        """

Set the pixel to the given color.
@param x: the horizontal position.
@param y: the vertical position.
@param color: the L{color<RGB>} to set the pixel.
@type x: int
@type y: int
@type color: int or (r,g,b) tuple

        """
        return self._append(_EMR._SETPIXELV(x,y,_normalizeColor(color)))

    def Polyline(self,points):
        """

Draw a sequence of connected lines.
@param points: list of x,y tuples
@return: true if polyline is successfully rendered.
@rtype: int
@type points: tuple

        """
        return self._appendOptimize16(points,_EMR._POLYLINE16,_EMR._POLYLINE)
    

    def PolyPolyline(self,polylines):
        """

Draw multiple polylines.  The polylines argument is a list of lists,
where each inner list represents a single polyline.  Each polyline is
described by a list of x,y tuples as in L{Polyline}.  For example::

  lines=[[(100,100),(200,100)],
         [(300,100),(400,100)]]
  emf.PolyPolyline(lines)

draws two lines, one from 100,100 to 200,100, and another from 300,100
to 400,100.

@param polylines: list of lines, where each line is a list of x,y tuples
@type polylines: list
@return: true if polypolyline is successfully rendered.
@rtype: int

        """
        return self._appendOptimizePoly16(polylines,_EMR._POLYPOLYLINE16,_EMR._POLYPOLYLINE)
    

    def Polygon(self,points):
        """

Draw a closed figure bounded by straight line segments.  A polygon is
defined by a list of points that define the endpoints for a series of
connected straight line segments.  The end of the last line segment is
automatically connected to the beginning of the first line segment,
the border is drawn with the current pen, and the interior is filled
with the current brush.  See L{SetPolyFillMode} for the fill effects
when an overlapping polygon is defined.

@param points: list of x,y tuples
@return: true if polygon is successfully rendered.
@rtype: int
@type points: tuple

        """
        if len(points)==4:
            if points[0][0]==points[1][0] and points[2][0]==points[3][0] and points[0][1]==points[3][1] and points[1][1]==points[2][1]:
                if self.verbose: print "converting to rectangle, option 1:"
                return self.Rectangle(points[0][0],points[0][1],points[2][0],points[2][1])
            elif points[0][1]==points[1][1] and points[2][1]==points[3][1] and points[0][0]==points[3][0] and points[1][0]==points[2][0]:
                if self.verbose: print "converting to rectangle, option 2:"
                return self.Rectangle(points[0][0],points[0][1],points[2][0],points[2][1])
        return self._appendOptimize16(points,_EMR._POLYGON16,_EMR._POLYGON)


    def PolyPolygon(self,polygons):
        """

Draw multiple polygons.  The polygons argument is a list of lists,
where each inner list represents a single polygon.  Each polygon is
described by a list of x,y tuples as in L{Polygon}.  For example::

  lines=[[(100,100),(200,100),(200,200),(100,200)],
         [(300,100),(400,100),(400,200),(300,200)]]
  emf.PolyPolygon(lines)

draws two squares.

B{Note:} Currently partially supported in OpenOffice.  The line width
is ignored and the polygon border is not closed (the final point is
not connected to the starting point in each polygon).

@param polygons: list of polygons, where each polygon is a list of x,y tuples
@type polygons: list
@return: true if polypolygon is successfully rendered.
@rtype: int

        """
        return self._appendOptimizePoly16(polygons,_EMR._POLYPOLYGON16,_EMR._POLYPOLYGON)
    

    def Ellipse(self,left,top,right,bottom):
        """

Draw an ellipse using the current pen.
@param left: x position of left side of ellipse bounding box.
@param top: y position of top side of ellipse bounding box.
@param right: x position of right edge of ellipse bounding box.
@param bottom: y position of bottom edge of ellipse bounding box.
@return: true if rectangle was successfully rendered.
@rtype: int
@type left: int
@type top: int
@type right: int
@type bottom: int

        """
        return self._append(_EMR._ELLIPSE((left,top,right,bottom)))
        
    def Rectangle(self,left,top,right,bottom):
        """

Draw a rectangle using the current pen.
@param left: x position of left side of ellipse bounding box.
@param top: y position of top side of ellipse bounding box.
@param right: x position of right edge of ellipse bounding box.
@param bottom: y position of bottom edge of ellipse bounding box.
@return: true if rectangle was successfully rendered.
@rtype: int
@type left: int
@type top: int
@type right: int
@type bottom: int

        """
        return self._append(_EMR._RECTANGLE((left,top,right,bottom)))

    def RoundRect(self,left,top,right,bottom,cornerwidth,cornerheight):
        """

Draw a rectangle with rounded corners using the current pen.
@param left: x position of left side of ellipse bounding box.
@param top: y position of top side of ellipse bounding box.
@param right: x position of right edge of ellipse bounding box.
@param bottom: y position of bottom edge of ellipse bounding box.
@param cornerwidth: width of the ellipse that defines the roundness of the corner.
@param cornerheight: height of ellipse
@return: true if rectangle was successfully rendered.
@rtype: int
@type left: int
@type top: int
@type right: int
@type bottom: int
@type cornerwidth: int
@type cornerheight: int

        """
        return self._append(_EMR._ROUNDRECT((left,top,right,bottom,
                                           cornerwidth,cornerheight)))

    def Arc(self,left,top,right,bottom,xstart,ystart,xend,yend):
        """

Draw an arc of an ellipse.  The ellipse is specified by its bounding
rectange and two lines from its center to indicate the start and end
angles.  left, top, right, bottom describe the bounding rectangle of
the ellipse.  The start point given by xstart,ystert defines a ray
from the center of the ellipse through the point and out to infinity.
The point at which this ray intersects the ellipse is the starting
point of the arc.  Similarly, the infinite radial ray from the center
through the end point defines the end point of the ellipse.  The arc
is drawn in a counterclockwise direction, and if the start and end
rays are coincident, a complete ellipse is drawn.

@param left: x position of left edge of arc box.
@param top: y position of top edge of arc box.
@param right: x position of right edge of arc box.
@param bottom: y position bottom edge of arc box.
@param xstart: x position of arc start.
@param ystart: y position of arc start.
@param xend: x position of arc end.
@param yend: y position of arc end.
@return: true if arc was successfully rendered.
@rtype: int
@type left: int
@type top: int
@type right: int
@type bottom: int
@type xstart: int
@type ystart: int
@type xend: int
@type yend: int

        """
        return self._append(_EMR._ARC(left,top,right,bottom,
                                    xstart,ystart,xend,yend))

    def Chord(self,left,top,right,bottom,xstart,ystart,xend,yend):
        """

Draw a chord of an ellipse.  A chord is a closed region bounded by an
arc and the [straight] line between the two points that define the arc
start and end.  The arc start and end points are defined as in L{Arc}.

@param left: x position of left edge of arc box.
@param top: y position of top edge of arc box.
@param right: x position of right edge of arc box.
@param bottom: y position bottom edge of arc box.
@param xstart: x position of arc start.
@param ystart: y position of arc start.
@param xend: x position of arc end.
@param yend: y position of arc end.
@return: true if arc was successfully rendered.
@rtype: int
@type left: int
@type top: int
@type right: int
@type bottom: int
@type xstart: int
@type ystart: int
@type xend: int
@type yend: int

        """
        return self._append(_EMR._CHORD(left,top,right,bottom,
                                    xstart,ystart,xend,yend))

    def Pie(self,left,top,right,bottom,xstart,ystart,xend,yend):
        """

Draw a pie slice of an ellipse.  The ellipse is specified as in
L{Arc}, and it is filled with the current brush.

@param left: x position of left edge of arc box.
@param top: y position of top edge of arc box.
@param right: x position of right edge of arc box.
@param bottom: y position bottom edge of arc box.
@param xstart: x position of arc start.
@param ystart: y position of arc start.
@param xend: x position of arc end.
@param yend: y position of arc end.
@return: true if arc was successfully rendered.
@rtype: int
@type left: int
@type top: int
@type right: int
@type bottom: int
@type xstart: int
@type ystart: int
@type xend: int
@type yend: int

        """
        if xstart==xend and ystart==yend:
            # Fix for OpenOffice: doesn't render a full ellipse when
            # the start and end angles are the same
            e=_EMR._ELLIPSE((left,top,right,bottom))
        else:
            e=_EMR._PIE(left,top,right,bottom,xstart,ystart,xend,yend)
        return self._append(e)

    def PolyBezier(self,points):
        """

Draw cubic Bezier curves using the list of points as both endpoints
and control points.  The first point is used as the starting point,
the second and thrird points are control points, and the fourth point
is the end point of the first curve.  Subsequent curves need three
points each: two control points and an end point, as the ending point
of the previous curve is used as the starting point for the next
curve.

@param points: list of x,y tuples that are either end points or control points
@return: true if bezier curve was successfully rendered.
@rtype: int
@type points: tuple

        """
        return self._appendOptimize16(points,_EMR._POLYBEZIER16,_EMR._POLYBEZIER)

    def BeginPath(self):
        """

Begin defining a path.  Any previous unclosed paths are discarded.
@return: true if successful.
@rtype: int

        """
        # record next record number as first item in path
        self.pathstart=len(self.records)
        return self._append(_EMR._BEGINPATH())

    def EndPath(self):
        """

End the path definition.
@return: true if successful.
@rtype: int

        """
        return self._append(_EMR._ENDPATH())

    def MoveTo(self,x,y):
        """

Move the current point to the given position and implicitly begin a
new figure or path.

@param x: new x position.
@param y: new y position.
@return: true if position successfully changed (can this fail?)
@rtype: int
@type x: int
@type y: int
        """
        return self._append(_EMR._MOVETOEX(x,y))

    def LineTo(self,x,y):
        """

Draw a straight line using the current pen from the current point to
the given position.

@param x: x position of line end.
@param y: y position of line end.
@return: true if line is drawn (can this fail?)
@rtype: int
@type x: int
@type y: int

        """
        return self._append(_EMR._LINETO(x,y))

    def PolylineTo(self,points):
        """

Draw a sequence of connected lines starting from the current
position and update the position to the final point in the list.

@param points: list of x,y tuples
@return: true if polyline is successfully rendered.
@rtype: int
@type points: tuple

        """
        return self._appendOptimize16(points,_EMR._POLYLINETO16,_EMR._POLYLINETO)

    def ArcTo(self,left,top,right,bottom,xstart,ystart,xend,yend):
        """

Draw an arc and update the current position.  The arc is drawn as
described in L{Arc}, but in addition the start of the arc will be
connected to the previous position and the current position is updated
to the end of the arc so subsequent path operations such as L{LineTo},
L{PolylineTo}, etc. will connect to the end.

B{Note:} Currently appears unsupported in OpenOffice.

@param left: x position of left edge of arc box.
@param top: y position of top edge of arc box.
@param right: x position of right edge of arc box.
@param bottom: y position bottom edge of arc box.
@param xstart: x position of arc start.
@param ystart: y position of arc start.
@param xend: x position of arc end.
@param yend: y position of arc end.
@return: true if arc was successfully rendered.
@rtype: int
@type left: int
@type top: int
@type right: int
@type bottom: int
@type xstart: int
@type ystart: int
@type xend: int
@type yend: int

        """
        return self._append(_EMR._ARCTO(left,top,right,bottom,
                                    xstart,ystart,xend,yend))

    def PolyBezierTo(self,points):
        """

Draw cubic Bezier curves, as described in L{PolyBezier}, but in
addition draw a line from the previous position to the start of the
curve.  If the arc is successfully rendered, the current position is
updated so that subsequent path operations such as L{LineTo},
L{PolylineTo}, etc. will follow from the end of the curve.

@param points: list of x,y tuples that are either end points or control points
@return: true if bezier curve was successfully rendered.
@rtype: int
@type points: tuple

        """
        return self._appendOptimize16(points,_EMR._POLYBEZIERTO16,_EMR._POLYBEZIERTO)

    def CloseFigure(self):
        """

Close a currently open path, which connects the current position to the starting position of a figure.  Usually the starting position is the most recent call to L{MoveTo} after L{BeginPath}.

@return: true if successful

@rtype: int

        """
        return self._append(_EMR._CLOSEFIGURE())

    def FillPath(self):
        """

Close any currently open path and fills it using the currently
selected brush and polygon fill mode.

@return: true if successful.
@rtype: int

        """
        bounds=self._getPathBounds()
        return self._append(_EMR._FILLPATH(bounds))

    def StrokePath(self):
        """

Close any currently open path and outlines it using the currently
selected pen.

@return: true if successful.
@rtype: int

        """
        bounds=self._getPathBounds()
        return self._append(_EMR._STROKEPATH(bounds))

    def StrokeAndFillPath(self):
        """

Close any currently open path, outlines it using the currently
selected pen, and fills it using the current brush.  Same as stroking
and filling using both the L{FillPath} and L{StrokePath} options,
except that the pixels that would be in the overlap region to be both
stroked and filled are optimized to be only stroked.

B{Note:} Supported in OpenOffice 2.*, unsupported in OpenOffice 1.*.

@return: true if successful.
@rtype: int

        """
        bounds=self._getPathBounds()
        return self._append(_EMR._STROKEANDFILLPATH(bounds))

    def SelectClipPath(self,mode=RGN_COPY):
        """

Use the current path as the clipping path.  The current path must be a
closed path (i.e. with L{CloseFigure} and L{EndPath})

B{Note:} Currently unsupported in OpenOffice -- it apparently uses the
bounding rectangle of the path as the clip area, not the path itself.

@param mode: one of the following values that specifies how to modify the clipping path
 - RGN_AND: the new clipping path becomes the intersection of the old path and the current path
 - RGN_OR: the new clipping path becomes the union of the old path and the current path
 - RGN_XOR: the new clipping path becomes the union of the old path and the current path minus the intersection of the old and current path
 - RGN_DIFF: the new clipping path becomes the old path where any overlapping region of the current path is removed
 - RGN_COPY: the new clipping path is set to the current path and the old path is thrown away

@return: true if successful.
@rtype: int

        """
        return self._append(_EMR._SELECTCLIPPATH(mode))

    def SaveDC(self):
        """

Saves the current state of the graphics mode (such as line and fill
styles, font, clipping path, drawing mode and any transformations) to
a stack.  This state can be restored by L{RestoreDC}.

B{Note:} Currently unsupported in OpenOffice -- it apparently uses the
bounding rectangle of the path as the clip area, not the path itself.

@return: value of the saved state.
@rtype: int

        """
        return self._append(_EMR._SAVEDC())

    def RestoreDC(self,stackid):
        """

Restores the state of the graphics mode to a stack.  The L{stackid}
parameter is either a value returned by L{SaveDC}, or if negative, is
the number of states relative to the top of the save stack.  For
example, C{stackid == -1} is the most recently saved state.

B{Note:} If the retrieved state is not at the top of the stack, any
saved states above it are thrown away.

B{Note:} Currently unsupported in OpenOffice -- it apparently uses the
bounding rectangle of the path as the clip area, not the path itself.

@param stackid: stack id number from L{SaveDC} or negative number for relative stack location
@type stackid: int
@return: nonzero for success
@rtype: int

        """
        return self._append(_EMR._RESTOREDC(-1))

    def SetTextAlign(self,alignment):
        """

Set the subsequent alignment of drawn text. You can also pass a flag
indicating whether or not to update the current point to the end of the
text. Alignment may have the (sum of) values:
 - TA_NOUPDATECP
 - TA_UPDATECP
 - TA_LEFT
 - TA_RIGHT
 - TA_CENTER
 - TA_TOP
 - TA_BOTTOM
 - TA_BASELINE
 - TA_RTLREADING
@param alignment: new text alignment.
@return: previous text alignment value.
@rtype: int
@type alignment: int

        """
        return self._append(_EMR._SETTEXTALIGN(alignment))

    def SetTextColor(self,color):
        """

Set the text foreground color.
@param color: text foreground L{color<RGB>}.
@return: previous text foreground L{color<RGB>}.
@rtype: int
@type color: int

        """
        e=_EMR._SETTEXTCOLOR(_normalizeColor(color))
        if not self._append(e):
            return 0
        return 1

    def CreateFont(self,height,width=0,escapement=0,orientation=0,weight=FW_NORMAL,italic=0,underline=0,strike_out=0,charset=ANSI_CHARSET,out_precision=OUT_DEFAULT_PRECIS,clip_precision=CLIP_DEFAULT_PRECIS,quality=DEFAULT_QUALITY,pitch_family='DEFAULT_PITCH|FF_DONTCARE',name='Times New Roman'):
        """

Create a new font object. Presumably, when rendering the EMF the
system tries to find a reasonable approximation to all the requested
attributes.

@param height: specified one of two ways:
 - if height>0: locate the font using the specified height as the typical cell height
 - if height<0: use the absolute value of the height as the typical glyph height.
@param width: typical glyph width.  If zero, the typical aspect ratio of the font is used.
@param escapement: angle, in degrees*10, of rendered string rotation.  Note that escapement and orientation must be the same.
@param orientation: angle, in degrees*10, of rendered string rotation.  Note that escapement and orientation must be the same.
@param weight: weight has (at least) the following values:
 - FW_DONTCARE
 - FW_THIN
 - FW_EXTRALIGHT
 - FW_ULTRALIGHT
 - FW_LIGHT
 - FW_NORMAL
 - FW_REGULAR
 - FW_MEDIUM
 - FW_SEMIBOLD
 - FW_DEMIBOLD
 - FW_BOLD
 - FW_EXTRABOLD
 - FW_ULTRABOLD
 - FW_HEAVY
 - FW_BLACK
@param italic: non-zero means try to find an italic version of the face.
@param underline: non-zero means to underline the glyphs.
@param strike_out: non-zero means to strike-out the glyphs.
@param charset: select the character set from the following list:
 - ANSI_CHARSET
 - DEFAULT_CHARSET
 - SYMBOL_CHARSET
 - SHIFTJIS_CHARSET
 - HANGEUL_CHARSET
 - HANGUL_CHARSET
 - GB2312_CHARSET
 - CHINESEBIG5_CHARSET
 - GREEK_CHARSET
 - TURKISH_CHARSET
 - HEBREW_CHARSET
 - ARABIC_CHARSET
 - BALTIC_CHARSET
 - RUSSIAN_CHARSET
 - EE_CHARSET
 - EASTEUROPE_CHARSET
 - THAI_CHARSET
 - JOHAB_CHARSET
 - MAC_CHARSET
 - OEM_CHARSET
@param out_precision: the precision of the face may have on of the
following values:
 - OUT_DEFAULT_PRECIS
 - OUT_STRING_PRECIS
 - OUT_CHARACTER_PRECIS
 - OUT_STROKE_PRECIS
 - OUT_TT_PRECIS
 - OUT_DEVICE_PRECIS
 - OUT_RASTER_PRECIS
 - OUT_TT_ONLY_PRECIS
 - OUT_OUTLINE_PRECIS
@param clip_precision: the precision of glyph clipping may have one of the
following values:
 - CLIP_DEFAULT_PRECIS
 - CLIP_CHARACTER_PRECIS
 - CLIP_STROKE_PRECIS
 - CLIP_MASK
 - CLIP_LH_ANGLES
 - CLIP_TT_ALWAYS
 - CLIP_EMBEDDED
@param quality: (subjective) quality of the font. Choose from the following
values:
 - DEFAULT_QUALITY
 - DRAFT_QUALITY
 - PROOF_QUALITY
 - NONANTIALIASED_QUALITY
 - ANTIALIASED_QUALITY
@param pitch_family: the pitch and family of the font face if the named font can't be found. Combine the pitch and style using a binary or.
 - Pitch:
   - DEFAULT_PITCH
   - FIXED_PITCH
   - VARIABLE_PITCH
   - MONO_FONT
 - Style:
   - FF_DONTCARE
   - FF_ROMAN
   - FF_SWISS
   - FF_MODERN
   - FF_SCRIPT
   - FF_DECORATIVE
@param name: ASCII string containing the name of the font face.
@return: handle of font.
@rtype: int
@type height: int
@type width: int
@type escapement: int
@type orientation: int
@type weight: int
@type italic: int
@type underline: int
@type strike_out: int
@type charset: int
@type out_precision: int
@type clip_precision: int
@type quality: int
@type pitch_family: int
@type name: string

        """
        return self._appendHandle(_EMR._EXTCREATEFONTINDIRECTW(height,width,escapement,orientation,weight,italic,underline,strike_out,charset,out_precision,clip_precision,quality,pitch_family,name))

    def TextOut(self,x,y,text):
        """

Draw a string of text at the given position using the current FONT and
other text attributes.
@param x: x position of text.
@param y: y position of text.
@param text: ASCII text string to render.
@return: true of string successfully drawn.

@rtype: int
@type x: int
@type y: int
@type text: string

        """
        e=_EMR._EXTTEXTOUTA(x,y,text)
        if not self._append(e):
            return 0
        return 1
 




if __name__ == "__main__":
    try:
        from optparse import OptionParser

        parser=OptionParser(usage="usage: %prog [options] emf-files...")
        parser.add_option("-v", action="store_true", dest="verbose", default=False)
        parser.add_option("-s", action="store_true", dest="save", default=False)
        parser.add_option("-o", action="store", dest="outputfile", default=None)
        (options, args) = parser.parse_args()
        # print options
    except:
        # hackola to work with Python 2.2, but this shouldn't be a
        # factor when imported in normal programs because __name__
        # will never equal "__main__", so this will never get called.
        class data:
            verbose=True
            save=True
            outputfile=None

        options=data()
        args=sys.argv[1:]

    if len(args)>0:
        for filename in args:
            e=EMF(verbose=options.verbose)
            e.load(filename)
            if options.save:
                if not options.outputfile:
                    options.outputfile=filename+".out.emf"
                print "Saving %s..." % options.outputfile
                ret=e.save(options.outputfile)
                if ret:
                    print "%s saved successfully." % options.outputfile
                else:
                    print "problem saving %s!" % options.outputfile
    else:
        e=EMF(verbose=options.verbose)
        e.save("new.emf")
