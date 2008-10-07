import sys
sys.argv = ""
import tables
import numpy as N

class Recording():
    def __init__(self, channels, comment, date, time):
        self.channels = channels
        self.comment = comment
        self.date = date
        self.time = time

    def get_list( self ):
        l = list()
        for c in self.channels:
            l.append([ s.data for s in c.sections ])
        return l

class Channel():
    def __init__(self, sections, name):
        self.sections = sections
        self.name = name

class Section():
    def __init__(self, data, dt, xunits, yunits):
        self.data = data
        self.dt = dt
        self.xunits = xunits
        self.yunits = yunits

class RecordingDescription(tables.IsDescription):
    channels = tables.Int32Col()
    date = tables.StringCol(128)
    time = tables.StringCol(128)

class ChannelDescription(tables.IsDescription):
    n_sections = tables.Int32Col()

class SectionDescription(tables.IsDescription):
    dt = tables.Float64Col()
    xunits = tables.StringCol(16)
    yunits = tables.StringCol(16)

def export_hdf5( filename="" ):
    """
    Exports a file in hdf5 format using PyTables.
    """
    stf = __import__("stf")

    if filename=="":
        filename = "%s.h5" % (stf.get_filename())

    h5file = tables.openFile(filename, mode = "w", title = "%s, converted to hdf5" % stf.get_filename())
    # write global file description
    root_table = h5file.createTable(h5file.root, "description", RecordingDescription, "Description of %s" % filename)
    root_row = root_table.row
    root_row['channels'] = stf.get_size_recording()
    root_row['date'] = stf.get_recording_date()
    root_row['time'] = stf.get_recording_time()
    root_row.append()
    root_table.flush()

    # write comment
    comment_group = h5file.createGroup("/", "comment", "multiline file comment")
    strarray = h5file.createArray(comment_group, "comment", [stf.get_recording_comment(),], "multiline file comment")
    
    # create group for channel names:
    chroot_group = h5file.createGroup("/", "channels", "channel names")

    # loop through channels:
    for n_c in range(stf.get_size_recording()):
        channel_name = stf.get_channel_name(n_c);
        if channel_name == "":
            channel_name = "ch%d" % (n_c)
            
        channel_group = h5file.createGroup("/", channel_name, "channel%d" % (n_c))

        # write channel name to root group:
        strarray = h5file.createArray(chroot_group, "ch%d" % n_c, [channel_name,], "channel name")
        
        channel_table = h5file.createTable(channel_group, "description", ChannelDescription, "Description of %s" % channel_name)
        channel_row = channel_table.row
        channel_row['n_sections'] = stf.get_size_channel(n_c)
        channel_row.append()
        channel_table.flush()
        
        if (stf.get_size_channel(n_c)==1):
            max_log10 = 0
        else:
            max_log10 = int(N.log10(stf.get_size_channel(n_c)-1))
        
        for n_s in range(stf.get_size_channel(n_c)):
            # construct a number with leading zeros:
            if n_s==0:
                n10 = 0
            else:
                n10 = int(N.log10(n_s))
            strZero = ""
            for n_z in range(n10,max_log10):
                strZero += "0"

            # construct a section name:
            section_name = stf.get_trace_name(n_s,n_c);
            if section_name == "":
                section_name = "sec%d" % (n_s)

            # create a child group in the channel:
            section_group = h5file.createGroup(channel_group, "section_%s%d" % (strZero, n_s), section_name)
            
            # add data and description:
            array = h5file.createArray(section_group, "data", N.array(stf.get_trace(n_s, n_c), dtype=N.float32), "data in %s" % section_name)
            desc_table = h5file.createTable(section_group, "description", SectionDescription, "description of %s" % section_name)
            desc_row = desc_table.row
            desc_row['dt'] = stf.get_sampling_interval()
            desc_row['xunits'] = stf.get_xunits(n_s,n_c)
            desc_row['yunits'] = stf.get_yunits(n_s,n_c)
            desc_row.append()
            desc_table.flush()

    h5file.close()
    return True

def import_hdf5( filename ):
    """
    Imports a file in hdf5 format stored by stimfit using PyTables, returns a numpy array.
    """
    h5file = tables.openFile( filename, mode='r' )

    # read global file description
    root_node = h5file.getNode("/", "description")
    date = root_node.col("date")[0]
    time = root_node.col("time")[0]
    n_channels = root_node.col("channels")[0]

    # read comment
    com = h5file.getNode("/", "comment")
    comment = ""
    for entry in h5file.walkNodes(com,classname='Array'):
        comment += "%s" % entry.read()[0]

    # read channel names
    channel_names = list()
    chan = h5file.getNode("/", "channels")
    for entry in h5file.walkNodes(chan, classname='Array'):
        channel_names.append( "%s" % entry.read()[0] )

    # read data from channels: 
    channel_list = list()
    for n_c in range(n_channels):
        channel_node = h5file.getNode("/", channel_names[n_c])
        chdesc_node = h5file.getNode( channel_node, "description" )
        n_sections = chdesc_node.col("n_sections")[0]

        # required number of zeros:
        if n_sections==1:
            max_log10 = 0
        else:
            max_log10 = int(N.log10(n_sections-1))

        # loop through sections:
        section_list = list()
        for n_s in range(n_sections):
            # construct a number with leading zeros:
            if n_s==0:
                n10 = 0
            else:
                n10 = int(N.log10(n_s))
            strZero = ""
            for n_z in range(n10,max_log10):
                strZero += "0"

            section_name = "section_%s%d" % ( strZero, n_s)
            section_node = h5file.getNode( channel_node, section_name )
            secdesc_node = h5file.getNode( section_node, "description" )
            dt = secdesc_node.col("dt")[0]
            xunits = secdesc_node.col("xunits")[0]
            yunits = secdesc_node.col("yunits")[0]
            data = h5file.getNode( section_node, "data").read()
            section_list.append( Section(data, dt, xunits, yunits) )

        channel_list.append( Channel(section_list, channel_names[n_c]) )

    h5file.close()

    return Recording( channel_list, comment, date, time )

def open_hdf5( filename ):
    """
    Opens and shows an hdf5 file with stimfit
    """
    rec = import_hdf5( filename )
    stf = __import__("stf")

    stf._gNames_resize( len(rec.channels) )
    for n_c in range(len(rec.channels)):
        stf._gNames_at( rec.channels[n_c].name, n_c )
    
    stf.new_window_list( rec.get_list() )
    n_channels = stf.get_size_recording()
    dt = rec.channels[0].sections[0].dt
    stf.set_sampling_interval( dt )
    stf.set_recording_comment( rec.comment )
    stf.set_recording_date( rec.date )
    stf.set_recording_time( rec.time )
    for n_c in range(stf.get_size_recording()):
        for n_s in range(stf.get_size_channel(n_c)):
            xunits = rec.channels[n_c].sections[n_s].xunits
            yunits = rec.channels[n_c].sections[n_s].yunits

            stf.set_xunits( xunits, n_s, n_c )
            stf.set_yunits( yunits, n_s, n_c )

    return True

def test():
    export_hdf5()
    open_hdf5("/home/cs/data/EE07_04_11_2AC.dat.h5")
