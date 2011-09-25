import sys
import numpy as np

def read_heka(filename):
    ascfile = open(filename)

    nchannels = 0
    nsweeps = 0
    newsweep = True
    header = True
    channels = []
    channelnames = []
    channelunits = []
    channeldt = []
    istext=False
    sys.stdout.write("Reading")
    sys.stdout.flush()

    for line in ascfile:
        words = line.replace('\r','').replace('\n','').split(",")
        try:
            np = int(words[0])
            istext=False
        except:
            istext = True
            if not header:
                newsweep=True
            else:
                prevline = words
            
        if not istext:
            if header:
                nchannels = (len(words)-1)/2
                channels = [list() for i in range(nchannels)]
                for nc in range(nchannels):
                    channelnames.append(
                        prevline[nc*2+2].replace("\"",'').strip()[:-3])
                    channelunits.append(
                        prevline[nc*2+2][prevline[nc*2+2].find('[')+1: \
                                         prevline[nc*2+2].find(']')])
                
                header=False
            if newsweep:
                for channel in channels:
                    channel.append(list())
                nsweeps += 1
                sys.stdout.write(".")
                sys.stdout.flush()
                newsweep=False
            if len(channels[-1][-1])==0:
                dt0 = float(words[1])
            if len(channels[-1][-1])==1:
                dt1 = float(words[1])
                channeldt.append(dt1-dt0)
            for nc, channel in enumerate(channels):
                channel[-1].append(float(words[nc*2+2]))

    return channels, channelnames, channelunits, channeldt

def read_heka_stf(filename):
    channels, channelnames, channelunits, channeldt = read_heka(filename)
    for nc, channel in enumerate(channels):
        if channelunits[nc]=="V":
            for ns, sweep in enumerate(channel):
                channels[nc][ns] = np.array(channels[nc][ns])
                channels[nc][ns] *= 1.0e3
            channelunits[nc]="mV"
        if channelunits[nc]=="A":
            for ns, sweep in enumerate(channel):
                channels[nc][ns] = np.array(channels[nc][ns])
                channels[nc][ns] *= 1.0e12
            channelunits[nc]="pA"

    import stf
    stf.new_window_list(channels)
    for nc, name in enumerate(channelnames):
        stf.set_channel_name(name, nc)
    for nc, units in enumerate(channelunits):
        for ns in range(stf.get_size_channel()):
            stf.set_yunits(units, ns, nc)
    stf.set_sampling_interval(channeldt[0]*1e3)

if __name__=="__main__":
    read_heka("JK100205aa.asc")
