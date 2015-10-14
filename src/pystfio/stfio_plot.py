"""
Some plotting utilities to use scale bars rather than coordinate axes.
04 Feb 2011, C. Schmidt-Hieber, University College London

From the stfio module:
http://code.google.com/p/stimfit
"""

# TODO: Pin scale bars to their position
# TODO: Implement 2-channel plots

import os
import sys

import numpy as np
import numpy.ma as ma

HAS_MPL = True
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid.axislines import Subplot
except ImportError as err:
    HAS_MPL = False
    MPL_ERR = err

    # dummy class
    class Subplot(object):
        pass

scale_dist_x = 0.04
scale_dist_y = 0.04
graph_width = 6.0
graph_height = 4.0
key_dist = 0.04


def save_ma(ftrunk, marr):
    if not isinstance(marr, ma.core.MaskedArray):
        marr = ma.array(marr, mask=False)
    data = np.array(marr)
    mask = np.array(marr.mask)
    np.save(ftrunk + ".data.npy", data)
    np.save(ftrunk + ".mask.npy", mask)


def load_ma(ftrunk):
    data = np.load(ftrunk + ".data.npy")
    mask = np.load(ftrunk + ".mask.npy")
    return ma.array(data, mask=mask)


class Timeseries(object):
    # it this is 2d, the second axis (shape[1]) is time
    def __init__(self, *args, **kwargs):
        if len(args) > 2:
            raise RuntimeError(
                "Timeseries accepts at most two non-keyworded arguments")
        fromFile = False
        # First argument has to be either data or file_trunk
        if isinstance(args[0], str):
            if len(args) > 1:
                raise RuntimeError(
                    "Timeseries accepts only one non-keyworded "
                    "argument if instantiated from file")
            if os.path.exists("%s_data.npy" % args[0]):
                self.data = np.load("%s_data.npy" % args[0])
            else:
                self.data = load_ma("%s_data.npy" % args[0])

            self.dt = np.load("%s_dt.npy" % args[0])

            fxu = open("%s_xunits" % args[0], 'r')
            self.xunits = fxu.read()
            fxu.close()

            fyu = open("%s_yunits" % args[0], 'r')
            self.yunits = fyu.read()
            fyu.close()
            fromFile = True
        else:
            self.data = args[0]
            self.dt = args[1]

        if len(kwargs) > 0 and fromFile:
            raise RuntimeError(
                "Can't set keyword arguments if Timeseries was "
                "instantiated from file")

        for key in kwargs:
            if key == "xunits":
                self.xunits = kwargs["xunits"]
            elif key == "yunits":
                self.yunits = kwargs["yunits"]
            elif key == "linestyle":
                self.linestyle = kwargs["linestyle"]
            elif key == "linewidth":
                self.linewidth = kwargs["linewidth"]
            elif key == "color":
                self.color = kwargs["color"]
            elif key == "colour":
                self.color = kwargs["colour"]
            else:
                raise RuntimeError("Unknown keyword argument: " + key)

        if "xunits" not in kwargs and not fromFile:
            self.xunits = "ms"
        if "yunits" not in kwargs and not fromFile:
            self.yunits = "mV"
        if "linestyle" not in kwargs:
            self.linestyle = "-"
        if "linewidth" not in kwargs:
            self.linewidth = 1.0
        if "color" not in kwargs and "colour" not in kwargs:
            self.color = 'k'

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def __add__(self, other):
        if isinstance(other, Timeseries):
            result = self.data + other.data
        else:
            result = self.data + other

        return self.copy_attributes(result)

    def __mul__(self, other):
        if isinstance(other, Timeseries):
            result = self.data * other.data
        else:
            result = self.data * other

        return self.copy_attributes(result)

    def __sub__(self, other):
        if isinstance(other, Timeseries):
            result = self.data - other.data
        else:
            result = self.data - other

        return self.copy_attributes(result)

    def __div__(self, other):
        if isinstance(other, Timeseries):
            result = self.data / other.data
        else:
            result = self.data / other

        return self.copy_attributes(result)

    def __truediv__(self, other):
        return self.__div__(other)

    def copy_attributes(self, data):
        return Timeseries(
            data, self.dt, xunits=self.xunits, yunits=self.yunits,
            linestyle=self.linestyle, linewidth=self.linewidth,
            color=self.color)

    def x_trange(self, tstart, tend):
        return np.arange(int(tstart/self.dt), int(tend/self.dt), 1.0,
                         dtype=np.float) * self.dt

    def y_trange(self, tstart, tend):
        return self.data[int(tstart/self.dt):int(tend/self.dt)]

    def timearray(self):
        return np.arange(0.0, self.data.shape[-1]) * self.dt

    def duration(self):
        return self.data.shape[-1] * self.dt

    def interpolate(self, newtime, newdt):
        if len(self.data.shape) == 1:
            return Timeseries(np.interp(newtime, self.timearray(), self.data,
                                        left=np.nan, right=np.nan), newdt)
        else:
            # interpolate each row individually:
            # iparray = ma.zeros((self.data.shape[0], len(newtime)))
            # for nrow, row in enumerate(self.data):
            #     flin = \
            #         interpolate.interp1d(
            #            self.timearray(), row,
            #            bounds_error=False, fill_value=np.nan, kind=kind)
            #     iparray[nrow,:]=flin(newtime)
            iparray = ma.array([
                np.interp(
                    newtime, self.timearray(), row, left=np.nan, right=np.nan)
                for nrow, row in enumerate(self.data)])
            return Timeseries(iparray, newdt)

    def maskedarray(self, center, left, right):
        # check whether we have enough data left and right:
        if len(self.data.shape) > 1:
            mask = \
                np.zeros((self.data.shape[0], int((right+left)/self.dt)))
            maskedarray = \
                ma.zeros((self.data.shape[0], int((right+left)/self.dt)))
        else:
            mask = np.zeros((int((right+left)/self.dt)))
            maskedarray = ma.zeros((int((right+left)/self.dt)))
        offset = 0
        if center - left < 0:
            if len(self.data.shape) > 1:
                mask[:, :int((left-center)/self.dt)] = 1
            else:
                mask[:int((left-center)/self.dt)] = 1
            leftindex = 0
            offset = int((left-center)/self.dt)
        else:
            leftindex = int((center-left)/self.dt)
        if center + right >= len(self.data) * self.dt:
            endtime = len(self.data) * self.dt
            if len(self.data.shape) > 1:
                mask[:, -int((center+right-endtime)/self.dt):] = 1
            else:
                mask[-int((center+right-endtime)/self.dt):] = 1
            rightindex = int(endtime/self.dt)
        else:
            rightindex = int((center+right)/self.dt)
        for timest in range(leftindex, rightindex):
                if len(self.data.shape) > 1:
                    if timest-leftindex+offset < maskedarray.shape[1] and \
                       timest < self.data.shape[1]:
                        maskedarray[:, timest-leftindex+offset] = \
                            self.data[:, timest]
                else:
                    if timest-leftindex+offset < len(maskedarray):
                        maskedarray[timest-leftindex+offset] = \
                            self.data[timest]
        maskedarray.mask = ma.make_mask(mask)
        return Timeseries(maskedarray, self.dt)

    def save(self, file_trunk):
        if isinstance(self.data, ma.MaskedArray):
            save_ma("%s_data.npy" % file_trunk, self.data)
        else:
            np.save("%s_data.npy" % file_trunk, self.data)

        np.save("%s_dt.npy" % file_trunk, self.dt)

        fxu = open("%s_xunits" % file_trunk, 'w')
        fxu.write(self.xunits)
        fxu.close()

        fyu = open("%s_yunits" % file_trunk, 'w')
        fyu.write(self.yunits)
        fyu.close()

    def plot(self):
        fig = plt.figure(figsize=(8, 6))

        ax = StandardAxis(fig, 111, hasx=True)
        ax.plot(self.timearray(), self.data, '-k')


class timeseries(Timeseries):
    def __init__(self, *args, **kwargs):
        super(timeseries, self).__init__(*args, **kwargs)
        sys.stderr.write("stfio_plot.timeseries is deprecated. "
                         "Use stfio_plot.Timeseries instead.\n")


class StandardAxis(Subplot):
    def __init__(self, *args, **kwargs):
        if not HAS_MPL:
            raise MPL_ERR

        hasx = kwargs.pop('hasx', False)
        hasy = kwargs.pop('hasy', True)
        kwargs['frameon'] = False

        super(StandardAxis, self).__init__(*args, **kwargs)

        args[0].add_axes(self)
        self.axis["right"].set_visible(False)
        self.axis["top"].set_visible(False)

        if not hasx:
            self.axis["bottom"].set_visible(False)
        if not hasy:
            self.axis["left"].set_visible(False)


def average(tsl):
    # find fastest dt:
    dt_common = 1e12
    for ts in tsl:
        if ts.dt < dt_common:
            newtime = ts.timearray()
            dt_common = ts.dt

    # interpolate all series to new dt:
    tslip = [ts.interpolate(newtime, dt_common) for ts in tsl]
    if len(tslip[0].data.shape) == 1:
        ave = np.empty((len(tslip), len(tslip[0].data)))
    else:
        ave = np.empty(
            (len(tslip), tslip[0].data.shape[0], tslip[0].data.shape[1]))

    for its, ts in enumerate(tslip):
        if len(ts.data.shape) == 1:
            ave[its] = ts.data
        else:
            ave[its, :, :] = ts.data[:, :]

    if len(ts.data.shape) == 1:
        return Timeseries(ma.mean(ave, axis=0), dt_common)
    else:
        avef = ma.zeros((tslip[0].data.shape[0], tslip[0].data.shape[1]))
        for nrow, row in enumerate(avef):
            avef[nrow, :] = ma.mean(ave[:, nrow, :], axis=0)
        return Timeseries(avef, dt_common)


def prettyNumber(f):
    fScaled = f
    if fScaled < 1:
        correct = 10.0
    else:
        correct = 1.0

    # set stepsize
    try:
        nZeros = int(np.log10(fScaled))
    except OverflowError:
        nZeros = 0

    prev10e = 10.0**nZeros / correct
    next10e = prev10e * 10

    if fScaled / prev10e > 7.5:
        return next10e
    elif fScaled / prev10e > 5.0:
        return 5 * prev10e
    else:
        return round(fScaled/prev10e) * prev10e


def plot_scalebars(ax, div=3.0, labels=True,
                   xunits="", yunits="", nox=False,
                   sb_xoff=0, sb_yoff=0,
                   sb_ylabel_xoff_comp=False, sb_ylabel_yoff=0,
                   rotate_yslabel=False,
                   linestyle="-k", linewidth=4.0,
                   textcolor='k', textweight='normal',
                   xmin=None, xmax=None, ymin=None, ymax=None):

    # print dir(ax.dataLim)
    if xmin is None:
        xmin = ax.dataLim.xmin
    if xmax is None:
        xmax = ax.dataLim.xmax
    if ymin is None:
        ymin = ax.dataLim.ymin
    if ymax is None:
        ymax = ax.dataLim.ymax
    xscale = xmax-xmin
    yscale = ymax-ymin

    xoff = (scale_dist_x + sb_xoff) * xscale
    if sb_ylabel_xoff_comp:
        xoff_ylabel = scale_dist_x * xscale
    else:
        xoff_ylabel = xoff

    yoff = (scale_dist_y - sb_yoff) * yscale

    # plot scale bars:
    xlength = prettyNumber((xmax-xmin)/div)
    xend_x, xend_y = xmax, ymin
    if not nox:
        xstart_x, xstart_y = xmax-xlength, ymin
        scalebarsx = [xstart_x+xoff, xend_x+xoff]
        scalebarsy = [xstart_y-yoff, xend_y-yoff]
    else:
        scalebarsx = [xend_x+xoff, ]
        scalebarsy = [xend_y-yoff]

    ylength = prettyNumber((ymax-ymin)/div)
    yend_x, yend_y = xmax, ymin+ylength
    scalebarsx.append(yend_x+xoff)
    scalebarsy.append(yend_y-yoff)

    ax.plot(scalebarsx, scalebarsy, linestyle, linewidth=linewidth,
            solid_joinstyle='miter')

    if labels:
        # if textcolor is not None:
        #     color = "\color{%s}" % textcolor
        # else:
        #     color = ""
        if not nox:
            # xlabel
            if xlength >= 1:
                xlabel = r"%d$\,$%s" % (xlength, xunits)
            else:
                xlabel = r"%g$\,$%s" % (xlength, xunits)
            xlabel_x, xlabel_y = xmax-xlength/2.0, ymin
            xlabel_y -= key_dist*yscale
            ax.text(
                xlabel_x+xoff, xlabel_y-yoff, xlabel, ha='center', va='top',
                weight=textweight, color=textcolor)
        # ylabel
        if ylength >= 1:
            ylabel = r"%d$\,$%s" % (ylength, yunits)
        else:
            ylabel = r"%g$\,$%s" % (ylength, yunits)
        if not rotate_yslabel:
            ylabel_x, ylabel_y = \
                xmax, ymin + ylength/2.0 + sb_ylabel_yoff*yscale
            ylabel_x += key_dist*xscale
            ax.text(ylabel_x+xoff_ylabel, ylabel_y-yoff, ylabel,
                    ha='left', va='center',
                    weight=textweight, color=textcolor)
        else:
            ylabel_x, ylabel_y = xmax, ymin + ylength/2.0 + sb_ylabel_yoff
            ylabel_x += key_dist*xscale
            ax.text(ylabel_x+xoff_ylabel, ylabel_y-yoff, ylabel,
                    ha='left', va='center', rotation=90,
                    weight=textweight, color=textcolor)


def xFormat(x, res, data_len, width):
    points = int(width/2.5 * res)
    part = float(x) / data_len
    return int(part*points)


def yFormat(y):
    return y


def reduce(ydata, dy, maxres, xoffset=0, width=graph_width):
    x_last = xFormat(0, maxres, len(ydata), width)
    y_last = yFormat(ydata[0])
    y_max = y_last
    y_min = y_last
    x_next = 0
    y_next = 0
    xrange = list()
    yrange = list()
    xrange.append(x_last)
    yrange.append(y_last)
    for (n, pt) in enumerate(ydata[:-1]):
        x_next = xFormat(n+1, maxres, len(ydata), width)
        y_next = yFormat(ydata[n+1])
        # if we are still at the same pixel column,
        # only draw if this is an extremum:
        if (x_next == x_last):
            if (y_next < y_min):
                y_min = y_next
            if (y_next > y_max):
                y_max = y_next
        else:
            # else, always draw and reset extrema:
            if (y_min != y_next):
                xrange.append(x_last)
                yrange.append(y_min)
                y_last = y_min
            if (y_max != y_next):
                xrange.append(x_last)
                yrange.append(y_max)
                y_last = y_max
            xrange.append(x_next)
            yrange.append(y_next)
            y_min = y_next
            y_max = y_next
            x_last = x_next
            y_last = y_next
    trace_len_pts = width/2.5 * maxres
    trace_len_time = len(ydata) * dy
    dt_per_pt = trace_len_time / trace_len_pts
    xrange = np.array(xrange)*dt_per_pt + xoffset

    return xrange, yrange


def plot_traces(traces, traces2=None, ax=None, Fig=None, pulses=None,
                xmin=None, xmax=None, ymin=None, ymax=None,
                y2min=None, y2max=None, xoffset=0,
                maxres=None,
                plot_sb=True, sb_yoff=0, sb_xoff=0, linestyle_sb="-k",
                dashedline=None, sagline=None, rotate_yslabel=False,
                textcolor='k', textweight='normal',
                textcolor2='r',
                figsize=None,
                pulseprop=0.05, border=0.2):

    if ax is None:
        if figsize is None:
            Fig = plt.figure(dpi=maxres)
        else:
            Fig = plt.figure(dpi=maxres, figsize=figsize)

        Fig.patch.set_alpha(0.0)

        if pulses is not None and len(pulses) > 0:
            prop = 1.0-pulseprop-border
        else:
            prop = 1.0-border
        ax = Fig.add_axes([0.0, (1.0-prop), 1.0-border, prop], alpha=0.0)

    # This is a hack to find the y-limits of the
    # traces. It should be supplied by stf.plot_ymin() & stf.plot_ymax(). But
    # this doesn't work for a 2nd channel, and if the active channel is 2 (IN
    # 2). It seems like stf.plot_ymin/max always gets data from channel 0 (IN
    # 0), irrespective of which is choosen as the active channel.
    if y2min is not None and y2max is not None:
        ymin, ymax = np.inf, -np.inf

    for trace in traces:
        if maxres is None:
            xrange = trace.timearray()+xoffset
            yrange = trace.data
        else:
            xrange, yrange = reduce(trace.data, trace.dt, maxres=maxres)
            xrange += xoffset
        if y2min is not None and y2max is not None:
            ymin, ymax = min(ymin, yrange.min()), max(ymax, yrange.max())
            # Hack to get y-limits of data. See comment above.
        ax.plot(
            xrange, yrange, trace.linestyle, lw=trace.linewidth,
            color=trace.color)

    y2min, y2max = np.inf, -np.inf
    # Hack to get y-limit of data, see comment above.
    if traces2 is not None:
        copy_ax = ax.twinx()
        for trace in traces2:
            if maxres is None:
                xrange = trace.timearray()+xoffset
                yrange = trace.data
            else:
                xrange, yrange = reduce(trace.data, trace.dt, maxres=maxres)
                xrange += xoffset
            y2min, y2max = min(y2min, yrange.min()), max(y2max, yrange.max())
            # Hack to get y-limit of data, see comment above.
            copy_ax.plot(xrange, yrange, trace.linestyle, lw=trace.linewidth,
                         color=trace.color)
    else:
        copy_ax = None

    if xmin is not None:
        phantomrect_x0 = xmin
    else:
        phantomrect_x0 = ax.dataLim.xmin

    if xmax is not None:
        phantomrect_x1 = xmax
    else:
        phantomrect_x1 = ax.dataLim.xmax

    if ymin is not None:
        phantomrect_y0 = ymin
    else:
        phantomrect_y0 = ax.dataLim.ymin

    if ymax is not None:
        phantomrect_y1 = ymax
    else:
        phantomrect_y1 = ax.dataLim.ymax

    ax.plot(
        [phantomrect_x0, phantomrect_x1], [phantomrect_y0, phantomrect_y1],
        alpha=0.0)

    if traces2 is not None:
        if y2min is not None:
            phantomrect_y20 = y2min
        else:
            phantomrect_y20 = copy_ax.dataLim.ymin

        if y2max is not None:
            phantomrect_y21 = y2max
        else:
            phantomrect_y21 = copy_ax.dataLim.ymax

        copy_ax.plot(
            [phantomrect_x0, phantomrect_x1],
            [phantomrect_y20, phantomrect_y21], alpha=0.0)

    xscale = ax.dataLim.xmax-ax.dataLim.xmin
    if dashedline is not None:
        ax.plot([ax.dataLim.xmin, ax.dataLim.xmax], [dashedline, dashedline],
                "--k", linewidth=traces[0].linewidth*2.0)
        gridline_x = ax.dataLim.xmax
        gridline_x += key_dist*xscale

    if sagline is not None:
        ax.plot([ax.dataLim.xmin, ax.dataLim.xmax],[sagline, sagline],
                "--k", linewidth=traces[0].linewidth*2.0)
        gridline_x = ax.dataLim.xmax
        gridline_x += key_dist*xscale

    if xmin is None:
        xmin = ax.dataLim.xmin
    if xmax is None:
        xmax = ax.dataLim.xmax
    if ymin is None:
        ymin = ax.dataLim.ymin
    if ymax is None:
        ymax = ax.dataLim.ymax

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if traces2 is not None:
        if y2min is None:
            y2min = copy_ax.dataLim.ymin
        if y2max is None:
            y2max = copy_ax.dataLim.ymax
        copy_ax.set_ylim(y2min, y2max)
        sb_xoff_total = -0.03+sb_xoff
        sb_yl_yoff = 0.025
    else:
        sb_xoff_total = sb_xoff
        sb_yl_yoff = 0

    if plot_sb:
        plot_scalebars(
            ax, linestyle=linestyle_sb, xunits=traces[0].xunits,
            yunits=traces[0].yunits, textweight=textweight,
            textcolor=textcolor, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            rotate_yslabel=rotate_yslabel, sb_xoff=sb_xoff_total,
            sb_ylabel_xoff_comp=(traces2 is not None),
            sb_ylabel_yoff=sb_yl_yoff)
        if traces2 is not None:
            plot_scalebars(
                copy_ax, linestyle=traces2[0].linestyle,
                xunits=traces2[0].xunits, yunits=traces2[0].yunits,
                textweight=textweight, textcolor=textcolor2, xmin=xmin,
                xmax=xmax, ymin=y2min, ymax=y2max,
                rotate_yslabel=rotate_yslabel, nox=True, sb_xoff=-0.01+sb_xoff,
                sb_ylabel_xoff_comp=True, sb_ylabel_yoff=-sb_yl_yoff)

    if pulses is not None and len(pulses) > 0:
        axp = Fig.add_axes(
            [0.0, 0.0, 1.0-border, pulseprop+border/4.0], sharex=ax)
        for pulse in pulses:
            xrange = pulse.timearray()
            yrange = pulse.data
            axp.plot(
                xrange, yrange, pulse.linestyle, linewidth=pulse.linewidth,
                color=pulse.color)
        plot_scalebars(
            axp, linestyle=linestyle_sb, nox=True, yunits=pulses[0].yunits,
            textweight=textweight, textcolor=textcolor)
        for o in axp.findobj():
            o.set_clip_on(False)
        axp.axis('off')

    for o in ax.findobj():
        o.set_clip_on(False)
    ax.axis('off')
    if traces2 is not None:
        copy_ax.axis('off')

    if ax is None:
        return Fig

    return ax  # NOTE copy_ax HKT mod


def standard_axis(fig, subplot, sharex=None, sharey=None,
                  hasx=False, hasy=True):

    sys.stderr.write("This method is deprecated. "
                     "Use stfio_plot.StandardAxis instead.\n")
    try:
        iter(subplot)
        if isinstance(subplot, matplotlib.gridspec.GridSpec):
            ax1 = Subplot(
                fig, subplot, frameon=False, sharex=sharex, sharey=sharey)
        else:
            ax1 = Subplot(
                fig, subplot[0], subplot[1], subplot[2], frameon=False,
                sharex=sharex, sharey=sharey)
    except:
        ax1 = Subplot(
            fig, subplot, frameon=False, sharex=sharex, sharey=sharey)

    fig.add_axes(ax1)
    ax1.axis["right"].set_visible(False)
    ax1.axis["top"].set_visible(False)
    if not hasx:
        ax1.axis["bottom"].set_visible(False)
    if not hasy:
        ax1.axis["left"].set_visible(False)

    return ax1
