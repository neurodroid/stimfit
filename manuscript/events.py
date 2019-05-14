from __future__ import print_function

import sys
import os

import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from scipy.optimize import leastsq

import stfio
import stfio_plot

try:
    import spectral
except ImportError:
    pass

class Bardata(object):
    def __init__(self, mean, err=None, data=None, title="", color='k'):
        self.mean = mean
        self.err = err
        self.data = data
        self.title = title
        self.color = color

def bargraph(datasets, ax, ylabel=None, labelpos=0, ylim=None, paired=False):

    if paired:
        assert(len(datasets)==2)
        assert(datasets[0].data is not None and datasets[1].data is not None)
        assert(len(datasets[0].data)==len(datasets[0].data))

    ax.axis["right"].set_visible(False)
    ax.axis["top"].set_visible(False)
    ax.axis["bottom"].set_visible(False)

    bar_width = 0.6
    gap2 = 0.15         # gap between series
    pos = 0
    xys = []
    for data in datasets:
        pos += gap2
        ax.bar(pos, data.mean, width=bar_width, color=data.color, edgecolor='k')
        if data.data is not None:
            ax.plot([pos+bar_width/2.0 for dat in data.data], 
                    data.data, 'o', ms=15, mew=0, lw=1.0, alpha=0.5, mfc='grey', color='grey')#grey')
            if paired:
                xys.append([[pos+bar_width/2.0, dat] for dat in data.data])

        if data.err is not None:
            yerr_offset = data.err/2.0
            if data.mean < 0:
                sign=-1
            else:
                sign=1
            erb = ax.errorbar(pos+bar_width/2.0, data.mean+sign*yerr_offset, yerr=sign*data.err/2.0, fmt=None, ecolor='k', capsize=6)
            if data.err==0:
                for erbs in erb[1]:
                    erbs.set_visible(False)
            erb[1][0].set_visible(False) # make lower error cap invisible

        ax.text(pos+bar_width, labelpos, data.title, ha='right', va='top', rotation=20)

        pos += bar_width+gap2

    if paired:
        for nxy in range(len(datasets[0].data)):
            ax.plot([xys[0][nxy][0],xys[1][nxy][0]], [xys[0][nxy][1],xys[1][nxy][1]], '-k')#grey')

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)

def leastsq_helper(p,y,lsfunc,x):
    return y - lsfunc(p,x)

def fexpbde(p, x):
    offset, delay, tau1, amp, tau2 = p

    if delay < 0:
        return np.ones(x.shape) * 1e9
    e1 = np.exp((delay-x)/tau1);
    e2 = np.exp((delay-x)/tau2);
    y = amp*e1 - amp*e2 + offset;
    y[x<delay] = offset

    return y

def find_peaks(data, dt, threshold, min_interval=None):
    """
    Finds peaks in data that are above a threshold.

    Arguments:
    data --          1D NumPy array
    dt --            Sampling interval
    threshold --     Threshold for peak detection
    min_interval --  Minimal interval between peaks

    Returns:
    Peak indices within data.
    """
    peak_start_i = np.where(np.diff((data > threshold)*1.0)==1)[0]
    peak_end_i = np.where(np.diff(
        (data[peak_start_i[0]:] > threshold)*1.0)==-1)[0] + peak_start_i[0]
    peak_i = np.array([
        np.argmax(data[peak_start_i[ni]:peak_end_i[ni]])+peak_start_i[ni]
        for ni, psi in enumerate(peak_start_i[:len(peak_end_i)])])
    if min_interval is not None:
        while np.any(np.diff(peak_i)*dt <= min_interval):
            peak_i = peak_i[np.diff(peak_i)*dt > min_interval]

    return peak_i

def correct_peaks(data, peak_i, i_before, i_after):
    """
    Finds and corrects misplaced peak indices.

    Arguments:
    data --          1D NumPy array
    peak_i --        Peak indices
    i_before --      Sampling points to be considered before the peak
    i_before --      Sampling points to be considered after the peak

    Returns:
    Corrected peak indices within data.
    """
    new_peak_i = []
    for pi in peak_i:
        old_pi = pi
        real_pi = np.argmax(data[pi-i_before:
                                        pi+i_after])+pi-i_before
        while real_pi != old_pi:
            old_pi = real_pi
            real_pi = np.argmax(data[real_pi-i_before:
                                     real_pi+i_after])+real_pi-i_before

        new_peak_i.append(real_pi)
    # Remove duplicates
    new_peak_i = np.array(list(set(new_peak_i)))

    return new_peak_i

def generate_events(t, mean_f):
    dt = t[1]-t[0]
    assert(np.sum(np.diff(t)-dt) < 1e-15)

    prob_per_dt = mean_f * dt
    event_prob = np.random.uniform(0, 1, (len(t)))
    event_times_i = np.where(event_prob < prob_per_dt)[0]
    assert(np.all(np.diff(event_times_i)))

    return event_times_i * dt

def ball_and_stick(h):
    soma = h.Section()
    soma.L = 20.0
    soma.diam = 20.0
    soma.nseg = 5

    dend = h.Section()
    dend.L = 500.0
    dend.diam = 5.0
    dend.nseg = 31

    dend.connect(soma)

    for sec in [soma, dend]:
        sec.insert('pas')
        sec.Ra = 150.0
        for seg in sec:
            seg.pas.e = -80.0
            seg.pas.g = 1.0/25000.0

    soma.push()

    return soma, dend

def add_noise(data, dt, mean_amp, snr=5.0):
    sigma = mean_amp/snr
    noise = np.random.normal(0, sigma, data.shape[0])
    std_orig = noise.std()
    noise = spectral.lowpass(spectral.Timeseries(noise, dt), 1.0)
    std_new = noise.data.std()
    noise.data *= std_orig/std_new
    return data+noise.data

def run(dt_nrn, tstop, mean_f):

    module_dir = os.path.dirname(__file__)
    if os.path.exists("%s/dat/events.h5" % module_dir):
        rec = stfio.read("%s/dat/events.h5" % module_dir)
        spiketimes = np.load("%s/dat/spiketimes.npy" % module_dir)
        return rec, spiketimes

    if os.path.exists("%s/dat/events_nonoise.npy" % module_dir):
        mrec = np.load("%s/dat/events_nonoise.npy" % module_dir)
        spiketimes = np.load("%s/dat/spiketimes.npy" % module_dir)
    else:
        from neuron import h
        h.load_file('stdrun.hoc')

        soma, dend = ball_and_stick(h)
        h.tstop = tstop

        trange = np.arange(0, tstop+dt_nrn, dt_nrn)
        spiketimes = generate_events(trange, mean_f)
        np.save("%s/dat/spiketimes.npy" % module_dir, spiketimes)

        syn_AMPA, spiketimes_nrn, vecstim, netcon = [], [], [], []
        for spike in spiketimes:
            loc = 0.8 * np.random.normal(1.0, 0.03)
            if loc < 0:
                loc = 0
            if loc > 1:
                loc = 1
            syn_AMPA.append(h.Exp2Syn(dend(loc), sec=dend))
            spiketimes_nrn.append(h.Vector([spike]))
            vecstim.append(h.VecStim())
            netcon.append(h.NetCon(vecstim[-1], syn_AMPA[-1]))

            syn_AMPA[-1].tau1 = 0.2 * np.random.normal(1.0, 0.3)
            if syn_AMPA[-1].tau1 < 0.05:
                syn_AMPA[-1].tau1 = 0.05
            syn_AMPA[-1].tau2 = 2.5 * np.random.normal(1.0, 0.3)
            if syn_AMPA[-1].tau2 < syn_AMPA[-1].tau1*1.5:
                syn_AMPA[-1].tau2 = syn_AMPA[-1].tau1 * 1.5
            syn_AMPA[-1].e = 0

            vecstim[-1].play(spiketimes_nrn[-1])
            netcon[-1].weight[0] = np.random.normal(1.0e-3, 1.0e-4)
            if netcon[-1].weight[0] < 0:
                netcon[-1].weight[0] = 0
            netcon[-1].threshold = 0.0

        vclamp = h.SEClamp(soma(0.5), sec=soma)
        vclamp.dur1 = tstop
        vclamp.amp1 = -80.0
        vclamp.rs = 5.0
        mrec = h.Vector()
        mrec.record(vclamp._ref_i)

        h.dt = dt_nrn # ms
        h.steps_per_ms = 1.0/h.dt
        h.v_init = -80.0
        h.run()

        mrec = np.array(mrec)
        np.save("%s/dat/events_nonoise.npy" % module_dir, mrec)

    plt.plot(np.arange(len(mrec), dtype=np.float) * dt_nrn, mrec)

    peak_window_i = 20.0 / dt_nrn
    amps_i = np.array([int(np.argmin(mrec[onset_i:onset_i+peak_window_i])+onset_i)
                       for onset_i in spiketimes/dt_nrn], dtype=np.int)

    plt.plot(amps_i * dt_nrn, mrec[amps_i], 'o')

    mean_amp = np.abs(mrec[amps_i].mean())
    print(mean_amp)
    mrec = add_noise(mrec, dt_nrn, mean_amp)
    plt.plot(np.arange(len(mrec), dtype=np.float) * dt_nrn, mrec)

    seclist = [stfio.Section(mrec),]
    chlist = [stfio.Channel(seclist),]
    chlist[0].yunits = "pA"
    rec = stfio.Recording(chlist)
    rec.dt = dt_nrn
    rec.xunits = "ms"
    rec.write("%s/dat/events.h5" % module_dir)

    return rec, spiketimes

def template(pre_event=5.0, post_event=15.0, sd_factor=4.0, min_interval=5.0,
             tau1_guess=0.5, tau2_guess=3.0):

    module_dir = os.path.dirname(__file__)

    if os.path.exists("%s/dat/template.npy" % module_dir):
        return np.load("%s/dat/template.npy" % module_dir), \
            np.load("%s/dat/template_epscs.npy" % module_dir), \
            np.load("%s/dat/spiketimes.npy" % module_dir)

    rec, spiketimes = run(0.01, 60000.0, 0.005)
    
    i_before = int(pre_event/rec.dt)
    i_after = int(post_event/rec.dt)

    # Find large peaks:
    trace = -np.array(rec[0][0])
    print(trace.mean(), trace.min(), trace.max())
    rec_threshold = trace.mean() + trace.std()*sd_factor
    peak_i = find_peaks(trace, rec.dt, rec_threshold, min_interval)

    # Correct for wrongly placed peaks after min_interval check:
    peak_i = correct_peaks(trace, peak_i, i_before, i_after)

    print("    Aligning events... ", end="")
    sys.stdout.flush()
    # offset, delay, tau1, amp, tau2 = p

    epscs = []
    for pi in peak_i:
        # Fit a function to each event to estimate its timing
        epsc = trace[pi-i_before:pi+i_after]
        t_epsc = np.arange(len(epsc)) * rec.dt
        p0 = [0, pre_event, tau1_guess, np.max(epsc)*4.0, tau2_guess]
        try:
            plsq = leastsq(leastsq_helper, p0, 
                           args = (epsc, fexpbde, t_epsc))
        except RuntimeWarning:
            pass
        delay_i = int(plsq[0][1]/rec.dt+pi-i_before)
        new_epsc = trace[delay_i-i_before:delay_i+i_after]
        # Reject badly fitted events:
        if np.argmax(new_epsc)*rec.dt > 0.8*pre_event:
            epscs.append(new_epsc)

    epscs = np.array(epscs)
    print("done")

    print("    Computing mean epsc ... ", end="")
    sys.stdout.flush()
    mean_epsc = np.mean(epscs, axis=0)
    p0 = [0, pre_event, tau1_guess, np.max(mean_epsc)*4.0, tau2_guess]
    plsq = leastsq(leastsq_helper, p0, 
                   args = (mean_epsc, fexpbde, t_epsc))

    sys.stdout.write(" done\n")

    templ = fexpbde(plsq[0], t_epsc)[plsq[0][1]/rec.dt:]
    np.save("%s/dat/template.npy" % module_dir, templ)
    np.save("%s/dat/template_epscs.npy" % module_dir, epscs)

    print("done")

    return templ, epscs, spiketimes

def figure():
    sd_factor=5.0
    # to yield a low total number of false positive and negative events:
    deconv_th=4.0
    matching_th=2.5
    deconv_min_int=5.0
    matching_min_int=5.0

    module_dir = os.path.dirname(__file__)

    import stf
    if not stf.file_open("%s/dat/events.h5" % module_dir):
        sys.stderr.write("Couldn't open %s/dat/events.h5; aborting now.\n" % 
                         module_dir)
        return
    dt = stf.get_sampling_interval()
    trace = stf.get_trace() * 1e3
    plot_start_t = 55310.0
    plot_end_t = 55640.0
    plot_hi_start_t = 55489.0
    plot_hi_end_t = 55511.0
    plot_start_i = int(plot_start_t/dt)
    plot_end_i = int(plot_end_t/dt)
    plot_hi_start_i = int(plot_hi_start_t/dt)
    plot_hi_end_i = int(plot_hi_end_t/dt)
    plot_trace = trace[plot_start_i:plot_end_i]
    plot_hi_trace = trace[plot_hi_start_i:plot_hi_end_i]
    trange = np.arange(len(plot_trace)) * dt
    trange_hi = np.arange(len(plot_hi_trace)) * dt
    templ, templ_epscs, spiketimes = template(sd_factor=sd_factor)
    plot_templ = templ * 1e3
    templ_epscs *= 1e3
    rec_threshold = trace.mean() - trace.std()*sd_factor
    t_templ = np.arange(templ_epscs.shape[1]) * dt

    # subtract baseline and normalize template:
    templ -= templ[0]
    if np.abs(templ.min()) > np.abs(templ.max()):
        templ /= np.abs(templ.min())
    else:
        templ /= templ.max()
    deconv_amps, deconv_onsets, deconv_crit, \
        matching_amps, matching_onsets, matching_crit = \
        events(-templ, deconv_th=deconv_th, matching_th=matching_th, 
               deconv_min_int=deconv_min_int, matching_min_int=matching_min_int)

    theoretical_ieis = np.diff(spiketimes)
    theoretical_peaks_t = spiketimes # + np.argmax(templ)*dt
    theoretical_peaks_t_plot = theoretical_peaks_t[
        (theoretical_peaks_t > plot_start_i*dt) & 
        (theoretical_peaks_t < plot_end_i*dt)] - plot_start_i*dt + 1.0
    theoretical_peaks_t_plot_hi = theoretical_peaks_t[
        (theoretical_peaks_t > plot_hi_start_i*dt) & 
        (theoretical_peaks_t < plot_hi_end_i*dt)] - plot_hi_start_i*dt + 1.0

    deconv_peaks_t = deconv_onsets# + np.argmax(templ)*dt
    deconv_peaks_t_plot = deconv_peaks_t[
        (deconv_peaks_t > plot_start_i*dt) & 
        (deconv_peaks_t < plot_end_i*dt)] - plot_start_i*dt
    deconv_peaks_t_plot_hi = deconv_peaks_t[
        (deconv_peaks_t > plot_hi_start_i*dt) & 
        (deconv_peaks_t < plot_hi_end_i*dt)] - plot_hi_start_i*dt
    matching_peaks_t = matching_onsets# + np.argmax(templ)*dt
    matching_peaks_t_plot = matching_peaks_t[
        (matching_peaks_t > plot_start_i*dt) & 
        (matching_peaks_t < plot_end_i*dt)] - plot_start_i*dt
    matching_peaks_t_plot_hi = matching_peaks_t[
        (matching_peaks_t > plot_hi_start_i*dt) & 
        (matching_peaks_t < plot_hi_end_i*dt)] - plot_hi_start_i*dt

    deconv_correct = np.zeros((deconv_peaks_t.shape[0]))
    matching_correct = np.zeros((matching_peaks_t.shape[0]))
    for theor in theoretical_peaks_t:
        if (np.abs(deconv_peaks_t-theor)).min() < deconv_min_int:
            deconv_correct[(np.abs(deconv_peaks_t-theor)).argmin()] = True
        if (np.abs(matching_peaks_t-theor)).min() < matching_min_int:
            matching_correct[(np.abs(matching_peaks_t-theor)).argmin()] = True

    total_events = spiketimes.shape[0]
    deconv_TP = deconv_correct.sum()/deconv_correct.shape[0]
    deconv_FP = (deconv_correct.shape[0]-deconv_correct.sum())/deconv_correct.shape[0]
    deconv_FN = (total_events - deconv_correct.sum())/total_events
    sys.stdout.write("True positives deconv: %.2f\n" % (deconv_TP*100.0))
    sys.stdout.write("False positives deconv: %.2f\n" % (deconv_FP*100.0))
    sys.stdout.write("False negatives deconv: %.2f\n" % (deconv_FN*100.0))
    matching_TP = matching_correct.sum()/matching_correct.shape[0]
    matching_FP = (matching_correct.shape[0]-matching_correct.sum())/matching_correct.shape[0]
    matching_FN = (total_events - matching_correct.sum())/total_events
    sys.stdout.write("True positives matching: %.2f\n" % (matching_TP*100.0))
    sys.stdout.write("False positives matching: %.2f\n" % (matching_FP*100.0))
    sys.stdout.write("False negatives matching: %.2f\n" % (matching_FN*100.0))
        
    gs = gridspec.GridSpec(11, 13)
    fig = plt.figure(figsize=(16,12))

    ax = stfio_plot.StandardAxis(fig, gs[:5,:6], hasx=False, hasy=False)
    ax.plot(trange, plot_trace, '-k', lw=2)
    ax.plot(theoretical_peaks_t_plot, 
            theoretical_peaks_t_plot**0*np.max(plot_trace), 
            'v', ms=12, mew=2.0, mec='k', mfc='None')
    ax.axhline(rec_threshold, ls='--', color='r', lw=2.0)
    stfio_plot.plot_scalebars(ax, xunits="ms", yunits="pA")

    ax_templ = stfio_plot.StandardAxis(fig, gs[:5,7:], hasx=False, hasy=False, sharey=ax)
    for epsc in templ_epscs:
        ax_templ.plot(t_templ, -epsc, '-', color='0.5', alpha=0.5)
    ax_templ.plot(t_templ, -templ_epscs.mean(axis=0), '-k', lw=2)
    ax_templ.plot(t_templ[-plot_templ.shape[0]:], -plot_templ, '-r', lw=4, alpha=0.5)
    stfio_plot.plot_scalebars(ax_templ, xunits="ms", yunits="pA", sb_yoff=0.1)

    ax_matching = stfio_plot.StandardAxis(fig, gs[5:7,:6], hasx=False, hasy=False, 
                                          sharex=ax)
    ax_matching.plot(trange, matching_crit[plot_start_i:plot_end_i], '-g')
    stfio_plot.plot_scalebars(ax_matching, xunits="ms", yunits="SD", nox=True)
    ax_matching.axhline(matching_th, ls='--', color='r', lw=2.0)
    ax_matching.plot(theoretical_peaks_t_plot, 
                     theoretical_peaks_t_plot**0*1.25*np.max(
                         matching_crit[plot_start_i:plot_end_i]), 
                     'v', ms=12, mew=2.0, mec='k', mfc='None')
    ax_matching.plot(matching_peaks_t_plot, 
                     matching_peaks_t_plot**0*np.max(
                         matching_crit[plot_start_i:plot_end_i]), 
                     'v', ms=12, mew=2.0, mec='g', mfc='None')
    ax_matching.set_ylim(None, 1.37*np.max(
        matching_crit[plot_start_i:plot_end_i]))
    ax_matching.set_title(r"Template matching")

    ax_deconv = stfio_plot.StandardAxis(fig, gs[7:9,:6], hasx=False, hasy=False, 
                                          sharex=ax)
    ax_deconv.plot(trange, deconv_crit[plot_start_i:plot_end_i], '-b')
    stfio_plot.plot_scalebars(ax_deconv, xunits="ms", yunits="SD")
    ax_deconv.axhline(deconv_th, ls='--', color='r', lw=2.0)
    ax_deconv.plot(theoretical_peaks_t_plot, 
                     theoretical_peaks_t_plot**0*1.2*np.max(
                         deconv_crit[plot_start_i:plot_end_i]), 
                     'v', ms=12, mew=2.0, mec='k', mfc='None')
    ax_deconv.plot(deconv_peaks_t_plot, 
                     deconv_peaks_t_plot**0*np.max(
                         deconv_crit[plot_start_i:plot_end_i]), 
                     'v', ms=12, mew=2.0, mec='b', mfc='None')
    ax_deconv.set_ylim(None, 1.3*np.max(
        deconv_crit[plot_start_i:plot_end_i]))
    ax_deconv.set_title(r"Deconvolution")

    ax_hi = stfio_plot.StandardAxis(fig, gs[9:11,2:5], hasx=False, hasy=False)
    ax_hi.plot(trange_hi, plot_hi_trace, '-k', lw=2)
    ax_hi.plot(theoretical_peaks_t_plot_hi, 
               theoretical_peaks_t_plot_hi*0 + 30.0, 
               'v', ms=12, mew=2.0, mec='k', mfc='None')
    ax_hi.plot(matching_peaks_t_plot_hi, 
               matching_peaks_t_plot_hi*0 + 20.0,
               'v', ms=12, mew=2.0, mec='g', mfc='None')
    ax_hi.plot(deconv_peaks_t_plot_hi, 
               deconv_peaks_t_plot_hi*0 + 10.0, 
               'v', ms=12, mew=2.0, mec='b', mfc='None')
    stfio_plot.plot_scalebars(ax_hi, xunits="ms", yunits="pA")

    xA = plot_hi_start_t - plot_start_t
    yA = deconv_crit[plot_start_i:plot_end_i].min()
    con = ConnectionPatch(xyA=(xA, yA), xyB=(0, 1.0),
                          coordsA="data", coordsB="axes fraction", 
                          axesA=ax_deconv, axesB=ax_hi,
                          arrowstyle="-", linewidth=1, color="k")
    ax_deconv.add_artist(con)
    xA += (plot_hi_end_t - plot_hi_start_t) * 0.9
    con = ConnectionPatch(xyA=(xA, yA), xyB=(0.9, 1.0),
                          coordsA="data", coordsB="axes fraction", 
                          axesA=ax_deconv, axesB=ax_hi,
                          arrowstyle="-", linewidth=1, color="k")
    ax_deconv.add_artist(con)

    ax_bars_matching = stfio_plot.StandardAxis(fig, gs[5:10,7:9])
    matching_bars_FP = Bardata(matching_FP*1e2, title="False positives", color='g')
    matching_bars_FN = Bardata(matching_FN*1e2, title="False negatives", color='g')
    bargraph([matching_bars_FP, matching_bars_FN], ax_bars_matching, 
             ylabel=r'Rate ($\%$)')
    ax_bars_matching.set_title(r"Template matching")

    ax_bars_deconv = stfio_plot.StandardAxis(fig, gs[5:10,10:12], hasy=False, sharey=ax_bars_matching)
    deconv_bars_FP = Bardata(deconv_FP*1e2, title="False positives", color='b')
    deconv_bars_FN = Bardata(deconv_FN*1e2, title="False negatives", color='b')
    bargraph([deconv_bars_FP, deconv_bars_FN], ax_bars_deconv, 
             ylabel=r'Error rate $\%$')
    ax_bars_deconv.set_title(r"Deconvolution")
    
    fig.text(0.09, 0.9, "A", size='x-large', weight='bold', ha='left', va='top')
    fig.text(0.53, 0.9, "B", size='x-large', weight='bold', ha='left', va='top')
    fig.text(0.09, 0.58, "C", size='x-large', weight='bold', ha='left', va='top')
    fig.text(0.53, 0.58, "D", size='x-large', weight='bold', ha='left', va='top')

    plt.savefig("%s/../../manuscript/figures/Fig5/Fig5.svg" % module_dir)
    
    fig = plt.figure()
    ieis_ax = fig.add_subplot(111)
    ieis_ax.hist([np.diff(deconv_onsets), np.diff(matching_onsets), 
                  theoretical_ieis], 
                 bins=len(theoretical_ieis)/1.0, 
                 cumulative=True, normed=True, histtype='step')
    ieis_ax.set_xlabel("Interevent intervals (ms)")
    ieis_ax.set_ylabel("Cumulative probability")
    ieis_ax.set_xlim(0,800.0)
    ieis_ax.set_ylim(0,1.0)

def events(template, deconv_th=4.5, matching_th=3.0, deconv_min_int=5.0,
           matching_min_int=5.0):
    """
    Detects events using both deconvolution and template matching. Requires
    an arbitrary template waveform as input. Thresholds and minimal intervals
    between events can be adjusted for both algorithms. Plots cumulative 
    distribution functions.
    """

    module_dir = os.path.dirname(__file__)

    if os.path.exists("%s/dat/deconv_amps.npy" % module_dir):
        return np.load("%s/dat/deconv_amps.npy" % module_dir), \
            np.load("%s/dat/deconv_onsets.npy" % module_dir), \
            np.load("%s/dat/deconv_crit.npy" % module_dir), \
            np.load("%s/dat/matching_amps.npy" % module_dir), \
            np.load("%s/dat/matching_onsets.npy" % module_dir), \
            np.load("%s/dat/matching_crit.npy" % module_dir)

    # Compute criteria
    deconv_amps, deconv_onsets, deconv_crit = \
        detect(template, "deconvolution", deconv_th, 
               deconv_min_int)
    matching_amps, matching_onsets, matching_crit = \
        detect(template, "criterion", matching_th, 
               matching_min_int)

    fig = plt.figure()

    amps_ax = fig.add_subplot(121)
    amps_ax.hist([deconv_amps, matching_amps], bins=50, cumulative=True, 
                 normed=True, histtype='step')
    amps_ax.set_xlabel("Amplitudes (pA)")
    amps_ax.set_ylabel("Cumulative probability")

    ieis_ax = fig.add_subplot(122)
    ieis_ax.hist([np.diff(deconv_onsets), np.diff(matching_onsets)], bins=50, 
                 cumulative=True, normed=True, histtype='step')
    ieis_ax.set_xlabel("Interevent intervals (ms)")
    ieis_ax.set_ylabel("Cumulative probability")

    np.save("%s/dat/deconv_amps.npy" % module_dir, deconv_amps)
    np.save("%s/dat/deconv_onsets.npy" % module_dir, deconv_onsets)
    np.save("%s/dat/deconv_crit.npy" % module_dir, deconv_crit)
    np.save("%s/dat/matching_amps.npy" % module_dir, matching_amps)
    np.save("%s/dat/matching_onsets.npy" % module_dir, matching_onsets)
    np.save("%s/dat/matching_crit.npy" % module_dir, matching_crit)

    return deconv_amps, deconv_onsets, deconv_crit, \
        matching_amps, matching_onsets, matching_crit

def detect(template, mode, th, min_int):
    """
    Detect events using the given template and the algorithm specified in
    'mode' with a threshold 'th' and a minimal interval of 'min_int' between
    events. Returns amplitudes and interevent intervals.
    """
    import stf

    # Compute criterium
    crit = stf.detect_events(template, mode=mode, norm=False, lowpass=0.1, 
                             highpass=0.001)

    dt = stf.get_sampling_interval()

    # Find event onset times (corresponding to peaks in criteria)
    onsets_i = stf.peak_detection(crit, th, int(min_int/dt))

    trace = stf.get_trace()

    # Use event onset times to find event amplitudes (negative for epscs)
    peak_window_i = min_int / dt
    amps_i = np.array([int(np.argmin(trace[onset_i:onset_i+peak_window_i])+onset_i)
                       for onset_i in onsets_i], dtype=np.int)

    amps = trace[amps_i]
    onsets = onsets_i * dt

    return amps, onsets, crit

if __name__=="__main__":
    figure()
