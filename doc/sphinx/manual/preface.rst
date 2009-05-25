*******
Preface
*******

:Author: Christoph Schmidt-Hieber (christsc at gmx.de)
:Date: |today|

``Stimfit`` was originally written by Peter Jonas, University of Freiburg,
in the early 1990s. It was primarily designed to analyze the kinetics of
evoked excitatory postsynaptic potentials (EPSCs; Jonas et al., 1993).
The name ``Stimfit`` was chosen because the program allowed to *fit*
exponential functions to the decay of EPSCs evoked by extracellular
*stim*-ulation. The program was written in Borland Pascal, running under
DOS and entirely controlled using keyboards shortcuts. The user
interface was similar to a digital oscilloscope, with vertical cursors
defining measurement windows for baseline calculation, peak detection
and curve fitting. This allowed to analyze data with surprising
efficiency once the keyboard shortcuts were mastered. However, the
Borland Pascal compiler imposed some significant restrictions which
became apparent with increasing data size and computing power: for
instance, arrays were not allowed to be longer than :math:`10^{4}`  elements, and
faster processors had to be artificially slowed down to avoid runtime
errors.

    .. figure:: images/stimfit_dos.png
        :align: center        
        :alt: The original Stimfit for DOS

        **Fig. 1:** The original Stimfit for DOS.

When I converted the original Pascal program to C/C++, I rewrote the
code almost entirely from scratch. Only the algorithms to calculate
latencies, rise times, half durations and slopes are direct translations
of the original Pascal code. By contrast, I tried to preserve the user
interface as far as possible. Therefore, the program only poorly adheres
to common conventions for graphical user interfaces: for instance,
clicking the right mouse button will usually set a cursor position
rather than popping up a context menu.

A number of people have contributed to the program: First, I would like
to thank Peter Jonas for the original ``Stimfit`` code. Josef
Bischofberger has added some functions to the DOS version which I have
adopted. Bill Anderson has made helpful suggestions concerning the user
interface and provided some very large files that have been recorded
with his free program `WinLTP <http://www.winltp.com>`_. A large amount of helpful comments and bug
reports were filed by Emmanuel Eggermann and Daniel Boischer. The
`Levenberg-Marquartdt-algorithm <http://www.ics.forth.gr/~lourakis/levmar>`_  used for curve fitting was implemented
by Manolis Lourakis.

