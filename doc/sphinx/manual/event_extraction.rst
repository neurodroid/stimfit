*************************************
Event extraction by template matching
*************************************

:Author: Christoph Schmidt-Hieber (christsc at gmx.de)
:Date: |today|

Introduction
=============================
`Stimfit <http://www.stimfit.org>`_ can analyze spontaneous events such as EPSCs or EPSPs using a template matching algorithm (described by Jonas et al. (1993) [#Jonas1993]_ and with some implementations from Clemens and Bekkers (1997) [#ClemensBekkers1997]_). A template is a waveform :math:`p(t)` of length :math:`n` that represents the time course of a typical or exemplary event. The template is slid over the recording trace :math:`r(t)`, and at each sampling point (:math:`s`) is multiplied by a scaling factor :math:`m`, and an offset :math:`c` is added or subtracted so that the sum of squared errors :math:`\chi^2(t_s)` between the trace and the template is minimized:

.. math::

    {\displaystyle \chi^2(t_s)= \sum_{k=0}^{n-1}\left[r(t_{s+k})-\left(m{\cdot}p(t_k)+c\right)\right]^2}

This equation amounts to the fairly simple operation of fitting a straight line that relates :math:`p(t)` and :math:`r(t)` at every sampling point. 

A detection criterion is applied to decide whether an event has occurred at a sampling point. Two options are available in `Stimfit <http://www.stimfit.org>`_: Jonas et al. (1993) [#Jonas1993]_ suggest to use the linear correlation coefficient between the optimally scaled template and the data, whereas Clements and Bekkers (1997) [#ClemensBekkers1997]_ compare the scaling factor with the noise standard deviation.

A practical guide to event detection
====================================
The following steps allow to extract events with `Stimfit <http://www.stimfit.org>`_:

1. Create a preliminary template by fitting a function to a single, large and isolated event.

2. Use this preliminary template to extract some more exemplary large and isolated events using a high detection threshold.

3. Create a final template by fitting a function to the average of the exemplary events.

4. Extract all events with the final template using a low detection criterion threshold.

5. Eliminate false-positive, add false-negative events.

This procedure will be explained in some more detail in the following sections.

1. Create a preliminary template
--------------------------------

A typical way of creating the template is to fit an exemplary or typical event to a model with similar kinetics. For example, EPSCs can typically be modeled with the sum or the product of two exponential functions [#f1]_. In practice, a robust estimate for a template can be obtained fitting an event to a biexponential function, selecting the detected events and repeating this operation iteratively. You can try this iterative approach with a file that you can download `here <http://stimfit.org/tutorial/minis.dat>`_.

    .. figure:: images/bait_template.png
        :align: center

        **Fig. 21:** Creation of a "bait" template.

First, we fit a function to a single large an isolated event to create a preliminary "bait" template. In this case, we will use an event that can be found roughly between t = 20990 ms and t = 21050 ms. Then, we fit the sum of two exponential functions with a delay to this EPSC. To obtain the same template as in the example, you can call the function ``preliminary`` from the ``minidemo`` module that comes  bundled with `Stimfit <http://www.stimfit.org>`_.

::

    >>> import minidemo
    >>> minidemo.preliminary()

This will take care of the appropriate cursor positions and the biexponential fit. If you prefer, you can use the fit settings dialog, as described in chapter 1 (Fig. 16).


2. Extract exemplary events
---------------------------

We now use the bait example to fish some more large and isolate events. Choose "Analysis"->"Event detection"->"Template matching..." from the menu. 


    .. figure:: images/eventdetectionsettings.png
        :align: center

        **Fig. 22:** Event detection settings.

In the dialog that will pop up (Fig. 22), you can set the threshold for the detection criterion. Since we want to extract some large and isolated events during this first pass, we set this to a high number, say 10, using the template scaling factor (Clemens and Bekkers, 1997). Click "OK" to start the event detection. When finished, press **F** to fit the whole trace to the window. The detected events will be marked by blue arrows in the upper part of the window, and blue circles will indicate the peak values of the detected events. With a threshold of 10, only two events are detected. Repeating the same procedure with a threshold of 4 will detect 33 events (Fig 23). 


    .. figure:: images/events.png
        :align: center

        **Fig. 23:**  Detected events.

To have a new window with the isolated events, switch to the event editing mode by pressing **E** or by activating the corresponding button in the toolbar (Fig. 24). Clicking on the trace with the right mouse button will allow a menu to appear. Select "Extract selected events" from this menu to put the exemplary events into a new window.


    .. figure:: images/eventbutton.png
        :align: center

        **Fig. 24:** Switching to event editing mode.



3. Create the final template
----------------------------

To create the final template, we need to get the average of all detected events and fit it to a biexponential function. Remember to set the baseline, peak and fit window cursors appropriately before performing the fit, and to update all calculations. Again, you can make use of a function from the ``minidemo`` module to set the cursors and perform the fit:

::

    >>> import minidemo # if you have not imported it already
    >>> minidemo.final()

The final template should look similar as shown in Fig. 25 and give time constants of 0.13 and 15.36 ms respectively.


    .. figure:: images/finaltemplate.png
        :align: center

        **Fig. 25:** Creating a final template.

4. Extract all events
---------------------

In the original file (minis.dat) we will extract all events with the final template. As before, select in Analysis -> Event detection -> Template matching, but now the final template is the second on the menu list (Fig. 26). For this final run, we will lower the detection threshold to a value of 3, as suggested by Clements and Bekkers (1997).


    .. figure:: images/selectfinaltemplate.png
        :align: center

        **Fig. 26:** Selecting the final template.

5. Edit detected events
-----------------------

The detected events have to be visually inspected to remove false-positives and add false-negatives. Remove false-positives with the unselected the checkbox next to the arrow indicating an event (Fig. 23). To add false-negatives, switch to the event-editing mode (Fig. 24) and right-click on the trace at the position where the event starts (Fig. 27). To screen the whole trace, it is convenient to use **Shift**  and left arrow at the same time. Once you are done with editing, choose "Extract selected events" from the context menu.


    .. figure:: images/falsenegative.png
        :align: center

        **Fig. 27:** Adding a false-negative event.

Analyze extracted events
------------------------

With the settings as suggested above, 83 events are extracted. You will find a table on the left of the traces: This will show you the time of onset of the events and the inter-event intervals. To apply some further analysis to the extracted events, you can set the baseline, peak and fit cursors and perform a batch analysis. Use ``minidemo.bath_cursors()`` to set cursors for the batch analysis:

    .. figure:: images/batchanalysis.png
        :align: center

        **Fig. 28:** Batch analysis settings.

From the dialog (Fig 28) choose the analysis functions that you want to apply to your data. Click "OK" once you are done. A new table will appear to the left of the traces. You can copy and paste values from the tables to spreadsheet programs for further analysis.


Adjusting event detection settings
----------------------------------

+-----------------------------------------------------------------------------+----------------------------------------------------------+
| **Problem**                                                                 | **Solution**                                             |
+-----------------------------------------------------------------------------+----------------------------------------------------------+
| 1. Too many false-positive events have been detected.                       | 1. Increase the detection threshold                      |
| 2. Too many events have been missed (false-negatives)                       | 2. Decrease the detection threshold                      |
| 3. One and the same event is detected multiple times at short time intervals| 3. Increase the number of sampling points between events | 
| 4. Closely spaced events are not detected separately                        | 4. Decrease the number of sampling points between events |
+-----------------------------------------------------------------------------+----------------------------------------------------------+


.. [#Jonas1993]  Jonas P, Major G, Sakman B. (1993) Quantal components of unitary EPSCs at the mossy fibre synapse on CA3 pyramidal cells of rat hippocampus. J Physiol. 472, 615-663.

.. [#ClemensBekkers1997] Clements JD, Bekkers JM. (1997) Detection of spontaneous synaptic events with an optimally scaled template. Biophys J 73:220–229.

.. [#f1] Note that the product of two exponentials :math:`{\displaystyle f(t)=a(1-e^{-\frac{t}{\tau_1}})e^{-\frac{t}{\tau_2}}}` can equivalently be expressed as the sum of two exponentials: :math:`{\displaystyle f(t)=a(e^{-\frac{t}{\tau_2}}-e^{-\frac{t}{\tau_3}}) }`, with :math:`{\displaystyle \tau_3=\frac{\tau_2 \tau_1}{\tau_2-\tau_1}}`.
