*********************
Calculating latencies
*********************

:Author: Jose Guzman
:Date:  |today|

`Stimfit <http://www.stimfit.org>`_ was originally used to calculate synaptic latencies (Katz and Miledi, 1965 [#KatzMiledi1965]_) but now can be used to calculate synaptic latencies and latencies between events or action potentials in the same or between different channels (see :doc:`/manual/latency_measurements` in the :doc:`/manual/index`). `Stimfit <http://www.stimfit.org>`_ also provides a very useful collection of Python functions which allow us to easily adapt the latency calculation for our particular conditions. We will use these functions to calculate the latency between two signals in two different channels (e.g. one corresponding to the soma, and another to the dendrite). We will use the object oriented programming paradigm (OOP) to solve this problem and applied it in the embedded Python shell of `Stimfit <http://www.stimfit.org>`_ .  


===============
The Spike class
===============

We will create a class to calculate basic action potential (AP) kinetics in the current/active channel. AP peak and half-width will be calculated from a threshold (in mV/ms) defined by the user, as described in Stuart et al. (1997) [#Stuart1997]_. In principle, this can be easily adjusted in the `Stimfit <http://www.stimfit.org>`_ menu toolbar (see Edit->Cursor settings and select Peak tab). However, as the number of AP to analyze increase, the manipulation of the menu becomes unnecessarily repetitive and prone to errors. We  will use an object to access different AP parameters (i.e baseline, peak, half-width and maximum rise-time) all them calculated from the threshold value Note these values are accessible in `Stimfit <http://www.stimfit.org>`_ result table (see Fig. 9 in the :doc:`/manual/index`), but we will access them within Python. Once the threshold is set, it can be accessed in terms of time with :func:`stf.get_threshold_time()`) or voltage with :func:`stf.get_threshold_value()`. 

Additionally, some other methods will be necessary to calculate the AP latencies. For example, we may want to calculate **onset latency** (i.e time difference between the beginning of the action potential in two different recordings) or **peak latency** (i.e difference in time between the peak of two APs in different recordings). More interestingly, we can calculate the **half-width latency** according to Schmidt-Hieber et al., (2008) [#Schmidt-Hieber2008]_ which calculates the AP latency by the time different between two APs by subtracting the time at the half-maximal amplitudes. 

::

    import stf
    import numpy as N
    from math import ceil, floor

    class Spike(object):
        """ 
        A collection of methods to calculate AP properties 
        from threshold (see Stuart et al, 1997). Note that all 
        calculations are performed in the active/current channel!!!
        """

        def __init__(self, threshold):
            """ 
            create a Spike instance with sampling rate and threshold. 
            measurements are performed in the current/active channel!!!

            Arguments:
            threshold    -- slope threshold to measure AP kinetics  
            """

            self._thr = threshold
            self.update()

        def update(self):
            """ 
            update current trace sampling rate,
            cursors position and measurements (peak, baseline & AP kinetics)
            according to the threshold value set at construction.
            """

            # set threshold
            stf.set_slope(self._thr)
            
            # update sampling rate
            self._dt = stf.get_sampling_interval()

            # update cursors and AP kinetics (peak and half-width)
            stf.measure()

        def get_base(self):
            """ 
            get baseline according to current cursor position in the 
            current/active trace            
            
            """

            return stf.get_trace(trace = -1,channel =-1)[stf.get_base_start():stf.get_base_end()+1].mean()

        def get_peak(self):
            """
            calculate peak measured from threshold in the current trace
            (see Stuart et al, 1997)
            """

            stf.set_peak_mean(1) # a single point for the peak value
            stf.set_peak_direction("up")

            self.update()

            peak = stf.get_peak() - stf.get_threshold_value()

            return peak

        def get_t50(self):
            """ calculates the half-width in ms in the current trace"""
            
            self.update()

            # current t50's difference gives the half-width
            return (stf.t50right_index-stf.t50left_index)*self._dt

        def get_max_rise(self):
            """ 
            maximum rate of rise (dV/dt) of AP in the current trace
            this depends on the available Na+ conductance, 
            see Mainen et al, 1995 or Schmidt-Hieber et al, 2008
            """

            self.update()
            pmaxrise = stf.maxrise_index() # in active channel

            trace = stf.get_trace()
        
            dV = trace[int(ceil(pmaxrise))]-trace[int(floor(pmaxrise))]

            return dV/self._dt

        def get_amplitude(self):
            """ returns the time at the peak in the current trace"""
            
            # stf.peak_index() does not update cursors!!!
            self.update()

            return stf.peak_index()*self._dt

        def get_threshold(self):
            """ returns the threshold (mV/ms) set at construction """
            return self._thr

        def get_threshold_value(self):
            """ returns the value (in y-units) at the threshold """

            self.update() #stf.get_threshold_value() does not update !!!
            return stf.get_threshold_value()

        def get_threshold_time(self):
            """ returns the value (in x-units) at the threshold """

            self.update() # stf.get_threshold_time() does not update!!!
            return stf.get_threshold_time('True')


==============
Code commented
==============

Note that all methods but **get_base()** and **get_threshold()** are preceded by **self.update()**. This is to update the sampling rate of the current trace (necessary to transform index points into time). This method will also update the cursors position (necessary to calculate the peak, half-widths and maximal slope of rise). Besides that, **both threshold_time()** and **threshold_value()** would change depending on the current trace. The method **get_base()** strongly depends on :func:`stf.get_base_start()` and :func:`stf.get_base_end()`. Fortunately, these functions return updated values when we change the trace. The method get_threshold() simply returns the threshold value set at construction, so we do not need any update. 



=====
Usage
=====

To use this class we have to create an object in the current trace with a threshold value as argument. Do not forget to set both baseline and peak cursors before creating the object.

>>> mySpike = AP.Spike(50)

Now we can calculate the parameters with the methods available to this object. Note that these values change as we change the trace (i.e, we do not need to type update() or use :func:`stf.measure()`). This means that the method mySpike.get_base() will return different values if we call it in different traces. Compare the values obtained with the functions with the corresponding values in the result table.

>>> mySpike.get_base() # correspond to baseline in the results table
>>> mySpike.get_peak() # correspond to Peak (from threshold) in the results table
>>> mySpike.get_t50() # correspond to t50 in the results table
>>> mySpike.get_max_rise() # correspond to slope (rise) in the results table
>>> mySpike.get_threshold_value() # correspond to Threshold in the results table

Additionally, we created the methods get_tamplitude(), get_threshold() and get_threshold_time() to help us to calculate the latencies with different methods. For example, if we have two different AP, one corresponding to the soma in channel 0, and the other corresponding to the dendrite in channel 1, we could calculate. We subtract the dendritic values from the somatic one (this means that positive values indicate that somatic AP would precede the dendritic AP).


* 1.- **Onset latency:** this is the latency between the beginning of 2 APs. We can calculate it as follows:

>>> APsoma = AP.Spike(50) # threshold of somatic AP is 50mV/ms
>>> t1 = APsoma.get_threshold_time()
>>> stf.set_channel(1)
>>> APdend = AP.Spike(20) # threshold for dendritic AP is 20mV/ms
>>> t2 = APdend.get_threshold_time()
>>> latency = t2-t1

* 2.- **Peak latency:** this is the latency between the peaks of 2 APs. Similarly to the previous calculate, we can use:

>>> stf.set_channel(0) # to the to channel 
>>> t1 = APsoma.get_tamplitude()
>>> stf.set_channel(1)
>>> t2 = APdend.get_tamplitude() # note this object has a different threshold
>>> latency = t2-t1

* 3.- **T50 latency:** this method is included in the Edit option of the `Stimfit <http://www.stimfit.org>`_ menu toolbar. However, this menu assumes that both thresholds are the same. We can calculate the t50 latency easily with the built-in python functions of `Stimfit <http://www.stimfit.org>`_ .For that, we type:

We calculate the time for the dendritic recording.

>>> stf.set_channel(1)
>>> stf.set_slope(20)
>>> t2 = stf.t50left_index()*stf.get_sampling_interval()

Now we change the channel and do the same for the somatic recording

>>> stf.set_channel(0)
>>> stf.set_slope(50)
>>> t1 = stf.t50left_index()*stf.get_sampling_interval()

and finally we calculate the latency.

>>> latency = t2-t1

In your current `Stimfit <http://www.stimfit.org>`_ version you can find a module called AP which contains the class Spike described bellow. Additionally, this module contains a function which creates a result table (see Figure bellow) with all the parameters described previously, and the latency calculate with the 3 methods described here. To use it simple type:

>>> import AP
>>> AP.calc(50,20) # 50 is the somatic threshold, 20 is dendritic

note that this function assumes that you set the cursors property in your trace, and that the dendritic and somatic AP are in different channels, being the somatic in the active channel.


    .. figure:: APmodule.png
        :align: center
        :alt: result table returned by AP.calc()

        Result table returned by the AP.calc() function. 

.. note::

    In the figure, the cell highlighted represent the latency calculated as the difference between the times at the half-width of the AP (as we did previously), and NOT the difference between the half-widths!!!

.. [#KatzMiledi1965] Katz B, Miledi R (1965). The measurement of synaptic delay, and the time course of acetylcholine release at the neuromuscular junction. Proc R Soc Lond B Biol Sci. 161, 483-495

.. [#Stuart1997] Stuart G, Schiller J, Sakmann B (1997). Action potential initiation and propagation in rat neocortical pyramidal neurons. J Physiol. 505, 617-632

.. [#Schmidt-Hieber2008] Schmidt-Hieber C, Jonas P, Bischofberger J (2008). Action potential initiation and propagation in hippocampal mossy fibre axons. J Physiol. 586, 1849-1857.

.. [#Mainen1995] Mainen ZF, Joerges J, Huguenard JR, Sejnowski TJ (1995). A model of spike initiation in neocortical pyramidal neurons. Neuron 15, 1427-1439.
