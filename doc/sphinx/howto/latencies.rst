*********************
Calculating latencies
*********************

:Author: Jose Guzman
:Date:  |today|

`Stimfit <http://www.stimfit.org>`_ was originally used to calculate synaptic latencies (Katz and Miledi, 1965 [#KatzMiledi1965]_) but now can be used to calculate synaptic latencies and latencies between events or action potentials in the same or between different channels (see :doc:`/manual/latency_measurements`). `Stimfit <http://www.stimfit.org>`_ also provides a very usefull collection of Python functions which allow us to easily calculate the latency for our particular conditions. We will use these functions to calculate the latency between two signals in two different channels (e.g. one corresponding to the soma, and another to the dendrite). Thereby we will introduce the object oriented programming paradigm (OOP) and its use in the embedded Python shell of `Stimfit <http://www.stimfit.org>`_ .

.. note::

    moving from the procedural/functional programming paradigm to the object oriented programming paradigm requires some mind re-wiring. In principle, everything what you can do in OOP can be done in functional programming. However, large programs would benefit from the OOP approach as their code is more reutilisable.

**Object oriented programmging**

Object oriented programming (OOP) is a software philosopy where the problems are solved with objects. These objects  behave similarly to objects in the physical world. For example, you may want to   . The key point is that an object has certain attributes (associated variables) and methods (associated functions). Interestingly, object attributes are dynamic (i.e change as the object is involved in different task). This is sometimes refeered as "they have an state".

Here is a list of some key concepts in OOP.


* **Class** is the blueprint used to generate objects. This is the master plan to define an object. After a class definition, this can be used to generate one or more objects. A class describes how to create an object in general, but does not describe the particular object.

* **Object** is the particular concept described in the class. It is the practical application of the class. It combines state (i.e variables) with behaviour (i.e functions, algorithms).

* **Encapsulation** because objects are exposed to the user, attributes and functions may be easily modified without permission. In order to prevent accidental overwritting some atttributes and methods are hiddend to the user.

* **Inheritance** a common mistake is creating a class definition for every object. To avoid extreme redundancy, classes may inherit properties from other classes.

To show how to create a class and an object in Python, we will create an object which simply collects the sampling interval, channel index and trace index of the current trace. We need to create a class as follows: 

::

    import stf

    class Spike(object):
        """ 
        A class to generate a Spike object with contains attributes
        of the current trace. 
        """
        def __init__(self, owner):
            """ 
            create instance with dt, trace and channel as attributes.
            Arguments:
            type    -- string containing the name of the user 
            """

            self.dt = stf.get_sampling_interval()
            self.trace = stf.get_trace_index()
            self.channel = stf.get_channel_index()
            self.owner = owner 


We can save this class in a file called test.py and import into Python. You will see that nothing happens after import. This is because we simply loaded the class (i.e instructions of how to create the object), but not the object itself. We can try to create an object called **mySpike** with this class with:

>>> mySpike= test.Spike('root') # test.py contains the class Spike()

Object attributes can be now accessed with the dot notation. To test the attributes of the object "myRecord" simply do:

>>> mySpike.dt
>>> 0.05000000074505806
>>> mySpike.trace
>>> 7 
>>> mySpike.channel
>>> 1 
>>> mySpike.owner
>>> 'root'

This tells us that the trace 8 in the channel 2 has a sampling rate of 0.05 msec. As you can see, nothing would prevent us to assign a value to any of the object attributes:

>>> mySpike.dt = 3

For that reason, it is a very good practice to hide these variables unless you definitely want them to be modified by the user during execution (i.e encapsulate them). To hide the object attributes just insert a single underscore before the attribute. This simply means, "look, but do not touch!"

.. note::

   Python strongly relies on convention rather than on enforcement. For example, encapsulated attributes are not really private (i.e user can overwrite them if necessary), but the underscore notation is used to indicate internal use only. If you find a good reason to overwrite them, Python is not going to avoid it. However, it is a good programming practice to keep the Python conventions if you want to share your programms with other users.
    
Additionally, we could give the user the opportunity to retreive these values without the dot notation by simply creating some functions available to this object. For example, we can create 3 functions called get_sampling_interval(), get_trace_index(), and get_channel_index() inside the class. These will be the methods of the object.

::


    import stf

    class Spike(object):
        """ 
        A class to generate a Spike object with contains attributes
        of the current trace. 
        """
        def __init__(self, owner):
            """ 
            create instance with dt, trace and channel as attributes.
            Arguments:
            type    -- string containing the name of the user 
            """

            self._dt = stf.get_sampling_interval()
            self._trace = stf.get_trace_index()
            self._channel = stf.get_channel_index()
            self.owner = owner 

        def get_sampling_interval(self):
            return self._dt
        
        def get_trace_index(self):
            return self._trace

        def get_channel_index(self):
            return self._channel


Now we can import/reload test.py and create a new object.

>>> mySpike2 = test.Spike('user')

and test its attributes as follows:

>>> mySpike2.get_sampling_interval()
>>> 0.05000000074505806
>>> mySpike2.get_trace_index()
>>> 7 
>>> mySpike2.get_channel_index()
>>> 1 
>>> mySpike.owner
>>> 'user'

.. note::

    do not confuse methods/attributes that start and end with two underscores with those which only start with a single underscores. The firsts are spetial methods and customize the standard python behaviour (like __init__), whereas the lasts are encapsulated methods.

There is still one problem to solve. As soon as we move through the recording, the trace (and maybe the channel) may change. However, if we call the methods get_trace_index() and get_channel_index() of the object they will return the attributes in the old status. We need need a new to update the object attributes everytime that we change the trace/channel, and this is where the dynamic nature of the objects come to 


=======================
The latency function
=======================

Note that this function assumes that current is recorded in pA. It sets the stf cursors (peak and baseline) to calculate the current deviation in response to the voltage difference. Finally, the voltage **amplitude** should be entered in mV. 


::

    import numpy as N
    
    # stimfit python module:
    import stf
    
    def resistance( base_start, base_end, peak_start, peak_end, amplitude):
        """Calculates the resistance from a series of voltage clamp traces.
        
        Keyword arguments:
        base_start -- Starting index (zero-based) of the baseline cursors.
        base_end   -- End index (zero-based) of the baseline cursors.
        peak_start -- Starting index (zero-based) of the peak cursors.
        peak_end   -- End index (zero-based) of the peak cursors.
        amplitude  -- Amplitude of the voltage command.
        
        Returns:
        The resistance.
        """

        if not stf.check_doc():
            print "Couldn't find an open file; aborting now."
            return 0

        #A temporary array to calculate the average:
        set = N.empty( (stf.get_size_channel(), stf.get_size_trace()) )
        for n in range( 0,  stf.get_size_channel() ):
            # Add this trace to set:
            set[n] = stf.get_trace( n )


        # calculate average and create a new section from it:
        stf.new_window( N.average(set,0) )
        
        # set peak cursors:
        if not stf.set_peak_mean(-1): return 0 # -1 means all points within peak window.
        if not stf.set_peak_start(peak_start): return 0
        if not stf.set_peak_end(peak_end): return 0
    
        # set base cursors:
        if not stf.set_base_start(base_start): return 0
        if not stf.set_base_end(base_end): return 0
    
        # measure everything:
        stf.measure()
    
        # calculate r_seal and return:
        return amplitude / (stf.get_peak()-stf.get_base())

==============
Code commented
==============

>>> if not setf.set_base_start(base_start,True) : return 0


.. note::
    :func:`stf.set_base_start()`, :func:`stf.set_base_end()`, :func:`stf.set_peak_start()` and :func:`stf.set_peak_end()` do not upgrade the measurements. For that reason, we call :func:`stf.measure()` (this is analogous to hit **Enter** in the main window). Thereby the values of :func:`stf.get_peak()` and :func:`stf.get_base()` are updated. 
  
=====
Usage
=====

>>> spells.resistance(0,999,10700,1999,-5)

Note that **charlie.py** has a routine called **r_in(amplitude=-5)** that does exactly this.

In the same way, if you wanted to calculate the value of the seal resistance (assuming this is the smallest resistance in the circuit, so no current will flow through any other resistance), you could test it with a larger voltage pulse.

>>> spells.resistance(0,199,1050,1199,50)

.. [#KatzMiledi1965] Katz B, Miledi R. (1965) The measurement of synaptic delay, and the time course of acetylcholine release at the neuromuscular junction. Proc R Soc Lond B Biol Sci. 161:483-495.

