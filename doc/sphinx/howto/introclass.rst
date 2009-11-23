****************************************
Object oriented programming with Stimfit
****************************************

:Author: Jose Guzman
:Date:  |today|


Object oriented programming (OOP) is a software philosophy where the problems are solved with objects rather than with simple functions. These objects  behave similarly to objects in the physical world. For example, you may want to travel from Freiburg (Germany) to London (UK). For that, you will need to use a transport (e.g car, airplane, etc...) which certain properties (e.g airplanes are much faster than cars). The key concept is that an object has distint attributes (associated variables) and methods (associated functions). Interestingly, object attributes are extremely dynamic. They may change as they are involved in different tasks. Because object *have an state* they provide much more versability to solve problems than an static function alone. Thus, in OOP the desing of the code is near to the question, and much more far away from the machine and the hardware details.

.. note::

    moving from the procedural/functional programming paradigm to the object oriented programming paradigm requires some mind re-wiring. In principle, everything what you can do in OOP can be done in functional programming. However, large programs would benefit from the OOP approach as their code is more reutilisable. Abstraction level is higher, because we will work with concepts rather than with complex software algorithms. This require a higher level of abstraction.



Here is a list of some key concepts in OOP.


* **Class** is the blueprint used to generate objects. This is the master plan to define an object. After a class definition, this can be used to generate one or more objects. A class describes how to create an object in general, but does not describe the particular object.

* **Object** is the particular concept described in the class. It is the practical application of the class. It combines state (i.e variables) with behaviour (i.e functions, algorithms).

* **Encapsulation** because objects are exposed to the user, attributes and functions may be easily modified without permission. In order to prevent accidental overwritting some atttributes and methods are hiddend to the user.

* **Inheritance** a common mistake is creating a class definition for every object. To avoid extreme redundancy, classes may inherit properties from other classes.

====================
Classes and  objects
====================

To start using objects in the embedded Python shell, we will first start with a basic example. We will need to collect the sampling interval, channel index and trace index of a series of traces in our recordings. For that, we will create a class, which defines an object that collects this information at once for us. To create that class we can use this code: 

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


We can save this class in a file called test.py and import into Python. You will see that nothing happens after import. This is because we simply loaded the class (i.e instructions of how to create the object), but not the object itself. We can try to create an object called **mySpike** with the instructions described in that class with:

>>> mySpike= test.Spike('root') # test.py contains the class Spike()

mySpike is now the object created with the instructions given in the class spike. This is commonly refered as *mySpike is an instance of the class Spike*.



=================
Object attributes
=================

Object attributes can be now accessed with the dot notation. To test the attributes of the object "mySpike" simply type:

>>> mySpike.dt
>>> 0.05000000074505806
>>> mySpike.trace
>>> 7 
>>> mySpike.channel
>>> 1 
>>> mySpike.owner
>>> 'root'

This tells us that the trace 8 in the channel 2 has a sampling rate of 0.05 msec. 

=============
Encapsulation
=============

As you can see, nothing would prevent us to assign a new value to any of thecurrent object attributes. For example, we could type:

>>> mySpike.dt = 3

and set erroneously the new sampling rate to 3. For that reason, it is a very good practice to hide some object attributes to the user (unless you definitely want them to be modified during execution). This is called **encapsulation**. To hide the object attributes, we have just to insert a single underscore before the attribute in the class. This simply means, "look, but do not touch!"

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
            """ get sampling interval """
            return self._dt
        
        def get_trace_index(self):
            """ get trace index"""
            return self._trace

        def get_channel_index(self):
            """ get channel index"""
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

=========================
Dynamic nature of objects
=========================

There is still one problem to solve. As soon as we move through the recording, the trace (and maybe the channel) may change. However, if we call the methods get_trace_index() and get_channel_index() of the object they will return the attributes in the old status. We need need a new method to update the object attributes everytime that we change the trace/channel. This is where the dynamic nature of the objects come handy.

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
            self.owner = owner 
            self.update()

        def update(self):
            """
            update dt, trace and channel possition 
            """

            self._trace = stf.get_trace_index()
            self._channel = stf.get_channel_index()
            self._dt = stf.get_sampling_interval()

        def get_sampling_interval(self):
            """ get sampling interval """
            return self._dt
        
        def get_trace_index(self):
            """ get trace index """
            return self._trace

        def get_channel_index(self):
            """ get channel index """
            return self._channel

After reloading this class, and creating the object (e.g mySpike) the object will call the update() function, which simply collects the channel, trace and sampling interval of the current trace. Note that if we change the trace, channel or even the window, the attributes of the object will be updated when we type:

>>> mySpike.update()


