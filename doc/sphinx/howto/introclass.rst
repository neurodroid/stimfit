****************************************
Object oriented programming with Stimfit
****************************************

:Authors: Jose Guzman, Alois Schlögl and Christoph Schmidt-Hieber
:Updated: |today|

Object-oriented programming (OOP) is a software philosophy where the problems are solved with objects rather than with simple functions. These objects  behave similarly to objects in the physical world. For example, image you may want to travel from Freiburg (Germany) to London (UK). For that, you will need to use a transport (e.g car, airplane, etc...) which certain properties (e.g airplanes are much faster than cars, whereas cars are more flexible in terms of schedule). The key concept here is that an object has distinct attributes (associated variables) and methods (associated functions). Interestingly, object attributes are extremely dynamic. They may change as they are involved in different tasks. Because object *have an state* they provide much more versatility to solve problems than static function alone. Thus, in OOP the design of the code is near to the question, and much more far away from the machine and the hardware details.

.. note::

    moving from the procedural/functional programming paradigm to the object oriented programming paradigm requires some mind re-wiring. In principle, everything what you can do in OOP can be done in functional programming. However, large programs would benefit from the OOP approach as their code is more reusable. Abstraction level is higher, because we will work with concepts rather than with complex software algorithms. 


There are some key concepts in OOP.


* **Class** is the blueprint used to generate objects. It contains the instructions to generate an object. Although a class describes how to create an object, it may not describe the particular properties of an object.

* **Object** is the practical application of the class. It combines state (i.e variables) and behavior (i.e functions, algorithms).

* **Encapsulation** because objects are exposed to the user, attributes and functions may be easily modified without permission. In order to prevent accidental overwriting,  some attributes and methods may be hidden to the user, and this is called encapsulation.

* **Inheritance** a common mistake when creating classes is to define a class for every object that we want to use. To avoid extreme redundancy, classes may inherit properties from other classes, providing thereby a way of creating more complex objects without having to re-write all the known instructions of a class inside another class.

====================
Classes and  objects
====================

We will start with a basic example to start using objects in the embedded Python shell. We will use an object to collect the sampling interval and trace index in our recording. For that, we will create a class, which defines an object that collects this information at once for us. To create that class we can use this code: 

::

    import stf

    class Trace(object):
        """ 
        A class to generate a Trace object with contains attributes
        of the current trace. 
        """
        def __init__(self, owner):
            """ 
            create instance with dt and trace index and as attributes.
            Arguments:
            type    -- string containing the name of the user 
            """

            self.dt = stf.get_sampling_interval()
            self.trace = stf.get_trace_index()
            self.owner = owner 


We can save this class in a file called test.py and import into our Python session. After importing the file, nothing will happen. This is because we simply loaded the class (i.e instructions of how to create the object), but not the object itself. Now, we can create an object called **myTrace** with the instructions described in that class as follows:

>>> myTrace= test.Trace('root') # test.py contains the class Trace()

myTrace is now my particular object. It was created with the instructions given in the class Trace. This is commonly refereed as *myTrace is an instance of the class Trace*.

=================
Object attributes
=================

Object attributes can be accessed with the dot notation. To test the attributes of "myTrace" we simply type:

>>> myTrace.dt
>>> 0.05000000074505806
>>> myTrace.trace
>>> 7 
>>> myTrace.owner
>>> 'root'

This tells us that the trace 8 has a sampling rate of 0.05 msec. The owner was set at construction, and it is a user called root. 

=============
Encapsulation
=============

As you can see bellow, nothing would prevent us to assign a new value to any of the current object attributes. For example, if we now type:

>>> myTrace.dt = 3

This potentially very dangerous (imagine the consecuences of setting the sampling rate to 3 in further calculations). For that reason, it is a very good programming practice to hide some object attributes to the user. This is called **encapsulation**. To hide the attributes of "myTrace", we have just to insert a single underscore before the attribute in the class. These objects are **private** which simply means, "look, but do not touch!"

.. note::

    Python strongly relies on convention rather than on enforcement. For example, encapsulated attributes are not really private (i.e user can overwrite them if necessary), but the underscore notation is used to indicate internal use only. If you find a good reason to overwrite them, Python is not going to stop you. However, it is a good programming practice to keep the Python conventions if you want to share your programs with other users.
    
Additionally, we could give the user the opportunity to retrieve these values without the dot notation by simply creating some functions available to this object. These would be the object methods. For example, we can create 2 functions called get_sampling_interval() and get_trace_index() inside the class. These are the methods of the object.

::

    import stf

    class Trace(object):
        """ 
        A class to generate a Trace object which contains attributes
        of the current trace. 
        """
        def __init__(self, owner):
            """ 
            create instance with dt and trace as attributes.
            Arguments:
            type    -- string containing the name of the user 
            """

            # please, note that underscore attributes are private
            self._dt = stf.get_sampling_interval()
            self._trace = stf.get_trace_index()
            self.owner = owner 

        def get_sampling_interval(self):
            """ get sampling interval """
            return self._dt
        
        def get_trace_index(self):
            """ get trace index"""
            return self._trace

Now we can import/reload test.py and create a new object.

>>> myTrace2 = test.Trace('user')

and test its attributes as follows:

>>> myTrace2.get_sampling_interval()
>>> 0.05000000074505806
>>> myTrace2.get_trace_index()
>>> 7 
>>> myTrace.owner
>>> 'user'

.. note::

    do not confuse methods/attributes that start and end with two underscores with those which only start with a single underscores. The firsts are special methods and customize the standard python behavior (like __init__), whereas the lasts are encapsulated methods.

=========================
Dynamic nature of objects
=========================

As soon as we move through the recording, the trace index may change. However, if we call the methods get_trace_index() or get_sampling_interval() of the object they will return the object attributes in the old status. We need a new method to update the object attributes every time that we change the trace. This is where the dynamic nature of the objects come handy.

::

    import stf

    class Trace(object):
        """ 
        A class to generate a Trace object which contains attributes
        of the current trace. 
        """
        def __init__(self, owner):
            """ 
            create instance with dt and trace as attributes.
            Arguments:
            type    -- string containing the name of the user 
            """
            self.owner = owner 
            self.update()

        def update(self):
            """
            update dt and  trace according to the current position 
            """

            self._trace = stf.get_trace_index()
            self._dt = stf.get_sampling_interval()

        def get_sampling_interval(self):
            """ get sampling interval """
            return self._dt
        
        def get_trace_index(self):
            """ get trace index """
            return self._trace

After reloading this class, and creating "myTrace" we can use the update() method. This simply collects the current trace index and sampling interval. If we change the trace or even the window, we have to call update() again to retreive the current index and sampling interval.

>>> myTrace3 = test.myTrace('user')
>>> myTrace3.get_trace_index()
>>> 0
>>> stf.set_trace_index(3)
>>> myTrace3.get_trace_index() # this returns the old state!!!
>>> 0
>>> myTrace3.update() # update attributes
>>> myTrace3.get_trace_index() # this returns the updated state!!!
>>> 3

=================
Class inheritance
=================

Object-oriented languages like Python support class inheritance. This means that we can inherit attributes and methods from a pre-existing class. Thus, we do not need to rewrite again this code. We can simply inherit from another class (called mother class). 
To inherit code from another class, we have to add the name of the mother class in the class headline. For example:

>>> class Channel(Trace):

The class Channel will automatically inherit the code from the class Trace.  We say that the class Channel is a subclass of the superclass Trace. If we want to extend the functionality of our now class, we can add new methods and/or attributes, or even overwrite the existing inherited methods. We can create a new Channel class in the same file like this:

::

    class Channel(Trace):
        """ 
        A class derived from Trace class
        """
        def __init__(self,owner):
            Trace.__init__(self.owner) # let Trace to get owner
            self._channel = stf.get_channel_index() 

        def update(self):
            """ update dt, trace and channel index """
            Trace.update(self) # update dt and trace
            self._channel = stf.get_channel_index()

        def get_channel_index(self):
            """ get channel index """
            return self._channel

From this example we can see that the class Channel not only inherits, but extends its functionality to the current channel. We have not only functions to calculate the sampling rate (get_sampling_rate() and trace get_trace_index() ) but also a new function called get_channel_index(). A new attribute is also added (self._channel). The update() function that we used to update the sampling interval and the trace in the Trace class, is now extended to include the updated channel number. We can now test it:

>>> stf.set_trace(3), stf.set_channel(1)
>>> True, True # remember, True if successful
>>> myChannel = test.Channel('user') # create a instance of Channel
>>> myChannel.get_trace_index() # this methods is inherited from Trace
>>> 3
>>> myChannel.get_sampling_interval() # inherited from Trace
>>> 0.05000000074505806
>>> myChannel.get_channel_index() # this is only for Channel
>>> 1

We can change trace and channel to test the update function

>>> stf.set_trace(5), stf.set_channel(0)
>>> True, True
>>> myChannel.get_trace_index() 
>>> 5  # this value was updated!
>>> myChannel.get_sampling_interval() # inherited from Trace
>>> 0.05000000074505806
>>> myChannel.get_channel_index() # this is only for Channel
>>> 0 # this is the updated value!

Finally, we can check if an object belongs to certain class with the function isinstance(object,class). For example

>>> isinstance(myChannel, test.Channel)
>>> True
>>> isinstance(myChannel, test.Trace)
>>> True # This is because Channel inherits from Trace
>>> isinstance(myTrace, test.Channel)
>>> False

or we can use the __class__ method included in every instance to check the type of the object:

>>> myChannel.__class__
>>> <class 'test.Channel'>   

we can get this class definition as string with:

>>> myChannel.__class__.__name__
>>> 'Channel'

