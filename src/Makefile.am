if WITH_BIOSIGLITE
SUBDIRS = libbiosiglite
else
SUBDIRS =
endif
# The order is important here because libstimfit.la will be used by stfswig.
if BUILD_MODULE
SUBDIRS += libstfio libstfnum pystfio
else
SUBDIRS += libstfio libstfnum stimfit stimfit/py
endif
