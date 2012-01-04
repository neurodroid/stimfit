"""
User-defined Python extensions that can be called from the menu.
"""

import spells

# submenu
class Extension(object):
    """
    A Python extension that can be added as a submenu in 
    the Extensions menu of Stimfit.
    """
    def __init__(self, menuEntryString, pyFunc, description="", 
                 requiresFile=True, parentEntry=None):
        """
        Arguments:
        menuEntryString -- This will be shown as a menu entry.
        pyFunc -- The Python function that is to be called.
                  Takes no arguments and returns a boolean.
        description -- A more verbose description of the function.
        requiresFile -- Whether pyFunc requires a file to be opened.
        """
        self.menuEntryString = menuEntryString
        self.pyFunc = pyFunc
        self.description = description
        self.requiresFile = requiresFile
        self.parentEntry = parentEntry

# define an Extension: it will appear as a submenu in the Extensions Menu
myExt = Extension("Count APs", spells.count_aps, "Count events >0 mV in selected files", True)

extensionList = [myExt]
