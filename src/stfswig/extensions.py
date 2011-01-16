"""
User-defined Python extensions that can be called from the menu.
"""

import spells

class Extension(object):
    def __init__(self, menuEntryString, pyFunc, description="", 
                 requiresFile=True):
        self.menuEntryString = menuEntryString
        self.pyFunc = pyFunc
        self.description = description
        self.requiresFile = requiresFile

extensionList = [
    Extension("Count APs", spells.count_aps, 
              "Counts APs in selected files", True),
]
