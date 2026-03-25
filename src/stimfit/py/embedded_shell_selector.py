"""Runtime shell selector for embedded Stimfit shell backends."""

import importlib
import os
import sys
import traceback

import wx

import stimfit_shell_api


class _NamespaceShellAdapter(object):
    def __init__(self, namespace):
        self.user_ns = namespace


class MyPanel(wx.Panel):
    _SHELLS = [
        ("legacy", "Legacy", "embedded_stf"),
        ("modern", "Modern", "embedded_shell_modern"),
        ("ipython", "IPython", "embedded_shell_ipython"),
    ]

    def __init__(self, parent):
        super(MyPanel, self).__init__(parent, wx.ID_ANY, style=wx.BORDER_NONE)
        self._current_panel = None
        self._shell_map = dict((key, (label, module_name)) for key, label, module_name in self._SHELLS)

        root_sizer = wx.BoxSizer(wx.VERTICAL)
        self._selector_row = wx.BoxSizer(wx.HORIZONTAL)
        self._selector_row.AddStretchSpacer(1)
        self._choice = wx.Choice(self, wx.ID_ANY, choices=[label for _, label, _ in self._SHELLS])
        self._selector_row.Add(self._choice, 0, wx.ALIGN_CENTER_VERTICAL)
        root_sizer.Add(self._selector_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 6)

        self._content_sizer = wx.BoxSizer(wx.VERTICAL)
        root_sizer.Add(self._content_sizer, 1, wx.EXPAND | wx.BOTTOM | wx.LEFT | wx.RIGHT, 10)
        self.SetSizer(root_sizer)

        self._choice.Bind(wx.EVT_CHOICE, self._on_shell_changed)
        if os.environ.get("STF_PY_SHELL_HIDE_INTERNAL_SELECTOR", "0") == "1":
            self.GetSizer().Show(self._selector_row, False)

        stimfit_shell_api.register_shell_backend_setter(self.set_shell_backend)

        default_key = os.environ.get("STF_PY_SHELL_DEFAULT", "modern").lower()
        if default_key not in self._shell_map:
            default_key = "modern"
        self.set_shell_backend(default_key)

    def _on_shell_changed(self, event):
        selection = self._choice.GetSelection()
        if selection != wx.NOT_FOUND:
            self.set_shell_backend(self._SHELLS[selection][0])
        event.Skip()

    def _register_shell_namespace(self, panel):
        shell = getattr(panel, "_console", None)
        if shell is not None and hasattr(shell, "user_ns"):
            stimfit_shell_api.register_shell(shell)
            return

        if shell is not None and hasattr(shell, "locals"):
            stimfit_shell_api.register_shell(_NamespaceShellAdapter(shell.locals))
            return

        pycrust = getattr(panel, "pycrust", None)
        if pycrust is not None:
            interp = getattr(pycrust, "interp", None)
            namespace = getattr(interp, "locals", None)
            if namespace is not None:
                stimfit_shell_api.register_shell(_NamespaceShellAdapter(namespace))

    def set_shell_backend(self, shell_key):
        if shell_key not in self._shell_map:
            return

        if self._current_panel is not None:
            self._content_sizer.Detach(self._current_panel)
            self._current_panel.Destroy()
            self._current_panel = None

        label, module_name = self._shell_map[shell_key]
        self._choice.SetSelection([k for k, _, _ in self._SHELLS].index(shell_key))

        if shell_key in ("legacy", "modern"):
            sys.ps1 = ">>> "
            sys.ps2 = "... "

        try:
            module = importlib.import_module(module_name)
            panel = module.MyPanel(self)
            self._current_panel = panel
            self._content_sizer.Add(panel, 1, wx.EXPAND)
            self._register_shell_namespace(panel)
        except Exception:
            self._current_panel = wx.TextCtrl(
                self,
                wx.ID_ANY,
                "Failed to initialize %s shell:\n\n%s" % (label, traceback.format_exc()),
                style=wx.TE_MULTILINE | wx.TE_READONLY,
            )
            self._content_sizer.Add(self._current_panel, 1, wx.EXPAND)

        self.Layout()
