#===========================================================================
# embedded_shell_modern.py
# Modern embedded shell fallback for wxPython Phoenix + Python 3.x
#===========================================================================

import code
import contextlib
import io
import traceback

import re

import wx
try:
    from wx.py import shell as wx_py_shell
except ImportError:
    wx_py_shell = None

from embedded_init import intro_msg


try:
    import stf_init  # noqa: F401
except ImportError:
    LOADED = " "
except SyntaxError:
    LOADED = " Syntax error in custom initialization script stf_init.py"
else:
    LOADED = " Successfully loaded custom initializaton script stf_init.py"


class _InteractiveTextCtrl(wx.TextCtrl):
    def __init__(self, parent, console):
        super(_InteractiveTextCtrl, self).__init__(
            parent,
            wx.ID_ANY,
            style=wx.TE_MULTILINE | wx.TE_RICH2 | wx.TE_PROCESS_ENTER,
        )
        self._console = console
        self._input_start = 0
        self.Bind(wx.EVT_CHAR_HOOK, self._on_char)

    def append_text(self, text):
        if text:
            self.AppendText(text)

    def begin_input(self, prompt):
        self.append_text(prompt)
        self._input_start = self.GetLastPosition()
        self.SetInsertionPointEnd()

    def _on_char(self, event):
        key = event.GetKeyCode()
        pos = self.GetInsertionPoint()

        if key in (wx.WXK_BACK, wx.WXK_DELETE) and pos <= self._input_start:
            return

        if key == wx.WXK_LEFT and pos <= self._input_start:
            return

        if key == wx.WXK_RETURN:
            current = self.GetValue()
            line = current[self._input_start :]
            self.AppendText("\n")
            self._console.run_command(line)
            return

        event.Skip()


class _StfConsole(code.InteractiveConsole):
    def __init__(self, text_ctrl):
        super(_StfConsole, self).__init__()
        self._text = text_ctrl
        self.locals["__name__"] = "__console__"

    def write(self, data):
        self._text.append_text(data)

    def run_command(self, line):
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                more = self.push(line)
            except Exception:
                traceback.print_exc()
                more = False

        self._text.append_text(out.getvalue())
        self._text.append_text(err.getvalue())
        self._text.begin_input("... " if more else ">>> ")


class MyPanel(wx.Panel):
    """Modern embedded shell panel with wx.py.shell fallback support."""

    def __init__(self, parent):
        super(MyPanel, self).__init__(parent, wx.ID_ANY, style=wx.BORDER_NONE)

        sizer = wx.BoxSizer(wx.VERTICAL)

        if wx_py_shell is not None:
            # Prefer the wx shell when available: it supports calltips and tab completion.
            self.pycrust = wx_py_shell.Shell(self, -1, introText=intro_msg() + LOADED)
            # Suppress terminal control-sequence writes from prompt-toolkit integration
            # in non-terminal embedded wx shell widgets.
            try:
                if hasattr(self.pycrust, "interp") and hasattr(self.pycrust.interp, "write"):
                    _orig_write = self.pycrust.interp.write

                    def _filtered_write(text):
                        if not text:
                            return
                        # Strip C0 control chars (except newline/tab) and CSI escapes.
                        text = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)
                        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
                        if text:
                            _orig_write(text)

                    self.pycrust.interp.write = _filtered_write
            except Exception:
                pass
            self.pycrust.push("from embedded_init import *", silent=True)
            sizer.Add(self.pycrust, 1, wx.EXPAND | wx.BOTTOM | wx.LEFT | wx.RIGHT, 10)
        else:
            # Keep a minimal fallback so the shell still works without wx.py.
            self.pycrust = None
            self._text = _InteractiveTextCtrl(self, None)
            self._console = _StfConsole(self._text)
            self._text._console = self._console

            banner = intro_msg() + LOADED + "\n"
            self._text.append_text(banner)
            self._console.push("from embedded_init import *")
            self._text.begin_input(">>> ")
            sizer.Add(self._text, 1, wx.EXPAND | wx.BOTTOM | wx.LEFT | wx.RIGHT, 10)

        self.SetSizer(sizer)
