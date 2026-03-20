# ===========================================================================
# embedded_shell_ipython.py
# In-process IPython-backed embedded shell for Stimfit.
# ===========================================================================

import code
import contextlib
import io
import re
import rlcompleter
import traceback

import wx

from embedded_init import intro_msg
import stimfit_shell_api

try:
    import stf_init  # noqa: F401
except ImportError:
    LOADED = " "
except SyntaxError:
    LOADED = " Syntax error in custom initialization script stf_init.py"
else:
    LOADED = " Successfully loaded custom initializaton script stf_init.py"

try:
    from IPython.core.interactiveshell import InteractiveShell
    HAVE_IPYTHON = True
except ImportError:
    InteractiveShell = None
    HAVE_IPYTHON = False


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

        if key == wx.WXK_TAB:
            current = self.GetValue()
            line = current[self._input_start :]
            match = re.search(r"([A-Za-z_][A-Za-z0-9_\.]*)$", line)
            if match is None:
                return

            token = match.group(1)
            matches = self._console.complete(token)
            if not matches:
                return

            if len(matches) == 1:
                completion = matches[0]
                replace_start = self._input_start + match.start(1)
                replace_end = self._input_start + match.end(1)
                self.SetSelection(replace_start, replace_end)
                self.Replace(replace_start, replace_end, completion)
                self.SetInsertionPoint(self._input_start + len(line) - len(token) + len(completion))
                return

            self.AppendText("\n" + "  ".join(matches) + "\n")
            self.begin_input(self._console.prompt())
            self.AppendText(line)
            self._input_start = self.GetLastPosition() - len(line)
            return

        event.Skip()


class _FallbackConsole(code.InteractiveConsole):
    def __init__(self, text_ctrl):
        super(_FallbackConsole, self).__init__()
        self._text = text_ctrl
        stimfit_shell_api.bootstrap_namespace(self.locals)

    @property
    def user_ns(self):
        return self.locals

    def prompt(self):
        return ">>> "

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

    def complete(self, token):
        if token.startswith("stf."):
            try:
                import stf as _stf_module
            except Exception:
                _stf_module = None
            if _stf_module is not None:
                prefix = token[4:]
                return ["stf." + name for name in dir(_stf_module) if name.startswith(prefix)]

        completer = rlcompleter.Completer(self.locals)
        matches = []
        index = 0
        while True:
            candidate = completer.complete(token, index)
            if candidate is None:
                break
            matches.append(candidate)
            index += 1
        return sorted(set(matches))


class _IPythonConsole(object):
    def __init__(self, text_ctrl):
        self._text = text_ctrl
        self._buffer = []
        self._shell = InteractiveShell.instance()
        self.user_ns = self._shell.user_ns
        stimfit_shell_api.bootstrap_namespace(self.user_ns)

    def prompt(self):
        if self._buffer:
            return "   ...: "
        return "In [%d]: " % (self._shell.execution_count + 1)

    def run_command(self, line):
        self._buffer.append(line)
        source = "\n".join(self._buffer)

        try:
            compiled = code.compile_command(source, symbol="exec")
        except (OverflowError, SyntaxError, ValueError):
            self._text.append_text(traceback.format_exc())
            self._buffer = []
            self._text.begin_input(self.prompt())
            return

        if compiled is None:
            self._text.begin_input(self.prompt())
            return

        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = self._shell.run_cell(source, store_history=True)

        self._buffer = []
        self._text.append_text(out.getvalue())
        self._text.append_text(err.getvalue())
        if getattr(result, "error_in_exec", None) is not None and not err.getvalue():
            self._text.append_text("Execution failed.\n")
        self._text.begin_input(self.prompt())

    def complete(self, token):
        if token.startswith("stf."):
            try:
                import stf as _stf_module
            except Exception:
                _stf_module = None
            if _stf_module is not None:
                prefix = token[4:]
                return ["stf." + name for name in dir(_stf_module) if name.startswith(prefix)]

        completer = rlcompleter.Completer(self.user_ns)
        matches = []
        index = 0
        while True:
            candidate = completer.complete(token, index)
            if candidate is None:
                break
            matches.append(candidate)
            index += 1
        return sorted(set(matches))


class MyPanel(wx.Panel):
    """Embedded shell panel backed by IPython when available."""

    def __init__(self, parent):
        super(MyPanel, self).__init__(parent, wx.ID_ANY, style=wx.BORDER_NONE)

        sizer = wx.BoxSizer(wx.VERTICAL)
        self._text = _InteractiveTextCtrl(self, None)

        if HAVE_IPYTHON:
            self._console = _IPythonConsole(self._text)
            banner = stimfit_shell_api.startup_banner(intro_msg(), LOADED)
        else:
            self._console = _FallbackConsole(self._text)
            banner = stimfit_shell_api.startup_banner(
                intro_msg(),
                LOADED + " IPython is unavailable; using the minimal fallback shell.",
            )

        stimfit_shell_api.register_shell(self._console)
        self._text._console = self._console
        self._text.append_text(banner + "\n")
        self._text.begin_input(self._console.prompt())
        sizer.Add(self._text, 1, wx.EXPAND | wx.BOTTOM | wx.LEFT | wx.RIGHT, 10)
        self.SetSizer(sizer)
