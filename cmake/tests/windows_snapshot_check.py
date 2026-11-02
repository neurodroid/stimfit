"""Run inside an installed Stimfit through its temporary stf_init.py hook.

Use check-windows-snapshot.ps1 to isolate the interpreter and collect the report.
"""

import importlib
import faulthandler
import functools
import json
import os
from pathlib import Path
import sys
import traceback

import wx
import wx.aui


def descendants(window):
    for child in window.GetChildren():
        yield child
        yield from descendants(child)


def menu_items(menu):
    for item in menu.GetMenuItems():
        yield item
        if item.GetSubMenu() is not None:
            yield from menu_items(item.GetSubMenu())


def write_edf(path, channels):
    """Write a calibrated recording without requiring an external test file."""
    import numpy as np

    def field(value, width):
        value = str(value).encode("ascii")
        assert len(value) <= width
        return value.ljust(width, b" ")

    sample_count = 512
    header_size = 256 * (channels + 1)
    header = b"".join(field(value, width) for value, width in [
        ("0", 8), ("Stimfit snapshot regression", 80), ("Known signals", 80),
        ("01.01.26", 8), ("00.00.00", 8), (header_size, 8), ("", 44),
        (1, 8), (1, 8), (channels, 4)])
    for values, width in [
            (["Signal %d" % channel for channel in range(channels)], 16),
            ([""] * channels, 80), (["uV"] * channels, 8),
            ([-32768] * channels, 8), ([32767] * channels, 8),
            ([-32768] * channels, 8), ([32767] * channels, 8),
            ([""] * channels, 80), ([sample_count] * channels, 8),
            ([""] * channels, 32)]:
        header += b"".join(field(value, width) for value in values)
    assert len(header) == header_size
    phase = np.arange(sample_count) * (2 * np.pi / sample_count)
    signals = [np.round(8000 * np.sin((channel + 1) * phase)).astype("<i2")
               for channel in range(channels)]
    path.write_bytes(header + b"".join(signal.tobytes() for signal in signals))
    return signals


def check_snapshot():
    check_dir = Path(os.environ["STF_SNAPSHOT_CHECK_DIR"])
    install_root = Path(os.environ["STF_SNAPSHOT_INSTALL_ROOT"]).resolve()
    report = {"passed": False, "install_root": str(install_root),
              "python": sys.version, "python_prefix": sys.prefix,
              "dialogs": [], "snapshots": []}
    main_window = next((window for window in wx.GetTopLevelWindows()
                        if window.GetTitle() == "Stimfit"), None)
    watchdog = None
    traceback_log = None

    def progress(step):
        report["step"] = step
        (check_dir / "snapshot-progress.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")

    try:
        traceback_log = (check_dir / "python-tracebacks.log").open("w", encoding="utf-8")
        faulthandler.dump_traceback_later(20, repeat=True, file=traceback_log)
        progress("importing bundled plotting modules")
        assert main_window is not None, "Stimfit main window was not created"
        report["window_handle"] = int(main_window.GetHandle())
        import numpy as np
        import stf
        import embedded_mpl
        from PIL import Image

        report["stimfit"] = stf.get_versionstring()
        expected_version = os.environ.get("STF_SNAPSHOT_EXPECTED_VERSION", "")
        if expected_version:
            assert expected_version in report["stimfit"], report["stimfit"]
        assert Path(sys.prefix).resolve().is_relative_to(install_root), sys.prefix
        # Import every plotting dependency from the installed bundle. This also
        # exercises extension modules and their companion DLLs with a clean PATH.
        report["modules"] = {}
        for name in ("numpy", "wx", "matplotlib", "matplotlib.ft2font", "PIL",
                     "PIL.Image", "contourpy", "cycler", "dateutil",
                     "fontTools", "six", "kiwisolver", "packaging", "pyparsing", "stf",
                     "embedded_mpl", "stfio_plot", "mpl_toolkits.axisartist"):
            module = importlib.import_module(name)
            module_path = Path(module.__file__).resolve()
            assert module_path.is_relative_to(install_root), (name, module_path)
            report["modules"][name] = str(module_path)
        report["matplotlib"] = importlib.import_module("matplotlib").__version__
        # Fail safely on old Python 3 releases before invoking their missing
        # C++ window factory, which may otherwise open an error dialog.
        assert hasattr(embedded_mpl, "FigCanvas"), "WXAgg canvas was not enabled for Python 3"
        assert hasattr(embedded_mpl, "NavigationToolbar"), "WXAgg toolbar is missing"

        # C++ redirects Python exceptions to the embedded shell before displaying
        # an error box. Save them to disk as well, including constructor errors.
        def record_context(label):
            context = {"where": label, "has_document": bool(stf.check_doc(False))}
            if context["has_document"]:
                context.update(filename=stf.get_filename(),
                               channel=stf.get_channel_index(),
                               trace=stf.get_trace_index(),
                               channels=stf.get_size_recording())
                if context["channels"] > 1:
                    context["reference_channel"] = stf.get_channel_index(False)
            report.setdefault("contexts", []).append(context)
            progress(label)

        def capture_errors(method, label=None):
            label = label or "MplPanel." + method.__name__
            @functools.wraps(method)
            def checked(*args, **kwargs):
                record_context("before " + label)
                try:
                    result = method(*args, **kwargs)
                except BaseException:
                    report["plot_error"] = traceback.format_exc()
                    record_context("failed in " + label)
                    raise
                record_context("after " + label)
                return result
            return checked

        for method_name in ("__init__", "plot_screen", "plot_spectrum"):
            setattr(embedded_mpl.MplPanel, method_name,
                    capture_errors(getattr(embedded_mpl.MplPanel, method_name)))
        embedded_mpl.stfio_plot.plot_traces = capture_errors(
            embedded_mpl.stfio_plot.plot_traces, "stfio_plot.plot_traces")

        manager = wx.aui.AuiManager.GetManager(main_window)
        assert manager is not None, "Stimfit window manager is unavailable"
        snapshot_tool = None
        fit_tool = None
        for window in descendants(main_window):
            if isinstance(window, wx.aui.AuiToolBar):
                for index in range(window.GetToolCount()):
                    tool = window.FindToolByIndex(index)
                    if tool.GetLabel() == "Snapshot":
                        snapshot_tool = (window, tool.GetId())
                    if "Fit traces to window" in tool.GetShortHelp():
                        fit_tool = (window, tool.GetId())
        assert snapshot_tool is not None, "Snapshot toolbar action is missing"
        assert fit_tool is not None, "Fit traces to window action is missing"
        report["toolbar_command"] = snapshot_tool[1]

        # wx.GetTopLevelWindows does not expose native Windows MessageBox
        # dialogs. Enumerate this process's visible dialogs so their error text
        # is captured and their standard OK button can dismiss them as well.
        native_dialogs = set()
        native_dismiss = lambda: None
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            user32.EnumWindows.argtypes = (callback_type, wintypes.LPARAM)
            user32.EnumChildWindows.argtypes = (wintypes.HWND, callback_type, wintypes.LPARAM)
            user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
            user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
            user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
            user32.IsWindowVisible.argtypes = (wintypes.HWND,)
            user32.PostMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

            def window_text(handle):
                value = ctypes.create_unicode_buffer(2048)
                user32.GetWindowTextW(handle, value, len(value))
                return value.value

            @callback_type
            def inspect_dialog(handle, unused):
                process_id = wintypes.DWORD()
                user32.GetWindowThreadProcessId(handle, ctypes.byref(process_id))
                if process_id.value != os.getpid() or not user32.IsWindowVisible(handle):
                    return True
                class_name = ctypes.create_unicode_buffer(256)
                user32.GetClassNameW(handle, class_name, len(class_name))
                if class_name.value != "#32770":
                    return True
                if handle not in native_dialogs:
                    text = [window_text(handle)]

                    @callback_type
                    def inspect_child(child, unused):
                        text.append(window_text(child))
                        return True

                    user32.EnumChildWindows(handle, inspect_child, 0)
                    report["dialogs"].append(" ".join(part for part in text if part))
                    native_dialogs.add(handle)
                    progress("dismissing native error dialog")
                user32.PostMessageW(handle, 0x0111, 1, 0)  # WM_COMMAND, IDOK
                return True

            def native_dismiss():
                user32.EnumWindows(inspect_dialog, 0)

        # Any C++ error message enters a nested event loop. Close those dialogs
        # and record their text, so a broken build produces a failed report.
        def dismiss_errors(event):
            for window in wx.GetTopLevelWindows():
                if isinstance(window, wx.Dialog) and window.IsModal():
                    text = " ".join(child.GetLabel() for child in descendants(window)
                                    if isinstance(child, wx.StaticText))
                    report["dialogs"].append(text or window.GetTitle())
                    window.EndModal(wx.ID_CANCEL)
                    progress("dismissing wx error dialog")
            native_dismiss()

        watchdog = wx.Timer(main_window)
        main_window.Bind(wx.EVT_TIMER, dismiss_errors, watchdog)
        watchdog.Start(250)

        def render(panel, name):
            progress("drawing " + name)
            wx.YieldIfNeeded()
            panel.canvas.draw()
            png = check_dir / (name + ".png")
            progress("saving " + name)
            panel.fig.savefig(str(png), dpi=100)
            assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            with Image.open(png) as image:
                image.load()
                assert image.width >= 300 and image.height >= 200, image.size
                pixels = np.asarray(image.convert("RGB"))
                assert np.count_nonzero(np.min(pixels, axis=2) < 200) > 100, "PNG is blank"
                return {"png": str(png), "size": list(image.size), "bytes": png.stat().st_size}

        def close_panel(panel):
            progress("closing Matplotlib pane")
            assert manager.DetachPane(panel)
            panel.Destroy()
            manager.Update()
            wx.YieldIfNeeded()

        def new_panel(command_id, source):
            before = {int(pane.window.GetHandle()) for pane in manager.GetAllPanes()}
            command = wx.CommandEvent(wx.wxEVT_MENU, command_id)
            command.SetEventObject(source)
            progress("dispatching plot command %d" % command_id)
            assert main_window.GetEventHandler().ProcessEvent(command), "Plot command was not handled"
            assert not report["dialogs"], report["dialogs"]
            panels = [pane.window for pane in manager.GetAllPanes()
                      if int(pane.window.GetHandle()) not in before
                      and isinstance(pane.window, embedded_mpl.MplPanel)]
            assert len(panels) == 1, "Plot command did not create one Matplotlib panel"
            return panels[0]

        channel_order = tuple(int(value) for value in os.environ.get(
            "STF_SNAPSHOT_CHANNEL_ORDER", "1,2").split(","))
        assert channel_order and all(value in (1, 2) for value in channel_order), channel_order
        report["channel_order"] = list(channel_order)
        for channels in channel_order:
            fixture = check_dir / ("snapshot-%d-channel.edf" % channels)
            signals = write_edf(fixture, channels)
            progress("opening %d-channel EDF" % channels)
            assert stf.file_open(str(fixture)), "Cannot open snapshot fixture"
            wx.YieldIfNeeded()
            progress("validating %d-channel EDF" % channels)
            assert stf.get_size_recording() == channels
            for channel, samples in enumerate(signals):
                np.testing.assert_allclose(stf.get_trace(0, channel),
                                           samples / 1000.0, atol=1e-9)
            progress("fitting %d-channel recording to its view" % channels)
            scale_toolbar, fit_id = fit_tool
            for index in range(scale_toolbar.GetToolCount()):
                tool = scale_toolbar.FindToolByIndex(index)
                if tool.GetLabel() in ("Ch 1", "Ch 2"):
                    scale_toolbar.ToggleTool(tool.GetId(), tool.GetLabel() == "Ch 1" or channels > 1)
            fit_command = wx.CommandEvent(wx.wxEVT_TOOL, fit_id)
            fit_command.SetEventObject(scale_toolbar)
            assert main_window.GetEventHandler().ProcessEvent(fit_command), "Fit command was not handled"
            wx.YieldIfNeeded()
            bounds = {"x": [stf.plot_xmin(), stf.plot_xmax()],
                      "y": [stf.plot_ymin(), stf.plot_ymax()]}
            assert np.isfinite(list(bounds.values())).all(), bounds
            duration = len(signals[0]) * stf.get_sampling_interval()
            assert 0 < bounds["x"][1] - bounds["x"][0] <= 2 * duration, bounds
            assert 0 < bounds["y"][1] - bounds["y"][0] <= 4 * np.ptp(signals[0] / 1000.0), bounds
            for channel in range(channels):
                record_context("before preflight get_fit(0,%d)" % channel)
                assert stf.get_fit(0, channel) is None, "Unfitted fixture unexpectedly contains a fit"
                record_context("after preflight get_fit(0,%d)" % channel)
            progress("creating %d-channel snapshot with fitted bounds %s" % (channels, bounds))
            toolbar, tool_id = snapshot_tool
            panel = new_panel(tool_id, toolbar)
            for channel in range(channels):
                expected = stf.get_trace(0, channel)
                matching = [(axes, line) for axes in panel.fig.axes for line in axes.lines
                            if np.asarray(line.get_ydata()).shape == expected.shape
                            and np.allclose(line.get_ydata(), expected)]
                assert matching, "Snapshot omitted channel %d" % channel
                axes, line = matching[0]
                # An intact trace with stale view limits can render as a dot.
                # Require the waveform to occupy a useful fraction of both axes.
                assert np.ptp(expected) / abs(np.diff(axes.get_ylim())[0]) >= 0.2, axes.get_ylim()
                assert np.ptp(line.get_xdata()) / abs(np.diff(axes.get_xlim())[0]) >= 0.5, axes.get_xlim()
            result = render(panel, "snapshot-%d-channel" % channels)
            result["channels"] = channels
            result["source_view_bounds"] = bounds
            report["snapshots"].append(result)
            # Keep the test independent of layout saved by earlier runs.
            close_panel(panel)

        # The public Python API uses a separate C++ factory from the toolbar.
        progress("calling stf.mpl_panel")
        panel = stf.mpl_panel([4.0, 3.0])
        assert not report["dialogs"], report["dialogs"]
        assert isinstance(panel, embedded_mpl.MplPanel), "stf.mpl_panel did not return its panel"
        values = stf.get_trace()
        panel.axes.plot(np.arange(len(values)) * stf.get_sampling_interval(), values)
        panel.axes.set_xlabel("Time (ms)")
        panel.axes.set_ylabel("Voltage (mV)")
        panel.axes.set_title(r"Stimfit $f=1$ Hz")
        report["mpl_panel"] = render(panel, "mpl-panel-api")
        close_panel(panel)

        # Native MDI keeps document commands on the child's menu bar while the
        # parent's menu can contain only the base File/Help commands.
        spectrum_commands = set()
        for window in (main_window, *descendants(main_window)):
            if isinstance(window, wx.Frame):
                menu_bar = window.GetMenuBar()
                if menu_bar is not None:
                    spectrum_commands.update(
                        item.GetId() for index in range(menu_bar.GetMenuCount())
                        for item in menu_items(menu_bar.GetMenu(index))
                        if "spectrum" in item.GetItemLabelText().lower())
        assert len(spectrum_commands) == 1, "Power spectrum menu action is missing"
        panel = new_panel(spectrum_commands.pop(), main_window)
        assert panel.axes.get_xlabel() == "Frequency (Hz)"
        assert panel.axes.get_ylabel() == "Power spectral density (dB/Hz)"
        assert len(panel.axes.lines) == 1, "Power spectrum did not plot its PSD"
        line = panel.axes.lines[0]
        assert len(line.get_xdata()) > 100
        assert np.all(np.diff(line.get_xdata()) > 0)
        assert np.count_nonzero(np.isfinite(line.get_ydata())) > 100
        report["spectrum"] = render(panel, "power-spectrum")
        close_panel(panel)
        progress("all plotting checks passed")
        report["passed"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
    finally:
        faulthandler.cancel_dump_traceback_later()
        if traceback_log is not None:
            traceback_log.close()
        if watchdog is not None:
            watchdog.Stop()
            main_window.Unbind(wx.EVT_TIMER, source=watchdog)
        temporary = check_dir / "snapshot-check.json.tmp"
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(check_dir / "snapshot-check.json")


wx.CallLater(1500, check_snapshot)
