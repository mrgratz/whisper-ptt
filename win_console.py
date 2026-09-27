"""Suppress phantom console-window popups from child processes on Windows.

A headless Python process (launched via ``pythonw``, ``DETACHED_PROCESS``, or
by a GUI parent — i.e. it owns no console) that shells out to a console-subsystem
program (python.exe, ffmpeg, yt-dlp, git, node, ...) makes Windows allocate a
fresh, focus-stealing console window for each child. Call
``suppress_child_windows()`` once near the top of any such entry point.

Self-gating: it only patches when the current process has NO console of its own,
so it is a no-op — and preserves child console output — when run from a real
terminal. It respects an explicit console disposition the caller already chose
(``CREATE_NEW_CONSOLE`` / ``DETACHED_PROCESS`` / ``CREATE_NO_WINDOW``). Idempotent
and safe to import/call from anywhere; no-op on non-Windows.

This lives in shipped code on purpose (DL-457): the fix must reach buyers, so a
machine-local ``usercustomize.py`` was rejected — the vault is the product.
"""
from __future__ import annotations

import subprocess
import sys

_installed = False


def _process_has_console() -> bool:
    """True if this process owns a console window (so children inherit it and
    no popup occurs). Fail-safe to True so we never suppress when unsure."""
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:
        return True


def suppress_child_windows() -> bool:
    """Patch ``subprocess.Popen`` to add ``CREATE_NO_WINDOW`` to child processes
    when this process is headless on Windows.

    Returns True if the patch was installed, False if skipped (non-Windows,
    this process has a console, or already installed).
    """
    global _installed
    if _installed or sys.platform != "win32":
        return False
    if _process_has_console():
        return False

    CREATE_NO_WINDOW = 0x08000000
    # If the caller already picked a console disposition, leave it alone:
    # a visible new console (/remote), a detached spawn, or an explicit hide.
    disposition = (
        subprocess.CREATE_NEW_CONSOLE
        | subprocess.DETACHED_PROCESS
        | CREATE_NO_WINDOW
    )
    _orig_init = subprocess.Popen.__init__

    def _init(self, *args, **kwargs):
        flags = kwargs.get("creationflags", 0)
        if not (flags & disposition):
            kwargs["creationflags"] = flags | CREATE_NO_WINDOW
        _orig_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _init  # type: ignore[method-assign]
    _installed = True
    return True
