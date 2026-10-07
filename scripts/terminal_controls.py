"""Read pause-toggle keys and terminal mouse clicks without blocking."""

from __future__ import annotations

import os
import re
import select
import sys


_MOUSE_PRESS = re.compile(r"\x1b\[<([0-9]+);([0-9]+);([0-9]+)M")
_PARTIAL_MOUSE = re.compile(r"\x1b(?:\[<?[0-9;]*)?$")


class TerminalControls:
    """Enable terminal mouse reports and poll Space, Enter, and clicks."""

    def __init__(self) -> None:
        self._active = False
        self._buffer = ""
        self._windows = os.name == "nt"
        self._console_handle = None
        self._old_console_mode = None
        self._old_tty_mode = None

    def __enter__(self) -> "TerminalControls":
        if not sys.stdin.isatty():
            return self

        if self._windows:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.GetStdHandle.restype = ctypes.c_void_p
            kernel32.GetStdHandle.argtypes = [ctypes.c_ulong]
            kernel32.GetConsoleMode.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
            ]
            kernel32.SetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            handle = kernel32.GetStdHandle(ctypes.c_ulong(0xFFFFFFF6))
            old_mode = ctypes.c_uint()
            if kernel32.GetConsoleMode(handle, ctypes.byref(old_mode)):
                # VT input lets Windows Terminal and compatible consoles
                # deliver ANSI mouse reports through the standard input.
                new_mode = (
                    old_mode.value & ~(0x0002 | 0x0004 | 0x0040)
                ) | 0x0080 | 0x0200
                if kernel32.SetConsoleMode(handle, new_mode):
                    self._console_handle = handle
                    self._old_console_mode = old_mode.value
                    self._active = True
        else:
            import termios
            import tty

            fd = sys.stdin.fileno()
            self._old_tty_mode = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            self._active = True

        if self._active:
            sys.stdout.write("\033[?1000h\033[?1006h")
            sys.stdout.flush()
        return self

    def poll_toggles(self) -> int:
        """Return the number of pause-toggle inputs currently waiting."""
        if not self._active:
            return 0

        incoming = ""
        if self._windows:
            import msvcrt

            while msvcrt.kbhit():
                incoming += msvcrt.getwch()
        else:
            fd = sys.stdin.fileno()
            while select.select([fd], [], [], 0)[0]:
                incoming += os.read(fd, 256).decode("utf-8", errors="ignore")

        self._buffer += incoming
        mouse_presses = [
            match
            for match in _MOUSE_PRESS.finditer(self._buffer)
            if int(match.group(1)) & 3 != 3
        ]
        key_presses = sum(self._buffer.count(key) for key in (" ", "\r", "\n"))
        toggles = len(mouse_presses) + key_presses
        self._buffer = _MOUSE_PRESS.sub("", self._buffer)

        # Retain only a possible incomplete mouse sequence across polls.
        escape_index = self._buffer.rfind("\033")
        if escape_index >= 0:
            suffix = self._buffer[escape_index:]
            self._buffer = suffix if _PARTIAL_MOUSE.fullmatch(suffix) else ""
        else:
            self._buffer = ""
        return toggles

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._active:
            sys.stdout.write("\033[?1006l\033[?1000l")
            sys.stdout.flush()

        if self._old_console_mode is not None:
            import ctypes

            ctypes.windll.kernel32.SetConsoleMode(
                self._console_handle,
                self._old_console_mode,
            )
        elif self._old_tty_mode is not None:
            import termios

            termios.tcsetattr(
                sys.stdin.fileno(),
                termios.TCSADRAIN,
                self._old_tty_mode,
            )
