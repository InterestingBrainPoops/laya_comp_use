"""Actuation: turns a Decision into mouse and keyboard events. Nothing else lives here."""
from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator

import pyautogui
import pyperclip

from laya_agent.config import Config
from laya_agent.models import (
    ACTION_SCROLL_DOWN,
    ACTION_SCROLL_UP,
    Decision,
    UIElement,
)

pyautogui.FAILSAFE = True  # slam the mouse into the top-left corner to abort
pyautogui.PAUSE = 0.05


class Executor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    # -- high level -----------------------------------------------------------
    def execute(self, decision: Decision) -> str:
        """Perform the decision. Returns a one-line history entry. Honors dry_run."""
        if decision.element is not None:
            note = f"clicked {decision.element.label()}"
            if not self.cfg.dry_run:
                self.click(decision.element)
            return note
        if decision.action == ACTION_SCROLL_DOWN:
            if not self.cfg.dry_run:
                self.scroll(-600)
            return "scrolled down"
        if decision.action == ACTION_SCROLL_UP:
            if not self.cfg.dry_run:
                self.scroll(600)
            return "scrolled up"
        return f"no-op ({decision.action})"

    # -- primitives -----------------------------------------------------------
    def click(self, el: UIElement, double: bool = False) -> None:
        x, y = el.center
        pyautogui.moveTo(x, y, duration=0.12)
        if double:
            pyautogui.doubleClick()
        else:
            pyautogui.click()

    def scroll(self, clicks: int) -> None:
        pyautogui.scroll(clicks)

    def type_text(self, text: str, press_enter: bool = False) -> None:
        """Paste via clipboard so Unicode and long strings arrive intact."""
        if self.cfg.dry_run:
            return
        old = _safe_paste()
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.1)
        if press_enter:
            pyautogui.press("enter")
        if old is not None:
            pyperclip.copy(old)

    def hotkey(self, *keys: str) -> None:
        if not self.cfg.dry_run:
            pyautogui.hotkey(*keys)


def _safe_paste() -> str | None:
    try:
        return pyperclip.paste()
    except Exception:
        return None


# -- own-window hiding ----------------------------------------------------------
@contextlib.contextmanager
def hidden_window(hwnd: int | None, settle_s: float = 0.25) -> Iterator[None]:
    """Minimize our own window while we look at and act on the screen."""
    if not hwnd:
        yield
        return
    import win32con
    import win32gui

    was_visible = win32gui.IsWindowVisible(hwnd) and not win32gui.IsIconic(hwnd)
    if was_visible:
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        time.sleep(settle_s)
    try:
        yield
    finally:
        if was_visible:
            # Restore without stealing focus so the target app stays in the foreground.
            win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)
