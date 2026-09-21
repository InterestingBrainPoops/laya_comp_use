"""Windows UI Automation parser: accessibility trees of every visible window -> UIElement list.

Windows are walked in z-order, front first. The foreground window gets the full node budget;
background windows get `uia_background_budget` each, enough for their tabs, toolbars and
menus but not their page content. Every window is also offered as a `Window` element so the
agent can switch to it by name. Minimized windows appear only as `Window` elements.
"""
from __future__ import annotations

import io
import time
from collections import deque
from dataclasses import replace

import mss
import uiautomation as auto
from PIL import Image

from laya_agent.config import Config
from laya_agent.models import Rect, Snapshot, UIElement

# Control types that a user clicks. Text/Image/Pane count only when they expose Invoke.
CLICKABLE_TYPES = {
    "ButtonControl", "HyperlinkControl", "EditControl", "ComboBoxControl",
    "CheckBoxControl", "RadioButtonControl", "MenuItemControl", "TabItemControl",
    "ListItemControl", "TreeItemControl", "SplitButtonControl",
    "SliderControl", "SpinnerControl", "DataItemControl", "HeaderItemControl",
}
MAYBE_TYPES = {"TextControl", "ImageControl", "PaneControl", "GroupControl", "CustomControl"}
SELECTABLE_TYPES = {"TabItemControl", "ListItemControl", "RadioButtonControl", "TreeItemControl"}
# Ancestors worth naming in the element path.
PATH_TYPES = {
    "WindowControl", "PaneControl", "GroupControl", "TabControl", "MenuControl", "MenuBarControl",
    "ListControl", "TreeControl", "ToolBarControl", "DialogControl", "DocumentControl",
}
# Top-level windows that are never targets.
SKIP_WINDOW_CLASSES = {"Shell_TrayWnd", "Shell_SecondaryTrayWnd", "Progman", "WorkerW", "Windows.UI.Core.CoreWindow"}
SKIP_WINDOW_NAMES = {"Program Manager", "Windows Input Experience", "Microsoft Text Input Application"}


def _kind(control_type_name: str) -> str:
    return control_type_name.removesuffix("Control")


def _clean_name(raw: str) -> str:
    """Drop icon-font glyphs (private use area) and control chars; collapse whitespace."""
    out = "".join(ch for ch in raw if not (0xE000 <= ord(ch) <= 0xF8FF or ord(ch) < 32))
    return " ".join(out.split())


def _is_readable(name: str) -> bool:
    """True for human labels, false for class-name noise like DesktopWindowXamlSource."""
    return bool(name) and (" " in name or len(name) <= 14)


APP_NAMES = {
    "chrome": "Chrome", "msedge": "Edge", "firefox": "Firefox", "windowsterminal": "Terminal",
    "explorer": "File Explorer", "code": "VS Code", "spotify": "Spotify", "discord": "Discord",
    "notepad": "Notepad", "powershell": "PowerShell", "pwsh": "PowerShell", "python": "Python",
    "steamwebhelper": "Steam", "obsidian": "Obsidian", "zed": "Zed", "slack": "Slack", "teams": "Teams",
}
_app_cache: dict[int, str] = {}


def _app_name(hwnd: int, title: str) -> str:
    """Owning process's executable, prettified ('chrome.exe' -> 'Chrome'). Titles lie: a
    browser's title is the page title. Falls back to the title's last ' - ' segment."""
    if hwnd in _app_cache:
        return _app_cache[hwnd]
    name = ""
    try:
        import os

        import win32api
        import win32con
        import win32process

        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        h = win32api.OpenProcess(win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        try:
            exe = os.path.splitext(os.path.basename(win32process.GetModuleFileNameEx(h, 0)))[0]
        finally:
            win32api.CloseHandle(h)
        if exe.lower() != "applicationframehost":  # UWP host: the title is the app
            name = APP_NAMES.get(exe.lower(), exe)
    except Exception:
        pass
    if not name:
        name = title.rsplit(" - ", 1)[-1].strip()[:30] if " - " in title else title[:30]
    _app_cache[hwnd] = name
    return name


class UIAScreenParser:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._nodes = 0
        auto.SetGlobalSearchTimeout(2.0)

    # -- public ---------------------------------------------------------------
    def parse(self, exclude_hwnd: int | None = None, window_title: str | None = None) -> Snapshot:
        t0 = time.perf_counter()
        png, w, h = self._screenshot()
        t1 = time.perf_counter()
        timings: dict[str, float] = {"screenshot_ms": round((t1 - t0) * 1000, 1)}
        self._nodes = 0
        with auto.UIAutomationInitializerInThread(debug=False):
            windows = self._windows(exclude_hwnd, window_title)
            timings["uia_windows_ms"] = round((time.perf_counter() - t1) * 1000, 1)
            elements: list[UIElement] = []
            titles: list[str] = []
            deadline = time.monotonic() + self.cfg.uia_deadline_s
            for i, win in enumerate(windows):
                title = _clean_name(win.Name)
                hwnd = win.NativeWindowHandle
                minimized = self._is_minimized(win)
                titles.append(title)
                elements.append(self._window_element(win, title, hwnd, len(elements), i == 0, minimized))
                if minimized or time.monotonic() > deadline:
                    continue
                budget = self.cfg.uia_max_nodes if i == 0 else self.cfg.uia_background_budget
                tw = time.perf_counter()
                elements.extend(
                    self._walk(win, exclude_hwnd, title, len(elements), budget, deadline, hwnd, i == 0)
                )
                if i == 0:
                    timings["uia_front_ms"] = round((time.perf_counter() - tw) * 1000, 1)
        timings["uia_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        timings["uia_nodes"] = self._nodes
        timings["parse_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return Snapshot(
            png=png, width=w, height=h, window_title=titles[0] if titles else "",
            elements=elements, source="uia", windows=titles, timings=timings,
        )

    # -- windows ----------------------------------------------------------------
    def _windows(self, exclude_hwnd: int | None, window_title: str | None) -> list:
        """Visible top-level windows in z-order (front first). `window_title` narrows to one.

        Uses Win32 EnumWindows (microseconds) and only then wraps the survivors as UIA
        controls. Walking the UIA desktop root's children instead cost ~4s per parse
        (measured: uia_windows_ms 4049 vs a 30ms element walk)."""
        import win32con
        import win32gui

        needle = window_title.lower() if window_title else None
        candidates: list[tuple[int, str]] = []

        def visit(hwnd: int, _: object) -> bool:
            if hwnd == exclude_hwnd or not win32gui.IsWindowVisible(hwnd):
                return True
            if win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) & win32con.WS_EX_TOOLWINDOW:
                return True
            title = _clean_name(win32gui.GetWindowText(hwnd) or "")
            if not title or title in SKIP_WINDOW_NAMES or win32gui.GetClassName(hwnd) in SKIP_WINDOW_CLASSES:
                return True
            if win32gui.GetWindow(hwnd, win32con.GW_OWNER):  # owned popups are not app windows
                return True
            candidates.append((hwnd, title))
            return True

        win32gui.EnumWindows(visit, None)  # z-order, front first
        out = []
        for hwnd, title in candidates:
            if needle is not None:
                # Title or app name: a Terminal's title follows its active tab, an app's name does not.
                if needle not in title.lower() and needle not in _app_name(hwnd, title).lower():
                    continue
            if not win32gui.IsIconic(hwnd):
                l, t, r, b = win32gui.GetWindowRect(hwnd)
                if r - l <= 0 or b - t <= 0:
                    continue
            try:
                ctrl = auto.ControlFromHandle(hwnd)
            except Exception:
                continue
            if ctrl is None:
                continue
            out.append(ctrl)
            if needle is not None or len(out) >= self.cfg.max_windows:
                break
        return out

    @staticmethod
    def _is_minimized(win) -> bool:
        try:
            import win32gui

            return bool(win32gui.IsIconic(win.NativeWindowHandle))
        except Exception:
            return False

    @staticmethod
    def _window_element(win, title: str, hwnd: int, idx: int, foreground: bool, minimized: bool) -> UIElement:
        try:
            r = win.BoundingRectangle
            rect = Rect(r.left, r.top, r.right, min(r.bottom, r.top + 36))  # title bar strip
        except Exception:
            rect = Rect(0, 0, 0, 0)
        if minimized:
            rect = Rect(0, 0, 0, 0)
        return UIElement(
            id=idx + 1, kind="Window", name=title, rect=rect, hwnd=hwnd, window=_app_name(hwnd, title),
            foreground=foreground, minimized=minimized, selected=foreground,
        )

    # -- internals ------------------------------------------------------------
    def _screenshot(self) -> tuple[bytes, int, int]:
        with mss.mss() as sct:
            mon = sct.monitors[self.cfg.monitor_index]
            shot = sct.grab(mon)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return buf.getvalue(), img.width, img.height

    def _walk(self, root, exclude_hwnd, window_title, start_idx, budget, deadline, hwnd, foreground) -> list[UIElement]:
        """Breadth-first so shallow chrome (tabs, toolbars, menus) is found before deep page
        content. Each Document subtree (a web page) gets its own node budget so one huge
        page cannot starve the rest of the window."""
        elements: list[UIElement] = []
        seen: set[tuple[str, str, int, int]] = set()
        visited = 0
        doc_budget: dict[int, int] = {}
        short = _app_name(hwnd, window_title)
        # queue items: control, depth, path, document id, inside-a-selected-item flag
        queue: deque[tuple[object, int, list[str], int | None, bool]] = deque([(root, 0, [], None, False)])
        while queue:
            ctrl, depth, path, doc, in_selected = queue.popleft()
            visited += 1
            if visited > budget or time.monotonic() > deadline:
                break
            self._nodes += 1
            if doc is not None:
                if doc_budget[doc] <= 0:
                    continue
                doc_budget[doc] -= 1
            try:
                if exclude_hwnd and ctrl.NativeWindowHandle == exclude_hwnd:
                    continue
                ct = ctrl.ControlTypeName
                name = _clean_name(ctrl.Name or "")
                rect = ctrl.BoundingRectangle
                offscreen = ctrl.IsOffscreen
            except Exception:
                continue
            r = Rect(rect.left, rect.top, rect.right, rect.bottom)
            selected = self._is_selected(ctrl) if ct in SELECTABLE_TYPES else False
            if not offscreen and r.area > 0 and depth > 0:
                el = self._maybe_element(ctrl, ct, name, r, path, start_idx + len(elements), hwnd, short, foreground)
                if el is not None:
                    if in_selected and not el.selected:
                        el = replace(el, selected=True)  # the close button of the current tab
                    key = (el.kind, el.name, *el.center)
                    if key not in seen:
                        seen.add(key)
                        elements.append(el)
            if depth >= self.cfg.uia_max_depth:
                continue
            if ct == "DocumentControl" and doc is None:
                doc = id(ctrl)
                doc_budget[doc] = self.cfg.uia_document_budget
            # Chrome repeats the window title on nested panes; that wastes option tokens.
            is_title = bool(window_title) and (name.startswith(window_title[:20]) or window_title.startswith(name[:20]))
            names_path = ct in PATH_TYPES and _is_readable(name) and depth > 0 and not is_title
            child_path = path + [name[:30]] if names_path else path
            try:
                children = ctrl.GetChildren()
            except Exception:
                children = []
            for child in children:
                queue.append((child, depth + 1, child_path, doc, in_selected or (selected and ct == "TabItemControl")))
        return elements

    def _maybe_element(self, ctrl, ct, name, r, path, idx, hwnd, window, foreground) -> UIElement | None:
        aid = ""
        try:
            aid = ctrl.AutomationId or ""
        except Exception:
            pass
        if ct in CLICKABLE_TYPES:
            if not name and not aid.isalnum():
                return None
        elif ct in MAYBE_TYPES:
            if not name or not self._has_invoke(ctrl):
                return None
        else:
            return None
        if r.width > 1600 and r.height > 900:  # whole-window panes are not targets
            return None
        return UIElement(
            id=idx + 1,
            kind=_kind(ct),
            name=name or aid,
            rect=r,
            path=" > ".join(p for p in path[-2:] if p),
            automation_id=aid,
            selected=self._is_selected(ctrl) if ct in SELECTABLE_TYPES else False,
            hwnd=hwnd,
            window=window,
            foreground=foreground,
        )

    @staticmethod
    def _is_selected(ctrl) -> bool:
        try:
            pat = ctrl.GetSelectionItemPattern()
            return bool(pat and pat.IsSelected)
        except Exception:
            return False

    @staticmethod
    def _has_invoke(ctrl) -> bool:
        try:
            return ctrl.GetInvokePattern() is not None
        except Exception:
            return False
