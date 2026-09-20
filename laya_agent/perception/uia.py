"""Windows UI Automation parser: accessibility tree -> UIElement list + screenshot."""
from __future__ import annotations

import io
import time

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


def _kind(control_type_name: str) -> str:
    return control_type_name.removesuffix("Control")


def _clean_name(raw: str) -> str:
    """Drop icon-font glyphs (private use area) and control chars; collapse whitespace."""
    out = "".join(ch for ch in raw if not (0xE000 <= ord(ch) <= 0xF8FF or ord(ch) < 32))
    return " ".join(out.split())


def _is_readable(name: str) -> bool:
    """True for human labels, false for class-name noise like DesktopWindowXamlSource."""
    return bool(name) and (" " in name or len(name) <= 14)


class UIAScreenParser:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        auto.SetGlobalSearchTimeout(2.0)

    # -- public ---------------------------------------------------------------
    def parse(self, exclude_hwnd: int | None = None, window_title: str | None = None) -> Snapshot:
        png, w, h = self._screenshot()
        with auto.UIAutomationInitializerInThread(debug=False):
            root = self._find_window(window_title, exclude_hwnd) if window_title else self._target_window(exclude_hwnd)
            title = _clean_name(root.Name) if root is not None else ""
            elements = self._walk(root, exclude_hwnd, title) if root is not None else []
        return Snapshot(png=png, width=w, height=h, window_title=title, elements=elements, source="uia")

    @staticmethod
    def _find_window(title_substr: str, exclude_hwnd: int | None):
        needle = title_substr.lower()
        for win in auto.GetRootControl().GetChildren():
            if win.NativeWindowHandle != exclude_hwnd and needle in (win.Name or "").lower():
                return win
        return None

    # -- internals ------------------------------------------------------------
    def _screenshot(self) -> tuple[bytes, int, int]:
        with mss.mss() as sct:
            mon = sct.monitors[self.cfg.monitor_index]
            shot = sct.grab(mon)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return buf.getvalue(), img.width, img.height

    def _target_window(self, exclude_hwnd: int | None):
        """Foreground window, unless it is ours; then the first visible top-level window."""
        fg = auto.GetForegroundControl()
        top = fg.GetTopLevelControl() if fg is not None else None
        if top is not None and top.NativeWindowHandle != exclude_hwnd:
            return top
        for win in auto.GetRootControl().GetChildren():
            if win.NativeWindowHandle == exclude_hwnd or win.IsOffscreen:
                continue
            if win.ControlTypeName == "WindowControl" and win.BoundingRectangle.width() > 0:
                return win
        return None

    def _walk(self, root, exclude_hwnd: int | None, window_title: str = "") -> list[UIElement]:
        """Breadth-first so shallow chrome (tabs, toolbars, menus) is found before deep page
        content. Each Document subtree (a web page) gets its own node budget so one huge
        page cannot starve the rest of the window."""
        from collections import deque

        deadline = time.monotonic() + self.cfg.uia_deadline_s
        elements: list[UIElement] = []
        seen: set[tuple[str, str, int, int]] = set()
        visited = 0
        doc_budget: dict[int, int] = {}  # id(document control) -> nodes left
        queue: deque[tuple[object, int, list[str], int | None]] = deque([(root, 0, [], None)])
        while queue:
            ctrl, depth, path, doc = queue.popleft()
            visited += 1
            if visited > self.cfg.uia_max_nodes or time.monotonic() > deadline:
                break
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
            if not offscreen and r.area > 0:
                el = self._maybe_element(ctrl, ct, name, r, path, len(elements))
                if el is not None:
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
                queue.append((child, depth + 1, child_path, doc))
        return elements

    def _maybe_element(self, ctrl, ct: str, name: str, r: Rect, path: list[str], idx: int) -> UIElement | None:
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
