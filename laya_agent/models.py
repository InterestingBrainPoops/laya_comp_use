"""Shared dataclasses. Every layer talks only through these."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass(frozen=True)
class UIElement:
    """One actionable thing on screen, as text plus a place to click."""

    id: int
    kind: str  # "Button", "Hyperlink", "Edit", ...
    name: str
    rect: Rect
    path: str = ""  # ancestor names, "Settings > Navigation"
    automation_id: str = ""
    selected: bool = False  # the active tab / checked item
    hwnd: int = 0  # top-level window that owns this element
    window: str = ""  # that window's title (short)
    foreground: bool = True  # owner window was in front when parsed
    minimized: bool = False  # only meaningful for kind == "Window"

    @property
    def center(self) -> tuple[int, int]:
        return self.rect.center

    @property
    def is_window(self) -> bool:
        return self.kind == "Window"

    def label(self) -> str:
        """Compact text the decision model sees as an option, in plain words."""
        s = f"{KIND_WORDS.get(self.kind, self.kind.lower())} '{self.name}'"
        if self.is_window and self.window and self.window.lower() not in self.name.lower():
            s += f" of app {self.window}"  # Spotify's title is the playing song
        if self.selected:
            s += " (current)"
        if self.minimized:
            s += " (minimized)"
        if self.path:
            s += f" in {self.path}"
        if not self.foreground and self.window and not self.is_window:
            s += f" in window {self.window}"
        return s


KIND_WORDS = {
    "TabItem": "tab", "Hyperlink": "link", "Edit": "text field", "MenuItem": "menu item",
    "ListItem": "list item", "TreeItem": "tree item", "ComboBox": "dropdown", "CheckBox": "checkbox",
    "RadioButton": "radio button", "SplitButton": "button", "Button": "button", "Window": "window",
}


@dataclass
class Snapshot:
    """What perception hands to the brain."""

    png: bytes
    width: int
    height: int
    window_title: str  # foreground window
    elements: list[UIElement]
    source: str = "uia"  # which ScreenParser produced it
    windows: list[str] = field(default_factory=list)  # every window parsed, z-order, front first

    def by_id(self, element_id: int) -> UIElement | None:
        return next((e for e in self.elements if e.id == element_id), None)


# Non-click actions the policy may choose. Element clicks are "click_<id>".
ACTION_SCROLL_DOWN = "scroll_down"
ACTION_SCROLL_UP = "scroll_up"
ACTION_DONE = "done"
ACTION_NEED_TEXT = "need_text"
META_ACTIONS: dict[str, str] = {
    ACTION_SCROLL_DOWN: "scroll the page down to reveal more content",
    ACTION_SCROLL_UP: "scroll the page up",
    ACTION_DONE: "the goal is already fully achieved on this screen, stop",
    ACTION_NEED_TEXT: "the next step is typing text into a field, not clicking",
}


@dataclass
class Decision:
    action: str  # "click_3" | one of META_ACTIONS
    element: UIElement | None
    confidence: float
    done_prob: float
    top_k: list[tuple[str, float]]  # (human label, probability) best first
    top_actions: list[str] = field(default_factory=list)  # action ids aligned with top_k
    raw: dict[str, Any] = field(default_factory=dict)  # model output; "_elements" maps action -> UIElement

    @property
    def is_click(self) -> bool:
        return self.element is not None


@dataclass
class StepResult:
    step: int
    snapshot: Snapshot
    decision: Decision
    executed: bool
    note: str = ""


@dataclass
class InputRequest:
    """Loop pauses and asks the human. Reply with a pick index or free text."""

    reason: str
    options: list[tuple[str, float]]  # same shape as Decision.top_k
    step: int
    kind: str = "pick"  # "pick" (choose/hint) | "text" (type literal text)
