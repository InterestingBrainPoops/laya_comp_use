"""Evaluation cases for the decision pipeline. Extend freely; keep answers unambiguous.

Each case: (kind, goal, candidate elements, 1-based index of the correct one).
`kind` is "literal" (a goal word appears in the answer's name) or "semantic" (no overlap).
EQUIV lists other on-screen names that are equally correct when the case is embedded in a
real screen's distractors (Chrome names its back button 'Go back', Terminal 'MinimizeButton').
"""
from __future__ import annotations

import json
from pathlib import Path

from laya_agent.models import Rect, UIElement

HERE = Path(__file__).parent


def els(*labels: str) -> list[UIElement]:
    out = []
    for i, l in enumerate(labels):
        kind, name = l.split(":", 1)
        out.append(UIElement(id=i + 1, kind=kind, name=name, rect=Rect(0, i * 10, 50, i * 10 + 9)))
    return out


CASES: list[tuple[str, str, list[UIElement], int]] = [
    # literal
    ("literal", "open the edit menu", els("MenuItem:File", "MenuItem:Edit", "MenuItem:View", "MenuItem:Help"), 2),
    ("literal", "switch to the NandhaKishorM tab", els("TabItem:NandhaKishorM/laya", "TabItem:convaiinnovations/laya · Hugging Face", "Button:New Tab", "Button:Install GitHub"), 1),
    ("literal", "click install github", els("TabItem:NandhaKishorM/laya", "TabItem:convaiinnovations/laya · Hugging Face", "Button:New Tab", "Button:Install GitHub"), 4),
    ("literal", "open a new tab", els("Button:New Tab", "Button:Close Tab", "Button:Tab search", "TabItem:Windows PowerShell", "Button:Minimize"), 1),
    ("literal", "press delete", els("MenuItem:Delete", "MenuItem:Rename", "MenuItem:Copy", "MenuItem:Paste"), 1),
    # semantic: synonyms / intent, no word overlap with the right answer
    ("semantic", "log in", els("Hyperlink:Sign in", "Hyperlink:Register", "Hyperlink:Forgot password", "Button:Search"), 1),
    ("semantic", "go back to the previous page", els("Button:Back", "Button:Forward", "Button:Reload this page", "Button:Home"), 1),
    ("semantic", "make it louder", els("Slider:Volume", "Button:Mute", "Button:Play", "Button:Next"), 1),
    ("semantic", "play the music", els("Button:Previous", "Button:Play", "Button:Next", "Button:Shuffle", "Slider:Volume"), 2),
    ("semantic", "get rid of this file", els("MenuItem:Open", "MenuItem:Rename", "MenuItem:Delete", "MenuItem:Properties"), 3),
    ("semantic", "type a web address", els("Edit:Address and search bar", "Button:Bookmark this tab", "Button:New Tab", "Button:Search tabs"), 1),
    ("semantic", "open the browser's settings", els("Button:Back", "Button:Reload this page", "Button:Customize and control Google Chrome", "Button:Bookmark this tab"), 3),
    ("semantic", "send it", els("Edit:Message", "Button:Attach", "Button:Send", "Button:Emoji"), 3),
    ("semantic", "switch to the github tab", els("TabItem:NandhaKishorM/laya", "TabItem:convaiinnovations/laya · Hugging Face", "Button:New Tab", "Button:Install GitHub"), 1),
    ("semantic", "save my work", els("MenuItem:File", "MenuItem:Edit", "Button:Save", "Button:Close"), 3),
    ("semantic", "make the window smaller", els("Button:Minimize", "Button:Maximize", "Button:Close", "Button:Restore"), 1),
]

EQUIV: dict[str, list[str]] = {
    "go back to the previous page": ["go back", "back"],
    "make the window smaller": ["minimize", "minimizebutton"],
    "make it louder": ["volume"],
    "get rid of this file": ["delete"],
    "open the browser's settings": ["customize and control google chrome", "settings and more"],
    "type a web address": ["address and search bar", "address bar"],
}


def correct_names(case) -> set[str]:
    kind, goal, elements, correct = case
    return {elements[correct - 1].name.lower(), *EQUIV.get(goal, [])}


def load_distractors() -> list[UIElement]:
    """Real element labels captured once from a busy desktop (11 windows), see capture()."""
    rows = json.loads((HERE / "distractors.json").read_text(encoding="utf-8"))
    return [UIElement(id=i + 1, kind=r["kind"], name=r["name"], rect=Rect(0, 0, 10, 10), window=r.get("window", ""),
                      foreground=r.get("foreground", True), path=r.get("path", "")) for i, r in enumerate(rows)]


def capture() -> None:
    """Refresh distractors.json from the live screen. Run: uv run python -m tests.eval.cases"""
    from laya_agent.config import Config
    from laya_agent.perception.base import make_screen_parser

    snap = make_screen_parser(Config()).parse()
    names = {e.name.lower() for _, _, elements, _ in CASES for e in elements}
    rows = [{"kind": e.kind, "name": e.name, "window": e.window, "foreground": e.foreground, "path": e.path}
            for e in snap.elements if e.name.lower() not in names and not e.is_window]
    (HERE / "distractors.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"captured {len(rows)} distractors from {len(snap.windows)} windows")


if __name__ == "__main__":
    capture()
