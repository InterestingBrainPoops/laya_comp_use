"""Learned aliases: when the user picks an element for a goal, remember the goal's words as
extra names for that element. Next time "github tab" matches 'NandhaKishorM/laya' lexically.
Stored as JSON next to the project (gitignored)."""
from __future__ import annotations

import json
from pathlib import Path

from laya_agent.brain.lexical import tokens


class AliasStore:
    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path else None
        self._data: dict[str, list[str]] = {}
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def remember(self, goal: str, element_name: str) -> None:
        phrase = " ".join(tokens(goal))
        if not phrase or not element_name:
            return
        names = self._data.setdefault(element_name, [])
        if phrase not in names:
            names.append(phrase)
            self._save()

    def as_dict(self) -> dict[str, list[str]]:
        return self._data

    def _save(self) -> None:
        if self.path:
            self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
