"""v1.5 hook: vision-based parser (microsoft/OmniParser-v2.0). Not implemented yet.

Same contract as UIAScreenParser. Intended for canvas-only UIs where the
accessibility tree is empty (games, some Electron apps, remote desktops).
"""
from __future__ import annotations

from laya_agent.config import Config
from laya_agent.models import Snapshot


class OmniParserScreenParser:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def parse(self, exclude_hwnd: int | None = None) -> Snapshot:
        raise NotImplementedError(
            "OmniParser perception is a v1.5 feature. Set config.screen_parser to 'uia'."
        )
