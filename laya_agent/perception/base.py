"""Perception boundary. Any screen parser implements this and returns a Snapshot."""
from __future__ import annotations

from typing import Protocol

from laya_agent.config import Config
from laya_agent.models import Snapshot


class ScreenParser(Protocol):
    def parse(self, exclude_hwnd: int | None = None) -> Snapshot:
        """Capture the screen and list actionable elements.

        exclude_hwnd: our own window; its elements are never returned.
        """
        ...


def make_screen_parser(cfg: Config) -> ScreenParser:
    if cfg.screen_parser == "uia":
        from laya_agent.perception.uia import UIAScreenParser

        return UIAScreenParser(cfg)
    if cfg.screen_parser == "omniparser":
        from laya_agent.perception.omniparser import OmniParserScreenParser

        return OmniParserScreenParser(cfg)
    raise ValueError(f"unknown screen_parser {cfg.screen_parser!r}")
