"""Freeform text generation boundary (v2 slot).

v1 ships NullTextGenerator: the loop asks the human to type the text.
v2 swaps in HFTextGenerator (a ~1B causal LM) by setting config.text_generator = "hf".
Nothing outside this module changes.
"""
from __future__ import annotations

from typing import Any, Protocol

from laya_agent.config import Config


class TextNeedsHuman(Exception):
    """Raised by generators that cannot produce text; the loop asks the user."""


class TextGenerator(Protocol):
    def generate(self, task: str, screen_state: dict[str, Any]) -> str:
        """Return the literal text to type for `task` given the current screen state."""
        ...


class NullTextGenerator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def generate(self, task: str, screen_state: dict[str, Any]) -> str:
        raise TextNeedsHuman(task)


class HFTextGenerator:
    """v2: transformers causal LM (default Qwen/Qwen3-1.7B, bf16, fits beside Laya on 6GB).

    Planned prompt: system = "You type exactly what should go in the focused field. Reply
    with the text only." user = goal + window title + focused field label. Strip thinking
    tags, take the first line.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        raise NotImplementedError(
            "HFTextGenerator is a v2 feature. Set config.text_generator to 'null'."
        )

    def generate(self, task: str, screen_state: dict[str, Any]) -> str:  # pragma: no cover
        raise NotImplementedError


def make_text_generator(cfg: Config) -> TextGenerator:
    if cfg.text_generator == "null":
        return NullTextGenerator(cfg)
    if cfg.text_generator == "hf":
        return HFTextGenerator(cfg)
    raise ValueError(f"unknown text_generator {cfg.text_generator!r}")
