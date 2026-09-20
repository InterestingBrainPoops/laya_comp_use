"""Runtime configuration. Swapping perception or the text model is a value here."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # Decision model
    model_id: str = "convaiinnovations/laya"
    device: str | None = None  # None -> laya picks cuda if available
    head_max_len: int = 320  # option budget; state gets the rest of 512 tokens
    conf_threshold: float = 0.7  # below this the loop asks the human
    done_threshold: float = 0.8  # noul P(goal achieved) above this ends the task
    max_elements: int = 20  # options shown to Laya per step

    # Perception
    screen_parser: str = "uia"  # "uia" | "omniparser" (v1.5)
    monitor_index: int = 1  # mss monitor index (1 = primary)
    uia_max_depth: int = 14
    uia_max_nodes: int = 2500

    # Text generation (v2)
    text_generator: str = "null"  # "null" | "hf"
    text_model_id: str = "Qwen/Qwen3-1.7B"

    # Loop
    max_steps: int = 12
    step_delay_s: float = 0.8
    history_len: int = 4
    dry_run: bool = False  # decide but never act

    extra: dict = field(default_factory=dict)
