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
    max_elements: int = 120  # hard cap on elements considered per step (bounds latency)
    coarse_chunk: int = 20  # elements per shortlisting forward pass
    coarse_keep: int = 3  # survivors per chunk
    alias_path: str | None = ".laya_aliases.json"  # learned goal -> element names; None disables

    # Perception
    screen_parser: str = "uia"  # "uia" | "omniparser" (v1.5)
    target_window: str | None = None  # title substring; None = foreground window
    monitor_index: int = 1  # mss monitor index (1 = primary)
    uia_max_depth: int = 16
    uia_max_nodes: int = 1500
    uia_document_budget: int = 500  # nodes per web page (Document control) before moving on
    uia_deadline_s: float = 4.0  # total across all windows
    max_windows: int = 12  # top-level windows parsed per step, z-order front first
    uia_background_budget: int = 250  # nodes per background window (tabs/toolbars, not page content)

    # Text generation (v2)
    text_generator: str = "null"  # "null" | "hf"
    text_model_id: str = "Qwen/Qwen3-1.7B"

    # Loop
    max_steps: int = 12
    step_delay_s: float = 0.8
    history_len: int = 4
    dry_run: bool = False  # decide but never act

    # Session logging
    log_dir: str | None = "logs"  # None disables; JSONL + .log per session
    log_screenshots: bool = True  # save step_N.png and step_N_annotated.png per step

    extra: dict = field(default_factory=dict)
