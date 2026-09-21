"""Fast, offline-first loading of the Laya checkpoint.

`laya.load(repo_id)` calls snapshot_download, which contacts the hub to re-verify a cached
model on every start (measured 22s online vs 5s offline on this machine). We resolve the
cached folder with local files only and hand Laya a path, so the network is touched once,
on the very first run.
"""
from __future__ import annotations

import contextlib
import os
import threading
import time
from typing import Any, Callable

from laya_agent.config import Config

PredictFn = Callable[[Any, dict[str, Any]], dict[str, Any]]
_lock = threading.Lock()


def resolve_model_dir(model_id: str, log: Callable[[str], None] = lambda s: None) -> str:
    """Local folder for `model_id`. Cached -> no network. Not cached -> download once."""
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(model_id, local_files_only=True)
    except (LocalEntryNotFoundError, FileNotFoundError, OSError):
        log(f"downloading {model_id} (first run only, ~850MB)")
        return snapshot_download(model_id)


@contextlib.contextmanager
def skip_random_init():
    """Make torch.nn.init.* no-ops. Laya builds ModernBERT-large from config, which randomly
    initialises 395M weights (measured 7-13s) that the checkpoint then overwrites. With init
    skipped the build takes ~3s and outputs are bit-identical."""
    import torch.nn.init as init

    names = [n for n in dir(init) if n.endswith("_") and not n.startswith("_") and callable(getattr(init, n))]
    saved = {n: getattr(init, n) for n in names}
    for n in names:
        setattr(init, n, lambda tensor, *a, **k: tensor)
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(init, n, f)


def load_predict(cfg: Config, log: Callable[[str], None] = lambda s: None) -> PredictFn:
    """Load Laya from the local cache, apply config, run one warmup pass. Thread-safe."""
    with _lock:
        t0 = time.perf_counter()
        model_dir = resolve_model_dir(cfg.model_id, log)
        import laya

        with skip_random_init():
            agent = laya.load(model_dir, device=cfg.device)
        agent.cfg["head_max_len"] = cfg.head_max_len
        agent.predict({"goal": "warmup"}, {"q": {"type": "noul", "instructions": "warmup"}})
        log(f"model ready on {agent.device} in {time.perf_counter() - t0:.1f}s")
        return agent.predict
