"""Per-session debug log. One JSONL file with every event, a readable .log mirror, and
optional per-step screenshots. Framework-free; the loop is the only writer.

    logs/2026-09-20_14-05-33.jsonl   machine-readable, one JSON object per line
    logs/2026-09-20_14-05-33.log     the same events, one readable line each
    logs/2026-09-20_14-05-33/        step_3.png, step_3_annotated.png (if log_screenshots)
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from laya_agent.models import Decision, InputRequest, Snapshot, StepResult, UIElement


def _jsonable(o: Any) -> Any:
    if isinstance(o, UIElement):
        return {"id": o.id, "kind": o.kind, "name": o.name, "rect": asdict(o.rect), "window": o.window,
                "selected": o.selected, "foreground": o.foreground}
    if is_dataclass(o):
        return {k: _jsonable(v) for k, v in asdict(o).items()}
    if isinstance(o, bytes):
        return f"<{len(o)} bytes>"
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    return o


class SessionLog:
    def __init__(self, log_dir: str | Path | None, screenshots: bool = False) -> None:
        self.enabled = log_dir is not None
        self.screenshots = screenshots
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self.path: Path | None = None
        if not self.enabled:
            return
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{stamp}.jsonl"
        self._text = d / f"{stamp}.log"
        self._shots = d / stamp
        self.event("session_start", when=datetime.now().isoformat(timespec="seconds"))

    # -- generic --------------------------------------------------------------
    def event(self, kind: str, **data: Any) -> None:
        if not self.enabled:
            return
        rec = {"t": round(time.perf_counter() - self._t0, 3), "kind": kind, **_jsonable(data)}
        line = json.dumps(rec, ensure_ascii=False)
        text = self._render(rec)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            with open(self._text, "a", encoding="utf-8") as f:
                f.write(text + "\n")

    # -- typed helpers ----------------------------------------------------------
    def goal(self, goal: str) -> None:
        self.event("goal", goal=goal)

    def parsed(self, step: int, snap: Snapshot) -> None:
        self.event(
            "parsed", step=step, front=snap.window_title, windows=snap.windows,
            n_elements=len(snap.elements), timings=snap.timings,
            elements=[e.label() + f" @{e.center}" for e in snap.elements],
        )

    def decided(self, step: int, goal: str, d: Decision) -> None:
        probs = d.raw.get("answers", {}).get("action", {}).get("probabilities", {})
        self.event(
            "decided", step=step, goal=goal, action=d.action, decided_by=d.raw.get("_decided_by"),
            confidence=d.confidence, done_prob=d.done_prob, top_k=d.top_k,
            element=d.element, lexical_top=d.raw.get("_lexical"), shortlist=d.raw.get("_shortlist"),
            fine=d.raw.get("_fine"), laya_probabilities=probs, timings=d.timings,
        )

    def asked(self, req: InputRequest, reply: str) -> None:
        self.event("human", step=req.step, input_kind=req.kind, reason=req.reason, options=req.options, reply=reply)

    def acted(self, step: int, note: str, ms: float) -> None:
        self.event("acted", step=step, note=note, act_ms=round(ms, 1))

    def step(self, r: StepResult) -> None:
        self.event("step", step=r.step, executed=r.executed, note=r.note, timings=r.timings)
        if self.screenshots and r.snapshot.png:
            self._save_shots(r)

    def finished(self, summary: str, steps: int) -> None:
        self.event("finished", summary=summary, steps=steps)

    def error(self, where: str, err: BaseException) -> None:
        self.event("error", where=where, error=repr(err))

    # -- internals ------------------------------------------------------------
    def _save_shots(self, r: StepResult) -> None:
        from laya_agent.debug import annotate_png

        self._shots.mkdir(exist_ok=True)
        (self._shots / f"step_{r.step}.png").write_bytes(r.snapshot.png)
        d = r.decision
        png = annotate_png(r.snapshot, set(d.raw.get("_shortlist", [])), set(d.raw.get("_fine", [])),
                           d.element.id if d.element else None)
        (self._shots / f"step_{r.step}_annotated.png").write_bytes(png)

    @staticmethod
    def _render(rec: dict[str, Any]) -> str:
        t, kind = rec["t"], rec["kind"]
        if kind == "parsed":
            tm = rec["timings"]
            return (f"[{t:8.3f}] step {rec['step']} parsed {rec['n_elements']} elements in {len(rec['windows'])} windows "
                    f"(front {rec['front']!r}) parse={tm.get('parse_ms')}ms uia={tm.get('uia_ms')}ms nodes={tm.get('uia_nodes')}")
        if kind == "decided":
            tm = rec["timings"]
            top = "; ".join(f"{p:.2f} {l}" for l, p in rec["top_k"][:3])
            return (f"[{t:8.3f}] step {rec['step']} decided {rec['action']} by {rec['decided_by']} conf={rec['confidence']:.2f} "
                    f"done={rec['done_prob']:.2f} laya={tm.get('laya_passes')}x/{tm.get('laya_ms')}ms lexical={tm.get('lexical_ms')}ms | {top}")
        if kind == "human":
            return f"[{t:8.3f}] step {rec['step']} asked ({rec['reason']}) -> reply {rec['reply']!r}"
        if kind == "acted":
            return f"[{t:8.3f}] step {rec['step']} {rec['note']} ({rec['act_ms']}ms)"
        if kind == "step":
            return f"[{t:8.3f}] step {rec['step']} done executed={rec['executed']} {rec['timings']}"
        rest = {k: v for k, v in rec.items() if k not in ("t", "kind")}
        return f"[{t:8.3f}] {kind} {json.dumps(rest, ensure_ascii=False)[:300]}"
