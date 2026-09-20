"""Live smoke test. Parses the foreground window; with --decide also asks Laya (no click).

    uv run python tests/scripts/smoke.py                          # list elements
    uv run python tests/scripts/smoke.py --decide "open the File menu"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from laya_agent.config import Config
from laya_agent.perception.base import make_screen_parser


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decide", metavar="GOAL", default=None)
    ap.add_argument("--delay", type=float, default=2.0, help="seconds to switch to the target window")
    ap.add_argument("--save", action="store_true", help="save screenshot to tests/out/smoke.png")
    args = ap.parse_args()
    cfg = Config()
    print(f"focus the target window... {args.delay:.0f}s")
    time.sleep(args.delay)
    parser = make_screen_parser(cfg)
    t0 = time.perf_counter()
    snap = parser.parse()
    dt = time.perf_counter() - t0
    print(f"window={snap.window_title!r} elements={len(snap.elements)} parse={dt * 1000:.0f}ms size={snap.width}x{snap.height}")
    for e in snap.elements[:60]:
        print(f"  [{e.id:>3}] {e.label()}  @{e.center}")
    if args.save:
        out = Path("tests/out")
        out.mkdir(exist_ok=True)
        (out / "smoke.png").write_bytes(snap.png)
    if args.decide:
        from laya_agent.brain.laya_policy import LayaPolicy

        policy = LayaPolicy(cfg)
        t0 = time.perf_counter()
        d = policy.decide(args.decide, snap, history=[])
        ms = (time.perf_counter() - t0) * 1000
        print(f"\ndecision={d.action} conf={d.confidence:.2f} done_p={d.done_prob:.2f} ({ms:.0f}ms)")
        for label, p in d.top_k:
            print(f"  {p:6.3f}  {label}")


if __name__ == "__main__":
    main()
