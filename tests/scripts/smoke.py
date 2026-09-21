"""Live smoke test. Parses every visible window; with --decide also asks the policy (no click).

    uv run python tests/scripts/smoke.py --all --annotate
    uv run python tests/scripts/smoke.py --decide "switch to the github tab" --annotate
    uv run python tests/scripts/smoke.py --window chrome --all        # one window only

--annotate writes tests/out/annotated.png: grey = every element, orange = shortlisted,
green = fine pass, red = chosen. --all prints every element instead of the first 60.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from laya_agent.config import Config
from laya_agent.debug import annotate_png, element_listing
from laya_agent.perception.base import make_screen_parser


def _own_window_hwnd() -> int | None:
    """The chat window, if it is open, so the smoke test sees what the agent sees."""
    try:
        import win32gui

        return win32gui.FindWindow(None, "Laya screen agent") or None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decide", metavar="GOAL", default=None)
    ap.add_argument("--window", metavar="TITLE_SUBSTR", default=None, help="parse only the window whose title contains this")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to arrange windows first")
    ap.add_argument("--all", action="store_true", help="print every element")
    ap.add_argument("--annotate", action="store_true", help="save tests/out/annotated.png")
    args = ap.parse_args()
    cfg = Config(target_window=args.window)
    if args.delay:
        print(f"arrange your windows... {args.delay:.0f}s")
        time.sleep(args.delay)
    parser = make_screen_parser(cfg)
    t0 = time.perf_counter()
    snap = parser.parse(exclude_hwnd=_own_window_hwnd(), window_title=cfg.target_window)
    dt = time.perf_counter() - t0
    print(f"front={snap.window_title!r} windows={len(snap.windows)} elements={len(snap.elements)} parse={dt * 1000:.0f}ms size={snap.width}x{snap.height}")

    shortlist: set[int] = set()
    fine: set[int] = set()
    chosen: int | None = None
    if args.decide:
        from laya_agent.brain.laya_policy import LayaPolicy

        policy = LayaPolicy(cfg)
        policy.preload(lambda m: print(f"  {m}"))
        t0 = time.perf_counter()
        d = policy.decide(args.decide, snap, history=[])
        ms = (time.perf_counter() - t0) * 1000
        shortlist, fine = set(d.raw["_shortlist"]), set(d.raw["_fine"])
        chosen = d.element.id if d.element else None
        print(f"\ndecision={d.action} by {d.raw['_decided_by']} conf={d.confidence:.2f} done_p={d.done_prob:.2f} ({ms:.0f}ms)")
        for label, p in d.top_k:
            print(f"  {p:6.3f}  {label}")
        print()

    listing = element_listing(snap, shortlist, fine)
    lines = listing.splitlines()
    print("\n".join(lines if args.all else lines[:61]))
    if not args.all and len(lines) > 61:
        print(f"  ... {len(lines) - 61} more (use --all)")
    if args.annotate:
        out = Path("tests/out")
        out.mkdir(exist_ok=True)
        (out / "annotated.png").write_bytes(annotate_png(snap, shortlist, fine, chosen))
        print(f"\nwrote {out / 'annotated.png'}")


if __name__ == "__main__":
    main()
