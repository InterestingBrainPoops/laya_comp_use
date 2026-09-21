"""Headless REPL. Alt-tab to the target app during the countdown.

    uv run python -m laya_agent.cli "open the Edit menu"
    uv run python -m laya_agent.cli            # interactive; '? question' asks about the screen
"""
from __future__ import annotations

import argparse
import sys
import time

from laya_agent.agent.loop import build_loop
from laya_agent.config import Config
from laya_agent.models import InputRequest, StepResult


class ConsoleEvents:
    def on_log(self, msg: str) -> None:
        print(f"  {msg}")

    def on_step(self, result: StepResult) -> None:
        tm = result.timings
        if result.executed:
            print(f"  -> {result.note}")
        print(f"     timing: parse {tm.get('parse_ms', 0):.0f}ms (uia {tm.get('uia_ms', 0):.0f}ms, {tm.get('uia_nodes', 0):.0f} nodes) "
              f"decide {tm.get('decide_ms', 0):.0f}ms (laya {tm.get('laya_passes', 0):.0f}x {tm.get('laya_ms', 0):.0f}ms) "
              f"act {tm.get('act_ms', 0):.0f}ms")

    def on_input_needed(self, req: InputRequest) -> str:
        print(f"\n  ? {req.reason}")
        for i, (label, p) in enumerate(req.options, 1):
            print(f"    {i}. {label}  ({p:.2f})")
        prompt = "  type the text> " if req.kind == "text" else "  pick a number, give a hint, or 'stop'> "
        try:
            return input(prompt)
        except EOFError:
            return "stop"

    def on_finished(self, summary: str) -> None:
        print(f"  == {summary}\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("goal", nargs="?", default=None)
    ap.add_argument("--delay", type=float, default=3.0, help="seconds to focus the target window")
    ap.add_argument("--dry-run", action="store_true", help="decide but never click")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--window", metavar="TITLE_SUBSTR", default=None, help="act only in the window whose title contains this")
    args = ap.parse_args(argv)

    cfg = Config(dry_run=args.dry_run, target_window=args.window)
    if args.max_steps:
        cfg.max_steps = args.max_steps
    loop = build_loop(cfg, ConsoleEvents())
    print(f"  session log: {loop.log.path or 'off'}")
    loop.policy.preload(lambda m: print(f"  {m}"))

    def run_once(text: str) -> None:
        print(f"focus the target window... {args.delay:.0f}s")
        time.sleep(args.delay)
        if text.startswith("?"):
            p = loop.answer(text[1:].strip())
            print(f"  P(true) = {p:.2f}\n")
        else:
            loop.run(text)

    if args.goal:
        run_once(args.goal)
        return
    print("goal> (empty line quits)")
    while True:
        try:
            text = input("goal> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        run_once(text)


if __name__ == "__main__":
    sys.exit(main())
