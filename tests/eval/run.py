"""Decision-pipeline evaluation. Measures the real code path (LayaPolicy.decide / shortlist).

    uv run python -m tests.eval.run            # all three tables
    uv run python -m tests.eval.run --quick    # skip the broad end-to-end table

Tables:
  1. narrow top-1: policy.decide on each case's 4-5 candidates alone (does the decider pick right?)
  2. broad recall@FINE_K: does the correct element survive policy.shortlist among ~90 real
     distractors from 11 windows, 3 shuffles per case?
  3. broad top-1: policy.decide on the same broad lists (the number the user feels)
Also prints ms per case and, for wrong answers, the top probability (the gate's input).
"""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import replace

from laya_agent.brain.laya_policy import LayaPolicy
from laya_agent.brain.lexical import lexical_scores
from laya_agent.config import Config
from laya_agent.models import Snapshot
from tests.eval.cases import CASES, correct_names, load_distractors

SEEDS = (1, 2, 3)
args_verbose = False


def _snap(elements) -> Snapshot:
    return Snapshot(png=b"", width=1, height=1, window_title="desktop", elements=elements, windows=["desktop"])


def _broad(case_els, distractors, seed):
    rng = random.Random(seed)
    els = list(case_els) + distractors
    rng.shuffle(els)
    return [replace(e, id=i + 1) for i, e in enumerate(els)]


def _row(name, kind, hits, total, extra=""):
    print(f"  {name:<14} {kind:<9} {hits:>2}/{total:<2} ({hits / total:4.0%}){extra}")


def table_narrow(policy: LayaPolicy) -> dict:
    print("\n1. narrow top-1 (candidates only)")
    out = {}
    for kind in ("literal", "semantic"):
        cases = [c for c in CASES if c[0] == kind]
        hits, wrong_p, t0 = 0, [], time.perf_counter()
        for c in cases:
            _, goal, elements, correct = c
            d = policy.decide(goal, _snap(elements), [])
            ok = d.element is not None and d.element.name.lower() in correct_names(c)
            hits += ok
            if not ok:
                wrong_p.append(d.top_k[0][1])
                print(f"      x {goal!r}: chose {d.top_k[0][0]} p={d.top_k[0][1]:.2f} by {d.raw['_decided_by']}")
        ms = (time.perf_counter() - t0) * 1000 / len(cases)
        wp = f"  p(top) when wrong {sum(wrong_p) / len(wrong_p):.2f}" if wrong_p else ""
        _row("narrow top-1", kind, hits, len(cases), f"  {ms:4.0f}ms/case{wp}")
        out[kind] = (hits, len(cases))
    return out


def table_broad(policy: LayaPolicy, distractors, do_top1=True) -> dict:
    print(f"\n2/3. broad: {len(distractors)} distractors, {len(SEEDS)} shuffles per case")
    out = {}
    for kind in ("literal", "semantic"):
        cases = [c for c in CASES if c[0] == kind]
        rec, top1, wrong_p, t_short, t_dec = 0, 0, [], 0.0, 0.0
        missed_rec, missed_top, confident_wrong = {}, {}, {}
        nag = 0  # correct pick but below the gate: the user is asked needlessly
        for c in cases:
            _, goal, case_els, _ = c
            ok_names = correct_names(c)
            for seed in SEEDS:
                elements = _broad(case_els, distractors, seed)
                by_id = {e.id: e for e in elements}
                t = time.perf_counter()
                lex = lexical_scores(goal, elements, policy.aliases.as_dict())
                by_lex = sorted(elements, key=lambda e: (lex[e.id], e.selected, -e.id), reverse=True)
                kept = policy.shortlist(goal, policy.build_state(goal, _snap(elements), []), elements, by_lex, lex)
                t_short += time.perf_counter() - t
                ok = any(e.name.lower() in ok_names for e in kept)
                rec += ok
                if not ok:
                    missed_rec[goal] = missed_rec.get(goal, 0) + 1
                if do_top1:
                    t = time.perf_counter()
                    d = policy.decide(goal, _snap(elements), [])
                    t_dec += time.perf_counter() - t
                    ok1 = d.element is not None and d.element.name.lower() in ok_names
                    top1 += ok1
                    if ok1 and d.confidence < policy.cfg.conf_threshold:
                        nag += 1
                    if not ok1:
                        wrong_p.append(d.top_k[0][1])
                        missed_top[goal] = missed_top.get(goal, 0) + 1
                        if d.confidence >= policy.cfg.conf_threshold:
                            confident_wrong[goal] = confident_wrong.get(goal, 0) + 1
                        if args_verbose:
                            in_fine = any(by_id[i].name.lower() in ok_names for i in d.raw["_fine"])
                            print(f"      x {goal!r} seed {seed}: chose {d.top_k[0][0][:50]} p={d.top_k[0][1]:.2f}"
                                  f" (correct {'in' if in_fine else 'NOT in'} fine set)")
        n = len(cases) * len(SEEDS)
        _row("recall@8", kind, rec, n, f"  {t_short * 1000 / n:4.0f}ms/case")
        if missed_rec:
            print("      missed: " + "; ".join(f"{g} x{k}" for g, k in missed_rec.items()))
        out[f"recall_{kind}"] = (rec, n)
        if do_top1:
            wp = f"  p(top) when wrong {sum(wrong_p) / len(wrong_p):.2f}" if wrong_p else ""
            _row("broad top-1", kind, top1, n, f"  {t_dec * 1000 / n:4.0f}ms/case{wp}")
            if missed_top:
                print("      wrong: " + "; ".join(f"{g} x{k}" for g, k in missed_top.items()))
            cw = sum(confident_wrong.values())
            _row("wrong+confident", kind, cw, n, "  (acted wrongly instead of asking)" + (
                ": " + "; ".join(f"{g} x{k}" for g, k in confident_wrong.items()) if cw else ""))
            _row("correct+asked", kind, nag, n, f"  (needless question, gate {policy.cfg.conf_threshold})")
            out[f"top1_{kind}"] = (top1, n)
            out[f"confident_wrong_{kind}"] = (cw, n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fine-k", type=int, default=None)
    ap.add_argument("--fusion", type=float, default=None)
    ap.add_argument("--shortlist", default=None, help="retriever | chunks")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each wrong broad pick")
    ap.add_argument("--gate", type=float, default=None, help="conf_threshold to score wrong+confident / correct+asked against")
    args = ap.parse_args()
    global args_verbose
    args_verbose = args.verbose
    cfg = Config(alias_path=None, log_dir=None)
    if args.fine_k:
        cfg.fine_k = args.fine_k
    if args.fusion is not None:
        cfg.fusion_weight = args.fusion
    if args.shortlist:
        cfg.shortlist = args.shortlist
    if args.gate is not None:
        cfg.conf_threshold = args.gate
    print(f"config: shortlist={cfg.shortlist} fine_k={cfg.fine_k} fusion={cfg.fusion_weight} retriever_top={cfg.retriever_top}")
    policy = LayaPolicy(cfg)
    policy.preload(lambda m: print("  " + m))
    table_narrow(policy)
    table_broad(policy, load_distractors(), do_top1=not args.quick)


if __name__ == "__main__":
    main()
