# CLAUDE.md

Read `docs/ARCHITECTURE.md` before changing anything. It is the source of truth for why
the code is shaped the way it is.

## Design ethos

1. **Laya decides, it does not perceive or generate.** It is a text-only, calibrated
   option picker. Anything that needs eyes goes in `perception/`. Anything that needs
   generated text goes through `brain/text_gen.py`. Do not try to make Laya do either.
2. **Deterministic before probabilistic.** If a rule can resolve the case (a name match,
   a learned alias, a duplicate tie-break), it runs before the model. The model handles
   what rules cannot, and the human handles what the model cannot. Every decision must be
   attributable: the log says which decider chose.
3. **Calibration is the product.** The low-confidence gate only works when probabilities
   are honest. Keep the fine pass at ≤ 10 options. Do not feed the gate numbers from an
   11+ option pass. Do not "fix" a wrong answer by raising confidence.
4. **Corrections must stick.** When the user picks an option, the system learns it
   (`aliases.py`). New failure modes should get the same treatment: a cheap way for one
   correction to prevent the repeat.
5. **Layers talk through `models.py` and protocols only.** No cross-layer imports of
   internals. Swaps are config values. If a change needs a new field, add it to the
   dataclass, not to a side channel.
6. **Everything measurable gets measured.** Timings and probabilities in the docs come
   from runs on this machine, with the command that produced them. Replace numbers when
   they change; do not leave stale ones.
7. **Visible over clever.** The user must be able to see every element the parser found
   and every option the model weighed. Do not add a pruning step without adding it to the
   debug overlay and the step log.

## Measuring decider changes

Any change to `brain/` (framing, question layout, shortlist, lexical rules, confidence,
thresholds) is measured before and after with the same command, and both tables go in
`docs/ARCHITECTURE.md`:

```
PYTHONHASHSEED=0 uv run python -m tests.eval.run -v
```

Read four numbers: broad semantic top-1 (accuracy the user feels), recall@fine_k (did
pruning lose the answer), wrong+confident (acted instead of asking: the safety number),
correct+asked (needless questions). A change that raises top-1 but raises wrong+confident
is a regression. Add a case to `tests/eval/cases.py` for every real failure you fix, with
an `EQUIV` entry when the screen has an equally correct control. Refresh
`tests/eval/distractors.json` only deliberately (`uv run python -m tests.eval.cases`), and
say so in the doc, because it changes every number.

## Keeping the docs aligned

- Any change to a layer boundary, a decider rule, a config default, a measured number, or
  a known limitation updates `docs/ARCHITECTURE.md` **in the same commit**. Add a row to
  its decisions log with the date and the reason.
- `README.md` stays the short "how to run" page. Detail belongs in the architecture doc.
- Before finishing a task, re-read the sections of `docs/ARCHITECTURE.md` you touched and
  check they match the code.

## Workflow rules

- Python only through `uv` (`uv run`, `uv add`). Never bare `pip` or `python`: on this
  machine they resolve to different interpreters.
- One git commit per layer or decision; each commit leaves `uv run pytest` green.
- Tests use fakes at layer boundaries. No unit test downloads the model or touches the
  screen. Live checks use `tests/scripts/smoke.py --window <title> --decide "<goal>"`.
- Console output needs `PYTHONUTF8=1` on Windows or window titles crash `print`.
- Keep `Config` the only home for knobs. No magic numbers in layer code.

## Where things live

```
laya_agent/models.py      shared dataclasses (the layer contract)
laya_agent/config.py      every knob
laya_agent/perception/    ScreenParser: uia.py (v1), omniparser.py (stub)
laya_agent/brain/         lexical.py, aliases.py, laya_policy.py, text_gen.py, model_loader.py
laya_agent/actuation/     executor.py (pyautogui, own-window hiding)
laya_agent/agent/loop.py  framework-free loop, LoopEvents protocol
laya_agent/cli.py         console adapter
laya_agent/ui/            PySide6 adapter
laya_agent/debug.py       annotated overlay, element listing
tests/                    unit tests (fakes) and scripts/smoke.py (live)
docs/ARCHITECTURE.md      design decisions, measured numbers, decisions log
```
