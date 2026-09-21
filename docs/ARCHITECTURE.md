# Architecture and design decisions

Status: living document. Update it in the same commit as any change to a layer boundary,
a decision rule, or a measured number below. `CLAUDE.md` holds the rules for keeping it current.

## 1. What this is

A Windows desktop agent with a chat window. The user types a goal. The agent lists the
clickable elements on screen as text, decides which one to click, clicks it, repeats, and
asks the user when it is unsure. Freeform text generation (URLs, notes) is a v2 slot.

## 2. The constraint that shaped everything: Laya is text-only

[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) is a 421M
ModernBERT-large encoder with a decision head. Facts that drive the design:

| fact | consequence |
|---|---|
| No vision. Input is text or JSON. | A perception layer must turn the screen into text. |
| Never generates text. One forward pass returns a label with calibrated probabilities. | Cannot type URLs or notes. Needs a separate generator (v2). |
| 512-token window: `[CLS] instructions [SEP] options [SEP] state [SEP]`. Options get `head_max_len` tokens, the state gets the rest. | Option labels carry the element description. The state stays small (goal, window, last actions). |
| Calibration temperature for 11+ options is 0.1 (near-argmax). 6-10 options use 1.0. | Only a 10-option pass gives usable confidence. Hence coarse-to-fine. |
| Trained on email/ticket triage. Weak at literal string matching (measured: "NandhaKishorM" vs tab `NandhaKishorM/laya` scored 0.46 against 0.45 for the other tab). | Explicit names are resolved by a deterministic matcher, not by Laya. |
| ~35ms per pass on the RTX 3050, ~300ms on CPU. | Several passes per step are affordable. |

## 3. Layers

```
chat input ──► agent/loop.py  (perceive → decide → gate → act → settle)
                    │
   perception/      brain/                     actuation/
   ScreenParser ──► LayaPolicy ──────────────► Executor
   Snapshot         Decision                   pyautogui + own-window hiding
                    ├─ lexical.py  name match
                    ├─ aliases.py  learned picks
                    └─ text_gen.py TextGenerator (v2 slot)
```

Rules:

- Layers exchange only the dataclasses in `laya_agent/models.py` (`UIElement`, `Snapshot`,
  `Decision`, `StepResult`, `InputRequest`) and the protocols `ScreenParser`,
  `TextGenerator`, `LoopEvents`.
- No layer imports another layer's internals. The loop never imports Qt. The UI never
  imports uiautomation, laya, or pyautogui.
- Swapping an implementation is a value in `config.py` (`screen_parser`,
  `text_generator`), never a code change elsewhere.
- Every knob has one home: `Config`. No magic numbers in layer code.

## 4. Perception (`perception/uia.py`)

Windows UI Automation over `uiautomation`. Chosen over a vision model because it gives
exact click coordinates with no GPU cost and works for browsers and native apps. Its gap
is canvas-only UIs; `perception/omniparser.py` is the stub for that.

Decisions:

- **Breadth-first walk.** Tabs, toolbars, and menus are shallow; page content is deep and
  huge. BFS finds the chrome first, and a per-Document node budget (`uia_document_budget`)
  stops one web page from starving the rest. Deadline `uia_deadline_s`.
- **Kept control types:** buttons, links, edits, combo boxes, check/radio, menu/tab/list/tree
  items, sliders. Text/Image/Pane only when they expose Invoke.
- **Names are cleaned.** Icon-font glyphs (private use area) and control chars are dropped.
  Path segments that repeat the window title or look like class names are dropped.
- **`selected`** is read for tabs/list/radio/tree items and shown as "(current)" so
  "switch to" makes sense.
- Target window is the foreground window, or `config.target_window` by title substring.
  Our own window is excluded by handle and minimized during capture and click, restored
  with `SW_SHOWNOACTIVATE` so focus stays on the target app.

Measured: Windows Terminal 17 elements in 200ms; Chrome with a large page ~100 elements
in ~4s (the COM property calls dominate; bulk `FindAll` with a cache request is the known
speedup, not done yet).

## 5. Decision (`brain/`)

Order of deciders in `LayaPolicy.decide`:

1. **Lexical** (`lexical.py`). Goal tokens minus UI stop words are matched against element
   name tokens (exact 1.0, containment 0.85, difflib ≥ 0.8). Score blends strength and
   coverage; a distinctive token ≥ 6 chars matching exactly is decisive. If the goal names
   a kind ("tab", "button", "link"), other kinds are multiplied by 0.6 unless the kind word
   is part of the element's own name ("New Tab"). Best ≥ 0.9 with a 0.25 margin over the
   nearest differently-named rival decides without Laya. Same-named duplicates are not
   rivals; the first in tree order wins.
2. **Laya, coarse-to-fine.** Elements in chunks of `coarse_chunk` (20), keep `coarse_keep`
   (3) per chunk. Guaranteed survivors: lexical ≥ 0.5 and kind matches. Merge, cut again
   if needed, then the fine pass: 8 elements + `done` + `scroll_down` = 10 options.
   `need_text` is a separate yes/no question, not an option, so it does not steal
   probability from elements.
3. **Ask the human.** Gate reads only the fine pass. Confidence is the top-1 margin
   (`margin_confidence`), not Laya's entropy score, which collapses with many options.
   Threshold `conf_threshold` (0.7). Also asks on two identical consecutive actions and on
   `done` with P(done) < 0.5.

**Aliases** (`aliases.py`). A human pick stores the goal's tokens as extra names for the
chosen element in `.laya_aliases.json`. The lexical matcher reads them, so a vague goal
becomes explicit after one correction. This is the intended way the agent gets better.

Instruction wording (measured on the Chrome tab strip): putting the goal inside the
question text ("The user asked: ... Which screen element is the user asking to click?")
beats a generic "which action moves closest to the goal" framing.

## 6. Loop and gating (`agent/loop.py`)

Framework-free. `LoopEvents` is implemented by the console (`cli.py`) and by Qt signals
(`ui/main_window.py`). The Qt adapter blocks the worker thread on a `threading.Event`
until the user replies. Replies: a digit picks an option (and teaches an alias), `stop`
aborts, anything else is a hint appended to the state's `previous_actions`.

`need_text` calls `TextGenerator.generate`. v1's `NullTextGenerator` raises and the loop
asks the user for the literal text. v2 swaps in `HFTextGenerator` (planned Qwen3-1.7B,
bf16, fits beside Laya on 6GB) by config only.

## 7. Startup

`brain/model_loader.py` does three things, each measured on this machine (RTX 3050):

| cost | before | after | how |
|---|---|---|---|
| hub re-check of a cached model | 22s | 0.2s | `snapshot_download(local_files_only=True)`, hand Laya a path; network only on first run |
| random init of ModernBERT-large before weights load | 7-13s | ~3s | `skip_random_init()` makes `torch.nn.init.*` no-ops during `laya.load`; outputs bit-identical |
| first CUDA pass | 0.9s | 0.3s | warmup predict inside the loader |

Unavoidable: ~1s torch import, ~1-2s CUDA init, ~2s moving weights to the GPU. The UI
starts a `Preloader` thread on open, so the first goal does not pay the load.
Reproduce with `uv run python -m laya_agent.cli "x" --dry-run --max-steps 1`, which prints
"model ready on cuda in Ns".

## 8. Debug views

- **All elements** button in the UI, or `tests/scripts/smoke.py --annotate --all`: every
  parsed element boxed and numbered. Grey parsed, orange shortlisted, green fine pass, red
  chosen.
- Step log line: window, element count, shortlist count, decider ("name match" or "laya"),
  top probability, confidence, P(done).

## 9. Tooling and process

- `uv` for everything Python. Never bare `pip` or `python` (they resolve to different
  interpreters on the dev machine).
- Git with one commit per layer or decision, each leaving the repo runnable.
- Tests use fakes at the layer boundaries (`tests/test_loop.py`, `tests/test_policy.py`).
  No test downloads the model or touches the screen. Live checks go through
  `tests/scripts/smoke.py`.

## 10. Decisions log

| date | decision | why |
|---|---|---|
| 2026-09-19 | UI Automation over a vision model for v1 | exact coordinates, no GPU, 200ms |
| 2026-09-19 | PySide6 desktop window, not a browser UI | can hide itself during capture and click |
| 2026-09-19 | Coarse-to-fine Laya passes | 11+ option calibration is near-argmax |
| 2026-09-19 | Margin confidence instead of Laya's entropy confidence | entropy collapses with many options |
| 2026-09-20 | Lexical decider before Laya, aliases learned from picks | Laya fails literal matching; user corrections should stick |
| 2026-09-20 | BFS walk with per-page budget | Chrome page trees blew the parse deadline and hid the tab strip |
| 2026-09-20 | Offline-first model resolve, background preload | 22s hub re-check on every start |
| 2026-09-20 | Skip torch random init during `laya.load` | 7-13s spent initialising weights the checkpoint overwrites |

## 11. Not done yet

- OmniParser vision parser for canvas-only UIs (`perception/omniparser.py` stub).
- `HFTextGenerator` for freeform text (`brain/text_gen.py` stub).
- Faster UIA parse via bulk `FindAll` with cached properties.
- Multi-monitor (`monitor_index` exists, untested beyond the primary).
