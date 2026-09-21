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
- **All visible windows, not just the foreground one.** Top-level windows are walked in
  z-order, front first. The front window gets `uia_max_nodes`; each background window gets
  `uia_background_budget` (enough for tabs, toolbars, menus, not page content). Every
  window is also emitted as a `Window` element so "switch to spotify" is a click target;
  minimized windows appear only as `Window` elements. `config.target_window` narrows to
  one window by title substring.
- **App name from the process, not the title.** Spotify's title is the playing song and a
  browser's title is the page. `_app_name` reads the owning executable (`chrome.exe` ->
  Chrome) and it is carried on every element as `window`, shown in labels as
  "in window Spotify" / "of app Spotify".
- **Selection propagates.** Children of the selected tab inherit `selected`, so the current
  tab's close button is "button 'Close Tab' (current)" and "close the current tab" cannot
  hit the first tab's button.
- Our own window is excluded by handle and minimized during capture and click, restored
  with `SW_SHOWNOACTIVATE` so focus stays on the target app.

Measured (2026-09-20, after the timing tab exposed it): the "~4s parse" blamed on Chrome
page content was mostly window enumeration. Walking the UIA desktop root's children cost
~4.0s per parse; the element walk itself was 30ms. Enumeration now uses Win32
`EnumWindows` and wraps only the survivors with `ControlFromHandle`.

| parse | before | after |
|---|---|---|
| Terminal only (`--window Terminal`, 42 nodes) | 4200ms | 190ms |
| all 12 visible windows (596 nodes) | ~4400ms | 580-680ms, of which screenshot 110-150ms |

Remaining cost is ~0.8ms per UIA node; bulk `FindAll` with a cache request is the next
step if it matters.

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

### 5b. Lexical scoring across windows

- Elements match on their name (+ aliases). A word matching only the owning app's name
  still counts, but an element matched *only* that way is scaled by 0.4, so "spotify"
  picks the Spotify window while "play in spotify" picks the Play button inside it.
- Windows match on their app name; title words never make a window explicit (cap 0.7), so
  "open a new tab in chrome" picks the New Tab button, not the window titled
  "New Tab - Google Chrome".
- Any unmatched goal word caps the score at 0.7: a strong shortlist candidate, never an
  explicit decision. Explicit means every goal word matched.
- Selected elements answer to "current", "active", "selected", "focused". Ties between
  same-named elements prefer the selected one, then the front-most.

## 5c. Click targeting (`actuation/targeting.py`)

The centre of a bounding box is wrong for composite controls: Windows Terminal's New Tab
is a SplitButton whose centre (x=868) lands on the divider between its primary part
(ends at 867) and the dropdown arrow. Rules, all pure geometry with tests:

1. Descend into the main child: the largest contained child covering ≥ 40% of the target
   (PrimaryButton inside the SplitButton). Up to 3 levels.
2. Obstacles are overlapping elements that neither contain the target nor are its main
   part (a tab's Close button). Pick the grid point inside the target, inset 4px from its
   edges, with the largest clearance from every obstacle; ties go to the point nearest the
   centre.
3. Elements in a background window get their window activated first (`alt` press then
   `SetForegroundWindow`, restore if minimized), so the click lands on them and not on
   whatever covered them.

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
| first real decision after warmup | 513ms | 170ms | warm up with the real shapes: a 20-option chunk pass and a 10-option fine pass, each with two noul questions |

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
- **Session log** (`session_log.py`, `config.log_dir`, default `logs/`). Per session:
  `<stamp>.jsonl` with one event per line (`goal`, `parsed` with every element label,
  `decided` with lexical scores, shortlist, fine pass, the full Laya probability table and
  the decider, `human` with the options shown and the reply, `acted` with the click point,
  `step` with merged timings, `finished`, `error`), a readable `<stamp>.log` mirror, and
  with `log_screenshots` a folder of `step_N.png` and `step_N_annotated.png`. The loop is
  the only writer; layers never log.
- **Timing tab** in the UI, and a timing line per step in the CLI. Every layer reports
  into `timings` on its own dataclass (`Snapshot.timings`: screenshot, UIA walk, node
  count; `Decision.timings`: lexical, Laya pass count and ms; loop adds human wait, text
  generation, act, settle) and the loop merges them into `StepResult.timings`. The tab
  shows one row per step plus averages and the model load time. Text generation is the
  v2 column and stays empty until `HFTextGenerator` exists.

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
| 2026-09-20 | Parse all visible windows in z-order, offer windows as targets | user works across apps; foreground-only could not "switch to spotify" |
| 2026-09-20 | App name from the process executable | Spotify's and browsers' titles do not contain the app name |
| 2026-09-20 | Click targeting: main child + obstacle clearance | New Tab split button's centre hit the divider; tab centres can graze close buttons |
| 2026-09-20 | Selection inherited by a tab's children; selected wins same-name ties | "close the current tab" must never close another tab |
| 2026-09-20 | Timings live on the dataclasses; loop merges; JSONL session log written only by the loop | one source of numbers for the UI tab, CLI, and log; layers stay side-effect free |
| 2026-09-20 | Win32 `EnumWindows` for window enumeration | the timing tab showed 4.0s of every parse was the UIA desktop-root walk |
| 2026-09-20 | Warm up with real question shapes | first decision paid 500ms of kernel setup |

## 11. Not done yet

- OmniParser vision parser for canvas-only UIs (`perception/omniparser.py` stub).
- `HFTextGenerator` for freeform text (`brain/text_gen.py` stub).
- Faster UIA walk via bulk `FindAll` with cached properties (only if ~0.8ms/node matters).
- Multi-monitor (`monitor_index` exists, untested beyond the primary).
