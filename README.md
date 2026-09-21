# Laya screen agent

Type a goal in a chat window. The agent reads the on-screen buttons through Windows UI
Automation, asks [Laya](https://huggingface.co/convaiinnovations/laya) which one moves
toward the goal, clicks it, and repeats. When Laya is unsure it pauses and asks you.

Laya is a 421M text-only decision model (ModernBERT backbone). It never sees pixels and
never generates text. It picks one option out of a typed list with calibrated probabilities
in one ~35ms forward pass. That is why the architecture below has a perception layer that
turns the screen into text, and a separate (v2) slot for a small LLM when freeform text is
needed.

## Run

```
uv sync
uv run python -m laya_agent.ui                  # chat window
uv run python -m laya_agent.cli "open the Edit menu"   # headless, 3s countdown to focus the app
uv run python tests/scripts/smoke.py --decide "open a new tab"   # one decision, no click
uv run pytest
```

First run downloads the ~850MB checkpoint to the Hugging Face cache. Needs an NVIDIA GPU
for ~35ms decisions; CPU works at ~300ms.

Chat commands:

- `open the File menu` runs a goal.
- `? is a dialog open` asks a yes/no question about the current screen, answered with P(yes).
- When asked, reply with an option number, a free-text hint, or `stop`.
- Mouse to the top-left corner aborts any click (pyautogui failsafe).

## Layers

```
laya_agent/
  models.py            dataclasses every layer shares (UIElement, Snapshot, Decision, ...)
  config.py            all knobs; swapping a layer is a value here
  perception/          ScreenParser protocol. uia.py (v1), omniparser.py (v1.5 stub)
  brain/laya_policy.py Snapshot + goal -> Decision. Coarse-to-fine so 11+ options stay calibrated
  brain/text_gen.py    TextGenerator protocol. NullTextGenerator (v1, asks you), HFTextGenerator (v2 stub)
  actuation/           pyautogui executor, own-window hiding
  agent/loop.py        perceive -> decide -> gate -> act. Framework-free, LoopEvents protocol
  cli.py               console adapter
  ui/                  PySide6 adapter
```

Rules: layers talk only through `models.py` and the two protocols. No layer imports another
layer's internals.

## How a decision is made

1. **Name match** (`brain/lexical.py`). If your goal names an element ("the NandhaKishorM
   tab", "open a new tab"), a deterministic token matcher decides. Laya is a triage model
   and cannot do literal string matching reliably, so explicit references never reach it.
   Saying a kind ("tab", "button", "link") penalises other kinds.
2. **Laya** for the vague cases. Shortlisting runs Laya over chunks of 20 elements, keeping
   3 per chunk plus every name/kind hit, then a final pass over 8 elements + done/scroll in
   the 10-option bucket where Laya's probabilities are calibrated (its 11+ option temperature
   is 0.1, near-argmax, so only the final pass feeds the confidence gate).
3. **Ask you** when neither is confident. Your pick is saved to `.laya_aliases.json`, so
   "github tab" maps to that tab's title next time and step 1 resolves it.

## Seeing what the agent sees

- Chat window: **All elements** button parses the target window now and shows every
  element boxed and numbered (grey = parsed, orange = shortlisted, green = final pass,
  red = chosen) with the full list in the chat.
- CLI: `uv run python tests/scripts/smoke.py --window chrome --decide "..." --annotate --all`
  writes `tests/out/annotated.png` and prints every element, marking `*` shortlisted and
  `**` final-pass ones.
- Each step's log line says how many elements were parsed and shortlisted and whether the
  choice came from the name match or Laya.
- Every session writes `logs/<stamp>.jsonl` (every parse, decision with the full
  probability table, question asked, reply given, click point, timings) plus a readable
  `logs/<stamp>.log` and per-step screenshots. Set `log_dir = None` in `config.py` to turn
  it off.
- The **Timing** tab shows per-step milliseconds for screenshot, UI Automation walk,
  name matching, Laya (pass count and time), waiting on you, text generation (v2), and
  the click, with averages and the model load time.

## v2 hooks

- `config.text_generator = "hf"` activates `HFTextGenerator` (planned Qwen/Qwen3-1.7B, bf16,
  fits beside Laya on a 6GB GPU) for typing URLs, notes, and other freeform text.
- `config.screen_parser = "omniparser"` activates the vision parser for canvas-only UIs.
