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

## Why coarse-to-fine

Laya's calibration temperature for 11+ options is 0.1, which turns every answer into
near-certainty. The policy first shortlists among up to 20 elements, then re-scores the
top 6 plus the 4 meta actions (scroll up/down, done, need_text) in the 6-10 bucket where
probabilities are calibrated. The low-confidence gate reads the second pass.

## v2 hooks

- `config.text_generator = "hf"` activates `HFTextGenerator` (planned Qwen/Qwen3-1.7B, bf16,
  fits beside Laya on a 6GB GPU) for typing URLs, notes, and other freeform text.
- `config.screen_parser = "omniparser"` activates the vision parser for canvas-only UIs.
