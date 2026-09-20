"""Deterministic name matching between a goal and screen elements.

Laya is a semantic triage model; it is weak at literal string matching (it cannot tell
'NandhaKishorM' names the tab 'NandhaKishorM/laya'). So explicit references are resolved
here, and Laya only arbitrates the vague cases.
"""
from __future__ import annotations

import difflib
import re

from laya_agent.models import UIElement

_WORD = re.compile(r"[a-z0-9]+")
STOP = {
    "the", "a", "an", "to", "on", "in", "of", "and", "click", "open", "go", "then", "press", "select",
    "switch", "my", "me", "it", "that", "this", "tab", "tabs", "button", "link", "menu", "window",
    "please", "now", "into", "onto", "with", "for", "at", "over", "back", "up", "down",
}


def tokens(s: str, keep_stop: bool = False) -> list[str]:
    ws = _WORD.findall(s.lower())
    return ws if keep_stop else [w for w in ws if w not in STOP]


def token_similarity(a: str, b: str) -> float:
    """1.0 exact, 0.85 one contains the other (len >= 4), else difflib ratio if >= 0.8."""
    if a == b:
        return 1.0
    if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
        return 0.85
    if min(len(a), len(b)) >= 4:
        r = difflib.SequenceMatcher(None, a, b).ratio()
        if r >= 0.8:
            return r * 0.9
    return 0.0


def score_element(goal_tokens: list[str], e: UIElement, extra_names: list[str] = ()) -> float:
    """Fraction of goal tokens that find a match in the element's name (or alias names),
    weighted so a single strong distinctive match still scores high."""
    if not goal_tokens:
        return 0.0
    names = [e.name, *extra_names]
    name_toks = {t for n in names for t in tokens(n, keep_stop=True)}
    if not name_toks:
        return 0.0
    best = []
    for g in goal_tokens:
        best.append(max((token_similarity(g, n) for n in name_toks), default=0.0))
    matched = [b for b in best if b > 0]
    if not matched:
        return 0.0
    coverage = len(matched) / len(goal_tokens)
    strength = max(best)
    # a long distinctive token (>= 6 chars) matching exactly is decisive on its own
    distinctive = any(b >= 0.85 and len(g) >= 6 for g, b in zip(goal_tokens, best))
    return round(min(1.0, 0.5 * strength + 0.5 * coverage + (0.3 if distinctive else 0.0)), 3)


KIND_WORDS_IN_GOAL = {
    "tab": "TabItem", "tabs": "TabItem", "link": "Hyperlink", "button": "Button", "menu": "MenuItem",
    "field": "Edit", "box": "Edit", "input": "Edit", "checkbox": "CheckBox", "dropdown": "ComboBox",
}
KIND_MISMATCH = 0.6  # "the github tab" must not lock onto a button named 'Install GitHub'


def kinds_in_goal(goal: str) -> set[str]:
    return {KIND_WORDS_IN_GOAL[w] for w in tokens(goal, keep_stop=True) if w in KIND_WORDS_IN_GOAL}


def lexical_scores(goal: str, elements: list[UIElement], aliases: dict[str, list[str]] | None = None) -> dict[int, float]:
    """element id -> [0, 1]. `aliases` maps element name -> extra names learned from the user.
    If the goal names a control kind, elements of other kinds are penalised."""
    gt = tokens(goal)
    aliases = aliases or {}
    kinds = kinds_in_goal(goal)
    kind_words = {w for w in tokens(goal, keep_stop=True) if w in KIND_WORDS_IN_GOAL}
    out = {}
    for e in elements:
        s = score_element(gt, e, aliases.get(e.name, []))
        # "open a new tab" -> button 'New Tab': the kind word is part of the name, no penalty.
        name_has_kind_word = bool(kind_words & set(tokens(e.name, keep_stop=True)))
        if kinds and e.kind not in kinds and not name_has_kind_word:
            s = round(s * KIND_MISMATCH, 3)
        out[e.id] = s
    return out
