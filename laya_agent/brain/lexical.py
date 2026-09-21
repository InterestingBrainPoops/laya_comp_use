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
    "please", "now", "into", "onto", "with", "for", "at", "over", "back", "up", "down", "app",
    "navigate", "show", "find", "focus", "bring", "activate", "launch", "start", "use", "get",
    "make", "set", "put", "move", "see", "look", "check", "hit", "tap", "choose", "pick",
    "want", "need", "i", "you", "we", "can", "could", "would", "is", "are", "be", "do",
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


SELECTED_WORDS = {"current", "active", "selected", "focused"}  # extra names for selected elements
PARTIAL_CAP = 0.7  # any goal word unmatched: strong shortlist candidate, never an explicit decision
FUZZY_CAP = 0.85  # a containment/difflib match ("play" in "Player") never makes an explicit decision
WINDOW_ONLY = 0.4  # element whose only matches are its owning window's name ("spotify" -> Play button)
WINDOW_BY_TITLE = 0.7  # a Window matched through its title, not its app name ("new tab" -> Chrome window)


def score_element(goal_tokens: list[str], e: UIElement, extra_names: list[str] = ()) -> float:
    """[0, 1]. Explicit (>= 0.9) only when every goal word matches.

    - Elements: match the name (plus aliases). Words matching only the owning window's app
      name also count, but an element matched *only* that way is scaled by WINDOW_ONLY, so
      "play in spotify" -> Play button, "spotify" -> Spotify window.
    - Windows: match the app name and the title. A window matched only through its title is
      scaled by WINDOW_BY_TITLE, so "open a new tab in chrome" -> the New Tab button, not the
      window titled 'New Tab - Google Chrome'; "switch to chrome" -> the window.
    """
    if not goal_tokens:
        return 0.0
    app_toks = set(tokens(e.window, keep_stop=True)) if e.window else set()
    if e.is_window:
        # learned aliases ("browser" -> app:Chrome) count at app strength, the title does not
        name_toks = app_toks | {t for n in extra_names for t in tokens(n, keep_stop=True)}
        alt_toks = set(tokens(e.name, keep_stop=True)) - name_toks
        alt_scale = WINDOW_BY_TITLE
    else:
        name_toks = {t for n in [e.name, *extra_names] for t in tokens(n, keep_stop=True)}
        if e.selected:
            name_toks |= SELECTED_WORDS  # "close the current tab" -> the active tab's close button
        alt_toks = app_toks - name_toks
        alt_scale = WINDOW_ONLY
    if not name_toks and not alt_toks:
        return 0.0
    best: list[float] = []
    via_alt: list[bool] = []
    for g in goal_tokens:
        s = max((token_similarity(g, n) for n in name_toks), default=0.0)
        alt = s == 0.0 and bool(alt_toks)
        if alt:
            s = max(token_similarity(g, n) for n in alt_toks)
        best.append(s)
        via_alt.append(alt and s > 0)
    matched = [b for b in best if b > 0]
    if not matched:
        return 0.0
    coverage = len(matched) / len(goal_tokens)
    strength = max(best)
    distinctive = any(b >= 0.85 and len(g) >= 6 for g, b in zip(goal_tokens, best))
    score = 0.4 * strength + 0.6 * coverage + (0.2 if distinctive else 0.0)
    if coverage < 1.0:
        score = min(score, PARTIAL_CAP)
    if any(0 < b < 1.0 for b in best):
        # "play the music" vs tab 'Spotify - Web Player: Music ...' matched play~Player and
        # music=Music and was declared explicit over the real Play button (measured).
        score = min(score, FUZZY_CAP)
    if all(a for a, b in zip(via_alt, best) if b > 0):
        score *= alt_scale
    elif e.is_window and any(via_alt):
        # "open a new tab in chrome": the Chrome window titled 'New Tab - ...' matches 'new'
        # only through its title. Title words never make a window an explicit choice.
        score = min(score, PARTIAL_CAP)
    return round(min(1.0, score), 3)


KIND_WORDS_IN_GOAL = {
    "tab": "TabItem", "tabs": "TabItem", "link": "Hyperlink", "button": "Button", "menu": "MenuItem",
    "field": "Edit", "box": "Edit", "input": "Edit", "checkbox": "CheckBox", "dropdown": "ComboBox",
    "window": "Window", "app": "Window",
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
        extra = list(aliases.get(e.name, []))
        if e.is_window and e.window:
            extra += aliases.get(f"app:{e.window}", [])
        s = score_element(gt, e, extra)
        # "open a new tab" -> button 'New Tab': the kind word is part of the name, no penalty.
        name_has_kind_word = bool(kind_words & set(tokens(e.name, keep_stop=True)))
        if kinds and e.kind not in kinds and not name_has_kind_word:
            s = round(s * KIND_MISMATCH, 3)
        out[e.id] = s
    return out
