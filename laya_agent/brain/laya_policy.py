"""Decision layer: Snapshot + goal -> Decision.

Two deciders, in order:
  1. lexical (brain/lexical.py): the goal names an element explicitly ("the NandhaKishorM tab").
     Deterministic, explainable, learns aliases from the user's picks.
  2. Laya: the vague cases. Coarse-to-fine so 11+ options stay calibrated:
       pass A  every element in chunks of `coarse_chunk`, keep `coarse_keep` per chunk
       pass B  one more cut if the merged shortlist exceeds FINE_K
       pass C  FINE_K elements + 2 meta actions = 10 options, Laya's calibrated bucket
     Laya's 11+ option temperature is 0.1 (near-argmax) so the gate only trusts pass C.

Laya packs `[CLS] instructions [SEP] options... [SEP] state [SEP]` into 512 tokens, so each
option label carries the element description and the state stays small.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from laya_agent.brain.aliases import AliasStore
from laya_agent.brain.lexical import kinds_in_goal, lexical_scores, tokens
from laya_agent.config import Config
from laya_agent.models import ACTION_DONE, ACTION_NEED_TEXT, ACTION_SCROLL_DOWN, META_ACTIONS, Decision, Snapshot, UIElement

PredictFn = Callable[[Any, dict[str, Any]], dict[str, Any]]

FINE_META = {ACTION_DONE: META_ACTIONS[ACTION_DONE], ACTION_SCROLL_DOWN: META_ACTIONS[ACTION_SCROLL_DOWN]}


class LayaPolicy:
    FINE_K = 8  # 8 elements + 2 meta actions = 10 options
    LEXICAL_DECIDE = 0.9  # best lexical score at or above this decides without Laya
    LEXICAL_MARGIN = 0.25  # ... provided the runner-up is this far behind

    def __init__(self, cfg: Config, predict: PredictFn | None = None, aliases: AliasStore | None = None) -> None:
        """predict: injectable for tests. Default loads the Laya checkpoint lazily."""
        self.cfg = cfg
        self._predict = predict
        self._agent = None
        self.aliases = aliases if aliases is not None else AliasStore(cfg.alias_path)

    # -- public ---------------------------------------------------------------
    def decide(self, goal: str, snap: Snapshot, history: list[str]) -> Decision:
        elements = self.rank_elements(goal, snap.elements)[: self.cfg.max_elements]
        state = self.build_state(goal, snap, history)
        lex = lexical_scores(goal, elements, self.aliases.as_dict())
        by_lex = sorted(elements, key=lambda e: lex[e.id], reverse=True)

        explicit = self._explicit_match(by_lex, lex)
        shortlist = self.shortlist(goal, state, elements, by_lex, lex)
        fine = sorted(shortlist, key=lambda e: e.id)
        raw = self.predict(state, self._questions(fine, goal))
        ranked = self._ranked(raw)
        ans = raw["answers"]
        need_text = float(ans.get("need_text", {}).get("noul", 0.0))
        done_prob = float(ans["done"]["noul"])

        if explicit is not None:
            # Lexical decides the element; Laya still informs done/need_text.
            action = f"click_{explicit.id}"
            top = [(action, lex[explicit.id])] + [(a, p) for a, p in ranked if a != action][:4]
            confidence = max(lex[explicit.id], 0.9)
            decided_by = "name match"
        else:
            action = ranked[0][0]
            top = ranked[:5]
            confidence = self.margin_confidence(ranked, float(ans["action"]["confidence"]))
            decided_by = "laya"
        if need_text >= 0.8 and explicit is None:
            action, confidence, decided_by = ACTION_NEED_TEXT, need_text, "laya"

        raw = dict(raw)
        raw["_elements"] = {f"click_{e.id}": e for e in elements}
        raw["_shortlist"] = [e.id for e in shortlist]
        raw["_fine"] = [e.id for e in fine]
        raw["_lexical"] = {e.id: lex[e.id] for e in by_lex[:5] if lex[e.id] > 0}
        raw["_decided_by"] = decided_by
        return Decision(
            action=action,
            element=self._element_for(action, elements),
            confidence=round(confidence, 4),
            done_prob=done_prob,
            top_k=[(self.describe(label, elements), p) for label, p in top],
            top_actions=[label for label, _ in top],
            raw=raw,
        )

    def remember(self, goal: str, element: UIElement) -> None:
        """The user picked `element` for `goal`: learn the alias."""
        self.aliases.remember(goal, element.name)

    def ask(self, question: str, snap: Snapshot, goal: str = "", history: list[str] | None = None) -> float:
        """Freeform yes/no question about the current screen. Returns P(true)."""
        state = self.build_state(goal, snap, history or [], include_elements=True)
        raw = self.predict(state, {"q": {"type": "noul", "instructions": question}})
        return float(raw["answers"]["q"]["noul"])

    # -- deciders -----------------------------------------------------------------
    def _explicit_match(self, by_lex: list[UIElement], lex: dict[int, float]) -> UIElement | None:
        if not by_lex:
            return None
        best = lex[by_lex[0].id]
        if best < self.LEXICAL_DECIDE:
            return None
        # Runner-up with the same name (two 'New Tab' buttons) is a duplicate, not ambiguity:
        # take the first in tree order, which is the shallower, more prominent one.
        rivals = [e for e in by_lex[1:] if e.name.lower() != by_lex[0].name.lower()]
        second = lex[rivals[0].id] if rivals else 0.0
        if best - second >= self.LEXICAL_MARGIN:
            return by_lex[0]
        return None

    def shortlist(self, goal: str, state: dict, elements: list[UIElement], by_lex: list[UIElement], lex: dict[int, float]) -> list[UIElement]:
        """Cut to FINE_K. Guaranteed: lexical hits and kind matches. Rest: Laya over chunks."""
        if len(elements) <= self.FINE_K:
            return list(elements)
        kinds = self.kinds_in_goal(goal)
        kept: dict[int, UIElement] = {}
        for e in by_lex:
            if lex[e.id] >= 0.5 and len(kept) < self.FINE_K // 2:
                kept[e.id] = e
        for e in elements:
            if e.kind in kinds and len(kept) < self.FINE_K - 2:
                kept.setdefault(e.id, e)
        chunk = max(self.FINE_K, self.cfg.coarse_chunk)
        for start in range(0, len(elements), chunk):
            part = elements[start : start + chunk]
            for e in self._top_clicks(state, goal, part, self.cfg.coarse_keep):
                kept.setdefault(e.id, e)
        merged = [e for e in elements if e.id in kept]
        guaranteed = [e for e in merged if lex[e.id] >= 0.5 or e.kind in kinds][: self.FINE_K - 2]
        while len(merged) > self.FINE_K:
            rest = [e for e in merged if e not in guaranteed]
            cut = self._top_clicks(state, goal, rest[:chunk], self.FINE_K - len(guaranteed))
            merged = guaranteed + cut + rest[chunk:]
        return merged

    def _top_clicks(self, state: dict, goal: str, elements: list[UIElement], k: int) -> list[UIElement]:
        raw = self.predict(state, self._questions(elements, goal))
        by_id = {e.id: e for e in elements}
        out = []
        for action, _ in self._ranked(raw):
            if action.startswith("click_"):
                out.append(by_id[int(action.split("_", 1)[1])])
                if len(out) == k:
                    break
        return out

    # -- building blocks ------------------------------------------------------
    def rank_elements(self, goal: str, elements: list[UIElement]) -> list[UIElement]:
        """Stable order: kind and word matches first, then screen order. Only affects which
        elements survive `max_elements` and how chunks are formed."""
        g = set(tokens(goal))
        kinds = self.kinds_in_goal(goal)

        def score(e: UIElement) -> float:
            overlap = len(g & set(tokens(e.name)))
            return (1.5 if e.kind in kinds else 0.0) + overlap * 1.0 - 0.0001 * e.id

        return sorted(elements, key=score, reverse=True)

    kinds_in_goal = staticmethod(kinds_in_goal)

    def build_state(self, goal: str, snap: Snapshot, history: list[str], include_elements: bool = False) -> dict:
        state: dict[str, Any] = {
            "goal": goal,
            "window": snap.window_title[:80],
            "previous_actions": history[-self.cfg.history_len:],
        }
        if include_elements:
            state["visible"] = [e.label() for e in snap.elements[:40]]
        return state

    @staticmethod
    def _questions(elements: list[UIElement], goal: str) -> dict[str, Any]:
        criteria = {f"click_{e.id}": e.label() for e in elements}
        criteria.update(FINE_META)
        return {
            "action": {
                "type": "choice",
                "instructions": f'The user asked: "{goal}". Which screen element is the user asking to '
                "click next? Choose 'done' only if the request is already satisfied.",
                "criteria": criteria,
            },
            "done": {
                "type": "noul",
                "instructions": f'Is the request "{goal}" already fully satisfied on the current screen?',
            },
            "need_text": {
                "type": "noul",
                "instructions": f'Does the next step for "{goal}" require typing text rather than clicking?',
            },
        }

    @staticmethod
    def _ranked(raw: dict[str, Any]) -> list[tuple[str, float]]:
        probs: dict[str, float] = raw["answers"]["action"]["probabilities"]
        return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

    @staticmethod
    def margin_confidence(ranked: list[tuple[str, float]], laya_conf: float) -> float:
        """Entropy confidence collapses with many options; use the top-1 margin instead."""
        if not ranked:
            return 0.0
        top = ranked[0][1]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        return round(max(min(top, 1.0), min(1.0, (top - second) * 2 + top * 0.5), laya_conf), 4)

    @staticmethod
    def _element_for(action: str, candidates: list[UIElement]) -> UIElement | None:
        if not action.startswith("click_"):
            return None
        eid = int(action.split("_", 1)[1])
        return next((e for e in candidates if e.id == eid), None)

    def describe(self, action: str, candidates: list[UIElement]) -> str:
        e = self._element_for(action, candidates)
        if e is not None:
            return f"click {e.label()}"
        return action.replace("_", " ")

    # -- model ------------------------------------------------------------------
    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        if self._predict is None:
            self._predict = self._load()
        return self._predict(state, questions)

    def _load(self) -> PredictFn:
        import laya

        agent = laya.load(self.cfg.model_id, device=self.cfg.device)
        agent.cfg["head_max_len"] = self.cfg.head_max_len
        self._agent = agent
        return agent.predict
