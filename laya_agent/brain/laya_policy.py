"""Decision layer: Snapshot + goal -> Decision, via Laya typed questions.

Laya packs `[CLS] instructions [SEP] options... [SEP] state [SEP]` into 512 tokens.
Options get `head_max_len` tokens, the state gets the rest. So each option label
carries the element description and the state stays small (goal, window, history).
"""
from __future__ import annotations

import re
from typing import Any, Callable

from laya_agent.config import Config
from laya_agent.models import ACTION_DONE, META_ACTIONS, Decision, Snapshot, UIElement

PredictFn = Callable[[Any, dict[str, Any]], dict[str, Any]]

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "to", "on", "in", "of", "and", "click", "open", "go", "then", "press", "select"}
_KIND_BONUS = {"Button": 0.5, "Hyperlink": 0.5, "MenuItem": 0.5, "TabItem": 0.4, "Edit": 0.3}


def _tokens(s: str) -> set[str]:
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP}


class LayaPolicy:
    def __init__(self, cfg: Config, predict: PredictFn | None = None) -> None:
        """predict: injectable for tests. Default loads the Laya checkpoint lazily."""
        self.cfg = cfg
        self._predict = predict
        self._agent = None

    # -- public ---------------------------------------------------------------
    def decide(self, goal: str, snap: Snapshot, history: list[str]) -> Decision:
        """Coarse-to-fine. Laya's calibration for 11+ options is near-argmax (temperature
        0.1), which would make every decision look certain. So: pass 1 shortlists among up
        to max_elements, pass 2 re-scores the top FINE_K elements plus meta actions in the
        calibrated 6-10 option bucket. The gate reads pass 2."""
        candidates = self.rank_elements(goal, snap.elements)[: self.cfg.max_elements]
        state = self.build_state(goal, snap, history)
        raw = self.predict(state, self._questions(candidates))
        ranked = self._ranked(raw)
        if len(candidates) > self.FINE_K:
            keep_ids = {int(a.split("_", 1)[1]) for a, _ in ranked if a.startswith("click_")}
            fine = [e for e in candidates if e.id in keep_ids][: self.FINE_K]
            fine.sort(key=lambda e: e.id)
            raw = self.predict(state, self._questions(fine))
            ranked = self._ranked(raw)
        ans = raw["answers"]
        action = ranked[0][0]
        element = self._element_for(action, candidates)
        top_k = [(self.describe(label, candidates), p) for label, p in ranked[:5]]
        return Decision(
            action=action,
            element=element,
            confidence=self.margin_confidence(ranked, float(ans["action"]["confidence"])),
            done_prob=float(ans["done"]["noul"]),
            top_k=top_k,
            raw=raw,
        )

    FINE_K = 6  # 6 elements + 4 meta actions = 10 options, Laya's calibrated bucket

    @staticmethod
    def _questions(elements: list[UIElement]) -> dict[str, Any]:
        criteria = {f"click_{e.id}": e.label() for e in elements}
        criteria.update(META_ACTIONS)
        return {
            "action": {
                "type": "choice",
                "instructions": "You control a Windows desktop with the mouse. Given the goal and the "
                "current screen, which single action moves closest to completing the goal?",
                "criteria": criteria,
            },
            "done": {
                "type": "noul",
                "instructions": "Is the goal already fully accomplished on the current screen?",
            },
        }

    @staticmethod
    def _ranked(raw: dict[str, Any]) -> list[tuple[str, float]]:
        probs: dict[str, float] = raw["answers"]["action"]["probabilities"]
        return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

    def ask(self, question: str, snap: Snapshot, goal: str = "", history: list[str] | None = None) -> float:
        """Freeform yes/no question about the current screen. Returns P(true)."""
        state = self.build_state(goal, snap, history or [], include_elements=True)
        raw = self.predict(state, {"q": {"type": "noul", "instructions": question}})
        return float(raw["answers"]["q"]["noul"])

    # -- building blocks ------------------------------------------------------
    def rank_elements(self, goal: str, elements: list[UIElement]) -> list[UIElement]:
        g = _tokens(goal)

        def score(e: UIElement) -> float:
            t = _tokens(e.name) | _tokens(e.path)
            overlap = len(g & t)
            return overlap * 2.0 + _KIND_BONUS.get(e.kind, 0.0) - 0.001 * e.id

        return sorted(elements, key=score, reverse=True)

    def build_state(self, goal: str, snap: Snapshot, history: list[str], include_elements: bool = False) -> dict:
        state: dict[str, Any] = {
            "goal": goal,
            "window": snap.window_title[:80],
            "previous_actions": history[-self.cfg.history_len:],
        }
        if include_elements:
            state["visible"] = [e.label() for e in snap.elements[: self.cfg.max_elements]]
        return state

    @staticmethod
    def margin_confidence(ranked: list[tuple[str, float]], laya_conf: float) -> float:
        """Entropy confidence collapses with 20+ options; use the top-1 margin instead."""
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
