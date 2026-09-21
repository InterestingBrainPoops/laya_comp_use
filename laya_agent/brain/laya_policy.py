"""Decision layer: Snapshot + goal -> Decision.

Pipeline (measured in tests/eval/run.py; numbers in docs/ARCHITECTURE.md):
  1. lexical (brain/lexical.py): the goal names an element explicitly and unambiguously
     ("the NandhaKishorM tab"). Deterministic, learns aliases from the user's picks.
  2. shortlist: cut every element on screen to FINE_K candidates (see `shortlist`).
  3. Laya fine pass: FINE_K elements as the only options of one `choice` question, in the
     6-10 option bucket where Laya's probabilities are calibrated. "Is the goal done?" and
     "does the next step need typing?" are separate `noul` questions in the same forward
     pass, never options in the list: as options they stole up to half the probability mass
     (narrow semantic top-1 36% with them, 91% without).
  4. gate (agent/loop.py): low margin -> ask the human; the offered list carries a
     "scroll down" option so scrolling is a human choice, not a competing action.

Laya packs `[CLS] instructions [SEP] options... [SEP] state [SEP]` into 512 tokens, so each
option label carries the element description and the state stays small.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

from laya_agent.brain.aliases import AliasStore
from laya_agent.brain.lexical import kinds_in_goal, lexical_scores, tokens
from laya_agent.brain.retriever import Retriever, make_retriever
from laya_agent.config import Config
from laya_agent.models import ACTION_NEED_TEXT, ACTION_SCROLL_DOWN, Decision, Snapshot, UIElement

PredictFn = Callable[[Any, dict[str, Any]], dict[str, Any]]

INSTRUCTIONS = (
    "You control a Windows desktop with the mouse. Given the goal and the current screen, "
    "which single action moves closest to completing the goal?"
)


class LayaPolicy:
    LEXICAL_DECIDE = 0.9  # best lexical score at or above this decides without Laya
    LEXICAL_MARGIN = 0.25  # ... provided the runner-up is this far behind

    @property
    def FINE_K(self) -> int:  # options in the final pass: Laya's calibrated 6-10 bucket
        return self.cfg.fine_k

    def __init__(
        self,
        cfg: Config,
        predict: PredictFn | None = None,
        aliases: AliasStore | None = None,
        retriever: Retriever | None | str = "auto",
    ) -> None:
        """predict / retriever: injectable for tests. Defaults load lazily per config."""
        self.cfg = cfg
        self._predict = predict
        self._load_lock = threading.Lock()
        self._passes, self._laya_ms = 0, 0.0
        self._retrieved: list[int] = []
        self._sims: dict[int, float] = {}
        self._goal_tokens: list[str] = []
        self._retrieve_ms = 0.0
        self.load_ms: float | None = None
        self.aliases = aliases if aliases is not None else AliasStore(cfg.alias_path)
        self.retriever: Retriever | None = make_retriever(cfg) if retriever == "auto" else retriever

    # -- public ---------------------------------------------------------------
    def decide(self, goal: str, snap: Snapshot, history: list[str], hint: str | None = None) -> Decision:
        """`hint`: the user's free-text reply when asked ("open the laya_comp_use tab"). It
        names the target more precisely than the goal, so it drives the name match and the
        retrieval query; the goal still frames Laya's question. Measured: a session where
        the hint named the tab exactly and was ignored because it only reached Laya's state."""
        t0 = time.perf_counter()
        self._passes, self._laya_ms = 0, 0.0
        self._retrieved, self._sims, self._retrieve_ms = [], {}, 0.0
        match_text = hint or goal
        query = f"{hint}. {goal}" if hint else goal
        elements = self.rank_elements(match_text, snap.elements)[: self.cfg.max_elements]
        state = self.build_state(goal, snap, history, hint=hint)
        lex = lexical_scores(match_text, elements, self.aliases.as_dict())
        self._goal_tokens = tokens(match_text)
        by_lex = sorted(elements, key=lambda e: (lex[e.id], *self._tie_key(e)), reverse=True)
        explicit = self._explicit_match(by_lex, lex)
        lexical_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        shortlist = self.shortlist(query, state, elements, by_lex, lex, kinds=kinds_in_goal(match_text) | kinds_in_goal(goal))
        shortlist_ms = (time.perf_counter() - t1) * 1000
        fine = sorted(shortlist, key=lambda e: e.id)
        raw = self.predict(state, self._questions(fine, goal))
        ranked = self._fused(self._ranked(raw), fine)
        ans = raw["answers"]
        need_text = float(ans.get("need_text", {}).get("noul", 0.0))
        done_prob = float(ans["done"]["noul"])

        if explicit is not None:
            # Lexical decides the element; Laya still informs done/need_text.
            action = f"click_{explicit.id}"
            top = [(action, lex[explicit.id])] + [(a, p) for a, p in ranked if a != action][: self.FINE_K - 1]
            confidence = max(lex[explicit.id], 0.9)
            decided_by = "name match"
        else:
            action = ranked[0][0] if ranked else ACTION_SCROLL_DOWN
            top = ranked[: self.FINE_K]  # everything Laya weighed: the human sees all of it when asked
            confidence = self.margin_confidence(ranked, float(ans["action"]["confidence"])) if ranked else 0.0
            decided_by = "laya"
        if need_text >= 0.8 and explicit is None:
            action, confidence, decided_by = ACTION_NEED_TEXT, need_text, "laya"
        # Scrolling is offered to the human, never chosen over an element automatically.
        top = top + [(ACTION_SCROLL_DOWN, 0.0)]

        raw = dict(raw)
        raw["_elements"] = {f"click_{e.id}": e for e in elements}
        raw["_shortlist"] = self._retrieved or [e.id for e in shortlist]  # orange in the overlay
        raw["_fine"] = [e.id for e in fine]  # green
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
            timings={
                "lexical_ms": round(lexical_ms, 1),
                "shortlist_ms": round(shortlist_ms, 1),
                "retrieve_ms": round(self._retrieve_ms, 1),
                "laya_passes": self._passes,
                "laya_ms": round(self._laya_ms, 1),
                "decide_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        )

    def remember(self, goal: str, element: UIElement) -> None:
        """The user picked `element` for `goal`: learn the alias. Windows are keyed by app,
        their titles change with the page or song (measured: an alias stored against
        'New Tab - Google Chrome' could never match again)."""
        self.aliases.remember(goal, f"app:{element.window}" if element.is_window and element.window else element.name)

    def ask(self, question: str, snap: Snapshot, goal: str = "", history: list[str] | None = None) -> float:
        """Freeform yes/no question about the current screen. Returns P(true)."""
        state = self.build_state(goal, snap, history or [], include_elements=True)
        raw = self.predict(state, {"q": {"type": "noul", "instructions": question}})
        return float(raw["answers"]["q"]["noul"])

    @staticmethod
    def _tie_key(e: UIElement) -> tuple[bool, bool, bool, int]:
        """Same-name ties, most attractive first: the front window over background ones
        (Terminal's 'New Tab' button over Chrome's tab named 'New Tab'); an already-selected
        tab last, clicking it does nothing; a selected control otherwise first (the current
        tab's own 'Close Tab'); then tree order (shallower, more prominent)."""
        already_active = e.selected and e.kind in ("TabItem", "ListItem", "RadioButton")
        return (e.foreground, not already_active, e.selected, -e.id)

    # -- deciders -----------------------------------------------------------------
    def _explicit_match(self, by_lex: list[UIElement], lex: dict[int, float]) -> UIElement | None:
        if not by_lex:
            return None
        best = lex[by_lex[0].id]
        if best < self.LEXICAL_DECIDE:
            return None
        # Runner-up with the same name (two 'New Tab' buttons) or another window of the same
        # app (two Chrome windows) is a duplicate, not ambiguity: take the first in z/tree
        # order, which is the front-most or shallower one.
        top = by_lex[0]
        rivals = [
            e for e in by_lex[1:]
            if e.name.lower() != top.name.lower() and not (e.is_window and top.is_window and e.window == top.window)
        ]
        second = lex[rivals[0].id] if rivals else 0.0
        if best - second >= self.LEXICAL_MARGIN:
            return top
        if second >= self.LEXICAL_DECIDE:
            # Several differently-named elements match every goal word ('Save' vs 'Save As';
            # two tabs of the same repo). The tightest name, fewest tokens beyond the goal's,
            # is the literal reading; deferring to Laya here measured worse.
            tied = [top] + [e for e in rivals if lex[e.id] >= self.LEXICAL_DECIDE]
            return min(tied, key=lambda e: (len(tokens(e.name, keep_stop=True)), tied.index(e)))
        # "switch to spotify" ties the Spotify window with a Chrome tab titled 'Spotify - Web
        # Player' (which sorts first when it is Chrome's current tab). Naming an app is a
        # request for that app's window (measured live).
        for e in by_lex:
            if lex[e.id] < best:
                break
            if e.is_window and self._names_app(e):
                return e
        return None

    def _names_app(self, win: UIElement) -> bool:
        goal_toks = set(self._goal_tokens)
        return bool(goal_toks) and goal_toks <= set(tokens(win.window))

    def _fused(self, ranked: list[tuple[str, float]], fine: list[UIElement]) -> list[tuple[str, float]]:
        """Blend Laya's probabilities with retriever similarity: p' ∝ p^(1-w) · s^w, where s is
        the similarity rescaled to [0.05, 1] over the fine set. w = 0 leaves Laya alone."""
        w = self.cfg.fusion_weight
        if w <= 0 or not self._sims or not ranked:
            return ranked
        sims = {f"click_{e.id}": self._sims.get(e.id, 0.0) for e in fine}
        lo, hi = min(sims.values()), max(sims.values())
        span = (hi - lo) or 1.0
        fused = {a: (max(p, 1e-6) ** (1 - w)) * (0.05 + 0.95 * (sims.get(a, lo) - lo) / span) ** w for a, p in ranked}
        z = sum(fused.values()) or 1.0
        return sorted(((a, round(v / z, 4)) for a, v in fused.items()), key=lambda kv: kv[1], reverse=True)

    # -- shortlisting -------------------------------------------------------------
    def shortlist(self, goal: str, state: dict, elements: list[UIElement], by_lex: list[UIElement], lex: dict[int, float], kinds: set[str] | None = None) -> list[UIElement]:
        """Cut to FINE_K. Guaranteed: lexical hits and kind matches. Rest, by config:
        "retriever": embedding top-`retriever_top`, order-independent, then one Laya cut.
        "chunks":    Laya over chunks of `coarse_chunk` keeping `coarse_keep` each, then a cut.
        If the goal names a kind ("tab") and the screen has at least FINE_K of that kind, only
        that kind is retrieved: with 23 open tabs the human must see tabs, not buttons."""
        if len(elements) <= self.FINE_K:
            return list(elements)
        kinds = kinds if kinds is not None else kinds_in_goal(goal)
        pool = self._pool(elements, kinds)
        guaranteed = self._guaranteed(goal, pool, by_lex, lex, kinds)
        kept: dict[int, UIElement] = {e.id: e for e in guaranteed}
        chunk = max(self.FINE_K, self.cfg.coarse_chunk)
        if self.retriever is not None:
            # Fill to FINE_K in similarity order. No Laya cut pass: that would be another
            # 11+ option ranking, which is exactly what the retriever replaces.
            t = time.perf_counter()
            ranked = self.retriever.rank(goal, pool)
            self._retrieve_ms = (time.perf_counter() - t) * 1000
            self._sims = {e.id: s for e, s in ranked}
            self._retrieved = [e.id for e, _ in ranked[: self.cfg.retriever_top]]
            for e, _sim in ranked:
                if len(kept) >= self.FINE_K:
                    break
                kept.setdefault(e.id, e)
            return [e for e in elements if e.id in kept]
        for start in range(0, len(elements), chunk):
            part = elements[start : start + chunk]
            for e in self._top_clicks(state, goal, part, self.cfg.coarse_keep):
                kept.setdefault(e.id, e)
        return self._final_cut(state, goal, elements, kept, guaranteed, chunk)

    def _pool(self, elements: list[UIElement], kinds: set[str]) -> list[UIElement]:
        """Narrow the retrieval pool by what the goal names, when enough remain:
        a kind ("tab" -> TabItems) and/or an app ("chrome" -> elements of Chrome windows).
        Measured: "switch to the github chrome tab" chose the Terminal's current tab at
        p=0.77 when the pool was all 21 tabs across apps."""
        goal_toks = set(self._goal_tokens)
        pool = elements
        if kinds:
            by_kind = [e for e in pool if e.kind in kinds]
            if len(by_kind) >= self.FINE_K:
                pool = by_kind
        apps = {e.window for e in elements if e.window and set(tokens(e.window)) & goal_toks}
        if apps:
            by_app = [e for e in pool if e.window in apps]
            if len(by_app) >= self.FINE_K:
                pool = by_app
        return pool

    def _guaranteed(self, goal, pool, by_lex, lex, kinds: set[str] | None = None) -> list[UIElement]:
        """Lexical hits from anywhere on screen (they are named); kind matches only from the
        narrowed pool (otherwise ten Terminal tabs fill the list for "the github chrome tab")."""
        kinds = kinds if kinds is not None else kinds_in_goal(goal)
        out: list[UIElement] = []
        for e in by_lex:
            if lex[e.id] >= 0.5 and len(out) < self.FINE_K // 2:
                out.append(e)
        for e in pool:
            if e.kind in kinds and e not in out and len(out) < self.FINE_K - 2:
                out.append(e)
        return out

    def _final_cut(self, state, goal, elements, kept, guaranteed, chunk) -> list[UIElement]:
        merged = [e for e in elements if e.id in kept]
        while len(merged) > self.FINE_K:
            rest = [e for e in merged if e not in guaranteed]
            cut = self._top_clicks(state, goal, rest[:chunk], self.FINE_K - len(guaranteed))
            merged = guaranteed + cut + rest[chunk:]
        return merged

    def _top_clicks(self, state: dict, goal: str, elements: list[UIElement], k: int) -> list[UIElement]:
        if not elements or k <= 0:
            return []
        raw = self.predict(state, self._questions(elements, goal, extras=False))
        by_id = {e.id: e for e in elements}
        out = []
        for action, _ in self._ranked(raw):
            out.append(by_id[int(action.split("_", 1)[1])])
            if len(out) == k:
                break
        return out

    # -- building blocks ------------------------------------------------------
    def rank_elements(self, goal: str, elements: list[UIElement]) -> list[UIElement]:
        """Stable order: kind and word matches first, then screen order. Only affects which
        elements survive `max_elements` and how chunks are formed."""
        g = set(tokens(goal))
        kinds = kinds_in_goal(goal)

        def score(e: UIElement) -> float:
            overlap = len(g & set(tokens(e.name)))
            return (1.5 if e.kind in kinds else 0.0) + overlap * 1.0 - 0.0001 * e.id

        return sorted(elements, key=score, reverse=True)

    kinds_in_goal = staticmethod(kinds_in_goal)

    def build_state(self, goal: str, snap: Snapshot, history: list[str], include_elements: bool = False, hint: str | None = None) -> dict:
        state: dict[str, Any] = {
            "goal": goal,
            "window": snap.window_title[:80],
            "previous_actions": history[-self.cfg.history_len:],
        }
        if hint:
            state["user_clarification"] = hint
        if include_elements:
            state["visible"] = [e.label() for e in snap.elements[:40]]
        return state

    @staticmethod
    def _questions(elements: list[UIElement], goal: str, extras: bool = True) -> dict[str, Any]:
        """Elements are the only options. done / need_text are separate yes-no questions."""
        qs: dict[str, Any] = {
            "action": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {f"click_{e.id}": e.label() for e in elements},
            }
        }
        if extras:
            qs["done"] = {"type": "noul", "instructions": f'Is the request "{goal}" already fully satisfied on the current screen?'}
            qs["need_text"] = {"type": "noul", "instructions": f'Does the next step for "{goal}" require typing text rather than clicking?'}
        return qs

    @staticmethod
    def _ranked(raw: dict[str, Any]) -> list[tuple[str, float]]:
        probs: dict[str, float] = raw["answers"]["action"]["probabilities"]
        return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

    @staticmethod
    def margin_confidence(ranked: list[tuple[str, float]], laya_conf: float) -> float:
        """Confidence = the top option's probability. The final pass runs in Laya's calibrated
        6-10 option bucket, so p(top) is the honest number. An earlier margin formula
        ((top - second) * 2 + top / 2) turned p=0.55 into 0.97 and let a wrong click through
        (measured: "make it louder" -> Play). Laya's entropy `confidence` is ignored: it
        collapses as the option count grows."""
        if not ranked:
            return 0.0
        return round(min(ranked[0][1], 1.0), 4)

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
            self.preload()
        t = time.perf_counter()
        try:
            return self._predict(state, questions)
        finally:
            self._passes += 1
            self._laya_ms += (time.perf_counter() - t) * 1000

    def preload(self, log: Callable[[str], None] = lambda s: None) -> None:
        """Load the model now (call from a background thread at startup). Idempotent."""
        with self._load_lock:
            if self._predict is None:
                from laya_agent.brain.model_loader import load_predict

                t = time.perf_counter()
                self._predict = load_predict(self.cfg, log)
                self.load_ms = round((time.perf_counter() - t) * 1000, 1)
            if self.retriever is not None and hasattr(self.retriever, "preload"):
                self.retriever.preload(log)

    @property
    def ready(self) -> bool:
        return self._predict is not None
