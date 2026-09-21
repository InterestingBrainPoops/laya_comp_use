"""The perceive -> decide -> gate -> act loop. Framework-free.

UI frameworks adapt `LoopEvents`: the CLI uses input(), the Qt window uses signals
plus a threading.Event. The loop itself never imports Qt. Every step carries a merged
timing breakdown (parse, decide, human, act, settle) and is written to the session log.
"""
from __future__ import annotations

import threading
import time
from typing import Protocol

from laya_agent.actuation.executor import Executor, hidden_window
from laya_agent.brain.laya_policy import LayaPolicy
from laya_agent.brain.text_gen import TextGenerator, TextNeedsHuman
from laya_agent.config import Config
from laya_agent.models import ACTION_DONE, ACTION_NEED_TEXT, Decision, InputRequest, Snapshot, StepResult
from laya_agent.perception.base import ScreenParser
from laya_agent.session_log import SessionLog

STOP_WORDS = {"stop", "abort", "cancel", "quit"}


class LoopEvents(Protocol):
    def on_log(self, msg: str) -> None: ...
    def on_step(self, result: StepResult) -> None: ...
    def on_input_needed(self, req: InputRequest) -> str:
        """Block until the human answers. A digit picks an option, text is a hint,
        'stop' aborts. For kind == 'text' the reply is typed literally."""
        ...
    def on_finished(self, summary: str) -> None: ...


class AgentLoop:
    def __init__(
        self,
        cfg: Config,
        parser: ScreenParser,
        policy: LayaPolicy,
        executor: Executor,
        text_gen: TextGenerator,
        events: LoopEvents,
        session_log: SessionLog | None = None,
    ) -> None:
        self.cfg = cfg
        self.parser = parser
        self.policy = policy
        self.executor = executor
        self.text_gen = text_gen
        self.events = events
        self.log = session_log or SessionLog(None)
        self.stop_event = threading.Event()
        self.own_hwnd: int | None = None
        self.last_snapshot: Snapshot | None = None
        self.last_decision: Decision | None = None

    # -- entry points ---------------------------------------------------------
    def run(self, goal: str) -> str:
        self.stop_event.clear()
        self.log.goal(goal)
        history: list[str] = []
        last_action: str | None = None
        repeats = 0
        step = 0
        actions = 0
        done_streak = 0
        hint: str | None = None  # the user's last free-text reply; drives matching until an action runs
        while step < self.cfg.max_steps:
            if self.stop_event.is_set():
                return self._finish("stopped by user", actions)
            step += 1
            tm: dict[str, float] = {}

            t = time.perf_counter()
            snap = self._perceive()
            tm["parse_ms"] = _ms(t)
            tm.update({k: v for k, v in snap.timings.items() if k != "parse_ms"})
            self.log.parsed(step, snap)

            t = time.perf_counter()
            decision = self.policy.decide(goal, snap, history, hint=hint)
            tm["decide_ms"] = _ms(t)
            tm.update({k: v for k, v in decision.timings.items() if k != "decide_ms"})
            self.last_snapshot, self.last_decision = snap, decision
            self.log.decided(step, goal, decision)
            self._log_decision(step, snap, decision)

            done_streak = done_streak + 1 if decision.done_prob >= self.cfg.done_threshold else 0
            if done_streak >= self.cfg.done_confirmations:
                self._emit(StepResult(step, snap, decision, executed=False, note="goal achieved", timings=tm))
                return self._finish(f"done after {actions} action(s)", actions)
            if done_streak:
                self.events.on_log(f"looks done (p={decision.done_prob:.2f}), checking once more")
                self._emit(StepResult(step, snap, decision, executed=False, note="done? confirming", timings=tm))
                time.sleep(self.cfg.step_delay_s)
                continue

            repeats = repeats + 1 if decision.action == last_action else 0
            needs_human = (
                decision.confidence < self.cfg.conf_threshold
                or repeats >= 2
                or (decision.action == ACTION_DONE and decision.done_prob < 0.5)
            )
            if needs_human:
                reason = "repeating the same action" if repeats >= 2 else "not sure what to click"
                req = InputRequest(reason=reason, options=decision.top_k, step=step, kind="pick")
                t = time.perf_counter()
                reply = self.events.on_input_needed(req).strip()
                tm["human_ms"] = _ms(t)
                self.log.asked(req, reply)
                if reply.lower() in STOP_WORDS:
                    return self._finish("stopped by user", actions)
                picked = self._pick(reply, decision)
                if picked is None:
                    hint = reply
                    history.append(f"user hint: {reply}")
                    self._emit(StepResult(step, snap, decision, executed=False, note="hint", timings=tm))
                    repeats = 0
                    last_action = None
                    continue
                decision = picked
                if decision.element is not None:
                    self.policy.remember(hint or goal, decision.element)
                    self.events.on_log(f"learned: '{hint or goal}' -> {decision.element.label()}")
                    self.log.event("alias_learned", goal=hint or goal, element=decision.element)

            if decision.action == ACTION_DONE:
                self._emit(StepResult(step, snap, decision, executed=False, note="done", timings=tm))
                return self._finish(f"done after {actions} action(s)", actions)

            t = time.perf_counter()
            note = self._act(goal, snap, decision, step, tm)
            tm["act_ms"] = _ms(t)
            if note is None:
                return self._finish("stopped by user", actions)
            self.log.acted(step, note, tm["act_ms"])
            actions += 1
            if hint and decision.element is not None and decision.raw.get("_decided_by") == "name match":
                # The hint named it; remember the original goal's words for this element so
                # "the github chrome tab" is a name match next time.
                self.policy.remember(goal, decision.element)
                self.events.on_log(f"learned: '{goal}' -> {decision.element.label()}")
                self.log.event("alias_learned", goal=goal, element=decision.element)
            hint = None
            history.append(note)
            last_action = decision.action
            self._emit(StepResult(step, snap, decision, executed=True, note=note, timings=tm))
            time.sleep(self.cfg.step_delay_s)
            tm["settle_ms"] = round(self.cfg.step_delay_s * 1000, 1)
        return self._finish(f"gave up after {self.cfg.max_steps} steps", actions)

    def answer(self, question: str, goal: str = "") -> float:
        """'? is the download finished' -> P(true) about the current screen."""
        snap = self._perceive()
        p = self.policy.ask(question, snap, goal=goal)
        self.log.event("question", question=question, p_true=p, front=snap.window_title)
        return p

    def inspect(self) -> Snapshot:
        """Parse only, no decision. For the 'show all elements' debug view."""
        return self._perceive()

    def stop(self) -> None:
        self.stop_event.set()

    # -- internals ------------------------------------------------------------
    def _perceive(self) -> Snapshot:
        with hidden_window(self.own_hwnd):
            snap = self.parser.parse(exclude_hwnd=self.own_hwnd, window_title=self.cfg.target_window)
        self.last_snapshot = snap
        return snap

    def _act(self, goal: str, snap: Snapshot, decision: Decision, step: int, tm: dict[str, float]) -> str | None:
        if decision.action == ACTION_NEED_TEXT:
            task = f"text to type for goal: {goal}"
            t = time.perf_counter()
            try:
                text = self.text_gen.generate(task, self.policy.build_state(goal, snap, [], include_elements=True))
                tm["text_gen_ms"] = _ms(t)
            except TextNeedsHuman:
                req = InputRequest(reason="what should I type?", options=[], step=step, kind="text")
                t = time.perf_counter()
                text = self.events.on_input_needed(req)
                tm["human_ms"] = tm.get("human_ms", 0.0) + _ms(t)
                self.log.asked(req, text)
                if text.strip().lower() in STOP_WORDS:
                    return None
            with hidden_window(self.own_hwnd):
                self.executor.type_text(text)
            return f"typed '{text[:40]}'"
        with hidden_window(self.own_hwnd):
            return self.executor.execute(decision)

    def _log_decision(self, step: int, snap: Snapshot, decision: Decision) -> None:
        n_short = len(decision.raw.get("_shortlist", []))
        by = decision.raw.get("_decided_by", "laya")
        self.events.on_log(
            f"step {step} in '{snap.window_title[:40]}': {len(snap.elements)} elements, "
            f"{n_short} shortlisted. {decision.top_k[0][0]} by {by} (p={decision.top_k[0][1]:.2f}, "
            f"conf={decision.confidence:.2f}, done={decision.done_prob:.2f})"
        )

    def _emit(self, result: StepResult) -> None:
        self.log.step(result)
        self.events.on_step(result)

    @staticmethod
    def _pick(reply: str, decision: Decision) -> Decision | None:
        """Digit reply -> a Decision for that top_k option, marked fully confident."""
        if not reply.isdigit():
            return None
        idx = int(reply) - 1
        if not 0 <= idx < len(decision.top_actions):
            return None
        action = decision.top_actions[idx]
        element = decision.raw.get("_elements", {}).get(action)
        return Decision(
            action=action,
            element=element,
            confidence=1.0,
            done_prob=decision.done_prob,
            top_k=[decision.top_k[idx]],
            top_actions=[action],
            raw=decision.raw,
            timings=decision.timings,
        )

    def _finish(self, summary: str, actions: int) -> str:
        self.log.finished(summary, actions)
        self.events.on_finished(summary)
        return summary


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def build_loop(cfg: Config, events: LoopEvents) -> AgentLoop:
    from laya_agent.brain.text_gen import make_text_generator
    from laya_agent.perception.base import make_screen_parser

    return AgentLoop(
        cfg,
        parser=make_screen_parser(cfg),
        policy=LayaPolicy(cfg),
        executor=Executor(cfg),
        text_gen=make_text_generator(cfg),
        events=events,
        session_log=SessionLog(cfg.log_dir, cfg.log_screenshots),
    )
