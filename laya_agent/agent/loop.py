"""The perceive -> decide -> gate -> act loop. Framework-free.

UI frameworks adapt `LoopEvents`: the CLI uses input(), the Qt window uses signals
plus a threading.Event. The loop itself never imports Qt.
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
    ) -> None:
        self.cfg = cfg
        self.parser = parser
        self.policy = policy
        self.executor = executor
        self.text_gen = text_gen
        self.events = events
        self.stop_event = threading.Event()
        self.own_hwnd: int | None = None

    # -- entry points ---------------------------------------------------------
    def run(self, goal: str) -> str:
        self.stop_event.clear()
        history: list[str] = []
        last_action: str | None = None
        repeats = 0
        step = 0
        while step < self.cfg.max_steps:
            if self.stop_event.is_set():
                return self._finish("stopped by user")
            step += 1
            snap = self._perceive()
            decision = self.policy.decide(goal, snap, history)
            self.events.on_log(
                f"step {step}: {decision.top_k[0][0]} (p={decision.top_k[0][1]:.2f}, "
                f"conf={decision.confidence:.2f}, done={decision.done_prob:.2f})"
            )

            if decision.done_prob >= self.cfg.done_threshold:
                self.events.on_step(StepResult(step, snap, decision, executed=False, note="goal achieved"))
                return self._finish(f"done after {step - 1} action(s)")

            repeats = repeats + 1 if decision.action == last_action else 0
            needs_human = (
                decision.confidence < self.cfg.conf_threshold
                or repeats >= 2
                or (decision.action == ACTION_DONE and decision.done_prob < 0.5)
            )
            if needs_human:
                reason = "repeating the same action" if repeats >= 2 else "not sure what to click"
                reply = self.events.on_input_needed(
                    InputRequest(reason=reason, options=decision.top_k, step=step, kind="pick")
                ).strip()
                if reply.lower() in STOP_WORDS:
                    return self._finish("stopped by user")
                picked = self._pick(reply, decision)
                if picked is None:
                    history.append(f"user hint: {reply}")
                    self.events.on_step(StepResult(step, snap, decision, executed=False, note="hint"))
                    repeats = 0
                    last_action = None
                    continue
                decision = picked

            if decision.action == ACTION_DONE:
                self.events.on_step(StepResult(step, snap, decision, executed=False, note="done"))
                return self._finish(f"done after {step - 1} action(s)")

            note = self._act(goal, snap, decision, step)
            if note is None:
                return self._finish("stopped by user")
            history.append(note)
            last_action = decision.action
            self.events.on_step(StepResult(step, snap, decision, executed=True, note=note))
            time.sleep(self.cfg.step_delay_s)
        return self._finish(f"gave up after {self.cfg.max_steps} steps")

    def answer(self, question: str, goal: str = "") -> float:
        """'? is the download finished' -> P(true) about the current screen."""
        snap = self._perceive()
        return self.policy.ask(question, snap, goal=goal)

    def stop(self) -> None:
        self.stop_event.set()

    # -- internals ------------------------------------------------------------
    def _perceive(self) -> Snapshot:
        with hidden_window(self.own_hwnd):
            return self.parser.parse(exclude_hwnd=self.own_hwnd)

    def _act(self, goal: str, snap: Snapshot, decision: Decision, step: int) -> str | None:
        if decision.action == ACTION_NEED_TEXT:
            task = f"text to type for goal: {goal}"
            try:
                text = self.text_gen.generate(task, self.policy.build_state(goal, snap, [], include_elements=True))
            except TextNeedsHuman:
                text = self.events.on_input_needed(
                    InputRequest(reason="what should I type?", options=[], step=step, kind="text")
                )
                if text.strip().lower() in STOP_WORDS:
                    return None
            with hidden_window(self.own_hwnd):
                self.executor.type_text(text)
            return f"typed '{text[:40]}'"
        with hidden_window(self.own_hwnd):
            return self.executor.execute(decision)

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
        )

    def _finish(self, summary: str) -> str:
        self.events.on_finished(summary)
        return summary


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
    )
