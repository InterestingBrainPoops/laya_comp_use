"""Loop behaviour with fake parser, policy, executor, and events. No model, no screen."""
from laya_agent.agent.loop import AgentLoop
from laya_agent.brain.text_gen import NullTextGenerator
from laya_agent.config import Config
from laya_agent.models import ACTION_DONE, ACTION_NEED_TEXT, Decision, Rect, Snapshot, UIElement


EL = UIElement(id=1, kind="Button", name="File", rect=Rect(0, 0, 10, 10))
EL2 = UIElement(id=2, kind="Button", name="Edit", rect=Rect(0, 20, 10, 30))


class FakeParser:
    def parse(self, exclude_hwnd=None, window_title=None):
        return Snapshot(png=b"", width=10, height=10, window_title="w", elements=[EL, EL2])


def _dec(action, element, conf, done=0.1):
    top = [("click Button 'File'", 0.6), ("click Button 'Edit'", 0.3), ("done", 0.1)]
    return Decision(action=action, element=element, confidence=conf, done_prob=done, top_k=top,
                    top_actions=["click_1", "click_2", ACTION_DONE], raw={"_elements": {"click_1": EL, "click_2": EL2}})


class FakePolicy:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def decide(self, goal, snap, history):
        self.calls.append(list(history))
        return self.decisions.pop(0)

    def build_state(self, goal, snap, history, include_elements=False):
        return {"goal": goal}

    def ask(self, question, snap, goal="", history=None):
        return 0.5

    def remember(self, goal, element):
        self.calls.append(("remember", goal, element.name))


class FakeExecutor:
    def __init__(self):
        self.done = []

    def execute(self, decision):
        self.done.append(decision.action)
        return f"clicked {decision.element.name}"

    def type_text(self, text, press_enter=False):
        self.done.append(f"type:{text}")


class FakeEvents:
    def __init__(self, replies=()):
        self.replies = list(replies)
        self.log, self.steps, self.asked, self.finished = [], [], [], None

    def on_log(self, m): self.log.append(m)
    def on_step(self, r): self.steps.append(r)
    def on_input_needed(self, req):
        self.asked.append(req)
        return self.replies.pop(0)
    def on_finished(self, s): self.finished = s


def _loop(decisions, replies=(), **cfg):
    c = Config(step_delay_s=0, **cfg)
    ev = FakeEvents(replies)
    ex = FakeExecutor()
    loop = AgentLoop(c, FakeParser(), FakePolicy(decisions), ex, NullTextGenerator(c), ev)
    return loop, ex, ev


def test_confident_click_then_done():
    loop, ex, ev = _loop([_dec("click_1", EL, 0.95), _dec(ACTION_DONE, None, 0.9, done=0.9)])
    loop.run("open file")
    assert ex.done == ["click_1"]
    assert ev.finished.startswith("done after 1")
    assert ev.asked == []


def test_low_confidence_asks_and_pick_executes():
    loop, ex, ev = _loop([_dec("click_1", EL, 0.2), _dec(ACTION_DONE, None, 0.9, done=0.9)], replies=["2"])
    loop.run("x")
    assert ex.done == ["click_2"]  # user picked option 2
    assert ev.asked[0].kind == "pick"


def test_hint_reruns_decide_with_history():
    loop, ex, ev = _loop([_dec("click_1", EL, 0.2), _dec("click_1", EL, 0.95), _dec(ACTION_DONE, None, 0.9, done=0.9)],
                         replies=["try the toolbar"])
    loop.run("x")
    assert ex.done == ["click_1"]
    assert loop.policy.calls[1] == ["user hint: try the toolbar"]


def test_stop_reply_aborts():
    loop, ex, ev = _loop([_dec("click_1", EL, 0.2)], replies=["stop"])
    assert loop.run("x") == "stopped by user" and ex.done == []


def test_need_text_defers_to_human_in_v1():
    loop, ex, ev = _loop([_dec(ACTION_NEED_TEXT, None, 0.95), _dec(ACTION_DONE, None, 0.9, done=0.9)],
                         replies=["https://example.com"])
    loop.run("go to example.com")
    assert ex.done == ["type:https://example.com"] and ev.asked[0].kind == "text"


def test_repeat_same_action_asks():
    decs = [_dec("click_1", EL, 0.95)] * 3 + [_dec(ACTION_DONE, None, 0.9, done=0.9)]
    loop, ex, ev = _loop(decs, replies=["stop"])
    loop.run("x")
    assert ex.done == ["click_1", "click_1"] and ev.asked[0].reason.startswith("repeating")


def test_max_steps_gives_up():
    loop, ex, ev = _loop([_dec("click_1", EL, 0.95)] * 5, max_steps=2)
    # repeats trigger a question at step 3, but max_steps=2 ends first
    assert loop.run("x").startswith("gave up") and len(ex.done) == 2
