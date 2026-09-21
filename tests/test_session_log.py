import json

from laya_agent.models import Decision, InputRequest, Rect, Snapshot, StepResult, UIElement
from laya_agent.session_log import SessionLog

EL = UIElement(id=1, kind="Button", name="OK", rect=Rect(0, 0, 10, 10))


def _snap():
    return Snapshot(png=b"", width=10, height=10, window_title="w", elements=[EL], windows=["w"], timings={"parse_ms": 12.0})


def _dec():
    return Decision(action="click_1", element=EL, confidence=0.9, done_prob=0.1, top_k=[("click button 'OK'", 0.9)],
                    top_actions=["click_1"], raw={"_decided_by": "laya", "answers": {"action": {"probabilities": {"click_1": 0.9}}}},
                    timings={"laya_ms": 40.0, "laya_passes": 1})


def test_writes_jsonl_and_text_mirror(tmp_path):
    log = SessionLog(tmp_path, screenshots=False)
    log.goal("press ok")
    log.parsed(1, _snap())
    log.decided(1, "press ok", _dec())
    log.asked(InputRequest(reason="not sure", options=[("a", 0.5)], step=1), "1")
    log.acted(1, "clicked button 'OK' at (5, 5)", 30.0)
    log.step(StepResult(1, _snap(), _dec(), executed=True, note="clicked", timings={"parse_ms": 12.0, "act_ms": 30.0}))
    log.finished("done after 1 action(s)", 1)
    lines = [json.loads(l) for l in log.path.read_text(encoding="utf-8").splitlines()]
    kinds = [l["kind"] for l in lines]
    assert kinds == ["session_start", "goal", "parsed", "decided", "human", "acted", "step", "finished"]
    dec = lines[3]
    assert dec["element"]["name"] == "OK" and dec["laya_probabilities"] == {"click_1": 0.9} and dec["timings"]["laya_ms"] == 40.0
    text = log.path.with_suffix(".log").read_text(encoding="utf-8")
    assert "decided click_1 by laya" in text and "asked (not sure) -> reply '1'" in text


def test_disabled_log_is_a_no_op(tmp_path):
    log = SessionLog(None)
    log.goal("x")
    log.step(StepResult(1, _snap(), _dec(), executed=True))
    assert log.path is None and not any(tmp_path.iterdir())
