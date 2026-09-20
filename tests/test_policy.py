from laya_agent.brain.laya_policy import LayaPolicy
from laya_agent.brain.text_gen import NullTextGenerator, TextNeedsHuman, make_text_generator
from laya_agent.config import Config
from laya_agent.models import ACTION_DONE, META_ACTIONS, Rect, Snapshot, UIElement

import pytest


def _els(names):
    return [UIElement(id=i + 1, kind="Button", name=n, rect=Rect(0, i * 10, 50, i * 10 + 9)) for i, n in enumerate(names)]


def _snap(names):
    return Snapshot(png=b"", width=100, height=100, window_title="Notepad", elements=_els(names))


def _fake_predict(winner: str, done: float = 0.1):
    calls = []

    def predict(state, questions):
        calls.append((state, questions))
        keys = list(questions["action"]["criteria"].keys())
        n = len(keys)
        probs = {k: (0.7 if k == winner else 0.3 / (n - 1)) for k in keys}
        return {
            "answers": {
                "action": {"choice": winner, "probabilities": probs, "confidence": 0.5},
                "done": {"noul": done},
            }
        }

    predict.calls = calls
    return predict


def test_decide_picks_element_and_exposes_top_k():
    fake = _fake_predict("click_2")
    policy = LayaPolicy(Config(max_elements=5), predict=fake)
    d = policy.decide("open the edit menu", _snap(["File", "Edit", "View"]), history=[])
    assert d.action == "click_2" and d.element.name == "Edit" and d.is_click
    assert d.top_k[0][0] == "click Button 'Edit'" and abs(d.top_k[0][1] - 0.7) < 1e-6
    assert d.confidence >= 0.7
    state, questions = fake.calls[0]
    assert state["goal"] == "open the edit menu" and state["window"] == "Notepad"
    assert set(META_ACTIONS) <= set(questions["action"]["criteria"])


def test_decide_meta_action_has_no_element():
    policy = LayaPolicy(Config(), predict=_fake_predict(ACTION_DONE, done=0.95))
    d = policy.decide("nothing", _snap(["File"]), history=[])
    assert d.action == ACTION_DONE and d.element is None and d.done_prob == 0.95


def test_rank_puts_goal_words_first_and_caps():
    policy = LayaPolicy(Config(max_elements=2), predict=_fake_predict("click_3"))
    names = ["Zzz", "Yyy", "Save As", "Xxx"]
    ranked = policy.rank_elements("save the file as", _els(names))
    assert ranked[0].name == "Save As"
    d = policy.decide("save the file as", _snap(names), history=[])
    assert len([k for k in d.raw["answers"]["action"]["probabilities"] if k.startswith("click_")]) == 2


def test_coarse_to_fine_second_pass_when_many_elements():
    fake = _fake_predict("click_9")
    policy = LayaPolicy(Config(max_elements=20), predict=fake)
    names = [f"Item {i}" for i in range(12)]
    d = policy.decide("pick item 8", _snap(names), history=[])
    assert len(fake.calls) == 2
    coarse = [k for k in fake.calls[0][1]["action"]["criteria"] if k.startswith("click_")]
    fine = [k for k in fake.calls[1][1]["action"]["criteria"] if k.startswith("click_")]
    assert len(coarse) == 12 and len(fine) == LayaPolicy.FINE_K and "click_9" in fine
    assert d.action == "click_9" and d.element.name == "Item 8"


def test_single_pass_when_few_elements():
    fake = _fake_predict("click_1")
    LayaPolicy(Config(), predict=fake).decide("x", _snap(["A", "B"]), history=[])
    assert len(fake.calls) == 1


def test_margin_confidence_low_when_flat():
    flat = [("a", 0.26), ("b", 0.25), ("c", 0.25), ("d", 0.24)]
    assert LayaPolicy.margin_confidence(flat, 0.05) < 0.3
    sharp = [("a", 0.9), ("b", 0.05)]
    assert LayaPolicy.margin_confidence(sharp, 0.05) >= 0.9


def test_ask_returns_probability():
    def predict(state, questions):
        assert "visible" in state
        return {"answers": {"q": {"noul": 0.83}}}

    policy = LayaPolicy(Config(), predict=predict)
    assert policy.ask("is a dialog open?", _snap(["OK"])) == 0.83


def test_null_text_generator_defers_to_human():
    gen = make_text_generator(Config())
    assert isinstance(gen, NullTextGenerator)
    with pytest.raises(TextNeedsHuman):
        gen.generate("type the url", {})
