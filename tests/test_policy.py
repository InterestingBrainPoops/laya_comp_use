from laya_agent.brain.laya_policy import LayaPolicy
from laya_agent.brain.text_gen import NullTextGenerator, TextNeedsHuman, make_text_generator
from laya_agent.config import Config
from laya_agent.models import Rect, Snapshot, UIElement

import pytest


def _els(names, kind="Button"):
    return [UIElement(id=i + 1, kind=kind, name=n, rect=Rect(0, i * 10, 50, i * 10 + 9)) for i, n in enumerate(names)]


def _snap(names, kind="Button"):
    return Snapshot(png=b"", width=100, height=100, window_title="Notepad", elements=_els(names, kind))


def _fake_predict(winner: str, done: float = 0.1):
    """Winner gets 0.7 when present in the question, else the first click option does."""
    calls = []

    def predict(state, questions):
        calls.append((state, questions))
        keys = list(questions["action"]["criteria"].keys())
        w = winner if winner in keys else next(k for k in keys if k.startswith("click_"))
        n = len(keys)
        probs = {k: (0.7 if k == w else 0.3 / (n - 1)) for k in keys}
        return {"answers": {"action": {"choice": w, "probabilities": probs, "confidence": 0.5}, "done": {"noul": done}}}

    predict.calls = calls
    return predict


def _click_keys(call):
    return [k for k in call[1]["action"]["criteria"] if k.startswith("click_")]


def test_decide_picks_element_and_exposes_top_k():
    fake = _fake_predict("click_2")
    policy = LayaPolicy(Config(alias_path=None, shortlist="chunks"), predict=fake)
    d = policy.decide("open the edit menu", _snap(["File", "Edit", "View"]), history=[])
    assert d.action == "click_2" and d.element.name == "Edit" and d.is_click
    assert d.top_k[0][0] == "click button 'Edit'" and abs(d.top_k[0][1] - 0.7) < 1e-6
    assert d.confidence >= 0.7
    state, questions = fake.calls[0]
    assert state["goal"] == "open the edit menu" and state["window"] == "Notepad"
    assert all(k.startswith("click_") for k in questions["action"]["criteria"])  # elements are the only options
    assert questions["done"]["type"] == "noul" and questions["need_text"]["type"] == "noul"
    assert d.top_actions[-1] == "scroll_down"  # offered to the human, never auto-chosen
    assert set(d.raw["_shortlist"]) == {1, 2, 3} and d.raw["_fine"] == [1, 2, 3]


def test_done_is_a_separate_probability_not_an_action():
    policy = LayaPolicy(Config(alias_path=None, shortlist="chunks"), predict=_fake_predict("click_1", done=0.95))
    d = policy.decide("nothing", _snap(["File"]), history=[])
    assert d.done_prob == 0.95 and d.action == "click_1"  # the loop reads done_prob against done_threshold


def test_single_pass_when_few_elements():
    fake = _fake_predict("click_1")
    LayaPolicy(Config(alias_path=None, shortlist="chunks"), predict=fake).decide("x", _snap(["A", "B"]), history=[])
    assert len(fake.calls) == 1


def test_shortlist_uses_laya_over_chunks_and_keeps_winner_without_keyword():
    """96 tabs named by page title; goal says 'github' but the tab is 'NandhaKishorM/laya'."""
    names = [f"Page {i}" for i in range(95)] + ["NandhaKishorM/laya"]
    fake = _fake_predict("click_96")
    policy = LayaPolicy(Config(alias_path=None, shortlist="chunks", coarse_chunk=20, coarse_keep=3), predict=fake)
    d = policy.decide("switch to the github tab", _snap(names, kind="TabItem"), history=[])
    assert d.element is not None and d.element.name == "NandhaKishorM/laya"
    chunk_calls = [c for c in fake.calls if len(_click_keys(c)) > policy.FINE_K]
    assert len(chunk_calls) >= 5  # 96 / 20 chunks, then a merge cut
    assert len(_click_keys(fake.calls[-1])) == policy.FINE_K  # calibrated fine pass
    assert 96 in d.raw["_fine"] and len(d.raw["_fine"]) == policy.FINE_K


def test_keyword_hits_survive_shortlist():
    names = [f"Thing {i}" for i in range(30)] + ["Save As"]
    fake = _fake_predict("click_1")  # Laya "prefers" other things; keyword must still survive
    policy = LayaPolicy(Config(alias_path=None, shortlist="chunks"), predict=fake)
    d = policy.decide("save as", _snap(names), history=[])
    assert 31 in d.raw["_fine"]


def test_kind_named_in_goal_survives_to_fine_pass():
    """Page buttons word-match 'github'; the tab does not. 'tab' in the goal must keep TabItems."""
    els = _els([f"Install GitHub {i}" for i in range(30)]) + [
        UIElement(id=31, kind="TabItem", name="NandhaKishorM/laya", rect=Rect(0, 0, 5, 5)),
        UIElement(id=32, kind="TabItem", name="Hugging Face", rect=Rect(6, 0, 11, 5)),
    ]
    snap = Snapshot(png=b"", width=1, height=1, window_title="Chrome", elements=els)
    d = LayaPolicy(Config(alias_path=None, shortlist="chunks"), predict=_fake_predict("click_1")).decide("switch to the github tab", snap, history=[])
    assert {31, 32} <= set(d.raw["_fine"])


def test_same_name_tie_prefers_selected_element():
    """'close this tab' with two 'Close Tab' buttons must hit the current tab's, not the first."""
    els = [
        UIElement(id=1, kind="Button", name="Close Tab", rect=Rect(0, 0, 1, 1)),
        UIElement(id=2, kind="Button", name="Close Tab", rect=Rect(5, 0, 6, 1), selected=True),
    ]
    snap = Snapshot(png=b"", width=1, height=1, window_title="Terminal", elements=els)
    d = LayaPolicy(Config(alias_path=None, shortlist="chunks"), predict=_fake_predict("click_1")).decide("close this tab", snap, history=[])
    assert d.element.id == 2 and d.raw["_decided_by"] == "name match"


GITHUB_TAB = "laya_comp_use/docs/ARCHITECTURE.md at main · InterestingBrainPoops/laya_comp_use"


def test_hint_drives_the_name_match():
    """Session log: goal 'open github tab', hint 'open the laya_comp_use tab' was ignored."""
    tabs = ["New Tab", "Coed dorm problem!! Pls help!!", "Silent_Coffeee (u/Silent_Coffeee) - Reddit", GITHUB_TAB]
    policy = LayaPolicy(Config(alias_path=None, shortlist="chunks"), predict=_fake_predict("click_1"))
    d = policy.decide("open github tab", _snap(tabs, kind="TabItem"), history=[], hint="open the laya_comp_use tab")
    assert d.element.name == GITHUB_TAB and d.raw["_decided_by"] == "name match"


def test_kind_named_in_goal_restricts_the_pool_when_plenty():
    """23 tabs and 100 buttons: 'switch to the ... tab' must offer tabs, not buttons."""
    from laya_agent.brain.retriever import EmbeddingRetriever
    from tests.test_retriever import fake_embed

    els = _els([f"Button {i}" for i in range(40)]) + [
        UIElement(id=100 + i, kind="TabItem", name=f"Page {i}", rect=Rect(0, 0, 5, 5)) for i in range(20)
    ]
    snap = Snapshot(png=b"", width=1, height=1, window_title="Chrome", elements=els)
    policy = LayaPolicy(Config(alias_path=None), predict=_fake_predict("click_1"), retriever=EmbeddingRetriever(Config(), embed=fake_embed))
    d = policy.decide("switch to the github tab", snap, history=[])
    kinds = {d.raw["_elements"][f"click_{i}"].kind for i in d.raw["_fine"]}
    assert kinds == {"TabItem"} and len(d.raw["_fine"]) == policy.FINE_K


def test_app_named_in_goal_restricts_the_pool():
    """Live: 'switch to the github chrome tab' picked the Terminal's current tab at p=0.77."""
    from laya_agent.brain.retriever import EmbeddingRetriever
    from tests.test_retriever import fake_embed

    els = [UIElement(id=i + 1, kind="TabItem", name=f"Term {i}", rect=Rect(0, 0, 5, 5), window="Terminal", selected=(i == 0)) for i in range(10)]
    els += [UIElement(id=50 + i, kind="TabItem", name=f"Page {i}", rect=Rect(0, 0, 5, 5), window="Chrome", foreground=False) for i in range(12)]
    snap = Snapshot(png=b"", width=1, height=1, window_title="Terminal", elements=els)
    policy = LayaPolicy(Config(alias_path=None), predict=_fake_predict("click_1"), retriever=EmbeddingRetriever(Config(), embed=fake_embed))
    d = policy.decide("switch to the github chrome tab", snap, history=[])
    assert {d.raw["_elements"][f"click_{i}"].window for i in d.raw["_fine"]} == {"Chrome"}


def test_full_match_tie_goes_to_the_tightest_name():
    policy = LayaPolicy(Config(alias_path=None, shortlist="chunks"), predict=_fake_predict("click_1"))
    d = policy.decide("save", _snap(["Save As", "Save", "Save All"]), history=[])
    assert d.element.name == "Save" and d.raw["_decided_by"] == "name match"
    tabs = [GITHUB_TAB, "InterestingBrainPoops/laya_comp_use", "New Tab"]
    d = policy.decide("open the laya_comp_use tab", _snap(tabs, kind="TabItem"), history=[])
    assert d.element.name == "InterestingBrainPoops/laya_comp_use" and d.raw["_decided_by"] == "name match"


def test_window_alias_is_keyed_by_app(tmp_path):
    from laya_agent.brain.aliases import AliasStore

    store = AliasStore(tmp_path / "a.json")
    policy = LayaPolicy(Config(alias_path=None, shortlist="chunks"), predict=_fake_predict("click_1"), aliases=store)
    win = UIElement(id=1, kind="Window", name="New Tab - Google Chrome", rect=Rect(0, 0, 5, 5), window="Chrome")
    policy.remember("switch to the browser", win)
    assert store.as_dict() == {"app:Chrome": ["browser"]}
    later = UIElement(id=1, kind="Window", name="Some other page - Google Chrome", rect=Rect(0, 0, 5, 5), window="Chrome")
    ok = UIElement(id=2, kind="Button", name="OK", rect=Rect(0, 0, 5, 5))
    snap = Snapshot(png=b"", width=1, height=1, window_title="x", elements=[later, ok])
    d = policy.decide("switch to the browser", snap, history=[])
    assert d.element is later and d.raw["_decided_by"] == "name match"


def test_rank_boosts_kind_named_in_goal():
    els = _els(["Zzz", "Yyy"], kind="Button") + [UIElement(id=3, kind="TabItem", name="Xxx", rect=Rect(0, 0, 5, 5))]
    ranked = LayaPolicy(Config(alias_path=None, shortlist="chunks"), predict=_fake_predict("click_1")).rank_elements("switch to the next tab", els)
    assert ranked[0].kind == "TabItem"


def test_margin_confidence_low_when_flat():
    flat = [("a", 0.26), ("b", 0.25), ("c", 0.25), ("d", 0.24)]
    assert LayaPolicy.margin_confidence(flat, 0.05) < 0.3
    sharp = [("a", 0.9), ("b", 0.05)]
    assert LayaPolicy.margin_confidence(sharp, 0.05) >= 0.9


def test_ask_returns_probability():
    def predict(state, questions):
        assert "visible" in state
        return {"answers": {"q": {"noul": 0.83}}}

    policy = LayaPolicy(Config(alias_path=None, shortlist="chunks"), predict=predict)
    assert policy.ask("is a dialog open?", _snap(["OK"])) == 0.83


def test_null_text_generator_defers_to_human():
    gen = make_text_generator(Config(alias_path=None, shortlist="chunks"))
    assert isinstance(gen, NullTextGenerator)
    with pytest.raises(TextNeedsHuman):
        gen.generate("type the url", {})
